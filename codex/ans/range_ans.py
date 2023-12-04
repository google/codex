# Copyright 2022 CoDeX authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Range ANS implementation in Jax."""

from __future__ import annotations

from collections.abc import Sequence
import functools
from typing import Union

import chex
import jax
import numpy as np

jnp = jax.numpy
lax = jax.lax

LOG2_MAX_STATE = 32
LOG2_MIN_STATE = LOG2_MAX_STATE - 16


@chex.dataclass(frozen=True)
class RangeAnsStack:
  """Stack class encapsulates rANS encode/decode states."""
  state: chex.Array  # uint32 scalar.
  stream: chex.Array  # uint8 vector.
  stream_loc: chex.Array  # int32 scalar.

  @classmethod
  def make(cls,
           stream: chex.Array,
           initial_state: ... = 0) -> RangeAnsStack:
    """Initializes the stack for encoding.

    Jax requires static array shapes. Therefore encoding procedure cannot append
    bytes to the byte sink. Instead, a large enough buffer is allocated and the
    encode/decode procedures keep track of the number of bytes written to the
    stream so far.

    If the encode procedure overruns the buffer, there is no way to recover.
    Either the user should give up or attempt to re-encode everything with a
    larger buffer.

    Args:
      stream: An uint8 buffer to write out the rANS encoded stream.
      initial_state: The initial state of rANS encoder.

    Returns:
      An empty stack.
    """
    state = jnp.asarray(initial_state, dtype=jnp.uint32)
    chex.assert_rank(state, 0)

    stream = jnp.asarray(stream)
    chex.assert_type(stream, jnp.uint8)
    chex.assert_rank(stream, 1)
    stream_loc = jnp.array(0, dtype=jnp.int32)

    assert cls is RangeAnsStack, cls
    return cls(state=state, stream=stream, stream_loc=stream_loc)

  def _write(self, n_bytes: int) -> RangeAnsStack:
    chex.assert_scalar_in(n_bytes, 1, 4)
    shift = 8 * lax.iota(jnp.uint32, n_bytes)
    update = (jnp.expand_dims(self.state, -1) >> shift) & 0xFF
    stream = lax.dynamic_update_slice_in_dim(
        self.stream, update.astype(jnp.uint8), self.stream_loc, axis=-1)
    return self.replace(stream=stream, stream_loc=self.stream_loc + n_bytes)

  def write(self, n_bytes: int) -> RangeAnsStack:
    chex.assert_scalar_in(n_bytes, 1, 3)
    return self._write(n_bytes).replace(state=self.state >> (8 * n_bytes))

  def _read(self, n_bytes: int) -> RangeAnsStack:
    chex.assert_scalar_in(n_bytes, 1, 4)
    stream_loc = self.stream_loc - n_bytes
    top = lax.dynamic_slice(self.stream, [stream_loc], [n_bytes])

    shift = 8 * lax.iota(jnp.uint32, n_bytes)
    top = (top.astype(jnp.uint32) << shift).sum()
    return self.replace(state=top, stream_loc=stream_loc)

  def read(self, n_bytes: int) -> RangeAnsStack:
    chex.assert_scalar_in(n_bytes, 1, 3)
    stack = self._read(n_bytes)
    return stack.replace(state=(self.state << (8 * n_bytes)) + stack.state)

  def serialize(self) -> tuple[chex.Array, chex.Array]:
    """Flushes the state to the stream and returns the stream.

    Returns:
      A tuple of jnp.uint8 array and jnp.int32 scalar. The scalar value is the
      number of bytes actually written to the stream. If the scalar value is
      greater than the array size, then it means buffer overflow has happened,
      and `max_size_in_bytes` in the initialization was too small.
    """
    stack = lax.cond(self.state < (1 << 24),
                     lambda s: s._write(3),        # pylint: disable=protected-access
                     lambda s: s._write(4), self)  # pylint: disable=protected-access
    return stack.stream, stack.stream_loc

  @classmethod
  def deserialize(cls,
                  stream: chex.Array,
                  stream_size: chex.Array) -> RangeAnsStack:
    """Initializes the stack for decoding."""
    chex.assert_type(stream, jnp.uint8)
    chex.assert_rank(stream, 1)
    chex.assert_type(stream_size, jnp.int32)
    chex.assert_rank(stream_size, 0)
    assert cls is RangeAnsStack, cls

    stack = cls(state=jnp.zeros((), dtype=jnp.uint32),
                stream=stream, stream_loc=stream_size)
    # REQUIRES: 3 <= stream_size and stream_size <= stream.size
    # rANS state is uint32 and 3 or 4 bytes are going to be read.
    return lax.cond(stream_size & 1 == 0,
                    lambda s: s._read(4), lambda s: s._read(3), stack)  # pylint: disable=protected-access

  def decode_scalar(
      self,
      cdf: chex.Array,
      precision: chex.Array) -> tuple[RangeAnsStack, jax.Array]:
    """Decodes one symbol from rANS stack.

    The coding alphabet is assumed to be `{0, 1, ..., n - 1}`, and `n` is the
    coding alphabet size.

    `cdf` is a 1-D array of size n + 1 where `cdf[a]` is the quantized value of
    `Pr(X < a)` for `a = 0, 1, ..., n - 1` and `cdf[n] == 2**precision`.

    Note that
    - `cdf[0] == 0` because no coding symbol is less than 0.
    - `cdf[a] <= cdf[a + 1]` for a = 0, ..., n - 1.

    Args:
      cdf: An int32 array. See above for explanation.
      precision: A positive integer <= 16.

    Returns:
      A `RangeAnsStack` and a jnp.int32 scalar. The scalar value is the decoded
      symbol.
    """
    dtype = jnp.uint32
    precision = jnp.asarray(precision, dtype=dtype)
    chex.assert_rank(precision, 0)
    chex.assert_type(cdf, int)
    chex.assert_rank(cdf, 1)

    stack = lax.cond(
        jnp.logical_and(self.stream_loc != 0,
                        self.state < (1 << LOG2_MIN_STATE)),
        lambda s: s.read(2), lambda s: s, self)

    state = stack.state
    loword = state & ((1 << precision) - 1)
    # NOTE: searchsorted() is a binary search.
    # REQUIRES: len(cdf) != 0 and cdf[0] == 0 and cdf[-1] == 2**precision
    # Therefore `index` should be in the open interval (0, cdf.size - 1).
    index = jnp.searchsorted(cdf, loword, side="right")
    lower, upper = cdf[index - 1], cdf[index]

    size = (upper - lower).astype(dtype)
    state = (state >> precision) * size + (loword - lower.astype(dtype))
    chex.assert_type(state, dtype)
    return stack.replace(state=state), index - 1

  def encode_scalar(self,
                    bound: tuple[chex.Array, chex.Array],
                    precision: chex.Array | int) -> RangeAnsStack:
    """Encodes one symbol to rANS stack.

    The argument section refers to `cdf`. See `decode_scalar()` documentation
    for this `cdf` argument.

    Args:
      bound: A tuple of two int32 scalar: `lower` and `upper`. When `a` is the
        value to be encoded, these elements should be `cdf[a]` and `cdf[a+1]`.
        These two values are the quantized values of `Pr(X < a)` and
        `Pr(X <= a)`. Note that there difference is `Pr(X=a)`. Therefore strict
        inequality `lower < upper` must hold.
      precision: A positive integer <= 16.

    Returns:
      A new updated stack.
    """
    dtype = jnp.uint32
    lower, upper = jax.tree_map(lambda u: jnp.asarray(u, dtype), bound)
    precision = jnp.asarray(precision, dtype)
    chex.assert_rank([lower, upper, precision], 0)

    size = upper - lower

    # Approximately, if MAX_STATE <= (state / size) * 2**precision:
    stack = lax.cond(
        size <= (self.state >> (LOG2_MAX_STATE - precision)),
        lambda s: s.write(2), lambda s: s, self)

    # TODO(ssjhv): Try pre-computed multipliers to avoid divisions.
    quotient = stack.state // size
    remainder = stack.state % size
    state = (quotient << precision) + (lower + remainder)
    chex.assert_type(state, dtype)
    return stack.replace(state=state)

  def decode(self,
             index: chex.Array,
             cdf: chex.Array,
             precision: chex.Array | int) -> tuple[RangeAnsStack, jax.Array]:
    """Decodes many values in a Jax loop.

    Args:
      index: A vector of indices. When `cdf` is 1-D vector, `index` contents are
        not used (or equivalently they are assumed to be all zeros). Instead,
        its shape is the output shape, i.e., `index.size` is the number of
        elements to be decoded. When `cdf` is 2-D and is a stack of 1-D vectors,
        then `index` indicates what `cdf` row is used to decode each value.
      cdf: When 1-D, this is a regular quantized CDF, see `decode_scalar()`.
        When 2-D, this is a stack of 1-D quantized CDF vectors, and `index`
        argument determines the row to be used during decode.
      precision: A positive integer <= 16.

    Returns:
      The same as `decode_scalar()`, i.e., a new updated stack and the decoded
      values, except that decoded symbols form a vector instead of a scalar.
    """
    cdf = jnp.asarray(cdf)
    chex.assert_rank(cdf, {1, 2})

    if cdf.ndim == 1:
      chex.assert_rank(index, 1)
      return lax.scan(
          lambda s, _: s.decode_scalar(cdf, precision),
          self, (), length=index.shape[-1])
    else:
      return lax.scan(
          lambda s, i: s.decode_scalar(cdf[i], precision),
          self, index)

  def encode(self,
             value: Union[tuple[chex.Array, chex.Array],
                          tuple[chex.Array, chex.Array, chex.Array]],
             precision: chex.Array | int) -> RangeAnsStack:
    """Encodes many values in a Jax loop.

    Note that ANS is a LIFO system, and it encodes and decodes in reverse order.
    This implementation encodes the input in reverse order and the paired decode
    function decodes in regular order.

    `value` argument can be either a tuple of 2 or 3 elements.

    When it has 2 elements, the tuple is interpreted as `bound` argument of
    `encode_single()`, except that the contained arrays are vectors instead of
    scalars.

    When it has 3 elements, the tuple elements are interpreted as:
    - value: Vector of values to be encoded.
    - index: Vector of same shape as `value`. `value[i]` is encoded using
        quantized CDF `cdf[index[i]]`.
    - cdf: A stack of quantized CDFs. See `decode_scalar()` for CDF description.

    Args:
      value: See above.
      precision: A positive integer <= 16.

    Returns:
      A new updated stack.
    """
    value = jax.tree_map(jnp.asarray, value)
    value, _ = jax.tree_util.tree_flatten(value)
    if len(value) == 2:
      def body_fn(stack, value):
        return stack.encode_scalar(value, precision), None

      stack, _ = lax.scan(body_fn, self, value, reverse=True)
      return stack
    else:
      value, index, cdf = value
      def body_fn(stack, value_and_index):
        value, index = value_and_index
        # Verify if performance changes if we move this gather out of loop.
        lower, upper = lax.dynamic_slice(cdf[index], [value], [2])
        return stack.encode_scalar((lower, upper), precision), None

      stack, _ = lax.scan(body_fn, self, (value, index), reverse=True)
      return stack

  @classmethod
  def make_decoder(cls, qmfs: Sequence[np.ndarray]) -> np.ndarray:
    """Makes decoder lookup from quantized pmf arrays."""
    return _range_ans_stack_make_decoder(qmfs)

  @classmethod
  def make_encoder(cls, decoder: np.ndarray) -> np.ndarray:
    """Makes encoder lookup from decoder lookup produced by `make_decoder`."""
    return _range_ans_stack_make_encoder(decoder)

  def pop_scalar(
      self,
      decoder: chex.Array,
      index: chex.Array | int) -> tuple[RangeAnsStack, jax.Array]:
    """Pops one value from rANS stack.

    Decodes a value from the stack with decoder lookup table. The decoder lookup
    table contains multiple distributions and `index` is the distribution index.

    Args:
      decoder: A lookup table. Created by RangeAnsStack.make_decoder().
      index: A non-negative int32 scalar. The distribution index of the value to
        be decoded.

    Returns:
      An updated `RangeAnsStack` and a jnp.int32 scalar. The scalar value is the
      decoded value.
    """
    stack = lax.cond(
        jnp.logical_and(self.stream_loc != 0,
                        self.state < (1 << LOG2_MIN_STATE)),
        lambda s: s.read(2), lambda s: s, self)

    chex.assert_rank(index, 0)
    chex.assert_rank(decoder, 1)
    chex.assert_type(decoder, jnp.uint32)

    header = lax.dynamic_slice(decoder, [4 * index], [4])
    addr = 4 * index + 2 * header[0]

    # header[1] is unused padded space.
    lower_mask = header[2] & 0xFFFF
    upper_mask = header[3] & 0xFFFF
    precision = header[2] >> 16
    shift = header[3] >> 16
    chex.assert_type([lower_mask, upper_mask, precision, shift], jnp.uint32)

    state = stack.state
    major = state & lower_mask
    minor = state & upper_mask
    state >>= precision

    e0, e1 = lax.dynamic_slice(decoder, [addr + 2 * major], [2])
    cutoff = e0 & upper_mask
    value1 = e0 & lower_mask

    is_value1 = (cutoff <= minor)
    state *= jnp.where(is_value1, e1, e0) >> 16
    state += jnp.where(is_value1, e1, 0) & 0xFFFF
    state += minor >> shift
    chex.assert_type(state, jnp.uint32)

    return stack.replace(state=state), jnp.where(is_value1, value1, major)

  def pop(self,
          decoder: chex.Array,
          index: chex.Array) -> tuple[RangeAnsStack, jax.Array]:
    """Pops many values in a Jax loop.

    Args:
      decoder: The same as `decoder` in `pop_scalar()`.
      index: The same as `index` in `pop_scalar()`, except that this is a 1-D
        array instead of a scalar.

    Returns:
      The same as `pop_scalar()`, i.e., a new updated stack and the decoded
      values, except that decoded values form a vector instead of a scalar.
    """
    return lax.scan(lambda s, i: s.pop_scalar(decoder, i), self, index)

  def push_scalar(self,
                  encoder: chex.Array,
                  index: chex.Array | int,
                  value: chex.Array | int) -> RangeAnsStack:
    """Pushes one value into rANS stack.

    Encodes a value into the stack with encoder lookup table. The encoder lookup
    table contains multiple distributions and `index` is the distribution index.

    Args:
      encoder: A lookup table. Created by RangeAnsStack.make_encoder().
      index: A non-negative int32 scalar. The distribution index of the value to
        be decoded.
      value: A non-negative int32 scalar. The maximum allowed value may be
        different depending on `index`.

    Returns:
      An updated `RangeAnsStack`.
    """
    chex.assert_rank(encoder, 1)
    chex.assert_type(encoder, jnp.uint32)
    offset0, precision = lax.dynamic_slice(encoder, [2 * index], [2])
    offset0 += value

    value_lookup = encoder[2 * index + offset0]
    offset1 = value_lookup & 0xFFFF
    size = value_lookup >> 16

    # Approximately, if MAX_STATE <= (state / size) * 2**precision:
    stack = lax.cond(
        size <= (self.state >> (LOG2_MAX_STATE - precision)),
        lambda s: s.write(2), lambda s: s, self)

    quotient = stack.state // size
    remainder = stack.state % size

    offset1 += remainder
    shuffle = encoder[2 * index + offset0 + offset1 // 2]
    # TODO(ssjhv): This depends on endianness.
    remainder = jnp.where(offset1 % 2 == 0, shuffle & 0xFFFF, shuffle >> 16)

    state = (quotient << precision) + remainder
    chex.assert_type(state, jnp.uint32)
    return stack.replace(state=state)

  def push(self,
           encoder: chex.Array,
           index: chex.Array,
           value: chex.Array) -> RangeAnsStack:
    """Pushes many values in a Jax loop.

    Args:
      encoder: The same as `encoder` in `push_scalar()`.
      index: The same as `index` in `push_scalar()`, except that this is a 1-D
        array instead of a scalar.
      value: The same as `value` in `push_scalar()`, except that this is a 1-D
        array instead of a scalar. Must have the same shape as `index`.

    Returns:
      The same as `push_scalar()`, i.e., a new updated stack.
    """
    chex.assert_rank([index, value], 1)
    chex.assert_equal_shape([index, value])

    def body_fn(stack, index_and_value):
      return stack.push_scalar(encoder, *index_and_value), None

    stack, _ = lax.scan(body_fn, self, (index, value), reverse=True)
    return stack


def _range_ans_stack_make_decoder(qmfs: Sequence[np.ndarray]) -> np.ndarray:
  """Makes rANS decoder lookup from quantized pmf arrays."""
  chex.assert_type(qmfs, int)
  chex.assert_rank(qmfs, 1)

  if not qmfs:
    raise ValueError("No quantized pmf in the list.")

  n = len(qmfs)
  qmfs = jax.tree_map(np.asarray, qmfs)

  if any(np.any(qmf < 0) for qmf in qmfs):
    raise ValueError("Quantized pmf contains a negative number.")

  qsum = np.asarray([qmf.sum() for qmf in qmfs])
  precision = np.asarray([int(x).bit_length() - 1 for x in qsum])
  if not np.all((0 < precision) & (precision < 16)):
    raise ValueError(f"Quantized pmf sum is not in range (1, 32768]: {qsum}")
  if not np.all(qsum == (1 << precision)):
    raise ValueError(f"Quantized pmf sum must be a power of 2: {qsum}")

  qsize = np.asarray([qmf.size for qmf in qmfs])
  shift = np.asarray([int(qmf.size).bit_length() - 1 for qmf in qmfs])
  shift += np.where((1 << shift) != qsize, 1, 0)
  assert np.all(qsize <= (1 << shift)) and np.all((1 << shift) < 2 * qsize)
  if not np.all(shift <= precision):
    raise ValueError(f"Invalid pmf.size={qsize}, {precision=}")

  size = 1 << shift
  if np.iinfo(np.int32).max < 2 * n + np.sum(size):
    # Displacement does not fit in int32 dtype.
    raise ValueError("Decoder lookup size is too large")

  buffer = np.zeros(2 * n + np.sum(size), dtype=np.uint64)
  header = buffer[:2 * n].view(np.uint32)

  header[::4] = (np.cumsum(size) - size + 2 * n) - 2 * np.arange(n)

  lower_mask = np.uint32((1 << shift) - 1)
  upper_mask = np.uint32((1 << precision) - 1) ^ lower_mask
  header[2::4] = lower_mask | (np.uint32(precision) << 16)
  header[3::4] = upper_mask | (np.uint32(shift) << 16)

  offset = 2 * n
  for qmf, precision, shift in zip(qmfs, precision, shift):
    size = 1 << shift
    lookup = buffer[offset:offset + size].view(np.uint32)
    offset += size

    lookup0 = lookup[0::2].view()
    lookup1 = lookup[1::2].view()

    assert qmf.sum() == (1 << precision)
    assert shift <= precision
    capacity = 1 << (precision - shift)

    lookup0[:qmf.size] = np.where(qmf != capacity, qmf, np.arange(qmf.size))
    lookup1[:qmf.size] = qmf << 16

    overfull = [i for i, v in enumerate(qmf) if capacity < v]
    underfull = [i for i, v in enumerate(qmf) if v < capacity]
    underfull.extend(range(qmf.size, lookup0.size))

    while underfull:
      i_under = underfull.pop()
      i_over = overfull[-1]

      cutoff = lookup0[i_under]
      lookup0[i_under] = i_over | (cutoff << shift)
      lookup1[i_under] = (lookup0[i_over] - capacity) | (qmf[i_over] << 16)

      lookup0[i_over] -= capacity - cutoff
      if lookup0[i_over] <= capacity: overfull.pop()
      if lookup0[i_over] < capacity:
        underfull.append(i_over)
      elif lookup0[i_over] == capacity:
        lookup0[i_over] = i_over

    assert not overfull
    lookup0[:qmf.size] |= np.uint32(qmf) << 16

  buffer.setflags(write=False)
  return buffer.view(np.uint32)


def _range_ans_stack_make_encoder(decoder: np.ndarray) -> np.ndarray:
  """Makes encoder lookup from decoder lookup."""
  chex.assert_type(decoder, np.uint64)
  chex.assert_rank(decoder, 1)

  decoder_header = decoder[:np.uint32(decoder[0])]
  precision = np.int32((decoder_header[1::2] >> 16) & 0xFFFF)
  shift = np.int32((decoder_header[1::2] >> 48) & 0xFFFF)
  assert np.all(shift <= precision)
  assert np.all(precision < 16)

  size = (1 << shift) + ((1 << precision) + 1) // 2
  if np.iinfo(np.int32).max < np.sum(2 + size):
    raise ValueError("Encoder lookup size is too large")

  buffer = np.zeros(np.sum(2 + size), dtype=np.uint32)
  header = buffer[:decoder_header.size].view()

  offset = np.cumsum(size) - size  # Exclusive scan.
  header[0::2] = offset + header.size - np.arange(0, header.size, 2)
  header[1::2] = precision

  doffset = int(decoder_header.size)
  offset = header.size
  for precision, shift in zip(precision, shift):
    assert shift <= precision < 16
    capacity = 1 << (precision - shift)

    dlookup = decoder[doffset:doffset + (1 << shift)].view(np.int32)
    doffset += 1 << shift

    qmf = (dlookup[::2] >> 16) & 0xFFFF
    assert np.sum(qmf) == (1 << precision)
    cdf = np.cumsum(qmf) - qmf  # Exclusive scan.

    lookup = buffer[offset:offset + (1 << shift)].view(np.int32)
    offset += 1 << shift

    if np.any(np.iinfo(np.uint16).max < (1 << precision) + (2 << shift)):
      raise ValueError("Encoder lookup size is too large")

    lookup[:] = cdf + 2 * (lookup.size - np.arange(lookup.size))
    lookup[:] |= qmf << 16

    size = ((1 << precision) + 1) // 2
    lookup = buffer[offset:offset + size].view(np.uint16)
    if precision == 0: lookup = lookup[:-1]
    offset += size

    cutoff = (dlookup[::2] & 0xFFFF) >> shift
    assert np.all(0 <= cutoff) and np.all(cutoff < capacity)

    value1 = dlookup[::2] & ((1 << shift) - 1)  # Lower shift bits.
    offset1 = dlookup[1::2] & 0xFFFF

    # Transposing lhs/rhs below may have better consecutive pieces.
    lhs = np.expand_dims(np.arange(capacity), -1)
    lhs = (lhs + np.where(lhs < cutoff, cdf, cdf[value1] + offset1)).ravel()
    rhs = np.arange(1 << shift)
    rhs = (rhs + np.expand_dims(np.arange(capacity) << shift, -1)).ravel()
    lookup[:] = rhs[np.argsort(lhs)]

    # Check if the lookup table is bijective.
    np.testing.assert_array_equal(lookup[0], 0)
    np.testing.assert_array_equal(
        np.sort(lookup), np.arange(lookup.size, dtype=lookup.dtype))

  buffer.setflags(write=False)
  return buffer.view(np.uint32)


def quantize_distribution(pmf: chex.Array, normalizer: int) -> chex.Array:
  """Quantizes pmf for entropy coding.

  Jax does not have a way to return error status, in particular when jit
  compiled. Therefore in case of error, instead of raising exceptions, the
  returned array shall be filled with negative values.
  (We may consider using `jax.experimental.checkify` in the future.)

  Let `qmf` denote the returned quantized array.

  - For index i if pmf[i] > 0, then qmf[i] > 0.
  - For index i if pmf[i] <= 0, then qmf[i] == 0. This is different from C++
    version, that negative values (including -inf) are treated as zero here.

  This implies:

  - If `pmf[i] > 0`, even it is a tiny value, `qmf[i]` is at least one.
    When there are many such indices, the quantized distribution may represent
    somewhat different distribution from `pmf`.

  - If `pmf[i] <= 0`, including exact zero, then `qmf[i] == 0`. This means,
    the character `i` cannot be entropy encoded and decoded.

  Args:
    pmf: 1-D float array of probability mass function. See above.
    normalizer: A positive integer. The returned array has the sum equal to
      this value.

  Returns:
    A 1-D int32 array of the same shape as `pmf`.
  """
  pmf = jnp.asarray(pmf)
  chex.assert_type(pmf, float)
  chex.assert_rank(pmf, 1)
  chex.assert_scalar_in(normalizer, 0, 2 ** 20, included=False)

  if pmf.size == 0: return -jnp.ones_like(pmf, dtype=jnp.int32)

  pmf = pmf.clip(min=0)
  psum = pmf.sum()
  nonzero = (pmf > 0)  # Find locations for qmf != 0.

  # The last error case is when the precision value is too small. In that case,
  # the lower bound on the quantized pmf sum exceeds the normalizer value.
  no_error = jnp.isfinite(psum)  # `pmf` and its sum should be finite.
  no_error &= 0 < nonzero.sum()  # At least one entry should be nonzero.
  no_error &= nonzero.sum() <= normalizer  # Not too many nonzeros.

  qmf = jnp.round(normalizer * (pmf / psum)).astype(jnp.int32)
  qmf = qmf.clip(min=1) * nonzero

  # Maybe helpful to repeat the step above for extremely skewed distributions.

  branch_index = (qmf.sum() < normalizer).astype(jnp.int32)
  increment_branch = functools.partial(_increment_branch, normalizer=normalizer)
  decrement_branch = functools.partial(_decrement_branch, normalizer=normalizer)

  return lax.switch(
      jnp.where(no_error, branch_index, 2),
      # Return array of -1s in case of error.
      [decrement_branch, increment_branch, lambda *_: jnp.full_like(qmf, -1)],
      pmf, qmf,
  )


# REQUIRES: `qmf.sum() <= normalizer`.
def _increment_branch(pmf, qmf, normalizer: int):
  """Iteratively increments qmf for qmf.sum() to reach normalizer."""
  diff = pmf * jnp.log(qmf / (qmf + 1))
  diff = jnp.where(qmf > 0, diff, jnp.inf)

  def body_fn(_, carry):
    diff, qmf = carry
    pos = jnp.argmin(diff)
    mass = qmf[pos]  # assert mass != 0
    new_diff = pmf[pos] * jnp.log((mass + 1) / (mass + 2))
    return diff.at[pos].set(new_diff), qmf.at[pos].set(mass + 1)

  _, qmf = lax.fori_loop(qmf.sum(), normalizer, body_fn, (diff, qmf))
  return qmf


# REQUIRES: `normalizer <= qmf.sum()`.
def _decrement_branch(pmf, qmf, normalizer: int):
  """Iteratively decrements qmf for qmf.sum() to reach normalizer."""
  diff = pmf * jnp.log(qmf / (qmf - 1))
  diff = jnp.where(qmf > 1, diff, jnp.inf)

  def body_fn(_, carry):
    diff, qmf = carry
    pos = jnp.argmin(diff)
    mass = qmf[pos]  # assert mass > 1
    new_diff = pmf[pos] * jnp.log((mass - 1) / (mass - 2))
    new_diff = jnp.where(mass > 2, new_diff, jnp.inf)
    return diff.at[pos].set(new_diff), qmf.at[pos].set(mass - 1)

  _, qmf = lax.fori_loop(normalizer, qmf.sum(), body_fn, (diff, qmf))
  return qmf
