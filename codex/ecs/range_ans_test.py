# Copyright 2023 CoDeX authors.
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
"""Tests for rANS in Numpy/Jax."""

import collections
import functools
import math

from absl import logging
import chex
from codex.cc import range_ans_pybind
from codex.ecs import range_ans
import jax
import jax.numpy as jnp
import numpy as np
import pytest


rng = np.random.default_rng()


def _make_decoder_encoder(qmf):
  decoder = range_ans.RangeAnsStack.make_decoder(list(qmf))
  encoder = range_ans.RangeAnsStack.make_encoder(decoder.view("u8"))

  cpp_decoder = range_ans_pybind.RangeAnsStack.make_decoder(list(qmf))
  cpp_encoder = range_ans_pybind.RangeAnsStack.make_encoder(cpp_decoder)

  # TODO(ssjhv): This works only on little-endian machines.
  cpp_decoder = cpp_decoder.view("u4")

  # Fix difference between C++ and Python versions.
  sentinel = np.asarray([q.size - 1 for q in qmf], dtype=np.uint32)
  decoder_for_test = decoder.copy()
  encoder_for_test = encoder.copy()
  decoder_for_test[:4 * sentinel.size][1::4] = sentinel
  encoder_for_test[:2 * sentinel.size][1::2] |= (sentinel << 16)

  np.testing.assert_array_equal(decoder_for_test, cpp_decoder)
  np.testing.assert_array_equal(encoder_for_test, cpp_encoder)

  return jnp.asarray(decoder), jnp.asarray(encoder)


def _precision_and_alphabet_size(alphabet_size):
  precision = int(rng.integers(3, 15, endpoint=True))
  if not alphabet_size:
    alphabet_size = rng.integers(3, min(1 << precision, 4096), endpoint=True)
  return precision, int(alphabet_size)


def _make_random_quantized_cdf_1d(n, precision):
  arr = np.zeros(n + 1, dtype=np.int32)
  arr[n] = 1 << precision

  queue = collections.deque([(0, n + 1)])
  while queue:
    begin, end = queue.popleft()
    if end - begin <= 2:
      continue

    lower = arr[begin]
    upper = arr[end - 1]
    mid = (begin + end) // 2
    assert begin < mid < end - 1
    margin = (mid - begin, end - 2 - mid)

    arr[mid] = rng.integers(lower + margin[0], upper - margin[1])
    queue.append((begin, mid + 1))
    queue.append((mid, end))

  assert np.all(arr[:-1] < arr[1:]), arr
  return arr


def _make_random_quantized_cdf(shape, precision):
  *shape, n = shape
  arr = np.stack([_make_random_quantized_cdf_1d(n, precision)
                  for _ in range(math.prod(shape))])
  return arr.reshape(*shape, n + 1)


def _sample_data(cdf, shape):
  if cdf.ndim == 1:
    p = np.asarray(cdf[1:] - cdf[:-1], dtype=float) / cdf[-1]
    return rng.choice(p.size, shape, p=p)

  assert cdf.shape[0] == shape[0], (cdf.shape, shape)
  return np.stack([_sample_data(row, shape[1:]) for row in cdf])


# Tests for range_ans.RangeAnsStack.


def _check_encoded_size(encoded, cdf, x, index=None):
  _, stream_size = encoded

  # Un-quantize PMF.
  p = np.asarray(cdf[..., 1:] - cdf[..., :-1], dtype=float) / cdf[..., -1:]
  if index is None:
    optimal_size = np.take_along_axis(-np.log2(p), x, axis=-1)
  else:
    optimal_size = -np.log2(p)[(index, x)]
  optimal_size = np.sum(optimal_size, axis=-1) / 8.0

  # Due to the last state spill, the stream size cannot be less than 4.
  optimal_size = np.maximum(optimal_size, 4.0)
  ratio = stream_size / optimal_size

  for optimal_size, stream_size, ratio in zip(
      *np.atleast_1d(optimal_size, stream_size, ratio)
  ):
    logging.info(
        "Theoretical bound=%f, stream size=%d, ratio=%f",
        optimal_size,
        stream_size,
        ratio,
    )

  # The actual stream size has some skew because it has to spill the 4-byte
  # state at the end. When the whole information entropy is too small, this
  # bias can skew the result. So skip when the optimal size is small.
  np.testing.assert_array_less(
      stream_size * (15.0 < optimal_size), 1 + np.ceil(1.1 * optimal_size)
  )


def _check_encoded(encoded):
  stream, stream_size = encoded
  np.testing.assert_array_less(stream_size, 1 + stream.shape[-1])


def _check_decoded(stack):
  np.testing.assert_array_equal(stack.state, 0)
  np.testing.assert_array_equal(stack.stream_loc, 0)


@pytest.mark.parametrize("m,n", [(0, 200), (2, 1200)])
@pytest.mark.parametrize("batch", [1, 4])
def test_encode_and_decode(m, n, batch):
  batch = () if batch == 1 else (batch,)
  precision, m = _precision_and_alphabet_size(m)

  cdf = _make_random_quantized_cdf((*batch, m), precision)
  x = _sample_data(cdf, (*batch, n))

  maybe_vmap = jax.vmap if batch else lambda f: f

  @jax.jit
  def encode_fn(lower, upper):
    buffer = jnp.empty(n * 4, dtype=jnp.uint8)
    stack = range_ans.RangeAnsStack.make(buffer)
    stack = stack.encode((lower, upper), precision)
    return stack.serialize()

  lower = np.take_along_axis(cdf, x + 0, axis=-1)
  upper = np.take_along_axis(cdf, x + 1, axis=-1)
  encoded = maybe_vmap(encode_fn)(lower, upper)
  _check_encoded(encoded)

  @jax.jit
  def decode_fn(encoded, cdf):
    stack = range_ans.RangeAnsStack.deserialize(*encoded)
    empty = jnp.empty(n, dtype=jnp.int32)
    return stack.decode(empty, cdf, precision)

  stack, decoded = maybe_vmap(decode_fn)(encoded, cdf)
  _check_decoded(stack)
  np.testing.assert_array_equal(x, decoded)

  _check_encoded_size(encoded, cdf, x)


@pytest.mark.parametrize("m,n,k", [(0, 200, 64), (2, 1200, 1)])
@pytest.mark.parametrize("batch", [1, 4])
def test_encode_and_decode_index(m, n, k, batch):
  batch = () if batch == 1 else (batch,)
  precision, m = _precision_and_alphabet_size(m)

  cdf = _make_random_quantized_cdf((k, m), precision)
  index = rng.integers(k, size=(*batch, n))
  x = np.stack([_sample_data(row, (*batch, n)) for row in cdf])
  x = np.take_along_axis(x, index[None, ...], axis=0).squeeze(axis=0)

  maybe_vmap = jax.vmap if batch else lambda f: f

  @jax.jit
  def encode_fn(x, index):
    buffer = jnp.empty(n * 4, dtype=jnp.uint8)
    stack = range_ans.RangeAnsStack.make(buffer)
    stack = stack.encode((x, index, cdf), precision)
    return stack.serialize()

  encoded = maybe_vmap(encode_fn)(x, index)
  _check_encoded(encoded)

  @jax.jit
  def decode_fn(encoded, index):
    stack = range_ans.RangeAnsStack.deserialize(*encoded)
    return stack.decode(index, cdf, precision)

  stack, decoded = maybe_vmap(decode_fn)(encoded, index)
  _check_decoded(stack)
  np.testing.assert_array_equal(x, decoded)

  _check_encoded_size(encoded, cdf, x, index=index)


@pytest.mark.parametrize("m,n", [(0, 31), (2, 735)])
def test_encode_and_decode_python_loop(m, n):
  precision, m = _precision_and_alphabet_size(m)
  cdf = _make_random_quantized_cdf((m,), precision)
  x = _sample_data(cdf, (n,))

  encode_single_fn = jax.jit(
      functools.partial(
          range_ans.RangeAnsStack.encode_scalar, precision=precision
      )
  )
  decode_single_fn = jax.jit(
      functools.partial(
          range_ans.RangeAnsStack.decode_scalar, precision=precision
      )
  )

  lower = np.take_along_axis(cdf, x + 0, axis=-1)
  upper = np.take_along_axis(cdf, x + 1, axis=-1)

  buffer = jnp.empty(n * 4, dtype=jnp.uint8)
  stack = range_ans.RangeAnsStack.make(buffer)
  for i in reversed(range(n)):
    stack = encode_single_fn(stack, (lower[i], upper[i]))
  encoded = stack.serialize()
  _check_encoded(encoded)

  stack = range_ans.RangeAnsStack.deserialize(*encoded)
  cdf = jnp.asarray(cdf)

  decoded = np.full_like(x, -1)
  for i in range(n):
    stack, decoded[i] = decode_single_fn(stack, cdf)

  _check_decoded(stack)
  np.testing.assert_array_equal(x, decoded)

  _check_encoded_size(encoded, cdf, x)


@pytest.mark.parametrize("m,n,k", [(0, 200, 64), (2, 1200, 1)])
@pytest.mark.parametrize("batch", [1, 4])
def test_stack_encode_and_decode_index(m, n, k, batch):
  batch = () if batch == 1 else (batch,)
  precision, m = _precision_and_alphabet_size(m)

  cdf = _make_random_quantized_cdf((k, m), precision)
  index = rng.integers(k, size=(*batch, n))
  x = np.stack([_sample_data(row, (*batch, n)) for row in cdf])
  x = np.take_along_axis(x, index[None, ...], axis=0).squeeze(axis=0)

  qmf = cdf[..., 1:] - cdf[..., :-1]
  decoder, encoder = _make_decoder_encoder(qmf)

  maybe_vmap = jax.vmap if batch else lambda f: f

  @jax.jit
  def encode_fn(index, x):
    buffer = jnp.empty(n * 4, dtype=jnp.uint8)
    stack = range_ans.RangeAnsStack.make(buffer)
    stack = stack.push(encoder, index, x)
    return stack.serialize()

  encoded = maybe_vmap(encode_fn)(index, x)
  _check_encoded(encoded)

  @jax.jit
  def decode_fn(encoded, index):
    stack = range_ans.RangeAnsStack.deserialize(*encoded)
    return stack.pop(decoder, index)

  stack, decoded = maybe_vmap(decode_fn)(encoded, index)
  _check_decoded(stack)
  np.testing.assert_array_equal(x, decoded)

  _check_encoded_size(encoded, cdf, x, index=index)


def test_range_ans_stack_error_condition():
  with pytest.raises(ValueError):
    _ = range_ans.RangeAnsStack.make_decoder([])

  with pytest.raises(ValueError):
    _ = range_ans.RangeAnsStack.make_decoder([np.asarray([0])])

  with pytest.raises(ValueError):
    _ = range_ans.RangeAnsStack.make_decoder([np.asarray([1])])


# Tests for range_ans.quantize_distribution().


def _test_output(pmf, qmf, normalizer: int, threshold: float):
  chex.assert_equal_shape([pmf, qmf])

  assert qmf.sum() == normalizer

  np.testing.assert_array_equal(qmf[pmf == 0], 0)
  np.testing.assert_array_less(0, qmf[pmf != 0])

  pmf /= pmf.sum()
  h = -(pmf * np.maximum(np.log2(pmf), -100)).sum()
  qh = np.log2(normalizer) - (pmf * np.maximum(np.log2(qmf), -100)).sum()

  logging.info("Expected code length using pmf: %f", h)
  logging.info("Expected code length using quantized pmf: %f", qh)
  logging.info("Relative difference: %f", (h - qh) / h)
  assert qh > h * (1 - np.finfo(h.dtype).eps)
  assert (qh - h) / h < threshold


def test_under_sum():
  normalizer = 100

  pmf = np.r_[[0.013] * 49, 0, 0]
  pmf[-1] = 1.0 - pmf.sum()
  np.testing.assert_almost_equal(pmf.sum(), 1.0)

  qmf = np.round(pmf * normalizer).astype(int)
  assert qmf.sum() < normalizer

  qmf = range_ans.quantize_distribution(pmf, normalizer)
  _test_output(pmf, qmf, normalizer, threshold=0.03)


def test_over_sum():
  normalizer = 100

  pmf = np.r_[[0.017] * 49, 0, 0]
  pmf[-1] = 1.0 - pmf.sum()
  np.testing.assert_almost_equal(pmf.sum(), 1.0)

  qmf = np.round(pmf * normalizer).astype(int)
  assert qmf.sum() > normalizer

  qmf = range_ans.quantize_distribution(pmf, normalizer)
  _test_output(pmf, qmf, normalizer, threshold=0.03)


@pytest.mark.parametrize("jit", [True, False])
def test_random_normalizer(jit):
  normalizer = int(rng.integers(100, 600))
  size = rng.integers(40, 70)

  fn = lambda pmf: range_ans.quantize_distribution(pmf, normalizer)
  if jit:
    fn = jax.jit(fn)

  pmf = rng.random((size,))
  # Make sure some values are nonzeros.
  pmf[:10] += np.finfo(np.float32).eps

  qmf = fn(pmf)
  _test_output(pmf, qmf, normalizer, threshold=0.03)


def test_random_normalizer_with_vmap():
  normalizer = int(rng.integers(100, 600))
  size = rng.integers(40, 70)

  fn = lambda pmf: range_ans.quantize_distribution(pmf, normalizer)

  pmf = rng.random((5, size))
  # Produce some zeros.
  pmf *= rng.random(pmf.shape) < 0.9
  # Make sure no row is entirely zero.
  pmf[:, :10] += np.finfo(np.float32).eps

  qmf = jax.vmap(fn)(pmf)
  chex.assert_equal_shape([pmf, qmf])
  for pmf_row, qmf_row in zip(pmf, qmf):
    _test_output(pmf_row, qmf_row, normalizer, threshold=0.03)


def test_random_power_of_2_normalizer():
  normalizer = int(1 << rng.integers(8, 20))
  size = rng.integers(40, 70)

  pmf = rng.random((size,))
  pmf[:10] += np.finfo(np.float32).eps
  qmf = range_ans.quantize_distribution(pmf, normalizer)
  _test_output(pmf, qmf, normalizer, threshold=0.03)


def test_many_small_values():
  normalizer = int(rng.integers(100, 600))
  size = rng.integers(40, 70)

  eps = np.finfo(np.float32).eps
  pmf = rng.random((size,)).clip(min=eps)
  # The second half of pmf are replaced with epsilons.
  pmf[size // 2 :] = eps

  # The following procedure makes pmf.sum() ~= 1.
  scale = (1.0 - ((size + 1) // 2) * eps) / pmf[: size // 2].sum()
  pmf[: size // 2] *= scale

  qmf = range_ans.quantize_distribution(pmf, normalizer)
  _test_output(pmf, qmf, normalizer, threshold=0.3)  # Higher threshold.


def test_many_zeros():
  normalizer = int(rng.integers(100, 600))
  size = rng.integers(40, 70)

  pmf = rng.random((size,)).clip(min=np.finfo(np.float32).eps)
  # The second half of pmf are replaced with zeros.
  pmf[size // 2 :] = 0

  qmf = range_ans.quantize_distribution(pmf, normalizer)
  _test_output(pmf, qmf, normalizer, threshold=0.03)


def test_quantize_distribution_error_condition():
  # `normalizer` is smaller than pmf size.
  qmf = range_ans.quantize_distribution(np.ones(20), 10)
  np.testing.assert_array_equal(qmf, -1)

  # `pmf` is all zero.
  qmf = range_ans.quantize_distribution(np.zeros(8), 100)
  np.testing.assert_array_equal(qmf, -1)

  # vmap above.
  pmf = rng.random((2, 8)).clip(min=np.finfo(np.float32).eps)
  pmf[0] = 0  # Make the first row all zero.
  qmf = jax.vmap(lambda u: range_ans.quantize_distribution(u, 100))(pmf)
  np.testing.assert_array_equal(qmf[0], -1)
  _test_output(pmf[1], qmf[1], 100, threshold=0.03)

  # Zero size.
  qmf = range_ans.quantize_distribution(np.zeros(0), 100)
  chex.assert_shape(qmf, (0,))

  # Negative `pmf` considered as zero prob.
  qmf = range_ans.quantize_distribution(np.r_[1.0, 1.0, -1.0], 100)
  _test_output(np.r_[1.0, 1.0, 0.0], qmf, 100, threshold=0.03)

  # Infinity.
  qmf = range_ans.quantize_distribution(np.r_[1.0, 1.0, np.inf], 100)
  np.testing.assert_array_equal(qmf, -1)

  # Negative infinity considered as zero prob.
  qmf = range_ans.quantize_distribution(np.r_[1.0, 1.0, -np.inf], 100)
  _test_output(np.r_[1.0, 1.0, 0.0], qmf, 100, threshold=0.03)

  # Nan.
  qmf = range_ans.quantize_distribution(np.r_[1.0, 1.0, np.nan], 100)
  np.testing.assert_array_equal(qmf, -1)
