"""Tests for python interface."""

from codex.cc import range_ans_pybind
import numpy as np
import pytest

rng = np.random.default_rng()


def _make_random_qmf(size, precision):
  pmf = rng.random((size,), dtype=np.float32).clip(min=1e-3)
  pmf /= np.maximum(pmf.sum(), 1e-10)
  qmf = np.bincount(rng.choice(size, 2 ** precision - size, p=pmf),
                    minlength=size)
  assert qmf.shape == (size,)
  qmf = np.asarray(qmf + 1, np.int32)
  assert qmf.sum() == 2 ** precision
  return qmf


@pytest.mark.parametrize("n_distribution", [1, 2, 5, 128])
@pytest.mark.parametrize("with_fallback", [False, True])
def test_encode_decode(n_distribution, with_fallback):
  n_distribution = rng.integers(1, n_distribution, endpoint=True)
  index = rng.integers(0, n_distribution, (1024,), dtype=np.int32)
  data = -np.ones((1024,), dtype=np.int32)

  qmfs = []
  for i in range(n_distribution):
    precision = rng.integers(9, 14, endpoint=True)
    alphabet_size = rng.integers(
        257, min(2 ** precision, 4096), endpoint=True)

    qmfs.append(_make_random_qmf(alphabet_size, precision))

    data = np.where(
        index != i, data,
        rng.integers(0, alphabet_size, data.shape, dtype=np.int32))

  decoder = range_ans_pybind.RangeAnsStack.make_decoder(qmfs)
  encoder = range_ans_pybind.RangeAnsStack.make_encoder(decoder)

  np.testing.assert_array_less(-1, data)
  if with_fallback:
    # Replace some values with out-of-range ones to invoke fallbacks.
    data[rng.choice(data.size, 50)] = -rng.integers(1, 2**31, size=50)
    data[rng.choice(data.size, 50)] = rng.integers(2**31, size=50)
    assert np.any(data < 0)

    # Test the smallest possible value.
    # NOTE: -2**31 is not encodeable.
    data[rng.choice(data.size, 2)] = -2**31 + 1

  seed = rng.integers(2 ** 20)

  stack = range_ans_pybind.RangeAnsStack(seed, reserve=2048)
  if with_fallback:
    stack.push_with_fallback(encoder, index, data)
  else:
    stack.push(encoder, index, data)
  encoded = stack.serialize()
  assert encoded, len(encoded)

  stack = range_ans_pybind.RangeAnsStack.deserialize(encoded)
  if with_fallback:
    decoded = stack.pop_with_fallback(decoder, index)
  else:
    decoded = stack.pop(decoder, index)

  np.testing.assert_array_equal(decoded, data)
  assert stack.state == seed


@pytest.mark.parametrize("with_fallback", [False, True])
def test_error_index_and_value_mismatch(with_fallback):
  qmf = _make_random_qmf(32, 7)
  decoder = range_ans_pybind.RangeAnsStack.make_decoder([qmf])
  encoder = range_ans_pybind.RangeAnsStack.make_encoder(decoder)

  stack = range_ans_pybind.RangeAnsStack()
  # Mismatch in index and value size.
  with pytest.raises(range_ans_pybind.AnsError) as cm:
    if with_fallback:
      stack.push_with_fallback(encoder,
                               np.zeros(3, dtype=np.int32),
                               np.zeros(2, dtype=np.int32))
    else:
      stack.push(encoder,
                 np.zeros(3, dtype=np.int32),
                 np.zeros(2, dtype=np.int32))
  assert "different sizes" in str(cm.value).casefold()


def test_error_deserialize_short_stream():
  with pytest.raises(range_ans_pybind.AnsError) as cm:
    _ = range_ans_pybind.RangeAnsStack.deserialize(b"")

  assert "INVALID_ARGUMENT" in str(cm.value)
  assert "rANS stack is too short" in str(cm.value)


def test_error_when_qmf_list_is_empty():
  with pytest.raises(range_ans_pybind.AnsError) as cm:
    _ = range_ans_pybind.RangeAnsStack.make_decoder([])
  assert "no pmf" in str(cm.value).casefold()


def test_error_when_contains_zerolen_qmf():
  _ = range_ans_pybind.RangeAnsStack.make_decoder([[4, 4]])  # This is ok.
  with pytest.raises(range_ans_pybind.AnsError) as cm:
    _ = range_ans_pybind.RangeAnsStack.make_decoder([[4, 4], []])
  assert "contains an empty pmf" in str(cm.value).casefold()


# Decoder jump table overflow.
def test_error_too_long_qmf():
  too_long_qmf = np.zeros(2**20, dtype=np.int32)
  too_long_qmf[:2] = [4, 4]
  with pytest.raises(range_ans_pybind.AnsError) as cm:
    _ = range_ans_pybind.RangeAnsStack.make_decoder([too_long_qmf] * (2**14))
  assert "decoder lookup is too large" in str(cm.value).casefold()


# Encoder jump table overflow.
def test_error_too_many_qmf():
  qmf15 = np.asarray([2**14, 2**14], dtype=np.int32)  # Precision: 15
  decoder = range_ans_pybind.RangeAnsStack.make_decoder([qmf15] * (2**17))
  with pytest.raises(range_ans_pybind.AnsError) as cm:
    _ = range_ans_pybind.RangeAnsStack.make_encoder(decoder)
  assert "encoder lookup is too large" in str(cm.value).casefold()


# Invalid last value in the decoder.
def test_error_invalid_last_value():
  qmf = np.asarray([4, 4], dtype=np.int32)  # Precision: 15
  decoder = range_ans_pybind.RangeAnsStack.make_decoder([qmf]).copy()
  assert int(decoder[0]) >> 32 == 1
  decoder[0] += 2 ** (32 + 17)
  with pytest.raises(range_ans_pybind.AnsError) as cm:
    _ = range_ans_pybind.RangeAnsStack.make_encoder(decoder)
  assert "INTERNAL" in str(cm.value)
