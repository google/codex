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


@pytest.mark.parametrize("n_distribution", [1, 2, 128])
def test_encode_decode(n_distribution):
  index = rng.integers(0, n_distribution, (1024,), dtype=np.int32)
  data = -np.ones((1024,), dtype=np.int32)

  qmfs = []
  for i in range(n_distribution):
    precision = rng.integers(9, 14, endpoint=True)
    alphabet_size = rng.integers(
        257, min(2 ** precision, 4096), endpoint=True)

    data = np.where(
        index != i, data,
        rng.integers(0, alphabet_size, data.shape, dtype=np.int32))

    qmfs.append(_make_random_qmf(alphabet_size, precision))

  np.testing.assert_array_less(-1, data)

  decoder = range_ans_pybind.RangeAnsStack.make_decoder(qmfs)
  encoder = range_ans_pybind.RangeAnsStack.make_encoder(decoder)

  seed = rng.integers(2 ** 20)

  stack = range_ans_pybind.RangeAnsStack(seed, reserve=2048)
  stack.push(encoder, index, data)
  encoded = stack.serialize()
  assert encoded, len(encoded)

  stack = range_ans_pybind.RangeAnsStack.deserialize(encoded)
  decoded = stack.pop(decoder, index)

  np.testing.assert_array_equal(decoded, data)
  assert stack.state == seed


def test_error_deserialize_short_stream():
  with pytest.raises(range_ans_pybind.AnsError) as cm:
    _ = range_ans_pybind.RangeAnsStack.deserialize(b"")

  assert "INVALID_ARGUMENT" in str(cm.value)
  assert "rANS stack is too short" in str(cm.value)

