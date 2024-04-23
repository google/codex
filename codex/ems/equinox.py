# Copyright 2024 CoDeX authors.
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
"""Entropy models as Equinox modules."""

from typing import List, Tuple
from codex.ems import deep_factorized as _deep_factorized
from codex.ems import fourier as _fourier
import equinox as _eqx
import jax as _jax

# pylint:disable=unused-import,g-importing-member,g-bad-import-order
from codex.ems.continuous import ContinuousEntropyModel
from codex.ems.distribution import DistributionEntropyModel
from codex.ems.distribution import scale_param
# pylint:enable=unused-import,g-importing-member,g-bad-import-order

Array = _jax.Array


class MonotonicMLP(_deep_factorized.MonotonicMLPBase, _eqx.Module):
  """MLP that implements monotonically increasing functions by construction."""
  matrices: List[Array]
  biases: List[Array]
  factors: List[Array]

  def __init__(self,
               rng,
               num_mlps: int,
               num_units: Tuple[int, ...],
               init_scale: float):
    """Initializes the MLP.

    Args:
      rng: Random number generator key for initialization.
      num_mlps: Integer. Number of independent MLPs.
      num_units: Iterable of integers. The number of filters for each of the
        hidden layers. The first and last layer of the network implementing the
        cumulative distribution are not included (they are assumed to be 1).
      init_scale: Float. Scale factor for the density at initialization. It is
        recommended to choose a large enough scale factor such that most values
        initially lie within a region of high likelihood. This improves
        training.
    """
    scale = init_scale ** (1 / (1 + len(num_units)))
    num_units = (1,) + num_units + (1,)
    self.matrices = []
    self.biases = []
    self.factors = []
    for k, shape in enumerate(zip(num_units[1:], num_units[:-1])):
      # shape == (out_dims, in_dims)
      shape = (num_mlps,) + shape
      self.matrices.append(_deep_factorized.matrix_init(shape, scale))
      self.biases.append(_jax.random.uniform(
          rng, shape[:2], minval=-.5, maxval=.5))
      if k < len(num_units) - 2:
        self.factors.append(_jax.numpy.zeros(shape[:2]))


class DeepFactorizedEntropyModel(
    _deep_factorized.DeepFactorizedEntropyModelBase,
    _eqx.Module):
  r"""Fully factorized entropy model based on neural network cumulative.

  This is a flexible, nonparametric entropy model, described in appendix 6.1 of
  the paper:

  > "Variational image compression with a scale hyperprior"<br />
  > J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston<br />
  > https://openreview.net/forum?id=rkcQFMZRb

  convolved with a unit-width uniform density, as described in appendix 6.2 of
  the same paper. Please cite the paper if you use this code for scientific
  work.

  This model learns a factorized distribution. For example, if the input has
  64 channels (the last dimension size), then the distribution has the form

     CDF(x) = \prod_{i=1}^64 CDF_i(x_i)

  where each function CDF_i is modeled by MLPs of different parameters.
  """
  cdf_logits: _eqx.Module

  def __init__(self,
               rng,
               num_pdfs: int,
               num_units: Tuple[int, ...] = (3, 3, 3),
               init_scale: float = 10.):
    """Initializes the model.

    Args:
      rng: Random number generator key for initialization.
      num_pdfs: Integer. The number of distinct scalar PDFs on the right of the
        input array. These are treated as independent, but non-identically
        distributed. The remaining array elements on the left are treated as
        i.i.d. (like in a batch dimension).
      num_units: Iterable of integers. The number of filters for each of the
        hidden layers. The first and last layer of the network implementing the
        cumulative distribution are not included (they are assumed to be 1).
      init_scale: Float. Scale factor for the density at initialization. It is
        recommended to choose a large enough scale factor such that most values
        initially lie within a region of high likelihood. This improves
        training.
    """
    self.cdf_logits = MonotonicMLP(rng, num_pdfs, num_units, init_scale)


class PeriodicFourierEntropyModel(_fourier.PeriodicFourierEntropyModelBase,
                                  _eqx.Module):
  """Fourier basis entropy model for periodic distributions."""
  period: float
  real: _jax.Array
  imag: _jax.Array

  def __init__(self,
               rng,
               period: float,
               num_pdfs: int,
               num_freqs: int = 10,
               init_scale: float = 1e-3):
    """Initializes the entropy model.

    Args:
      rng: Random number generator key for initialization.
      period: Float. Length of interval on `x` over which entropy model is
        periodic.
      num_pdfs: Integer. The number of distinct scalar PDFs on the right of the
        input array. These are treated as independent, but non-identically
        distributed. The remaining array elements on the left are treated as
        i.i.d. (like in a batch dimension).
      num_freqs: Integer. Number of frequency components of the Fourier series.
      init_scale: Float. Scale of normal distribution for random initialization
        of coefficients.
    """
    super().__init__()
    self.period = period
    self.real, self.imag = init_scale * _jax.random.normal(
        rng, (2, num_pdfs, num_freqs))


class RealMappedFourierEntropyModel(_fourier.RealMappedFourierEntropyModelBase,
                                    _eqx.Module):
  """Fourier basis entropy model mapped to the real line."""
  scale: _jax.Array
  offset: _jax.Array
  real: _jax.Array
  imag: _jax.Array

  def __init__(self,
               rng,
               num_pdfs: int,
               num_freqs: int = 10,
               init_scale: float = 1e-3):
    """Initializes the entropy model.

    Args:
      rng: Random number generator key for initialization.
      num_pdfs: Integer. The number of distinct scalar PDFs on the right of the
        input array. These are treated as independent, but non-identically
        distributed. The remaining array elements on the left are treated as
        i.i.d. (like in a batch dimension).
      num_freqs: Integer. Number of frequency components of the Fourier series.
      init_scale: Float. Scale of normal distribution for random initialization
        of coefficients.
    """
    super().__init__()
    self.real, self.imag = init_scale * _jax.random.normal(
        rng, (2, num_pdfs, num_freqs))
    self.scale = _jax.numpy.ones((num_pdfs,))
    self.offset = _jax.numpy.zeros((num_pdfs,))
