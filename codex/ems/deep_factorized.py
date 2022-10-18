# Copyright 2022 CoDeX authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Deep fully factorized entropy model based on cumulative."""

from typing import Tuple
from codex.ems import continuous
from codex.ops import rounding
import flax.linen as nn
import jax
import jax.numpy as jnp


def _matrix_init(key, shape, scale):
  """Initializes matrix with constant value according to init_scale.

  This initialization function was designed to avoid the early training
  stall due to too large or too small probability mass values computed
  with the initial parameters.

  Softplus is applied to the matrix during the forward pass, to ensure the
  matrix elements are all positive. With the current initialization, the
  matrix has constant values and after matrix-vector multiplication with
  the input, the output vector elements are constant

  ```
  x / init_scale ** (1 / 1 + len(self.features)).
  ```

  Assuming zero bias initialization and ignoring the factor residual
  paths, the CDF output of the entire distribution network is constant
  vector with value

  ```
  sigmoid(x / init_scale).
  ```

  Therefore `init_scale` should be in the order of the expected magnitude
  of `x`.

  Args:
    key: The Jax PRNG key for this matrix initialization.
    shape: Sequence of integers. The shape of the matrix.
    scale: Scale factor for initial value.

  Returns:
    The initial matrix value.
  """
  del key  # Unused.
  return jnp.full(shape, jnp.log(jnp.expm1(1 / scale / shape[-1])))


def _bias_init(key, shape):
  return jax.random.uniform(key, shape, minval=-.5, maxval=.5)


def _factor_init(key, shape):
  return jax.nn.initializers.zeros(key, shape)


class MonotonicMLP(nn.Module):
  """MLP that implements monotonically increasing functions by construction."""
  features: Tuple[int, ...]
  init_scale: float

  @nn.compact
  def __call__(self, x):
    scale = self.init_scale ** (1 / (1 + len(self.features)))
    u = x.reshape((-1, 1))
    features = (1,) + self.features + (1,)
    for k, shape in enumerate(zip(features[:-1], features[1:])):
      matrix = self.param(f"matrix_{k}", _matrix_init, shape, scale)
      bias = self.param(f"bias_{k}", _bias_init, shape[-1:])
      u = u @ jax.nn.softplus(matrix) + bias
      if k < len(self.features):
        factor = self.param(f"factor_{k}", _factor_init, shape[-1:])
        u += jnp.tanh(u) * jnp.tanh(factor)
    return u.reshape(x.shape)


class DeepFactorizedEntropyModel(continuous.ContinuousEntropyModel):
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

  Attributes:
    features: Iterable of integers. The number of filters for each of the
      hidden layers. The first and last layer of the network implementing the
      cumulative distribution are not included (they are assumed to be 1).
    init_scale: Float. Scale factor for the density at initialization. It is
      recommended to choose a large enough scale factor such that most values
      initially lie within a region of high likelihood. This improves
      training.
  """
  features: Tuple[int, ...] = (3, 3, 3)
  init_scale: float = 10.

  # TODO(jonycgn): This class is currently limited to only modeling distinct
  # distributions in the last dimension. Consider adding more shaping options.

  def setup(self):
    super().setup()
    self.cdf_logits = nn.vmap(
        MonotonicMLP,
        in_axes=-1,
        out_axes=-1,
        variable_axes={"params": 0},
        split_rngs={"params": True})(self.features, self.init_scale)

  def bin_bits(self, center, temperature=jnp.inf):
    upper = rounding.soft_round_inverse(center + .5, temperature)
    lower = upper - 1.
    logits_upper = self.cdf_logits(upper)
    logits_lower = self.cdf_logits(lower)
    # sigmoid(u) - sigmoid(l) = sigmoid(-l) - sigmoid(-u)
    condition = logits_upper <= -logits_lower
    big = nn.log_sigmoid(jnp.where(condition, logits_upper, -logits_lower))
    small = nn.log_sigmoid(jnp.where(condition, logits_lower, -logits_upper))
    return continuous.logsum_expbig_minus_expsmall(big, small) / -jnp.log(2.)

  def bin_prob(self, center, temperature=jnp.inf):
    upper = rounding.soft_round_inverse(center + .5, temperature)
    lower = upper - 1.
    logits_upper = self.cdf_logits(upper)
    logits_lower = self.cdf_logits(lower)
    # sigmoid(u) - sigmoid(l) = sigmoid(-l) - sigmoid(-u)
    sgn = -jnp.sign(logits_upper + logits_lower)
    return abs(nn.sigmoid(sgn * logits_upper) - nn.sigmoid(sgn * logits_lower))
