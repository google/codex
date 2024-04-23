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
"""Deep fully factorized entropy model based on cumulative density."""

from typing import Optional, Tuple
from codex.ems import continuous
from codex.ops import quantization
import jax
from jax import nn
import jax.numpy as jnp

Array = jax.Array
ArrayLike = jax.typing.ArrayLike


def matrix_init(shape, scale):
  """Initializes matrix with constant value according to init_scale.

  This initialization function was designed to avoid the early training
  stall due to too large or too small probability mass values computed
  with the initial parameters.

  Softplus is applied to the matrix during the forward pass, to ensure the
  matrix elements are all positive. With the current initialization, the
  matrix has constant values and after matrix-vector multiplication with
  the input, the output vector elements are constant

  ```
  x / init_scale ** (1 / 1 + len(num_units)).
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
    shape: Sequence of integers. The shape of the matrix.
    scale: Scale factor for initial value.

  Returns:
    The initial matrix value.
  """
  return jnp.full(shape, jnp.log(jnp.expm1(1 / scale / shape[-1])))


# pytype:disable=attribute-error
class MonotonicMLPBase:
  """MLP that implements monotonically increasing functions by construction."""

  def __call__(self, x):
    x = x[..., None]
    assert len(self.matrices) == len(self.biases) == len(self.factors) + 1
    for k in range(len(self.factors)):
      x = jnp.einsum("...i,...ji->...j", x, jax.nn.softplus(self.matrices[k]))
      x += self.biases[k]
      x += jnp.tanh(x) * jnp.tanh(self.factors[k])
    x = jnp.einsum("...i,...ji->...j", x, jax.nn.softplus(self.matrices[-1]))
    x += self.biases[-1]
    return jnp.squeeze(x, axis=-1)


class DeepFactorizedEntropyModelBase(continuous.ContinuousEntropyModel):
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

  def _upper_lower_logits(
      self,
      center: ArrayLike,
      temperature: Optional[ArrayLike] = None,
  ) -> Tuple[Array, ...]:
    upper = quantization.soft_round_inverse(center + .5, temperature)
    lower = upper - 1.
    logits_upper = self.cdf_logits(upper)
    logits_lower = self.cdf_logits(lower)
    return self._maybe_upcast((logits_upper, logits_lower))

  def bin_bits(self,
               center: ArrayLike,
               temperature: Optional[ArrayLike] = None) -> Array:
    logits_upper, logits_lower = self._upper_lower_logits(center, temperature)
    # sigmoid(u) - sigmoid(l) = sigmoid(-l) - sigmoid(-u)
    condition = logits_upper <= -logits_lower
    big = nn.log_sigmoid(jnp.where(condition, logits_upper, -logits_lower))
    small = nn.log_sigmoid(jnp.where(condition, logits_lower, -logits_upper))
    return continuous.logsum_expbig_minus_expsmall(big, small) / -jnp.log(2.)

  def bin_prob(self,
               center: ArrayLike,
               temperature: Optional[ArrayLike] = None) -> Array:
    logits_upper, logits_lower = self._upper_lower_logits(center, temperature)
    # sigmoid(u) - sigmoid(l) = sigmoid(-l) - sigmoid(-u)
    sgn = -jnp.sign(logits_upper + logits_lower)
    return abs(nn.sigmoid(sgn * logits_upper) - nn.sigmoid(sgn * logits_lower))
# pytype:enable=attribute-error
