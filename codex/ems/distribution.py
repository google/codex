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
"""Entropy model wrapping a Distrax/TFP Distribution object."""

from typing import ClassVar, Optional, Tuple
from codex.ems import continuous
from codex.ops import gradient
from codex.ops import quantization
import jax
import jax.numpy as jnp

Array = jax.Array
ArrayLike = jax.typing.ArrayLike


def bin_prob(distribution,
             center: ArrayLike,
             temperature: Optional[ArrayLike] = None,
             even_symmetric: bool = False) -> Array:
  """Functional version of `DistributionEntropyModel.bin_prob`."""
  # Note that soft_round_inverse corresponds to identity for temperature = None.
  if even_symmetric:
    upper = quantization.soft_round_inverse(.5 - abs(center), temperature)
    lower = upper - 1.
    return distribution.cdf(upper) - distribution.cdf(lower)
  upper = quantization.soft_round_inverse(center + .5, temperature)
  lower = upper - 1.
  sf_upper = distribution.survival_function(upper)
  sf_lower = distribution.survival_function(lower)
  cdf_upper = distribution.cdf(upper)
  cdf_lower = distribution.cdf(lower)
  return jnp.where(
      sf_upper < cdf_upper, sf_lower - sf_upper, cdf_upper - cdf_lower)


def bin_bits(distribution,
             center: ArrayLike,
             temperature: Optional[ArrayLike] = None,
             even_symmetric: bool = False) -> Array:
  """Functional version of `DistributionEntropyModel.bin_bits`."""
  # Note that soft_round_inverse corresponds to identity for temperature = None.
  if even_symmetric:
    upper = quantization.soft_round_inverse(.5 - abs(center), temperature)
    lower = upper - 1.
    big = distribution.log_cdf(upper)
    small = distribution.log_cdf(lower)
    return continuous.logsum_expbig_minus_expsmall(big, small) / -jnp.log(2.)
  upper = quantization.soft_round_inverse(center + .5, temperature)
  lower = upper - 1.
  logsf_upper = distribution.log_survival_function(upper)
  logsf_lower = distribution.log_survival_function(lower)
  logcdf_upper = distribution.log_cdf(upper)
  logcdf_lower = distribution.log_cdf(lower)
  condition = logsf_upper < logcdf_upper
  big = jnp.where(condition, logsf_lower, logcdf_upper)
  small = jnp.where(condition, logsf_upper, logcdf_lower)
  return continuous.logsum_expbig_minus_expsmall(big, small) / -jnp.log(2.)


def scale_param(param, levels, log_scale_min=-10., log_scale_max=10.):
  """Returns a scale parameter for a `Distribution`.

  Input could be a neural network output or a model parameter. The function
  limits the scale to a finite range, which prevents numerical issues, and sets
  up the scale for quantization, which is needed for range coding.

  Example usage in a conditional entropy model:

  ```
  class ConditionalEntropyModel(cdx.ems.DistributionEntropyModel):
    param: Array

    @property
    def distribution(self):
      loc = self.param[..., 0::2]
      scale = cdx.ems.scale_param(self.param[..., 1::2], 20)
      return tfp.distributions.Normal(loc=loc, scale=scale)
  ```

  Args:
    param: An array with expected values in the range [0, `levels`]. Values
      outside this range are clipped by this function. For example, this could
      be the output of a neural network ending in a linear layer.
    levels: Integer. The upper range limit for `param`.
    log_scale_min: Float. Logarithm of minimal output value.
    log_scale_max: Float. Logarithm of maximal output value.

  Returns:
    `jnp.exp(A + B * param)`, where `A` and `B` are chosen such that the active
    range for `param` is [0, `levels`], and the output of the function
    is in the range [`jnp.exp(log_scale_min)`, `jnp.exp(log_scale_max)`].
  """
  param = gradient.lower_limit(param, 0)
  param = gradient.upper_limit(param, levels)
  factor = (log_scale_max - log_scale_min) / levels
  return jnp.exp(log_scale_min + factor * param)


class DistributionEntropyModel(continuous.ContinuousEntropyModel):
  """Entropy model wrapping a continuous Distrax/TFP Distribution object.

  Attributes:
    distribution: The `Distribution` object. It needs to implement
      `log_survival_function` and `log_cdf`.
    even_symmetric: Boolean. If `True`, indicates that `distribution` is
      guaranteed to be symmetric around zero (p(x) = p(-x) for any x). This
      simplifies computations in `bin_prob`/`bin_bits`. Defaults to `False`.
  """

  even_symmetric: ClassVar[bool] = False

  @property
  def distribution(self):
    """Continuous Distrax/TFP `Distribution` representing this entropy model."""
    raise NotImplementedError("Subclass must define distribution.")

  def bin_prob(self,
               center: ArrayLike,
               temperature: Optional[ArrayLike] = None) -> Array:
    center, temperature = self._maybe_upcast((center, temperature))
    return bin_prob(self.distribution, center, temperature, self.even_symmetric)

  def bin_bits(self,
               center: ArrayLike,
               temperature: Optional[ArrayLike] = None) -> Array:
    center, temperature = self._maybe_upcast((center, temperature))
    return bin_bits(self.distribution, center, temperature, self.even_symmetric)

  def quantization_offset(self) -> Array:
    return self.distribution.mode()

  def tail_locations(self, tail_mass: ArrayLike) -> Tuple[Array, Array]:
    tail_mass = self._maybe_upcast(tail_mass)
    return (self.distribution.quantile(tail_mass),
            self.distribution.quantile(1 - tail_mass))
