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

from typing import ClassVar
from codex.ems import continuous
from codex.ops import rounding
import jax.numpy as jnp


def _bin_prob_even(distribution, center, temperature):
  """Computes probability mass of quant. bins for symmetric distribution."""
  upper = rounding.soft_round_inverse(.5 - abs(center), temperature)
  lower = upper - 1.
  return distribution.cdf(upper) - distribution.cdf(lower)


def _bin_prob(distribution, center, temperature):
  """Computes probability mass of quantization bins."""
  upper = rounding.soft_round_inverse(center + .5, temperature)
  lower = upper - 1.
  sf_upper = distribution.survival_function(upper)
  sf_lower = distribution.survival_function(lower)
  cdf_upper = distribution.cdf(upper)
  cdf_lower = distribution.cdf(lower)
  return jnp.where(
      sf_upper < cdf_upper, sf_lower - sf_upper, cdf_upper - cdf_lower)


def _bin_bits_even(distribution, center, temperature):
  """Computes information content of quant. bins for symmetric distribution."""
  upper = rounding.soft_round_inverse(.5 - abs(center), temperature)
  lower = upper - 1.
  big = distribution.log_cdf(upper)
  small = distribution.log_cdf(lower)
  return continuous.logsum_expbig_minus_expsmall(big, small) / -jnp.log(2.)


def _bin_bits(distribution, center, temperature):
  """Computes information content of quantization bins."""
  upper = rounding.soft_round_inverse(center + .5, temperature)
  lower = upper - 1.
  logsf_upper = distribution.log_survival_function(upper)
  logsf_lower = distribution.log_survival_function(lower)
  logcdf_upper = distribution.log_cdf(upper)
  logcdf_lower = distribution.log_cdf(lower)
  condition = logsf_upper < logcdf_upper
  big = jnp.where(condition, logsf_lower, logcdf_upper)
  small = jnp.where(condition, logsf_upper, logcdf_lower)
  return continuous.logsum_expbig_minus_expsmall(big, small) / -jnp.log(2.)


class DistributionEntropyModel(continuous.ContinuousEntropyModel):
  """Entropy model wrapping a Distrax/TFP Distribution object.

  Attributes:
    even_symmetric: Boolean. If `True`, indicates that `distribution` is
      guaranteed to be symmetric around zero (p(x) = p(-x) for any x). This
      simplifies computations in `bin_prob`/`bin_bits`. Defaults to `False`.
  """

  even_symmetric: ClassVar[bool] = False

  @property
  def distribution(self):
    """Distrax/TFP `Distribution` representing this entropy model.

    TODO(jonycgn): Document what parts of the TFP/Distrax API need to be
    implemented.
    """
    raise NotImplementedError("Subclass must define distribution.")

  def bin_prob(self, center, temperature=jnp.inf):
    if self.even_symmetric:
      return _bin_prob_even(self.distribution, center, temperature)
    return _bin_prob(self.distribution, center, temperature)

  def bin_bits(self, center, temperature=jnp.inf):
    if self.even_symmetric:
      return _bin_bits_even(self.distribution, center, temperature)
    return _bin_bits(self.distribution, center, temperature)

  def quantization_offset(self):
    return self.distribution.mode()

  def tail_locations(self, tail_mass):
    return (self.distribution.quantile(tail_mass),
            self.distribution.quantile(1 - tail_mass))
