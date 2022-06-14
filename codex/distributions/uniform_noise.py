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
"""Distributions that model additive uniform noise."""

from codex.distributions import distribution
from jax import lax


def _logsum_expbig_minus_expsmall(big, small):
  """Numerically stable evaluation of `log(exp(big) - exp(small))`.

  This assumes `small <= big` and arguments that can be broadcast against each
  other.
  Args:
    big: Floating-point `Tensor`.
    small: Floating-point `Tensor`.

  Returns:
    `tf.Tensor` containing the result.
  """
  return lax.log1p(-lax.exp(small - big)) + big


class UniformNoiseAdapter(distribution.Distribution):
  """Additive i.i.d.

  uniform noise adapter distribution.
  Given a base `tfp.distributions.Distribution` object, this distribution
  models the base distribution after addition of independent uniform noise.
  Effectively, the base density function is convolved with a box kernel of width
  one. The resulting density can be efficiently evaluated via the relation:
  ```
  (p * u)(x) = c(x + .5) - c(x - .5)
  ```
  where `p` and `u` are the base density and the unit-width uniform density,
  respectively, and `c` is the cumulative distribution function (CDF)
  corresponding to `p`. This is described in appendix 6.2 of the paper:
  > "Variational image compression with a scale hyperprior"<br />
  > J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston<br />
  > https://openreview.net/forum?id=rkcQFMZRb
  For best numerical stability, the base `Distribution` should implement both
  `cdf()` and `survival_function()` and/or their `log_*` equivalents.
  """

  def __init__(self, base):
    """Initializer.

    Args:
      base: A `Distribution` object representing a continuous-valued random
        variable.
    """
    self._base = base

  @property
  def base(self):
    """The base distribution (without uniform noise)."""
    return self._base

  def log_prob(self, y):
    return self._log_prob_with_logsf_and_logcdf(y)

  def _log_prob_with_logsf_and_logcdf(self, y):
    """Compute log_prob(y) using log survival_function and cdf together."""
    # There are two options that would be equal if we had infinite precision:
    # Log[ sf(y - .5) - sf(y + .5) ]
    #   = Log[ exp{logsf(y - .5)} - exp{logsf(y + .5)} ]
    # Log[ cdf(y + .5) - cdf(y - .5) ]
    #   = Log[ exp{logcdf(y + .5)} - exp{logcdf(y - .5)} ]
    logsf_y_plus = self.base.log_survival_function(y + .5)
    logsf_y_minus = self.base.log_survival_function(y - .5)
    logcdf_y_plus = self.base.log_cdf(y + .5)
    logcdf_y_minus = self.base.log_cdf(y - .5)

    # Important:  Here we use select in a way such that no input is inf, this
    # prevents the troublesome case where the output of select can be finite,
    # but the output of grad(select) will be NaN.

    # In either case, we are doing Log[ exp{big} - exp{small} ]
    # We want to use the sf items precisely when we are on the right side of the
    # median, which occurs when logsf_y < logcdf_y.
    condition = logsf_y_plus < logcdf_y_plus
    big = lax.select(condition, logsf_y_minus, logcdf_y_plus)
    small = lax.select(condition, logsf_y_plus, logcdf_y_minus)
    return _logsum_expbig_minus_expsmall(big, small)

  def prob(self, y):
    return lax.exp(self.log_prob(y))


class NoisyNormal(UniformNoiseAdapter):
  """Gaussian distribution with additive i.i.d. uniform noise."""

  def __init__(self, **kwargs):
    super().__init__(distribution.Normal(**kwargs))


class NoisyLogistic(UniformNoiseAdapter):
  """Logistic distribution with additive i.i.d. uniform noise."""

  def __init__(self, **kwargs):
    super().__init__(distribution.Logistic(**kwargs))
