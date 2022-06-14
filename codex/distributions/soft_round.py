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
"""Distributions that model soft rounding."""

from codex.distributions import distribution
from codex.distributions import uniform_noise
from codex.ops import rounding


class MonotonicAdapter(distribution.Distribution):
  """Adapt a continuous distribution via an ascending monotonic function.

  This is described in Appendix E. in the paper
  > "Universally Quantized Neural Compression"<br />
  > Eirikur Agustsson & Lucas Theis<br />
  > https://arxiv.org/abs/2006.09952
  """

  def __init__(self, base):
    """Initializer.

    Args:
      base: A `tfp.distributions.Distribution` object representing a
        continuous-valued random variable.
    """
    self._base = base

  @property
  def base(self):
    """The base distribution."""
    return self._base

  def transform(self, x):
    """The forward transform."""
    raise NotImplementedError()

  def inverse_transform(self, y):
    """The backward transform."""
    # Let f(x) = self.transform(x)
    # Then g(y) = self.inverse_transform(y) is defined as
    # g(y) := inf_x { x : f(x) >= y }
    # which is just the inverse of `f` if it is invertible.
    raise NotImplementedError()

  def prob(self, x):
    raise NotImplementedError()

  def log_prob(self, x):
    raise NotImplementedError()

  # pylint: disable=protected-access
  def cdf(self, y):
    # Let f be the forward transform and g the inverse.
    # Then we have:
    #   P( f(x) <= y )
    #   P( g(f(x)) <= g(y) )
    # = P(  x <= g(y) )
    return self.base.cdf(self.inverse_transform(y))

  def log_cdf(self, y):
    return self.base.log_cdf(self.inverse_transform(y))

  def survival_function(self, y):
    return self.base.survival_function(self.inverse_transform(y))

  def log_survival_function(self, y):
    return self.base.log_survival_function(self.inverse_transform(y))


class SoftRoundAdapter(MonotonicAdapter):
  """Differentiable approximation to round."""

  def __init__(self, base, alpha):
    """Initializer.

    Args:
      base: A `tfp.distributions.Distribution` object representing a
        continuous-valued random variable.
      alpha: Float or tf.Tensor. Controls smoothness of the approximation.
    """
    super().__init__(base=base)
    self._alpha = alpha

  def transform(self, x):
    return rounding.soft_round(x, self._alpha)

  def inverse_transform(self, y):
    return rounding.soft_round_inverse(y, self._alpha)


class NoisySoftRoundAdapter(uniform_noise.UniformNoiseAdapter):
  """Uniform noise + differentiable approximation to round."""

  def __init__(self, base, alpha):
    """Initializer.

    Args:
      base: A `Distribution` object representing a continuous-valued random
        variable.
      alpha: Float or Tensor. Controls smoothness of soft round.
    """
    super().__init__(SoftRoundAdapter(base, alpha))


class NoisySoftRoundedNormal(NoisySoftRoundAdapter):
  """Soft rounded normal distribution + uniform noise."""

  def __init__(self, alpha=5.0, **kwargs):
    super().__init__(base=distribution.Normal(**kwargs), alpha=alpha)


class NoisySoftRoundedLogistic(NoisySoftRoundAdapter):
  """Soft rounded normal distribution + uniform noise."""

  def __init__(self, alpha=5.0, **kwargs):
    super().__init__(base=distribution.Logistic(**kwargs), alpha=alpha)
