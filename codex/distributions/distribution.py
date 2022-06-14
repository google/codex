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
"""Scalar distributions for entropy modeling."""

from tensorflow_probability.substrates import jax as tfp

# TODO(relational): Break up into modules?


class Distribution:
  """Specifies the API of a scalar distribution."""

  def prob(self, x):
    raise NotImplementedError()

  def log_prob(self, x):
    raise NotImplementedError()

  def cdf(self, y):
    raise NotImplementedError()

  def log_cdf(self, y):
    raise NotImplementedError()

  def survival_function(self, y):
    raise NotImplementedError()

  def log_survival_function(self, y):
    raise NotImplementedError()


class WrappedDistribution(Distribution):
  """Wrap an object that already implements the `Distribution` API.

  Attributes:
    distribution: An object that implements the methods specified in
      `Distribution`.
  """
  # TODO(relational): Fix.
  # pytype: disable=attribute-error

  def prob(self, x):
    return self.distribution.prob(x)

  def log_prob(self, x):
    return self.distribution.log_prob(x)

  def cdf(self, y):
    return self.distribution.cdf(y)

  def log_cdf(self, y):
    return self.distribution.log_cdf(y)

  def survival_function(self, y):
    return self.distribution.survival_function(y)

  def log_survival_function(self, y):
    return self.distribution.log_survival_function(y)
  # pytype: enable=attribute-error


class Normal(WrappedDistribution):

  def __init__(self, loc, scale):
    self.distribution = tfp.distributions.Normal(loc=loc, scale=scale)


class Logistic(WrappedDistribution):

  def __init__(self, loc, scale):
    self.distribution = tfp.distributions.Logistic(loc=loc, scale=scale)
