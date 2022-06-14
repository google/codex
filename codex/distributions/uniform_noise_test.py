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
"""Tests for uniform noise distributions."""

# TODO(jonycgn): Convert to pytest.

from absl.testing import absltest
from absl.testing import parameterized
import chex
from codex.distributions import distribution
from codex.distributions import uniform_noise
import jax
import jax.numpy as jnp


class NoisySoftRoundedLocScaleTest(parameterized.TestCase):

  @parameterized.parameters(uniform_noise.NoisyNormal,
                            uniform_noise.NoisyLogistic)
  def test_can_instantiate_and_run_scalar(self, dist_cls):
    xx = jax.random.normal(jax.random.PRNGKey(0), (3, 4, 5))
    dist = dist_cls(loc=0.0, scale=1.0)
    chex.assert_equal_shape((xx, dist.prob(xx)))

  @parameterized.parameters(uniform_noise.NoisyNormal,
                            uniform_noise.NoisyLogistic)
  def test_can_instantiate_and_run_batched(self, dist_cls):
    xx = jax.random.normal(jax.random.PRNGKey(0), (3, 4, 2))
    dist = dist_cls(loc=[3.0, 2.0], scale=0.5)
    chex.assert_equal_shape((xx, dist.prob(xx)))

  @parameterized.parameters(uniform_noise.NoisyNormal,
                            uniform_noise.NoisyLogistic)
  def test_methods(self, dist_cls):
    xx = jax.random.normal(jax.random.PRNGKey(0), (3, 4, 5))
    dist = dist_cls(loc=0.0, scale=1.0)
    chex.assert_equal_shape((xx, dist.prob(xx)))
    chex.assert_equal_shape((xx, dist.log_prob(xx)))
    with self.assertRaises(NotImplementedError):
      dist.cdf(xx)
    with self.assertRaises(NotImplementedError):
      dist.log_cdf(xx)
    with self.assertRaises(NotImplementedError):
      dist.survival_function(xx)
    with self.assertRaises(NotImplementedError):
      dist.log_survival_function(xx)

  @parameterized.parameters(uniform_noise.NoisyNormal,
                            uniform_noise.NoisyLogistic)
  def test_uniform_is_special_case(self, dist_cls):
    # With the scale parameter going to zero, the adapted distribution should
    # approach a unit-width uniform distribution.
    dist = dist_cls(loc=5.0, scale=1e-7)
    x = jnp.linspace(5.0 - 1, 5.0 + 1, 10)
    chex.assert_tree_all_close(
        dist.prob(x), jnp.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0]))

  @parameterized.parameters(
      (uniform_noise.NoisyNormal, distribution.Normal),
      (uniform_noise.NoisyLogistic, distribution.Logistic),
      )
  def test_non_noisy_special_case(self, dist_cls, plain_dist_cls):
    # With the scale parameter going to infinity, the adapted distribution
    # should approach a non-noisy distribution.
    dist = dist_cls(loc=5.0, scale=300.0)
    dist2 = plain_dist_cls(loc=5.0, scale=300.0)
    x = jnp.linspace(2, 7)
    chex.assert_tree_all_close(dist.prob(x), dist2.prob(x), atol=1e-6)


if __name__ == '__main__':
  absltest.main()
