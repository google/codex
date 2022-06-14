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
"""Tests for distributions."""

# TODO(jonycgn): Convert to pytest.

from absl.testing import absltest
from absl.testing import parameterized
import chex
from codex.distributions import distribution
import jax


class DistributionsTest(parameterized.TestCase):

  @parameterized.parameters(distribution.Normal, distribution.Logistic)
  def test_loc_scale_can_instantiate_and_run(self, cls):
    xx = jax.random.normal(jax.random.PRNGKey(0), (3, 4, 5))
    dist = cls(loc=0.0, scale=1.0)
    chex.assert_equal_shape((xx, dist.prob(xx)))
    chex.assert_equal_shape((xx, dist.log_prob(xx)))
    chex.assert_equal_shape((xx, dist.cdf(xx)))
    chex.assert_equal_shape((xx, dist.log_cdf(xx)))
    chex.assert_equal_shape((xx, dist.survival_function(xx)))
    chex.assert_equal_shape((xx, dist.log_survival_function(xx)))


if __name__ == '__main__':
  absltest.main()
