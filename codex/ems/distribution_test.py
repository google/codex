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
"""Tests for distribution entropy model."""

from typing import Any
import chex
from codex.ems import distribution
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp


# TODO(jonycgn): Improve unit tests.


class TestNormalDistribution:

  class EntropyModel(distribution.DistributionEntropyModel):
    loc: Any
    scale: Any

    @property
    def distribution(self):
      return tfp.distributions.Normal(loc=self.loc, scale=self.scale)

  def test_can_instantiate_and_evaluate_scalar(self):
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 4, 5))
    em = self.EntropyModel(loc=0., scale=1.)
    chex.assert_equal_shape((x, em.bin_prob(x)))
    chex.assert_equal_shape((x, em.bin_bits(x)))

  def test_can_instantiate_and_evaluate_array(self):
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 4, 2))
    em = self.EntropyModel(loc=jnp.array([3., 2.]), scale=.5)
    chex.assert_equal_shape((x, em.bin_prob(x)))
    chex.assert_equal_shape((x, em.bin_bits(x)))

  def test_uniform_is_special_case(self):
    # With the scale parameter going to zero, the adapted distribution should
    # approach a unit-width uniform distribution.
    em = self.EntropyModel(loc=5., scale=1e-7)
    x = jnp.linspace(4., 6., 10)
    chex.assert_trees_all_close(
        em.bin_prob(x),
        jnp.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0]))

  def test_plain_noisy_is_special_case(self):
    # With the temperature parameter going to infinity, the adapted distribution
    # should approach a non-soft-rounded distribution.
    em = self.EntropyModel(loc=5., scale=3.)
    x = jnp.linspace(-7., -2., 50)
    chex.assert_trees_all_close(
        em.bin_prob(x),
        em.distribution.cdf(x + .5) - em.distribution.cdf(x - .5))

  def test_non_noisy_is_special_case(self):
    # With the scale parameter going to infinity, the adapted distribution
    # should approach a non-noisy distribution.
    em = self.EntropyModel(loc=5., scale=3000.)
    x = jnp.linspace(2., 7., 50)
    chex.assert_trees_all_close(
        em.bin_prob(x),
        em.distribution.prob(x),
        atol=1e-6)

  def test_scale_param_yields_finite_output(self):
    p = jnp.linspace(-10, 30, 25)
    x = jnp.linspace(-1e5, 1e5, 23)

    em = self.EntropyModel(loc=0., scale=distribution.scale_param(p, 25))

    bits = em.apply({}, x[:, None], method="bin_bits")
    assert jnp.isfinite(bits).all(), bits
    prob = em.apply({}, x[:, None], method="bin_prob")
    assert jnp.isfinite(prob).all(), prob

  def test_scale_param_yields_finite_gradient(self):
    p = jnp.linspace(-10, 30, 25)
    x = jnp.linspace(-1e5, 1e5, 23)

    def bits(x, p):
      em = self.EntropyModel(loc=0., scale=distribution.scale_param(p, 25))
      return em.apply({}, x, method="bin_bits")

    grad_x = jax.grad(bits, argnums=0)
    dbdx = jax.vmap(jax.vmap(grad_x, (None, 0), 0), (0, None), 1)(x, p)
    assert jnp.isfinite(dbdx).all(), dbdx

    grad_p = jax.grad(bits, argnums=1)
    dbdp = jax.vmap(jax.vmap(grad_p, (None, 0), 0), (0, None), 1)(x, p)
    assert jnp.isfinite(dbdp).all(), dbdp


class TestLogisticDistribution(TestNormalDistribution):

  class EntropyModel(distribution.DistributionEntropyModel):
    loc: Any
    scale: Any

    @property
    def distribution(self):
      return tfp.distributions.Logistic(loc=self.loc, scale=self.scale)


class TestLaplaceDistribution(TestNormalDistribution):

  class EntropyModel(distribution.DistributionEntropyModel):
    loc: Any
    scale: Any

    @property
    def distribution(self):
      return tfp.distributions.Laplace(loc=self.loc, scale=self.scale)
