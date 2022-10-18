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
"""Tests of deep factorized entropy model."""

import chex
from codex.ems import deep_factorized
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

# TODO(jonycgn): Improve unit tests, e.g. check that the distribution is
# normalized, that expected value of `bin_bits` with noise is an upper bound
# on it without noise, ...


def test_logistic_is_special_case():
  # With no hidden units, the density should collapse to a logistic
  # distribution convolved with a standard uniform distribution.
  em = deep_factorized.DeepFactorizedEntropyModel(features=(), init_scale=1)
  x = jnp.linspace(-5., 5., 30)[:, None]
  prob_em, v = em.init_with_output(jax.random.PRNGKey(0), x, method=em.bin_prob)
  logistic = tfp.distributions.Logistic(
      loc=-v["params"]["cdf_logits"]["bias_0"][0, 0], scale=1.)
  prob_log = logistic.cdf(x + .5) - logistic.cdf(x - .5)
  chex.assert_trees_all_close(prob_em, prob_log, atol=1e-7)


def test_uniform_is_special_case():
  # With the scale parameter going to zero, the density should approach a
  # unit-width uniform distribution.
  em = deep_factorized.DeepFactorizedEntropyModel(init_scale=1e-6)
  x = jnp.linspace(-1., 1., 10)[:, None]
  prob, _ = em.init_with_output(jax.random.PRNGKey(0), x, method=em.bin_prob)
  chex.assert_trees_all_close(prob[:, 0],
                              jnp.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0]))


def test_bin_prob_and_bits_are_consistent():
  em = deep_factorized.DeepFactorizedEntropyModel()
  x = jnp.linspace(-5., 5., 30)[:, None]
  prob, v = em.init_with_output(jax.random.PRNGKey(0), x, method=em.bin_prob)
  bits = em.apply(v, x, method=em.bin_bits)
  chex.assert_trees_all_close(prob, 2 ** -bits, atol=1e-7)
