# Copyright 2024 CoDeX authors.
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
"""Tests of Equinox entropy models."""

import chex
from codex.ems import equinox as ems
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp

Array = jax.Array


class TestDistributionEntropyModel:

  def get_em(self, rng):
    class EntropyModel(ems.DistributionEntropyModel, eqx.Module):
      loc: Array
      log_scale: Array

      def __init__(self, rng):
        rngs = jax.random.split(rng)
        self.loc = jax.random.normal(rngs[0], (3, 1))
        self.log_scale = jax.random.normal(rngs[1], (2,))

      @property
      def distribution(self):
        return distrax.Normal(loc=self.loc, scale=jnp.exp(self.log_scale))

    return EntropyModel(rng)

  expected_param_shapes = [(3, 1), (2,)]

  def test_shapes_are_correct(self):
    get_em = self.get_em

    class OuterModel(eqx.Module):
      em: eqx.Module

      def __init__(self, rng):
        self.em = get_em(rng)

      def __call__(self, x):
        return self.em.bin_bits(x)

    x = jnp.ones((5, 3, 2))

    model = OuterModel(jax.random.PRNGKey(0))
    param_shapes = jax.tree.map(
        lambda p: p.shape,
        jax.tree.flatten(eqx.filter(model, eqx.is_array))[0])
    chex.assert_trees_all_equal(
        param_shapes,
        self.expected_param_shapes)

    y = model(x)
    assert y.shape == x.shape

    grads = eqx.filter_grad(lambda m, x: m(x).sum())(model, x)
    grad_shapes = jax.tree.map(
        lambda g: g.shape,
        jax.tree.flatten(eqx.filter(grads, eqx.is_array))[0])
    chex.assert_trees_all_equal(
        grad_shapes,
        self.expected_param_shapes)


class TestDeepFactorizedEntropyModel(TestDistributionEntropyModel):

  def get_em(self, rng):
    return ems.DeepFactorizedEntropyModel(rng, num_pdfs=2, num_units=(3, 5))

  expected_param_shapes = [
      (2, 3, 1), (2, 5, 3), (2, 1, 5),  # matrices
      (2, 3), (2, 5), (2, 1),  # biases
      (2, 3), (2, 5),  # factors
  ]


class TestPeriodicFourierEntropyModel(TestDistributionEntropyModel):

  def get_em(self, rng):
    return ems.PeriodicFourierEntropyModel(
        rng, period=2., num_pdfs=2, num_freqs=5)

  expected_param_shapes = [(2, 5), (2, 5)]


class TestRealMappedFourierEntropyModel(TestDistributionEntropyModel):

  def get_em(self, rng):
    return ems.RealMappedFourierEntropyModel(rng, num_pdfs=2, num_freqs=5)

  expected_param_shapes = [
      (2,), (2,),  # scale and offset
      (2, 5), (2, 5),  # real and imag
  ]
