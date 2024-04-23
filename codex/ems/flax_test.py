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
"""Tests of Flax entropy models."""

import chex
from codex.ems import flax as ems
import distrax
from flax import linen as nn
import jax
import jax.numpy as jnp


class TestDistributionEntropyModel:

  def get_em(self):
    class EntropyModel(ems.DistributionEntropyModel, nn.Module):

      def setup(self):
        self.loc = self.param("loc", nn.initializers.normal(), (3, 1))
        self.log_scale = self.param("log_scale", nn.initializers.normal(), (2,))

      @property
      def distribution(self):
        return distrax.Normal(loc=self.loc, scale=jnp.exp(self.log_scale))

    return EntropyModel()

  expected_param_shapes = dict(
      params=dict(
          EntropyModel_0=dict(
              loc=(3, 1),
              log_scale=(2,),
          )
      )
  )

  def test_shapes_are_correct(self):
    get_em = self.get_em

    class OuterModel(nn.Module):

      @nn.compact
      def __call__(self, x):
        return get_em().bin_bits(x)

    x = jnp.ones((5, 3, 2))

    init_params = OuterModel().init(jax.random.PRNGKey(0), x)
    chex.assert_trees_all_equal(
        jax.tree.map(lambda p: p.shape, init_params),
        self.expected_param_shapes)

    y = OuterModel().apply(init_params, x)
    assert y.shape == x.shape

    grads = jax.grad(lambda p: OuterModel().apply(p, x).sum())(init_params)
    chex.assert_trees_all_equal(
        jax.tree.map(lambda g: g.shape, grads),
        self.expected_param_shapes)


class TestDeepFactorizedEntropyModel(TestDistributionEntropyModel):

  def get_em(self):
    return ems.DeepFactorizedEntropyModel(num_pdfs=2, num_units=(3, 5))

  expected_param_shapes = dict(
      params=dict(
          DeepFactorizedEntropyModel_0=dict(
              cdf_logits=dict(
                  matrix_0=(2, 3, 1),
                  matrix_1=(2, 5, 3),
                  matrix_2=(2, 1, 5),
                  bias_0=(2, 3),
                  bias_1=(2, 5),
                  bias_2=(2, 1),
                  factor_0=(2, 3),
                  factor_1=(2, 5),
              ),
          )
      )
  )


class TestPeriodicFourierEntropyModel(TestDistributionEntropyModel):

  def get_em(self):
    return ems.PeriodicFourierEntropyModel(period=2., num_pdfs=2, num_freqs=5)

  expected_param_shapes = dict(
      params=dict(
          PeriodicFourierEntropyModel_0=dict(
              real=(2, 5),
              imag=(2, 5),
          )
      )
  )


class TestRealMappedFourierEntropyModel(TestDistributionEntropyModel):

  def get_em(self):
    return ems.RealMappedFourierEntropyModel(num_pdfs=2, num_freqs=5)

  expected_param_shapes = dict(
      params=dict(
          RealMappedFourierEntropyModel_0=dict(
              real=(2, 5),
              imag=(2, 5),
              offset=(2,),
              scale=(2,),
          )
      )
  )
