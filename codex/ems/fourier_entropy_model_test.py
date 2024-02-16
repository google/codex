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
"""Test for fourier entropy model."""

import chex
from codex.ems import fourier_entropy_model
import jax
import jax.numpy as jnp


def test_fourier_entropy_model_output_shape():
  em = fourier_entropy_model.FourierSeriesEntropyModel(num_freqs=10)
  num_dims, length = 3, 20
  x = jax.random.normal(jax.random.PRNGKey(0), (length, num_dims))
  prob, _ = em.init_with_output(jax.random.PRNGKey(0), x, method=em.bin_prob)
  chex.assert_shape(prob, (length, num_dims))


def test_fourier_entropy_model_bin_prob_sum_values():
  em = fourier_entropy_model.FourierSeriesEntropyModel(num_freqs=10)
  x = jnp.linspace(-10, 10, 21)
  x = jnp.moveaxis(
      jnp.stack(
          (x, x),
          axis=0,
      ),
      -1,
      0,
  )
  bin_prob_values, _ = em.init_with_output(
      jax.random.PRNGKey(0), x, method=em.bin_prob
  )
  integral = jnp.round(bin_prob_values.sum(axis=0), 3)
  chex.assert_trees_all_equal(integral, jnp.array([1.0, 1.0]))


def test_fourier_bin_prob_and_bin_prob_are_consistent():
  em = fourier_entropy_model.FourierSeriesEntropyModel(num_freqs=15)
  num_dims, length = 3, 20
  x = jax.random.normal(jax.random.PRNGKey(0), (length, num_dims))
  prob, v = em.init_with_output(jax.random.PRNGKey(0), x, method=em.bin_prob)
  bits_values = em.apply(v, x, method=em.bin_bits)
  chex.assert_trees_all_close(-jnp.log(prob), bits_values, atol=1e-7)
