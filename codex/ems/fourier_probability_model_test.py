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
"""Test for fourier probability model."""

import chex
from codex.ems import fourier_probability_model
import jax
import jax.numpy as jnp
import numpy as np


def test_periodic_fourier_density_model_output_shape():
  em = fourier_probability_model.PeriodicFourierDensityModel(
      num_freqs=10, period=2.0 * jnp.pi, num_dims=3
  )
  num_dims, length = 3, 20
  x = jax.random.normal(jax.random.PRNGKey(0), (length, num_dims))
  nll, _ = em.init_with_output(jax.random.PRNGKey(0), x, method=em.neg_log_prob)
  chex.assert_shape(nll, (length, num_dims))


def test_generalized_fourier_density_model_output_shape():
  em = fourier_probability_model.GeneralizedFourierDensityModel(
      num_freqs=10, num_dims=3
  )
  num_dims, length = 3, 20
  x = jax.random.normal(jax.random.PRNGKey(0), (length, num_dims))
  nll, _ = em.init_with_output(jax.random.PRNGKey(0), x, method=em.neg_log_prob)
  chex.assert_shape(nll, (length, num_dims))


def test_generalized_fourier_density_model_integral_equal_one():
  xlim, length = 50, 10000
  center = jnp.linspace(-xlim, xlim, length)
  x = jnp.moveaxis(jnp.tile(center, (2, 1)), -1, 0)
  em = fourier_probability_model.GeneralizedFourierDensityModel(
      num_freqs=10, num_dims=2
  )
  nll, _ = em.init_with_output(jax.random.PRNGKey(0), x, method=em.neg_log_prob)
  pdf = jnp.exp(-nll)
  integral = jnp.round(np.trapz(pdf, dx=2.0 * xlim / length, axis=0), 3)
  assert jnp.all(integral == 1.0)
