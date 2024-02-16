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
"""Test for fourier functions."""

from codex.ems import fourier
import jax
import jax.numpy as jnp
import numpy as np


def test_autocorrelate_shape():
  rng = jax.random.PRNGKey(0)
  batch, length = 3, 5
  rnd = jax.random.normal(rng, (batch, length))
  assert fourier.autocorrelate(rnd).shape == (batch, length)


def test_autocorrelate_correctness():
  rng = jax.random.PRNGKey(0)
  ix = jax.random.normal(rng, (10,))
  auto_corr1 = jax.numpy.correlate(ix, ix, mode='full')[len(ix) - 1 :]
  auto_corr2 = fourier.autocorrelate(ix.reshape(1, -1))[0]

  assert jnp.all(auto_corr1 == auto_corr2)
  ix = jnp.array([[1, 0, 0, 0], [1j, 0, 0, 0]]).astype(jnp.complex64)
  ox = jnp.array([[1, 0, 0, 0], [1, 0, 0, 0]]).astype(jnp.complex64)
  assert jnp.all(fourier.autocorrelate(ix) == ox)


def test_build_periodic_pdf_shape():
  num_freq, num_dims, length, period = 4, 2, 5, 1.0
  rng = jax.random.PRNGKey(0)
  coef = jax.random.normal(rng, (num_dims, num_freq))
  centers = jax.random.normal(rng, (length, num_dims))
  pdf = fourier.build_periodic_pdf(coef, centers, period)
  assert pdf.shape == (length, num_dims)


def test_build_pdf_shape():
  num_freq, num_dims, length = 4, 2, 5
  rng = jax.random.PRNGKey(0)
  coef = jax.random.normal(rng, (num_dims, num_freq))
  center = jax.random.normal(rng, (length, num_dims))
  offset = jnp.zeros((1, num_dims))
  scale = jnp.ones((1, num_dims))
  pdf = fourier.build_pdf(coef, center, scale, offset)
  assert pdf.shape == (length, num_dims)


def test_build_pdf_large_scale():
  num_freq, num_dims, length = 4, 2, 5
  rng = jax.random.PRNGKey(0)
  coef = jax.random.normal(rng, (num_dims, num_freq))
  center = jax.random.normal(rng, (length, num_dims))
  offset = jnp.zeros((1, num_dims))
  scale = 1e6 * jnp.ones((1, num_dims))
  pdf = fourier.build_pdf(coef, center, scale, offset, 1e-4)
  assert jnp.all(pdf == 1e-4)


def test_build_pdf_small_scale():
  num_freq, num_dims, length = 4, 2, 5
  rng = jax.random.PRNGKey(0)
  coef = jax.random.normal(rng, (num_dims, num_freq))
  center = jax.random.normal(rng, (length, num_dims))
  offset = jnp.zeros((1, num_dims))
  scale = 1e-6 * jnp.ones((1, num_dims))
  pdf = fourier.build_pdf(coef, center, scale, offset)
  assert jnp.all(pdf == 1e-20)


def test_build_pdf_integral_equal_one():
  num_freq, num_dims = 4, 2
  xlim, length = 20, 10000
  rng = jax.random.PRNGKey(0)
  center = jnp.linspace(-xlim, xlim, length)
  center = jnp.moveaxis(jnp.tile(center, (num_dims, 1)), -1, 0)
  coef = jax.random.normal(rng, (num_dims, num_freq))
  offset = jnp.zeros((1, num_dims))
  scale = jnp.ones((1, num_dims))
  pdf = fourier.build_pdf(coef, center, scale, offset)
  integral = jnp.round(np.trapz(pdf, dx=2.0 * xlim / length, axis=0), 3)
  assert jnp.all(integral == 1.0)


def test_build_pdf_non_negative():
  num_freq, num_dims = 4, 2
  xlim, length = 20, 10000
  rng = jax.random.PRNGKey(0)
  center = jnp.linspace(-xlim, xlim, length)
  center = jnp.moveaxis(jnp.tile(center, (num_dims, 1)), -1, 0)
  coef = jax.random.normal(rng, (num_dims, num_freq))
  offset = jnp.zeros((1, num_dims))
  scale = jnp.ones((1, num_dims))
  pdf = fourier.build_pdf(coef, center, scale, offset)
  assert jnp.all(pdf >= 0)
