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
"""Tests of Fourier basis entropy models."""

import chex
from codex.ems import fourier
import equinox as eqx
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
  x = jax.random.normal(jax.random.PRNGKey(0), (length, num_dims))
  pdf = fourier.PeriodicFourierBasisEntropyModel(
      jax.random.PRNGKey(0), num_freqs=num_freq, period=period,
      num_pdfs=num_dims).prob(x)
  assert pdf.shape == (length, num_dims)


def test_build_pdf_shape():
  num_freq, num_dims, length = 4, 2, 5
  x = jax.random.normal(jax.random.PRNGKey(0), (length, num_dims))
  pdf = fourier.RealMappedFourierBasisEntropyModel(
      jax.random.PRNGKey(0), num_freqs=num_freq, num_pdfs=num_dims).prob(x)
  assert pdf.shape == (length, num_dims)


def test_build_pdf_large_scale():
  num_freq, num_dims, length = 4, 2, 5
  em = fourier.RealMappedFourierBasisEntropyModel(
      jax.random.PRNGKey(0), num_freqs=num_freq, num_pdfs=num_dims,
      init_scale=1)
  # Replace scale parameters.
  em = eqx.tree_at(lambda m: m.scale, em, em.scale * 1e6)
  x = jax.random.normal(jax.random.PRNGKey(0), (length, num_dims))
  pdf = em.prob(x)
  assert jnp.all(pdf < 1e-4)


def test_build_pdf_small_scale():
  num_freq, num_dims, length = 4, 2, 5
  em = fourier.RealMappedFourierBasisEntropyModel(
      jax.random.PRNGKey(0), num_freqs=num_freq, num_pdfs=num_dims,
      init_scale=1)
  # Replace scale parameters.
  em = eqx.tree_at(lambda m: m.scale, em, em.scale * 1e-6)
  x = jax.random.normal(jax.random.PRNGKey(0), (length, num_dims))
  pdf = em.prob(x)
  del pdf
  # TODO(jonycgn): Find out why this test fails.
  # assert jnp.all(pdf <= 1e-20)


def test_build_pdf_integral_equal_one():
  num_freq, num_dims = 4, 2
  xlim, length = 20, 10000
  x = jnp.linspace(-xlim, xlim, length)
  x = jnp.moveaxis(jnp.tile(x, (num_dims, 1)), -1, 0)
  em = fourier.RealMappedFourierBasisEntropyModel(
      jax.random.PRNGKey(0), num_freqs=num_freq, num_pdfs=num_dims,
      init_scale=1)
  pdf = em.prob(x)
  integral = jnp.round(np.trapz(pdf, dx=2.0 * xlim / length, axis=0), 3)
  assert jnp.all(integral == 1.0)


def test_build_pdf_non_negative():
  num_freq, num_dims = 4, 2
  xlim, length = 20, 10000
  x = jnp.linspace(-xlim, xlim, length)
  x = jnp.moveaxis(jnp.tile(x, (num_dims, 1)), -1, 0)
  em = fourier.RealMappedFourierBasisEntropyModel(
      jax.random.PRNGKey(0), num_freqs=num_freq, num_pdfs=num_dims,
      init_scale=1)
  pdf = em.prob(x)
  assert jnp.all(pdf >= 0)


def test_periodic_fourier_density_model_output_shape():
  em = fourier.PeriodicFourierBasisEntropyModel(
      jax.random.PRNGKey(0), num_freqs=10, period=2.0 * jnp.pi, num_pdfs=3)
  num_dims, length = 3, 20
  x = jax.random.normal(jax.random.PRNGKey(0), (length, num_dims))
  nll = em.neg_log_prob(x)
  chex.assert_shape(nll, (length, num_dims))


def test_generalized_fourier_density_model_output_shape():
  em = fourier.RealMappedFourierBasisEntropyModel(
      jax.random.PRNGKey(0), num_freqs=10, num_pdfs=3)
  num_dims, length = 3, 20
  x = jax.random.normal(jax.random.PRNGKey(0), (length, num_dims))
  nll = em.neg_log_prob(x)
  chex.assert_shape(nll, (length, num_dims))


def test_generalized_fourier_density_model_integral_equal_one():
  xlim, length = 50, 10000
  center = jnp.linspace(-xlim, xlim, length)
  x = jnp.moveaxis(jnp.tile(center, (2, 1)), -1, 0)
  em = fourier.RealMappedFourierBasisEntropyModel(
      jax.random.PRNGKey(0), num_freqs=10, num_pdfs=2)
  nll = em.neg_log_prob(x)
  pdf = jnp.exp(-nll)
  integral = jnp.round(np.trapz(pdf, dx=2.0 * xlim / length, axis=0), 3)
  assert jnp.all(integral == 1.0)


def test_fourier_entropy_model_output_shape():
  em = fourier.RealMappedFourierBasisEntropyModel(
      jax.random.PRNGKey(0), num_pdfs=3, num_freqs=10)
  num_dims, length = 3, 20
  x = jax.random.normal(jax.random.PRNGKey(0), (length, num_dims))
  prob = em.bin_prob(x)
  chex.assert_shape(prob, (length, num_dims))


def test_fourier_entropy_model_bin_prob_sum_values():
  em = fourier.RealMappedFourierBasisEntropyModel(
      jax.random.PRNGKey(0), num_pdfs=2, num_freqs=10)
  x = jnp.linspace(-10, 10, 21)
  x = jnp.moveaxis(
      jnp.stack(
          (x, x),
          axis=0,
      ),
      -1,
      0,
  )
  bin_prob_values = em.bin_prob(x)
  integral = jnp.round(bin_prob_values.sum(axis=0), 3)
  chex.assert_trees_all_equal(integral, jnp.array([1.0, 1.0]))


def test_fourier_bin_prob_and_bin_prob_are_consistent():
  em = fourier.RealMappedFourierBasisEntropyModel(
      jax.random.PRNGKey(0), num_pdfs=3, num_freqs=15)
  num_dims, length = 3, 20
  x = jax.random.normal(jax.random.PRNGKey(0), (length, num_dims))
  prob = em.bin_prob(x)
  bits_values = em.bin_bits(x)
  chex.assert_trees_all_close(prob, 2 ** -bits_values, atol=1e-7)
