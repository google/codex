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
"""Helper functions for Fourier basis models."""

import jax
import jax.numpy as jnp


def autocorrelate(sequence, precision=None):
  """Computes batched complex-valued autocorrelation.

  Args:
    sequence: Array of shape `(batch, length)`, where `batch` is the number of
      independent sequences to correlate with themselves, and `length` is the
      length of each sequence.
    precision: See `jax.lax.conv_general_dilated`.

  Returns:
    Array of shape `(batch, length)`. The right half of each autocorrelation
    sequence. The left half is redundant due to symmetry (even for the real
    part, odd for the imaginary part).
  """
  batch, length = sequence.shape
  return jax.lax.conv_general_dilated(
      sequence[None, :, :],
      sequence[:, None, :].conj(),
      window_strides=(1,),
      padding=[(0, length - 1)],
      feature_group_count=batch,
      precision=precision,
  )[0]


def build_periodic_pdf(complex_coef, center, period, precision=1e-20):
  """Computes PDF for periodic distributions for each value in array center.

  Args:
    complex_coef: Complex typed array of shape `(num_dims, num_freq)`, where
      `num_dims` is the number of independent sequences to correlate with
      themselves, and `num_freq` is the number of frequency terms in the Fourier
      reconstruction.
    center: Array of shape `(length, num_dims)`, where `length` is the number of
      points where we want to evaluate the PDF, and `num_dims` is the number of
      independent dimensions.
    period: float. Applicable for periodic circular distributions.
    precision: float. Minimum value for clipping the distribution.

  Returns:
    Array of shape `(length, num_dims)` which contains the values of PDF.
  """
  # Coefficients are first autocorrelated to ensure a non-negative density.
  coef = autocorrelate(complex_coef)

  # Fitting density on a finite interval - evaluate density
  _, num_freq = coef.shape
  # The DC coefficient is special: it is the normalizer of the density.
  dc = coef[:, 0].real
  ac = coef[:, 1:]
  freq = (2j * jnp.pi) * jnp.arange(1, num_freq) / period
  center = center[..., None]
  # We can take the real part here because the sequence is assumed to have
  # Hermitian symmetry, so the Fourier series is always real.
  density = 0.5 + (ac * jnp.exp(freq * center)).real.sum(axis=-1) / dc
  density /= period / 2.0
  return jnp.clip(density, precision, None)


def build_pdf(complex_coef, center, scale, offset, precision=1e-20):
  """Computes PDF for general distributions for each value in array center.

  Args:
    complex_coef: Complex typed array of shape `(num_dims, num_freq)`, where
      `num_dims` is the number of independent sequences to correlate with
      themselves, and `num_freq` is the number of terms in the Fourier
      reconstruction.
    center: Array of shape `(length, num_dims)`, where `length` is the number of
      points where we want to evaluate the PDF, and `num_dims` is the number of
      independent dimensions.
    scale: Array of shape `(1, num_dims)`, where  `num_dims` is the number of
      independent dimensions.
    offset: Array of shape `(1, num_dims)`, where `num_dims` is the number of
      independent dimensions.
    precision: float. Minimum value for clipping the distribution.

  Returns:
    Array of shape `(length, num_dims)` which contains the values of PDF.
  """
  # Coefficients are first autocorrelated to ensure a non-negative density.
  coef = autocorrelate(complex_coef)

  # Fitting density on a finite interval - evaluate density
  _, num_freq = coef.shape
  # The DC coefficient is special: it is the normalizer of the density.
  dc = coef[:, 0].real
  ac = coef[:, 1:]
  freq = (1j * jnp.pi) * jnp.arange(1, num_freq)

  # Apply change of variables
  center = (center - offset) / scale
  tanh_x = jnp.tanh(center)
  center = center[..., None]
  factor = (1.0 - jnp.square(tanh_x)) / scale
  # We can take the real part here because the sequence is assumed to have
  # Hermitian symmetry, so the Fourier series is always real.
  density = 0.5 + (ac * jnp.exp(freq * jnp.tanh(center))).real.sum(axis=-1) / dc
  density *= factor
  return jnp.clip(density, precision, None)
