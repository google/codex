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
"""Fourier series density models."""

from codex.ems import fourier
import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jax.Array
ArrayLike = jax.typing.ArrayLike


class PeriodicFourierDensityModel(nn.Module):
  """Fully factorized Fourier Basis probablity model for periodic distributions on [-period/2, period/2].

  Attributes:
    num_freqs: Integer. Number of frequency components of the Fourier series.
    num_dims: Integer. Number of dimensions of the probability distribution.
    period: Float. Period of the Fourier series centered at zero.
  """

  num_freqs: int = 5
  num_dims: int = 1
  period: float = 2.0 * jnp.pi

  @nn.compact
  def neg_log_prob(self, center: ArrayLike) -> Array:

    # Raw series float coefficients.
    coef = self.param(
        "coef",
        nn.initializers.normal(0.001),
        (self.num_dims, 2 * self.num_freqs),
    )

    # Cast as complex
    coef = jax.lax.complex(coef[:, : self.num_freqs], coef[:, self.num_freqs :])

    # Calculate PDF
    pdf = fourier.build_periodic_pdf(coef, center, self.period)

    # Return Negative Log probability
    return -jnp.log(pdf)


class GeneralizedFourierDensityModel(nn.Module):
  """Fully factorized Fourier probability model defined for the real line.

  Attributes:
    num_freqs: Integer. Number of frequency components of the Fourier series.
    num_dims: Integer. Number of dimensions of the probability distribution.
  """

  num_freqs: int = 5
  num_dims: int = 1

  @nn.compact
  def neg_log_prob(self, center: ArrayLike) -> Array:

    # Raw series float coefficients.
    coef = self.param(
        "coef",
        nn.initializers.normal(0.001),
        (self.num_dims, 2 * self.num_freqs),
    )

    # Initialize scale / offset
    scale = self.param("scale", nn.initializers.ones, (1, self.num_dims))
    offset = self.param("offset", nn.initializers.zeros, (1, self.num_dims))
    scale = nn.softplus(scale)

    # Cast as complex
    coef = jax.lax.complex(coef[:, : self.num_freqs], coef[:, self.num_freqs :])

    # Calculate PDF
    pdf = fourier.build_pdf(coef, center, scale, offset)

    # Return Negative Log Probability
    return -jnp.log(pdf)
