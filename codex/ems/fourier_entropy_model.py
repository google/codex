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
"""Fourier series entropy models."""

from typing import Optional

from codex.ems import continuous
from codex.ems import fourier
from codex.ops import quantization
import flax.linen as nn
import jax
import jax.numpy as jnp

Array = jax.Array
ArrayLike = jax.typing.ArrayLike


class FourierSeriesEntropyModel(continuous.ContinuousEntropyModel):
  """Fully factorized entropy model based on Fourier series.

  Attributes:
    num_freqs: Integer. Number of frequency components of the Fourier series.
  """

  num_freqs: int = 5

  @nn.compact
  def bin_prob(
      self, center: ArrayLike, temperature: Optional[ArrayLike] = None
  ) -> Array:
    center, temperature = self._maybe_upcast((center, temperature))

    num_dims = center.shape[-1]

    # Fourier series coefficients. The actual coefficients are first
    # autocorrelated to ensure a non-negative density.
    coef = self.param(
        "coef",
        nn.initializers.normal(0.001),
        (num_dims, 2 * self.num_freqs),
    )

    # Cast as complex
    coef = jax.lax.complex(coef[:, : self.num_freqs], coef[:, self.num_freqs :])

    # Autocorrelate
    coef = fourier.autocorrelate(coef)

    # Initialize scale / offset
    scale = self.param("scale", nn.initializers.ones, (1, num_dims))
    offset = self.param("offset", nn.initializers.zeros, (1, num_dims))
    scale = nn.softplus(scale)

    # Transformation for soft rounding.
    upper = quantization.soft_round_inverse(center + 0.5, temperature)
    # soft_round is periodic with period 1, so we don't need to call it again.
    lower = upper - 1.0

    # Transformation for real line (offset and scale)
    upper = jnp.tanh((upper - offset) / scale)
    lower = jnp.tanh((lower - offset) / scale)

    # The DC coefficient is special: it is the normalizer of the density.
    dc = coef[:, 0].real
    ac = coef[:, 1:]
    freq = 1j * jnp.arange(1, self.num_freqs) * jnp.pi

    # Note: we can take the real part below because the coefficient sequence is
    # assumed to have Hermitian symmetry, so the Fourier series is always real.
    exp_diff = jnp.exp(freq * upper[..., None]) - jnp.exp(
        freq * lower[..., None]
    )
    prob = ((ac / freq) * exp_diff).real.sum(axis=-1) / dc + (
        upper - lower
    ) / 2.0
    return jnp.clip(prob, 1e-20, None)

  def bin_bits(
      self, center: ArrayLike, temperature: Optional[ArrayLike] = None
  ) -> Array:
    return -jnp.log(self.bin_prob(center, temperature))
