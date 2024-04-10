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
import jax
from jax import nn
import jax.numpy as jnp

Array = jax.Array
ArrayLike = jax.typing.ArrayLike


class FourierSeriesEntropyModel(continuous.ContinuousEntropyModel):
  """Fully factorized entropy model based on Fourier series."""
  real: Array
  imag: Array
  scale: Array
  offset: Array
  num_freqs: int

  def __init__(self, key, num_pdfs: int, num_freqs: int = 10):
    """Initializes the entropy model.

    Args:
      key: Random key for initialization.
      num_pdfs: Integer. The number of distinct scalar PDFs on the right of the
        input array. These are treated as independent, but non-identically
        distributed. The remaining array elements on the left are treated as
        i.i.d. (like in a batch dimension).
      num_freqs: Integer. Number of frequency components of the Fourier series.
    """
    super().__init__()
    self.num_freqs = num_freqs
    self.real, self.imag = 1e-3 * jax.random.normal(
        key, (2, num_pdfs, num_freqs))
    self.scale = jnp.ones((num_pdfs,))
    self.offset = jnp.zeros((num_pdfs,))

  def bin_prob(self,
               center: ArrayLike,
               temperature: Optional[ArrayLike] = None) -> Array:
    center, temperature = self._maybe_upcast((center, temperature))

    coef = jax.lax.complex(self.real, self.imag)
    coef = fourier.autocorrelate(coef)

    # Transformation for soft rounding.
    upper = quantization.soft_round_inverse(center + 0.5, temperature)
    # soft_round is periodic with period 1, so we don't need to call it again.
    lower = upper - 1.0

    # Transformation for real line (offset and scale)
    scale = nn.softplus(self.scale)
    offset = self.offset
    upper = jnp.tanh((upper - offset) / scale)
    lower = jnp.tanh((lower - offset) / scale)

    # The DC coefficient is special: it is the normalizer of the density.
    dc = coef[:, 0].real
    ac = coef[:, 1:]
    freq = (1j * jnp.pi) * jnp.arange(1, self.num_freqs)

    # Note: we can take the real part below because the coefficient sequence is
    # assumed to have Hermitian symmetry, so the Fourier series is always real.
    exp_diff = (jnp.exp(freq * upper[..., None]) -
                jnp.exp(freq * lower[..., None]))
    prob = ((ac / freq) * exp_diff).real.sum(axis=-1) / dc
    prob += (upper - lower) / 2.0
    return jnp.maximum(prob, 1e-20)

  def bin_bits(self,
               center: ArrayLike,
               temperature: Optional[ArrayLike] = None) -> Array:
    return jnp.log(self.bin_prob(center, temperature)) / -jnp.log(2.)
