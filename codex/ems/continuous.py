# Copyright 2022 CoDeX authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base class for entropy models of continuous distributions."""

from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

# TODO(jonycgn): Think through shape contracts and broadcasting for all methods
# of this interface.


ArrayLike = jax.typing.ArrayLike


def logsum_expbig_minus_expsmall(big, small):
  """Numerically stable evaluation of `log(exp(big) - exp(small))`."""
  return jnp.log1p(-jnp.exp(small - big)) + big


class ContinuousEntropyModel(nn.Module):
  """Entropy model for continuous random variables."""

  def bin_prob(
      self, center: jax.Array, temperature: Optional[ArrayLike] = None
  ):
    """Computes probability mass of bins, see `bin_bits` for explanation."""
    # Default implementation may work in most cases, but may be overridden for
    # performance/stability reasons.
    return 2 ** -self.bin_bits(center, temperature)

  def bin_bits(
      self, center: jax.Array, temperature: Optional[ArrayLike] = jnp.inf
  ):
    """Computes information content of unit-width quantization bins in bits.

    Args:
      center: n-D Array. Locations of quantization bin centers.
      temperature: Scalar. Temperature parameter for soft quantization.
        Should be `None` or `jnp.inf` if no soft quantization was used.

    This function models the distribution of the bottleneck tensor AFTER it has
    been subjected to (soft or hard) quantization.  NOTE: the function does not
    quantize the `center`, this is the responsibility of the caller.

    Use cases:

      em = ConditionalEntropyModel(...)
      y = ...  # Some continuous valued data.

      # Case A: caller uses noise during training.
      u = jax.random.uniform(rng, y.shape, minval=-0.5, maxval=0.5)
      y_bits = cdx.ops.perturb_and_apply(em.bin_bits, y, u)

      # Case B: caller uses soft round.
      temperature = ...
      y_hat = cdx.ops.soft_round(y, temperature)
      y_bits = em.bin_bits(y_hat_inv, temperature)

    For training a compression model, it is recommended to either
    - Train without soft rounding (`temperature = None`), and then switch to
      hard quantization for inference. Since the loss function is then agnostic
      to the quantization offset, a suitable quantization offset should be
      determined after training using a heuristic, grid search, etc.
    - Anneal `temperature` towards zero during the course of training, and then
      switch to hard quantization for inference. Since the loss function is then
      aware of the quantization offset, it can simply be set to zero during
      inference.

    Returns:
      `-log_2 E_u p(Q(center, temperature) + u)`, where `Q` is the soft rounding
      function, `u` is additive uniform noise, `p` is `self.distribution`, and
      `E_u` is the expectation with respect to `u`.
    """
    raise NotImplementedError()

  def quantization_offset(self):
    """Determines a quantization offset using a heuristic.

    Returns a good heuristic value of `o` to use for `jnp.around(x - o) + o`
    when quantizing.

    According to [1], the optimal quantizer for Laplacian sources centers one
    reconstruction value at the mode of the distribution. This can also be a
    good heuristic for other distributions.

    > [1] "Efficient Scalar Quantization of Exponential and Laplacian Random
    > Variables"<br />
    > G. J. Sullivan<br />
    > https://doi.org/10.1109/18.532878
    """
    raise NotImplementedError()

  def tail_locations(self, tail_mass):
    """Determines approximate tail quantiles.

    Args:
      tail_mass: Scalar between 0 and 1. The total probability mass of the tails
        excluding the central interval between `left` and `right`.

    Returns:
      `(left, right)`, where `left` and `right` are the approximate locations of
      the left and right distribution tails, respectively.
    """
    raise NotImplementedError()
