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
"""Base class for continuous distribution entropy models."""

import flax.linen as nn
import jax.numpy as jnp

# TODO(jonycgn): Think through shape contracts and broadcasting for all methods
# of this interface.


def logsum_expbig_minus_expsmall(big, small):
  """Numerically stable evaluation of `log(exp(big) - exp(small))`."""
  return jnp.log1p(-jnp.exp(small - big)) + big


class ContinuousEntropyModel(nn.Module):
  """Entropy model for continuous random variables."""

  def bin_prob(self, center, temperature=jnp.inf):
    """Computes probability mass of unit-width quantization bins.

    Args:
      center: n-D Array. Locations of quantization bin centers.
      temperature: Scalar. Temperature parameter for soft quantization.

    This function models the distribution of the bottleneck tensor after it is
    subjected to (soft or hard) quantization. In a nutshell:

    - `temperature == inf` corresponds to dithered quantization (or
      equivalently, additive uniform noise). In this case, this function is
      typically continuously differentiable.

    - `temperature == 0` corresponds to hard quantization (rounding). In this
      case, this function is piecewise constant for every unit-width interval
      around an integer location.

    Values between 0 and infinity interpolate between the two cases. If `center`
    only contains integer values, the temperature parameter has no effect.

    Returns:
      `E_u p(Q(center, temperature) + u)`, where `Q` is the soft rounding
      function, `u` is additive uniform noise, and `E_u` is the expectation
      with respect to `u`.
    """
    # Default implementation may work in most cases, but may be overridden for
    # performance/stability reasons.
    return 2 ** -self.bin_bits(center, temperature)

  def bin_bits(self, center, temperature=jnp.inf):
    """Computes information content of unit-width quantization bins in bits.

    Args:
      center: n-D Array. Locations of quantization bin centers.
      temperature: Scalar. Temperature parameter for soft quantization.

    This function models the distribution of the bottleneck tensor after it is
    subjected to (soft or hard) quantization. In a nutshell:

    - `temperature == inf` corresponds to dithered quantization (or
      equivalently, additive uniform noise). In this case, this function is
      typically continuously differentiable.

    - `temperature == 0` corresponds to hard quantization (rounding). In this
      case, this function is piecewise constant for every unit-width interval
      around an integer location.

    Values between 0 and infinity interpolate between the two cases.

    The expected value of this function for `temperature == inf` is an upper
    bound on the expected value for `temperature == 0` (which is the Shannon
    cross entropy).

    For training a compression model, it is recommended to either
    - Train with `temperature = inf`, and then switch to hard quantization for
      inference. Since the loss function is then agnostic to the quantization
      offset, a suitable quantization offset should be determined after
      training using a heuristic, grid search, etc.
    - Anneal `temperature` towards zero during the course of training, and then
      switch to hard quantization for inference. Since the loss function is then
      aware of the quantization offset, it can simply be set to zero during
      inference.

    Returns:
      `-log_2 E_u p(Q(center, temperature) + u)`, where `Q` is the soft rounding
      function, `u` is additive uniform noise, and `E_u` is the expectation with
      respect to `u`.
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
