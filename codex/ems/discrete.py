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
"""Entropy models of discrete distributions."""

from typing import Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp

# TODO(jonycgn): Think through shape contracts and broadcasting for all methods
# of this interface.


class DiscreteEntropyModel(nn.Module):
  """Entropy model for discrete distributions."""

  @property
  def logits(self):
    """Logits of a categorical distribution representing this entropy model."""
    raise NotImplementedError("Subclass must define logits.")

  def prob(self):
    """Returns probability mass.

    Returns:
      P(x)
    """
    return jax.nn.softmax(self.logits, axis=-1)

  def bits(self, logits=None, index=None, temperature=1.):
    """Computes information content in bits.

    Args:
      logits: n-D Array. Logits representing concrete distribution Q.
      index: n-D Array. Samples from Q.
      temperature: Scalar. Temperature parameter for Q. Not used when `index` is
        given.

    Returns:
      -sum_i Q_{logits}(i, temperature) log_2 P(i), or -log_2 P(index)
    """
    if logits is None == index is None:
      raise ValueError("Need exactly one of {index, logits}.")

    bits = jax.nn.log_softmax(self.logits, axis=-1) / -jnp.log(2.)

    def monte_carlo(index):
      index = jnp.expand_dims(index, axis=-1)
      take = jnp.expand_dims(bits, range(len(index.shape) - len(bits.shape)))
      take = jnp.take_along_axis(take, index, axis=-1)
      return jnp.squeeze(take, axis=-1)

    def analytical(logits):
      probs = jax.nn.softmax(logits / temperature, axis=-1)
      return jnp.einsum("...i,...i->...", probs, bits)

    def low_temperature(logits):
      return monte_carlo(jnp.argmax(logits, axis=-1))

    if index is not None:
      return monte_carlo(index)
    return jax.lax.cond(temperature < 1e-4, low_temperature, analytical, logits)


class LearnedDiscreteEntropyModel(DiscreteEntropyModel):
  """A discrete entropy model with learned probabilities."""
  cardinality: int
  shape: Tuple[int, ...]

  def setup(self):
    super().setup()
    self._logits = self.param(
        "_logits",
        jax.nn.initializers.uniform(1.),
        (*self.shape, self.cardinality))

  @property
  def logits(self):
    return self._logits
