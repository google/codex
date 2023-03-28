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
"""Rounding operations."""

import jax
from jax import numpy as jnp


def soft_round(x, temperature):
  """Differentiable approximation to `jnp.round`.

  Lower temperatures correspond to closer approximations of the round function.
  For temperatures approaching infinity, this function resembles the identity.

  This function is described in Sec. 4.1 of the paper
  > "Universally Quantized Neural Compression"<br />
  > Eirikur Agustsson & Lucas Theis<br />
  > https://arxiv.org/abs/2006.09952

  The temperature argument is the reciprocal of `alpha` in the paper.

  Args:
    x: Array. Inputs to the function.
    temperature: Float >= 0. Controls smoothness of the approximation.

  Returns:
    Array of same shape as `x`.
  """
  def _soft_round(x, t):
    m = jnp.floor(x) + .5
    z = 2 * jnp.tanh(.5 / t)
    r = jnp.tanh((x - m) / t) / z
    return m + r

  identity_t = lambda z, t: z
  round_t = lambda z, t: jnp.round(z)

  return jax.lax.cond(
      temperature < 1e-4,
      round_t,
      lambda z, t: jax.lax.cond(t > 1e4, identity_t, _soft_round, z, t),
      x, temperature)


def soft_round_inverse(x, temperature):
  """Inverse of `soft_round`.

  This function is described in Sec. 4.1 of the paper
  > "Universally Quantized Neural Compression"<br />
  > Eirikur Agustsson & Lucas Theis<br />
  > https://arxiv.org/abs/2006.09952

  The temperature argument is the reciprocal of `alpha` in the paper.

  Args:
    x: Array. Inputs to the function.
    temperature: Float >= 0. Controls smoothness of the approximation.

  Returns:
    Array of same shape as `x`.
  """
  def _sr_inverse(x, t):
    m = jnp.floor(x) + .5
    z = 2 * jnp.tanh(.5 / t)
    r = jnp.arctanh((x - m) * z) * t
    return m + r

  identity_t = lambda z, t: z
  round_inverse_t = lambda z, t: jnp.ceil(z) - .5

  return jax.lax.cond(
      temperature < 1e-4,
      round_inverse_t,
      lambda z, t: jax.lax.cond(t > 1e4, identity_t, _sr_inverse, z, t),
      x, temperature)


def soft_round_conditional_mean(x, temperature):
  """Conditional mean of inputs given noisy soft rounded values.

  Computes `g(z) = E[X | Q(X) + U = z]` where `Q` is the soft-rounding function,
  `U` is uniform between -0.5 and 0.5 and `X` is considered uniform when
  truncated to the interval `[z - 0.5, z + 0.5]`.

  This is described in Sec. 4.1. in the paper
  > "Universally Quantized Neural Compression"<br />
  > Eirikur Agustsson & Lucas Theis<br />
  > https://arxiv.org/abs/2006.09952

  Args:
    x: The input tensor.
    temperature: Float >= 0. Controls smoothness of the approximation.

  Returns:
    Array of same shape as `x`.
  """
  return soft_round_inverse(x - .5, temperature) + .5


@jax.custom_jvp
def ste_round(x):
  """`jnp.round` with straight-through gradient estimation."""
  return jnp.round(x)


@ste_round.defjvp
def ste_round_jvp(primals, tangents):
  x, = primals
  x_dot, = tangents
  return ste_round(x), x_dot
