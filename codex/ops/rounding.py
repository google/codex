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

from jax import numpy as jnp


def soft_round(x, temperature):
  """Differentiable approximation to `round`.

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
  if temperature < 1e-4:
    return jnp.around(x)
  if temperature > 1e4:
    return x
  m = jnp.floor(x) + .5
  z = 2 * jnp.tanh(.5 / temperature)
  r = jnp.tanh((x - m) / temperature) / z
  return m + r


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
  if temperature < 1e-4:
    return jnp.ceil(x) - .5
  if temperature > 1e4:
    return x
  m = jnp.floor(x) + .5
  z = 2 * jnp.tanh(.5 / temperature)
  r = jnp.arctanh((x - m) * z) * temperature
  return m + r


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
