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

from jax import lax


def soft_round(x, alpha, eps=1e-3):
  """Differentiable approximation to round().

  Larger alphas correspond to closer approximations of the round function.
  If alpha is close to zero, this function reduces to the identity.
  This is described in Sec. 4.1. in the paper
  > "Universally Quantized Neural Compression"<br />
  > Eirikur Agustsson & Lucas Theis<br />
  > https://arxiv.org/abs/2006.09952
  Args:
    x: Tensor. Inputs to the rounding function.
    alpha: Tensor. Controls smoothness of the approximation.
    eps: Float. Threshold below which soft_round() will return identity.

  Returns:
    Tensor
  """
  # This guards the gradient of tf.where below against NaNs, while maintaining
  # correctness, as for alpha < eps the result is ignored.
  alpha_bounded = lax.max(alpha, eps)

  m = lax.floor(x) + .5
  r = x - m
  z = lax.tanh(alpha_bounded / 2.) * 2.
  y = m + lax.tanh(alpha_bounded * r) / z

  # For very low alphas, soft_round behaves like identity
  return lax.select(alpha < eps, x, y)


def soft_round_inverse(y, alpha, eps=1e-3):
  """Inverse of soft_round().

  This is described in Sec. 4.1. in the paper
  > "Universally Quantized Neural Compression"<br />
  > Eirikur Agustsson & Lucas Theis<br />
  > https://arxiv.org/abs/2006.09952
  Args:
    y: tf.Tensor. Inputs to this function.
    alpha: Float or tf.Tensor. Controls smoothness of the approximation.
    eps: Float. Threshold below which soft_round() is assumed to equal the
      identity function.

  Returns:
    tf.Tensor
  """
  # This guards the gradient of tf.where below against NaNs, while maintaining
  # correctness, as for alpha < eps the result is ignored.
  alpha_bounded = lax.max(alpha, eps)
  m = lax.floor(y) + .5
  s = (y - m) * (lax.tanh(alpha_bounded / 2.) * 2.)
  r = lax.atanh(s) / alpha_bounded
  # `r` must be between -.5 and .5 by definition. In case atanh becomes +-inf
  # due to numerical instability, this prevents the forward pass from yielding
  # infinite values. Note that it doesn't prevent the backward pass from
  # returning non-finite values.
  r = lax.clamp(-.5, r, .5)

  # For very low alphas, soft_round behaves like identity.
  return lax.select(alpha < eps, y, m + r)


def soft_round_conditional_mean(inputs, alpha):
  """Conditional mean of inputs given noisy soft rounded values.

  Computes g(z) = E[Y | s(Y) + U = z] where s is the soft-rounding function,
  U is uniform between -0.5 and 0.5 and `Y` is considered uniform when truncated
  to the interval [z-0.5, z+0.5].
  This is described in Sec. 4.1. in the paper
  > "Universally Quantized Neural Compression"<br />
  > Eirikur Agustsson & Lucas Theis<br />
  > https://arxiv.org/abs/2006.09952
  Args:
    inputs: The input tensor.
    alpha: The softround alpha.

  Returns:
    The conditional mean, of same shape as `inputs`.
  """
  return soft_round_inverse(inputs - .5, alpha) + .5
