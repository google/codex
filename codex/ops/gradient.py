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
"""Special gradient operations."""

import jax


def perturb_and_apply(f, x, u, *args):
  """Perturbs the inputs of a pointwise function using JAX.

  This function adds uniform noise in the range -0.5 to 0.5 to the first
  argument of the given function.
  It further replaces derivatives of the function with (analytically computed)
  expected derivatives w.r.t. the noise.

  This is further described in Sec. 4.2. in the paper:
  > "Universally Quantized Neural Compression"<br />
  > Eirikur Agustsson & Lucas Theis<br />
  > https://arxiv.org/abs/2006.09952

  Args:
    f: Callable. JAX transformable pointwise function.
    x: Array. The inputs.
    u: Array. The noise realization to perturb x with.
    *args: Optional, additional arguments of f.

  Returns:
   A tuple (y, x+u) where y=(f(x+u, *args), f'(x+u, *args)), and u is uniform
   noise. The gradient of `f` w.r.t. `x` uses expected derivatives w.r.t.
   the distribution of u.
  """

  @jax.custom_jvp
  def _perturb_and_apply(x, args):
    return f(x + u, *args)

  @_perturb_and_apply.defjvp
  def _perturb_and_apply_jvp(primals, tangents):
    x, args = primals
    grad_x, grad_args = tangents
    dy = (f(x + 0.5, *args) - f(x - 0.5, *args)) * grad_x
    dy += jax.jvp(lambda t: f(x + u, *t), (args,), (grad_args,))[1]
    return _perturb_and_apply(x, args), dy

  return _perturb_and_apply(x, args)
