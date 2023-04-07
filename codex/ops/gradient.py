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
import jax.numpy as jnp


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
    u: Array. The noise realization to perturb `x` with. Must be a sample from a
      uniform distribution.
    *args: Optional, additional arguments of `f`.

  Returns:
    `y=f(x+u, *args)`. The gradient of `y` w.r.t. `x` takes the expectation over
    the derivatives w.r.t. the distribution of `u` in closed form.
  """
  # This is the correct output of the function, and allows automatically
  # computing the gradient wrt. all arguments and closures of f, except x.
  output = f(jax.lax.stop_gradient(x) + u, *args)

  # Capture all closures as extra arguments of a new function. Then disable
  # gradient propagation to all arguments except x. Note: closure_convert fails
  # if f is a module since it tries to hash it, and Flax disallows hashing of
  # modules with variables. So we have to wrap it in a lambda function.
  new_f, extra_args = jax.closure_convert(lambda *a: f(*a), x, *args)  # pylint:disable=unnecessary-lambda
  new_args = jax.lax.stop_gradient(args + tuple(extra_args))

  # Define a function that returns zeros in the forward pass, but defines the
  # closed-form derivative of f with respect to x.
  @jax.custom_jvp
  def zeros_with_df_dx(x):
    return jnp.zeros_like(x)

  @zeros_with_df_dx.defjvp
  def zeros_with_df_dx_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    f_dot = (new_f(x + .5, *new_args) - new_f(x - .5, *new_args)) * x_dot
    return zeros_with_df_dx(x), f_dot

  # Add the custom function (zeros) to the output, so that gradients wrt. x
  # flow through the custom function, and all others flow through f itself.
  return output + zeros_with_df_dx(x)
