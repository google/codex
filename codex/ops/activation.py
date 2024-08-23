# Copyright 2024 CoDeX authors.
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
"""Activation functions."""

import jax
from jax import numpy as jnp

Array = jax.Array
ArrayLike = jax.typing.ArrayLike


def verysoftplus(x: ArrayLike) -> Array:
  """An activation function symmetric wrt. division and multiplication.

  Defined as `1+x` for `x > 0` and `1/(1-x)` for `x < 0`. This implies:
  ```
  1/verysoftplus(x) = verysoftplus(-x)
  ```
  Args:
    x: Array, the argument of the function.

  Returns:
    the function value of `x`, evaluated elementwise.
  """
  x_neg = jnp.minimum(x, 0)
  return jnp.where(x > 0, x + 1, 1 / (1 - x_neg))
