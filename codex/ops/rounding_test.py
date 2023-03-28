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
"""Tests for rounding operations."""

import chex
from codex.ops import rounding
import jax
import jax.numpy as jnp
import pytest

# TODO(jonycgn): Improve unit tests.


@pytest.mark.parametrize("t", [.2, 1., jnp.inf])
def test_soft_round_inverse_is_consistent(t):
  x = jnp.linspace(.9, 2.1, 50)
  y = rounding.soft_round_inverse(rounding.soft_round(x, t), t)
  chex.assert_trees_all_close(x, y)


def test_ste_rounds():
  x = jnp.linspace(-1.2, 3.4, 50)
  chex.assert_trees_all_close(rounding.ste_round(x), jnp.round(x))


def test_ste_gradients_are_identity():
  ste_grad = jax.grad(lambda x: rounding.ste_round(x).sum())
  dydx = ste_grad(jnp.linspace(-1.2, 3.4, 50))
  chex.assert_trees_all_close(dydx, jnp.ones((50,)))
