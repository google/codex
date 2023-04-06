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
"""Tests for quantization operations."""

import chex
from codex.ops import quantization
import jax
from jax import nn
from jax import random
import jax.numpy as jnp
import pytest

# TODO(jonycgn): Improve unit tests.


@pytest.mark.parametrize("t", [0.2, 1.0, jnp.inf])
def test_soft_round_is_inverted_correctly(t):
  x = jnp.linspace(0.9, 2.1, 50)
  y = quantization.soft_round_inverse(quantization.soft_round(x, t), t)
  chex.assert_trees_all_close(x, y)


def test_large_temperature_soft_round_consistent_with_identity():
  x = jnp.linspace(-2.0, 2.0, 6)
  y = quantization.soft_round(x, jnp.inf)
  chex.assert_trees_all_close(x, y)


def test_low_temperature_soft_round_near_half_integer():
  # Test that soft round is accurate exactly near half-integer values
  for offset in range(-5, 5):
    x = jnp.linspace(offset - 0.499, offset + 0.499, 100)
    y = quantization.soft_round(x, 0.0005)
    chex.assert_trees_all_close(jnp.round(x), y, atol=0.05)


def test_ste_round_is_consistent_with_jnp_round():
  x = jnp.linspace(-1.2, 3.4, 50)
  chex.assert_trees_all_close(quantization.ste_round(x), jnp.round(x))


def test_ste_round_gradient_is_identity():
  ste_grad = jax.grad(lambda x: quantization.ste_round(x).sum())
  dydx = ste_grad(jnp.linspace(-1.2, 3.4, 50))
  chex.assert_trees_all_close(dydx, jnp.ones((50,)))


@pytest.mark.parametrize("t", [1e-1, 1e0, 1e1])
def test_ste_argmax_gradient_is_computed_correctly(t):
  softmax = lambda l, t: nn.softmax(l / t, axis=-1)
  x = random.uniform(random.PRNGKey(0), (3, 4, 5))
  primals = (x, t)
  tangents = (jnp.ones_like(x), jnp.zeros_like(t))
  _, grad_sm = jax.jvp(softmax, primals, tangents)
  _, grad_ste_am = jax.jvp(quantization.ste_argmax, primals, tangents)
  chex.assert_trees_all_close(grad_sm, grad_ste_am)
