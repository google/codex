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
"""Tests for perturb and apply operations."""

import chex
from codex.ops import gradient
from codex.ops import rounding
import jax
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize("t", ([7.0]))
def test_function_evalation_is_consistent(t):

  def f(x, args):
    return rounding.soft_round(x, args)

  rng = jax.random.PRNGKey(0)
  x = jnp.linspace(-2.0, 2.0, 5)
  u = jax.random.uniform(
      rng, jnp.shape(x), minval=-.5, maxval=.5, dtype=x.dtype)
  y = gradient.perturb_and_apply(f, x, u, t)
  chex.assert_trees_all_close(y, f(x + u, t))


@pytest.mark.parametrize("t", [[7.0]])
def test_gradient_x_calculation_is_correct(t):

  def f(x, args):
    return jnp.multiply(jnp.square(x), args[0])

  t = jnp.asarray(t)
  rng = jax.random.PRNGKey(0)
  x = jnp.linspace(-2.0, 2.0, 5)
  u = jax.random.uniform(
      rng, jnp.shape(x), minval=-.5, maxval=.5, dtype=x.dtype)
  loss = lambda x_: jnp.sum(gradient.perturb_and_apply(f, x_, u, t))
  gradx_1 = jax.grad(loss, argnums=(0,))(x)[0]
  gradx_2 = jnp.diagonal(jax.jacfwd(f)(x, t))
  chex.assert_trees_all_close(gradx_1, gradx_2)


@pytest.mark.parametrize("t", [[7.0, 1.0]])
def test_gradient_args_calculation_is_correct(t):

  def f(x, args):
    return jnp.multiply(jnp.square(x), args[0]) + args[1]

  t = jnp.asarray(t)
  rng = jax.random.PRNGKey(0)
  x = jnp.linspace(-2.0, 2.0, 5)
  u = jax.random.uniform(
      rng, jnp.shape(x), minval=-.5, maxval=.5, dtype=x.dtype)
  loss = lambda t_: jnp.sum(gradient.perturb_and_apply(f, x, u, t_))
  gradx_1 = jax.grad(loss, argnums=(0,))(t)[0]
  gradx_2 = jax.jacfwd(f, argnums=(1,))(x + u, t)[0].T @ jnp.ones_like(x)
  chex.assert_trees_all_close(gradx_1, gradx_2)


@pytest.mark.parametrize("t", [[7.0]])
def test_gradient_x_is_consistent(t):

  def f(x, args):
    return jnp.multiply(jnp.square(x), args[0])

  t = jnp.asarray(t)
  rng = jax.random.PRNGKey(0)
  x = jnp.linspace(-2.0, 2.0, 50)
  u = jax.random.uniform(
      rng, jnp.shape(x), minval=-.5, maxval=.5, dtype=x.dtype)
  loss = lambda x_: jnp.sum(gradient.perturb_and_apply(f, x_, u, t))
  grad_x = jax.grad(loss, argnums=(0,))(x)[0]
  dydx = f(x + 0.5, t) - f(x - 0.5, t)
  chex.assert_trees_all_close(grad_x, dydx)


@pytest.mark.parametrize("t", [])
def test_empty_args(t):

  def f(x):
    return jnp.multiply(jnp.square(x))

  rng = jax.random.PRNGKey(0)
  x = jnp.linspace(-2.0, 2.0, 50)
  u = jax.random.uniform(
      rng, jnp.shape(x), minval=-.5, maxval=.5, dtype=x.dtype)
  loss = lambda x_: jnp.sum(gradient.perturb_and_apply(f, x_, u, t))
  grad_x = jax.grad(loss, argnums=(0,))(x)[0]
  dydx = f(x + 0.5) - f(x - 0.5)
  chex.assert_trees_all_close(grad_x, dydx)
