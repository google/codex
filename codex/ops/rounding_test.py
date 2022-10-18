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
import jax.numpy as jnp
import pytest

# TODO(jonycgn): Improve unit tests.


@pytest.mark.parametrize("t", [.2, 1., jnp.inf])
def test_inverse_is_consistent(t):
  x = jnp.linspace(.9, 2.1, 50)
  y = rounding.soft_round_inverse(rounding.soft_round(x, t), t)
  chex.assert_trees_all_close(x, y)
