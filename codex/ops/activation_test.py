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
"""Tests for neural network operations."""

from codex.ops import activation
from jax import random


def test_verysoftplus():
  shape = (1, 2, 3)
  x = 100 * random.normal(random.PRNGKey(0), shape)
  y = activation.verysoftplus(x)
  assert y.shape == shape
  assert (y > 0).all()
