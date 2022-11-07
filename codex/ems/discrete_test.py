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
"""Tests for discrete entropy model."""

from codex.ems import discrete
import jax


# TODO(jonycgn): Improve unit tests.


def test_can_instantiate_and_evaluate_scalar():
  rng = jax.random.PRNGKey(0)
  em = discrete.LearnedDiscreteEntropyModel(5, ())
  em = em.bind(em.init(rng, method=em.prob))
  assert em.prob().shape == (5,)
  index = jax.random.randint(rng, (3, 4), 0, 4)
  assert em.bits(index=index).shape == (3, 4)
  logits = jax.random.normal(rng, (3, 4, 5))
  assert em.bits(logits=logits).shape == (3, 4)
  assert em.bits(logits=logits, temperature=0).shape == (3, 4)


def test_can_instantiate_and_evaluate_array():
  rng = jax.random.PRNGKey(0)
  em = discrete.LearnedDiscreteEntropyModel(5, (3,))
  em = em.bind(em.init(rng, method=em.prob))
  assert em.prob().shape == (3, 5)
  index = jax.random.randint(rng, (4, 3), 0, 4)
  assert em.bits(index=index).shape == (4, 3)
  logits = jax.random.normal(rng, (4, 3, 5))
  assert em.bits(logits=logits).shape == (4, 3)
  assert em.bits(logits=logits, temperature=0).shape == (4, 3)
