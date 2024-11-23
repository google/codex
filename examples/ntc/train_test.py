"""Tests of the training code."""

import jax
from jax import numpy as jnp

import config
import train_lib


def mock_dataset(shape):
  while True:
    yield jnp.zeros(shape)


def test_train(tmp_path):
  c = config.get_config()
  c.num_epochs = 1
  c.num_steps_per_epoch = 1
  c.num_eval_steps = 1
  c.batch_size = 1
  c.patch_size = 256

  iterator = mock_dataset((c.batch_size, 3, c.patch_size, c.patch_size))
  rng = jax.random.PRNGKey(0)

  path = tmp_path / "train"
  train_lib.train(c, path, iterator, rng)
