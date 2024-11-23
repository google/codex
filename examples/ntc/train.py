"""Runs an NTC training loop."""

import os
from absl import app
from absl import flags
from absl import logging
import jax
from ml_collections import config_flags
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import train_lib

config_flags.DEFINE_config_file("config")
flags.DEFINE_string(
    "checkpoint_path", "/tmp/train",
    "Directory where to write checkpoints.")
flags.DEFINE_string(
    "start_path", None,
    "Directory to read initial checkpoint from.")

FLAGS = flags.FLAGS


def load_training_set(patch_size, batch_size, shuffle_size):
  """Returns a tf.Dataset with training images."""
  def image_filter(item):
    shape = tf.shape(item["image"])
    return ((shape[0] >= patch_size) and
            (shape[1] >= patch_size) and
            (shape[2] == 3))

  def image_preprocess(item):
    """Preprocesses an image from the CLIC dataset."""
    image = item["image"]
    shape = tf.cast(tf.shape(image), dtype=tf.float32)
    min_factor = float(patch_size) / tf.math.minimum(shape[0], shape[1])
    scale_factor = tf.random.uniform((), minval=min_factor, maxval=1.)
    shape = scale_factor * shape[:2]
    shape = tf.math.minimum(tf.cast(tf.round(shape), tf.int32), patch_size)
    image = tf.image.resize(image, shape, method="bilinear", antialias=True)
    image = tf.image.random_crop(image, (patch_size, patch_size, 3))
    image = tf.transpose(image, (2, 0, 1)) / 255
    return image

  ds = tfds.load("clic", split="train", shuffle_files=True)
  ds = ds.repeat()
  ds = ds.filter(image_filter)
  ds = ds.map(image_preprocess)
  ds = ds.shuffle(shuffle_size)
  ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.prefetch(2)
  return ds


def main(_):
  tf.config.experimental.set_visible_devices([], "GPU")
  logging.info(
      "JAX devices: %s, TF devices: %s",
      jax.devices(),
      tf.config.get_visible_devices(),
  )

  jax.config.update("jax_debug_nans", FLAGS.config.debug_nans)

  host_count = jax.process_count()
  # host_id = jax.process_index()
  local_device_count = jax.local_device_count()
  logging.info(
      "Device count: %d, host count: %d, local device count: %d",
      jax.device_count(),
      host_count,
      local_device_count,
  )

  seed, = np.frombuffer(os.getrandom(8), dtype=np.int64)
  rng = jax.random.PRNGKey(seed)

  train_set = load_training_set(
      FLAGS.config.patch_size, FLAGS.config.batch_size,
      FLAGS.config.shuffle_size)
  train_iterator = train_set.as_numpy_iterator()

  train_lib.train(
      FLAGS.config,
      FLAGS.checkpoint_path,
      train_iterator,
      rng,
      start_path=FLAGS.start_path,
  )


if __name__ == "__main__":
  app.run(main)
