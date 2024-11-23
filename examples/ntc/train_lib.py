"""Runs an NTC training loop."""

import collections
import math
import os
from absl import logging
import equinox as eqx
import jax
import optax

import ntc


@eqx.filter_jit
def evaluate(model, x):
  return model(x, None, None)


def save_state(path, model, epoch, opt_state):
  state = (model, epoch, opt_state)
  fn_state = f"{path}/state.eqx"
  with open(fn_state, "wb") as f:
    eqx.tree_serialise_leaves(f, state)  # pytype: disable=wrong-arg-types


def load_state(path, model, opt_state=None):
  if opt_state is None:
    state = (model, 0)
  else:
    state = (model, 0, opt_state)
  fn_state = f"{path}/state.eqx"
  with open(fn_state, "rb") as f:
    return eqx.tree_deserialise_leaves(f, state)  # pytype: disable=wrong-arg-types


def instantiate_model(rng, config):
  cls = getattr(ntc, config.model_cls)
  kwargs = config.model_kwargs[config.model_cls]
  return cls(rng, **kwargs)


def checkify(fn):
  error_set = jax.experimental.checkify.all_checks
  error_set -= jax.experimental.checkify.div_checks
  checkified = jax.experimental.checkify.checkify(fn, errors=error_set)

  def new_fn(*args, **kwargs):
    err, result = checkified(*args, **kwargs)
    err.throw()
    return result

  return new_fn


def train(config, checkpoint_path, train_iterator, rng, start_path=None):
  """The main training loop."""
  if start_path is None:
    start_path = checkpoint_path

  lr_schedule = optax.schedules.piecewise_constant_schedule(
      config.learning_rate, {})
  t_schedule = optax.schedules.piecewise_constant_schedule(
      config.temperature, {})
  optimizer = optax.adam(learning_rate=lr_schedule)

  os.makedirs(checkpoint_path, exist_ok=True)

  rng, init_rng = jax.random.split(rng)
  model = instantiate_model(init_rng, config)
  opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
  try:
    model, start_epoch, opt_state = load_state(start_path, model, opt_state)
  except IOError:
    start_epoch = 0

  @eqx.filter_jit
  def train_step(model, opt_state, x, rng):
    logging.info("Compiling train_step.")
    lr = lr_schedule(opt_state[0].count)
    t = t_schedule(opt_state[0].count)
    grad_fn = eqx.filter_grad(ntc.batched_loss_fn, has_aux=True)
    rng = jax.random.split(rng, x.shape[0])
    grads, metrics = grad_fn(model, x, config.lmbda, rng, t)
    update, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, update)
    metrics.update(lr=lr, t=t)
    return model, opt_state, metrics

  @eqx.filter_jit
  def eval_step(model, x):
    logging.info("Compiling eval_step.")
    _, metrics = ntc.batched_loss_fn(model, x, config.lmbda, None, None)
    return {f"val_{k}": v for k, v in metrics.items()}

  if config.checkify:
    train_step = checkify(train_step)
    eval_step = checkify(eval_step)

  for epoch in range(start_epoch, config.num_epochs):
    logging.info("Starting epoch %d.", epoch)
    save_state(checkpoint_path, model, epoch, opt_state)

    metrics = collections.defaultdict(lambda: 0.)
    step_metrics = dict()

    for _ in range(config.num_steps_per_epoch):
      rng, train_rng = jax.random.split(rng)
      model, opt_state, step_metrics = train_step(
          model, opt_state, next(train_iterator), train_rng)
      for k in step_metrics:
        metrics[k] += float(step_metrics[k])
    for k in step_metrics:
      metrics[k] /= config.num_steps_per_epoch

    for _ in range(config.num_eval_steps):
      step_metrics = eval_step(model, next(train_iterator))
      for k in step_metrics:
        metrics[k] += float(step_metrics[k])
    for k in step_metrics:
      metrics[k] /= config.num_eval_steps

    logging.info("Epoch %d metrics: %s", epoch, metrics)

    nan_metrics = [k for k, v in metrics.items() if math.isnan(v)]
    if nan_metrics:
      raise RuntimeError(
          f"Encountered NaN in metrics: {nan_metrics}. Stopping training.")

  save_state(checkpoint_path, model, config.num_epochs, opt_state)
