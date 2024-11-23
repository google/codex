"""Nonlinear transform coders.

This is a simple reimplementation of two NTC coders in JAX/CoDeX.

`FactorizedPriorModel` is an implementation of the fully factorized model
published in:

J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston
"Variational Image Compression with a Scale Hyperprior"
https://openreview.net/forum?id=rkcQFMZRb

`HyperPriorModel` is an implementation of the mean-scale hyperprior model
published in:

D. Minnen, J. Ballé, G. D. Toderici:
"Joint Autoregressive and Hierarchical Priors for Learned Image Compression"
https://proceedings.neurips.cc/paper/2018/hash/53edebc543333dfbf7c5933af792c9c4-Abstract.html

Both models make use of the soft-rounding implemented in CoDeX, as published in:

E. Agustsson, L. Theis:
"Universally Quantized Neural Compression"
https://proceedings.neurips.cc/paper/2020/hash/92049debbe566ca5782a3045cf300a3c-Abstract.html
"""

import codex
from codex.ems import equinox as ems
import distrax
import equinox as eqx
import jax
from jax import numpy as jnp


Array = jax.Array


class AnalysisTransform(eqx.nn.Sequential):

  def __init__(self, rng, in_channels, out_channels):
    rng = jax.random.split(rng, 3)
    super().__init__([
        eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            stride=4,
            padding=2,
            use_bias=True,
            key=rng[0],
        ),
        eqx.nn.Lambda(jax.nn.leaky_relu),
        eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            use_bias=True,
            key=rng[1],
        ),
        eqx.nn.Lambda(jax.nn.leaky_relu),
        eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            use_bias=True,
            key=rng[2],
        ),
    ])


class SynthesisTransform(eqx.nn.Sequential):

  def __init__(self, rng, in_channels, out_channels):
    rng = jax.random.split(rng, 3)
    super().__init__([
        eqx.nn.ConvTranspose(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            use_bias=True,
            key=rng[0],
        ),
        eqx.nn.Lambda(jax.nn.leaky_relu),
        eqx.nn.ConvTranspose(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            use_bias=True,
            key=rng[1],
        ),
        eqx.nn.Lambda(jax.nn.leaky_relu),
        eqx.nn.ConvTranspose(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            stride=4,
            padding=2,
            output_padding=3,
            use_bias=True,
            key=rng[2],
        ),
    ])


class HyperAnalysisTransform(eqx.nn.Sequential):

  def __init__(self, rng, in_channels, out_channels):
    rng = jax.random.split(rng, 2)
    super().__init__([
        eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            use_bias=True,
            key=rng[0],
        ),
        eqx.nn.Lambda(jax.nn.leaky_relu),
        eqx.nn.Conv(
            num_spatial_dims=2,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            use_bias=True,
            key=rng[1],
        ),
    ])


class HyperSynthesisTransform(eqx.nn.Sequential):

  def __init__(self, rng, in_channels, out_channels):
    rng = jax.random.split(rng, 2)
    super().__init__([
        eqx.nn.ConvTranspose(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            use_bias=True,
            key=rng[0],
        ),
        eqx.nn.Lambda(jax.nn.leaky_relu),
        eqx.nn.ConvTranspose(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            use_bias=True,
            key=rng[1],
        ),
    ])


class ConditionalLogisticEntropyModel(ems.DistributionEntropyModel, eqx.Module):
  scale_param: Array

  @property
  def distribution(self):
    return distrax.Logistic(loc=0, scale=ems.scale_param(self.scale_param, 20))


class FactorizedPriorModel(eqx.Module):
  """Nonlinear transform coder with factorized entropy model."""
  analysis: eqx.Module
  synthesis: eqx.Module
  em_y: ems.ContinuousEntropyModel

  def __init__(self, rng, x_channels, y_channels, em_y="fourier"):
    super().__init__()
    rng = jax.random.split(rng, 3)
    self.analysis = AnalysisTransform(rng[0], x_channels, y_channels)
    self.synthesis = SynthesisTransform(rng[1], y_channels, x_channels)
    em_cls = dict(
        fourier=ems.RealMappedFourierEntropyModel,
        deep=ems.DeepFactorizedEntropyModel,
    )[em_y]
    self.em_y = em_cls(
        rng=rng[2],
        num_pdfs=y_channels,
    )

  def __call__(self, x, rng=None, t=None):
    num_pixels = x.shape[-1] * x.shape[-2]

    y = self.analysis(x)

    if rng is not None:
      y = codex.ops.soft_round(y, t)
      y += jax.random.uniform(rng, y.shape, minval=-.5, maxval=.5)
    else:
      y = jnp.round(y)
    rate = self.em_y.bin_bits(y.transpose((1, 2, 0))).sum() / num_pixels
    if rng is not None:
      y = codex.ops.soft_round_conditional_mean(y, t)

    x_rec = self.synthesis(y)
    x_rec = x_rec[:, :x.shape[-2], :x.shape[-1]]

    distortion = jnp.square(x - x_rec).sum() / num_pixels

    return x_rec, dict(
        rate=rate,
        distortion=distortion,
    )


class HyperPriorModel(eqx.Module):
  """Nonlinear transform coder with hyperprior entropy model."""
  analysis: eqx.Module
  synthesis: eqx.Module
  hyper_analysis: eqx.Module
  hyper_synthesis: eqx.Module
  em_z: ems.ContinuousEntropyModel

  def __init__(self, rng, x_channels, y_channels, z_channels, em_z="fourier"):
    super().__init__()
    rng = jax.random.split(rng, 5)
    self.analysis = AnalysisTransform(
        rng[0], x_channels, y_channels)
    self.synthesis = SynthesisTransform(
        rng[1], y_channels, x_channels)
    self.hyper_analysis = HyperAnalysisTransform(
        rng[2], y_channels, z_channels)
    self.hyper_synthesis = HyperSynthesisTransform(
        rng[3], z_channels, 2 * y_channels)
    em_cls = dict(
        fourier=ems.RealMappedFourierEntropyModel,
        deep=ems.DeepFactorizedEntropyModel,
    )[em_z]
    self.em_z = em_cls(
        rng=rng[4],
        num_pdfs=z_channels,
    )

  def __call__(self, x, rng=None, t=None):
    num_pixels = x.shape[-1] * x.shape[-2]

    y = self.analysis(x)
    z = self.hyper_analysis(y)

    if rng is not None:
      rng = jax.random.split(rng, 2)
      z = codex.ops.soft_round(z, t)
      z += jax.random.uniform(rng[0], z.shape, minval=-.5, maxval=.5)
    else:
      z = jnp.round(z)
    rate_z = self.em_z.bin_bits(z.transpose((1, 2, 0))).sum() / num_pixels
    if rng is not None:
      z = codex.ops.soft_round_conditional_mean(z, t)

    offset, scale = jnp.split(
        self.hyper_synthesis(z)[:, :y.shape[-2], :y.shape[-1]], 2, axis=0)
    em_y = ConditionalLogisticEntropyModel(scale)

    if rng is not None:
      y = codex.ops.soft_round(y - offset, t)
      y += jax.random.uniform(rng[1], y.shape, minval=-.5, maxval=.5)
    else:
      y = jnp.round(y - offset)
    rate_y = em_y.bin_bits(y).sum() / num_pixels
    if rng is not None:
      y = codex.ops.soft_round_conditional_mean(y, t) + offset
    else:
      y += offset

    x_rec = self.synthesis(y)
    x_rec = x_rec[:, :x.shape[-2], :x.shape[-1]]

    rate = rate_y + rate_z

    distortion = jnp.square(x - x_rec).sum() / num_pixels

    return x_rec, dict(
        rate=rate,
        rate_y=rate_y,
        rate_z=rate_z,
        distortion=distortion,
    )


def loss_fn(model, x, lmbda, rng, t):
  _, metrics = model(x, rng, t)
  loss = metrics["rate"] + lmbda * metrics["distortion"]
  metrics.update(loss=loss)
  return loss, metrics


def batched_loss_fn(model, x, lmbda, rng, t):
  _, metrics = jax.vmap(model, (0, 0, None))(x, rng, t)
  metrics = jax.tree.map(jnp.mean, metrics)
  loss = metrics["rate"] + lmbda * metrics["distortion"]
  metrics.update(loss=loss)
  return loss, metrics
