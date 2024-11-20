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
"""Implementation of Wasserstein Distortion.

Please refer to https://arxiv.org/abs/2310.03629 for details.
"""

import collections
import functools
from typing import Any, Collection, Union
from codex.ops import gradient
import jax
from jax import numpy as jnp


Array = jax.Array


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def safe_sqrt(x: Array, limit: float) -> Array:
  """`jnp.sqrt` with capped gradient."""
  del limit  # unused in forward pass
  return jnp.sqrt(x)


@safe_sqrt.defjvp
def safe_sqrt_jvp(limit, primals, tangents):
  x, = primals
  x_dot, = tangents
  sqrt_dot = jnp.minimum(.5 / jnp.sqrt(x), limit) * x_dot
  return safe_sqrt(x, limit), sqrt_dot


def lowpass(inputs: Array, stride: int) -> Array:
  """Lowpass filters an array of shape (batch, height, width).

  Args:
    inputs: The input array of shape (batch, height, width).
    stride: The stride length of the convolution. Typically either 1 or 2.

  Returns:
    The lowpass filtered array of shape (batch, height, width). Height and width
    are the same as the input array if stride is 1.
  """
  # Lowpass filter. Weights sum to one so that the output dynamic range is the
  # same as the input's. Its frequency response will cause some alias when
  # downsampling, but we accept that because a 3x3 filter is reasonably fast.
  kernel = jnp.array([.25, .5, .25])
  kernel = jnp.outer(kernel, kernel)[None, None]  # shape: (1, 1, 3, 3)
  return jax.lax.conv(
      jnp.expand_dims(inputs, 1),
      kernel,
      window_strides=(stride, stride),
      padding="same",
  ).squeeze(1)


def compute_multiscale_stats(
    features: Array,
    num_levels: int,
) -> tuple[list[Array], list[Array]]:
  """Computes local mean and variance of a feature array."""
  squared = jnp.square(features)
  means = []
  variances = []
  for _ in range(num_levels):
    m = lowpass(features, stride=1)
    p = lowpass(squared, stride=1)
    means.append(m)
    variances.append(p - jnp.square(m))
    features = m[..., ::2, ::2]
    squared = p[..., ::2, ::2]
  return means, variances


def wasserstein_distortion(
    features_a: Array,
    features_b: Array,
    log2_sigma: Array,
    num_levels: int = 5,
    sqrt_grad_limit: float = 1e6,
    return_intermediates: bool = False,
) -> Union[Array, tuple[Array, dict[str, Any]]]:
  """Evaluates Wasserstein Distortion between two feature arrays.

  Args:
    features_a: Array, shape `(channels, height, width)`. The first feature
      array to be compared.
    features_b: Array, shape `(channels, height, width)`. The second feature
      array to be compared.
    log2_sigma: Array, shape `(height, width)`. The base two logarithm of the
      sigma map, which indicates the amount of summarization in each location.
      Must have the same height and width as the feature arrays.
    num_levels: Integer. The number of multi-scale levels of the feature
      statistics to compute. Must be greater or equal to the maximum of
      `log2_sigma`.
    sqrt_grad_limit: Float. Upper limit for the gradient of the square root
      applied to the empirical feature variance estimates, for numerical
      stability.
    return_intermediates: Boolean. If `True`, returns intermediate computations
      in a dictionary, besides the distortion value.

  Returns:
    Distortion value, except if `return_intermediates`, then returns a tuple of
    distortion value and a dictionary containing intermediate computations.
  """
  if features_a.shape != features_b.shape:
    raise ValueError(
        f"`features_a` and `features_b` must have same shape, but received "
        f"{features_a.shape} and {features_b.shape}, respectively.")
  if features_a.shape[-2:] != log2_sigma.shape:
    raise ValueError(
        f"features and `log2_sigma` must have same spatial shape, but received "
        f"{features_a.shape[-2:]} and {log2_sigma.shape}, respectively.")
  means_a, variances_a = compute_multiscale_stats(features_a, num_levels)
  means_b, variances_b = compute_multiscale_stats(features_b, num_levels)
  assert len(means_a) == len(means_b) == len(variances_a) == len(variances_b)

  wd_maps = [jnp.square(features_a - features_b)]
  for ma, mb, va, vb in zip(means_a, means_b, variances_a, variances_b):
    assert ma.shape == mb.shape == va.shape == vb.shape
    # Variance estimates can turn out slightly negative due to numerics. This
    # brings such estimates up to zero, but passes through a useful gradient.
    va = gradient.lower_limit(va, 0)
    vb = gradient.lower_limit(vb, 0)
    # The square root has unbounded gradients near zero. This limits the
    # gradient to a finite value.
    sa = safe_sqrt(va, sqrt_grad_limit)
    sb = safe_sqrt(vb, sqrt_grad_limit)
    wd_maps.append(jnp.square(ma - mb) + jnp.square(sa - sb))
  assert len(wd_maps) == num_levels + 1

  dist = jnp.zeros(())
  intermediates = collections.defaultdict(list)
  intermediates.update(wd_maps=wd_maps)
  for i, wd_map in enumerate(wd_maps):
    assert wd_map.shape[-2:] == log2_sigma.shape
    weight = jax.nn.relu(1 - abs(log2_sigma - i))
    intermediates["weights"].append(weight)
    dist += jnp.mean(weight * wd_map)
    if i > 0:
      log2_sigma = lowpass(log2_sigma[None], stride=2).squeeze(0)

  if return_intermediates:
    return dist, intermediates
  return dist


def multi_wasserstein_distortion(
    features_a: Collection[Array],
    features_b: Collection[Array],
    log2_sigma: Array,
    num_levels: int = 5,
    sqrt_grad_limit: float = 1e6,
    return_intermediates: bool = False,
) -> Union[Array, tuple[Array, dict[str, Any]]]:
  """Wasserstein Distortion between multiple feature arrays of two images.

  This function accepts more than one feature array per image. The arrays don't
  need to be all the same shape, but the nth array in `features_a` must have the
  same shape as the nth array in `features_b`. The aspect ratio of all the
  arrays should be approximately the same.

  Args:
    features_a: Multiple feature arrays of format `(channels, height, width)`,
      corresponding to the first image to be compared.
    features_b: Multiple feature arrays of format `(channels, height, width)`,
      corresponding to the second image to be compared.
    log2_sigma: Array, shape `(height, width)`. The base two logarithm of the
      sigma map, which indicates the amount of summarization in each location.
      Doesn't have to have the same shape as the feature arrays.
    num_levels: Integer. The number of multi-scale levels of the feature
      statistics to compute. Must be greater or equal to the maximum of
      `log2_sigma`.
    sqrt_grad_limit: Float. Upper limit for the gradient of the square root
      applied to the empirical feature variance estimates, for numerical
      stability.
    return_intermediates: Boolean. If `True`, returns intermediate computations
      in a dictionary, besides the distortion value.

  Returns:
    Distortion value, except if `return_intermediates`, then returns a tuple of
    distortion value and a dictionary containing intermediate computations.
  """
  if len(features_a) != len(features_b):
    raise ValueError(
        f"`features_a` and `features_b` must have same length, but received "
        f"{len(features_a)} and {len(features_b)}, respectively.")

  dist = jnp.zeros(())
  intermediates = collections.defaultdict(list)
  for fa, fb in zip(features_a, features_b):
    if fa.shape != fb.shape:
      raise ValueError(
          f"Found feature arrays with incompatible shapes. "
          f"A: {fa.shape}, B: {fb.shape}.")
    # Resize the sigma map to match the feature arrays.
    ls = jax.image.resize(log2_sigma, fa.shape[-2:], "linear", antialias=True)
    # Rescale sigma to match the feature arrays. For example, if a feature array
    # has a very low spatial resolution, we make sigma correspondingly smaller,
    # because each element in the feature array covers a larger portion of the
    # image. Since we are in log space, we subtract the log of the size ratio
    # and then cap at zero.
    log_ratio_h = jnp.log2(log2_sigma.shape[-2] / fa.shape[-2])
    log_ratio_w = jnp.log2(log2_sigma.shape[-1] / fa.shape[-1])
    mean_log_ratio = (log_ratio_h + log_ratio_w) / 2
    ls = jax.nn.relu(ls - mean_log_ratio)
    d, s = wasserstein_distortion(
        fa, fb, ls, num_levels=num_levels, sqrt_grad_limit=sqrt_grad_limit,
        return_intermediates=True)
    dist += d
    for i in s:
      intermediates[i].append(s[i])

  if return_intermediates:
    return dist, intermediates
  return dist
