# Copyright 2023 CoDeX authors.
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
"""RangeAns pybind11 interface."""

from collections.abc import Sequence

import numpy as np


class RangeAnsStack:
  """rANS stack.

  This is a fork of table-based rANS [1] in JpegXL project [2].
  The original ideas belong to the authors of [1] and [2].
  Special thanks to Luca Versari for the directions.

  [1] Duda, "Asymmetric numeral systems: entropy coding combining speed of
  Huffman coding with compression rate of arithmetic coding", 2013.
  https://arxiv.org/abs/1311.2540
  [2] - https://gitlab.com/wg1/jpeg-xl/-/blob/master/lib/jxl/ans_common.h
  """

  def __init__(self, initial_state: int = 0, *, reserve: int = 0): ...

  @staticmethod
  def make_decoder(qmfs : Sequence[np.ndarray]) -> np.ndarray: ...

  @staticmethod
  def make_encoder(decoder: np.ndarray) -> np.ndarray: ...

  @property
  def state(self) -> int: ...

  def pop(self,
          decoder: np.ndarray,
          index: np.ndarray) -> np.ndarray: ...

  def push(self,
           encoder: np.ndarray,
           index: np.ndarray,
           value: np.ndarray) -> None: ...

  # A variant pop/push pair that supports out-of-range values with fallback
  # encoding scheme.
  def pop_with_fallback(self,
                        decoder: np.ndarray,
                        index: np.ndarray) -> np.ndarray: ...

  def push_with_fallback(self,
                         encoder: np.ndarray,
                         index: np.ndarray,
                         value: np.ndarray) -> None: ...

  def serialize(self) -> bytes: ...

  @staticmethod
  def deserialize(source: bytes) -> RangeAnsStack: ...

