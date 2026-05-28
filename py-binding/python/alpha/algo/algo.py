# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
from . import _algo


def EMA(
  input: np.ndarray | list[np.ndarray], period: int
) -> np.ndarray | list[np.ndarray]:
  """
  Exponential Moving Average (variant of EMA)

  alpha = 2 / (n + 1)

  https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average

  Args:
    input: input array
    period: period

  Returns:
    output array
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.ema(r, input, period)
    return r
  else:
    r = np.empty_like(input)
    _algo.ema(r, input, period)
    return r
