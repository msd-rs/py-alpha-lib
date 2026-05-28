# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

import talib
import numpy as np
import talib
import time
import alpha.algo as algo
import timeit
import pandas as pd


def test_performance():
  close = np.array([7.2, 6.97, 7.08, 6.74, 6.49, 5.9, 6.26, 5.9, 5.35, 5.63])
  close = np.tile(close, 10_000)
  print(f"{len(close):,}")
  x = timeit.timeit(lambda: talib.EMA(close, 30), number=10000)
  print(x)

  x = timeit.timeit(lambda: algo.EMA(close, 30), number=10000)
  print(x)


def test_correct():
  close = np.array([7.2, 6.97, 7.08, 6.74, 6.49, 5.9, 6.26, 5.9, 5.35, 5.63])
  close = np.tile(close, 10)

  x1 = talib.EMA(close, 30)
  x2 = algo.EMA(close, 30)

  x3 = algo.EMA([close, close], 30)
  assert np.allclose(x1[-30:], x2[-30:], rtol=1e-3)
  assert np.allclose(x1[-30:], x3[0][-30:], rtol=1e-3)
  assert np.allclose(x1[-30:], x3[1][-30:], rtol=1e-3)


def test_sma_dma_smoke():
  close = np.random.randn(100)
  sma = algo.SMA(close, 10, 2)
  dma = algo.DMA(close, 0.5)
  assert sma.shape == close.shape
  assert dma.shape == close.shape

  # Check list input
  sma_list = algo.SMA([close, close], 10, 2)
  assert len(sma_list) == 2
  assert sma_list[0].shape == close.shape


if __name__ == "__main__":
  test_correct()
  test_sma_dma_smoke()
  test_performance()
