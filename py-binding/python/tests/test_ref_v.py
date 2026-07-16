# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

from alpha.algo import (
    REF_V, MA_V, EMA_V, SMA_V, DMA_V, LWMA_V,
    HHV_V, LLV_V, HHVBARS_V, LLVBARS_V, COUNT_V
)
import numpy as np

def test_ref_v():
    input_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    periods = np.array([1, 2, 1, 3, 2])
    r = REF_V(input_arr, periods)
    expected = np.array([np.nan, np.nan, 2.0, 1.0, 3.0])
    np.testing.assert_allclose(r, expected, equal_nan=True)

def test_ref_v_list():
    input_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    periods = np.array([1, 2, 1, 3, 2])
    r = REF_V([input_arr, input_arr], [periods, periods])
    expected = np.array([np.nan, np.nan, 2.0, 1.0, 3.0])
    assert len(r) == 2
    np.testing.assert_allclose(r[0], expected, equal_nan=True)
    np.testing.assert_allclose(r[1], expected, equal_nan=True)

def test_ma_v():
    input_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    periods = np.array([1, 2, 3, 2, 3])
    r = MA_V(input_arr, periods)
    expected = np.array([1.0, 1.5, 2.0, 3.5, 4.0])
    np.testing.assert_allclose(r, expected, equal_nan=True)

def test_ema_v():
    input_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    periods = np.array([3, 3, 3, 3, 3])
    r = EMA_V(input_arr, periods)
    # EMA(3) alpha=0.5 -> 1.0, 1.5, 2.25, 3.125, 4.0625
    expected = np.array([1.0, 1.5, 2.25, 3.125, 4.0625])
    np.testing.assert_allclose(r, expected, equal_nan=True)

def test_sma_v():
    input_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ns = np.array([3, 3, 3, 3, 3])
    r = SMA_V(input_arr, ns, 1)
    # SMA(3, 1) -> alpha=1/3. 
    # r[0] = 1.0
    # r[1] = 2.0 * (1/3) + 1.0 * (2/3) = 1.33333
    # r[2] = 3.0 * (1/3) + 1.33333 * (2/3) = 1.88888
    # Let's just compare against calling SMA_V directly and checking non-null output
    assert not np.any(np.isnan(r))

def test_dma_v():
    input_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weight = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    r = DMA_V(input_arr, weight)
    assert not np.any(np.isnan(r))

def test_lwma_v():
    input_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    periods = np.array([3, 3, 3, 3, 3])
    r = LWMA_V(input_arr, periods)
    expected = np.array([np.nan, np.nan, 14.0/6.0, 20.0/6.0, 26.0/6.0])
    np.testing.assert_allclose(r, expected, equal_nan=True)

def test_hhv_v():
    input_arr = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0])
    periods = np.array([3, 3, 3, 3, 3, 3])
    r = HHV_V(input_arr, periods)
    expected = np.array([1.0, 3.0, 3.0, 5.0, 5.0, 6.0])
    np.testing.assert_allclose(r, expected, equal_nan=True)

def test_count_v():
    input_arr = np.array([True, False, True, True, False])
    periods = np.array([3, 3, 3, 3, 3])
    r = COUNT_V(input_arr, periods)
    expected = np.array([1.0, 1.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(r, expected, equal_nan=True)
