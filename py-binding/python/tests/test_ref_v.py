# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

from alpha.algo import REF_V
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
