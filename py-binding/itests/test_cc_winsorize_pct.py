# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""CC_WINSORIZE_PCT (横截面分位数去极值) 集成测试.

算法简介
--------
对每个截面有效有限值按线性插值计算 ``[pct_low, pct_high]`` 分位数并裁剪,
等价于 ``np.clip(x, np.nanpercentile(x, pct_low * 100), np.nanpercentile(x, pct_high * 100))``.

场景与结果示例
--------------
    >>> import numpy as np
    >>> import alpha as al
    >>> al.set_ctx(start=0, end=0, groups=5, flags=0)
    >>> x = np.array([0., 10., 20., 30., 100.])
    >>> np.round(al.CC_WINSORIZE_PCT(x, 0.1, 0.9), 4).tolist()
    [4.0, 10.0, 20.0, 30.0, 72.0]
"""

import numpy as np
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
GROUPS = [2, 3, 4, 6]


def _cc_winsorize_pct_ref(
  x: np.ndarray, groups: int, pct_low: float, pct_high: float
) -> np.ndarray:
  x = np.asarray(x, dtype=np.float64)
  if x.size == 0:
    return np.empty(0, dtype=np.float64)
  mat = x.reshape(groups, -1)
  out = np.full_like(mat, np.nan)
  for j in range(mat.shape[1]):
    col = mat[:, j]
    fin = np.isfinite(col)
    if not np.any(fin):
      continue
    vals = col[fin]
    lo = float(np.percentile(vals, pct_low * 100.0))
    hi = float(np.percentile(vals, pct_high * 100.0))
    out[fin, j] = np.clip(vals, lo, hi)
  return out.reshape(-1)


@pytest.mark.parametrize("groups", GROUPS)
def test_cc_winsorize_pct_matches_numpy_percentile(groups):
  rng = np.random.default_rng(20261127)
  x = rng.normal(0.0, 5.0, N)
  x[[0, 7, 33, 88]] = np.nan
  x[12] = np.inf

  al.set_ctx(start=0, end=0, groups=groups, flags=0)
  actual = al.CC_WINSORIZE_PCT(x, 0.05, 0.95)
  expected = _cc_winsorize_pct_ref(x, groups, 0.05, 0.95)
  np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True)


def test_cc_winsorize_pct_invalid_bounds():
  al.set_ctx(start=0, end=0, groups=4, flags=0)
  x = np.array([1.0, 2.0, 3.0, 4.0])
  with pytest.raises(ValueError):
    al.CC_WINSORIZE_PCT(x, 0.9, 0.1)
  with pytest.raises(ValueError):
    al.CC_WINSORIZE_PCT(x, -0.01, 0.99)
  with pytest.raises(ValueError):
    al.CC_WINSORIZE_PCT(x, 0.01, 1.01)
