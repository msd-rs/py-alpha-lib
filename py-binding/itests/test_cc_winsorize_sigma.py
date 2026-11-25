# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""CC_WINSORIZE_SIGMA (横截面标准差去极值) 集成测试.

算法简介
--------
对每个截面计算有限值的均值 ``mean`` 与总体标准差 ``std`` (``ddof=0``),
将有限值裁剪至 ``[mean - sigma_k * std, mean + sigma_k * std]``.
当截面有效数 ``<= 1`` 或 ``std <= 1e-12`` 时保持原有限值不变.

场景与结果示例
--------------
    >>> import numpy as np
    >>> import alpha as al
    >>> al.set_ctx(start=0, end=0, groups=4, flags=0)
    >>> x = np.array([0., 0., 0., 100.])
    >>> np.round(al.CC_WINSORIZE_SIGMA(x, 1.0), 4).tolist()
    [0.0, 0.0, 0.0, 68.3013]
"""

import numpy as np
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
GROUPS = [2, 3, 4, 6]


def _cc_winsorize_sigma_ref(x: np.ndarray, groups: int, sigma_k: float) -> np.ndarray:
  x = np.asarray(x, dtype=np.float64)
  if x.size == 0:
    return np.empty(0, dtype=np.float64)
  mat = x.reshape(groups, -1)
  out = np.full_like(mat, np.nan)
  for j in range(mat.shape[1]):
    col = mat[:, j]
    fin = np.isfinite(col)
    cnt = int(np.sum(fin))
    if cnt == 0:
      continue
    vals = col[fin]
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=0))
    if cnt > 1 and np.isfinite(std) and std > 1e-12:
      out[fin, j] = np.clip(vals, mean - sigma_k * std, mean + sigma_k * std)
    else:
      out[fin, j] = vals
  return out.reshape(-1)


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("sigma_k", [1.5, 3.0])
def test_cc_winsorize_sigma_matches_reference(groups, sigma_k):
  rng = np.random.default_rng(20261126)
  x = rng.normal(5.0, 3.0, N)
  x[1] = 100.0
  x[[4, 19, 63]] = np.nan
  x[30] = -np.inf

  al.set_ctx(start=0, end=0, groups=groups, flags=0)
  actual = al.CC_WINSORIZE_SIGMA(x, sigma_k)
  expected = _cc_winsorize_sigma_ref(x, groups, sigma_k)
  np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True)


def test_cc_winsorize_sigma_degenerate_and_f32():
  al.set_ctx(start=0, end=0, groups=4, flags=0)
  x = np.array([3.0, np.nan, 3.0, np.nan, 3.0, np.nan, 3.0, 9.0], dtype=np.float32)
  actual = al.CC_WINSORIZE_SIGMA(x, 2.0)
  expected = _cc_winsorize_sigma_ref(x.astype(np.float64), 4, 2.0)
  np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5, equal_nan=True)
