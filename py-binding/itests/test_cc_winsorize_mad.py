# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""CC_WINSORIZE_MAD (横截面 MAD 去极值) 集成测试.

算法简介
--------
数据按 group-major 排布: ``groups`` 个证券各自占一段长度为 ``group_size``
的连续数据, 位置 ``i * group_size + j`` 表示证券 ``i`` 在时刻 ``j`` 的值.
对每个时刻 ``j``, 计算截面有效有限值的中位数 ``med`` 与绝对中位差
``MAD = median(|x - med|)``, 将有限值裁剪至::

    [med - 1.4826 * mad_k * MAD, med + 1.4826 * mad_k * MAD]

当 ``1.4826 * mad_k * MAD <= 1e-12`` (如超过半数取值相同导致 ``MAD = 0``)
时保持原有限值不变, 防止截面信号坍塌. NaN 与 Inf 输出 NaN.

场景与结果示例
--------------
    >>> import numpy as np
    >>> import alpha as al
    >>> al.set_ctx(start=0, end=0, groups=5, flags=0)
    >>> x = np.array([1., 2., 3., 4., 100.])
    >>> np.round(al.CC_WINSORIZE_MAD(x, 3.0), 4).tolist()
    [1.0, 2.0, 3.0, 4.0, 7.4478]
"""

import numpy as np
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
GROUPS = [2, 3, 4, 6]


def _cc_winsorize_mad_ref(x: np.ndarray, groups: int, mad_k: float) -> np.ndarray:
  x = np.asarray(x, dtype=np.float64)
  if x.size == 0:
    return np.empty(0, dtype=np.float64)
  mat = x.reshape(groups, -1).copy()
  out = np.full_like(mat, np.nan)
  for j in range(mat.shape[1]):
    col = mat[:, j]
    fin = np.isfinite(col)
    if not np.any(fin):
      continue
    vals = col[fin]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    scaled = 1.4826 * mad * float(mad_k)
    if np.isfinite(scaled) and scaled > 1e-12:
      out[fin, j] = np.clip(vals, med - scaled, med + scaled)
    else:
      out[fin, j] = vals
  return out.reshape(-1)


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("mad_k", [1.5, 3.0])
def test_cc_winsorize_mad_matches_reference(groups, mad_k):
  rng = np.random.default_rng(20261125)
  x = rng.normal(0.0, 2.0, N)
  x[0] = 50.0
  x[N // 2] = -60.0
  x[[3, 17, 45]] = np.nan
  x[22] = np.inf

  al.set_ctx(start=0, end=0, groups=groups, flags=0)
  actual = al.CC_WINSORIZE_MAD(x, mad_k)
  expected = _cc_winsorize_mad_ref(x, groups, mad_k)
  np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True)


def test_cc_winsorize_mad_zero_mad_guard_and_all_nan():
  al.set_ctx(start=0, end=0, groups=5, flags=0)
  x = np.array([10.0, 2.0, 10.0, np.nan, 10.0, 2.0, 10.0, np.nan, 50.0, 8.0])
  actual = al.CC_WINSORIZE_MAD(x, 3.0)
  expected = _cc_winsorize_mad_ref(x, 5, 3.0)
  np.testing.assert_allclose(actual, expected, atol=1e-12, equal_nan=True)

  all_nan = np.full(10, np.nan)
  assert np.isnan(al.CC_WINSORIZE_MAD(all_nan, 3.0)).all()

  al.set_ctx(start=0, end=0, groups=1, flags=0)
  empty = np.empty(0, dtype=np.float64)
  assert al.CC_WINSORIZE_MAD(empty, 3.0).size == 0
