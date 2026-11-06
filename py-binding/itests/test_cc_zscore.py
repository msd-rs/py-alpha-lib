# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""CC_ZSCORE (横截面 Z-Score) 集成测试.

算法简介
--------
数据按 group-major 排布: ``groups`` 个证券各自占一段长度为 ``group_size``
的连续数据, 位置 ``i * group_size + j`` 表示证券 ``i`` 在时刻 ``j`` 的值.
对每个时刻 ``j``, 用该时刻所有证券的有效值计算均值与样本标准差
(``ddof = 1``), 再做标准化::

    CC_ZSCORE[i, j] = (x[i, j] - mean(j)) / std(j)
    std(j) = sqrt(sum((x - mean)^2) / (valid_count(j) - 1))

nan 位置输出 nan, 且不参与均值/方差; 有效值少于 2 个时该时刻全部输出 nan;
方差为 0 (截面是常量) 时有效位置输出 0. 只有一个证券时无法计算样本标准差,
结果应为全 nan.

四种 ``flags`` 组合都不改变结果: 截面计算本身总是排除 nan,
``FLAG_SKIP_NAN`` / ``FLAG_STRICTLY_CYCLE`` 没有额外语义. ``start`` / ``end``
是 chunk 内相对下标, 当前实现忽略它们而输出全量结果, 因此测试只校验
``[start, end)`` 区间内的值. ``groups < 2`` 时实现回退到时序
``ZSCORE(input, 0)``, 同样输出全 nan, 与上述算法逻辑一致.

参考实现 ``_cc_zscore_ref`` 用 pandas 的 ``mean`` / ``std(ddof=1)`` 加上 numpy
逐行标准化独立复现公式, 不调用 alpha, 也不实现 flags / start-end 机制.
``set_ctx`` 只覆盖传入的参数, 所以每次调用都显式重置 start/end/groups/flags.
浮点比较容差 1e-6.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> x = np.arange(1., 13.)          # 3 个证券, 每个 4 个时刻
    >>> al.set_ctx(start=0, end=0, groups=3, flags=0)

1. 每个时刻的横截面是 ``[j+1, j+5, j+9]``, 均值 j+5, 标准差 4::

    >>> al.CC_ZSCORE(x).tolist()
    [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

2. ``groups = 2`` 时两个值的 Z-Score 必然是 -1/sqrt(2) 与 1/sqrt(2)::

    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.CC_ZSCORE(x), 4).tolist()
    [-0.7071, -0.7071, -0.7071, -0.7071, -0.7071, -0.7071, 0.7071, 0.7071, 0.7071, 0.7071, 0.7071, 0.7071]

3. 常量截面方差为 0, 全部输出 0::

    >>> al.set_ctx(start=0, end=0, groups=3, flags=0)
    >>> al.CC_ZSCORE(np.full(12, 3.)).tolist()
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

4. nan 被排除: 时刻 1 的截面是 ``[nan, 6, 10]``, 均值 8, 标准差 2*sqrt(2)::

    >>> xn = x.copy(); xn[1] = np.nan
    >>> np.round(al.CC_ZSCORE(xn)[[1, 5, 9]], 4).tolist()
    [nan, -0.7071, 0.7071]

5. ``groups = 1`` 只有一个证券, 样本标准差无法计算, 输出全 nan::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> bool(np.isnan(al.CC_ZSCORE(x)).all())
    True

6. ``flags`` 与 ``start`` / ``end`` 不改变当前实现的全量输出::

    >>> al.set_ctx(start=2, end=4, groups=3,
    ...            flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> al.CC_ZSCORE(x).tolist()
    [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
"""

import numpy as np
import pandas as pd
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
EPS = np.finfo(np.float64).eps
N = 120
NAN_POS = np.array([0, 1, 3, 17, 18, 29, 44, 52, 77, 101])
GROUPS = [2, 3, 4, 6]
START_END = [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5), (-10, 0), (40, 20)]
FLAG_COMBOS = [
  0,
  al.FLAG_STRICTLY_CYCLE,
  al.FLAG_SKIP_NAN,
  al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
]


# ---------------------------------------------------------------------------
# pandas / numpy 对比实现
# ---------------------------------------------------------------------------


def _cc_zscore_ref(x, groups):
  """横截面 Z-Score 参考实现, 只描述算法逻辑.

  pandas 的 ``mean`` / ``std(ddof=1)`` 默认跳过 nan; 有效值不足 2 个或
  方差为 0 时分别输出 nan / 0, nan 输入位置始终输出 nan.
  """
  x = np.asarray(x, dtype=np.float64)
  if x.size == 0:
    return np.empty(0, dtype=np.float64)
  assert x.size % groups == 0
  group_size = x.size // groups
  mat = x.reshape(groups, group_size).T
  df = pd.DataFrame(mat)
  means = df.mean(axis=1).to_numpy()
  stds = df.std(axis=1, ddof=1).to_numpy()
  counts = df.count(axis=1).to_numpy()
  valid = ~np.isnan(mat)
  out = np.full(mat.shape, np.nan)
  for j in range(mat.shape[0]):
    if counts[j] < 2:
      continue
    if np.isnan(stds[j]) or stds[j] ** 2 < EPS:
      out[j, valid[j]] = 0.0
    else:
      out[j, valid[j]] = (mat[j, valid[j]] - means[j]) / stds[j]
  return out.T.reshape(-1)


def _expected(x, *, groups=1):
  """参考值总是全量计算; start/end/flags 由测试单独处理."""
  return _cc_zscore_ref(x, groups)


def _calc_range(n, start, end):
  """对应 Rust ``Context::start`` / ``Context::end`` 在单个 chunk 内的解析."""
  if n == 0:
    return 0, 0
  s = min(n - 1, start) if start >= 0 else max(0, n + start)
  if end == 0:
    e = n
  elif end > 0:
    e = min(n, end)
  else:
    e = max(0, n + end)
  return s, e


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(x, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.CC_ZSCORE(np.asarray(x, dtype=np.float64)), dtype=np.float64
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261106)
  return rng.normal(10.0, 2.0, n).astype(np.float64)


def _tie_data(n=N):
  rng = np.random.default_rng(20261107)
  return rng.integers(0, 6, size=n).astype(np.float64)


def _nan_data(n=N):
  x = _clean_data(n)
  x[NAN_POS[NAN_POS < n]] = np.nan
  return x


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


def _assert_range_close(actual, expected, start, end, groups):
  """只校验每个 chunk 内 [start, end) 区间的值."""
  size = len(actual) // groups
  s, e = _calc_range(size, start, end)
  for g in range(groups):
    lo = g * size
    _assert_close(actual[lo + s : lo + e], expected[lo + s : lo + e])


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("groups", GROUPS)
def test_cc_zscore_matches_pandas_reference(groups):
  """无 nan 数据在各 groups 下与 pandas mean/std 参考实现逐点比较."""
  x = _clean_data()
  actual = _calc(x, groups=groups)
  _assert_close(actual, _expected(x, groups=groups))
  assert np.isfinite(actual).all()


@pytest.mark.parametrize("groups", GROUPS)
def test_cc_zscore_zero_mean_unit_std_per_cross_section(groups):
  """每个横截面的 Z-Score 均值为 0, 样本标准差为 1."""
  x = _clean_data()
  actual = _calc(x, groups=groups).reshape(groups, -1).T
  for row in actual:
    assert abs(row.mean()) < ATOL
    assert abs(row.std(ddof=1) - 1.0) < ATOL


@pytest.mark.parametrize("groups", GROUPS)
def test_cc_zscore_ties(groups):
  """离散化的重复值数据同样与参考实现一致."""
  x = _tie_data()
  _assert_close(_calc(x, groups=groups), _expected(x, groups=groups))


@pytest.mark.parametrize("groups", GROUPS)
def test_cc_zscore_constant_cross_section_is_zero(groups):
  """常量截面方差为 0, 有效位置一律输出 0 而不是 nan."""
  const = np.full(N, 3.0)
  actual = _calc(const, groups=groups)
  _assert_close(actual, np.zeros(N))
  _assert_close(actual, _expected(const, groups=groups))


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_cc_zscore_nan_excluded_from_moments(flags):
  """nan 位置输出 nan, 其余有效值按剩余有效值计算均值/标准差."""
  x = _nan_data()
  actual = _calc(x, groups=3, flags=flags)
  _assert_close(actual, _expected(x, groups=3))
  assert np.isnan(actual[np.isnan(x)]).all()
  assert np.isfinite(actual[~np.isnan(x)]).any()


def test_cc_zscore_too_few_valid_values():
  """某时刻只剩 1 个有效值时该时刻全部输出 nan, 不影响其它时刻."""
  x = np.arange(1., 13.)
  x[1], x[9] = np.nan, np.nan
  actual = _calc(x, groups=3)
  _assert_close(actual, _expected(x, groups=3))
  assert np.isnan(actual[[1, 5, 9]]).all()
  assert np.isfinite(actual[[0, 4, 8]]).all()


def test_cc_zscore_all_nan_cross_section():
  """整列都是 nan 的时刻全部输出 nan."""
  x = np.arange(1., 13.)
  x[[1, 5, 9]] = np.nan
  actual = _calc(x, groups=3, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, groups=3))
  assert np.isnan(actual[[1, 5, 9]]).all()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_cc_zscore_flags_do_not_change_result(flags):
  """四种 flag 组合结果与 flags=0 完全一致 (含 nan 数据)."""
  x = _nan_data()
  base = _calc(x, groups=3, flags=0)
  _assert_close(_calc(x, groups=3, flags=flags), base)


@pytest.mark.parametrize("flags", [al.FLAG_SKIP_NAN, al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE])
def test_cc_zscore_skip_nan_does_not_use_nan(flags):
  """FLAG_SKIP_NAN: 结果等于排除 nan 后的横截面 Z-Score 参考值."""
  x = _nan_data()
  actual = _calc(x, groups=3, flags=flags)
  _assert_close(actual, _expected(x, groups=3))
  assert np.isnan(actual[np.isnan(x)]).all()


@pytest.mark.parametrize("start,end", START_END)
@pytest.mark.parametrize("groups", GROUPS)
def test_cc_zscore_start_end(start, end, groups):
  """start/end: 仅校验 [start, end) 区间内与全量参考一致."""
  x = _nan_data()
  expected = _expected(x, groups=groups)
  actual = _calc(x, start=start, end=end, groups=groups)
  _assert_range_close(actual, expected, start, end, groups)


@pytest.mark.parametrize("groups", [0, 1])
def test_cc_zscore_single_group_is_nan(groups):
  """groups<2 只有一个证券, 样本标准差不可计算, 输出全 nan."""
  x = _clean_data(12)
  actual = _calc(x, groups=groups)
  assert np.isnan(actual).all()
  _assert_close(actual, _expected(x, groups=max(1, groups)))


def test_cc_zscore_edge_cases():
  """空输入 / 长度不整除 / 单元素 / 全 nan."""
  empty = np.empty(0, dtype=np.float64)
  _assert_close(_calc(empty, groups=1), empty)
  _assert_close(_expected(empty, groups=1), empty)

  with pytest.raises(ValueError):
    _calc(np.ones(10), groups=3)

  one = np.array([3.0])
  assert np.isnan(_calc(one, groups=1)).all()

  all_nan = np.full(12, np.nan)
  assert np.isnan(_calc(all_nan, groups=3, flags=al.FLAG_SKIP_NAN)).all()
