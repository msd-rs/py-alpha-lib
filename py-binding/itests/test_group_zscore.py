# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""GROUP_ZSCORE (类别内横截面 Z-Score) 集成测试.

算法简介
--------
数据按 group-major 排布: ``groups`` 个证券各自占一段长度为 ``group_size``
的连续数据, 位置 ``i * group_size + j`` 表示证券 ``i`` 在时刻 ``j`` 的值.
对每个时刻 ``j``, 先按 ``category`` 把证券分类, 再在每一类内部用 ``input``
的有效值计算均值与样本标准差 (``ddof = 1``), 最后标准化::

    GROUP_ZSCORE[i, j] = (x[i, j] - mean(cat, j)) / std(cat, j)
    std(cat, j) = sqrt(sum((x - mean)^2) / (count(cat, j) - 1))

``category`` 或 ``input`` 为 nan 的项不属于任何类别, 输出 nan; 类别内有效值
少于 2 个时该类别全部输出 nan; 类别内方差为 0 (常量) 时有效位置输出 0.

四种 ``flags`` 组合都不改变结果: 分类计算本身总是排除 nan,
``FLAG_SKIP_NAN`` / ``FLAG_STRICTLY_CYCLE`` 没有额外语义. ``start`` / ``end``
是 chunk 内相对下标, 当前实现忽略它们而输出全量结果, 因此测试只校验
``[start, end)`` 区间内的值. 当所有 ``category`` 相同且 ``groups >= 2`` 时,
本算子应等价于 ``CC_ZSCORE`` (见交叉验证测试).

参考实现 ``_group_zscore_ref`` 用 pandas ``groupby().transform('mean'/'std')``
加 numpy 逐类别标准化独立复现公式, 不调用 alpha, 也不实现 flags /
start-end 机制. ``set_ctx`` 只覆盖传入的参数, 所以每次调用都显式重置
start/end/groups/flags. 浮点比较容差 1e-6.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> c = np.array([1., 1., 2., 2.])       # 4 个证券, 每个 1 个时刻
    >>> v = np.array([10., 30., 5., 15.])
    >>> al.set_ctx(start=0, end=0, groups=4, flags=0)

1. 两类各自标准化, 两元素类别的 Z-Score 为 -1/sqrt(2) 与 1/sqrt(2)::

    >>> np.round(al.GROUP_ZSCORE(c, v), 4).tolist()
    [-0.7071, 0.7071, -0.7071, 0.7071]

2. 多时刻 (3 个证券, 每个 2 个时刻), 类别 2 只有一个证券, 输出 nan::

    >>> c = np.array([1., 1., 1., 1., 2., 2.])
    >>> v = np.array([10., 20., 30., 40., 50., 60.])
    >>> al.set_ctx(start=0, end=0, groups=3, flags=0)
    >>> np.round(al.GROUP_ZSCORE(c, v), 4)
    array([-0.7071, -0.7071,  0.7071,  0.7071,     nan,     nan])

3. 类别内是常量时方差为 0, 输出 0::

    >>> al.set_ctx(start=0, end=0, groups=3, flags=0)
    >>> al.GROUP_ZSCORE(np.ones(3), np.full(3, 5.)).tolist()
    [0.0, 0.0, 0.0]

4. 单元素类别输出 nan, 两元素类别正常标准化::

    >>> np.round(al.GROUP_ZSCORE(np.array([1., 1., 2.]),
    ...                          np.array([10., 20., 100.])), 4).tolist()
    [-0.7071, 0.7071, nan]

5. 所有类别相同时结果与 ``CC_ZSCORE`` 完全一致::

    >>> x = np.arange(1., 13.)
    >>> al.set_ctx(start=0, end=0, groups=3, flags=0)
    >>> np.allclose(al.GROUP_ZSCORE(np.ones(12), x), al.CC_ZSCORE(x))
    True

6. 类别为 nan 的项不参与任何计算, 自身输出 nan::

    >>> cn = np.array([1., 1., 2.]); cn[2] = np.nan
    >>> bool(np.isnan(al.GROUP_ZSCORE(cn, np.array([10., 20., 30.])))[2])
    True

7. ``flags`` 与 ``start`` / ``end`` 不改变当前实现的全量输出::

    >>> al.set_ctx(start=2, end=4, groups=3,
    ...            flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> al.GROUP_ZSCORE(np.ones(12), x).tolist()
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
GROUPS = [1, 2, 3, 4, 6]
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


def _group_zscore_ref(category, x, groups):
  """类别内 Z-Score 参考实现, 只描述算法逻辑.

  逐时刻按 category 分组: 组内少于 2 个有效值 -> nan; 组内方差为 0 -> 0;
  其余按 (x - mean) / std(ddof=1) 标准化. nan 输入位置始终输出 nan.
  """
  category = np.asarray(category, dtype=np.float64)
  x = np.asarray(x, dtype=np.float64)
  if x.size == 0:
    return np.empty(0, dtype=np.float64)
  assert x.size % groups == 0
  group_size = x.size // groups
  cm = category.reshape(groups, group_size).T
  xm = x.reshape(groups, group_size).T
  out = np.full(xm.shape, np.nan)
  for j in range(group_size):
    valid = ~np.isnan(cm[j]) & ~np.isnan(xm[j])
    idx = np.flatnonzero(valid)
    if idx.size == 0:
      continue
    df = pd.DataFrame({"cat": cm[j, idx], "val": xm[j, idx]})
    grouped = df.groupby("cat")["val"]
    means = grouped.transform("mean").to_numpy()
    stds = grouped.transform("std").to_numpy()
    sizes = grouped.transform("size").to_numpy()
    vals = df["val"].to_numpy()
    z = np.full(vals.shape, np.nan)
    enough = sizes >= 2
    zero_var = enough & (np.isnan(stds) | (np.nan_to_num(stds) ** 2 < EPS))
    normal = enough & ~zero_var
    z[normal] = (vals[normal] - means[normal]) / stds[normal]
    z[zero_var] = 0.0
    out[j, idx] = z
  return out.T.reshape(-1)


def _expected(category, x, *, groups=1):
  """参考值总是全量计算; start/end/flags 由测试单独处理."""
  return _group_zscore_ref(category, x, groups)


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


def _calc(category, x, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.GROUP_ZSCORE(
      np.asarray(category, dtype=np.float64),
      np.asarray(x, dtype=np.float64),
    ),
    dtype=np.float64,
  )


def _clean_pair(n=N):
  rng = np.random.default_rng(20261106)
  x = rng.normal(10.0, 2.0, n).astype(np.float64)
  cat_rng = np.random.default_rng(20261108)
  category = cat_rng.integers(0, 3, size=n).astype(np.float64)
  return category, x


def _tie_pair(n=N):
  rng = np.random.default_rng(20261107)
  x = rng.integers(0, 6, size=n).astype(np.float64)
  cat_rng = np.random.default_rng(20261108)
  category = cat_rng.integers(0, 3, size=n).astype(np.float64)
  return category, x


def _nan_pair(n=N):
  category, x = _clean_pair(n)
  x[NAN_POS[NAN_POS < n]] = np.nan
  shifted = NAN_POS + 2
  category[shifted[shifted < n]] = np.nan
  return category, x


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
def test_group_zscore_matches_pandas_reference(groups):
  """无 nan 数据在各 groups 下与 pandas groupby 参考实现逐点比较."""
  category, x = _clean_pair()
  actual = _calc(category, x, groups=groups)
  _assert_close(actual, _expected(category, x, groups=groups))


@pytest.mark.parametrize("groups", GROUPS)
def test_group_zscore_ties(groups):
  """离散重复值数据 (易产生零方差类别) 仍与参考实现一致."""
  category, x = _tie_pair()
  _assert_close(_calc(category, x, groups=groups), _expected(category, x, groups=groups))


@pytest.mark.parametrize("groups", [2, 3, 4, 6])
def test_group_zscore_zero_mean_unit_std_within_category(groups):
  """每个非退化类别内部的 Z-Score 均值为 0, 样本标准差为 1."""
  category, x = _clean_pair()
  gs = len(x) // groups
  values = _calc(category, x, groups=groups)
  out = values.reshape(groups, gs).T
  cm = category.reshape(groups, gs).T
  for j in range(gs):
    row = out[j]
    for cat in np.unique(cm[j][~np.isnan(cm[j])]):
      sel = (cm[j] == cat) & ~np.isnan(row)
      vals = row[sel]
      if vals.size >= 2 and vals.std(ddof=1) > ATOL:
        assert abs(vals.mean()) < ATOL
        assert abs(vals.std(ddof=1) - 1.0) < ATOL


@pytest.mark.parametrize("groups", GROUPS)
def test_group_zscore_constant_category_is_zero(groups):
  """类别内是常量时方差为 0, 有效位置输出 0 而不是 nan."""
  category = np.ones(N)
  x = np.full(N, 3.0)
  actual = _calc(category, x, groups=groups)
  _assert_close(actual, _expected(category, x, groups=groups))
  if groups >= 2:
    _assert_close(actual, np.zeros(N))


def test_group_zscore_single_item_category_is_nan():
  """类别只有一个有效值时该类别输出 nan, 两元素类别正常标准化."""
  category = np.array([1., 1., 2.])
  x = np.array([10., 20., 100.])
  expected = np.array([-1 / np.sqrt(2), 1 / np.sqrt(2), np.nan])
  actual = _calc(category, x, groups=3)
  _assert_close(actual, expected)
  _assert_close(actual, _expected(category, x, groups=3))


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_group_zscore_nan_in_input_or_category(flags):
  """input / category 中任一为 nan 的位置输出 nan, 其余与参考一致."""
  category, x = _nan_pair()
  actual = _calc(category, x, groups=3, flags=flags)
  _assert_close(actual, _expected(category, x, groups=3))
  invalid = np.isnan(category) | np.isnan(x)
  assert np.isnan(actual[invalid]).all()
  assert np.isfinite(actual[~invalid]).any()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_group_zscore_flags_do_not_change_result(flags):
  """四种 flag 组合结果与 flags=0 完全一致 (含 nan 数据)."""
  category, x = _nan_pair()
  base = _calc(category, x, groups=3, flags=0)
  _assert_close(_calc(category, x, groups=3, flags=flags), base)


@pytest.mark.parametrize("flags", [al.FLAG_SKIP_NAN, al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE])
def test_group_zscore_skip_nan_does_not_use_nan(flags):
  """FLAG_SKIP_NAN: 结果等于排除所有无效项后的类别内 Z-Score 参考值."""
  category, x = _nan_pair()
  actual = _calc(category, x, groups=3, flags=flags)
  _assert_close(actual, _expected(category, x, groups=3))
  assert np.isnan(actual[np.isnan(category) | np.isnan(x)]).all()


@pytest.mark.parametrize("groups", [2, 3, 4, 6])
def test_group_zscore_single_category_equals_cc_zscore(groups):
  """category 全相同时, 类别内 Z-Score 退化为横截面 CC_ZSCORE."""
  x = _clean_pair()[1]
  category = np.ones(len(x))
  actual = _calc(category, x, groups=groups)
  expected = np.asarray(al.CC_ZSCORE(x), dtype=np.float64)
  _assert_close(actual, expected)
  _assert_close(actual, _expected(category, x, groups=groups))


@pytest.mark.parametrize("start,end", START_END)
@pytest.mark.parametrize("groups", [2, 3, 4, 6])
def test_group_zscore_start_end(start, end, groups):
  """start/end: 仅校验 [start, end) 区间内与全量参考一致."""
  category, x = _nan_pair()
  expected = _expected(category, x, groups=groups)
  actual = _calc(category, x, start=start, end=end, groups=groups)
  _assert_range_close(actual, expected, start, end, groups)


@pytest.mark.parametrize("groups", [0, 1])
def test_group_zscore_single_group_is_nan(groups):
  """groups<2 时每个类别只有一个证券, 样本标准差不可计算, 输出全 nan."""
  category, x = _clean_pair(12)
  actual = _calc(category, x, groups=groups)
  assert np.isnan(actual).all()
  _assert_close(actual, _expected(category, x, groups=max(1, groups)))


def test_group_zscore_edge_cases():
  """空输入 / 长度不匹配 / 长度不整除 / 全 nan."""
  empty = np.empty(0, dtype=np.float64)
  with pytest.raises(ValueError):
    _calc(empty, empty, groups=1)

  with pytest.raises(ValueError):
    _calc(np.ones(11), np.ones(12), groups=3)
  with pytest.raises(ValueError):
    _calc(np.ones(10), np.ones(10), groups=3)

  all_nan = np.full(12, np.nan)
  assert np.isnan(
    _calc(np.ones(12), all_nan, groups=3, flags=al.FLAG_SKIP_NAN)
  ).all()
  assert np.isnan(
    _calc(all_nan, np.ones(12), groups=3, flags=al.FLAG_SKIP_NAN)
  ).all()
