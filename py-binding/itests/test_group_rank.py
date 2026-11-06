# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""GROUP_RANK (类别内横截面排名百分比) 集成测试.

算法简介
--------
数据按 group-major 排布: ``groups`` 个证券各自占一段长度为 ``group_size``
的连续数据, 位置 ``i * group_size + j`` 表示证券 ``i`` 在时刻 ``j`` 的值.
对每个时刻 ``j``, 先按 ``category`` 把证券分类, 再在每一类内部对 ``input``
做升序排名 (并列取平均排名, 1-based), 除以该类有效值个数::

    GROUP_RANK[i, j] = rank_avg(i, j) / count(category(j))   # 取值 (0, 1]

只有一个证券的类别定义为 0.5 (中性排名), 而不是 1.0. ``category`` 或
``input`` 为 nan 的项不参与任何类别, 输出 nan; 某时刻没有有效项时全部输出 nan.

四种 ``flags`` 组合都不改变结果: 分类计算本身总是排除 nan,
``FLAG_SKIP_NAN`` / ``FLAG_STRICTLY_CYCLE`` 没有额外语义. ``start`` / ``end``
是 chunk 内相对下标, 当前实现忽略它们而输出全量结果, 因此测试只校验
``[start, end)`` 区间内的值. 当所有 ``category`` 相同且 ``groups >= 2`` 时,
本算子应等价于 ``CC_RANK`` (见交叉验证测试).

参考实现 ``_group_rank_ref`` 用 pandas ``groupby(...).rank(pct=True)`` 加
单元素类别修正独立复现算法逻辑, 不调用 alpha, 也不实现 flags / start-end
机制. ``set_ctx`` 只覆盖传入的参数, 所以每次调用都显式重置
start/end/groups/flags. 浮点比较容差 1e-6.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> c = np.array([1., 1., 2., 2.])       # 4 个证券, 每个 1 个时刻
    >>> v = np.array([10., 30., 5., 15.])
    >>> al.set_ctx(start=0, end=0, groups=4, flags=0)

1. 类别 1 内 ``10 < 30``, 类别 2 内 ``5 < 15``, 各自是 0.5 / 1.0::

    >>> al.GROUP_RANK(c, v).tolist()
    [0.5, 1.0, 0.5, 1.0]

2. 多时刻 (3 个证券, 每个 2 个时刻), 类别 2 只有一个证券时恒为 0.5::

    >>> c = np.array([1., 1., 1., 1., 2., 2.])
    >>> v = np.array([10., 20., 30., 40., 50., 60.])
    >>> al.set_ctx(start=0, end=0, groups=3, flags=0)
    >>> al.GROUP_RANK(c, v).tolist()
    [0.5, 0.5, 1.0, 1.0, 0.5, 0.5]

3. 并列值取平均排名: 两个 5 并列, ``rank_avg = 1.5``, 百分比 0.5::

    >>> al.set_ctx(start=0, end=0, groups=3, flags=0)
    >>> al.GROUP_RANK(np.ones(3), np.array([5., 5., 10.])).tolist()
    [0.5, 0.5, 1.0]

4. 类别为 nan 的项被剔除, 其它类别的排名不受影响::

    >>> cn = np.array([1., 1., 2.]); cn[2] = np.nan
    >>> bool(np.isnan(al.GROUP_RANK(cn, np.array([10., 20., 30.])))[2])
    True

5. 所有类别相同时结果与 ``CC_RANK`` 完全一致::

    >>> x = np.arange(1., 13.)
    >>> al.set_ctx(start=0, end=0, groups=3, flags=0)
    >>> np.allclose(al.GROUP_RANK(np.ones(12), x), al.CC_RANK(x))
    True

6. ``flags`` 与 ``start`` / ``end`` 不改变当前实现的全量输出::

    >>> al.set_ctx(start=2, end=4, groups=3, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.GROUP_RANK(np.ones(12), x), 4).tolist()
    [0.3333, 0.3333, 0.3333, 0.3333, 0.6667, 0.6667, 0.6667, 0.6667, 1.0, 1.0, 1.0, 1.0]
"""

import numpy as np
import pandas as pd
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
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
# pandas 对比实现
# ---------------------------------------------------------------------------


def _group_rank_ref(category, x, groups):
  """类别内排名百分比参考实现, 只描述算法逻辑.

  每个时刻单独构造 DataFrame, 按 category 分组后用
  ``rank(method='average', pct=True)``; 单元素类别按定义输出 0.5.
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
    ranks = grouped.rank(method="average", pct=True)
    sizes = grouped.transform("size")
    out[j, idx] = ranks.where(sizes > 1, 0.5).to_numpy()
  return out.T.reshape(-1)


def _expected(category, x, *, groups=1):
  """参考值总是全量计算; start/end/flags 由测试单独处理."""
  return _group_rank_ref(category, x, groups)


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
    al.GROUP_RANK(
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
def test_group_rank_matches_pandas_reference(groups):
  """无 nan 数据在各 groups 下与 pandas groupby+rank 参考实现逐点比较."""
  category, x = _clean_pair()
  actual = _calc(category, x, groups=groups)
  _assert_close(actual, _expected(category, x, groups=groups))
  assert ((actual[~np.isnan(actual)] > 0) & (actual[~np.isnan(actual)] <= 1)).all()


@pytest.mark.parametrize("groups", GROUPS)
def test_group_rank_ties(groups):
  """离散重复值在各类别内按平均排名计算."""
  category, x = _tie_pair()
  _assert_close(_calc(category, x, groups=groups), _expected(category, x, groups=groups))


def test_group_rank_average_rank_and_single_item_category():
  """小数组精确验证并列平均排名与单元素类别的 0.5 约定."""
  _assert_close(
    _calc(np.ones(3), np.array([5., 5., 10.]), groups=3),
    np.array([0.5, 0.5, 1.0]),
  )
  _assert_close(
    _calc(np.array([1., 1., 2.]), np.array([10., 20., 100.]), groups=3),
    np.array([0.5, 1.0, 0.5]),
  )


@pytest.mark.parametrize("groups", [2, 3, 4, 6])
def test_group_rank_all_same_category_ties(groups):
  """所有类别相同且值也相同时, 全部并列, 排名为 (n+1)/(2n)."""
  category = np.ones(N)
  x = np.full(N, 3.0)
  expected_pct = (groups + 1) / (2 * groups)
  actual = _calc(category, x, groups=groups)
  _assert_close(actual, np.full(N, expected_pct))
  _assert_close(actual, _expected(category, x, groups=groups))


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_group_rank_nan_in_input_or_category(flags):
  """input / category 中任一为 nan 的位置输出 nan, 其余与参考一致."""
  category, x = _nan_pair()
  actual = _calc(category, x, groups=3, flags=flags)
  _assert_close(actual, _expected(category, x, groups=3))
  invalid = np.isnan(category) | np.isnan(x)
  assert np.isnan(actual[invalid]).all()
  assert np.isfinite(actual[~invalid]).all()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_group_rank_flags_do_not_change_result(flags):
  """四种 flag 组合结果与 flags=0 完全一致 (含 nan 数据)."""
  category, x = _nan_pair()
  base = _calc(category, x, groups=3, flags=0)
  _assert_close(_calc(category, x, groups=3, flags=flags), base)


@pytest.mark.parametrize("flags", [al.FLAG_SKIP_NAN, al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE])
def test_group_rank_skip_nan_does_not_use_nan(flags):
  """FLAG_SKIP_NAN: 结果等于排除所有无效项后的类别内排名参考值."""
  category, x = _nan_pair()
  actual = _calc(category, x, groups=3, flags=flags)
  _assert_close(actual, _expected(category, x, groups=3))
  assert np.isnan(actual[np.isnan(category) | np.isnan(x)]).all()


@pytest.mark.parametrize("groups", [2, 3, 4, 6])
def test_group_rank_single_category_equals_cc_rank(groups):
  """category 全相同时, 类别内排名退化为横截面排名 CC_RANK."""
  x = _clean_pair()[1]
  category = np.ones(len(x))
  actual = _calc(category, x, groups=groups)
  expected = np.asarray(al.CC_RANK(x), dtype=np.float64)
  _assert_close(actual, expected)
  _assert_close(actual, _expected(category, x, groups=groups))


@pytest.mark.parametrize("groups", GROUPS)
def test_group_rank_group_major_layout(groups):
  """group-major: 相同 chunk 的 category/value 构成全并列截面."""
  category, x = _clean_pair()
  _assert_close(
    _calc(category, x, groups=groups),
    _expected(category, x, groups=groups),
  )

  size = len(x) // groups
  repeated_x = np.tile(x[:size], groups)
  repeated_cat = np.tile(category[:size], groups)
  actual = _calc(repeated_cat, repeated_x, groups=groups)
  _assert_close(actual, _expected(repeated_cat, repeated_x, groups=groups))


@pytest.mark.parametrize("start,end", START_END)
@pytest.mark.parametrize("groups", [2, 3, 4, 6])
def test_group_rank_start_end(start, end, groups):
  """start/end: 仅校验 [start, end) 区间内与全量参考一致."""
  category, x = _nan_pair()
  expected = _expected(category, x, groups=groups)
  actual = _calc(category, x, start=start, end=end, groups=groups)
  _assert_range_close(actual, expected, start, end, groups)


def test_group_rank_edge_cases():
  """空输入 / 长度不匹配 / 长度不整除 / 全 nan / 单元素."""
  empty = np.empty(0, dtype=np.float64)
  with pytest.raises(ValueError):
    _calc(empty, empty, groups=1)

  with pytest.raises(ValueError):
    _calc(np.ones(11), np.ones(12), groups=3)
  with pytest.raises(ValueError):
    _calc(np.ones(10), np.ones(10), groups=3)

  one = np.array([1.0])
  _assert_close(_calc(one, np.array([3.0]), groups=1), np.array([0.5]))

  all_nan = np.full(12, np.nan)
  assert np.isnan(
    _calc(np.ones(12), all_nan, groups=3, flags=al.FLAG_SKIP_NAN)
  ).all()
  assert np.isnan(
    _calc(all_nan, np.ones(12), groups=3, flags=al.FLAG_SKIP_NAN)
  ).all()
