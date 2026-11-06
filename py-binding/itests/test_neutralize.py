# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""NEUTRALIZE (类别中性化后的残差排名) 集成测试.

算法简介
--------
数据按 group-major 排布: ``groups`` 个证券各自占一段长度为 ``group_size``
的连续数据, 位置 ``i * group_size + j`` 表示证券 ``i`` 在时刻 ``j`` 的值.
对每个时刻 ``j``:

1. 按 ``category`` 把有效项 (category / input 都非 nan) 分组, 计算组内均值,
   用 ``resid = input - group_mean`` 去掉类别效应 (中性化);
2. 把该时刻所有残差放在一起做升序排名 (并列取平均排名, 1-based), 再归一化::

    NEUTRALIZE = (rank_avg - 1) / (valid_count - 1)     # 取值 [0, 1]

有效项只有 1 个时定义为 0.5; 所有残差并列时同样得到 0.5. 最小残差为 0,
最大残差为 1. ``category`` 或 ``input`` 为 nan 的项不属于任何类别, 输出 nan;
某时刻没有有效项时该时刻全部输出 nan. ``groups < 2`` 时每个时刻只有一个
证券, 恒为 0.5 (有效位置) 或 nan.

四种 ``flags`` 组合都不改变结果: 分类计算本身总是排除 nan,
``FLAG_SKIP_NAN`` / ``FLAG_STRICTLY_CYCLE`` 没有额外语义. ``start`` / ``end``
是 chunk 内相对下标, ``groups >= 2`` 时当前实现忽略它们而输出全量结果,
因此测试只校验 ``[start, end)`` 区间内的值. 当所有 ``category`` 相同时,
退化关系为 ``NEUTRALIZE = (CC_RANK * groups - 1) / (groups - 1)``
(见交叉验证测试).

参考实现 ``_neutralize_ref`` 用 pandas ``groupby().transform('mean')`` 与
``rank(method='average')`` 独立复现算法逻辑, 不调用 alpha, 也不实现
flags / start-end 机制. ``set_ctx`` 只覆盖传入的参数, 所以每次调用都显式
重置 start/end/groups/flags. 浮点比较容差 1e-6.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> c = np.array([1.] * 6 + [2.] * 6)   # 4 个证券, 每个 3 个时刻
    >>> v = np.array([10., 20., 30., 12., 18., 33.,
    ...               5., 7., 9., 6., 8., 10.])
    >>> al.set_ctx(start=0, end=0, groups=4, flags=0)

1. 类别 1 的残差为 -1/1, 类别 2 的残差为 -0.5/0.5; 全部残差排序后
   ``(rank_avg - 1) / 3`` 得到结果::

    >>> np.round(al.NEUTRALIZE(c, v), 5).tolist()
    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.33333, 0.33333, 0.33333, 0.66667, 0.66667, 0.66667]

2. 所有类别相同时, 残差排名等于原始值排名, 结果 0 / 0.5 / 1::

    >>> x = np.arange(1., 13.)
    >>> al.set_ctx(start=0, end=0, groups=3, flags=0)
    >>> al.NEUTRALIZE(np.ones(12), x).tolist()
    [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]

3. 常量输入的所有残差并列, 结果恒为 0.5::

    >>> np.unique(al.NEUTRALIZE(np.ones(12), np.full(12, 3.))).tolist()
    [0.5]

4. nan 被剔除: 类别 1 剩下 1 与 3, 类别 2 只有 10; 时刻 1 的两个残差
   都是 0 而并列::

    >>> cn = np.array([1., 1., 1., 1., 2., 2.])
    >>> vn = np.array([1., np.nan, 3., 5., 10., 20.])
    >>> al.set_ctx(start=0, end=0, groups=3, flags=0)
    >>> al.NEUTRALIZE(cn, vn).tolist()
    [0.0, nan, 1.0, 0.5, 0.5, 0.5]

5. ``groups = 1`` 时每个时刻只有一个证券, 输出 0.5::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.unique(al.NEUTRALIZE(c, v)).tolist()
    [0.5]

6. 单类别时与 ``CC_RANK`` 的换算关系::

    >>> al.set_ctx(start=0, end=0, groups=3, flags=0)
    >>> np.allclose(al.NEUTRALIZE(np.ones(12), x), (al.CC_RANK(x) * 3 - 1) / 2)
    True

7. ``flags`` 与 ``start`` / ``end`` 不改变当前实现的全量输出::

    >>> al.set_ctx(start=2, end=10, groups=3, flags=al.FLAG_SKIP_NAN)
    >>> al.NEUTRALIZE(np.ones(12), x).tolist()
    [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
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


def _neutralize_ref(category, x, groups):
  """类别中性化 + 残差排名参考实现, 只描述算法逻辑.

  逐时刻按 category 求组内均值得到残差, 再对全部残差做平均排名并归一化到
  [0, 1]; 只有一个有效项时按定义输出 0.5.
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
    residual = df["val"] - df.groupby("cat")["val"].transform("mean")
    if idx.size == 1:
      out[j, idx] = 0.5
      continue
    ranks = residual.rank(method="average").to_numpy()
    out[j, idx] = (ranks - 1.0) / (idx.size - 1)
  return out.T.reshape(-1)


def _expected(category, x, *, groups=1):
  """参考值总是全量计算; start/end/flags 由测试单独处理."""
  return _neutralize_ref(category, x, groups)


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
    al.NEUTRALIZE(
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
def test_neutralize_matches_pandas_reference(groups):
  """无 nan 数据在各 groups 下与 pandas 残差排名参考实现逐点比较."""
  category, x = _clean_pair()
  actual = _calc(category, x, groups=groups)
  _assert_close(actual, _expected(category, x, groups=groups))
  finite = actual[np.isfinite(actual)]
  assert ((finite >= 0) & (finite <= 1)).all()


@pytest.mark.parametrize("groups", GROUPS)
def test_neutralize_ties(groups):
  """离散重复值产生大量并列残差, 仍与参考实现一致."""
  category, x = _tie_pair()
  _assert_close(_calc(category, x, groups=groups), _expected(category, x, groups=groups))


def test_neutralize_unit_examples():
  """Rust 单元测试中的两个精确算例."""
  category = np.array([1.] * 6 + [2.] * 6)
  x = np.array([10., 20., 30., 12., 18., 33., 5., 7., 9., 6., 8., 10.])
  _assert_close(
    _calc(category, x, groups=4),
    np.array([0., 1., 0., 1., 0., 1., 1/3, 1/3, 1/3, 2/3, 2/3, 2/3]),
  )

  category = np.array([1., 1., 1., 1., 2., 2.])
  x = np.array([1., np.nan, 3., 5., 10., 20.])
  _assert_close(
    _calc(category, x, groups=3),
    np.array([0., np.nan, 1., 0.5, 0.5, 0.5]),
  )


@pytest.mark.parametrize("groups", [2, 3, 4, 6])
def test_neutralize_single_category_equals_rank_pct(groups):
  """单类别: 残差排名与 pandas 平均排名归一化完全一致."""
  x = _clean_pair()[1]
  category = np.ones(len(x))
  gs = len(x) // groups
  mat = x.reshape(groups, gs).T
  rank_avg = pd.DataFrame(mat).rank(axis=1, method="average").to_numpy()
  expected = ((rank_avg - 1.0) / (groups - 1)).T.reshape(-1)
  actual = _calc(category, x, groups=groups)
  _assert_close(actual, expected)
  _assert_close(actual, _expected(category, x, groups=groups))


@pytest.mark.parametrize("groups", [2, 3, 4, 6])
def test_neutralize_single_category_equals_cc_rank_transform(groups):
  """单类别: NEUTRALIZE = (CC_RANK * groups - 1) / (groups - 1)."""
  x = _clean_pair()[1]
  category = np.ones(len(x))
  actual = _calc(category, x, groups=groups)
  cc_rank = np.asarray(al.CC_RANK(x), dtype=np.float64)
  _assert_close(actual, (cc_rank * groups - 1.0) / (groups - 1.0))


@pytest.mark.parametrize("groups", GROUPS)
def test_neutralize_all_residuals_tied_is_half(groups):
  """常量输入的残差全为 0, 整组并列, 结果恒为 0.5."""
  category = np.ones(N)
  x = np.full(N, 3.0)
  actual = _calc(category, x, groups=groups)
  _assert_close(actual, np.full(N, 0.5))
  _assert_close(actual, _expected(category, x, groups=groups))


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_neutralize_nan_in_input_or_category(flags):
  """input / category 中任一为 nan 的位置输出 nan, 其余与参考一致."""
  category, x = _nan_pair()
  actual = _calc(category, x, groups=3, flags=flags)
  _assert_close(actual, _expected(category, x, groups=3))
  invalid = np.isnan(category) | np.isnan(x)
  assert np.isnan(actual[invalid]).all()
  assert np.isfinite(actual[~invalid]).all()


def test_neutralize_single_valid_item_in_cross_section():
  """某时刻只有一个有效项时输出 0.5, 无效项输出 nan."""
  category = np.ones(12)
  category[[4, 8]] = np.nan
  x = np.arange(1., 13.)
  actual = _calc(category, x, groups=3)
  _assert_close(actual, _expected(category, x, groups=3))
  assert actual[0] == pytest.approx(0.5, abs=ATOL)
  assert np.isnan(actual[[4, 8]]).all()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_neutralize_flags_do_not_change_result(flags):
  """四种 flag 组合结果与 flags=0 完全一致 (含 nan 数据)."""
  category, x = _nan_pair()
  base = _calc(category, x, groups=3, flags=0)
  _assert_close(_calc(category, x, groups=3, flags=flags), base)


@pytest.mark.parametrize("flags", [al.FLAG_SKIP_NAN, al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE])
def test_neutralize_skip_nan_does_not_use_nan(flags):
  """FLAG_SKIP_NAN: 结果等于排除所有无效项后的残差排名参考值."""
  category, x = _nan_pair()
  actual = _calc(category, x, groups=3, flags=flags)
  _assert_close(actual, _expected(category, x, groups=3))
  assert np.isnan(actual[np.isnan(category) | np.isnan(x)]).all()


@pytest.mark.parametrize("groups", [0, 1])
def test_neutralize_groups_less_than_two(groups):
  """groups<2 时只有单个证券: 有效位置 0.5, nan 位置 nan."""
  category, x = _nan_pair(12)
  actual = _calc(category, x, groups=groups)
  _assert_close(actual, _expected(category, x, groups=max(1, groups)))
  valid = ~np.isnan(category) & ~np.isnan(x)
  _assert_close(actual[valid], np.full(valid.sum(), 0.5))
  assert np.isnan(actual[~valid]).all()


@pytest.mark.parametrize("start,end", START_END)
@pytest.mark.parametrize("groups", [2, 3, 4, 6])
def test_neutralize_start_end(start, end, groups):
  """start/end: 仅校验 [start, end) 区间内与全量参考一致."""
  category, x = _nan_pair()
  expected = _expected(category, x, groups=groups)
  actual = _calc(category, x, start=start, end=end, groups=groups)
  _assert_range_close(actual, expected, start, end, groups)


@pytest.mark.parametrize("groups", GROUPS)
def test_neutralize_group_major_layout(groups):
  """group-major: 相同 chunk 的 category/value 在每个时刻构成相同截面."""
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


def test_neutralize_edge_cases():
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
