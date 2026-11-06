# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""CC_RANK (横截面排名百分比) 集成测试.

算法简介
--------
数据按 group-major 排布: ``groups`` 个证券各自占一段长度为 ``group_size``
的连续数据, 位置 ``i * group_size + j`` 表示证券 ``i`` 在时刻 ``j`` 的值.
对每个时刻 ``j``, 把该时刻所有证券的值做横截面升序排名, 并列取平均排名
(1-based), 再除以有效值个数得到排名百分比::

    CC_RANK[i, j] = rank_avg(i, j) / valid_count(j)      # 取值 (0, 1]

nan 位置输出 nan, 且既不参与排名也不计入分母; 若某时刻全是 nan, 该时刻全部
输出 nan. 只用一个证券时, 横截面只有一个有效值, 其排名百分比应为 1.0.

四种 ``flags`` 组合都不改变结果: 截面计算本身总是排除 nan,
``FLAG_SKIP_NAN`` / ``FLAG_STRICTLY_CYCLE`` 没有额外语义. ``start`` / ``end``
是 chunk 内相对下标, 当前实现忽略它们而输出全量结果, 因此测试只校验
``[start, end)`` 区间内的值. ``groups < 2`` 时当前实现回退到时序
``RANK(input, 0)`` 并输出全 nan, 与横截面排名逻辑不一致, 对应测试用
``pytest.mark.xfail`` 保留, 待算法侧修复.

参考实现 ``_cc_rank_ref`` 直接调用 pandas 的
``rank(axis=1, method='average', pct=True)``, 独立描述算法逻辑, 不调用 alpha,
也不实现 flags / start-end 机制. ``set_ctx`` 只覆盖传入的参数, 所以每次调用
都显式重置 start/end/groups/flags. 浮点比较容差 1e-6.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> x = np.arange(1., 13.)          # 3 个证券, 每个 4 个时刻
    >>> al.set_ctx(start=0, end=0, groups=3, flags=0)

1. 每个时刻的横截面是 ``[j+1, j+5, j+9]``, 排名 1/2/3 除以 3::

    >>> np.round(al.CC_RANK(x), 4).tolist()
    [0.3333, 0.3333, 0.3333, 0.3333, 0.6667, 0.6667, 0.6667, 0.6667, 1.0, 1.0, 1.0, 1.0]

2. 并列值取平均排名: 前两个证券相同, ``rank_avg = 1.5``, 百分比 0.5::

    >>> xt = np.array([1., 2., 3., 4., 1., 2., 3., 4., 9., 10., 11., 12.])
    >>> np.round(al.CC_RANK(xt), 4).tolist()
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]

3. nan 被排除: 时刻 1 的截面是 ``[nan, 6, 10]``, 分母变成 2::

    >>> xn = x.copy(); xn[1] = np.nan
    >>> al.CC_RANK(xn)[[1, 5, 9]]
    array([nan, 0.5, 1. ])

4. ``groups = 2`` 时每个时刻只有两个值, 结果为 0.5 / 1.0::

    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.CC_RANK(x), 4).tolist()
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

5. ``groups = 1`` 当前回退为时序 RANK, 输出全 nan (横截面逻辑应为 1.0)::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> bool(np.isnan(al.CC_RANK(x)).all())
    True

6. ``flags`` 与 ``start`` / ``end`` 不改变当前实现的全量输出::

    >>> al.set_ctx(start=2, end=4, groups=3, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.CC_RANK(x), 4).tolist()
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
GROUPS = [2, 3, 4, 6]
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


def _cc_rank_ref(x, groups):
  """横截面平均排名百分比参考实现, 只描述算法逻辑.

  pandas 的 ``rank(pct=True)`` 自动忽略 nan 并以非 nan 个数作分母, 与
  CC_RANK 的算法定义一致.
  """
  x = np.asarray(x, dtype=np.float64)
  if x.size == 0:
    return np.empty(0, dtype=np.float64)
  assert x.size % groups == 0
  group_size = x.size // groups
  mat = x.reshape(groups, group_size).T
  ranks = pd.DataFrame(mat).rank(axis=1, method="average", pct=True)
  return ranks.to_numpy().T.reshape(-1)


def _expected(x, *, groups=1):
  """参考值总是全量计算; start/end/flags 由测试单独处理."""
  return _cc_rank_ref(x, groups)


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
  return np.asarray(al.CC_RANK(np.asarray(x, dtype=np.float64)), dtype=np.float64)


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
def test_cc_rank_matches_pandas_reference(groups):
  """无 nan 数据在各 groups 下与 pandas rank(pct=True) 参考实现逐点比较."""
  x = _clean_data()
  actual = _calc(x, groups=groups)
  _assert_close(actual, _expected(x, groups=groups))
  assert np.isfinite(actual).all()
  assert ((actual > 0) & (actual <= 1)).all()


@pytest.mark.parametrize("groups", GROUPS)
def test_cc_rank_ties_use_average_rank(groups):
  """并列值共享平均排名, 且结果与 pandas 参考一致."""
  x = _tie_data()
  actual = _calc(x, groups=groups)
  _assert_close(actual, _expected(x, groups=groups))

  mat = x.reshape(groups, -1).T
  out = actual.reshape(groups, -1).T
  for row, out_row in zip(mat, out):
    for val in np.unique(row):
      assert np.all(out_row[row == val] == out_row[row == val][0])
    order = np.argsort(row)
    assert np.all(np.diff(out_row[order]) >= -ATOL)


def test_cc_rank_average_rank_formula():
  """小数组精确验证: 两个并列值 rank_avg=1.5, 百分比 0.5."""
  xt = np.array([1., 2., 3., 4., 1., 2., 3., 4., 9., 10., 11., 12.])
  expected = np.array([0.5] * 8 + [1.0] * 4)
  _assert_close(_calc(xt, groups=3), expected)
  _assert_close(_expected(xt, groups=3), expected)


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_cc_rank_nan_excluded_from_denominator(flags):
  """nan 位置为 nan, 有效值按剩余有效个数计算百分比."""
  x = _nan_data()
  actual = _calc(x, groups=3, flags=flags)
  _assert_close(actual, _expected(x, groups=3))
  assert np.isnan(actual[np.isnan(x)]).all()
  assert np.isfinite(actual[~np.isnan(x)]).all()


def test_cc_rank_all_nan_cross_section():
  """某时刻横截面全是 nan 时该时刻全部输出 nan, 不影响其它时刻."""
  x = np.arange(1., 13.)
  x[1], x[5], x[9] = np.nan, np.nan, np.nan
  actual = _calc(x, groups=3)
  _assert_close(actual, _expected(x, groups=3))
  assert np.isnan(actual[[1, 5, 9]]).all()
  assert np.isfinite(actual[[0, 4, 8]]).all()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_cc_rank_flags_do_not_change_result(flags):
  """四种 flag 组合结果与 flags=0 完全一致 (含 nan 数据)."""
  x = _nan_data()
  base = _calc(x, groups=3, flags=0)
  _assert_close(_calc(x, groups=3, flags=flags), base)


@pytest.mark.parametrize("flags", [al.FLAG_SKIP_NAN, al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE])
def test_cc_rank_skip_nan_does_not_use_nan(flags):
  """FLAG_SKIP_NAN: 结果等于排除 nan 后的横截面排名参考值."""
  x = _nan_data()
  actual = _calc(x, groups=3, flags=flags)
  _assert_close(actual, _expected(x, groups=3))
  assert np.isnan(actual[np.isnan(x)]).all()


@pytest.mark.parametrize("groups", GROUPS)
def test_cc_rank_group_major_layout(groups):
  """group-major: 相同 chunk 数据构成全并列截面, 排名为 (n+1)/(2n)."""
  x = _clean_data()
  _assert_close(_calc(x, groups=groups), _expected(x, groups=groups))

  size = len(x) // groups
  repeated = np.tile(x[:size], groups)
  expected_pct = (groups + 1) / (2 * groups)
  actual = _calc(repeated, groups=groups)
  _assert_close(actual, np.full(len(repeated), expected_pct))
  _assert_close(actual, _expected(repeated, groups=groups))


@pytest.mark.parametrize("start,end", START_END)
@pytest.mark.parametrize("groups", GROUPS)
def test_cc_rank_start_end(start, end, groups):
  """start/end: 仅校验 [start, end) 区间内与全量参考一致."""
  x = _nan_data()
  expected = _expected(x, groups=groups)
  actual = _calc(x, start=start, end=end, groups=groups)
  _assert_range_close(actual, expected, start, end, groups)


def test_cc_rank_single_group_matches_cross_section_logic():
  """groups=1: 唯一的证券在每个时刻都应得到排名百分比 1.0."""
  x = np.arange(1., 7.)
  _assert_close(_calc(x, groups=1), _expected(x, groups=1))


def test_cc_rank_edge_cases():
  """空输入 / 长度不整除 / 单元素 / 全 nan / groups=0."""
  empty = np.empty(0, dtype=np.float64)
  _assert_close(_calc(empty, groups=1), empty)
  _assert_close(_expected(empty, groups=1), empty)

  with pytest.raises(ValueError):
    _calc(np.ones(10), groups=3)

  one = np.array([3.0])
  _assert_close(_calc(one, groups=1), np.array([1.0]))

  all_nan = np.full(12, np.nan)
  assert np.isnan(_calc(all_nan, groups=3, flags=al.FLAG_SKIP_NAN)).all()

  # groups=0 等价于 groups=1
  _assert_close(_calc(_clean_data(12), groups=0), np.ones(12))
