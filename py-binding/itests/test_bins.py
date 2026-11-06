# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""BINS (横截面等宽分箱) 集成测试.

算法简介
--------
数据按 group-major 排布: ``groups`` 个证券各自占一段长度为 ``group_size``
的连续数据, 位置 ``i * group_size + j`` 表示证券 ``i`` 在时刻 ``j`` 的值.
对每个时刻 ``j``, 把该时刻所有证券的值做横截面升序排名 (并列取平均排名,
1-based), 再映射成 ``bins`` 个 0-based 等宽箱::

    bin = floor((rank_avg - 1) * bins / total)   # clamp 到 [0, bins - 1]

``total`` 为该时刻的有效值个数. 相同值一定落在同一个箱; ``bins >= groups``
时部分箱会为空; ``bins = 1`` 时所有有效值都是 0; ``bins = 0`` 是退化输入,
当前实现直接返回全 0 (连 nan 位置也是 0).

缺失值: 按算法逻辑 nan 位置输出 nan, 且 nan 既不参与排名也不计入 ``total``,
因此不会影响同一时刻其它证券的分箱. 注意当前实现让 nan 占据排序位置且
``total`` 仍取 ``groups``, 含 nan 时与上述参考不一致, 对应测试用
``pytest.mark.xfail`` 保留, 待算法侧修复. 四种 ``flags`` 组合都不改变结果:
截面计算本身总是排除 nan, ``FLAG_SKIP_NAN`` / ``FLAG_STRICTLY_CYCLE`` 无额外
语义. ``start`` / ``end`` 是 chunk 内相对下标, 当前实现忽略它们而输出全量结果,
因此测试只校验 ``[start, end)`` 区间内的值.

参考实现 ``_bins_ref`` 用 pandas 的 ``rank(method='average')`` 与 numpy 的
``floor`` 独立复现分箱公式, 不调用 alpha, 也不实现 flags / start-end 机制.
``set_ctx`` 只覆盖传入的参数, 所以每次调用都显式重置 start/end/groups/flags.
浮点比较容差 1e-6.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> x = np.arange(1., 13.)          # 3 个证券, 每个 4 个时刻
    >>> al.set_ctx(start=0, end=0, groups=3, flags=0)

1. 每个时刻的横截面是 ``[j+1, j+5, j+9]``, 排名 1/2/3 分别进箱 0/1/2::

    >>> al.BINS(x, 3).astype(int).tolist()
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

2. 并列值取平均排名: 前两个证券取值相同, ``rank_avg = 1.5``, 进箱 0;
   第三个证券排名 3, 进箱 2, 箱 1 为空::

    >>> xt = np.array([1., 2., 3., 4., 1., 2., 3., 4., 9., 10., 11., 12.])
    >>> al.BINS(xt, 3).astype(int).tolist()
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]

3. 退化分箱: ``bins = 0`` 与 ``bins = 1`` 都输出全 0::

    >>> al.BINS(x, 0).astype(int).tolist()
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    >>> al.BINS(x, 1).astype(int).tolist()
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

4. nan 位置输出 nan, 其它位置仍是合法箱号::

    >>> xn = x.copy(); xn[1] = np.nan
    >>> bool(np.isnan(al.BINS(xn, 3))[1])
    True
    >>> bool(np.isfinite(al.BINS(xn, 3)[~np.isnan(xn)]).all())
    True

5. ``groups = 2`` 时每个时刻只有 2 个值, 箱号只能是 0 或 1::

    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> al.BINS(x, 3).astype(int).tolist()
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

6. ``flags`` 与 ``start`` / ``end`` 不改变当前实现的全量输出::

    >>> al.set_ctx(start=2, end=4, groups=3,
    ...            flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> al.BINS(x, 3).astype(int).tolist()
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
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
# pandas / numpy 对比实现
# ---------------------------------------------------------------------------


def _bins_ref(x, bins, groups):
  """横截面等宽分箱参考实现, 只描述算法逻辑.

  nan 不参与排名也不计入 total; ``bins = 0`` 是退化输入, 直接返回全 0.
  """
  x = np.asarray(x, dtype=np.float64)
  if x.size == 0:
    return np.empty(0, dtype=np.float64)
  assert x.size % groups == 0
  group_size = x.size // groups
  mat = x.reshape(groups, group_size).T
  if bins == 0:
    return np.zeros(x.size, dtype=np.float64)

  valid = ~np.isnan(mat)
  counts = valid.sum(axis=1, keepdims=True)
  ranks = pd.DataFrame(mat).rank(axis=1, method="average").to_numpy()
  denom = np.where(counts == 0, 1, counts)
  values = np.minimum(np.floor((ranks - 1.0) * bins / denom), bins - 1)
  out = np.where(valid, values, np.nan)
  return out.T.reshape(-1)


def _expected(x, bins, *, groups=1):
  """参考值总是全量计算; start/end/flags 由测试单独处理."""
  return _bins_ref(x, bins, groups)


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


def _calc(x, bins, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.BINS(np.asarray(x, dtype=np.float64), bins), dtype=np.float64
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


@pytest.mark.parametrize("bins", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("groups", GROUPS)
def test_bins_matches_pandas_reference(bins, groups):
  """无 nan 数据在各 groups / bins 下与 pandas 平均排名参考实现逐点比较."""
  x = _clean_data()
  actual = _calc(x, bins, groups=groups)
  _assert_close(actual, _bins_ref(x, bins, groups))
  assert np.isfinite(actual).all()
  assert ((actual >= 0) & (actual <= bins - 1)).all()


@pytest.mark.parametrize("bins", [1, 2, 3, 5, 7])
def test_bins_ties_share_same_bin(bins):
  """同一时刻的相同值必须落进同一个箱."""
  x = _tie_data()
  groups = 4
  actual = _calc(x, bins, groups=groups)
  _assert_close(actual, _bins_ref(x, bins, groups))

  mat = x.reshape(groups, -1).T
  out = actual.reshape(groups, -1).T
  for row, out_row in zip(mat, out):
    for val in np.unique(row):
      assert np.all(out_row[row == val] == out_row[row == val][0])


def test_bins_average_rank_formula():
  """并列值的平均排名参与 floor 运算, 而不是最小/最大排名."""
  xt = np.array(
    [1., 2., 3., 4., 1., 2., 3., 4., 9., 10., 11., 12.]
  )
  expected = np.array([0.] * 8 + [2.] * 4)
  _assert_close(_calc(xt, 3, groups=3), expected)
  _assert_close(_expected(xt, 3, groups=3), expected)


@pytest.mark.parametrize("bins", [0, 1])
def test_bins_degenerate_bins(bins):
  """bins=0 / bins=1 全 0; bins=0 连 nan 位置也填 0."""
  x = _clean_data()
  _assert_close(_calc(x, bins, groups=3), np.zeros(len(x)))
  _assert_close(_expected(x, bins, groups=3), np.zeros(len(x)))

  if bins == 0:
    xn = _nan_data()
    _assert_close(_calc(xn, 0, groups=3), np.zeros(len(xn)))


@pytest.mark.parametrize("bins", [1, 3, 5, 10])
@pytest.mark.parametrize("groups", [2, 3, 4, 6])
def test_bins_constant_cross_section(bins, groups):
  """常量截面全是并列值, 平均排名 (1 + groups) / 2 决定统一的箱号."""
  const = np.full(N, 3.0)
  actual = _calc(const, bins, groups=groups)
  _assert_close(actual, _bins_ref(const, bins, groups))
  assert np.unique(actual).size == 1


def test_bins_more_bins_than_groups():
  """bins 大于证券数时箱号仍受公式约束, 且不超过 bins-1."""
  x = _clean_data()
  actual = _calc(x, 10, groups=3)
  _assert_close(actual, _bins_ref(x, 10, 3))
  assert np.array_equal(np.unique(actual), np.array([0., 3., 6.]))


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_bins_nan_positions_are_nan(flags):
  """含 nan 数据时 nan 位置输出 nan, 其余位置是有限箱号."""
  x = _nan_data()
  actual = _calc(x, 3, groups=3, flags=flags)
  assert np.isnan(actual[np.isnan(x)]).all()
  assert np.isfinite(actual[~np.isnan(x)]).all()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_bins_flags_do_not_change_result(flags):
  """四种 flag 组合结果与 flags=0 完全一致 (含 nan 数据)."""
  x = _nan_data()
  base = _calc(x, 3, groups=3, flags=0)
  _assert_close(_calc(x, 3, groups=3, flags=flags), base)


@pytest.mark.parametrize(
  "flags", [al.FLAG_SKIP_NAN, al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE]
)
def test_bins_skip_nan_excludes_nan_from_ranking(flags):
  """FLAG_SKIP_NAN 应确保 nan 不参与排名/计数, 结果等于排除 nan 的参考值.

  当前实现中 ``[2, 6, nan]`` 仍按 3 个位置排名, 得到 1/2 而不是参考的 0/1.
  """
  x = np.arange(1., 13.)
  x[9] = np.nan
  actual = _calc(x, 3, groups=3, flags=flags)
  _assert_close(actual, _expected(x, 3, groups=3))


@pytest.mark.parametrize("groups", GROUPS)
def test_bins_group_major_layout(groups):
  """group-major: 每个 chunk 是一个证券, 相同 chunk 数据构成全并列截面."""
  x = _clean_data()
  _assert_close(_calc(x, 3, groups=groups), _expected(x, 3, groups=groups))

  size = len(x) // groups
  repeated = np.tile(x[:size], groups)
  actual = _calc(repeated, 3, groups=groups)
  _assert_close(actual, _bins_ref(repeated, 3, groups))
  assert np.unique(actual).size == 1


@pytest.mark.parametrize("start,end", START_END)
@pytest.mark.parametrize("groups", [2, 3, 4, 6])
def test_bins_start_end(start, end, groups):
  """start/end: 仅校验 [start, end) 区间内与全量参考一致."""
  x = _clean_data()
  expected = _expected(x, 3, groups=groups)
  actual = _calc(x, 3, start=start, end=end, groups=groups)
  _assert_range_close(actual, expected, start, end, groups)


def test_bins_edge_cases():
  """空输入 / 长度不整除 / 负 bins / 单元素 / 全 nan."""
  with pytest.raises(ValueError):
    _calc(np.empty(0, dtype=np.float64), 3)
  with pytest.raises(ValueError):
    _calc(np.ones(10), 3, groups=3)
  with pytest.raises(OverflowError):
    _calc(np.ones(12), -1, groups=3)

  one = np.array([3.0])
  _assert_close(_calc(one, 3, groups=1), np.array([0.0]))

  all_nan = np.full(12, np.nan)
  actual = _calc(all_nan, 3, groups=3, flags=al.FLAG_SKIP_NAN)
  assert np.isnan(actual).all()
