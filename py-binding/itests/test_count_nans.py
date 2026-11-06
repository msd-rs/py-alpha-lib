# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""COUNT_NANS (滚动窗口内 nan 计数) 集成测试.

算法简介
--------
输出最近 `periods` 个原始位置中 nan 的个数 (以浮点形式表示)::

    COUNT_NANS[i] = sum(isnan(x[k]) for k in max(0, i-periods+1) .. i)

窗口未满 (``i < periods-1``) 时, ``flags=0`` 仍按已有的 1..periods 个值
给出部分计数, 等价于 pandas ``rolling(periods, min_periods=1).sum()``; 设置
``FLAG_STRICTLY_CYCLE`` 后前 ``periods-1`` 个位置输出 nan, 等价于
``min_periods=periods``.

与其它滚动算子不同, **当前位置是 nan 时输出仍然是数字**: 它自己也要被计入
窗口, 结果不会变成 nan; 无 nan 的序列结果恒为 0, 全 nan 的序列结果恒为
``min(i+1, periods)``. ``FLAG_SKIP_NAN`` 对该算子没有作用 (源码未使用该
flag), 四种 flag 组合里只有 STRICTLY_CYCLE 会影响输出.

参考实现 ``_count_nans_ref`` 用 pandas rolling sum 独立复现算法逻辑, 不调用
alpha, 也不实现 SKIP_NAN / Group / start-end 机制; 这些上下文语义由
``_expected`` 单独描述: Group 通过按 chunk 分块计算, start/end 只校验
[start, end) 区间, 窗口本身仍可引用 start 之前的数据 (算法内部用 pre_fill
继承计数), 所以先在整个 chunk 上全量计算再屏蔽区间外.

注意 ``periods=0`` 在 Rust 实现里会触发 ``periods - 1`` 的 usize 下溢, 属于
实现陷阱, 测试不覆盖该取值, 只使用 ``periods >= 1``.

`set_ctx` 只覆盖传入的参数, 所以每次调用都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=120)
    >>> c = np.array([1., np.nan, 3., np.nan, np.nan])
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算 (3 周期窗口). i=1 的窗口 [1, nan] 含 1 个 nan, i=3 的窗口
   [nan, 3, nan] 含 2 个::

    >>> al.COUNT_NANS(c, 3)
    array([0., 1., 1., 2., 2.])

2. ``FLAG_STRICTLY_CYCLE``: 前 2 个位置窗口未满, 输出 nan::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> al.COUNT_NANS(c, 3)
    array([nan, nan,  1.,  2.,  2.])

3. ``FLAG_SKIP_NAN`` 无影响 (当前位置是 nan 也照样输出计数)::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> al.COUNT_NANS(c, 3)
    array([0., 1., 1., 2., 2.])

4. ``periods=1`` 时窗口只含当前值, 输出即 nan 指示::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> al.COUNT_NANS(c, 1)
    array([0., 1., 0., 1., 1.])

5. ``groups`` 把一维数据按 group-major 切块, 每块的窗口不跨边界::

    >>> c2 = np.array([1., np.nan, np.nan, 4., 5., np.nan, 7., 8., np.nan, 10.])
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> al.COUNT_NANS(c2, 3)
    array([0., 1., 2., 2., 1., 1., 1., 1., 1., 1.])

6. ``start`` / ``end`` 只计算 [start, end) 区间, 区间外全部为 nan,
   但 i=2 的窗口 [1, nan, nan] 仍引用了 start 之前的数据::

    >>> al.set_ctx(start=2, end=8, groups=1, flags=0)
    >>> al.COUNT_NANS(c2, 3)
    array([nan, nan,  2.,  2.,  1.,  1.,  1.,  1., nan, nan])
"""

import numpy as np
import pandas as pd
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
NAN_POS = np.array([0, 1, 3, 17, 18, 29, 44, 52, 77, 101])
PERIODS = [1, 2, 3, 4, 5, 10, 20]
GROUPS = [1, 2, 3, 4, 6]
START_END = [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5), (-10, 0), (40, 20)]
FLAG_COMBOS = [
  0,
  al.FLAG_STRICTLY_CYCLE,
  al.FLAG_SKIP_NAN,
  al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
]


# ---------------------------------------------------------------------------
# numpy / pandas 对比实现
# ---------------------------------------------------------------------------


def _count_nans_ref(x, periods, *, strict):
  """pandas rolling sum 复现窗口内 nan 计数."""
  x = np.asarray(x, dtype=np.float64)
  is_nan = np.isnan(x).astype(np.float64)
  min_periods = periods if strict else 1
  return (
    pd.Series(is_nan)
    .rolling(periods, min_periods=min_periods)
    .sum()
    .to_numpy(dtype=np.float64)
    .copy()
  )


def _calc_range(n, start, end):
  """对应 Rust ``Context::start`` / ``Context::end`` 在单个 chunk 内的解析."""
  s = min(n - 1, start) if start >= 0 else max(0, n + start)
  if end == 0:
    e = n
  elif end > 0:
    e = min(n, end)
  else:
    e = max(0, n + end)
  return s, e


def _expected(x, periods, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 全量计算参考值, 再只保留 [start, end) 区间.

  窗口可引用 start 之前的数据, 因此先在整个 chunk 上计算 rolling 计数,
  再把区间外屏蔽成 nan; SKIP_NAN 被忽略, 只有 STRICTLY_CYCLE 生效.
  """
  x = np.asarray(x, dtype=np.float64)
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  parts = []
  for chunk in np.split(x, groups):
    n = len(chunk)
    s, e = _calc_range(n, start, end)
    ref = _count_nans_ref(chunk, periods, strict=strict)
    ref[:s] = np.nan
    ref[e:] = np.nan
    parts.append(ref)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(x, periods, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.COUNT_NANS(np.asarray(x, dtype=np.float64), periods),
    dtype=np.float64,
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261106)
  return rng.uniform(5.0, 20.0, size=n).astype(np.float64)


def _nan_data(n=N):
  x = _clean_data(n)
  x[NAN_POS[NAN_POS < n]] = np.nan
  return x


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("periods", PERIODS)
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_count_nans_matches_pandas_rolling(periods, flags):
  """含 nan 数据上与 pandas rolling sum 逐点比较."""
  x = _nan_data()
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  _assert_close(
    actual,
    _count_nans_ref(x, periods, strict=bool(flags & al.FLAG_STRICTLY_CYCLE)),
  )


@pytest.mark.parametrize("periods", PERIODS)
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_count_nans_no_nan_data(periods, flags):
  """无 nan 序列: 常规模式恒为 0, 严格模式前 periods-1 个为 nan."""
  x = _clean_data()
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  if flags & al.FLAG_STRICTLY_CYCLE:
    assert np.isnan(actual[: periods - 1]).all()
    _assert_close(actual[periods - 1 :], np.zeros(len(x) - periods + 1))
  else:
    _assert_close(actual, np.zeros(len(x)))


@pytest.mark.parametrize("periods", PERIODS)
def test_count_nans_strictly_cycle(periods):
  """FLAG_STRICTLY_CYCLE: 等价于 pandas min_periods=periods."""
  x = _nan_data()
  flags = al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  _assert_close(actual, _count_nans_ref(x, periods, strict=True))
  assert np.isnan(actual[: periods - 1]).all()
  assert np.isfinite(actual[periods - 1 :]).all()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_count_nans_skip_nan_has_no_effect(flags):
  """四种 flag 组合: SKIP_NAN 不改变结果, 只有 STRICTLY_CYCLE 生效."""
  x = _nan_data()
  periods = 6
  actual = _calc(x, periods, flags=flags)
  plain_flags = flags & ~al.FLAG_SKIP_NAN
  _assert_close(actual, _expected(x, periods, flags=flags))
  _assert_close(actual, _calc(x, periods, flags=plain_flags))


def test_count_nans_current_nan_is_still_counted():
  """当前位置是 nan 时输出仍是计数 (数字), 不是 nan."""
  x = _nan_data()
  periods = 4
  actual = _calc(x, periods)
  nan_positions = np.isnan(x)
  assert np.isfinite(actual[nan_positions]).all()
  _assert_close(actual, _expected(x, periods))
  for i in np.flatnonzero(nan_positions):
    expected = np.isnan(x[max(0, i - periods + 1) : i + 1]).sum()
    _assert_close(actual[i : i + 1], np.array([float(expected)]))


@pytest.mark.parametrize("periods", PERIODS)
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_count_nans_all_nan_data(periods, flags):
  """全 nan 序列: 计数为 min(i+1, periods)."""
  x = np.full(N, np.nan)
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  ref = np.minimum(np.arange(1, len(x) + 1, dtype=np.float64), periods)
  if flags & al.FLAG_STRICTLY_CYCLE:
    ref[: periods - 1] = np.nan
  _assert_close(actual, ref)


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_count_nans_groups(groups, flags):
  """group-major 分块: 每个 chunk 的窗口不跨边界, 各自重新 warm-up."""
  x = _nan_data()
  periods = 4
  actual = _calc(x, periods, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, periods, groups=groups, flags=flags))

  chunk_size = len(x) // groups
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  for g in range(groups):
    lo = g * chunk_size
    chunk = x[lo : lo + chunk_size]
    _assert_close(
      actual[lo : lo + chunk_size],
      _count_nans_ref(chunk, periods, strict=strict),
    )


@pytest.mark.parametrize("start,end", START_END)
def test_count_nans_start_end(start, end):
  """start/end: 仅 [start, end) 有结果, 区间外必须全是 nan."""
  x = _nan_data()
  periods = 4
  actual = _calc(x, periods, start=start, end=end)
  _assert_close(actual, _expected(x, periods, start=start, end=end))
  s, e = _calc_range(len(x), start, end)
  if s >= e:
    assert np.isnan(actual).all()
  else:
    assert np.isnan(actual[:s]).all()
    assert np.isnan(actual[e:]).all()
    assert np.isfinite(actual[s:e]).all()


@pytest.mark.parametrize("groups", [1, 2, 3])
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_count_nans_start_end_with_groups(groups, flags):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  x = _nan_data()
  periods = 4
  start, end = 6, 25
  ctx = dict(start=start, end=end, groups=groups, flags=flags)
  actual = _calc(x, periods, **ctx)
  _assert_close(actual, _expected(x, periods, **ctx))
  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


def test_count_nans_window_uses_pre_start_data():
  """start>0 时窗口仍引用 start 之前的 nan (pre_fill 继承计数)."""
  x = np.arange(1.0, 21.0)
  x[9] = np.nan
  periods = 3
  start, end = 10, 18
  actual = _calc(x, periods, start=start, end=end)
  _assert_close(actual, _expected(x, periods, start=start, end=end))
  _assert_close(actual[10:13], np.array([1.0, 1.0, 0.0]))


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_count_nans_skip_nan_with_negative_end_and_groups(groups):
  """SKIP_NAN + 负数 end + groups 组合."""
  x = _nan_data()
  periods = 5
  ctx = dict(start=5, end=-5, groups=groups, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, periods, **ctx)
  _assert_close(actual, _expected(x, periods, **ctx))


def test_count_nans_edge_cases():
  """periods=1、窗口长于数据、连续 nan 段、前导 nan、长度不匹配."""
  x = _nan_data()
  _assert_close(_calc(x, 1), np.isnan(x).astype(np.float64))

  periods = len(x) + 1
  _assert_close(_calc(x, periods), _expected(x, periods))
  assert np.isnan(_calc(x, periods, flags=al.FLAG_STRICTLY_CYCLE)).all()

  dense = np.full(30, np.nan)
  dense[10:20] = np.arange(10.0, 20.0)
  for periods in (3, 8):
    _assert_close(_calc(dense, periods), _expected(dense, periods))
    _assert_close(
      _calc(dense, periods, flags=al.FLAG_STRICTLY_CYCLE),
      _expected(dense, periods, flags=al.FLAG_STRICTLY_CYCLE),
    )

  one = np.array([np.nan])
  _assert_close(_calc(one, 1), np.array([1.0]))
  _assert_close(_calc(np.ones(N), 3), np.zeros(N))
