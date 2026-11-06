# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""LLVBARS (距滚动最低值的周期数) 集成测试.

算法简介
--------
LLVBARS 给出当前位置距最近 `periods` 个值中最低值所在位置有多少根 bar::

    LLVBARS[i] = i - argmin_last(x[i - periods + 1 : i + 1])

``argmin_last`` 表示并列最低时取最靠后的位置, 因此当前值刷新最低值时结果为
0. 与 LLV 共用同一套极值框架, 与 HHVBARS 对称. ``flags=0`` 时窗口未满也按
已有值计算; 设置 ``FLAG_STRICTLY_CYCLE`` 后窗口未满输出 nan. `periods=0` 是
特例: 从头开始的累积最低值, 输出距最近一次刷新累积最低值的 bar 数.

不设置 ``FLAG_SKIP_NAN`` 时, 窗口内出现 nan 会使该窗口结果为 nan; 设置后
nan 不参与极值的选取, 窗口改为最近 `periods` 个有效值, 但 bar 距离仍按原始
位置计算 (nan 也占一根 bar), nan 位置本身输出 nan. 再组合
``FLAG_STRICTLY_CYCLE`` 时还要求最近 `periods` 个原始位置全部有效.

参考实现只在无 nan 数据上用 pandas ``rolling`` 配合 numpy ``argmin`` 计算,
SKIP_NAN 与累积模式用独立的 numpy 实现; 均不实现 Group / start-end 机制.
测试里 Group 通过按 chunk 分块计算实现, start/end 只校验 [start, end) 区间,
区间外必须是 nan; 窗口本身仍可引用 start 之前的数据 (SKIP_NAN 和 periods=0
除外, 它们只在 [start, end) 内计算). ``set_ctx`` 只覆盖传入的参数, 所以每次
调用都显式重置 start/end/groups/flags. 浮点比较容差 1e-6.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=150)
    >>> x = np.array([1., 3., 2., 5., 4., 6., 8., 7., 9., 10.])
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算 (窗口未满也按已有值计算, 并列取最靠后)::

    >>> np.round(al.LLVBARS(x, 4), 3)
    array([0., 1., 2., 3., 2., 3., 2., 3., 3., 2.])

   i=3 的窗口 [1, 3, 2, 5] 最低值 1 在 3 根 bar 之前, 结果为 3.

2. ``FLAG_STRICTLY_CYCLE`` (窗口未满输出 nan)::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.LLVBARS(x, 4), 3)
    array([nan, nan, nan,  3.,  2.,  3.,  2.,  3.,  3.,  2.])

3. 不跳过 nan: 窗口内任一 nan 都会污染整个窗口::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.LLVBARS(xn, 4), 3)
    array([ 0.,  1.,  2., nan, nan, nan, nan,  3.,  3.,  2.])

4. ``FLAG_SKIP_NAN``: nan 不参与极值选取, 但距离仍按原始位置计算 (i=4 的
   最低值 1 在 4 根 bar 之前)::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.LLVBARS(xn, 4), 3)
    array([ 0.,  1.,  2., nan,  4.,  3.,  4.,  3.,  3.,  2.])

5. ``FLAG_SKIP_NAN | FLAG_STRICTLY_CYCLE``: 要求连续 `periods` 个原始位置
   全部有效, i=7 的窗口 [4, 6, 8, 7] 才重新完整::

    >>> al.set_ctx(start=0, end=0, groups=1,
    ...            flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.LLVBARS(xn, 4), 3)
    array([nan, nan, nan, nan, nan, nan, nan,  3.,  3.,  2.])

6. ``groups`` 把一维数据按 group-major 切块独立计算, 每块重新计窗口::

    >>> y = np.array([3., 1., 4., 1., 5., 9., 2., 6., 5., 3., 5., 8.])
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.LLVBARS(y, 4), 3)
    array([0., 0., 1., 0., 1., 2., 0., 1., 2., 3., 1., 2.])

7. ``start`` / ``end`` 只计算 [start, end) 区间, 区间外全部为 nan,
   但窗口仍可引用 start 之前的数据::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.LLVBARS(x, 4), 3)
    array([nan, nan, nan,  3.,  2.,  3.,  2.,  3., nan, nan])

8. ``periods=0`` 计算距最近一次刷新累积最低值的 bar 数 (首个值 1 一直是最
   低值)::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.LLVBARS(x, 0), 3)
    array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
"""

import numpy as np
import pandas as pd
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
NAN_POS = np.array([0, 1, 3, 17, 18, 29, 44, 52, 77, 101])


# ---------------------------------------------------------------------------
# numpy / pandas 对比实现
# ---------------------------------------------------------------------------


def _bars_ref(x, periods, *, strict):
  """无 nan 数据上距窗口最低值的 bar 数 (并列取最靠后)."""
  x = np.asarray(x, dtype=np.float64)
  assert not np.isnan(x).any()
  assert periods > 0
  min_periods = periods if strict else 1
  return (
    pd.Series(x)
    .rolling(periods, min_periods=min_periods)
    .apply(lambda w: float(np.argmin(w[::-1])), raw=True)
    .to_numpy(dtype=np.float64)
  )


def _bars_cum_ref(x, *, skip):
  """periods=0: 距最近一次刷新累积最低值的原始 bar 数.

  不跳过 nan 时累积窗口含 nan 即无法定义, 首个 nan 及其后均为 nan; 跳过 nan
  时忽略 nan, nan 位置也照常按原始索引计数.
  """
  x = np.asarray(x, dtype=np.float64)
  n = len(x)
  valid = ~np.isnan(x)
  running = np.minimum.accumulate(np.where(valid, x, np.inf))
  idx = np.where(valid & (x == running), np.arange(n), -1)
  best = np.maximum.accumulate(idx)
  bars = np.arange(n, dtype=np.float64) - best
  bars[best < 0] = np.nan
  if not skip:
    clean_prefix = np.cumsum((~valid).astype(np.int64)) == 0
    bars[~clean_prefix] = np.nan
  return bars


def _bars_skip_ref(seg, periods, *, strict):
  """SKIP_NAN: 窗口取最近 periods 个有效值, bar 距离按原始位置计算."""
  seg = np.asarray(seg, dtype=np.float64)
  out = np.full(len(seg), np.nan)
  valid = ~np.isnan(seg)
  window = []
  for i in range(len(seg)):
    if not valid[i]:
      continue
    window.append(i)
    if len(window) > periods:
      window.pop(0)
    if strict and (
      len(window) != periods or i - window[0] + 1 != periods
    ):
      continue
    vals = seg[window]
    last = len(vals) - 1 - int(np.argmin(vals[::-1]))
    out[i] = i - window[last]
  return out


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


def _no_nan_window_mask(x, periods):
  """每个位置的滚动窗口 (最多 periods 个原始值) 是否完全不含 nan."""
  nan_count = (
    pd.Series(np.isnan(x).astype(np.float64))
    .rolling(periods, min_periods=1)
    .sum()
    .to_numpy()
  )
  return nan_count == 0


def _compact_ref(x, periods, strict):
  """把有效值压紧后跑参考实现, 再映射回有效位置; nan 位置保持 nan."""
  out = np.full(len(x), np.nan)
  valid = ~np.isnan(x)
  if valid.any():
    out[valid] = _bars_ref(x[valid], periods, strict=strict)
  return out


def _expected(x, periods, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 全量计算参考值, 再只保留 alpha 应计算的 [start, end) 区间."""
  x = np.asarray(x, dtype=np.float64)
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  skip = bool(flags & al.FLAG_SKIP_NAN)
  parts = []
  for chunk in np.split(x, groups):
    n = len(chunk)
    s, e = _calc_range(n, start, end)
    ref = np.full(n, np.nan)
    if s < e:
      if periods == 0:
        ref[s:e] = _bars_cum_ref(chunk[s:e], skip=skip)
      elif skip:
        ref[s:e] = _bars_skip_ref(chunk[s:e], periods, strict=strict)
      else:
        ref = _compact_ref(chunk, periods, strict)
        ref[~_no_nan_window_mask(chunk, periods)] = np.nan
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
    al.LLVBARS(np.asarray(x, dtype=np.float64), periods), dtype=np.float64
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261106)
  return rng.normal(loc=10.0, scale=2.0, size=n).astype(np.float64)


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


@pytest.mark.parametrize("periods", [2, 3, 4, 5, 10, 20])
def test_llvbars_matches_pandas_rolling(periods):
  """flags=0, groups=1: 与 pandas rolling + argmin 逐点比较."""
  x = _clean_data()
  actual = _calc(x, periods, flags=0)
  _assert_close(actual, _expected(x, periods, flags=0))
  _assert_close(actual, _bars_ref(x, periods, strict=False))
  assert np.isfinite(actual).all()


@pytest.mark.parametrize("periods", [2, 3, 4, 5, 10, 20])
def test_llvbars_strictly_cycle_matches_pandas(periods):
  """FLAG_STRICTLY_CYCLE: 窗口未满输出 nan."""
  x = _clean_data()
  flags = al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  _assert_close(actual, _bars_ref(x, periods, strict=True))
  assert np.isnan(actual[: periods - 1]).all()


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_llvbars_nan_without_skip_propagates(flags):
  """无 SKIP_NAN 时, 窗口内任一 nan 都会使结果变成 nan."""
  x = _nan_data()
  periods = 4
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))

  for i in range(len(x)):
    if np.isnan(x[max(0, i - periods + 1) : i + 1]).any():
      assert np.isnan(actual[i])


def test_llvbars_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: nan 不参与极值选取, 距离按原始位置计算."""
  x = _nan_data()
  periods = 4
  actual = _calc(x, periods, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, periods, flags=al.FLAG_SKIP_NAN))
  _assert_close(actual, _bars_skip_ref(x, periods, strict=False))
  assert np.isnan(actual[np.isnan(x)]).all()


def test_llvbars_skip_nan_on_clean_data():
  """无 nan 数据上 SKIP_NAN 与不跳过 nan 的结果一致."""
  x = _clean_data()
  periods = 5
  for flags in (0, al.FLAG_SKIP_NAN):
    _assert_close(
      _calc(x, periods, flags=flags), _bars_ref(x, periods, strict=False)
    )
  for flags in (
    al.FLAG_STRICTLY_CYCLE,
    al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
  ):
    _assert_close(
      _calc(x, periods, flags=flags), _bars_ref(x, periods, strict=True)
    )


def test_llvbars_skip_nan_and_strictly_cycle():
  """两个 flag 组合: 要求连续无 nan 的完整窗口."""
  x = _nan_data()
  periods = 4
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  _assert_close(actual, _bars_skip_ref(x, periods, strict=True))
  assert np.isnan(actual[np.isnan(x)]).all()


@pytest.mark.parametrize(
  "flags",
  [
    0,
    al.FLAG_STRICTLY_CYCLE,
    al.FLAG_SKIP_NAN,
    al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
  ],
)
def test_llvbars_all_flag_combinations_with_nan(flags):
  """四种 flag 组合在含 nan 数据上都与参考实现一致."""
  x = _nan_data()
  periods = 6
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_llvbars_groups(groups, flags):
  """group-major 分块: 每个 chunk 独立计算, 开头重新计窗口."""
  x = _clean_data()
  periods = 4
  actual = _calc(x, periods, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, periods, groups=groups, flags=flags))

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    chunk = x[lo : lo + chunk_size]
    if flags & al.FLAG_STRICTLY_CYCLE:
      assert np.isnan(actual[lo : lo + periods - 1]).all()
    else:
      _assert_close(
        actual[lo : lo + chunk_size],
        _bars_ref(chunk, periods, strict=False),
      )


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
def test_llvbars_skip_nan_within_each_group(groups):
  """SKIP_NAN 的有效值窗口按 chunk 独立进行, 不会跨 group 借用数据."""
  x = _nan_data()
  periods = 5
  flags = al.FLAG_SKIP_NAN
  actual = _calc(x, periods, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, periods, groups=groups, flags=flags))

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    got = actual[lo : lo + chunk_size]
    _assert_close(
      got, _bars_skip_ref(x[lo : lo + chunk_size], periods, strict=False)
    )


@pytest.mark.parametrize(
  "start,end",
  [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5), (-10, 0), (40, 20)],
)
def test_llvbars_start_end(start, end):
  """start/end: 仅校验 [start, end) 区间, 区间外必须全是 nan."""
  x = _clean_data()
  periods = 4
  actual = _calc(x, periods, start=start, end=end, flags=0)
  _assert_close(actual, _expected(x, periods, start=start, end=end, flags=0))

  s, e = _calc_range(len(x), start, end)
  if s >= e:
    assert np.isnan(actual).all()
  else:
    assert np.isnan(actual[:s]).all()
    assert np.isnan(actual[e:]).all()
    assert np.isfinite(actual[s:e]).all()


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_llvbars_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  x = _clean_data()
  periods = 4
  start, end = 6, 25
  actual = _calc(x, periods, start=start, end=end, groups=groups, flags=0)
  _assert_close(
    actual,
    _expected(x, periods, start=start, end=end, groups=groups, flags=0),
  )

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


def test_llvbars_skip_nan_with_start_end():
  """SKIP_NAN + start/end: 只有 [start, end) 内的有效值参与窗口计算."""
  x = _nan_data()
  periods = 4
  ctx = dict(start=10, end=90, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, periods, **ctx)
  _assert_close(actual, _expected(x, periods, **ctx))
  assert np.isnan(actual[:10]).all()
  assert np.isnan(actual[90:]).all()
  _assert_close(actual[10:90], _bars_skip_ref(x[10:90], periods, strict=False))


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_llvbars_skip_nan_start_end_with_groups(groups):
  """SKIP_NAN + 负数 end + groups: 每个 chunk 内独立解析区间并压紧."""
  x = _nan_data()
  periods = 3
  ctx = dict(start=5, end=-5, groups=groups, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, periods, **ctx)
  _assert_close(actual, _expected(x, periods, **ctx))


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_llvbars_skip_nan_strict_start_end_with_groups(groups):
  """SKIP_NAN | STRICTLY_CYCLE 与 start/end/groups 组合."""
  x = _nan_data()
  periods = 4
  ctx = dict(
    start=8,
    end=-8,
    groups=groups,
    flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
  )
  actual = _calc(x, periods, **ctx)
  _assert_close(actual, _expected(x, periods, **ctx))


@pytest.mark.parametrize(
  "flags",
  [
    0,
    al.FLAG_STRICTLY_CYCLE,
    al.FLAG_SKIP_NAN,
    al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
  ],
)
def test_llvbars_periods_zero_cumulative(flags):
  """periods=0: 距最近一次刷新累积最低值的 bar 数 (无 nan 数据)."""
  x = _clean_data()
  actual = _calc(x, 0, flags=flags)
  _assert_close(actual, _expected(x, 0, flags=flags))
  _assert_close(actual, _bars_cum_ref(x, skip=False))
  assert np.isfinite(actual).all()


def test_llvbars_periods_zero_nan_poison_without_skip():
  """periods=0 且不跳过 nan: 首个 nan 及其后全部输出 nan."""
  x = _clean_data().copy()
  x[7] = np.nan
  actual = _calc(x, 0, flags=0)
  _assert_close(actual, _expected(x, 0, flags=0))
  assert np.isfinite(actual[:7]).all()
  assert np.isnan(actual[7:]).all()
  _assert_close(actual[:7], _bars_cum_ref(x[:7], skip=False))


def test_llvbars_periods_zero_skip_nan():
  """periods=0 且跳过 nan: nan 位置也按原始索引累计 bar 距离."""
  x = _nan_data()
  actual = _calc(x, 0, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, 0, flags=al.FLAG_SKIP_NAN))
  _assert_close(actual, _bars_cum_ref(x, skip=True))
  assert np.isnan(actual[:2]).all()


@pytest.mark.parametrize("groups", [1, 2, 3, 4])
def test_llvbars_periods_zero_groups_and_range(groups):
  """periods=0: 每个 chunk 在 [start, end) 内从头开始独立累积."""
  x = _clean_data()
  ctx = dict(start=7, end=-9, groups=groups, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, 0, **ctx)
  _assert_close(actual, _expected(x, 0, **ctx))

  chunk_size = len(x) // groups
  e = chunk_size - 9
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + 7]).all()
    assert np.isnan(actual[lo + e : lo + chunk_size]).all()
    _assert_close(
      actual[lo + 7 : lo + e],
      _bars_cum_ref(x[lo + 7 : lo + e], skip=True),
    )


def test_llvbars_repeated_values():
  """并列最低值取最靠后的位置: 常数序列的 bars 恒为 0."""
  x = np.repeat(np.array([1.0, 3.0, 2.0, 3.0, 5.0]), 7)
  x = np.concatenate([x, np.arange(20.0, 30.0)])
  for periods in (1, 3, 5, 9):
    for flags in (0, al.FLAG_STRICTLY_CYCLE, al.FLAG_SKIP_NAN):
      actual = _calc(x, periods, flags=flags)
      _assert_close(actual, _expected(x, periods, flags=flags))
  assert np.all(_calc(np.full(20, 2.0), 5, flags=0) == 0)


def test_llvbars_edge_cases():
  """periods=1 恒为 0; 窗口长于数据时非严格模式仍输出部分窗口."""
  x = _nan_data()
  _assert_close(_calc(x, 1, flags=0), _expected(x, 1, flags=0))
  assert np.all(_calc(np.ones(N), 1, flags=0) == 0)

  periods = len(x) + 1
  _assert_close(_calc(x, periods, flags=0), _expected(x, periods, flags=0))
  assert np.isnan(_calc(x, periods, flags=al.FLAG_STRICTLY_CYCLE)).all()
  _assert_close(
    _calc(x, periods, flags=al.FLAG_SKIP_NAN),
    _expected(x, periods, flags=al.FLAG_SKIP_NAN),
  )
