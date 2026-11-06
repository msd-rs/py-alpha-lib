# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""MA (Simple Moving Average, 简单移动平均) 集成测试.

算法简介
--------
MA 对最近 `periods` 个值取算术平均::

    MA[i] = sum(x[i - periods + 1 : i + 1]) / periods

与 LWMA 不同, ``flags=0`` 时窗口未满也会用已有的 1..periods 个值给出
部分均值 (warm-up); 设置 ``FLAG_STRICTLY_CYCLE`` 后窗口未满输出 nan,
与 pandas ``rolling(periods, min_periods=periods).mean()`` 一致. 参考实现
``_ma_ref`` 只在无 nan 数据上用 pandas ``rolling`` 计算, 不实现 SKIP_NAN /
Group / start-end 机制.

不设置 ``FLAG_SKIP_NAN`` 时, 窗口内出现 nan 会使该窗口结果为 nan; 设置后
nan 不参与运算, 等价于把有效值压紧 (compact) 后跑参考实现, 再映射回有效
位置 (nan 位置本身仍为 nan). 再组合 ``FLAG_STRICTLY_CYCLE`` 时, 还要求最近
`periods` 个原始位置全部有效, 因此等价于 pandas 直接在含 nan 的数据上做
``rolling(periods).mean()``.

测试里 Group 通过按 chunk 分块计算实现, start/end 只校验 [start, end) 区间,
区间外必须是 nan; 窗口本身仍可引用 start 之前的数据 (SKIP_NAN 除外, 它只
在 [start, end) 内压紧有效值). ``set_ctx`` 只覆盖传入的参数, 所以每次调用
都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=120)
    >>> x = np.arange(1., 11.)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算 (窗口未满输出部分均值)::

    >>> np.round(al.MA(x, 3), 3)
    array([1. , 1.5, 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9. ])

   前三个输出依次为 1, (1+2)/2=1.5, (1+2+3)/3=2.

2. ``FLAG_STRICTLY_CYCLE`` (窗口未满输出 nan)::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.MA(x, 3), 3)
    array([nan, nan,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])

3. 不跳过 nan: 窗口内任一 nan 都会污染整个窗口::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.MA(xn, 3), 3)
    array([1. , 1.5, 2. , nan, nan, nan, 6. , 7. , 8. , 9. ])

4. ``FLAG_SKIP_NAN``: nan 完全不参与运算. 压紧后的有效值为
   [1, 2, 3, 5, 6, 7, 8, 9, 10], 其 3 周期部分均值依次是
   [1, 1.5, 2, 3.333, 4.667, 6, 7, 8, 9]::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.MA(xn, 3), 3)
    array([1.   , 1.5  , 2.   ,   nan, 3.333, 4.667, 6.   , 7.   , 8.   , 9.   ])

5. ``FLAG_SKIP_NAN | FLAG_STRICTLY_CYCLE``: 要求连续 `periods` 个原始位置
   全部有效. i=2 的窗口 [1, 2, 3] 完整, i=3..5 的窗口含 nan, i=6 的窗口
   [5, 6, 7] 才重新完整::

    >>> al.set_ctx(start=0, end=0, groups=1,
    ...            flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.MA(xn, 3), 3)
    array([nan, nan,  2., nan, nan, nan,  6.,  7.,  8.,  9.])

6. ``groups`` 把一维数据按 group-major 切块独立计算, 每块重新 warm-up::

    >>> y = np.arange(1., 13.)
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.MA(y, 3), 3)
    array([ 1. ,  1.5,  2. ,  3. ,  4. ,  5. ,  7. ,  7.5,  8. ,  9. , 10. , 11. ])

7. ``start`` / ``end`` 只计算 [start, end) 区间, 区间外全部为 nan,
   但窗口仍可引用 start 之前的数据::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.MA(x, 3), 3)
    array([nan, nan, nan,  3.,  4.,  5.,  6.,  7., nan, nan])

8. ``periods=1`` 直接复制输入, 忽略 ctx (含 nan 位置也原样保留)::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.MA(xn, 1), 3)
    array([ 1.,  2.,  3., nan,  5.,  6.,  7.,  8.,  9., 10.])
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


def _ma_ref(x, periods, *, strict):
  """无 nan 数据上的 MA 参考实现 (pandas rolling mean)."""
  x = np.asarray(x, dtype=np.float64)
  assert not np.isnan(x).any()
  min_periods = periods if strict else 1
  return (
    pd.Series(x)
    .rolling(periods, min_periods=min_periods)
    .mean()
    .to_numpy(dtype=np.float64)
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


def _no_nan_window_mask(x, periods):
  """每个位置的滚动窗口 (最多 periods 个原始值) 是否完全不含 nan."""
  nan_count = (
    pd.Series(np.isnan(x).astype(np.float64))
    .rolling(periods, min_periods=1)
    .sum()
    .to_numpy()
  )
  return nan_count == 0


def _full_valid_window_mask(x, periods):
  """最近 periods 个原始位置是否全部有效 (SKIP_NAN + STRICTLY_CYCLE)."""
  valid_count = (
    pd.Series((~np.isnan(x)).astype(np.float64))
    .rolling(periods, min_periods=periods)
    .sum()
    .to_numpy()
  )
  return valid_count == periods


def _compact_ref(x, periods, strict):
  """把有效值压紧后跑参考实现, 再映射回有效位置; nan 位置保持 nan."""
  out = np.full(len(x), np.nan)
  valid = ~np.isnan(x)
  if valid.any():
    out[valid] = _ma_ref(x[valid], periods, strict=strict)
  return out


def _expected(x, periods, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 全量计算参考值, 再只保留 alpha 应计算的 [start, end) 区间.

  无 SKIP_NAN 时, 窗口可以引用 start 之前的数据, 但窗口内有 nan 就让结果为
  nan; 有 SKIP_NAN 时, 只在 [start, end) 内压紧有效值, 不跨边界借用数据.
  """
  x = np.asarray(x, dtype=np.float64)
  if periods == 1:
    # Rust 对 periods=1 直接 copy_from_slice, 完全忽略 ctx
    return x.copy()

  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  skip = bool(flags & al.FLAG_SKIP_NAN)
  parts = []
  for chunk in np.split(x, groups):
    n = len(chunk)
    s, e = _calc_range(n, start, end)
    ref = np.full(n, np.nan)
    if s < e:
      if skip:
        seg = chunk[s:e]
        seg_ref = _compact_ref(seg, periods, strict)
        if strict:
          seg_ref[~_full_valid_window_mask(seg, periods)] = np.nan
        ref[s:e] = seg_ref
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
    al.MA(np.asarray(x, dtype=np.float64), periods), dtype=np.float64
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
def test_ma_matches_pandas_rolling(periods):
  """flags=0, groups=1: 与 pandas rolling(min_periods=1) 逐点比较."""
  x = _clean_data()
  actual = _calc(x, periods, flags=0)
  _assert_close(actual, _expected(x, periods, flags=0))
  _assert_close(actual, _ma_ref(x, periods, strict=False))
  assert np.isfinite(actual).all()


@pytest.mark.parametrize("periods", [2, 3, 4, 5, 10, 20])
def test_ma_strictly_cycle_matches_pandas(periods):
  """FLAG_STRICTLY_CYCLE: 窗口未满输出 nan, 之后与 pandas rolling 一致."""
  x = _clean_data()
  flags = al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  _assert_close(actual, _ma_ref(x, periods, strict=True))
  assert np.isnan(actual[: periods - 1]).all()
  assert np.isfinite(actual[periods - 1 :]).all()


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_ma_nan_without_skip_propagates(flags):
  """无 SKIP_NAN 时, 窗口内任一 nan 都会使结果变成 nan."""
  x = _nan_data()
  periods = 4
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))

  for i in range(len(x)):
    if np.isnan(x[max(0, i - periods + 1) : i + 1]).any():
      assert np.isnan(actual[i])
  if flags & al.FLAG_STRICTLY_CYCLE:
    assert np.isnan(actual[: periods - 1]).all()


def test_ma_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 有效位置等于「未插入 nan」数据压紧后的参考结果."""
  x_clean = _clean_data()
  x = x_clean.copy()
  x[NAN_POS] = np.nan
  periods = 4
  actual = _calc(x, periods, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, periods, flags=al.FLAG_SKIP_NAN))

  valid = ~np.isnan(x)
  assert np.isnan(actual[~valid]).all()
  _assert_close(actual[valid], _ma_ref(x_clean[valid], periods, strict=False))


def test_ma_skip_nan_and_strictly_cycle():
  """两个 flag 组合: 要求连续无 nan 的完整窗口, 等价于 pandas 原始 rolling."""
  x = _nan_data()
  periods = 4
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  _assert_close(
    actual,
    pd.Series(x).rolling(periods, min_periods=periods).mean().to_numpy(),
  )
  assert np.isnan(actual[: periods - 1]).all()
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
def test_ma_all_flag_combinations_with_nan(flags):
  """四种 flag 组合在含 nan 数据上都与参考实现一致."""
  x = _nan_data()
  periods = 6
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_ma_groups(groups, flags):
  """group-major 分块: 每个 chunk 独立计算, 开头重新 warm-up."""
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
        actual[lo : lo + periods - 1],
        _ma_ref(chunk, periods, strict=False)[: periods - 1],
      )


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
def test_ma_skip_nan_within_each_group(groups):
  """SKIP_NAN 的压紧过程按 chunk 独立进行, 不会跨 group 借用数据."""
  x = _nan_data()
  periods = 5
  flags = al.FLAG_SKIP_NAN
  actual = _calc(x, periods, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, periods, groups=groups, flags=flags))

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    chunk = x[lo : lo + chunk_size]
    valid = ~np.isnan(chunk)
    got = actual[lo : lo + chunk_size]
    assert np.isnan(got[~valid]).all()
    _assert_close(got[valid], _ma_ref(chunk[valid], periods, strict=False))


@pytest.mark.parametrize(
  "start,end",
  [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5), (-10, 0), (40, 20)],
)
def test_ma_start_end(start, end):
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
def test_ma_start_end_with_groups(groups):
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


def test_ma_skip_nan_with_start_end():
  """SKIP_NAN + start/end: 只有 [start, end) 内的有效值参与压紧计算."""
  x_clean = _clean_data()
  x = x_clean.copy()
  x[NAN_POS] = np.nan
  periods = 4
  ctx = dict(start=10, end=90, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, periods, **ctx)
  _assert_close(actual, _expected(x, periods, **ctx))
  assert np.isnan(actual[:10]).all()
  assert np.isnan(actual[90:]).all()
  assert np.isnan(actual[np.isnan(x)]).all()

  seg = x[10:90]
  valid = ~np.isnan(seg)
  _assert_close(
    actual[10:90][valid],
    _ma_ref(x_clean[10:90][valid], periods, strict=False),
  )


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_ma_skip_nan_start_end_with_groups(groups):
  """SKIP_NAN + 负数 end + groups: 每个 chunk 内独立解析区间并压紧."""
  x = _nan_data()
  periods = 3
  ctx = dict(start=5, end=-5, groups=groups, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, periods, **ctx)
  _assert_close(actual, _expected(x, periods, **ctx))


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_ma_skip_nan_strict_start_end_with_groups(groups):
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


def test_ma_edge_cases():
  """periods=1 直接复制输入; 窗口长于数据时非严格模式为扩展均值."""
  x = _nan_data()
  for ctx in (
    {},
    dict(start=10, end=50, groups=3, flags=al.FLAG_SKIP_NAN),
  ):
    _assert_close(_calc(x, 1, **ctx), x)

  periods = len(x) + 1
  _assert_close(_calc(x, periods, flags=0), _expected(x, periods, flags=0))
  assert np.isnan(_calc(x, periods, flags=al.FLAG_STRICTLY_CYCLE)).all()
  assert np.isnan(
    _calc(
      x,
      periods,
      flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
    )
  ).all()

  actual = _calc(x, periods, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, periods, flags=al.FLAG_SKIP_NAN))
  valid = ~np.isnan(x)
  _assert_close(actual[valid], _ma_ref(x[valid], periods, strict=False))
