# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""WEIGHTED_DELAY (Weighted Delay, 加权延迟) 集成测试.

算法简介
--------
WEIGHTED_DELAY 对最近 `periods` 个**滞后**值做线性加权平均, 越近的滞后值
权重越大::

    WEIGHTED_DELAY(x, k)[i] =
        (k*x[i-1] + (k-1)*x[i-2] + ... + 1*x[i-k]) / (k*(k+1)/2)

等价于对滞后一期的序列做 LWMA: ``LWMA(REF(x, 1), k)``. 分母恒为
``k*(k+1)/2``, 且必须凑齐 k 个滞后值才输出, 因此前 k 个位置恒为 nan —
从这个意义上算法本身已满足 ``FLAG_STRICTLY_CYCLE``, 该 flag 不改变结果.
当前值 ``x[i]`` 不参与运算, 只有 ``x[i-k..i-1]`` 参与.

参考实现 ``_weighted_delay_ref`` 按定义用 numpy 逐点加权求和, 不实现
SKIP_NAN 的压紧 / Group / start-end 机制: Group 通过按 chunk 分块计算,
start/end 只校验 [start, end) 区间 (滞后窗口仍可引用 start 之前的数据).
不设置 ``FLAG_SKIP_NAN`` 时, 滞后窗口内任一 nan 都会使结果为 nan; 设置后
语义是 nan 完全不参与运算 (有效值压紧后计算), Rust 实现已支持该 flag,
与参考实现一致.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=120)
    >>> x = np.arange(1., 11.)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 3 期加权延迟: i=3 用 (3*3 + 2*2 + 1*1)/6 = 14/6 = 2.333::

    >>> np.round(al.WEIGHTED_DELAY(x, 3), 3)
    array([  nan,   nan,   nan, 2.333, 3.333, 4.333, 5.333, 6.333, 7.333, 8.333])

2. ``FLAG_STRICTLY_CYCLE`` 不改变结果 (算法本身要求完整滞后窗口)::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.WEIGHTED_DELAY(x, 3), 3)
    array([  nan,   nan,   nan, 2.333, 3.333, 4.333, 5.333, 6.333, 7.333, 8.333])

3. 滞后窗口内出现 nan 会污染结果; 当前值本身为 nan 不影响结果
   (i=3 只用 x[0..2], 所以仍是 2.333)::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.WEIGHTED_DELAY(xn, 3), 3)
    array([  nan,   nan,   nan, 2.333,   nan,   nan,   nan, 6.333, 7.333, 8.333])

4. ``FLAG_SKIP_NAN`` 的理想语义: 压紧有效值
   [1, 2, 3, 5, 6, 7, 8, 9, 10] 后按上式计算, nan 位置仍为 nan.

5. ``groups`` 按 group-major 切块独立计算, 每块前 k 个位置为 nan::

    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.WEIGHTED_DELAY(x, 3), 3)
    array([  nan,   nan,   nan, 2.333, 3.333,   nan,   nan,   nan, 7.333, 8.333])

6. ``start`` / ``end`` 只计算 [start, end) 区间, 区间外全部为 nan,
   但滞后窗口仍可引用 start 之前的数据::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.WEIGHTED_DELAY(x, 3), 3)
    array([  nan,   nan,   nan, 2.333, 3.333, 4.333, 5.333, 6.333,   nan,   nan])

7. ``periods=1`` 时权重为 1, 结果就是滞后一期的值 x[i-1]::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.WEIGHTED_DELAY(x, 1), 3)
    array([nan,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
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


def _weights(periods):
  """滞后 1..periods 的权重 (periods, periods-1, ..., 1)."""
  return np.arange(periods, 0, -1, dtype=np.float64)


def _weighted_delay_ref(x, periods, *, strict=False):
  """按算法定义逐点对最近 periods 个滞后值做线性加权平均.

  ``x`` 可以含 nan: 滞后窗口内任一 nan 会让结果为 nan; 当前值 x[i] 不参与
  运算, 因此 x[i] 为 nan 不影响输出. 前 periods 个位置恒为 nan.
  """
  x = np.asarray(x, dtype=np.float64)
  out = np.full(len(x), np.nan)
  if periods == 0 or len(x) <= periods:
    return out
  w = _weights(periods)
  denom = w.sum()
  for i in range(periods, len(x)):
    win = x[i - periods : i]
    if np.isnan(win).any():
      continue
    out[i] = np.dot(win[::-1], w) / denom
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


def _full_lag_window_mask(x, periods):
  """最近 periods 个滞后位置 [i-periods, i) 是否全部有效."""
  valid = ~np.isnan(x)
  mask = np.zeros(len(x), dtype=bool)
  for i in range(periods, len(x)):
    mask[i] = bool(valid[i - periods : i].all())
  return mask


def _compact_ref(x, periods, strict):
  """把有效值压紧后跑参考实现, 再映射回有效位置; nan 位置保持 nan."""
  out = np.full(len(x), np.nan)
  valid = ~np.isnan(x)
  if valid.any():
    out[valid] = _weighted_delay_ref(
      x[valid], periods, strict=strict
    )
  return out


def _expected(x, periods, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 全量计算参考值, 再只保留 alpha 应计算的 [start, end) 区间.

  无 SKIP_NAN 时滞后窗口引用原始位置 (可跨 start 借用数据), 窗口内 nan 会
  传播; 有 SKIP_NAN 时按理想语义只在 [start, end) 内压紧有效值后计算.
  """
  x = np.asarray(x, dtype=np.float64)
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
          seg_ref[~_full_lag_window_mask(seg, periods)] = np.nan
        ref[s:e] = seg_ref
      else:
        ref = _weighted_delay_ref(chunk, periods, strict=strict)
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
    al.WEIGHTED_DELAY(np.asarray(x, dtype=np.float64), periods),
    dtype=np.float64,
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


@pytest.mark.parametrize("periods", [1, 2, 3, 4, 5, 10, 20])
def test_weighted_delay_matches_numpy(periods):
  """flags=0, groups=1: 每个位置都与逐点加权求和一致."""
  x = _clean_data()
  actual = _calc(x, periods, flags=0)
  _assert_close(actual, _expected(x, periods, flags=0))
  _assert_close(actual, _weighted_delay_ref(x, periods))
  assert np.isnan(actual[:periods]).all()
  assert np.isfinite(actual[periods:]).all()


@pytest.mark.parametrize("periods", [2, 3, 5, 10])
def test_weighted_delay_matches_pandas_rolling_apply(periods):
  """独立校验: 对 shift(1) 序列做 rolling 加权平均."""
  x = _clean_data()
  w = _weights(periods)
  expected = (
    pd.Series(x)
    .shift(1)
    .rolling(periods)
    .apply(lambda win: np.dot(win, w[::-1]) / w.sum(), raw=True)
    .to_numpy(dtype=np.float64)
  )
  _assert_close(_calc(x, periods, flags=0), expected)


def test_weighted_delay_equals_lwma_of_lagged_series():
  """与库内 LWMA(REF(x, 1), k) 交叉校验."""
  x = _clean_data()
  periods = 6
  al.set_ctx(start=0, end=0, groups=1, flags=0)
  lagged = np.asarray(al.REF(x, 1), dtype=np.float64)
  lwma = np.asarray(al.LWMA(lagged, periods), dtype=np.float64)
  _assert_close(_calc(x, periods, flags=0)[periods:], lwma[periods:])


@pytest.mark.parametrize("periods", [1, 2, 3, 4, 5, 10, 20])
def test_weighted_delay_strictly_cycle_is_same(periods):
  """算法本身要求完整滞后窗口, FLAG_STRICTLY_CYCLE 不改变结果."""
  x = _clean_data()
  flags = al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  _assert_close(actual, _calc(x, periods, flags=0))


def test_weighted_delay_constant_and_linear_input():
  """常数列结果仍为该常数; 线性序列的加权延迟仍为等间距线性序列."""
  x = np.full(20, 4.25)
  _assert_close(_calc(x, 3, flags=0), _weighted_delay_ref(x, 3))
  _assert_close(_calc(x, 3, flags=0)[3:], np.full(17, 4.25))

  linear = np.arange(1.0, 21.0)
  k = 3
  actual = _calc(linear, k, flags=0)
  _assert_close(actual, _weighted_delay_ref(linear, k))
  assert np.allclose(np.diff(actual[k:]), 1.0)


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_weighted_delay_nan_without_skip_propagates(flags):
  """无 SKIP_NAN: 滞后窗口内任一 nan 都会污染结果, 当前值 nan 不影响结果."""
  x = _nan_data()
  periods = 4
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))

  for i in range(periods, len(x)):
    if np.isnan(x[i - periods : i]).any():
      assert np.isnan(actual[i])
  assert np.isnan(actual[:periods]).all()

  current_nan = [
    i
    for i in range(periods, len(x))
    if np.isnan(x[i]) and not np.isnan(x[i - periods : i]).any()
  ]
  assert current_nan
  assert np.isfinite(actual[current_nan]).all()


def test_weighted_delay_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 有效位置应等于「未插入 nan」数据压紧后的参考结果."""
  x_clean = _clean_data()
  x = x_clean.copy()
  x[NAN_POS] = np.nan
  periods = 4
  actual = _calc(x, periods, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, periods, flags=al.FLAG_SKIP_NAN))

  valid = ~np.isnan(x)
  assert np.isnan(actual[~valid]).all()
  _assert_close(
    actual[valid], _weighted_delay_ref(x_clean[valid], periods)
  )


def test_weighted_delay_skip_nan_and_strictly_cycle():
  """两个 flag 组合: 要求连续无 nan 的完整滞后窗口后才输出."""
  x = _nan_data()
  periods = 4
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  assert np.isnan(actual[:periods]).all()
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
def test_weighted_delay_all_flag_combinations_with_nan(flags):
  """四种 flag 组合在含 nan 数据上都与参考实现一致."""
  x = _nan_data()
  periods = 6
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_weighted_delay_groups(groups, flags):
  """group-major 分块: 每个 chunk 独立计算, 开头 k 个位置为 nan."""
  x = _clean_data()
  periods = 4
  actual = _calc(x, periods, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, periods, groups=groups, flags=flags))

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    chunk = x[lo : lo + chunk_size]
    _assert_close(
      actual[lo : lo + chunk_size],
      _weighted_delay_ref(chunk, periods, strict=bool(flags & al.FLAG_STRICTLY_CYCLE)),
    )


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
def test_weighted_delay_skip_nan_within_each_group(groups):
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
    _assert_close(got[valid], _weighted_delay_ref(chunk[valid], periods))


@pytest.mark.parametrize(
  "start,end",
  [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5), (-10, 0), (40, 20)],
)
def test_weighted_delay_start_end(start, end):
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
    lo = max(s, periods)
    if lo < e:
      assert np.isfinite(actual[lo:e]).all()


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_weighted_delay_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  x = _clean_data()
  periods = 4
  start, end = 6, 25
  actual = _calc(x, periods, start=start, end=end, groups=groups, flags=0)
  _assert_close(
    actual, _expected(x, periods, start=start, end=end, groups=groups, flags=0)
  )

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


def test_weighted_delay_skip_nan_with_start_end():
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
    _weighted_delay_ref(x_clean[10:90][valid], periods),
  )


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_weighted_delay_skip_nan_start_end_with_groups(groups):
  """SKIP_NAN + 负数 end + groups: 每个 chunk 内独立解析区间并压紧."""
  x = _nan_data()
  periods = 3
  ctx = dict(start=5, end=-5, groups=groups, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, periods, **ctx)
  _assert_close(actual, _expected(x, periods, **ctx))


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_weighted_delay_skip_nan_strict_start_end_with_groups(groups):
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


def test_weighted_delay_edge_cases():
  """periods=0 全 nan; periods=1 为滞后一期; 窗口长于数据全 nan."""
  x = _clean_data()
  assert np.isnan(_calc(x, 0, flags=0)).all()

  lagged = np.concatenate(([np.nan], x[:-1]))
  _assert_close(_calc(x, 1, flags=0), lagged)

  periods = len(x) + 1
  assert np.isnan(_calc(x, periods, flags=0)).all()
  assert np.isnan(_calc(x, periods, flags=al.FLAG_STRICTLY_CYCLE)).all()
