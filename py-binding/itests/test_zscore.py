# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""ZSCORE (滚动 Z 分数) 集成测试.

算法简介
--------
ZSCORE 用最近 `periods` 个值的样本均值与样本标准差 (ddof=1) 标准化当前值::

    ZSCORE[i] = (x[i] - mean(window)) / std(window, ddof=1)

分母为 0 (常数窗口) 时结果为 0. 样本标准差只有在窗口收集满 `periods` 个值
后才有定义, 因此 ``flags=0`` 与 ``FLAG_STRICTLY_CYCLE`` 都要求完整窗口,
前 `periods-1` 个位置为 nan; `periods=1` 时全部为 nan. 参考实现
``_zscore_ref`` 只在无 nan 数据上用 pandas ``rolling`` 的 mean/std 计算, 不
实现 SKIP_NAN / Group / start-end 机制.

``FLAG_SKIP_NAN`` 让 nan 不参与运算, 窗口按有效值计算: 非严格模式下窗口内
至少有 2 个有效值就给出 Z 分数, 等价于把有效值压紧 (compact) 后跑参考实现
再映射回有效位置 (nan 位置本身仍为 nan); 组合 ``FLAG_STRICTLY_CYCLE`` 时还
要求最近 `periods` 个原始位置全部有效. 不设置 ``FLAG_SKIP_NAN`` 时, 窗口内
出现 nan 会使该窗口结果为 nan.

测试里 Group 通过按 chunk 分块计算实现, start/end 只校验 [start, end) 区间,
区间外必须是 nan; 窗口本身仍可引用 start 之前的数据 (SKIP_NAN 除外, 它只在
[start, end) 内压紧有效值). ``set_ctx`` 只覆盖传入的参数, 所以每次调用都
显式重置 start/end/groups/flags. 浮点比较容差 1e-6.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=150)
    >>> x = np.array([1., 3., 2., 5., 4., 6., 8., 7., 9., 10.])
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算 (窗口未满输出 nan)::

    >>> np.round(al.ZSCORE(x, 3), 3)
    array([  nan,   nan, 0.   , 1.091, 0.218, 1.   , 1.   , 0.   , 1.   , 0.873])

   i=2 的窗口 [1, 3, 2] 均值与标准差都是 2 和 1, 当前值 2 的 Z 分数为 0.

2. ``FLAG_STRICTLY_CYCLE`` 与默认一致: Z 分数总是需要完整窗口::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.ZSCORE(x, 3), 3)
    array([  nan,   nan, 0.   , 1.091, 0.218, 1.   , 1.   , 0.   , 1.   , 0.873])

3. 不跳过 nan: 窗口内任一 nan 都会污染整个窗口::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.ZSCORE(xn, 3), 3)
    array([  nan,   nan, 0.   ,   nan,   nan,   nan, 1.   , 0.   , 1.   , 0.873])

4. ``FLAG_SKIP_NAN``: 压紧有效值后计算, 允许部分窗口 (i=1 的窗口 [1, 3]
   给出 (3-2)/sqrt(2)=0.707)::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.ZSCORE(xn, 3), 3)
    array([  nan, 0.707, 0.   ,   nan, 1.   , 1.   , 1.   , 0.   , 1.   , 0.873])

5. ``FLAG_SKIP_NAN | FLAG_STRICTLY_CYCLE``: 要求连续 `periods` 个原始位置
   全部有效, 结果与不跳过 nan 时一致::

    >>> al.set_ctx(start=0, end=0, groups=1,
    ...            flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.ZSCORE(xn, 3), 3)
    array([  nan,   nan, 0.   ,   nan,   nan,   nan, 1.   , 0.   , 1.   , 0.873])

6. ``groups`` 把一维数据按 group-major 切块独立计算, 每块重新计窗口::

    >>> y = np.array([3., 1., 4., 1., 5., 9., 2., 6., 5., 3., 5., 8.])
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.ZSCORE(y, 3), 3)
    array([   nan,    nan,  0.873, -0.577,  0.801,  1.   ,    nan,    nan,  0.32 , -1.091,  0.577,  1.06 ])

7. ``start`` / ``end`` 只计算 [start, end) 区间, 区间外全部为 nan,
   但窗口仍可引用 start 之前的数据::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.ZSCORE(x, 3), 3)
    array([  nan,   nan,   nan, 1.091, 0.218, 1.   , 1.   , 0.   ,   nan,   nan])

8. ``periods=1`` 时 Z 分数无定义, 全部为 nan::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.ZSCORE(x, 1), 3)
    array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])
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


def _zscore_ref(x, periods, min_periods):
  """无 nan 数据上的滚动 Z 分数参考实现 (pandas rolling mean/std).

  std 为 0 的常数窗口按算法约定输出 0, 而不是除零得到的 inf.
  """
  x = np.asarray(x, dtype=np.float64)
  assert not np.isnan(x).any()
  assert periods > 0
  if min_periods > periods:
    return np.full(len(x), np.nan)
  s = pd.Series(x)
  mean = (
    s.rolling(periods, min_periods=min_periods)
    .mean()
    .to_numpy(dtype=np.float64)
  )
  std = (
    s.rolling(periods, min_periods=min_periods)
    .std(ddof=1)
    .to_numpy(dtype=np.float64)
  )
  out = np.full(len(x), np.nan)
  ok = np.isfinite(mean) & np.isfinite(std) & (std != 0)
  out[ok] = (x[ok] - mean[ok]) / std[ok]
  out[np.isfinite(mean) & (std == 0)] = 0.0
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


def _full_valid_window_mask(x, periods):
  """最近 periods 个原始位置是否全部有效 (SKIP_NAN + STRICTLY_CYCLE)."""
  valid_count = (
    pd.Series((~np.isnan(x)).astype(np.float64))
    .rolling(periods, min_periods=periods)
    .sum()
    .to_numpy()
  )
  return valid_count == periods


def _compact_ref(x, periods, min_periods):
  """把有效值压紧后跑参考实现, 再映射回有效位置; nan 位置保持 nan."""
  out = np.full(len(x), np.nan)
  valid = ~np.isnan(x)
  if valid.any():
    out[valid] = _zscore_ref(x[valid], periods, min_periods)
  return out


def _expected(x, periods, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 全量计算参考值, 再只保留 alpha 应计算的 [start, end) 区间.

  无 SKIP_NAN 时窗口需要完整 `periods` 个值, 且窗口内有 nan 就让结果为 nan;
  有 SKIP_NAN 时只在 [start, end) 内压紧有效值, 非严格模式允许 2 个有效值
  起的部分窗口, 严格模式另外要求连续 `periods` 个原始位置全部有效.
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
        seg_ref = _compact_ref(seg, periods, periods if strict else 2)
        if strict:
          seg_ref[~_full_valid_window_mask(seg, periods)] = np.nan
        ref[s:e] = seg_ref
      else:
        ref = _compact_ref(chunk, periods, periods)
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
    al.ZSCORE(np.asarray(x, dtype=np.float64), periods), dtype=np.float64
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
def test_zscore_matches_pandas_rolling(periods):
  """flags=0, groups=1: 与 pandas rolling mean/std 组合逐点比较."""
  x = _clean_data()
  actual = _calc(x, periods, flags=0)
  _assert_close(actual, _expected(x, periods, flags=0))
  _assert_close(actual, _zscore_ref(x, periods, periods))
  assert np.isnan(actual[: periods - 1]).all()
  assert np.isfinite(actual[periods - 1 :]).all()


@pytest.mark.parametrize("periods", [2, 3, 4, 5, 10, 20])
def test_zscore_strictly_cycle_matches_pandas(periods):
  """FLAG_STRICTLY_CYCLE 与默认一致: Z 分数总是要求完整窗口."""
  x = _clean_data()
  flags = al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  _assert_close(actual, _zscore_ref(x, periods, periods))


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_zscore_nan_without_skip_propagates(flags):
  """无 SKIP_NAN 时, 窗口内任一 nan 都会使结果变成 nan."""
  x = _nan_data()
  periods = 4
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))

  for i in range(len(x)):
    if np.isnan(x[max(0, i - periods + 1) : i + 1]).any():
      assert np.isnan(actual[i])


def test_zscore_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 有效位置等于「未插入 nan」数据压紧后的参考结果."""
  x_clean = _clean_data()
  x = x_clean.copy()
  x[NAN_POS] = np.nan
  periods = 4
  actual = _calc(x, periods, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, periods, flags=al.FLAG_SKIP_NAN))

  valid = ~np.isnan(x)
  assert np.isnan(actual[~valid]).all()
  _assert_close(actual[valid], _zscore_ref(x_clean[valid], periods, 2))


def test_zscore_skip_nan_partial_window():
  """SKIP_NAN 非严格模式: 2 个有效值起就输出部分窗口的 Z 分数."""
  x = _clean_data()
  periods = 5
  actual = _calc(x, periods, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, periods, flags=al.FLAG_SKIP_NAN))
  _assert_close(actual, _zscore_ref(x, periods, 2))
  assert np.isnan(actual[0])
  assert np.isfinite(actual[1:]).all()


def test_zscore_skip_nan_and_strictly_cycle():
  """两个 flag 组合: 等价于 pandas 直接在含 nan 数据上做 rolling Z 分数."""
  x = _nan_data()
  periods = 4
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))

  s = pd.Series(x)
  mean = s.rolling(periods, min_periods=periods).mean()
  std = s.rolling(periods, min_periods=periods).std(ddof=1)
  ref = ((s - mean) / std).to_numpy(dtype=np.float64).copy()
  ref[std.to_numpy(dtype=np.float64) == 0] = 0.0
  _assert_close(actual, ref)
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
def test_zscore_all_flag_combinations_with_nan(flags):
  """四种 flag 组合在含 nan 数据上都与参考实现一致."""
  x = _nan_data()
  periods = 6
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_zscore_groups(groups, flags):
  """group-major 分块: 每个 chunk 独立计算, 开头重新计窗口."""
  x = _clean_data()
  periods = 4
  actual = _calc(x, periods, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, periods, groups=groups, flags=flags))

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + periods - 1]).all()
    _assert_close(
      actual[lo : lo + chunk_size],
      _zscore_ref(x[lo : lo + chunk_size], periods, periods),
    )


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
def test_zscore_skip_nan_within_each_group(groups):
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
    _assert_close(got[valid], _zscore_ref(chunk[valid], periods, 2))


@pytest.mark.parametrize(
  "start,end",
  [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5), (-10, 0), (40, 20)],
)
def test_zscore_start_end(start, end):
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
    assert np.isfinite(actual[max(s, periods - 1) : e]).all()


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_zscore_start_end_with_groups(groups):
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


def test_zscore_skip_nan_with_start_end():
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
    actual[10:90][valid], _zscore_ref(x_clean[10:90][valid], periods, 2)
  )


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_zscore_skip_nan_start_end_with_groups(groups):
  """SKIP_NAN + 负数 end + groups: 每个 chunk 内独立解析区间并压紧."""
  x = _nan_data()
  periods = 3
  ctx = dict(start=5, end=-5, groups=groups, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, periods, **ctx)
  _assert_close(actual, _expected(x, periods, **ctx))


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_zscore_skip_nan_strict_start_end_with_groups(groups):
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
def test_zscore_periods_one_is_all_nan(flags):
  """periods=1: 单个样本无法标准化, 全部输出 nan."""
  x = _clean_data()
  _assert_close(_calc(x, 1, flags=flags), np.full(len(x), np.nan))
  _assert_close(_calc(x, 1, flags=flags), _expected(x, 1, flags=flags))


def test_zscore_constant_series_is_zero():
  """常数窗口的标准差为 0, 按算法约定 Z 分数输出 0 而不是 inf."""
  x = np.full(N, 3.5)
  for periods in (2, 5, 10):
    for flags in (0, al.FLAG_SKIP_NAN, al.FLAG_STRICTLY_CYCLE):
      actual = _calc(x, periods, flags=flags)
      _assert_close(actual, _expected(x, periods, flags=flags))
    assert np.all(actual[periods - 1 :] == 0)


def test_zscore_edge_cases():
  """窗口长于数据时: 严格/默认全 nan, SKIP_NAN 非严格仍输出部分窗口."""
  x = _nan_data()
  periods = len(x) + 1
  assert np.isnan(_calc(x, periods, flags=0)).all()
  assert np.isnan(_calc(x, periods, flags=al.FLAG_STRICTLY_CYCLE)).all()
  assert np.isnan(
    _calc(x, periods, flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
  ).all()

  actual = _calc(x, periods, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, periods, flags=al.FLAG_SKIP_NAN))
  valid = ~np.isnan(x)
  _assert_close(actual[valid], _zscore_ref(x[valid], periods, 2))
