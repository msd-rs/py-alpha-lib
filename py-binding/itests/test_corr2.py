# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""CORR2 (两序列滚动相关系数) 集成测试.

算法简介
--------
最近 ``periods`` 个位置上 x 与 y 的 Pearson 相关系数::

    CORR2[i] = (sum(x*y) - sum(x)*sum(y)/n)
               / sqrt((sum(x^2) - sum(x)^2/n) * (sum(y^2) - sum(y)^2/n))

分子分母都没有除以 ``n-1`` (ddof 因子在比值中约掉). 默认模式要求窗口填满
``periods`` 个值 (至少 2 个), 窗口未满、窗口内任一序列出现 nan、或任一序列
为常数时输出 nan. ``FLAG_STRICTLY_CYCLE`` 单独使用不改变结果 (默认已是完整
窗口); 与 ``FLAG_SKIP_NAN`` 组合时要求连续 ``periods`` 个原始位置全部有效.

注意 ``CORR`` 是「序列与时间下标」的相关, ``CORR2`` 才是两序列之间的相关.

对比基准使用 pandas ``rolling`` 的分组和 (sum/count) 按上述公式计算, 只处理
无 nan 的数据, 不实现 SKIP_NAN / Group / start-end 机制: group 按 chunk 分块
分别计算, start/end 只校验 [start, end) 区间. ``FLAG_SKIP_NAN`` 等价于把
成对有效 (x 与 y 同时非 nan) 的值压紧后跑参考实现再映射回原始位置.

`set_ctx` 只覆盖传入参数, 因此每个用例都显式重置 start/end/groups/flags.

场景与结果示例
--------------
1. 常规计算 (近似线性关系 y ~ 2x)::

    >>> import numpy as np
    >>> import alpha as al
    >>> x = np.arange(1., 11.)
    >>> y = np.array([2.1, 3.8, 6.2, 7.9, 10.5, 11.8, 14.2, 15.9, 18.1, 19.8])
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.CORR2(x, y, 4), 4)
    array([nan, nan, nan, 0.9975, 0.9971, 0.9926, 0.9928, 0.9942, 0.9981,
           0.9986])

2. ``FLAG_SKIP_NAN`` 允许部分窗口 (2 个成对有效值即可), 且让 nan 不参与
   运算; ``FLAG_STRICTLY_CYCLE`` 单独使用结果与默认一致::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.CORR2(x, y, 4), 4)
    array([nan, 1.    , 0.9952, 0.9975, 0.9971, 0.9926, 0.9928, 0.9942,
           0.9981, 0.9986])

    >>> xn = x.copy(); yn = y.copy(); xn[3] = np.nan; yn[6] = np.nan
    >>> np.round(al.CORR2(xn, yn, 4), 4)
    array([nan, 1.    , 0.9952, nan, 0.9988, 0.9959, nan, 0.9978, 0.9968,
           0.9992])

   不设置 ``FLAG_SKIP_NAN`` 时, 窗口内任一 nan 会使结果变成 nan.

3. ``FLAG_SKIP_NAN | FLAG_STRICTLY_CYCLE`` 要求连续 4 个原始位置全部有效::

    >>> al.set_ctx(start=0, end=0, groups=1,
    ...            flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.CORR2(xn, yn, 4), 4)
    array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])

4. ``groups`` 按 group-major 切块, 每块重新 warm-up (完全线性序列相关为 1)::

    >>> z = np.arange(1., 13.)
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.CORR2(z, 2 * z + 1, 4), 4)
    array([nan, nan, nan, 1., 1., 1., nan, nan, nan, 1., 1., 1.])

5. ``start`` / ``end`` 只计算 [start, end) 区间::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.CORR2(x, y, 4), 4)
    array([nan, nan, nan, 0.9975, 0.9971, 0.9926, 0.9928, 0.9942, nan, nan])
"""

import numpy as np
import pandas as pd
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
EPS = np.finfo(np.float64).eps
N = 120
NAN_POS_X = np.array([3, 17, 18, 29, 44, 52, 77, 101])
NAN_POS_Y = np.array([5, 17, 30, 44, 60, 77, 95])


# ---------------------------------------------------------------------------
# numpy / pandas 对比实现
# ---------------------------------------------------------------------------


def _rolling_sums(x, y, periods, min_periods):
  """无 nan 数据上的滚动一阶/二阶和 (pandas rolling)."""
  x = np.asarray(x, dtype=np.float64)
  y = np.asarray(y, dtype=np.float64)
  df = pd.DataFrame(
    {"x": x, "y": y, "xy": x * y, "x2": x * x, "y2": y * y}
  )
  roll = df.rolling(periods, min_periods=min_periods)
  n = roll.count()["x"].to_numpy(dtype=np.float64)
  s = roll.sum()
  return (
    n,
    s["x"].to_numpy(dtype=np.float64),
    s["y"].to_numpy(dtype=np.float64),
    s["xy"].to_numpy(dtype=np.float64),
    s["x2"].to_numpy(dtype=np.float64),
    s["y2"].to_numpy(dtype=np.float64),
  )


def _corr2_ref(x, y, periods, min_periods):
  """无 nan 数据上的 CORR2 参考实现 (中心化协方差 / 标准差乘积)."""
  n, sx, sy, sxy, sx2, sy2 = _rolling_sums(x, y, periods, min_periods)
  with np.errstate(divide="ignore", invalid="ignore"):
    numerator = sxy - sx * sy / n
    var_x = np.maximum(sx2 - sx * sx / n, 0.0)
    var_y = np.maximum(sy2 - sy * sy / n, 0.0)
    denominator = np.sqrt(var_x * var_y)
    out = np.where(denominator > EPS, numerator / denominator, np.nan)
  out[n < 2] = np.nan
  return out


def _paired_nan(x, y):
  """任一序列为 nan 的位置标记成 nan, 供滚动窗口 mask 使用."""
  return np.where((~np.isnan(x)) & (~np.isnan(y)), 0.0, np.nan)


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


def _no_nan_window_mask(paired, periods):
  """每个位置的滚动窗口是否全部为成对有效值."""
  nan_count = (
    pd.Series(np.isnan(paired).astype(np.float64))
    .rolling(periods, min_periods=1)
    .sum()
    .to_numpy()
  )
  return nan_count == 0


def _full_valid_window_mask(paired, periods):
  """最近 periods 个原始位置是否全部成对有效 (SKIP_NAN + STRICT)."""
  valid_count = (
    pd.Series((~np.isnan(paired)).astype(np.float64))
    .rolling(periods, min_periods=periods)
    .sum()
    .to_numpy()
  )
  return valid_count == periods


def _expected(x, y, periods, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 全量计算参考值, 再只保留 alpha 应计算的 [start, end) 区间.

  无 SKIP_NAN 时窗口内任一 nan 就让结果为 nan; 有 SKIP_NAN 时只在
  [start, end) 内压紧成对有效值, 不跨边界借用数据. 严格模式另外要求连续
  ``periods`` 个原始位置全部成对有效.
  """
  x = np.asarray(x, dtype=np.float64)
  y = np.asarray(y, dtype=np.float64)
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  skip = bool(flags & al.FLAG_SKIP_NAN)
  min_periods = 2 if (skip and not strict) else periods
  parts = []
  for chunk_x, chunk_y in zip(np.split(x, groups), np.split(y, groups)):
    n = len(chunk_x)
    s, e = _calc_range(n, start, end)
    paired = _paired_nan(chunk_x, chunk_y)
    ref = np.full(n, np.nan)
    if s < e:
      if skip:
        seg_x = chunk_x[s:e]
        seg_y = chunk_y[s:e]
        valid = (~np.isnan(seg_x)) & (~np.isnan(seg_y))
        seg_ref = np.full(e - s, np.nan)
        if valid.any():
          seg_ref[valid] = _corr2_ref(
            seg_x[valid], seg_y[valid], periods, min_periods
          )
        if strict:
          mask = _full_valid_window_mask(_paired_nan(seg_x, seg_y), periods)
          seg_ref[~mask] = np.nan
        ref[s:e] = seg_ref
      else:
        ref = _corr2_ref(chunk_x, chunk_y, periods, min_periods)
        ref[~_no_nan_window_mask(paired, periods)] = np.nan
        ref[:s] = np.nan
        ref[e:] = np.nan
    parts.append(ref)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(x, y, periods, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.CORR2(
      np.asarray(x, dtype=np.float64),
      np.asarray(y, dtype=np.float64),
      periods,
    ),
    dtype=np.float64,
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261107)
  x = rng.normal(loc=10.0, scale=2.0, size=n).astype(np.float64)
  y = 1.5 * x + rng.normal(loc=0.0, scale=1.5, size=n)
  return x, y.astype(np.float64)


def _nan_data(n=N):
  x, y = _clean_data(n)
  x[NAN_POS_X[NAN_POS_X < n]] = np.nan
  y[NAN_POS_Y[NAN_POS_Y < n]] = np.nan
  return x, y


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("periods", [2, 3, 4, 5, 10, 20])
def test_corr2_matches_pandas_rolling(periods):
  """flags=0, groups=1: 与 pandas rolling 中心化公式逐点比较."""
  x, y = _clean_data()
  actual = _calc(x, y, periods, flags=0)
  _assert_close(actual, _expected(x, y, periods, flags=0))
  assert np.isnan(actual[: periods - 1]).all()
  assert np.isfinite(actual[periods - 1 :]).all()
  assert (np.abs(actual[periods - 1 :]) <= 1 + ATOL).all()


@pytest.mark.parametrize("periods", [2, 4, 10])
def test_corr2_strictly_cycle_alone(periods):
  """FLAG_STRICTLY_CYCLE 单独使用: 默认已是完整窗口, 结果与 flags=0 一致."""
  x, y = _clean_data()
  actual = _calc(x, y, periods, flags=al.FLAG_STRICTLY_CYCLE)
  _assert_close(actual, _expected(x, y, periods, flags=al.FLAG_STRICTLY_CYCLE))
  _assert_close(actual, _calc(x, y, periods, flags=0))


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
@pytest.mark.parametrize("with_nan", [False, True])
def test_corr2_without_skip_nan_propagates(flags, with_nan):
  """无 SKIP_NAN 时, 任一序列在窗口内出现 nan 都会使结果为 nan."""
  x, y = _nan_data() if with_nan else _clean_data()
  periods = 5
  actual = _calc(x, y, periods, flags=flags)
  _assert_close(actual, _expected(x, y, periods, flags=flags))

  if with_nan:
    paired = _paired_nan(x, y)
    for i in range(periods - 1, len(x)):
      if np.isnan(paired[i - periods + 1 : i + 1]).any():
        assert np.isnan(actual[i])


def test_corr2_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 结果等于成对压紧后的无 nan 数据参考结果."""
  x, y = _nan_data()
  periods = 4
  actual = _calc(x, y, periods, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, y, periods, flags=al.FLAG_SKIP_NAN))

  valid = (~np.isnan(x)) & (~np.isnan(y))
  assert np.isnan(actual[~valid]).all()
  _assert_close(actual[valid], _corr2_ref(x[valid], y[valid], periods, 2))


def test_corr2_skip_nan_and_strictly_cycle():
  """两个 flag 组合: 要求连续 periods 个位置都成对有效."""
  x, y = _nan_data()
  periods = 4
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, y, periods, flags=flags)
  _assert_close(actual, _expected(x, y, periods, flags=flags))
  assert np.isnan(
    actual[~_full_valid_window_mask(_paired_nan(x, y), periods)]
  ).all()


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
def test_corr2_groups(groups):
  """group-major 分块: 两个序列按同样的 chunk 切分, 各自独立计算."""
  x, y = _clean_data()
  periods = 4
  actual = _calc(x, y, periods, groups=groups, flags=0)
  _assert_close(actual, _expected(x, y, periods, groups=groups, flags=0))

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + periods - 1]).all()


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_corr2_skip_nan_within_each_group(groups):
  """SKIP_NAN 的成对压紧按 chunk 独立进行, 不会跨 group 借用数据."""
  x, y = _nan_data()
  periods = 5
  actual = _calc(x, y, periods, groups=groups, flags=al.FLAG_SKIP_NAN)
  _assert_close(
    actual, _expected(x, y, periods, groups=groups, flags=al.FLAG_SKIP_NAN)
  )


@pytest.mark.parametrize(
  "start,end", [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5)]
)
def test_corr2_start_end(start, end):
  """start/end: 窗口可引用 start 之前的数据, 区间外全部为 nan."""
  x, y = _clean_data()
  periods = 4
  actual = _calc(x, y, periods, start=start, end=end, flags=0)
  expected = _expected(x, y, periods, start=start, end=end, flags=0)
  s, e = _calc_range(len(x), start, end)
  _assert_close(actual[s:e], expected[s:e])
  assert np.isnan(actual[:s]).all()
  assert np.isnan(actual[e:]).all()


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_corr2_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  x, y = _clean_data()
  periods = 4
  start, end = 6, 25
  actual = _calc(x, y, periods, start=start, end=end, groups=groups, flags=0)
  expected = _expected(
    x, y, periods, start=start, end=end, groups=groups, flags=0
  )
  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    _assert_close(actual[lo + start : lo + end], expected[lo + start : lo + end])
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


@pytest.mark.parametrize("flags", [0, al.FLAG_SKIP_NAN])
def test_corr2_start_end_with_nan(flags):
  """含 nan 数据的 start/end: 只校验 [start, end) 内的结果."""
  x, y = _nan_data()
  periods = 4
  start, end = 10, 90
  actual = _calc(x, y, periods, start=start, end=end, flags=flags)
  expected = _expected(x, y, periods, start=start, end=end, flags=flags)
  _assert_close(actual[start:end], expected[start:end])
  assert np.isnan(actual[:start]).all()
  assert np.isnan(actual[end:]).all()
  assert np.isnan(actual[np.isnan(_paired_nan(x, y))]).all()


def test_corr2_edge_cases():
  """periods < 2 / 超长窗口 / 常数序列全部 nan; 完全线性序列相关为 1."""
  x, y = _clean_data()
  assert np.isnan(_calc(x, y, 1, flags=0)).all()
  assert np.isnan(_calc(x, y, len(x) + 1, flags=0)).all()

  constant = np.full(20, 3.0)
  assert np.isnan(_calc(constant, y[:20], 4, flags=0)).all()

  linear = np.arange(20.0)
  actual = _calc(linear, 2.5 * linear - 3.0, 4, flags=0)
  _assert_close(actual[3:], np.ones(17))
  assert np.isnan(actual[:3]).all()
