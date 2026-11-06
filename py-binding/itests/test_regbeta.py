# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""REGBETA (滚动回归斜率 / Beta) 集成测试.

算法简介
--------
把 ``y`` 对 ``x`` 在最近 ``periods`` 个位置上做最小二乘回归, 返回斜率::

    REGBETA(y, x, d)[i] = (sum(x*y) - sum(x)*sum(y)/n)
                          / (sum(x^2) - sum(x)^2/n)

注意参数顺序是 ``(y, x)``, 分母用的是**第二个参数 x** 的方差. 分子分母都
不做 ``n-1`` 归一化 (ddof 因子约掉). 默认模式要求窗口填满 ``periods`` 个值
(至少 2 个), 窗口未满、窗口内任一序列出现 nan、或 x 为常数时输出 nan.
``FLAG_STRICTLY_CYCLE`` 单独使用不改变结果; 与 ``FLAG_SKIP_NAN`` 组合时
要求连续 ``periods`` 个原始位置全部有效.

对比基准使用 pandas ``rolling`` 的分组和按上述公式计算, 只处理无 nan 的
数据, 不实现 SKIP_NAN / Group / start-end 机制: group 按 chunk 分块分别
计算, start/end 只校验 [start, end) 区间. ``FLAG_SKIP_NAN`` 等价于把成对
有效 (x 与 y 同时非 nan) 的值压紧后跑参考实现再映射回原始位置.

`set_ctx` 只覆盖传入参数, 因此每个用例都显式重置 start/end/groups/flags.

场景与结果示例
--------------
1. 常规计算 (y ~ 2x 带噪声)::

    >>> import numpy as np
    >>> import alpha as al
    >>> x = np.arange(1., 11.)
    >>> y = np.array([2.1, 3.8, 6.2, 7.9, 10.5, 11.8, 14.2, 15.9, 18.1, 19.8])
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.REGBETA(y, x, 4), 4)
    array([nan, nan, nan, 1.98, 2.18, 1.94, 2.02, 1.86, 2.06, 1.9 ])

2. ``FLAG_SKIP_NAN`` 允许部分窗口, 且让 nan 不参与运算::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.REGBETA(y, x, 4), 4)
    array([nan, 1.7 , 2.05, 1.98, 2.18, 1.94, 2.02, 1.86, 2.06, 1.9 ])

    >>> xn = x.copy(); yn = y.copy(); xn[3] = np.nan; yn[6] = np.nan
    >>> np.round(al.REGBETA(yn, xn, 4), 4)
    array([   nan, 1.7   , 2.05  ,    nan, 2.1314, 2.03  ,    nan, 1.9154,
            1.93  , 2.0229])

   不设置 ``FLAG_SKIP_NAN`` 时, 窗口内任一 nan 会使结果变成 nan;
   ``FLAG_STRICTLY_CYCLE`` 单独使用结果与默认一致.

3. ``FLAG_SKIP_NAN | FLAG_STRICTLY_CYCLE`` 要求连续 4 个原始位置全部有效::

    >>> al.set_ctx(start=0, end=0, groups=1,
    ...            flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.REGBETA(yn, xn, 4), 4)
    array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])

4. ``groups`` 按 group-major 切块, 每块重新 warm-up (y = 2x + 1 时 beta=2)::

    >>> z = np.arange(1., 13.)
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.REGBETA(2 * z + 1, z, 4), 4)
    array([nan, nan, nan, 2., 2., 2., nan, nan, nan, 2., 2., 2.])

5. ``start`` / ``end`` 只计算 [start, end) 区间::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.REGBETA(y, x, 4), 4)
    array([nan, nan, nan, 1.98, 2.18, 1.94, 2.02, 1.86, nan, nan])
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
    {"x": x, "y": y, "xy": x * y, "x2": x * x}
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
  )


def _regbeta_ref(y, x, periods, min_periods):
  """无 nan 数据上的 REGBETA 参考实现 (协方差 / x 的方差)."""
  n, sx, sy, sxy, sx2 = _rolling_sums(x, y, periods, min_periods)
  with np.errstate(divide="ignore", invalid="ignore"):
    numerator = sxy - sx * sy / n
    var_x = sx2 - sx * sx / n
    out = np.where(np.abs(var_x) > EPS, numerator / var_x, np.nan)
  out[n < 2] = np.nan
  return out


def _paired_nan(y, x):
  """任一序列为 nan 的位置标记成 nan, 供滚动窗口 mask 使用."""
  return np.where((~np.isnan(y)) & (~np.isnan(x)), 0.0, np.nan)


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


def _expected(y, x, periods, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 全量计算参考值, 再只保留 alpha 应计算的 [start, end) 区间.

  无 SKIP_NAN 时窗口内任一 nan 就让结果为 nan; 有 SKIP_NAN 时只在
  [start, end) 内压紧成对有效值, 不跨边界借用数据. 严格模式另外要求连续
  ``periods`` 个原始位置全部成对有效.
  """
  y = np.asarray(y, dtype=np.float64)
  x = np.asarray(x, dtype=np.float64)
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  skip = bool(flags & al.FLAG_SKIP_NAN)
  min_periods = 2 if (skip and not strict) else periods
  parts = []
  for chunk_y, chunk_x in zip(np.split(y, groups), np.split(x, groups)):
    n = len(chunk_y)
    s, e = _calc_range(n, start, end)
    paired = _paired_nan(chunk_y, chunk_x)
    ref = np.full(n, np.nan)
    if s < e:
      if skip:
        seg_y = chunk_y[s:e]
        seg_x = chunk_x[s:e]
        valid = (~np.isnan(seg_y)) & (~np.isnan(seg_x))
        seg_ref = np.full(e - s, np.nan)
        if valid.any():
          seg_ref[valid] = _regbeta_ref(
            seg_y[valid], seg_x[valid], periods, min_periods
          )
        if strict:
          mask = _full_valid_window_mask(_paired_nan(seg_y, seg_x), periods)
          seg_ref[~mask] = np.nan
        ref[s:e] = seg_ref
      else:
        ref = _regbeta_ref(chunk_y, chunk_x, periods, min_periods)
        ref[~_no_nan_window_mask(paired, periods)] = np.nan
        ref[:s] = np.nan
        ref[e:] = np.nan
    parts.append(ref)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(y, x, periods, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.REGBETA(
      np.asarray(y, dtype=np.float64),
      np.asarray(x, dtype=np.float64),
      periods,
    ),
    dtype=np.float64,
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261108)
  x = rng.normal(loc=10.0, scale=2.0, size=n).astype(np.float64)
  y = 2.5 * x + rng.normal(loc=0.0, scale=1.5, size=n)
  return y.astype(np.float64), x


def _nan_data(n=N):
  y, x = _clean_data(n)
  y[NAN_POS_Y[NAN_POS_Y < n]] = np.nan
  x[NAN_POS_X[NAN_POS_X < n]] = np.nan
  return y, x


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("periods", [2, 3, 4, 5, 10, 20])
def test_regbeta_matches_pandas_rolling(periods):
  """flags=0, groups=1: 与 pandas rolling 协方差/方差公式逐点比较."""
  y, x = _clean_data()
  actual = _calc(y, x, periods, flags=0)
  _assert_close(actual, _expected(y, x, periods, flags=0))
  assert np.isnan(actual[: periods - 1]).all()
  assert np.isfinite(actual[periods - 1 :]).all()


@pytest.mark.parametrize("periods", [2, 4, 10])
def test_regbeta_strictly_cycle_alone(periods):
  """FLAG_STRICTLY_CYCLE 单独使用: 默认已是完整窗口, 结果与 flags=0 一致."""
  y, x = _clean_data()
  actual = _calc(y, x, periods, flags=al.FLAG_STRICTLY_CYCLE)
  _assert_close(actual, _expected(y, x, periods, flags=al.FLAG_STRICTLY_CYCLE))
  _assert_close(actual, _calc(y, x, periods, flags=0))


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
@pytest.mark.parametrize("with_nan", [False, True])
def test_regbeta_without_skip_nan_propagates(flags, with_nan):
  """无 SKIP_NAN 时, 任一序列在窗口内出现 nan 都会使结果为 nan."""
  y, x = _nan_data() if with_nan else _clean_data()
  periods = 5
  actual = _calc(y, x, periods, flags=flags)
  _assert_close(actual, _expected(y, x, periods, flags=flags))

  if with_nan:
    paired = _paired_nan(y, x)
    for i in range(periods - 1, len(x)):
      if np.isnan(paired[i - periods + 1 : i + 1]).any():
        assert np.isnan(actual[i])


def test_regbeta_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 结果等于成对压紧后的无 nan 数据参考结果."""
  y, x = _nan_data()
  periods = 4
  actual = _calc(y, x, periods, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(y, x, periods, flags=al.FLAG_SKIP_NAN))

  valid = (~np.isnan(y)) & (~np.isnan(x))
  assert np.isnan(actual[~valid]).all()
  _assert_close(actual[valid], _regbeta_ref(y[valid], x[valid], periods, 2))


def test_regbeta_skip_nan_and_strictly_cycle():
  """两个 flag 组合: 要求连续 periods 个位置都成对有效."""
  y, x = _nan_data()
  periods = 4
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(y, x, periods, flags=flags)
  _assert_close(actual, _expected(y, x, periods, flags=flags))
  assert np.isnan(
    actual[~_full_valid_window_mask(_paired_nan(y, x), periods)]
  ).all()


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
def test_regbeta_groups(groups):
  """group-major 分块: 两个序列按同样的 chunk 切分, 各自独立计算."""
  y, x = _clean_data()
  periods = 4
  actual = _calc(y, x, periods, groups=groups, flags=0)
  _assert_close(actual, _expected(y, x, periods, groups=groups, flags=0))

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + periods - 1]).all()


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_regbeta_skip_nan_within_each_group(groups):
  """SKIP_NAN 的成对压紧按 chunk 独立进行, 不会跨 group 借用数据."""
  y, x = _nan_data()
  periods = 5
  actual = _calc(y, x, periods, groups=groups, flags=al.FLAG_SKIP_NAN)
  _assert_close(
    actual, _expected(y, x, periods, groups=groups, flags=al.FLAG_SKIP_NAN)
  )


@pytest.mark.parametrize(
  "start,end", [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5)]
)
def test_regbeta_start_end(start, end):
  """start/end: 窗口可引用 start 之前的数据, 区间外全部为 nan."""
  y, x = _clean_data()
  periods = 4
  actual = _calc(y, x, periods, start=start, end=end, flags=0)
  expected = _expected(y, x, periods, start=start, end=end, flags=0)
  s, e = _calc_range(len(x), start, end)
  _assert_close(actual[s:e], expected[s:e])
  assert np.isnan(actual[:s]).all()
  assert np.isnan(actual[e:]).all()


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_regbeta_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  y, x = _clean_data()
  periods = 4
  start, end = 6, 25
  actual = _calc(y, x, periods, start=start, end=end, groups=groups, flags=0)
  expected = _expected(
    y, x, periods, start=start, end=end, groups=groups, flags=0
  )
  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    _assert_close(actual[lo + start : lo + end], expected[lo + start : lo + end])
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


@pytest.mark.parametrize("flags", [0, al.FLAG_SKIP_NAN])
def test_regbeta_start_end_with_nan(flags):
  """含 nan 数据的 start/end: 只校验 [start, end) 内的结果."""
  y, x = _nan_data()
  periods = 4
  start, end = 10, 90
  actual = _calc(y, x, periods, start=start, end=end, flags=flags)
  expected = _expected(y, x, periods, start=start, end=end, flags=flags)
  _assert_close(actual[start:end], expected[start:end])
  assert np.isnan(actual[:start]).all()
  assert np.isnan(actual[end:]).all()
  assert np.isnan(actual[np.isnan(_paired_nan(y, x))]).all()


def test_regbeta_edge_cases():
  """periods < 2 / 超长窗口 / x 为常数全部 nan; 线性序列 beta 为斜率."""
  y, x = _clean_data()
  assert np.isnan(_calc(y, x, 1, flags=0)).all()
  assert np.isnan(_calc(y, x, len(x) + 1, flags=0)).all()

  constant_x = np.full(20, 3.0)
  assert np.isnan(_calc(y[:20], constant_x, 4, flags=0)).all()

  linear_x = np.arange(20.0)
  actual = _calc(2.5 * linear_x - 3.0, linear_x, 4, flags=0)
  _assert_close(actual[3:], np.full(17, 2.5))
  assert np.isnan(actual[:3]).all()

  constant_y = np.full(20, 4.0)
  _assert_close(_calc(constant_y, linear_x, 4, flags=0)[3:], np.zeros(17))
