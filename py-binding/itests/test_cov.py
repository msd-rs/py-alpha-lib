# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""COV (两序列滚动协方差) 集成测试.

算法简介
--------
最近 ``periods`` 个位置上 x 与 y 的样本协方差 (ddof=1)::

    COV[i] = (sum(x*y) - sum(x)*sum(y)/n) / (n - 1)

默认模式要求窗口填满 ``periods`` 个值 (至少 2 个), 窗口未满或任一序列在
窗口内出现 nan 时输出 nan; ``FLAG_STRICTLY_CYCLE`` 在 COV 上不改变结果.

对比基准使用 pandas ``rolling().cov(ddof=1)``, 只处理无 nan 的数据, 不实现
SKIP_NAN / Group / start-end 机制: group 按 chunk 分块分别计算, start/end
只校验 [start, end) 区间. ``FLAG_SKIP_NAN`` 的语义是「最近 periods 个成对
有效 (x 与 y 同时非 nan) 的位置构成窗口」, 等价于把成对有效值压紧后跑参考
实现再映射回原始位置; 只有一个序列为 nan 的位置同样不参与运算.

`set_ctx` 只覆盖传入参数, 因此每个用例都显式重置 start/end/groups/flags.

场景与结果示例
--------------
1. 常规计算 (y = 2x, 协方差恒为样本方差的 2 倍)::

    >>> import numpy as np
    >>> import alpha as al
    >>> x = np.arange(1., 11.)
    >>> y = np.arange(2., 21., 2.)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.COV(x, y, 4), 4)
    array([nan, nan, nan, 3.3333, 3.3333, 3.3333, 3.3333, 3.3333, 3.3333,
           3.3333])

   另一组非完全相关的数据::

    >>> xa = np.array([1., 3., 2., 6., 5., 9., 8., 12., 11., 15.])
    >>> ya = np.array([2., 1., 4., 3., 8., 5., 12., 9., 14., 11.])
    >>> np.round(al.COV(xa, ya, 4), 4)
    array([nan, nan, nan, 0.3333, 1.6667, 0.3333, 1., 0., 1., -2.])

2. x 中插入 nan 且不设置 ``FLAG_SKIP_NAN`` 时, nan 污染所在窗口::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.COV(xn, y, 4), 4)
    array([nan, nan, nan, nan, nan, nan, nan, 3.3333, 3.3333, 3.3333])

3. ``FLAG_SKIP_NAN`` 使该位置在两条序列上都被跳过, 等价于成对压紧::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.COV(xn, y, 4), 4)
    array([nan, 1.    , 2.    , nan, 5.8333, 6.6667, 5.8333, 3.3333,
           3.3333, 3.3333])

   非严格模式下 2 个成对有效值即可计算; nan 位置本身仍输出 nan. 组合
   ``FLAG_STRICTLY_CYCLE`` 时要求连续 4 个原始位置全部成对有效, 结果回到
   ``[nan, nan, nan, nan, nan, nan, nan, 3.3333, 3.3333, 3.3333]``.

4. ``groups`` 按 group-major 切块, 每块重新 warm-up::

    >>> z = np.arange(1., 13.)
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.COV(z, z * 2, 4), 4)
    array([nan, nan, nan, 3.3333, 3.3333, 3.3333, nan, nan, nan, 3.3333,
           3.3333, 3.3333])

5. ``start`` / ``end`` 只计算 [start, end) 区间::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.COV(x, y, 4), 4)
    array([nan, nan, nan, 3.3333, 3.3333, 3.3333, 3.3333, 3.3333, nan, nan])
"""

import numpy as np
import pandas as pd
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
NAN_POS_X = np.array([3, 17, 18, 29, 44, 52, 77, 101])
NAN_POS_Y = np.array([5, 17, 30, 44, 60, 77, 95])


# ---------------------------------------------------------------------------
# numpy / pandas 对比实现
# ---------------------------------------------------------------------------


def _cov_ref(x, y, periods, min_periods):
  """无 nan 数据上的 COV 参考实现 (pandas rolling.cov, ddof=1)."""
  sx = pd.Series(np.asarray(x, dtype=np.float64))
  sy = pd.Series(np.asarray(y, dtype=np.float64))
  return (
    sx.rolling(periods, min_periods=min_periods)
    .cov(sy, ddof=1)
    .to_numpy(dtype=np.float64)
    .copy()
  )


def _paired_nan(x, y):
  """任一序列为 nan 的位置标记成 nan, 供滚动窗口 mask 使用."""
  return np.where(
    (~np.isnan(x)) & (~np.isnan(y)), 0.0, np.nan
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


def _compact_ref(x, y, periods, min_periods):
  """压紧成对有效值后跑参考实现, 再映射回有效位置."""
  out = np.full(len(x), np.nan)
  valid = (~np.isnan(x)) & (~np.isnan(y))
  if valid.any():
    out[valid] = _cov_ref(x[valid], y[valid], periods, min_periods)
  return out


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
        seg_ref = _compact_ref(seg_x, seg_y, periods, min_periods)
        if strict:
          mask = _full_valid_window_mask(_paired_nan(seg_x, seg_y), periods)
          seg_ref[~mask] = np.nan
        ref[s:e] = seg_ref
      else:
        ref = _cov_ref(chunk_x, chunk_y, periods, min_periods)
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
    al.COV(
      np.asarray(x, dtype=np.float64),
      np.asarray(y, dtype=np.float64),
      periods,
    ),
    dtype=np.float64,
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20260610)
  x = rng.normal(loc=10.0, scale=2.0, size=n).astype(np.float64)
  y = 0.6 * x + rng.normal(loc=1.0, scale=1.5, size=n)
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
def test_cov_matches_pandas_rolling(periods):
  """flags=0, groups=1: 与 pandas rolling.cov(ddof=1) 逐点比较."""
  x, y = _clean_data()
  actual = _calc(x, y, periods, flags=0)
  _assert_close(actual, _expected(x, y, periods, flags=0))
  assert np.isnan(actual[: periods - 1]).all()
  assert np.isfinite(actual[periods - 1 :]).all()


@pytest.mark.parametrize("periods", [2, 4, 10])
def test_cov_strictly_cycle(periods):
  """FLAG_STRICTLY_CYCLE: COV 默认已要求完整窗口, 两者结果一致."""
  x, y = _clean_data()
  actual = _calc(x, y, periods, flags=al.FLAG_STRICTLY_CYCLE)
  _assert_close(actual, _expected(x, y, periods, flags=al.FLAG_STRICTLY_CYCLE))
  _assert_close(actual, _calc(x, y, periods, flags=0))


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
@pytest.mark.parametrize("with_nan", [False, True])
def test_cov_without_skip_nan_propagates(flags, with_nan):
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


def test_cov_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 结果与成对压紧后的无 nan 数据参考结果一致."""
  x, y = _nan_data()
  periods = 4
  actual = _calc(x, y, periods, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, y, periods, flags=al.FLAG_SKIP_NAN))

  valid = (~np.isnan(x)) & (~np.isnan(y))
  assert np.isnan(actual[~valid]).all()
  _assert_close(actual[valid], _cov_ref(x[valid], y[valid], periods, 2))


def test_cov_skip_nan_and_strictly_cycle():
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
def test_cov_groups(groups):
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
def test_cov_skip_nan_within_each_group(groups):
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
def test_cov_start_end(start, end):
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
def test_cov_start_end_with_groups(groups):
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
def test_cov_start_end_with_nan(flags):
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


def test_cov_equals_variance_for_identical_series():
  """x == y 时协方差等于样本方差 (ddof=1)."""
  x, _ = _clean_data()
  periods = 6
  actual = _calc(x, x, periods, flags=0)
  variance = (
    pd.Series(x)
    .rolling(periods, min_periods=periods)
    .var(ddof=1)
    .to_numpy(dtype=np.float64)
  )
  _assert_close(actual, variance)


def test_cov_edge_cases():
  """periods < 2 / 超长窗口全部 nan; 常数序列协方差为 0."""
  x, y = _clean_data()
  assert np.isnan(_calc(x, y, 1, flags=0)).all()
  assert np.isnan(_calc(x, y, len(x) + 1, flags=0)).all()

  constant = np.full(20, 3.0)
  actual = _calc(constant, constant * 2, 4, flags=0)
  _assert_close(actual[3:], np.zeros(17))
  assert np.isnan(actual[:3]).all()
