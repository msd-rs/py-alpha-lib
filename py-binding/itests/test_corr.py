# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""CORR (序列与时间下标的滚动相关系数) 集成测试.

算法简介
--------
计算最近 ``periods`` 个值与时间下标 ``t = 0, 1, ..., n-1`` 的 Pearson
相关系数 (也称时间序列自相关/趋势相关)::

    CORR[i] = sum((t - mean(t)) * (x - mean(x)))
              / sqrt(sum((t - mean(t)) ** 2) * sum((x - mean(x)) ** 2))

结果在 [-1, 1] 内; 窗口内 x 为常数 (方差为 0) 时输出 nan. 默认模式就要求
窗口填满 ``periods`` 个值 (至少 2 个), 窗口未满或窗口内含 nan 时输出 nan;
``FLAG_STRICTLY_CYCLE`` 在 CORR 上不改变结果. 注意这不是两序列相关系数
``CORR2``, 而是「序列与自身时间轴」的相关性.

对比基准使用 pandas ``rolling().apply()`` 计算中心化相关系数, 只处理无 nan
的数据, 不实现 SKIP_NAN / Group / start-end 机制: group 按 chunk 分块分别
计算, start/end 只校验 [start, end) 区间. ``FLAG_SKIP_NAN`` 等价于把有效值
压紧后跑参考实现再映射回原始有效位置, 非严格模式 2 个有效值即可输出.

`set_ctx` 只覆盖传入参数, 因此每个用例都显式重置 start/end/groups/flags.

场景与结果示例
--------------
1. 常规计算 (完全线性的序列相关系数恒为 1)::

    >>> import numpy as np
    >>> import alpha as al
    >>> x = np.arange(1., 11.)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.CORR(x, 4), 4)
    array([nan, nan, nan, 1., 1., 1., 1., 1., 1., 1.])

   非等差序列::

    >>> y = np.array([1., 3., 2., 6., 5., 9., 8., 12., 11., 15.])
    >>> np.round(al.CORR(y, 4), 4)
    array([   nan,    nan,    nan, 0.8367, 0.7071, 0.8944, 0.7071, 0.8944,
            0.7071, 0.8944])

2. 插入 nan 且不设置 ``FLAG_SKIP_NAN`` 时, nan 污染所在窗口::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.CORR(xn, 4), 4)
    array([nan, nan, nan, nan, nan, nan, nan, 1., 1., 1.])

3. ``FLAG_SKIP_NAN`` 让 nan 完全不参与运算, 等价于压紧有效值::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.CORR(xn, 4), 4)
    array([   nan, 1.    , 1.    ,    nan, 0.9827, 0.9899, 0.9827, 1.    ,
            1.    , 1.    ])

   nan 所在位置仍输出 nan; 组合 ``FLAG_STRICTLY_CYCLE`` 时要求连续 4 个
   原始位置全部有效, 结果回到 ``[nan, nan, nan, nan, nan, nan, nan, 1.,
   1., 1.]``.

4. ``groups`` 按 group-major 切块, 每块重新 warm-up::

    >>> z = np.arange(1., 13.)
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.CORR(z, 4), 4)
    array([nan, nan, nan, 1., 1., 1., nan, nan, nan, 1., 1., 1.])

5. ``start`` / ``end`` 只计算 [start, end) 区间 (测试只校验区间内结果)::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.CORR(x, 4)[3:8], 4)
    array([1., 1., 1., 1., 1.])
"""

import numpy as np
import pandas as pd
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
NAN_POS = np.array([3, 17, 18, 29, 44, 52, 77, 101])


# ---------------------------------------------------------------------------
# numpy / pandas 对比实现
# ---------------------------------------------------------------------------


def _corr_ref(x, periods, min_periods):
  """无 nan 数据上的 CORR 参考实现 (pandas rolling + Pearson 相关)."""
  x = np.asarray(x, dtype=np.float64)

  def _fit(w):
    t = np.arange(len(w), dtype=np.float64)
    dt = t - t.mean()
    dw = w - w.mean()
    sxx = float(np.dot(dt, dt))
    syy = float(np.dot(dw, dw))
    if sxx <= 0 or syy <= 0:
      return np.nan
    return float(np.dot(dt, dw) / np.sqrt(sxx * syy))

  return (
    pd.Series(x)
    .rolling(periods, min_periods=min_periods)
    .apply(_fit, raw=True)
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
  """压紧有效值后跑参考实现, 再映射回有效位置; nan 位置保持 nan."""
  out = np.full(len(x), np.nan)
  valid = ~np.isnan(x)
  if valid.any():
    out[valid] = _corr_ref(x[valid], periods, min_periods)
  return out


def _expected(x, periods, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 全量计算参考值, 再只保留 alpha 应计算的 [start, end) 区间.

  无 SKIP_NAN 时窗口内出现 nan 就让结果为 nan; 有 SKIP_NAN 时只在
  [start, end) 内压紧有效值, 不跨边界借用数据. 严格模式另外要求连续
  ``periods`` 个原始位置全部有效.
  """
  x = np.asarray(x, dtype=np.float64)
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  skip = bool(flags & al.FLAG_SKIP_NAN)
  min_periods = 2 if (skip and not strict) else periods
  parts = []
  for chunk in np.split(x, groups):
    n = len(chunk)
    s, e = _calc_range(n, start, end)
    ref = np.full(n, np.nan)
    if s < e:
      if skip:
        seg = chunk[s:e]
        seg_ref = _compact_ref(seg, periods, min_periods)
        if strict:
          seg_ref[~_full_valid_window_mask(seg, periods)] = np.nan
        ref[s:e] = seg_ref
      else:
        ref = _corr_ref(chunk, periods, min_periods)
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
    al.CORR(np.asarray(x, dtype=np.float64), periods), dtype=np.float64
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20260609)
  return rng.normal(loc=5.0, scale=2.0, size=n).astype(np.float64)


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
def test_corr_matches_pandas_rolling(periods):
  """flags=0, groups=1: 与 pandas rolling Pearson 相关参考实现逐点比较."""
  x = _clean_data()
  actual = _calc(x, periods, flags=0)
  _assert_close(actual, _expected(x, periods, flags=0))
  assert np.isnan(actual[: periods - 1]).all()
  assert ((actual[periods - 1 :] >= -1 - ATOL) &
          (actual[periods - 1 :] <= 1 + ATOL)).all()


@pytest.mark.parametrize("periods", [2, 4, 10])
def test_corr_strictly_cycle(periods):
  """FLAG_STRICTLY_CYCLE: CORR 默认已要求完整窗口, 两者结果一致."""
  x = _clean_data()
  actual = _calc(x, periods, flags=al.FLAG_STRICTLY_CYCLE)
  _assert_close(actual, _expected(x, periods, flags=al.FLAG_STRICTLY_CYCLE))
  _assert_close(actual, _calc(x, periods, flags=0))


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
@pytest.mark.parametrize("with_nan", [False, True])
def test_corr_without_skip_nan_propagates(flags, with_nan):
  """无 SKIP_NAN 时, 窗口内任一 nan 都会使结果变成 nan."""
  x = _nan_data() if with_nan else _clean_data()
  periods = 5
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))

  if with_nan:
    for i in range(periods - 1, len(x)):
      if np.isnan(x[i - periods + 1 : i + 1]).any():
        assert np.isnan(actual[i])


def test_corr_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 结果等于压紧后的无 nan 数据上的参考结果."""
  x = _nan_data()
  periods = 4
  actual = _calc(x, periods, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, periods, flags=al.FLAG_SKIP_NAN))

  valid = ~np.isnan(x)
  assert np.isnan(actual[~valid]).all()
  _assert_close(actual[valid], _corr_ref(x[valid], periods, 2))


def test_corr_skip_nan_and_strictly_cycle():
  """两个 flag 组合: 要求连续无 nan 的完整窗口."""
  x = _nan_data()
  periods = 4
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  assert np.isnan(actual[~_full_valid_window_mask(x, periods)]).all()
  assert np.isnan(actual[np.isnan(x)]).all()


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
def test_corr_groups(groups):
  """group-major 分块: 每个 chunk 独立计算, 开头重新 warm-up."""
  x = _clean_data()
  periods = 4
  actual = _calc(x, periods, groups=groups, flags=0)
  _assert_close(actual, _expected(x, periods, groups=groups, flags=0))

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + periods - 1]).all()


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_corr_skip_nan_within_each_group(groups):
  """SKIP_NAN 的压紧过程按 chunk 独立进行, 不会跨 group 借用数据."""
  x = _nan_data()
  periods = 5
  actual = _calc(x, periods, groups=groups, flags=al.FLAG_SKIP_NAN)
  _assert_close(
    actual, _expected(x, periods, groups=groups, flags=al.FLAG_SKIP_NAN)
  )


@pytest.mark.parametrize(
  "start,end", [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5)]
)
def test_corr_start_end(start, end):
  """start/end: 窗口可引用 start 之前的数据, 只校验 [start, end) 区间."""
  x = _clean_data()
  periods = 4
  actual = _calc(x, periods, start=start, end=end, flags=0)
  expected = _expected(x, periods, start=start, end=end, flags=0)
  s, e = _calc_range(len(x), start, end)
  _assert_close(actual[s:e], expected[s:e])
  assert np.isnan(actual[:s]).all()


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_corr_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  x = _clean_data()
  periods = 4
  start, end = 6, 25
  actual = _calc(x, periods, start=start, end=end, groups=groups, flags=0)
  expected = _expected(
    x, periods, start=start, end=end, groups=groups, flags=0
  )
  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    _assert_close(actual[lo + start : lo + end], expected[lo + start : lo + end])
    assert np.isnan(actual[lo : lo + start]).all()


@pytest.mark.parametrize("flags", [0, al.FLAG_SKIP_NAN])
def test_corr_start_end_with_nan(flags):
  """含 nan 数据的 start/end: 只校验 [start, end) 内的结果."""
  x = _nan_data()
  periods = 4
  start, end = 10, 90
  actual = _calc(x, periods, start=start, end=end, flags=flags)
  expected = _expected(x, periods, start=start, end=end, flags=flags)
  _assert_close(actual[start:end], expected[start:end])
  assert np.isnan(actual[:start]).all()
  assert np.isnan(actual[np.isnan(x)]).all()


def test_corr_edge_cases():
  """periods < 2 / 超长窗口 / 常数序列全部 nan; 完全线性序列相关为 1."""
  x = np.arange(1.0, 21.0)
  assert np.isnan(_calc(x, 1, flags=0)).all()
  assert np.isnan(_calc(x, len(x) + 1, flags=0)).all()
  assert np.isnan(_calc(np.full(20, 3.0), 4, flags=0)).all()

  linear = np.arange(20.0) * 2.5 - 3.0
  actual = _calc(linear, 4, flags=0)
  _assert_close(actual[3:], np.ones(17))
  assert np.isnan(actual[:3]).all()
