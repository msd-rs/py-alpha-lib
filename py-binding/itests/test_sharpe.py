# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""SHARPE (滚动夏普比率) 集成测试.

算法简介
--------
滚动窗口内收益率的夏普比率 (不做年化)::

    SHARPE[i] = mean(x[i-d+1..i]) / std_pop(x[i-d+1..i])

标准差为总体标准差 (ddof=0). 非严格模式下窗口未满时用已有数据计算, 至少
需要 2 个值; 方差为 0 时输出 nan. ``FLAG_STRICTLY_CYCLE`` 要求完整窗口.

对比基准使用 pandas ``rolling`` 的 mean / std(ddof=0), 只处理无 nan 的数据,
不实现 SKIP_NAN / Group / start-end 机制: group 按 chunk 分块分别计算,
start/end 只校验 [start, end) 区间. ``FLAG_SKIP_NAN`` 的语义是「最近 d 个
有效值构成窗口」, 等价于把有效值压紧 (compact) 后跑参考实现再映射回原位.

`set_ctx` 只覆盖传入参数, 因此每个用例都显式重置 start/end/groups/flags.

场景与结果示例
--------------
1. 常规计算, 窗口未满时允许部分窗口 (至少 2 个值)::

    >>> import numpy as np
    >>> import alpha as al
    >>> x = np.arange(1., 11.)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.SHARPE(x, 4), 4)
    array([nan, 3.    , 2.4495, 2.2361, 3.1305, 4.0249, 4.9193, 5.8138,
           6.7082, 7.6026])

   前两个值 [1, 2] 的均值为 1.5、总体标准差为 0.5, 结果为 3.0.

2. ``FLAG_STRICTLY_CYCLE`` 要求窗口填满才输出::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.SHARPE(x, 4), 4)
    array([nan, nan, nan, 2.2361, 3.1305, 4.0249, 4.9193, 5.8138, 6.7082,
           7.6026])

3. 不设置 ``FLAG_SKIP_NAN`` 时, nan 会污染其所在的整个窗口::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.SHARPE(xn, 4), 4)
    array([nan, 3.    , 2.4495, nan, nan, nan, nan, 5.8138, 6.7082, 7.6026])

4. ``FLAG_SKIP_NAN`` 使 nan 完全不参与运算, 等价于压紧有效值::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.SHARPE(xn, 4), 4)
    array([nan, 3.    , 2.4495, nan, 1.8593, 2.5298, 3.5496, 5.8138,
           6.7082, 7.6026])

   nan 所在位置本身仍输出 nan. 组合 ``FLAG_STRICTLY_CYCLE`` 时还要求最近
   4 个原始位置全部有效, 结果为 ``[nan, nan, nan, nan, nan, nan, nan,
   5.8138, 6.7082, 7.6026]``.

5. ``groups`` 按 group-major 切块, 每块重新 warm-up::

    >>> y = np.arange(1., 13.)
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.SHARPE(y, 4), 4)
    array([nan, 3.    , 2.4495, 2.2361, 3.1305, 4.0249, nan, 15.    ,
           9.798 , 7.6026, 8.4971, 9.3915])

6. ``start`` / ``end`` 只计算 [start, end) 区间::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.SHARPE(x, 4), 4)
    array([nan, nan, nan, 2.2361, 3.1305, 4.0249, 4.9193, 5.8138, nan, nan])
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


def _sharpe_ref(x, periods, min_periods):
  """无 nan 数据上的 SHARPE 参考实现 (pandas rolling mean / std(ddof=0))."""
  s = pd.Series(np.asarray(x, dtype=np.float64))
  mean = s.rolling(periods, min_periods=min_periods).mean().to_numpy()
  std = s.rolling(periods, min_periods=min_periods).std(ddof=0).to_numpy()
  with np.errstate(divide="ignore", invalid="ignore"):
    out = mean / std
  # 常数窗口方差为 0, 按算法约定输出 nan.
  out[np.isfinite(std) & (std <= 0)] = np.nan
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
  """压紧有效值后跑参考实现, 再映射回有效位置; nan 位置保持 nan."""
  out = np.full(len(x), np.nan)
  valid = ~np.isnan(x)
  if valid.any():
    out[valid] = _sharpe_ref(x[valid], periods, min_periods)
  return out


def _expected(x, periods, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 全量计算参考值, 再只保留 alpha 应计算的 [start, end) 区间.

  无 SKIP_NAN 时窗口内出现 nan 就让结果为 nan; 有 SKIP_NAN 时窗口由最近
  ``periods`` 个有效值构成 (回溯整个 chunk, 不受 start 限制), nan 位置输出
  nan. 严格模式另外要求连续 ``periods`` 个原始位置全部有效.
  """
  x = np.asarray(x, dtype=np.float64)
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  skip = bool(flags & al.FLAG_SKIP_NAN)
  min_periods = periods if strict else 2
  parts = []
  for chunk in np.split(x, groups):
    n = len(chunk)
    s, e = _calc_range(n, start, end)
    if skip:
      ref = _compact_ref(chunk, periods, min_periods)
      if strict:
        ref[~_full_valid_window_mask(chunk, periods)] = np.nan
    else:
      ref = _sharpe_ref(chunk, periods, min_periods)
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
    al.SHARPE(np.asarray(x, dtype=np.float64), periods), dtype=np.float64
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261106)
  return rng.normal(loc=0.5, scale=2.0, size=n).astype(np.float64)


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
def test_sharpe_matches_pandas_rolling(periods):
  """flags=0, groups=1: 与 pandas rolling mean/std 参考实现逐点比较."""
  x = _clean_data()
  actual = _calc(x, periods, flags=0)
  _assert_close(actual, _expected(x, periods, flags=0))
  assert np.isnan(actual[0])
  assert np.isfinite(actual[1:]).all()


@pytest.mark.parametrize("periods", [2, 3, 5, 10, 20])
def test_sharpe_strictly_cycle(periods):
  """FLAG_STRICTLY_CYCLE: 窗口填满 periods 个值之前全部为 nan."""
  x = _clean_data()
  actual = _calc(x, periods, flags=al.FLAG_STRICTLY_CYCLE)
  _assert_close(actual, _expected(x, periods, flags=al.FLAG_STRICTLY_CYCLE))
  assert np.isnan(actual[: periods - 1]).all()
  assert np.isfinite(actual[periods - 1 :]).all()


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
@pytest.mark.parametrize("with_nan", [False, True])
def test_sharpe_without_skip_nan_propagates(flags, with_nan):
  """无 SKIP_NAN 时, 窗口内任一 nan 都会使结果变成 nan."""
  x = _nan_data() if with_nan else _clean_data()
  periods = 5
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))

  if with_nan:
    for i in range(len(x)):
      if np.isnan(x[max(0, i - periods + 1) : i + 1]).any():
        assert np.isnan(actual[i])


def test_sharpe_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 结果与「未插入 nan」的压紧数据上的参考结果一致."""
  x = _nan_data()
  periods = 4
  actual = _calc(x, periods, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, periods, flags=al.FLAG_SKIP_NAN))

  valid = ~np.isnan(x)
  assert np.isnan(actual[~valid]).all()
  _assert_close(actual[valid], _sharpe_ref(x[valid], periods, 2))


def test_sharpe_skip_nan_and_strictly_cycle():
  """两个 flag 组合: 要求连续无 nan 的完整窗口."""
  x = _nan_data()
  periods = 4
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  assert np.isnan(actual[~_full_valid_window_mask(x, periods)]).all()
  assert np.isnan(actual[np.isnan(x)]).all()


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
def test_sharpe_groups(groups):
  """group-major 分块: 每个 chunk 独立计算, 开头重新 warm-up."""
  x = _clean_data()
  periods = 4
  actual = _calc(x, periods, groups=groups, flags=0)
  _assert_close(actual, _expected(x, periods, groups=groups, flags=0))

  chunk_size = len(x) // groups
  for g in range(groups):
    assert np.isnan(actual[g * chunk_size])


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_sharpe_skip_nan_within_each_group(groups):
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
def test_sharpe_start_end(start, end):
  """start/end: 只校验 [start, end) 区间, 区间外必须全是 nan."""
  x = _clean_data()
  periods = 4
  actual = _calc(x, periods, start=start, end=end, flags=0)
  expected = _expected(x, periods, start=start, end=end, flags=0)
  s, e = _calc_range(len(x), start, end)
  _assert_close(actual[s:e], expected[s:e])
  assert np.isnan(actual[:s]).all()
  assert np.isnan(actual[e:]).all()


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_sharpe_start_end_with_groups(groups):
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
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


@pytest.mark.parametrize("flags", [0, al.FLAG_SKIP_NAN])
def test_sharpe_start_end_with_nan(flags):
  """含 nan 数据的 start/end: 只校验 [start, end) 内的结果."""
  x = _nan_data()
  periods = 4
  start, end = 10, 90
  actual = _calc(x, periods, start=start, end=end, flags=flags)
  expected = _expected(x, periods, start=start, end=end, flags=flags)
  _assert_close(actual[start:end], expected[start:end])
  assert np.isnan(actual[:start]).all()
  assert np.isnan(actual[end:]).all()
  assert np.isnan(actual[np.isnan(x)]).all()


def test_sharpe_edge_cases():
  """periods < 2 报错; 常数序列方差为 0 输出 nan; 超长窗口仍可用部分窗口."""
  x = _clean_data()
  with pytest.raises(ValueError):
    _calc(x, 1, flags=0)

  constant = np.full(20, 2.0)
  assert np.isnan(_calc(constant, 4, flags=0)).all()

  long_period = len(x) + 10
  _assert_close(
    _calc(x, long_period, flags=0),
    _expected(x, long_period, flags=0),
  )
