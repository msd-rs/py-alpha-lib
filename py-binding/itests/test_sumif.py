# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""SUMIF (条件滚动求和) 集成测试.

算法简介
--------
最近 ``periods`` 个位置上, 只对 ``condition`` 为 True 的 bar 累加输入::

    SUMIF(x, cond, d)[i] = sum(x[k] for k in [i-d+1, i] if cond[k])

``condition`` 是 bool 数组; ``cond=False`` 的位置仍然占用一个窗口位置, 只是
贡献 0. ``FLAG_STRICTLY_CYCLE`` 要求窗口填满 ``periods`` 个值 (前 d-1 个
位置输出 nan), 默认模式允许渐长的部分窗口. 无 ``FLAG_SKIP_NAN`` 时, 窗口内
任一「cond=True 且 x 为 nan」的 bar 会让结果变成 nan (cond=False 位置的 nan
被忽略); 有 ``FLAG_SKIP_NAN`` 时, x 为 nan 的 bar 完全不参与运算, 窗口由最近
``periods`` 个有效值构成, 等价于把 x (及对应的 cond) 压紧后跑参考实现.

对比基准使用 pandas ``rolling().sum()``, 只处理无 nan 的数据, 不实现
SKIP_NAN / Group / start-end 机制: group 按 chunk 分块分别计算, start/end
只校验 [start, end) 区间. ``set_ctx`` 只覆盖传入参数, 因此每个用例都显式
重置 start/end/groups/flags.

场景与结果示例
--------------
1. 常规计算 (允许渐长的部分窗口)::

    >>> import numpy as np
    >>> import alpha as al
    >>> x = np.arange(1., 11.)
    >>> cond = np.array([True, False, True, True, False, True, True, False,
    ...                  True, False])
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.SUMIF(x, cond, 3), 4)
    array([ 1.,  1.,  4.,  7.,  7., 10., 13., 13., 16.,  9.])

   前 3 个位置的窗口 [1], [1,2], [1,2,3] 中只有 1、3 被计入.

2. ``FLAG_STRICTLY_CYCLE`` 要求窗口填满::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.SUMIF(x, cond, 3), 4)
    array([nan, nan,  4.,  7.,  7., 10., 13., 13., 16.,  9.])

3. x 中插入 nan 时不设置 ``FLAG_SKIP_NAN`` 会污染所在窗口::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.SUMIF(xn, cond, 3), 4)
    array([ 1.,  1.,  4., nan, nan, nan, 13., 13., 16.,  9.])

4. ``FLAG_SKIP_NAN`` 让 nan 完全不参与运算, 等价于压紧有效值::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.SUMIF(xn, cond, 3), 4)
    array([ 1.,  1.,  4., nan,  3.,  9., 13., 13., 16.,  9.])

   nan 所在位置本身仍输出 nan. 组合 ``FLAG_STRICTLY_CYCLE`` 时还要求连续
   3 个原始位置都有效, 结果为 ``[nan, nan, 4, nan, nan, nan, 13, 13,
   16, 9]``.

5. ``groups`` 按 group-major 切块, 每块重新 warm-up::

    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.SUMIF(np.arange(1., 13.), np.tile(cond, 2)[:12], 3), 4)
    array([ 1.,  1.,  4.,  7.,  7., 10.,  7.,  7., 16.,  9., 20., 11.])

6. ``start`` / ``end`` 只计算 [start, end) 区间::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.SUMIF(x, cond, 3), 4)
    array([nan, nan, nan,  7.,  7., 10., 13., 13., nan, nan])
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


def _sumif_ref(x, cond, periods):
  """无 nan 数据上的 SUMIF 参考实现 (pandas rolling sum)."""
  v = np.where(cond, np.asarray(x, dtype=np.float64), 0.0)
  return (
    pd.Series(v)
    .rolling(periods, min_periods=1)
    .sum()
    .to_numpy(dtype=np.float64)
    .copy()
  )


def _bad_window_mask(x, cond, periods):
  """窗口内是否存在 cond=True 且 x 为 nan 的 bar (存在则结果为 nan)."""
  bad = (cond & np.isnan(x)).astype(np.float64)
  nan_count = (
    pd.Series(bad).rolling(periods, min_periods=1).sum().to_numpy()
  )
  return nan_count > 0


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


def _full_valid_window_mask(x, periods):
  """最近 periods 个原始位置是否全部有效 (SKIP_NAN + STRICTLY_CYCLE)."""
  valid_count = (
    pd.Series((~np.isnan(x)).astype(np.float64))
    .rolling(periods, min_periods=periods)
    .sum()
    .to_numpy()
  )
  return valid_count == periods


def _compact_ref(x, cond, periods, strict):
  """压紧 x 中的 nan (连同 cond) 后跑参考实现, 再映射回原始位置."""
  out = np.full(len(x), np.nan)
  valid = ~np.isnan(x)
  if valid.any():
    out[valid] = _sumif_ref(x[valid], cond[valid], periods)
  if strict:
    out[~_full_valid_window_mask(x, periods)] = np.nan
  return out


def _expected(x, cond, periods, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 全量计算参考值, 再只保留 alpha 应计算的 [start, end) 区间.

  无 SKIP_NAN 时窗口内 cond=True 的 nan 会让结果为 nan; 有 SKIP_NAN 时只在
  [start, end) 内压紧有效值, 不跨边界借用数据. 严格模式在非 SKIP_NAN 分支
  要求窗口填满, 在 SKIP_NAN 分支还要求连续 ``periods`` 个原始位置全部有效.
  """
  x = np.asarray(x, dtype=np.float64)
  cond = np.asarray(cond, dtype=bool)
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  skip = bool(flags & al.FLAG_SKIP_NAN)
  parts = []
  for chunk_x, chunk_c in zip(np.split(x, groups), np.split(cond, groups)):
    n = len(chunk_x)
    s, e = _calc_range(n, start, end)
    ref = np.full(n, np.nan)
    if s < e:
      if skip:
        ref[s:e] = _compact_ref(
          chunk_x[s:e], chunk_c[s:e], periods, strict
        )
      else:
        ref = _sumif_ref(chunk_x, chunk_c, periods)
        ref[_bad_window_mask(chunk_x, chunk_c, periods)] = np.nan
        if strict:
          ref[: periods - 1] = np.nan
        ref[:s] = np.nan
        ref[e:] = np.nan
    parts.append(ref)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(x, cond, periods, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.SUMIF(
      np.asarray(x, dtype=np.float64),
      np.asarray(cond, dtype=bool),
      periods,
    ),
    dtype=np.float64,
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261110)
  x = rng.uniform(0.5, 5.0, size=n).astype(np.float64)
  cond = rng.random(n) < 0.6
  return x, cond


def _nan_data(n=N):
  x, cond = _clean_data(n)
  x[NAN_POS[NAN_POS < n]] = np.nan
  return x, cond


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("periods", [1, 2, 3, 4, 5, 10, 20])
def test_sumif_matches_pandas_rolling(periods):
  """flags=0, groups=1: 与 pandas rolling 条件求和逐点比较."""
  x, cond = _clean_data()
  actual = _calc(x, cond, periods, flags=0)
  _assert_close(actual, _expected(x, cond, periods, flags=0))
  assert np.isfinite(actual).all()


@pytest.mark.parametrize("periods", [2, 3, 5, 10])
def test_sumif_strictly_cycle(periods):
  """FLAG_STRICTLY_CYCLE: 窗口填满 periods 个值之前全部为 nan."""
  x, cond = _clean_data()
  actual = _calc(x, cond, periods, flags=al.FLAG_STRICTLY_CYCLE)
  _assert_close(
    actual, _expected(x, cond, periods, flags=al.FLAG_STRICTLY_CYCLE)
  )
  assert np.isnan(actual[: periods - 1]).all()
  assert np.isfinite(actual[periods - 1 :]).all()


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
@pytest.mark.parametrize("with_nan", [False, True])
def test_sumif_without_skip_nan_propagates(flags, with_nan):
  """无 SKIP_NAN 时, 窗口内 cond=True 的 nan 会让结果变成 nan."""
  x, cond = _nan_data() if with_nan else _clean_data()
  periods = 5
  actual = _calc(x, cond, periods, flags=flags)
  _assert_close(actual, _expected(x, cond, periods, flags=flags))

  if with_nan:
    bad = cond & np.isnan(x)
    for i in range(len(x)):
      if bad[max(0, i - periods + 1) : i + 1].any():
        assert np.isnan(actual[i])


def test_sumif_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 结果等于压紧后的无 nan 数据参考结果."""
  x, cond = _nan_data()
  periods = 4
  actual = _calc(x, cond, periods, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, cond, periods, flags=al.FLAG_SKIP_NAN))

  valid = ~np.isnan(x)
  assert np.isnan(actual[~valid]).all()
  _assert_close(actual[valid], _sumif_ref(x[valid], cond[valid], periods))


def test_sumif_skip_nan_and_strictly_cycle():
  """两个 flag 组合: 要求连续无 nan 的完整窗口."""
  x, cond = _nan_data()
  periods = 4
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, cond, periods, flags=flags)
  _assert_close(actual, _expected(x, cond, periods, flags=flags))
  assert np.isnan(actual[~_full_valid_window_mask(x, periods)]).all()


@pytest.mark.parametrize("cond_value", [False, True])
def test_sumif_constant_condition(cond_value):
  """cond 全 False 时输出 0; cond 全 True 时等价于普通滚动求和."""
  x, _ = _clean_data()
  cond = np.full(len(x), cond_value, dtype=bool)
  periods = 4
  actual = _calc(x, cond, periods, flags=0)
  _assert_close(actual, _expected(x, cond, periods, flags=0))
  if not cond_value:
    _assert_close(actual, np.zeros(len(x)))


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
def test_sumif_groups(groups):
  """group-major 分块: x 与 cond 按相同 chunk 切分, 各自独立计算."""
  x, cond = _clean_data()
  periods = 4
  actual = _calc(x, cond, periods, groups=groups, flags=0)
  _assert_close(actual, _expected(x, cond, periods, groups=groups, flags=0))


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_sumif_skip_nan_within_each_group(groups):
  """SKIP_NAN 的压紧过程按 chunk 独立进行, 不会跨 group 借用数据."""
  x, cond = _nan_data()
  periods = 5
  actual = _calc(x, cond, periods, groups=groups, flags=al.FLAG_SKIP_NAN)
  _assert_close(
    actual, _expected(x, cond, periods, groups=groups, flags=al.FLAG_SKIP_NAN)
  )


@pytest.mark.parametrize(
  "start,end", [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5)]
)
def test_sumif_start_end(start, end):
  """start/end: 窗口可引用 start 之前的数据, 区间外全部为 nan."""
  x, cond = _clean_data()
  periods = 4
  actual = _calc(x, cond, periods, start=start, end=end, flags=0)
  expected = _expected(x, cond, periods, start=start, end=end, flags=0)
  s, e = _calc_range(len(x), start, end)
  _assert_close(actual[s:e], expected[s:e])
  assert np.isnan(actual[:s]).all()
  assert np.isnan(actual[e:]).all()


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_sumif_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  x, cond = _clean_data()
  periods = 4
  start, end = 6, 25
  actual = _calc(x, cond, periods, start=start, end=end, groups=groups, flags=0)
  expected = _expected(
    x, cond, periods, start=start, end=end, groups=groups, flags=0
  )
  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    _assert_close(actual[lo + start : lo + end], expected[lo + start : lo + end])
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


@pytest.mark.parametrize("flags", [0, al.FLAG_SKIP_NAN])
def test_sumif_start_end_with_nan(flags):
  """含 nan 数据的 start/end: 只校验 [start, end) 内的结果."""
  x, cond = _nan_data()
  periods = 4
  start, end = 10, 90
  actual = _calc(x, cond, periods, start=start, end=end, flags=flags)
  expected = _expected(x, cond, periods, start=start, end=end, flags=flags)
  _assert_close(actual[start:end], expected[start:end])
  assert np.isnan(actual[:start]).all()
  assert np.isnan(actual[end:]).all()
  if flags & al.FLAG_SKIP_NAN:
    assert np.isnan(actual[np.isnan(x)]).all()


def test_sumif_edge_cases():
  """periods=1 只累计当前 bar; 窗口长于数据时仍能输出部分和."""
  x, cond = _clean_data()
  actual = _calc(x, cond, 1, flags=0)
  _assert_close(actual, np.where(cond, x, 0.0))

  long_period = len(x) + 10
  _assert_close(
    _calc(x, cond, long_period, flags=0),
    _expected(x, cond, long_period, flags=0),
  )
  assert np.isnan(
    _calc(x, cond, long_period, flags=al.FLAG_STRICTLY_CYCLE)
  ).all()
