# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""RANK (Time-Series Rank, 滚动窗口排名) 集成测试.

算法简介
--------
RANK 返回当前值在最近 `periods` 个值构成的窗口中的名次 (从 1 开始), 并列
值取最小名次 (min-rank, 同 ``pandas.rank(method="min")``)::

    RANK[i] = 1 + count(x[j] < x[i] for j in [i-periods+1, i])

窗口未满时 (``flags=0``) 用已有的 1..periods 个值计算部分排名; 设置
``FLAG_STRICTLY_CYCLE`` 后窗口未满输出 nan, 与 pandas ``rolling()`` 默认
行为一致. 单调递增序列在窗口填满后名次恒为 `periods`.

参考实现 ``_rank_ref`` 只在无 nan 数据上逐窗口计数, 不实现 SKIP_NAN /
Group / start-end 机制: Group 通过按 chunk 分块计算, start/end 只校验
[start, end) 区间, 窗口本身仍应引用 start 之前的历史数据 (与其他滚动算子的
``Context`` 语义一致). 理想语义下 nan 是缺失值: 窗口内出现 nan 会污染结果,
``FLAG_SKIP_NAN`` 应把有效值压紧后再计算, nan 位置输出 nan.

当前 Rust 实现有两处与上述算法逻辑不一致, 相关用例保留失败以暴露差异,
不应为了让测试通过而修改基准:

* 无论 flags 如何, 窗口内 nan 都会参与计数, nan 位置返回窗口大小而不是 nan;
* ``start`` 之前的数据不进入窗口, [start, end) 开头几个值会从 1 重新计数.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=120)
    >>> x = np.arange(1., 11.)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算 (窗口未满也会给出部分名次)::

    >>> np.round(al.RANK(x, 4), 3)
    array([1., 2., 3., 4., 4., 4., 4., 4., 4., 4.])

2. ``FLAG_STRICTLY_CYCLE``: 窗口未满输出 nan::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.RANK(x, 4), 3)
    array([nan, nan, nan,  4.,  4.,  4.,  4.,  4.,  4.,  4.])

3. 并列值取最小名次: ``[3, 1, 3, 2, 3, 5]`` 的 3 周期 RANK 为
   ``[1, 1, 2, 2, 2, 3]`` (i=2 的窗口 [3, 1, 3] 中只有 1 小于 3)::

    >>> t = np.array([3., 1., 3., 2., 3., 5.])
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.RANK(t, 3), 3)
    array([1., 1., 2., 2., 2., 3.])

4. ``groups`` 按 group-major 切块独立计算, 每块重新 warm-up::

    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.RANK(x, 4), 3)
    array([1., 2., 3., 4., 4., 1., 2., 3., 4., 4.])

5. ``start`` / ``end`` 只计算 [start, end) 区间, 区间外全部为 nan. 当前实现
   在 start 处重新累积窗口, 因此前几个值从 1 开始; 按算法逻辑窗口应能引用
   start 之前的数据 (输出 4)::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.RANK(x, 4), 3)
    array([nan, nan, nan,  1.,  2.,  3.,  4.,  4., nan, nan])

6. ``periods=1`` 时窗口只含当前值, 名次恒为 1, 与 ctx 无关::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.RANK(x, 1), 3)
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
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


def _rank_ref(x, periods, *, strict):
  """无 nan 数据上的 RANK 参考实现 (窗口内小于当前值的个数 + 1)."""
  x = np.asarray(x, dtype=np.float64)
  assert not np.isnan(x).any()
  out = np.full(len(x), np.nan)
  if periods == 0:
    return out

  for i in range(len(x)):
    win = x[max(0, i - periods + 1) : i + 1]
    if strict and win.size < periods:
      continue
    out[i] = 1.0 + np.count_nonzero(win < x[i])
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


def _compact_ref(x, periods, strict):
  """把有效值压紧后跑参考实现, 再映射回有效位置; nan 位置保持 nan."""
  out = np.full(len(x), np.nan)
  valid = ~np.isnan(x)
  if valid.any():
    out[valid] = _rank_ref(x[valid], periods, strict=strict)
  return out


def _expected(x, periods, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 全量计算参考值, 再只保留 alpha 应计算的 [start, end) 区间.

  无 SKIP_NAN 时窗口可以引用 start 之前的数据, 但窗口内有 nan 就让结果为
  nan; 有 SKIP_NAN 时, 只在 [start, end) 内压紧有效值, 不跨边界借用数据.
  """
  x = np.asarray(x, dtype=np.float64)
  if periods == 1:
    return np.ones_like(x)

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
    al.RANK(np.asarray(x, dtype=np.float64), periods), dtype=np.float64
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
def test_rank_matches_numpy(periods):
  """flags=0, groups=1: 每个窗口的 min-rank 都与逐窗口计数一致."""
  x = _clean_data()
  actual = _calc(x, periods, flags=0)
  _assert_close(actual, _expected(x, periods, flags=0))
  _assert_close(actual, _rank_ref(x, periods, strict=False))
  assert np.isfinite(actual).all()
  assert np.all((actual >= 1.0) & (actual <= periods))


@pytest.mark.parametrize("periods", [2, 3, 4, 5, 10, 20])
def test_rank_matches_pandas_rolling_rank(periods):
  """与 pandas rolling().rank(method="min") 独立校验."""
  x = _clean_data()
  actual = _calc(x, periods, flags=0)
  expected = (
    pd.Series(x)
    .rolling(periods, min_periods=1)
    .rank(method="min")
    .to_numpy(dtype=np.float64)
  )
  _assert_close(actual, expected)


def test_rank_ties_use_min_rank():
  """重复值取最小名次: 名次 = 1 + 严格小于当前值的元素个数."""
  x = np.array([3.0, 1.0, 3.0, 2.0, 3.0, 5.0, 5.0, 1.0, 4.0, 2.0])
  actual = _calc(x, 3, flags=0)
  _assert_close(actual, _rank_ref(x, 3, strict=False))
  _assert_close(actual[:6], np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0]))


@pytest.mark.parametrize("periods", [2, 3, 4, 5, 10, 20])
def test_rank_strictly_cycle_matches_pandas(periods):
  """FLAG_STRICTLY_CYCLE: 窗口未满输出 nan, 之后与 pandas rolling rank 一致."""
  x = _clean_data()
  flags = al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  _assert_close(actual, _rank_ref(x, periods, strict=True))
  _assert_close(
    actual,
    pd.Series(x)
    .rolling(periods, min_periods=periods)
    .rank(method="min")
    .to_numpy(dtype=np.float64),
  )
  assert np.isnan(actual[: periods - 1]).all()
  assert np.isfinite(actual[periods - 1 :]).all()


def test_rank_nan_position_is_nan():
  """nan 是缺失值: 当前值为 nan 时结果也应为 nan."""
  x = _nan_data()
  actual = _calc(x, 4, flags=0)
  assert np.isnan(actual[np.isnan(x)]).all()


def test_rank_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 有效位置应等于「未插入 nan」数据压紧后的参考结果."""
  x_clean = _clean_data()
  x = x_clean.copy()
  x[NAN_POS] = np.nan
  periods = 4
  actual = _calc(x, periods, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, periods, flags=al.FLAG_SKIP_NAN))

  valid = ~np.isnan(x)
  assert np.isnan(actual[~valid]).all()
  _assert_close(actual[valid], _rank_ref(x_clean[valid], periods, strict=False))


def test_rank_skip_nan_and_strictly_cycle():
  """两个 flag 组合: 要求连续无 nan 的完整窗口, 等价于 pandas rolling rank."""
  x = _nan_data()
  periods = 4
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
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
def test_rank_all_flag_combinations_with_nan(flags):
  """四种 flag 组合在含 nan 数据上都与参考实现一致."""
  x = _nan_data()
  periods = 6
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_rank_groups(groups, flags):
  """group-major 分块: 每个 chunk 独立计算, 开头重新 warm-up."""
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
      _rank_ref(chunk, periods, strict=bool(flags & al.FLAG_STRICTLY_CYCLE)),
    )


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_rank_skip_nan_within_each_group(groups):
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
    _assert_close(got[valid], _rank_ref(chunk[valid], periods, strict=False))


@pytest.mark.parametrize(
  "start,end",
  [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5), (-10, 0), (40, 20)],
)
def test_rank_start_end(start, end):
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
def test_rank_start_end_with_groups(groups):
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


def test_rank_skip_nan_with_start_end():
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
    _rank_ref(x_clean[10:90][valid], periods, strict=False),
  )


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_rank_skip_nan_start_end_with_groups(groups):
  """SKIP_NAN + 负数 end + groups: 每个 chunk 内独立解析区间并压紧."""
  x = _nan_data()
  periods = 3
  ctx = dict(start=5, end=-5, groups=groups, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, periods, **ctx)
  _assert_close(actual, _expected(x, periods, **ctx))


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_rank_skip_nan_strict_start_end_with_groups(groups):
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


def test_rank_edge_cases():
  """periods=1 恒为 1; 窗口长于数据时非严格模式按已有值排名."""
  x = _clean_data()
  for ctx in (
    {},
    dict(start=10, end=50, groups=3, flags=al.FLAG_SKIP_NAN),
  ):
    _assert_close(_calc(x, 1, **ctx), np.ones_like(x))

  periods = len(x) + 1
  _assert_close(_calc(x, periods, flags=0), _expected(x, periods, flags=0))
  assert np.isnan(_calc(x, periods, flags=al.FLAG_STRICTLY_CYCLE)).all()
