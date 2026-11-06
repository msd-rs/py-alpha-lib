# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""LLV (Lowest Low Value, 滚动最小值) 集成测试.

算法简介
--------
LLV 取最近 `periods` 个值中的最小值::

    LLV[i] = min(x[i - periods + 1 : i + 1])

`periods=0` 是特例: 从头开始的累积最小值. 不设置 ``FLAG_SKIP_NAN`` 时,
累积窗口一旦出现 nan, 该位置及之后都输出 nan; 设置后 nan 被忽略, nan
位置沿用之前的累积最小值.

与 HHV 对称, ``flags=0`` 时窗口未满也给出已有 1..periods 个值的部分最小值
(warm-up); 设置 ``FLAG_STRICTLY_CYCLE`` 后窗口未满输出 nan, 与 pandas
``rolling(periods, min_periods=periods).min()`` 一致. 参考实现 ``_llv_ref``
只在无 nan 数据上用 pandas ``rolling`` 计算, ``_llv_cum_ref`` 用 numpy
``minimum.accumulate`` 计算累积最小值; 两者都不实现 SKIP_NAN / Group /
start-end 机制.

不设置 ``FLAG_SKIP_NAN`` 时, 窗口内出现 nan 会使该窗口结果为 nan; 设置后
nan 不参与运算, 等价于把有效值压紧 (compact) 后跑参考实现, 再映射回有效
位置 (nan 位置本身仍为 nan). 再组合 ``FLAG_STRICTLY_CYCLE`` 时, 还要求最近
`periods` 个原始位置全部有效.

测试里 Group 通过按 chunk 分块计算实现, start/end 只校验 [start, end) 区间,
区间外必须是 nan; 窗口本身仍可引用 start 之前的数据 (SKIP_NAN 和 periods=0
除外, 它们只在 [start, end) 内计算). ``set_ctx`` 只覆盖传入的参数, 所以每次
调用都显式重置 start/end/groups/flags. 浮点比较容差 1e-6.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=150)
    >>> x = np.array([1., 3., 2., 5., 4., 6., 8., 7., 9., 10.])
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算 (窗口未满输出部分最小值)::

    >>> np.round(al.LLV(x, 3), 3)
    array([1., 1., 1., 2., 2., 4., 4., 6., 7., 7.])

   前三个输出依次为 1, min(1, 3)=1, min(1, 3, 2)=1.

2. ``FLAG_STRICTLY_CYCLE`` (窗口未满输出 nan)::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.LLV(x, 3), 3)
    array([nan, nan,  1.,  2.,  2.,  4.,  4.,  6.,  7.,  7.])

3. 不跳过 nan: 窗口内任一 nan 都会污染整个窗口::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.LLV(xn, 3), 3)
    array([ 1.,  1.,  1., nan, nan, nan,  4.,  6.,  7.,  7.])

4. ``FLAG_SKIP_NAN``: nan 完全不参与运算. 压紧后的有效值为
   [1, 3, 2, 4, 6, 8, 7, 9, 10], 其 3 周期滚动最小值依次是
   [1, 1, 1, 2, 2, 4, 6, 7, 7]::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.LLV(xn, 3), 3)
    array([ 1.,  1.,  1., nan,  2.,  2.,  4.,  6.,  7.,  7.])

5. ``FLAG_SKIP_NAN | FLAG_STRICTLY_CYCLE``: 要求连续 `periods` 个原始位置
   全部有效. i=2 的窗口 [1, 3, 2] 完整, i=3..5 的窗口含 nan, i=6 的窗口
   [4, 6, 8] 才重新完整::

    >>> al.set_ctx(start=0, end=0, groups=1,
    ...            flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.LLV(xn, 3), 3)
    array([nan, nan,  1., nan, nan, nan,  4.,  6.,  7.,  7.])

6. ``groups`` 把一维数据按 group-major 切块独立计算, 每块重新 warm-up::

    >>> y = np.array([3., 1., 4., 1., 5., 9., 2., 6., 5., 3., 5., 8.])
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.LLV(y, 3), 3)
    array([3., 1., 1., 1., 1., 1., 2., 2., 2., 3., 3., 3.])

   第二块从 2 重新开始: [2, 2, 2, 3, 3, 3].

7. ``start`` / ``end`` 只计算 [start, end) 区间, 区间外全部为 nan,
   但窗口仍可引用 start 之前的数据::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.LLV(x, 3), 3)
    array([nan, nan, nan,  2.,  2.,  4.,  4.,  6., nan, nan])

8. ``periods=0`` 计算累积最小值; 不跳过 nan 时, 第一个 nan 之后全部为 nan::

    >>> z = np.array([5., 4., 6., 3., 7., 2., 8., 1., 9., 10.])
    >>> zn = z.copy(); zn[3] = np.nan
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.LLV(z, 0), 3)
    array([5., 4., 4., 3., 3., 2., 2., 1., 1., 1.])
    >>> np.round(al.LLV(zn, 0), 3)
    array([ 5.,  4.,  4., nan, nan, nan, nan, nan, nan, nan])
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


def _llv_ref(x, periods, *, strict):
  """无 nan 数据上的滚动最小值参考实现 (pandas rolling min)."""
  x = np.asarray(x, dtype=np.float64)
  assert not np.isnan(x).any()
  assert periods > 0
  min_periods = periods if strict else 1
  return (
    pd.Series(x)
    .rolling(periods, min_periods=min_periods)
    .min()
    .to_numpy(dtype=np.float64)
  )


def _llv_cum_ref(x, *, skip):
  """累积最小值参考实现.

  不跳过 nan 时, 累积窗口含 nan 即无法定义, 首个 nan 及其后均为 nan;
  跳过 nan 时, nan 位置沿用之前的累积最小值, 首个有效值之前为 nan.
  """
  x = np.asarray(x, dtype=np.float64)
  if skip:
    out = np.minimum.accumulate(np.where(np.isnan(x), np.inf, x))
    out[np.isposinf(out)] = np.nan
    return out
  clean_prefix = np.cumsum(np.isnan(x).astype(np.int64)) == 0
  out = np.full(len(x), np.nan)
  if clean_prefix.any():
    out[clean_prefix] = np.minimum.accumulate(x[clean_prefix])
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
    out[valid] = _llv_ref(x[valid], periods, strict=strict)
  return out


def _expected(x, periods, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 全量计算参考值, 再只保留 alpha 应计算的 [start, end) 区间.

  ``periods=0`` 在每个 chunk 的 [start, end) 段上独立累积; 无 SKIP_NAN 时,
  滚动窗口可以引用 start 之前的数据, 但窗口内有 nan 就让结果为 nan; 有
  SKIP_NAN 时, 只在 [start, end) 内压紧有效值, 不跨边界借用数据.
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
      if periods == 0:
        ref[s:e] = _llv_cum_ref(chunk[s:e], skip=skip)
      elif skip:
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
    al.LLV(np.asarray(x, dtype=np.float64), periods), dtype=np.float64
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
def test_llv_matches_pandas_rolling(periods):
  """flags=0, groups=1: 与 pandas rolling(min_periods=1) 逐点比较."""
  x = _clean_data()
  actual = _calc(x, periods, flags=0)
  _assert_close(actual, _expected(x, periods, flags=0))
  _assert_close(actual, _llv_ref(x, periods, strict=False))
  assert np.isfinite(actual).all()


@pytest.mark.parametrize("periods", [2, 3, 4, 5, 10, 20])
def test_llv_strictly_cycle_matches_pandas(periods):
  """FLAG_STRICTLY_CYCLE: 窗口未满输出 nan, 之后与 pandas rolling 一致."""
  x = _clean_data()
  flags = al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  _assert_close(actual, _llv_ref(x, periods, strict=True))
  assert np.isnan(actual[: periods - 1]).all()
  assert np.isfinite(actual[periods - 1 :]).all()


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_llv_nan_without_skip_propagates(flags):
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


def test_llv_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 有效位置等于「未插入 nan」数据压紧后的参考结果."""
  x_clean = _clean_data()
  x = x_clean.copy()
  x[NAN_POS] = np.nan
  periods = 4
  actual = _calc(x, periods, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, periods, flags=al.FLAG_SKIP_NAN))

  valid = ~np.isnan(x)
  assert np.isnan(actual[~valid]).all()
  _assert_close(actual[valid], _llv_ref(x_clean[valid], periods, strict=False))


def test_llv_skip_nan_on_clean_data():
  """无 nan 数据上 SKIP_NAN 不应改变任何结果."""
  x = _clean_data()
  periods = 5
  for flags in (0, al.FLAG_SKIP_NAN):
    _assert_close(
      _calc(x, periods, flags=flags), _llv_ref(x, periods, strict=False)
    )
  for flags in (
    al.FLAG_STRICTLY_CYCLE,
    al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
  ):
    _assert_close(
      _calc(x, periods, flags=flags), _llv_ref(x, periods, strict=True)
    )


def test_llv_skip_nan_and_strictly_cycle():
  """两个 flag 组合: 要求连续无 nan 的完整窗口, 等价于 pandas 原始 rolling."""
  x = _nan_data()
  periods = 4
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  _assert_close(
    actual,
    pd.Series(x).rolling(periods, min_periods=periods).min().to_numpy(),
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
def test_llv_all_flag_combinations_with_nan(flags):
  """四种 flag 组合在含 nan 数据上都与参考实现一致."""
  x = _nan_data()
  periods = 6
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_llv_groups(groups, flags):
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
        _llv_ref(chunk, periods, strict=False)[: periods - 1],
      )


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
def test_llv_skip_nan_within_each_group(groups):
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
    _assert_close(got[valid], _llv_ref(chunk[valid], periods, strict=False))


@pytest.mark.parametrize(
  "start,end",
  [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5), (-10, 0), (40, 20)],
)
def test_llv_start_end(start, end):
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
def test_llv_start_end_with_groups(groups):
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


def test_llv_skip_nan_with_start_end():
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
    _llv_ref(x_clean[10:90][valid], periods, strict=False),
  )


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_llv_skip_nan_start_end_with_groups(groups):
  """SKIP_NAN + 负数 end + groups: 每个 chunk 内独立解析区间并压紧."""
  x = _nan_data()
  periods = 3
  ctx = dict(start=5, end=-5, groups=groups, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, periods, **ctx)
  _assert_close(actual, _expected(x, periods, **ctx))


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_llv_skip_nan_strict_start_end_with_groups(groups):
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
def test_llv_periods_zero_cumulative_min(flags):
  """periods=0: 从头累积最小值, 不受 flags 影响 (无 nan 数据)."""
  x = _clean_data()
  actual = _calc(x, 0, flags=flags)
  _assert_close(actual, _expected(x, 0, flags=flags))
  _assert_close(actual, _llv_cum_ref(x, skip=False))
  _assert_close(actual, pd.Series(x).expanding(min_periods=1).min().to_numpy())
  assert np.isfinite(actual).all()


def test_llv_periods_zero_nan_poison_without_skip():
  """periods=0 且不跳过 nan: 首个 nan 及其后全部输出 nan."""
  x = _clean_data().copy()
  x[7] = np.nan
  actual = _calc(x, 0, flags=0)
  _assert_close(actual, _expected(x, 0, flags=0))
  assert np.isfinite(actual[:7]).all()
  assert np.isnan(actual[7:]).all()
  _assert_close(actual[:7], _llv_cum_ref(x[:7], skip=False))


def test_llv_periods_zero_skip_nan():
  """periods=0 且跳过 nan: 有效值压紧后累积, nan 位置沿用之前的累积值."""
  x = _nan_data()
  actual = _calc(x, 0, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, 0, flags=al.FLAG_SKIP_NAN))
  _assert_close(actual, _llv_cum_ref(x, skip=True))
  assert np.isnan(actual[:2]).all()

  valid = ~np.isnan(x)
  _assert_close(
    actual[valid],
    pd.Series(x[valid]).expanding(min_periods=1).min().to_numpy(),
  )


@pytest.mark.parametrize("groups", [1, 2, 3, 4])
def test_llv_periods_zero_groups_and_range(groups):
  """periods=0: 每个 chunk 在 [start, end) 内从零开始独立累积."""
  x = _clean_data()
  ctx = dict(start=7, end=-9, groups=groups, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, 0, **ctx)
  _assert_close(actual, _expected(x, 0, **ctx))

  chunk_size = len(x) // groups
  e = chunk_size - 9
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + 7]).all()
    assert np.isnan(actual[lo + e : lo + chunk_size]).all()
    _assert_close(
      actual[lo + 7 : lo + e], _llv_cum_ref(x[lo + 7 : lo + e], skip=True)
    )


def test_llv_repeated_values():
  """重复值 / 单调序列上, 最小值仍与 pandas rolling 一致."""
  x = np.repeat(np.array([1.0, 3.0, 2.0, 3.0, 5.0]), 7)
  x = np.concatenate([x, np.arange(20.0, 30.0)])
  for periods in (1, 3, 5, 9):
    for flags in (0, al.FLAG_STRICTLY_CYCLE, al.FLAG_SKIP_NAN):
      actual = _calc(x, periods, flags=flags)
      _assert_close(actual, _expected(x, periods, flags=flags))


def test_llv_edge_cases():
  """periods=1 逐点返回输入; 窗口长于数据时非严格模式为扩展最小值."""
  x = _nan_data()
  _assert_close(_calc(x, 1, flags=0), x)
  _assert_close(_calc(x, 1, flags=0), _expected(x, 1, flags=0))

  periods = len(x) + 1
  _assert_close(_calc(x, periods, flags=0), _expected(x, periods, flags=0))
  assert np.isnan(_calc(x, periods, flags=al.FLAG_STRICTLY_CYCLE)).all()
  assert np.isnan(
    _calc(x, periods, flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
  ).all()

  actual = _calc(x, periods, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, periods, flags=al.FLAG_SKIP_NAN))
  valid = ~np.isnan(x)
  _assert_close(actual[valid], _llv_ref(x[valid], periods, strict=False))
