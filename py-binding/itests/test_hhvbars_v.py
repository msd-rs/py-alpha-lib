# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""HHVBARS_V / LLVBARS_V (逐点可变窗口的距极值 bar 数) 集成测试.

算法简介
--------
两个算子是 HHVBARS / LLVBARS 的变周期版本, 每个位置 ``i`` 用
``periods[i]`` 决定窗口长度, 返回当前位置距窗口内最高 / 最低值所在位置
有多少根 bar::

    q = periods[i]
    lo = 0                      (q == 0, 从 chunk 开头累计)
    lo = max(0, i + 1 - q)      (q > 0)
    HHVBARS_V[i] = i - argmax_last(x[lo : i + 1])
    LLVBARS_V[i] = i - argmin_last(x[lo : i + 1])

``argmax_last`` / ``argmin_last`` 表示并列极值时取最靠后的位置, 因此当前值
刷新极值时结果为 0. ``q = 0`` 是累计模式, 返回距最近一次刷新累计极值的
bar 数. 周期恒定时分别与 ``HHVBARS(x, q)`` / ``LLVBARS(x, q)`` 一致.

窗口未满 (``i < q - 1``) 时按已有的部分窗口给出结果; 设置
``FLAG_STRICTLY_CYCLE`` 后这些位置输出 nan (累计模式不受影响). 不设置
``FLAG_SKIP_NAN`` 时, 窗口内出现 nan 会使该位置输出 nan. 设置后 nan 不
参与极值选取, 窗口改为 ``[start, end)`` 内最近 ``q`` 个有效值, 但 bar
距离仍按原始位置计算 (nan 也占一根 bar), nan 位置本身输出 nan. 再组合
``FLAG_STRICTLY_CYCLE`` 时, 还要求这 ``q`` 个有效值恰好占据 ``q`` 个
连续原始位置 (窗口内不能有 nan 间隙).

``groups`` 按 group-major 切成等长 chunk, 每个 chunk 独立计算, ``periods``
数组同样按 chunk 切分. start/end 是每个 chunk 内的相对下标, 只计算
``[start, end)`` 区间: 普通模式下窗口可引用 start 之前的数据, SKIP_NAN
模式下有效值只在区间内收集; 区间外一律为 nan.

参考实现 ``_bars_v_ref`` / ``_bars_v_skip_ref`` 用 numpy 独立实现, 只描述
算法逻辑, 不调用 alpha, 也不实现 SKIP_NAN / Group / start-end 机制; 这些
上下文语义由 ``_expected`` 单独描述. 固定周期场景另外用 pandas
``rolling`` 配合 ``argmax`` 做交叉校验. ``set_ctx`` 只覆盖传入的参数,
因此每次调用都显式重置 start/end/groups/flags. 浮点比较容差 1e-6.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> x = np.array([5., 1., 2., 3., 4., 6., 8., 7., 9., 10.])
    >>> p = np.array([1, 2, 3, 4, 5, 2, 3, 1, 2, 4], dtype=np.uint64)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算, 每个位置使用自己的窗口长度::

    >>> al.HHVBARS_V(x, p).tolist()
    [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    >>> al.LLVBARS_V(x, p).tolist()
    [0.0, 0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 0.0, 1.0, 2.0]

   i=4 时窗口为 ``[5, 1, 2, 3, 4]``, 最高值 5 在 4 根 bar 之前,
   最低值 1 在 3 根 bar 之前.

2. ``FLAG_STRICTLY_CYCLE``: 固定周期 3 时窗口未满的位置输出 nan;
   ``p = 0`` 为累计模式::

    >>> y = np.array([2., 1., 2., 2., 1., 2.])
    >>> p3 = np.full(6, 3, dtype=np.uint64)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.HHVBARS_V(y, p3), 3).tolist()
    [nan, nan, 0.0, 0.0, 1.0, 0.0]
    >>> np.round(al.LLVBARS_V(y, p3), 3).tolist()
    [nan, nan, 1.0, 2.0, 0.0, 1.0]
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> al.HHVBARS_V(y, np.zeros(6, dtype=np.uint64)).tolist()
    [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]

3. 不跳过 nan: 窗口内出现 nan 的位置输出 nan::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> np.round(al.HHVBARS_V(xn, p), 3).tolist()
    [0.0, 1.0, 2.0, nan, nan, 0.0, 0.0, 0.0, 0.0, 0.0]

4. ``FLAG_SKIP_NAN``: nan 不参与极值选取, 但距离仍按原始位置计算::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.HHVBARS_V(xn, p), 3).tolist()
    [0.0, 1.0, 2.0, nan, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    >>> np.round(al.LLVBARS_V(xn, p), 3).tolist()
    [0.0, 0.0, 1.0, nan, 3.0, 1.0, 2.0, 0.0, 1.0, 2.0]

   ``FLAG_SKIP_NAN | FLAG_STRICTLY_CYCLE`` 额外要求窗口内无 nan 间隙,
   i=4 重新变为 nan::

    >>> al.set_ctx(start=0, end=0, groups=1,
    ...            flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.HHVBARS_V(xn, p), 3).tolist()
    [0.0, 1.0, 2.0, nan, nan, 0.0, 0.0, 0.0, 0.0, 0.0]

5. ``groups`` 把数据按 group-major 切块, 每块窗口独立::

    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> al.LLVBARS_V(x, p).tolist()
    [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0]

6. ``start`` / ``end`` 只输出 ``[start, end)`` 区间, 区间外全部为 nan::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.HHVBARS_V(x, p), 3).tolist()
    [nan, nan, nan, 3.0, 4.0, 0.0, 0.0, 0.0, nan, nan]
"""

import numpy as np
import pandas as pd
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
NAN_POS = np.array([0, 3, 17, 18, 29, 44, 52, 77, 101])
GROUPS = [1, 2, 3, 4, 6]
START_END = [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5), (-10, 0), (40, 20)]
FLAG_COMBOS = [
  0,
  al.FLAG_STRICTLY_CYCLE,
  al.FLAG_SKIP_NAN,
  al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
]
OPS = ["max", "min"]


# ---------------------------------------------------------------------------
# numpy / pandas 对比实现
# ---------------------------------------------------------------------------


def _last_extremum_index(window, op):
  """并列极值时取最靠后的位置 (反转后用 argmax)."""
  reversed_vals = window[::-1] if op == "max" else -window[::-1]
  return len(window) - 1 - int(np.argmax(reversed_vals))


def _bars_v_ref(x, periods, op, *, strict=False):
  """按算法逻辑逐点取窗口极值位置的 bar 距离, 不依赖 alpha.

  窗口为 ``x[lo : i + 1]``, ``q = 0`` 时从序列开头累计; 窗口内含 nan
  则输出 nan; ``strict`` 时窗口未满 ``q`` 个值的位置输出 nan.
  """
  x = np.asarray(x, dtype=np.float64)
  p = np.asarray(periods, dtype=np.int64)
  out = np.full(len(x), np.nan)
  for i in range(len(x)):
    q = int(p[i])
    if strict and q > 0 and i < q - 1:
      continue
    lo = 0 if q == 0 else max(0, i + 1 - q)
    window = x[lo : i + 1]
    if np.isnan(window).any():
      continue
    out[i] = i - (lo + _last_extremum_index(window, op))
  return out


def _bars_v_skip_ref(seg, periods, op, *, strict=False):
  """SKIP_NAN 参考实现: 极值取自最近 q 个有效值, 距离按原始位置计算.

  有效值不足 ``q`` 个时用已有的部分窗口; ``strict`` 还要求这 ``q`` 个
  有效值占据 ``q`` 个连续原始位置 (窗口内不能有 nan 间隙).
  """
  seg = np.asarray(seg, dtype=np.float64)
  p = np.asarray(periods, dtype=np.int64)
  out = np.full(len(seg), np.nan)
  vals = []
  idxs = []
  for i in range(len(seg)):
    if np.isnan(seg[i]):
      continue
    vals.append(seg[i])
    idxs.append(i)
    q = int(p[i])
    l = len(vals)
    start_hist = 0 if q == 0 else max(0, l - q)
    if strict and q > 0 and (
      l < q or i - idxs[start_hist] + 1 != q
    ):
      continue
    window = np.asarray(vals[start_hist:], dtype=np.float64)
    best = idxs[start_hist + _last_extremum_index(window, op)]
    out[i] = i - best
  return out


def _bars_pandas_ref(x, periods, op, *, strict=False):
  """固定周期下用 pandas rolling + numpy argmax 做独立交叉校验."""
  min_periods = periods if strict else 1
  return (
    pd.Series(np.asarray(x, dtype=np.float64))
    .rolling(periods, min_periods=min_periods)
    .apply(
      lambda w: float(np.argmax(w[::-1] if op == "max" else -w[::-1])),
      raw=True,
    )
    .to_numpy(dtype=np.float64)
  )


def _calc_range(n, start, end):
  """对应 Rust ``Context::start`` / ``Context::end`` 在单个 chunk 内的解析."""
  if n == 0:
    return 0, 0
  s = min(n - 1, start) if start >= 0 else max(0, n + start)
  if end == 0:
    e = n
  elif end > 0:
    e = min(n, end)
  else:
    e = max(0, n + end)
  return s, e


def _expected(x, periods, op, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 计算参考值, 只保留 [start, end) 区间.

  ``FLAG_SKIP_NAN`` 时只在 ``[s, e)`` 段内收集有效值 (历史不跨 start);
  普通模式全量计算窗口, 再把区间外填 nan.
  """
  x = np.asarray(x, dtype=np.float64)
  p = np.asarray(periods, dtype=np.int64)
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  skip = bool(flags & al.FLAG_SKIP_NAN)
  parts = []
  for chunk_x, chunk_p in zip(np.split(x, groups), np.split(p, groups)):
    n = len(chunk_x)
    s, e = _calc_range(n, start, end)
    ref = np.full(n, np.nan)
    if skip:
      ref[s:e] = _bars_v_skip_ref(
        chunk_x[s:e], chunk_p[s:e], op, strict=strict
      )
    else:
      ref = _bars_v_ref(chunk_x, chunk_p, op, strict=strict)
      ref[:s] = np.nan
      ref[e:] = np.nan
    parts.append(ref)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(x, periods, op, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  fn = al.HHVBARS_V if op == "max" else al.LLVBARS_V
  return np.asarray(
    fn(
      np.asarray(x, dtype=np.float64),
      np.asarray(periods, dtype=np.uint64),
    ),
    dtype=np.float64,
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261112)
  return rng.normal(loc=10.0, scale=2.0, size=n).astype(np.float64)


def _nan_data(n=N):
  x = _clean_data(n)
  x[NAN_POS[NAN_POS < n]] = np.nan
  return x


def _periods_data(n=N, max_p=12):
  rng = np.random.default_rng(20261113)
  return rng.integers(0, max_p + 1, size=n).astype(np.uint64)


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_hhvbars_v_matches_reference(op, flags):
  """变周期数据在无 nan 时与 numpy 逐点窗口参考实现比较."""
  x = _clean_data()
  p = _periods_data()
  actual = _calc(x, p, op, flags=flags)
  _assert_close(actual, _expected(x, p, op, flags=flags))
  _assert_close(
    actual,
    _bars_v_ref(
      x, p, op, strict=bool(flags & al.FLAG_STRICTLY_CYCLE)
    ),
  )
  finite = np.isfinite(actual)
  assert (actual[finite] >= 0).all()
  assert (actual[finite] <= np.arange(len(x))[finite]).all()


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("periods", [1, 2, 3, 4, 5, 10, 20])
@pytest.mark.parametrize("strict", [False, True])
def test_hhvbars_v_constant_periods_matches_pandas(op, periods, strict):
  """固定周期时与 pandas rolling + argmax 的参考实现比较."""
  x = _clean_data()
  p = np.full(len(x), periods, dtype=np.uint64)
  flags = al.FLAG_STRICTLY_CYCLE if strict else 0
  actual = _calc(x, p, op, flags=flags)
  _assert_close(actual, _bars_pandas_ref(x, periods, op, strict=strict))
  _assert_close(actual, _expected(x, p, op, flags=flags))
  if strict and periods > 1:
    assert np.isnan(actual[: periods - 1]).all()
    assert np.isfinite(actual[periods - 1 :]).all()


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("periods", [0, 1, 2, 3, 5, 10])
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_hhvbars_v_constant_periods_equals_hhvbars(op, periods, flags):
  """周期恒定时与固定周期版 HHVBARS / LLVBARS 在四种 flag 下都一致."""
  x = _clean_data()
  p = np.full(len(x), periods, dtype=np.uint64)
  al.set_ctx(start=0, end=0, groups=1, flags=flags)
  fn = al.HHVBARS if op == "max" else al.LLVBARS
  expected = np.asarray(fn(x, periods), dtype=np.float64)
  _assert_close(_calc(x, p, op, flags=flags), expected)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_hhvbars_v_cumulative_mode(op, flags):
  """p=0 累计模式: 距最近一次刷新累计极值的 bar 数."""
  x = _nan_data()
  p = np.zeros(len(x), dtype=np.uint64)
  actual = _calc(x, p, op, flags=flags)
  _assert_close(actual, _expected(x, p, op, flags=flags))
  if flags & al.FLAG_SKIP_NAN:
    assert np.isnan(actual[np.isnan(x)]).all()

  xs = np.array([2.0, 1.0, 2.0, 2.0, 1.0, 2.0])
  ps = np.zeros(len(xs), dtype=np.uint64)
  if op == "max":
    expected = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
  else:
    expected = [0.0, 0.0, 1.0, 2.0, 0.0, 1.0]
  _assert_close(_calc(xs, ps, op, flags=0), np.asarray(expected))


def test_hhvbars_v_ties_take_latest():
  """并列极值时取最靠后的位置: 常数序列上所有结果都是 0."""
  const = np.full(20, 3.0)
  p = _periods_data(20)
  for op in OPS:
    _assert_close(_calc(const, p, op, flags=0), np.zeros(20))


@pytest.mark.parametrize("op", OPS)
def test_hhvbars_v_strictly_cycle(op):
  """FLAG_STRICTLY_CYCLE: 窗口未满的位置输出 nan."""
  x = np.array([2.0, 1.0, 2.0, 2.0, 1.0, 2.0])
  p = np.full(len(x), 3, dtype=np.uint64)
  actual = _calc(x, p, op, flags=al.FLAG_STRICTLY_CYCLE)
  _assert_close(
    actual, _expected(x, p, op, flags=al.FLAG_STRICTLY_CYCLE)
  )
  assert np.isnan(actual[:2]).all()
  assert np.isfinite(actual[2:]).all()
  _assert_close(
    actual[2:],
    _calc(x, p, op, flags=0)[2:],
  )


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_hhvbars_v_nan_without_skip_pollutes_window(op, flags):
  """无 SKIP_NAN: 窗口内任一个 nan 都会让该位置输出 nan."""
  x = _nan_data()
  p = _periods_data()
  actual = _calc(x, p, op, flags=flags)
  _assert_close(actual, _expected(x, p, op, flags=flags))
  assert np.isnan(actual[np.isnan(x)]).all()

  for i in range(len(x)):
    q = int(p[i])
    lo = 0 if q == 0 else max(0, i + 1 - q)
    if np.isnan(x[lo : i + 1]).any():
      assert np.isnan(actual[i])


@pytest.mark.parametrize("op", OPS)
def test_hhvbars_v_skip_nan_ignores_nan_values(op):
  """FLAG_SKIP_NAN: nan 位置为 nan, 有效位置与参考实现一致."""
  x = _nan_data()
  p = _periods_data()
  actual = _calc(x, p, op, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, p, op, flags=al.FLAG_SKIP_NAN))
  assert np.isnan(actual[np.isnan(x)]).all()

  valid = ~np.isnan(x)
  assert np.isfinite(actual[valid]).sum() > 0
  _assert_close(
    actual[valid], _bars_v_skip_ref(x, p, op)[valid]
  )


@pytest.mark.parametrize("op", OPS)
def test_hhvbars_v_skip_nan_and_strictly_cycle(op):
  """两个 flag 组合: 严格模式要求窗口内无 nan 间隙."""
  x = _nan_data()
  p = _periods_data()
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, p, op, flags=flags)
  _assert_close(actual, _expected(x, p, op, flags=flags))
  assert np.isnan(actual[np.isnan(x)]).all()

  loose = _calc(x, p, op, flags=al.FLAG_SKIP_NAN)
  finite = np.isfinite(actual)
  _assert_close(actual[finite], loose[finite])
  assert finite.sum() < np.isfinite(loose).sum()


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_hhvbars_v_all_flag_combinations_with_nan(op, flags):
  """四种 flag 组合在含 nan 数据上都与参考实现比较."""
  x = _nan_data()
  p = _periods_data()
  actual = _calc(x, p, op, flags=flags)
  _assert_close(actual, _expected(x, p, op, flags=flags))


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_hhvbars_v_groups(op, groups, flags):
  """group-major 分块: 每个 chunk 独立计算, periods 也按块切分."""
  x = _clean_data()
  p = _periods_data()
  actual = _calc(x, p, op, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, p, op, groups=groups, flags=flags))

  chunk_size = len(x) // groups
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  for g in range(groups):
    lo = g * chunk_size
    _assert_close(
      actual[lo : lo + chunk_size],
      _bars_v_ref(
        x[lo : lo + chunk_size],
        p[lo : lo + chunk_size],
        op,
        strict=strict,
      ),
    )


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("groups", GROUPS)
def test_hhvbars_v_skip_nan_within_each_group(op, groups):
  """SKIP_NAN 的有效值收集按 chunk 独立进行, 不会跨 group 借用数据."""
  x = _nan_data()
  p = _periods_data()
  flags = al.FLAG_SKIP_NAN
  actual = _calc(x, p, op, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, p, op, groups=groups, flags=flags))
  assert np.isnan(actual[np.isnan(x)]).all()


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("start,end", START_END)
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_hhvbars_v_start_end(op, start, end, flags):
  """start/end: 只校验 [start, end) 区间, 区间外全部为 nan."""
  x = _nan_data()
  p = _periods_data()
  actual = _calc(x, p, op, start=start, end=end, flags=flags)
  _assert_close(
    actual, _expected(x, p, op, start=start, end=end, flags=flags)
  )
  s, e = _calc_range(len(x), start, end)
  assert np.isnan(actual[:s]).all()
  assert np.isnan(actual[e:]).all()


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("groups", GROUPS)
def test_hhvbars_v_start_end_with_groups(op, groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  x = _clean_data()
  p = _periods_data()
  start, end = 6, 25
  actual = _calc(
    x, p, op, start=start, end=end, groups=groups, flags=0
  )
  _assert_close(
    actual,
    _expected(x, p, op, start=start, end=end, groups=groups, flags=0),
  )

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


@pytest.mark.parametrize("op", OPS)
def test_hhvbars_v_skip_nan_with_start_end(op):
  """SKIP_NAN + start/end: 只有区间内的有效值参与极值选取."""
  x = _nan_data()
  p = _periods_data()
  ctx = dict(start=10, end=90, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, p, op, **ctx)
  _assert_close(actual, _expected(x, p, op, **ctx))
  assert np.isnan(actual[:10]).all()
  assert np.isnan(actual[90:]).all()
  assert np.isnan(actual[np.isnan(x)]).all()


@pytest.mark.parametrize("op", OPS)
def test_hhvbars_v_edge_cases(op):
  """空数组 / 单元素 / 全 nan / p=1 / 长度不匹配报错."""
  empty = np.empty(0, dtype=np.float64)
  empty_p = np.empty(0, dtype=np.uint64)
  _assert_close(_calc(empty, empty_p, op, flags=0), empty)
  _assert_close(
    _calc(empty, empty_p, op, flags=al.FLAG_SKIP_NAN), empty
  )

  one = np.array([3.0])
  _assert_close(
    _calc(one, np.array([1], dtype=np.uint64), op, flags=0),
    np.zeros(1),
  )

  all_nan = np.full(10, np.nan)
  assert np.isnan(
    _calc(all_nan, _periods_data(10), op, flags=al.FLAG_SKIP_NAN)
  ).all()

  x = _clean_data(20)
  p1 = np.ones(20, dtype=np.uint64)
  _assert_close(_calc(x, p1, op, flags=0), np.zeros(20))

  with pytest.raises(ValueError):
    _calc(x, _periods_data(10), op, flags=0)
