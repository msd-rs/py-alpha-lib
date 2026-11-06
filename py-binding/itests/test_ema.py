# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""EMA (指数移动平均) 集成测试.

算法简介
--------
EMA 是递推型平均, 权重 ``w = 2 / (periods + 1)``::

    r[0] = x[0]
    r[i] = w * x[i] + (1 - w) * r[i - 1]

``FLAG_STRICTLY_CYCLE`` 要求窗口未满时不输出: 前 ``periods - 1`` 个位置为
nan, 且 warm-up 期间不做递推, 只把当前原始值作为后续递推的种子, 因此
``EMA(3)`` 在严格周期下第一个输出是 ``0.5 * 3 + 0.5 * 2 = 2.5`` (而不是
从头递推得到的 2.25). ``FLAG_SKIP_NAN`` 使 nan 完全不参与运算, 等价于把
有效值压紧 (compact) 后跑参考实现, 再映射回有效位置; nan 位置本身仍为 nan.

参考实现 ``_ema_ref`` 用 numpy 独立实现递推公式, 只描述算法逻辑, 不调用
alpha, 也不实现 Group / start-end 机制: Group 由测试按 chunk 分块计算,
start/end 只校验 ``[start, end)`` 区间的值. 不设置 ``FLAG_SKIP_NAN`` 时,
按递推公式 nan 会污染后续所有输出; 当前实现遇到 nan 后会重启递推, 这类
用例保留失败以暴露差异. 同样地, 当前实现从 ``start`` 处重启递推 (种子为
chunk 的第一个值) 且忽略 ``end``, 与「全量递推后只输出区间」的定义不同,
相关用例也保留失败.

`set_ctx` 只覆盖传入的参数, 因此每次调用前都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=120, suppress=True)
    >>> x = np.arange(1., 11.)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算 (``w = 0.5``, 从头递推)::

    >>> np.round(al.EMA(x, 3), 3)
    array([1.   , 1.5  , 2.25 , 3.125, 4.062, 5.031, 6.016, 7.008, 8.004, 9.002])

2. ``FLAG_STRICTLY_CYCLE``: 前 2 个位置为 nan, 递推以 warm-up 期的原始值
   为种子重新起步::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.EMA(x, 3), 3)
    array([  nan,   nan, 2.5  , 3.25 , 4.125, 5.062, 6.031, 7.016, 8.008, 9.004])

3. 不跳过 nan: 窗口中出现 nan 时该位置输出 nan::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.EMA(xn, 3), 3)
    array([1.  , 1.5 , 2.25,  nan, 5.  , 5.5 , 6.25, 7.125, 8.062, 9.031])

   按标准递推公式 nan 应污染后续所有值, 当前实现在 nan 之后重启了递推.

4. ``FLAG_SKIP_NAN``: 压紧后的有效值为 ``[1, 2, 3, 5, 6, 7, 8, 9, 10]``,
   其 EMA(3) 依次是 ``[1, 1.5, 2.25, 3.625, 4.8125, ...]``::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.EMA(xn, 3), 3)
    array([1.   , 1.5  , 2.25 ,   nan, 3.625, 4.812, 5.906, 6.953, 7.977, 8.988])

5. ``FLAG_SKIP_NAN | FLAG_STRICTLY_CYCLE``: 在压紧后的有效序列上做 warm-up::

    >>> al.set_ctx(start=0, end=0, groups=1,
    ...            flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.EMA(xn, 3), 3)
    array([  nan,   nan, 2.5  ,   nan, 3.75 , 4.875, 5.938, 6.969, 7.984, 8.992])

6. ``groups`` 把一维数据按 group-major 切块, 每块重新以该块首值起步::

    >>> y = np.arange(1., 13.)
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.EMA(y, 3), 3)
    array([ 1.   ,  1.5  ,  2.25 ,  3.125,  4.062,  5.031,
             7.   ,  7.5  ,  8.25 ,  9.125, 10.062, 11.031])

7. ``start`` / ``end``: 测试只校验 ``[start, end)`` 区间, 该区间之外当前
   实现并不会填 nan (递推从 start 重启且忽略 end)::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.EMA(x, 3)[3:8], 3)
    array([2.5  , 3.75 , 4.875, 5.938, 6.969])
"""

import numpy as np
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
NAN_POS = np.array([9, 26, 43, 57, 74, 88, 109])


# ---------------------------------------------------------------------------
# numpy 对比实现
# ---------------------------------------------------------------------------


def _ema_ref(x, periods, *, strict=False):
  """EMA 递推参考实现: r[0] = x[0], r[i] = w*x[i] + (1-w)*r[i-1].

  无 nan 数据上 ``strict=False`` 即标准 EMA; 允许含 nan 时 nan 会沿递推
  传播到后续所有位置. ``strict=True`` 时前 ``periods - 1`` 个位置输出 nan,
  且 warm-up 期间只把当前值作为递推种子 (严格周期定义).
  """
  x = np.asarray(x, dtype=np.float64)
  if len(x) == 0:
    return np.empty(0, dtype=np.float64)

  w = 2.0 / (periods + 1)
  out = np.empty(len(x), dtype=np.float64)
  prev = x[0]
  for i, val in enumerate(x):
    if strict and i < periods - 1:
      out[i] = np.nan
      prev = val
    else:
      prev = w * val + (1.0 - w) * prev
      out[i] = prev
  return out


def _compact_ref(x, periods, *, strict=False):
  """把有效值压紧后跑参考实现, 再映射回有效位置; nan 位置保持 nan."""
  x = np.asarray(x, dtype=np.float64)
  out = np.full(len(x), np.nan)
  valid = ~np.isnan(x)
  if valid.any():
    out[valid] = _ema_ref(x[valid], periods, strict=strict)
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


def _expected(x, periods, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 全量计算参考值, 只保留 [start, end) 区间.

  ``FLAG_SKIP_NAN`` 时在整个 chunk 上压紧有效值, nan 不参与递推; 其余情况
  直接按递推公式全量计算 (含 nan 会向后传播).
  """
  x = np.asarray(x, dtype=np.float64)
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  skip = bool(flags & al.FLAG_SKIP_NAN)
  parts = []
  for chunk in np.split(x, groups):
    n = len(chunk)
    s, e = _calc_range(n, start, end)
    if skip:
      ref = _compact_ref(chunk, periods, strict=strict)
    else:
      ref = _ema_ref(chunk, periods, strict=strict)
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
    al.EMA(np.asarray(x, dtype=np.float64), periods), dtype=np.float64
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


def _assert_range_close(actual, expected, *, start=0, end=0, groups=1):
  """递推型算子只校验 [start, end) 区间, 区间外的值未定义."""
  assert actual.shape == expected.shape
  for a_chunk, e_chunk in zip(
    np.split(actual, groups), np.split(expected, groups)
  ):
    s, e = _calc_range(len(a_chunk), start, end)
    np.testing.assert_allclose(
      a_chunk[s:e], e_chunk[s:e], atol=ATOL, rtol=RTOL, equal_nan=True
    )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("periods", [1, 2, 3, 4, 5, 10, 20])
def test_ema_matches_reference(periods):
  """flags=0, groups=1: 与 numpy 递推参考实现逐点比较."""
  x = _clean_data()
  actual = _calc(x, periods, flags=0)
  _assert_close(actual, _expected(x, periods, flags=0))
  _assert_close(actual, _ema_ref(x, periods))
  assert np.isfinite(actual).all()


@pytest.mark.parametrize("periods", [1, 2, 3, 4, 5, 10, 20])
def test_ema_strictly_cycle_matches_reference(periods):
  """FLAG_STRICTLY_CYCLE: 前 periods-1 个 nan, 之后按 warm-up 种子递推."""
  x = _clean_data()
  flags = al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  _assert_close(actual, _ema_ref(x, periods, strict=True))
  if periods > 1:
    assert np.isnan(actual[: periods - 1]).all()
    assert np.isfinite(actual[periods - 1 :]).all()


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_ema_nan_without_skip_propagates(flags):
  """无 SKIP_NAN: 按递推公式 nan 应污染后续所有输出."""
  x = _nan_data()
  periods = 4
  actual = _calc(x, periods, flags=flags)
  _assert_range_close(actual, _expected(x, periods, flags=flags))
  assert np.isnan(actual[np.isnan(x)]).all()


def test_ema_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 有效位置等于压紧有效值后的参考结果."""
  x = _nan_data()
  periods = 4
  actual = _calc(x, periods, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, periods, flags=al.FLAG_SKIP_NAN))

  valid = ~np.isnan(x)
  assert np.isnan(actual[~valid]).all()
  _assert_close(actual[valid], _ema_ref(x[valid], periods))
  assert np.isfinite(actual[valid]).all()


def test_ema_skip_nan_and_strictly_cycle():
  """两个 flag 组合: 在压紧后的有效序列上做 warm-up."""
  x = _nan_data()
  periods = 4
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
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
def test_ema_all_flag_combinations_with_nan(flags):
  """四种 flag 组合在含 nan 数据上都与参考实现比较."""
  x = _nan_data()
  periods = 6
  actual = _calc(x, periods, flags=flags)
  _assert_range_close(actual, _expected(x, periods, flags=flags))


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_ema_groups(groups, flags):
  """group-major 分块: 每个 chunk 独立递推, 重新以块首值起步."""
  x = _clean_data()
  periods = 4
  actual = _calc(x, periods, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, periods, groups=groups, flags=flags))

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    chunk = x[lo : lo + chunk_size]
    _assert_close(
      actual[lo : lo + chunk_size], _ema_ref(chunk, periods, strict=bool(flags))
    )


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
def test_ema_skip_nan_within_each_group(groups):
  """SKIP_NAN 的压紧过程按 chunk 独立进行, 不会跨 group 借用数据."""
  x = _nan_data()
  periods = 5
  flags = al.FLAG_SKIP_NAN
  actual = _calc(x, periods, groups=groups, flags=flags)
  _assert_close(
    actual, _expected(x, periods, groups=groups, flags=flags)
  )
  assert np.isnan(actual[np.isnan(x)]).all()


@pytest.mark.parametrize(
  "start,end", [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5)]
)
def test_ema_start_end(start, end):
  """start/end: 仅校验 [start, end) 区间, 参考值来自全量递推."""
  x = _clean_data()
  periods = 4
  actual = _calc(x, periods, start=start, end=end, flags=0)
  _assert_range_close(
    actual,
    _expected(x, periods, start=start, end=end, flags=0),
    start=start,
    end=end,
  )


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_ema_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  x = _clean_data()
  periods = 4
  start, end = 6, 25
  actual = _calc(
    x, periods, start=start, end=end, groups=groups, flags=0
  )
  _assert_range_close(
    actual,
    _expected(
      x, periods, start=start, end=end, groups=groups, flags=0
    ),
    start=start,
    end=end,
    groups=groups,
  )


def test_ema_skip_nan_with_start_end():
  """SKIP_NAN + start/end: nan 不参与 [start, end) 区间的递推."""
  x = _nan_data()
  periods = 4
  ctx = dict(start=10, end=90, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, periods, **ctx)
  _assert_range_close(
    actual, _expected(x, periods, **ctx), start=10, end=90
  )
  assert np.isnan(actual[np.isnan(x)]).all()


def test_ema_edge_cases():
  """periods=1 时 w=1, 输出等于输入; SKIP_NAN 下 nan 位置仍为 nan."""
  x = _clean_data()
  _assert_close(_calc(x, 1, flags=0), x)

  xn = _nan_data()
  actual = _calc(xn, 1, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _compact_ref(xn, 1))
