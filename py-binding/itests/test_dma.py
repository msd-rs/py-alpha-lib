# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""DMA (动态移动平均) 集成测试.

算法简介
--------
DMA 是权重由调用方直接给出的递推平均, 要求 ``0 <= weight <= 1``::

    r[0] = x[0]
    r[i] = weight * x[i] + (1 - weight) * r[i - 1]

``weight = 1`` 时输出等于输入, ``weight = 0`` 时输出恒为首值;
``SMA(x, n, m)`` 等价于 ``DMA(x, m / n)``, ``EMA(x, 3)`` 等价于
``DMA(x, 0.5)``. DMA 本身没有窗口长度, 因此 ``FLAG_STRICTLY_CYCLE`` 按
算法逻辑不应改变递推结果; 当前实现把内部周期置为 0, 严格周期下输出全
nan, 相关用例保留失败.

``FLAG_SKIP_NAN`` 使 nan 完全不参与运算, 等价于把有效值压紧 (compact) 后
跑参考实现, 再映射回有效位置; nan 位置本身仍为 nan. 不设置该 flag 时,
按递推公式 nan 会污染后续所有输出; 当前实现遇到 nan 后会重启递推.

参考实现 ``_dma_ref`` 用 numpy 独立实现递推公式, 只描述算法逻辑, 不调用
alpha, 也不实现 Group / start-end 机制: Group 由测试按 chunk 分块计算,
start/end 只校验 ``[start, end)`` 区间的值. 当前实现从 ``start`` 处重启
递推 (种子为 chunk 首值) 且忽略 ``end``, 相关用例保留失败.

`set_ctx` 只覆盖传入的参数, 因此每次调用前都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=120, suppress=True)
    >>> x = np.arange(1., 11.)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. ``DMA(x, 0.5)`` 与 ``EMA(x, 3)`` 完全一致::

    >>> np.round(al.DMA(x, 0.5), 3)
    array([1.   , 1.5  , 2.25 , 3.125, 4.062, 5.031, 6.016, 7.008, 8.004, 9.002])

2. 权重越小越平滑, ``DMA(x, 0.25)`` 等价于 ``SMA(x, 4, 1)``::

    >>> np.round(al.DMA(x, 0.25), 3)
    array([1.   , 1.25 , 1.688, 2.266, 2.949, 3.712, 4.534, 5.4  , 6.3  , 7.225])

3. 权重边界: ``weight = 1`` 复制输入, ``weight = 0`` 恒为首值::

    >>> al.DMA(x, 1.0)
    array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
    >>> np.round(al.DMA(x, 0.0), 3)
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

4. ``FLAG_SKIP_NAN``: 压紧后的有效值为 ``[1, 2, 3, 5, 6, 7, 8, 9, 10]``::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.DMA(xn, 0.5), 3)
    array([1.   , 1.5  , 2.25 ,   nan, 3.625, 4.812, 5.906, 6.953, 7.977, 8.988])

   不设置该 flag 时 nan 会参与运算::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.DMA(xn, 0.5), 3)
    array([1.  , 1.5 , 2.25,  nan, 5.  , 5.5 , 6.25, 7.125, 8.062, 9.031])

5. ``FLAG_STRICTLY_CYCLE`` 按算法逻辑不应改变递推, 当前实现输出全 nan::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.DMA(x, 0.5), 3)
    array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])

6. ``groups`` 把一维数据按 group-major 切块, 每块重新以块首值起步::

    >>> y = np.arange(1., 13.)
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.DMA(y, 0.5), 3)
    array([ 1.   ,  1.5  ,  2.25 ,  3.125,  4.062,  5.031,
             7.   ,  7.5  ,  8.25 ,  9.125, 10.062, 11.031])

7. ``start`` / ``end``: 测试只校验 ``[start, end)`` 区间::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.DMA(x, 0.5)[3:8], 3)
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


def _dma_ref(x, weight):
  """DMA 递推参考实现: r[0] = x[0], r[i] = w*x[i] + (1-w)*r[i-1].

  只描述算法逻辑: 含 nan 时 nan 沿递推传播到后续所有位置; 没有窗口长度,
  因此这里不实现 STRICTLY_CYCLE.
  """
  x = np.asarray(x, dtype=np.float64)
  if len(x) == 0:
    return np.empty(0, dtype=np.float64)

  out = np.empty(len(x), dtype=np.float64)
  prev = x[0]
  for i, val in enumerate(x):
    prev = weight * val + (1.0 - weight) * prev
    out[i] = prev
  return out


def _compact_ref(x, weight):
  """把有效值压紧后跑参考实现, 再映射回有效位置; nan 位置保持 nan."""
  x = np.asarray(x, dtype=np.float64)
  out = np.full(len(x), np.nan)
  valid = ~np.isnan(x)
  if valid.any():
    out[valid] = _dma_ref(x[valid], weight)
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


def _expected(x, weight, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 全量计算参考值, 只保留 [start, end) 区间.

  ``FLAG_SKIP_NAN`` 时在整个 chunk 上压紧有效值; 严格周期不改变递推结果.
  """
  x = np.asarray(x, dtype=np.float64)
  skip = bool(flags & al.FLAG_SKIP_NAN)
  parts = []
  for chunk in np.split(x, groups):
    n = len(chunk)
    s, e = _calc_range(n, start, end)
    if skip:
      ref = _compact_ref(chunk, weight)
    else:
      ref = _dma_ref(chunk, weight)
    ref[:s] = np.nan
    ref[e:] = np.nan
    parts.append(ref)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(x, weight, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.DMA(np.asarray(x, dtype=np.float64), weight), dtype=np.float64
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


@pytest.mark.parametrize("weight", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
def test_dma_matches_reference(weight):
  """flags=0, groups=1: 与 numpy 递推参考实现逐点比较."""
  x = _clean_data()
  actual = _calc(x, weight, flags=0)
  _assert_close(actual, _expected(x, weight, flags=0))
  _assert_close(actual, _dma_ref(x, weight))
  assert np.isfinite(actual).all()


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_dma_strictly_cycle_does_not_change_recursion(weight):
  """DMA 无窗口长度, FLAG_STRICTLY_CYCLE 不应改变递推结果."""
  x = _clean_data()
  flags = al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, weight, flags=flags)
  _assert_close(actual, _expected(x, weight, flags=flags))
  _assert_close(actual, _dma_ref(x, weight))
  assert np.isfinite(actual).all()


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_dma_nan_without_skip_propagates(flags):
  """无 SKIP_NAN: 按递推公式 nan 应污染后续所有输出."""
  x = _nan_data()
  weight = 0.5
  actual = _calc(x, weight, flags=flags)
  _assert_range_close(actual, _expected(x, weight, flags=flags))
  assert np.isnan(actual[np.isnan(x)]).all()


def test_dma_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 有效位置等于压紧有效值后的参考结果."""
  x = _nan_data()
  weight = 0.5
  actual = _calc(x, weight, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, weight, flags=al.FLAG_SKIP_NAN))

  valid = ~np.isnan(x)
  assert np.isnan(actual[~valid]).all()
  _assert_close(actual[valid], _dma_ref(x[valid], weight))
  assert np.isfinite(actual[valid]).all()


def test_dma_skip_nan_and_strictly_cycle():
  """两个 flag 组合: 严格周期不应改变 SKIP_NAN 的压紧递推结果."""
  x = _nan_data()
  weight = 0.5
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, weight, flags=flags)
  _assert_close(actual, _expected(x, weight, flags=flags))
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
def test_dma_all_flag_combinations_with_nan(flags):
  """四种 flag 组合在含 nan 数据上都与参考实现比较."""
  x = _nan_data()
  weight = 0.35
  actual = _calc(x, weight, flags=flags)
  _assert_range_close(actual, _expected(x, weight, flags=flags))


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
def test_dma_groups(groups):
  """group-major 分块: 每个 chunk 独立递推, 重新以块首值起步."""
  x = _clean_data()
  weight = 0.5
  actual = _calc(x, weight, groups=groups, flags=0)
  _assert_close(actual, _expected(x, weight, groups=groups, flags=0))

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    chunk = x[lo : lo + chunk_size]
    _assert_close(actual[lo : lo + chunk_size], _dma_ref(chunk, weight))


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
def test_dma_skip_nan_within_each_group(groups):
  """SKIP_NAN 的压紧过程按 chunk 独立进行, 不会跨 group 借用数据."""
  x = _nan_data()
  weight = 0.4
  flags = al.FLAG_SKIP_NAN
  actual = _calc(x, weight, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, weight, groups=groups, flags=flags))
  assert np.isnan(actual[np.isnan(x)]).all()


@pytest.mark.parametrize(
  "start,end", [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5)]
)
def test_dma_start_end(start, end):
  """start/end: 仅校验 [start, end) 区间, 参考值来自全量递推."""
  x = _clean_data()
  weight = 0.5
  actual = _calc(x, weight, start=start, end=end, flags=0)
  _assert_range_close(
    actual,
    _expected(x, weight, start=start, end=end, flags=0),
    start=start,
    end=end,
  )


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_dma_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  x = _clean_data()
  weight = 0.5
  start, end = 6, 25
  actual = _calc(x, weight, start=start, end=end, groups=groups, flags=0)
  _assert_range_close(
    actual,
    _expected(x, weight, start=start, end=end, groups=groups, flags=0),
    start=start,
    end=end,
    groups=groups,
  )


def test_dma_skip_nan_with_start_end():
  """SKIP_NAN + start/end: nan 不参与 [start, end) 区间的递推."""
  x = _nan_data()
  weight = 0.5
  ctx = dict(start=10, end=90, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, weight, **ctx)
  _assert_range_close(
    actual, _expected(x, weight, **ctx), start=10, end=90
  )
  assert np.isnan(actual[np.isnan(x)]).all()


def test_dma_edge_cases():
  """weight=1 复制输入; weight=0 恒为首值; SKIP_NAN 下 nan 位置仍为 nan."""
  x = _clean_data()
  _assert_close(_calc(x, 1.0, flags=0), x)
  _assert_close(_calc(x, 0.0, flags=0), np.full_like(x, x[0]))

  xn = _nan_data()
  _assert_close(
    _calc(xn, 0.5, flags=al.FLAG_SKIP_NAN), _compact_ref(xn, 0.5)
  )
