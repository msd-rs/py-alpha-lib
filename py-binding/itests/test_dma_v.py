# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""DMA_V (逐点可变权重的动态移动平均) 集成测试.

算法简介
--------
DMA_V 是 DMA 的变权重版本, 每个位置 ``i`` 用自己的权重 ``w[i]`` 做一阶
递推 (要求 ``0 <= w[i] <= 1``)::

    r[0] = x[0]
    r[i] = w[i] * x[i] + (1 - w[i]) * r[i - 1]

``w[i] = 1`` 时该位置直接取当前值, ``w[i] = 0`` 时该位置保持前一个结果.
权重全为常数时与 ``DMA(x, w)`` 完全相同. 递推型算子没有窗口长度, 因此
``FLAG_STRICTLY_CYCLE`` 按算法逻辑不应改变结果.

不设置 ``FLAG_SKIP_NAN`` 时, nan 会沿递推公式污染后续所有输出; 设置后 nan
完全不参与运算, 等价于把有效值压紧 (compact) 后跑参考实现, 再映射回有效
位置, nan 位置本身仍输出 nan.

``groups`` 按 group-major 切成等长 chunk, 每个 chunk 独立以块首值起步,
``weight`` 数组同样按 chunk 切分. start/end 是每个 chunk 内的相对下标, 只
计算 ``[start, end)`` 区间; 递推本身仍从 chunk 开头起步, 因此区间内结果
与全量递推一致, 区间外为 nan.

参考实现 ``_dma_v_ref`` 用 numpy 独立实现递推公式, 只描述算法逻辑, 不调用
alpha, 也不实现 SKIP_NAN / Group / start-end 机制; 这些上下文语义由
``_expected`` 单独描述. ``set_ctx`` 只覆盖传入的参数, 因此每次调用都显式
重置 start/end/groups/flags. 浮点比较容差 1e-6.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> x = np.arange(1., 11.)
    >>> w = np.array([0.5, 0.5, 0.2, 0.8, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5])
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算, 每个位置使用自己的权重; i=3 处 ``0.8 * 4 + 0.2 * 1.8 = 3.56``::

    >>> np.round(al.DMA_V(x, w), 3).tolist()
    [1.0, 1.5, 1.8, 3.56, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0]

2. 权重边界: 全 1 复制输入, 全 0 恒为首值; 常数权重等价于 DMA::

    >>> al.DMA_V(x, np.ones(10)).tolist()
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    >>> np.round(al.DMA_V(x, np.zeros(10)), 3).tolist()
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    >>> np.round(al.DMA_V(x, np.full(10, 0.5)), 3).tolist()
    [1.0, 1.5, 2.25, 3.125, 4.062, 5.031, 6.016, 7.008, 8.004, 9.002]

3. 不跳过 nan: 按递推公式 nan 污染后续全部输出::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> np.round(al.DMA_V(xn, w), 3).tolist()
    [1.0, 1.5, 1.8, nan, nan, nan, nan, nan, nan, nan]

4. ``FLAG_SKIP_NAN``: 压紧后的有效值为 ``[1, 2, 3, 5, 6, 7, 8, 9, 10]``,
   nan 位置本身仍是 nan; ``FLAG_STRICTLY_CYCLE`` 不改变递推结果::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.DMA_V(xn, w), 3).tolist()
    [1.0, 1.5, 1.8, nan, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.DMA_V(x, w), 3).tolist()
    [1.0, 1.5, 1.8, 3.56, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0]

5. ``groups`` 把数据按 group-major 切块, 每块重新以块首值起步::

    >>> y = np.arange(1., 13.)
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.DMA_V(y, np.full(12, 0.5)), 3).tolist()
    [1.0, 1.5, 2.25, 3.125, 4.062, 5.031, 7.0, 7.5, 8.25, 9.125, 10.062, 11.031]

6. ``start`` / ``end`` 只输出 ``[start, end)`` 区间, 区间外全部为 nan::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.DMA_V(x, w), 3).tolist()
    [nan, nan, nan, 3.56, 5.0, 5.0, 6.0, 7.0, nan, nan]
"""

import numpy as np
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


# ---------------------------------------------------------------------------
# numpy 对比实现
# ---------------------------------------------------------------------------


def _dma_v_ref(x, weight):
  """DMA_V 递推参考实现: r[0] = x[0], r[i] = w[i]*x[i] + (1-w[i])*r[i-1].

  只描述算法逻辑: 含 nan 时 nan 沿递推传播到后续所有位置; 没有窗口长度,
  因此这里不实现 STRICTLY_CYCLE.
  """
  x = np.asarray(x, dtype=np.float64)
  w = np.asarray(weight, dtype=np.float64)
  if len(x) == 0:
    return np.empty(0, dtype=np.float64)

  out = np.empty(len(x), dtype=np.float64)
  prev = x[0]
  for i in range(len(x)):
    prev = w[i] * x[i] + (1.0 - w[i]) * prev
    out[i] = prev
  return out


def _compact_ref(x, weight):
  """把有效值压紧后跑参考实现, 再映射回有效位置; nan 位置保持 nan."""
  x = np.asarray(x, dtype=np.float64)
  w = np.asarray(weight, dtype=np.float64)
  out = np.full(len(x), np.nan)
  valid = ~np.isnan(x)
  if valid.any():
    out[valid] = _dma_v_ref(x[valid], w[valid])
  return out


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


def _expected(x, weight, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 全量计算参考值, 只保留 [start, end) 区间.

  ``FLAG_SKIP_NAN`` 时在整个 chunk 上压紧有效值; 严格周期不改变递推结果.
  """
  x = np.asarray(x, dtype=np.float64)
  w = np.asarray(weight, dtype=np.float64)
  skip = bool(flags & al.FLAG_SKIP_NAN)
  parts = []
  for chunk_x, chunk_w in zip(np.split(x, groups), np.split(w, groups)):
    n = len(chunk_x)
    s, e = _calc_range(n, start, end)
    ref = (
      _compact_ref(chunk_x, chunk_w)
      if skip
      else _dma_v_ref(chunk_x, chunk_w)
    )
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
    al.DMA_V(
      np.asarray(x, dtype=np.float64),
      np.asarray(weight, dtype=np.float64),
    ),
    dtype=np.float64,
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261106)
  return rng.normal(loc=10.0, scale=2.0, size=n).astype(np.float64)


def _nan_data(n=N):
  x = _clean_data(n)
  x[NAN_POS[NAN_POS < n]] = np.nan
  return x


def _weights_data(n=N):
  rng = np.random.default_rng(20261107)
  return rng.uniform(0.0, 1.0, size=n).astype(np.float64)


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_dma_v_matches_reference(flags):
  """四种 flag 组合在无 nan 数据上都与 numpy 递推参考实现逐点比较."""
  x = _clean_data()
  w = _weights_data()
  actual = _calc(x, w, flags=flags)
  _assert_close(actual, _expected(x, w, flags=flags))
  _assert_close(actual, _dma_v_ref(x, w))
  assert np.isfinite(actual).all()


def test_dma_v_weight_boundaries():
  """w=1 复制输入, w=0 恒为首值, 常数序列保持不变."""
  x = _clean_data()
  _assert_close(_calc(x, np.ones(len(x)), flags=0), x)
  _assert_close(
    _calc(x, np.zeros(len(x)), flags=0), np.full(len(x), x[0])
  )

  const = np.full(20, 3.0)
  _assert_close(_calc(const, _weights_data(20), flags=0), const)


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_dma_v_constant_weight_equals_dma(weight):
  """常数权重时 DMA_V 与固定权重的 DMA 完全一致."""
  x = _clean_data()
  w = np.full(len(x), weight)
  _assert_close(_calc(x, w, flags=0), _calc_dma(x, weight))


def _calc_dma(x, weight):
  al.set_ctx(start=0, end=0, groups=1, flags=0)
  return np.asarray(al.DMA(x, weight), dtype=np.float64)


def test_dma_v_nan_without_skip_propagates():
  """无 SKIP_NAN: 按递推公式 nan 污染后续所有输出."""
  x = _nan_data()
  w = _weights_data()
  actual = _calc(x, w, flags=0)
  _assert_close(actual, _expected(x, w, flags=0))
  assert np.isnan(actual[np.isnan(x)]).all()
  first_nan = int(np.flatnonzero(np.isnan(x))[0])
  assert np.isnan(actual[first_nan:]).all()


def test_dma_v_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 有效位置等于压紧有效值后的参考结果."""
  x = _nan_data()
  w = _weights_data()
  actual = _calc(x, w, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, w, flags=al.FLAG_SKIP_NAN))

  valid = ~np.isnan(x)
  assert np.isnan(actual[~valid]).all()
  _assert_close(actual[valid], _dma_v_ref(x[valid], w[valid]))
  assert np.isfinite(actual[valid]).all()


def test_dma_v_skip_nan_matches_compacted_clean_data():
  """SKIP_NAN 等价于把有效位置压紧后, 在无 nan 序列上做 DMA_V."""
  x_clean = _clean_data()
  x = _nan_data()
  w = _weights_data()
  valid = ~np.isnan(x)
  actual = _calc(x, w, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual[valid], _dma_v_ref(x_clean[valid], w[valid]))
  assert np.isnan(actual[~valid]).all()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_dma_v_all_flag_combinations_with_nan(flags):
  """四种 flag 组合在含 nan 数据上都与参考实现比较."""
  x = _nan_data()
  w = _weights_data()
  actual = _calc(x, w, flags=flags)
  _assert_close(actual, _expected(x, w, flags=flags))
  assert np.isnan(actual[np.isnan(x)]).all()


def test_dma_v_strictly_cycle_has_no_effect():
  """递推型算子没有窗口长度, STRICTLY_CYCLE 不改变任何输出."""
  x = _clean_data()
  w = _weights_data()
  base = _calc(x, w, flags=0)
  _assert_close(_calc(x, w, flags=al.FLAG_STRICTLY_CYCLE), base)

  xn = _nan_data()
  for flags in [al.FLAG_SKIP_NAN, al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE]:
    _assert_close(_calc(xn, w, flags=flags), _compact_ref(xn, w))


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_dma_v_groups(groups, flags):
  """group-major 分块: 每个 chunk 独立递推, 权重也按块切分."""
  x = _clean_data()
  w = _weights_data()
  actual = _calc(x, w, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, w, groups=groups, flags=flags))

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    _assert_close(
      actual[lo : lo + chunk_size],
      _dma_v_ref(x[lo : lo + chunk_size], w[lo : lo + chunk_size]),
    )


@pytest.mark.parametrize("groups", GROUPS)
def test_dma_v_skip_nan_within_each_group(groups):
  """SKIP_NAN 的压紧过程按 chunk 独立进行, 不会跨 group 借用数据."""
  x = _nan_data()
  w = _weights_data()
  actual = _calc(x, w, groups=groups, flags=al.FLAG_SKIP_NAN)
  _assert_close(
    actual, _expected(x, w, groups=groups, flags=al.FLAG_SKIP_NAN)
  )
  assert np.isnan(actual[np.isnan(x)]).all()


@pytest.mark.parametrize("start,end", START_END)
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_dma_v_start_end(start, end, flags):
  """start/end: 只计算 [start, end) 区间, 区间外全部为 nan."""
  x = _clean_data()
  w = _weights_data()
  actual = _calc(x, w, start=start, end=end, flags=flags)
  _assert_close(
    actual, _expected(x, w, start=start, end=end, flags=flags)
  )
  s, e = _calc_range(len(x), start, end)
  assert np.isnan(actual[:s]).all()
  assert np.isnan(actual[e:]).all()


@pytest.mark.parametrize("groups", GROUPS)
def test_dma_v_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  x = _clean_data()
  w = _weights_data()
  start, end = 6, 25
  actual = _calc(
    x, w, start=start, end=end, groups=groups, flags=0
  )
  _assert_close(
    actual,
    _expected(x, w, start=start, end=end, groups=groups, flags=0),
  )

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


def test_dma_v_skip_nan_with_start_end():
  """SKIP_NAN + start/end: 递推仍从 chunk 开头起步, 只输出区间内结果."""
  x = _nan_data()
  w = _weights_data()
  ctx = dict(start=10, end=90, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, w, **ctx)
  _assert_close(actual, _expected(x, w, **ctx))
  assert np.isnan(actual[:10]).all()
  assert np.isnan(actual[90:]).all()
  assert np.isnan(actual[np.isnan(x)]).all()


def test_dma_v_edge_cases():
  """空数组 / 单元素 / 全 nan / 长度不匹配 / 非法权重报错."""
  empty = np.empty(0, dtype=np.float64)
  _assert_close(_calc(empty, empty, flags=0), empty)
  _assert_close(
    _calc(empty, empty, flags=al.FLAG_SKIP_NAN), empty
  )

  one = np.array([3.0])
  _assert_close(_calc(one, np.array([0.5]), flags=0), one)

  all_nan = np.full(10, np.nan)
  assert np.isnan(
    _calc(all_nan, _weights_data(10), flags=al.FLAG_SKIP_NAN)
  ).all()

  x = _clean_data(20)
  with pytest.raises(ValueError):
    _calc(x, _weights_data(10), flags=0)
  with pytest.raises(ValueError):
    _calc(x, np.full(20, -0.1), flags=0)
  with pytest.raises(ValueError):
    _calc(x, np.full(20, 1.1), flags=0)
