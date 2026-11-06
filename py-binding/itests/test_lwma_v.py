# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""LWMA_V (逐点可变周期的线性加权移动平均) 集成测试.

算法简介
--------
LWMA_V 是 LWMA 的变周期版本, 每个位置 ``i`` 用 ``periods[i]`` 决定窗口
长度, 窗口内按线性递增权重 ``1, 2, ..., q`` 加权求平均::

    q = periods[i]
    LWMA_V[i] = sum(j * x[i - q + j] for j in 1..q) / (q * (q + 1) / 2)

窗口未满 (``i < q - 1``) 时输出 nan; ``q = 0`` 在 LWMA_V 上没有累计模式,
直接输出 nan (这一点与 HHV_V / LLV_V 不同). 周期恒定时与 ``LWMA(x, q)``
完全相同.

不设置 ``FLAG_SKIP_NAN`` 时, 窗口内任一个 nan (包括 ``x[i]`` 本身) 都会
让该位置输出 nan. 设置后 nan 完全不参与运算: 把 ``[start, end)`` 区间内的
有效值压紧 (compact), 每个有效位置用它之前最近的 ``q`` 个有效值按权重
``1..q`` 求平均, nan 位置本身输出 nan. 再组合 ``FLAG_STRICTLY_CYCLE``
时, 还要求这 ``q`` 个有效值恰好占据 ``q`` 个连续的原始位置 (窗口内不能
有 nan 间隙), 否则输出 nan. 由于窗口未满本来就不输出, 单独设置
``FLAG_STRICTLY_CYCLE`` 不改变结果.

``groups`` 按 group-major 切成等长 chunk, 每个 chunk 独立计算, ``periods``
数组同样按 chunk 切分. start/end 是每个 chunk 内的相对下标, 只计算
``[start, end)`` 区间: 普通模式下窗口可引用 start 之前的数据, SKIP_NAN
模式下有效值只在区间内压紧; 区间外一律为 nan.

参考实现 ``_lwma_v_ref`` / ``_lwma_v_skip_ref`` 用 numpy 独立实现, 只描述
算法逻辑, 不调用 alpha, 也不实现 SKIP_NAN / Group / start-end 机制; 这些
上下文语义由 ``_expected`` 单独描述. 固定周期场景另外用 pandas
``rolling`` 加 numpy 权重向量做交叉校验. ``set_ctx`` 只覆盖传入的参数,
因此每次调用都显式重置 start/end/groups/flags. 浮点比较容差 1e-6.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> x = np.arange(1., 11.)
    >>> p = np.array([1, 2, 3, 4, 5, 2, 3, 1, 2, 10], dtype=np.uint64)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算, 每个位置使用自己的窗口长度; 权重和为 ``q*(q+1)/2``::

    >>> np.round(al.LWMA_V(x, p), 3).tolist()
    [1.0, 1.667, 2.333, 3.0, 3.667, 5.667, 6.333, 8.0, 8.667, 7.0]

   i=9 的窗口是 ``[1..10]``, 结果 ``(1*1 + ... + 10*10) / 55 = 7.0``.

2. ``p = 1`` 复制输入; ``p = 0`` 全部输出 nan; 固定周期 3 时窗口未满的
   前 2 个位置输出 nan::

    >>> al.LWMA_V(x, np.ones(10, dtype=np.uint64)).tolist()
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    >>> np.round(al.LWMA_V(x, np.zeros(10, dtype=np.uint64)), 3).tolist()
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
    >>> np.round(al.LWMA_V(x, np.full(10, 3, dtype=np.uint64)), 3).tolist()
    [nan, nan, 2.333, 3.333, 4.333, 5.333, 6.333, 7.333, 8.333, 9.333]

3. 不跳过 nan: 窗口内出现 nan 的位置输出 nan::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> np.round(al.LWMA_V(xn, p), 3).tolist()
    [1.0, 1.667, 2.333, nan, nan, 5.667, 6.333, 8.0, 8.667, nan]

4. ``FLAG_SKIP_NAN``: 固定周期 3 时压紧后的有效值为
   ``[1, 2, 3, 5, 6, 7, 8, 9, 10]``, i=4 的窗口是 ``[2, 3, 5]``,
   结果 ``(1*2 + 2*3 + 3*5) / 6 = 3.833``::

    >>> p3 = np.full(10, 3, dtype=np.uint64)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.LWMA_V(xn, p3), 3).tolist()
    [nan, nan, 2.333, nan, 3.833, 5.167, 6.333, 7.333, 8.333, 9.333]

   ``FLAG_SKIP_NAN | FLAG_STRICTLY_CYCLE`` 额外要求窗口内无 nan 间隙,
   i=4、i=5 重新变为 nan::

    >>> al.set_ctx(start=0, end=0, groups=1,
    ...            flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.LWMA_V(xn, p3), 3).tolist()
    [nan, nan, 2.333, nan, nan, nan, 6.333, 7.333, 8.333, 9.333]

5. ``groups`` 把数据按 group-major 切块, 每块窗口独立::

    >>> y = np.arange(1., 13.)
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.LWMA_V(y, np.full(12, 3, dtype=np.uint64)), 3).tolist()
    [nan, nan, 2.333, 3.333, 4.333, 5.333, nan, nan, 8.333, 9.333, 10.333, 11.333]

6. ``start`` / ``end`` 只输出 ``[start, end)`` 区间, 区间外全部为 nan::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.LWMA_V(x, p), 3).tolist()
    [nan, nan, nan, 3.0, 3.667, 5.667, 6.333, 8.0, nan, nan]
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


# ---------------------------------------------------------------------------
# numpy / pandas 对比实现
# ---------------------------------------------------------------------------


def _lwma_v_ref(x, periods):
  """按算法逻辑逐点取窗口的线性加权均值, 不依赖 alpha.

  窗口为 ``x[i + 1 - q : i + 1]``; ``q = 0`` 或窗口未满 (``i < q - 1``)
  时输出 nan; 窗口内含 nan 也输出 nan.
  """
  x = np.asarray(x, dtype=np.float64)
  p = np.asarray(periods, dtype=np.int64)
  out = np.full(len(x), np.nan)
  for i in range(len(x)):
    q = int(p[i])
    if q == 0 or i < q - 1:
      continue
    window = x[i + 1 - q : i + 1]
    if np.isnan(window).any():
      continue
    weights = np.arange(1, q + 1, dtype=np.float64)
    out[i] = float(np.dot(window, weights) / weights.sum())
  return out


def _lwma_v_skip_ref(seg, periods, *, strict=False):
  """SKIP_NAN 参考实现: 压紧有效值后取最近 q 个有效值的线性加权均值.

  有效值不足 ``q`` 个时输出 nan; ``strict`` 还要求这 ``q`` 个有效值占据
  ``q`` 个连续原始位置 (窗口内不能有 nan 间隙).
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
    if q == 0 or l < q:
      continue
    if strict and i - idxs[l - q] + 1 != q:
      continue
    weights = np.arange(1, q + 1, dtype=np.float64)
    out[i] = float(np.dot(vals[l - q :], weights) / weights.sum())
  return out


def _lwma_pandas_ref(x, periods):
  """固定周期下用 pandas rolling + numpy 权重做独立交叉校验."""
  weights = np.arange(1, periods + 1, dtype=np.float64)
  return (
    pd.Series(np.asarray(x, dtype=np.float64))
    .rolling(periods, min_periods=periods)
    .apply(lambda w: float(np.dot(w, weights) / weights.sum()), raw=True)
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


def _expected(x, periods, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 计算参考值, 只保留 [start, end) 区间.

  ``FLAG_SKIP_NAN`` 时只在 ``[s, e)`` 段内压紧有效值 (历史不跨 start);
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
      ref[s:e] = _lwma_v_skip_ref(
        chunk_x[s:e], chunk_p[s:e], strict=strict
      )
    else:
      ref = _lwma_v_ref(chunk_x, chunk_p)
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
    al.LWMA_V(
      np.asarray(x, dtype=np.float64),
      np.asarray(periods, dtype=np.uint64),
    ),
    dtype=np.float64,
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261110)
  return rng.normal(loc=10.0, scale=2.0, size=n).astype(np.float64)


def _nan_data(n=N):
  x = _clean_data(n)
  x[NAN_POS[NAN_POS < n]] = np.nan
  return x


def _periods_data(n=N, max_p=12):
  rng = np.random.default_rng(20261111)
  return rng.integers(0, max_p + 1, size=n).astype(np.uint64)


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_lwma_v_matches_reference(flags):
  """变周期数据在无 nan 时与 numpy 逐点窗口加权参考实现比较."""
  x = _clean_data()
  p = _periods_data()
  actual = _calc(x, p, flags=flags)
  _assert_close(actual, _expected(x, p, flags=flags))
  _assert_close(actual, _lwma_v_ref(x, p))
  assert np.isnan(actual[p == 0]).all()


@pytest.mark.parametrize("periods", [1, 2, 3, 4, 5, 10, 20])
def test_lwma_v_constant_periods_matches_pandas_rolling(periods):
  """固定周期时与 pandas rolling + 线性权重的参考实现比较."""
  x = _clean_data()
  p = np.full(len(x), periods, dtype=np.uint64)
  actual = _calc(x, p, flags=0)
  _assert_close(actual, _lwma_pandas_ref(x, periods))
  _assert_close(actual, _expected(x, p, flags=0))
  assert np.isnan(actual[: periods - 1]).all()
  assert np.isfinite(actual[periods - 1 :]).all()


@pytest.mark.parametrize("periods", [1, 2, 3, 5, 10])
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_lwma_v_constant_periods_equals_lwma(periods, flags):
  """周期恒定时 LWMA_V 与固定周期的 LWMA 在四种 flag 下都一致."""
  x = _clean_data()
  p = np.full(len(x), periods, dtype=np.uint64)
  al.set_ctx(start=0, end=0, groups=1, flags=flags)
  expected = np.asarray(al.LWMA(x, periods), dtype=np.float64)
  _assert_close(_calc(x, p, flags=flags), expected)


def test_lwma_v_period_boundaries():
  """p=1 复制输入, p=0 全 nan, 超大窗口全 nan."""
  x = _clean_data()
  _assert_close(_calc(x, np.ones(len(x), dtype=np.uint64), flags=0), x)
  assert np.isnan(
    _calc(x, np.zeros(len(x), dtype=np.uint64), flags=0)
  ).all()
  assert np.isnan(
    _calc(x, np.full(len(x), len(x) + 1, dtype=np.uint64), flags=0)
  ).all()


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_lwma_v_nan_without_skip_pollutes_window(flags):
  """无 SKIP_NAN: 窗口内任一个 nan 都会让该位置输出 nan."""
  x = _nan_data()
  p = _periods_data()
  actual = _calc(x, p, flags=flags)
  _assert_close(actual, _expected(x, p, flags=flags))
  _assert_close(actual, _lwma_v_ref(x, p))
  assert np.isnan(actual[np.isnan(x)]).all()

  for i in range(len(x)):
    q = int(p[i])
    if q and i >= q - 1 and np.isnan(x[i + 1 - q : i + 1]).any():
      assert np.isnan(actual[i])


def test_lwma_v_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 有效位置等于压紧有效值后的参考结果."""
  x = _nan_data()
  p = _periods_data()
  actual = _calc(x, p, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, p, flags=al.FLAG_SKIP_NAN))
  assert np.isnan(actual[np.isnan(x)]).all()

  valid = ~np.isnan(x)
  _assert_close(
    actual[valid], _lwma_v_skip_ref(x[valid], p[valid])
  )


def test_lwma_v_skip_nan_matches_compacted_clean_data():
  """SKIP_NAN 等价于把有效位置压紧后, 在无 nan 序列上做 LWMA_V."""
  x_clean = _clean_data()
  x = _nan_data()
  p = _periods_data()
  valid = ~np.isnan(x)
  actual = _calc(x, p, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual[valid], _lwma_v_ref(x_clean[valid], p[valid]))
  assert np.isnan(actual[~valid]).all()


def test_lwma_v_skip_nan_and_strictly_cycle():
  """两个 flag 组合: 严格模式要求最近 q 个有效值占据连续原始位置."""
  x = _nan_data()
  p = _periods_data()
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, p, flags=flags)
  _assert_close(actual, _expected(x, p, flags=flags))
  assert np.isnan(actual[np.isnan(x)]).all()

  loose = _calc(x, p, flags=al.FLAG_SKIP_NAN)
  finite = np.isfinite(actual)
  _assert_close(actual[finite], loose[finite])
  assert finite.sum() < np.isfinite(loose).sum()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_lwma_v_all_flag_combinations_with_nan(flags):
  """四种 flag 组合在含 nan 数据上都与参考实现比较."""
  x = _nan_data()
  p = _periods_data()
  actual = _calc(x, p, flags=flags)
  _assert_close(actual, _expected(x, p, flags=flags))


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_lwma_v_groups(groups, flags):
  """group-major 分块: 每个 chunk 独立计算, periods 也按块切分."""
  x = _clean_data()
  p = _periods_data()
  actual = _calc(x, p, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, p, groups=groups, flags=flags))

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    _assert_close(
      actual[lo : lo + chunk_size],
      _lwma_v_ref(x[lo : lo + chunk_size], p[lo : lo + chunk_size]),
    )


@pytest.mark.parametrize("groups", GROUPS)
def test_lwma_v_skip_nan_within_each_group(groups):
  """SKIP_NAN 的压紧过程按 chunk 独立进行, 不会跨 group 借用数据."""
  x = _nan_data()
  p = _periods_data()
  flags = al.FLAG_SKIP_NAN
  actual = _calc(x, p, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, p, groups=groups, flags=flags))
  assert np.isnan(actual[np.isnan(x)]).all()


@pytest.mark.parametrize("start,end", START_END)
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_lwma_v_start_end(start, end, flags):
  """start/end: 只校验 [start, end) 区间, 区间外全部为 nan."""
  x = _nan_data()
  p = _periods_data()
  actual = _calc(x, p, start=start, end=end, flags=flags)
  _assert_close(
    actual, _expected(x, p, start=start, end=end, flags=flags)
  )
  s, e = _calc_range(len(x), start, end)
  assert np.isnan(actual[:s]).all()
  assert np.isnan(actual[e:]).all()


@pytest.mark.parametrize("groups", GROUPS)
def test_lwma_v_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  x = _clean_data()
  p = _periods_data()
  start, end = 6, 25
  actual = _calc(x, p, start=start, end=end, groups=groups, flags=0)
  _assert_close(
    actual,
    _expected(x, p, start=start, end=end, groups=groups, flags=0),
  )

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


def test_lwma_v_skip_nan_with_start_end():
  """SKIP_NAN + start/end: 只有区间内的有效值参与压紧计算."""
  x = _nan_data()
  p = _periods_data()
  ctx = dict(start=10, end=90, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, p, **ctx)
  _assert_close(actual, _expected(x, p, **ctx))
  assert np.isnan(actual[:10]).all()
  assert np.isnan(actual[90:]).all()
  assert np.isnan(actual[np.isnan(x)]).all()
  assert np.isnan(actual[10:90][np.isnan(x[10:90])]).all()


def test_lwma_v_edge_cases():
  """空数组 / 单元素 / 全 nan / 常数序列 / 长度不匹配报错."""
  empty = np.empty(0, dtype=np.float64)
  empty_p = np.empty(0, dtype=np.uint64)
  _assert_close(_calc(empty, empty_p, flags=0), empty)
  _assert_close(_calc(empty, empty_p, flags=al.FLAG_SKIP_NAN), empty)

  one = np.array([3.0])
  _assert_close(
    _calc(one, np.array([1], dtype=np.uint64), flags=0), one
  )
  assert np.isnan(
    _calc(one, np.array([0], dtype=np.uint64), flags=0)
  ).all()

  all_nan = np.full(10, np.nan)
  assert np.isnan(
    _calc(all_nan, _periods_data(10), flags=al.FLAG_SKIP_NAN)
  ).all()

  const = np.full(20, 3.0)
  p3 = np.full(20, 4, dtype=np.uint64)
  expected = _lwma_pandas_ref(const, 4)
  _assert_close(_calc(const, p3, flags=0), expected)

  with pytest.raises(ValueError):
    _calc(const, _periods_data(10), flags=0)
