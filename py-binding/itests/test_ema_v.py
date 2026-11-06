# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""EMA_V (逐点可变周期的指数移动平均) 集成测试.

算法简介
--------
EMA_V 是 EMA 的变周期版本, 每个位置 ``i`` 用 ``periods[i]`` 决定平滑权重::

    a[i] = 2 / (p[i] + 1)          (p[i] > 0)
    a[i] = 0                       (p[i] == 0)
    r[0] = x[0]
    r[i] = a[i] * x[i] + (1 - a[i]) * r[i - 1]

``p[i] = 1`` 时权重为 1, 该位置直接取当前值; ``p[i] = 0`` 时权重为 0,
保持前一个结果. 周期恒定时与 ``EMA(x, p)`` 完全相同.

``FLAG_STRICTLY_CYCLE`` 要求窗口未满时不输出: 若当前位置的有效值计数
``k < p[i] - 1`` (且 ``p[i] > 1``), 该位置输出 nan, 并把当前原始值作为
后续递推的种子 (warm-up 期间不做递推). 周期恒定 ``p`` 时即前 ``p - 1``
个位置为 nan, 与 EMA 的严格周期语义一致.

不设置 ``FLAG_SKIP_NAN`` 时, nan 沿递推公式污染后续所有输出; 设置后 nan
完全不参与运算, 等价于把有效值压紧 (compact) 后跑参考实现, 再映射回有效
位置, nan 位置本身仍输出 nan. 组合两个 flag 时在压紧后的有效序列上做
warm-up.

``groups`` 按 group-major 切成等长 chunk, 每个 chunk 独立以块首值起步,
``periods`` 数组同样按 chunk 切分. start/end 是每个 chunk 内的相对下标,
只计算 ``[start, end)`` 区间; 递推本身仍从 chunk 开头起步, 因此区间内结果
与全量递推一致, 区间外为 nan.

参考实现 ``_ema_v_ref`` 用 numpy 独立实现递推公式, 只描述算法逻辑, 不调用
alpha, 也不实现 SKIP_NAN / Group / start-end 机制; 这些上下文语义由
``_expected`` 单独描述. ``set_ctx`` 只覆盖传入的参数, 因此每次调用都显式
重置 start/end/groups/flags. 浮点比较容差 1e-6.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> x = np.arange(1., 11.)
    >>> p = np.array([1, 2, 3, 4, 5, 2, 3, 1, 2, 10], dtype=np.uint64)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算, 每个位置使用自己的平滑权重::

    >>> np.round(al.EMA_V(x, p), 3).tolist()
    [1.0, 1.667, 2.333, 3.0, 3.667, 5.222, 6.111, 8.0, 8.667, 8.909]

   i=1 处 ``a = 2/3``, 结果为 ``2/3 * 2 + 1/3 * 1 = 1.667``.

2. ``p = 1`` 复制输入, ``p = 0`` 恒为首值::

    >>> al.EMA_V(x, np.ones(10, dtype=np.uint64)).tolist()
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    >>> al.EMA_V(x, np.zeros(10, dtype=np.uint64)).tolist()
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

3. ``FLAG_STRICTLY_CYCLE``: 固定周期 4 时前 3 个位置为 nan, warm-up 以
   原始值 ``x[2] = 3`` 为种子, 首个输出 ``0.4 * 4 + 0.6 * 3 = 3.4``;
   与 ``al.EMA(x, 4)`` 的严格模式一致::

    >>> p4 = np.full(10, 4, dtype=np.uint64)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.EMA_V(x, p4), 3).tolist()
    [nan, nan, nan, 3.4, 4.04, 4.824, 5.694, 6.617, 7.57, 8.542]

4. 不跳过 nan: 按递推公式 nan 污染后续全部输出::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.EMA_V(xn, p), 3).tolist()
    [1.0, 1.667, 2.333, nan, nan, nan, nan, nan, nan, nan]

5. ``FLAG_SKIP_NAN``: 压紧后的有效值为 ``[1, 2, 3, 5, 6, 7, 8, 9, 10]``::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.EMA_V(xn, p), 3).tolist()
    [1.0, 1.667, 2.333, nan, 3.222, 5.074, 6.037, 8.0, 8.667, 8.909]

   ``FLAG_SKIP_NAN | FLAG_STRICTLY_CYCLE`` 在压紧序列上做 warm-up,
   末位 ``p = 10`` 超过有效值个数, 输出 nan::

    >>> al.set_ctx(start=0, end=0, groups=1,
    ...            flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.EMA_V(xn, p), 3).tolist()
    [1.0, 1.667, 2.333, nan, nan, 5.667, 6.333, 8.0, 8.667, nan]

6. ``groups`` 把数据按 group-major 切块, 每块重新以块首值起步::

    >>> y = np.arange(1., 13.)
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.EMA_V(y, np.full(12, 3, dtype=np.uint64)), 3).tolist()
    [1.0, 1.5, 2.25, 3.125, 4.062, 5.031, 7.0, 7.5, 8.25, 9.125, 10.062, 11.031]

7. ``start`` / ``end`` 只输出 ``[start, end)`` 区间, 区间外全部为 nan::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.EMA_V(x, p), 3).tolist()
    [nan, nan, nan, 3.0, 3.667, 5.222, 6.111, 8.0, nan, nan]
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


def _ema_v_ref(x, periods, *, strict=False):
  """EMA_V 递推参考实现, 只描述算法逻辑.

  含 nan 时 nan 沿递推传播到后续所有位置; ``strict`` 时有效值计数不足
  ``p[i] - 1`` 的位置输出 nan, 且 warm-up 期间只把当前值作为递推种子.
  """
  x = np.asarray(x, dtype=np.float64)
  p = np.asarray(periods, dtype=np.int64)
  if len(x) == 0:
    return np.empty(0, dtype=np.float64)

  out = np.empty(len(x), dtype=np.float64)
  prev = x[0]
  for i, val in enumerate(x):
    a = 2.0 / (p[i] + 1) if p[i] > 0 else 0.0
    if strict and p[i] > 1 and i < p[i] - 1:
      out[i] = np.nan
      prev = val
    else:
      prev = a * val + (1.0 - a) * prev
      out[i] = prev
  return out


def _compact_ref(x, periods, *, strict=False):
  """把有效值压紧后跑参考实现, 再映射回有效位置; nan 位置保持 nan."""
  x = np.asarray(x, dtype=np.float64)
  p = np.asarray(periods, dtype=np.int64)
  out = np.full(len(x), np.nan)
  valid = ~np.isnan(x)
  if valid.any():
    out[valid] = _ema_v_ref(x[valid], p[valid], strict=strict)
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


def _expected(x, periods, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 全量计算参考值, 只保留 [start, end) 区间.

  ``FLAG_SKIP_NAN`` 时在整个 chunk 上压紧有效值; ``FLAG_STRICTLY_CYCLE``
  交给参考实现的 warm-up 语义处理.
  """
  x = np.asarray(x, dtype=np.float64)
  p = np.asarray(periods, dtype=np.int64)
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  skip = bool(flags & al.FLAG_SKIP_NAN)
  parts = []
  for chunk_x, chunk_p in zip(np.split(x, groups), np.split(p, groups)):
    n = len(chunk_x)
    s, e = _calc_range(n, start, end)
    ref = (
      _compact_ref(chunk_x, chunk_p, strict=strict)
      if skip
      else _ema_v_ref(chunk_x, chunk_p, strict=strict)
    )
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
    al.EMA_V(
      np.asarray(x, dtype=np.float64),
      np.asarray(periods, dtype=np.uint64),
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


def _periods_data(n=N, max_p=12):
  rng = np.random.default_rng(20261108)
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
def test_ema_v_matches_reference(flags):
  """变周期数据在无 nan 时与 numpy 递推参考实现逐点比较."""
  x = _clean_data()
  p = _periods_data()
  actual = _calc(x, p, flags=flags)
  _assert_close(actual, _expected(x, p, flags=flags))
  _assert_close(
    actual, _ema_v_ref(x, p, strict=bool(flags & al.FLAG_STRICTLY_CYCLE))
  )


def test_ema_v_period_boundaries():
  """p=1 复制输入, p=0 恒为首值, 常数序列保持不变."""
  x = _clean_data()
  _assert_close(_calc(x, np.ones(len(x), dtype=np.uint64), flags=0), x)
  _assert_close(
    _calc(x, np.zeros(len(x), dtype=np.uint64), flags=0),
    np.full(len(x), x[0]),
  )

  const = np.full(20, 3.0)
  _assert_close(_calc(const, _periods_data(20), flags=0), const)


@pytest.mark.parametrize("periods", [1, 2, 3, 4, 5, 10, 20])
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_ema_v_constant_periods_equals_ema(periods, flags):
  """周期恒定时 EMA_V 与固定周期的 EMA 完全一致 (含严格周期)."""
  x = _clean_data()
  p = np.full(len(x), periods, dtype=np.uint64)
  al.set_ctx(start=0, end=0, groups=1, flags=flags)
  expected = np.asarray(al.EMA(x, periods), dtype=np.float64)
  _assert_close(_calc(x, p, flags=flags), expected)


def test_ema_v_strictly_cycle_warm_up():
  """FLAG_STRICTLY_CYCLE: warm-up 位置输出 nan, 之后以原始值作种子递推."""
  x = np.arange(1.0, 11.0)
  p = np.full(len(x), 4, dtype=np.uint64)
  actual = _calc(x, p, flags=al.FLAG_STRICTLY_CYCLE)
  _assert_close(actual, _ema_v_ref(x, p, strict=True))
  _assert_close(actual, _expected(x, p, flags=al.FLAG_STRICTLY_CYCLE))
  assert np.isnan(actual[:3]).all()
  assert np.isfinite(actual[3:]).all()
  assert actual[3] == pytest.approx(3.4, abs=ATOL)


def test_ema_v_nan_without_skip_propagates():
  """无 SKIP_NAN: 按递推公式 nan 污染后续所有输出."""
  x = _nan_data()
  p = _periods_data()
  actual = _calc(x, p, flags=0)
  _assert_close(actual, _expected(x, p, flags=0))
  assert np.isnan(actual[np.isnan(x)]).all()
  first_nan = int(np.flatnonzero(np.isnan(x))[0])
  assert np.isnan(actual[first_nan:]).all()


def test_ema_v_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 有效位置等于压紧有效值后的参考结果."""
  x = _nan_data()
  p = _periods_data()
  actual = _calc(x, p, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, p, flags=al.FLAG_SKIP_NAN))

  valid = ~np.isnan(x)
  assert np.isnan(actual[~valid]).all()
  _assert_close(actual[valid], _ema_v_ref(x[valid], p[valid]))
  assert np.isfinite(actual[valid]).all()


def test_ema_v_skip_nan_matches_compacted_clean_data():
  """SKIP_NAN 等价于把有效位置压紧后, 在无 nan 序列上做 EMA_V."""
  x_clean = _clean_data()
  x = _nan_data()
  p = _periods_data()
  valid = ~np.isnan(x)
  actual = _calc(x, p, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual[valid], _ema_v_ref(x_clean[valid], p[valid]))
  assert np.isnan(actual[~valid]).all()


def test_ema_v_skip_nan_and_strictly_cycle():
  """两个 flag 组合: 在压紧后的有效序列上做 warm-up."""
  x = _nan_data()
  p = _periods_data()
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, p, flags=flags)
  _assert_close(actual, _expected(x, p, flags=flags))
  assert np.isnan(actual[np.isnan(x)]).all()

  valid = ~np.isnan(x)
  _assert_close(
    actual[valid], _ema_v_ref(x[valid], p[valid], strict=True)
  )


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_ema_v_all_flag_combinations_with_nan(flags):
  """四种 flag 组合在含 nan 数据上都与参考实现比较."""
  x = _nan_data()
  p = _periods_data()
  actual = _calc(x, p, flags=flags)
  _assert_close(actual, _expected(x, p, flags=flags))
  assert np.isnan(actual[np.isnan(x)]).all()


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_ema_v_groups(groups, flags):
  """group-major 分块: 每个 chunk 独立递推, periods 也按块切分."""
  x = _clean_data()
  p = _periods_data()
  actual = _calc(x, p, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, p, groups=groups, flags=flags))

  chunk_size = len(x) // groups
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  for g in range(groups):
    lo = g * chunk_size
    _assert_close(
      actual[lo : lo + chunk_size],
      _ema_v_ref(
        x[lo : lo + chunk_size],
        p[lo : lo + chunk_size],
        strict=strict,
      ),
    )


@pytest.mark.parametrize("groups", GROUPS)
def test_ema_v_skip_nan_within_each_group(groups):
  """SKIP_NAN 的压紧过程按 chunk 独立进行, 不会跨 group 借用数据."""
  x = _nan_data()
  p = _periods_data()
  flags = al.FLAG_SKIP_NAN
  actual = _calc(x, p, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, p, groups=groups, flags=flags))
  assert np.isnan(actual[np.isnan(x)]).all()


@pytest.mark.parametrize("start,end", START_END)
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_ema_v_start_end(start, end, flags):
  """start/end: 只计算 [start, end) 区间, 区间外全部为 nan."""
  x = _clean_data()
  p = _periods_data()
  actual = _calc(x, p, start=start, end=end, flags=flags)
  _assert_close(actual, _expected(x, p, start=start, end=end, flags=flags))
  s, e = _calc_range(len(x), start, end)
  assert np.isnan(actual[:s]).all()
  assert np.isnan(actual[e:]).all()


@pytest.mark.parametrize("groups", GROUPS)
def test_ema_v_start_end_with_groups(groups):
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


def test_ema_v_skip_nan_with_start_end():
  """SKIP_NAN + start/end: 递推仍从 chunk 开头起步, 只输出区间内结果."""
  x = _nan_data()
  p = _periods_data()
  ctx = dict(start=10, end=90, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, p, **ctx)
  _assert_close(actual, _expected(x, p, **ctx))
  assert np.isnan(actual[:10]).all()
  assert np.isnan(actual[90:]).all()
  assert np.isnan(actual[np.isnan(x)]).all()


def test_ema_v_edge_cases():
  """空数组 / 单元素 / 全 nan / 超大周期 / 长度不匹配报错."""
  empty = np.empty(0, dtype=np.float64)
  empty_p = np.empty(0, dtype=np.uint64)
  _assert_close(_calc(empty, empty_p, flags=0), empty)
  _assert_close(_calc(empty, empty_p, flags=al.FLAG_SKIP_NAN), empty)

  one = np.array([3.0])
  _assert_close(_calc(one, np.array([2], dtype=np.uint64), flags=0), one)

  all_nan = np.full(10, np.nan)
  assert np.isnan(
    _calc(all_nan, _periods_data(10), flags=al.FLAG_SKIP_NAN)
  ).all()

  x = _clean_data(20)
  big = np.full(20, 100, dtype=np.uint64)
  assert np.isnan(
    _calc(x, big, flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
  ).all()

  with pytest.raises(ValueError):
    _calc(x, _periods_data(10), flags=0)
