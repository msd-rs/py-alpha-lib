# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""FRET (未来收益 / Future Return) 集成测试.

算法简介
--------
以 t+delay 日的开盘价买入, 在 t+delay+periods-1 日的收盘价卖出::

    FRET[i] = (Close[i + delay + periods - 1] - Open[i + delay]) / Open[i + delay]
    exit_offset = delay + periods - 1

``is_calc[i + delay]`` 为 0 或 nan 时该位置不计算 (输出 nan); 非零值 (含负数)
都视为「可交易」. ``Open[i + delay]`` 为 nan / 0, 或
``Close[i + exit_offset]`` 为 nan 时同样输出 nan. 窗口需要未来数据, 因此每个
chunk 的尾部 ``exit_offset`` 个位置恒为 nan; ``periods = 0`` 时全部输出 nan.

``end`` 表示本次可用数据的截止位置: 窗口不得引用下标 >= ``end`` 的数据,
所以输出只可能出现在 ``[start, end - exit_offset)``; 区间外全部为 nan.
``groups`` 按 group-major 切成等长 chunk 独立计算, start/end 是每个 chunk 内的
相对下标. 四种 ``flags`` 组合都不改变结果: 本算子是固定前瞻窗口而非滚动窗口,
没有可「跳过」的 nan, ``FLAG_SKIP_NAN`` / ``FLAG_STRICTLY_CYCLE`` 无额外语义;
被窗口引用的 nan 会让该位置输出 nan, 未被引用的 nan 完全不参与运算.

参考实现 ``_fret_ref`` 用 numpy 按定义逐个位置复现公式 (只描述算法逻辑), 不调用
alpha; ``_expected`` 按 chunk 分块并解析 start/end 的数据边界.
``set_ctx`` 只覆盖传入的参数, 所以每次调用都显式重置 start/end/groups/flags.
浮点比较容差 1e-6.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> o = np.arange(10., 15.)
    >>> c = o + 0.5
    >>> m = np.ones(5)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. ``delay=1, periods=1``: 次日开盘买入、次日收盘卖出, 末位无未来数据::

    >>> np.round(al.FRET(o, c, m, 1, 1), 5).tolist()
    [0.04545, 0.04167, 0.03846, 0.03571, nan]

2. ``delay=2, periods=1``: 延后两天入场, 尾部两个位置为 nan::

    >>> np.round(al.FRET(o, c, m, 2, 1), 5).tolist()
    [0.04167, 0.03846, 0.03571, nan, nan]

3. ``delay=1, periods=2``: 持有两天, 用 ``Close[t+2]`` 卖出::

    >>> np.round(al.FRET(o, c, m, 1, 2), 5).tolist()
    [0.13636, 0.125, 0.11538, nan, nan]

4. ``delay=0, periods=1``: 当日开盘买入、当日收盘卖出::

    >>> np.round(al.FRET(o, c, m, 0, 1), 5).tolist()
    [0.05, 0.04545, 0.04167, 0.03846, 0.03571]

5. ``is_calc`` 门控: 0 与 nan 跳过, 负数非零仍参与计算::

    >>> m2 = np.array([1., 0., np.nan, -2., 1.])
    >>> np.round(al.FRET(o, c, m2, 1, 1), 5).tolist()
    [nan, nan, 0.03846, 0.03571, nan]

6. ``periods = 0`` 全部输出 nan::

    >>> bool(np.isnan(al.FRET(o, c, m, 1, 0)).all())
    True

7. ``groups = 3``: 每个 chunk 各自在块尾留下 nan::

    >>> o2 = np.arange(10., 22.); c2 = o2 + 0.5; m3 = np.ones(12)
    >>> al.set_ctx(start=0, end=0, groups=3, flags=0)
    >>> np.round(al.FRET(o2, c2, m3, 1, 1), 5).tolist()
    [0.04545, 0.04167, 0.03846, nan, 0.03333, 0.03125, 0.02941, nan, 0.02632, 0.025, 0.02381, nan]

8. ``start`` / ``end`` 与 ``flags``: 只在 [start, end) 内计算, 且窗口不得
   越过 end; 区间外全部为 nan::

    >>> al.set_ctx(start=2, end=4, groups=3, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.FRET(o2, c2, m3, 1, 1), 5).tolist()
    [nan, nan, 0.03846, nan, nan, nan, 0.02941, nan, nan, nan, 0.02381, nan]
"""

import numpy as np
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
NAN_POS = np.array([0, 1, 3, 17, 18, 29, 44, 52, 77, 101])
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


def _fret_ref(open_, close, is_calc, delay, periods, *, data_end=None, start=0):
  """未来收益参考实现, 只描述算法逻辑.

  ``data_end`` 是可用数据的截止下标 (窗口不得引用其后的数据), ``start`` 是
  输出起始下标; 两者默认覆盖整个序列.
  """
  open_ = np.asarray(open_, dtype=np.float64)
  close = np.asarray(close, dtype=np.float64)
  is_calc = np.asarray(is_calc, dtype=np.float64)
  n = len(open_)
  if data_end is None:
    data_end = n
  out = np.full(n, np.nan)
  if periods == 0:
    return out

  exit_offset = delay + periods - 1
  last = max(0, data_end - exit_offset)
  for i in range(start, last):
    entry = i + delay
    exit_i = i + exit_offset
    if (
      not np.isnan(is_calc[entry])
      and is_calc[entry] != 0
      and not np.isnan(open_[entry])
      and open_[entry] != 0
      and not np.isnan(close[exit_i])
    ):
      out[i] = (close[exit_i] - open_[entry]) / open_[entry]
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


def _expected(
  open_, close, is_calc, delay, periods, *, start=0, end=0, groups=1, flags=0
):
  """按 chunk 分块计算参考值, 每个 chunk 独立解析 start/end; flags 无影响."""
  open_ = np.asarray(open_, dtype=np.float64)
  close = np.asarray(close, dtype=np.float64)
  is_calc = np.asarray(is_calc, dtype=np.float64)
  parts = []
  for chunk_o, chunk_c, chunk_m in zip(
    np.split(open_, groups), np.split(close, groups), np.split(is_calc, groups)
  ):
    s, e = _calc_range(len(chunk_o), start, end)
    parts.append(
      _fret_ref(
        chunk_o, chunk_c, chunk_m, delay, periods, data_end=e, start=s
      )
    )
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(open_, close, is_calc, delay, periods, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.FRET(
      np.asarray(open_, dtype=np.float64),
      np.asarray(close, dtype=np.float64),
      np.asarray(is_calc, dtype=np.float64),
      delay,
      periods,
    ),
    dtype=np.float64,
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261106)
  open_ = rng.uniform(50.0, 150.0, n)
  close = open_ + rng.normal(0.0, 3.0, n)
  is_calc = np.ones(n)
  return open_, close, is_calc


def _gated_data(n=N):
  open_, close, is_calc = _clean_data(n)
  rng = np.random.default_rng(20261109)
  is_calc[rng.random(n) < 0.2] = 0.0
  is_calc[rng.random(n) < 0.1] = -2.0
  return open_, close, is_calc


def _nan_data(n=N):
  open_, close, is_calc = _clean_data(n)
  open_[NAN_POS[NAN_POS < n]] = np.nan
  shifted = NAN_POS + 2
  close[shifted[shifted < n]] = np.nan
  shifted2 = NAN_POS + 4
  is_calc[shifted2[shifted2 < n]] = np.nan
  return open_, close, is_calc


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


def _assert_range_close(actual, expected, start, end, groups):
  """只校验每个 chunk 内 [start, end) 区间的值."""
  size = len(actual) // groups
  s, e = _calc_range(size, start, end)
  for g in range(groups):
    lo = g * size
    _assert_close(actual[lo + s : lo + e], expected[lo + s : lo + e])


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("delay", [0, 1, 2, 3])
@pytest.mark.parametrize("periods", [1, 2, 3, 5])
def test_fret_matches_numpy_reference(delay, periods):
  """各种 delay / periods 组合都与 numpy 公式参考实现逐点比较."""
  o, c, m = _clean_data()
  actual = _calc(o, c, m, delay, periods)
  _assert_close(actual, _fret_ref(o, c, m, delay, periods))
  _assert_close(actual, _expected(o, c, m, delay, periods))

  exit_offset = delay + periods - 1
  assert np.isfinite(actual[: len(o) - exit_offset]).all()
  assert np.isnan(actual[len(o) - exit_offset :]).all()


@pytest.mark.parametrize("delay", [0, 1, 2, 3])
@pytest.mark.parametrize("periods", [1, 2, 3, 5])
def test_fret_formula_direct_check(delay, periods):
  """直接按定义逐点核对公式, 不依赖 _fret_ref."""
  o, c, m = _clean_data()
  actual = _calc(o, c, m, delay, periods)
  exit_offset = delay + periods - 1
  expected = np.array(
    [
      (c[i + exit_offset] - o[i + delay]) / o[i + delay]
      for i in range(len(o) - exit_offset)
    ]
  )
  _assert_close(actual[: len(o) - exit_offset], expected)


def test_fret_unit_examples():
  """Rust 单元测试中的精确算例: 不同 delay / periods 的收益."""
  o = np.arange(10., 15.)
  c = o + 0.5
  m = np.ones(5)

  _assert_close(
    _calc(o, c, m, 1, 1),
    np.array([0.5 / 11, 0.5 / 12, 0.5 / 13, 0.5 / 14, np.nan]),
  )
  _assert_close(
    _calc(o, c, m, 2, 1),
    np.array([0.5 / 12, 0.5 / 13, 0.5 / 14, np.nan, np.nan]),
  )
  _assert_close(
    _calc(o, c, m, 1, 2),
    np.array([1.5 / 11, 1.5 / 12, 1.5 / 13, np.nan, np.nan]),
  )
  _assert_close(
    _calc(o, c, m, 0, 1),
    np.array([0.5 / 10, 0.5 / 11, 0.5 / 12, 0.5 / 13, 0.5 / 14]),
  )


@pytest.mark.parametrize("delay", [0, 1, 3])
def test_fret_periods_zero_is_all_nan(delay):
  """periods=0 不做任何计算, 输出全 nan."""
  o, c, m = _clean_data(20)
  actual = _calc(o, c, m, delay, 0)
  assert np.isnan(actual).all()
  _assert_close(actual, _expected(o, c, m, delay, 0))


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_fret_is_calc_gate(flags):
  """is_calc[i+delay] 为 0 / nan 时跳过, 负数等非零值仍参与计算."""
  o, c, m = _clean_data()
  m[3], m[7], m[19] = 0.0, np.nan, -2.0
  actual = _calc(o, c, m, 1, 1, flags=flags)
  _assert_close(actual, _expected(o, c, m, 1, 1))
  assert np.isnan(actual[[2, 6]]).all()
  assert np.isfinite(actual[18])


def test_fret_is_calc_gate_unit_example():
  """小数组精确验证门控: 0 与 nan 跳过, 负数非零有效."""
  o = np.arange(10., 15.)
  c = o + 0.5
  m2 = np.array([1., 0., np.nan, -2., 1.])
  _assert_close(
    _calc(o, c, m2, 1, 1),
    np.array([np.nan, np.nan, 0.5 / 13, 0.5 / 14, np.nan]),
  )


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_fret_nan_and_zero_open(flags):
  """Open / Close 为 nan, 或 Open 为 0 时该位置输出 nan."""
  o, c, m = _clean_data()
  o[5] = np.nan
  c[12] = np.nan
  o[22] = 0.0
  actual = _calc(o, c, m, 1, 2, flags=flags)
  _assert_close(actual, _expected(o, c, m, 1, 2))
  # delay=1, periods=2 -> entry=i+1, exit=i+2
  assert np.isnan(actual[[4, 10, 21]]).all()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_fret_flags_do_not_change_result(flags):
  """四种 flag 组合结果与 flags=0 完全一致 (含 nan 数据)."""
  o, c, m = _nan_data()
  base = _calc(o, c, m, 1, 2, flags=0)
  _assert_close(_calc(o, c, m, 1, 2, flags=flags), base)
  _assert_close(base, _expected(o, c, m, 1, 2))


@pytest.mark.parametrize("flags", [al.FLAG_SKIP_NAN, al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE])
def test_fret_skip_nan_unreferenced_nan_has_no_effect(flags):
  """FLAG_SKIP_NAN: 未被窗口引用的 nan 不参与运算, 结果与无 nan 数据一致."""
  o, c, m = _clean_data()
  o_nan, c_nan, m_nan = o.copy(), c.copy(), m.copy()
  delay, periods = 1, 3
  # entry 最大为 n - periods, 因此其后的 Open / is_calc 位置不会被引用
  unused_from = len(o) - periods + 1
  o_nan[unused_from:] = np.nan
  m_nan[unused_from:] = np.nan
  actual = _calc(o_nan, c_nan, m_nan, delay, periods, flags=flags)
  _assert_close(actual, _fret_ref(o, c, m, delay, periods))
  _assert_close(actual, _expected(o_nan, c_nan, m_nan, delay, periods, flags=flags))


@pytest.mark.parametrize("groups", GROUPS)
def test_fret_groups(groups):
  """group-major 分块: 每个 chunk 独立计算, 各自在块尾留下 nan."""
  o, c, m = _clean_data()
  delay, periods = 1, 2
  actual = _calc(o, c, m, delay, periods, groups=groups)
  _assert_close(actual, _expected(o, c, m, delay, periods, groups=groups))

  size = len(o) // groups
  exit_offset = delay + periods - 1
  for g in range(groups):
    lo = g * size
    assert np.isnan(actual[lo + size - exit_offset : lo + size]).all()
    _assert_close(
      actual[lo : lo + size],
      _fret_ref(o[lo : lo + size], c[lo : lo + size], m[lo : lo + size], delay, periods),
    )


@pytest.mark.parametrize("start,end", START_END)
@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
def test_fret_start_end(start, end, groups):
  """start/end: 仅校验 [start, end) 区间内的值, 且与全区间参考一致."""
  o, c, m = _clean_data()
  delay, periods = 1, 2
  ctx = dict(start=start, end=end, groups=groups)
  actual = _calc(o, c, m, delay, periods, **ctx)
  expected = _expected(o, c, m, delay, periods, **ctx)
  _assert_close(actual, expected)
  _assert_range_close(actual, expected, start, end, groups)


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_fret_start_end_with_nan_and_flags(flags):
  """含 nan 数据下 start/end 与四种 flags 组合共同生效."""
  o, c, m = _nan_data()
  ctx = dict(start=7, end=95, groups=3, flags=flags)
  actual = _calc(o, c, m, 1, 2, **ctx)
  _assert_close(actual, _expected(o, c, m, 1, 2, **ctx))


@pytest.mark.parametrize("delay,periods", [(0, 200), (5, 30), (200, 1)])
def test_fret_window_larger_than_data(delay, periods):
  """窗口越过数据末尾时没有任何有效输出."""
  o, c, m = _clean_data(20)
  actual = _calc(o, c, m, delay, periods)
  assert np.isnan(actual).all()
  _assert_close(actual, _expected(o, c, m, delay, periods))


def test_fret_edge_cases():
  """空输入 / 长度不匹配 / 长度不整除 / 负参数 / 单元素."""
  empty = np.empty(0, dtype=np.float64)
  with pytest.raises(ValueError):
    _calc(empty, empty, empty, 1, 1)

  with pytest.raises(ValueError):
    _calc(np.ones(12), np.ones(11), np.ones(12), 1, 1)
  with pytest.raises(ValueError):
    _calc(np.ones(12), np.ones(12), np.ones(11), 1, 1)
  with pytest.raises(ValueError):
    _calc(np.ones(10), np.ones(10), np.ones(10), 1, 1, groups=3)

  with pytest.raises(OverflowError):
    _calc(np.ones(12), np.ones(12), np.ones(12), -1, 1)
  with pytest.raises(OverflowError):
    _calc(np.ones(12), np.ones(12), np.ones(12), 1, -1)

  one = np.array([10.0])
  _assert_close(_calc(one, np.array([11.0]), one, 0, 1), np.array([0.1]))
  assert np.isnan(_calc(one, np.array([11.0]), one, 1, 1)).all()
