# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""LONGCROSS / RLONGCROSS (持续 N 期后的上穿 / 下穿) 集成测试.

算法简介
--------
两个算子在两条等长序列上输出布尔信号::

    LONGCROSS(a, b, n)[i]  = 前 n 期 a < b 且当前 a >= b   # 持续 n 期在下方后上穿
    RLONGCROSS(a, b, n)[i] = 前 n 期 a > b 且当前 a <= b   # 持续 n 期在上方后下穿

「前 n 期」按「紧邻当前位置之前**连续**至少 n 期满足」理解, 连续段更长同样
成立; 当前位置满足反向比较时连续计数归零. ``n=0`` 时前置条件为空, 只要
当前 ``a >= b`` (或 ``a <= b``) 就输出 True. 相等位置属于 ``>=`` / ``<=``,
不属于严格的前置状态.

``flags=0`` 时要求参与比较的四个值都非 nan, nan 位置输出 False 并**打断**
连续状态; 窗口可以引用 ``start`` 之前的数据, 即状态从 chunk 头累计.
``FLAG_SKIP_NAN`` 改为只在 a、b **同时有效**的配对上维护连续状态: nan 位置
输出 False 但既不打断状态、也不参与比较, 等价于把有效配对压紧后再跑参考
实现再映射回有效位置, 且该状态只在 ``[start, end)`` 内维护.
``FLAG_STRICTLY_CYCLE`` 被忽略; bool 输出无法表达 nan, 区间外与无效位置
一律用 False 表示. ``groups`` 按 group-major 切成等长 chunk 独立计算.

对比基准 ``_longcross_ref`` / ``_longcross_skip_ref`` 用 numpy 独立复现连续
计数状态机, 不调用 alpha, 也不实现 Group / start-end 机制; 这些上下文语义
由 ``_expected`` 单独描述: 按 chunk 分块, flags=0 时先全量计算再屏蔽区间外,
SKIP_NAN 时只在 [start, end) 内压紧有效配对.

`set_ctx` 只覆盖传入的参数, 所以每次调用都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> a = np.array([1., 2., 3., 6., 7., 5., 4., 8.])
    >>> b = np.full(8, 5.)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. ``LONGCROSS(a, b, 2)``: i=3 之前连续 3 期 a<b, 当前 6>=5::

    >>> al.LONGCROSS(a, b, 2).tolist()
    [False, False, False, True, False, False, False, False]

2. ``RLONGCROSS(a, b, 2)``: i=3、4 连续 2 期 a>b, i=5 处 5<=5 下穿::

    >>> al.RLONGCROSS(a, b, 2).tolist()
    [False, False, False, False, False, True, False, False]

3. ``n=0``: 前置条件为空, 当前 a>=b 的位置即为 True::

    >>> al.LONGCROSS(a, b, 0).tolist()
    [False, False, False, True, True, True, False, True]

4. 不跳过 nan 时, a[2] 的 nan 打断连续状态; SKIP_NAN 则忽略该位置,
   结果回到无 nan 时的输出::

    >>> an = a.copy(); an[2] = np.nan
    >>> al.LONGCROSS(an, b, 2).tolist()
    [False, False, False, False, False, False, False, False]
    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> al.LONGCROSS(an, b, 2).tolist()
    [False, False, False, True, False, False, False, False]

5. ``groups``: 每块独立重新累计连续状态, 跨块的历史不生效::

    >>> a2 = np.array([1., 2., 3., 4., 6., 2., 3., 4.])
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> al.LONGCROSS(a2, b, 3).tolist()
    [False, False, False, False, True, False, False, False]
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> al.LONGCROSS(a2, b, 3).tolist()
    [False, False, False, False, False, False, False, False]

6. ``start`` / ``end`` 只输出 [start, end) 区间::

    >>> a3 = np.array([1., 2., 3., 6., 1., 2., 3., 6.])
    >>> al.set_ctx(start=4, end=8, groups=1, flags=0)
    >>> al.LONGCROSS(a3, b, 2).tolist()
    [False, False, False, False, False, False, False, True]
"""

import numpy as np
import pytest

import alpha as al

N = 120
NS = [0, 1, 2, 3, 5, 10]
REVERSE = [False, True]
NAN_POS_A = np.array([3, 17, 18, 29, 44, 52, 77, 101])
NAN_POS_B = np.array([5, 18, 31, 45, 60, 78, 99])
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


def _longcross_ref(a, b, n, reverse):
  """flags=0 的参考实现: 连续前置状态计数, nan 打断计数并输出 False."""
  a = np.asarray(a, dtype=np.float64)
  b = np.asarray(b, dtype=np.float64)
  valid = ~(np.isnan(a) | np.isnan(b))
  below = valid & (a > b if reverse else a < b)
  cross = valid & (a <= b if reverse else a >= b)

  out = np.zeros(len(a), dtype=bool)
  streak = 0
  for i in range(len(a)):
    if cross[i] and streak >= n:
      out[i] = True
    streak = streak + 1 if below[i] else 0
  return out


def _longcross_skip_ref(a, b, n, reverse, lo, hi):
  """SKIP_NAN 参考实现: 只在 [lo, hi) 内的有效配对上维护连续状态."""
  a = np.asarray(a, dtype=np.float64)
  b = np.asarray(b, dtype=np.float64)
  out = np.zeros(len(a), dtype=bool)
  streak = 0
  for i in range(lo, hi):
    if np.isnan(a[i]) or np.isnan(b[i]):
      continue
    is_below = a[i] > b[i] if reverse else a[i] < b[i]
    if is_below:
      streak += 1
    else:
      if streak >= n:
        out[i] = True
      streak = 0
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


def _expected(a, b, n, reverse, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 计算参考值, 只保留 alpha 应输出的 [start, end) 区间."""
  a = np.asarray(a, dtype=np.float64)
  b = np.asarray(b, dtype=np.float64)
  skip = bool(flags & al.FLAG_SKIP_NAN)
  parts = []
  for chunk_a, chunk_b in zip(np.split(a, groups), np.split(b, groups)):
    size = len(chunk_a)
    s, e = _calc_range(size, start, end)
    if skip:
      ref = _longcross_skip_ref(chunk_a, chunk_b, n, reverse, s, e)
    else:
      ref = _longcross_ref(chunk_a, chunk_b, n, reverse)
      ref[:s] = False
      ref[e:] = False
    parts.append(ref)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(a, b, n, reverse, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  fn = al.RLONGCROSS if reverse else al.LONGCROSS
  return np.asarray(
    fn(
      np.asarray(a, dtype=np.float64),
      np.asarray(b, dtype=np.float64),
      n,
    ),
    dtype=bool,
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261111)
  a = rng.normal(loc=10.0, scale=2.0, size=n).astype(np.float64)
  b = rng.normal(loc=10.0, scale=2.0, size=n).astype(np.float64)
  return a, b


def _nan_data(n=N):
  a, b = _clean_data(n)
  a[NAN_POS_A[NAN_POS_A < n]] = np.nan
  b[NAN_POS_B[NAN_POS_B < n]] = np.nan
  return a, b


def _assert_eq(actual, expected):
  assert actual.shape == expected.shape
  assert actual.dtype == np.bool_
  np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("reverse", REVERSE)
def test_longcross_matches_reference(n, reverse):
  """flags=0, groups=1: 与 numpy 连续状态机参考实现逐点比较."""
  a, b = _clean_data()
  actual = _calc(a, b, n, reverse, flags=0)
  _assert_eq(actual, _longcross_ref(a, b, n, reverse))
  _assert_eq(actual, _expected(a, b, n, reverse, flags=0))


@pytest.mark.parametrize("reverse", REVERSE)
def test_longcross_zero_periods_is_current_comparison(reverse):
  """n=0: 前置条件为空, 输出等于当前位置的 >= / <= 比较."""
  a, b = _clean_data()
  actual = _calc(a, b, 0, reverse, flags=0)
  expected = a <= b if reverse else a >= b
  _assert_eq(actual, expected)
  _assert_eq(actual, _longcross_ref(a, b, 0, reverse))


@pytest.mark.parametrize("flags", FLAG_COMBOS)
@pytest.mark.parametrize("reverse", REVERSE)
def test_longcross_strictly_cycle_has_no_effect(flags, reverse):
  """STRICTLY_CYCLE 被忽略; 四种 flag 组合在干净数据上结果一致."""
  a, b = _clean_data()
  n = 3
  _assert_eq(_calc(a, b, n, reverse, flags=flags), _calc(a, b, n, reverse, flags=0))
  _assert_eq(
    _calc(a, b, n, reverse, flags=flags),
    _expected(a, b, n, reverse, flags=flags),
  )


@pytest.mark.parametrize("reverse", REVERSE)
def test_longcross_without_skip_nan_propagates(reverse):
  """无 SKIP_NAN: nan 位置输出 False, 且打断连续状态."""
  a, b = _nan_data()
  n = 3
  flags = 0
  actual = _calc(a, b, n, reverse, flags=flags)
  _assert_eq(actual, _expected(a, b, n, reverse, flags=flags))
  invalid = np.isnan(a) | np.isnan(b)
  assert not actual[invalid].any()


@pytest.mark.parametrize("reverse", REVERSE)
def test_longcross_skip_nan_ignores_nan_values(reverse):
  """FLAG_SKIP_NAN: nan 不参与运算、不打断连续状态, 有效位置的结果不变."""
  a_clean, b_clean = _clean_data()
  a, b = _nan_data()
  n = 3
  actual = _calc(a, b, n, reverse, flags=al.FLAG_SKIP_NAN)
  _assert_eq(
    actual,
    _expected(a, b, n, reverse, flags=al.FLAG_SKIP_NAN),
  )

  valid = ~(np.isnan(a) | np.isnan(b))
  assert not actual[~valid].any()
  compact_ref = _longcross_ref(a_clean[valid], b_clean[valid], n, reverse)
  _assert_eq(actual[valid], compact_ref)


@pytest.mark.parametrize("flags", FLAG_COMBOS)
@pytest.mark.parametrize("reverse", REVERSE)
def test_longcross_all_flag_combinations_with_nan(flags, reverse):
  """四种 flag 组合在含 nan 数据上都与参考实现比较."""
  a, b = _nan_data()
  n = 4
  actual = _calc(a, b, n, reverse, flags=flags)
  _assert_eq(actual, _expected(a, b, n, reverse, flags=flags))


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("reverse", REVERSE)
def test_longcross_groups(groups, reverse):
  """group-major 分块: 每个 chunk 独立从块首累计连续状态."""
  a, b = _clean_data()
  n = 3
  actual = _calc(a, b, n, reverse, groups=groups, flags=0)
  _assert_eq(actual, _expected(a, b, n, reverse, groups=groups, flags=0))

  chunk_size = len(a) // groups
  for g in range(groups):
    lo = g * chunk_size
    _assert_eq(
      actual[lo : lo + chunk_size],
      _longcross_ref(a[lo : lo + chunk_size], b[lo : lo + chunk_size], n, reverse),
    )


@pytest.mark.parametrize("groups", GROUPS)
def test_longcross_skip_nan_within_each_group(groups):
  """SKIP_NAN 的压紧过程按 chunk 独立进行, 不会跨 group 借用数据."""
  a, b = _nan_data()
  n = 4
  flags = al.FLAG_SKIP_NAN
  actual = _calc(a, b, n, False, groups=groups, flags=flags)
  _assert_eq(actual, _expected(a, b, n, False, groups=groups, flags=flags))
  invalid = np.isnan(a) | np.isnan(b)
  assert not actual[invalid].any()


@pytest.mark.parametrize("start,end", START_END)
@pytest.mark.parametrize("flags", [0, al.FLAG_SKIP_NAN])
def test_longcross_start_end(start, end, flags):
  """start/end: 只校验 [start, end) 区间, 区间外全部为 False."""
  a, b = _nan_data()
  n = 3
  actual = _calc(a, b, n, False, start=start, end=end, flags=flags)
  _assert_eq(
    actual,
    _expected(a, b, n, False, start=start, end=end, flags=flags),
  )
  s, e = _calc_range(len(a), start, end)
  assert not actual[:s].any()
  assert not actual[e:].any()


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("flags", [0, al.FLAG_SKIP_NAN])
def test_longcross_start_end_with_groups(groups, flags):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  a, b = _nan_data()
  n = 3
  start, end = 6, 25
  actual = _calc(
    a, b, n, False, start=start, end=end, groups=groups, flags=flags
  )
  _assert_eq(
    actual,
    _expected(
      a, b, n, False, start=start, end=end, groups=groups, flags=flags
    ),
  )
  chunk_size = len(a) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert not actual[lo : lo + start].any()
    assert not actual[lo + end : lo + chunk_size].any()


def test_longcross_edge_cases():
  """相等序列 / 超长 n / 空数组 / 单元素 / 手工示例 / 长度不匹配报错."""
  equal = np.full(20, 3.0)
  assert not _calc(equal, equal, 2, False, flags=0).any()
  assert not _calc(equal, equal, 2, True, flags=0).any()
  assert _calc(equal, equal, 0, False, flags=0).all()
  assert _calc(equal, equal, 0, True, flags=0).all()

  a, b = _clean_data()
  assert not _calc(a, b, len(a) + 5, False, flags=0).any()
  assert not _calc(a, b, len(a) + 5, True, flags=0).any()

  empty = np.empty(0, dtype=np.float64)
  _assert_eq(_calc(empty, empty, 2, False, flags=0), empty.astype(bool))

  single = np.array([3.0])
  _assert_eq(_calc(single, single, 0, False, flags=0), np.array([True]))
  _assert_eq(_calc(single, single, 1, False, flags=0), np.array([False]))

  manual_a = np.array([1.0, 2.0, 3.0, 6.0])
  manual_b = np.full(4, 5.0)
  _assert_eq(
    _calc(manual_a, manual_b, 2, False, flags=0),
    np.array([False, False, False, True]),
  )

  with pytest.raises(ValueError):
    _calc(np.ones(5), np.ones(3), 2, False)
