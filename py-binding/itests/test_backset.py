# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""BACKSET (信号向前回填) 集成测试.

算法简介
--------
BACKSET 把每个成立的信号向前回填 `periods` 个 bar (含信号本身), 多个信号的
回填区间取并集::

    BACKSET[i] = any(cond[i : i + periods])

即位置 i 为真当且仅当它之后的 `periods` 个位置内出现过信号. ``periods = 1``
时输出等于输入, ``periods = 0`` 时全部为 False. 回填只在 ``[start, end)``
区间内进行: ``end`` 之后 (含 end) 的信号不可见, ``start`` 之前的位置不写,
信号在 start 附近的回填区间会被 start 截断.

输入与输出都是布尔数组, 没有 nan, 因此 ``FLAG_SKIP_NAN`` /
``FLAG_STRICTLY_CYCLE`` 都不应改变结果; 输出是 bool 而非浮点, 区间外用 False
表示 (bool 无法表达 nan). 参考实现 ``_backset_ref`` 用"向前看窗口"的 numpy
实现, ``_backset_backward_ref`` 用"逐信号回填"的独立实现交叉验证; 两者只描述
算法逻辑, 不调用 alpha, 也不实现 Group 机制: Group 由测试按 chunk 分块计算
(每个 chunk 的回填不跨边界), start/end 只校验 ``[start, end)`` 区间的值.

`set_ctx` 只覆盖传入的参数, 因此每次调用前都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=120, suppress=True)
    >>> cond = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype=bool)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算: 信号在下标 2 / 5 / 8, 各自向前回填 3 个 bar::

    >>> al.BACKSET(cond, 3)
    array([ True,  True,  True,  True,  True,  True,  True,  True,  True, False])

2. ``periods = 2`` 时回填区间更短::

    >>> al.BACKSET(cond, 2)
    array([False,  True,  True, False,  True,  True, False,  True,  True, False])

3. ``periods = 1`` 时输出等于输入::

    >>> al.BACKSET(cond, 1)
    array([False, False,  True, False, False,  True, False, False,  True, False])

4. ``periods = 0`` 时全部为 False::

    >>> al.BACKSET(cond, 0)
    array([False, False, False, False, False, False, False, False, False, False])

5. 两个 flag 同时设置也不改变布尔序列的结果::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> al.BACKSET(cond, 3)
    array([ True,  True,  True,  True,  True,  True,  True,  True,  True, False])

6. ``groups`` 把一维数据按 group-major 切块, 回填不跨 chunk 边界::

    >>> s12 = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0], dtype=bool)
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> al.BACKSET(s12, 3)
    array([ True,  True,  True, False, False, False,  True,  True,  True,  True, False, False])

7. ``start`` / ``end`` 只计算 ``[start, end)`` 区间: 只有 ``cond[5]`` 可见,
   回填 [3, 5]; ``cond[8]`` 在 end 之外不参与, 区间外全部为 False::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> al.BACKSET(cond, 3)
    array([False, False, False,  True,  True,  True, False, False, False, False])
"""

import numpy as np
import pytest

import alpha as al

N = 120
PERIODS = [1, 2, 3, 4, 5, 10, 20]
GROUPS = [1, 2, 3, 4, 6]
START_END = [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5)]
FLAG_COMBOS = [
  0,
  al.FLAG_STRICTLY_CYCLE,
  al.FLAG_SKIP_NAN,
  al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
]


# ---------------------------------------------------------------------------
# numpy 对比实现
# ---------------------------------------------------------------------------


def _backset_ref(cond, periods, start, end):
  """BACKSET 参考实现: r[i] = any(cond[i : i + periods]), 只看 [start, end).

  窗口在 end 处截断, 区间外的位置保持 False; ``periods = 0`` 时全 False.
  """
  cond = np.asarray(cond, dtype=bool)
  out = np.zeros(len(cond), dtype=bool)
  if periods == 0:
    return out
  for i in range(start, end):
    out[i] = cond[i : min(i + periods, end)].any()
  return out


def _backset_backward_ref(cond, periods, start, end):
  """独立参考实现: 每个信号把 [j - periods + 1, j] 回填为 True."""
  cond = np.asarray(cond, dtype=bool)
  out = np.zeros(len(cond), dtype=bool)
  if periods == 0:
    return out
  for j in range(start, end):
    if cond[j]:
      out[max(start, j - periods + 1) : j + 1] = True
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


def _expected(cond, periods, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 独立计算参考值; 回填区间被 [start, end) 与 chunk 边界截断."""
  cond = np.asarray(cond, dtype=bool)
  parts = []
  for chunk in np.split(cond, groups):
    n = len(chunk)
    s, e = _calc_range(n, start, end)
    parts.append(_backset_ref(chunk, periods, s, e))
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(cond, periods, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.BACKSET(np.asarray(cond, dtype=bool), periods), dtype=bool
  )


def _bool_data(n=N):
  rng = np.random.default_rng(20261110)
  return (rng.random(n) < 0.25).astype(bool)


def _assert_equal(actual, expected):
  assert actual.shape == expected.shape
  assert actual.dtype == np.bool_
  np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("periods", PERIODS)
def test_backset_matches_reference(periods):
  """flags=0, groups=1: 与向前看窗口及逐信号回填实现逐点比较."""
  cond = _bool_data()
  actual = _calc(cond, periods, flags=0)
  _assert_equal(actual, _expected(cond, periods, flags=0))
  _assert_equal(actual, _backset_ref(cond, periods, 0, len(cond)))
  _assert_equal(
    actual, _backset_backward_ref(cond, periods, 0, len(cond))
  )


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_backset_periods_one(flags):
  """periods=1: 每个信号只回填自身, 输出等于输入."""
  cond = _bool_data()
  actual = _calc(cond, 1, flags=flags)
  _assert_equal(actual, cond)
  _assert_equal(actual, _expected(cond, 1, flags=flags))


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("start,end", [(0, 0), (10, 80)])
def test_backset_periods_zero(groups, start, end):
  """periods=0: 全部为 False, 与 groups / start-end 无关."""
  cond = _bool_data()
  actual = _calc(cond, 0, start=start, end=end, groups=groups, flags=0)
  _assert_equal(actual, np.zeros(len(cond), dtype=bool))
  _assert_equal(
    actual, _expected(cond, 0, start=start, end=end, groups=groups, flags=0)
  )


@pytest.mark.parametrize("flags", FLAG_COMBOS)
@pytest.mark.parametrize("periods", [3, 6])
def test_backset_all_flag_combinations(periods, flags):
  """四种 flag 组合对布尔输入的结果完全一致."""
  cond = _bool_data()
  actual = _calc(cond, periods, flags=flags)
  _assert_equal(actual, _expected(cond, periods, flags=flags))
  _assert_equal(actual, _calc(cond, periods, flags=0))


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("periods", [2, 4])
def test_backset_groups(groups, periods):
  """group-major 分块: 回填区间不跨越 chunk 边界."""
  cond = _bool_data()
  actual = _calc(cond, periods, groups=groups, flags=0)
  _assert_equal(actual, _expected(cond, periods, groups=groups, flags=0))

  chunk_size = len(cond) // groups
  for g in range(groups):
    lo = g * chunk_size
    chunk = cond[lo : lo + chunk_size]
    _assert_equal(
      actual[lo : lo + chunk_size],
      _backset_ref(chunk, periods, 0, chunk_size),
    )


@pytest.mark.parametrize("start,end", START_END)
def test_backset_start_end(start, end):
  """start/end: 仅 [start, end) 有结果, 区间外必须全是 False."""
  cond = _bool_data()
  periods = 4
  actual = _calc(cond, periods, start=start, end=end, flags=0)
  _assert_equal(actual, _expected(cond, periods, start=start, end=end, flags=0))

  s, e = _calc_range(len(cond), start, end)
  assert not actual[:s].any()
  assert not actual[e:].any()


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_backset_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  cond = _bool_data()
  periods = 4
  start, end = 6, 25
  actual = _calc(cond, periods, start=start, end=end, groups=groups, flags=0)
  _assert_equal(
    actual,
    _expected(cond, periods, start=start, end=end, groups=groups, flags=0),
  )

  chunk_size = len(cond) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert not actual[lo : lo + start].any()
    assert not actual[lo + end : lo + chunk_size].any()


def test_backset_ignores_signals_after_end():
  """end 之后的信号不可见; 全量计算时该信号正常回填."""
  cond = np.zeros(N, dtype=bool)
  cond[60] = True
  periods = 5
  partial = _calc(cond, periods, start=10, end=50, flags=0)
  assert not partial.any()
  _assert_equal(
    partial, _expected(cond, periods, start=10, end=50, flags=0)
  )

  full = _calc(cond, periods, flags=0)
  expected = np.zeros(N, dtype=bool)
  expected[56:61] = True
  _assert_equal(full, expected)


def test_backset_signal_near_start_is_clipped():
  """信号紧邻 start 时, 回填区间被 start 截断, 不写到区间外."""
  cond = np.zeros(N, dtype=bool)
  cond[20] = True
  start, end = 18, 60
  periods = 6
  actual = _calc(cond, periods, start=start, end=end, flags=0)
  _assert_equal(
    actual, _expected(cond, periods, start=start, end=end, flags=0)
  )
  assert actual[start:21].all()
  assert not actual[:start].any()
  assert not actual[21:end].any()


def test_backset_skip_nan_with_start_end():
  """SKIP_NAN + start/end: 结果与不带 flag 时相同."""
  cond = _bool_data()
  periods = 4
  ctx = dict(start=10, end=90, flags=al.FLAG_SKIP_NAN)
  actual = _calc(cond, periods, **ctx)
  _assert_equal(actual, _expected(cond, periods, **ctx))
  _assert_equal(actual, _calc(cond, periods, start=10, end=90, flags=0))
  assert not actual[:10].any()
  assert not actual[90:].any()


def test_backset_edge_cases():
  """窗口长于数据; 全 True / 全 False 序列; 相邻信号的回填区间重叠."""
  cond = _bool_data()
  long_periods = len(cond) + 1
  _assert_equal(
    _calc(cond, long_periods, flags=0),
    _expected(cond, long_periods, flags=0),
  )

  for value in (False, True):
    dense = np.full(len(cond), value, dtype=bool)
    for periods in (3, 6):
      for flags in FLAG_COMBOS:
        _assert_equal(
          _calc(dense, periods, flags=flags),
          _expected(dense, periods, flags=flags),
        )

  overlap = np.zeros(N, dtype=bool)
  overlap[20] = True
  overlap[22] = True
  overlap[23] = True
  for periods in (3, 7):
    _assert_equal(
      _calc(overlap, periods, flags=0), _expected(overlap, periods, flags=0)
    )


def test_backset_example_values():
  """固定小样例: 显式校验多个信号的回填并集."""
  cond = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype=bool)
  expected = np.array(
    [False, True, True, False, True, True, False, True, True, False],
    dtype=bool,
  )
  _assert_equal(_calc(cond, 2, flags=0), expected)
  _assert_equal(_backset_ref(cond, 2, 0, len(cond)), expected)
  _assert_equal(_backset_backward_ref(cond, 2, 0, len(cond)), expected)
