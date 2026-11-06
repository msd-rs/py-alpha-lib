# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""ALL (滚动窗口内条件全部成立) 集成测试.

算法简介
--------
ALL 判断布尔序列在最近 `periods` 个位置中是否全部为真::

    ALL[i] = all(cond[i - periods + 1 : i + 1])

窗口未满 (``i < periods - 1``) 时只用已有的 1..periods 个值判断, 等价于
pandas ``rolling(periods, min_periods=1).min() == 1``. ``periods = 0`` 表示
从序列开头累计判断: ``ALL[i] = all(cond[: i + 1])``, 一旦出现过 False, 之后
恒为 False. 设置 ``FLAG_STRICTLY_CYCLE`` 后窗口未满的前 ``periods - 1`` 个
位置输出 False; 累计模式没有窗口, 不受该 flag 影响.

输入与输出都是布尔数组, 没有 nan, 因此 ``FLAG_SKIP_NAN`` 不应改变结果; 输出
是 bool 而非浮点, ``[start, end)`` 区间外以及窗口未满的位置用 False 表示
(bool 无法表达 nan). 参考实现 ``_all_ref`` 用 pandas ``rolling`` / numpy
``logical_and.accumulate`` 独立实现, 只描述算法逻辑, 不调用 alpha, 也不实现
Group / start-end 机制: Group 由测试按 chunk 分块计算 (每个 chunk 的窗口不跨
边界), start/end 只校验 ``[start, end)`` 区间的值, 但滚动窗口与累计状态仍可
引用 start 之前的数据.

`set_ctx` 只覆盖传入的参数, 因此每次调用前都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=120, suppress=True)
    >>> cond = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 1], dtype=bool)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算 (窗口未满时用已有值判断)::

    >>> al.ALL(cond, 3)
    array([ True,  True,  True, False, False, False, False, False, False,  True])

2. ``FLAG_STRICTLY_CYCLE``: 前 2 个位置 (窗口未满) 输出 False::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> al.ALL(cond, 3)
    array([False, False,  True, False, False, False, False, False, False,  True])

3. 窗口为 2, 只有连续两个 True 的位置为真::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> al.ALL(cond, 2)
    array([ True,  True,  True, False, False,  True, False, False,  True,  True])

4. ``periods = 0`` 为累计判断, 出现过 False 后恒为 False::

    >>> al.ALL(cond, 0)
    array([ True,  True,  True, False, False, False, False, False, False, False])

5. ``FLAG_SKIP_NAN`` 对布尔输入无影响::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> al.ALL(cond, 3)
    array([ True,  True,  True, False, False, False, False, False, False,  True])

6. ``groups`` 把一维数据按 group-major 切块, 每块的窗口不跨边界::

    >>> c12 = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1], dtype=bool)
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> al.ALL(c12, 2)
    array([ True,  True,  True, False, False, False,  True, False, False, False, False,  True])

7. ``start`` / ``end`` 只计算 ``[start, end)`` 区间, 区间外全部为 False::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> al.ALL(cond, 2)
    array([False, False, False, False, False,  True, False, False, False, False])
"""

import numpy as np
import pandas as pd
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
# numpy / pandas 对比实现
# ---------------------------------------------------------------------------


def _all_ref(cond, periods, *, strict=False):
  """ALL 参考实现: rolling(periods, min_periods=1).min() == 1 或累计与.

  ``periods = 0`` 表示从序列开头累计; ``strict`` 时前 ``periods - 1`` 个
  位置输出 False (累计模式没有窗口, 不受影响).
  """
  cond = np.asarray(cond, dtype=bool)
  if periods == 0:
    return np.logical_and.accumulate(cond)

  out = (
    pd.Series(cond.astype(np.float64))
    .rolling(periods, min_periods=1)
    .min()
    .to_numpy()
    == 1.0
  )
  if strict:
    out[: periods - 1] = False
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
  """按 chunk 全量计算参考值, 再只保留 [start, end) 区间.

  布尔输出无法表达 nan, 区间外一律为 False; 窗口与累计状态仍可引用 start
  之前的数据, 因此先全量计算再屏蔽区间外.
  """
  cond = np.asarray(cond, dtype=bool)
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  parts = []
  for chunk in np.split(cond, groups):
    n = len(chunk)
    s, e = _calc_range(n, start, end)
    ref = _all_ref(chunk, periods, strict=strict)
    ref[:s] = False
    ref[e:] = False
    parts.append(ref)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(cond, periods, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.ALL(np.asarray(cond, dtype=bool), periods), dtype=bool
  )


def _bool_data(n=N):
  rng = np.random.default_rng(20261106)
  return (rng.random(n) < 0.55).astype(bool)


def _assert_equal(actual, expected):
  assert actual.shape == expected.shape
  assert actual.dtype == np.bool_
  np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("periods", PERIODS)
def test_all_matches_pandas_rolling(periods):
  """flags=0, groups=1: 与 pandas rolling min 逐点比较."""
  cond = _bool_data()
  actual = _calc(cond, periods, flags=0)
  _assert_equal(actual, _expected(cond, periods, flags=0))
  _assert_equal(actual, _all_ref(cond, periods))


@pytest.mark.parametrize("periods", PERIODS)
def test_all_strictly_cycle(periods):
  """FLAG_STRICTLY_CYCLE: 前 periods-1 个位置为 False, 其余与参考一致."""
  cond = _bool_data()
  flags = al.FLAG_STRICTLY_CYCLE
  actual = _calc(cond, periods, flags=flags)
  _assert_equal(actual, _expected(cond, periods, flags=flags))
  _assert_equal(actual, _all_ref(cond, periods, strict=True))
  if periods > 1:
    assert not actual[: periods - 1].any()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_all_cumulative(flags):
  """periods=0: 从序列开头累计, 不受任何 flag 影响."""
  cond = _bool_data()
  actual = _calc(cond, 0, flags=flags)
  _assert_equal(actual, _expected(cond, 0, flags=flags))
  _assert_equal(actual, np.logical_and.accumulate(cond))


@pytest.mark.parametrize("flags", FLAG_COMBOS)
@pytest.mark.parametrize("periods", [3, 6])
def test_all_all_flag_combinations(periods, flags):
  """四种 flag 组合: SKIP_NAN 对布尔输入无影响."""
  cond = _bool_data()
  actual = _calc(cond, periods, flags=flags)
  _assert_equal(actual, _expected(cond, periods, flags=flags))
  plain_flags = flags & ~al.FLAG_SKIP_NAN
  _assert_equal(actual, _calc(cond, periods, flags=plain_flags))


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_all_groups(groups, flags):
  """group-major 分块: 每个 chunk 的滚动窗口不跨边界."""
  cond = _bool_data()
  periods = 4
  actual = _calc(cond, periods, groups=groups, flags=flags)
  _assert_equal(actual, _expected(cond, periods, groups=groups, flags=flags))

  chunk_size = len(cond) // groups
  for g in range(groups):
    lo = g * chunk_size
    chunk = cond[lo : lo + chunk_size]
    _assert_equal(
      actual[lo : lo + chunk_size],
      _all_ref(chunk, periods, strict=bool(flags)),
    )


@pytest.mark.parametrize("groups", GROUPS)
def test_all_cumulative_with_groups(groups):
  """累计判断也按 chunk 独立, 每块从块首重新开始."""
  cond = _bool_data()
  actual = _calc(cond, 0, groups=groups, flags=0)
  _assert_equal(actual, _expected(cond, 0, groups=groups, flags=0))

  chunk_size = len(cond) // groups
  for g in range(groups):
    lo = g * chunk_size
    chunk = cond[lo : lo + chunk_size]
    _assert_equal(
      actual[lo : lo + chunk_size], np.logical_and.accumulate(chunk)
    )


@pytest.mark.parametrize("start,end", START_END)
def test_all_start_end(start, end):
  """start/end: 仅 [start, end) 有结果, 区间外必须全是 False."""
  cond = _bool_data()
  periods = 4
  actual = _calc(cond, periods, start=start, end=end, flags=0)
  _assert_equal(actual, _expected(cond, periods, start=start, end=end, flags=0))

  s, e = _calc_range(len(cond), start, end)
  assert not actual[:s].any()
  assert not actual[e:].any()


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_all_start_end_with_groups(groups):
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


def test_all_window_uses_pre_start_data():
  """滚动窗口可引用 start 之前的数据: start 处不因缺历史而误判."""
  cond = np.ones(N, dtype=bool)
  periods = 5
  start, end = 10, 30
  actual = _calc(cond, periods, start=start, end=end, flags=0)
  _assert_equal(
    actual, _expected(cond, periods, start=start, end=end, flags=0)
  )
  assert actual[start:end].all()


def test_all_cumulative_uses_pre_start_data():
  """累计状态包含 start 之前的数据: 早先的 False 不会因 start 被丢弃."""
  cond = np.ones(N, dtype=bool)
  cond[2] = False
  start, end = 10, 30
  actual = _calc(cond, 0, start=start, end=end, flags=0)
  _assert_equal(actual, _expected(cond, 0, start=start, end=end, flags=0))
  assert not actual[start:end].any()
  assert not actual[:start].any()  # bool 输出的区间外一律为 False


def test_all_edge_cases():
  """periods=1 等于输入; 窗口长于数据等于累计; 全 True / 全 False 序列."""
  cond = _bool_data()
  _assert_equal(_calc(cond, 1, flags=0), cond)
  _assert_equal(
    _calc(cond, len(cond) + 1, flags=0), np.logical_and.accumulate(cond)
  )

  for value in (False, True):
    dense = np.full(len(cond), value, dtype=bool)
    for periods in (3, 6):
      for flags in (0, al.FLAG_STRICTLY_CYCLE, al.FLAG_SKIP_NAN):
        _assert_equal(
          _calc(dense, periods, flags=flags),
          _expected(dense, periods, flags=flags),
        )
