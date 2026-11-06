# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""FILTER (过滤连续重复信号) 集成测试.

算法简介
--------
FILTER 对布尔条件序列做「信号冷却」: 输出 True 后, 接下来 ``periods`` 个
位置的信号全部被抑制, 直到冷却结束才允许下一个 True::

    FILTER(cond, p)[i] = cond[i] and (i - last_signal > p)

其中 ``last_signal`` 是最近一次发出信号的位置 (初始为 ``-p-1``, 保证序列
开头即可触发). 等价地说, 两个相邻输出 True 的位置间隔必须大于 ``periods``.
这是通达信 FILTER 的语义: 条件连续成立时只保留第一次和冷却结束后的第一次.

``periods`` 只影响冷却长度, 不引入滚动窗口, 因此 ``FLAG_STRICTLY_CYCLE``
没有作用; 输入是布尔序列, 没有 nan 概念, ``FLAG_SKIP_NAN`` 同样不改变结果.
输出始终是布尔值, 区间外一律为 False.

``groups`` 按 group-major 切成等长 chunk, 每个 chunk **独立**从块首开始
维护冷却状态; start/end 是每个 chunk 内的相对下标. 注意 start 之前的信号
虽然不输出, 但仍会影响 ``[start, end)`` 区间的冷却状态, 即「全量计算后只
保留区间」. 测试只校验 ``[start, end)`` 区间的值.

对比基准 ``_filter_ref`` 用 numpy 独立复现上述信号间隔规则, 不调用 alpha,
也不实现 SKIP_NAN / Group / start-end 机制; 这些上下文语义由 ``_expected``
单独描述: 按 chunk 全量计算再屏蔽区间外.

`set_ctx` 只覆盖传入的参数, 所以每次调用都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> cond = np.array([True, False, True, True, False, True, False, True, True, True])
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. ``periods=2``: 每个 True 之后 2 期内的信号被抑制::

    >>> al.FILTER(cond, 2).tolist()
    [True, False, False, True, False, False, False, True, False, False]

2. ``periods=1`` 时冷却更短, 允许更密集的信号::

    >>> al.FILTER(cond, 1).tolist()
    [True, False, True, False, False, True, False, True, False, True]

3. 全 True 序列 + ``groups``: 每块长度为 4, 各自从块首重新起步::

    >>> ones = np.ones(8, dtype=bool)
    >>> al.FILTER(ones, 2).tolist()
    [True, False, False, True, False, False, True, False]
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> al.FILTER(ones, 2).tolist()
    [True, False, False, True, True, False, False, True]

4. ``start`` / ``end``: start 之前的信号仍影响冷却状态::

    >>> al.set_ctx(start=2, end=6, groups=1, flags=0)
    >>> al.FILTER(ones, 2).tolist()
    [False, False, False, True, False, False, False, False]

5. 四种 flag 组合结果一致::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> al.FILTER(cond, 2).tolist()
    [True, False, False, True, False, False, False, True, False, False]
"""

import numpy as np
import pytest

import alpha as al

N = 120
PERIODS = [1, 2, 3, 5, 10, 25]
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


def _filter_ref(cond, periods):
  """信号冷却参考实现: 相邻两个 True 的间隔必须大于 periods."""
  cond = np.asarray(cond, dtype=bool)
  out = np.zeros(len(cond), dtype=bool)
  last = -periods - 1
  for i, flag in enumerate(cond):
    if flag and i - last > periods:
      out[i] = True
      last = i
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
  """按 chunk 全量计算参考值, 再把 [start, end) 之外全部置 False."""
  cond = np.asarray(cond, dtype=bool)
  parts = []
  for chunk in np.split(cond, groups):
    n = len(chunk)
    s, e = _calc_range(n, start, end)
    ref = _filter_ref(chunk, periods)
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
    al.FILTER(np.asarray(cond, dtype=bool), periods), dtype=bool
  )


def _cond_data(n=N):
  rng = np.random.default_rng(20261109)
  return rng.random(n) < 0.35


def _assert_eq(actual, expected):
  assert actual.shape == expected.shape
  assert actual.dtype == np.bool_
  np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("periods", PERIODS)
def test_filter_matches_reference(periods):
  """flags=0, groups=1: 与 numpy 信号冷却参考实现逐点比较."""
  cond = _cond_data()
  actual = _calc(cond, periods, flags=0)
  _assert_eq(actual, _filter_ref(cond, periods))
  _assert_eq(actual, _expected(cond, periods, flags=0))
  assert (cond[actual]).all()

  fired = np.flatnonzero(actual)
  if len(fired) > 1:
    assert (np.diff(fired) > periods).all()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_filter_flags_have_no_effect(flags):
  """四种 flag 组合: 布尔输入没有 nan, FILTER 也没有滚动窗口."""
  cond = _cond_data()
  periods = 3
  _assert_eq(_calc(cond, periods, flags=flags), _calc(cond, periods, flags=0))
  _assert_eq(_calc(cond, periods, flags=flags), _filter_ref(cond, periods))


@pytest.mark.parametrize("periods", PERIODS)
def test_filter_all_true_signals_are_periodic(periods):
  """全 True 输入: 输出位置为 0, periods+1, 2*(periods+1), ..."""
  cond = np.ones(N, dtype=bool)
  actual = _calc(cond, periods, flags=0)
  expected = _filter_ref(cond, periods)
  _assert_eq(actual, expected)
  np.testing.assert_array_equal(
    np.flatnonzero(actual), np.arange(0, N, periods + 1)
  )


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("periods", [1, 2, 5])
def test_filter_groups(groups, periods):
  """group-major 分块: 每个 chunk 独立维护冷却状态, 从块首重新起步."""
  cond = _cond_data()
  actual = _calc(cond, periods, groups=groups, flags=0)
  _assert_eq(actual, _expected(cond, periods, groups=groups, flags=0))

  chunk_size = len(cond) // groups
  for g in range(groups):
    lo = g * chunk_size
    _assert_eq(
      actual[lo : lo + chunk_size], _filter_ref(cond[lo : lo + chunk_size], periods)
    )


@pytest.mark.parametrize("start,end", START_END)
def test_filter_start_end(start, end):
  """start/end: 状态从 chunk 头累计, 只保留 [start, end) 的输出."""
  cond = _cond_data()
  periods = 4
  actual = _calc(cond, periods, start=start, end=end, flags=0)
  _assert_eq(
    actual, _expected(cond, periods, start=start, end=end, flags=0)
  )
  s, e = _calc_range(len(cond), start, end)
  assert not actual[:s].any()
  assert not actual[e:].any()


@pytest.mark.parametrize("groups", GROUPS)
def test_filter_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  cond = _cond_data()
  periods = 3
  start, end = 6, 25
  actual = _calc(
    cond, periods, start=start, end=end, groups=groups, flags=0
  )
  _assert_eq(
    actual,
    _expected(
      cond, periods, start=start, end=end, groups=groups, flags=0
    ),
  )
  chunk_size = len(cond) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert not actual[lo : lo + start].any()
    assert not actual[lo + end : lo + chunk_size].any()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_filter_skip_nan_with_start_end(flags):
  """四种 flag 组合叠加 start/end, 结果仍等于全量计算后屏蔽区间外."""
  cond = _cond_data()
  ctx = dict(start=10, end=90, flags=flags)
  actual = _calc(cond, 4, **ctx)
  _assert_eq(actual, _expected(cond, 4, **ctx))


def test_filter_edge_cases():
  """全 False / 单元素 / 超长冷却 / 空数组 / list 输入."""
  zeros = np.zeros(N, dtype=bool)
  assert not _calc(zeros, 3, flags=0).any()

  one = np.array([True])
  _assert_eq(_calc(one, 5, flags=0), np.array([True]))

  long_periods = _calc(np.ones(N, dtype=bool), N + 10, flags=0)
  expected = np.zeros(N, dtype=bool)
  expected[0] = True
  _assert_eq(long_periods, expected)

  empty = np.empty(0, dtype=bool)
  _assert_eq(_calc(empty, 2, flags=0), empty)

  al.set_ctx(start=0, end=0, groups=1, flags=0)
  result = al.FILTER(
    [np.ones(4, dtype=bool), np.array([True, False, True, True])], 2
  )
  assert isinstance(result, list)
  _assert_eq(np.asarray(result[0]), _filter_ref(np.ones(4, dtype=bool), 2))
  _assert_eq(
    np.asarray(result[1]),
    _filter_ref(np.array([True, False, True, True]), 2),
  )
