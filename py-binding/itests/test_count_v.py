# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""COUNT_V (逐点可变窗口的条件计数) 集成测试.

算法简介
--------
COUNT_V 是 COUNT 的变周期版本, 每个位置 ``i`` 用 ``periods[i]`` 决定窗口
长度, 统计该窗口内条件为真的个数::

    COUNT_V(cond, p)[i] = sum(cond[max(0, i + 1 - p[i]) : i + 1])

``p[i] = 0`` 表示从序列 (chunk) 开头累计计数, 即 ``cumsum(cond)[i]``;
``p[i] > 0`` 但窗口尚未填满时用已有的部分窗口计数. 设置
``FLAG_STRICTLY_CYCLE`` 后, 窗口未满 (``i < p[i] - 1``) 的位置输出 nan;
累计模式没有窗口, 不受该标志影响. 输入是布尔数组, 没有 nan 值, 因此
``FLAG_SKIP_NAN`` 不应改变结果; 测试仍覆盖四种 flag 组合以确认这一点.

``groups`` 按 group-major 切成等长 chunk, 每个 chunk 的窗口与累计都独立,
不跨边界; ``periods`` 数组同样按 chunk 切分. start/end 是每个 chunk 内的
相对下标, 只计算 [start, end) 区间, 区间外一律为 nan, 但窗口仍可引用
start 之前的数据.

对比基准 ``_count_v_ref`` 用 numpy 前缀和独立实现, 只描述算法逻辑, 不调用
alpha, 也不实现 SKIP_NAN / Group / start-end 机制; 这些上下文语义由
``_expected`` 单独描述: 按 chunk 全量计算再屏蔽区间外.

`set_ctx` 只覆盖传入的参数, 因此每次调用前都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> cond = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=bool)
    >>> p = np.array([0, 1, 2, 3, 0, 1, 2, 3, 4, 5], dtype=np.uint64)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算, 每个位置使用自己的窗口长度::

    >>> al.COUNT_V(cond, p).tolist()
    [1.0, 0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 2.0, 3.0]

2. ``FLAG_STRICTLY_CYCLE``: 窗口未满的位置输出 nan::

    >>> p3 = np.full(10, 3, dtype=np.uint64)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> al.COUNT_V(cond, p3).tolist()
    [nan, nan, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0]

3. ``p = 0`` 为从头累计计数::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> al.COUNT_V(cond, np.zeros(10, dtype=np.uint64)).tolist()
    [1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 6.0]

4. ``FLAG_SKIP_NAN`` 对布尔输入无影响::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> al.COUNT_V(cond, p).tolist()
    [1.0, 0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 2.0, 3.0]

5. ``groups`` 把数据按 group-major 切块, 每块重新累计::

    >>> c12 = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1], dtype=bool)
    >>> p12 = np.full(12, 3, dtype=np.uint64)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> al.COUNT_V(c12, p12).tolist()
    [1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 2.0]
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> al.COUNT_V(c12, p12).tolist()
    [1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0]

6. ``start`` / ``end`` 只输出 [start, end) 区间::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> al.COUNT_V(cond, p).tolist()
    [nan, nan, nan, 2.0, 3.0, 1.0, 1.0, 1.0, nan, nan]
"""

import numpy as np
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
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


def _count_v_ref(cond, periods, *, strict=False):
  """COUNT_V 参考实现: 前缀和上的逐点窗口计数.

  ``p = 0`` 表示累计计数; ``strict`` 时窗口未满的前 ``p - 1`` 个位置
  输出 nan (累计模式不受影响).
  """
  c = np.asarray(cond, dtype=np.float64)
  p = np.asarray(periods, dtype=np.int64)
  i = np.arange(len(c))
  pref = np.concatenate(([0.0], np.cumsum(c)))
  lo = np.where(p == 0, 0, np.maximum(i + 1 - p, 0))
  out = pref[i + 1] - pref[lo]
  if strict:
    out[(p > 0) & (i < p - 1)] = np.nan
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

  布尔输入没有 nan, ``FLAG_SKIP_NAN`` 不改变参考值; 窗口在每个 chunk 内
  独立, 但可引用 start 之前的数据.
  """
  cond = np.asarray(cond, dtype=bool)
  p = np.asarray(periods, dtype=np.int64)
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  parts = []
  for chunk_cond, chunk_p in zip(np.split(cond, groups), np.split(p, groups)):
    n = len(chunk_cond)
    s, e = _calc_range(n, start, end)
    ref = _count_v_ref(chunk_cond, chunk_p, strict=strict)
    ref[:s] = np.nan
    ref[e:] = np.nan
    parts.append(ref)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(cond, periods, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.COUNT_V(
      np.asarray(cond, dtype=bool),
      np.asarray(periods, dtype=np.uint64),
    ),
    dtype=np.float64,
  )


def _bool_data(n=N):
  rng = np.random.default_rng(20261114)
  return rng.random(n) < 0.4


def _periods_data(n=N, max_p=12):
  rng = np.random.default_rng(20261115)
  return rng.integers(0, max_p + 1, size=n).astype(np.uint64)


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


def test_count_v_matches_reference():
  """flags=0, groups=1: 与 numpy 前缀和参考实现逐点比较."""
  cond = _bool_data()
  p = _periods_data()
  actual = _calc(cond, p, flags=0)
  _assert_close(actual, _count_v_ref(cond, p))
  _assert_close(actual, _expected(cond, p, flags=0))
  assert np.isfinite(actual).all()
  assert (actual >= 0).all()
  np.testing.assert_array_equal(actual, np.floor(actual))


def test_count_v_strictly_cycle():
  """FLAG_STRICTLY_CYCLE: 窗口未满 p[i] 的位置输出 nan."""
  cond = _bool_data()
  p = _periods_data()
  flags = al.FLAG_STRICTLY_CYCLE
  actual = _calc(cond, p, flags=flags)
  _assert_close(actual, _count_v_ref(cond, p, strict=True))
  _assert_close(actual, _expected(cond, p, flags=flags))

  short = (p > 0) & (np.arange(len(cond)) < p.astype(np.int64) - 1)
  assert np.isnan(actual[short]).all()
  assert np.isfinite(actual[~short]).all()


def test_count_v_zero_periods_is_cumsum():
  """p 全为 0: 结果等于 cumsum, 累计模式不受 STRICTLY_CYCLE 影响."""
  cond = _bool_data()
  p = np.zeros(len(cond), dtype=np.uint64)
  expected = np.cumsum(cond, dtype=np.float64)
  for flags in FLAG_COMBOS:
    _assert_close(_calc(cond, p, flags=flags), expected)
    _assert_close(_calc(cond, p, flags=flags), _expected(cond, p, flags=flags))


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_count_v_all_flag_combinations(flags):
  """四种 flag 组合: SKIP_NAN 无效, STRICTLY_CYCLE 决定是否输出部分窗口."""
  cond = _bool_data()
  p = _periods_data()
  actual = _calc(cond, p, flags=flags)
  _assert_close(actual, _expected(cond, p, flags=flags))
  _assert_close(
    actual, _count_v_ref(cond, p, strict=bool(flags & al.FLAG_STRICTLY_CYCLE))
  )


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_count_v_groups(groups, flags):
  """group-major 分块: 每个 chunk 独立累计, periods 也按块切分."""
  cond = _bool_data()
  p = _periods_data()
  actual = _calc(cond, p, groups=groups, flags=flags)
  _assert_close(actual, _expected(cond, p, groups=groups, flags=flags))

  chunk_size = len(cond) // groups
  for g in range(groups):
    lo = g * chunk_size
    _assert_close(
      actual[lo : lo + chunk_size],
      _count_v_ref(
        cond[lo : lo + chunk_size],
        p[lo : lo + chunk_size],
        strict=bool(flags & al.FLAG_STRICTLY_CYCLE),
      ),
    )


@pytest.mark.parametrize("start,end", START_END)
def test_count_v_start_end(start, end):
  """start/end: 只校验 [start, end) 区间, 窗口可引用 start 之前的数据."""
  cond = _bool_data()
  p = _periods_data()
  actual = _calc(cond, p, start=start, end=end, flags=0)
  _assert_close(actual, _expected(cond, p, start=start, end=end, flags=0))
  s, e = _calc_range(len(cond), start, end)
  assert np.isnan(actual[:s]).all()
  assert np.isnan(actual[e:]).all()


@pytest.mark.parametrize("groups", GROUPS)
def test_count_v_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  cond = _bool_data()
  p = _periods_data()
  start, end = 6, 25
  actual = _calc(cond, p, start=start, end=end, groups=groups, flags=0)
  _assert_close(
    actual, _expected(cond, p, start=start, end=end, groups=groups, flags=0)
  )
  chunk_size = len(cond) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


def test_count_v_edge_cases():
  """全 False / 全 True / 超大 p / 单元素 / 空数组 / 长度不匹配报错."""
  cond = _bool_data(20)
  p = _periods_data(20)

  zeros = np.zeros(20, dtype=bool)
  assert (_calc(zeros, p, flags=0) == 0).all()

  ones = np.ones(20, dtype=bool)
  p1 = np.ones(20, dtype=np.uint64)
  _assert_close(_calc(ones, p1, flags=0), np.ones(20))

  p_big = np.full(20, 100, dtype=np.uint64)
  cum = np.cumsum(cond, dtype=np.float64)
  _assert_close(_calc(cond, p_big, flags=0), cum)
  assert np.isnan(_calc(cond, p_big, flags=al.FLAG_STRICTLY_CYCLE)).all()

  one = np.array([True])
  _assert_close(_calc(one, np.array([0], dtype=np.uint64), flags=0), np.array([1.0]))

  empty = np.empty(0, dtype=bool)
  _assert_close(_calc(empty, np.empty(0, dtype=np.uint64), flags=0), np.empty(0))

  with pytest.raises(ValueError):
    _calc(cond, np.zeros(10, dtype=np.uint64), flags=0)
