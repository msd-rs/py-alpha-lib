# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""LAST (连续条件成立周期数) 集成测试.

算法简介
--------
LAST 统计到当前位置为止条件**连续**成立的周期数::

    LAST(cond)[0] = 1 if cond[0] else 0
    LAST(cond)[i] = LAST(cond)[i-1] + 1   if cond[i]
                    0                      otherwise

输出是浮点计数 (0, 1, 2, ...), 条件一旦为 False 立即归零. 输入是布尔序列,
没有 nan 概念, 算子也没有滚动窗口, 因此 ``FLAG_SKIP_NAN`` /
``FLAG_STRICTLY_CYCLE`` 都不改变结果; 四种 flag 组合结果相同.

``groups`` 按 group-major 切成等长 chunk, 每个 chunk 独立从块首重新计数;
start/end 是每个 chunk 内的相对下标. start 之前的条件虽然不输出, 但仍会
影响 ``[start, end)`` 区间的连续计数, 即「全量计算后只保留区间」, 区间外
为 nan. 测试只校验 ``[start, end)`` 区间的值.

对比基准 ``_last_ref`` 用 numpy 独立复现连续计数规则, 不调用 alpha, 也不
实现 SKIP_NAN / Group / start-end 机制; 这些上下文语义由 ``_expected``
单独描述: 按 chunk 全量计算再屏蔽区间外.

`set_ctx` 只覆盖传入的参数, 所以每次调用都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> cond = np.array([True, True, False, True, True, True, False, True])
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算, 条件为 False 处归零::

    >>> al.LAST(cond).tolist()
    [1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0]

2. ``FLAG_SKIP_NAN`` / ``FLAG_STRICTLY_CYCLE`` 不改变结果::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> al.LAST(cond).tolist()
    [1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0]

3. ``start`` / ``end``: start 之前的连续状态仍参与计数::

    >>> al.set_ctx(start=2, end=6, groups=1, flags=0)
    >>> al.LAST(cond).tolist()
    [nan, nan, 0.0, 1.0, 2.0, 3.0, nan, nan]

4. ``groups`` 让每个长度为 4 的 chunk 从 1 重新起步::

    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> al.LAST(cond).tolist()
    [1.0, 2.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0]

5. 全 True 序列等于 1..n::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> al.LAST(np.ones(5, dtype=bool)).tolist()
    [1.0, 2.0, 3.0, 4.0, 5.0]
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


def _last_ref(cond):
  """连续条件成立周期数的参考实现."""
  cond = np.asarray(cond, dtype=bool)
  out = np.empty(len(cond), dtype=np.float64)
  count = 0
  for i, flag in enumerate(cond):
    count = count + 1 if flag else 0
    out[i] = count
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


def _expected(cond, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 全量计算参考值, 再把 [start, end) 之外全部置 nan."""
  cond = np.asarray(cond, dtype=bool)
  parts = []
  for chunk in np.split(cond, groups):
    n = len(chunk)
    s, e = _calc_range(n, start, end)
    ref = _last_ref(chunk)
    ref[:s] = np.nan
    ref[e:] = np.nan
    parts.append(ref)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(cond, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.LAST(np.asarray(cond, dtype=bool)), dtype=np.float64
  )


def _cond_data(n=N):
  rng = np.random.default_rng(20261110)
  return rng.random(n) < 0.55


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


def test_last_matches_reference():
  """flags=0, groups=1: 与 numpy 连续计数参考实现逐点比较."""
  cond = _cond_data()
  actual = _calc(cond, flags=0)
  _assert_close(actual, _last_ref(cond))
  _assert_close(actual, _expected(cond, flags=0))
  assert np.isfinite(actual).all()
  np.testing.assert_array_equal(actual, np.floor(actual))
  _assert_close(actual[~cond], np.zeros(np.count_nonzero(~cond)))
  assert (actual[cond] >= 1).all()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_last_flags_have_no_effect(flags):
  """四种 flag 组合: 布尔输入没有 nan, LAST 也没有滚动窗口."""
  cond = _cond_data()
  _assert_close(_calc(cond, flags=flags), _calc(cond, flags=0))
  _assert_close(_calc(cond, flags=flags), _last_ref(cond))


def test_last_all_true_and_all_false():
  """全 True 输出 1..n; 全 False 输出全 0."""
  ones = np.ones(N, dtype=bool)
  _assert_close(_calc(ones, flags=0), np.arange(1, N + 1, dtype=np.float64))
  zeros = np.zeros(N, dtype=bool)
  _assert_close(_calc(zeros, flags=0), np.zeros(N))


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_last_groups(groups, flags):
  """group-major 分块: 每个 chunk 独立从块首重新计数."""
  cond = _cond_data()
  actual = _calc(cond, groups=groups, flags=flags)
  _assert_close(actual, _expected(cond, groups=groups, flags=flags))

  chunk_size = len(cond) // groups
  for g in range(groups):
    lo = g * chunk_size
    _assert_close(actual[lo : lo + chunk_size], _last_ref(cond[lo : lo + chunk_size]))


@pytest.mark.parametrize("start,end", START_END)
def test_last_start_end(start, end):
  """start/end: 连续状态从 chunk 头累计, 只校验 [start, end) 区间."""
  cond = _cond_data()
  actual = _calc(cond, start=start, end=end, flags=0)
  _assert_close(actual, _expected(cond, start=start, end=end, flags=0))
  s, e = _calc_range(len(cond), start, end)
  assert np.isnan(actual[:s]).all()
  assert np.isnan(actual[e:]).all()


@pytest.mark.parametrize("groups", GROUPS)
def test_last_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  cond = _cond_data()
  start, end = 6, 25
  actual = _calc(cond, start=start, end=end, groups=groups, flags=0)
  _assert_close(
    actual, _expected(cond, start=start, end=end, groups=groups, flags=0)
  )
  chunk_size = len(cond) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_last_skip_nan_with_start_end(flags):
  """四种 flag 组合叠加 start/end, 计数仍从 chunk 头累计."""
  cond = _cond_data()
  ctx = dict(start=10, end=90, flags=flags)
  actual = _calc(cond, **ctx)
  _assert_close(actual, _expected(cond, **ctx))


def test_last_edge_cases():
  """空数组 / 单元素 / 交替条件 / list 输入."""
  empty = np.empty(0, dtype=bool)
  _assert_close(_calc(empty, flags=0), np.empty(0, dtype=np.float64))

  _assert_close(_calc(np.array([True]), flags=0), np.array([1.0]))
  _assert_close(_calc(np.array([False]), flags=0), np.array([0.0]))

  alternate = np.arange(10) % 2 == 0
  _assert_close(_calc(alternate, flags=0), _last_ref(alternate))

  al.set_ctx(start=0, end=0, groups=1, flags=0)
  result = al.LAST(
    [np.array([True, True]), np.array([True, False, True])]
  )
  assert isinstance(result, list)
  _assert_close(np.asarray(result[0]), np.array([1.0, 2.0]))
  _assert_close(np.asarray(result[1]), np.array([1.0, 0.0, 1.0]))
