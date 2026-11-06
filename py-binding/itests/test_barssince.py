# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""BARSSINCE (距第一次条件成立的周期数) 集成测试.

算法简介
--------
BARSSINCE 顺序扫描布尔序列, 输出当前位置距第一次条件成立的 bar 数::

    BARSSINCE[i] = i - min{j in [start, i] : cond[j] 为真}

首次成立的位置输出 0, 它之前 (尚无信号) 输出 nan; 与 BARSLAST 不同, 后续
信号不会重置计数, 计数始终相对首次信号. 输出是浮点数组, 因此
``[start, end)`` 区间外以及首个信号之前用 nan 表示. 算法是带状态的扫描:
状态从 ``start`` 处重新初始化, ``start`` 之前的信号不参与本次计算.

输入是布尔数组 (``nan`` 会被转换, 没有可跳过的浮点 nan), 因此
``FLAG_STRICTLY_CYCLE`` / ``FLAG_SKIP_NAN`` 都不应改变结果; 测试仍覆盖四种
flag 组合. 参考实现 ``_barssince_ref`` 用 numpy 顺序扫描独立实现, 并额外提供
``_barssince_vectorized`` 用首个信号下标交叉验证; 两者只描述算法逻辑, 不调用
alpha, 也不实现 Group 机制: Group 由测试按 chunk 分块计算 (每个 chunk 的状态
独立), start/end 只校验 ``[start, end)`` 区间的值.

`set_ctx` 只覆盖传入的参数, 因此每次调用前都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=120, suppress=True)
    >>> cond = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 1], dtype=bool)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算: 首个信号在下标 2, 之后计数连续递增, ``cond[4]`` / ``cond[9]``
   的后续信号不重置计数::

    >>> al.BARSSINCE(cond)
    array([nan, nan,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.])

2. 两个 flag 同时设置也不改变布尔序列的结果::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> al.BARSSINCE(cond)
    array([nan, nan,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.])

3. 全 False 序列没有任何信号, 全部为 nan::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> al.BARSSINCE(np.zeros(10, dtype=bool))
    array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])

4. ``groups`` 把一维数据按 group-major 切块, 每块从块首重新初始化状态::

    >>> c12 = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1], dtype=bool)
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> al.BARSSINCE(c12)
    array([0., 1., 2., 3., 4., 5., 0., 1., 2., 3., 4., 5.])

5. ``start`` / ``end`` 只计算 ``[start, end)`` 区间, 且首次信号按区间内
   的下标 4 计算::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> al.BARSSINCE(cond)
    array([nan, nan, nan, nan,  0.,  1.,  2.,  3., nan, nan])

6. ``FLAG_SKIP_NAN`` + start/end 与不带 flag 的结果相同::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> al.BARSSINCE(cond)
    array([nan, nan, nan, nan,  0.,  1.,  2.,  3., nan, nan])

7. 状态从 start 处重新开始, start 之前的信号 (``c2[0]``) 不参与 (沿用
   ``start=3, end=8``)::

    >>> c2 = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=bool)
    >>> al.BARSSINCE(c2)
    array([nan, nan, nan, nan, nan, nan,  0.,  1., nan, nan])
"""

import numpy as np
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
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


def _barssince_ref(cond, start, end):
  """BARSSINCE 参考实现: 从 start 顺序扫描, 记录第一次信号下标.

  只在 ``[start, end)`` 内写结果, 其余位置保持 nan; 后续信号不重置计数.
  """
  cond = np.asarray(cond, dtype=bool)
  out = np.full(len(cond), np.nan)
  first = None
  for i in range(start, end):
    if first is None and cond[i]:
      first = i
    if first is not None:
      out[i] = float(i - first)
  return out


def _barssince_vectorized(cond, start, end):
  """向量化参考实现: 找区间内首个信号, 之后输出连续的 bar 数."""
  cond = np.asarray(cond, dtype=bool)
  out = np.full(len(cond), np.nan)
  idx = np.flatnonzero(cond[start:end])
  if idx.size:
    first = start + int(idx[0])
    out[first:end] = np.arange(end - first, dtype=np.float64)
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
  """按 chunk 独立计算参考值; 每个 chunk 的状态只在 [start, end) 内建立."""
  cond = np.asarray(cond, dtype=bool)
  parts = []
  for chunk in np.split(cond, groups):
    n = len(chunk)
    s, e = _calc_range(n, start, end)
    parts.append(_barssince_ref(chunk, s, e))
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(cond, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.BARSSINCE(np.asarray(cond, dtype=bool)), dtype=np.float64
  )


def _bool_data(n=N):
  rng = np.random.default_rng(20261109)
  return (rng.random(n) < 0.3).astype(bool)


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


def test_barssince_matches_reference():
  """flags=0, groups=1: 与顺序扫描及向量化参考实现逐点比较."""
  cond = _bool_data()
  actual = _calc(cond, flags=0)
  _assert_close(actual, _expected(cond, flags=0))
  _assert_close(actual, _barssince_ref(cond, 0, len(cond)))
  _assert_close(actual, _barssince_vectorized(cond, 0, len(cond)))


def test_barssince_counts_from_first_signal_only():
  """计数始终相对首个信号, 后续信号不会重置."""
  cond = _bool_data()
  actual = _calc(cond, flags=0)
  _assert_close(actual, _expected(cond, flags=0))

  first = int(np.flatnonzero(cond)[0])
  expected = np.full(len(cond), np.nan)
  expected[first:] = np.arange(len(cond) - first, dtype=np.float64)
  _assert_close(actual, expected)
  assert np.isnan(actual[:first]).all()
  assert actual[first] == pytest.approx(0.0, abs=ATOL)


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_barssince_all_flag_combinations(flags):
  """四种 flag 组合对布尔输入的结果完全一致."""
  cond = _bool_data()
  actual = _calc(cond, flags=flags)
  _assert_close(actual, _expected(cond, flags=flags))
  _assert_close(actual, _calc(cond, flags=0))


@pytest.mark.parametrize("groups", GROUPS)
def test_barssince_groups(groups):
  """group-major 分块: 每个 chunk 的状态独立, 计数在块内从 0 重新开始."""
  cond = _bool_data()
  actual = _calc(cond, groups=groups, flags=0)
  _assert_close(actual, _expected(cond, groups=groups, flags=0))

  chunk_size = len(cond) // groups
  for g in range(groups):
    lo = g * chunk_size
    chunk = cond[lo : lo + chunk_size]
    _assert_close(
      actual[lo : lo + chunk_size], _barssince_ref(chunk, 0, chunk_size)
    )


@pytest.mark.parametrize("start,end", START_END)
def test_barssince_start_end(start, end):
  """start/end: 仅校验 [start, end) 区间, 区间外必须全是 nan."""
  cond = _bool_data()
  actual = _calc(cond, start=start, end=end, flags=0)
  _assert_close(actual, _expected(cond, start=start, end=end, flags=0))

  s, e = _calc_range(len(cond), start, end)
  assert np.isnan(actual[:s]).all()
  assert np.isnan(actual[e:]).all()
  if cond[s:e].any():
    assert np.isfinite(actual[s:e]).any()


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_barssince_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  cond = _bool_data()
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


def test_barssince_state_starts_at_start():
  """状态从 start 处初始化: start 之前的信号不参与, 首个信号前为 nan."""
  cond = np.zeros(N, dtype=bool)
  cond[0] = True
  cond[46] = True
  cond[55] = True  # 后续信号不应重置计数
  start, end = 30, 70
  actual = _calc(cond, start=start, end=end, flags=0)
  _assert_close(actual, _expected(cond, start=start, end=end, flags=0))
  assert np.isnan(actual[start:46]).all()
  assert actual[46:end] == pytest.approx(np.arange(end - 46), abs=ATOL)
  assert np.isnan(actual[:start]).all()
  assert np.isnan(actual[end:]).all()


def test_barssince_skip_nan_with_start_end():
  """SKIP_NAN + start/end: 结果与不带 flag 时相同."""
  cond = _bool_data()
  ctx = dict(start=10, end=90, flags=al.FLAG_SKIP_NAN)
  actual = _calc(cond, **ctx)
  _assert_close(actual, _expected(cond, **ctx))
  _assert_close(actual, _calc(cond, start=10, end=90, flags=0))
  assert np.isnan(actual[:10]).all()
  assert np.isnan(actual[90:]).all()


def test_barssince_edge_cases():
  """全 False 全部 nan; 全 True 等于 0..N-1; 信号只在尾."""
  all_false = np.zeros(N, dtype=bool)
  assert np.isnan(_calc(all_false, flags=0)).all()

  all_true = np.ones(N, dtype=bool)
  _assert_close(_calc(all_true, flags=0), np.arange(N, dtype=np.float64))

  tail = np.zeros(N, dtype=bool)
  tail[-1] = True
  expected = np.full(N, np.nan)
  expected[-1] = 0.0
  _assert_close(_calc(tail, flags=0), expected)


def test_barssince_example_values():
  """固定小样例: 显式校验 nan 前缀与后续信号不重置."""
  cond = np.array([0, 0, 1, 0, 0, 1, 0, 1, 0, 0], dtype=bool)
  expected = np.array(
    [np.nan, np.nan, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
  )
  _assert_close(_calc(cond, flags=0), expected)
  _assert_close(_barssince_ref(cond, 0, len(cond)), expected)
  _assert_close(_barssince_vectorized(cond, 0, len(cond)), expected)
