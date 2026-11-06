# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""BARSLAST (距上一次条件成立的周期数) 集成测试.

算法简介
--------
BARSLAST 顺序扫描布尔序列, 输出当前位置距最近一次条件成立的 bar 数::

    BARSLAST[i] = i - max{j <= i : cond[j] 为真}

条件成立的位置本身输出 0, 条件成立之前 (尚未出现任何信号) 输出 nan, 后续
信号会把计数重置为 0. 输出是浮点数组, 因此 ``[start, end)`` 区间外以及首个
信号之前用 nan 表示. 算法是一个带状态的扫描: 状态从 ``start`` 处重新初始化,
``start`` 之前的信号不参与本次计算, ``end`` 之后的位置不计算.

输入是布尔数组 (``nan`` 会被转换, 没有可跳过的浮点 nan), 因此
``FLAG_STRICTLY_CYCLE`` / ``FLAG_SKIP_NAN`` 都不应改变结果; 测试仍覆盖四种
flag 组合. 参考实现 ``_barslast_ref`` 用 numpy 顺序扫描独立实现, 并额外提供
``_barslast_vectorized`` 用 ``maximum.accumulate`` 交叉验证; 两者只描述算法
逻辑, 不调用 alpha, 也不实现 Group 机制: Group 由测试按 chunk 分块计算 (每个
chunk 的状态独立), start/end 只校验 ``[start, end)`` 区间的值.

`set_ctx` 只覆盖传入的参数, 因此每次调用前都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=120, suppress=True)
    >>> cond = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=bool)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算: 信号处为 0, 之后逐 bar 递增, 遇到信号重置::

    >>> al.BARSLAST(cond)
    array([0., 1., 2., 0., 1., 2., 3., 0., 1., 2.])

2. 两个 flag 同时设置也不改变布尔序列的结果::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> al.BARSLAST(cond)
    array([0., 1., 2., 0., 1., 2., 3., 0., 1., 2.])

3. 全 False 序列没有任何信号, 全部为 nan::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> al.BARSLAST(np.zeros(10, dtype=bool))
    array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])

4. ``groups`` 把一维数据按 group-major 切块, 每块从块首重新初始化状态::

    >>> c12 = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1], dtype=bool)
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> al.BARSLAST(c12)
    array([0., 0., 0., 1., 2., 0., 0., 1., 0., 1., 0., 0.])

5. ``start`` / ``end`` 只计算 ``[start, end)`` 区间, 区间外全部为 nan::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> al.BARSLAST(cond)
    array([nan, nan, nan,  0.,  1.,  2.,  3.,  0., nan, nan])

6. 状态从 start 处重新开始, start 之前的信号 (``c2[0]``) 不参与::

    >>> c2 = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=bool)
    >>> al.BARSLAST(c2)
    array([nan, nan, nan, nan, nan, nan,  0.,  1., nan, nan])

7. ``FLAG_SKIP_NAN`` + start/end 与不带 flag 的结果相同::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> al.BARSLAST(cond)
    array([nan, nan, nan,  0.,  1.,  2.,  3.,  0., nan, nan])
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


def _barslast_ref(cond, start, end):
  """BARSLAST 参考实现: 从 start 顺序扫描, 记录最近一次信号下标.

  只在 ``[start, end)`` 内写结果, 其余位置保持 nan; start 之前没有状态.
  """
  cond = np.asarray(cond, dtype=bool)
  out = np.full(len(cond), np.nan)
  last = None
  for i in range(start, end):
    if cond[i]:
      last = i
    if last is not None:
      out[i] = float(i - last)
  return out


def _barslast_vectorized(cond, start, end):
  """向量化参考实现: 用 maximum.accumulate 前推最近一次信号下标."""
  cond = np.asarray(cond, dtype=bool)
  out = np.full(len(cond), np.nan)
  idx = np.arange(start, end)
  last = np.maximum.accumulate(np.where(cond[start:end], idx, -1))
  out[start:end] = np.where(last >= 0, idx - last, np.nan)
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
    parts.append(_barslast_ref(chunk, s, e))
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(cond, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.BARSLAST(np.asarray(cond, dtype=bool)), dtype=np.float64
  )


def _bool_data(n=N):
  rng = np.random.default_rng(20261108)
  return (rng.random(n) < 0.3).astype(bool)


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


def test_barslast_matches_reference():
  """flags=0, groups=1: 与顺序扫描及向量化参考实现逐点比较."""
  cond = _bool_data()
  actual = _calc(cond, flags=0)
  _assert_close(actual, _expected(cond, flags=0))
  _assert_close(actual, _barslast_ref(cond, 0, len(cond)))
  _assert_close(actual, _barslast_vectorized(cond, 0, len(cond)))


def test_barslast_resets_at_each_signal():
  """信号处为 0, 两个信号之间的计数逐 bar 递增."""
  cond = _bool_data()
  actual = _calc(cond, flags=0)
  _assert_close(actual, _expected(cond, flags=0))

  signal_idx = np.flatnonzero(cond)
  assert np.all(actual[signal_idx] == 0.0)
  for i in range(signal_idx[0], len(cond)):
    expected_i = i - signal_idx[signal_idx <= i][-1]
    assert actual[i] == pytest.approx(expected_i, abs=ATOL)
  assert np.isnan(actual[: signal_idx[0]]).all()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_barslast_all_flag_combinations(flags):
  """四种 flag 组合对布尔输入的结果完全一致."""
  cond = _bool_data()
  actual = _calc(cond, flags=flags)
  _assert_close(actual, _expected(cond, flags=flags))
  _assert_close(actual, _calc(cond, flags=0))


@pytest.mark.parametrize("groups", GROUPS)
def test_barslast_groups(groups):
  """group-major 分块: 每个 chunk 的状态独立, 不会跨边界借用信号."""
  cond = _bool_data()
  actual = _calc(cond, groups=groups, flags=0)
  _assert_close(actual, _expected(cond, groups=groups, flags=0))

  chunk_size = len(cond) // groups
  for g in range(groups):
    lo = g * chunk_size
    chunk = cond[lo : lo + chunk_size]
    _assert_close(
      actual[lo : lo + chunk_size], _barslast_ref(chunk, 0, chunk_size)
    )


@pytest.mark.parametrize("start,end", START_END)
def test_barslast_start_end(start, end):
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
def test_barslast_start_end_with_groups(groups):
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


def test_barslast_state_starts_at_start():
  """状态从 start 处初始化: start 之前的信号不参与, 首个信号前为 nan."""
  cond = np.zeros(N, dtype=bool)
  cond[0] = True
  cond[46] = True
  start, end = 30, 70
  actual = _calc(cond, start=start, end=end, flags=0)
  _assert_close(actual, _expected(cond, start=start, end=end, flags=0))
  assert np.isnan(actual[start:46]).all()
  assert actual[46] == pytest.approx(0.0, abs=ATOL)
  assert actual[47:end] == pytest.approx(np.arange(1, end - 46), abs=ATOL)
  assert np.isnan(actual[:start]).all()
  assert np.isnan(actual[end:]).all()


def test_barslast_skip_nan_with_start_end():
  """SKIP_NAN + start/end: 结果与不带 flag 时相同."""
  cond = _bool_data()
  ctx = dict(start=10, end=90, flags=al.FLAG_SKIP_NAN)
  actual = _calc(cond, **ctx)
  _assert_close(actual, _expected(cond, **ctx))
  _assert_close(
    actual, _calc(cond, start=10, end=90, flags=0)
  )
  assert np.isnan(actual[:10]).all()
  assert np.isnan(actual[90:]).all()


def test_barslast_edge_cases():
  """全 False 全部 nan; 全 True 全部 0; 信号只在首 / 尾."""
  all_false = np.zeros(N, dtype=bool)
  assert np.isnan(_calc(all_false, flags=0)).all()

  all_true = np.ones(N, dtype=bool)
  _assert_close(_calc(all_true, flags=0), np.zeros(N))

  head = np.zeros(N, dtype=bool)
  head[0] = True
  _assert_close(_calc(head, flags=0), np.arange(N, dtype=np.float64))

  tail = np.zeros(N, dtype=bool)
  tail[-1] = True
  expected = np.full(N, np.nan)
  expected[-1] = 0.0
  _assert_close(_calc(tail, flags=0), expected)


def test_barslast_example_values():
  """固定小样例: 用显式数组校验信号重置与 nan 前缀."""
  cond = np.array([0, 0, 1, 0, 0, 0, 1, 0, 1, 0], dtype=bool)
  expected = np.array(
    [np.nan, np.nan, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0]
  )
  _assert_close(_calc(cond, flags=0), expected)
  _assert_close(_barslast_ref(cond, 0, len(cond)), expected)
  _assert_close(_barslast_vectorized(cond, 0, len(cond)), expected)
