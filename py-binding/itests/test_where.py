# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""WHERE (按条件逐元素二选一) 集成测试.

算法简介
--------
WHERE 对三条等长序列逐位置选择, 没有滚动窗口、没有 warm-up::

    WHERE(cond, a, b)[i] = a[i] if cond[i] else b[i]

直接复制被选中的值, **不做任何有效性检查**: 被选中的 nan 原样输出, 未被
选中的 nan 完全不参与运算, 未被选中的 ±Inf 也不会影响结果. 该算子不读取
``FLAG_SKIP_NAN`` / ``FLAG_STRICTLY_CYCLE``, 四种 flag 组合结果相同 —— 设置
SKIP_NAN 不会把「被选中但为 nan」的位置替换成另一条序列的值.

``groups`` 按 group-major 切成等长 chunk; start/end 是每个 chunk 内的相对
下标, 只计算 [start, end) 区间, 区间外一律为 nan.

对比基准 ``_where_ref`` 用 numpy ``np.where`` 独立复现算法逻辑, 只处理无
nan 数据, 不实现 SKIP_NAN / Group / start-end 机制; 这些上下文语义由
``_expected`` 单独描述: 按 chunk 分块, 只在 [s, e) 内按条件取值.

`set_ctx` 只覆盖传入的参数, 所以每次调用都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=120)
    >>> cond = np.array([True, False, True, False])
    >>> a = np.array([1., 2., np.nan, 4.])
    >>> b = np.array([10., 20., 30., np.nan])
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算: 只复制被选中的值::

    >>> al.WHERE(cond, a, b).tolist()
    [1.0, 20.0, nan, nan]

2. 未被选中的 nan 不影响结果::

    >>> al.WHERE(np.zeros(4, dtype=bool), a, np.arange(4.)).tolist()
    [0.0, 1.0, 2.0, 3.0]

3. ``FLAG_SKIP_NAN`` 不会用另一条序列填补被选中的 nan::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> al.WHERE(cond, a, b).tolist()
    [1.0, 20.0, nan, nan]

4. ``start`` / ``end`` 只计算 [start, end) 区间::

    >>> c2 = np.array([True, False, True, False, True, False, True, False])
    >>> a2 = np.arange(1., 9.)
    >>> b2 = -a2
    >>> al.set_ctx(start=2, end=6, groups=1, flags=0)
    >>> al.WHERE(c2, a2, b2).tolist()
    [nan, nan, 3.0, -4.0, 5.0, -6.0, nan, nan]

5. ``groups`` 让 start/end 在每个长度为 4 的 chunk 内各自生效::

    >>> al.set_ctx(start=1, end=3, groups=2, flags=0)
    >>> al.WHERE(c2, a2, b2).tolist()
    [nan, -2.0, 3.0, nan, nan, -6.0, 7.0, nan]
"""

import numpy as np
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
NAN_POS_A = np.array([0, 3, 17, 18, 29, 44, 52, 77, 101])
NAN_POS_B = np.array([1, 5, 18, 31, 45, 60, 78, 99])
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


def _where_ref(cond, a, b):
  """无 nan 数据上的逐元素条件选择参考实现."""
  cond = np.asarray(cond, dtype=bool)
  a = np.asarray(a, dtype=np.float64)
  b = np.asarray(b, dtype=np.float64)
  assert not np.isnan(a).any()
  assert not np.isnan(b).any()
  return np.where(cond, a, b)


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


def _expected(cond, a, b, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 逐点选择参考值, 只保留 [start, end) 区间.

  flags 不参与运算: 直接复制被选中的位置 (可以是 nan), 区间外输出 nan.
  参考实现本身只处理无 nan 数据.
  """
  cond = np.asarray(cond, dtype=bool)
  a = np.asarray(a, dtype=np.float64)
  b = np.asarray(b, dtype=np.float64)
  parts = []
  chunks = zip(np.split(cond, groups), np.split(a, groups), np.split(b, groups))
  for chunk_c, chunk_a, chunk_b in chunks:
    n = len(chunk_a)
    s, e = _calc_range(n, start, end)
    out = np.full(n, np.nan)
    for i in range(s, e):
      out[i] = chunk_a[i] if chunk_c[i] else chunk_b[i]
    parts.append(out)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(cond, a, b, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.WHERE(
      np.asarray(cond, dtype=bool),
      np.asarray(a, dtype=np.float64),
      np.asarray(b, dtype=np.float64),
    ),
    dtype=np.float64,
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261108)
  cond = rng.random(n) < 0.45
  a = rng.normal(loc=4.0, scale=3.0, size=n).astype(np.float64)
  b = rng.normal(loc=-4.0, scale=3.0, size=n).astype(np.float64)
  return cond, a, b


def _nan_data(n=N):
  cond, a, b = _clean_data(n)
  a[NAN_POS_A[NAN_POS_A < n]] = np.nan
  b[NAN_POS_B[NAN_POS_B < n]] = np.nan
  return cond, a, b


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


def test_where_matches_reference():
  """flags=0, groups=1: 与 numpy np.where 参考实现逐点比较."""
  cond, a, b = _clean_data()
  actual = _calc(cond, a, b, flags=0)
  _assert_close(actual, _where_ref(cond, a, b))
  _assert_close(actual, _expected(cond, a, b, flags=0))
  assert np.isfinite(actual).all()
  _assert_close(actual[cond], a[cond])
  _assert_close(actual[~cond], b[~cond])


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_where_selected_nan_is_copied(flags):
  """四种 flag 组合: 被选中位置的 nan 原样输出, 不被另一条序列覆盖."""
  cond, a, b = _nan_data()
  actual = _calc(cond, a, b, flags=flags)
  _assert_close(actual, _expected(cond, a, b, flags=flags))

  selected_nan = (cond & np.isnan(a)) | (~cond & np.isnan(b))
  assert np.isnan(actual[selected_nan]).all()
  untouched = ~selected_nan
  assert np.isfinite(actual[untouched]).all()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_where_flags_have_no_effect(flags):
  """SKIP_NAN / STRICTLY_CYCLE 都不改变 WHERE 的结果."""
  cond, a, b = _nan_data()
  _assert_close(_calc(cond, a, b, flags=flags), _calc(cond, a, b, flags=0))


def test_where_unselected_nan_never_leaks():
  """未被选中的序列即使全为 nan, 结果也完全等于被选中的序列."""
  cond, a, b = _clean_data()
  nan = np.full_like(a, np.nan)

  all_from_a = _calc(np.ones(len(a), dtype=bool), a, nan, flags=0)
  _assert_close(all_from_a, a)

  all_from_b = _calc(np.zeros(len(a), dtype=bool), nan, b, flags=al.FLAG_SKIP_NAN)
  _assert_close(all_from_b, b)


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_where_groups(groups, flags):
  """逐元素运算与 groups 无关, 分块只影响 start/end 下标解析."""
  cond, a, b = _nan_data()
  actual = _calc(cond, a, b, groups=groups, flags=flags)
  _assert_close(
    actual, _expected(cond, a, b, groups=groups, flags=flags)
  )
  _assert_close(actual, _calc(cond, a, b, groups=1, flags=flags))


@pytest.mark.parametrize("start,end", START_END)
def test_where_start_end(start, end):
  """start/end: 只计算 [start, end) 区间, 区间外全部为 nan."""
  cond, a, b = _nan_data()
  actual = _calc(cond, a, b, start=start, end=end, flags=0)
  _assert_close(
    actual, _expected(cond, a, b, start=start, end=end, flags=0)
  )
  s, e = _calc_range(len(a), start, end)
  assert np.isnan(actual[:s]).all()
  assert np.isnan(actual[e:]).all()


@pytest.mark.parametrize("groups", GROUPS)
def test_where_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  cond, a, b = _nan_data()
  start, end = 5, 17
  actual = _calc(cond, a, b, start=start, end=end, groups=groups, flags=0)
  _assert_close(
    actual,
    _expected(cond, a, b, start=start, end=end, groups=groups, flags=0),
  )
  chunk_size = len(a) // groups
  for g in range(groups):
    lo = g * chunk_size
    s, e = _calc_range(chunk_size, start, end)
    assert np.isnan(actual[lo : lo + s]).all()
    assert np.isnan(actual[lo + e : lo + chunk_size]).all()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_where_skip_nan_with_start_end(flags):
  """含 nan 数据 + start/end: 区间内仍按条件原样复制."""
  cond, a, b = _nan_data()
  ctx = dict(start=10, end=90, flags=flags)
  actual = _calc(cond, a, b, **ctx)
  _assert_close(actual, _expected(cond, a, b, **ctx))


def test_where_edge_cases():
  """空数组 / 全 True / 全 False / ±Inf / list 输入."""
  empty = np.empty(0, dtype=np.float64)
  _assert_close(_calc(empty.astype(bool), empty, empty, flags=0), empty)

  a = np.array([-1.0, 2.0, 3.0])
  b = np.array([10.0, 20.0, 30.0])
  _assert_close(
    _calc(np.ones(3, dtype=bool), a, b, flags=0), a
  )
  _assert_close(
    _calc(np.zeros(3, dtype=bool), a, b, flags=0), b
  )

  infs = np.array([-np.inf, np.inf, 0.0])
  finite = np.array([1.0, 2.0, 3.0])
  cond = np.array([True, False, True])
  _assert_close(_calc(cond, infs, finite, flags=0), _where_ref(cond, infs, finite))

  al.set_ctx(start=0, end=0, groups=1, flags=0)
  result = al.WHERE(
    [np.array([True, False]), np.array([False, True])],
    [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
    [np.array([10.0, 20.0]), np.array([30.0, 40.0])],
  )
  assert isinstance(result, list)
  _assert_close(np.asarray(result[0]), np.array([1.0, 20.0]))
  _assert_close(np.asarray(result[1]), np.array([30.0, 4.0]))
