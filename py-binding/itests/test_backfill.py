# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""BACKFILL (用最近有效观测前向填充 nan) 集成测试.

算法简介
--------
BACKFILL 逐个位置扫描, 把 nan 替换为之前最近的一个有效值::

    last = nan
    BACKFILL(x)[i] = x[i]  if x[i] is valid, last = x[i]
                      last  otherwise

序列开头在任何有效值出现之前的 nan 无法填充, 保持 nan; 连续 nan 都会被
同一个最近有效值填充. 实现中的 ``is_normal`` 只排除 NaN, 所以 ±Inf 属于
有效值, 会作为填充源. 算子不读取 ``FLAG_SKIP_NAN`` /
``FLAG_STRICTLY_CYCLE``, 四种 flag 组合结果完全相同 (它本身就是在处理
nan).

``groups`` 按 group-major 切成等长 chunk, 每个 chunk 独立从块首开始扫描;
start/end 是每个 chunk 内的相对下标. 计算从 ``start`` 起步, ``last_valid``
不引用 ``[start, end)`` 之外的观测 —— 因此当 start 落在一段 nan 中时, 这些
位置仍为 nan, 而不是被 start 之前的有效值填充. ``end`` 之后的输出未定义
(实现不初始化该段), 测试只校验 ``[start, end)`` 区间的值.

对比基准 ``_backfill_ref`` 用 numpy 独立实现前向填充, 只描述算法逻辑, 不
调用 alpha, 也不实现 Group / start-end 机制; 这些上下文语义由 ``_expected``
单独描述: 按 chunk 取 [s, e) 段独立填充.

`set_ctx` 只覆盖传入的参数, 因此每次调用前都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> x = np.array([np.nan, 2., np.nan, np.nan, 5.,
    ...               np.nan, 7., np.nan, np.nan, 10.])
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算: 前导 nan 保持 nan, 其余 nan 由最近有效值填充::

    >>> al.BACKFILL(x).tolist()
    [nan, 2.0, 2.0, 2.0, 5.0, 5.0, 7.0, 7.0, 7.0, 10.0]

2. 四种 flag 组合结果一致::

    >>> al.set_ctx(start=0, end=0, groups=1,
    ...            flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> al.BACKFILL(x).tolist()
    [nan, 2.0, 2.0, 2.0, 5.0, 5.0, 7.0, 7.0, 7.0, 10.0]

3. ``groups`` 把数据按 group-major 切块, 每块重新起步, i=5 不再被前一块
   的值填充::

    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> al.BACKFILL(x).tolist()
    [nan, 2.0, 2.0, 2.0, 5.0, nan, 7.0, 7.0, 7.0, 10.0]

4. ``start`` / ``end``: 只观察 [start, end) 区间, 该段从 start 重新起步,
   因此 i=3 的 nan 不会被 i=1 的值填充::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> al.BACKFILL(x)[3:8].tolist()
    [nan, 5.0, 5.0, 7.0, 7.0]

5. ±Inf 是有效值, 可作为填充源::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> al.BACKFILL(np.array([np.inf, np.nan, -np.inf, np.nan])).tolist()
    [inf, inf, -inf, -inf]
"""

import numpy as np
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
NAN_POS = np.array([0, 1, 3, 17, 18, 29, 44, 45, 52, 77, 101, 102])
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


def _backfill_ref(seg):
  """前向填充参考实现: 用最近的有效值替换 nan, 前导 nan 保持 nan."""
  seg = np.asarray(seg, dtype=np.float64)
  out = np.full(len(seg), np.nan)
  last = np.nan
  for i, val in enumerate(seg):
    if np.isnan(val):
      out[i] = last
    else:
      last = val
      out[i] = val
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


def _expected(x, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 在 [s, e) 段内独立做前向填充; 段外输出 nan (不校验)."""
  x = np.asarray(x, dtype=np.float64)
  parts = []
  for chunk in np.split(x, groups):
    n = len(chunk)
    s, e = _calc_range(n, start, end)
    ref = np.full(n, np.nan)
    ref[s:e] = _backfill_ref(chunk[s:e])
    parts.append(ref)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(x, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.BACKFILL(np.asarray(x, dtype=np.float64)), dtype=np.float64
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261119)
  return rng.normal(loc=10.0, scale=2.0, size=n).astype(np.float64)


def _nan_data(n=N):
  x = _clean_data(n)
  x[NAN_POS[NAN_POS < n]] = np.nan
  return x


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


def _assert_range_close(actual, expected, s, e):
  """只校验 [s, e) 区间: end 之后的输出未被实现初始化."""
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual[s:e], expected[s:e], atol=ATOL, rtol=RTOL, equal_nan=True
  )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


def test_backfill_matches_reference():
  """flags=0, groups=1: 与 numpy 前向填充参考实现逐点比较."""
  x = _nan_data()
  actual = _calc(x, flags=0)
  _assert_close(actual, _backfill_ref(x))
  _assert_close(actual, _expected(x, flags=0))

  first_valid = int(np.flatnonzero(~np.isnan(x))[0])
  assert np.isnan(actual[:first_valid]).all()
  assert np.isfinite(actual[first_valid:]).all()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_backfill_flags_have_no_effect(flags):
  """四种 flag 组合: BACKFILL 本身处理 nan, 不读取任何 flag."""
  x = _nan_data()
  _assert_close(_calc(x, flags=flags), _calc(x, flags=0))
  _assert_close(_calc(x, flags=flags), _backfill_ref(x))


def test_backfill_leading_and_consecutive_nan():
  """前导 nan 保持 nan; 连续 nan 全部由最近有效值填充."""
  x = np.array([np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0, np.nan])
  expected = np.array([np.nan, np.nan, 3.0, 3.0, 3.0, 3.0, 7.0, 7.0])
  _assert_close(_calc(x, flags=0), expected)
  _assert_close(_calc(x, flags=al.FLAG_SKIP_NAN), expected)


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_backfill_groups(groups, flags):
  """group-major 分块: 每个 chunk 独立从块首起步, 不跨块借用有效值."""
  x = _nan_data()
  actual = _calc(x, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, groups=groups, flags=flags))

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    _assert_close(
      actual[lo : lo + chunk_size], _backfill_ref(x[lo : lo + chunk_size])
    )


@pytest.mark.parametrize("start,end", START_END)
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_backfill_start_end(start, end, flags):
  """start/end: 只校验 [start, end) 区间, 该段从 start 重新起步."""
  x = _nan_data()
  actual = _calc(x, start=start, end=end, flags=flags)
  expected = _expected(x, start=start, end=end, flags=flags)
  s, e = _calc_range(len(x), start, end)
  _assert_range_close(actual, expected, s, e)
  assert np.isnan(actual[:s]).all()


@pytest.mark.parametrize("groups", GROUPS)
def test_backfill_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  x = _nan_data()
  start, end = 6, 25
  actual = _calc(x, start=start, end=end, groups=groups, flags=0)
  expected = _expected(x, start=start, end=end, groups=groups, flags=0)
  chunk_size = len(x) // groups
  s, e = _calc_range(chunk_size, start, end)
  for g in range(groups):
    lo = g * chunk_size
    _assert_range_close(
      actual[lo : lo + chunk_size], expected[lo : lo + chunk_size], s, e
    )
    assert np.isnan(actual[lo : lo + s]).all()


def test_backfill_edge_cases():
  """全 nan / 无 nan / 单元素 / 空数组 / ±Inf / list 输入."""
  all_nan = np.full(10, np.nan)
  assert np.isnan(_calc(all_nan, flags=0)).all()

  clean = _clean_data(20)
  _assert_close(_calc(clean, flags=0), clean)

  _assert_close(_calc(np.array([3.0]), flags=0), np.array([3.0]))
  assert np.isnan(_calc(np.array([np.nan]), flags=0)).all()

  empty = np.empty(0, dtype=np.float64)
  _assert_close(_calc(empty, flags=0), empty)

  infs = np.array([np.inf, np.nan, -np.inf, np.nan])
  _assert_close(_calc(infs, flags=0), np.array([np.inf, np.inf, -np.inf, -np.inf]))

  al.set_ctx(start=0, end=0, groups=1, flags=0)
  result = al.BACKFILL(
    [np.array([1.0, np.nan, np.nan]), np.array([np.nan, 2.0])]
  )
  assert isinstance(result, list)
  _assert_close(np.asarray(result[0]), np.array([1.0, 1.0, 1.0]))
  _assert_close(np.asarray(result[1]), np.array([np.nan, 2.0]))
