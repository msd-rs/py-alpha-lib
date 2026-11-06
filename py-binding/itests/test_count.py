# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""COUNT (滚动窗口内条件成立次数) 集成测试.

算法简介
--------
COUNT 统计布尔序列在最近 `periods` 个位置中为真的个数::

    COUNT[i] = sum(cond[i - periods + 1 : i + 1])

窗口未满 (``i < periods - 1``) 时用已有的 1..periods 个值给出部分计数,
等价于 pandas ``rolling(periods, min_periods=1).sum()``.
``periods = 0`` 表示从序列开头累计计数, 即 ``cumsum(cond)``.
设置 ``FLAG_STRICTLY_CYCLE`` 后窗口未满的前 ``periods - 1`` 个位置输出 nan
(累计模式下没有窗口, 不受影响).

输入是布尔数组, 没有 nan 值, 因此 ``FLAG_SKIP_NAN`` 不应改变结果; 测试仍
覆盖四种 flag 组合以确认这一点. 参考实现 ``_count_ref`` 用 pandas
``rolling`` / numpy ``cumsum`` 独立实现, 只描述算法逻辑, 不调用 alpha,
也不实现 Group / start-end 机制: Group 由测试按 chunk 分块计算 (每个 chunk
的窗口不跨边界), start/end 只校验 ``[start, end)`` 区间的值, 区间外 alpha
一律输出 nan, 但窗口仍可引用 start 之前的数据.

`set_ctx` 只覆盖传入的参数, 因此每次调用前都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=120, suppress=True)
    >>> cond = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=bool)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算 (窗口未满时给出部分计数)::

    >>> al.COUNT(cond, 3)
    array([1., 1., 2., 2., 2., 2., 1., 1., 1., 2.])

2. ``FLAG_STRICTLY_CYCLE``: 前 2 个位置输出 nan::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> al.COUNT(cond, 3)
    array([nan, nan,  2.,  2.,  2.,  2.,  1.,  1.,  1.,  2.])

3. ``periods = 0`` 为累计计数::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> al.COUNT(cond, 0)
    array([1., 1., 2., 3., 3., 4., 4., 4., 5., 6.])

4. ``FLAG_SKIP_NAN`` 对布尔输入无影响::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> al.COUNT(cond, 3)
    array([1., 1., 2., 2., 2., 2., 1., 1., 1., 2.])

5. ``groups`` 把一维数据按 group-major 切块, 每块窗口独立计数::

    >>> c12 = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1], dtype=bool)
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> al.COUNT(c12, 3)
    array([1., 2., 2., 1., 1., 2., 1., 2., 2., 1., 1., 2.])

6. ``start`` / ``end`` 只计算 ``[start, end)`` 区间, 区间外全部为 nan::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> al.COUNT(cond, 3)
    array([nan, nan, nan,  2.,  2.,  2.,  1.,  1., nan, nan])
"""

import numpy as np
import pandas as pd
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120


# ---------------------------------------------------------------------------
# numpy / pandas 对比实现
# ---------------------------------------------------------------------------


def _count_ref(cond, periods, *, strict=False):
  """COUNT 参考实现: rolling(periods, min_periods=1).sum() 或 cumsum.

  ``periods = 0`` 表示累计计数; ``strict`` 时前 ``periods - 1`` 个位置
  输出 nan (累计模式没有窗口, 不受影响).
  """
  cond = np.asarray(cond, dtype=bool)
  if periods == 0:
    return np.cumsum(cond, dtype=np.float64)

  out = (
    pd.Series(cond.astype(np.float64))
    .rolling(periods, min_periods=1)
    .sum()
    .to_numpy(dtype=np.float64)
    .copy()
  )
  if strict:
    out[: periods - 1] = np.nan
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
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  parts = []
  for chunk in np.split(cond, groups):
    n = len(chunk)
    s, e = _calc_range(n, start, end)
    ref = _count_ref(chunk, periods, strict=strict)
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
    al.COUNT(np.asarray(cond, dtype=bool), periods), dtype=np.float64
  )


def _bool_data(n=N):
  rng = np.random.default_rng(20261106)
  return (rng.random(n) < 0.4).astype(bool)


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("periods", [1, 2, 3, 4, 5, 10, 20])
def test_count_matches_pandas_rolling(periods):
  """flags=0, groups=1: 与 pandas rolling sum 逐点比较."""
  cond = _bool_data()
  actual = _calc(cond, periods, flags=0)
  _assert_close(actual, _expected(cond, periods, flags=0))
  _assert_close(actual, _count_ref(cond, periods))
  assert np.isfinite(actual).all()


@pytest.mark.parametrize("periods", [1, 2, 3, 4, 5, 10, 20])
def test_count_strictly_cycle(periods):
  """FLAG_STRICTLY_CYCLE: 前 periods-1 个 nan, 之后与 pandas 一致."""
  cond = _bool_data()
  flags = al.FLAG_STRICTLY_CYCLE
  actual = _calc(cond, periods, flags=flags)
  _assert_close(actual, _expected(cond, periods, flags=flags))
  _assert_close(actual, _count_ref(cond, periods, strict=True))
  if periods > 1:
    assert np.isnan(actual[: periods - 1]).all()
    assert np.isfinite(actual[periods - 1 :]).all()


@pytest.mark.parametrize(
  "flags",
  [
    0,
    al.FLAG_STRICTLY_CYCLE,
    al.FLAG_SKIP_NAN,
    al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
  ],
)
def test_count_cumulative(flags):
  """periods=0: 从序列开头累计计数, 不受任何 flag 影响."""
  cond = _bool_data()
  actual = _calc(cond, 0, flags=flags)
  _assert_close(actual, _expected(cond, 0, flags=flags))
  _assert_close(actual, np.cumsum(cond, dtype=np.float64))
  assert np.isfinite(actual).all()


@pytest.mark.parametrize(
  "flags",
  [
    0,
    al.FLAG_STRICTLY_CYCLE,
    al.FLAG_SKIP_NAN,
    al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
  ],
)
@pytest.mark.parametrize("periods", [3, 6])
def test_count_all_flag_combinations(periods, flags):
  """四种 flag 组合: SKIP_NAN 对布尔输入无影响."""
  cond = _bool_data()
  actual = _calc(cond, periods, flags=flags)
  _assert_close(actual, _expected(cond, periods, flags=flags))
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  _assert_close(actual, _count_ref(cond, periods, strict=strict))


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_count_groups(groups, flags):
  """group-major 分块: 每个 chunk 的滚动窗口不跨边界."""
  cond = _bool_data()
  periods = 4
  actual = _calc(cond, periods, groups=groups, flags=flags)
  _assert_close(actual, _expected(cond, periods, groups=groups, flags=flags))

  chunk_size = len(cond) // groups
  for g in range(groups):
    lo = g * chunk_size
    chunk = cond[lo : lo + chunk_size]
    _assert_close(
      actual[lo : lo + chunk_size],
      _count_ref(chunk, periods, strict=bool(flags)),
    )


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
def test_count_cumulative_with_groups(groups):
  """累计计数也按 chunk 独立, 每块从 0 重新开始计数."""
  cond = _bool_data()
  actual = _calc(cond, 0, groups=groups, flags=0)
  _assert_close(actual, _expected(cond, 0, groups=groups, flags=0))

  chunk_size = len(cond) // groups
  for g in range(groups):
    lo = g * chunk_size
    _assert_close(
      actual[lo : lo + chunk_size],
      np.cumsum(cond[lo : lo + chunk_size], dtype=np.float64),
    )


@pytest.mark.parametrize(
  "start,end", [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5)]
)
def test_count_start_end(start, end):
  """start/end: 仅校验 [start, end) 区间, 区间外必须全是 nan."""
  cond = _bool_data()
  periods = 4
  actual = _calc(cond, periods, start=start, end=end, flags=0)
  _assert_close(actual, _expected(cond, periods, start=start, end=end, flags=0))

  s, e = _calc_range(len(cond), start, end)
  assert np.isnan(actual[:s]).all()
  assert np.isnan(actual[e:]).all()
  assert np.isfinite(actual[s:e]).all()


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_count_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  cond = _bool_data()
  periods = 4
  start, end = 6, 25
  actual = _calc(cond, periods, start=start, end=end, groups=groups, flags=0)
  _assert_close(
    actual,
    _expected(cond, periods, start=start, end=end, groups=groups, flags=0),
  )

  chunk_size = len(cond) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


def test_count_edge_cases():
  """窗口长于数据时等于累计计数; 全 False / 全 True 序列与参考实现一致."""
  cond = _bool_data()
  long_periods = len(cond) + 1
  _assert_close(
    _calc(cond, long_periods, flags=0),
    np.cumsum(cond, dtype=np.float64),
  )

  for value in (False, True):
    dense = np.full(len(cond), value, dtype=bool)
    for periods in (3, 6):
      _assert_close(
        _calc(dense, periods, flags=al.FLAG_STRICTLY_CYCLE),
        _count_ref(dense, periods, strict=True),
      )
