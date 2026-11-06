# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""ABS (逐元素绝对值) 集成测试.

算法简介
--------
ABS 对输入序列逐位置取绝对值, 没有滚动窗口、没有 warm-up::

    ABS(x)[i] = |x[i]|

输入为 nan 时输出 nan; 实现里的 ``is_normal`` 只排除 NaN, 因此 ±Inf 属于
有效值, ``ABS(-inf) = inf``. 该算子不读取 ``FLAG_SKIP_NAN`` /
``FLAG_STRICTLY_CYCLE``, 四种 flag 组合结果完全相同: 即使设置 SKIP_NAN,
nan 也不会被"跳过"或由邻近值填充, 它只是原样保留.

``groups`` 按 group-major 切成等长 chunk; start/end 是每个 chunk 内的相对
下标, 只计算 [start, end) 区间, 区间外一律为 nan. 由于是逐元素运算, 分块
只影响 start/end 下标的解析, 不存在跨位置借数据的情况.

对比基准 ``_abs_ref`` 用 numpy ``np.abs`` 独立复现算法逻辑, 只处理无 nan
数据, 不实现 SKIP_NAN / Group / start-end 机制; 这些上下文语义由
``_expected`` 单独描述: 按 chunk 分块, 只在 [s, e) 内逐点取绝对值, nan
输入与区间外都输出 nan.

`set_ctx` 只覆盖传入的参数, 所以每次调用都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=120)
    >>> x = np.array([1., -2., np.nan, -4., 5.])
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算, nan 位置保持 nan::

    >>> al.ABS(x).tolist()
    [1.0, 2.0, nan, 4.0, 5.0]

2. ``FLAG_SKIP_NAN`` 不会填充 nan 位置 (结果同 flags=0)::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> al.ABS(x).tolist()
    [1.0, 2.0, nan, 4.0, 5.0]

3. ``start`` / ``end`` 只计算 [start, end) 区间, 区间外全部为 nan::

    >>> y = np.array([1., -2., 3., -4., 5., -6., 7., -8.])
    >>> al.set_ctx(start=2, end=6, groups=1, flags=0)
    >>> al.ABS(y).tolist()
    [nan, nan, 3.0, 4.0, 5.0, 6.0, nan, nan]

4. ``groups`` 让 start/end 在每个长度为 4 的 chunk 内各自生效::

    >>> al.set_ctx(start=1, end=3, groups=2, flags=0)
    >>> al.ABS(y).tolist()
    [nan, 2.0, 3.0, nan, nan, 6.0, 7.0, nan]

5. 无穷值按普通数值处理::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> al.ABS(np.array([-np.inf, np.inf, -0.])).tolist()
    [inf, inf, 0.0]
"""

import numpy as np
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
NAN_POS = np.array([0, 3, 17, 18, 29, 44, 52, 77, 101])
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


def _abs_ref(x):
  """无 nan 数据上的逐元素绝对值参考实现."""
  x = np.asarray(x, dtype=np.float64)
  assert not np.isnan(x).any()
  return np.abs(x)


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
  """按 chunk 逐点计算参考值, 只保留 [start, end) 区间.

  flags 不参与运算: nan 输入位置与区间外都输出 nan. 参考实现本身只处理
  无 nan 数据.
  """
  x = np.asarray(x, dtype=np.float64)
  parts = []
  for chunk in np.split(x, groups):
    n = len(chunk)
    s, e = _calc_range(n, start, end)
    out = np.full(n, np.nan)
    for i in range(s, e):
      if not np.isnan(chunk[i]):
        out[i] = abs(chunk[i])
    parts.append(out)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(x, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(al.ABS(np.asarray(x, dtype=np.float64)), dtype=np.float64)


def _clean_data(n=N):
  rng = np.random.default_rng(20261107)
  return rng.normal(loc=-2.0, scale=6.0, size=n).astype(np.float64)


def _nan_data(n=N):
  x = _clean_data(n)
  x[NAN_POS[NAN_POS < n]] = np.nan
  return x


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


def test_abs_matches_reference():
  """flags=0, groups=1: 与 numpy np.abs 参考实现逐点比较."""
  x = _clean_data()
  actual = _calc(x, flags=0)
  _assert_close(actual, _abs_ref(x))
  _assert_close(actual, _expected(x, flags=0))
  assert np.isfinite(actual).all()
  assert (actual >= 0).all()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_abs_nan_positions_stay_nan(flags):
  """四种 flag 组合: nan 输入位置一律输出 nan, 有效位置为绝对值."""
  x = _nan_data()
  actual = _calc(x, flags=flags)
  _assert_close(actual, _expected(x, flags=flags))

  valid = ~np.isnan(x)
  assert np.isnan(actual[~valid]).all()
  _assert_close(actual[valid], _abs_ref(x[valid]))
  assert np.isfinite(actual[valid]).all()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_abs_flags_have_no_effect(flags):
  """SKIP_NAN / STRICTLY_CYCLE 都不改变 ABS 的结果."""
  x = _nan_data()
  _assert_close(_calc(x, flags=flags), _calc(x, flags=0))


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_abs_groups(groups, flags):
  """逐元素运算与 groups 无关, 分块只影响 start/end 下标解析."""
  x = _nan_data()
  actual = _calc(x, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, groups=groups, flags=flags))
  _assert_close(actual, _calc(x, groups=1, flags=flags))


@pytest.mark.parametrize("start,end", START_END)
def test_abs_start_end(start, end):
  """start/end: 只计算 [start, end) 区间, 区间外全部为 nan."""
  x = _nan_data()
  actual = _calc(x, start=start, end=end, flags=0)
  _assert_close(actual, _expected(x, start=start, end=end, flags=0))
  s, e = _calc_range(len(x), start, end)
  assert np.isnan(actual[:s]).all()
  assert np.isnan(actual[e:]).all()


@pytest.mark.parametrize("groups", GROUPS)
def test_abs_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  x = _nan_data()
  start, end = 5, 17
  actual = _calc(x, start=start, end=end, groups=groups, flags=0)
  _assert_close(
    actual, _expected(x, start=start, end=end, groups=groups, flags=0)
  )
  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    s, e = _calc_range(chunk_size, start, end)
    assert np.isnan(actual[lo : lo + s]).all()
    assert np.isnan(actual[lo + e : lo + chunk_size]).all()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_abs_skip_nan_with_start_end(flags):
  """含 nan 数据 + start/end: 区间内有效值仍是绝对值, nan 不被填充."""
  x = _nan_data()
  ctx = dict(start=10, end=90, flags=flags)
  actual = _calc(x, **ctx)
  _assert_close(actual, _expected(x, **ctx))
  assert np.isnan(actual[np.isnan(x)]).all()


def test_abs_edge_cases():
  """空数组 / 全 nan / 单个元素 / ±Inf / list 输入."""
  empty = np.empty(0, dtype=np.float64)
  _assert_close(_calc(empty, flags=0), empty)

  all_nan = np.full(10, np.nan)
  assert np.isnan(_calc(all_nan, flags=0)).all()
  assert np.isnan(_calc(all_nan, flags=al.FLAG_SKIP_NAN)).all()

  one = np.array([-3.5])
  _assert_close(_calc(one, flags=0), np.array([3.5]))

  infs = np.array([-np.inf, np.inf, -0.0, 2.0])
  _assert_close(_calc(infs, flags=0), _abs_ref(infs))
  _assert_close(_calc(infs, flags=al.FLAG_SKIP_NAN), _abs_ref(infs))

  al.set_ctx(start=0, end=0, groups=1, flags=0)
  result = al.ABS([np.array([-1.0, 2.0]), np.array([-3.0, -4.0])])
  assert isinstance(result, list)
  _assert_close(np.asarray(result[0]), np.array([1.0, 2.0]))
  _assert_close(np.asarray(result[1]), np.array([3.0, 4.0]))
