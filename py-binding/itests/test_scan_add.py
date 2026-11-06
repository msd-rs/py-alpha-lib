# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""SCAN_ADD / SCAN_MUL (条件累加 / 条件累乘) 集成测试.

算法简介
--------
两个算子都在每个 chunk 内维护一个累加器, 只在条件为真且输入值有效时更新,
但**每个位置都写出当前累加器**::

    SCAN_ADD: acc = 0;  r[i] = acc += x[i] if cond[i] and x[i] is not nan
    SCAN_MUL: acc = 1;  r[i] = acc *= x[i] if cond[i] and x[i] is not nan

关键语义:

* 输入为 nan 的位置输出**不是 nan**, 而是之前的累积值, nan 不会毒化状态;
  condition 为假的位置同样只是保持累积值.
* 累加器只在 ``[start, end)`` 区间内迭代, **不引用 start 之前的数据**,
  与 MA/REF 等滚动窗口算子不同; 区间外一律 nan.
* 两个 flag (``FLAG_SKIP_NAN`` / ``FLAG_STRICTLY_CYCLE``) 都被忽略,
  四种组合结果完全相同.
* ``groups`` 按 group-major 切成等长 chunk, 每块 acc 重新初始化为 0 / 1;
  start/end 是每个 chunk 内的相对下标.

参考实现 ``_scan_ref`` 只处理无 nan 数据, 用 numpy ``cumsum`` / ``cumprod``
配合 ``np.where`` 独立复现算法逻辑, 不调用 alpha, 也不实现 Group / start-end
机制; 这些上下文语义由 ``_expected`` 单独描述: 按 chunk 切分, 只在 [s, e)
段内累加, 区间外填 nan.

`set_ctx` 只覆盖传入的参数, 所以每次调用都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=120)
    >>> x = np.arange(1., 6.)
    >>> cond = np.array([True, False, True, False, True])
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. ``SCAN_ADD``: 只在 cond 为真的 1、3、5 号位置累加::

    >>> al.SCAN_ADD(x, cond)
    array([1., 1., 4., 4., 9.])

2. ``SCAN_MUL``: 同样条件下从 1 开始累乘::

    >>> al.SCAN_MUL(np.array([2., 3., 0.5, 4., 2.]), cond)
    array([2., 2., 1., 1., 2.])

3. 输入 nan 不毒化累积状态, 该位置仍输出之前的累积值::

    >>> xn = np.array([1., np.nan, 3., 4., np.nan])
    >>> al.SCAN_ADD(xn, np.ones(5, dtype=bool))
    array([1., 1., 4., 8., 8.])

4. flags 被忽略, 结果与 ``flags=0`` 相同::

    >>> al.set_ctx(start=0, end=0, groups=1,
    ...            flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> al.SCAN_ADD(x, cond)
    array([1., 1., 4., 4., 9.])

5. ``groups=2``: 第二块从 0 重新累加 (输入 1..10, cond 全真)::

    >>> t = np.arange(1., 11.)
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> al.SCAN_ADD(t, np.ones(10, dtype=bool))
    array([ 1.,  3.,  6., 10., 15.,  6., 13., 21., 30., 40.])

6. ``start`` / ``end``: 区间外为 nan, 且 start 之前的数据不参与累加.
   i=2 的输出是 3 而不是 1+2+3=6::

    >>> al.set_ctx(start=2, end=8, groups=1, flags=0)
    >>> al.SCAN_ADD(t, np.ones(10, dtype=bool))
    array([nan, nan,  3.,  7., 12., 18., 25., 33., nan, nan])

7. ``SCAN_MUL`` 一旦乘过 0, 之后的累积值恒为 0::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> al.SCAN_MUL(np.array([2., 0., 3., 4.]), np.ones(4, dtype=bool))
    array([2., 0., 0., 0.])
"""

import numpy as np
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
NAN_POS = np.array([0, 1, 3, 17, 18, 29, 44, 52, 77, 101])
GROUPS = [1, 2, 3, 4, 6]
START_END = [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5), (-10, 0), (40, 20)]
FLAG_COMBOS = [
  0,
  al.FLAG_STRICTLY_CYCLE,
  al.FLAG_SKIP_NAN,
  al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
]
KINDS = ["add", "mul"]


# ---------------------------------------------------------------------------
# numpy 对比实现
# ---------------------------------------------------------------------------


def _scan_ref(x, cond, kind):
  """无 nan 数据上的条件累加 / 累乘参考实现."""
  x = np.asarray(x, dtype=np.float64)
  cond = np.asarray(cond, dtype=bool)
  assert not np.isnan(x).any()
  valid = cond & ~np.isnan(x)
  if kind == "add":
    return np.cumsum(np.where(valid, x, 0.0))
  return np.cumprod(np.where(valid, x, 1.0))


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


def _expected(x, cond, kind, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 独立累加, 只在 [start, end) 内写值, 区间外为 nan.

  ``flags`` 对这两个算子没有作用, 保留参数只是为了与其它测试文件一致.
  """
  x = np.asarray(x, dtype=np.float64)
  cond = np.asarray(cond, dtype=bool)
  parts = []
  for chunk_x, chunk_c in zip(np.split(x, groups), np.split(cond, groups)):
    n = len(chunk_x)
    s, e = _calc_range(n, start, end)
    out = np.full(n, np.nan)
    if s < e:
      seg = chunk_x[s:e].copy()
      cseg = chunk_c[s:e] & ~np.isnan(seg)
      neutral = 0.0 if kind == "add" else 1.0
      seg = np.where(np.isnan(seg), neutral, seg)
      out[s:e] = _scan_ref(seg, cseg, kind)
    parts.append(out)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(x, cond, kind, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  x = np.asarray(x, dtype=np.float64)
  cond = np.asarray(cond, dtype=bool)
  fn = al.SCAN_ADD if kind == "add" else al.SCAN_MUL
  return np.asarray(fn(x, cond), dtype=np.float64)


def _clean_data(n=N):
  rng = np.random.default_rng(20261106)
  return rng.uniform(0.5, 1.5, size=n).astype(np.float64)


def _nan_data(n=N):
  x = _clean_data(n)
  x[NAN_POS[NAN_POS < n]] = np.nan
  return x


def _cond_data(n=N):
  rng = np.random.default_rng(20261106)
  return (rng.random(n) < 0.55).astype(bool)


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_scan_add_matches_cumsum(flags):
  """SCAN_ADD: 与 numpy cumsum 参考实现逐点比较."""
  x, cond = _clean_data(), _cond_data()
  actual = _calc(x, cond, "add", flags=flags)
  _assert_close(actual, _expected(x, cond, "add", flags=flags))
  _assert_close(actual, _scan_ref(x, cond, "add"))
  assert np.isfinite(actual).all()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_scan_mul_matches_cumprod(flags):
  """SCAN_MUL: 与 numpy cumprod 参考实现逐点比较."""
  x, cond = _clean_data(), _cond_data()
  actual = _calc(x, cond, "mul", flags=flags)
  _assert_close(actual, _expected(x, cond, "mul", flags=flags))
  _assert_close(actual, _scan_ref(x, cond, "mul"))
  assert np.isfinite(actual).all()


@pytest.mark.parametrize("kind", KINDS)
def test_scan_nan_input_does_not_poison_state(kind):
  """输入 nan 的位置输出仍是累积值, 且不改变累加器状态."""
  x = _nan_data()
  cond = np.ones(len(x), dtype=bool)
  actual = _calc(x, cond, kind)
  _assert_close(actual, _expected(x, cond, kind))
  assert np.isfinite(actual).all()

  for i in range(len(x)):
    if np.isnan(x[i]):
      before = 0.0 if kind == "add" else 1.0
      if i > 0:
        before = actual[i - 1]
      _assert_close(actual[i : i + 1], np.array([before]))


@pytest.mark.parametrize("kind", KINDS)
def test_scan_all_flag_combinations_identical(kind):
  """四种 flag 组合: SKIP_NAN 与 STRICTLY_CYCLE 都被忽略."""
  x, cond = _nan_data(), _cond_data()
  baseline = _calc(x, cond, kind, flags=0)
  for flags in FLAG_COMBOS:
    _assert_close(_calc(x, cond, kind, flags=flags), baseline)
    _assert_close(
      _calc(x, cond, kind, flags=flags), _expected(x, cond, kind)
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("all_true", [True, False])
def test_scan_condition_extremes(kind, all_true):
  """condition 全真 / 全假: 全假时输出恒为 0 (add) 或 1 (mul)."""
  x = _clean_data()
  cond = np.full(len(x), all_true, dtype=bool)
  actual = _calc(x, cond, kind)
  _assert_close(actual, _expected(x, cond, kind))
  if not all_true:
    neutral = 0.0 if kind == "add" else 1.0
    _assert_close(actual, np.full(len(x), neutral))


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("kind", KINDS)
def test_scan_groups(groups, kind):
  """group-major 分块: 每块 acc 重新初始化, 不跨 group 累积."""
  x, cond = _clean_data(), _cond_data()
  actual = _calc(x, cond, kind, groups=groups)
  _assert_close(actual, _expected(x, cond, kind, groups=groups))

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    _assert_close(
      actual[lo : lo + chunk_size],
      _scan_ref(x[lo : lo + chunk_size], cond[lo : lo + chunk_size], kind),
    )


@pytest.mark.parametrize("start,end", START_END)
@pytest.mark.parametrize("kind", KINDS)
def test_scan_start_end(start, end, kind):
  """start/end: 区间外全 nan, 且 start 之前的数据不参与累积."""
  x, cond = _clean_data(), _cond_data()
  actual = _calc(x, cond, kind, start=start, end=end)
  _assert_close(actual, _expected(x, cond, kind, start=start, end=end))
  s, e = _calc_range(len(x), start, end)
  if s >= e:
    assert np.isnan(actual).all()
  else:
    assert np.isnan(actual[:s]).all()
    assert np.isnan(actual[e:]).all()
    assert np.isfinite(actual[s:e]).all()


def test_scan_start_ignores_pre_start_data():
  """显式核对: start 处的首个输出只含 x[start], 不含更早的数据."""
  x = np.arange(1.0, 11.0)
  cond = np.ones(10, dtype=bool)
  actual = _calc(x, cond, "add", start=3, end=8)
  _assert_close(actual[3:8], np.array([4.0, 9.0, 15.0, 22.0, 30.0]))
  _assert_close(
    actual,
    _expected(x, cond, "add", start=3, end=8),
  )


@pytest.mark.parametrize("groups", [1, 2, 3])
@pytest.mark.parametrize("kind", KINDS)
def test_scan_start_end_with_groups(groups, kind):
  """start/end 是 chunk 内的相对下标, 每个 group 各自重置 acc."""
  x, cond = _clean_data(), _cond_data()
  start, end = 6, 25
  actual = _calc(x, cond, kind, start=start, end=end, groups=groups)
  _assert_close(
    actual,
    _expected(x, cond, kind, start=start, end=end, groups=groups),
  )
  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("kind", KINDS)
def test_scan_skip_nan_flags_with_start_end_and_groups(groups, kind):
  """flags + 负数 end + groups 组合下仍与参考实现一致."""
  x, cond = _nan_data(), _cond_data()
  ctx = dict(
    start=5,
    end=-5,
    groups=groups,
    flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
  )
  actual = _calc(x, cond, kind, **ctx)
  _assert_close(actual, _expected(x, cond, kind, **ctx))


def test_scan_mul_zero_locks_accumulator():
  """SCAN_MUL 乘过 0 之后累积值恒为 0, 输入 nan 也不会改写它."""
  x = np.array([2.0, 0.0, np.nan, 4.0, 5.0])
  cond = np.ones(5, dtype=bool)
  actual = _calc(x, cond, "mul")
  _assert_close(actual, np.array([2.0, 0.0, 0.0, 0.0, 0.0]))
  _assert_close(actual, _expected(x, cond, "mul"))


def test_scan_edge_cases():
  """单元素数组、全 nan 输入、输入长度不匹配报错."""
  for kind, neutral in (("add", 0.0), ("mul", 1.0)):
    one = np.array([3.0])
    _assert_close(_calc(one, np.array([True]), kind), np.array([3.0]))
    _assert_close(
      _calc(one, np.array([False]), kind), np.array([neutral])
    )

    all_nan = np.full(10, np.nan)
    _assert_close(
      _calc(all_nan, np.ones(10, dtype=bool), kind),
      np.full(10, neutral),
    )

    with pytest.raises(ValueError):
      _calc(np.arange(1.0, 6.0), np.ones(3, dtype=bool), kind)
