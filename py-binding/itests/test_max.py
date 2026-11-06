# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""MAX / MIN (逐元素最大值 / 最小值) 集成测试.

算法简介
--------
两个算子对两条等长序列逐位置取最大 / 最小值, **没有滚动窗口、没有
warm-up**::

    MAX(a, b)[i] = max(a[i], b[i])
    MIN(a, b)[i] = min(a[i], b[i])

``flags=0`` 时只要 ``a[i]`` 或 ``b[i]`` 任一为 nan, 输出就是 nan. 设置
``FLAG_SKIP_NAN`` 后改为「忽略单侧 nan」: 一侧为 nan 时直接取另一侧的有效
值, 两侧都是 nan 才输出 nan. ``FLAG_STRICTLY_CYCLE`` 对该算子没有作用, 四种
flag 组合里只有 SKIP_NAN 会改变结果. 实现中的 ``is_normal`` 只排除 NaN,
所以 ±Inf 属于有效值, 会正常参与 max / min 比较.

``groups`` 按 group-major 切成等长 chunk; start/end 是每个 chunk 内的相对
下标, 只计算 [start, end) 区间, 区间外一律为 nan. 由于是逐元素运算, 分块
只影响 start/end 下标的解析, 不存在跨位置借数据的情况.

参考实现 ``_max_ref`` 只处理无 nan 数据, 用 numpy ``maximum`` / ``minimum``
独立复现算法逻辑, 不调用 alpha, 也不实现 SKIP_NAN / Group / start-end 机制;
这些上下文语义由 ``_expected`` 单独描述: 按 chunk 分块, 只在 [s, e) 内按
nan 语义逐点计算, 区间外填 nan.

`set_ctx` 只覆盖传入的参数, 所以每次调用都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=120)
    >>> a = np.array([1., np.nan, 5., 2.])
    >>> b = np.array([3., 4., np.nan, 2.])
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. ``flags=0``: 任一 nan 污染该位置, 相等时返回该值::

    >>> al.MAX(a, b)
    array([ 3., nan, nan,  2.])
    >>> al.MIN(a, b)
    array([ 1., nan, nan,  2.])

2. ``FLAG_SKIP_NAN``: 单侧 nan 时取另一侧, 两侧都为 nan 才输出 nan::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> al.MAX(a, b)
    array([3., 4., 5., 2.])
    >>> al.MIN(a, b)
    array([1., 4., 5., 2.])

3. 两侧都是 nan 的位置仍是 nan::

    >>> al.MAX(np.array([np.nan, 2.]), np.array([np.nan, 3.]))
    array([nan,  3.])

4. 逐元素运算与 groups 无关 (只影响 start/end 下标解析)::

    >>> a3 = np.array([1., 5., 3., 7., 2., 6., 4., 8.])
    >>> b3 = np.array([4., 2., 6., 1., 8., 3., 7., 2.])
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> al.MAX(a3, b3)
    array([4., 5., 6., 7., 8., 6., 7., 8.])

5. ``start`` / ``end`` 只计算 [start, end) 区间, 区间外全部为 nan::

    >>> al.set_ctx(start=2, end=6, groups=1, flags=0)
    >>> al.MAX(a3, b3)
    array([nan, nan,  6.,  7.,  8.,  6., nan, nan])
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
OPS = ["max", "min"]


# ---------------------------------------------------------------------------
# numpy 对比实现
# ---------------------------------------------------------------------------


def _max_ref(a, b, op):
  """无 nan 数据上的逐元素 max / min 参考实现."""
  a = np.asarray(a, dtype=np.float64)
  b = np.asarray(b, dtype=np.float64)
  assert not np.isnan(a).any()
  assert not np.isnan(b).any()
  return np.maximum(a, b) if op == "max" else np.minimum(a, b)


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


def _expected(a, b, op, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 逐点计算参考值, 只保留 [start, end) 区间.

  SKIP_NAN 语义: 单侧 nan 取另一侧有效值, 双侧 nan 输出 nan; 不跳过时任一
  nan 就输出 nan. 参考实现本身只处理无 nan 数据.
  """
  a = np.asarray(a, dtype=np.float64)
  b = np.asarray(b, dtype=np.float64)
  skip = bool(flags & al.FLAG_SKIP_NAN)
  parts = []
  for chunk_a, chunk_b in zip(np.split(a, groups), np.split(b, groups)):
    n = len(chunk_a)
    s, e = _calc_range(n, start, end)
    out = np.full(n, np.nan)
    for i in range(s, e):
      av, bv = chunk_a[i], chunk_b[i]
      na, nb = np.isnan(av), np.isnan(bv)
      if na and nb:
        continue
      if na or nb:
        if skip:
          out[i] = bv if na else av
        continue
      out[i] = _max_ref(np.array([av]), np.array([bv]), op)[0]
    parts.append(out)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(a, b, op, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  fn = al.MAX if op == "max" else al.MIN
  return np.asarray(
    fn(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)),
    dtype=np.float64,
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261106)
  return rng.normal(10.0, 2.0, n), rng.normal(10.0, 2.0, n)


def _nan_data(n=N):
  a, b = _clean_data(n)
  a[NAN_POS[NAN_POS < n]] = np.nan
  shifted = NAN_POS + 2
  b[shifted[shifted < n]] = np.nan
  b[NAN_POS[:3][NAN_POS[:3] < n]] = np.nan
  return a, b


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_max_matches_numpy_elementwise(op, flags):
  """无 nan 数据: 与 numpy maximum / minimum 逐点比较."""
  a, b = _clean_data()
  actual = _calc(a, b, op, flags=flags)
  _assert_close(actual, _expected(a, b, op, flags=flags))
  _assert_close(actual, _max_ref(a, b, op))
  assert np.isfinite(actual).all()


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_max_all_flag_combinations_with_nan(op, flags):
  """四种 flag 组合在含 nan 数据上都与参考实现一致."""
  a, b = _nan_data()
  actual = _calc(a, b, op, flags=flags)
  _assert_close(actual, _expected(a, b, op, flags=flags))


@pytest.mark.parametrize("op", OPS)
def test_max_skip_nan_takes_other_side(op):
  """FLAG_SKIP_NAN: 单侧 nan 时结果等于另一侧的有效值."""
  a, b = _nan_data()
  flags = al.FLAG_SKIP_NAN
  actual = _calc(a, b, op, flags=flags)
  _assert_close(actual, _expected(a, b, op, flags=flags))

  na, nb = np.isnan(a), np.isnan(b)
  _assert_close(actual[na & ~nb], b[na & ~nb])
  _assert_close(actual[nb & ~na], a[nb & ~na])
  assert np.isnan(actual[na & nb]).all()
  _assert_close(
    actual[~na & ~nb], _max_ref(a[~na & ~nb], b[~na & ~nb], op)
  )


@pytest.mark.parametrize("op", OPS)
def test_max_both_nan_stays_nan(op):
  """两侧都是 nan 的位置在 SKIP_NAN 下仍输出 nan."""
  a = np.array([np.nan, 2.0, np.nan, 4.0])
  b = np.array([np.nan, np.nan, 3.0, 1.0])
  for flags in FLAG_COMBOS:
    actual = _calc(a, b, op, flags=flags)
    _assert_close(actual, _expected(a, b, op, flags=flags))
    assert np.isnan(actual[0])


@pytest.mark.parametrize("op", OPS)
def test_max_strictly_cycle_has_no_effect(op):
  """FLAG_STRICTLY_CYCLE 被忽略: 与相同 skip 设置的结果一致."""
  a, b = _nan_data()
  for skip in (0, al.FLAG_SKIP_NAN):
    plain = _calc(a, b, op, flags=skip)
    strict = _calc(a, b, op, flags=skip | al.FLAG_STRICTLY_CYCLE)
    _assert_close(strict, plain)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("flags", [0, al.FLAG_SKIP_NAN])
@pytest.mark.parametrize("groups", GROUPS)
def test_max_groups(op, flags, groups):
  """group-major 分块: 逐元素结果不变, 只影响 start/end 下标解析."""
  a, b = _nan_data()
  actual = _calc(a, b, op, groups=groups, flags=flags)
  _assert_close(
    actual, _expected(a, b, op, groups=groups, flags=flags)
  )
  _assert_close(actual, _expected(a, b, op, flags=flags))


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("flags", [0, al.FLAG_SKIP_NAN])
@pytest.mark.parametrize("start,end", START_END)
def test_max_start_end(op, flags, start, end):
  """start/end: 仅 [start, end) 有结果, 区间外必须全是 nan."""
  a, b = _nan_data()
  actual = _calc(a, b, op, start=start, end=end, flags=flags)
  _assert_close(
    actual, _expected(a, b, op, start=start, end=end, flags=flags)
  )
  s, e = _calc_range(len(a), start, end)
  if s >= e:
    assert np.isnan(actual).all()
  else:
    assert np.isnan(actual[:s]).all()
    assert np.isnan(actual[e:]).all()


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("flags", [0, al.FLAG_SKIP_NAN])
@pytest.mark.parametrize("groups", [1, 2, 3])
def test_max_start_end_with_groups(op, flags, groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  a, b = _nan_data()
  start, end = 6, 25
  ctx = dict(start=start, end=end, groups=groups, flags=flags)
  actual = _calc(a, b, op, **ctx)
  _assert_close(actual, _expected(a, b, op, **ctx))
  chunk_size = len(a) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


@pytest.mark.parametrize("op", OPS)
def test_max_no_window_semantics(op):
  """没有滚动窗口: 打乱数据顺序后结果只是同样被打乱."""
  a, b = _clean_data()
  rng = np.random.default_rng(20261106)
  order = rng.permutation(len(a))
  actual = _calc(a, b, op, flags=al.FLAG_SKIP_NAN)
  shuffled = _calc(
    a[order], b[order], op, flags=al.FLAG_SKIP_NAN
  )
  _assert_close(shuffled, actual[order])


def test_max_edge_cases():
  """相等值、常数列、全 nan、单元素、±Inf、长度不匹配报错."""
  ones = np.ones(10)
  twos = np.full(10, 2.0)
  for op in OPS:
    _assert_close(_calc(ones, ones, op), ones)
    expected = twos if op == "max" else ones
    _assert_close(_calc(ones, twos, op), expected)

    all_nan = np.full(10, np.nan)
    for flags in FLAG_COMBOS:
      assert np.isnan(_calc(all_nan, all_nan, op, flags=flags)).all()

    _assert_close(
      _calc(np.array([3.0]), np.array([5.0]), op),
      np.array([5.0 if op == "max" else 3.0]),
    )

  # is_normal 只排除 NaN, ±Inf 是有效值
  a = np.array([np.inf, -np.inf, 1.0, np.nan])
  b = np.array([1.0, np.inf, -np.inf, np.nan])
  _assert_close(_calc(a, b, "max"), np.array([np.inf, np.inf, 1.0, np.nan]))
  _assert_close(_calc(a, b, "min"), np.array([1.0, -np.inf, -np.inf, np.nan]))
  _assert_close(
    _calc(a, b, "max", flags=al.FLAG_SKIP_NAN),
    np.array([np.inf, np.inf, 1.0, np.nan]),
  )

  with pytest.raises(ValueError):
    _calc(np.ones(5), np.ones(3), "max")
