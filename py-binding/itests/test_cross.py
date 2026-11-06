# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""CROSS / RCROSS (金叉 / 死叉) 集成测试.

算法简介
--------
两个算子比较相邻两个位置的两条序列, 输出布尔信号 (i=0 恒为 False)::

    CROSS(a, b)[i]  = a[i-1] < b[i-1] and a[i] >= b[i]   # 上穿
    RCROSS(a, b)[i] = a[i-1] > b[i-1] and a[i] <= b[i]   # 下穿

``flags=0`` 时要求 ``a[i], b[i], a[i-1], b[i-1]`` 四个值都非 nan, 任一为
nan 则该位置为 False; 窗口可引用 start 之前的数据. 设置 ``FLAG_SKIP_NAN``
后改为「上一个有效配对」状态机: 只保留 a、b 同时有效的位置, 在这些有效配对
构成的序列上比较相邻两项, nan 位置输出 False 且不打断状态. 该状态只在
``[start, end)`` 内维护, 不引用 start 之前的数据.

``FLAG_STRICTLY_CYCLE`` 被忽略; bool 输出无法表达 nan, 区间外与无效位置一律
用 False 表示. ``groups`` 按 group-major 切成等长 chunk 独立计算, 每块的首个
位置恒为 False; start/end 是每个 chunk 内的相对下标.

参考实现 ``_cross_ref`` / ``_cross_skip_ref`` 用 numpy 向量化比较独立复现
算法逻辑, 不调用 alpha, 也不实现 Group / start-end 机制; 这些上下文语义由
``_expected`` 单独描述: 按 chunk 分块, flags=0 时先全量计算再屏蔽区间外,
SKIP_NAN 时只在 [start, end) 内压紧有效配对.

`set_ctx` 只覆盖传入的参数, 所以每次调用都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=120)
    >>> a = np.array([1., 2., 3., 2., 4., 1.])
    >>> b = np.full(6, 2.)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. ``CROSS``: 只有 i=1 满足 a[0]=1 < b[0]=2 且 a[1]=2 >= b[1]=2::

    >>> al.CROSS(a, b)
    array([False,  True, False, False, False, False])

2. ``RCROSS``: i=3 (3>2 且 2<=2) 与 i=5 (4>2 且 1<=2) 为真::

    >>> al.RCROSS(a, b)
    array([False, False, False,  True, False,  True])

3. 不跳过 nan: 把 a[2] 换成 nan 后, 相邻窗口 i=2、i=3 都失效::

    >>> an = a.copy(); an[2] = np.nan
    >>> al.CROSS(an, b)
    array([False,  True, False, False, False, False])

4. ``FLAG_SKIP_NAN``: nan 被跳过, i=3 的上一个有效配对是 i=1 (a=2 >= b=2),
   因此不会误报上穿::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> al.CROSS(an, b)
    array([False,  True, False, False, False, False])

5. ``groups=2``: 第二块从块首重新开始, i=5 的跨块比较不算信号::

    >>> a2 = np.array([1., 2., 3., 2., 1., 2., 3., 2., 1., 2.])
    >>> b2 = np.full(10, 2.)
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> al.CROSS(a2, b2)
    array([False,  True, False, False, False, False, False, False, False,  True])

6. ``start`` / ``end`` 只计算 [start, end) 区间, 区间外全部为 False::

    >>> al.set_ctx(start=2, end=8, groups=1, flags=0)
    >>> al.CROSS(a2, b2)
    array([False, False, False, False, False,  True, False, False, False, False])

7. 相等序列: 上穿/下穿都不成立 (需要严格小于 / 严格大于的前置状态)::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> al.CROSS(np.ones(4), np.ones(4))
    array([False, False, False, False])
"""

import numpy as np
import pytest

import alpha as al

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
REVERSE = [False, True]


# ---------------------------------------------------------------------------
# numpy 对比实现
# ---------------------------------------------------------------------------


def _cross_ref(a, b, reverse):
  """flags=0 的向量化参考实现: 比较相邻两个原始位置."""
  a = np.asarray(a, dtype=np.float64)
  b = np.asarray(b, dtype=np.float64)
  out = np.zeros(len(a), dtype=bool)
  if len(a) > 1:
    prev = a[:-1] > b[:-1] if reverse else a[:-1] < b[:-1]
    curr = a[1:] <= b[1:] if reverse else a[1:] >= b[1:]
    valid = ~(
      np.isnan(a[:-1])
      | np.isnan(b[:-1])
      | np.isnan(a[1:])
      | np.isnan(b[1:])
    )
    out[1:] = prev & curr & valid
  return out


def _cross_skip_ref(a, b, reverse, lo, hi):
  """SKIP_NAN 参考实现: 只在 [lo, hi) 内压紧有效配对后比较相邻项."""
  a = np.asarray(a, dtype=np.float64)
  b = np.asarray(b, dtype=np.float64)
  out = np.zeros(len(a), dtype=bool)
  valid = ~np.isnan(a) & ~np.isnan(b)
  idx = np.flatnonzero(valid[lo:hi]) + lo
  if len(idx) > 1:
    va, vb = a[idx], b[idx]
    prev = va[:-1] > vb[:-1] if reverse else va[:-1] < vb[:-1]
    curr = va[1:] <= vb[1:] if reverse else va[1:] >= vb[1:]
    sig = np.zeros(len(idx), dtype=bool)
    sig[1:] = prev & curr
    out[idx] = sig
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


def _expected(a, b, reverse, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 分块计算参考值, 再只保留 [start, end) 区间.

  flags=0 时先在整个 chunk 上全量计算 (窗口可引用 start 之前的数据), 再把
  区间外屏蔽成 False; SKIP_NAN 时状态只在区间内维护.
  """
  a = np.asarray(a, dtype=np.float64)
  b = np.asarray(b, dtype=np.float64)
  skip = bool(flags & al.FLAG_SKIP_NAN)
  parts = []
  for chunk_a, chunk_b in zip(np.split(a, groups), np.split(b, groups)):
    n = len(chunk_a)
    s, e = _calc_range(n, start, end)
    if skip:
      ref = _cross_skip_ref(chunk_a, chunk_b, reverse, s, e)
    else:
      ref = _cross_ref(chunk_a, chunk_b, reverse)
      ref[:s] = False
      ref[e:] = False
    parts.append(ref)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(a, b, reverse, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  fn = al.RCROSS if reverse else al.CROSS
  return np.asarray(
    fn(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)),
    dtype=bool,
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261106)
  return rng.normal(10.0, 2.0, n), rng.normal(10.0, 2.0, n)


def _nan_data(n=N):
  a, b = _clean_data(n)
  a[NAN_POS[NAN_POS < n]] = np.nan
  shifted = NAN_POS + 2
  b[shifted[shifted < n]] = np.nan
  return a, b


def _assert_equal(actual, expected):
  assert actual.shape == expected.shape
  assert actual.dtype == np.bool_
  np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reverse", REVERSE)
def test_cross_matches_numpy_reference(reverse):
  """flags=0, groups=1: 与 numpy 相邻比较的参考实现逐点比较."""
  a, b = _clean_data()
  actual = _calc(a, b, reverse)
  _assert_equal(actual, _expected(a, b, reverse))
  _assert_equal(actual, _cross_ref(a, b, reverse))
  assert not actual[0]
  assert actual.any()


@pytest.mark.parametrize("reverse", REVERSE)
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_cross_nan_without_skip_invalidates_pairs(reverse, flags):
  """无 SKIP_NAN 时, 相邻两对四个值中任一 nan 都让该位置为 False."""
  a, b = _nan_data()
  actual = _calc(a, b, reverse, flags=flags)
  _assert_equal(actual, _expected(a, b, reverse, flags=flags))
  invalid = (
    np.isnan(a)
    | np.isnan(b)
    | np.roll(np.isnan(a) | np.isnan(b), 1)
  )
  invalid[0] = True
  assert not actual[invalid].any()


@pytest.mark.parametrize("reverse", REVERSE)
def test_cross_skip_nan_compacts_valid_pairs(reverse):
  """FLAG_SKIP_NAN: 等价于删除无效配对后再比较相邻项."""
  a, b = _nan_data()
  flags = al.FLAG_SKIP_NAN
  actual = _calc(a, b, reverse, flags=flags)
  _assert_equal(actual, _expected(a, b, reverse, flags=flags))
  _assert_equal(actual, _cross_skip_ref(a, b, reverse, 0, len(a)))

  invalid = np.isnan(a) | np.isnan(b)
  assert not actual[invalid].any()
  idx = np.flatnonzero(~invalid)
  va, vb = a[idx], b[idx]
  prev = va[:-1] > vb[:-1] if reverse else va[:-1] < vb[:-1]
  curr = va[1:] <= vb[1:] if reverse else va[1:] >= vb[1:]
  _assert_equal(actual[idx[1:]], prev & curr)


@pytest.mark.parametrize("flags", FLAG_COMBOS)
@pytest.mark.parametrize("reverse", REVERSE)
def test_cross_all_flag_combinations_with_nan(flags, reverse):
  """四种 flag 组合在含 nan 数据上都与参考实现一致."""
  a, b = _nan_data()
  actual = _calc(a, b, reverse, flags=flags)
  _assert_equal(actual, _expected(a, b, reverse, flags=flags))


@pytest.mark.parametrize("reverse", REVERSE)
def test_cross_strictly_cycle_has_no_effect(reverse):
  """FLAG_STRICTLY_CYCLE 被忽略: 与相同 skip 设置的 flags 结果一致."""
  a, b = _nan_data()
  for skip in (0, al.FLAG_SKIP_NAN):
    plain = _calc(a, b, reverse, flags=skip)
    strict = _calc(a, b, reverse, flags=skip | al.FLAG_STRICTLY_CYCLE)
    _assert_equal(strict, plain)


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("flags", [0, al.FLAG_SKIP_NAN])
@pytest.mark.parametrize("reverse", REVERSE)
def test_cross_groups(groups, flags, reverse):
  """group-major 分块: 每块首位置为 False, 窗口/状态不跨边界."""
  a, b = _clean_data()
  actual = _calc(a, b, reverse, groups=groups, flags=flags)
  _assert_equal(
    actual, _expected(a, b, reverse, groups=groups, flags=flags)
  )
  chunk_size = len(a) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert not actual[lo]
    _assert_equal(
      actual[lo : lo + chunk_size],
      _expected(
        a[lo : lo + chunk_size],
        b[lo : lo + chunk_size],
        reverse,
        flags=flags,
      ),
    )


@pytest.mark.parametrize("start,end", START_END)
@pytest.mark.parametrize("reverse", REVERSE)
def test_cross_start_end(start, end, reverse):
  """start/end: 仅 [start, end) 可能有信号, 区间外必须全 False."""
  a, b = _clean_data()
  actual = _calc(a, b, reverse, start=start, end=end)
  _assert_equal(actual, _expected(a, b, reverse, start=start, end=end))
  s, e = _calc_range(len(a), start, end)
  assert not actual[:s].any()
  assert not actual[e:].any()


def test_cross_window_uses_pre_start_data():
  """flags=0 时 i=start 会与 start-1 比较, 历史数据仍参与判断."""
  a = np.array([1.0, 2.0, 3.0, 1.0, 2.0])
  b = np.full(5, 2.0)
  actual = _calc(a, b, False, start=1, end=4)
  _assert_equal(actual, np.array([False, True, False, False, False]))
  _assert_equal(actual, _expected(a, b, False, start=1, end=4))


def test_cross_skip_nan_state_starts_at_start():
  """SKIP_NAN 状态只在 [start, end) 内维护, 不借用 start 之前的数据."""
  a = np.array([1.0, 3.0, 1.0, 3.0, 1.0])
  b = np.full(5, 2.0)
  flags = al.FLAG_SKIP_NAN
  actual = _calc(a, b, False, start=1, end=5, flags=flags)
  _assert_equal(
    actual, np.array([False, False, False, True, False])
  )
  _assert_equal(
    actual, _expected(a, b, False, start=1, end=5, flags=flags)
  )
  assert not actual[1]  # i=1 没有区间内的上一个有效配对


@pytest.mark.parametrize("groups", [1, 2, 3])
@pytest.mark.parametrize("flags", [0, al.FLAG_SKIP_NAN])
def test_cross_start_end_with_groups(groups, flags):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  a, b = _nan_data()
  start, end = 6, 25
  ctx = dict(start=start, end=end, groups=groups, flags=flags)
  actual = _calc(a, b, False, **ctx)
  _assert_equal(actual, _expected(a, b, False, **ctx))
  chunk_size = len(a) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert not actual[lo : lo + start].any()
    assert not actual[lo + end : lo + chunk_size].any()


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_cross_skip_nan_with_negative_end_and_groups(groups):
  """SKIP_NAN + 负数 end + groups 组合."""
  a, b = _nan_data()
  ctx = dict(start=5, end=-5, groups=groups, flags=al.FLAG_SKIP_NAN)
  actual = _calc(a, b, False, **ctx)
  _assert_equal(actual, _expected(a, b, False, **ctx))


def test_cross_edge_cases():
  """恒小于 / 恒大于 / 相等序列, 单元素数组, 长度不匹配报错."""
  ones = np.ones(20)
  twos = np.full(20, 2.0)
  zeros = np.zeros(20)

  for reverse in REVERSE:
    _assert_equal(_calc(ones, twos, reverse), np.zeros(20, dtype=bool))
    _assert_equal(_calc(zeros, ones, reverse), np.zeros(20, dtype=bool))
    _assert_equal(_calc(ones, ones, reverse), np.zeros(20, dtype=bool))
    _assert_equal(
      _calc(np.array([3.0]), np.array([1.0]), reverse),
      np.zeros(1, dtype=bool),
    )

  # 相等值也属于「>= / <=」, 前置状态必须严格小于 / 大于
  a = np.array([1.0, 2.0, 2.0, 2.0])
  b = np.full(4, 2.0)
  _assert_equal(_calc(a, b, False), np.array([False, True, False, False]))
  _assert_equal(_calc(a, b, True), np.array([False, False, False, False]))

  with pytest.raises(ValueError):
    _calc(np.ones(5), np.ones(3), False)
