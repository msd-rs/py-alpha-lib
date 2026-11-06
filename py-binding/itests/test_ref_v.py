# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""REF_V (逐点可变周期的引用) 集成测试.

算法简介
--------
REF_V 是 REF 的变周期版本, 每个位置 ``i`` 用 ``periods[i]`` 决定回看几期::

    REF_V(x, p)[i] = x[max(0, i - p[i])]

历史不足 ``p[i]`` 期时, 普通模式用块首值 ``x[0]`` 填充 (等价于把下标
clamp 到 0); ``FLAG_STRICTLY_CYCLE`` 下这些位置没有完整历史, 输出 nan.
``p[i] = 0`` 时直接复制 ``x[i]``. 注意普通模式只读取被引用的位置, 当前
``x[i]`` 是否为 nan 并不影响 ``p[i] > 0`` 时的结果.

``FLAG_SKIP_NAN`` 改为在「有效值序列」上做引用: 只在 ``[start, end)``
区间内收集非 nan 的值, nan 位置本身输出 nan; 第 k 个有效值用它前面第
``p[i]`` 个有效值, 前 ``p[i]`` 个有效位置没有结果 (nan). 这等价于把该区间
的有效值压紧 (compact) 后跑参考实现, 再映射回有效位置; 组合
``FLAG_STRICTLY_CYCLE`` 不改变这一压紧行为.

对比基准 ``_ref_v_ref`` 用 numpy 索引独立实现, 只描述算法逻辑, 不调用
alpha, 也不实现 Group / start-end 机制: Group 由测试按 chunk 分块计算,
start/end 只校验 ``[start, end)`` 区间的值 (区间外 alpha 一律输出 nan).
普通模式下窗口可以引用 start 之前的数据; SKIP_NAN 模式下有效值只在区间内
压紧.

`set_ctx` 只覆盖传入的参数, 因此每次调用前都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> x = np.arange(1., 11.)
    >>> p = np.array([0, 1, 2, 3, 2, 1, 3, 0, 2, 4], dtype=np.uint64)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算, 每个位置使用自己的回看期数::

    >>> al.REF_V(x, p).tolist()
    [1.0, 1.0, 1.0, 1.0, 3.0, 5.0, 4.0, 8.0, 7.0, 6.0]

2. 历史不足时用块首值填充; ``FLAG_STRICTLY_CYCLE`` 则输出 nan::

    >>> p3 = np.full(10, 3, dtype=np.uint64)
    >>> al.REF_V(x, p3).tolist()
    [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> al.REF_V(x, p3).tolist()
    [nan, nan, nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

3. 不跳过 nan 时, 被引用位置是 nan 结果才是 nan::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> al.REF_V(xn, p).tolist()
    [1.0, 1.0, 1.0, 1.0, 3.0, 5.0, nan, 8.0, 7.0, 6.0]

4. ``FLAG_SKIP_NAN``: 压紧后的有效值为 ``[1, 2, 3, 5, 6, 7, 8, 9, 10]``::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> al.REF_V(xn, p).tolist()
    [1.0, 1.0, 1.0, nan, 2.0, 5.0, 3.0, 8.0, 7.0, 6.0]

5. ``groups`` 把数据按 group-major 切块, 每块重新以块首值填充::

    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> al.REF_V(x, p).tolist()
    [1.0, 1.0, 1.0, 1.0, 3.0, 6.0, 6.0, 8.0, 7.0, 6.0]

6. ``start`` / ``end`` 只输出 [start, end) 区间, 但窗口仍可引用之前的数据::

    >>> al.set_ctx(start=2, end=8, groups=1, flags=0)
    >>> al.REF_V(x, p).tolist()
    [nan, nan, 1.0, 1.0, 3.0, 5.0, 4.0, 8.0, nan, nan]
"""

import numpy as np
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
NAN_POS = np.array([3, 17, 18, 29, 44, 52, 77, 101])
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


def _ref_v_ref(x, periods, *, strict=False):
  """REF_V 参考实现: r[i] = x[max(0, i - p[i])].

  ``strict`` 时历史不足 ``p[i]`` 期的位置输出 nan. 允许输入含 nan:
  被引用位置是 nan, 输出就是 nan.
  """
  x = np.asarray(x, dtype=np.float64)
  p = np.asarray(periods, dtype=np.int64)
  idx = np.maximum(np.arange(len(x)) - p, 0)
  out = x[idx].copy()
  if strict:
    out[np.arange(len(x)) < p] = np.nan
  return out


def _compact_ref(seg_x, seg_p):
  """区间内压紧有效值后引用, 再映射回有效位置; nan 位置保持 nan."""
  seg_x = np.asarray(seg_x, dtype=np.float64)
  seg_p = np.asarray(seg_p, dtype=np.int64)
  out = np.full(len(seg_x), np.nan)
  valid = ~np.isnan(seg_x)
  if not valid.any():
    return out

  cx = seg_x[valid]
  cp = seg_p[valid]
  idx = np.maximum(np.arange(len(cx)) - cp, 0)
  ref = cx[idx].copy()
  ref[np.arange(len(cx)) < cp] = np.nan
  out[valid] = ref
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


def _expected(x, periods, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 全量计算参考值, 再只保留 [start, end) 区间."""
  x = np.asarray(x, dtype=np.float64)
  p = np.asarray(periods, dtype=np.int64)
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  skip = bool(flags & al.FLAG_SKIP_NAN)
  parts = []
  for chunk_x, chunk_p in zip(np.split(x, groups), np.split(p, groups)):
    n = len(chunk_x)
    s, e = _calc_range(n, start, end)
    ref = np.full(n, np.nan)
    if s < e:
      if skip:
        ref[s:e] = _compact_ref(chunk_x[s:e], chunk_p[s:e])
      else:
        ref = _ref_v_ref(chunk_x, chunk_p, strict=strict)
        ref[:s] = np.nan
        ref[e:] = np.nan
    parts.append(ref)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(x, periods, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.REF_V(
      np.asarray(x, dtype=np.float64),
      np.asarray(periods, dtype=np.uint64),
    ),
    dtype=np.float64,
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261112)
  return rng.normal(loc=10.0, scale=2.0, size=n).astype(np.float64)


def _nan_data(n=N):
  x = _clean_data(n)
  x[NAN_POS[NAN_POS < n]] = np.nan
  return x


def _periods_data(n=N, max_p=12):
  rng = np.random.default_rng(20261113)
  return rng.integers(0, max_p + 1, size=n).astype(np.uint64)


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


def test_ref_v_matches_reference():
  """flags=0, groups=1: 与 numpy 索引参考实现逐点比较."""
  x = _clean_data()
  p = _periods_data()
  actual = _calc(x, p, flags=0)
  _assert_close(actual, _ref_v_ref(x, p))
  _assert_close(actual, _expected(x, p, flags=0))
  assert np.isfinite(actual).all()


def test_ref_v_strictly_cycle():
  """FLAG_STRICTLY_CYCLE: 历史不足 p[i] 期的位置输出 nan."""
  x = _clean_data()
  p = _periods_data()
  flags = al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, p, flags=flags)
  _assert_close(actual, _ref_v_ref(x, p, strict=True))
  _assert_close(actual, _expected(x, p, flags=flags))

  short = np.arange(len(x)) < p.astype(np.int64)
  assert np.isnan(actual[short]).all()
  assert np.isfinite(actual[~short]).all()


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_ref_v_nan_without_skip(flags):
  """无 SKIP_NAN: 结果只取决于被引用位置, 当前位置的 nan 不传播."""
  x = _nan_data()
  p = _periods_data()
  actual = _calc(x, p, flags=flags)
  _assert_close(actual, _expected(x, p, flags=flags))
  _assert_close(actual, _ref_v_ref(x, p, strict=bool(flags)))

  idx = np.maximum(np.arange(len(x)) - p.astype(np.int64), 0)
  assert np.isnan(actual[np.isnan(x[idx])]).all()


def test_ref_v_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: nan 位置输出 nan, 有效位置等于压紧后的参考结果."""
  x = _nan_data()
  p = _periods_data()
  actual = _calc(x, p, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, p, flags=al.FLAG_SKIP_NAN))

  valid = ~np.isnan(x)
  assert np.isnan(actual[~valid]).all()
  compact = _compact_ref(x, p)
  _assert_close(actual[valid], compact[valid])


def test_ref_v_skip_nan_matches_compacted_clean_data():
  """SKIP_NAN 等价于把有效位置压紧后, 在无 nan 序列上做引用."""
  x_clean = _clean_data()
  x = _nan_data()
  p = _periods_data()
  valid = ~np.isnan(x)
  actual = _calc(x, p, flags=al.FLAG_SKIP_NAN)

  cx = x_clean[valid]
  cp = p[valid].astype(np.int64)
  idx = np.maximum(np.arange(len(cx)) - cp, 0)
  ref = cx[idx].copy()
  ref[np.arange(len(cx)) < cp] = np.nan
  _assert_close(actual[valid], ref)
  assert np.isnan(actual[~valid]).all()


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_ref_v_all_flag_combinations_with_nan(flags):
  """四种 flag 组合在含 nan 数据上都与参考实现比较."""
  x = _nan_data()
  p = _periods_data()
  actual = _calc(x, p, flags=flags)
  _assert_close(actual, _expected(x, p, flags=flags))


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_ref_v_groups(groups, flags):
  """group-major 分块: 每块独立以块首值填充, periods 也按块切分."""
  x = _nan_data()
  p = _periods_data()
  actual = _calc(x, p, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, p, groups=groups, flags=flags))


@pytest.mark.parametrize("start,end", START_END)
@pytest.mark.parametrize("flags", [0, al.FLAG_SKIP_NAN])
def test_ref_v_start_end(start, end, flags):
  """start/end: 只校验 [start, end) 区间, 普通模式窗口可越界引用."""
  x = _nan_data()
  p = _periods_data()
  actual = _calc(x, p, start=start, end=end, flags=flags)
  _assert_close(actual, _expected(x, p, start=start, end=end, flags=flags))
  s, e = _calc_range(len(x), start, end)
  assert np.isnan(actual[:s]).all()
  assert np.isnan(actual[e:]).all()


@pytest.mark.parametrize("groups", GROUPS)
def test_ref_v_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  x = _nan_data()
  p = _periods_data()
  start, end = 6, 25
  actual = _calc(x, p, start=start, end=end, groups=groups, flags=0)
  _assert_close(
    actual, _expected(x, p, start=start, end=end, groups=groups, flags=0)
  )
  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


def test_ref_v_edge_cases():
  """p=0 / 超大 p / 单元素 / 空数组 / 长度不匹配报错."""
  x = _clean_data(20)
  p0 = np.zeros(20, dtype=np.uint64)
  _assert_close(_calc(x, p0, flags=0), x)

  p_big = np.full(20, 100, dtype=np.uint64)
  _assert_close(_calc(x, p_big, flags=0), np.full(20, x[0]))
  assert np.isnan(_calc(x, p_big, flags=al.FLAG_STRICTLY_CYCLE)).all()
  assert np.isnan(_calc(x, p_big, flags=al.FLAG_SKIP_NAN)).all()

  one = np.array([3.0])
  _assert_close(_calc(one, np.array([0], dtype=np.uint64), flags=0), one)
  _assert_close(_calc(one, np.array([5], dtype=np.uint64), flags=0), one)

  empty = np.empty(0, dtype=np.float64)
  _assert_close(_calc(empty, np.empty(0, dtype=np.uint64), flags=0), empty)

  with pytest.raises(ValueError):
    _calc(x, np.zeros(10, dtype=np.uint64), flags=0)
