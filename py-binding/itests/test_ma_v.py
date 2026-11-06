# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""MA_V (逐点可变窗口的简单移动平均) 集成测试.

算法简介
--------
MA_V 是 MA 的变周期版本, 每个位置 ``i`` 用 ``periods[i]`` 决定窗口长度::

    MA_V(x, p)[i] = mean(x[max(0, i + 1 - p[i]) : i + 1])

``p[i] = 0`` 时没有窗口, 输出 nan. 窗口未满 (``i < p[i] - 1``) 时, 普通
模式用已有的部分窗口求均值; ``FLAG_STRICTLY_CYCLE`` 要求窗口填满
``p[i]`` 个值, 这些位置输出 nan. 不跳过 nan 时, 窗口内任一个 nan (包括
``x[i]`` 本身) 都会让该位置输出 nan.

``FLAG_SKIP_NAN`` 让 nan 完全不参与运算: 只在 ``[start, end)`` 区间内收集
有效值, 第 k 个有效值用它之前最近的 ``min(k + 1, p[i])`` 个有效值求均值,
nan 位置本身输出 nan. 这等价于把有效值压紧 (compact) 后跑参考实现再映射
回有效位置. 组合 ``FLAG_STRICTLY_CYCLE`` 时还额外要求最近 ``p[i]`` 个
有效值恰好占据 ``p[i]`` 个连续原始位置 (即窗口内不能有 nan 间隙), 否则
输出 nan.

``groups`` 按 group-major 切成等长 chunk, 每个 chunk 独立计算, ``periods``
数组同样按 chunk 切分. start/end 是每个 chunk 内的相对下标, 只计算
[start, end) 区间; 普通模式下窗口可引用 start 之前的数据, SKIP_NAN 模式下
有效值只在区间内压紧.

对比基准 ``_ma_v_ref`` / ``_ma_v_skip_ref`` 用 numpy 独立实现, 只描述算法
逻辑, 不调用 alpha, 也不实现 Group / start-end 机制; 这些上下文语义由
``_expected`` 单独描述.

`set_ctx` 只覆盖传入的参数, 因此每次调用前都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> x = np.arange(1., 11.)
    >>> p = np.array([0, 1, 2, 3, 2, 1, 3, 0, 2, 4], dtype=np.uint64)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算, 每个位置使用自己的窗口长度; ``p[i] = 0`` 输出 nan::

    >>> np.round(al.MA_V(x, p), 3).tolist()
    [nan, 2.0, 2.5, 3.0, 4.5, 6.0, 6.0, nan, 8.5, 8.5]

2. ``FLAG_STRICTLY_CYCLE``: 窗口未满的位置输出 nan::

    >>> p3 = np.full(10, 3, dtype=np.uint64)
    >>> np.round(al.MA_V(x, p3), 3).tolist()
    [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.MA_V(x, p3), 3).tolist()
    [nan, nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

3. 不跳过 nan: 窗口内出现 nan 的位置输出 nan::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.MA_V(xn, p), 3).tolist()
    [nan, 2.0, 2.5, nan, nan, 6.0, 6.0, nan, 8.5, 8.5]

4. ``FLAG_SKIP_NAN``: 压紧后的有效值为 ``[1, 2, 3, 5, 6, 7, 8, 9, 10]``::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.MA_V(xn, p), 3).tolist()
    [nan, 2.0, 2.5, nan, 4.0, 6.0, 6.0, nan, 8.5, 8.5]

   ``FLAG_SKIP_NAN | FLAG_STRICTLY_CYCLE`` 额外要求窗口内无 nan 间隙,
   i=4 处重新变为 nan::

    >>> al.set_ctx(start=0, end=0, groups=1,
    ...            flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.MA_V(xn, p), 3).tolist()
    [nan, 2.0, 2.5, nan, nan, 6.0, 6.0, nan, 8.5, 8.5]

5. ``groups`` 把数据按 group-major 切块, 每块窗口独立::

    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.MA_V(x, p), 3).tolist()
    [nan, 2.0, 2.5, 3.0, 4.5, 6.0, 6.5, nan, 8.5, 8.5]

6. ``start`` / ``end`` 只输出 [start, end) 区间::

    >>> al.set_ctx(start=2, end=8, groups=1, flags=0)
    >>> np.round(al.MA_V(x, p), 3).tolist()
    [nan, nan, 2.5, 3.0, 4.5, 6.0, 6.0, nan, nan, nan]
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


def _ma_v_ref(x, periods, *, strict=False):
  """flags=0 的参考实现: 每个位置按自己的 p[i] 取部分窗口均值.

  窗口内含 nan 则输出 nan; ``strict`` 时窗口未满 p[i] 个值的位置输出 nan.
  """
  x = np.asarray(x, dtype=np.float64)
  p = np.asarray(periods, dtype=np.int64)
  out = np.full(len(x), np.nan)
  for i in range(len(x)):
    q = int(p[i])
    if q == 0:
      continue
    lo = max(0, i + 1 - q)
    w = x[lo : i + 1]
    if np.isnan(w).any():
      continue
    if strict and i < q - 1:
      continue
    out[i] = w.sum() / q if strict else w.mean()
  return out


def _ma_v_skip_ref(chunk, periods, s, e, strict):
  """SKIP_NAN 参考实现: 只在 [s, e) 内压紧有效值后按 p[i] 求均值.

  非严格模式取最近 ``min(l, p)`` 个有效值; 严格模式要求最近 ``p`` 个有效
  值恰好占据 ``p`` 个连续原始位置 (窗口内无 nan 间隙).
  """
  chunk = np.asarray(chunk, dtype=np.float64)
  p = np.asarray(periods, dtype=np.int64)
  out = np.full(len(chunk), np.nan)
  vals = []
  idxs = []
  for i in range(s, e):
    if np.isnan(chunk[i]):
      continue
    vals.append(chunk[i])
    idxs.append(i)
    q = int(p[i])
    if q == 0:
      continue
    l = len(vals)
    if strict:
      if l >= q and (i - idxs[l - q] + 1 == q):
        out[i] = sum(vals[l - q :]) / q
    else:
      cnt = min(l, q)
      out[i] = sum(vals[l - cnt :]) / cnt
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
  """按 chunk 计算参考值, 只保留 alpha 应输出的 [start, end) 区间."""
  x = np.asarray(x, dtype=np.float64)
  p = np.asarray(periods, dtype=np.int64)
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  skip = bool(flags & al.FLAG_SKIP_NAN)
  parts = []
  for chunk_x, chunk_p in zip(np.split(x, groups), np.split(p, groups)):
    n = len(chunk_x)
    s, e = _calc_range(n, start, end)
    if skip:
      ref = _ma_v_skip_ref(chunk_x, chunk_p, s, e, strict)
    else:
      ref = _ma_v_ref(chunk_x, chunk_p, strict=strict)
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
    al.MA_V(
      np.asarray(x, dtype=np.float64),
      np.asarray(periods, dtype=np.uint64),
    ),
    dtype=np.float64,
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261116)
  return rng.normal(loc=10.0, scale=2.0, size=n).astype(np.float64)


def _nan_data(n=N):
  x = _clean_data(n)
  x[NAN_POS[NAN_POS < n]] = np.nan
  return x


def _periods_data(n=N, max_p=12):
  rng = np.random.default_rng(20261117)
  return rng.integers(0, max_p + 1, size=n).astype(np.uint64)


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


def test_ma_v_matches_reference():
  """flags=0, groups=1: 与 numpy 逐点窗口均值参考实现比较."""
  x = _clean_data()
  p = _periods_data()
  actual = _calc(x, p, flags=0)
  _assert_close(actual, _ma_v_ref(x, p))
  _assert_close(actual, _expected(x, p, flags=0))
  assert np.isfinite(actual[p > 0]).all()
  assert np.isnan(actual[p == 0]).all()


def test_ma_v_strictly_cycle():
  """FLAG_STRICTLY_CYCLE: 窗口未满 p[i] 个值的位置输出 nan."""
  x = _clean_data()
  p = _periods_data()
  flags = al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, p, flags=flags)
  _assert_close(actual, _ma_v_ref(x, p, strict=True))
  _assert_close(actual, _expected(x, p, flags=flags))

  short = (p > 0) & (np.arange(len(x)) < p.astype(np.int64) - 1)
  assert np.isnan(actual[short]).all()


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_ma_v_nan_without_skip(flags):
  """无 SKIP_NAN: 窗口内任一个 nan 都让该位置输出 nan."""
  x = _nan_data()
  p = _periods_data()
  actual = _calc(x, p, flags=flags)
  _assert_close(actual, _expected(x, p, flags=flags))
  _assert_close(actual, _ma_v_ref(x, p, strict=bool(flags)))
  assert np.isnan(actual[np.isnan(x)]).all()


def test_ma_v_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 有效位置等于压紧有效值后的参考结果."""
  x = _nan_data()
  p = _periods_data()
  actual = _calc(x, p, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, p, flags=al.FLAG_SKIP_NAN))

  valid = ~np.isnan(x)
  assert np.isnan(actual[~valid]).all()
  _assert_close(actual[valid], _ma_v_ref(x[valid], p[valid], strict=False))


def test_ma_v_skip_nan_matches_compacted_clean_data():
  """SKIP_NAN 等价于把有效位置压紧后, 在无 nan 序列上做 MA_V."""
  x_clean = _clean_data()
  x = _nan_data()
  p = _periods_data()
  valid = ~np.isnan(x)
  actual = _calc(x, p, flags=al.FLAG_SKIP_NAN)
  ref = _ma_v_ref(x_clean[valid], p[valid], strict=False)
  _assert_close(actual[valid], ref)
  assert np.isnan(actual[~valid]).all()


def test_ma_v_skip_nan_and_strictly_cycle():
  """两个 flag 组合: 严格模式要求最近 p 个有效值占据连续原始位置."""
  x = _nan_data()
  p = _periods_data()
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, p, flags=flags)
  _assert_close(actual, _expected(x, p, flags=flags))
  assert np.isnan(actual[np.isnan(x)]).all()

  loose = _calc(x, p, flags=al.FLAG_SKIP_NAN)
  finite = np.isfinite(actual)
  _assert_close(actual[finite], loose[finite])


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_ma_v_all_flag_combinations_with_nan(flags):
  """四种 flag 组合在含 nan 数据上都与参考实现比较."""
  x = _nan_data()
  p = _periods_data()
  actual = _calc(x, p, flags=flags)
  _assert_close(actual, _expected(x, p, flags=flags))


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_ma_v_groups(groups, flags):
  """group-major 分块: 每个 chunk 独立计算, periods 也按块切分."""
  x = _nan_data()
  p = _periods_data()
  actual = _calc(x, p, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, p, groups=groups, flags=flags))


@pytest.mark.parametrize("groups", GROUPS)
def test_ma_v_skip_nan_within_each_group(groups):
  """SKIP_NAN 的压紧过程按 chunk 独立进行, 不会跨 group 借用数据."""
  x = _nan_data()
  p = _periods_data()
  flags = al.FLAG_SKIP_NAN
  actual = _calc(x, p, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, p, groups=groups, flags=flags))
  assert np.isnan(actual[np.isnan(x)]).all()


@pytest.mark.parametrize("start,end", START_END)
@pytest.mark.parametrize("flags", [0, al.FLAG_SKIP_NAN])
def test_ma_v_start_end(start, end, flags):
  """start/end: 只校验 [start, end) 区间, 区间外全部为 nan."""
  x = _nan_data()
  p = _periods_data()
  actual = _calc(x, p, start=start, end=end, flags=flags)
  _assert_close(actual, _expected(x, p, start=start, end=end, flags=flags))
  s, e = _calc_range(len(x), start, end)
  assert np.isnan(actual[:s]).all()
  assert np.isnan(actual[e:]).all()


@pytest.mark.parametrize("groups", GROUPS)
def test_ma_v_start_end_with_groups(groups):
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


def test_ma_v_edge_cases():
  """p=0 / p=1 / 超大 p / 常数序列 / 单元素 / 空数组 / 长度不匹配报错."""
  x = _clean_data(20)
  p0 = np.zeros(20, dtype=np.uint64)
  assert np.isnan(_calc(x, p0, flags=0)).all()

  p1 = np.ones(20, dtype=np.uint64)
  _assert_close(_calc(x, p1, flags=0), x)

  p_big = np.full(20, 100, dtype=np.uint64)
  expected = np.cumsum(x) / np.arange(1, 21)
  _assert_close(_calc(x, p_big, flags=0), expected)
  assert np.isnan(_calc(x, p_big, flags=al.FLAG_STRICTLY_CYCLE)).all()

  const = np.full(20, 3.0)
  p3 = np.full(20, 4, dtype=np.uint64)
  _assert_close(_calc(const, p3, flags=0), const)

  one = np.array([3.0])
  _assert_close(_calc(one, np.array([1], dtype=np.uint64), flags=0), one)

  empty = np.empty(0, dtype=np.float64)
  _assert_close(_calc(empty, np.empty(0, dtype=np.uint64), flags=0), empty)

  with pytest.raises(ValueError):
    _calc(x, np.zeros(10, dtype=np.uint64), flags=0)
