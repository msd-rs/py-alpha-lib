# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""LWMA (线性加权移动平均) 集成测试.

算法简介
--------
LWMA 对最近 `periods` 个价格按线性递增的权重 1, 2, ..., periods 加权求平均::

    LWMA[i] = sum(j * x[i - periods + j] for j in 1..periods) / (periods * (periods + 1) / 2)

窗口未满 (i < periods - 1) 时输出 nan. 参考实现使用 pandas 的 ``rolling``
(``min_periods=periods``) 配合 numpy 权重向量, 不实现 SKIP_NAN / Group / start-end
机制: 对比按 chunk 分块计算, 只校验 alpha 在 [start, end) 区间内的结果.

`set_ctx` 只会覆盖传入的参数, 因此每次调用前都显式重置 start/end/groups/flags.

场景与结果示例
--------------
1. 常规计算 (窗口未满时输出 nan)::

    >>> import numpy as np
    >>> import alpha as al
    >>> x = np.arange(1., 11.)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> al.LWMA(x, 4)
    array([nan, nan, nan,  3.,  4.,  5.,  6.,  7.,  8.,  9.])

   权重和为 1+2+3+4=10, 第一个窗口 (1*1 + 2*2 + 3*3 + 4*4) / 10 = 3.0.

2. ``FLAG_STRICTLY_CYCLE`` 在 LWMA 上同样是「窗口未满输出 nan」::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> al.LWMA(x, 4)
    array([nan, nan, nan,  3.,  4.,  5.,  6.,  7.,  8.,  9.])

   不设置 ``FLAG_SKIP_NAN`` 时, 窗口内出现 nan 会使该窗口结果为 nan::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> al.LWMA(xn, 4)
    array([nan, nan, nan, nan, nan, nan, nan,  7.,  8.,  9.])

3. ``FLAG_SKIP_NAN`` 让 nan 完全不参与运算, 等价于把有效值压紧
   (compact) 后再计算参考值, 再映射回原始有效位置::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> al.LWMA(xn, 4)
    array([nan, nan, nan, nan, 3.4, 4.7, 5.9, 7. , 8. , 9. ])

   压紧后的数据为 [1, 2, 3, 5, 6, 7, 8, 9, 10], 其参考 LWMA(4) 为
   [nan, nan, nan, 3.4, 4.7, 5.9, 7.0, 8.0, 9.0], 与上面有效位置的结果一致.
   nan 所在位置本身仍输出 nan.

4. ``FLAG_SKIP_NAN | FLAG_STRICTLY_CYCLE`` 要求窗口是连续的、无 nan 的完整窗口,
   因此结果与 pandas 直接在含 nan 的原始数据上做 rolling 一致::

    >>> al.set_ctx(start=0, end=0, groups=1,
    ...            flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
    >>> al.LWMA(xn, 4)
    array([nan, nan, nan, nan, nan, nan, nan,  7.,  8.,  9.])

5. ``groups`` 把一维数据按 group-major 切块独立计算, 每块重新开始计数::

    >>> y = np.arange(1., 13.)
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> al.LWMA(y, 4)
    array([nan, nan, nan,  3.,  4.,  5., nan, nan, nan,  9., 10., 11.])

6. ``start`` / ``end`` 只计算 [start, end) 区间, 区间外全部为 nan,
   但窗口仍可引用 start 之前的数据::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> al.LWMA(x, 4)
    array([nan, nan, nan,  3.,  4.,  5.,  6.,  7., nan, nan])
"""

import sys
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import alpha as al  # noqa: E402

ATOL = 1e-6
RTOL = 1e-6
N = 120
NAN_POS = np.array([3, 17, 29, 44, 52, 77, 101])


# ---------------------------------------------------------------------------
# numpy / pandas 对比实现
# ---------------------------------------------------------------------------


def _lwma_ref(x, periods):
  """无 nan 数据上的 LWMA 参考实现 (pandas rolling + numpy 权重)."""
  weights = np.arange(1, periods + 1, dtype=np.float64)
  return (
    pd.Series(np.asarray(x, dtype=np.float64))
    .rolling(periods, min_periods=periods)
    .apply(lambda w: float(np.dot(w, weights) / weights.sum()), raw=True)
    .to_numpy(dtype=np.float64)
    .copy()
  )


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
  """按 chunk 全量计算参考值, 再只保留 alpha 应计算的 [start, end) 区间.

  ``FLAG_SKIP_NAN`` 时把 chunk 中 [start, end) 段的有效值压紧, 用不含 nan 的
  数据跑参考实现, 再把结果映射回有效位置; 组合 ``FLAG_STRICTLY_CYCLE`` 时
  改为直接在该段上做 rolling (任一 nan 都使窗口无效).
  """
  parts = []
  for chunk in np.split(np.asarray(x, dtype=np.float64), groups):
    n = len(chunk)
    s, e = _calc_range(n, start, end)
    ref = np.full(n, np.nan)

    if flags & al.FLAG_SKIP_NAN:
      seg = chunk[s:e]
      if flags & al.FLAG_STRICTLY_CYCLE:
        ref[s:e] = _lwma_ref(seg, periods)
      else:
        valid = ~np.isnan(seg)
        compacted = np.full(len(seg), np.nan)
        compacted[valid] = _lwma_ref(seg[valid], periods)
        ref[s:e] = compacted
    else:
      ref = _lwma_ref(chunk, periods)
      ref[:s] = np.nan
      ref[e:] = np.nan

    parts.append(ref)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(x, periods, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(al.LWMA(x, periods), dtype=np.float64)


def _clean_data(n=N):
  rng = np.random.default_rng(20260606)
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


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("periods", [2, 3, 4, 5, 10, 20])
def test_lwma_matches_pandas_rolling(periods):
  """flags=0, groups=1: 与 pandas rolling 参考实现逐点比较."""
  x = _clean_data()
  actual = _calc(x, periods, flags=0)
  _assert_close(actual, _expected(x, periods, flags=0))
  assert np.isnan(actual[: periods - 1]).all()
  assert np.isfinite(actual[periods - 1 :]).all()


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
@pytest.mark.parametrize("with_nan", [False, True])
def test_lwma_cycle_flags_without_skip(flags, with_nan):
  """无 SKIP_NAN 时, 窗口内任一 nan 都会使结果变成 nan."""
  x = _nan_data() if with_nan else _clean_data()
  periods = 5
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))

  if with_nan:
    for i in range(periods - 1, len(x)):
      if np.isnan(x[i - periods + 1 : i + 1]).any():
        assert np.isnan(actual[i])


def test_lwma_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 结果等于「未插入 nan」的数据上的参考结果."""
  x = _nan_data()
  periods = 4
  actual = _calc(x, periods, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, periods, flags=al.FLAG_SKIP_NAN))

  valid = ~np.isnan(x)
  assert np.isnan(actual[~valid]).all()
  # 压紧有效值即等价于从未插入过 nan 的数据, 直接跑参考实现.
  _assert_close(actual[valid], _lwma_ref(x[valid], periods))


def test_lwma_skip_nan_and_strictly_cycle():
  """两个 flag 组合: 要求连续无 nan 的完整窗口, 等价于 pandas 原始 rolling."""
  x = _nan_data()
  periods = 4
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  _assert_close(actual, _lwma_ref(x, periods))

  for i in range(periods - 1, len(x)):
    if np.isnan(x[i - periods + 1 : i + 1]).any():
      assert np.isnan(actual[i])


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
def test_lwma_groups(groups):
  """group-major 分块: 每个 chunk 独立计算, 开头的窗口重新计数."""
  x = _clean_data()
  periods = 4
  actual = _calc(x, periods, groups=groups, flags=0)
  _assert_close(actual, _expected(x, periods, groups=groups, flags=0))

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + periods - 1]).all()


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_lwma_skip_nan_within_each_group(groups):
  """SKIP_NAN 的压紧过程按 chunk 独立进行, 不会跨 group 借用数据."""
  x = _nan_data()
  periods = 5
  flags = al.FLAG_SKIP_NAN
  actual = _calc(x, periods, groups=groups, flags=flags)
  _assert_close(
    actual, _expected(x, periods, groups=groups, flags=flags)
  )


@pytest.mark.parametrize(
  "start,end", [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5)]
)
def test_lwma_start_end(start, end):
  """start/end: 仅校验 [start, end) 区间, 区间外必须全是 nan."""
  x = _clean_data()
  periods = 4
  actual = _calc(x, periods, start=start, end=end, flags=0)
  _assert_close(actual, _expected(x, periods, start=start, end=end, flags=0))

  s, e = _calc_range(len(x), start, end)
  assert np.isnan(actual[:s]).all()
  assert np.isnan(actual[e:]).all()
  assert np.isfinite(actual[max(s, periods - 1) : e]).all()


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_lwma_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  x = _clean_data()
  periods = 4
  start, end = 6, 25
  actual = _calc(
    x, periods, start=start, end=end, groups=groups, flags=0
  )
  _assert_close(
    actual,
    _expected(
      x, periods, start=start, end=end, groups=groups, flags=0
    ),
  )

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


def test_lwma_skip_nan_with_start_end():
  """SKIP_NAN + start/end: 只有 [start, end) 内的有效值参与压紧计算."""
  x = _nan_data()
  periods = 4
  ctx = dict(start=10, end=90, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, periods, **ctx)
  _assert_close(actual, _expected(x, periods, **ctx))
  assert np.isnan(actual[:10]).all()
  assert np.isnan(actual[90:]).all()
  assert np.isnan(actual[np.isnan(x)]).all()


def test_lwma_edge_cases():
  """periods=1 直接返回输入; 窗口长于数据时全部为 nan."""
  x = _nan_data()
  _assert_close(_calc(x, 1, flags=0), x)
  assert np.isnan(_calc(x, len(x) + 1, flags=0)).all()
