# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""QUANTILE (Rolling Quantile, 滚动分位数) 集成测试.

算法简介
--------
QUANTILE 对最近 `periods` 个值求 q 分位数 (0 <= q <= 1), 采用线性插值::

    QUANTILE[i] = np.quantile(x[i - periods + 1 : i + 1], q)

排序后位置 ``pos = q * (n - 1)``, 取 ``floor(pos)`` 与 ``ceil(pos)`` 两个
元素按小数部分插值, 与 numpy ``np.quantile(..., method="linear")`` 以及
pandas ``rolling(periods).quantile(q)`` 一致. q=0 / q=1 分别退化为窗口最小
值 / 最大值. 窗口未满时 (``flags=0``) 用已有的 1..periods 个值计算部分分位
数; ``FLAG_STRICTLY_CYCLE`` 要求窗口完整才输出, 与 pandas ``rolling`` 默认
行为一致. 文档中说明窗口内的 nan 会被排除, 但当前值本身为 nan 时输出 nan.

参考实现 ``_quantile_ref`` 直接按算法定义逐窗口调用 ``np.quantile``, 不实现
SKIP_NAN 的压紧 / Group / start-end 机制: Group 通过按 chunk 分块计算, start
/ end 只校验 [start, end) 区间 (窗口本身仍可引用 start 之前的数据).

理想语义下 ``FLAG_SKIP_NAN`` 应让 nan 完全不参与运算: 有效位置上的结果等于
把有效值压紧后跑参考实现. 当前 Rust 实现没有区分该 flag (窗口始终按原始
位置取, 只是排除 nan), 因此 :func:`test_quantile_skip_nan_ignores_nan_values`
等用例会失败并被保留, 用于暴露实现差异, 不应为了让测试通过而修改基准.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=120)
    >>> x = np.arange(1., 11.)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 中位数 (窗口未满给出部分分位数)::

    >>> np.round(al.QUANTILE(x, 3, 0.5), 3)
    array([1. , 1.5, 2. , 3. , 4. , 5. , 6. , 7. , 8. , 9. ])

   i=1 的窗口 [1, 2] 中位数为 1.5; i=3 的窗口 [2, 3, 4] 中位数为 3.

2. ``FLAG_STRICTLY_CYCLE``: 前 `periods-1` 个位置输出 nan::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.QUANTILE(x, 3, 0.5), 3)
    array([nan, nan,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])

3. q=0 / q=1 等价于滚动最小 / 最大值::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.QUANTILE(x, 3, 0.0), 3)
    array([nan, nan,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])
    >>> np.round(al.QUANTILE(x, 3, 1.0), 3)
    array([nan, nan,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])

4. 窗口内 nan 被排除, nan 位置本身输出 nan::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.QUANTILE(xn, 3, 0.5), 3)
    array([1. , 1.5, 2. , nan, 4. , 5.5, 6. , 7. , 8. , 9. ])

   i=4 的原始窗口是 [3, nan, 5], 排除 nan 后只剩 [3, 5], 中位数 4;
   i=5 的窗口 [nan, 5, 6] 剩 [5, 6], 中位数 5.5.

5. ``groups`` 把一维数据按 group-major 切块独立计算, 每块重新 warm-up::

    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.QUANTILE(x, 3, 0.5), 3)
    array([1. , 1.5, 2. , 3. , 4. , 6. , 6.5, 7. , 8. , 9. ])

6. ``start`` / ``end`` 只计算 [start, end) 区间, 区间外全部为 nan,
   但窗口仍可引用 start 之前的数据::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.QUANTILE(x, 3, 0.5), 3)
    array([nan, nan, nan,  3.,  4.,  5.,  6.,  7., nan, nan])
"""

import numpy as np
import pandas as pd
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
NAN_POS = np.array([0, 1, 3, 17, 18, 29, 44, 52, 77, 101])


# ---------------------------------------------------------------------------
# numpy / pandas 对比实现
# ---------------------------------------------------------------------------


def _quantile_ref(x, periods, q, *, strict):
  """按算法定义逐窗口求线性插值分位数.

  ``x`` 不含 nan: 窗口是 ``x[max(0, i-periods+1) : i+1]``; 窗口内出现 nan
  时按文档排除 nan, 但当前值为 nan 时输出 nan. ``strict`` 要求窗口完整且
  全部有效.
  """
  x = np.asarray(x, dtype=np.float64)
  out = np.full(len(x), np.nan)
  if periods == 0 or not (0.0 <= q <= 1.0):
    return out

  for i in range(len(x)):
    if np.isnan(x[i]):
      continue
    win = x[max(0, i - periods + 1) : i + 1]
    valid = win[~np.isnan(win)]
    if valid.size == 0:
      continue
    if strict and (win.size < periods or valid.size < periods):
      continue
    out[i] = np.quantile(valid, q)
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


def _full_valid_window_mask(x, periods):
  """最近 periods 个原始位置是否全部有效 (SKIP_NAN + STRICTLY_CYCLE)."""
  valid_count = (
    pd.Series((~np.isnan(x)).astype(np.float64))
    .rolling(periods, min_periods=periods)
    .sum()
    .to_numpy()
  )
  return valid_count == periods


def _compact_ref(x, periods, q, strict):
  """把有效值压紧后跑参考实现, 再映射回有效位置; nan 位置保持 nan."""
  out = np.full(len(x), np.nan)
  valid = ~np.isnan(x)
  if valid.any():
    out[valid] = _quantile_ref(x[valid], periods, q, strict=strict)
  return out


def _expected(x, periods, q, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 全量计算参考值, 再只保留 alpha 应计算的 [start, end) 区间.

  无 SKIP_NAN 时窗口引用原始位置 (可跨 start 借用数据), 窗口内 nan 被排除;
  有 SKIP_NAN 时按理想语义只在 [start, end) 内压紧有效值后计算.
  """
  x = np.asarray(x, dtype=np.float64)
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  skip = bool(flags & al.FLAG_SKIP_NAN)
  parts = []
  for chunk in np.split(x, groups):
    n = len(chunk)
    s, e = _calc_range(n, start, end)
    ref = np.full(n, np.nan)
    if s < e:
      if skip:
        seg = chunk[s:e]
        seg_ref = _compact_ref(seg, periods, q, strict)
        if strict:
          seg_ref[~_full_valid_window_mask(seg, periods)] = np.nan
        ref[s:e] = seg_ref
      else:
        ref = _quantile_ref(chunk, periods, q, strict=strict)
        ref[:s] = np.nan
        ref[e:] = np.nan
    parts.append(ref)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(x, periods, q, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.QUANTILE(np.asarray(x, dtype=np.float64), periods, q), dtype=np.float64
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261106)
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
def test_quantile_matches_numpy(periods):
  """flags=0, groups=1: 每个窗口都与 np.quantile 线性插值结果一致."""
  x = _clean_data()
  q = 0.5
  actual = _calc(x, periods, q, flags=0)
  _assert_close(actual, _expected(x, periods, q, flags=0))
  _assert_close(actual, _quantile_ref(x, periods, q, strict=False))
  assert np.isfinite(actual).all()


@pytest.mark.parametrize("q", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_quantile_q_values(q):
  """不同分位点 (含 0 / 1 端点) 都与 numpy 一致."""
  x = _clean_data()
  periods = 7
  actual = _calc(x, periods, q, flags=0)
  _assert_close(actual, _expected(x, periods, q, flags=0))
  if q == 0.0:
    _assert_close(
      actual, pd.Series(x).rolling(periods, min_periods=1).min().to_numpy()
    )
  if q == 1.0:
    _assert_close(
      actual, pd.Series(x).rolling(periods, min_periods=1).max().to_numpy()
    )


@pytest.mark.parametrize("periods", [2, 3, 4, 5, 10, 20])
def test_quantile_strictly_cycle_matches_pandas(periods):
  """FLAG_STRICTLY_CYCLE: 窗口未满输出 nan, 之后与 pandas rolling 一致."""
  x = _clean_data()
  q = 0.3
  flags = al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, q, flags=flags)
  _assert_close(actual, _expected(x, periods, q, flags=flags))
  _assert_close(actual, _quantile_ref(x, periods, q, strict=True))
  _assert_close(
    actual,
    pd.Series(x)
    .rolling(periods, min_periods=periods)
    .quantile(q)
    .to_numpy(dtype=np.float64),
  )
  assert np.isnan(actual[: periods - 1]).all()
  assert np.isfinite(actual[periods - 1 :]).all()


def test_quantile_nan_position_is_nan():
  """当前值为 nan 时输出 nan, 其余位置仍按窗口有效值计算."""
  x = _nan_data()
  periods, q = 4, 0.5
  actual = _calc(x, periods, q, flags=0)
  assert np.isnan(actual[np.isnan(x)]).all()
  assert np.isfinite(actual[~np.isnan(x)]).all()
  _assert_close(actual, _expected(x, periods, q, flags=0))


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_quantile_nan_without_skip(flags):
  """无 SKIP_NAN 时窗口内 nan 被排除, 严格模式还要求窗口全有效."""
  x = _nan_data()
  periods, q = 5, 0.25
  actual = _calc(x, periods, q, flags=flags)
  _assert_close(actual, _expected(x, periods, q, flags=flags))
  if flags & al.FLAG_STRICTLY_CYCLE:
    assert np.isnan(actual[: periods - 1]).all()
    for i in range(len(x)):
      if np.isnan(x[max(0, i - periods + 1) : i + 1]).any():
        assert np.isnan(actual[i])


def test_quantile_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 有效位置应等于「未插入 nan」数据压紧后的参考结果."""
  x_clean = _clean_data()
  x = x_clean.copy()
  x[NAN_POS] = np.nan
  periods, q = 4, 0.5
  actual = _calc(x, periods, q, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, periods, q, flags=al.FLAG_SKIP_NAN))

  valid = ~np.isnan(x)
  assert np.isnan(actual[~valid]).all()
  _assert_close(
    actual[valid], _quantile_ref(x_clean[valid], periods, q, strict=False)
  )


def test_quantile_skip_nan_and_strictly_cycle():
  """两个 flag 组合: 要求连续无 nan 的完整窗口后才输出."""
  x = _nan_data()
  periods, q = 4, 0.75
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, q, flags=flags)
  _assert_close(actual, _expected(x, periods, q, flags=flags))
  assert np.isnan(actual[: periods - 1]).all()
  assert np.isnan(actual[np.isnan(x)]).all()


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_quantile_groups(groups, flags):
  """group-major 分块: 每个 chunk 独立计算, 开头重新 warm-up."""
  x = _clean_data()
  periods, q = 4, 0.5
  actual = _calc(x, periods, q, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, periods, q, groups=groups, flags=flags))

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    if flags & al.FLAG_STRICTLY_CYCLE:
      assert np.isnan(actual[lo : lo + periods - 1]).all()
    else:
      _assert_close(
        actual[lo : lo + periods - 1],
        _quantile_ref(
          x[lo : lo + chunk_size], periods, q, strict=False
        )[: periods - 1],
      )


@pytest.mark.parametrize(
  "start,end",
  [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5), (-10, 0), (40, 20)],
)
def test_quantile_start_end(start, end):
  """start/end: 仅校验 [start, end) 区间, 区间外必须全是 nan."""
  x = _clean_data()
  periods, q = 4, 0.5
  actual = _calc(x, periods, q, start=start, end=end, flags=0)
  _assert_close(
    actual, _expected(x, periods, q, start=start, end=end, flags=0)
  )

  s, e = _calc_range(len(x), start, end)
  if s >= e:
    assert np.isnan(actual).all()
  else:
    assert np.isnan(actual[:s]).all()
    assert np.isnan(actual[e:]).all()
    assert np.isfinite(actual[s:e]).all()


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_quantile_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  x = _clean_data()
  periods, q = 4, 0.25
  start, end = 6, 25
  actual = _calc(
    x, periods, q, start=start, end=end, groups=groups, flags=0
  )
  _assert_close(
    actual,
    _expected(
      x, periods, q, start=start, end=end, groups=groups, flags=0
    ),
  )

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


def test_quantile_skip_nan_with_start_end():
  """SKIP_NAN + start/end: 只有 [start, end) 内的有效值参与压紧计算."""
  x_clean = _clean_data()
  x = x_clean.copy()
  x[NAN_POS] = np.nan
  periods, q = 4, 0.5
  ctx = dict(start=10, end=90, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, periods, q, **ctx)
  _assert_close(actual, _expected(x, periods, q, **ctx))
  assert np.isnan(actual[:10]).all()
  assert np.isnan(actual[90:]).all()
  assert np.isnan(actual[np.isnan(x)]).all()

  seg = x[10:90]
  valid = ~np.isnan(seg)
  _assert_close(
    actual[10:90][valid],
    _quantile_ref(x_clean[10:90][valid], periods, q, strict=False),
  )


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_quantile_skip_nan_start_end_with_groups(groups):
  """SKIP_NAN + 负数 end + groups: 每个 chunk 内独立解析区间并压紧."""
  x = _nan_data()
  periods, q = 3, 0.5
  ctx = dict(start=5, end=-5, groups=groups, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, periods, q, **ctx)
  _assert_close(actual, _expected(x, periods, q, **ctx))


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_quantile_skip_nan_strict_start_end_with_groups(groups):
  """SKIP_NAN | STRICTLY_CYCLE 与 start/end/groups 组合."""
  x = _nan_data()
  periods, q = 4, 0.5
  ctx = dict(
    start=8,
    end=-8,
    groups=groups,
    flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
  )
  actual = _calc(x, periods, q, **ctx)
  _assert_close(actual, _expected(x, periods, q, **ctx))


def test_quantile_edge_cases():
  """periods=1 复制输入; periods=0 / q 越界全 nan; 超长窗口按已有值计算."""
  x = _nan_data()
  _assert_close(_calc(x, 1, 0.5, flags=0), x)
  assert np.isnan(_calc(x, 0, 0.5, flags=0)).all()
  assert np.isnan(_calc(x, 4, -0.1, flags=0)).all()
  assert np.isnan(_calc(x, 4, 1.5, flags=0)).all()
  assert np.isnan(_calc(x, 4, np.nan, flags=0)).all()

  periods = len(x) + 1
  q = 0.5
  _assert_close(_calc(x, periods, q, flags=0), _expected(x, periods, q, flags=0))
  assert np.isnan(_calc(x, periods, q, flags=al.FLAG_STRICTLY_CYCLE)).all()
