# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""ENTROPY (Rolling Shannon Entropy, 滚动香农熵) 集成测试.

算法简介
--------
ENTROPY 在最近 `periods` 个值构成的窗口中, 先把有效值按 [min, max] 等宽
分成 `bins` 个桶, 再计算桶分布的香农熵 (自然对数)::

    bin(v)  = min(bins - 1, floor((v - min) / ((max - min) / bins)))
    p_b     = count_b / n
    H[i]    = -sum(p_b * ln(p_b))

窗口内有效值少于 2 个、或所有值相等时熵为 0. ``bins=0`` 按 10 处理. 窗口
未满时 (``flags=0``) 用已有的值计算; ``FLAG_STRICTLY_CYCLE`` 要求窗口完整
才输出. 窗口内 nan 不参与分桶, 但当前值本身为 nan 时输出 nan.

参考实现 ``_entropy_ref`` 直接按上述定义用 numpy 分桶计数, 不实现 SKIP_NAN
的压紧 / Group / start-end 机制: Group 通过按 chunk 分块计算, start/end 只
校验 [start, end) 区间 (窗口仍可引用 start 之前的数据).

理想语义下 ``FLAG_SKIP_NAN`` 把有效值压紧后再计算, nan 位置输出 nan.
Rust 实现通过 ``SkipNanWindow`` 支持了该 flag, 与参考实现一致.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=120)
    >>> x = np.arange(1., 11.)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 4 周期 / 4 桶: 单值窗口熵为 0, 两值窗口 ln(2)=0.693, 三值窗口
   ln(3)=1.099, 四值且各占一桶时 ln(4)=1.386::

    >>> np.round(al.ENTROPY(x, 4, 4), 3)
    array([0.   , 0.693, 1.099, 1.386, 1.386, 1.386, 1.386, 1.386, 1.386, 1.386])

2. ``FLAG_STRICTLY_CYCLE``: 前 `periods-1` 个位置输出 nan::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.ENTROPY(x, 4, 4), 3)
    array([  nan,   nan,   nan, 1.386, 1.386, 1.386, 1.386, 1.386, 1.386, 1.386])

3. 桶数决定离散粒度: 2 桶时 [1, 2, 3, 4] 落在两个桶里, 熵为 ln(2)::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> float(np.round(al.ENTROPY(x, 4, 2)[3], 3))
    0.693

4. 常数列所有窗口熵都是 0::

    >>> np.round(al.ENTROPY(np.full(5, 7.0), 3, 10), 3)
    array([0., 0., 0., 0., 0.])

5. 窗口内 nan 被排除, nan 位置本身输出 nan::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> np.round(al.ENTROPY(xn, 4, 4), 3)
    array([0.   , 0.693, 1.099,   nan, 1.099, 1.099, 1.099, 1.386, 1.386, 1.386])

6. ``groups`` 按 group-major 切块独立计算, 每块重新 warm-up::

    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.ENTROPY(x, 4, 4), 3)
    array([0.   , 0.693, 1.099, 1.386, 1.386, 0.   , 0.693, 1.099, 1.386, 1.386])

7. ``start`` / ``end`` 只计算 [start, end) 区间, 区间外全部为 nan::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.ENTROPY(x, 4, 4), 3)
    array([  nan,   nan,   nan, 1.386, 1.386, 1.386, 1.386, 1.386,   nan,   nan])
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
# numpy 对比实现
# ---------------------------------------------------------------------------


def _entropy_ref(x, periods, bins, *, strict):
  """按算法定义逐窗口做等宽分桶并求香农熵.

  窗口 ``x[max(0, i-periods+1) : i+1]`` 内的 nan 被排除; 当前值为 nan 时
  输出 nan. ``strict`` 要求窗口完整 (i >= periods-1).
  """
  x = np.asarray(x, dtype=np.float64)
  out = np.full(len(x), np.nan)
  bins = 10 if bins == 0 else bins
  if periods == 0:
    return out

  for i in range(len(x)):
    if np.isnan(x[i]):
      continue
    if strict and i < periods - 1:
      continue
    win = x[max(0, i - periods + 1) : i + 1]
    valid = win[~np.isnan(win)]
    n = valid.size
    if n < 2:
      out[i] = 0.0
      continue
    lo, hi = valid.min(), valid.max()
    if hi == lo:
      out[i] = 0.0
      continue
    idx = np.minimum(
      ((valid - lo) / ((hi - lo) / bins)).astype(np.int64), bins - 1
    )
    counts = np.bincount(idx, minlength=bins)
    p = counts[counts > 0] / n
    out[i] = -np.sum(p * np.log(p))
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


def _compact_ref(x, periods, bins, strict):
  """把有效值压紧后跑参考实现, 再映射回有效位置; nan 位置保持 nan."""
  out = np.full(len(x), np.nan)
  valid = ~np.isnan(x)
  if valid.any():
    out[valid] = _entropy_ref(x[valid], periods, bins, strict=strict)
  return out


def _expected(x, periods, bins, *, start=0, end=0, groups=1, flags=0):
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
        seg_ref = _compact_ref(seg, periods, bins, strict)
        if strict:
          seg_ref[~_full_valid_window_mask(seg, periods)] = np.nan
        ref[s:e] = seg_ref
      else:
        ref = _entropy_ref(chunk, periods, bins, strict=strict)
        ref[:s] = np.nan
        ref[e:] = np.nan
    parts.append(ref)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(x, periods, bins, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.ENTROPY(np.asarray(x, dtype=np.float64), periods, bins),
    dtype=np.float64,
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
def test_entropy_matches_numpy(periods):
  """flags=0, groups=1: 每个窗口的分桶熵都与 numpy 参考实现一致."""
  x = _clean_data()
  bins = 5
  actual = _calc(x, periods, bins, flags=0)
  _assert_close(actual, _expected(x, periods, bins, flags=0))
  _assert_close(actual, _entropy_ref(x, periods, bins, strict=False))
  assert np.isfinite(actual).all()
  assert np.all(actual >= 0.0)
  assert np.all(actual <= np.log(bins) + 1e-9)


@pytest.mark.parametrize("bins", [2, 3, 4, 5, 10, 20])
def test_entropy_bins_control_resolution(bins):
  """不同桶数得到不同的离散分布, 且 bins=0 等价于默认 10 桶."""
  x = _clean_data()
  periods = 8
  actual = _calc(x, periods, bins, flags=0)
  _assert_close(actual, _expected(x, periods, bins, flags=0))
  _assert_close(actual, _entropy_ref(x, periods, bins, strict=False))
  if bins == 10:
    _assert_close(actual, _calc(x, periods, 0, flags=0))


def test_entropy_constant_and_single_value_windows():
  """常数列 / 单值窗口的熵恒为 0."""
  x = np.full(20, 7.0)
  for periods in (1, 3, 8):
    _assert_close(_calc(x, periods, 10, flags=0), np.zeros_like(x))
  _assert_close(
    _calc(np.full(5, 7.0), 3, 10, flags=al.FLAG_STRICTLY_CYCLE),
    _expected(np.full(5, 7.0), 3, 10, flags=al.FLAG_STRICTLY_CYCLE),
  )


def test_entropy_uniform_window_is_log_bins():
  """值均匀落在 bins 个不同桶时熵等于 ln(bins)."""
  x = np.arange(1.0, 5.0)
  actual = _calc(x, 4, 4, flags=0)
  _assert_close(actual[-1], np.log(4.0))
  short = _calc(x, 4, 2, flags=0)
  _assert_close(short[-1], np.log(2.0))


@pytest.mark.parametrize("periods", [2, 3, 4, 5, 10, 20])
def test_entropy_strictly_cycle(periods):
  """FLAG_STRICTLY_CYCLE: 前 periods-1 个位置输出 nan, 之后照常计算."""
  x = _clean_data()
  bins = 4
  flags = al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, bins, flags=flags)
  _assert_close(actual, _expected(x, periods, bins, flags=flags))
  _assert_close(actual, _entropy_ref(x, periods, bins, strict=True))
  assert np.isnan(actual[: periods - 1]).all()
  assert np.isfinite(actual[periods - 1 :]).all()


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_entropy_nan_without_skip(flags):
  """无 SKIP_NAN 时窗口内 nan 被排除, nan 位置输出 nan."""
  x = _nan_data()
  periods, bins = 6, 5
  actual = _calc(x, periods, bins, flags=flags)
  _assert_close(actual, _expected(x, periods, bins, flags=flags))
  assert np.isnan(actual[np.isnan(x)]).all()
  valid = ~np.isnan(x)
  if flags & al.FLAG_STRICTLY_CYCLE:
    valid[: periods - 1] = False
    assert np.isnan(actual[: periods - 1]).all()
  assert np.isfinite(actual[valid]).all()


def test_entropy_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 有效位置应等于「未插入 nan」数据压紧后的参考结果."""
  x_clean = _clean_data()
  x = x_clean.copy()
  x[NAN_POS] = np.nan
  periods, bins = 5, 4
  actual = _calc(x, periods, bins, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, periods, bins, flags=al.FLAG_SKIP_NAN))

  valid = ~np.isnan(x)
  assert np.isnan(actual[~valid]).all()
  _assert_close(
    actual[valid],
    _entropy_ref(x_clean[valid], periods, bins, strict=False),
  )


def test_entropy_skip_nan_and_strictly_cycle():
  """两个 flag 组合: 要求连续无 nan 的完整窗口后才输出."""
  x = _nan_data()
  periods, bins = 4, 4
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, bins, flags=flags)
  _assert_close(actual, _expected(x, periods, bins, flags=flags))
  assert np.isnan(actual[: periods - 1]).all()
  assert np.isnan(actual[np.isnan(x)]).all()


@pytest.mark.parametrize(
  "flags",
  [
    0,
    al.FLAG_STRICTLY_CYCLE,
    al.FLAG_SKIP_NAN,
    al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
  ],
)
def test_entropy_all_flag_combinations_with_nan(flags):
  """四种 flag 组合在含 nan 数据上都与参考实现一致."""
  x = _nan_data()
  periods, bins = 6, 5
  actual = _calc(x, periods, bins, flags=flags)
  _assert_close(actual, _expected(x, periods, bins, flags=flags))


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_entropy_groups(groups, flags):
  """group-major 分块: 每个 chunk 独立计算, 开头重新 warm-up."""
  x = _clean_data()
  periods, bins = 5, 4
  actual = _calc(x, periods, bins, groups=groups, flags=flags)
  _assert_close(
    actual, _expected(x, periods, bins, groups=groups, flags=flags)
  )

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    _assert_close(
      actual[lo : lo + chunk_size],
      _entropy_ref(
        x[lo : lo + chunk_size],
        periods,
        bins,
        strict=bool(flags & al.FLAG_STRICTLY_CYCLE),
      ),
    )


@pytest.mark.parametrize(
  "start,end",
  [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5), (-10, 0), (40, 20)],
)
def test_entropy_start_end(start, end):
  """start/end: 仅校验 [start, end) 区间, 区间外必须全是 nan."""
  x = _clean_data()
  periods, bins = 5, 4
  actual = _calc(x, periods, bins, start=start, end=end, flags=0)
  _assert_close(
    actual, _expected(x, periods, bins, start=start, end=end, flags=0)
  )

  s, e = _calc_range(len(x), start, end)
  if s >= e:
    assert np.isnan(actual).all()
  else:
    assert np.isnan(actual[:s]).all()
    assert np.isnan(actual[e:]).all()
    assert np.isfinite(actual[s:e]).all()


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_entropy_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  x = _clean_data()
  periods, bins = 5, 4
  start, end = 6, 25
  actual = _calc(
    x, periods, bins, start=start, end=end, groups=groups, flags=0
  )
  _assert_close(
    actual,
    _expected(
      x, periods, bins, start=start, end=end, groups=groups, flags=0
    ),
  )

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


def test_entropy_skip_nan_with_start_end():
  """SKIP_NAN + start/end: 只有 [start, end) 内的有效值参与压紧计算."""
  x_clean = _clean_data()
  x = x_clean.copy()
  x[NAN_POS] = np.nan
  periods, bins = 5, 4
  ctx = dict(start=10, end=90, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, periods, bins, **ctx)
  _assert_close(actual, _expected(x, periods, bins, **ctx))
  assert np.isnan(actual[:10]).all()
  assert np.isnan(actual[90:]).all()
  assert np.isnan(actual[np.isnan(x)]).all()

  seg = x[10:90]
  valid = ~np.isnan(seg)
  _assert_close(
    actual[10:90][valid],
    _entropy_ref(x_clean[10:90][valid], periods, bins, strict=False),
  )


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_entropy_skip_nan_start_end_with_groups(groups):
  """SKIP_NAN + 负数 end + groups: 每个 chunk 内独立解析区间并压紧."""
  x = _nan_data()
  periods, bins = 4, 4
  ctx = dict(start=5, end=-5, groups=groups, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, periods, bins, **ctx)
  _assert_close(actual, _expected(x, periods, bins, **ctx))


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_entropy_skip_nan_strict_start_end_with_groups(groups):
  """SKIP_NAN | STRICTLY_CYCLE 与 start/end/groups 组合."""
  x = _nan_data()
  periods, bins = 4, 5
  ctx = dict(
    start=8,
    end=-8,
    groups=groups,
    flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
  )
  actual = _calc(x, periods, bins, **ctx)
  _assert_close(actual, _expected(x, periods, bins, **ctx))


def test_entropy_edge_cases():
  """periods=1 时每个窗口只有一个值, 熵恒为 0 (nan 位置仍为 nan)."""
  x = _clean_data()
  _assert_close(_calc(x, 1, 10, flags=0), np.zeros_like(x))

  xn = _nan_data()
  actual = _calc(xn, 1, 10, flags=0)
  assert np.isnan(actual[np.isnan(xn)]).all()
  _assert_close(actual[~np.isnan(xn)], np.zeros(np.count_nonzero(~np.isnan(xn))))

  periods = len(x) + 1
  _assert_close(
    _calc(x, periods, 4, flags=0), _expected(x, periods, 4, flags=0)
  )
  assert np.isnan(_calc(x, periods, 4, flags=al.FLAG_STRICTLY_CYCLE)).all()
