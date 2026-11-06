# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""BETA / ALPHA (滚动回归 Beta 与 Jensen's Alpha) 集成测试.

算法简介
--------
两个算子都在最近 ``periods`` 个位置上做最小二乘回归, 以 ``benchmark`` 为
自变量 ``y``, ``input`` 为因变量 ``x``::

    n     = min(i + 1, periods)
    denom = n * sum(y^2) - (sum(y))^2
    BETA  = (n * sum(x*y) - sum(x) * sum(y)) / denom
    ALPHA = mean(x) - BETA * mean(y)

即 ALPHA 是回归截距 (不含无风险利率项). 两个算子的窗口、nan 与 flag
语义完全相同, 只有输出量不同. 窗口未填满时用已有的部分窗口 (至少 2 个点);
``FLAG_STRICTLY_CYCLE`` 要求窗口填满 ``periods`` 个值, 前 ``periods - 1``
个位置输出 nan. benchmark 在窗口内为常数时 ``denom = 0``, 输出 nan.

``flags=0`` 时窗口内任一位置只要 x 或 y 为 nan, 该位置输出 nan; 窗口可以
引用 ``start`` 之前的数据. ``FLAG_SKIP_NAN`` 改为从当前位置往回收集最近
``periods`` 个「x、y 同时有效」的配对, nan 完全不参与运算, 等价于把有效
配对压紧 (compact) 后跑参考实现再映射回有效位置; nan 位置本身输出 nan.
组合 ``FLAG_STRICTLY_CYCLE`` 时还要求最近 ``periods`` 个**原始**位置全部
有效 (窗口内不能有 nan 间隙), 否则输出 nan.

``groups`` 按 group-major 切成等长 chunk 独立计算, 每块重新 warm-up;
start/end 是每个 chunk 内的相对下标, 只计算 [start, end) 区间, 区间外为
nan. 对比基准用 numpy 独立实现 OLS 公式, 不调用 alpha, 也不实现 Group /
start-end 机制; 这些上下文语义由 ``_expected`` 单独描述.

`set_ctx` 只覆盖传入的参数, 因此每次调用前都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> x = np.arange(1., 11.)
    >>> y = 2 * x
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 完全线性关系 ``x = 0.5 * y``: BETA 恒为 0.5, ALPHA 恒为 0::

    >>> np.round(al.BETA(x, y, 3), 4).tolist()
    [nan, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    >>> np.round(al.ALPHA(x, y, 3), 4).tolist()
    [nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

2. ``FLAG_STRICTLY_CYCLE``: 前 2 个位置输出 nan::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.BETA(x, y, 3), 4).tolist()
    [nan, nan, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

3. 不跳过 nan 时, 含 nan 的窗口输出 nan::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.BETA(xn, y, 3), 4).tolist()
    [nan, 0.5, 0.5, nan, nan, nan, 0.5, 0.5, 0.5, 0.5]

4. ``FLAG_SKIP_NAN``: nan 不参与运算, i=4 起恢复 0.5::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.BETA(xn, y, 3), 4).tolist()
    [nan, 0.5, 0.5, nan, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

5. ``groups`` 把数据按 group-major 切块, 每块重新 warm-up::

    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.BETA(x, y, 3), 4).tolist()
    [nan, 0.5, 0.5, 0.5, 0.5, nan, 0.5, 0.5, 0.5, 0.5]

6. ``start`` / ``end`` 只输出 [start, end) 区间::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.BETA(x, y, 3), 4).tolist()
    [nan, nan, nan, 0.5, 0.5, 0.5, 0.5, 0.5, nan, nan]
"""

import numpy as np
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
PERIODS = [2, 3, 4, 5, 10, 20]
IS_ALPHA = [False, True]
NAN_POS_X = np.array([3, 17, 18, 29, 44, 52, 77, 101])
NAN_POS_Y = np.array([5, 18, 31, 45, 60, 78, 99])
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


def _ols_stats(wx, wy):
  """窗口内最小二乘: 返回 (beta, mean_x, mean_y), denom 为 0 时返回 None."""
  n = len(wx)
  sx = float(np.sum(wx))
  sy = float(np.sum(wy))
  sxy = float(np.dot(wx, wy))
  sy2 = float(np.dot(wy, wy))
  denom = n * sy2 - sy * sy
  if denom == 0 or not np.isfinite(denom):
    return None
  beta = (n * sxy - sx * sy) / denom
  return beta, sx / n, sy / n


def _ols_ref(x, y, periods, *, strict=False, alpha=False):
  """flags=0 的参考实现: 每个位置取最近 periods 个原始值做 OLS."""
  x = np.asarray(x, dtype=np.float64)
  y = np.asarray(y, dtype=np.float64)
  out = np.full(len(x), np.nan)
  for i in range(len(x)):
    lo = max(0, i + 1 - periods)
    wx, wy = x[lo : i + 1], y[lo : i + 1]
    if strict and i < periods - 1:
      continue
    if np.isnan(wx).any() or np.isnan(wy).any():
      continue
    if len(wx) < 2:
      continue
    stats = _ols_stats(wx, wy)
    if stats is None:
      continue
    beta, mx, my = stats
    out[i] = mx - beta * my if alpha else beta
  return out


def _ols_skip_ref(x, y, periods, lo, hi, *, strict=False, alpha=False):
  """SKIP_NAN 参考实现: 往回收集最近 periods 个同时有效的配对做 OLS.

  严格模式额外要求最近 periods 个原始位置全部有效, 不允许 nan 间隙.
  """
  x = np.asarray(x, dtype=np.float64)
  y = np.asarray(y, dtype=np.float64)
  out = np.full(len(x), np.nan)
  for i in range(lo, hi):
    if np.isnan(x[i]) or np.isnan(y[i]):
      continue
    if strict:
      wl = max(0, i + 1 - periods)
      if i < periods - 1:
        continue
      wx, wy = x[wl : i + 1], y[wl : i + 1]
      if np.isnan(wx).any() or np.isnan(wy).any():
        continue
    else:
      xs, ys = [], []
      j = i
      while j >= 0 and len(xs) < periods:
        if not (np.isnan(x[j]) or np.isnan(y[j])):
          xs.append(x[j])
          ys.append(y[j])
        j -= 1
      wx, wy = np.asarray(xs), np.asarray(ys)
    if len(wx) < 2:
      continue
    stats = _ols_stats(wx, wy)
    if stats is None:
      continue
    beta, mx, my = stats
    out[i] = mx - beta * my if alpha else beta
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


def _expected(x, y, periods, *, start=0, end=0, groups=1, flags=0, alpha=False):
  """按 chunk 计算参考值, 只保留 alpha 应输出的 [start, end) 区间."""
  x = np.asarray(x, dtype=np.float64)
  y = np.asarray(y, dtype=np.float64)
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  skip = bool(flags & al.FLAG_SKIP_NAN)
  parts = []
  for chunk_x, chunk_y in zip(np.split(x, groups), np.split(y, groups)):
    n = len(chunk_x)
    s, e = _calc_range(n, start, end)
    if skip:
      ref = _ols_skip_ref(
        chunk_x, chunk_y, periods, s, e, strict=strict, alpha=alpha
      )
    else:
      ref = _ols_ref(chunk_x, chunk_y, periods, strict=strict, alpha=alpha)
      ref[:s] = np.nan
      ref[e:] = np.nan
    parts.append(ref)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(x, y, periods, *, alpha=False, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  fn = al.ALPHA if alpha else al.BETA
  return np.asarray(
    fn(
      np.asarray(x, dtype=np.float64),
      np.asarray(y, dtype=np.float64),
      periods,
    ),
    dtype=np.float64,
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261118)
  x = rng.normal(loc=0.0, scale=1.0, size=n).astype(np.float64)
  y = 0.6 * x + rng.normal(loc=0.0, scale=1.0, size=n).astype(np.float64)
  return x, y


def _nan_data(n=N):
  x, y = _clean_data(n)
  x[NAN_POS_X[NAN_POS_X < n]] = np.nan
  y[NAN_POS_Y[NAN_POS_Y < n]] = np.nan
  return x, y


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("periods", PERIODS)
@pytest.mark.parametrize("alpha", IS_ALPHA)
def test_ols_matches_reference(periods, alpha):
  """flags=0, groups=1: 与 numpy OLS 参考实现逐点比较."""
  x, y = _clean_data()
  actual = _calc(x, y, periods, alpha=alpha, flags=0)
  _assert_close(actual, _ols_ref(x, y, periods, alpha=alpha))
  _assert_close(actual, _expected(x, y, periods, alpha=alpha, flags=0))
  assert np.isnan(actual[0])
  assert np.isfinite(actual[1:]).all()


@pytest.mark.parametrize("periods", PERIODS)
@pytest.mark.parametrize("alpha", IS_ALPHA)
def test_ols_strictly_cycle(periods, alpha):
  """FLAG_STRICTLY_CYCLE: 前 periods-1 个位置输出 nan, 之后与参考一致."""
  x, y = _clean_data()
  flags = al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, y, periods, alpha=alpha, flags=flags)
  _assert_close(actual, _ols_ref(x, y, periods, strict=True, alpha=alpha))
  _assert_close(actual, _expected(x, y, periods, alpha=alpha, flags=flags))
  assert np.isnan(actual[: periods - 1]).all()
  assert np.isfinite(actual[periods - 1 :]).all()


@pytest.mark.parametrize("alpha", IS_ALPHA)
def test_ols_linear_relationship(alpha):
  """完全线性关系 x = 0.5*y: BETA 恒为 0.5, ALPHA 恒为 0."""
  x = np.arange(1.0, 21.0)
  y = 2 * x
  periods = 4
  actual = _calc(x, y, periods, alpha=alpha, flags=0)
  expected = np.full(len(x), 0.0 if alpha else 0.5)
  expected[0] = np.nan
  _assert_close(actual, expected)


@pytest.mark.parametrize("alpha", IS_ALPHA)
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_ols_nan_without_skip(alpha, flags):
  """无 SKIP_NAN: 窗口内任一配对含 nan, 该位置输出 nan."""
  x, y = _nan_data()
  periods = 5
  actual = _calc(x, y, periods, alpha=alpha, flags=flags)
  _assert_close(
    actual, _expected(x, y, periods, alpha=alpha, flags=flags)
  )
  invalid = np.isnan(x) | np.isnan(y)
  assert np.isnan(actual[invalid]).all()
  for i in range(periods - 1, len(x)):
    wx, wy = x[max(0, i + 1 - periods) : i + 1], y[max(0, i + 1 - periods) : i + 1]
    if np.isnan(wx).any() or np.isnan(wy).any():
      assert np.isnan(actual[i])


@pytest.mark.parametrize("alpha", IS_ALPHA)
def test_ols_skip_nan_ignores_nan_values(alpha):
  """FLAG_SKIP_NAN: nan 位置输出 nan, 有效位置按压紧后的配对计算."""
  x, y = _nan_data()
  periods = 5
  actual = _calc(x, y, periods, alpha=alpha, flags=al.FLAG_SKIP_NAN)
  _assert_close(
    actual, _expected(x, y, periods, alpha=alpha, flags=al.FLAG_SKIP_NAN)
  )
  invalid = np.isnan(x) | np.isnan(y)
  assert np.isnan(actual[invalid]).all()
  assert np.isfinite(actual[~invalid][2:]).all()


@pytest.mark.parametrize("alpha", IS_ALPHA)
def test_ols_skip_nan_matches_compacted_clean_data(alpha):
  """SKIP_NAN 等价于压紧有效配对后, 在无 nan 序列上做 OLS."""
  x_clean, y_clean = _clean_data()
  x, y = _nan_data()
  periods = 5
  valid = ~(np.isnan(x) | np.isnan(y))
  actual = _calc(x, y, periods, alpha=alpha, flags=al.FLAG_SKIP_NAN)
  ref = _ols_ref(x_clean[valid], y_clean[valid], periods, alpha=alpha)
  _assert_close(actual[valid], ref)
  assert np.isnan(actual[~valid]).all()


@pytest.mark.parametrize("alpha", IS_ALPHA)
def test_ols_skip_nan_and_strictly_cycle(alpha):
  """两个 flag 组合: 要求最近 periods 个原始位置全部有效."""
  x, y = _nan_data()
  periods = 5
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, y, periods, alpha=alpha, flags=flags)
  _assert_close(actual, _expected(x, y, periods, alpha=alpha, flags=flags))

  invalid = np.isnan(x) | np.isnan(y)
  assert np.isnan(actual[invalid]).all()
  loose = _calc(x, y, periods, alpha=alpha, flags=al.FLAG_SKIP_NAN)
  finite = np.isfinite(actual)
  _assert_close(actual[finite], loose[finite])


@pytest.mark.parametrize("flags", FLAG_COMBOS)
@pytest.mark.parametrize("alpha", IS_ALPHA)
def test_ols_all_flag_combinations_with_nan(flags, alpha):
  """四种 flag 组合在含 nan 数据上都与参考实现比较."""
  x, y = _nan_data()
  periods = 6
  actual = _calc(x, y, periods, alpha=alpha, flags=flags)
  _assert_close(actual, _expected(x, y, periods, alpha=alpha, flags=flags))


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("alpha", IS_ALPHA)
def test_ols_groups(groups, alpha):
  """group-major 分块: 每个 chunk 独立计算, 开头重新 warm-up."""
  x, y = _clean_data()
  periods = 4
  actual = _calc(x, y, periods, alpha=alpha, groups=groups, flags=0)
  _assert_close(
    actual, _expected(x, y, periods, alpha=alpha, groups=groups, flags=0)
  )
  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo])
    _assert_close(
      actual[lo : lo + chunk_size],
      _ols_ref(x[lo : lo + chunk_size], y[lo : lo + chunk_size], periods, alpha=alpha),
    )


@pytest.mark.parametrize("groups", GROUPS)
def test_ols_skip_nan_within_each_group(groups):
  """SKIP_NAN 的有效配对按 chunk 独立收集, 不会跨 group 借用数据."""
  x, y = _nan_data()
  periods = 5
  flags = al.FLAG_SKIP_NAN
  actual = _calc(x, y, periods, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, y, periods, groups=groups, flags=flags))


@pytest.mark.parametrize("start,end", START_END)
@pytest.mark.parametrize("flags", [0, al.FLAG_SKIP_NAN])
def test_ols_start_end(start, end, flags):
  """start/end: 窗口可引用 start 之前的数据, 只校验 [start, end) 区间."""
  x, y = _nan_data()
  periods = 4
  actual = _calc(x, y, periods, start=start, end=end, flags=flags)
  _assert_close(
    actual, _expected(x, y, periods, start=start, end=end, flags=flags)
  )
  s, e = _calc_range(len(x), start, end)
  assert np.isnan(actual[:s]).all()
  assert np.isnan(actual[e:]).all()


@pytest.mark.parametrize("groups", GROUPS)
def test_ols_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  x, y = _nan_data()
  periods = 4
  start, end = 6, 25
  actual = _calc(
    x, y, periods, start=start, end=end, groups=groups, flags=0
  )
  _assert_close(
    actual,
    _expected(x, y, periods, start=start, end=end, groups=groups, flags=0),
  )
  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


@pytest.mark.parametrize("alpha", IS_ALPHA)
def test_ols_edge_cases(alpha):
  """periods < 2 报错 / 常数 benchmark / 空数组 / 长度不匹配报错."""
  x, y = _clean_data(20)

  with pytest.raises(ValueError):
    _calc(x, y, 1, alpha=alpha, flags=0)

  flat = np.full(20, 3.0)
  assert np.isnan(_calc(x, flat, 4, alpha=alpha, flags=0)).all()

  empty = np.empty(0, dtype=np.float64)
  _assert_close(_calc(empty, empty, 3, alpha=alpha, flags=0), empty)

  with pytest.raises(ValueError):
    _calc(x, np.ones(10), 4, alpha=alpha, flags=0)
