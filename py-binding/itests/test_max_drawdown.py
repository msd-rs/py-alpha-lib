# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""MAX_DRAWDOWN (Rolling Maximum Drawdown, 滚动最大回撤) 集成测试.

算法简介
--------
输入是一条价格/净值曲线, 输出是最近 `periods` 个原始位置构成的窗口内的
最大峰谷回撤, 以负收益表示 (-0.2 表示从峰值回撤 20%)::

    w = x[max(0, i-periods+1) : i+1]
    peak = maximum.accumulate(w)
    MDD[i] = min(0, min((w - peak) / peak))

即先更新 ``peak = max(peak, val)`` 再计算 ``(val - peak) / peak``, 取窗口内
最小值并与 0 取较小者. 单调递增序列恒为 0, ``periods=1`` 时窗口只含当前值
也恒为 0, 结果恒 <= 0. 窗口未满 (``flags=0``) 时用已有的 1..periods 个值
给出部分回撤; 设置 ``FLAG_STRICTLY_CYCLE`` 后窗口未满输出 nan.

``periods`` 必须 >= 1, 传 0 会抛出 ``ValueError``. 不设置 ``FLAG_SKIP_NAN``
时, 窗口内任一 nan (含当前位置) 都会让该窗口结果为 nan; 设置后当前位置为
nan 仍输出 nan, 但窗口内的 nan 被忽略: 窗口仍按**原始下标**截取, 只是把其中
的 nan 丢弃, 再用剩下的有效值按原顺序计算回撤 (与 MA 那种"压紧成新序列再
取最近 periods 个有效值"的语义不同). 两个 flag 可以组合.

参考实现 ``_mdd_ref`` / ``_mdd_series_ref`` 只处理无 nan 的正价格数据, 用
numpy ``maximum.accumulate`` 独立复现算法逻辑, 不调用 alpha, 也不实现
SKIP_NAN / Group / start-end 机制; 这些上下文语义由 ``_expected`` 单独描述:
Group 通过按 chunk 分块计算, start/end 只校验 [start, end) 区间, 窗口本身仍
可引用 start 之前的数据. 严格模式的"窗口是否已满"按每个 chunk 的计算起点
``start`` 起算, 即 ``i - start + 1 >= periods`` 才输出, 与本库其它滚动算法
一致.

`set_ctx` 只覆盖传入的参数, 所以每次调用都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=120)
    >>> x = np.array([10., 12., 15., 13., 9., 14., 16.])
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算 (4 周期窗口). i=3 的窗口 [10,12,15,13] 峰值 15, 回撤
   (13-15)/15=-0.133; i=4 的窗口 [12,15,13,9] 回撤 (9-15)/15=-0.4::

    >>> np.round(al.MAX_DRAWDOWN(x, 4), 3)
    array([ 0.   ,  0.   ,  0.   , -0.133, -0.4  , -0.4  , -0.308])

2. ``FLAG_STRICTLY_CYCLE``: 前 3 个位置窗口未满, 输出 nan::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> np.round(al.MAX_DRAWDOWN(x, 4), 3)
    array([   nan,    nan,    nan, -0.133, -0.4  , -0.4  , -0.308])

3. 不跳过 nan: 窗口内任一 nan 污染整个窗口 (把 15 换成 nan)::

    >>> xn = x.copy(); xn[2] = np.nan
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.MAX_DRAWDOWN(xn, 3), 3)
    array([ 0.   ,  0.   ,    nan,    nan,    nan, -0.308,  0.   ])

4. ``FLAG_SKIP_NAN``: nan 位置仍为 nan, 但窗口内的 nan 不参与运算.
   i=3 的窗口 [12, nan, 13] 只剩 [12, 13], 单调上涨故回撤为 0::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.MAX_DRAWDOWN(xn, 3), 3)
    array([ 0.   ,  0.   ,    nan,  0.   , -0.308, -0.308,  0.   ])

5. ``groups`` 把一维数据按 group-major 切块, 每块的窗口不跨边界::

    >>> y = np.array([10., 12., 9., 11., 8., 10., 12., 6., 8., 10., 7., 9.])
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.MAX_DRAWDOWN(y, 3), 3)
    array([ 0.   ,  0.   , -0.25 , -0.25 , -0.273, -0.273,  0.   , -0.5  , -0.5  ,  0.   , -0.3  , -0.3  ])

6. ``start`` / ``end`` 只计算 [start, end) 区间, 区间外全部为 nan,
   但窗口仍可引用 start 之前的数据::

    >>> al.set_ctx(start=2, end=6, groups=1, flags=0)
    >>> np.round(al.MAX_DRAWDOWN(x, 3), 3)
    array([   nan,    nan,  0.   , -0.133, -0.4  , -0.308,    nan])

7. ``periods=1`` 时窗口只含当前值, 回撤恒为 0::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.MAX_DRAWDOWN(x, 1), 3)
    array([0., 0., 0., 0., 0., 0., 0.])
"""

import numpy as np
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
NAN_POS = np.array([0, 1, 3, 17, 18, 29, 44, 52, 77, 101])
PERIODS = [1, 2, 3, 4, 5, 10, 20]
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


def _mdd_ref(w):
  """无 nan 的正价格窗口上的最大回撤 (非正值)."""
  w = np.asarray(w, dtype=np.float64)
  assert not np.isnan(w).any()
  peak = np.maximum.accumulate(w)
  dd = (w - peak) / peak
  return min(0.0, float(np.min(dd)))


def _mdd_series_ref(x, periods):
  """无 nan 序列上的逐点滚动最大回撤."""
  x = np.asarray(x, dtype=np.float64)
  assert not np.isnan(x).any()
  return np.array(
    [_mdd_ref(x[max(0, i - periods + 1) : i + 1]) for i in range(len(x))],
    dtype=np.float64,
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

  窗口下标始终按原始位置截取; SKIP_NAN 只丢弃窗口内的 nan 值, 不改变窗口
  范围. 严格模式的窗口计数从每个 chunk 的 start 起算.
  """
  x = np.asarray(x, dtype=np.float64)
  strict = bool(flags & al.FLAG_STRICTLY_CYCLE)
  skip = bool(flags & al.FLAG_SKIP_NAN)
  parts = []
  for chunk in np.split(x, groups):
    n = len(chunk)
    s, e = _calc_range(n, start, end)
    out = np.full(n, np.nan)
    for i in range(s, e):
      if strict and (i - s + 1) < periods:
        continue
      w = chunk[max(0, i - periods + 1) : i + 1]
      if skip:
        if np.isnan(chunk[i]):
          continue
        w = w[~np.isnan(w)]
        if len(w) == 0:
          continue
      elif np.isnan(w).any():
        continue
      out[i] = _mdd_ref(w)
    parts.append(out)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(x, periods, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.MAX_DRAWDOWN(np.asarray(x, dtype=np.float64), periods),
    dtype=np.float64,
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261106)
  return rng.uniform(5.0, 20.0, size=n).astype(np.float64)


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


@pytest.mark.parametrize("periods", PERIODS)
def test_mdd_matches_numpy_reference(periods):
  """flags=0, groups=1: 与 numpy 逐窗口参考实现逐点比较, 结果恒非正."""
  x = _clean_data()
  actual = _calc(x, periods)
  _assert_close(actual, _expected(x, periods))
  _assert_close(actual, _mdd_series_ref(x, periods))
  assert np.isfinite(actual).all()
  assert (actual <= ATOL).all()


@pytest.mark.parametrize("periods", PERIODS)
def test_mdd_strictly_cycle(periods):
  """FLAG_STRICTLY_CYCLE: 前 periods-1 个位置为 nan, 其余与参考一致."""
  x = _clean_data()
  flags = al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  _assert_close(
    actual[periods - 1 :], _mdd_series_ref(x, periods)[periods - 1 :]
  )
  assert np.isnan(actual[: periods - 1]).all()
  assert np.isfinite(actual[periods - 1 :]).all()


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_mdd_nan_without_skip_propagates(flags):
  """无 SKIP_NAN 时, 窗口内任一 nan 都会使结果变成 nan."""
  x = _nan_data()
  periods = 4
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  for i in range(len(x)):
    if np.isnan(x[max(0, i - periods + 1) : i + 1]).any():
      assert np.isnan(actual[i])


def test_mdd_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: nan 位置输出 nan, 窗口内其它 nan 不参与回撤计算."""
  x = _nan_data()
  periods = 4
  flags = al.FLAG_SKIP_NAN
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  assert np.isnan(actual[np.isnan(x)]).all()
  assert np.isfinite(actual[~np.isnan(x)]).all()

  # 手工核对: 窗口按原始下标截取, 丢掉其中的 nan 后交给无 nan 参考实现
  for i in range(len(x)):
    w = x[max(0, i - periods + 1) : i + 1]
    if np.isnan(x[i]):
      assert np.isnan(actual[i])
    else:
      _assert_close(actual[i : i + 1], np.array([_mdd_ref(w[~np.isnan(w)])]))


@pytest.mark.parametrize("flags", FLAG_COMBOS)
def test_mdd_all_flag_combinations_with_nan(flags):
  """四种 flag 组合在含 nan 数据上都与参考实现一致."""
  x = _nan_data()
  periods = 6
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))


@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_mdd_groups(groups, flags):
  """group-major 分块: 每个 chunk 的窗口不跨边界, 各自重新 warm-up."""
  x = _clean_data()
  periods = 4
  actual = _calc(x, periods, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, periods, groups=groups, flags=flags))

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    chunk = x[lo : lo + chunk_size]
    got = actual[lo : lo + chunk_size]
    ref = _mdd_series_ref(chunk, periods)
    if flags & al.FLAG_STRICTLY_CYCLE:
      assert np.isnan(got[: periods - 1]).all()
      _assert_close(got[periods - 1 :], ref[periods - 1 :])
    else:
      _assert_close(got, ref)


@pytest.mark.parametrize("groups", GROUPS)
def test_mdd_skip_nan_within_each_group(groups):
  """SKIP_NAN 的窗口按 chunk 内的原始下标截取, 不跨 group 借用数据."""
  x = _nan_data()
  periods = 5
  flags = al.FLAG_SKIP_NAN
  actual = _calc(x, periods, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, periods, groups=groups, flags=flags))

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    chunk = x[lo : lo + chunk_size]
    assert np.isnan(actual[lo : lo + chunk_size][np.isnan(chunk)]).all()


@pytest.mark.parametrize("start,end", START_END)
def test_mdd_start_end(start, end):
  """start/end: 仅校验 [start, end) 区间, 区间外必须全是 nan."""
  x = _clean_data()
  periods = 4
  actual = _calc(x, periods, start=start, end=end)
  _assert_close(actual, _expected(x, periods, start=start, end=end))
  s, e = _calc_range(len(x), start, end)
  if s >= e:
    assert np.isnan(actual).all()
  else:
    assert np.isnan(actual[:s]).all()
    assert np.isnan(actual[e:]).all()
    assert np.isfinite(actual[s:e]).all()


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_mdd_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  x = _clean_data()
  periods = 4
  start, end = 6, 25
  actual = _calc(x, periods, start=start, end=end, groups=groups)
  _assert_close(
    actual,
    _expected(x, periods, start=start, end=end, groups=groups),
  )
  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


def test_mdd_window_uses_pre_start_data():
  """滚动窗口可引用 start 之前的数据: start 处的结果与全量计算一致."""
  x = np.linspace(10.0, 20.0, N)
  periods = 5
  start, end = 10, 50
  actual = _calc(x, periods, start=start, end=end)
  _assert_close(
    actual[start:end], _mdd_series_ref(x, periods)[start:end]
  )
  _assert_close(actual, _expected(x, periods, start=start, end=end))


def test_mdd_skip_nan_with_start_end():
  """SKIP_NAN + start/end: 窗口仍按原始下标, 只丢弃区间内的 nan."""
  x = _nan_data()
  periods = 4
  ctx = dict(start=10, end=90, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, periods, **ctx)
  _assert_close(actual, _expected(x, periods, **ctx))
  assert np.isnan(actual[:10]).all()
  assert np.isnan(actual[90:]).all()
  assert np.isnan(actual[np.isnan(x)]).all()


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_mdd_skip_nan_strict_start_end_with_groups(groups):
  """SKIP_NAN | STRICTLY_CYCLE 与负数 end、groups 组合."""
  x = _nan_data()
  periods = 4
  ctx = dict(
    start=8,
    end=-8,
    groups=groups,
    flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
  )
  actual = _calc(x, periods, **ctx)
  _assert_close(actual, _expected(x, periods, **ctx))


def test_mdd_edge_cases():
  """periods=1 恒为 0; 窗口长于数据; 单调/常数列; periods=0 报错."""
  x = _nan_data()
  _assert_close(_calc(x, 1), np.where(np.isnan(x), np.nan, 0.0))

  periods = len(x) + 1
  _assert_close(_calc(x, periods), _expected(x, periods))
  assert np.isnan(_calc(x, periods, flags=al.FLAG_STRICTLY_CYCLE)).all()
  assert np.isnan(
    _calc(x, periods, flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE)
  ).all()

  up = np.linspace(10.0, 20.0, 30)
  down = up[::-1].copy()
  constant = np.full(30, 12.5)
  for data in (up, constant):
    _assert_close(_calc(data, 6), np.zeros(len(data)))
  _assert_close(_calc(down, 6), _mdd_series_ref(down, 6))
  assert (_calc(down, 6) < 0).any()

  with pytest.raises(ValueError):
    _calc(x, 0)
