# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""REF (引用前 N 期值) 集成测试.

算法简介
--------
REF 把序列右移 `periods` 个位置, 开头不足 `periods` 个历史值时用首值填充::

    REF[i] = x[max(0, i - periods)]

设置 ``FLAG_STRICTLY_CYCLE`` 后前 `periods` 个位置 (尚无完整历史) 输出 nan,
而不是用首值填充. ``periods = 0`` 时输出等于输入.

``FLAG_SKIP_NAN`` 改为在「有效值序列」上做引用: 只在 ``[start, end)``
区间内收集非 nan 的值, nan 位置本身输出 nan, 第 k 个有效值输出它前面第
`periods` 个有效值, 因此前 `periods` 个有效位置没有结果 (nan). 这等价于把
该区间的有效值压紧 (compact) 后跑参考实现, 再映射回有效位置. 组合
``FLAG_STRICTLY_CYCLE`` 不改变这一压紧行为.

参考实现 ``_ref_ref`` 用 numpy 索引独立实现, 只描述算法逻辑, 不调用 alpha,
也不实现 Group / start-end 机制: Group 由测试按 chunk 分块计算, start/end
只校验 ``[start, end)`` 区间的值 (区间外 alpha 一律输出 nan). 普通模式下
窗口可以引用 start 之前的数据; SKIP_NAN 模式下有效值只在区间内压紧.

`set_ctx` 只覆盖传入的参数, 因此每次调用前都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> np.set_printoptions(linewidth=120, suppress=True)
    >>> x = np.arange(1., 11.)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 常规计算: 前 3 个位置用首值填充, 之后取 ``x[i - 3]``::

    >>> al.REF(x, 3)
    array([1., 1., 1., 1., 2., 3., 4., 5., 6., 7.])

2. ``FLAG_STRICTLY_CYCLE``: 前 3 个位置输出 nan::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_STRICTLY_CYCLE)
    >>> al.REF(x, 3)
    array([nan, nan, nan,  1.,  2.,  3.,  4.,  5.,  6.,  7.])

3. 不跳过 nan: 被引用位置是 nan 时结果也是 nan::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> al.REF(xn, 3)
    array([ 1.,  1.,  1.,  1.,  2.,  3., nan,  5.,  6.,  7.])

4. ``FLAG_SKIP_NAN``: 压紧后的有效值为 ``[1, 2, 3, 5, 6, 7, 8, 9, 10]``,
   前 3 个有效位置为 nan, 之后依次引用 ``[1, 2, 3, 5, 6, 7]``::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> al.REF(xn, 3)
    array([nan, nan, nan, nan,  1.,  2.,  3.,  5.,  6.,  7.])

   ``FLAG_SKIP_NAN | FLAG_STRICTLY_CYCLE`` 结果与上面相同.

5. ``periods = 0`` 直接复制输入::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> al.REF(x, 0)
    array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])

6. ``groups`` 把一维数据按 group-major 切块, 每块重新用块首值填充::

    >>> y = np.arange(1., 13.)
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> al.REF(y, 3)
    array([1., 1., 1., 1., 2., 3., 7., 7., 7., 7., 8., 9.])

7. ``start`` / ``end`` 只计算 ``[start, end)`` 区间, 区间外全部为 nan,
   但被引用值仍可来自 start 之前::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> al.REF(x, 3)
    array([nan, nan, nan,  1.,  2.,  3.,  4.,  5., nan, nan])
"""

import numpy as np
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
NAN_POS = np.array([3, 17, 29, 44, 52, 77, 101])


# ---------------------------------------------------------------------------
# numpy 对比实现
# ---------------------------------------------------------------------------


def _ref_ref(x, periods, *, strict=False):
  """REF 参考实现: r[i] = x[max(0, i - periods)].

  ``strict`` 时前 ``periods`` 个位置输出 nan (没有完整历史). 允许输入含
  nan: 被引用位置是 nan, 输出就是 nan.
  """
  x = np.asarray(x, dtype=np.float64)
  idx = np.maximum(np.arange(len(x)) - periods, 0)
  out = x[idx].copy()
  if strict:
    out[:periods] = np.nan
  return out


def _compact_ref(seg, periods):
  """区间内压紧有效值后引用, 再映射回有效位置; nan 位置保持 nan."""
  seg = np.asarray(seg, dtype=np.float64)
  out = np.full(len(seg), np.nan)
  valid = ~np.isnan(seg)
  if valid.any():
    out[valid] = _ref_ref(seg[valid], periods, strict=True)
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
  """按 chunk 全量计算参考值, 再只保留 [start, end) 区间.

  无 SKIP_NAN 时窗口可引用 start 之前的数据; SKIP_NAN 时只在 [start, end)
  段内压紧有效值, 不跨边界借用数据.
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
        ref[s:e] = _compact_ref(chunk[s:e], periods)
      else:
        ref = _ref_ref(chunk, periods, strict=strict)
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
    al.REF(np.asarray(x, dtype=np.float64), periods), dtype=np.float64
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


@pytest.mark.parametrize("periods", [0, 1, 2, 3, 4, 5, 10, 20])
def test_ref_matches_reference(periods):
  """flags=0, groups=1: 与 numpy 索引参考实现逐点比较."""
  x = _clean_data()
  actual = _calc(x, periods, flags=0)
  _assert_close(actual, _expected(x, periods, flags=0))
  _assert_close(actual, _ref_ref(x, periods))
  assert np.isfinite(actual).all()


@pytest.mark.parametrize("periods", [0, 1, 2, 3, 4, 5, 10, 20])
def test_ref_strictly_cycle(periods):
  """FLAG_STRICTLY_CYCLE: 前 periods 个位置输出 nan, 之后与参考实现一致."""
  x = _clean_data()
  flags = al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  _assert_close(actual, _ref_ref(x, periods, strict=True))
  if periods > 0:
    assert np.isnan(actual[:periods]).all()
    assert np.isfinite(actual[periods:]).all()


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_ref_nan_without_skip(flags):
  """无 SKIP_NAN: 被引用位置是 nan 时结果也是 nan."""
  x = _nan_data()
  periods = 4
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  for i in range(periods, len(x)):
    assert np.isnan(actual[i]) == np.isnan(x[i - periods])


def test_ref_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 有效位置等于压紧有效值后的参考结果."""
  x = _nan_data()
  periods = 4
  actual = _calc(x, periods, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, periods, flags=al.FLAG_SKIP_NAN))

  valid = ~np.isnan(x)
  assert np.isnan(actual[~valid]).all()
  _assert_close(actual[valid], _ref_ref(x[valid], periods, strict=True))
  assert np.isnan(actual[valid][:periods]).all()


def test_ref_skip_nan_and_strictly_cycle():
  """两个 flag 组合: 严格周期不改变压紧引用行为."""
  x = _nan_data()
  periods = 4
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))
  _assert_close(actual, _calc(x, periods, flags=al.FLAG_SKIP_NAN))
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
def test_ref_all_flag_combinations_with_nan(flags):
  """四种 flag 组合在含 nan 数据上都与参考实现一致."""
  x = _nan_data()
  periods = 6
  actual = _calc(x, periods, flags=flags)
  _assert_close(actual, _expected(x, periods, flags=flags))


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
def test_ref_groups(groups, flags):
  """group-major 分块: 每个 chunk 独立计算, 开头重新用块首值填充."""
  x = _clean_data()
  periods = 4
  actual = _calc(x, periods, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, periods, groups=groups, flags=flags))

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    chunk = x[lo : lo + chunk_size]
    _assert_close(
      actual[lo : lo + chunk_size],
      _ref_ref(chunk, periods, strict=bool(flags)),
    )


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
def test_ref_skip_nan_within_each_group(groups):
  """SKIP_NAN 的压紧过程按 chunk 独立进行, 不会跨 group 借用数据."""
  x = _nan_data()
  periods = 5
  flags = al.FLAG_SKIP_NAN
  actual = _calc(x, periods, groups=groups, flags=flags)
  _assert_close(actual, _expected(x, periods, groups=groups, flags=flags))
  assert np.isnan(actual[np.isnan(x)]).all()


@pytest.mark.parametrize(
  "start,end", [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5)]
)
def test_ref_start_end(start, end):
  """start/end: 仅校验 [start, end) 区间, 区间外必须全是 nan."""
  x = _clean_data()
  periods = 4
  actual = _calc(x, periods, start=start, end=end, flags=0)
  _assert_close(actual, _expected(x, periods, start=start, end=end, flags=0))

  s, e = _calc_range(len(x), start, end)
  assert np.isnan(actual[:s]).all()
  assert np.isnan(actual[e:]).all()
  assert np.isfinite(actual[s:e]).all()


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_ref_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  x = _clean_data()
  periods = 4
  start, end = 6, 25
  actual = _calc(x, periods, start=start, end=end, groups=groups, flags=0)
  _assert_close(
    actual,
    _expected(x, periods, start=start, end=end, groups=groups, flags=0),
  )

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


def test_ref_skip_nan_with_start_end():
  """SKIP_NAN + start/end: 只有 [start, end) 内的有效值参与压紧引用."""
  x = _nan_data()
  periods = 4
  ctx = dict(start=10, end=90, flags=al.FLAG_SKIP_NAN)
  actual = _calc(x, periods, **ctx)
  _assert_close(actual, _expected(x, periods, **ctx))
  assert np.isnan(actual[:10]).all()
  assert np.isnan(actual[90:]).all()
  assert np.isnan(actual[np.isnan(x)]).all()


def test_ref_edge_cases():
  """periods=0 直接复制输入; 窗口长于数据时按首值填充或输出 nan."""
  x = _clean_data()
  _assert_close(_calc(x, 0, flags=0), x)
  _assert_close(_calc(x, 0, flags=al.FLAG_STRICTLY_CYCLE), x)
  _assert_close(_calc(x, 0, flags=al.FLAG_SKIP_NAN), x)

  long_periods = len(x) + 1
  _assert_close(
    _calc(x, long_periods, flags=0), np.full_like(x, x[0])
  )
  assert np.isnan(_calc(x, long_periods, flags=al.FLAG_STRICTLY_CYCLE)).all()

  xn = _nan_data()
  _assert_close(
    _calc(xn, 0, flags=al.FLAG_SKIP_NAN), _compact_ref(xn, 0)
  )
