# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""SUMBARS (向后累计达到目标额所需 bar 数) 集成测试.

算法简介
--------
对每个位置 ``i`` 从当前位置向序列起点回溯累加, 返回累计和首次达到
``amount`` 时经过的 bar 数; 若回溯到起点仍不足 ``amount`` 则输出 nan::

    SUMBARS(x, amount)[i] = min(b : sum(x[i-b+1..i]) >= amount)

与 ``SUM`` 不同, 这里的窗口长度不是固定的, 而是由 ``amount`` 决定. 没有
滚动周期概念, 因此 ``FLAG_STRICTLY_CYCLE`` 不改变结果. 不设置
``FLAG_SKIP_NAN`` 时, 回溯途中遇到 nan 会让累计和变成 nan, 之后再也无法
满足条件 (输出 nan); 设置 ``FLAG_SKIP_NAN`` 时, nan 被跳过且不计入 bar 数,
等价于把有效值压紧后再回溯累计.

对比基准用 numpy 逐点回溯实现上述算法逻辑, 只处理无 nan 的数据, 不实现
SKIP_NAN / Group / start-end 机制: group 按 chunk 分块分别计算 (回溯不跨
chunk), start/end 只校验 [start, end) 区间, 但窗口仍可引用 start 之前的数据.

`set_ctx` 只覆盖传入参数, 因此每个用例都显式重置 start/end/groups/flags.

场景与结果示例
--------------
1. 常规计算::

    >>> import numpy as np
    >>> import alpha as al
    >>> x = np.arange(1., 11.)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)
    >>> np.round(al.SUMBARS(x, 10.0), 4)
    array([nan, nan, nan,  4.,  3.,  2.,  2.,  2.,  2.,  1.])

   i=3 时 4+3+2+1=10, 需要 4 根 bar; i=9 时 10 本身已达标, 需要 1 根.

2. 改变 ``amount``::

    >>> np.round(al.SUMBARS(x, 5.0), 4)
    array([nan, nan,  2.,  2.,  1.,  1.,  1.,  1.,  1.,  1.])

3. x 中插入 nan, 不设置 ``FLAG_SKIP_NAN`` 时 nan 会截断回溯累计::

    >>> xn = x.copy(); xn[3] = np.nan
    >>> np.round(al.SUMBARS(xn, 10.0), 4)
    array([nan, nan, nan, nan, nan,  2.,  2.,  2.,  2.,  1.])

   i=5 时 6+5=11 已达标, 尚未回溯到 nan 所在的 i=3, 因此仍有结果.

4. ``FLAG_SKIP_NAN`` 让 nan 不参与运算也不计入 bar 数, 等价于压紧有效值::

    >>> al.set_ctx(start=0, end=0, groups=1, flags=al.FLAG_SKIP_NAN)
    >>> np.round(al.SUMBARS(xn, 10.0), 4)
    array([nan, nan, nan, nan,  3.,  2.,  2.,  2.,  2.,  1.])

   i=4 时跳过 nan 后 5+3+2=10, 需要 3 根有效 bar.

5. ``groups`` 按 group-major 切块, 回溯不跨 chunk::

    >>> z = np.arange(1., 13.)
    >>> al.set_ctx(start=0, end=0, groups=2, flags=0)
    >>> np.round(al.SUMBARS(z, 10.0), 4)
    array([nan, nan, nan,  4.,  3.,  2., nan,  2.,  2.,  1.,  1.,  1.])

6. ``start`` / ``end`` 只计算 [start, end) 区间::

    >>> al.set_ctx(start=3, end=8, groups=1, flags=0)
    >>> np.round(al.SUMBARS(x, 10.0), 4)
    array([nan, nan, nan,  4.,  3.,  2.,  2.,  2., nan, nan])
"""

import numpy as np
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
N = 120
NAN_POS = np.array([3, 17, 18, 29, 44, 52, 77, 101])


# ---------------------------------------------------------------------------
# numpy 对比实现
# ---------------------------------------------------------------------------


def _sumbars_ref(x, amount, *, skip):
  """无 nan 数据上的 SUMBARS 参考实现 (逐点向后回溯累计)."""
  x = np.asarray(x, dtype=np.float64)
  out = np.full(len(x), np.nan)
  for i in range(len(x)):
    total = 0.0
    bars = 0
    for j in range(i, -1, -1):
      val = x[j]
      if skip:
        if not np.isnan(val):
          total += val
          bars += 1
      else:
        total += val
        bars += 1
      if not np.isnan(total) and total >= amount:
        out[i] = float(bars)
        break
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


def _expected(x, amount, *, start=0, end=0, groups=1, flags=0):
  """按 chunk 全量回溯计算参考值, 再只保留 [start, end) 区间.

  ``FLAG_STRICTLY_CYCLE`` 对没有固定窗口的 SUMBARS 不产生影响;
  ``FLAG_SKIP_NAN`` 跳过 nan 且不计入 bar 数.
  """
  x = np.asarray(x, dtype=np.float64)
  skip = bool(flags & al.FLAG_SKIP_NAN)
  parts = []
  for chunk in np.split(x, groups):
    n = len(chunk)
    s, e = _calc_range(n, start, end)
    ref = _sumbars_ref(chunk, amount, skip=skip)
    ref[:s] = np.nan
    ref[e:] = np.nan
    parts.append(ref)
  return np.concatenate(parts)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(x, amount, *, start=0, end=0, groups=1, flags=0):
  al.set_ctx(start=start, end=end, groups=groups, flags=flags)
  return np.asarray(
    al.SUMBARS(np.asarray(x, dtype=np.float64), float(amount)),
    dtype=np.float64,
  )


def _clean_data(n=N):
  rng = np.random.default_rng(20261111)
  return rng.uniform(0.5, 3.0, size=n).astype(np.float64)


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


@pytest.mark.parametrize("amount", [3.0, 5.0, 10.0, 20.0, 50.0])
def test_sumbars_matches_reference(amount):
  """flags=0, groups=1: 与逐点回溯累计参考实现逐点比较."""
  x = _clean_data()
  actual = _calc(x, amount, flags=0)
  _assert_close(actual, _expected(x, amount, flags=0))
  finite = np.isfinite(actual)
  assert finite.any()
  assert (actual[finite] >= 1).all()


@pytest.mark.parametrize(
  "flags",
  [0, al.FLAG_STRICTLY_CYCLE, al.FLAG_SKIP_NAN,
   al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE],
)
def test_sumbars_flags(flags):
  """无 nan 数据上各 flag 组合; STRICTLY_CYCLE 不改变结果."""
  x = _clean_data()
  actual = _calc(x, 10.0, flags=flags)
  _assert_close(actual, _expected(x, 10.0, flags=flags))
  _assert_close(actual, _calc(x, 10.0, flags=0))


@pytest.mark.parametrize("flags", [0, al.FLAG_STRICTLY_CYCLE])
@pytest.mark.parametrize("with_nan", [False, True])
def test_sumbars_without_skip_nan_propagates(flags, with_nan):
  """无 SKIP_NAN 时, 回溯途中遇到 nan 会使累计和失效."""
  x = _nan_data() if with_nan else _clean_data()
  actual = _calc(x, 10.0, flags=flags)
  _assert_close(actual, _expected(x, 10.0, flags=flags))

  if with_nan:
    # 第一个 nan 之前的回溯累计无法跨越它, 只有已提前达标的位置才有结果.
    nan_idx = int(np.flatnonzero(np.isnan(x))[0])
    assert np.isnan(actual[:nan_idx]).all()


def test_sumbars_skip_nan_ignores_nan_values():
  """FLAG_SKIP_NAN: 有效位置结果等于压紧后的无 nan 数据参考结果."""
  x = _nan_data()
  actual = _calc(x, 10.0, flags=al.FLAG_SKIP_NAN)
  _assert_close(actual, _expected(x, 10.0, flags=al.FLAG_SKIP_NAN))

  valid = ~np.isnan(x)
  compacted = _sumbars_ref(x[valid], 10.0, skip=False)
  _assert_close(actual[valid], compacted)


def test_sumbars_skip_nan_and_strictly_cycle():
  """两个 flag 组合: 与单独 SKIP_NAN 结果一致."""
  x = _nan_data()
  flags = al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE
  actual = _calc(x, 10.0, flags=flags)
  _assert_close(actual, _expected(x, 10.0, flags=flags))
  _assert_close(actual, _calc(x, 10.0, flags=al.FLAG_SKIP_NAN))


@pytest.mark.parametrize("groups", [1, 2, 3, 4, 6])
def test_sumbars_groups(groups):
  """group-major 分块: 回溯不跨 chunk, 每块从自己的起点开始."""
  x = _clean_data()
  actual = _calc(x, 10.0, groups=groups, flags=0)
  _assert_close(actual, _expected(x, 10.0, groups=groups, flags=0))

  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    assert np.isnan(actual[lo])


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_sumbars_skip_nan_within_each_group(groups):
  """SKIP_NAN 的压紧过程按 chunk 独立进行, 不会跨 group 借用数据."""
  x = _nan_data()
  actual = _calc(x, 10.0, groups=groups, flags=al.FLAG_SKIP_NAN)
  _assert_close(
    actual, _expected(x, 10.0, groups=groups, flags=al.FLAG_SKIP_NAN)
  )


@pytest.mark.parametrize(
  "start,end", [(3, 40), (0, 60), (17, 0), (0, -20), (5, -5)]
)
def test_sumbars_start_end(start, end):
  """start/end: 窗口可引用 start 之前的数据, 区间外全部为 nan."""
  x = _clean_data()
  actual = _calc(x, 10.0, start=start, end=end, flags=0)
  expected = _expected(x, 10.0, start=start, end=end, flags=0)
  s, e = _calc_range(len(x), start, end)
  _assert_close(actual[s:e], expected[s:e])
  assert np.isnan(actual[:s]).all()
  assert np.isnan(actual[e:]).all()


@pytest.mark.parametrize("groups", [1, 2, 3])
def test_sumbars_start_end_with_groups(groups):
  """start/end 是 chunk 内的相对下标, 每个 group 各自生效."""
  x = _clean_data()
  start, end = 6, 25
  actual = _calc(x, 10.0, start=start, end=end, groups=groups, flags=0)
  expected = _expected(
    x, 10.0, start=start, end=end, groups=groups, flags=0
  )
  chunk_size = len(x) // groups
  for g in range(groups):
    lo = g * chunk_size
    _assert_close(actual[lo + start : lo + end], expected[lo + start : lo + end])
    assert np.isnan(actual[lo : lo + start]).all()
    assert np.isnan(actual[lo + end : lo + chunk_size]).all()


@pytest.mark.parametrize("flags", [0, al.FLAG_SKIP_NAN])
def test_sumbars_start_end_with_nan(flags):
  """含 nan 数据的 start/end: 只校验 [start, end) 内的结果."""
  x = _nan_data()
  start, end = 10, 90
  actual = _calc(x, 10.0, start=start, end=end, flags=flags)
  expected = _expected(x, 10.0, start=start, end=end, flags=flags)
  _assert_close(actual[start:end], expected[start:end])
  assert np.isnan(actual[:start]).all()
  assert np.isnan(actual[end:]).all()


def test_sumbars_edge_cases():
  """amount <= 0 立即达标; amount 大于总和 / 全 nan 数据输出 nan."""
  x = _clean_data()
  _assert_close(_calc(x, 0.0, flags=0), np.ones(len(x)))
  assert np.isnan(_calc(x, x.sum() * 2, flags=0)).all()

  all_nan = np.full(20, np.nan)
  assert np.isnan(_calc(all_nan, 1.0, flags=0)).all()
  assert np.isnan(_calc(all_nan, 1.0, flags=al.FLAG_SKIP_NAN)).all()

  constant = np.full(20, 2.0)
  actual = _calc(constant, 6.0, flags=0)
  _assert_close(actual[2:], np.full(18, 3.0))
  assert np.isnan(actual[:2]).all()
