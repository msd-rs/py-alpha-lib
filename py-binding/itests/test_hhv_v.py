# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""HHV_V / LLV_V (逐点可变窗口极值) 集成测试 —— 变周期差异复现.

算法简介
--------
两个算子是 HHV / LLV 的变周期版本, 每个位置 ``i`` 用 ``periods[i]`` 决定
窗口长度, 取窗口内的最大 / 最小值::

    HHV_V(x, p)[i] = max(x[max(0, i + 1 - p[i]) : i + 1])
    LLV_V(x, p)[i] = min(x[max(0, i + 1 - p[i]) : i + 1])

``p[i] = 0`` 表示从序列 (chunk) 开头累计取极值. 窗口未填满时用已有的部分
窗口给出结果; 窗口内出现 nan 时该位置输出 nan. 按算法逻辑, 每个位置的
窗口只由其自身的 ``p[i]`` 决定, 与之前位置用过的窗口长度无关.

已知差异 (保留失败)
--------------------
Rust 实现用单调队列 (deque) 维护候选极值: 当 ``p[i]`` 变小、窗口起点后移
时, 会不可逆地 ``pop_front`` 掉队首的旧下标; 之后 ``p`` 再次变大, 窗口
起点前移, 这些旧下标已无法回到队列, 于是窗口缺少历史候选值. 结果是
HHV_V 偏小、LLV_V 偏大. 该问题只在 ``periods`` 数组**非单调**时出现:
固定周期 (含 ``p = 0`` 累计模式) 下窗口起点单调后移, 结果与窗口极值一致.

本文件的 ``_extremum_v_ref`` 严格按上述算法逻辑逐点取窗口 max / min, 不以
当前 alpha 实现为准. ``test_hhv_v_variable_periods_matches_window_extremum``
用一个最小复现和一组固定 seed 的随机变周期数据校验两个算子, 该用例预期
**不通过**, 保留以暴露差异, 待实现修复后会自动转为通过. 其余上下文
(groups / start-end / SKIP_NAN) 的完整集成测试待修复后补充.

`set_ctx` 只覆盖传入的参数, 因此每次调用前都显式重置 start/end/groups/flags.

场景与结果示例
--------------
所有示例先执行::

    >>> import numpy as np
    >>> import alpha as al
    >>> x = np.array([5., 1., 2., 3., 4.])
    >>> p = np.array([1, 1, 1, 5, 5], dtype=np.uint64)
    >>> al.set_ctx(start=0, end=0, groups=1, flags=0)

1. 最小复现: ``p`` 先小后大, i=3、4 的窗口应重新覆盖 x[0] = 5,
   但实现已把该下标从队列中弹出::

    >>> al.HHV_V(x, p).tolist()
    [5.0, 1.0, 2.0, 3.0, 4.0]
    >>> _extremum_v_ref(x, p, "max").tolist()
    [5.0, 1.0, 2.0, 5.0, 5.0]

   LLV_V 同理, i=3、4 的窗口最小值应为 x[1] = 1::

    >>> al.LLV_V(x, p).tolist()
    [5.0, 1.0, 2.0, 2.0, 2.0]
    >>> _extremum_v_ref(x, p, "min").tolist()
    [5.0, 1.0, 2.0, 1.0, 1.0]

2. 固定周期时窗口起点单调后移, 两者一致::

    >>> pf = np.full(5, 3, dtype=np.uint64)
    >>> al.HHV_V(x, pf).tolist()
    [5.0, 5.0, 5.0, 3.0, 4.0]
    >>> _extremum_v_ref(x, pf, "max").tolist()
    [5.0, 5.0, 5.0, 3.0, 4.0]
"""

import numpy as np
import pytest

import alpha as al

ATOL = 1e-6
RTOL = 1e-6
OPS = ["max", "min"]


# ---------------------------------------------------------------------------
# numpy 对比实现
# ---------------------------------------------------------------------------


def _extremum_v_ref(x, periods, op, *, strict=False):
  """按算法逻辑逐点取窗口极值的参考实现, 不依赖 alpha.

  窗口为 ``x[max(0, i + 1 - p[i]) : i + 1]``; ``p[i] = 0`` 时从头累计.
  窗口内含 nan 则输出 nan; ``strict`` 时窗口未满 ``p[i]`` 个值输出 nan.
  """
  x = np.asarray(x, dtype=np.float64)
  p = np.asarray(periods, dtype=np.int64)
  reduce = np.max if op == "max" else np.min
  out = np.full(len(x), np.nan)
  for i in range(len(x)):
    q = int(p[i])
    lo = 0 if q == 0 else max(0, i + 1 - q)
    w = x[lo : i + 1]
    if strict and q > 0 and i < q - 1:
      continue
    if np.isnan(w).any():
      continue
    out[i] = reduce(w)
  return out


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _calc(x, periods, op, *, flags=0):
  al.set_ctx(start=0, end=0, groups=1, flags=flags)
  fn = al.HHV_V if op == "max" else al.LLV_V
  return np.asarray(
    fn(
      np.asarray(x, dtype=np.float64),
      np.asarray(periods, dtype=np.uint64),
    ),
    dtype=np.float64,
  )


def _assert_close(actual, expected):
  assert actual.shape == expected.shape
  np.testing.assert_allclose(
    actual, expected, atol=ATOL, rtol=RTOL, equal_nan=True
  )


# ---------------------------------------------------------------------------
# 集成测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("periods_value", [0, 1, 2, 3, 5, 8])
def test_hhv_v_fixed_periods_matches_reference(op, periods_value):
  """对照组: 固定周期 (含 p=0 累计模式) 时与窗口极值参考实现一致."""
  rng = np.random.default_rng(20261120)
  x = rng.normal(loc=10.0, scale=2.0, size=60).astype(np.float64)
  p = np.full(len(x), periods_value, dtype=np.uint64)
  _assert_close(_calc(x, p, op, flags=0), _extremum_v_ref(x, p, op))


def test_hhv_v_variable_periods_matches_window_extremum():
  """变周期下结果必须等于逐点窗口极值 (已知实现差异, 预期不通过).

  数据分两组: 最小复现 (``p`` 先小后大) 和固定 seed 的随机变周期数据;
  两个算子 HHV_V / LLV_V 一起校验. Rust 侧的单调队列在窗口重新变大时
  无法恢复此前弹出的历史候选值, 因此该断言会失败. 参考实现严格按算法
  逻辑计算, 不为迁就实现而修改.
  """
  x_small = np.array([5.0, 1.0, 2.0, 3.0, 4.0])
  p_small = np.array([1, 1, 1, 5, 5], dtype=np.uint64)

  rng = np.random.default_rng(7)
  x_rand = rng.normal(size=20)
  p_rand = rng.integers(0, 7, size=20).astype(np.uint64)
  cases = [(x_small, p_small), (x_rand, p_rand)]

  actual_parts = []
  expected_parts = []
  for op in OPS:
    for x, p in cases:
      actual_parts.append(_calc(x, p, op, flags=0))
      expected_parts.append(_extremum_v_ref(x, p, op))

  np.testing.assert_allclose(
    np.concatenate(actual_parts),
    np.concatenate(expected_parts),
    atol=ATOL,
    rtol=RTOL,
    equal_nan=True,
    err_msg=(
      "HHV_V / LLV_V 在变周期下偏离逐点窗口极值: "
      "单调队列 pop_front 丢失的历史候选值无法在窗口变大时恢复"
    ),
  )
