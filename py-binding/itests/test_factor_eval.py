# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Integration tests for cross-sectional correlation and tranche/bucket primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import alpha as al


def _ref_cc_corr(x: np.ndarray, y: np.ndarray, min_obs: int = 2, lag: int = 0) -> np.ndarray:
  bars = x.shape[1]
  if bars <= lag:
    return np.empty(0, dtype=np.float64)
  out = np.full(bars - lag, np.nan, dtype=np.float64)
  for t in range(bars - lag):
    xv = x[:, t]
    yv = y[:, t + lag]
    ok = np.isfinite(xv) & np.isfinite(yv)
    if int(np.sum(ok)) < max(min_obs, 2):
      continue
    xa = xv[ok] - np.mean(xv[ok])
    ya = yv[ok] - np.mean(yv[ok])
    den = float(np.sqrt(np.dot(xa, xa) * np.dot(ya, ya)))
    if den > 1e-12:
      out[t] = float(np.dot(xa, ya) / den)
  return out


def _ref_cc_spearman(x: np.ndarray, y: np.ndarray, min_obs: int = 2, lag: int = 0) -> np.ndarray:
  bars = x.shape[1]
  if bars <= lag:
    return np.empty(0, dtype=np.float64)
  out = np.full(bars - lag, np.nan, dtype=np.float64)
  for t in range(bars - lag):
    xv = x[:, t]
    yv = y[:, t + lag]
    ok = np.isfinite(xv) & np.isfinite(yv)
    if int(np.sum(ok)) < max(min_obs, 2):
      continue
    rx = pd.Series(xv[ok]).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(yv[ok]).rank(method="average").to_numpy(dtype=np.float64)
    xa = rx - np.mean(rx)
    ya = ry - np.mean(ry)
    den = float(np.sqrt(np.dot(xa, xa) * np.dot(ya, ya)))
    if den > 1e-12:
      out[t] = float(np.dot(xa, ya) / den)
  return out


def _ref_tranche(
  bins: np.ndarray,
  entry_mask: np.ndarray | None,
  step_first: np.ndarray,
  step_cont: np.ndarray,
  periods: list[int],
  n_buckets: int,
  entry_lag: int,
) -> np.ndarray:
  n_objs, bars = bins.shape
  out = np.full((len(periods), n_buckets + 1, bars), np.nan, dtype=np.float64)
  em = np.ones((n_objs, bars), dtype=bool) if entry_mask is None else entry_mask

  for pi, period in enumerate(periods):
    for q in range(n_buckets + 1):
      if q < n_buckets:
        bmask = (bins == q) & em
      else:
        bmask = np.isfinite(bins) & (bins >= 0) & (bins < n_buckets) & em

      sum_ret = np.zeros(bars, dtype=np.float64)
      cnt_tr = np.zeros(bars, dtype=np.float64)
      for age in range(period):
        shift = entry_lag + age
        if shift >= bars:
          break
        m = bmask[:, : bars - shift]
        r_mat = step_first[:, shift:] if age == 0 else step_cont[:, shift:]
        valid = m & np.isfinite(r_mat)
        n_v = np.sum(valid, axis=0)
        s_v = np.sum(np.where(valid, r_mat, 0.0), axis=0)
        has_v = n_v > 0
        tr = np.zeros(bars - shift, dtype=np.float64)
        tr[has_v] = s_v[has_v] / n_v[has_v]
        sum_ret[shift:] += np.where(has_v, tr, 0.0)
        cnt_tr[shift:] += has_v.astype(np.float64)

      with np.errstate(divide="ignore", invalid="ignore"):
        out[pi, q] = np.where(cnt_tr > 0, sum_ret / cnt_tr, np.nan)
  return out


def test_cc_corr_spearman_ic_with_ties_nans_and_lag():
  rng = np.random.default_rng(101)
  n_objs, bars = 40, 35
  x = rng.normal(0.0, 1.0, (n_objs, bars))
  y = 0.6 * x + rng.normal(0.0, 0.8, (n_objs, bars))
  # Inject ties, NaNs, Inf, and an all-NaN bar
  x[0:5, 3] = 1.25
  y[2:7, 3] = -0.5
  x[rng.random((n_objs, bars)) < 0.18] = np.nan
  y[rng.random((n_objs, bars)) < 0.15] = np.nan
  x[:, 10] = np.nan
  x[1, 12] = np.inf
  y[2, 12] = -np.inf

  for lag in (0, 1, 5, 35):
    ic_ref = _ref_cc_corr(x, y, min_obs=10, lag=lag)
    ric_ref = _ref_cc_spearman(x, y, min_obs=10, lag=lag)

    ic_c = al.CC_CORR(x, y, min_obs=10, lag=lag)
    ric_s = al.CC_SPEARMAN(x, y, min_obs=10, lag=lag)
    ic_j, ric_j = al.CC_IC(x, y, min_obs=10, lag=lag)

    np.testing.assert_allclose(ic_c, ic_ref, atol=1e-12, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(ic_j, ic_ref, atol=1e-12, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(ric_s, ric_ref, atol=1e-12, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(ric_j, ric_ref, atol=1e-12, rtol=1e-12, equal_nan=True)

  # 1D + context groups dispatch
  al.set_ctx(start=0, end=0, groups=n_objs, flags=0)
  ic_1d, ric_1d = al.CC_IC(x.reshape(-1), y.reshape(-1), min_obs=10, lag=0)
  np.testing.assert_allclose(ic_1d, _ref_cc_corr(x, y, 10, 0), atol=1e-12, equal_nan=True)
  np.testing.assert_allclose(ric_1d, _ref_cc_spearman(x, y, 10, 0), atol=1e-12, equal_nan=True)


def test_cc_spearman_matrix_exact_on_mixed_masks():
  rng = np.random.default_rng(202)
  n_objs, bars = 50, 30
  f0 = rng.normal(size=(n_objs, bars))
  f1 = 0.7 * f0 + rng.normal(0.0, 0.5, (n_objs, bars))
  f2 = rng.integers(0, 4, size=(n_objs, bars)).astype(np.float64)
  f3 = -f0.copy()

  # Shared mask on half the bars, different masks on the rest
  common_nan = rng.random((n_objs, bars)) < 0.15
  f0[common_nan] = np.nan
  f1[common_nan] = np.nan
  f2[common_nan] = np.nan
  f3[common_nan] = np.nan
  sub_f1 = f1[:, :15]
  sub_f1[rng.random((n_objs, 15)) < 0.25] = np.nan
  f2[:, 25:] = np.nan  # all-NaN tail in f2

  factors = [f0, f1, f2, f3]
  actual = al.CC_SPEARMAN_MATRIX(factors, min_obs=8)
  expected = np.eye(4, dtype=np.float64)
  for i in range(4):
    for j in range(i + 1, 4):
      s = _ref_cc_spearman(factors[i], factors[j], min_obs=8, lag=0)
      v = s[np.isfinite(s)]
      val = float(np.mean(v)) if len(v) > 0 else np.nan
      expected[i, j] = val
      expected[j, i] = val

  np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("same_step", [True, False])
@pytest.mark.parametrize("use_mask", [True, False])
def test_tranche_returns_bucket_means_and_turnover(same_step, use_mask):
  rng = np.random.default_rng(303)
  n_objs, bars, n_buckets = 60, 40, 5
  factor = np.where(rng.random((n_objs, bars)) < 0.85, rng.normal(size=(n_objs, bars)), np.nan)
  with al.Context(groups=n_objs, flags=al.FLAG_SKIP_NAN, start=0, end=bars):
    bins = np.asarray(al.BINS(factor.reshape(-1), n_buckets), dtype=np.float64).reshape(
      n_objs, bars
    )

  step_cont = np.where(
    rng.random((n_objs, bars)) < 0.88, rng.normal(0.001, 0.02, (n_objs, bars)), np.nan
  )
  step_first = (
    step_cont
    if same_step
    else np.where(
      rng.random((n_objs, bars)) < 0.88, rng.normal(0.0005, 0.015, (n_objs, bars)), np.nan
    )
  )
  entry_mask = (rng.random((n_objs, bars)) < 0.9) if use_mask else None
  periods = [1, 5, 12]

  actual = al.TRANCHE_RETURNS(
    bins,
    step_first,
    step_cont,
    periods,
    n_buckets,
    entry_lag=1,
    entry_mask=entry_mask,
  )
  expected = _ref_tranche(bins, entry_mask, step_first, step_cont, periods, n_buckets, 1)
  np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12, equal_nan=True)

  # BUCKET_MEANS
  fwd = np.where(rng.random((n_objs, bars)) < 0.85, rng.normal(0.01, 0.05, (n_objs, bars)), np.nan)
  means_actual = al.BUCKET_MEANS(bins, fwd, n_buckets)
  means_exp = np.full((n_buckets + 1, bars), np.nan, dtype=np.float64)
  for q in range(n_buckets):
    v = (bins == q) & np.isfinite(fwd)
    c = np.sum(v, axis=0)
    s = np.sum(np.where(v, fwd, 0.0), axis=0)
    ok = c > 0
    means_exp[q, ok] = s[ok] / c[ok]
  vm = np.isfinite(bins) & (bins >= 0) & (bins < n_buckets) & np.isfinite(fwd)
  cm = np.sum(vm, axis=0)
  sm = np.sum(np.where(vm, fwd, 0.0), axis=0)
  okm = cm > 0
  means_exp[n_buckets, okm] = sm[okm] / cm[okm]
  np.testing.assert_allclose(means_actual, means_exp, atol=1e-12, rtol=1e-12, equal_nan=True)

  # BUCKET_TURNOVER
  for lag in (1, 5, 50):
    turn_actual = al.BUCKET_TURNOVER(bins, n_buckets, lag=lag, entry_mask=entry_mask)
    em = np.ones((n_objs, bars), dtype=bool) if entry_mask is None else entry_mask
    turn_exp = np.zeros(n_buckets, dtype=np.float64)
    if lag < bars:
      for q in range(n_buckets):
        mq = (bins == q) & em
        p = mq[:, :-lag]
        c = mq[:, lag:]
        pc = np.sum(p, axis=0)
        cc = np.sum(c, axis=0)
        both = np.sum(p & c, axis=0)
        v = (pc > 0) & (cc > 0)
        if np.any(v):
          turn_exp[q] = float(np.mean(np.clip(1.0 - both[v] / cc[v], 0.0, 1.0)))
    np.testing.assert_allclose(turn_actual, turn_exp, atol=1e-12, rtol=1e-12)


def test_tranche_returns_many_buckets_u16_path():
  rng = np.random.default_rng(404)
  n_objs, bars, n_buckets = 300, 10, 260
  bins = rng.integers(0, n_buckets, size=(n_objs, bars)).astype(np.float64)
  step = rng.normal(0.0, 0.01, size=(n_objs, bars))
  actual = al.TRANCHE_RETURNS(bins, step, step, [3], n_buckets, entry_lag=1)
  expected = _ref_tranche(bins, None, step, step, [3], n_buckets, 1)
  np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12, equal_nan=True)
