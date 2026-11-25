# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""High-performance cross-sectional correlation and bucket/tranche aggregation operators."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .algo import _algo
from .algo._algo import get_ctx

__all__ = [
  "CC_CORR",
  "CC_SPEARMAN",
  "CC_IC",
  "CC_SPEARMAN_MATRIX",
  "TRANCHE_RETURNS",
  "BUCKET_MEANS",
  "BUCKET_TURNOVER",
]


def _prepare_matrix(
  arr: np.ndarray,
  dtype: type | np.dtype,
  *,
  groups: int | None = None,
  name: str = "input",
) -> tuple[np.ndarray, int, int]:
  a = np.asarray(arr)
  if a.ndim == 2:
    g, b = int(a.shape[0]), int(a.shape[1])
  elif a.ndim == 1:
    g = int(groups) if groups is not None else max(1, int(get_ctx()[2]))
    if g <= 0:
      raise ValueError(f"{name}: groups must be positive, got {g}")
    if len(a) % g != 0:
      raise ValueError(
        f"{name}: 1D array length ({len(a)}) is not divisible by groups ({g})"
      )
    b = len(a) // g
  else:
    raise ValueError(f"{name}: expected 1D or 2D array, got shape {a.shape}")

  if a.dtype != dtype or not a.flags.c_contiguous:
    a = np.ascontiguousarray(a, dtype=dtype)
  return a.reshape(-1), g, b


def _check_same_grid(
  g1: int, b1: int, g2: int, b2: int, name1: str, name2: str
) -> None:
  if (g1, b1) != (g2, b2):
    raise ValueError(
      f"shape mismatch between {name1} ({g1}, {b1}) and {name2} ({g2}, {b2})"
    )


def CC_CORR(
  x: np.ndarray,
  y: np.ndarray,
  *,
  min_obs: int = 2,
  lag: int = 0,
  groups: int | None = None,
) -> np.ndarray:
  """Cross-sectional Pearson correlation per bar between ``x[:, t]`` and ``y[:, t + lag]``."""
  xf, gx, bx = _prepare_matrix(x, np.float64, groups=groups, name="x")
  yf, gy, by = _prepare_matrix(y, np.float64, groups=groups if groups is not None else gx, name="y")
  _check_same_grid(gx, bx, gy, by, "x", "y")
  return _algo.cc_corr(gx, bx, xf, yf, min_obs=int(min_obs), lag=int(lag))


def CC_SPEARMAN(
  x: np.ndarray,
  y: np.ndarray,
  *,
  min_obs: int = 2,
  lag: int = 0,
  groups: int | None = None,
) -> np.ndarray:
  """Cross-sectional Spearman rank correlation per bar between ``x[:, t]`` and ``y[:, t + lag]``."""
  xf, gx, bx = _prepare_matrix(x, np.float64, groups=groups, name="x")
  yf, gy, by = _prepare_matrix(y, np.float64, groups=groups if groups is not None else gx, name="y")
  _check_same_grid(gx, bx, gy, by, "x", "y")
  return _algo.cc_spearman(gx, bx, xf, yf, min_obs=int(min_obs), lag=int(lag))


def CC_IC(
  x: np.ndarray,
  y: np.ndarray,
  *,
  min_obs: int = 2,
  lag: int = 0,
  groups: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Jointly compute cross-sectional Pearson IC and Spearman RankIC series per bar."""
  xf, gx, bx = _prepare_matrix(x, np.float64, groups=groups, name="x")
  yf, gy, by = _prepare_matrix(y, np.float64, groups=groups if groups is not None else gx, name="y")
  _check_same_grid(gx, bx, gy, by, "x", "y")
  return _algo.cc_ic(gx, bx, xf, yf, min_obs=int(min_obs), lag=int(lag))


def CC_SPEARMAN_MATRIX(
  factors: Sequence[np.ndarray],
  *,
  min_obs: int = 2,
  groups: int | None = None,
) -> np.ndarray:
  """Compute pairwise time-averaged cross-sectional Spearman rank correlation matrix."""
  nf = len(factors)
  if nf == 0:
    return np.empty((0, 0), dtype=np.float64)

  flat_list: list[np.ndarray] = []
  g0, b0 = 0, 0
  for idx, mat in enumerate(factors):
    mf, gi, bi = _prepare_matrix(
      mat,
      np.float64,
      groups=groups if idx == 0 or groups is not None else g0,
      name=f"factors[{idx}]",
    )
    if idx == 0:
      g0, b0 = gi, bi
    else:
      _check_same_grid(g0, b0, gi, bi, "factors[0]", f"factors[{idx}]")
    flat_list.append(mf)

  raw = _algo.cc_spearman_matrix(g0, b0, flat_list, min_obs=int(min_obs))
  return raw.reshape(nf, nf)


def TRANCHE_RETURNS(
  bins: np.ndarray,
  step_first: np.ndarray,
  step_cont: np.ndarray,
  periods: Sequence[int],
  n_buckets: int,
  *,
  entry_lag: int = 1,
  entry_mask: np.ndarray | None = None,
  groups: int | None = None,
) -> np.ndarray:
  """Compute overlapping equal-weight tranche daily returns across all buckets and periods.

  Returns:
    Float64 array of shape ``(len(periods), n_buckets + 1, bars)``, where
    index ``n_buckets`` along axis 1 is the equal-weight union ("Market")
    across all valid binned entities.
  """
  bf, g, b = _prepare_matrix(bins, np.float64, groups=groups, name="bins")
  same_step = step_first is step_cont
  sf, g_sf, b_sf = _prepare_matrix(
    step_first, np.float64, groups=groups if groups is not None else g, name="step_first"
  )
  _check_same_grid(g, b, g_sf, b_sf, "bins", "step_first")
  if same_step:
    sc = sf
  else:
    sc, g_sc, b_sc = _prepare_matrix(
      step_cont, np.float64, groups=groups if groups is not None else g, name="step_cont"
    )
    _check_same_grid(g, b, g_sc, b_sc, "bins", "step_cont")

  em_f: np.ndarray | None = None
  if entry_mask is not None:
    em_f, g_em, b_em = _prepare_matrix(
      entry_mask, np.bool_, groups=groups if groups is not None else g, name="entry_mask"
    )
    _check_same_grid(g, b, g_em, b_em, "bins", "entry_mask")

  p_list = [int(p) for p in periods]
  raw = _algo.tranche_returns(
    g,
    b,
    bf,
    sf,
    sc,
    p_list,
    int(n_buckets),
    entry_lag=int(entry_lag),
    entry_mask=em_f,
  )
  return raw.reshape(len(p_list), int(n_buckets) + 1, b)


def BUCKET_MEANS(
  bins: np.ndarray,
  values: np.ndarray,
  n_buckets: int,
  *,
  groups: int | None = None,
) -> np.ndarray:
  """Compute per-bucket and overall cross-sectional means per bar.

  Returns:
    Float64 array of shape ``(n_buckets + 1, bars)`` where row ``n_buckets`` is
    the mean across all entities in buckets ``0..n_buckets - 1``.
  """
  bf, g, b = _prepare_matrix(bins, np.float64, groups=groups, name="bins")
  vf, g_v, b_v = _prepare_matrix(
    values, np.float64, groups=groups if groups is not None else g, name="values"
  )
  _check_same_grid(g, b, g_v, b_v, "bins", "values")
  raw = _algo.bucket_means(g, b, bf, vf, int(n_buckets))
  return raw.reshape(int(n_buckets) + 1, b)


def BUCKET_TURNOVER(
  bins: np.ndarray,
  n_buckets: int,
  *,
  lag: int = 1,
  entry_mask: np.ndarray | None = None,
  groups: int | None = None,
) -> np.ndarray:
  """Compute average one-way membership turnover per bucket at ``lag``."""
  bf, g, b = _prepare_matrix(bins, np.float64, groups=groups, name="bins")
  em_f: np.ndarray | None = None
  if entry_mask is not None:
    em_f, g_em, b_em = _prepare_matrix(
      entry_mask, np.bool_, groups=groups if groups is not None else g, name="entry_mask"
    )
    _check_same_grid(g, b, g_em, b_em, "bins", "entry_mask")
  return _algo.bucket_turnover(
    g, b, bf, int(n_buckets), lag=int(lag), entry_mask=em_f
  )
