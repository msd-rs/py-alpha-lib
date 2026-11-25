// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

//! High-performance cross-sectional factor evaluation primitives.
//!
//! All 2D arrays use the standard row-major ``(groups, bars)`` layout
//! (``idx = g * bars + t``).

use std::cmp::Ordering;

use rayon::prelude::*;

use crate::algo::Error;

#[derive(Clone, Copy)]
struct UnsafeF64Ptr {
  ptr: *mut f64,
  len: usize,
}

impl UnsafeF64Ptr {
  fn new(ptr: *mut f64, len: usize) -> Self {
    UnsafeF64Ptr { ptr, len }
  }

  #[allow(clippy::mut_from_ref)]
  fn get(&self) -> &mut [f64] {
    unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
  }
}

unsafe impl Send for UnsafeF64Ptr {}
unsafe impl Sync for UnsafeF64Ptr {}

#[inline]
fn validate_grid_len(groups: usize, bars: usize, slice: &[f64], name: &str) -> Result<usize, Error> {
  let expected = groups
    .checked_mul(bars)
    .ok_or_else(|| Error::InvalidParameter(format!("groups * bars overflows usize for {name}")))?;
  if slice.len() != expected {
    return Err(Error::LengthMismatch(slice.len(), expected));
  }
  Ok(expected)
}

#[inline]
fn assign_avg_ranks(vals: &[f64], order: &mut Vec<usize>, ranks: &mut Vec<f64>) {
  let n = vals.len();
  order.clear();
  order.extend(0..n);
  order.sort_unstable_by(|&a, &b| vals[a].partial_cmp(&vals[b]).unwrap_or(Ordering::Equal));
  ranks.resize(n, 0.0);

  let mut s = 0usize;
  while s < n {
    let v0 = vals[order[s]];
    let mut e = s + 1;
    while e < n && vals[order[e]] == v0 {
      e += 1;
    }
    let avg_rank = 0.5 * ((s + 1 + e) as f64);
    for &idx in &order[s..e] {
      ranks[idx] = avg_rank;
    }
    s = e;
  }
}

#[inline]
fn pearson_slice(xs: &[f64], ys: &[f64], min_obs: usize) -> f64 {
  let n = xs.len();
  if n < min_obs.max(2) {
    return f64::NAN;
  }
  let mut sx = 0.0_f64;
  let mut sy = 0.0_f64;
  for i in 0..n {
    sx += xs[i];
    sy += ys[i];
  }
  let nf = n as f64;
  let xm = sx / nf;
  let ym = sy / nf;

  let mut num = 0.0_f64;
  let mut sxx = 0.0_f64;
  let mut syy = 0.0_f64;
  for i in 0..n {
    let dx = xs[i] - xm;
    let dy = ys[i] - ym;
    num += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  let den = (sxx * syy).sqrt();
  if den > 1e-12 {
    num / den
  } else {
    f64::NAN
  }
}

#[inline]
fn pearson_ranks(rx: &[f64], ry: &[f64], min_obs: usize) -> f64 {
  let n = rx.len();
  if n < min_obs.max(2) {
    return f64::NAN;
  }
  let mean = 0.5 * ((n as f64) + 1.0);
  let mut num = 0.0_f64;
  let mut sxx = 0.0_f64;
  let mut syy = 0.0_f64;
  for i in 0..n {
    let dx = rx[i] - mean;
    let dy = ry[i] - mean;
    num += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  let den = (sxx * syy).sqrt();
  if den > 1e-12 {
    num / den
  } else {
    f64::NAN
  }
}

#[derive(Default)]
struct IcScratch {
  xs: Vec<f64>,
  ys: Vec<f64>,
  order: Vec<usize>,
  rx: Vec<f64>,
  ry: Vec<f64>,
}

/// Cross-sectional Pearson correlation per bar across ``groups`` entities.
///
/// Computes correlation between ``x[:, t]`` and ``y[:, t + lag]`` for
/// ``t in 0..(bars - lag)`` on mutually finite pairs. Bars with fewer than
/// ``max(min_obs, 2)`` valid pairs or near-zero variance yield ``NaN``.
pub fn cc_corr(
  groups: usize,
  bars: usize,
  x: &[f64],
  y: &[f64],
  min_obs: usize,
  lag: usize,
) -> Result<Vec<f64>, Error> {
  validate_grid_len(groups, bars, x, "x")?;
  validate_grid_len(groups, bars, y, "y")?;
  if bars <= lag {
    return Ok(Vec::new());
  }
  let out_len = bars - lag;
  if groups == 0 {
    return Ok(vec![f64::NAN; out_len]);
  }

  let out: Vec<f64> = (0..out_len)
    .into_par_iter()
    .map_init(
      || (Vec::with_capacity(groups), Vec::with_capacity(groups)),
      |(xs, ys), t| {
        xs.clear();
        ys.clear();
        let t_y = t + lag;
        for g in 0..groups {
          let xv = x[g * bars + t];
          let yv = y[g * bars + t_y];
          if xv.is_finite() && yv.is_finite() {
            xs.push(xv);
            ys.push(yv);
          }
        }
        pearson_slice(xs, ys, min_obs)
      },
    )
    .collect();

  Ok(out)
}

/// Cross-sectional Spearman rank correlation per bar across ``groups`` entities.
///
/// Computes rank correlation with average tie-breaking between ``x[:, t]`` and
/// ``y[:, t + lag]`` for ``t in 0..(bars - lag)`` on mutually finite pairs.
pub fn cc_spearman(
  groups: usize,
  bars: usize,
  x: &[f64],
  y: &[f64],
  min_obs: usize,
  lag: usize,
) -> Result<Vec<f64>, Error> {
  validate_grid_len(groups, bars, x, "x")?;
  validate_grid_len(groups, bars, y, "y")?;
  if bars <= lag {
    return Ok(Vec::new());
  }
  let out_len = bars - lag;
  if groups == 0 {
    return Ok(vec![f64::NAN; out_len]);
  }

  let min_req = min_obs.max(2);
  let out: Vec<f64> = (0..out_len)
    .into_par_iter()
    .map_init(
      || IcScratch {
        xs: Vec::with_capacity(groups),
        ys: Vec::with_capacity(groups),
        order: Vec::with_capacity(groups),
        rx: Vec::with_capacity(groups),
        ry: Vec::with_capacity(groups),
      },
      |ws, t| {
        ws.xs.clear();
        ws.ys.clear();
        let t_y = t + lag;
        for g in 0..groups {
          let xv = x[g * bars + t];
          let yv = y[g * bars + t_y];
          if xv.is_finite() && yv.is_finite() {
            ws.xs.push(xv);
            ws.ys.push(yv);
          }
        }
        if ws.xs.len() < min_req {
          return f64::NAN;
        }
        assign_avg_ranks(&ws.xs, &mut ws.order, &mut ws.rx);
        assign_avg_ranks(&ws.ys, &mut ws.order, &mut ws.ry);
        pearson_ranks(&ws.rx, &ws.ry, min_obs)
      },
    )
    .collect();

  Ok(out)
}

/// Jointly compute daily cross-sectional Pearson IC and Spearman RankIC series.
///
/// Collects mutually finite pairs once per bar and computes both correlations.
pub fn cc_ic(
  groups: usize,
  bars: usize,
  x: &[f64],
  y: &[f64],
  min_obs: usize,
  lag: usize,
) -> Result<(Vec<f64>, Vec<f64>), Error> {
  validate_grid_len(groups, bars, x, "x")?;
  validate_grid_len(groups, bars, y, "y")?;
  if bars <= lag {
    return Ok((Vec::new(), Vec::new()));
  }
  let out_len = bars - lag;
  if groups == 0 {
    return Ok((vec![f64::NAN; out_len], vec![f64::NAN; out_len]));
  }

  let min_req = min_obs.max(2);
  let pairs: Vec<(f64, f64)> = (0..out_len)
    .into_par_iter()
    .map_init(
      || IcScratch {
        xs: Vec::with_capacity(groups),
        ys: Vec::with_capacity(groups),
        order: Vec::with_capacity(groups),
        rx: Vec::with_capacity(groups),
        ry: Vec::with_capacity(groups),
      },
      |ws, t| {
        ws.xs.clear();
        ws.ys.clear();
        let t_y = t + lag;
        for g in 0..groups {
          let xv = x[g * bars + t];
          let yv = y[g * bars + t_y];
          if xv.is_finite() && yv.is_finite() {
            ws.xs.push(xv);
            ws.ys.push(yv);
          }
        }
        if ws.xs.len() < min_req {
          return (f64::NAN, f64::NAN);
        }
        let ic = pearson_slice(&ws.xs, &ws.ys, min_obs);
        assign_avg_ranks(&ws.xs, &mut ws.order, &mut ws.rx);
        assign_avg_ranks(&ws.ys, &mut ws.order, &mut ws.ry);
        let ric = pearson_ranks(&ws.rx, &ws.ry, min_obs);
        (ic, ric)
      },
    )
    .collect();

  let mut ic_out = Vec::with_capacity(out_len);
  let mut ric_out = Vec::with_capacity(out_len);
  for (ic, ric) in pairs {
    ic_out.push(ic);
    ric_out.push(ric);
  }
  Ok((ic_out, ric_out))
}

struct MatrixAccum {
  groups: usize,
  n_words: usize,
  mask_bits: Vec<Vec<u64>>,
  zrank: Vec<Vec<f64>>,
  valid_ok: Vec<bool>,
  vals: Vec<f64>,
  idxs: Vec<usize>,
  order: Vec<usize>,
  ranks: Vec<f64>,
  px: Vec<f64>,
  py: Vec<f64>,
  rx: Vec<f64>,
  ry: Vec<f64>,
  sum: Vec<f64>,
  cnt: Vec<usize>,
}

impl MatrixAccum {
  fn new(nf: usize, groups: usize) -> Self {
    let n_words = groups.div_ceil(64);
    MatrixAccum {
      groups,
      n_words,
      mask_bits: vec![vec![0u64; n_words]; nf],
      zrank: vec![vec![0.0_f64; groups]; nf],
      valid_ok: vec![false; nf],
      vals: Vec::with_capacity(groups),
      idxs: Vec::with_capacity(groups),
      order: Vec::with_capacity(groups),
      ranks: Vec::with_capacity(groups),
      px: Vec::with_capacity(groups),
      py: Vec::with_capacity(groups),
      rx: Vec::with_capacity(groups),
      ry: Vec::with_capacity(groups),
      sum: vec![0.0_f64; nf * nf],
      cnt: vec![0usize; nf * nf],
    }
  }
}

/// Pairwise time-averaged cross-sectional Spearman rank correlation matrix.
///
/// Uses exact bitset mask equality per bar: factor pairs with identical finite
/// masks on bar ``t`` share pre-ranked normalized vectors via a dot product,
/// while differing masks fall back to exact pairwise ranking on common finite pairs.
/// Returns a row-major ``(nf, nf)`` flattened matrix with ``1.0`` on the diagonal.
pub fn cc_spearman_matrix(
  groups: usize,
  bars: usize,
  factors: &[&[f64]],
  min_obs: usize,
) -> Result<Vec<f64>, Error> {
  let nf = factors.len();
  for (i, f) in factors.iter().enumerate() {
    validate_grid_len(groups, bars, f, &format!("factors[{i}]"))?;
  }
  if nf == 0 {
    return Ok(Vec::new());
  }

  let mut out = vec![f64::NAN; nf * nf];
  for i in 0..nf {
    out[i * nf + i] = 1.0;
  }
  if groups == 0 || bars == 0 {
    return Ok(out);
  }

  let min_req = min_obs.max(2);
  let (sum_all, cnt_all) = (0..bars)
    .into_par_iter()
    .fold(
      || MatrixAccum::new(nf, groups),
      |mut ws, t| {
        for f in 0..nf {
          ws.mask_bits[f].fill(0);
          ws.zrank[f].fill(0.0);
          ws.vals.clear();
          ws.idxs.clear();

          let f_slice = factors[f];
          for g in 0..ws.groups {
            let v = f_slice[g * bars + t];
            if v.is_finite() {
              ws.mask_bits[f][g >> 6] |= 1u64 << (g & 63);
              ws.vals.push(v);
              ws.idxs.push(g);
            }
          }

          let n = ws.vals.len();
          if n < min_req {
            ws.valid_ok[f] = false;
            continue;
          }

          assign_avg_ranks(&ws.vals, &mut ws.order, &mut ws.ranks);
          let mean = 0.5 * ((n as f64) + 1.0);
          let mut sxx = 0.0_f64;
          for &rk in &ws.ranks {
            let d = rk - mean;
            sxx += d * d;
          }
          let norm = sxx.sqrt();
          if norm <= 1e-12 {
            ws.valid_ok[f] = false;
            continue;
          }
          let inv_norm = 1.0 / norm;
          ws.valid_ok[f] = true;
          for (k, &g) in ws.idxs.iter().enumerate() {
            ws.zrank[f][g] = (ws.ranks[k] - mean) * inv_norm;
          }
        }

        for i in 0..nf {
          for j in (i + 1)..nf {
            let same_mask = ws.mask_bits[i] == ws.mask_bits[j];
            let corr = if same_mask {
              if !ws.valid_ok[i] || !ws.valid_ok[j] {
                f64::NAN
              } else {
                let zi = &ws.zrank[i];
                let zj = &ws.zrank[j];
                let mut dot = 0.0_f64;
                for w in 0..ws.n_words {
                  let mut bits = ws.mask_bits[i][w];
                  let base = w << 6;
                  while bits != 0 {
                    let tz = bits.trailing_zeros() as usize;
                    let g = base + tz;
                    dot += zi[g] * zj[g];
                    bits &= bits - 1;
                  }
                }
                dot.clamp(-1.0, 1.0)
              }
            } else {
              ws.px.clear();
              ws.py.clear();
              let fi = factors[i];
              let fj = factors[j];
              for g in 0..ws.groups {
                let vi = fi[g * bars + t];
                let vj = fj[g * bars + t];
                if vi.is_finite() && vj.is_finite() {
                  ws.px.push(vi);
                  ws.py.push(vj);
                }
              }
              if ws.px.len() < min_req {
                f64::NAN
              } else {
                assign_avg_ranks(&ws.px, &mut ws.order, &mut ws.rx);
                assign_avg_ranks(&ws.py, &mut ws.order, &mut ws.ry);
                pearson_ranks(&ws.rx, &ws.ry, min_obs)
              }
            };

            if corr.is_finite() {
              let idx = i * nf + j;
              ws.sum[idx] += corr;
              ws.cnt[idx] += 1;
            }
          }
        }
        ws
      },
    )
    .map(|ws| (ws.sum, ws.cnt))
    .reduce(
      || (vec![0.0_f64; nf * nf], vec![0usize; nf * nf]),
      |(mut s1, mut c1), (s2, c2)| {
        for k in 0..s1.len() {
          s1[k] += s2[k];
          c1[k] += c2[k];
        }
        (s1, c1)
      },
    );

  for i in 0..nf {
    for j in (i + 1)..nf {
      let idx = i * nf + j;
      let val = if cnt_all[idx] > 0 {
        sum_all[idx] / (cnt_all[idx] as f64)
      } else {
        f64::NAN
      };
      out[idx] = val;
      out[j * nf + i] = val;
    }
  }

  Ok(out)
}

#[inline]
fn decode_bucket(b_val: f64, n_buckets: usize) -> Option<usize> {
  if b_val.is_finite() && b_val >= 0.0 {
    let b = b_val as usize;
    if (b_val - (b as f64)) == 0.0 && b < n_buckets {
      return Some(b);
    }
  }
  None
}

fn tranche_returns_typed<B: Copy + Send + Sync>(
  groups: usize,
  bars: usize,
  bin_id_t: &[B],
  invalid_sentinel: B,
  to_usize: impl Fn(B) -> usize + Sync,
  step_first: &[f64],
  step_cont: &[f64],
  periods: &[usize],
  n_buckets: usize,
  entry_lag: usize,
) -> Vec<f64>
where
  B: PartialEq,
{
  let n_per = periods.len();
  let stride_b = n_buckets + 1;
  let mut out = vec![f64::NAN; n_per * stride_b * bars];
  let max_p = *periods.iter().max().unwrap();

  let mut periods_by_age: Vec<Vec<usize>> = vec![Vec::new(); max_p];
  for (pi, &p) in periods.iter().enumerate() {
    periods_by_age[p - 1].push(pi);
  }

  let same_step = std::ptr::eq(step_first.as_ptr(), step_cont.as_ptr());
  let out_ptr = UnsafeF64Ptr::new(out.as_mut_ptr(), out.len());

  (0..bars)
    .into_par_iter()
    .map_init(
      || {
        (
          Vec::<(usize, f64)>::with_capacity(groups),
          Vec::<(usize, f64)>::with_capacity(if same_step { 0 } else { groups }),
          vec![0.0_f64; stride_b],
          vec![0u32; stride_b],
          vec![0.0_f64; n_buckets],
          vec![0u32; n_buckets],
        )
      },
      |(valid_cont, valid_first, acc_sum, acc_cnt, sum_b, cnt_b), t| {
        valid_cont.clear();
        for g in 0..groups {
          let rv = step_cont[g * bars + t];
          if rv.is_finite() {
            valid_cont.push((g, rv));
          }
        }
        if !same_step {
          valid_first.clear();
          for g in 0..groups {
            let rv = step_first[g * bars + t];
            if rv.is_finite() {
              valid_first.push((g, rv));
            }
          }
        }

        acc_sum.fill(0.0);
        acc_cnt.fill(0);
        let out_slice = out_ptr.get();

        for age in 0..max_p {
          let shift = entry_lag.saturating_add(age);
          if t < shift {
            for rem_age in age..max_p {
              for &pi in &periods_by_age[rem_age] {
                for b in 0..stride_b {
                  if acc_cnt[b] > 0 {
                    out_slice[(pi * stride_b + b) * bars + t] =
                      acc_sum[b] / (acc_cnt[b] as f64);
                  }
                }
              }
            }
            break;
          }

          let s = t - shift;
          let b_row = &bin_id_t[s * groups..(s + 1) * groups];
          let r_pairs = if age == 0 && !same_step {
            valid_first.as_slice()
          } else {
            valid_cont.as_slice()
          };

          sum_b.fill(0.0);
          cnt_b.fill(0);
          for &(g, rv) in r_pairs {
            let b_raw = b_row[g];
            if b_raw != invalid_sentinel {
              let b = to_usize(b_raw);
              sum_b[b] += rv;
              cnt_b[b] += 1;
            }
          }

          let mut mkt_s = 0.0_f64;
          let mut mkt_n = 0u32;
          for b in 0..n_buckets {
            let c = cnt_b[b];
            if c > 0 {
              let sb = sum_b[b];
              acc_sum[b] += sb / (c as f64);
              acc_cnt[b] += 1;
              mkt_s += sb;
              mkt_n += c;
            }
          }
          if mkt_n > 0 {
            acc_sum[n_buckets] += mkt_s / (mkt_n as f64);
            acc_cnt[n_buckets] += 1;
          }

          for &pi in &periods_by_age[age] {
            for b in 0..stride_b {
              if acc_cnt[b] > 0 {
                out_slice[(pi * stride_b + b) * bars + t] = acc_sum[b] / (acc_cnt[b] as f64);
              }
            }
          }
        }
      },
    )
    .count();

  out
}

/// Compute overlapping equal-weight tranche daily returns across all buckets and periods in one pass.
///
/// Output is flattened row-major with logical shape ``(periods.len(), n_buckets + 1, bars)``
/// where bucket index ``n_buckets`` is the equal-weight union ("Market") across all
/// valid binned entities ``0..n_buckets``.
#[allow(clippy::too_many_arguments)]
pub fn tranche_returns(
  groups: usize,
  bars: usize,
  bins: &[f64],
  entry_mask: Option<&[bool]>,
  step_first: &[f64],
  step_cont: &[f64],
  periods: &[usize],
  n_buckets: usize,
  entry_lag: usize,
) -> Result<Vec<f64>, Error> {
  if n_buckets == 0 || n_buckets >= (u16::MAX as usize) {
    return Err(Error::InvalidParameter(format!(
      "n_buckets must be in 1..{}, got {n_buckets}",
      u16::MAX
    )));
  }
  if periods.is_empty() || periods.iter().any(|&p| p == 0) {
    return Err(Error::InvalidParameter(format!(
      "periods must be non-empty positive integers, got {periods:?}"
    )));
  }
  let total = validate_grid_len(groups, bars, bins, "bins")?;
  validate_grid_len(groups, bars, step_first, "step_first")?;
  validate_grid_len(groups, bars, step_cont, "step_cont")?;
  if let Some(em) = entry_mask {
    if em.len() != total {
      return Err(Error::LengthMismatch(em.len(), total));
    }
  }

  let stride_b = n_buckets + 1;
  if groups == 0 || bars == 0 {
    return Ok(vec![f64::NAN; periods.len() * stride_b * bars]);
  }

  if n_buckets <= (u8::MAX as usize) {
    let mut bin_id_t = vec![u8::MAX; total];
    bin_id_t
      .par_chunks_mut(groups)
      .enumerate()
      .for_each(|(s, row)| {
        if let Some(em) = entry_mask {
          for g in 0..groups {
            let idx = g * bars + s;
            if em[idx] {
              if let Some(b) = decode_bucket(bins[idx], n_buckets) {
                row[g] = b as u8;
              }
            }
          }
        } else {
          for g in 0..groups {
            if let Some(b) = decode_bucket(bins[g * bars + s], n_buckets) {
              row[g] = b as u8;
            }
          }
        }
      });

    Ok(tranche_returns_typed(
      groups,
      bars,
      &bin_id_t,
      u8::MAX,
      |b| b as usize,
      step_first,
      step_cont,
      periods,
      n_buckets,
      entry_lag,
    ))
  } else {
    let mut bin_id_t = vec![u16::MAX; total];
    bin_id_t
      .par_chunks_mut(groups)
      .enumerate()
      .for_each(|(s, row)| {
        if let Some(em) = entry_mask {
          for g in 0..groups {
            let idx = g * bars + s;
            if em[idx] {
              if let Some(b) = decode_bucket(bins[idx], n_buckets) {
                row[g] = b as u16;
              }
            }
          }
        } else {
          for g in 0..groups {
            if let Some(b) = decode_bucket(bins[g * bars + s], n_buckets) {
              row[g] = b as u16;
            }
          }
        }
      });

    Ok(tranche_returns_typed(
      groups,
      bars,
      &bin_id_t,
      u16::MAX,
      |b| b as usize,
      step_first,
      step_cont,
      periods,
      n_buckets,
      entry_lag,
    ))
  }
}

/// Compute cross-sectional mean of ``values`` per bucket ``0..n_buckets`` and overall union.
///
/// Output is flattened row-major with logical shape ``(n_buckets + 1, bars)`` where
/// row ``n_buckets`` is the mean across all entities with valid bucket ``0..n_buckets``
/// and finite ``values``. Empty cells produce ``NaN``.
pub fn bucket_means(
  groups: usize,
  bars: usize,
  bins: &[f64],
  values: &[f64],
  n_buckets: usize,
) -> Result<Vec<f64>, Error> {
  if n_buckets == 0 {
    return Err(Error::InvalidParameter(
      "n_buckets must be >= 1".to_string(),
    ));
  }
  validate_grid_len(groups, bars, bins, "bins")?;
  validate_grid_len(groups, bars, values, "values")?;

  let stride_b = n_buckets + 1;
  let mut out = vec![f64::NAN; stride_b * bars];
  if groups == 0 || bars == 0 {
    return Ok(out);
  }

  let out_ptr = UnsafeF64Ptr::new(out.as_mut_ptr(), out.len());
  (0..bars)
    .into_par_iter()
    .map_init(
      || (vec![0.0_f64; n_buckets], vec![0u32; n_buckets]),
      |(sum_b, cnt_b), t| {
        sum_b.fill(0.0);
        cnt_b.fill(0);
        for g in 0..groups {
          let idx = g * bars + t;
          let v = values[idx];
          if v.is_finite() {
            if let Some(b) = decode_bucket(bins[idx], n_buckets) {
              sum_b[b] += v;
              cnt_b[b] += 1;
            }
          }
        }
        let out_slice = out_ptr.get();
        let mut all_s = 0.0_f64;
        let mut all_c = 0u32;
        for b in 0..n_buckets {
          let c = cnt_b[b];
          if c > 0 {
            let sb = sum_b[b];
            out_slice[b * bars + t] = sb / (c as f64);
            all_s += sb;
            all_c += c;
          }
        }
        if all_c > 0 {
          out_slice[n_buckets * bars + t] = all_s / (all_c as f64);
        }
      },
    )
    .count();

  Ok(out)
}

/// Compute average one-way membership turnover ``1 - |A & B| / |B|`` per bucket at ``lag``.
///
/// Mirrors ``_bucket_turnover_lag`` across all ``0..n_buckets`` simultaneously in a single pass.
pub fn bucket_turnover(
  groups: usize,
  bars: usize,
  bins: &[f64],
  entry_mask: Option<&[bool]>,
  n_buckets: usize,
  lag: usize,
) -> Result<Vec<f64>, Error> {
  if n_buckets == 0 {
    return Err(Error::InvalidParameter(
      "n_buckets must be >= 1".to_string(),
    ));
  }
  let total = validate_grid_len(groups, bars, bins, "bins")?;
  if let Some(em) = entry_mask {
    if em.len() != total {
      return Err(Error::LengthMismatch(em.len(), total));
    }
  }
  if lag == 0 || bars <= lag || groups == 0 {
    return Ok(vec![0.0_f64; n_buckets]);
  }

  let (turn_sum, valid_cnt) = (lag..bars)
    .into_par_iter()
    .fold(
      || {
        (
          vec![0u32; n_buckets],
          vec![0u32; n_buckets],
          vec![0u32; n_buckets],
          vec![0.0_f64; n_buckets],
          vec![0usize; n_buckets],
        )
      },
      |(mut prev_c, mut curr_c, mut both_c, mut t_sum, mut v_cnt), t| {
        prev_c.fill(0);
        curr_c.fill(0);
        both_c.fill(0);
        let t_prev = t - lag;

        if let Some(em) = entry_mask {
          for g in 0..groups {
            let i_p = g * bars + t_prev;
            let i_c = g * bars + t;
            let bp = if em[i_p] {
              decode_bucket(bins[i_p], n_buckets)
            } else {
              None
            };
            let bc = if em[i_c] {
              decode_bucket(bins[i_c], n_buckets)
            } else {
              None
            };
            if let Some(b) = bp {
              prev_c[b] += 1;
            }
            if let Some(b) = bc {
              curr_c[b] += 1;
              if bp == Some(b) {
                both_c[b] += 1;
              }
            }
          }
        } else {
          for g in 0..groups {
            let bp = decode_bucket(bins[g * bars + t_prev], n_buckets);
            let bc = decode_bucket(bins[g * bars + t], n_buckets);
            if let Some(b) = bp {
              prev_c[b] += 1;
            }
            if let Some(b) = bc {
              curr_c[b] += 1;
              if bp == Some(b) {
                both_c[b] += 1;
              }
            }
          }
        }

        for b in 0..n_buckets {
          if prev_c[b] > 0 && curr_c[b] > 0 {
            let turn = (1.0 - (both_c[b] as f64) / (curr_c[b] as f64)).clamp(0.0, 1.0);
            t_sum[b] += turn;
            v_cnt[b] += 1;
          }
        }
        (prev_c, curr_c, both_c, t_sum, v_cnt)
      },
    )
    .map(|(_, _, _, t_sum, v_cnt)| (t_sum, v_cnt))
    .reduce(
      || (vec![0.0_f64; n_buckets], vec![0usize; n_buckets]),
      |(mut s1, mut c1), (s2, c2)| {
        for b in 0..n_buckets {
          s1[b] += s2[b];
          c1[b] += c2[b];
        }
        (s1, c1)
      },
    );

  let mut out = vec![0.0_f64; n_buckets];
  for b in 0..n_buckets {
    if valid_cnt[b] > 0 {
      out[b] = turn_sum[b] / (valid_cnt[b] as f64);
    }
  }
  Ok(out)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::algo::assert_vec_eq_nan;

  #[test]
  fn test_cc_corr_spearman_ic_and_lag() {
    // 4 stocks x 3 bars
    let x = vec![
      1.0, 4.0, 1.0, //
      2.0, 3.0, 2.0, //
      3.0, 2.0, 4.0, //
      4.0, 1.0, 3.0, //
    ];
    let y = vec![
      10.0, 1.0, 2.0, //
      20.0, 2.0, 4.0, //
      30.0, 3.0, 6.0, //
      40.0, 4.0, 8.0, //
    ];
    let (ic, ric) = cc_ic(4, 3, &x, &y, 2, 0).unwrap();
    assert_vec_eq_nan(&ic, &[1.0, -1.0, 0.8]);
    assert_vec_eq_nan(&ric, &[1.0, -1.0, 0.8]);

    let corr_lag1 = cc_corr(4, 3, &x, &y, 2, 1).unwrap();
    let spear_lag1 = cc_spearman(4, 3, &x, &y, 2, 1).unwrap();
    // t=0 x=[1,2,3,4], y(t=1)=[1,2,3,4] -> 1.0
    // t=1 x=[4,3,2,1], y(t=2)=[2,4,6,8] -> -1.0
    assert_vec_eq_nan(&corr_lag1, &[1.0, -1.0]);
    assert_vec_eq_nan(&spear_lag1, &[1.0, -1.0]);

    assert!(cc_corr(4, 3, &x, &y, 2, 3).unwrap().is_empty());
  }

  #[test]
  fn test_cc_spearman_ties_and_nan() {
    // 4 stocks x 2 bars, bar 0 has ties, bar 1 has NaN
    let x = vec![
      1.0, 10.0, //
      1.0, f64::NAN, //
      2.0, 20.0, //
      3.0, 30.0, //
    ];
    let y = vec![
      1.0, 100.0, //
      2.0, 200.0, //
      3.0, 300.0, //
      4.0, 400.0, //
    ];
    let ric = cc_spearman(4, 2, &x, &y, 3, 0).unwrap();
    // bar 0: rx=[1.5, 1.5, 3, 4], ry=[1, 2, 3, 4] -> 0.948683298...
    // bar 1: 3 valid pairs monotonically increasing -> 1.0
    assert!((ric[0] - 0.9486832980505138).abs() < 1e-12);
    assert!((ric[1] - 1.0).abs() < 1e-12);
  }

  #[test]
  fn test_cc_spearman_matrix_mixed_masks() {
    // 4 stocks x 2 bars, f0 and f1 share mask on bar 0, differ on bar 1
    let f0 = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
    let f1 = vec![2.0, 4.0, 4.0, f64::NAN, 6.0, 2.0, 8.0, 1.0];
    let f2 = vec![4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0];
    let mat = cc_spearman_matrix(4, 2, &[&f0, &f1, &f2], 3).unwrap();
    // f0 vs f2 = -1.0
    assert!((mat[0] - 1.0).abs() < 1e-12);
    assert!((mat[4] - 1.0).abs() < 1e-12);
    assert!((mat[8] - 1.0).abs() < 1e-12);
    assert!((mat[2] + 1.0).abs() < 1e-12);
    assert!((mat[6] + 1.0).abs() < 1e-12);
    // f0 vs f1: bar 0 = +1.0, bar 1 = -1.0 -> mean = 0.0
    assert!(mat[1].abs() < 1e-12);
    assert!(mat[3].abs() < 1e-12);
  }

  #[test]
  fn test_tranche_returns_and_bucket_helpers() {
    // 2 stocks x 5 bars, 2 buckets (0, 1), periods=[1, 2], entry_lag=1
    let bins = vec![
      0.0, 0.0, 1.0, 0.0, 1.0, //
      1.0, 1.0, 0.0, 1.0, 0.0, //
    ];
    let step = vec![
      0.01, 0.02, 0.03, 0.04, 0.05, //
      0.10, 0.20, 0.30, 0.40, 0.50, //
    ];
    let res = tranche_returns(2, 5, &bins, None, &step, &step, &[1, 2], 2, 1).unwrap();
    // Shape: (2, 3, 5). For period=1:
    // t=0: NaN
    // t=1 (s=0: g0 in b0, g1 in b1): b0=0.02, b1=0.20, mkt=0.11
    // t=2 (s=1: g0 in b0, g1 in b1): b0=0.03, b1=0.30, mkt=0.165
    // t=3 (s=2: g0 in b1, g1 in b0): b0=0.40, b1=0.04, mkt=0.22
    // t=4 (s=3: g0 in b0, g1 in b1): b0=0.05, b1=0.50, mkt=0.275
    assert_vec_eq_nan(&res[0..5], &[f64::NAN, 0.02, 0.03, 0.40, 0.05]);
    assert_vec_eq_nan(&res[5..10], &[f64::NAN, 0.20, 0.30, 0.04, 0.50]);
    assert_vec_eq_nan(&res[10..15], &[f64::NAN, 0.11, 0.165, 0.22, 0.275]);
    // For period=2:
    // t=1: only age=0 -> b0=0.02
    // t=2: age0(s=1,g0)=0.03, age1(s=0,g0)=0.03 -> 0.03
    // t=3: age0(s=2,g1)=0.40, age1(s=1,g0)=0.04 -> avg=0.22
    // t=4: age0(s=3,g0)=0.05, age1(s=2,g1)=0.50 -> avg=0.275
    assert_vec_eq_nan(&res[15..20], &[f64::NAN, 0.02, 0.03, 0.22, 0.275]);

    let means = bucket_means(2, 5, &bins, &step, 2).unwrap();
    assert_vec_eq_nan(&means[0..5], &[0.01, 0.02, 0.30, 0.04, 0.50]);
    assert_vec_eq_nan(&means[5..10], &[0.10, 0.20, 0.03, 0.40, 0.05]);
    assert_vec_eq_nan(&means[10..15], &[0.055, 0.11, 0.165, 0.22, 0.275]);

    // Turnover lag=1: t=1 same (0), t=2 flipped (1), t=3 flipped (1), t=4 flipped (1) -> 3/4 = 0.75
    let turn = bucket_turnover(2, 5, &bins, None, 2, 1).unwrap();
    assert_vec_eq_nan(&turn, &[0.75, 0.75]);
  }
}
