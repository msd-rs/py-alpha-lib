// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use std::{cmp::Ordering, fmt::Debug};

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error};

#[derive(Debug, Clone, Copy)]
struct UnsafePtr<NumT: Float> {
  ptr: *mut NumT,
  len: usize,
}

impl<NumT: Float> UnsafePtr<NumT> {
  fn new(ptr: *mut NumT, len: usize) -> Self {
    UnsafePtr { ptr, len }
  }

  #[allow(clippy::mut_from_ref)]
  fn get(&self) -> &mut [NumT] {
    unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
  }
}

unsafe impl<NumT: Float> Send for UnsafePtr<NumT> {}
unsafe impl<NumT: Float> Sync for UnsafePtr<NumT> {}

#[inline]
fn median_in_place(vals: &mut [f64]) -> f64 {
  let n = vals.len();
  if n == 0 {
    return f64::NAN;
  }
  if n == 1 {
    return vals[0];
  }
  let mid = n / 2;
  vals.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
  let hi = vals[mid];
  if n % 2 == 1 {
    hi
  } else {
    let lo = vals[..mid].iter().copied().fold(f64::NEG_INFINITY, f64::max);
    0.5 * (lo + hi)
  }
}

#[inline]
fn linear_quantile_in_place(vals: &mut [f64], p: f64) -> f64 {
  let n = vals.len();
  if n == 0 {
    return f64::NAN;
  }
  if n == 1 {
    return vals[0];
  }
  let pos = p * ((n - 1) as f64);
  let i0 = (pos.floor() as usize).min(n - 1);
  let i1 = (pos.ceil() as usize).min(n - 1);
  vals.select_nth_unstable_by(i0, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
  let v0 = vals[i0];
  if i1 == i0 {
    v0
  } else {
    let v1 = vals[i0 + 1..]
      .iter()
      .copied()
      .fold(f64::INFINITY, f64::min);
    v0 + (pos - (i0 as f64)) * (v1 - v0)
  }
}

/// Cross-sectional MAD winsorization across groups at each time step.
///
/// Clips finite values in each cross-section to
/// ``[median - 1.4826 * mad_k * MAD, median + 1.4826 * mad_k * MAD]``.
/// Zero-MAD cross-sections (``1.4826 * mad_k * MAD <= 1e-12``) preserve finite
/// values unchanged. Non-finite inputs (NaN/Inf) produce NaN output.
pub fn ta_cc_winsorize_mad<NumT: Float + Send + Sync + Debug>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  mad_k: NumT,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }
  if r.is_empty() {
    return Ok(());
  }

  let group_size = ctx.chunk_size(r.len());
  let groups = ctx.groups();
  if r.len() != group_size * groups {
    return Err(Error::LengthMismatch(r.len(), group_size * groups));
  }

  let k = mad_k.to_f64().unwrap_or(f64::NAN);
  let r_ptr = UnsafePtr::new(r.as_mut_ptr(), r.len());

  (0..group_size).into_par_iter().for_each(|j| {
    let r = r_ptr.get();
    let mut vals: Vec<f64> = Vec::with_capacity(groups);

    for i in 0..groups {
      let val = input[i * group_size + j];
      if val.is_finite() {
        if let Some(vf) = val.to_f64() {
          vals.push(vf);
        }
      }
    }

    if vals.is_empty() {
      for i in 0..groups {
        r[i * group_size + j] = NumT::nan();
      }
      return;
    }

    let med = median_in_place(&mut vals);
    for v in &mut vals {
      *v = (*v - med).abs();
    }
    let mad = median_in_place(&mut vals);
    let scaled = 1.4826 * mad * k;

    if med.is_finite() && scaled.is_finite() && scaled > 1e-12 {
      let lo = med - scaled;
      let hi = med + scaled;
      for i in 0..groups {
        let idx = i * group_size + j;
        let val = input[idx];
        if val.is_finite() {
          let vf = val.to_f64().unwrap();
          r[idx] = NumT::from(vf.clamp(lo, hi)).unwrap_or_else(NumT::nan);
        } else {
          r[idx] = NumT::nan();
        }
      }
    } else {
      for i in 0..groups {
        let idx = i * group_size + j;
        let val = input[idx];
        r[idx] = if val.is_finite() { val } else { NumT::nan() };
      }
    }
  });

  Ok(())
}

/// Cross-sectional standard-deviation (sigma) winsorization across groups at each time step.
///
/// Clips finite values in each cross-section to
/// ``[mean - sigma_k * std, mean + sigma_k * std]`` using population standard
/// deviation (``ddof = 0``). Degenerate cross-sections (``cnt <= 1`` or
/// ``std <= 1e-12``) preserve finite values unchanged. Non-finite inputs produce NaN.
pub fn ta_cc_winsorize_sigma<NumT: Float + Send + Sync + Debug>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  sigma_k: NumT,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }
  if r.is_empty() {
    return Ok(());
  }

  let group_size = ctx.chunk_size(r.len());
  let groups = ctx.groups();
  if r.len() != group_size * groups {
    return Err(Error::LengthMismatch(r.len(), group_size * groups));
  }

  let k = sigma_k.to_f64().unwrap_or(f64::NAN);
  let r_ptr = UnsafePtr::new(r.as_mut_ptr(), r.len());

  (0..group_size).into_par_iter().for_each(|j| {
    let r = r_ptr.get();
    let mut sum = 0.0_f64;
    let mut cnt = 0usize;

    for i in 0..groups {
      let val = input[i * group_size + j];
      if val.is_finite() {
        if let Some(vf) = val.to_f64() {
          sum += vf;
          cnt += 1;
        }
      }
    }

    if cnt == 0 {
      for i in 0..groups {
        r[i * group_size + j] = NumT::nan();
      }
      return;
    }

    let mean = sum / (cnt as f64);
    let mut ss = 0.0_f64;
    for i in 0..groups {
      let val = input[i * group_size + j];
      if val.is_finite() {
        let d = val.to_f64().unwrap() - mean;
        ss += d * d;
      }
    }
    let std = (ss / (cnt as f64)).sqrt();

    if cnt > 1 && mean.is_finite() && std.is_finite() && std > 1e-12 && k.is_finite() && k >= 0.0
    {
      let lo = mean - k * std;
      let hi = mean + k * std;
      for i in 0..groups {
        let idx = i * group_size + j;
        let val = input[idx];
        if val.is_finite() {
          let vf = val.to_f64().unwrap();
          r[idx] = NumT::from(vf.clamp(lo, hi)).unwrap_or_else(NumT::nan);
        } else {
          r[idx] = NumT::nan();
        }
      }
    } else {
      for i in 0..groups {
        let idx = i * group_size + j;
        let val = input[idx];
        r[idx] = if val.is_finite() { val } else { NumT::nan() };
      }
    }
  });

  Ok(())
}

/// Cross-sectional percentile winsorization across groups at each time step.
///
/// Clips finite values in each cross-section to the linear-interpolated
/// ``[pct_low, pct_high]`` quantiles (``0.0 <= pct_low < pct_high <= 1.0``).
/// Non-finite inputs (NaN/Inf) produce NaN output.
pub fn ta_cc_winsorize_pct<NumT: Float + Send + Sync + Debug>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  pct_low: NumT,
  pct_high: NumT,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }

  let lo_p = pct_low.to_f64().unwrap_or(f64::NAN);
  let hi_p = pct_high.to_f64().unwrap_or(f64::NAN);
  if !lo_p.is_finite() || !hi_p.is_finite() || lo_p < 0.0 || hi_p > 1.0 || lo_p >= hi_p {
    return Err(Error::InvalidParameter(format!(
      "invalid percentile bounds: pct_low={lo_p}, pct_high={hi_p}"
    )));
  }

  if r.is_empty() {
    return Ok(());
  }

  let group_size = ctx.chunk_size(r.len());
  let groups = ctx.groups();
  if r.len() != group_size * groups {
    return Err(Error::LengthMismatch(r.len(), group_size * groups));
  }

  let r_ptr = UnsafePtr::new(r.as_mut_ptr(), r.len());

  (0..group_size).into_par_iter().for_each(|j| {
    let r = r_ptr.get();
    let mut vals: Vec<f64> = Vec::with_capacity(groups);

    for i in 0..groups {
      let val = input[i * group_size + j];
      if val.is_finite() {
        if let Some(vf) = val.to_f64() {
          vals.push(vf);
        }
      }
    }

    if vals.is_empty() {
      for i in 0..groups {
        r[i * group_size + j] = NumT::nan();
      }
      return;
    }

    let lo = linear_quantile_in_place(&mut vals, lo_p);
    let hi = linear_quantile_in_place(&mut vals, hi_p);

    for i in 0..groups {
      let idx = i * group_size + j;
      let val = input[idx];
      if val.is_finite() {
        let vf = val.to_f64().unwrap();
        r[idx] = NumT::from(vf.clamp(lo, hi)).unwrap_or_else(NumT::nan);
      } else {
        r[idx] = NumT::nan();
      }
    }
  });

  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::algo::assert_vec_eq_nan;

  #[test]
  fn test_cc_winsorize_mad_basic() {
    // 5 groups x 2 bars:
    // bar 0: [1, 2, 3, 4, 100] -> med=3, dev=[2,1,0,1,97], mad=1, scaled=1.4826*1*3=4.4478
    // bar 1: [10, 10, 10, 10, 50] -> med=10, dev=[0,0,0,0,40], mad=0 -> unchanged!
    let ctx = Context::new(0, 5, 0);
    let input = vec![1.0, 10.0, 2.0, 10.0, 3.0, 10.0, 4.0, 10.0, 100.0, 50.0];
    let mut r = vec![0.0; input.len()];
    ta_cc_winsorize_mad(&ctx, &mut r, &input, 3.0).unwrap();

    let hi0 = 3.0 + 1.4826 * 3.0;
    assert_vec_eq_nan(
      &r,
      &[1.0, 10.0, 2.0, 10.0, 3.0, 10.0, 4.0, 10.0, hi0, 50.0],
    );
  }

  #[test]
  fn test_cc_winsorize_mad_nan_inf() {
    let ctx = Context::new(0, 5, 0);
    let input = vec![1.0, 2.0, 3.0, f64::NAN, f64::INFINITY];
    let mut r = vec![0.0; 5];
    ta_cc_winsorize_mad(&ctx, &mut r, &input, 1.0).unwrap();
    // finite=[1,2,3], med=2, mad=1, scaled=1.4826 -> [1,2,3,NaN,NaN]
    assert_vec_eq_nan(&r, &[1.0, 2.0, 3.0, f64::NAN, f64::NAN]);
  }

  #[test]
  fn test_cc_winsorize_sigma_basic() {
    // 4 groups x 1 bar: [0, 0, 0, 100] -> mean=25, pop std=sqrt((25^2*3+75^2)/4)=43.30127
    let ctx = Context::new(0, 4, 0);
    let input = vec![0.0, 0.0, 0.0, 100.0];
    let mut r = vec![0.0; 4];
    ta_cc_winsorize_sigma(&ctx, &mut r, &input, 1.0).unwrap();
    let mean = 25.0_f64;
    let std = (7500.0_f64 / 4.0).sqrt();
    assert_vec_eq_nan(&r, &[0.0, 0.0, 0.0, mean + std]);
  }

  #[test]
  fn test_cc_winsorize_pct_basic_and_errors() {
    let ctx = Context::new(0, 5, 0);
    let input = vec![0.0, 10.0, 20.0, 30.0, 100.0];
    let mut r = vec![0.0; 5];
    ta_cc_winsorize_pct(&ctx, &mut r, &input, 0.1, 0.9).unwrap();
    // pos_lo = 0.1*4 = 0.4 -> 4.0; pos_hi = 0.9*4 = 3.6 -> 30 + 0.6*(100-30) = 72.0
    assert_vec_eq_nan(&r, &[4.0, 10.0, 20.0, 30.0, 72.0]);

    assert!(ta_cc_winsorize_pct(&ctx, &mut r, &input, 0.9, 0.1).is_err());
    assert!(ta_cc_winsorize_pct(&ctx, &mut r, &input, -0.1, 0.9).is_err());
    assert!(ta_cc_winsorize_pct(&ctx, &mut r, &input, 0.1, 1.1).is_err());
  }

  #[test]
  fn test_cc_winsorize_empty_and_single_group() {
    let ctx1 = Context::new(0, 1, 0);
    let mut empty_r: Vec<f64> = Vec::new();
    let empty_in: Vec<f64> = Vec::new();
    ta_cc_winsorize_mad(&ctx1, &mut empty_r, &empty_in, 3.0).unwrap();
    ta_cc_winsorize_sigma(&ctx1, &mut empty_r, &empty_in, 3.0).unwrap();
    ta_cc_winsorize_pct(&ctx1, &mut empty_r, &empty_in, 0.01, 0.99).unwrap();

    let single_in = vec![5.0, f64::NAN, 8.0];
    let mut single_r = vec![0.0; 3];
    ta_cc_winsorize_mad(&ctx1, &mut single_r, &single_in, 3.0).unwrap();
    assert_vec_eq_nan(&single_r, &[5.0, f64::NAN, 8.0]);
  }
}
