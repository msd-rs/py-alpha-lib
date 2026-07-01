// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal};

/// Forward split and dividend adjustment
///
/// Adjusts prices forward (from earliest to latest event) using a loop for precise calculation.
pub fn ta_fw_split<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  price: &[NumT],
  dividend: &[NumT],
  transfer_shares: &[NumT],
  right_shares: &[NumT],
  right_price: &[NumT],
) -> Result<(), Error> {
  if r.len() != price.len()
    || r.len() != dividend.len()
    || r.len() != transfer_shares.len()
    || r.len() != right_shares.len()
    || r.len() != right_price.len()
  {
    return Err(Error::LengthMismatch(r.len(), price.len()));
  }

  if r.is_empty() {
    return Ok(());
  }

  let groups = ctx.groups();
  let group_size = ctx.chunk_size(r.len());
  if r.len() != group_size * groups {
    return Err(Error::LengthMismatch(r.len(), group_size * groups));
  }

  r.par_chunks_mut(group_size)
    .zip(price.par_chunks(group_size))
    .zip(dividend.par_chunks(group_size))
    .zip(transfer_shares.par_chunks(group_size))
    .zip(right_shares.par_chunks(group_size))
    .zip(right_price.par_chunks(group_size))
    .for_each(|(((((r, p), div), ts), rs), rp)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      if start >= end {
        return;
      }

      r[start..end].copy_from_slice(&p[start..end]);

      for t in start + 1..end {
        let d = if is_normal(&div[t]) {
          div[t]
        } else {
          NumT::zero()
        };
        let ts_val = if is_normal(&ts[t]) {
          ts[t]
        } else {
          NumT::zero()
        };
        let rs_val = if is_normal(&rs[t]) {
          rs[t]
        } else {
          NumT::zero()
        };
        let rp_val = if is_normal(&rp[t]) {
          rp[t]
        } else {
          NumT::zero()
        };

        if d != NumT::zero() || ts_val != NumT::zero() || rs_val != NumT::zero() {
          for j in start..t {
            if is_normal(&r[j]) {
              r[j] = (r[j] - d + rs_val * rp_val) / (NumT::one() + ts_val + rs_val);
            }
          }
        }
      }
    });

  Ok(())
}

/// Backward split and dividend adjustment
///
/// Adjusts prices backward (from latest to earliest event) using a loop for precise calculation.
pub fn ta_bw_split<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  price: &[NumT],
  dividend: &[NumT],
  transfer_shares: &[NumT],
  right_shares: &[NumT],
  right_price: &[NumT],
) -> Result<(), Error> {
  if r.len() != price.len()
    || r.len() != dividend.len()
    || r.len() != transfer_shares.len()
    || r.len() != right_shares.len()
    || r.len() != right_price.len()
  {
    return Err(Error::LengthMismatch(r.len(), price.len()));
  }

  if r.is_empty() {
    return Ok(());
  }

  let groups = ctx.groups();
  let group_size = ctx.chunk_size(r.len());
  if r.len() != group_size * groups {
    return Err(Error::LengthMismatch(r.len(), group_size * groups));
  }

  r.par_chunks_mut(group_size)
    .zip(price.par_chunks(group_size))
    .zip(dividend.par_chunks(group_size))
    .zip(transfer_shares.par_chunks(group_size))
    .zip(right_shares.par_chunks(group_size))
    .zip(right_price.par_chunks(group_size))
    .for_each(|(((((r, p), div), ts), rs), rp)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      if start >= end {
        return;
      }

      r[start..end].copy_from_slice(&p[start..end]);

      for t in (start + 1..end).rev() {
        let d = if is_normal(&div[t]) {
          div[t]
        } else {
          NumT::zero()
        };
        let ts_val = if is_normal(&ts[t]) {
          ts[t]
        } else {
          NumT::zero()
        };
        let rs_val = if is_normal(&rs[t]) {
          rs[t]
        } else {
          NumT::zero()
        };
        let rp_val = if is_normal(&rp[t]) {
          rp[t]
        } else {
          NumT::zero()
        };

        if d != NumT::zero() || ts_val != NumT::zero() || rs_val != NumT::zero() {
          for j in t..end {
            if is_normal(&r[j]) {
              r[j] = r[j] * (NumT::one() + ts_val + rs_val) + d - rs_val * rp_val;
            }
          }
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
  fn test_ta_fw_split_only_dividend() {
    let price = vec![10.0, 10.0, 10.0, 10.0];
    let dividend = vec![0.0, 1.0, 1.0, 0.0];
    let transfer_shares = vec![0.0; 4];
    let right_shares = vec![0.0; 4];
    let right_price = vec![0.0; 4];

    let mut r = vec![0.0; 4];
    let ctx = Context::default();

    ta_fw_split(
      &ctx,
      &mut r,
      &price,
      &dividend,
      &transfer_shares,
      &right_shares,
      &right_price,
    )
    .unwrap();

    assert_vec_eq_nan(&r, &vec![8.0, 9.0, 10.0, 10.0]);
  }

  #[test]
  fn test_ta_bw_split() {
    let price = vec![10.0, 10.0, 10.0];
    let dividend = vec![0.0, 0.0, 1.0];
    let transfer_shares = vec![0.0, 1.0, 0.0];
    let right_shares = vec![0.0; 3];
    let right_price = vec![0.0; 3];

    let mut r = vec![0.0; 3];
    let ctx = Context::default();

    ta_bw_split(
      &ctx,
      &mut r,
      &price,
      &dividend,
      &transfer_shares,
      &right_shares,
      &right_price,
    )
    .unwrap();

    // Order:
    // t=2: D_2=1.0 -> P_2 = 10.0 + 1.0 = 11.0
    // t=1: T_1=1.0 -> P_1 = 10.0 * 2 = 20.0, P_2 = 11.0 * 2 = 22.0
    // Result should be: [10.0, 20.0, 22.0]
    assert_vec_eq_nan(&r, &vec![10.0, 20.0, 22.0]);
  }

  #[test]
  fn test_ta_fw_split() {
    let price = vec![10.0, 10.0, 10.0];
    let dividend = vec![0.0, 0.0, 1.0];
    let transfer_shares = vec![0.0, 1.0, 0.0];
    let right_shares = vec![0.0; 3];
    let right_price = vec![0.0; 3];

    let mut r = vec![0.0; 3];
    let ctx = Context::default();

    ta_fw_split(
      &ctx,
      &mut r,
      &price,
      &dividend,
      &transfer_shares,
      &right_shares,
      &right_price,
    )
    .unwrap();

    // Order:
    // t=1: T_1=1.0 -> P_0 / 2 = 5.0
    // t=2: D_2=1.0 -> P_0 = 5.0 - 1.0 = 4.0, P_1 = 10.0 - 1.0 = 9.0
    // Result should be: [4.0, 9.0, 10.0]
    assert_vec_eq_nan(&r, &vec![4.0, 9.0, 10.0]);
  }

  #[test]
  fn test_ta_split_with_nan() {
    let price = vec![10.0, f64::NAN, 10.0];
    let dividend = vec![0.0, 0.0, 1.0];
    let transfer_shares = vec![0.0, 1.0, 0.0];
    let right_shares = vec![0.0; 3];
    let right_price = vec![0.0; 3];

    let mut r = vec![0.0; 3];
    let ctx = Context::default();

    ta_bw_split(
      &ctx,
      &mut r,
      &price,
      &dividend,
      &transfer_shares,
      &right_shares,
      &right_price,
    )
    .unwrap();
    assert_vec_eq_nan(&r, &vec![10.0, f64::NAN, 22.0]);

    ta_fw_split(
      &ctx,
      &mut r,
      &price,
      &dividend,
      &transfer_shares,
      &right_shares,
      &right_price,
    )
    .unwrap();
    assert_vec_eq_nan(&r, &vec![4.0, f64::NAN, 10.0]);
  }
}
