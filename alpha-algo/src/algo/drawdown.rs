// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal};

/// Rolling Maximum Drawdown.
///
/// MaxDrawdown = minimum peak-to-trough decline within the rolling window.
/// Result is expressed as a negative return (e.g. -0.2 means 20% drawdown from peak).
/// Input should be a price or equity curve series.
///
/// Ref: https://en.wikipedia.org/wiki/Drawdown_(economics)
///
pub fn ta_max_drawdown<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }

  if periods < 1 {
    return Err(Error::InvalidPeriod(format!(
      "max_drawdown requires periods >= 1, got {}",
      periods
    )));
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      for i in start..end {
        let win_start = if i >= periods { i - periods + 1 } else { 0 };

        if ctx.is_strictly_cycle() && (i - start + 1) < periods {
          continue;
        }

        if ctx.is_skip_nan() {
          if !is_normal(&x[i]) {
            continue;
          }

          // Find first valid value as initial peak
          let mut j = win_start;
          while j <= i && !is_normal(&x[j]) {
            j += 1;
          }
          if j > i {
            continue;
          }

          let mut peak = x[j];
          let mut mdd = NumT::zero();

          for k in j..=i {
            let val = x[k];
            if !is_normal(&val) {
              continue;
            }
            if val > peak {
              peak = val;
            }
            let dd = (val - peak) / peak;
            if dd < mdd {
              mdd = dd;
            }
          }

          r[i] = mdd;
        } else {
          if !is_normal(&x[i]) {
            continue;
          }

          // Scan window; any NaN invalidates the whole window
          let mut has_nan = false;
          let mut peak = NumT::zero();
          let mut peak_set = false;
          let mut mdd = NumT::zero();

          for k in win_start..=i {
            let val = x[k];
            if !is_normal(&val) {
              has_nan = true;
              break;
            }
            if !peak_set {
              peak = val;
              peak_set = true;
            } else if val > peak {
              peak = val;
            }
            let dd = (val - peak) / peak;
            if dd < mdd {
              mdd = dd;
            }
          }

          if has_nan {
            continue;
          }

          r[i] = mdd;
        }
      }
    });

  Ok(())
}

#[cfg(test)]
mod tests {
  use crate::algo::{
    assert_vec_eq_nan,
    context::{FLAG_SKIP_NAN, FLAG_STRICTLY_CYCLE},
  };

  use super::*;

  #[test]
  fn test_ta_max_drawdown_basic() {
    // Prices: [10, 12, 15, 13, 9, 14, 16]
    let input = vec![10.0, 12.0, 15.0, 13.0, 9.0, 14.0, 16.0];
    let periods = 4;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_max_drawdown(&ctx, &mut r, &input, periods).unwrap();

    // i=0: [10] mdd=0
    // i=1: [10,12] mdd=0
    // i=2: [10,12,15] mdd=0
    // i=3: [10,12,15,13] peak=15 dd=(13-15)/15=-0.13333, mdd=-0.13333
    // i=4: [12,15,13,9] peak=15 dd=(9-15)/15=-0.4, mdd=-0.4
    // i=5: [15,13,9,14] peak=15 dd max is (9-15)/15=-0.4, mdd=-0.4
    // i=6: [13,9,14,16] peak=16 (at end) but drawdown at 9 from 13: (9-13)/13=-0.3077
    assert_vec_eq_nan(
      &r,
      &vec![0.0, 0.0, 0.0, -0.13333333, -0.4, -0.4, -0.307692],
    );
  }

  #[test]
  fn test_ta_max_drawdown_strictly_cycle() {
    let input = vec![10.0, 12.0, 15.0, 13.0, 9.0, 14.0, 16.0];
    let periods = 4;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, FLAG_STRICTLY_CYCLE);
    ta_max_drawdown(&ctx, &mut r, &input, periods).unwrap();
    assert_vec_eq_nan(
      &r,
      &vec![f64::NAN, f64::NAN, f64::NAN, -0.13333333, -0.4, -0.4, -0.307692],
    );
  }

  #[test]
  fn test_ta_max_drawdown_monotonic() {
    // Strictly increasing → no drawdown
    let input = vec![10.0, 12.0, 14.0, 16.0, 18.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_max_drawdown(&ctx, &mut r, &input, periods).unwrap();
    assert_vec_eq_nan(&r, &vec![0.0, 0.0, 0.0, 0.0, 0.0]);
  }

  #[test]
  fn test_ta_max_drawdown_decreasing() {
    // Strictly decreasing → max decline each step
    let input = vec![16.0, 14.0, 12.0, 10.0, 8.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_max_drawdown(&ctx, &mut r, &input, periods).unwrap();
    // i=0: [16] mdd=0
    // i=1: [16,14] mdd=(14-16)/16=-0.125
    // i=2: [16,14,12] peak=16 mdd=(12-16)/16=-0.25
    // i=3: [14,12,10] peak=14 mdd=(10-14)/14=-0.2857
    // i=4: [12,10,8] peak=12 mdd=(8-12)/12=-0.3333
    assert_vec_eq_nan(
      &r,
      &vec![0.0, -0.125, -0.25, -0.285714, -0.333333],
    );
  }

  #[test]
  fn test_ta_max_drawdown_skip_nan() {
    let input = vec![10.0, f64::NAN, 15.0, 13.0, 9.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);
    ta_max_drawdown(&ctx, &mut r, &input, periods).unwrap();
    // i=0: [10] mdd=0
    // i=1: NaN → skip
    // i=2: [10, NaN, 15] skip NaN. [10,15] peak=15, mdd=(10-15)/15=-0.333
    // i=3: [NaN, 15, 13] skip NaN. [15,13] peak=15, mdd=(13-15)/15=-0.1333
    // i=4: [15, 13, 9] peak=15, mdd=(9-15)/15=-0.4
    assert_vec_eq_nan(&r, &vec![0.0, f64::NAN, 0.0, -0.133333, -0.4]);
  }

  #[test]
  fn test_ta_max_drawdown_nan_break() {
    // Non-skip-nan: any NaN in window → NaN output
    let input = vec![10.0, f64::NAN, 15.0, 13.0, 9.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_max_drawdown(&ctx, &mut r, &input, periods).unwrap();
    // i=0: [10] mdd=0
    // i=1: NaN
    // i=2: [10,NaN,15] NaN
    // i=3: [NaN,15,13] NaN
    // i=4: [15,13,9] mdd=(9-15)/15=-0.4
    assert_vec_eq_nan(&r, &vec![0.0, f64::NAN, f64::NAN, f64::NAN, -0.4]);
  }
}
