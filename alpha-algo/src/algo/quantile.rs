// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal};

/// Calculate rolling quantile over a moving window
///
/// QUANTILE(x, d, q) returns the q-th quantile (0 <= q <= 1) of values
/// in the preceding d periods. Uses linear interpolation between data points
/// (matching numpy/pandas percentile with interpolation='linear').
/// NaN values are excluded from the computation. Requires at least 1 valid value.
///
/// Ref: https://numpy.org/doc/stable/reference/generated/numpy.quantile.html
pub fn ta_quantile<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
  q: NumT,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }

  if periods == 0 {
    r.fill(NumT::nan());
    return Ok(());
  }

  if q < NumT::zero() || q > NumT::one() || q.is_nan() {
    r.fill(NumT::nan());
    return Ok(());
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      // Allocate a buffer for sorting within the window
      let mut buf = vec![NumT::zero(); periods];

      for i in start..end {
        let val = x[i];
        if !is_normal(&val) {
          continue;
        }

        // Determine window boundaries
        let win_start = if i >= periods { i - periods + 1 } else { 0 };
        let win_end = i + 1;

        // Collect valid (non-NaN) values into buffer
        let mut count = 0;
        for k in win_start..win_end {
          let v = x[k];
          if is_normal(&v) {
            buf[count] = v;
            count += 1;
          }
        }

        if count == 0 {
          continue;
        }

        // Check strict cycle: require full window of `periods` valid values
        if ctx.is_strictly_cycle() {
          if i < periods - 1 || count != periods || (win_end - win_start) != periods {
            continue;
          }
        }

        // Sort the buffer (insertion sort for small windows, efficient for typical d=5..50)
        let slice = &mut buf[..count];
        for j in 1..count {
          let key = slice[j];
          let mut k = j;
          while k > 0 && slice[k - 1] > key {
            slice[k] = slice[k - 1];
            k -= 1;
          }
          slice[k] = key;
        }

        // Linear interpolation quantile (matching numpy/pandas)
        let pos = q * NumT::from(count - 1).unwrap();
        let lo = pos.floor();
        let hi = lo + NumT::one();
        let lo_idx = lo.to_usize().unwrap_or(0).min(count - 1);
        let hi_idx = hi.to_usize().unwrap_or(0).min(count - 1);
        let frac = pos - lo;
        r[i] = slice[lo_idx] * (NumT::one() - frac) + slice[hi_idx] * frac;
      }
    });

  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::algo::{assert_vec_eq_nan, context::FLAG_STRICTLY_CYCLE};

  #[test]
  fn test_quantile_median() {
    // Median (q=0.5) of [1,2,3] = 2.0
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_quantile(&ctx, &mut r, &input, periods, 0.5).unwrap();

    // i=0: [1] -> median=1.0
    // i=1: [1,2] -> median=1.5
    // i=2: [1,2,3] -> median=2.0
    // i=3: [2,3,4] -> median=3.0
    // i=4: [3,4,5] -> median=4.0
    assert_vec_eq_nan(&r, &vec![1.0, 1.5, 2.0, 3.0, 4.0]);
  }

  #[test]
  fn test_quantile_strict() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, FLAG_STRICTLY_CYCLE);
    ta_quantile(&ctx, &mut r, &input, periods, 0.5).unwrap();

    assert_vec_eq_nan(
      &r,
      &vec![f64::NAN, f64::NAN, 2.0, 3.0, 4.0],
    );
  }

  #[test]
  fn test_quantile_q0_q1() {
    let input = vec![3.0, 1.0, 4.0, 1.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, FLAG_STRICTLY_CYCLE);

    // q=0 -> min
    ta_quantile(&ctx, &mut r, &input, periods, 0.0).unwrap();
    assert_vec_eq_nan(
      &r,
      &vec![f64::NAN, f64::NAN, 1.0, 1.0, 1.0],
    );

    // q=1 -> max
    ta_quantile(&ctx, &mut r, &input, periods, 1.0).unwrap();
    assert_vec_eq_nan(
      &r,
      &vec![f64::NAN, f64::NAN, 4.0, 4.0, 5.0],
    );
  }

  #[test]
  fn test_quantile_with_nan() {
    let input = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_quantile(&ctx, &mut r, &input, periods, 0.5).unwrap();

    // i=0: [1] -> 1.0
    // i=1: NaN -> NaN
    // i=2: [1, 3] (NaN skipped) -> median of [1,3] = 2.0
    // i=3: [3, 4] (NaN skipped) -> median of [3,4] = 3.5
    // i=4: [3, 4, 5] -> median = 4.0
    assert_vec_eq_nan(&r, &vec![1.0, f64::NAN, 2.0, 3.5, 4.0]);
  }

  #[test]
  fn test_quantile_groups() {
    // 2 groups of 3 each
    let input = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 2, FLAG_STRICTLY_CYCLE);
    ta_quantile(&ctx, &mut r, &input, periods, 0.5).unwrap();

    // Group 1: [1,2,3] -> median=2.0
    // Group 2: [10,20,30] -> median=20.0
    assert_vec_eq_nan(
      &r,
      &vec![f64::NAN, f64::NAN, 2.0, f64::NAN, f64::NAN, 20.0],
    );
  }

  #[test]
  fn test_quantile_periods_zero() {
    let input = vec![1.0, 2.0, 3.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_quantile(&ctx, &mut r, &input, 0, 0.5).unwrap();
    // periods=0 → all NaN
    assert!(r.iter().all(|x| x.is_nan()));
  }

  #[test]
  fn test_quantile_invalid_q() {
    let input = vec![1.0, 2.0, 3.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);

    // q < 0
    ta_quantile(&ctx, &mut r, &input, 3, -0.1).unwrap();
    assert!(r.iter().all(|x| x.is_nan()));

    // q > 1
    ta_quantile(&ctx, &mut r, &input, 3, 1.5).unwrap();
    assert!(r.iter().all(|x| x.is_nan()));

    // q = NaN
    ta_quantile(&ctx, &mut r, &input, 3, f64::NAN).unwrap();
    assert!(r.iter().all(|x| x.is_nan()));
  }
}
