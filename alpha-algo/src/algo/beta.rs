// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal};

/// Rolling Beta coefficient of asset returns against benchmark returns.
///
/// Beta = Covariance(input, benchmark) / Variance(benchmark)
/// Measures systematic risk of an asset relative to the market.
///
/// Ref: https://en.wikipedia.org/wiki/Beta_(finance)
///
pub fn ta_beta<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  benchmark: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != input.len() || r.len() != benchmark.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }

  if periods < 2 {
    return Err(Error::InvalidPeriod(format!(
      "beta requires periods >= 2, got {}",
      periods
    )));
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .zip(benchmark.par_chunks(ctx.chunk_size(benchmark.len())))
    .for_each(|((r, x), y)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      if ctx.is_skip_nan() {
        for i in start..end {
          let mut sum_x = NumT::zero();
          let mut sum_y = NumT::zero();
          let mut sum_xy = NumT::zero();
          let mut sum_y2 = NumT::zero();
          let mut n = 0usize;
          let mut gap = false;

          let mut j = i;
          loop {
            if is_normal(&x[j]) && is_normal(&y[j]) {
              sum_x = sum_x + x[j];
              sum_y = sum_y + y[j];
              sum_xy = sum_xy + x[j] * y[j];
              sum_y2 = sum_y2 + y[j] * y[j];
              n += 1;
              if n == periods {
                break;
              }
            } else {
              gap = true;
            }
            if j == 0 {
              break;
            }
            j -= 1;
          }

          if !is_normal(&x[i]) || !is_normal(&y[i]) {
            continue;
          }

          if ctx.is_strictly_cycle() {
            if n != periods || gap || (i - j + 1) != periods {
              continue;
            }
          }

          if n < 2 {
            continue;
          }

          let nf = NumT::from(n).unwrap();
          let denom = nf * sum_y2 - sum_y * sum_y;
          if denom == NumT::zero() {
            continue;
          }
          r[i] = (nf * sum_xy - sum_x * sum_y) / denom;
        }
      } else {
        let mut sum_x = NumT::zero();
        let mut sum_y = NumT::zero();
        let mut sum_xy = NumT::zero();
        let mut sum_y2 = NumT::zero();
        let mut nan_in_window = 0usize;

        let pre_start = if start >= periods { start - periods } else { 0 };
        for k in pre_start..start {
          if is_normal(&x[k]) && is_normal(&y[k]) {
            sum_x = sum_x + x[k];
            sum_y = sum_y + y[k];
            sum_xy = sum_xy + x[k] * y[k];
            sum_y2 = sum_y2 + y[k] * y[k];
          } else {
            nan_in_window += 1;
          }
        }

        for i in start..end {
          if is_normal(&x[i]) && is_normal(&y[i]) {
            sum_x = sum_x + x[i];
            sum_y = sum_y + y[i];
            sum_xy = sum_xy + x[i] * y[i];
            sum_y2 = sum_y2 + y[i] * y[i];
          } else {
            nan_in_window += 1;
          }

          if i >= periods {
            let old_x = x[i - periods];
            let old_y = y[i - periods];
            if is_normal(&old_x) && is_normal(&old_y) {
              sum_x = sum_x - old_x;
              sum_y = sum_y - old_y;
              sum_xy = sum_xy - old_x * old_y;
              sum_y2 = sum_y2 - old_y * old_y;
            } else {
              nan_in_window -= 1;
            }
          }

          if !is_normal(&x[i]) || !is_normal(&y[i]) {
            continue;
          }

          if ctx.is_strictly_cycle() && i < periods - 1 {
            continue;
          }

          if nan_in_window > 0 {
            continue;
          }

          let n = if i < periods { i + 1 } else { periods };
          if n < 2 {
            continue;
          }
          let nf = NumT::from(n).unwrap();
          let denom = nf * sum_y2 - sum_y * sum_y;
          if denom == NumT::zero() {
            continue;
          }
          r[i] = (nf * sum_xy - sum_x * sum_y) / denom;
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
  fn test_ta_beta_basic() {
    // y = 2x, so Beta(x on y) = 0.5
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let periods = 3;
    let mut r = vec![0.0; x.len()];
    let ctx = Context::new(0, 0, 0);
    ta_beta(&ctx, &mut r, &x, &y, periods).unwrap();
    assert_vec_eq_nan(&r, &vec![f64::NAN, 0.5, 0.5, 0.5, 0.5]);
  }

  #[test]
  fn test_ta_beta_strictly_cycle() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let periods = 3;
    let mut r = vec![0.0; x.len()];
    let ctx = Context::new(0, 0, FLAG_STRICTLY_CYCLE);
    ta_beta(&ctx, &mut r, &x, &y, periods).unwrap();
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 0.5, 0.5, 0.5]);
  }

  #[test]
  fn test_ta_beta_skip_nan() {
    let x = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let periods = 3;
    let mut r = vec![0.0; x.len()];

    // Without skip_nan
    let ctx = Context::new(0, 0, 0);
    ta_beta(&ctx, &mut r, &x, &y, periods).unwrap();
    assert_vec_eq_nan(&r, &vec![
      f64::NAN, f64::NAN, f64::NAN, f64::NAN, 0.5,
    ]);

    // With skip_nan
    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);
    ta_beta(&ctx, &mut r, &x, &y, periods).unwrap();
    // i=0: n=1 → NaN
    // i=1: NaN → skip
    // i=2: pairs (2,4) and (3,6) valid, n=2 → Beta=0.5
    // i=3: pairs (3,6) and (4,8), n=2 → Beta=0.5
    // i=4: pairs (3,6),(4,8),(5,10), n=3 → Beta=0.5
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 0.5, 0.5, 0.5]);
  }

  #[test]
  fn test_ta_beta_skip_nan_strictly() {
    let x = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let periods = 3;
    let mut r = vec![0.0; x.len()];

    let ctx = Context::new(0, 0, FLAG_SKIP_NAN | FLAG_STRICTLY_CYCLE);
    ta_beta(&ctx, &mut r, &x, &y, periods).unwrap();
    // skip_nan + strictly_cycle: no NaN gaps allowed AND exactly periods valid pairs
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN, 0.5]);
  }

  #[test]
  fn test_ta_beta_perfect_correlation() {
    // y = x, Beta should be 1.0
    let x = vec![2.0, 3.0, 5.0, 7.0, 4.0, 6.0, 8.0];
    let y = vec![2.0, 3.0, 5.0, 7.0, 4.0, 6.0, 8.0];
    let periods = 4;
    let mut r = vec![0.0; x.len()];
    let ctx = Context::new(0, 0, 0);
    ta_beta(&ctx, &mut r, &x, &y, periods).unwrap();
    // Window [0,1,2,3]: x=[2,3,5,7] y=[2,3,5,7]
    // sum_x=17 sum_y=17 sum_xy=4+9+25+49=87 sum_y2=87
    // Beta = (4*87-289)/(4*87-289) = 1.0
    assert_vec_eq_nan(&r, &vec![f64::NAN, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
  }

  #[test]
  fn test_ta_beta_benchmark_nan() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, f64::NAN, 6.0, 8.0, 10.0];
    let periods = 3;
    let mut r = vec![0.0; x.len()];
    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);
    ta_beta(&ctx, &mut r, &x, &y, periods).unwrap();
    // Only compute from pairs where both are valid
    // i=2: valid pairs at (0,0) and (2,2) → Beta=0.5
    // i=3: valid pairs at (0,0),(2,2),(3,3) → Beta=0.5
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 0.5, 0.5, 0.5]);
  }
}
