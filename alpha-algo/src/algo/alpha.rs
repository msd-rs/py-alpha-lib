// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal};

/// Rolling Jensen's Alpha of asset returns against benchmark returns.
///
/// Alpha = mean(input) - Beta * mean(benchmark)
/// Measures excess return of an asset relative to its expected return given beta.
///
/// Ref: https://en.wikipedia.org/wiki/Jensen%27s_alpha
///
pub fn ta_alpha<NumT: Float + Send + Sync>(
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
      "alpha requires periods >= 2, got {}",
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
          let beta = (nf * sum_xy - sum_x * sum_y) / denom;
          r[i] = (sum_x - beta * sum_y) / nf;
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
          let beta = (nf * sum_xy - sum_x * sum_y) / denom;
          r[i] = (sum_x - beta * sum_y) / nf;
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
  fn test_ta_alpha_basic() {
    // y = 2x, Alpha should be 0 (returns proportional, no excess return)
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let periods = 3;
    let mut r = vec![0.0; x.len()];
    let ctx = Context::new(0, 0, 0);
    ta_alpha(&ctx, &mut r, &x, &y, periods).unwrap();
    assert_vec_eq_nan(&r, &vec![f64::NAN, 0.0, 0.0, 0.0, 0.0]);
  }

  #[test]
  fn test_ta_alpha_positive() {
    // x has extra 1.0 above y*0.5
    let x = vec![2.0, 3.0, 4.0, 5.0, 6.0];   // y*0.5 + 1
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];   // y=2*(x-1)
    let periods = 3;
    let mut r = vec![0.0; x.len()];
    let ctx = Context::new(0, 0, 0);
    ta_alpha(&ctx, &mut r, &x, &y, periods).unwrap();
    // Alpha should be 1.0
    assert_vec_eq_nan(&r, &vec![f64::NAN, 1.0, 1.0, 1.0, 1.0]);
  }

  #[test]
  fn test_ta_alpha_strictly_cycle() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let periods = 3;
    let mut r = vec![0.0; x.len()];
    let ctx = Context::new(0, 0, FLAG_STRICTLY_CYCLE);
    ta_alpha(&ctx, &mut r, &x, &y, periods).unwrap();
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 0.0, 0.0, 0.0]);
  }

  #[test]
  fn test_ta_alpha_skip_nan() {
    let x = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let periods = 3;
    let mut r = vec![0.0; x.len()];

    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);
    ta_alpha(&ctx, &mut r, &x, &y, periods).unwrap();
    // i=0: n=1 → NaN
    // i=1: NaN → skip
    // i=2: pairs (0,0) and (2,2) valid, n=2 → Alpha=0
    // i=3: 3 valid pairs → Alpha=0
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 0.0, 0.0, 0.0]);
  }
}
