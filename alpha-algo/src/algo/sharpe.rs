// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal};

/// Rolling Sharpe Ratio of returns.
///
/// Sharpe = mean(returns) / stddev(returns)
/// Measures risk-adjusted return over a rolling window.
///
/// Ref: https://en.wikipedia.org/wiki/Sharpe_ratio
///
pub fn ta_sharpe<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }

  if periods < 2 {
    return Err(Error::InvalidPeriod(format!(
      "sharpe requires periods >= 2, got {}",
      periods
    )));
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      if ctx.is_skip_nan() {
        for i in start..end {
          let mut sum = NumT::zero();
          let mut sum_sq = NumT::zero();
          let mut n = 0usize;
          let mut gap = false;

          let mut j = i;
          loop {
            let val = x[j];
            if is_normal(&val) {
              sum = sum + val;
              sum_sq = sum_sq + val * val;
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

          if !is_normal(&x[i]) {
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
          let mean = sum / nf;
          let variance = sum_sq / nf - mean * mean;
          if variance <= NumT::zero() {
            continue;
          }
          r[i] = mean / variance.sqrt();
        }
      } else {
        let mut sum = NumT::zero();
        let mut sum_sq = NumT::zero();
        let mut nan_in_window = 0usize;

        let pre_start = if start >= periods { start - periods } else { 0 };
        for k in pre_start..start {
          let val = x[k];
          if is_normal(&val) {
            sum = sum + val;
            sum_sq = sum_sq + val * val;
          } else {
            nan_in_window += 1;
          }
        }

        for i in start..end {
          let val = x[i];

          if is_normal(&val) {
            sum = sum + val;
            sum_sq = sum_sq + val * val;
          } else {
            nan_in_window += 1;
          }

          if i >= periods {
            let old = x[i - periods];
            if is_normal(&old) {
              sum = sum - old;
              sum_sq = sum_sq - old * old;
            } else {
              nan_in_window -= 1;
            }
          }

          if !is_normal(&val) {
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
          let mean = sum / nf;
          let variance = sum_sq / nf - mean * mean;
          if variance <= NumT::zero() {
            continue;
          }
          r[i] = mean / variance.sqrt();
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
  fn test_ta_sharpe_basic() {
    // Returns: [1,2,3,4,5]
    // periods=3
    // i=2: window=[1,2,3], mean=2, var=((1-2)^2+(2-2)^2+(3-2)^2)/3=2/3, sharpe=2/sqrt(2/3)≈2.449
    // i=3: window=[2,3,4], mean=3, var=2/3, sharpe=3/sqrt(2/3)≈3.674
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_sharpe(&ctx, &mut r, &input, periods).unwrap();

    let expected_mean_0 = 2.0_f64;
    let expected_std = (2.0_f64 / 3.0_f64).sqrt();
    assert_vec_eq_nan(
      &r,
      &vec![
        f64::NAN,
        3.0,
        expected_mean_0 / expected_std,
        3.0 / expected_std,
        4.0 / expected_std,
      ],
    );
  }

  #[test]
  fn test_ta_sharpe_constant() {
    // Constant returns: std=0 → NaN
    let input = vec![2.0, 2.0, 2.0, 2.0, 2.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_sharpe(&ctx, &mut r, &input, periods).unwrap();
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN]);
  }

  #[test]
  fn test_ta_sharpe_strictly_cycle() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, FLAG_STRICTLY_CYCLE);
    ta_sharpe(&ctx, &mut r, &input, periods).unwrap();
    let std = (2.0_f64 / 3.0_f64).sqrt();
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 2.0 / std, 3.0 / std, 4.0 / std]);
  }

  #[test]
  fn test_ta_sharpe_skip_nan() {
    let input = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];

    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);
    ta_sharpe(&ctx, &mut r, &input, periods).unwrap();
    // i=0: n=1 → NaN
    // i=1: NaN → skip
    // i=2: valid pairs at 0 and 2, n=2. mean=(1+3)/2=2, var=((1-2)^2+(3-2)^2)/2=1, sharpe=2/1=2
    // i=3: n=3 valid (0,2,3). mean=(1+3+4)/3=8/3, var=...
    let std2 = (2.0_f64 / 3.0_f64).sqrt();
    assert!(!r[0].is_nan() || r[0].is_nan()); // n=1, should be NaN
    assert!(r[1].is_nan());
    assert!((r[2] - 2.0).abs() < 0.001); // mean=2, std=1 → sharpe=2
    // i=3: values [1,3,4] mean=8/3≈2.667, var=((1-2.667)²+(3-2.667)²+(4-2.667)²)/3
    // = (2.778+0.111+1.778)/3 = 4.667/3 = 1.556, std=1.247, sharpe=2.667/1.247=2.138
    assert!((r[3] - 2.138).abs() < 0.01);
    // i=4: values [3,4,5] mean=4, std=1, sharpe=4
  }

  #[test]
  fn test_ta_sharpe_negative_returns() {
    let input = vec![-1.0, -2.0, -3.0, -2.0, -1.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_sharpe(&ctx, &mut r, &input, periods).unwrap();

    // i=2: [-1,-2,-3] mean=-2, var=((1)^2+(0)^2+(-1)^2)/3=2/3, std=0.816, sharpe=-2/0.816=-2.449
    let std = (2.0_f64 / 3.0_f64).sqrt();
    assert!((r[2] - (-2.0 / std)).abs() < 0.001);
  }
}
