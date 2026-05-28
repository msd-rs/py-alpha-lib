// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal, skip_nan_window::SkipNanWindow};

fn linear_reg_core<NumT, F>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
  op: F,
) -> Result<(), Error>
where
  NumT: Float + Send + Sync,
  F: Fn(NumT, NumT, NumT, NumT, NumT, NumT) -> NumT + Sync + Send + Copy,
{
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }


  if periods < 2 {
    r.fill(NumT::nan());
    return Ok(());
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      if ctx.is_skip_nan() {
        let iter = SkipNanWindow::new(&x[..end], periods, start);
        let mut sum_y = NumT::zero();
        let mut sum_y2 = NumT::zero();
        let mut sum_xy_1based = NumT::zero();
        let mut count = 0;

        for i in iter {
          for k in i.prev_start..i.start {
            let old = x[k];
            if is_normal(&old) {
              sum_xy_1based = sum_xy_1based - sum_y;
              sum_y = sum_y - old;
              sum_y2 = sum_y2 - old * old;
              count -= 1;
            }
          }

          let val = x[i.end];
          if is_normal(&val) {
            count += 1;
            let n_t = NumT::from(count).unwrap();
            sum_y = sum_y + val;
            sum_y2 = sum_y2 + val * val;
            sum_xy_1based = sum_xy_1based + n_t * val;
          }

          if !is_normal(&val) {
            continue;
          }

          let mut should_output = true;
          if ctx.is_strictly_cycle() {
            if count != periods || (i.end - i.start + 1) != periods {
              should_output = false;
            }
          }

          if should_output && count >= 2 {
            let n = NumT::from(count).unwrap();
            let sum_x = n * (n - NumT::one()) / NumT::from(2.0).unwrap();
            let sum_x2 = n * (n - NumT::one()) * (NumT::from(2.0).unwrap() * n - NumT::one())
              / NumT::from(6.0).unwrap();

            let sum_xy = sum_xy_1based - sum_y;

            r[i.end] = op(n, sum_x, sum_x2, sum_y, sum_y2, sum_xy);
          }
        }
      } else {
        let mut sum_y = NumT::zero();
        let mut sum_y2 = NumT::zero();
        let mut sum_xy_1based = NumT::zero();
        let mut count = 0;
        let mut nan_in_window = 0;

        let pre_fill_start = if start >= periods { start - periods } else { 0 };
        for k in pre_fill_start..start {
          let val = x[k];
          if is_normal(&val) {
            count += 1;
            let n_t = NumT::from(count).unwrap();
            sum_y = sum_y + val;
            sum_y2 = sum_y2 + val * val;
            sum_xy_1based = sum_xy_1based + n_t * val;
          } else {
            count += 1;
            nan_in_window += 1;
          }
        }

        let total = r.len();
        for (n, (r, c)) in r
          .iter_mut()
          .zip(x.iter())
          .enumerate()
          .skip(ctx.start(total))
        {
          count += 1;
          let n_t = NumT::from(count).unwrap();

          if is_normal(c) {
            sum_y = sum_y + *c;
            sum_y2 = sum_y2 + *c * *c;
            sum_xy_1based = sum_xy_1based + n_t * *c;
          } else {
            nan_in_window += 1;
          }

          if count > periods {
            let old_idx = n - periods;
            let old = x[old_idx];

            sum_xy_1based = sum_xy_1based - sum_y;
            if is_normal(&old) {
              sum_y = sum_y - old;
              sum_y2 = sum_y2 - old * old;
            } else {
              nan_in_window -= 1;
            }
            count -= 1;
          }

          if count == periods {
            if nan_in_window > 0 {
              *r = NumT::nan();
            } else if ctx.is_strictly_cycle() && n < periods - 1 {
              *r = NumT::nan();
            } else {
              let n_val = NumT::from(periods).unwrap();
              let sum_x = n_val * (n_val - NumT::one()) / NumT::from(2.0).unwrap();
              let sum_x2 =
                n_val * (n_val - NumT::one()) * (NumT::from(2.0).unwrap() * n_val - NumT::one())
                  / NumT::from(6.0).unwrap();

              let sum_xy = sum_xy_1based - sum_y;

              *r = op(n_val, sum_x, sum_x2, sum_y, sum_y2, sum_xy);
            }
          } else {
            *r = NumT::nan();
          }
        }
      }
    });
  Ok(())
}

/// Linear Regression Slope
///
/// Calculates the slope of the linear regression line for a moving window.
///
pub fn ta_slope<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  linear_reg_core(
    ctx,
    r,
    input,
    periods,
    |n, sum_x, sum_x2, sum_y, _sum_y2, sum_xy| {
      // slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x^2)
      let numerator = n * sum_xy - sum_x * sum_y;
      let denominator = n * sum_x2 - sum_x * sum_x;

      if denominator != NumT::zero() {
        numerator / denominator
      } else {
        NumT::nan()
      }
    },
  )
}

/// Linear Regression Intercept
///
/// Calculates the intercept of the linear regression line for a moving window.
///
pub fn ta_intercept<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  linear_reg_core(
    ctx,
    r,
    input,
    periods,
    |n, sum_x, sum_x2, sum_y, _sum_y2, sum_xy| {
      // Intercept: b = (sum_y * sum_x2 - sum_x * sum_xy) / (n * sum_x2 - sum_x^2)
      let numerator = sum_y * sum_x2 - sum_x * sum_xy;
      let denominator = n * sum_x2 - sum_x * sum_x;

      if denominator != NumT::zero() {
        numerator / denominator
      } else {
        NumT::nan()
      }
    },
  )
}

/// Time Series Correlation in moving window on self
///
/// Calculates the correlation coefficient between the input series and the time index.
///
pub fn ta_corr<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  linear_reg_core(
    ctx,
    r,
    input,
    periods,
    |n, sum_x, sum_x2, sum_y, sum_y2, sum_xy| {
      // r = (n * sum_xy - sum_x * sum_y) / sqrt( (n * sum_x2 - sum_x^2) * (n * sum_y2 - sum_y^2) )
      let numerator = n * sum_xy - sum_x * sum_y;
      let var_x = n * sum_x2 - sum_x * sum_x;
      let var_y = n * sum_y2 - sum_y * sum_y;

      let denominator_sq = var_x * var_y;
      if denominator_sq > NumT::zero() {
        numerator / denominator_sq.sqrt()
      } else {
        NumT::nan()
      }
    },
  )
}

#[cfg(test)]
mod tests {
  use crate::algo::{assert_vec_eq_nan, context::FLAG_SKIP_NAN};

  use super::*;

  #[test]
  fn test_ta_slope() {
    // y = 2x + 1. Slope should be 2.
    // 0: 1
    // 1: 3
    // 2: 5
    // 3: 7
    // 4: 9
    let input = vec![1.0, 3.0, 5.0, 7.0, 9.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_slope(&ctx, &mut r, &input, periods).unwrap();

    // 0: NaN
    // 1: NaN
    // 2: [1, 3, 5] -> x=[0,1,2]. y=[1,3,5]. Slope 2.
    // 3: [3, 5, 7] -> Slope 2.
    // 4: [5, 7, 9] -> Slope 2.

    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 2.0, 2.0, 2.0]);
  }

  #[test]
  fn test_ta_slope_flat() {
    let input = vec![5.0, 5.0, 5.0, 5.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_slope(&ctx, &mut r, &input, periods).unwrap();

    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 0.0, 0.0, 0.0]);
  }

  #[test]
  fn test_ta_slope_skip_nan() {
    let input = vec![1.0, 3.0, f64::NAN, 5.0, 7.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);
    ta_slope(&ctx, &mut r, &input, periods).unwrap();

    // 0: 1. Count 1.
    // 1: 3. Count 2. Slope? periods=3. Not enough?
    // Wait, my implementation requires `count >= 2`.
    // But standard `ma` requires `count == periods`?
    // ta_ma output partial results?
    // ta_ma outputs partial results if NOT strictly cycle.
    // My implementation: `if should_output && count >= 2`.
    // `should_output` is true if not strictly cycle.
    // So for index 1: count=2. [1, 3]. x=[0, 1]. Slope = (3-1)/(1-0) = 2.

    // 2: NaN.

    // 3: 5. Window [1, 3, 5] (skipped NaN). Count 3. Slope 2.

    // 4: 7. Window [3, 5, 7] (1 dropped). Count 3. Slope 2.

    let expected = vec![f64::NAN, 2.0, f64::NAN, 2.0, 2.0];
    assert_vec_eq_nan(&r, &expected);
  }

  #[test]
  fn test_ta_intercept() {
    // y = 2x + 1.
    // Window 0: [1, 3, 5]. x=[0, 1, 2]. m=2. b=1.
    // Window 1: [3, 5, 7]. x=[0, 1, 2]. m=2. b=3.
    // Window 2: [5, 7, 9]. x=[0, 1, 2]. m=2. b=5.
    let input = vec![1.0, 3.0, 5.0, 7.0, 9.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_intercept(&ctx, &mut r, &input, periods).unwrap();

    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 1.0, 3.0, 5.0]);
  }

  #[test]
  fn test_ta_ts_corr() {
    let input = vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_corr(&ctx, &mut r, &input, periods).unwrap();

    let expected = vec![
      f64::NAN,
      f64::NAN,
      1.0,
      0.8660254037844386,
      -0.8660254037844386,
      -1.0,
    ];
    assert_vec_eq_nan(&r, &expected);
  }
}
