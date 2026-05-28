// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal, skip_nan_window::SkipNanWindow};

/// Calculate rolling sample Skewness over a moving window
///
/// Uses adjusted Fisher-Pearson formula (matches pandas):
/// skew = n / ((n-1)(n-2)) * sum(((x-mean)/std)^3)
/// Requires at least 3 valid values.
pub fn ta_skewness<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }


  let two = NumT::from(2.0).unwrap();
  let three = NumT::from(3.0).unwrap();

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      if ctx.is_skip_nan() {
        let iter = SkipNanWindow::new(&x[..end], periods, start);
        let mut sum = NumT::zero();
        let mut sum_sq = NumT::zero();
        let mut sum_cb = NumT::zero();

        for i in iter {
          let val = x[i.end];
          if is_normal(&val) {
            sum = sum + val;
            sum_sq = sum_sq + val * val;
            sum_cb = sum_cb + val * val * val;
          }

          for k in i.prev_start..i.start {
            let old = x[k];
            if is_normal(&old) {
              sum = sum - old;
              sum_sq = sum_sq - old * old;
              sum_cb = sum_cb - old * old * old;
            }
          }

          if !is_normal(&val) {
            continue;
          }

          let mut should_output = true;
          if ctx.is_strictly_cycle() {
            if i.no_nan_count != periods || (i.end - i.start + 1) != periods {
              should_output = false;
            }
          }

          if should_output && i.no_nan_count >= 3 {
            let n = NumT::from(i.no_nan_count).unwrap();
            let mean = sum / n;
            // m2 = sum((x-mean)^2) = sum_sq - n*mean^2
            let m2 = sum_sq - n * mean * mean;
            // m3 = sum((x-mean)^3) = sum_cb - 3*mean*sum_sq + 2*n*mean^3
            let m3 = sum_cb - three * mean * sum_sq + two * n * mean * mean * mean;

            let var = m2 / (n - NumT::one());
            if var.abs() < NumT::epsilon() {
              r[i.end] = NumT::zero();
            } else {
              // adjusted skewness = n/((n-1)(n-2)) * m3 / var^(3/2)
              let std = var.sqrt();
              r[i.end] = (n / ((n - NumT::one()) * (n - two))) * (m3 / (std * std * std));
            }
          }
        }
      } else {
        let mut sum = NumT::zero();
        let mut sum_sq = NumT::zero();
        let mut sum_cb = NumT::zero();
        let mut nan_in_window = 0;

        let pre_fill_start = if start >= periods { start - periods } else { 0 };

        for k in pre_fill_start..start {
          let val = x[k];
          if is_normal(&val) {
            sum = sum + val;
            sum_sq = sum_sq + val * val;
            sum_cb = sum_cb + val * val * val;
          } else {
            nan_in_window += 1;
          }
        }

        for i in start..end {
          let val = x[i];

          if is_normal(&val) {
            sum = sum + val;
            sum_sq = sum_sq + val * val;
            sum_cb = sum_cb + val * val * val;
          } else {
            nan_in_window += 1;
          }

          if i >= periods {
            let old = x[i - periods];
            if is_normal(&old) {
              sum = sum - old;
              sum_sq = sum_sq - old * old;
              sum_cb = sum_cb - old * old * old;
            } else {
              nan_in_window -= 1;
            }
          }

          if nan_in_window > 0 || !is_normal(&val) {
            // NaN
          } else if i >= periods - 1 && periods >= 3 {
            let n = NumT::from(periods).unwrap();
            let mean = sum / n;
            let m2 = sum_sq - n * mean * mean;
            let m3 = sum_cb - three * mean * sum_sq + two * n * mean * mean * mean;

            let var = m2 / (n - NumT::one());
            if var.abs() < NumT::epsilon() {
              r[i] = NumT::zero();
            } else {
              let std = var.sqrt();
              r[i] = (n / ((n - NumT::one()) * (n - two))) * (m3 / (std * std * std));
            }
          }
        }
      }
    });

  Ok(())
}

/// Calculate rolling sample excess Kurtosis over a moving window
///
/// Uses adjusted Fisher formula (matches pandas):
/// kurt = n(n+1)/((n-1)(n-2)(n-3)) * sum(((x-mean)/std)^4) - 3(n-1)^2/((n-2)(n-3))
/// Requires at least 4 valid values.
pub fn ta_kurtosis<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }


  let two = NumT::from(2.0).unwrap();
  let three = NumT::from(3.0).unwrap();
  let four = NumT::from(4.0).unwrap();
  let six = NumT::from(6.0).unwrap();

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      if ctx.is_skip_nan() {
        let iter = SkipNanWindow::new(&x[..end], periods, start);
        let mut sum = NumT::zero();
        let mut sum_sq = NumT::zero();
        let mut sum_cb = NumT::zero();
        let mut sum_ft = NumT::zero();

        for i in iter {
          let val = x[i.end];
          if is_normal(&val) {
            let v2 = val * val;
            sum = sum + val;
            sum_sq = sum_sq + v2;
            sum_cb = sum_cb + v2 * val;
            sum_ft = sum_ft + v2 * v2;
          }

          for k in i.prev_start..i.start {
            let old = x[k];
            if is_normal(&old) {
              let o2 = old * old;
              sum = sum - old;
              sum_sq = sum_sq - o2;
              sum_cb = sum_cb - o2 * old;
              sum_ft = sum_ft - o2 * o2;
            }
          }

          if !is_normal(&val) {
            continue;
          }

          let mut should_output = true;
          if ctx.is_strictly_cycle() {
            if i.no_nan_count != periods || (i.end - i.start + 1) != periods {
              should_output = false;
            }
          }

          if should_output && i.no_nan_count >= 4 {
            let n = NumT::from(i.no_nan_count).unwrap();
            let mean = sum / n;
            let mean2 = mean * mean;
            let mean4 = mean2 * mean2;

            // m2 = sum((x-mean)^2) = sum_sq - n*mean^2
            let m2 = sum_sq - n * mean2;
            // m4 = sum((x-mean)^4) = sum_ft - 4*mean*sum_cb + 6*mean^2*sum_sq - 3*n*mean^4
            let m4 = sum_ft - four * mean * sum_cb + six * mean2 * sum_sq - three * n * mean4;

            let var = m2 / (n - NumT::one());
            if var.abs() < NumT::epsilon() {
              r[i.end] = NumT::zero();
            } else {
              // Adjusted kurtosis (Fisher):
              // = n(n+1)/((n-1)(n-2)(n-3)) * m4/var^2 - 3(n-1)^2/((n-2)(n-3))
              let a = n * (n + NumT::one()) / ((n - NumT::one()) * (n - two) * (n - three));
              let b = three * (n - NumT::one()) * (n - NumT::one()) / ((n - two) * (n - three));
              r[i.end] = a * (m4 / (var * var)) - b;
            }
          }
        }
      } else {
        let mut sum = NumT::zero();
        let mut sum_sq = NumT::zero();
        let mut sum_cb = NumT::zero();
        let mut sum_ft = NumT::zero();
        let mut nan_in_window = 0;

        let pre_fill_start = if start >= periods { start - periods } else { 0 };

        for k in pre_fill_start..start {
          let val = x[k];
          if is_normal(&val) {
            let v2 = val * val;
            sum = sum + val;
            sum_sq = sum_sq + v2;
            sum_cb = sum_cb + v2 * val;
            sum_ft = sum_ft + v2 * v2;
          } else {
            nan_in_window += 1;
          }
        }

        for i in start..end {
          let val = x[i];

          if is_normal(&val) {
            let v2 = val * val;
            sum = sum + val;
            sum_sq = sum_sq + v2;
            sum_cb = sum_cb + v2 * val;
            sum_ft = sum_ft + v2 * v2;
          } else {
            nan_in_window += 1;
          }

          if i >= periods {
            let old = x[i - periods];
            if is_normal(&old) {
              let o2 = old * old;
              sum = sum - old;
              sum_sq = sum_sq - o2;
              sum_cb = sum_cb - o2 * old;
              sum_ft = sum_ft - o2 * o2;
            } else {
              nan_in_window -= 1;
            }
          }

          if nan_in_window > 0 || !is_normal(&val) {
            // NaN
          } else if i >= periods - 1 && periods >= 4 {
            let n = NumT::from(periods).unwrap();
            let mean = sum / n;
            let mean2 = mean * mean;
            let mean4 = mean2 * mean2;

            let m2 = sum_sq - n * mean2;
            let m4 = sum_ft - four * mean * sum_cb + six * mean2 * sum_sq - three * n * mean4;

            let var = m2 / (n - NumT::one());
            if var.abs() < NumT::epsilon() {
              r[i] = NumT::zero();
            } else {
              let a = n * (n + NumT::one()) / ((n - NumT::one()) * (n - two) * (n - three));
              let b = three * (n - NumT::one()) * (n - NumT::one()) / ((n - two) * (n - three));
              r[i] = a * (m4 / (var * var)) - b;
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
  use crate::algo::{assert_vec_eq_nan, context::FLAG_SKIP_NAN};

  #[test]
  fn test_skewness_symmetric() {
    // Evenly spaced values -> skewness = 0 for all windows
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_skewness(&ctx, &mut r, &input, periods).unwrap();

    // [1,2,3]: skew=0, [2,3,4]: skew=0, [3,4,5]: skew=0
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 0.0, 0.0, 0.0]);
  }

  #[test]
  fn test_skewness_positive() {
    // Right-skewed data
    let input = vec![1.0, 1.0, 10.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_skewness(&ctx, &mut r, &input, periods).unwrap();

    // Pandas: pd.Series([1,1,10]).skew() = 1.7320508075688774
    let expected = 3.0f64.sqrt();
    assert!(
      (r[2] - expected).abs() < 1e-5,
      "got {}, expected {}",
      r[2],
      expected
    );
  }

  #[test]
  fn test_skewness_skip_nan() {
    let input = vec![1.0, f64::NAN, 2.0, 3.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);
    ta_skewness(&ctx, &mut r, &input, periods).unwrap();

    // Position 3: valid window [1, 2, 3] -> skew = 0
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, f64::NAN, 0.0]);
  }

  #[test]
  fn test_kurtosis_uniform() {
    // pd.Series([1,2,3,4]).kurtosis() = -1.2
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let periods = 4;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_kurtosis(&ctx, &mut r, &input, periods).unwrap();

    assert!((r[3] - (-1.2)).abs() < 1e-5, "got {}, expected -1.2", r[3]);
  }

  #[test]
  fn test_kurtosis_skip_nan() {
    let input = vec![1.0, f64::NAN, 2.0, 3.0, 4.0];
    let periods = 4;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);
    ta_kurtosis(&ctx, &mut r, &input, periods).unwrap();

    // Position 4: valid values [1, 2, 3, 4] -> kurt = -1.2
    assert!((r[4] - (-1.2)).abs() < 1e-5, "got {}, expected -1.2", r[4]);
  }
}
