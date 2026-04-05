// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal, skip_nan_window::SkipNanWindow};

/// Calculate Standard Deviation over a moving window
///
/// Ref: https://en.wikipedia.org/wiki/Standard_deviation
pub fn ta_stddev<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }


  // If periods == 1, stddev is 0.0 (if valid) or NaN?
  // Sample stddev requires N >= 2 usually.
  // Population stddev can be defined for N=1 (it is 0).
  // "Standard" trading software usually uses Population or Sample?
  // Pandas uses ddof=1 (Sample) by default.
  // Polars uses ddof=1.
  // TA-Lib uses ddof=1?
  // Let's implement Sample StdDev (divisor N-1) effectively,
  // OR Population (divisor N).
  // Most financial libraries use Population (N) or Sample (N-1).
  // Let's stick to what formula implies.
  // User asked for "stddev".
  // Let's use Sample StdDev (divide by N-1) for statistical correctness on samples,
  // but many trading indicators use Population (divide by N).
  // Let's check `alpha101_alphalib.py`... `df.rolling(window).std()`.
  // Pandas `rolling().std()` uses ddof=1 (Sample).
  // So we should use N-1.

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

        for i in iter {
          let val = x[i.end];
          if is_normal(&val) {
            sum = sum + val;
            sum_sq = sum_sq + val * val;
          }

          for k in i.prev_start..i.start {
            let old = x[k];
            if is_normal(&old) {
              sum = sum - old;
              sum_sq = sum_sq - old * old;
            }
          }

          if !is_normal(&val) {
            continue;
          }

          // Calculate StdDev
          // we use actual count of valid numbers in window
          let count = NumT::from(i.no_nan_count).unwrap();

          let mut should_output = true;
          if ctx.is_strictly_cycle() {
            if i.no_nan_count != periods || (i.end - i.start + 1) != periods {
              should_output = false;
            }
          }

          if should_output {
            if i.no_nan_count > 1 {
              // Variance = (SumSq - (Sum^2)/N) / (N - 1)
              let var_num = sum_sq - (sum * sum / count);
              let var = var_num / (count - NumT::one());
              // Handle precision issues where var might be slightly negative
              if var < NumT::zero() {
                r[i.end] = NumT::zero();
              } else {
                r[i.end] = var.sqrt();
              }
            } else {
              r[i.end] = NumT::zero(); // or NaN? Pandas returns NaN for N=1. 
              // Let's return NaN for N=1 to be safe/consistent with Pandas
              r[i.end] = NumT::nan();
            }
          }
        }
      } else {
        let mut sum = NumT::zero();
        let mut sum_sq = NumT::zero();
        let mut nan_in_window = 0;

        let pre_fill_start = if start >= periods { start - periods } else { 0 };

        for k in pre_fill_start..start {
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

          if nan_in_window > 0 || !is_normal(&val) {
            // Result NaN
          } else {
            // For non-strictly cycle, we might output for i < periods if we wanted partial.
            // But stddev usually implies full window.
            // Let's output only when we have at least 'periods' elements (or loop index check).

            let output_idx = i;

            let mut valid = false;
            if ctx.is_strictly_cycle() {
              if i >= periods - 1 {
                valid = true;
              }
            } else {
              // For standard stddev, we usually wait for full window?
              // Or we verify we have enough data?
              // Let's assume full window is required.
              if i >= periods - 1 {
                valid = true;
              }
            }

            if valid {
              let count = NumT::from(periods).unwrap();
              if periods > 1 {
                let var_num = sum_sq - (sum * sum / count);
                let var = var_num / (count - NumT::one());
                if var < NumT::zero() {
                  r[output_idx] = NumT::zero();
                } else {
                  r[output_idx] = var.sqrt();
                }
              } else {
                r[output_idx] = NumT::nan();
              }
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
  fn test_stddev_sliding() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_stddev(&ctx, &mut r, &input, periods).unwrap();

    // N=3. Divisor=2.
    // 0: NaN
    // 1: NaN
    // 2: [1,2,3]. Mean=2. SumSq=1+4+9=14. Var=(14 - 36/3)/2 = (14-12)/2 = 1. StdDev=1.
    // 3: [2,3,4]. Mean=3. SumSq=4+9+16=29. Var=(29 - 81/3)/2 = (29-27)/2 = 1.
    // 4: [3,4,5]. Mean=4. SumSq=9+16+25=50. Var=(50 - 144/3)/2 = (50-48)/2 = 1.

    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 1.0, 1.0, 1.0]);
  }

  #[test]
  fn test_stddev_skip_nan() {
    let input = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];

    // Case 1: No skip (should be NaN when NaN in window)
    let ctx = Context::new(0, 0, 0);
    ta_stddev(&ctx, &mut r, &input, periods).unwrap();
    // 2: NaN (NaN in window)
    // 3: NaN (NaN in window)
    // 4: [NaN, 4, 5]? No window is indices 2,3,4 -> [NaN, 4, 5]. NaN.
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN]);

    // Case 2: Skip Nan
    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);
    ta_stddev(&ctx, &mut r, &input, periods).unwrap();
    // 2: [1,2]. N=2. Mean=1.5. SumSq=5. Var=(5 - 2*2.25)/(1) = 0.5. Std=0.707...
    // 3: [1,2,4]. N=3. Mean=7/3=2.333. SumSq=1+4+16=21. Var=(21 - 3*(49/9))/2 = (21 - 16.333)/2 = 2.333. Std=1.527...
    // 4: [2,4,5]. N=3. Mean=11/3=3.666. SumSq=4+16+25=45. Var=(45 - 3*(121/9))/2 = (45 - 40.333)/2 = 2.333. Std=1.527...

    // Let's calc manually:
    // 2: [1, 2]. N=2. Var=SampVar([1,2]) = 0.5. Sqrt(0.5)=0.70710678.
    // 3: [1, 2, 4]. N=3. Var=SampVar([1,2,4]). Mean=2.333. Diff=[-1.333, -0.333, 1.666]. Sq=[1.777, 0.111, 2.777]. Sum=4.666. Var=2.333. Sqrt=1.527525.
    // 4: [2, 4, 5]. N=3. Var=SampVar([2,4,5]). Mean=3.666. Diff=[-1.666, 0.333, 1.333]. Sq=[2.777, 0.111, 1.777]. Sum=4.666. Var=2.333. Sqrt=1.527525.

    let expected = vec![
      f64::NAN,
      (0.5f64).sqrt(),
      f64::NAN,
      1.5275252316519468,
      1.5275252316519468,
    ];
    assert_vec_eq_nan(&r, &expected);
  }
}
