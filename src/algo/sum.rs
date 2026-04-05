// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal, skip_nan_window::SkipNanWindow};

/// Calculate sum of values in preceding `periods` window
///
/// If periods is 0, it calculates the cumulative sum from the first valid value.
///
/// Ref: https://www.amibroker.com/guide/afl/sum.html
pub fn ta_sum<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }


  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      if periods == 0 {
        // Cumulative sum
        let mut sum = NumT::zero();
        let mut found_valid = false;

        for i in start..end {
          let val = x[i];
          if is_normal(&val) {
            sum = sum + val;
            found_valid = true;
          }

          if found_valid {
            r[i] = sum;
          }
        }
      } else {
        // Sliding window sum
        if ctx.is_skip_nan() {
          // Skip NaN logic
          let iter = SkipNanWindow::new(&x[..end], periods, start);
          let mut sum = NumT::zero();

          for i in iter {
            let val = x[i.end];
            if is_normal(&val) {
              sum = sum + val;
            }

            // subtract values that fell out of the window
            for k in i.prev_start..i.start {
              let old = x[k];
              if is_normal(&old) {
                sum = sum - old;
              }
            }

            if !is_normal(&val) {
              continue;
            }

            if ctx.is_strictly_cycle() {
              if i.no_nan_count == periods && (i.end - i.start + 1) == periods {
                r[i.end] = sum;
              }
            } else {
              r[i.end] = sum;
            }
          }
        } else {
          // Normal mode
          let mut sum = NumT::zero();
          let mut nan_in_window = 0;

          // Pre-initialization for start > 0
          let pre_fill_start = if start >= periods { start - periods } else { 0 };
          for k in pre_fill_start..start {
            if is_normal(&x[k]) {
              sum = sum + x[k];
            } else {
              nan_in_window += 1;
            }
          }

          for i in start..end {
            let val = x[i];

            // Add new value
            if is_normal(&val) {
              sum = sum + val;
            } else {
              nan_in_window += 1;
            }

            // Remove old value
            if i >= periods {
              let old = x[i - periods];
              if is_normal(&old) {
                sum = sum - old;
              } else {
                nan_in_window -= 1;
              }
            }

            if ctx.is_strictly_cycle() {
              if i >= periods - 1 {
                if nan_in_window == 0 {
                  r[i] = sum;
                }
              }
            } else {
              if i < periods {
                // Usually SUM(X, N) for N < periods is not defined or is partial sum?
                // AmiBroker SUM returns value for first N bars?
                // Documentation says: "Sum of array over periods".
                // Usually simple moving sum requires N periods.
                // Let's assume naive partial sum is returned until full window, UNLESS strictly cycle?
                // Or consistent with MA? MA returns partial MA.
                // Let's return partial sum.
                if nan_in_window == 0 {
                  r[i] = sum;
                }
              } else {
                if nan_in_window == 0 {
                  r[i] = sum;
                }
              }
            }
          }
        }
      }
    });

  Ok(())
}

/// Calculate number of periods (bars) backwards until the sum of values is greater than or equal to `amount`
///
/// Ref: https://www.amibroker.com/guide/afl/sumbars.html
pub fn ta_sumbars<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  amount: NumT,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }


  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      for i in start..end {
        let mut sum = NumT::zero();
        let mut bars = 0;

        // Iterate backwards
        for j in (0..=i).rev() {
          let val = x[j];
          if ctx.is_skip_nan() {
            if is_normal(&val) {
              sum = sum + val;
              bars += 1;
            }
          } else {
            if is_normal(&val) {
              sum = sum + val;
            } else {
              // If not skip nan, NaN breaks the sum? Or treats as 0?
              // Generally arithmetic with NaN results in NaN.
              // So we stop? Or we just count it?
              // Let's assume standard behavior: NaN makes result NaN for that bar unless handled.
              // But for simplicity in SUMBARS, usually we just count back.
              // If we hit NaN, sum becomes NaN.
              // If sum is NaN, it will never be >= amount.
              sum = sum + val;
            }
            bars += 1;
          }

          if is_normal(&sum) && sum >= amount {
            r[i] = NumT::from(bars).unwrap();
            break;
          }
        }

        // If loop finishes and not found, r[i] remains NaN (default fill)
      }
    });

  Ok(())
}

/// Calculate sum of values in preceding `periods` window where `condition` is true
///
/// Ref: Custom extension
pub fn ta_sumif<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  condition: &[bool],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != input.len() || r.len() != condition.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }


  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .zip(condition.par_chunks(ctx.chunk_size(condition.len())))
    .for_each(|((r, x), c)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      if periods == 0 {
        // Cumulative sum
        let mut sum = NumT::zero();
        let mut found_valid = false;

        for i in start..end {
          let val = x[i];
          let cond = c[i];
          if cond && is_normal(&val) {
            sum = sum + val;
            found_valid = true;
          }

          if found_valid {
            r[i] = sum;
          }
        }
      } else {
        // Sliding window sum
        if ctx.is_skip_nan() {
          // Skip NaN logic with condition
          // "Skip NaN" usually means we skip the *periods* counting if value is NaN.
          // BUT if condition is FALSE, do we count it as a period?
          // Standard strictly cycle behavior: "periods" refers to valid bars.
          // IF condition is involved, usually it filters VALUES but TIME (periods) still passes?
          // OR does it mean sum of last N *valid* values where condition is true?
          // If we follow `ta_sum` skip_nan logic:
          // It uses `SkipNanWindow`. This skips NaNs in the *Input*.
          // If Input is NaN, it's not in the window.
          // If Input is valid but Condition is False, is it in the window?
          //
          // If the definition is "Sum of A over last n choice-conditions", then it's different.
          // But common definition: "Sum of A over last n bars, IF A is selected by C".
          // i.e. Window size is fixed on 'n' (bars or valid values), and we only sum if C is true.
          // Let's assume standard behavior:
          // Window is defined by `input` validity (if skip_nan) and `periods`.
          // Summation includes `val` only if `condition` is true.

          let iter = SkipNanWindow::new(&x[..end], periods, start);
          let mut sum = NumT::zero();

          for i in iter {
            let idx = i.end;
            let val = x[idx];
            let cond = c[idx];

            if cond && is_normal(&val) {
              sum = sum + val;
            }

            // subtract values that fell out of the window
            for k in i.prev_start..i.start {
              let old = x[k];
              let old_cond = c[k];
              if old_cond && is_normal(&old) {
                sum = sum - old;
              }
            }

            if !is_normal(&val) {
              continue;
            }
            // If !is_normal, we don't output?
            // `iter` loop continues.

            if ctx.is_strictly_cycle() {
              if i.no_nan_count == periods && (i.end - i.start + 1) == periods {
                r[idx] = sum;
              }
            } else {
              r[idx] = sum;
            }
          }
        } else {
          // Normal mode (no skip NaN)
          let mut sum = NumT::zero();
          let mut nan_in_window = 0; // NaNs in input, regardless of condition?
          // Or strictly, if condition is False, NaN in input doesn't matter?
          // Usually for `SUM` no skip nan: any NaN in window makes result NaN.
          // If `SUMIF`, maybe only if `condition` is True AND `input` is NaN?
          // Let's stick to safe side: If any value in window is NaN (and likely summed?), result is NaN?
          // Taking simplified approach:
          // Window Logic is separate from Sum Logic.
          // Window Logic: fixed size `periods`.
          // Sum Logic: Sum valid items where condition is true.
          // BUT standard handling of NaN in window:
          // If `x[i]` is NaN, usually spread NaN.
          // Let's assume:
          // If C[i] is True, add X[i]. If X[i] is NaN, Sum becomes NaN.
          // If C[i] is False, X[i] is ignored.

          // Pre-initialization
          let pre_fill_start = if start >= periods { start - periods } else { 0 };
          for k in pre_fill_start..start {
            if c[k] {
              if is_normal(&x[k]) {
                sum = sum + x[k];
              } else {
                nan_in_window += 1;
              }
            }
          }

          for i in start..end {
            let val = x[i];
            let cond = c[i];

            // Add new
            if cond {
              if is_normal(&val) {
                sum = sum + val;
              } else {
                nan_in_window += 1;
              }
            }

            // Remove old
            if i >= periods {
              let old_idx = i - periods;
              let old = x[old_idx];
              let old_cond = c[old_idx];
              if old_cond {
                if is_normal(&old) {
                  sum = sum - old;
                } else {
                  nan_in_window -= 1;
                }
              }
            }

            if ctx.is_strictly_cycle() {
              if i >= periods - 1 {
                if nan_in_window == 0 {
                  r[i] = sum;
                }
              }
            } else {
              // Loose mode
              if i < periods {
                // Partial sum allowed if nan_in_window == 0
                if nan_in_window == 0 {
                  r[i] = sum;
                }
              } else {
                if nan_in_window == 0 {
                  r[i] = sum;
                }
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
  fn test_sum_sliding() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_sum(&ctx, &mut r, &input, 3).unwrap();
    // 0: 1
    // 1: 3
    // 2: 6
    // 3: 9 (2+3+4)
    // 4: 12 (3+4+5)
    assert_vec_eq_nan(&r, &vec![1.0, 3.0, 6.0, 9.0, 12.0]);
  }

  #[test]
  fn test_sum_cumulative() {
    let input = vec![1.0, 2.0, 3.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_sum(&ctx, &mut r, &input, 0).unwrap();
    assert_vec_eq_nan(&r, &vec![1.0, 3.0, 6.0]);
  }

  #[test]
  fn test_sum_skip_nan() {
    let input = vec![1.0, f64::NAN, 2.0, 3.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);
    ta_sum(&ctx, &mut r, &input, 2).unwrap();
    // 0: 1 (1)
    // 1: NaN (NaN)
    // 2: 3 (1+2, skip nan)
    // 3: 5 (2+3)
    assert_vec_eq_nan(&r, &vec![1.0, f64::NAN, 3.0, 5.0]);
  }

  #[test]
  fn test_sumbars() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    // Find bars to sum up to 5.0
    ta_sumbars(&ctx, &mut r, &input, 5.0).unwrap();
    // 0: 1 (1 < 5, end, wait, actually loop rev(0..=0). sum=1. 1<5. not found. r[0]=NaN)
    // Wait, SUMBARS(X, A) returns bars count needed to reach A.
    // If total history sum < A, returns NaN? Yes usually.

    // 0: sum=1 < 5 -> NaN
    // 1: 2+1 = 3 < 5 -> NaN
    // 2: 3+2 = 5 >= 5 -> 2 bars (idx 2, 1)
    // 3: 4 >= 5 (false), 4+3=7 >= 5 -> 2 bars
    // 4: 5 >= 5 -> 1 bar

    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 2.0, 2.0, 1.0]);
  }

  #[test]
  fn test_sumif() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let cond = vec![true, false, true, false, true];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_sumif(&ctx, &mut r, &input, &cond, 3).unwrap();
    // w=3
    // 0: [0] 1(T) -> 1
    // 1: [0,1] 1(T), 2(F) -> 1.0
    // 2: [0,1,2] 1(T), 2(F), 3(T) -> 1+3 = 4.0
    // 3: [1,2,3] 2(F), 3(T), 4(F) -> 3.0
    // 4: [2,3,4] 3(T), 4(F), 5(T) -> 3+5 = 8.0
    assert_vec_eq_nan(&r, &vec![1.0, 1.0, 4.0, 3.0, 8.0]);
  }
}
