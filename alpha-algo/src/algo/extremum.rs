// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use std::collections::VecDeque;

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal, skip_nan_window::SkipNanWindow};

/// Find highest value in a preceding `periods` window
///
/// Ref: https://www.amibroker.com/guide/afl/hhv.html
pub fn ta_hhv<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  run_extremum(ctx, r, input, periods, |a, b| a >= b, |_, _, val| val)
}

/// Find lowest value in a preceding `periods` window
///
/// Ref: https://www.amibroker.com/guide/afl/llv.html
pub fn ta_llv<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  run_extremum(ctx, r, input, periods, |a, b| a <= b, |_, _, val| val)
}

/// The number of periods that have passed since the array reached its `periods` period high
///
/// Ref: https://www.amibroker.com/guide/afl/hhvbars.html
pub fn ta_hhvbars<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  run_extremum(
    ctx,
    r,
    input,
    periods,
    |a, b| a >= b,
    |best_idx, curr_idx, _| NumT::from(curr_idx - best_idx).unwrap(),
  )
}

/// The number of periods that have passed since the array reached its periods period low
///
/// Ref: https://www.amibroker.com/guide/afl/llvbars.html
pub fn ta_llvbars<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  run_extremum(
    ctx,
    r,
    input,
    periods,
    |a, b| a <= b,
    |best_idx, curr_idx, _| NumT::from(curr_idx - best_idx).unwrap(),
  )
}

fn run_extremum<NumT, FComp, FOut>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
  compare: FComp,
  output: FOut,
) -> Result<(), Error>
where
  NumT: Float + Send + Sync,
  FComp: Fn(NumT, NumT) -> bool + Sync + Send,
  FOut: Fn(usize, usize, NumT) -> NumT + Sync + Send,
{
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
        // Cumulative mode
        let mut best_val: Option<NumT> = None;
        let mut best_idx: usize = 0;
        let mut has_nan_poison = false;

        for i in start..end {
          let val = x[i];
          if !is_normal(&val) {
            if !ctx.is_skip_nan() {
              has_nan_poison = true;
            }
            // If skip_nan is true, just ignore this value
            // If skip_nan is false, we set poison flag
          } else {
            // Valid value
            match best_val {
              None => {
                best_val = Some(val);
                best_idx = i;
              }
              Some(bv) => {
                if compare(val, bv) {
                  best_val = Some(val);
                  best_idx = i;
                }
              }
            }
          }

          if has_nan_poison {
            // Leave r[i] as NaN
            continue;
          }

          if let Some(bv) = best_val {
            r[i] = output(best_idx, i, bv);
          }
        }
      } else {
        // Sliding window mode
        // Logic similar within SkipNanWindow usage or manual loop
        let mut deque: VecDeque<usize> = VecDeque::new();

        if ctx.is_skip_nan() {
          let iter = SkipNanWindow::new(&x[..end], periods, start);
          for i in iter {
            let curr_val = x[i.end];
            if is_normal(&curr_val) {
              while let Some(&back_idx) = deque.back() {
                if compare(curr_val, x[back_idx]) {
                  deque.pop_back();
                } else {
                  break;
                }
              }
              deque.push_back(i.end);
            }

            // Remove indices that are strictly before the current window start
            // i.start is the first valid index in the window.
            while let Some(&front_idx) = deque.front() {
              if front_idx < i.start {
                deque.pop_front();
              } else {
                break;
              }
            }

            if !is_normal(&curr_val) {
              continue;
            }

            if ctx.is_strictly_cycle() {
              // Requires exactly `periods` valid numbers and compact window?
              // According to ma.rs logic:
              if i.no_nan_count == periods && (i.end - i.start + 1) == periods {
                if let Some(&best_idx) = deque.front() {
                  r[i.end] = output(best_idx, i.end, x[best_idx]);
                }
              }
            } else {
              // Normal skip_nan behavior: valid count <= periods
              // For min periods check? Usually partial windows are allowed at start if periods > i+1?
              // However, SkipNanWindow logic usually results in valid windows.
              // If we have at least 1 valid value?
              if let Some(&best_idx) = deque.front() {
                r[i.end] = output(best_idx, i.end, x[best_idx]);
              }
            }
          }
        } else {
          // !skip_nan
          let mut nan_in_window = 0;

          // Pre-scan for NaNs if start > 0 is tricky without window context.
          // But we assume start is beginning of relevance or we just start fresh.
          // To be consistent with ma.rs, we need to track NaNs in the moving window properly.

          // Replicating ma.rs logic structure
          let pre_fill_start = if start >= periods { start - periods } else { 0 };

          // Fill initial context for naive window tracking
          for k in pre_fill_start..start {
            if !is_normal(&x[k]) {
              nan_in_window += 1;
            }
          }

          for i in pre_fill_start..end {
            let val = x[i];
            let is_valid = is_normal(&val);

            // Add new value
            if is_valid {
              while let Some(&back_idx) = deque.back() {
                if compare(val, x[back_idx]) {
                  deque.pop_back();
                } else {
                  break;
                }
              }
              deque.push_back(i);
            } else {
              nan_in_window += 1;
            }

            // Remove old value
            if i >= periods {
              let falling_out_idx = i - periods;
              if !is_normal(&x[falling_out_idx]) {
                nan_in_window -= 1;
              }
              // Remove from deque if falling out
              if let Some(&front_idx) = deque.front() {
                if front_idx <= falling_out_idx {
                  deque.pop_front();
                }
              }
            }

            // Only output if we are in valid range
            if i >= start {
              // Logic for output
              if !is_valid {
                continue;
              }

              let can_write = if ctx.is_strictly_cycle() {
                if i >= periods - 1 {
                  nan_in_window == 0
                } else {
                  false
                }
              } else {
                // !strictly_cycle, !skip_nan
                // If nan_in_window > 0, we can't produce result (it's NaN)
                nan_in_window == 0
              };

              if can_write {
                if let Some(&best_idx) = deque.front() {
                  r[i] = output(best_idx, i, x[best_idx]);
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
  use crate::algo::{
    assert_vec_eq_nan,
    context::{FLAG_SKIP_NAN, FLAG_STRICTLY_CYCLE},
  };

  #[test]
  fn test_hhv_simple() {
    let input = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);

    ta_hhv(&ctx, &mut r, &input, 3).unwrap();
    // 1: 1
    // 3: 3
    // 2: 3 (1,3,2) -> 3
    // 5: 5 (3,2,5)
    // 4: 5 (2,5,4)
    // 6: 6 (5,4,6)
    assert_vec_eq_nan(&r, &vec![1.0, 3.0, 3.0, 5.0, 5.0, 6.0]);
  }

  #[test]
  fn test_hhvbars_simple() {
    let input = vec![10.0, 12.0, 11.0, 13.0, 8.0];
    // 0: 10 -> 0
    // 1: 12 -> 0
    // 2: 12 (10,12,11) -> idx 1 is max. current 2. diff 1.
    // 3: 13 -> idx 3 is max. diff 0.
    // 4: 13 (12,11,13,8) NO window 3. (11,13,8). Max 13 idx 3. curr 4. diff 1.
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_hhvbars(&ctx, &mut r, &input, 3).unwrap();
    assert_vec_eq_nan(&r, &vec![0.0, 0.0, 1.0, 0.0, 1.0]);
  }

  #[test]
  fn test_llv_simple() {
    let input = vec![5.0, 3.0, 4.0, 1.0, 2.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_llv(&ctx, &mut r, &input, 3).unwrap();
    assert_vec_eq_nan(&r, &vec![5.0, 3.0, 3.0, 1.0, 1.0]);
  }

  #[test]
  fn test_cumulative_n0() {
    let input = vec![1.0, 5.0, 3.0, 6.0, 2.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_hhv(&ctx, &mut r, &input, 0).unwrap();
    assert_vec_eq_nan(&r, &vec![1.0, 5.0, 5.0, 6.0, 6.0]);

    let mut r_bars = vec![0.0; input.len()];
    ta_hhvbars(&ctx, &mut r_bars, &input, 0).unwrap();
    // 0: 1(0) -> 0
    // 1: 5(1) -> 0
    // 2: 5(1) -> 2-1=1
    // 3: 6(3) -> 0
    // 4: 6(3) -> 4-3=1
    assert_vec_eq_nan(&r_bars, &vec![0.0, 0.0, 1.0, 0.0, 1.0]);
  }

  #[test]
  fn test_skip_nan() {
    let input = vec![1.0, 2.0, f64::NAN, 4.0, 0.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);
    // 3 periods valid
    // 0: 1 -> 1
    // 1: 2 -> 2
    // 2: NaN
    // 3: 4 -> 4. window (1,2,4)
    // 4: 0 -> 4. window (2,4,0)
    ta_hhv(&ctx, &mut r, &input, 3).unwrap();
    assert_vec_eq_nan(&r, &vec![1.0, 2.0, f64::NAN, 4.0, 4.0]);
  }

  #[test]
  fn test_strict_cycle() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, FLAG_STRICTLY_CYCLE);
    // periods 3.
    // 0: NaN
    // 1: NaN
    // 2: 3.0 (1,2,3)
    // 3: 4.0 (2,3,4)
    ta_hhv(&ctx, &mut r, &input, 3).unwrap();
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 3.0, 4.0]);
  }
}
