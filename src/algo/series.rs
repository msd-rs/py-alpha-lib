// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal};

/// Right shift input array by `periods`, r[i] = input[i - periods]
///
/// Ref: https://www.amibroker.com/guide/afl/ref.html
pub fn ta_ref<NumT: Float + Send + Sync>(
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

      if ctx.is_skip_nan() {
        let mut history = std::collections::VecDeque::new();
        // pre-fill logic if needed? NO, simple scan.
        for i in start..end {
          let val = x[i];
          if is_normal(&val) {
            history.push_back(val);
            if history.len() > periods {
              let res = history.pop_front().unwrap();
              r[i] = res;
            }
          }
        }
      } else {
        // Normal mode
        for i in start..end {
          if i >= periods {
            r[i] = x[i - periods];
          }
        }
      }
    });

  Ok(())
}

/// Calculate number of bars since last condition true
///
/// Ref: https://www.amibroker.com/guide/afl/barslast.html
pub fn ta_barslast<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[bool],
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

      let mut last_idx: Option<usize> = None;

      for i in start..end {
        let is_true = x[i];

        if is_true {
          last_idx = Some(i);
          r[i] = NumT::zero();
        } else if let Some(idx) = last_idx {
          r[i] = NumT::from(i - idx).unwrap();
        }
      }
    });

  Ok(())
}

/// Calculate number of bars since first condition true
///
/// Ref: https://www.amibroker.com/guide/afl/barssince.html
pub fn ta_barssince<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[bool],
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

      let mut first_idx: Option<usize> = None;

      for i in start..end {
        let is_true = x[i];

        if first_idx.is_none() {
          if is_true {
            first_idx = Some(i);
            r[i] = NumT::zero();
          }
        } else {
          if let Some(idx) = first_idx {
            r[i] = NumT::from(i - idx).unwrap();
          }
        }
      }
    });

  Ok(())
}

/// Calculate number of periods where condition is true in passed `periods` window
///
/// Ref: https://www.amibroker.com/guide/afl/count.html
pub fn ta_count<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[bool],
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
        // Cumulative count
        let mut count = 0;
        for i in start..end {
          let is_true = x[i];
          if is_true {
            count += 1;
          }
          r[i] = NumT::from(count).unwrap();
        }
      } else {
        // Sliding window
        // For bool input, skip_nan doesn't really apply in the sense of invalid inputs.
        // BUT context might imply we want to skip something?
        // Usually bools are dense.
        // However, if we assume standard behavior:
        // "Strictly cycle" might still apply to the window size logic.

        let mut current_true_count = 0;
        let pre_fill_start = if start >= periods { start - periods } else { 0 };

        // Preload
        for k in pre_fill_start..start {
          if x[k] {
            current_true_count += 1;
          }
        }

        for i in start..end {
          // Add new
          if x[i] {
            current_true_count += 1;
          }

          // Remove old
          if i >= periods {
            let old_idx = i - periods;
            if x[old_idx] {
              current_true_count -= 1;
            }
          }

          if i >= start {
            let mut valid = true;
            if ctx.is_strictly_cycle() {
              if i < periods - 1 {
                valid = false;
              }
            }

            if valid {
              r[i] = NumT::from(current_true_count).unwrap();
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
  fn test_ref() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_ref(&ctx, &mut r, &input, 2).unwrap();
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 1.0, 2.0, 3.0]);
  }

  #[test]
  fn test_ref_skip_nan() {
    let input = vec![1.0, f64::NAN, 2.0, 3.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);
    ta_ref(&ctx, &mut r, &input, 1).unwrap();
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 1.0, 2.0]);
  }

  #[test]
  fn test_barslast() {
    let input = vec![false, true, false, false, true, false];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_barslast(&ctx, &mut r, &input).unwrap();
    assert_vec_eq_nan(&r, &vec![f64::NAN, 0.0, 1.0, 2.0, 0.0, 1.0]);
  }

  #[test]
  fn test_barssince() {
    let input = vec![false, true, false, false, true, false];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_barssince(&ctx, &mut r, &input).unwrap();
    assert_vec_eq_nan(&r, &vec![f64::NAN, 0.0, 1.0, 2.0, 3.0, 4.0]);
  }

  #[test]
  fn test_count() {
    let input = vec![true, false, true, true, false];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_count(&ctx, &mut r, &input, 3).unwrap();
    // 0: 1
    // 1: 1
    // 2: 2
    // 3: 2 (window 1..3: F T T -> 2)
    // 4: 2 (window 2..4: T T F -> 2)
    assert_vec_eq_nan(&r, &vec![1.0, 1.0, 2.0, 2.0, 2.0]);
  }

  #[test]
  fn test_count_cumulative() {
    let input = vec![true, false, true, true, false];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_count(&ctx, &mut r, &input, 0).unwrap();
    assert_vec_eq_nan(&r, &vec![1.0, 1.0, 2.0, 3.0, 3.0]);
  }

  #[test]
  fn test_count_strictly_cycle() {
    let input = vec![true, false, true];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, FLAG_STRICTLY_CYCLE);
    ta_count(&ctx, &mut r, &input, 3).unwrap();
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 2.0]);
  }
}
