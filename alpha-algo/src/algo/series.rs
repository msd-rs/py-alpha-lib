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
          } else if !ctx.is_strictly_cycle() {
            r[i] = x[0];
          }
        }
      }
    });

  Ok(())
}

/// Right shift input array by variable `periods`, r[i] = input[i - periods[i]]
///
/// Ref: https://www.amibroker.com/guide/afl/ref.html
pub fn ta_ref_v<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: &[usize],
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }
  if r.len() != periods.len() {
    return Err(Error::LengthMismatch(r.len(), periods.len()));
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .zip(periods.par_chunks(ctx.chunk_size(periods.len())))
    .for_each(|((r, x), p)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      if ctx.is_skip_nan() {
        let mut history = Vec::with_capacity(r.len());
        for i in start..end {
          let val = x[i];
          if is_normal(&val) {
            history.push(val);
            let period = p[i];
            if history.len() > period {
              r[i] = history[history.len() - 1 - period];
            }
          }
        }
      } else {
        // Normal mode
        for i in start..end {
          let period = p[i];
          if i >= period {
            r[i] = x[i - period];
          } else if !ctx.is_strictly_cycle() {
            r[i] = x[0];
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

/// Calculate number of periods where condition is true in passed variable `periods` window
///
/// Ref: https://www.amibroker.com/guide/afl/count.html
pub fn ta_count_v<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[bool],
  periods: &[usize],
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }
  if r.len() != periods.len() {
    return Err(Error::LengthMismatch(r.len(), periods.len()));
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .zip(periods.par_chunks(ctx.chunk_size(periods.len())))
    .for_each(|((r, x), p)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      let mut pref = vec![0; r.len() + 1];
      for k in 0..r.len() {
        pref[k + 1] = pref[k] + (if x[k] { 1 } else { 0 });
      }

      for i in start..end {
        let period = p[i];
        let start_idx = if period == 0 {
          0
        } else {
          if i >= period { i + 1 - period } else { 0 }
        };

        let mut can_write = false;
        if period == 0 {
          can_write = true;
        } else {
          if ctx.is_strictly_cycle() {
            if i >= period - 1 {
              can_write = true;
            }
          } else {
            can_write = true;
          }
        }

        if can_write {
          r[i] = NumT::from(pref[i + 1] - pref[start_idx]).unwrap();
        }
      }
    });

  Ok(())
}

/// Check if condition is always true within moving window of `periods` (or cumulative if `periods` is 0)
///
/// ALL(cond, periods)
pub fn ta_all(
  ctx: &Context,
  r: &mut [bool],
  condition: &[bool],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != condition.len() {
    return Err(Error::LengthMismatch(r.len(), condition.len()));
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(condition.par_chunks(ctx.chunk_size(condition.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(false);

      if periods == 0 {
        let mut all_true = true;
        for i in 0..start {
          if !x[i] {
            all_true = false;
          }
        }
        for i in start..end {
          if !x[i] {
            all_true = false;
          }
          r[i] = all_true;
        }
      } else {
        let mut false_count = 0;
        let pre_start = if start >= periods { start - periods } else { 0 };
        for i in pre_start..start {
          if !x[i] {
            false_count += 1;
          }
        }

        for i in start..end {
          if !x[i] {
            false_count += 1;
          }

          if i >= periods {
            let old = x[i - periods];
            if !old {
              false_count -= 1;
            }
          }

          if ctx.is_strictly_cycle() {
            if i >= periods - 1 && false_count == 0 {
              r[i] = true;
            }
          } else if false_count == 0 {
            r[i] = true;
          }
        }
      }
    });

  Ok(())
}

/// Check if condition is true at least once within moving window of `periods` (or cumulative if `periods` is 0)
///
/// ANY(cond, periods)
pub fn ta_any(
  ctx: &Context,
  r: &mut [bool],
  condition: &[bool],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != condition.len() {
    return Err(Error::LengthMismatch(r.len(), condition.len()));
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(condition.par_chunks(ctx.chunk_size(condition.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(false);

      if periods == 0 {
        let mut any_true = false;
        for i in 0..start {
          if x[i] {
            any_true = true;
          }
        }
        for i in start..end {
          if x[i] {
            any_true = true;
          }
          r[i] = any_true;
        }
      } else {
        let mut true_count = 0;
        let pre_start = if start >= periods { start - periods } else { 0 };
        for i in pre_start..start {
          if x[i] {
            true_count += 1;
          }
        }

        for i in start..end {
          if x[i] {
            true_count += 1;
          }

          if i >= periods {
            let old = x[i - periods];
            if old {
              true_count -= 1;
            }
          }

          if ctx.is_strictly_cycle() {
            if i >= periods - 1 && true_count > 0 {
              r[i] = true;
            }
          } else if true_count > 0 {
            r[i] = true;
          }
        }
      }
    });

  Ok(())
}

/// If condition is true, set current position and previous `periods - 1` periods (total `periods` bars) to true
///
/// BACKSET(cond, periods)
pub fn ta_backset(
  ctx: &Context,
  r: &mut [bool],
  condition: &[bool],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != condition.len() {
    return Err(Error::LengthMismatch(r.len(), condition.len()));
  }

  if periods == 0 {
    r.fill(false);
    return Ok(());
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(condition.par_chunks(ctx.chunk_size(condition.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(false);

      if end <= start {
        return;
      }

      let mut remaining_backset = 0;
      for i in (start..end).rev() {
        if x[i] {
          remaining_backset = remaining_backset.max(periods);
        }
        if remaining_backset > 0 {
          r[i] = true;
          remaining_backset -= 1;
        }
      }
    });

  Ok(())
}

/// Filter consecutive signals: once condition is true, set subsequent `periods` periods to false
///
/// FILTER(cond, periods)
pub fn ta_filter(
  ctx: &Context,
  r: &mut [bool],
  condition: &[bool],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != condition.len() {
    return Err(Error::LengthMismatch(r.len(), condition.len()));
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(condition.par_chunks(ctx.chunk_size(condition.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(false);

      let mut cool_down = 0;
      for i in 0..start {
        if cool_down > 0 {
          cool_down -= 1;
        } else if x[i] {
          cool_down = if periods == 0 { usize::MAX } else { periods };
        }
      }

      for i in start..end {
        if cool_down > 0 {
          r[i] = false;
          cool_down -= 1;
        } else if x[i] {
          r[i] = true;
          cool_down = if periods == 0 { usize::MAX } else { periods };
        } else {
          r[i] = false;
        }
      }
    });

  Ok(())
}

/// Count consecutive periods where condition is true up to the current bar
///
/// LAST(cond)
pub fn ta_last<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  condition: &[bool],
) -> Result<(), Error> {
  if r.len() != condition.len() {
    return Err(Error::LengthMismatch(r.len(), condition.len()));
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(condition.par_chunks(ctx.chunk_size(condition.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      let mut count = 0;
      for i in 0..start {
        if x[i] {
          count += 1;
        } else {
          count = 0;
        }
      }

      for i in start..end {
        if x[i] {
          count += 1;
        } else {
          count = 0;
        }
        r[i] = NumT::from(count).unwrap();
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
  fn test_ref_v() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let periods = vec![1, 2, 1, 3, 2];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_ref_v(&ctx, &mut r, &input, &periods).unwrap();
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 2.0, 1.0, 3.0]);
  }

  #[test]
  fn test_ref_v_skip_nan() {
    let input = vec![1.0, f64::NAN, 2.0, 3.0, 4.0];
    let periods = vec![1, 1, 2, 1, 3];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);
    ta_ref_v(&ctx, &mut r, &input, &periods).unwrap();
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, f64::NAN, 2.0, 1.0]);
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

  #[test]
  fn test_ta_count_v() {
    let input = vec![true, false, true, true, false];
    let periods = vec![3, 3, 3, 3, 3];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_count_v(&ctx, &mut r, &input, &periods).unwrap();
    assert_vec_eq_nan(&r, &vec![1.0, 1.0, 2.0, 2.0, 2.0]);
  }

  #[test]
  fn test_ta_all() {
    let input = vec![true, true, true, false, true, true];
    let mut r = vec![false; input.len()];
    let ctx = Context::new(0, 0, 0);

    // Window = 3
    // i=0: [T] -> T
    // i=1: [T,T] -> T
    // i=2: [T,T,T] -> T
    // i=3: [T,T,F] -> F
    // i=4: [T,F,T] -> F
    // i=5: [F,T,T] -> F
    ta_all(&ctx, &mut r, &input, 3).unwrap();
    assert_eq!(r, vec![true, true, true, false, false, false]);

    // Cumulative (periods = 0)
    ta_all(&ctx, &mut r, &input, 0).unwrap();
    assert_eq!(r, vec![true, true, true, false, false, false]);

    // strictly cycle
    let ctx_strict = Context::new(0, 0, FLAG_STRICTLY_CYCLE);
    ta_all(&ctx_strict, &mut r, &input, 3).unwrap();
    assert_eq!(r, vec![false, false, true, false, false, false]);
  }

  #[test]
  fn test_ta_any() {
    let input = vec![false, false, true, false, false];
    let mut r = vec![false; input.len()];
    let ctx = Context::new(0, 0, 0);

    // Window = 2
    // i=0: [F] -> F
    // i=1: [F,F] -> F
    // i=2: [F,T] -> T
    // i=3: [T,F] -> T
    // i=4: [F,F] -> F
    ta_any(&ctx, &mut r, &input, 2).unwrap();
    assert_eq!(r, vec![false, false, true, true, false]);

    // Cumulative (periods = 0)
    ta_any(&ctx, &mut r, &input, 0).unwrap();
    assert_eq!(r, vec![false, false, true, true, true]);
  }

  #[test]
  fn test_ta_backset() {
    let input = vec![false, false, true, false, false];
    let mut r = vec![false; input.len()];
    let ctx = Context::new(0, 0, 0);

    // periods = 3 -> sets index 2, 1, 0 to true
    ta_backset(&ctx, &mut r, &input, 3).unwrap();
    assert_eq!(r, vec![true, true, true, false, false]);

    // periods = 2 on multiple signals
    let input2 = vec![true, false, false, true, false];
    let mut r2 = vec![false; input2.len()];
    ta_backset(&ctx, &mut r2, &input2, 2).unwrap();
    // i=0 (T) -> [0]
    // i=3 (T) -> [2, 3]
    assert_eq!(r2, vec![true, false, true, true, false]);
  }

  #[test]
  fn test_ta_filter() {
    let input = vec![true, true, false, true, false, true];
    let mut r = vec![false; input.len()];
    let ctx = Context::new(0, 0, 0);

    // periods = 2
    // i=0: T -> r[0]=T, cool_down=2
    // i=1: T -> cool_down(2)>0 -> r[1]=F, cool_down=1
    // i=2: F -> cool_down(1)>0 -> r[2]=F, cool_down=0
    // i=3: T -> r[3]=T, cool_down=2
    // i=4: F -> cool_down(2)>0 -> r[4]=F, cool_down=1
    // i=5: T -> cool_down(1)>0 -> r[5]=F, cool_down=0
    ta_filter(&ctx, &mut r, &input, 2).unwrap();
    assert_eq!(r, vec![true, false, false, true, false, false]);

    // periods = 0 (filter all subsequent)
    ta_filter(&ctx, &mut r, &input, 0).unwrap();
    assert_eq!(r, vec![true, false, false, false, false, false]);
  }

  #[test]
  fn test_ta_last() {
    let input = vec![false, true, true, false, true, true, true];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);

    ta_last(&ctx, &mut r, &input).unwrap();
    assert_vec_eq_nan(&r, &vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0]);
  }
}
