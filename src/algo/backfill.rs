// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal};

/// Forward-fill NaN values with the last valid observation
///
/// Iterates forward through each group; if x[i] is NaN, copies the last valid value.
/// Leading NaNs (before any valid value) remain NaN.
pub fn ta_backfill<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }


  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r[..start].fill(NumT::nan());

      let mut last_valid = NumT::nan();

      for i in start..end {
        let val = x[i];
        if is_normal(&val) {
          last_valid = val;
          r[i] = val;
        } else if is_normal(&last_valid) {
          r[i] = last_valid;
        } else {
          r[i] = NumT::nan();
        }
      }
    });

  Ok(())
}

/// Count number of NaN values in a rolling window
///
/// For each position, counts the number of NaN values in the preceding `periods` elements.
pub fn ta_count_nans<NumT: Float + Send + Sync>(
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

      let mut nan_count: usize = 0;

      let pre_fill_start = if start >= periods { start - periods } else { 0 };

      for k in pre_fill_start..start {
        if !is_normal(&x[k]) {
          nan_count += 1;
        }
      }

      for i in start..end {
        let val = x[i];
        if !is_normal(&val) {
          nan_count += 1;
        }

        if i >= periods {
          let old = x[i - periods];
          if !is_normal(&old) {
            nan_count -= 1;
          }
        }

        if i >= periods - 1 {
          r[i] = NumT::from(nan_count).unwrap();
        } else if !ctx.is_strictly_cycle() {
          r[i] = NumT::from(nan_count).unwrap();
        }
      }
    });

  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::algo::assert_vec_eq_nan;

  #[test]
  fn test_backfill_simple() {
    let input = vec![1.0, f64::NAN, f64::NAN, 4.0, f64::NAN];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_backfill(&ctx, &mut r, &input).unwrap();

    assert_vec_eq_nan(&r, &vec![1.0, 1.0, 1.0, 4.0, 4.0]);
  }

  #[test]
  fn test_backfill_leading_nan() {
    let input = vec![f64::NAN, f64::NAN, 3.0, f64::NAN];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_backfill(&ctx, &mut r, &input).unwrap();

    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 3.0, 3.0]);
  }

  #[test]
  fn test_backfill_no_nan() {
    let input = vec![1.0, 2.0, 3.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_backfill(&ctx, &mut r, &input).unwrap();

    assert_vec_eq_nan(&r, &vec![1.0, 2.0, 3.0]);
  }

  #[test]
  fn test_count_nans_simple() {
    let input = vec![1.0, f64::NAN, 3.0, f64::NAN, f64::NAN];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_count_nans(&ctx, &mut r, &input, periods).unwrap();

    // 0: [1] -> 0
    // 1: [1, NaN] -> 1
    // 2: [1, NaN, 3] -> 1
    // 3: [NaN, 3, NaN] -> 2
    // 4: [3, NaN, NaN] -> 2
    assert_vec_eq_nan(&r, &vec![0.0, 1.0, 1.0, 2.0, 2.0]);
  }

  #[test]
  fn test_count_nans_no_nan() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_count_nans(&ctx, &mut r, &input, periods).unwrap();

    assert_vec_eq_nan(&r, &vec![0.0, 0.0, 0.0, 0.0]);
  }

  #[test]
  fn test_count_nans_all_nan() {
    let input = vec![f64::NAN, f64::NAN, f64::NAN];
    let periods = 2;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_count_nans(&ctx, &mut r, &input, periods).unwrap();

    assert_vec_eq_nan(&r, &vec![1.0, 2.0, 2.0]);
  }
}
