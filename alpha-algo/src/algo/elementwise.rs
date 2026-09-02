// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal};

/// Calculate absolute value of input elements
///
/// ABS(x) = |x|
pub fn ta_abs<NumT: Float + Send + Sync>(
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
      r.fill(NumT::nan());

      for i in start..end {
        let val = x[i];
        if is_normal(&val) {
          r[i] = val.abs();
        }
      }
    });

  Ok(())
}

/// Calculate element-wise maximum of two arrays
///
/// MAX(a, b) = max(a, b)
pub fn ta_max<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  a: &[NumT],
  b: &[NumT],
) -> Result<(), Error> {
  if r.len() != a.len() || r.len() != b.len() {
    return Err(Error::LengthMismatch(r.len(), a.len()));
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(a.par_chunks(ctx.chunk_size(a.len())))
    .zip(b.par_chunks(ctx.chunk_size(b.len())))
    .for_each(|((r, a), b)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      if ctx.is_skip_nan() {
        for i in start..end {
          let val_a = a[i];
          let val_b = b[i];
          let a_ok = is_normal(&val_a);
          let b_ok = is_normal(&val_b);
          if a_ok && b_ok {
            r[i] = val_a.max(val_b);
          } else if a_ok {
            r[i] = val_a;
          } else if b_ok {
            r[i] = val_b;
          }
        }
      } else {
        for i in start..end {
          let val_a = a[i];
          let val_b = b[i];
          if is_normal(&val_a) && is_normal(&val_b) {
            r[i] = val_a.max(val_b);
          }
        }
      }
    });

  Ok(())
}

/// Calculate element-wise minimum of two arrays
///
/// MIN(a, b) = min(a, b)
pub fn ta_min<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  a: &[NumT],
  b: &[NumT],
) -> Result<(), Error> {
  if r.len() != a.len() || r.len() != b.len() {
    return Err(Error::LengthMismatch(r.len(), a.len()));
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(a.par_chunks(ctx.chunk_size(a.len())))
    .zip(b.par_chunks(ctx.chunk_size(b.len())))
    .for_each(|((r, a), b)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      if ctx.is_skip_nan() {
        for i in start..end {
          let val_a = a[i];
          let val_b = b[i];
          let a_ok = is_normal(&val_a);
          let b_ok = is_normal(&val_b);
          if a_ok && b_ok {
            r[i] = val_a.min(val_b);
          } else if a_ok {
            r[i] = val_a;
          } else if b_ok {
            r[i] = val_b;
          }
        }
      } else {
        for i in start..end {
          let val_a = a[i];
          let val_b = b[i];
          if is_normal(&val_a) && is_normal(&val_b) {
            r[i] = val_a.min(val_b);
          }
        }
      }
    });

  Ok(())
}

/// Choose elements from a or b depending on condition
///
/// WHERE(cond, a, b) = a if cond else b
pub fn ta_where<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  condition: &[bool],
  a: &[NumT],
  b: &[NumT],
) -> Result<(), Error> {
  if r.len() != condition.len() || r.len() != a.len() || r.len() != b.len() {
    return Err(Error::LengthMismatch(r.len(), condition.len()));
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(condition.par_chunks(ctx.chunk_size(condition.len())))
    .zip(a.par_chunks(ctx.chunk_size(a.len())))
    .zip(b.par_chunks(ctx.chunk_size(b.len())))
    .for_each(|(((r, c), a), b)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      for i in start..end {
        r[i] = if c[i] { a[i] } else { b[i] };
      }
    });

  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::algo::{assert_vec_eq_nan, context::FLAG_SKIP_NAN};

  #[test]
  fn test_abs() {
    let input = vec![-1.5, 2.0, -0.0, 0.0, f64::NAN, -3.7];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_abs(&ctx, &mut r, &input).unwrap();
    assert_vec_eq_nan(&r, &vec![1.5, 2.0, 0.0, 0.0, f64::NAN, 3.7]);
  }

  #[test]
  fn test_abs_groups_and_slice() {
    let input = vec![-1.0, -2.0, -3.0, -4.0];
    let mut r = vec![0.0; 4];
    let ctx = Context::new(1, 2, 0); // start at index 1 of each group of size 2
    ta_abs(&ctx, &mut r, &input).unwrap();
    // Group 1: [-1.0, -2.0] -> start 1 -> [NAN, 2.0]
    // Group 2: [-3.0, -4.0] -> start 1 -> [NAN, 4.0]
    assert_vec_eq_nan(&r, &vec![f64::NAN, 2.0, f64::NAN, 4.0]);
  }

  #[test]
  fn test_max() {
    let a = vec![1.0, 5.0, -2.0, f64::NAN, 3.0];
    let b = vec![2.0, 3.0, -1.0, 4.0, f64::NAN];
    let mut r = vec![0.0; a.len()];
    let ctx = Context::new(0, 0, 0);
    ta_max(&ctx, &mut r, &a, &b).unwrap();
    assert_vec_eq_nan(&r, &vec![2.0, 5.0, -1.0, f64::NAN, f64::NAN]);

    // Test with FLAG_SKIP_NAN
    let ctx_skip = Context::new(0, 0, FLAG_SKIP_NAN);
    ta_max(&ctx_skip, &mut r, &a, &b).unwrap();
    assert_vec_eq_nan(&r, &vec![2.0, 5.0, -1.0, 4.0, 3.0]);
  }

  #[test]
  fn test_min() {
    let a = vec![1.0, 5.0, -2.0, f64::NAN, 3.0];
    let b = vec![2.0, 3.0, -1.0, 4.0, f64::NAN];
    let mut r = vec![0.0; a.len()];
    let ctx = Context::new(0, 0, 0);
    ta_min(&ctx, &mut r, &a, &b).unwrap();
    assert_vec_eq_nan(&r, &vec![1.0, 3.0, -2.0, f64::NAN, f64::NAN]);

    // Test with FLAG_SKIP_NAN
    let ctx_skip = Context::new(0, 0, FLAG_SKIP_NAN);
    ta_min(&ctx_skip, &mut r, &a, &b).unwrap();
    assert_vec_eq_nan(&r, &vec![1.0, 3.0, -2.0, 4.0, 3.0]);
  }

  #[test]
  fn test_where() {
    let cond = vec![true, false, true, false, true];
    let a = vec![10.0, 20.0, 30.0, 40.0, f64::NAN];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut r = vec![0.0; cond.len()];
    let ctx = Context::new(0, 0, 0);
    ta_where(&ctx, &mut r, &cond, &a, &b).unwrap();
    assert_vec_eq_nan(&r, &vec![10.0, 2.0, 30.0, 4.0, f64::NAN]);
  }

  #[test]
  fn test_length_mismatch() {
    let ctx = Context::new(0, 0, 0);
    let mut r = vec![0.0; 3];
    let a = vec![1.0, 2.0];
    let b = vec![1.0, 2.0, 3.0];
    let cond = vec![true, false, true];

    assert!(matches!(ta_abs(&ctx, &mut r, &a), Err(Error::LengthMismatch(..))));
    assert!(matches!(ta_max(&ctx, &mut r, &a, &b), Err(Error::LengthMismatch(..))));
    assert!(matches!(ta_min(&ctx, &mut r, &a, &b), Err(Error::LengthMismatch(..))));
    assert!(matches!(ta_where(&ctx, &mut r, &cond, &a, &b), Err(Error::LengthMismatch(..))));
  }
}
