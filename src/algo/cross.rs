// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal};

/// For 2 arrays A and B, return true if A[i-1] < B[i-1] and A[i] >= B[i]
/// alias: golden_cross, cross_ge
///
pub fn ta_cross<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [bool],
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
      r.fill(false);

      if ctx.is_skip_nan() {
        // Skip NaN logic
        // We need to track the LAST VALID comparison state.
        // If last valid A < B, and current A >= B (and valid), then true.
        // We maintain `was_less`: Option<bool>
        // None: no valid previous state
        // Some(true): last valid A < B
        // Some(false): last valid A >= B

        let mut was_less: Option<bool> = None;

        for i in start..end {
          let val_a = a[i];
          let val_b = b[i];

          if is_normal(&val_a) && is_normal(&val_b) {
            let is_less = val_a < val_b;
            let is_ge = val_a >= val_b;

            if let Some(prev_less) = was_less {
              if prev_less && is_ge {
                r[i] = true;
              }
            }

            was_less = Some(is_less);
          }
        }
      } else {
        // Normal logic (look at i-1)
        for i in start..end {
          if i == 0 {
            continue;
          }

          let curr_a = a[i];
          let curr_b = b[i];
          let prev_a = a[i - 1];
          let prev_b = b[i - 1];

          if is_normal(&curr_a) && is_normal(&curr_b) && is_normal(&prev_a) && is_normal(&prev_b) {
            if prev_a < prev_b && curr_a >= curr_b {
              r[i] = true;
            }
          }
        }
      }
    });

  Ok(())
}

/// For 2 arrays A and B, return true if A[i-1] > B[i-1] and A[i] <= B[i]
/// alias: death_cross, cross_le
pub fn ta_rcross<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [bool],
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
      r.fill(false);

      if ctx.is_skip_nan() {
        let mut was_greater: Option<bool> = None;

        for i in start..end {
          let val_a = a[i];
          let val_b = b[i];

          if is_normal(&val_a) && is_normal(&val_b) {
            let is_greater = val_a > val_b;
            let is_le = val_a <= val_b;

            if let Some(prev_greater) = was_greater {
              if prev_greater && is_le {
                r[i] = true;
              }
            }

            was_greater = Some(is_greater);
          }
        }
      } else {
        for i in start..end {
          if i == 0 {
            continue;
          }

          let curr_a = a[i];
          let curr_b = b[i];
          let prev_a = a[i - 1];
          let prev_b = b[i - 1];

          if is_normal(&curr_a) && is_normal(&curr_b) && is_normal(&prev_a) && is_normal(&prev_b) {
            if prev_a > prev_b && curr_a <= curr_b {
              r[i] = true;
            }
          }
        }
      }
    });

  Ok(())
}

/// For 2 arrays A and B, return true if previous N periods A < B, Current A >= B
pub fn ta_longcross<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [bool],
  a: &[NumT],
  b: &[NumT],
  n: usize,
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
      r.fill(false);

      if n == 0 {
        // If N=0, "Previous 0 A < B" is vacuously true? Or doesn't make sense?
        // "Up to 0 periods ago" usually means just check current?
        // But strict reading: "Previous N A < B" implies the range [i-N, i-1].
        // If N=0, range is empty. Empty range condition is true?
        // If so, it becomes just Current A >= B?
        // Let's assume if N=0 it requires at least 1 previous period?
        // Usually N >= 1 for such funcs.
        // If N=0, let's treat as just A >= B
        for i in start..end {
          if is_normal(&a[i]) && is_normal(&b[i]) {
            if a[i] >= b[i] {
              r[i] = true;
            }
          }
        }
        return;
      }

      if ctx.is_skip_nan() {
        // Maintaining a history of comparisons is complex with skip_nan and arbitrary N.
        // We can maintain a count of consecutive "A < B" events.
        // `less_count`.
        // If A < B, less_count++.
        // If A >= B:
        //    Check if less_count >= N. If so, and we just switched, is this the switch?
        //    Wait, "Previous N A < B" means *exactly* the last N periods were A < B.
        //    It doesn't say "At least N". But usually implies "Last N periods".
        //    If we had N+1 periods of A < B, does that satisfy "Previous N A < B"?
        //    Usually yes.
        //    So we need consecutive count of (A < B) >= N immediately before current.

        let mut less_count = 0;

        for i in start..end {
          let val_a = a[i];
          let val_b = b[i];

          if is_normal(&val_a) && is_normal(&val_b) {
            if val_a < val_b {
              less_count += 1;
            } else {
              // A >= B
              if less_count >= n {
                r[i] = true;
              }
              less_count = 0; // Reset
            }
          }
        }
      } else {
        // Normal mode
        // Check window [i-N .. i-1] for A < B
        // And i for A >= B

        // We can optimize with a counter too.
        let mut less_count = 0;

        // Pre-fill counter if possible (start > 0)
        // Optimization: just run from loop
        let pre_start = if start > n { start - n } else { 0 };
        for i in pre_start..start {
          if is_normal(&a[i]) && is_normal(&b[i]) && a[i] < b[i] {
            less_count += 1;
          } else {
            less_count = 0;
          }
        }

        for i in start..end {
          let curr_a = a[i];
          let curr_b = b[i];

          if is_normal(&curr_a) && is_normal(&curr_b) {
            if curr_a >= curr_b {
              if less_count >= n {
                r[i] = true;
              }
              less_count = 0;
            } else {
              // curr_a < curr_b
              less_count += 1;
            }
          } else {
            less_count = 0; // reset on nan
          }
        }
      }
    });

  Ok(())
}

/// For 2 arrays A and B, return true if previous N periods A > B, Current A <= B
pub fn ta_rlongcross<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [bool],
  a: &[NumT],
  b: &[NumT],
  n: usize,
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
      r.fill(false);

      if n == 0 {
        for i in start..end {
          if is_normal(&a[i]) && is_normal(&b[i]) {
            if a[i] <= b[i] {
              r[i] = true;
            }
          }
        }
        return;
      }

      if ctx.is_skip_nan() {
        let mut greater_count = 0;

        for i in start..end {
          let val_a = a[i];
          let val_b = b[i];

          if is_normal(&val_a) && is_normal(&val_b) {
            if val_a > val_b {
              greater_count += 1;
            } else {
              // A <= B
              if greater_count >= n {
                r[i] = true;
              }
              greater_count = 0;
            }
          }
        }
      } else {
        let mut greater_count = 0;

        let pre_start = if start > n { start - n } else { 0 };
        for i in pre_start..start {
          if is_normal(&a[i]) && is_normal(&b[i]) && a[i] > b[i] {
            greater_count += 1;
          } else {
            greater_count = 0;
          }
        }

        for i in start..end {
          let curr_a = a[i];
          let curr_b = b[i];

          if is_normal(&curr_a) && is_normal(&curr_b) {
            if curr_a <= curr_b {
              if greater_count >= n {
                r[i] = true;
              }
              greater_count = 0;
            } else {
              // curr_a > curr_b
              greater_count += 1;
            }
          } else {
            greater_count = 0;
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
    // assert_vec_eq_nan, // bools don't need approx eq
    context::FLAG_SKIP_NAN,
  };

  fn assert_vec_eq_bool(a: &[bool], b: &[bool]) {
    assert_eq!(a, b, "\nLeft: {:?}\nRight: {:?}", a, b);
  }

  #[test]
  fn test_cross() {
    let a = vec![1.0, 2.0, 4.0, 2.0];
    let b = vec![2.0, 2.0, 3.0, 1.0];
    // i=0: 1vs2 (<). prev=None.
    // i=1: 2vs2 (>=). prev A<B(1<2). CROSS -> T
    // i=2: 4vs3 (>=). prev A<B? 2<2 is False (2>=2). prev was >=. CROSS -> F
    // i=3: 2vs1 (>=). prev 4vs3 (>=). CROSS -> F

    // Wait, check definition: CROSS(A,B). prev A<B, curr A>=B.
    // 0: N/a
    // 1: prev 1<2 (T), curr 2>=2 (T). -> T.
    // 2: prev 2<2 (F), curr 4>=3 (T). -> F.
    // 3: prev 4<3 (F). -> F.

    let mut r = vec![false; 4];
    let ctx = Context::new(0, 0, 0);
    ta_cross(&ctx, &mut r, &a, &b).unwrap();
    assert_vec_eq_bool(&r, &vec![false, true, false, false]);
  }

  #[test]
  fn test_cross_skip_nan() {
    let a = vec![1.0, f64::NAN, 4.0];
    let b = vec![2.0, f64::NAN, 3.0];
    // 0: 1<2 (T). last_less=T
    // 1: NaN. Skip.
    // 2: 4>=3 (T). last was T. -> T. last_less=F (4<3 is F).

    let mut r = vec![false; 3];
    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);
    ta_cross(&ctx, &mut r, &a, &b).unwrap();
    assert_vec_eq_bool(&r, &vec![false, false, true]);
  }

  #[test]
  fn test_longcross() {
    // Need N periods of A<B.
    // N=2
    // 0: 1<5 (T)
    // 1: 2<5 (T)
    // 2: 3<5 (T)
    // 3: 6>=5 (T). prev 2 periods (1,2) were < 5?
    //    Indices of A<B: 0, 1, 2.
    //    At idx 3: prev 2 are 2 and 1. A[2]<B[2], A[1]<B[1]. Yes. -> T.

    let a = vec![1.0, 2.0, 3.0, 6.0];
    let b = vec![5.0, 5.0, 5.0, 5.0];

    let mut r = vec![false; 4];
    let ctx = Context::new(0, 0, 0);
    ta_longcross(&ctx, &mut r, &a, &b, 2).unwrap();

    // 0: cnt=1
    // 1: cnt=2
    // 2: cnt=3
    // 3: 6>=5. cnt=3 >= 2. -> T. cnt=0.

    assert_vec_eq_bool(&r, &vec![false, false, false, true]);
  }
}
