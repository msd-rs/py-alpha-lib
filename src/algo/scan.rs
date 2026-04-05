// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal};

/// Conditional cumulative multiply: r[t] = r[t-1] * (cond[t] ? input[t] : 1)
///
/// Used for SELF-referencing alpha expressions like GTJA #143.
/// Serial within each stock, parallel across stocks via rayon.
pub fn ta_scan_mul<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  condition: &[bool],
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

      let mut acc = NumT::one();
      for i in start..end {
        if c[i] && is_normal(&x[i]) {
          acc = acc * x[i];
        }
        r[i] = acc;
      }
    });

  Ok(())
}

/// Conditional cumulative add: r[t] = r[t-1] + (cond[t] ? input[t] : 0)
///
/// Used for SELF-referencing alpha expressions with additive accumulation.
/// Serial within each stock, parallel across stocks via rayon.
pub fn ta_scan_add<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  condition: &[bool],
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

      let mut acc = NumT::zero();
      for i in start..end {
        if c[i] && is_normal(&x[i]) {
          acc = acc + x[i];
        }
        r[i] = acc;
      }
    });

  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::algo::assert_vec_eq_nan;

  #[test]
  fn test_scan_mul() {
    let input = vec![2.0, 3.0, 0.5, 4.0, 2.0];
    let cond = vec![true, false, true, true, false];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_scan_mul(&ctx, &mut r, &input, &cond).unwrap();
    // t0: cond=T, acc=1*2=2, r=2
    // t1: cond=F, acc=2, r=2
    // t2: cond=T, acc=2*0.5=1, r=1
    // t3: cond=T, acc=1*4=4, r=4
    // t4: cond=F, acc=4, r=4
    assert_vec_eq_nan(&r, &vec![2.0, 2.0, 1.0, 4.0, 4.0]);
  }

  #[test]
  fn test_scan_add() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let cond = vec![true, false, true, false, true];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_scan_add(&ctx, &mut r, &input, &cond).unwrap();
    // t0: cond=T, acc=0+1=1, r=1
    // t1: cond=F, acc=1, r=1
    // t2: cond=T, acc=1+3=4, r=4
    // t3: cond=F, acc=4, r=4
    // t4: cond=T, acc=4+5=9, r=9
    assert_vec_eq_nan(&r, &vec![1.0, 1.0, 4.0, 4.0, 9.0]);
  }
}
