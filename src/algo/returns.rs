// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal};

/// Future Return
///
/// Calculates the return from the open price of the delayed day (t+delay) to the close price of the future day (t+delay+periods-1).
/// Return = (Close[t+delay+periods-1] - Open[t+delay]) / Open[t+delay]
///
/// If n=1, delay=1, it calculates (Close[t+1] - Open[t+1]) / Open[t+1].
/// If High[t+delay] == Open[t+delay] == Low[t+delay] == Close[t+delay], returns NaN.
pub fn ta_fret<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  open: &[NumT],
  high: &[NumT],
  low: &[NumT],
  close: &[NumT],
  delay: usize,
  periods: usize,
) -> Result<(), Error> {
  if r.len() != open.len() || r.len() != high.len() || r.len() != low.len() || r.len() != close.len() {
    return Err(Error::LengthMismatch(r.len(), open.len()));
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(open.par_chunks(ctx.chunk_size(open.len())))
    .zip(high.par_chunks(ctx.chunk_size(high.len())))
    .zip(low.par_chunks(ctx.chunk_size(low.len())))
    .zip(close.par_chunks(ctx.chunk_size(close.len())))
    .for_each(|((((r, o), h), l), c)| {
      let start = ctx.start(r.len());
      r.fill(NumT::nan());

      // Future Return calculation
      // Return = (Close[t+delay+periods-1] - Open[t+delay]) / Open[t+delay]
      
      if periods == 0 {
          return;
      }

      let exit_offset = periods + delay - 1;
      let max_offset = std::cmp::max(exit_offset, delay);
      
      let end_idx = if c.len() > max_offset { c.len() - max_offset } else { 0 };

      for i in start..end_idx {
        let open_next = o[i + delay];
        let high_next = h[i + delay];
        let low_next = l[i + delay];
        let close_next = c[i + delay];
        
        let close_future = c[i + exit_offset];

        if is_normal(&open_next) && is_normal(&close_future) {
             // Check if H=O=L=C on the entry day (t+delay)
             if high_next == open_next && open_next == low_next && low_next == close_next {
                 r[i] = NumT::nan();
             } else {
                 if open_next != NumT::zero() {
                     r[i] = (close_future - open_next) / open_next;
                 } else {
                     r[i] = NumT::nan();
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
  use crate::algo::{assert_vec_eq_nan, Context};

  #[test]
  fn test_ta_fret() {
    let open = vec![10.0, 11.0, 12.0, 13.0, 14.0];
    let high = vec![10.5, 11.5, 12.5, 13.5, 14.5];
    let low  = vec![ 9.5, 10.5, 11.5, 12.5, 13.5];
    let close = vec![10.5, 11.5, 12.5, 13.5, 14.5];
    let mut r = vec![0.0; 5];
    let ctx = Context::default();

    // FRET(delay=1, periods=1) at t=0:
    // Entry = t+1 = 1 (Open[1]=11.0)
    // Exit = t+1+1-1 = 1 (Close[1]=11.5)
    // Return = (11.5-11.0)/11.0 = 0.5/11.0 = 0.045454545
    
    ta_fret(&ctx, &mut r, &open, &high, &low, &close, 1, 1).unwrap();
    assert_vec_eq_nan(&r, &vec![
        0.045454545454545456, 
        0.041666666666666664, 
        0.038461538461538464, 
        0.03571428571428571, 
        f64::NAN
    ]);
  }

  #[test]
  fn test_ta_fret_delayed() {
    let open = vec![10.0, 11.0, 12.0, 13.0, 14.0];
    let high = vec![10.5, 11.5, 12.5, 13.5, 14.5];
    let low  = vec![ 9.5, 10.5, 11.5, 12.5, 13.5];
    let close = vec![10.5, 11.5, 12.5, 13.5, 14.5];
    let mut r2 = vec![0.0; 5];
    let ctx = Context::default();

    // Test delay=2, periods=1
    // t=0: Entry t+2=2. Exit t+2+1-1=2. (Close[2]-Open[2])/Open[2] = (12.5-12)/12
    
    ta_fret(&ctx, &mut r2, &open, &high, &low, &close, 2, 1).unwrap();
    let expected2 = vec![
        (12.5 - 12.0) / 12.0,
        (13.5 - 13.0) / 13.0,
        (14.5 - 14.0) / 14.0,
        f64::NAN,
        f64::NAN
    ];
    assert_vec_eq_nan(&r2, &expected2);
  }

  #[test]
  fn test_ta_fret_ohlc_equal() {
    let open = vec![10.0, 11.0, 12.0];
    let high = vec![11.0, 11.0, 13.0];
    let low  = vec![ 9.0, 11.0, 11.0];
    let close = vec![10.5, 11.0, 12.5];
    
    // At i=0. Entry i+1=1.
    // Open[1]=11, High[1]=11, Low[1]=11, Close[1]=11.
    // So r[0] should be NaN.
    
    let mut r = vec![0.0; 3];
    let ctx = Context::default();
    
    ta_fret(&ctx, &mut r, &open, &high, &low, &close, 1, 1).unwrap();
    
    assert!(r[0].is_nan());
  }
}
