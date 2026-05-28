// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;

use super::{Context, Error, is_normal, skip_nan_window::SkipNanWindow};
use rayon::prelude::*;

/// Exponential Moving Average (variant of well-known EMA) weight = 2 / (n + 1)
///
/// Ref: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
///
pub fn ta_ema<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  let alpha = NumT::from(2.0).unwrap() / NumT::from(periods + 1).unwrap();

  ema_impl(ctx, r, input, alpha, periods)
}

/// Exponential Moving Average (variant of well-known EMA) weight = m / n
///
/// Ref: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
///
pub fn ta_sma<NumT: Float + Send + Sync>(
  _ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  n: usize,
  m: usize,
) -> Result<(), Error> {
  let alpha = NumT::from(m).unwrap() / NumT::from(n).unwrap();
  ta_dma(_ctx, r, input, alpha)
}

/// Exponential Moving Average
/// current = weight * current + (1 - weight) * previous
///
/// Ref: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
///
pub fn ta_dma<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  weight: NumT,
) -> Result<(), Error> {
  ema_impl(ctx, r, input, weight, 0)
}

/// Linear Weighted Moving Average
///
/// LWMA = SUM(Price * Weight) / SUM(Weight)
///
pub fn ta_lwma<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }


  if periods == 1 {
    r.copy_from_slice(input);
    return Ok(());
  }

  let n_t = NumT::from(periods).unwrap();
  let sum_weight = NumT::from(periods * (periods + 1) / 2).unwrap();

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      if ctx.is_skip_nan() {
        let iter = SkipNanWindow::new(&x[..end], periods, start);
        let mut simple_sum = NumT::zero();
        let mut weighted_sum = NumT::zero();

        for i in iter {
          let val = x[i.end];
          if is_normal(&val) {
            weighted_sum = weighted_sum + n_t * val - simple_sum;
            simple_sum = simple_sum + val;
          }

          for k in i.prev_start..i.start {
            let old = x[k];
            if is_normal(&old) {
              simple_sum = simple_sum - old;
            }
          }

          if !is_normal(&val) {
            continue;
          }

          if ctx.is_strictly_cycle() {
            if i.no_nan_count == periods && (i.end - i.start + 1) == periods {
              r[i.end] = weighted_sum / sum_weight;
            }
          } else {
            // LWMA is usually defined for full window.
            // If window not full, we could calculate partial LWMA but strict definition uses fixed weights N..1.
            // Standard behavior matches simple MA: only output when valid count matches?
            // Actually `ma.rs` produces result for partial window if NOT strictly cycle.
            // But for LWMA weights change.
            // If we have 2 items: weights 1, 2. sum_weight = 3.
            // Our algorithm:
            // 1st item: W=N*p1.
            // This is "wrong" if we wanted weight 1.
            // But if we want result when full, we wait.
            // Let's mimic other libs: usually NaN until full periods.
            // Assuming "standard" behavior strictly requires N periods.
            // If user wants partial, current alg gives specific projection (assuming Phantom zeros).

            if i.no_nan_count == periods {
              r[i.end] = weighted_sum / sum_weight;
            }
          }
        }
      } else {
        let mut simple_sum = NumT::zero();
        let mut weighted_sum = NumT::zero();
        let mut nan_in_window = 0;

        // Pre-fill
        let pre_fill_start = if start >= periods { start - periods } else { 0 };

        for k in pre_fill_start..start {
          let val = x[k];
          if is_normal(&val) {
            weighted_sum = weighted_sum + n_t * val - simple_sum;
            simple_sum = simple_sum + val;
          } else {
            nan_in_window += 1;
            // No shift in valid streams, but strictly positional LWMA shifts everything?
            // If LWMA is purely positional (like internal indices):
            // Then NaN occupies a weight slot.
            // Formula LWMAi = Sum / SumWeight where Sum = P_i * N + ...
            // If P_k is NaN?
            // Usually result is NaN.
            // If skip_nan=false, one NaN makes result NaN.
            // So we don't need to be clever.
            // Just check NaN count.
          }
        }

        for i in start..end {
          let val = x[i];

          // Add new
          if is_normal(&val) {
            // Check if ANY NaN in window -> Result is NaN anyway.
            // So logic can be simplified?
            // We need to maintain sums correctly to recover when NaNs leave.
            // But if NaN occupies a slot, our "valid number shift" logic breaks because we shifted for a NaN.
            // "Standard" LWMA on array with NaNs:
            // If any NaN in window, output NaN.
            // Logic for maintaining sums:
            // Treat NaNs as 0.0 for sum purposes?
            // No, because when they leave, we subtract 0.0.
            // BUT they occupied a weight.
            // If we treat NaN as 0, then:
            // Add val (0): W_new = W_old + N*0 - S_old.
            // This shifts valid numbers!
            // So Yes, differential update works even with 0 insertion.
            // We just need to track if we should output NaN.
            weighted_sum =
              weighted_sum + n_t * (if is_normal(&val) { val } else { NumT::zero() }) - simple_sum;
            simple_sum = simple_sum + (if is_normal(&val) { val } else { NumT::zero() });
          } else {
            nan_in_window += 1;
            // Shift happens
            weighted_sum = weighted_sum - simple_sum;
            // Add 0
          }

          // Remove old
          if i >= periods {
            let old = x[i - periods];
            if is_normal(&old) {
              simple_sum = simple_sum - old;
            } else {
              nan_in_window -= 1;
            }
          }

          if nan_in_window > 0 || !is_normal(&val) {
            // Result NaN
          } else {
            if i >= periods - 1 {
              r[i] = weighted_sum / sum_weight;
            }
          }
        }
      }
    });

  Ok(())
}

pub fn ema_impl<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  weight: NumT,
  periods: usize,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }


  if weight < NumT::zero() || weight > NumT::one() {
    return Err(Error::InvalidParameter(
      "alpha must be between 0 and 1".to_string(),
    ));
  }

  let k = NumT::one() - weight;

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, i)| {
      let mut prev = i[0];
      let total = r.len();
      for (n, (r, c)) in r
        .iter_mut()
        .zip(i.iter())
        .enumerate()
        .skip(ctx.start(total))
      {
        if ctx.is_skip_nan() && !is_normal(c) {
          *r = NumT::nan();
          continue;
        }
        if ctx.is_strictly_cycle() && n < periods - 1 {
          *r = NumT::nan();
          prev = *c;
        } else {
          *r = weight * *c + k * prev;
          prev = *r;
        }
      }
    });
  Ok(())
}

#[cfg(test)]
mod tests {
  use crate::algo::{
    assert_vec_eq_nan,
    context::{FLAG_SKIP_NAN, FLAG_STRICTLY_CYCLE},
  };

  use super::*;

  #[test]
  fn test_ta_ema() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    let _ = ta_ema(&ctx, &mut r, &input, periods);
    // EMA(3) -> alpha = 2/4 = 0.5.
    // 0: 1*0.5 + 0 = ? No, first element Init?
    // ema_impl uses prev = i[0].
    // n=0: r = 0.5*1 + 0.5*1 = 1.
    // n=1: r = 0.5*2 + 0.5*1 = 1.5.
    // n=2: r = 0.5*3 + 0.5*1.5 = 1.5 + 0.75 = 2.25.
    assert_eq!(r[0], 1.0);
    assert_eq!(r[1], 1.5);
    assert_eq!(r[2], 2.25);
  }

  #[test]
  fn test_ta_ema_strictly_cycle() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, FLAG_STRICTLY_CYCLE);
    let _ = ta_ema(&ctx, &mut r, &input, periods);
    // EMA(3) -> alpha = 2/4 = 0.5.
    // 0: NaN
    // 1: NaN
    // 2: 0.5*3 + 0.5*2 = 2.5
    // 3: 0.5*4 + 0.5*2.5 = 3.25
    // 4: 0.5*5 + 0.5*3.25 = 4.125
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 2.5, 3.25, 4.125]);
  }

  #[test]
  fn test_ta_lwma() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let periods = 3;
    // Weights: 1, 2, 3. SumW = 6.
    // 0: NaN
    // 1: NaN (Buildup)
    // 2: (1*1 + 2*2 + 3*3)/6 = (1+4+9)/6 = 14/6 = 2.3333
    // 3: (1*2 + 2*3 + 3*4)/6 = (2+6+12)/6 = 20/6 = 3.3333 (Check diff: W_old=14. Sum_old=6. W_new=14+3*4-6=20. OK)
    // 4: (1*3 + 2*4 + 3*5)/6 = (3+8+15)/6 = 26/6 = 4.3333

    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_lwma(&ctx, &mut r, &input, periods).unwrap();

    let expected = vec![f64::NAN, f64::NAN, 14.0 / 6.0, 20.0 / 6.0, 26.0 / 6.0];
    assert_vec_eq_nan(&r, &expected);
  }

  #[test]
  fn test_ta_lwma_skip_nan() {
    let input = vec![1.0, 2.0, f64::NAN, 3.0, 4.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];

    // Case 1: No skip nan
    let ctx = Context::new(0, 0, 0);
    ta_lwma(&ctx, &mut r, &input, periods).unwrap();
    // 0: NaN
    // 1: NaN
    // 2: NaN (Has NaN)
    // 3: NaN (Has NaN)
    // 4: (1*2 + 2*3 + 3*4)? No, window [NAN, 3, 4] -> NaN.
    // Wait, index 4 window is [NaN, 3, 4]? No. Indices 2,3,4.
    // input[2] is NaN. So r[2], r[3], r[4] involve NaN?
    // r[2]: [0,1,2] -> has NaN.
    // r[3]: [1,2,3] -> has NaN at 2.
    // r[4]: [2,3,4] -> has NaN at 2.
    // So all NaNs starting from 2?
    // r[0], r[1] are NaN due to periods.
    let expected_noskip = vec![f64::NAN; 5];
    assert_vec_eq_nan(&r, &expected_noskip);

    // Case 2: Skip Nan
    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);
    ta_lwma(&ctx, &mut r, &input, periods).unwrap();
    // Valid stream: 1, 2, 3, 4.
    // 0: 1. Count 1.
    // 1: 2. Count 2.
    // 2: NaN (Skip).
    // 3: 3. Count 3. Window [1, 2, 3]. Result (1*1+2*2+3*3)/6 = 2.3333
    // 4: 4. Count 3. Window [2, 3, 4]. Result (1*2+2*3+3*4)/6 = 3.3333

    let expected_skip = vec![f64::NAN, f64::NAN, f64::NAN, 14.0 / 6.0, 20.0 / 6.0];
    assert_vec_eq_nan(&r, &expected_skip);
  }
}
