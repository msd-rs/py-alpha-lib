// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal, skip_nan_window::SkipNanWindow};

/// Calculate rolling min-max difference (range) over a moving window
///
/// TS_MIN_MAX_DIFF = TS_MAX(x, d) - TS_MIN(x, d)
/// Single-pass using two monotonic deques for efficiency.
pub fn ta_min_max_diff<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }


  use std::collections::VecDeque;

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      let mut max_deque: VecDeque<usize> = VecDeque::new();
      let mut min_deque: VecDeque<usize> = VecDeque::new();
      let mut nan_in_window = 0;

      let pre_fill_start = if start >= periods { start - periods } else { 0 };

      for k in pre_fill_start..start {
        let val = x[k];
        if is_normal(&val) {
          while let Some(&back) = max_deque.back() {
            if val >= x[back] {
              max_deque.pop_back();
            } else {
              break;
            }
          }
          max_deque.push_back(k);
          while let Some(&back) = min_deque.back() {
            if val <= x[back] {
              min_deque.pop_back();
            } else {
              break;
            }
          }
          min_deque.push_back(k);
        } else {
          nan_in_window += 1;
        }
      }

      for i in start..end {
        let val = x[i];

        if is_normal(&val) {
          while let Some(&back) = max_deque.back() {
            if val >= x[back] {
              max_deque.pop_back();
            } else {
              break;
            }
          }
          max_deque.push_back(i);
          while let Some(&back) = min_deque.back() {
            if val <= x[back] {
              min_deque.pop_back();
            } else {
              break;
            }
          }
          min_deque.push_back(i);
        } else {
          nan_in_window += 1;
        }

        if i >= periods {
          let old = x[i - periods];
          if !is_normal(&old) {
            nan_in_window -= 1;
          }
          if let Some(&front) = max_deque.front() {
            if front <= i - periods {
              max_deque.pop_front();
            }
          }
          if let Some(&front) = min_deque.front() {
            if front <= i - periods {
              min_deque.pop_front();
            }
          }
        }

        if nan_in_window > 0 || !is_normal(&val) {
          continue;
        }

        if ctx.is_strictly_cycle() && i < periods - 1 {
          continue;
        }

        if let (Some(&max_idx), Some(&min_idx)) = (max_deque.front(), min_deque.front()) {
          r[i] = x[max_idx] - x[min_idx];
        }
      }
    });

  Ok(())
}

/// Calculate weighted delay (exponentially weighted lag)
///
/// WEIGHTED_DELAY(x, k) = (k * x[t-1] + (k-1) * x[t-2] + ... + 1 * x[t-k]) / (k*(k+1)/2)
/// This is essentially LWMA applied to the lagged (shifted by 1) series over k periods.
pub fn ta_weighted_delay<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }


  if periods == 0 {
    r.fill(NumT::nan());
    return Ok(());
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      // Weight denominator: sum of 1..=periods = periods*(periods+1)/2
      let denom = NumT::from(periods * (periods + 1) / 2).unwrap();

      for i in start..end {
        // We need x[i-1], x[i-2], ..., x[i-periods]
        if i < periods {
          continue;
        }

        let mut sum = NumT::zero();
        let mut has_nan = false;
        for k in 1..=periods {
          let val = x[i - k];
          if !is_normal(&val) {
            has_nan = true;
            break;
          }
          // Weight: periods - k + 1 for most recent lag getting highest weight
          // lag 1 (most recent) gets weight = periods
          // lag periods (oldest) gets weight = 1
          let w = NumT::from(periods - k + 1).unwrap();
          sum = sum + w * val;
        }

        if !has_nan {
          r[i] = sum / denom;
        }
      }
    });

  Ok(())
}

/// Calculate rolling k-th central moment over a moving window
///
/// MOMENT(x, d, k) = mean((x - mean)^k) over window of d periods.
/// This is the raw (non-adjusted) sample moment.
/// k=2 gives variance (population), k=3 gives raw third moment, etc.
pub fn ta_moment<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
  k: usize,
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
        let iter = SkipNanWindow::new(&x[..end], periods, start);
        let mut sum = NumT::zero();

        for i in iter {
          let val = x[i.end];
          if is_normal(&val) {
            sum = sum + val;
          }

          for j in i.prev_start..i.start {
            let old = x[j];
            if is_normal(&old) {
              sum = sum - old;
            }
          }

          if !is_normal(&val) {
            continue;
          }

          let mut should_output = true;
          if ctx.is_strictly_cycle() {
            if i.no_nan_count != periods || (i.end - i.start + 1) != periods {
              should_output = false;
            }
          }

          if should_output && i.no_nan_count >= 2 {
            let count = NumT::from(i.no_nan_count).unwrap();
            let mean = sum / count;

            // Compute sum of (x - mean)^k over the window
            let mut moment_sum = NumT::zero();
            // We need to iterate over the valid values in the window
            // The window is from i.start to i.end
            for j in i.start..=i.end {
              let v = x[j];
              if is_normal(&v) {
                let diff = v - mean;
                let mut power = diff;
                for _ in 1..k {
                  power = power * diff;
                }
                moment_sum = moment_sum + power;
              }
            }

            r[i.end] = moment_sum / count;
          }
        }
      } else {
        let mut sum = NumT::zero();
        let mut nan_in_window = 0;

        let pre_fill_start = if start >= periods { start - periods } else { 0 };

        for j in pre_fill_start..start {
          let val = x[j];
          if is_normal(&val) {
            sum = sum + val;
          } else {
            nan_in_window += 1;
          }
        }

        for i in start..end {
          let val = x[i];

          if is_normal(&val) {
            sum = sum + val;
          } else {
            nan_in_window += 1;
          }

          if i >= periods {
            let old = x[i - periods];
            if is_normal(&old) {
              sum = sum - old;
            } else {
              nan_in_window -= 1;
            }
          }

          if nan_in_window > 0 || !is_normal(&val) {
            // NaN
          } else if i >= periods - 1 && periods >= 2 {
            let count = NumT::from(periods).unwrap();
            let mean = sum / count;

            let win_start = i + 1 - periods;
            let mut moment_sum = NumT::zero();
            for j in win_start..=i {
              let v = x[j];
              let diff = v - mean;
              let mut power = diff;
              for _ in 1..k {
                power = power * diff;
              }
              moment_sum = moment_sum + power;
            }

            r[i] = moment_sum / count;
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
  fn test_min_max_diff_simple() {
    let input = vec![1.0, 5.0, 3.0, 7.0, 2.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_min_max_diff(&ctx, &mut r, &input, periods).unwrap();

    // 0: [1] -> 0
    // 1: [1,5] -> 4
    // 2: [1,5,3] -> 4
    // 3: [5,3,7] -> 4
    // 4: [3,7,2] -> 5
    assert_vec_eq_nan(&r, &vec![0.0, 4.0, 4.0, 4.0, 5.0]);
  }

  #[test]
  fn test_weighted_delay_simple() {
    let input = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_weighted_delay(&ctx, &mut r, &input, periods).unwrap();

    // i=3: lags are x[2]=30(w=3), x[1]=20(w=2), x[0]=10(w=1)
    //       = (90 + 40 + 10) / 6 = 140/6 = 23.333...
    // i=4: lags are x[3]=40(w=3), x[2]=30(w=2), x[1]=20(w=1)
    //       = (120 + 60 + 20) / 6 = 200/6 = 33.333...
    let expected_3 = 140.0 / 6.0;
    let expected_4 = 200.0 / 6.0;
    assert!(r[0].is_nan());
    assert!(r[1].is_nan());
    assert!(r[2].is_nan());
    assert!((r[3] - expected_3).abs() < 1e-5, "got {}", r[3]);
    assert!((r[4] - expected_4).abs() < 1e-5, "got {}", r[4]);
  }

  #[test]
  fn test_moment_k2_is_population_variance() {
    // k=2 moment = population variance = sum((x-mean)^2)/n
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_moment(&ctx, &mut r, &input, periods, 2).unwrap();

    // Window [1,2,3]: mean=2, pop_var = (1+0+1)/3 = 2/3 = 0.6667
    // Window [2,3,4]: mean=3, pop_var = 2/3
    // Window [3,4,5]: mean=4, pop_var = 2/3
    let expected = 2.0 / 3.0;
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, expected, expected, expected]);
  }

  #[test]
  fn test_moment_k3_symmetric() {
    // Symmetric data -> k=3 moment = 0
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_moment(&ctx, &mut r, &input, periods, 3).unwrap();

    // Evenly spaced -> 3rd moment = 0
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 0.0, 0.0, 0.0]);
  }

  #[test]
  fn test_moment_skip_nan() {
    let input = vec![1.0, f64::NAN, 2.0, 3.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);
    ta_moment(&ctx, &mut r, &input, periods, 2).unwrap();

    // Position 3: valid [1, 2, 3], mean=2, pop_var = 2/3
    let expected = 2.0 / 3.0;
    assert!((r[3] - expected).abs() < 1e-5, "got {}", r[3]);
  }
}
