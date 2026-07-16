// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal, skip_nan_window::SkipNanWindow};

/// Simple Moving Average, also known as arithmetic moving average
///
/// Ref: https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average
///
pub fn ta_ma<NumT: Float + Send + Sync>(
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

          // subtract values that fell out of the window
          for k in i.prev_start..i.start {
            let old = x[k];
            if is_normal(&old) {
              sum = sum - old;
            }
          }

          if !is_normal(&val) {
            continue;
          }

          if ctx.is_strictly_cycle() {
            // strict cycle with skip_nan implies we want 'periods' valid numbers,
            // BUT existing behavior implies we return NaN if there are any NaNs in the window.
            if i.no_nan_count == periods && (i.end - i.start + 1) == periods {
              r[i.end] = sum / NumT::from(periods).unwrap();
            }
          } else {
            r[i.end] = sum / NumT::from(i.no_nan_count).unwrap();
          }
        }
      } else {
        let mut sum = NumT::zero();
        let mut nan_in_window = 0;

        // Pre-initialization for start > 0
        let pre_fill_start = if start >= periods { start - periods } else { 0 };
        for k in pre_fill_start..start {
          if is_normal(&x[k]) {
            sum = sum + x[k];
          } else {
            nan_in_window += 1;
          }
        }

        for i in start..end {
          let val = x[i];

          // Add new value
          if is_normal(&val) {
            sum = sum + val;
          } else {
            nan_in_window += 1;
          }

          // Remove old value
          if i >= periods {
            let old = x[i - periods];
            if is_normal(&old) {
              sum = sum - old;
            } else {
              nan_in_window -= 1;
            }
          }

          if !is_normal(&val) {
            continue;
          }

          if ctx.is_strictly_cycle() {
            if i >= periods - 1 {
              if nan_in_window == 0 {
                r[i] = sum / NumT::from(periods).unwrap();
              }
            }
          } else {
            if i < periods {
              if nan_in_window == 0 {
                let count = if i < periods { i + 1 } else { periods };
                r[i] = sum / NumT::from(count).unwrap();
              }
            } else {
              if nan_in_window == 0 {
                r[i] = sum / NumT::from(periods).unwrap();
              }
            }
          }
        }
      }
    });

  Ok(())
}

/// Simple Moving Average with variable periods
///
/// Ref: https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average
pub fn ta_ma_v<NumT: Float + Send + Sync>(
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
        let mut history_indices = Vec::with_capacity(r.len());
        let mut history_vals = Vec::with_capacity(r.len());
        let mut history_pref = vec![NumT::zero()];
        for i in start..end {
          let val = x[i];
          if is_normal(&val) {
            history_indices.push(i);
            history_vals.push(val);
            let last_pref = *history_pref.last().unwrap();
            history_pref.push(last_pref + val);

            let period = p[i];
            if period == 0 {
              continue;
            }
            let l = history_vals.len();
            if ctx.is_strictly_cycle() {
              if l >= period {
                let start_hist_idx = l - period;
                if i - history_indices[start_hist_idx] + 1 == period {
                  let sum = history_pref[l] - history_pref[start_hist_idx];
                  r[i] = sum / NumT::from(period).unwrap();
                }
              }
            } else {
              let count = l.min(period);
              if count > 0 {
                let start_hist_idx = l - count;
                let sum = history_pref[l] - history_pref[start_hist_idx];
                r[i] = sum / NumT::from(count).unwrap();
              }
            }
          }
        }
      } else {
        // Normal mode
        let mut pref = vec![NumT::zero(); r.len() + 1];
        let mut nan_pref = vec![0; r.len() + 1];
        for k in 0..r.len() {
          let val = x[k];
          if is_normal(&val) {
            pref[k + 1] = pref[k] + val;
            nan_pref[k + 1] = nan_pref[k];
          } else {
            pref[k + 1] = pref[k];
            nan_pref[k + 1] = nan_pref[k] + 1;
          }
        }

        for i in start..end {
          let val = x[i];
          if !is_normal(&val) {
            continue;
          }
          let period = p[i];
          if period == 0 {
            continue;
          }
          let start_idx = if i >= period { i + 1 - period } else { 0 };
          let win_len = i + 1 - start_idx;
          let nan_in_window = nan_pref[i + 1] - nan_pref[start_idx];

          if nan_in_window == 0 {
            if ctx.is_strictly_cycle() {
              if i >= period - 1 {
                r[i] = (pref[i + 1] - pref[start_idx]) / NumT::from(period).unwrap();
              }
            } else {
              r[i] = (pref[i + 1] - pref[start_idx]) / NumT::from(win_len).unwrap();
            }
          }
        }
      }
    });

  Ok(())
}


/// Calculate product of values in preceding `periods` window
///
/// If periods is 0, it calculates the cumulative product from the first valid value.
///
/// Ref: https://www.amibroker.com/guide/afl/product.html
pub fn ta_product<NumT: Float + Send + Sync>(
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

      if periods == 0 {
        // Cumulative product
        let mut prod = NumT::one();
        let mut found_valid = false;

        for i in start..end {
          let val = x[i];
          if is_normal(&val) {
            prod = prod * val;
            found_valid = true;
          }

          if found_valid {
            r[i] = prod;
          }
        }
      } else {
        // Sliding window
        if ctx.is_skip_nan() {
          let iter = SkipNanWindow::new(&x[..end], periods, start);
          let mut prod_non_zero = NumT::one();
          let mut zero_count = 0;

          for i in iter {
            let val = x[i.end];
            if is_normal(&val) {
              if val == NumT::zero() {
                zero_count += 1;
              } else {
                prod_non_zero = prod_non_zero * val;
              }
            }

            // subtract values that fell out of the window
            for k in i.prev_start..i.start {
              let old = x[k];
              if is_normal(&old) {
                if old == NumT::zero() {
                  zero_count -= 1;
                } else {
                  prod_non_zero = prod_non_zero / old;
                }
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

            if should_output {
              if zero_count > 0 {
                r[i.end] = NumT::zero();
              } else {
                r[i.end] = prod_non_zero;
              }
            }
          }
        } else {
          let mut prod_non_zero = NumT::one();
          let mut zero_count = 0;
          let mut nan_in_window = 0;

          // Pre-initialization for start > 0
          let pre_fill_start = if start >= periods { start - periods } else { 0 };
          for k in pre_fill_start..start {
            let val = x[k];
            if is_normal(&val) {
              if val == NumT::zero() {
                zero_count += 1;
              } else {
                prod_non_zero = prod_non_zero * val;
              }
            } else {
              nan_in_window += 1;
            }
          }

          for i in start..end {
            let val = x[i];

            // Add new value
            if is_normal(&val) {
              if val == NumT::zero() {
                zero_count += 1;
              } else {
                prod_non_zero = prod_non_zero * val;
              }
            } else {
              nan_in_window += 1;
            }

            // Remove old value
            if i >= periods {
              let old = x[i - periods];
              if is_normal(&old) {
                if old == NumT::zero() {
                  zero_count -= 1;
                } else {
                  prod_non_zero = prod_non_zero / old;
                }
              } else {
                nan_in_window -= 1;
              }
            }

            if !is_normal(&val) {
              continue;
            }

            if ctx.is_strictly_cycle() {
              if i >= periods - 1 {
                if nan_in_window == 0 {
                  if zero_count > 0 {
                    r[i] = NumT::zero();
                  } else {
                    r[i] = prod_non_zero;
                  }
                }
              }
            } else {
              if nan_in_window == 0 {
                if zero_count > 0 {
                  r[i] = NumT::zero();
                } else {
                  r[i] = prod_non_zero;
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
  use crate::algo::{
    assert_vec_eq_nan,
    context::{FLAG_SKIP_NAN, FLAG_STRICTLY_CYCLE},
  };

  use super::*;

  #[test]
  fn test_ta_ma() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_ma(&ctx, &mut r, &input, periods).unwrap();
    assert_eq!(r, vec![1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

    let ctx = Context::new(0, 0, FLAG_STRICTLY_CYCLE);
    ta_ma(&ctx, &mut r, &input, periods).unwrap();
    assert_vec_eq_nan(
      &r,
      &vec![f64::NAN, f64::NAN, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
  }

  #[test]
  fn test_ta_ma_skip_nan() {
    let input = vec![1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];

    // don't skip nan
    let ctx = Context::new(0, 0, 0);
    ta_ma(&ctx, &mut r, &input, periods).unwrap();
    assert_vec_eq_nan(
      &r,
      &vec![
        1.0,
        1.5,
        2.0,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        6.0,
        7.0,
        8.0,
        9.0,
      ],
    );

    // skip nan
    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);
    ta_ma(&ctx, &mut r, &input, periods).unwrap();
    assert_vec_eq_nan(
      &r,
      &vec![
        1.0,
        1.5,
        2.0,
        f64::NAN,
        3.333333,
        4.666667,
        6.0,
        7.0,
        8.0,
        9.0,
      ],
    );

    // skip nan and strictly cycle
    let ctx = Context::new(2, 0, FLAG_SKIP_NAN | FLAG_STRICTLY_CYCLE);
    ta_ma(&ctx, &mut r, &input, periods).unwrap();
    assert_vec_eq_nan(
      &r,
      &vec![
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        6.0,
        7.0,
        8.0,
        9.0,
      ],
    );
  }

  #[test]
  fn test_ta_product() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_product(&ctx, &mut r, &input, periods).unwrap();

    // 0: 1*1*1 = 1 (partial) -> 1
    // 1: 1*2 = 2 (partial) -> 2
    // 2: 1*2*3 = 6
    // 3: 2*3*4 = 24
    // 4: 3*4*5 = 60
    assert_vec_eq_nan(&r, &vec![1.0, 2.0, 6.0, 24.0, 60.0]);
  }

  #[test]
  fn test_ta_product_zeros() {
    let input = vec![1.0, 2.0, 0.0, 4.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_product(&ctx, &mut r, &input, periods).unwrap();

    // 0: 1
    // 1: 2
    // 2: 1*2*0 = 0
    // 3: 2*0*4 = 0
    // 4: 0*4*5 = 0
    assert_vec_eq_nan(&r, &vec![1.0, 2.0, 0.0, 0.0, 0.0]);

    let input2 = vec![1.0, 2.0, 0.0, 4.0, 5.0, 2.0];
    // 5: 4*5*2 = 40 (0 left window)
    let mut r2 = vec![0.0; input2.len()];
    ta_product(&ctx, &mut r2, &input2, periods).unwrap();
    assert_vec_eq_nan(&r2, &vec![1.0, 2.0, 0.0, 0.0, 0.0, 40.0]);
  }

  #[test]
  fn test_ta_product_skip_nan() {
    let input = vec![1.0, f64::NAN, 2.0, 3.0, 4.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);
    ta_product(&ctx, &mut r, &input, periods).unwrap();

    // 0: 1
    // 1: NaN
    // 2: 1*2=2
    // 3: 1*2*3=6
    // 4: 2*3*4=24
    assert_vec_eq_nan(&r, &vec![1.0, f64::NAN, 2.0, 6.0, 24.0]);
  }

  #[test]
  fn test_ta_ma_v() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let periods = vec![1, 2, 3, 2, 3];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_ma_v(&ctx, &mut r, &input, &periods).unwrap();
    assert_eq!(r, vec![1.0, 1.5, 2.0, 3.5, 4.0]);
  }
}
