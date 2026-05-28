// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal, skip_nan_window::SkipNanWindow};

/// Calculate Variance over a moving window
///
/// Variance = (SumSq - (Sum^2)/N) / (N - 1)
pub fn ta_var<NumT: Float + Send + Sync>(
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
        let iter = SkipNanWindow::new(&x[..end], periods, start);
        let mut sum = NumT::zero();
        let mut sum_sq = NumT::zero();

        for i in iter {
          let val = x[i.end];
          if is_normal(&val) {
            sum = sum + val;
            sum_sq = sum_sq + val * val;
          }

          for k in i.prev_start..i.start {
            let old = x[k];
            if is_normal(&old) {
              sum = sum - old;
              sum_sq = sum_sq - old * old;
            }
          }

          if !is_normal(&val) {
            continue;
          }

          let count = NumT::from(i.no_nan_count).unwrap();

          let mut should_output = true;
          if ctx.is_strictly_cycle() {
            if i.no_nan_count != periods || (i.end - i.start + 1) != periods {
              should_output = false;
            }
          }

          if should_output {
            if i.no_nan_count > 1 {
              let var_num = sum_sq - (sum * sum / count);
              let var = var_num / (count - NumT::one());
              // Precision check
              if var < NumT::zero() {
                r[i.end] = NumT::zero();
              } else {
                r[i.end] = var;
              }
            } else {
              r[i.end] = NumT::nan();
            }
          }
        }
      } else {
        let mut sum = NumT::zero();
        let mut sum_sq = NumT::zero();
        let mut nan_in_window = 0;

        let pre_fill_start = if start >= periods { start - periods } else { 0 };

        for k in pre_fill_start..start {
          let val = x[k];
          if is_normal(&val) {
            sum = sum + val;
            sum_sq = sum_sq + val * val;
          } else {
            nan_in_window += 1;
          }
        }

        for i in start..end {
          let val = x[i];

          if is_normal(&val) {
            sum = sum + val;
            sum_sq = sum_sq + val * val;
          } else {
            nan_in_window += 1;
          }

          if i >= periods {
            let old = x[i - periods];
            if is_normal(&old) {
              sum = sum - old;
              sum_sq = sum_sq - old * old;
            } else {
              nan_in_window -= 1;
            }
          }

          if nan_in_window > 0 || !is_normal(&val) {
            // Result NaN
          } else {
            let output_idx = i;
            let mut valid = false;
            if ctx.is_strictly_cycle() {
              if i >= periods - 1 {
                valid = true;
              }
            } else {
              if i >= periods - 1 {
                valid = true;
              }
            }

            if valid {
              let count = NumT::from(periods).unwrap();
              if periods > 1 {
                let var_num = sum_sq - (sum * sum / count);
                let var = var_num / (count - NumT::one());
                if var < NumT::zero() {
                  r[output_idx] = NumT::zero();
                } else {
                  r[output_idx] = var;
                }
              } else {
                r[output_idx] = NumT::nan();
              }
            }
          }
        }
      }
    });

  Ok(())
}

/// Calculate Covariance over a moving window
///
/// Covariance = (SumXY - (SumX * SumY) / N) / (N - 1)
pub fn ta_cov<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  x: &[NumT],
  y: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != x.len() || x.len() != y.len() {
    return Err(Error::LengthMismatch(r.len(), x.len()));
  }


  let chunk_size = ctx.chunk_size(r.len());

  r.par_chunks_mut(chunk_size)
    .zip(x.par_chunks(chunk_size))
    .zip(y.par_chunks(chunk_size))
    .for_each(|((r, x), y)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      if ctx.is_skip_nan() {
        let mut sum_x = NumT::zero();
        let mut sum_y = NumT::zero();
        let mut sum_xy = NumT::zero();
        let mut no_nan_count = 0;

        let mut win_start = start;

        for i in start..end {
          let val_x = x[i];
          let val_y = y[i];
          let is_valid = is_normal(&val_x) && is_normal(&val_y);

          if is_valid {
            sum_x = sum_x + val_x;
            sum_y = sum_y + val_y;
            sum_xy = sum_xy + val_x * val_y;
            no_nan_count += 1;
          }

          while no_nan_count > periods {
            let old_x = x[win_start];
            let old_y = y[win_start];
            if is_normal(&old_x) && is_normal(&old_y) {
              sum_x = sum_x - old_x;
              sum_y = sum_y - old_y;
              sum_xy = sum_xy - old_x * old_y;
              no_nan_count -= 1;
            }
            win_start += 1;
          }

          while win_start <= i && !(is_normal(&x[win_start]) && is_normal(&y[win_start])) {
            win_start += 1;
          }

          if !is_valid {
            continue;
          }

          let count = NumT::from(no_nan_count).unwrap();

          let mut should_output = true;
          if ctx.is_strictly_cycle() {
            if no_nan_count != periods || (i - win_start + 1) != periods {
              should_output = false;
            }
          }

          if should_output {
            if no_nan_count > 1 {
              let cov_num = sum_xy - (sum_x * sum_y / count);
              let cov = cov_num / (count - NumT::one());
              r[i] = cov;
            } else {
              r[i] = NumT::nan();
            }
          }
        }
      } else {
        let mut sum_x = NumT::zero();
        let mut sum_y = NumT::zero();
        let mut sum_xy = NumT::zero();
        let mut nan_in_window = 0;

        let pre_fill_start = if start >= periods { start - periods } else { 0 };

        for k in pre_fill_start..start {
          let val_x = x[k];
          let val_y = y[k];
          if is_normal(&val_x) && is_normal(&val_y) {
            sum_x = sum_x + val_x;
            sum_y = sum_y + val_y;
            sum_xy = sum_xy + val_x * val_y;
          } else {
            nan_in_window += 1;
          }
        }

        for i in start..end {
          let val_x = x[i];
          let val_y = y[i];
          let is_valid = is_normal(&val_x) && is_normal(&val_y);

          if is_valid {
            sum_x = sum_x + val_x;
            sum_y = sum_y + val_y;
            sum_xy = sum_xy + val_x * val_y;
          } else {
            nan_in_window += 1;
          }

          if i >= periods {
            let old_x = x[i - periods];
            let old_y = y[i - periods];
            if is_normal(&old_x) && is_normal(&old_y) {
              sum_x = sum_x - old_x;
              sum_y = sum_y - old_y;
              sum_xy = sum_xy - old_x * old_y;
            } else {
              nan_in_window -= 1;
            }
          }

          if nan_in_window > 0 || !is_valid {
            // Result NaN
          } else {
            let mut valid = false;
            if i >= periods - 1 {
              valid = true;
            }

            if valid {
              let count = NumT::from(periods).unwrap();
              if periods > 1 {
                let cov_num = sum_xy - (sum_x * sum_y / count);
                let cov = cov_num / (count - NumT::one());
                r[i] = cov;
              } else {
                r[i] = NumT::nan();
              }
            }
          }
        }
      }
    });

  Ok(())
}

/// Calculate two series correlation over a moving window
///
/// Correlation = Cov(X, Y) / (StdDev(X) * StdDev(Y))
pub fn ta_corr2<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  x: &[NumT],
  y: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != x.len() || x.len() != y.len() {
    return Err(Error::LengthMismatch(r.len(), x.len()));
  }


  let chunk_size = ctx.chunk_size(r.len());

  r.par_chunks_mut(chunk_size)
    .zip(x.par_chunks(chunk_size))
    .zip(y.par_chunks(chunk_size))
    .for_each(|((r, x), y)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      if ctx.is_skip_nan() {
        let mut sum_x = NumT::zero();
        let mut sum_y = NumT::zero();
        let mut sum_xy = NumT::zero();
        let mut sum_x2 = NumT::zero();
        let mut sum_y2 = NumT::zero();
        let mut no_nan_count = 0;

        let mut win_start = start;

        for i in start..end {
          let val_x = x[i];
          let val_y = y[i];
          let is_valid = is_normal(&val_x) && is_normal(&val_y);

          if is_valid {
            sum_x = sum_x + val_x;
            sum_y = sum_y + val_y;
            sum_xy = sum_xy + val_x * val_y;
            sum_x2 = sum_x2 + val_x * val_x;
            sum_y2 = sum_y2 + val_y * val_y;
            no_nan_count += 1;
          }

          while no_nan_count > periods {
            let old_x = x[win_start];
            let old_y = y[win_start];
            if is_normal(&old_x) && is_normal(&old_y) {
              sum_x = sum_x - old_x;
              sum_y = sum_y - old_y;
              sum_xy = sum_xy - old_x * old_y;
              sum_x2 = sum_x2 - old_x * old_x;
              sum_y2 = sum_y2 - old_y * old_y;
              no_nan_count -= 1;
            }
            win_start += 1;
          }

          while win_start <= i && !(is_normal(&x[win_start]) && is_normal(&y[win_start])) {
            win_start += 1;
          }

          if !is_valid {
            continue;
          }

          let count = NumT::from(no_nan_count).unwrap();

          let mut should_output = true;
          if ctx.is_strictly_cycle() {
            if no_nan_count != periods || (i - win_start + 1) != periods {
              should_output = false;
            }
          }

          if should_output {
            if no_nan_count > 1 {
              let numerator = sum_xy - (sum_x * sum_y / count);

              let mut var_x_part = sum_x2 - (sum_x * sum_x / count);
              if var_x_part < NumT::zero() {
                var_x_part = NumT::zero();
              }

              let mut var_y_part = sum_y2 - (sum_y * sum_y / count);
              if var_y_part < NumT::zero() {
                var_y_part = NumT::zero();
              }

              let denominator = (var_x_part * var_y_part).sqrt();

              if denominator > NumT::epsilon() {
                r[i] = numerator / denominator;
              } else {
                r[i] = NumT::nan();
              }
            } else {
              r[i] = NumT::nan();
            }
          }
        }
      } else {
        let mut sum_x = NumT::zero();
        let mut sum_y = NumT::zero();
        let mut sum_xy = NumT::zero();
        let mut sum_x2 = NumT::zero();
        let mut sum_y2 = NumT::zero();
        let mut nan_in_window = 0;

        let pre_fill_start = if start >= periods { start - periods } else { 0 };

        for k in pre_fill_start..start {
          let val_x = x[k];
          let val_y = y[k];
          if is_normal(&val_x) && is_normal(&val_y) {
            sum_x = sum_x + val_x;
            sum_y = sum_y + val_y;
            sum_xy = sum_xy + val_x * val_y;
            sum_x2 = sum_x2 + val_x * val_x;
            sum_y2 = sum_y2 + val_y * val_y;
          } else {
            nan_in_window += 1;
          }
        }

        for i in start..end {
          let val_x = x[i];
          let val_y = y[i];
          let is_valid = is_normal(&val_x) && is_normal(&val_y);

          if is_valid {
            sum_x = sum_x + val_x;
            sum_y = sum_y + val_y;
            sum_xy = sum_xy + val_x * val_y;
            sum_x2 = sum_x2 + val_x * val_x;
            sum_y2 = sum_y2 + val_y * val_y;
          } else {
            nan_in_window += 1;
          }

          if i >= periods {
            let old_x = x[i - periods];
            let old_y = y[i - periods];
            if is_normal(&old_x) && is_normal(&old_y) {
              sum_x = sum_x - old_x;
              sum_y = sum_y - old_y;
              sum_xy = sum_xy - old_x * old_y;
              sum_x2 = sum_x2 - old_x * old_x;
              sum_y2 = sum_y2 - old_y * old_y;
            } else {
              nan_in_window -= 1;
            }
          }

          if nan_in_window > 0 || !is_valid {
            // Result NaN
          } else {
            let mut valid = false;
            // Standard check i >= periods - 1
            if i >= periods - 1 {
              valid = true;
            }

            if valid {
              let count = NumT::from(periods).unwrap();
              if periods > 1 {
                let numerator = sum_xy - (sum_x * sum_y / count);

                let mut var_x_part = sum_x2 - (sum_x * sum_x / count);
                if var_x_part < NumT::zero() {
                  var_x_part = NumT::zero();
                }

                let mut var_y_part = sum_y2 - (sum_y * sum_y / count);
                if var_y_part < NumT::zero() {
                  var_y_part = NumT::zero();
                }

                let denominator = (var_x_part * var_y_part).sqrt();

                if denominator > NumT::epsilon() {
                  r[i] = numerator / denominator;
                } else {
                  r[i] = NumT::nan();
                }
              } else {
                r[i] = NumT::nan();
              }
            }
          }
        }
      }
    });

  Ok(())
}

/// Calculate Regression Coefficient (Beta) of Y on X over a moving window
///
/// Beta = Cov(X, Y) / Var(X)
pub fn ta_regbeta<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  y: &[NumT],
  x: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != y.len() || y.len() != x.len() {
    return Err(Error::LengthMismatch(r.len(), y.len()));
  }


  let chunk_size = ctx.chunk_size(r.len());

  r.par_chunks_mut(chunk_size)
    .zip(y.par_chunks(chunk_size))
    .zip(x.par_chunks(chunk_size))
    .for_each(|((r, y), x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      if ctx.is_skip_nan() {
        let mut sum_x = NumT::zero();
        let mut sum_y = NumT::zero();
        let mut sum_xy = NumT::zero();
        let mut sum_x2 = NumT::zero();
        let mut no_nan_count = 0;
        let mut win_start = start;

        for i in start..end {
          let val_x = x[i];
          let val_y = y[i];
          let is_valid = is_normal(&val_x) && is_normal(&val_y);

          if is_valid {
            sum_x = sum_x + val_x;
            sum_y = sum_y + val_y;
            sum_xy = sum_xy + val_x * val_y;
            sum_x2 = sum_x2 + val_x * val_x;
            no_nan_count += 1;
          }

          while no_nan_count > periods {
            let old_x = x[win_start];
            let old_y = y[win_start];
            if is_normal(&old_x) && is_normal(&old_y) {
              sum_x = sum_x - old_x;
              sum_y = sum_y - old_y;
              sum_xy = sum_xy - old_x * old_y;
              sum_x2 = sum_x2 - old_x * old_x;
              no_nan_count -= 1;
            }
            win_start += 1;
          }

          while win_start <= i && !(is_normal(&x[win_start]) && is_normal(&y[win_start])) {
            win_start += 1;
          }

          if !is_valid {
            continue;
          }

          let mut should_output = true;
          if ctx.is_strictly_cycle() {
            if no_nan_count != periods || (i - win_start + 1) != periods {
              should_output = false;
            }
          }

          if should_output && no_nan_count > 1 {
            let count = NumT::from(no_nan_count).unwrap();
            let cov_num = sum_xy - (sum_x * sum_y / count);
            let var_x_part = sum_x2 - (sum_x * sum_x / count);

            if var_x_part.abs() > NumT::epsilon() {
              r[i] = cov_num / var_x_part;
            } else {
              r[i] = NumT::nan();
            }
          }
        }
      } else {
        let mut sum_x = NumT::zero();
        let mut sum_y = NumT::zero();
        let mut sum_xy = NumT::zero();
        let mut sum_x2 = NumT::zero();
        let mut nan_in_window = 0;

        let pre_fill_start = if start >= periods { start - periods } else { 0 };

        for k in pre_fill_start..start {
          let val_x = x[k];
          let val_y = y[k];
          if is_normal(&val_x) && is_normal(&val_y) {
            sum_x = sum_x + val_x;
            sum_y = sum_y + val_y;
            sum_xy = sum_xy + val_x * val_y;
            sum_x2 = sum_x2 + val_x * val_x;
          } else {
            nan_in_window += 1;
          }
        }

        for i in start..end {
          let val_x = x[i];
          let val_y = y[i];
          let is_valid = is_normal(&val_x) && is_normal(&val_y);

          if is_valid {
            sum_x = sum_x + val_x;
            sum_y = sum_y + val_y;
            sum_xy = sum_xy + val_x * val_y;
            sum_x2 = sum_x2 + val_x * val_x;
          } else {
            nan_in_window += 1;
          }

          if i >= periods {
            let old_x = x[i - periods];
            let old_y = y[i - periods];
            if is_normal(&old_x) && is_normal(&old_y) {
              sum_x = sum_x - old_x;
              sum_y = sum_y - old_y;
              sum_xy = sum_xy - old_x * old_y;
              sum_x2 = sum_x2 - old_x * old_x;
            } else {
              nan_in_window -= 1;
            }
          }

          if nan_in_window > 0 || !is_valid {
            // Result NaN
          } else {
            let mut valid = false;
            if i >= periods - 1 {
              valid = true;
            }

            if valid && periods > 1 {
              let count = NumT::from(periods).unwrap();
              let cov_num = sum_xy - (sum_x * sum_y / count);
              let var_x_part = sum_x2 - (sum_x * sum_x / count);

              if var_x_part.abs() > NumT::epsilon() {
                r[i] = cov_num / var_x_part;
              } else {
                r[i] = NumT::nan();
              }
            } else {
              r[i] = NumT::nan();
            }
          }
        }
      }
    });

  Ok(())
}

/// Calculate Regression Residual of Y on X over a moving window
///
/// Returns the residual of the last point: epsilon = Y - (alpha + beta * X)
pub fn ta_regresi<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  y: &[NumT],
  x: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != y.len() || y.len() != x.len() {
    return Err(Error::LengthMismatch(r.len(), y.len()));
  }


  let chunk_size = ctx.chunk_size(r.len());

  r.par_chunks_mut(chunk_size)
    .zip(y.par_chunks(chunk_size))
    .zip(x.par_chunks(chunk_size))
    .for_each(|((r, y), x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      if ctx.is_skip_nan() {
        let mut sum_x = NumT::zero();
        let mut sum_y = NumT::zero();
        let mut sum_xy = NumT::zero();
        let mut sum_x2 = NumT::zero();
        let mut no_nan_count = 0;
        let mut win_start = start;

        for i in start..end {
          let val_x = x[i];
          let val_y = y[i];
          let is_valid = is_normal(&val_x) && is_normal(&val_y);

          if is_valid {
            sum_x = sum_x + val_x;
            sum_y = sum_y + val_y;
            sum_xy = sum_xy + val_x * val_y;
            sum_x2 = sum_x2 + val_x * val_x;
            no_nan_count += 1;
          }

          while no_nan_count > periods {
            let old_x = x[win_start];
            let old_y = y[win_start];
            if is_normal(&old_x) && is_normal(&old_y) {
              sum_x = sum_x - old_x;
              sum_y = sum_y - old_y;
              sum_xy = sum_xy - old_x * old_y;
              sum_x2 = sum_x2 - old_x * old_x;
              no_nan_count -= 1;
            }
            win_start += 1;
          }

          while win_start <= i && !(is_normal(&x[win_start]) && is_normal(&y[win_start])) {
            win_start += 1;
          }

          if !is_valid {
            continue;
          }

          let mut should_output = true;
          if ctx.is_strictly_cycle() {
            if no_nan_count != periods || (i - win_start + 1) != periods {
              should_output = false;
            }
          }

          if should_output && no_nan_count > 1 {
            let count = NumT::from(no_nan_count).unwrap();
            let cov_num = sum_xy - (sum_x * sum_y / count);
            let var_x_part = sum_x2 - (sum_x * sum_x / count);

            if var_x_part.abs() > NumT::epsilon() {
              let beta = cov_num / var_x_part;
              let mean_x = sum_x / count;
              let mean_y = sum_y / count;
              let residual = (val_y - mean_y) - beta * (val_x - mean_x);
              r[i] = residual;
            } else {
              r[i] = NumT::nan();
            }
          }
        }
      } else {
        let mut sum_x = NumT::zero();
        let mut sum_y = NumT::zero();
        let mut sum_xy = NumT::zero();
        let mut sum_x2 = NumT::zero();
        let mut nan_in_window = 0;

        let pre_fill_start = if start >= periods { start - periods } else { 0 };

        for k in pre_fill_start..start {
          let val_x = x[k];
          let val_y = y[k];
          if is_normal(&val_x) && is_normal(&val_y) {
            sum_x = sum_x + val_x;
            sum_y = sum_y + val_y;
            sum_xy = sum_xy + val_x * val_y;
            sum_x2 = sum_x2 + val_x * val_x;
          } else {
            nan_in_window += 1;
          }
        }

        for i in start..end {
          let val_x = x[i];
          let val_y = y[i];
          let is_valid = is_normal(&val_x) && is_normal(&val_y);

          if is_valid {
            sum_x = sum_x + val_x;
            sum_y = sum_y + val_y;
            sum_xy = sum_xy + val_x * val_y;
            sum_x2 = sum_x2 + val_x * val_x;
          } else {
            nan_in_window += 1;
          }

          if i >= periods {
            let old_x = x[i - periods];
            let old_y = y[i - periods];
            if is_normal(&old_x) && is_normal(&old_y) {
              sum_x = sum_x - old_x;
              sum_y = sum_y - old_y;
              sum_xy = sum_xy - old_x * old_y;
              sum_x2 = sum_x2 - old_x * old_x;
            } else {
              nan_in_window -= 1;
            }
          }

          if nan_in_window > 0 || !is_valid {
            // Result NaN
          } else {
            let mut valid = false;
            if i >= periods - 1 {
              valid = true;
            }

            if valid && periods > 1 {
              let count = NumT::from(periods).unwrap();
              let cov_num = sum_xy - (sum_x * sum_y / count);
              let var_x_part = sum_x2 - (sum_x * sum_x / count);

              if var_x_part.abs() > NumT::epsilon() {
                let beta = cov_num / var_x_part;
                let mean_x = sum_x / count;
                let mean_y = sum_y / count;
                let residual = (val_y - mean_y) - beta * (val_x - mean_x);
                r[i] = residual;
              } else {
                r[i] = NumT::nan();
              }
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
  use crate::algo::{assert_vec_eq_nan, context::FLAG_SKIP_NAN};

  #[test]
  fn test_var() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_var(&ctx, &mut r, &input, periods).unwrap();

    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 1.0, 1.0, 1.0]);
  }

  #[test]
  fn test_cov() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![2.0, 3.0, 4.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; x.len()];
    let ctx = Context::new(0, 0, 0);

    ta_cov(&ctx, &mut r, &x, &y, periods).unwrap();

    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 1.0, 1.0]);
  }

  #[test]
  fn test_cov_neg() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![3.0, 2.0, 1.0];
    let periods = 3;
    let mut r = vec![0.0; x.len()];
    let ctx = Context::new(0, 0, 0);

    ta_cov(&ctx, &mut r, &x, &y, periods).unwrap();

    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, -1.0]);
  }

  #[test]
  fn test_cov_skip_nan() {
    let x = vec![1.0, f64::NAN, 2.0, 3.0];
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let periods = 2; // small window
    let mut r = vec![0.0; x.len()];
    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);

    ta_cov(&ctx, &mut r, &x, &y, periods).unwrap();

    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 1.0, 0.5]);
  }

  #[test]
  fn test_corr() {
    // x: [1, 2, 3] mean=2, std=1
    // y: [2, 3, 4] mean=3, std=1
    // Cov=1. Corr=1/1 = 1.
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];
    let periods = 3;
    let mut r = vec![0.0; x.len()];
    let ctx = Context::new(0, 0, 0);
    ta_corr2(&ctx, &mut r, &x, &y, periods).unwrap();

    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 1.0, 1.0, 1.0]);
  }

  #[test]
  fn test_corr_neg() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![3.0, 2.0, 1.0];
    let periods = 3;
    let mut r = vec![0.0; x.len()];
    let ctx = Context::new(0, 0, 0);
    ta_corr2(&ctx, &mut r, &x, &y, periods).unwrap();
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, -1.0]);
  }

  #[test]
  fn test_regbeta() {
    // x = [1, 2, 3]
    // y = [2, 4, 6] -> y = 2x. Beta = 2.
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![2.0, 4.0, 6.0, 8.0];
    let periods = 3;
    let mut r = vec![0.0; x.len()];
    let ctx = Context::new(0, 0, 0);
    ta_regbeta(&ctx, &mut r, &y, &x, periods).unwrap(); // Note: y, x order

    // Period 3 window: [1,2,3] vs [2,4,6]. Beta=2.
    // Period 4 window: [2,3,4] vs [4,6,8]. Beta=2.
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 2.0, 2.0]);
  }

  #[test]
  fn test_regresi() {
    // x = [1, 2, 3]
    // y = [2.1, 3.9, 6.1] -> approx y = 2x.
    // Regression on 3 points:
    // Window [0,1,2]: x=[1,2,3], y=[2.1, 3.9, 6.1]
    // mean_x = 2. mean_y = (2.1+3.9+6.1)/3 = 4.0333
    // Calculate beta manually or just check low residual.
    // But let's test perfect fit first.
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 4.0, 6.0];
    let periods = 3;
    let mut r = vec![0.0; x.len()];
    let ctx = Context::new(0, 0, 0);
    ta_regresi(&ctx, &mut r, &y, &x, periods).unwrap();

    // Perfect fit -> residual 0
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 0.0]);

    // Test with error
    // x=[1,2,3], y=[2,4,7]
    // mean_x = 2
    // mean_y = 13/3 = 4.333
    // var_x = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 2 = 2/2 = 1.
    // cov_xy = ((1-2)(2-4.333) + (2-2)(...) + (3-2)(7-4.333)) / 2
    //        = (-1 * -2.333 + 0 + 1 * 2.666) / 2 = (2.333 + 2.666) / 2 = 5/2 = 2.5
    // beta = 2.5 / 1 = 2.5
    // alpha = 4.333 - 2.5*2 = 4.333 - 5 = -0.666
    // residual_3 = y_3 - (alpha + beta*x_3) = 7 - (-0.666 + 2.5*3) = 7 - (-0.666 + 7.5) = 7 - 6.833 = 0.166...
    // Formula check: (y - mean_y) - beta*(x - mean_x)
    // (7 - 4.333) - 2.5 * (3 - 2) = 2.666 - 2.5 = 0.1666...
    let x2 = vec![1.0, 2.0, 3.0];
    let y2 = vec![2.0, 4.0, 7.0];
    ta_regresi(&ctx, &mut r, &y2, &x2, periods).unwrap();
    // 0.166666...
    assert!((r[2] - 1.0 / 6.0).abs() < 1e-5);
  }
}
