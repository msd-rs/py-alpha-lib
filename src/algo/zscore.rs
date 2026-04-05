// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use std::fmt::Debug;

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal, skip_nan_window::SkipNanWindow};

/// Calculate rolling Z-Score over a moving window
///
/// Z-Score = (x - mean) / stddev, computed over a rolling window of `periods`.
/// Uses sample stddev (ddof=1) to match pandas.
pub fn ta_zscore<NumT: Float + Send + Sync>(
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

          let mut should_output = true;
          if ctx.is_strictly_cycle() {
            if i.no_nan_count != periods || (i.end - i.start + 1) != periods {
              should_output = false;
            }
          }

          if should_output && i.no_nan_count > 1 {
            let count = NumT::from(i.no_nan_count).unwrap();
            let mean = sum / count;
            let var_num = sum_sq - (sum * sum / count);
            let var = var_num / (count - NumT::one());
            if var < NumT::zero() || var.abs() < NumT::epsilon() {
              r[i.end] = NumT::zero();
            } else {
              r[i.end] = (val - mean) / var.sqrt();
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
            // NaN
          } else if i >= periods - 1 && periods > 1 {
            let count = NumT::from(periods).unwrap();
            let mean = sum / count;
            let var_num = sum_sq - (sum * sum / count);
            let var = var_num / (count - NumT::one());
            if var < NumT::zero() || var.abs() < NumT::epsilon() {
              r[i] = NumT::zero();
            } else {
              r[i] = (val - mean) / var.sqrt();
            }
          }
        }
      }
    });

  Ok(())
}

#[derive(Debug, Clone, Copy)]
struct UnsafePtr<NumT: Float> {
  ptr: *mut NumT,
  len: usize,
}

impl<NumT: Float> UnsafePtr<NumT> {
  pub fn new(ptr: *mut NumT, len: usize) -> Self {
    UnsafePtr { ptr, len }
  }

  pub fn get(&self) -> &mut [NumT] {
    unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
  }
}

unsafe impl<NumT: Float> Send for UnsafePtr<NumT> {}
unsafe impl<NumT: Float> Sync for UnsafePtr<NumT> {}

/// Calculate cross-sectional Z-Score across groups at each time step
///
/// Z-Score = (x - mean) / stddev, computed across all groups for each time position.
/// NaN values are excluded from mean/stddev computation. NaN input produces NaN output.
pub fn ta_cc_zscore<NumT: Float + Send + Sync + Debug>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }


  let group_size = ctx.chunk_size(r.len()) as usize;
  let groups = ctx.groups() as usize;

  if ctx.groups() < 2 {
    // Fall back to rolling zscore with full window
    return ta_zscore(ctx, r, input, 0);
  }

  if r.len() != group_size * groups {
    return Err(Error::LengthMismatch(r.len(), group_size * groups));
  }


  let r = UnsafePtr::new(r.as_mut_ptr(), r.len());
  (0..group_size).into_par_iter().for_each(|j| {
    let r = r.get();

    // Collect values across groups for this time position
    let mut sum = NumT::zero();
    let mut sum_sq = NumT::zero();
    let mut valid_count: usize = 0;

    for i in 0..groups {
      let idx = i * group_size + j;
      let val = input[idx];
      if is_normal(&val) {
        sum = sum + val;
        sum_sq = sum_sq + val * val;
        valid_count += 1;
      }
    }

    if valid_count < 2 {
      // Not enough data for zscore, set all to NaN
      for i in 0..groups {
        r[i * group_size + j] = NumT::nan();
      }
      return;
    }

    let count = NumT::from(valid_count).unwrap();
    let mean = sum / count;
    let var_num = sum_sq - (sum * sum / count);
    let var = var_num / (count - NumT::one());

    if var < NumT::zero() || var.abs() < NumT::epsilon() {
      // Zero variance: all values same, zscore = 0
      for i in 0..groups {
        let idx = i * group_size + j;
        if is_normal(&input[idx]) {
          r[idx] = NumT::zero();
        } else {
          r[idx] = NumT::nan();
        }
      }
      return;
    }

    let std = var.sqrt();
    for i in 0..groups {
      let idx = i * group_size + j;
      let val = input[idx];
      if is_normal(&val) {
        r[idx] = (val - mean) / std;
      } else {
        r[idx] = NumT::nan();
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
  fn test_ts_zscore_simple() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_zscore(&ctx, &mut r, &input, periods).unwrap();

    // Window [1,2,3]: mean=2, std=1. zscore(3)=(3-2)/1=1
    // Window [2,3,4]: mean=3, std=1. zscore(4)=(4-3)/1=1
    // Window [3,4,5]: mean=4, std=1. zscore(5)=(5-4)/1=1
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 1.0, 1.0, 1.0]);
  }

  #[test]
  fn test_ts_zscore_negative() {
    let input = vec![3.0, 2.0, 1.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_zscore(&ctx, &mut r, &input, periods).unwrap();

    // Window [3,2,1]: mean=2, std=1. zscore(1)=(1-2)/1=-1
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, -1.0]);
  }

  #[test]
  fn test_ts_zscore_constant() {
    let input = vec![5.0, 5.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_zscore(&ctx, &mut r, &input, periods).unwrap();

    // All same -> std=0, zscore=0
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 0.0]);
  }

  #[test]
  fn test_zscore_cross_sectional() {
    // groups=3, group_size=2
    // matrix: [1, 10; 3, 20; 5, 30]
    let input = vec![1.0, 10.0, 3.0, 20.0, 5.0, 30.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 3, 0);

    ta_cc_zscore(&ctx, &mut r, &input).unwrap();

    // j=0: values [1, 3, 5]. mean=3, std=2. z = [-1, 0, 1]
    // j=1: values [10, 20, 30]. mean=20, std=10. z = [-1, 0, 1]
    assert_vec_eq_nan(&r, &vec![-1.0, -1.0, 0.0, 0.0, 1.0, 1.0]);
  }

  #[test]
  fn test_zscore_cross_with_nan() {
    // groups=3, group_size=1
    let input = vec![1.0, f64::NAN, 5.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 3, 0);

    ta_cc_zscore(&ctx, &mut r, &input).unwrap();

    // j=0: values [1, NaN, 5]. valid=[1,5], mean=3, std=sqrt(8)=2*sqrt(2)
    // z(1) = (1-3)/(2*sqrt(2)) = -2/(2*sqrt(2)) = -1/sqrt(2) = -0.707106...
    // z(NaN) = NaN
    // z(5) = (5-3)/(2*sqrt(2)) = 1/sqrt(2) = 0.707106...
    let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
    assert_vec_eq_nan(&r, &vec![-inv_sqrt2, f64::NAN, inv_sqrt2]);
  }
}
