// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use std::{cmp::Ordering, collections::BTreeMap, fmt::Debug};

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error};

#[derive(Copy, Clone, Default, PartialEq, PartialOrd)]
struct OrderedFloat<NumT: Float> {
  value: NumT,
}

impl<NumT: Float> OrderedFloat<NumT> {
  pub fn new(value: NumT) -> OrderedFloat<NumT> {
    OrderedFloat { value }
  }
}

impl<NumT: Float> Ord for OrderedFloat<NumT> {
  fn cmp(&self, other: &Self) -> Ordering {
    match self.value.partial_cmp(&other.value) {
      Some(ord) => ord,
      None => {
        if self.value.is_finite() {
          return Ordering::Greater;
        }
        if self.value.is_infinite() {
          return Ordering::Less;
        }
        return Ordering::Less;
      }
    }
  }
}
impl<NumT: Float> Eq for OrderedFloat<NumT> {}

impl<NumT: Float> From<NumT> for OrderedFloat<NumT> {
  fn from(value: NumT) -> Self {
    OrderedFloat::new(value)
  }
}

impl<NumT: Float + Debug> Debug for OrderedFloat<NumT> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{:?}", self.value)
  }
}

/// Calculate rank in a sliding window with size `periods`
pub fn ta_ts_rank<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }

  if periods == 1 {
    r.fill(NumT::from(1.0).unwrap());
    return Ok(());
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      r.fill(NumT::nan());
      let mut rank_window: BTreeMap<OrderedFloat<NumT>, usize> = BTreeMap::new();
      for i in start..x.len() {
        let val = x[i].into();
        if rank_window.len() < periods {
          rank_window.insert(val, i);
        } else {
          rank_window.remove(&x[i - periods].into());
          rank_window.insert(val, i);
        }

        let rank = rank_window
          .iter()
          .position(|v| val.eq(v.0))
          .unwrap_or(rank_window.len() - 1)
          + 1;
        if ctx.is_strictly_cycle() && rank_window.len() < periods {
          continue;
        }
        r[i] = NumT::from(rank).unwrap();
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

/// Calculate rank percentage cross group dimension, the ctx.groups() is the number of groups
/// Same value are averaged
pub fn ta_rank<NumT: Float + Send + Sync + Debug>(
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
    return ta_ts_rank(ctx, r, input, 0);
  }

  if r.len() != group_size * groups {
    // ensure data is complete
    return Err(Error::LengthMismatch(r.len(), group_size * groups));
  }

  let r = UnsafePtr::new(r.as_mut_ptr(), r.len());
  (0..group_size).into_par_iter().for_each(|j| {
    let mut rank_window: Vec<(OrderedFloat<NumT>, usize)> = Vec::new();
    for i in 0..groups {
      let idx = i * group_size + j;
      rank_window.push((input[idx].into(), idx));
    }
    rank_window.sort_by(|a, b| a.0.cmp(&b.0));
    let r = r.get();

    let mut prev_rank_value = rank_window[0].0.value;
    let mut s = 0;
    let total = NumT::from(rank_window.len()).unwrap();

    // chunk by same value
    for e in 0..rank_window.len() {
      if prev_rank_value == rank_window[e].0.value {
        continue;
      }
      let rank_avg = NumT::from(e + s + 1).unwrap() / NumT::from(2usize).unwrap();
      for i in s..e {
        r[rank_window[i].1] = rank_avg / total;
      }
      s = e;
      prev_rank_value = rank_window[e].0.value;
    }

    // the last chunk
    let rank_avg = NumT::from(rank_window.len() + s + 1).unwrap() / NumT::from(2usize).unwrap();
    for i in s..rank_window.len() {
      r[rank_window[i].1] = rank_avg / total;
    }
  });

  Ok(())
}

/// Discretize the input into n bins, the ctx.groups() is the number of groups
///
/// Bins are 0-based index.
/// Same value are assigned to the same bin.
pub fn ta_bins<NumT: Float + Send + Sync + Debug>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  bins: usize,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }

  let group_size = ctx.chunk_size(r.len()) as usize;
  let groups = ctx.groups() as usize;

  if r.len() != group_size * groups {
    return Err(Error::LengthMismatch(r.len(), group_size * groups));
  }

  // If bins is 0 or 1, everything is bin 0 (or error?)
  // If bins=1, all 0.
  if bins == 0 {
     // Return 0 or error? Let's return 0.
     r.fill(NumT::zero());
     return Ok(());
  }

  let r_ptr = UnsafePtr::new(r.as_mut_ptr(), r.len());
  (0..group_size).into_par_iter().for_each(|j| {
    let mut rank_window: Vec<(OrderedFloat<NumT>, usize)> = Vec::new();
    for i in 0..groups {
      let idx = i * group_size + j;
      // Handle NaN: usually we skip NaNs or put them in a separate bin?
      // ta_rank includes NaNs in sorting?
      // OrderedFloat handles NaNs (NaN > infinite > finite).
      // So NaNs will be at the end.
      // If we include NaNs in binning, they will be in the last bin.
      // Usually we want NaNs to remain NaN.
      // Let's check if we should skip NaNs.
      // ta_rank implementation sorts everything. OrderedFloat treats NaN as > all.
      // So NaNs get the highest rank.
      // If we want NaNs to be NaN in output, we should filter them.
      // But ta_rank output: `r[rank_window[i].1] = rank_avg / total;`.
      // So NaNs get a rank.
      // If user wants to skip NaNs, they should use FLAG_SKIP_NAN?
      // But ta_rank doesn't check ctx.is_skip_nan().
      // ta_ts_rank uses skip_nan_window if ctx.is_skip_nan() (implied? No, ta_ts_rank doesn't seem to check flag either in the snippet I saw, wait. ta_linear_reg checked.)
      // ta_ts_rank code:
      // if ctx.is_strictly_cycle() ...
      // It doesn't seem to handle SKIP_NAN explicitly in the snippet I saw.
      // But `OrderedFloat` puts NaN at the end.
      // Let's stick to ta_rank behavior: process everything.
      // However, for "bins", usually NaNs should be NaN.
      // If I sort and bin, NaNs will be in the top bin.
      // Let's check `is_normal` usage in other files.
      // In `returns.rs`, `ta_fret` checks `is_normal`.
      // In `rank.rs`, `ta_rank` does NOT check `is_normal`. It sorts everything.
      // So `ta_bins` should probably follow `ta_rank`.
      
      rank_window.push((input[idx].into(), idx));
    }
    rank_window.sort_by(|a, b| a.0.cmp(&b.0));
    let r = r_ptr.get();

    // We need to count valid values if we want to ignore NaNs?
    // But ta_rank counts everything.
    // I will stick to exact ta_rank logic but map to bins.
    
    let mut prev_rank_value = rank_window[0].0.value;
    let mut s = 0;
    let total = NumT::from(rank_window.len()).unwrap();
    let bins_t = NumT::from(bins).unwrap();

    // chunk by same value
    for e in 0..rank_window.len() {
      if prev_rank_value == rank_window[e].0.value {
        continue;
      }
      // Process chunk s..e
      let rank_avg = NumT::from(e + s + 1).unwrap() / NumT::from(2usize).unwrap();
      // rank_avg is 1-based average rank.
      // formula: floor((rank_avg - 1) * bins / total)
      // but ensure result is in [0, bins-1]
      
      let val = rank_window[s].0.value;
      let bin = if val.is_nan() {
          NumT::nan() // If value is NaN, return NaN
          // OrderedFloat treats NaN as largest.
          // If input has NaN, ta_rank assigns high rank.
          // If we want bins, maybe NaN should be NaN.
          // Let's assume we preserve NaN if value is not normal.
      } else {
          let b = ((rank_avg - NumT::one()) * bins_t / total).floor();
          // clamp to bins-1 (although logic says it should be < bins)
          if b >= bins_t { bins_t - NumT::one() } else { b }
      };

      for i in s..e {
        r[rank_window[i].1] = bin;
      }
      s = e;
      prev_rank_value = rank_window[e].0.value;
    }

    // the last chunk
    let rank_avg = NumT::from(rank_window.len() + s + 1).unwrap() / NumT::from(2usize).unwrap();
    let val = rank_window[s].0.value;
    let bin = if val.is_nan() {
        NumT::nan()
    } else {
        let b = ((rank_avg - NumT::one()) * bins_t / total).floor();
        if b >= bins_t { bins_t - NumT::one() } else { b }
    };
    
    for i in s..rank_window.len() {
      r[rank_window[i].1] = bin;
    }
  });

  Ok(())
}


#[cfg(test)]
mod tests {
  use super::*;
  use crate::algo::assert_vec_eq_nan;

  #[test]
  fn test_ta_ts_rank_simple() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);

    ta_ts_rank(&ctx, &mut r, &input, 3).unwrap();
    // Position 0: [1] -> rank 1
    // Position 1: [1,2] -> rank of 2 is 2
    // Position 2: [1,2,3] -> rank of 3 is 3
    // Position 3: [2,3,4] -> rank of 4 is 3
    // Position 4: [3,4,5] -> rank of 5 is 3
    assert_vec_eq_nan(&r, &vec![1.0, 2.0, 3.0, 3.0, 3.0]);
  }

  #[test]
  fn test_ta_ts_rank_periods_one() {
    let input = vec![1.0, 2.0, 3.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);

    ta_ts_rank(&ctx, &mut r, &input, 1).unwrap();
    assert_vec_eq_nan(&r, &vec![1.0, 1.0, 1.0]);
  }

  #[test]
  fn test_ta_ts_rank_with_nan() {
    let input = vec![1.0, f64::NAN, 3.0, 4.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);

    ta_ts_rank(&ctx, &mut r, &input, 3).unwrap();
    // NaN gets the highest rank since NaN != NaN
    // Position 0: [1] -> 1
    // Position 1: [1, NaN] -> NaN rank 2 (highest)
    // Position 2: [1, NaN, 3] -> 3 rank 3
    // Position 3: [NaN, 3, 4] -> 4 rank 3
    assert_vec_eq_nan(&r, &vec![1.0, 2.0, 3.0, 3.0]);
  }

  #[test]
  fn test_ta_rank_same_value() {
    let input = vec![1.0, 2.0, 1.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 3, 0);

    ta_rank(&ctx, &mut r, &input).unwrap();
    assert_vec_eq_nan(&r, &vec![0.5, 1.0, 0.5]);
  }

  #[test]
  fn test_ta_rank_simple() {
    let input = vec![3.0, 1.0, 2.0, 4.0]; // groups=2, matrix [3,2; 1,4]
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 2, 0);

    ta_rank(&ctx, &mut r, &input).unwrap();
    // j=0: values [3,2], sorted [2,3], ranks [2,1] at indices 0,2
    // j=1: values [1,4], sorted [1,4], ranks [1,2] at indices 1,3
    assert_vec_eq_nan(&r, &vec![1.0, 0.5, 0.5, 1.0]);
  }

  #[test]
  fn test_ta_rank_three_groups() {
    let input = vec![3.0, 1.0, 2.0, 5.0, 4.0, 6.0]; // groups=3, matrix [3,2; 1,5; 4,6]
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 3, 0);

    ta_rank(&ctx, &mut r, &input).unwrap();
    // j=0: values [3,2,4], sorted [2,3,4], ranks [2,1,3] at 0,2,4
    // j=1: values [1,5,6], sorted [1,5,6], ranks [1,2,3] at 1,3,5
    assert_vec_eq_nan(
      &r,
      &vec![
        2.0 / 3.0,
        1.0 / 3.0,
        1.0 / 3.0,
        2.0 / 3.0,
        3.0 / 3.0,
        3.0 / 3.0,
      ],
    );
  }
}
