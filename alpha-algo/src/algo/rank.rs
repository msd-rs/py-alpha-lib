// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use std::{cmp::Ordering, collections::BTreeMap, fmt::Debug};

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal, skip_nan_window::SkipNanWindow};

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
    match (self.value.is_nan(), other.value.is_nan()) {
      (true, true) => Ordering::Equal,
      (true, false) => Ordering::Less,
      (false, true) => Ordering::Greater,
      (false, false) => self
        .value
        .partial_cmp(&other.value)
        .unwrap_or(Ordering::Less),
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
///
/// Uses min-rank method for ties (same as pandas rankdata method='min').
/// Current NaN values produce NaN output.
pub fn ta_rank<NumT: Float + Send + Sync>(
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

  if periods == 1 {
    r.fill(NumT::from(1.0).unwrap());
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
        let mut rank_window: BTreeMap<OrderedFloat<NumT>, u32> = BTreeMap::new();
        for i in iter {
          let val = x[i.end];
          if is_normal(&val) {
            *rank_window.entry(val.into()).or_insert(0) += 1;
          }

          // remove values that fell out of window
          for k in i.prev_start..i.start {
            let old = x[k];
            if is_normal(&old) {
              let old_key: OrderedFloat<NumT> = old.into();
              if let Some(count) = rank_window.get_mut(&old_key) {
                *count -= 1;
                if *count == 0 {
                  rank_window.remove(&old_key);
                }
              }
            }
          }

          if !is_normal(&val) {
            continue;
          }

          if ctx.is_strictly_cycle() {
            if i.no_nan_count < periods || (i.end - i.start + 1) != periods {
              continue;
            }
          }

          let val_key: OrderedFloat<NumT> = val.into();
          let mut less_count: usize = 0;
          for (key, count) in rank_window.iter() {
            if *key < val_key {
              less_count += *count as usize;
            } else {
              break;
            }
          }
          r[i.end] = NumT::from(less_count + 1).unwrap();
        }
      } else {
        let mut rank_window: BTreeMap<OrderedFloat<NumT>, u32> = BTreeMap::new();
        let mut nan_in_window = 0;

        let pre_fill_start = if start >= periods { start - periods } else { 0 };
        for k in pre_fill_start..start {
          let val = x[k];
          if is_normal(&val) {
            *rank_window.entry(val.into()).or_insert(0) += 1;
          } else {
            nan_in_window += 1;
          }
        }

        for i in start..end {
          let val = x[i];
          if is_normal(&val) {
            *rank_window.entry(val.into()).or_insert(0) += 1;
          } else {
            nan_in_window += 1;
          }

          if i >= periods {
            let old = x[i - periods];
            if is_normal(&old) {
              let old_key: OrderedFloat<NumT> = old.into();
              if let Some(count) = rank_window.get_mut(&old_key) {
                *count -= 1;
                if *count == 0 {
                  rank_window.remove(&old_key);
                }
              }
            } else {
              nan_in_window -= 1;
            }
          }

          if !is_normal(&val) {
            continue;
          }

          if nan_in_window > 0 {
            continue;
          }

          if ctx.is_strictly_cycle() && i < periods - 1 {
            continue;
          }

          let val_key: OrderedFloat<NumT> = val.into();
          let mut less_count: usize = 0;
          for (key, count) in rank_window.iter() {
            if *key < val_key {
              less_count += *count as usize;
            } else {
              break;
            }
          }
          r[i] = NumT::from(less_count + 1).unwrap();
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

/// Calculate rank percentage cross group dimension, the ctx.groups() is the number of groups
/// Same value are averaged
pub fn ta_cc_rank<NumT: Float + Send + Sync + Debug>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }

  if r.is_empty() {
    return Ok(());
  }

  let group_size = ctx.chunk_size(r.len()) as usize;
  let groups = ctx.groups() as usize;

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

    // OrderedFloat sorts NaN to the beginning (NaN < everything).
    // Find where valid (non-NaN) values start.
    let nan_count = rank_window
      .iter()
      .take_while(|v| v.0.value.is_nan())
      .count();
    let valid_count = rank_window.len() - nan_count;

    // Assign NaN to NaN positions
    for i in 0..nan_count {
      r[rank_window[i].1] = NumT::nan();
    }

    if valid_count == 0 {
      return;
    }

    let total = NumT::from(valid_count).unwrap();

    // Rank only valid values (starting from nan_count)
    let mut prev_rank_value = rank_window[nan_count].0.value;
    let mut s = nan_count;

    // chunk by same value
    for e in nan_count..rank_window.len() {
      if prev_rank_value == rank_window[e].0.value {
        continue;
      }
      let rank_avg =
        NumT::from(e - nan_count + s - nan_count + 1).unwrap() / NumT::from(2usize).unwrap();
      for i in s..e {
        r[rank_window[i].1] = rank_avg / total;
      }
      s = e;
      prev_rank_value = rank_window[e].0.value;
    }

    // the last chunk of valid values
    let rank_avg =
      NumT::from(valid_count + s - nan_count + 1).unwrap() / NumT::from(2usize).unwrap();
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

  // If bins is 0, everything is 0
  if bins == 0 {
    r.fill(NumT::zero());
    return Ok(());
  }

  let r_ptr = UnsafePtr::new(r.as_mut_ptr(), r.len());
  (0..group_size).into_par_iter().for_each(|j| {
    let mut rank_window: Vec<(OrderedFloat<NumT>, usize)> = Vec::new();
    for i in 0..groups {
      let idx = i * group_size + j;
      rank_window.push((input[idx].into(), idx));
    }
    rank_window.sort_by(|a, b| a.0.cmp(&b.0));
    let r = r_ptr.get();

    // OrderedFloat sorts NaN to the beginning (NaN < everything).
    // Find where valid (non-NaN) values start.
    let nan_count = rank_window
      .iter()
      .take_while(|v| v.0.value.is_nan())
      .count();
    let valid_count = rank_window.len() - nan_count;

    // Assign NaN to NaN positions
    for i in 0..nan_count {
      r[rank_window[i].1] = NumT::nan();
    }

    if valid_count == 0 {
      return;
    }

    let total = NumT::from(valid_count).unwrap();
    let bins_t = NumT::from(bins).unwrap();

    let mut prev_rank_value = rank_window[nan_count].0.value;
    let mut s = nan_count;

    // chunk by same value
    for e in nan_count..rank_window.len() {
      if prev_rank_value == rank_window[e].0.value {
        continue;
      }
      let rank_avg =
        NumT::from(e - nan_count + s - nan_count + 1).unwrap() / NumT::from(2usize).unwrap();
      let b = ((rank_avg - NumT::one()) * bins_t / total).floor();
      let bin = if b >= bins_t {
        bins_t - NumT::one()
      } else if b < NumT::zero() {
        NumT::zero()
      } else {
        b
      };

      for i in s..e {
        r[rank_window[i].1] = bin;
      }
      s = e;
      prev_rank_value = rank_window[e].0.value;
    }

    // the last chunk
    let rank_avg =
      NumT::from(valid_count + s - nan_count + 1).unwrap() / NumT::from(2usize).unwrap();
    let b = ((rank_avg - NumT::one()) * bins_t / total).floor();
    let bin = if b >= bins_t {
      bins_t - NumT::one()
    } else if b < NumT::zero() {
      NumT::zero()
    } else {
      b
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

    ta_rank(&ctx, &mut r, &input, 3).unwrap();
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

    ta_rank(&ctx, &mut r, &input, 1).unwrap();
    assert_vec_eq_nan(&r, &vec![1.0, 1.0, 1.0]);
  }

  #[test]
  fn test_ta_ts_rank_with_nan() {
    let input = vec![1.0, f64::NAN, 3.0, 4.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);

    ta_rank(&ctx, &mut r, &input, 3).unwrap();
    // flags=0: NaN in window causes NaN output
    assert_vec_eq_nan(&r, &vec![1.0, f64::NAN, f64::NAN, f64::NAN]);

    // FLAG_SKIP_NAN: skips NaN and keeps non-NaN rank
    let ctx_skip = Context::new(0, 0, 1);
    ta_rank(&ctx_skip, &mut r, &input, 3).unwrap();
    assert_vec_eq_nan(&r, &vec![1.0, f64::NAN, 2.0, 3.0]);
  }

  #[test]
  fn test_ta_ts_rank_duplicates() {
    // Test that duplicate values in the window are handled correctly.
    // The old BTreeMap<key, index> approach would lose duplicates.
    let input = vec![1.0, 1.0, 2.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);

    ta_rank(&ctx, &mut r, &input, 3).unwrap();
    // Position 0: [1] -> rank of 1 is 1
    // Position 1: [1, 1] -> rank of 1 is 1 (min rank for ties)
    // Position 2: [1, 1, 2] -> rank of 2 is 3 (two 1s below it)
    assert_vec_eq_nan(&r, &vec![1.0, 1.0, 3.0]);
  }

  #[test]
  fn test_ta_ts_rank_duplicates_sliding() {
    // Verify correct sliding window behavior when duplicates enter and leave.
    let input = vec![3.0, 1.0, 2.0, 1.0, 3.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);

    ta_rank(&ctx, &mut r, &input, 3).unwrap();
    // Position 0: [3] -> rank 1
    // Position 1: [3, 1] -> rank of 1 is 1
    // Position 2: [3, 1, 2] -> rank of 2 is 2
    // Position 3: [1, 2, 1] -> rank of 1 is 1 (min rank, two 1s and one 2)
    // Position 4: [2, 1, 3] -> rank of 3 is 3
    assert_vec_eq_nan(&r, &vec![1.0, 1.0, 2.0, 1.0, 3.0]);
  }

  #[test]
  fn test_ta_rank_same_value() {
    let input = vec![1.0, 2.0, 1.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 3, 0);

    ta_cc_rank(&ctx, &mut r, &input).unwrap();
    assert_vec_eq_nan(&r, &vec![0.5, 1.0, 0.5]);
  }

  #[test]
  fn test_ta_rank_simple() {
    let input = vec![3.0, 1.0, 2.0, 4.0]; // groups=2, matrix [3,2; 1,4]
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 2, 0);

    ta_cc_rank(&ctx, &mut r, &input).unwrap();
    // j=0: values [3,2], sorted [2,3], ranks [2,1] at indices 0,2
    // j=1: values [1,4], sorted [1,4], ranks [1,2] at indices 1,3
    assert_vec_eq_nan(&r, &vec![1.0, 0.5, 0.5, 1.0]);
  }

  #[test]
  fn test_ta_rank_three_groups() {
    let input = vec![3.0, 1.0, 2.0, 5.0, 4.0, 6.0]; // groups=3, matrix [3,2; 1,5; 4,6]
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 3, 0);

    ta_cc_rank(&ctx, &mut r, &input).unwrap();
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

  #[test]
  fn test_ta_rank_with_nan() {
    // groups=3, matrix [3,NaN; 1,5; 4,6]
    let n = 32;
    let g = 32;
    let mut input = vec![];
    (0..n).for_each(|i| {
      if i % 2 == 0 {
        (0..g).for_each(|_j| input.push(f64::NAN));
      } else if i % 3 == 0 {
        (0..g).for_each(|_j| input.push(f64::INFINITY));
      } else if i % 5 == 0 {
        (0..g).for_each(|_j| input.push(f64::NEG_INFINITY));
      } else {
        (0..g).for_each(|_j| input.push(i as f64));
      }
    });
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, g, 0);

    ta_cc_rank(&ctx, &mut r, &input).unwrap();
    // j=0: values [3,1,4], all valid, sorted [1,3,4], ranks [2,1,3]
    // j=1: values [NaN,5,6], 2 valid, sorted [5,6,NaN]
    //   5 -> rank 1/2=0.5, 6 -> rank 2/2=1.0, NaN -> NaN
    // assert_vec_eq_nan(
    //   &r,
    //   &vec![
    //     2.0 / 3.0,
    //     f64::NAN,
    //     1.0 / 3.0,
    //     1.0 / 2.0,
    //     3.0 / 3.0,
    //     2.0 / 2.0,
    //   ],
    // );
    println!("{:?}", input);
    println!("{:?}", r);
  }

  #[test]
  fn test_ta_rank_with_nan_inf() {
    let mut a = vec![
      OrderedFloat::new(f64::NAN),
      OrderedFloat::new(f64::INFINITY),
      OrderedFloat::new(f64::NEG_INFINITY),
      OrderedFloat::new(f64::NAN),
      OrderedFloat::new(f64::INFINITY),
      OrderedFloat::new(f64::NEG_INFINITY),
      OrderedFloat::new(f64::NAN),
      OrderedFloat::new(f64::NAN),
    ];

    println!("{:?}", a);
    a.sort_unstable();
    println!("{:?}", a);
    a.sort();
    println!("{:?}", a);
  }

  #[test]
  fn test_float_order() {
    let mut v = vec![
      f64::NAN,
      1.0,
      f64::NAN,
      f64::NEG_INFINITY,
      f64::INFINITY,
      0.0,
      -1.0,
      f64::MAX,
      f64::MIN,
      f64::NAN,
      1.0,
      f64::NAN,
      f64::NEG_INFINITY,
      f64::INFINITY,
      0.0,
      -1.0,
      f64::MAX,
      f64::MIN,
      f64::NAN,
      1.0,
      f64::NAN,
      f64::NEG_INFINITY,
      f64::INFINITY,
      0.0,
      -1.0,
      f64::MAX,
      f64::MIN,
    ]
    .into_iter()
    .map(OrderedFloat::from)
    .collect::<Vec<_>>();

    v.sort_by(OrderedFloat::cmp);

    println!("{:?}", v);
  }
}
