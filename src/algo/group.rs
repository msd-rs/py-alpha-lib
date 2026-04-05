// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use std::{cmp::Ordering, fmt::Debug};

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal};

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
        Ordering::Less
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

/// Calculate rank percentage within each category group at each time step
///
/// For each time position, groups items by `category` value, then computes
/// rank percentage within each group. Same value gets averaged rank.
/// NaN in category or input produces NaN output.
pub fn ta_group_rank<NumT: Float + Send + Sync + Debug>(
  ctx: &Context,
  r: &mut [NumT],
  category: &[NumT],
  input: &[NumT],
) -> Result<(), Error> {
  if r.len() != input.len() || r.len() != category.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }


  let group_size = ctx.chunk_size(r.len()) as usize;
  let groups = ctx.groups() as usize;

  if r.len() != group_size * groups {
    return Err(Error::LengthMismatch(r.len(), group_size * groups));
  }


  let r_ptr = UnsafePtr::new(r.as_mut_ptr(), r.len());
  (0..group_size).into_par_iter().for_each(|j| {
    let r = r_ptr.get();

    // Initialize all to NaN
    for i in 0..groups {
      r[i * group_size + j] = NumT::nan();
    }

    // Collect (category, value, index) for valid items
    let mut items: Vec<(OrderedFloat<NumT>, OrderedFloat<NumT>, usize)> =
      Vec::with_capacity(groups);
    for i in 0..groups {
      let idx = i * group_size + j;
      let c = category[idx];
      let x = input[idx];
      if is_normal(&c) && is_normal(&x) {
        items.push((c.into(), x.into(), idx));
      }
    }

    if items.is_empty() {
      return;
    }

    // Sort by category first, then by value
    items.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    // Process each category group
    let mut cat_start = 0;
    while cat_start < items.len() {
      // Find end of this category
      let mut cat_end = cat_start + 1;
      while cat_end < items.len() && items[cat_end].0.value == items[cat_start].0.value {
        cat_end += 1;
      }

      let cat_count = cat_end - cat_start;
      if cat_count == 1 {
        r[items[cat_start].2] = NumT::from(0.5).unwrap();
        cat_start = cat_end;
        continue;
      }

      // Items in this category are already sorted by value
      // Compute averaged rank percentage
      let total = NumT::from(cat_count).unwrap();
      let mut s = cat_start;
      while s < cat_end {
        let mut e = s + 1;
        while e < cat_end && items[e].1.value == items[s].1.value {
          e += 1;
        }
        // Ranks (1-based) for this tie group: (s-cat_start+1) to (e-cat_start)
        let rank_avg =
          NumT::from((s - cat_start + 1) + (e - cat_start)).unwrap() / NumT::from(2usize).unwrap();
        for k in s..e {
          r[items[k].2] = rank_avg / total;
        }
        s = e;
      }

      cat_start = cat_end;
    }
  });

  Ok(())
}

/// Calculate Z-Score within each category group at each time step
///
/// For each time position, groups items by `category` value, then computes
/// (x - group_mean) / group_std within each group.
/// NaN in category or input produces NaN output.
/// Groups with fewer than 2 valid values produce NaN.
pub fn ta_group_zscore<NumT: Float + Send + Sync + Debug>(
  ctx: &Context,
  r: &mut [NumT],
  category: &[NumT],
  input: &[NumT],
) -> Result<(), Error> {
  if r.len() != input.len() || r.len() != category.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }


  let group_size = ctx.chunk_size(r.len()) as usize;
  let groups = ctx.groups() as usize;

  if r.len() != group_size * groups {
    return Err(Error::LengthMismatch(r.len(), group_size * groups));
  }


  let r_ptr = UnsafePtr::new(r.as_mut_ptr(), r.len());
  (0..group_size).into_par_iter().for_each(|j| {
    let r = r_ptr.get();

    for i in 0..groups {
      r[i * group_size + j] = NumT::nan();
    }

    // Collect (category, value, index) for valid items
    let mut items: Vec<(OrderedFloat<NumT>, NumT, usize)> = Vec::with_capacity(groups);
    for i in 0..groups {
      let idx = i * group_size + j;
      let c = category[idx];
      let x = input[idx];
      if is_normal(&c) && is_normal(&x) {
        items.push((c.into(), x, idx));
      }
    }

    if items.is_empty() {
      return;
    }

    // Sort by category
    items.sort_by(|a, b| a.0.cmp(&b.0));

    // Process each category group
    let mut cat_start = 0;
    while cat_start < items.len() {
      let mut cat_end = cat_start + 1;
      while cat_end < items.len() && items[cat_end].0.value == items[cat_start].0.value {
        cat_end += 1;
      }

      let n = cat_end - cat_start;
      if n < 2 {
        // Not enough data for zscore
        cat_start = cat_end;
        continue;
      }

      // Compute mean and std
      let mut sum = NumT::zero();
      let mut sum_sq = NumT::zero();
      for k in cat_start..cat_end {
        let v = items[k].1;
        sum = sum + v;
        sum_sq = sum_sq + v * v;
      }

      let count = NumT::from(n).unwrap();
      let mean = sum / count;
      let var_num = sum_sq - (sum * sum / count);
      let var = var_num / (count - NumT::one());

      if var.abs() < NumT::epsilon() {
        // Zero variance, all same value -> zscore = 0
        for k in cat_start..cat_end {
          r[items[k].2] = NumT::zero();
        }
      } else {
        let std = var.sqrt();
        for k in cat_start..cat_end {
          r[items[k].2] = (items[k].1 - mean) / std;
        }
      }

      cat_start = cat_end;
    }
  });

  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::algo::assert_vec_eq_nan;

  #[test]
  fn test_group_rank_simple() {
    // groups=4, group_size=1
    // category: [1, 1, 2, 2] — two category groups
    // values:   [10, 30, 5, 15]
    // cat1: [10, 30] -> rank: 10=0.5/1=0.5, 30=1.0
    // Actually with 2 items: rank_avg for 10 = 1, pct = 1/2 = 0.5; rank_avg for 30 = 2, pct = 2/2 = 1.0
    // cat2: [5, 15]  -> rank: 5=0.5, 15=1.0
    let ctx = Context::new(0, 4, 0);
    let category = vec![1.0, 1.0, 2.0, 2.0];
    let input = vec![10.0, 30.0, 5.0, 15.0];
    let mut r = vec![0.0; 4];
    ta_group_rank(&ctx, &mut r, &category, &input).unwrap();
    assert_vec_eq_nan(&r, &vec![0.5, 1.0, 0.5, 1.0]);
  }

  #[test]
  fn test_group_rank_with_ties() {
    // groups=3, group_size=1
    // all same category, values [5, 5, 10]
    let ctx = Context::new(0, 3, 0);
    let category = vec![1.0, 1.0, 1.0];
    let input = vec![5.0, 5.0, 10.0];
    let mut r = vec![0.0; 3];
    ta_group_rank(&ctx, &mut r, &category, &input).unwrap();
    // Two 5s: avg rank = 1.5, pct = 1.5/3 = 0.5
    // One 10: rank = 3, pct = 3/3 = 1.0
    assert_vec_eq_nan(&r, &vec![0.5, 0.5, 1.0]);
  }

  #[test]
  fn test_group_rank_with_nan() {
    let ctx = Context::new(0, 3, 0);
    let category = vec![1.0, 1.0, f64::NAN];
    let input = vec![10.0, 20.0, 30.0];
    let mut r = vec![0.0; 3];
    ta_group_rank(&ctx, &mut r, &category, &input).unwrap();
    assert_vec_eq_nan(&r, &vec![0.5, 1.0, f64::NAN]);
  }

  #[test]
  fn test_group_zscore_simple() {
    // groups=4, group_size=1
    // cat1: [10, 30] -> mean=20, std=~14.14, z=[-0.707, 0.707]
    // cat2: [5, 15]  -> mean=10, std=~7.07, z=[-0.707, 0.707]
    let ctx = Context::new(0, 4, 0);
    let category = vec![1.0, 1.0, 2.0, 2.0];
    let input = vec![10.0, 30.0, 5.0, 15.0];
    let mut r = vec![0.0; 4];
    ta_group_zscore(&ctx, &mut r, &category, &input).unwrap();

    let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
    assert_vec_eq_nan(&r, &vec![-inv_sqrt2, inv_sqrt2, -inv_sqrt2, inv_sqrt2]);
  }

  #[test]
  fn test_group_zscore_single_in_group() {
    // groups=3, group_size=1
    // cat1 has 2 items, cat2 has 1 item -> cat2 gets NaN
    let ctx = Context::new(0, 3, 0);
    let category = vec![1.0, 1.0, 2.0];
    let input = vec![10.0, 20.0, 100.0];
    let mut r = vec![0.0; 3];
    ta_group_zscore(&ctx, &mut r, &category, &input).unwrap();

    let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
    assert_vec_eq_nan(&r, &vec![-inv_sqrt2, inv_sqrt2, f64::NAN]);
  }

  #[test]
  fn test_group_zscore_zero_variance() {
    let ctx = Context::new(0, 3, 0);
    let category = vec![1.0, 1.0, 1.0];
    let input = vec![5.0, 5.0, 5.0];
    let mut r = vec![0.0; 3];
    ta_group_zscore(&ctx, &mut r, &category, &input).unwrap();
    assert_vec_eq_nan(&r, &vec![0.0, 0.0, 0.0]);
  }

  #[test]
  fn test_group_rank_multi_time() {
    // groups=3, group_size=2 (2 time steps, 3 stocks)
    // Layout: [s1_t1, s1_t2, s2_t1, s2_t2, s3_t1, s3_t2]
    // category: [1, 1, 1, 1, 2, 2] — s1,s2 in cat1; s3 in cat2
    // values:   [10, 20, 30, 40, 50, 60]
    // t1 (j=0): cat1=[10(idx0), 30(idx2)] -> 10=0.5, 30=1.0; cat2=[50(idx4)] -> 0.5
    // t2 (j=1): cat1=[20(idx1), 40(idx3)] -> 20=0.5, 40=1.0; cat2=[60(idx5)] -> 0.5
    let ctx = Context::new(0, 3, 0);
    let category = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0];
    let input = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let mut r = vec![0.0; 6];
    ta_group_rank(&ctx, &mut r, &category, &input).unwrap();
    assert_vec_eq_nan(&r, &vec![0.5, 0.5, 1.0, 1.0, 0.5, 0.5]);
  }
}
