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

pub fn ta_neutralize<NumT: Float + Send + Sync + Debug>(
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

  if groups < 2 {
    for i in 0..r.len() {
      if is_normal(&input[i]) && is_normal(&category[i]) {
        r[i] = NumT::from(0.5).unwrap();
      } else {
        r[i] = NumT::nan();
      }
    }
    return Ok(());
  }

  let r_ptr = UnsafePtr::new(r.as_mut_ptr(), r.len());
  (0..group_size).into_par_iter().for_each(|j| {
    let r = r_ptr.get();

    for i in 0..groups {
      r[i * group_size + j] = NumT::nan();
    }

    let mut items: Vec<(OrderedFloat<NumT>, usize, NumT)> = Vec::with_capacity(groups);
    for i in 0..groups {
      let idx = i * group_size + j;
      let c = category[idx];
      let x = input[idx];
      if is_normal(&c) && is_normal(&x) {
        items.push((c.into(), idx, x));
      }
    }

    if items.is_empty() {
      return;
    }

    items.sort_by(|a, b| a.0.cmp(&b.0));

    let mut rank_items: Vec<(OrderedFloat<NumT>, usize)> = Vec::with_capacity(items.len());
    let mut s = 0usize;
    while s < items.len() {
      let mut e = s + 1;
      while e < items.len() && items[e].0.value == items[s].0.value {
        e += 1;
      }

      let mut sum = NumT::zero();
      for k in s..e {
        sum = sum + items[k].2;
      }
      let mean = sum / NumT::from(e - s).unwrap();

      for k in s..e {
        let resid = items[k].2 - mean;
        rank_items.push((resid.into(), items[k].1));
      }

      s = e;
    }

    if rank_items.is_empty() {
      return;
    }

    rank_items.sort_by(|a, b| a.0.cmp(&b.0));
    if rank_items.len() == 1 {
      r[rank_items[0].1] = NumT::from(0.5).unwrap();
      return;
    }

    let total_minus_one = NumT::from(rank_items.len() - 1).unwrap();
    let mut prev_rank_value = rank_items[0].0.value;
    let mut s = 0usize;

    for e in 0..rank_items.len() {
      if prev_rank_value == rank_items[e].0.value {
        continue;
      }
      let rank_avg_1based = NumT::from(e + s + 1).unwrap() / NumT::from(2usize).unwrap();
      let pct = (rank_avg_1based - NumT::one()) / total_minus_one;
      for i in s..e {
        r[rank_items[i].1] = pct;
      }
      s = e;
      prev_rank_value = rank_items[e].0.value;
    }

    let rank_avg_1based =
      NumT::from(rank_items.len() + s + 1).unwrap() / NumT::from(2usize).unwrap();
    let pct = (rank_avg_1based - NumT::one()) / total_minus_one;
    for i in s..rank_items.len() {
      r[rank_items[i].1] = pct;
    }
  });

  Ok(())
}

#[cfg(test)]
mod tests {
  use crate::algo::assert_vec_eq_nan;

  use super::*;

  #[test]
  fn test_neutralize_simple() {
    let ctx = Context::new(0, 4, 0);
    let category = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
    let input = vec![10.0, 20.0, 30.0, 12.0, 18.0, 33.0, 5.0, 7.0, 9.0, 6.0, 8.0, 10.0];

    let mut r = vec![0.0; input.len()];
    ta_neutralize(&ctx, &mut r, &category, &input).unwrap();

    assert_vec_eq_nan(
      &r,
      &vec![
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.3333333333333333,
        0.3333333333333333,
        0.3333333333333333,
        0.6666666666666666,
        0.6666666666666666,
        0.6666666666666666,
      ],
    );
  }

  #[test]
  fn test_neutralize_with_nan() {
    let ctx = Context::new(0, 3, 0);
    let category = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0];
    let input = vec![1.0, f64::NAN, 3.0, 5.0, 10.0, 20.0];

    let mut r = vec![0.0; input.len()];
    ta_neutralize(&ctx, &mut r, &category, &input).unwrap();

    assert_vec_eq_nan(&r, &vec![0.0, f64::NAN, 1.0, 0.5, 0.5, 0.5]);
  }
}
