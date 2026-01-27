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
