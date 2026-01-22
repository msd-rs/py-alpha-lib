use std::{cell::UnsafeCell, cmp::Ordering, collections::BTreeMap, sync::Arc};

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error};

#[derive(Debug, Copy, Clone, Default, PartialEq, PartialOrd)]
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

/// rank by ts dim
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
}

impl<NumT: Float> UnsafePtr<NumT> {
  pub fn new(ptr: *mut NumT) -> Self {
    UnsafePtr { ptr }
  }

  pub fn get(&self) -> &mut [NumT] {
    let slice = unsafe { std::slice::from_raw_parts_mut(self.ptr, 0) };
    slice
  }
}

unsafe impl<NumT: Float> Send for UnsafePtr<NumT> {}
unsafe impl<NumT: Float> Sync for UnsafePtr<NumT> {}

/// rank by group dim
pub fn ta_rank<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }

  if ctx.groups() < 2 {
    return ta_ts_rank(ctx, r, input, 0);
  }

  let group_size = ctx.chunk_size(r.len()) as usize;
  let groups = ctx.groups() as usize;
  let r = UnsafePtr::new(r.as_mut_ptr());
  (0..group_size).into_par_iter().for_each(|j| {
    let mut rank_window: Vec<(OrderedFloat<NumT>, usize)> = Vec::new();
    for i in 0..groups {
      let idx = j * groups + i;
      rank_window.push((input[idx].into(), idx));
    }
    rank_window.sort_by(|a, b| a.0.cmp(&b.0));
    let r = r.get();
    rank_window.iter().enumerate().for_each(|(rank, (_v, i))| {
      r[*i] = NumT::from(rank + 1).unwrap();
    });
  });

  Ok(())
}
