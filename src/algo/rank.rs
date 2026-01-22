use std::{cmp::Ordering, collections::BTreeMap};

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

/// rank by group dim
pub fn ta_rank(ctx: &Context, r: &mut [f64], input: &[f64], periods: usize) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }

  if ctx.groups() < 2 {
    return ta_ts_rank(ctx, r, input, periods);
  }

  Ok(())
}
