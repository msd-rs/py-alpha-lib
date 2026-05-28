// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

mod alpha;
mod backfill;
mod beta;
mod context;
mod cross;
mod drawdown;
mod ema;
mod entropy;
mod error;
mod extremum;
mod group;
mod ma;
mod misc;
mod moments;
mod neutralize;
mod quantile;
mod rank;
mod returns;
mod scan;
mod series;
mod sharpe;
mod skip_nan_window;
mod slope;
mod stats;
mod stddev;
mod sum;
mod zscore;

pub use alpha::*;
pub use backfill::*;
pub use beta::*;
pub use context::Context;
pub use cross::*;
pub use drawdown::*;
pub use ema::*;
pub use entropy::*;
pub use error::Error;
pub use extremum::*;
pub use group::*;
pub use ma::*;
pub use misc::*;
pub use moments::*;
pub use neutralize::*;
pub use quantile::*;
pub use rank::*;
pub use returns::*;
pub use scan::*;
pub use series::*;
pub use sharpe::*;
pub use slope::*;
pub use stats::*;
pub use stddev::*;
pub use sum::*;
pub use zscore::*;

pub use num_traits::Float;

#[inline]
pub fn is_normal<T: Float>(a: &T) -> bool {
  !a.is_nan()
}

#[cfg(test)]
use std::fmt::{Debug, Display};
#[cfg(test)]
pub fn assert_vec_eq_nan<T: Float + Display + Debug>(a: &[T], b: &[T]) {
  assert!(
    a.len() == b.len(),
    "length mismatch, a.len(): {}, b.len(): {}",
    a.len(),
    b.len()
  );
  for i in 0..a.len() {
    assert!(
      (a[i] - b[i]).abs() < T::from(0.000001).unwrap() || (!is_normal(&a[i]) && !is_normal(&b[i])),
      "index {} mismatch, a: {}, b: {}\nvec a: {:?}\nvec b: {:?}",
      i,
      a[i],
      b[i],
      a,
      b
    );
  }
}
