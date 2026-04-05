// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal};

/// Calculate rolling Shannon entropy over a moving window
///
/// Discretizes values into `bins` equal-width buckets within the window's
/// [min, max] range, then computes -sum(p * ln(p)) where p is the frequency
/// of each occupied bin. Uses natural log (base e).
/// Requires at least 2 valid values. Single-value windows return 0.
pub fn ta_entropy<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
  bins: usize,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }


  let bins = if bins == 0 { 10 } else { bins };

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      r.fill(NumT::nan());

      // We need to collect window values to compute entropy,
      // since we need min/max for binning.
      let mut window: Vec<NumT> = Vec::with_capacity(periods);

      for i in start..end {
        let val = x[i];

        // Add new value
        if is_normal(&val) {
          window.push(val);
        }

        // Remove old values that fall out of window
        // We track by time index, so the window contains values from
        // x[max(0, i-periods+1)..=i] that are valid
        let win_start = if i >= periods { i - periods + 1 } else { 0 };

        // Rebuild window for simplicity (correct but not single-pass optimal)
        window.clear();
        for k in win_start..=i {
          if is_normal(&x[k]) {
            window.push(x[k]);
          }
        }

        if !is_normal(&val) {
          continue;
        }

        if ctx.is_strictly_cycle() && i < periods - 1 {
          continue;
        }

        let n = window.len();
        if n < 2 {
          r[i] = NumT::zero();
          continue;
        }

        // Find min and max
        let mut min_val = window[0];
        let mut max_val = window[0];
        for &v in &window[1..] {
          if v < min_val {
            min_val = v;
          }
          if v > max_val {
            max_val = v;
          }
        }

        if max_val == min_val {
          // All same value -> entropy = 0
          r[i] = NumT::zero();
          continue;
        }

        // Bin the values
        let range = max_val - min_val;
        let bin_width = range / NumT::from(bins).unwrap();
        let mut counts = vec![0usize; bins];

        for &v in &window {
          let mut bin_idx = ((v - min_val) / bin_width).to_usize().unwrap_or(bins - 1);
          if bin_idx >= bins {
            bin_idx = bins - 1;
          }
          counts[bin_idx] += 1;
        }

        // Compute entropy: -sum(p * ln(p))
        let total = NumT::from(n).unwrap();
        let mut entropy = NumT::zero();
        for &c in &counts {
          if c > 0 {
            let p = NumT::from(c).unwrap() / total;
            entropy = entropy - p * p.ln();
          }
        }

        r[i] = entropy;
      }
    });

  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::algo::assert_vec_eq_nan;

  #[test]
  fn test_entropy_uniform() {
    // 4 distinct values spread across bins -> high entropy
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let periods = 4;
    let bins = 4;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_entropy(&ctx, &mut r, &input, periods, bins).unwrap();

    // 4 values in 4 bins, each p=0.25, entropy = -4*(0.25*ln(0.25)) = ln(4) = 1.3862...
    let expected = 4.0f64.ln();
    assert!(
      (r[3] - expected).abs() < 1e-5,
      "got {}, expected {}",
      r[3],
      expected
    );
  }

  #[test]
  fn test_entropy_constant() {
    let input = vec![5.0, 5.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_entropy(&ctx, &mut r, &input, periods, 10).unwrap();

    // All same -> entropy = 0
    assert_vec_eq_nan(&r, &vec![0.0, 0.0, 0.0]);
  }

  #[test]
  fn test_entropy_two_groups() {
    // [1, 1, 2, 2] in 2 bins -> each bin has 2/4 = 0.5, entropy = ln(2)
    let input = vec![1.0, 1.0, 2.0, 2.0];
    let periods = 4;
    let bins = 2;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_entropy(&ctx, &mut r, &input, periods, bins).unwrap();

    let expected = 2.0f64.ln();
    assert!(
      (r[3] - expected).abs() < 1e-5,
      "got {}, expected {}",
      r[3],
      expected
    );
  }
}
