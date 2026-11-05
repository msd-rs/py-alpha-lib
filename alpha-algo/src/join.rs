// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum FillMethod {
  #[default]
  Zero,
  Nan,
  Backward,
  Forward,
}

impl From<i32> for FillMethod {
  fn from(value: i32) -> Self {
    match value {
      1 => Self::Nan,
      2 => Self::Backward,
      3 => Self::Forward,
      _ => Self::Zero,
    }
  }
}

impl From<FillMethod> for i32 {
  fn from(value: FillMethod) -> i32 {
    match value {
      FillMethod::Zero => 0,
      FillMethod::Nan => 1,
      FillMethod::Backward => 2,
      FillMethod::Forward => 3,
    }
  }
}

impl From<&str> for FillMethod {
  fn from(value: &str) -> Self {
    match value {
      "nan" => Self::Nan,
      "backward" => Self::Backward,
      "forward" => Self::Forward,
      _ => Self::Zero,
    }
  }
}

impl From<String> for FillMethod {
  fn from(value: String) -> Self {
    Self::from(value.as_str())
  }
}

impl From<&String> for FillMethod {
  fn from(value: &String) -> Self {
    Self::from(value.as_str())
  }
}

impl From<FillMethod> for &'static str {
  fn from(value: FillMethod) -> Self {
    match value {
      FillMethod::Zero => "zero",
      FillMethod::Nan => "nan",
      FillMethod::Backward => "backward",
      FillMethod::Forward => "forward",
    }
  }
}

/// Join `left` and `right` according to `method`, and return in `result` the indices of `right` that match `left`. Both `left` and `right` are already sorted in ascending order.
pub fn joined_index<T>(left: &[T], right: &[T], method: FillMethod, result: &mut Vec<usize>)
where
  T: PartialOrd + PartialEq,
{
  if result.len() != left.len() {
    result.resize(left.len(), usize::MAX);
  }

  if left.is_empty() {
    return;
  }

  if right.is_empty() {
    result.fill(usize::MAX);
    return;
  }

  match method {
    FillMethod::Backward => joined_index_backward(left, right, result),
    FillMethod::Forward => joined_index_forward(left, right, result),
    FillMethod::Zero | FillMethod::Nan => joined_index_exact(left, right, result),
  }
}

#[inline]
fn joined_index_backward<T: PartialOrd>(left: &[T], right: &[T], result: &mut [usize]) {
  let right_len = right.len();
  let mut curr: Option<usize> = None;

  for (x, out) in left.iter().zip(result.iter_mut()) {
    if x.partial_cmp(x).is_none() {
      *out = usize::MAX;
      continue;
    }

    let mut cand = match curr {
      Some(c) if right[c] <= *x => c,
      _ => {
        if right[0] > *x {
          *out = usize::MAX;
          continue;
        }
        0
      }
    };

    if cand + 1 < right_len && right[cand + 1] <= *x {
      cand += 1;
      if cand + 1 < right_len && right[cand + 1] <= *x {
        cand += 1;
        let mut step = 2usize;
        while cand + step < right_len && right[cand + step] <= *x {
          cand += step;
          step = step.saturating_mul(2);
        }
        let upper = (cand + step).min(right_len);
        let offset = right[cand + 1..upper].partition_point(|r| r <= x);
        cand += offset;
      }
    }

    *out = cand;
    curr = Some(cand);
  }
}

#[inline]
fn joined_index_exact<T: PartialOrd + PartialEq>(left: &[T], right: &[T], result: &mut [usize]) {
  let right_len = right.len();
  let mut curr: Option<usize> = None;

  for (x, out) in left.iter().zip(result.iter_mut()) {
    if x.partial_cmp(x).is_none() {
      *out = usize::MAX;
      continue;
    }

    let mut cand = match curr {
      Some(c) if right[c] <= *x => c,
      _ => {
        if right[0] > *x {
          *out = usize::MAX;
          continue;
        }
        0
      }
    };

    if cand + 1 < right_len && right[cand + 1] <= *x {
      cand += 1;
      if cand + 1 < right_len && right[cand + 1] <= *x {
        cand += 1;
        let mut step = 2usize;
        while cand + step < right_len && right[cand + step] <= *x {
          cand += step;
          step = step.saturating_mul(2);
        }
        let upper = (cand + step).min(right_len);
        let offset = right[cand + 1..upper].partition_point(|r| r <= x);
        cand += offset;
      }
    }

    curr = Some(cand);
    if right[cand] == *x {
      *out = cand;
    } else {
      *out = usize::MAX;
    }
  }
}

#[inline]
fn joined_index_forward<T: PartialOrd>(left: &[T], right: &[T], result: &mut [usize]) {
  let right_len = right.len();
  let mut curr = 0usize;

  for (x, out) in left.iter().zip(result.iter_mut()) {
    if x.partial_cmp(x).is_none() {
      *out = usize::MAX;
      continue;
    }

    if curr >= right_len {
      *out = usize::MAX;
      continue;
    }

    if right[curr] >= *x {
      *out = curr;
      continue;
    }

    if curr + 1 >= right_len {
      curr = right_len;
      *out = usize::MAX;
      continue;
    }
    if right[curr + 1] >= *x {
      curr += 1;
      *out = curr;
      continue;
    }

    if curr + 2 >= right_len {
      curr = right_len;
      *out = usize::MAX;
      continue;
    }
    if right[curr + 2] >= *x {
      curr += 2;
      *out = curr;
      continue;
    }

    let mut step = 4usize;
    curr += 2;
    while curr + step < right_len && right[curr + step] < *x {
      curr += step;
      step = step.saturating_mul(2);
    }

    let upper = (curr + step).min(right_len);
    let offset = right[curr + 1..upper].partition_point(|r| r < x);
    let found = curr + 1 + offset;

    if found < right_len && right[found] >= *x {
      curr = found;
      *out = found;
    } else {
      curr = right_len;
      *out = usize::MAX;
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_joined_index_empty() {
    let mut res = Vec::new();
    joined_index::<f64>(&[], &[1.0, 2.0], FillMethod::Backward, &mut res);
    assert_eq!(res, Vec::<usize>::new());

    joined_index::<f64>(&[1.0, 2.0], &[], FillMethod::Backward, &mut res);
    assert_eq!(res, vec![usize::MAX, usize::MAX]);
  }

  #[test]
  fn test_joined_index_exact() {
    let left = vec![1.0, 2.0, 2.5, 3.0, 4.0];
    let right = vec![1.0, 2.0, 3.0];
    let mut res = Vec::new();
    joined_index(&left, &right, FillMethod::Zero, &mut res);
    assert_eq!(res, vec![0, 1, usize::MAX, 2, usize::MAX]);

    joined_index(&left, &right, FillMethod::Nan, &mut res);
    assert_eq!(res, vec![0, 1, usize::MAX, 2, usize::MAX]);
  }

  #[test]
  fn test_joined_index_backward() {
    let left = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
    let right = vec![1.0, 2.0, 3.0];
    let mut res = Vec::new();
    joined_index(&left, &right, FillMethod::Backward, &mut res);
    assert_eq!(res, vec![usize::MAX, 0, 0, 1, 1, 2, 2]);
  }

  #[test]
  fn test_joined_index_forward() {
    let left = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
    let right = vec![1.0, 2.0, 3.0];
    let mut res = Vec::new();
    joined_index(&left, &right, FillMethod::Forward, &mut res);
    assert_eq!(res, vec![0, 0, 1, 1, 2, 2, usize::MAX]);
  }

  #[test]
  fn test_joined_index_duplicates() {
    let left = vec![2.0];
    let right = vec![1.0, 2.0, 2.0, 2.0, 3.0];
    let mut res = Vec::new();

    // Backward chooses the last element <= 2.0 -> index 3
    joined_index(&left, &right, FillMethod::Backward, &mut res);
    assert_eq!(res, vec![3]);

    // Forward chooses the first element >= 2.0 -> index 1
    joined_index(&left, &right, FillMethod::Forward, &mut res);
    assert_eq!(res, vec![1]);

    // Exact matches the last element == 2.0 -> index 3
    joined_index(&left, &right, FillMethod::Zero, &mut res);
    assert_eq!(res, vec![3]);
  }

  #[test]
  fn test_joined_index_nan() {
    let left = vec![1.0, f64::NAN, 2.0];
    let right = vec![1.0, 2.0];
    let mut res = Vec::new();

    joined_index(&left, &right, FillMethod::Backward, &mut res);
    assert_eq!(res, vec![0, usize::MAX, 1]);

    joined_index(&left, &right, FillMethod::Forward, &mut res);
    assert_eq!(res, vec![0, usize::MAX, 1]);
  }

  #[test]
  fn test_joined_index_stress_equivalence() {
    // Stress test comparing against naive binary search
    let left: Vec<i64> = (0..5000).map(|x| x * 3).collect();
    let right: Vec<i64> = (0..10000).map(|x| x * 2).collect();

    let mut res_backward = Vec::new();
    joined_index(&left, &right, FillMethod::Backward, &mut res_backward);

    for (i, &x) in left.iter().enumerate() {
      let idx = right.partition_point(|&r| r <= x);
      let expected = if idx > 0 { idx - 1 } else { usize::MAX };
      assert_eq!(res_backward[i], expected, "mismatch at left[{}]={}", i, x);
    }

    let mut res_forward = Vec::new();
    joined_index(&left, &right, FillMethod::Forward, &mut res_forward);

    for (i, &x) in left.iter().enumerate() {
      let idx = right.partition_point(|&r| r < x);
      let expected = if idx < right.len() { idx } else { usize::MAX };
      assert_eq!(res_forward[i], expected, "mismatch at left[{}]={}", i, x);
    }

    let mut res_exact = Vec::new();
    joined_index(&left, &right, FillMethod::Zero, &mut res_exact);

    for (i, &x) in left.iter().enumerate() {
      let idx = right.partition_point(|&r| r <= x);
      let expected = if idx > 0 && right[idx - 1] == x {
        idx - 1
      } else {
        usize::MAX
      };
      assert_eq!(res_exact[i], expected, "mismatch at left[{}]={}", i, x);
    }
  }

  #[test]
  fn test_joined_index_large_random_comparison() {
    use std::time::Instant;

    // Pseudo-random generator without external deps (LCG)
    let mut rng = 123456789u64;
    let mut rand = || {
      rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
      (rng >> 33) as i64
    };

    let n = 50_000;
    let m = 100_000;

    let mut left = Vec::with_capacity(n);
    let mut cur = 0i64;
    for _ in 0..n {
      cur += (rand().abs() % 10) + 1;
      left.push(cur);
    }

    let mut right = Vec::with_capacity(m);
    cur = 0;
    for _ in 0..m {
      cur += (rand().abs() % 5) + 1;
      right.push(cur);
    }

    let t0 = Instant::now();
    let mut res_backward = Vec::new();
    joined_index(&left, &right, FillMethod::Backward, &mut res_backward);
    let elapsed = t0.elapsed();
    println!("joined_index backward 50k x 100k elapsed: {:?}", elapsed);

    for (i, &x) in left.iter().enumerate() {
      let idx = right.partition_point(|&r| r <= x);
      let expected = if idx > 0 { idx - 1 } else { usize::MAX };
      assert_eq!(res_backward[i], expected);
    }

    let mut res_forward = Vec::new();
    joined_index(&left, &right, FillMethod::Forward, &mut res_forward);
    for (i, &x) in left.iter().enumerate() {
      let idx = right.partition_point(|&r| r < x);
      let expected = if idx < right.len() { idx } else { usize::MAX };
      assert_eq!(res_forward[i], expected);
    }
  }
}
