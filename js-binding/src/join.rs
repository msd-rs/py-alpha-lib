pub fn join_asof_impl(
  d1: &[f64],
  d2: &[f64],
  v2: &[f64],
  method: i32,
) -> Result<Box<[f64]>, String> {
  if d2.len() != v2.len() {
    return Err("d2 and v2 must have the same length".to_string());
  }

  let mut result = vec![0.0; d1.len()];
  if d1.is_empty() {
    return Ok(result.into_boxed_slice());
  }

  if d2.is_empty() {
    match method {
      0 => {}
      1 | 2 | -2 => {
        result.fill(f64::NAN);
      }
      _ => {
        return Err(format!("invalid method: {}", method));
      }
    }
    return Ok(result.into_boxed_slice());
  }

  let is_sorted = d2.windows(2).all(|w| w[0] <= w[1]);

  match method {
    0 => {
      for (i, &t) in d1.iter().enumerate() {
        let val = if is_sorted {
          let idx = d2.partition_point(|&x| x <= t);
          if idx > 0 && d2[idx - 1] == t {
            v2[idx - 1]
          } else {
            0.0
          }
        } else {
          d2.iter()
            .position(|&x| x == t)
            .map(|idx| v2[idx])
            .unwrap_or(0.0)
        };
        result[i] = val;
      }
    }
    1 => {
      for (i, &t) in d1.iter().enumerate() {
        let val = if is_sorted {
          let idx = d2.partition_point(|&x| x <= t);
          if idx > 0 && d2[idx - 1] == t {
            v2[idx - 1]
          } else {
            f64::NAN
          }
        } else {
          d2.iter()
            .position(|&x| x == t)
            .map(|idx| v2[idx])
            .unwrap_or(f64::NAN)
        };
        result[i] = val;
      }
    }
    2 => {
      for (i, &t) in d1.iter().enumerate() {
        let val = if is_sorted {
          let idx = d2.partition_point(|&x| x <= t);
          if idx > 0 { v2[idx - 1] } else { f64::NAN }
        } else {
          d2.iter()
            .enumerate()
            .filter(|&(_, &x)| x <= t)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| v2[idx])
            .unwrap_or(f64::NAN)
        };
        result[i] = val;
      }
    }
    -2 => {
      for (i, &t) in d1.iter().enumerate() {
        let val = if is_sorted {
          let idx = d2.partition_point(|&x| x < t);
          if idx < d2.len() { v2[idx] } else { f64::NAN }
        } else {
          d2.iter()
            .enumerate()
            .filter(|&(_, &x)| x >= t)
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| v2[idx])
            .unwrap_or(f64::NAN)
        };
        result[i] = val;
      }
    }
    _ => {
      return Err(format!("invalid method: {}", method));
    }
  }

  Ok(result.into_boxed_slice())
}
