use std::fmt::Display;

use pyo3::FromPyObject;

/// Context information for the how to calculate the result
#[derive(Debug, Default, Clone, Copy, FromPyObject)]
pub struct Context {
  /// only calculate from this index, 0 means from the first, -1 means from the last
  #[pyo3(attribute("start"))]
  pub _start: i32,
  /// number of groups, indicates the group count of the input data,
  /// the input data is chunked by group, and each group is calculated independently
  /// 0 means no group, which is the same as 1
  #[pyo3(attribute("groups"))]
  pub _groups: u32,
  /// flags, indicates the calculation flags
  #[pyo3(attribute("flags"))]
  pub _flags: u64,
}

/// skip nan when compute
pub const FLAG_SKIP_NAN: u64 = 1;
/// strictly follow cycle required, fill nan for not enough data
pub const FLAG_STRICTLY_CYCLE: u64 = 2;

impl Context {
  #[allow(dead_code)]
  pub fn new(start: i32, groups: u32, flags: u64) -> Self {
    Self {
      _start: start,
      _groups: groups,
      _flags: flags,
    }
  }

  /// get the start index to calculate by a given total length
  pub fn start(&self, total: usize) -> usize {
    if total == 0 {
      return 0;
    }
    if self._start >= 0 {
      (total - 1).min(self._start as usize)
    } else {
      0.max(total as i32 + self._start) as usize
    }
  }

  pub fn groups(&self) -> usize {
    if self._groups == 0 {
      1
    } else {
      self._groups as usize
    }
  }

  pub fn chunk_size(&self, total: usize) -> usize {
    total / self.groups()
  }

  /// should skip nan when compute
  pub fn is_skip_nan(&self) -> bool {
    self._flags & FLAG_SKIP_NAN != 0
  }

  /// strictly follow cycle required, fill nan for not enough data
  pub fn is_strictly_cycle(&self) -> bool {
    self._flags & FLAG_STRICTLY_CYCLE != 0
  }
}

impl From<(i32, u32, u64)> for Context {
  fn from((start, groups, flags): (i32, u32, u64)) -> Self {
    Self {
      _start: start,
      _groups: groups,
      _flags: flags,
    }
  }
}

impl Display for Context {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "Context {{ start: {}, groups: {}, flags: {} }}",
      self._start, self._groups, self._flags
    )
  }
}
