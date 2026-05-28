// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use thiserror::Error;

#[cfg(feature = "py-binding")]
use pyo3::{PyErr, exceptions::PyValueError};

#[derive(Error, Debug)]
pub enum Error {
  #[error("result length ({0}) should be the same as as input length ({1})")]
  LengthMismatch(usize, usize),
  #[error("invalid parameter: {0}")]
  InvalidParameter(String),
  #[error("invalid period: {0}")]
  InvalidPeriod(String),
}

#[cfg(feature = "py-binding")]
impl From<Error> for PyErr {
  fn from(err: Error) -> Self {
    PyValueError::new_err(err.to_string())
  }
}
