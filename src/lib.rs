mod algo;
use std::sync::{LazyLock, RwLock};

use pyo3::prelude::*;

use crate::algo::Context;

static _ALGO_CTX_: LazyLock<RwLock<Context>> = LazyLock::new(|| RwLock::new(Context::default()));

mod algo_impl {
  use log::debug;
  use numpy::{PyReadonlyArray1, PyReadwriteArray1};
  use pyo3::{exceptions::PyValueError, prelude::*, types::PyList};
  use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

  use crate::_ALGO_CTX_;

  use super::algo::*;

  fn ctx<'py>(_py: Python<'py>) -> Context {
    let ctx = match _ALGO_CTX_.try_read() {
      Ok(ctx) => ctx.clone(),
      Err(_e) => Context::default(),
    };
    debug!("current {}", ctx);
    ctx
  }

  #[pyfunction]
  #[pyo3(signature = (/, start=None, groups=None, flags=None))]
  pub fn set_ctx<'py>(
    _py: Python<'py>,
    start: Option<i32>,
    groups: Option<u32>,
    flags: Option<u64>,
  ) -> PyResult<()> {
    match _ALGO_CTX_.try_write() {
      Ok(mut ctx) => {
        start.map(|s| ctx._start = s);
        groups.map(|g| ctx._groups = g);
        flags.map(|f| ctx._flags = f);
        Ok(())
      }
      Err(_e) => Err(PyValueError::new_err("failed to set context")),
    }
  }

  /// Exponential Moving Average (variant of EMA)
  ///
  /// alpha = 2 / (n + 1)
  ///
  /// https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
  ///
  #[pyfunction]
  pub fn ema<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    let mut ctx = ctx(py);

    // 2. check input type and do dispatch
    if let Some((mut r, input)) = r
      .extract::<PyReadwriteArray1<'py, f64>>()
      .ok()
      .zip(input.extract::<PyReadonlyArray1<'py, f64>>().ok())
    {
      // input is f64 array
      let mut r = r.as_array_mut();
      let r = r
        .as_slice_mut()
        .ok_or(PyValueError::new_err("failed to get mutable slice"))?;

      let input = input.as_array();
      let input = input
        .as_slice()
        .ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_ema(&ctx, r, input, periods).map_err(|e| e.into())
    } else if let Some((mut r, input)) = r
      .extract::<PyReadwriteArray1<'py, f32>>()
      .ok()
      .zip(input.extract::<PyReadonlyArray1<'py, f32>>().ok())
    {
      // input is f32 array
      let mut r = r.as_array_mut();
      let r = r
        .as_slice_mut()
        .ok_or(PyValueError::new_err("invalid input"))?;

      let input = input.as_array();
      let input = input
        .as_slice()
        .ok_or(PyValueError::new_err("invalid input"))?;
      ta_ema(&ctx, r, input, periods).map_err(|e| e.into())
    } else if let Some((r, input)) = r.cast::<PyList>().ok().zip(input.cast::<PyList>().ok()) {
      // input is list of arrays

      // each array is a group, ensure groups is set to 1
      ctx._groups = 1;

      if r.len() != input.len() {
        return Err(PyValueError::new_err("length mismatch"));
      }

      // check if each array is f64 array
      if let Some((mut r, input)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f64>>>()
        .ok()
        .zip(input.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok())
      {
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let input = input.iter().map(|x| x.as_array()).collect::<Vec<_>>();

        let mut _r = vec![];
        r.into_par_iter()
          .zip(input.into_par_iter())
          .map(|(mut out, input)| {
            let out = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((out, input)) = out.zip(input) {
              ta_ema(&ctx, out, input, periods).map_err(|e| e.into())
            } else {
              Err(PyValueError::new_err("invalid input"))
            }
          })
          .collect_into_vec(&mut _r);

        match _r.into_iter().find(|x| x.is_err()) {
          Some(e) => e,
          None => Ok(()),
        }
      // check if each array is f32 array
      } else if let Some((mut r, input)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f32>>>()
        .ok()
        .zip(input.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok())
      {
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let input = input.iter().map(|x| x.as_array()).collect::<Vec<_>>();

        let mut _r = vec![];
        r.into_par_iter()
          .zip(input.into_par_iter())
          .map(|(mut out, input)| {
            let out = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((out, input)) = out.zip(input) {
              ta_ema(&ctx, out, input, periods).map_err(|e| e.into())
            } else {
              Err(PyValueError::new_err("invalid input"))
            }
          })
          .collect_into_vec(&mut _r);

        match _r.into_iter().find(|x| x.is_err()) {
          Some(e) => e,
          None => Ok(()),
        }
      } else {
        // NumT array can only be f64 or f32
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      // NumT array can only be f64 or f32
      Err(PyValueError::new_err("invalid input"))
    }
  }

  include!(concat!(env!("OUT_DIR"), "/algo_bindings.rs"));
}

#[pymodule]
fn _algo(m: &Bound<'_, PyModule>) -> PyResult<()> {
  use algo_impl::*;
  pyo3_log::init();

  m.add_function(wrap_pyfunction!(ema, m)?)?;
  m.add_function(wrap_pyfunction!(set_ctx, m)?)?;
  algo_impl::register_functions(m)?;
  Ok(())
}
