// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use alpha_algo::factor;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (groups, bars, x, y, *, min_obs = 2, lag = 0))]
pub fn cc_corr<'py>(
  py: Python<'py>,
  groups: usize,
  bars: usize,
  x: PyReadonlyArray1<'py, f64>,
  y: PyReadonlyArray1<'py, f64>,
  min_obs: usize,
  lag: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
  let xs = x.as_slice()?;
  let ys = y.as_slice()?;
  let out = py
    .detach(|| factor::cc_corr(groups, bars, xs, ys, min_obs, lag))
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
  Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (groups, bars, x, y, *, min_obs = 2, lag = 0))]
pub fn cc_spearman<'py>(
  py: Python<'py>,
  groups: usize,
  bars: usize,
  x: PyReadonlyArray1<'py, f64>,
  y: PyReadonlyArray1<'py, f64>,
  min_obs: usize,
  lag: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
  let xs = x.as_slice()?;
  let ys = y.as_slice()?;
  let out = py
    .detach(|| factor::cc_spearman(groups, bars, xs, ys, min_obs, lag))
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
  Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (groups, bars, x, y, *, min_obs = 2, lag = 0))]
pub fn cc_ic<'py>(
  py: Python<'py>,
  groups: usize,
  bars: usize,
  x: PyReadonlyArray1<'py, f64>,
  y: PyReadonlyArray1<'py, f64>,
  min_obs: usize,
  lag: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
  let xs = x.as_slice()?;
  let ys = y.as_slice()?;
  let (ic, ric) = py
    .detach(|| factor::cc_ic(groups, bars, xs, ys, min_obs, lag))
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
  Ok((ic.into_pyarray(py), ric.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(signature = (groups, bars, factors, *, min_obs = 2))]
pub fn cc_spearman_matrix<'py>(
  py: Python<'py>,
  groups: usize,
  bars: usize,
  factors: Vec<PyReadonlyArray1<'py, f64>>,
  min_obs: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
  let slices: Vec<&[f64]> = factors
    .iter()
    .map(|a| a.as_slice())
    .collect::<Result<Vec<_>, _>>()?;
  let out = py
    .detach(|| factor::cc_spearman_matrix(groups, bars, &slices, min_obs))
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
  Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (groups, bars, bins, step_first, step_cont, periods, n_buckets, *, entry_lag = 1, entry_mask = None))]
#[allow(clippy::too_many_arguments)]
pub fn tranche_returns<'py>(
  py: Python<'py>,
  groups: usize,
  bars: usize,
  bins: PyReadonlyArray1<'py, f64>,
  step_first: PyReadonlyArray1<'py, f64>,
  step_cont: PyReadonlyArray1<'py, f64>,
  periods: Vec<usize>,
  n_buckets: usize,
  entry_lag: usize,
  entry_mask: Option<PyReadonlyArray1<'py, bool>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
  let bs = bins.as_slice()?;
  let sf = step_first.as_slice()?;
  let sc = step_cont.as_slice()?;
  let em = entry_mask.as_ref().map(|a| a.as_slice()).transpose()?;
  let out = py
    .detach(|| {
      factor::tranche_returns(
        groups, bars, bs, em, sf, sc, &periods, n_buckets, entry_lag,
      )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
  Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (groups, bars, bins, values, n_buckets))]
pub fn bucket_means<'py>(
  py: Python<'py>,
  groups: usize,
  bars: usize,
  bins: PyReadonlyArray1<'py, f64>,
  values: PyReadonlyArray1<'py, f64>,
  n_buckets: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
  let bs = bins.as_slice()?;
  let vs = values.as_slice()?;
  let out = py
    .detach(|| factor::bucket_means(groups, bars, bs, vs, n_buckets))
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
  Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (groups, bars, bins, n_buckets, *, lag = 1, entry_mask = None))]
pub fn bucket_turnover<'py>(
  py: Python<'py>,
  groups: usize,
  bars: usize,
  bins: PyReadonlyArray1<'py, f64>,
  n_buckets: usize,
  lag: usize,
  entry_mask: Option<PyReadonlyArray1<'py, bool>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
  let bs = bins.as_slice()?;
  let em = entry_mask.as_ref().map(|a| a.as_slice()).transpose()?;
  let out = py
    .detach(|| factor::bucket_turnover(groups, bars, bs, em, n_buckets, lag))
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
  Ok(out.into_pyarray(py))
}

pub fn register_factor(m: &Bound<'_, PyModule>) -> PyResult<()> {
  m.add_function(wrap_pyfunction!(cc_corr, m)?)?;
  m.add_function(wrap_pyfunction!(cc_spearman, m)?)?;
  m.add_function(wrap_pyfunction!(cc_ic, m)?)?;
  m.add_function(wrap_pyfunction!(cc_spearman_matrix, m)?)?;
  m.add_function(wrap_pyfunction!(tranche_returns, m)?)?;
  m.add_function(wrap_pyfunction!(bucket_means, m)?)?;
  m.add_function(wrap_pyfunction!(bucket_turnover, m)?)?;
  Ok(())
}
