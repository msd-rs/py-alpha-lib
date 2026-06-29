  /// Rolling Jensen's Alpha of asset returns against benchmark returns.
  #[pyfunction]
  fn alpha<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    benchmark: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some(((mut r, input), benchmark)) = r
      .extract::<PyReadwriteArray1<'py, f64>>() .ok()
      .zip(input.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(benchmark.extract::<PyReadonlyArray1<'py, f64>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let input = input.as_array();
      let input = input.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let benchmark = benchmark.as_array();
      let benchmark = benchmark.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_alpha(&ctx, r, input, benchmark, periods).map_err(|e| e.into())
    } else if let Some(((mut r, input), benchmark)) = r
      .extract::<PyReadwriteArray1<'py, f32>>() .ok()
      .zip(input.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(benchmark.extract::<PyReadonlyArray1<'py, f32>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let input = input.as_array();
      let input = input.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let benchmark = benchmark.as_array();
      let benchmark = benchmark.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_alpha(&ctx, r, input, benchmark, periods).map_err(|e| e.into())
    } else if let Some(((r, input), benchmark)) = r.cast::<PyList>().ok().zip(input.cast::<PyList>().ok()).zip(benchmark.cast::<PyList>().ok()) {
      if r.len() != input.len() || benchmark.len() != input.len() { return Err(PyValueError::new_err("length mismatch")); }
      if let Some(((mut r, input), benchmark)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f64>>>().ok()
        .zip(input.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok())
        .zip(benchmark.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok()) {
        // ... list iter logic ...
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let input = input.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let benchmark = benchmark.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter().zip(input.into_par_iter()).zip(benchmark.into_par_iter())
          .map(|((mut out, a), b)| {
            let r = out.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
            let input = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            let benchmark = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            ta_alpha(&ctx, r, input, benchmark, periods).map_err(|e| e.into())
          }).collect_into_vec(&mut _r);
        match _r.into_iter().find(|x| x.is_err()) { Some(e) => e, None => Ok(()) }
      } else if let Some(((mut r, input), benchmark)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f32>>>().ok()
        .zip(input.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok())
        .zip(benchmark.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok()) {
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let input = input.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let benchmark = benchmark.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter().zip(input.into_par_iter()).zip(benchmark.into_par_iter())
          .map(|((mut out, a), b)| {
            let r = out.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
            let input = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            let benchmark = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            ta_alpha(&ctx, r, input, benchmark, periods).map_err(|e| e.into())
          }).collect_into_vec(&mut _r);
        match _r.into_iter().find(|x| x.is_err()) { Some(e) => e, None => Ok(()) }
      } else { Err(PyValueError::new_err("invalid input list")) }
    } else { Err(PyValueError::new_err("invalid input")) }
  }
  /// Forward-fill NaN values with the last valid observation
  #[pyfunction]
  fn backfill<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_backfill(&ctx, r, input).map_err(|e| e.into())
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
      ta_backfill(&ctx, r, input).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_backfill(&ctx, r, input).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_backfill(&ctx, r, input).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Calculate number of bars since last condition true
  #[pyfunction]
  fn barslast<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some((mut r, input)) = r
      .extract::<PyReadwriteArray1<'py, f64>>()
      .ok()
      .zip(input.extract::<PyReadonlyArray1<'py, bool>>().ok())
    {
      let mut r = r.as_array_mut();
      let r = r
        .as_slice_mut()
        .ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let input = input.as_array();
      let input = input
        .as_slice()
        .ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_barslast(&ctx, r, input).map_err(|e| e.into())
    } else if let Some((mut r, input)) = r
      .extract::<PyReadwriteArray1<'py, f32>>()
      .ok()
      .zip(input.extract::<PyReadonlyArray1<'py, bool>>().ok())
    {
      let mut r = r.as_array_mut();
      let r = r
        .as_slice_mut()
        .ok_or(PyValueError::new_err("invalid input"))?;
      let input = input.as_array();
      let input = input
        .as_slice()
        .ok_or(PyValueError::new_err("invalid input"))?;
      ta_barslast(&ctx, r, input).map_err(|e| e.into())
    } else if let Some((r, input)) = r.cast::<PyList>().ok().zip(input.cast::<PyList>().ok()) {
      // input is list of arrays
      // each array is a group, ensure groups is set to 1
      ctx._groups = 1;
      if r.len() != input.len() {
        return Err(PyValueError::new_err("length mismatch"));
      }
      // check if each array is f64 array output and bool input
      if let Some((mut r, input)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f64>>>()
        .ok()
        .zip(input.extract::<Vec<PyReadonlyArray1<'py, bool>>>().ok())
      {
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let input = input.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter()
          .zip(input.into_par_iter())
          .map(|(mut out, input)| {
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_barslast(&ctx, r, input).map_err(|e| e.into())
            } else {
              Err(PyValueError::new_err("invalid input"))
            }
          })
          .collect_into_vec(&mut _r);
        match _r.into_iter().find(|x| x.is_err()) {
          Some(e) => e,
          None => Ok(()),
        }
      // check if each array is f32 array output
      } else if let Some((mut r, input)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f32>>>()
        .ok()
        .zip(input.extract::<Vec<PyReadonlyArray1<'py, bool>>>().ok())
      {
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let input = input.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter()
          .zip(input.into_par_iter())
          .map(|(mut out, input)| {
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_barslast(&ctx, r, input).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Calculate number of bars since first condition true
  #[pyfunction]
  fn barssince<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some((mut r, input)) = r
      .extract::<PyReadwriteArray1<'py, f64>>()
      .ok()
      .zip(input.extract::<PyReadonlyArray1<'py, bool>>().ok())
    {
      let mut r = r.as_array_mut();
      let r = r
        .as_slice_mut()
        .ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let input = input.as_array();
      let input = input
        .as_slice()
        .ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_barssince(&ctx, r, input).map_err(|e| e.into())
    } else if let Some((mut r, input)) = r
      .extract::<PyReadwriteArray1<'py, f32>>()
      .ok()
      .zip(input.extract::<PyReadonlyArray1<'py, bool>>().ok())
    {
      let mut r = r.as_array_mut();
      let r = r
        .as_slice_mut()
        .ok_or(PyValueError::new_err("invalid input"))?;
      let input = input.as_array();
      let input = input
        .as_slice()
        .ok_or(PyValueError::new_err("invalid input"))?;
      ta_barssince(&ctx, r, input).map_err(|e| e.into())
    } else if let Some((r, input)) = r.cast::<PyList>().ok().zip(input.cast::<PyList>().ok()) {
      // input is list of arrays
      // each array is a group, ensure groups is set to 1
      ctx._groups = 1;
      if r.len() != input.len() {
        return Err(PyValueError::new_err("length mismatch"));
      }
      // check if each array is f64 array output and bool input
      if let Some((mut r, input)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f64>>>()
        .ok()
        .zip(input.extract::<Vec<PyReadonlyArray1<'py, bool>>>().ok())
      {
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let input = input.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter()
          .zip(input.into_par_iter())
          .map(|(mut out, input)| {
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_barssince(&ctx, r, input).map_err(|e| e.into())
            } else {
              Err(PyValueError::new_err("invalid input"))
            }
          })
          .collect_into_vec(&mut _r);
        match _r.into_iter().find(|x| x.is_err()) {
          Some(e) => e,
          None => Ok(()),
        }
      // check if each array is f32 array output
      } else if let Some((mut r, input)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f32>>>()
        .ok()
        .zip(input.extract::<Vec<PyReadonlyArray1<'py, bool>>>().ok())
      {
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let input = input.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter()
          .zip(input.into_par_iter())
          .map(|(mut out, input)| {
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_barssince(&ctx, r, input).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Rolling Beta coefficient of asset returns against benchmark returns.
  #[pyfunction]
  fn beta<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    benchmark: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some(((mut r, input), benchmark)) = r
      .extract::<PyReadwriteArray1<'py, f64>>() .ok()
      .zip(input.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(benchmark.extract::<PyReadonlyArray1<'py, f64>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let input = input.as_array();
      let input = input.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let benchmark = benchmark.as_array();
      let benchmark = benchmark.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_beta(&ctx, r, input, benchmark, periods).map_err(|e| e.into())
    } else if let Some(((mut r, input), benchmark)) = r
      .extract::<PyReadwriteArray1<'py, f32>>() .ok()
      .zip(input.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(benchmark.extract::<PyReadonlyArray1<'py, f32>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let input = input.as_array();
      let input = input.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let benchmark = benchmark.as_array();
      let benchmark = benchmark.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_beta(&ctx, r, input, benchmark, periods).map_err(|e| e.into())
    } else if let Some(((r, input), benchmark)) = r.cast::<PyList>().ok().zip(input.cast::<PyList>().ok()).zip(benchmark.cast::<PyList>().ok()) {
      if r.len() != input.len() || benchmark.len() != input.len() { return Err(PyValueError::new_err("length mismatch")); }
      if let Some(((mut r, input), benchmark)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f64>>>().ok()
        .zip(input.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok())
        .zip(benchmark.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok()) {
        // ... list iter logic ...
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let input = input.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let benchmark = benchmark.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter().zip(input.into_par_iter()).zip(benchmark.into_par_iter())
          .map(|((mut out, a), b)| {
            let r = out.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
            let input = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            let benchmark = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            ta_beta(&ctx, r, input, benchmark, periods).map_err(|e| e.into())
          }).collect_into_vec(&mut _r);
        match _r.into_iter().find(|x| x.is_err()) { Some(e) => e, None => Ok(()) }
      } else if let Some(((mut r, input), benchmark)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f32>>>().ok()
        .zip(input.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok())
        .zip(benchmark.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok()) {
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let input = input.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let benchmark = benchmark.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter().zip(input.into_par_iter()).zip(benchmark.into_par_iter())
          .map(|((mut out, a), b)| {
            let r = out.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
            let input = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            let benchmark = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            ta_beta(&ctx, r, input, benchmark, periods).map_err(|e| e.into())
          }).collect_into_vec(&mut _r);
        match _r.into_iter().find(|x| x.is_err()) { Some(e) => e, None => Ok(()) }
      } else { Err(PyValueError::new_err("invalid input list")) }
    } else { Err(PyValueError::new_err("invalid input")) }
  }
  /// Discretize the input into n bins, the ctx.groups() is the number of groups
  #[pyfunction]
  fn bins<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    bins: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_bins(&ctx, r, input, bins).map_err(|e| e.into())
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
      ta_bins(&ctx, r, input, bins).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_bins(&ctx, r, input, bins).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_bins(&ctx, r, input, bins).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Backward split and dividend adjustment
  #[pyfunction]
  fn bw_split<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    price: &'py Bound<'_, PyAny>,
    dividend: &'py Bound<'_, PyAny>,
    transfer_shares: &'py Bound<'_, PyAny>,
    right_shares: &'py Bound<'_, PyAny>,
    right_price: &'py Bound<'_, PyAny>,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some((((((mut r, price), dividend), transfer_shares), right_shares), right_price)) = r
      .extract::<PyReadwriteArray1<'py, f64>>().ok()
      .zip(price.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(dividend.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(transfer_shares.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(right_shares.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(right_price.extract::<PyReadonlyArray1<'py, f64>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let price = price.as_array();
      let price = price.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let dividend = dividend.as_array();
      let dividend = dividend.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let transfer_shares = transfer_shares.as_array();
      let transfer_shares = transfer_shares.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let right_shares = right_shares.as_array();
      let right_shares = right_shares.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let right_price = right_price.as_array();
      let right_price = right_price.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_bw_split(&ctx, r, price, dividend, transfer_shares, right_shares, right_price).map_err(|e| e.into())
    } else if let Some((((((mut r, price), dividend), transfer_shares), right_shares), right_price)) = r
      .extract::<PyReadwriteArray1<'py, f32>>().ok()
      .zip(price.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(dividend.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(transfer_shares.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(right_shares.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(right_price.extract::<PyReadonlyArray1<'py, f32>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let price = price.as_array();
      let price = price.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let dividend = dividend.as_array();
      let dividend = dividend.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let transfer_shares = transfer_shares.as_array();
      let transfer_shares = transfer_shares.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let right_shares = right_shares.as_array();
      let right_shares = right_shares.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let right_price = right_price.as_array();
      let right_price = right_price.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_bw_split(&ctx, r, price, dividend, transfer_shares, right_shares, right_price).map_err(|e| e.into())
    } else { Err(PyValueError::new_err("invalid input")) }
  }
  /// Calculate rank percentage cross group dimension, the ctx.groups() is the number of groups
  #[pyfunction]
  fn cc_rank<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_cc_rank(&ctx, r, input).map_err(|e| e.into())
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
      ta_cc_rank(&ctx, r, input).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_cc_rank(&ctx, r, input).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_cc_rank(&ctx, r, input).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Calculate cross-sectional Z-Score across groups at each time step
  #[pyfunction]
  fn cc_zscore<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_cc_zscore(&ctx, r, input).map_err(|e| e.into())
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
      ta_cc_zscore(&ctx, r, input).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_cc_zscore(&ctx, r, input).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_cc_zscore(&ctx, r, input).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Time Series Correlation in moving window on self
  #[pyfunction]
  fn corr<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_corr(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_corr(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_corr(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_corr(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Calculate two series correlation over a moving window
  #[pyfunction]
  fn corr2<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    x: &'py Bound<'_, PyAny>,
    y: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some(((mut r, x), y)) = r
      .extract::<PyReadwriteArray1<'py, f64>>() .ok()
      .zip(x.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(y.extract::<PyReadonlyArray1<'py, f64>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let x = x.as_array();
      let x = x.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let y = y.as_array();
      let y = y.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_corr2(&ctx, r, x, y, periods).map_err(|e| e.into())
    } else if let Some(((mut r, x), y)) = r
      .extract::<PyReadwriteArray1<'py, f32>>() .ok()
      .zip(x.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(y.extract::<PyReadonlyArray1<'py, f32>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let x = x.as_array();
      let x = x.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let y = y.as_array();
      let y = y.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_corr2(&ctx, r, x, y, periods).map_err(|e| e.into())
    } else if let Some(((r, x), y)) = r.cast::<PyList>().ok().zip(x.cast::<PyList>().ok()).zip(y.cast::<PyList>().ok()) {
      if r.len() != x.len() || y.len() != x.len() { return Err(PyValueError::new_err("length mismatch")); }
      if let Some(((mut r, x), y)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f64>>>().ok()
        .zip(x.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok())
        .zip(y.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok()) {
        // ... list iter logic ...
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let x = x.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let y = y.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter().zip(x.into_par_iter()).zip(y.into_par_iter())
          .map(|((mut out, a), b)| {
            let r = out.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
            let x = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            let y = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            ta_corr2(&ctx, r, x, y, periods).map_err(|e| e.into())
          }).collect_into_vec(&mut _r);
        match _r.into_iter().find(|x| x.is_err()) { Some(e) => e, None => Ok(()) }
      } else if let Some(((mut r, x), y)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f32>>>().ok()
        .zip(x.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok())
        .zip(y.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok()) {
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let x = x.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let y = y.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter().zip(x.into_par_iter()).zip(y.into_par_iter())
          .map(|((mut out, a), b)| {
            let r = out.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
            let x = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            let y = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            ta_corr2(&ctx, r, x, y, periods).map_err(|e| e.into())
          }).collect_into_vec(&mut _r);
        match _r.into_iter().find(|x| x.is_err()) { Some(e) => e, None => Ok(()) }
      } else { Err(PyValueError::new_err("invalid input list")) }
    } else { Err(PyValueError::new_err("invalid input")) }
  }
  /// Calculate number of periods where condition is true in passed `periods` window
  #[pyfunction]
  fn count<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some((mut r, input)) = r
      .extract::<PyReadwriteArray1<'py, f64>>()
      .ok()
      .zip(input.extract::<PyReadonlyArray1<'py, bool>>().ok())
    {
      let mut r = r.as_array_mut();
      let r = r
        .as_slice_mut()
        .ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let input = input.as_array();
      let input = input
        .as_slice()
        .ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_count(&ctx, r, input, periods).map_err(|e| e.into())
    } else if let Some((mut r, input)) = r
      .extract::<PyReadwriteArray1<'py, f32>>()
      .ok()
      .zip(input.extract::<PyReadonlyArray1<'py, bool>>().ok())
    {
      let mut r = r.as_array_mut();
      let r = r
        .as_slice_mut()
        .ok_or(PyValueError::new_err("invalid input"))?;
      let input = input.as_array();
      let input = input
        .as_slice()
        .ok_or(PyValueError::new_err("invalid input"))?;
      ta_count(&ctx, r, input, periods).map_err(|e| e.into())
    } else if let Some((r, input)) = r.cast::<PyList>().ok().zip(input.cast::<PyList>().ok()) {
      // input is list of arrays
      // each array is a group, ensure groups is set to 1
      ctx._groups = 1;
      if r.len() != input.len() {
        return Err(PyValueError::new_err("length mismatch"));
      }
      // check if each array is f64 array output and bool input
      if let Some((mut r, input)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f64>>>()
        .ok()
        .zip(input.extract::<Vec<PyReadonlyArray1<'py, bool>>>().ok())
      {
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let input = input.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter()
          .zip(input.into_par_iter())
          .map(|(mut out, input)| {
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_count(&ctx, r, input, periods).map_err(|e| e.into())
            } else {
              Err(PyValueError::new_err("invalid input"))
            }
          })
          .collect_into_vec(&mut _r);
        match _r.into_iter().find(|x| x.is_err()) {
          Some(e) => e,
          None => Ok(()),
        }
      // check if each array is f32 array output
      } else if let Some((mut r, input)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f32>>>()
        .ok()
        .zip(input.extract::<Vec<PyReadonlyArray1<'py, bool>>>().ok())
      {
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let input = input.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter()
          .zip(input.into_par_iter())
          .map(|(mut out, input)| {
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_count(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Count number of NaN values in a rolling window
  #[pyfunction]
  fn count_nans<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_count_nans(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_count_nans(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_count_nans(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_count_nans(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Calculate Covariance over a moving window
  #[pyfunction]
  fn cov<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    x: &'py Bound<'_, PyAny>,
    y: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some(((mut r, x), y)) = r
      .extract::<PyReadwriteArray1<'py, f64>>() .ok()
      .zip(x.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(y.extract::<PyReadonlyArray1<'py, f64>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let x = x.as_array();
      let x = x.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let y = y.as_array();
      let y = y.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_cov(&ctx, r, x, y, periods).map_err(|e| e.into())
    } else if let Some(((mut r, x), y)) = r
      .extract::<PyReadwriteArray1<'py, f32>>() .ok()
      .zip(x.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(y.extract::<PyReadonlyArray1<'py, f32>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let x = x.as_array();
      let x = x.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let y = y.as_array();
      let y = y.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_cov(&ctx, r, x, y, periods).map_err(|e| e.into())
    } else if let Some(((r, x), y)) = r.cast::<PyList>().ok().zip(x.cast::<PyList>().ok()).zip(y.cast::<PyList>().ok()) {
      if r.len() != x.len() || y.len() != x.len() { return Err(PyValueError::new_err("length mismatch")); }
      if let Some(((mut r, x), y)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f64>>>().ok()
        .zip(x.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok())
        .zip(y.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok()) {
        // ... list iter logic ...
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let x = x.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let y = y.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter().zip(x.into_par_iter()).zip(y.into_par_iter())
          .map(|((mut out, a), b)| {
            let r = out.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
            let x = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            let y = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            ta_cov(&ctx, r, x, y, periods).map_err(|e| e.into())
          }).collect_into_vec(&mut _r);
        match _r.into_iter().find(|x| x.is_err()) { Some(e) => e, None => Ok(()) }
      } else if let Some(((mut r, x), y)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f32>>>().ok()
        .zip(x.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok())
        .zip(y.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok()) {
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let x = x.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let y = y.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter().zip(x.into_par_iter()).zip(y.into_par_iter())
          .map(|((mut out, a), b)| {
            let r = out.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
            let x = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            let y = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            ta_cov(&ctx, r, x, y, periods).map_err(|e| e.into())
          }).collect_into_vec(&mut _r);
        match _r.into_iter().find(|x| x.is_err()) { Some(e) => e, None => Ok(()) }
      } else { Err(PyValueError::new_err("invalid input list")) }
    } else { Err(PyValueError::new_err("invalid input")) }
  }
  /// For 2 arrays A and B, return true if A[i-1] < B[i-1] and A[i] >= B[i]
  #[pyfunction]
  fn cross<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    a: &'py Bound<'_, PyAny>,
    b: &'py Bound<'_, PyAny>,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some(((mut r, a), b)) = r
      .extract::<PyReadwriteArray1<'py, bool>>() .ok()
      .zip(a.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(b.extract::<PyReadonlyArray1<'py, f64>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let a = a.as_array();
      let a = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let b = b.as_array();
      let b = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_cross(&ctx, r, a, b).map_err(|e| e.into())
    } else { Err(PyValueError::new_err("invalid input (expected bool, float, float)")) }
  }
  /// Exponential Moving Average
  #[pyfunction]
  fn dma<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    weight: f64,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_dma(&ctx, r, input, weight).map_err(|e| e.into())
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
      ta_dma(&ctx, r, input, weight as f32).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_dma(&ctx, r, input, weight).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_dma(&ctx, r, input, weight as f32).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Calculate rolling Shannon entropy over a moving window
  #[pyfunction]
  fn entropy<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
    bins: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_entropy(&ctx, r, input, periods, bins).map_err(|e| e.into())
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
      ta_entropy(&ctx, r, input, periods, bins).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_entropy(&ctx, r, input, periods, bins).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_entropy(&ctx, r, input, periods, bins).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Future Return
  #[pyfunction]
  fn fret<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    open: &'py Bound<'_, PyAny>,
    close: &'py Bound<'_, PyAny>,
    is_calc: &'py Bound<'_, PyAny>,
    delay: usize,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some((((mut r, open), close), is_calc)) = r
      .extract::<PyReadwriteArray1<'py, f64>>().ok()
      .zip(open.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(close.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(is_calc.extract::<PyReadonlyArray1<'py, f64>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let open = open.as_array();
      let open = open.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let close = close.as_array();
      let close = close.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let is_calc = is_calc.as_array();
      let is_calc = is_calc.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_fret(&ctx, r, open, close, is_calc, delay, periods).map_err(|e| e.into())
    } else if let Some((((mut r, open), close), is_calc)) = r
      .extract::<PyReadwriteArray1<'py, f32>>().ok()
      .zip(open.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(close.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(is_calc.extract::<PyReadonlyArray1<'py, f32>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let open = open.as_array();
      let open = open.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let close = close.as_array();
      let close = close.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let is_calc = is_calc.as_array();
      let is_calc = is_calc.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_fret(&ctx, r, open, close, is_calc, delay, periods).map_err(|e| e.into())
    } else { Err(PyValueError::new_err("invalid input")) }
  }
  /// Forward split and dividend adjustment
  #[pyfunction]
  fn fw_split<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    price: &'py Bound<'_, PyAny>,
    dividend: &'py Bound<'_, PyAny>,
    transfer_shares: &'py Bound<'_, PyAny>,
    right_shares: &'py Bound<'_, PyAny>,
    right_price: &'py Bound<'_, PyAny>,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some((((((mut r, price), dividend), transfer_shares), right_shares), right_price)) = r
      .extract::<PyReadwriteArray1<'py, f64>>().ok()
      .zip(price.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(dividend.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(transfer_shares.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(right_shares.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(right_price.extract::<PyReadonlyArray1<'py, f64>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let price = price.as_array();
      let price = price.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let dividend = dividend.as_array();
      let dividend = dividend.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let transfer_shares = transfer_shares.as_array();
      let transfer_shares = transfer_shares.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let right_shares = right_shares.as_array();
      let right_shares = right_shares.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let right_price = right_price.as_array();
      let right_price = right_price.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_fw_split(&ctx, r, price, dividend, transfer_shares, right_shares, right_price).map_err(|e| e.into())
    } else if let Some((((((mut r, price), dividend), transfer_shares), right_shares), right_price)) = r
      .extract::<PyReadwriteArray1<'py, f32>>().ok()
      .zip(price.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(dividend.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(transfer_shares.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(right_shares.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(right_price.extract::<PyReadonlyArray1<'py, f32>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let price = price.as_array();
      let price = price.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let dividend = dividend.as_array();
      let dividend = dividend.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let transfer_shares = transfer_shares.as_array();
      let transfer_shares = transfer_shares.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let right_shares = right_shares.as_array();
      let right_shares = right_shares.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let right_price = right_price.as_array();
      let right_price = right_price.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_fw_split(&ctx, r, price, dividend, transfer_shares, right_shares, right_price).map_err(|e| e.into())
    } else { Err(PyValueError::new_err("invalid input")) }
  }
  /// Calculate rank percentage within each category group at each time step
  #[pyfunction]
  fn group_rank<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    category: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some(((mut r, category), input)) = r
      .extract::<PyReadwriteArray1<'py, f64>>() .ok()
      .zip(category.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(input.extract::<PyReadonlyArray1<'py, f64>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let category = category.as_array();
      let category = category.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let input = input.as_array();
      let input = input.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_group_rank(&ctx, r, category, input).map_err(|e| e.into())
    } else if let Some(((mut r, category), input)) = r
      .extract::<PyReadwriteArray1<'py, f32>>() .ok()
      .zip(category.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(input.extract::<PyReadonlyArray1<'py, f32>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let category = category.as_array();
      let category = category.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let input = input.as_array();
      let input = input.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_group_rank(&ctx, r, category, input).map_err(|e| e.into())
    } else if let Some(((r, category), input)) = r.cast::<PyList>().ok().zip(category.cast::<PyList>().ok()).zip(input.cast::<PyList>().ok()) {
      if r.len() != category.len() || input.len() != category.len() { return Err(PyValueError::new_err("length mismatch")); }
      if let Some(((mut r, category), input)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f64>>>().ok()
        .zip(category.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok())
        .zip(input.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok()) {
        // ... list iter logic ...
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let category = category.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let input = input.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter().zip(category.into_par_iter()).zip(input.into_par_iter())
          .map(|((mut out, a), b)| {
            let r = out.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
            let category = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            let input = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            ta_group_rank(&ctx, r, category, input).map_err(|e| e.into())
          }).collect_into_vec(&mut _r);
        match _r.into_iter().find(|x| x.is_err()) { Some(e) => e, None => Ok(()) }
      } else if let Some(((mut r, category), input)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f32>>>().ok()
        .zip(category.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok())
        .zip(input.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok()) {
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let category = category.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let input = input.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter().zip(category.into_par_iter()).zip(input.into_par_iter())
          .map(|((mut out, a), b)| {
            let r = out.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
            let category = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            let input = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            ta_group_rank(&ctx, r, category, input).map_err(|e| e.into())
          }).collect_into_vec(&mut _r);
        match _r.into_iter().find(|x| x.is_err()) { Some(e) => e, None => Ok(()) }
      } else { Err(PyValueError::new_err("invalid input list")) }
    } else { Err(PyValueError::new_err("invalid input")) }
  }
  /// Calculate Z-Score within each category group at each time step
  #[pyfunction]
  fn group_zscore<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    category: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some(((mut r, category), input)) = r
      .extract::<PyReadwriteArray1<'py, f64>>() .ok()
      .zip(category.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(input.extract::<PyReadonlyArray1<'py, f64>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let category = category.as_array();
      let category = category.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let input = input.as_array();
      let input = input.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_group_zscore(&ctx, r, category, input).map_err(|e| e.into())
    } else if let Some(((mut r, category), input)) = r
      .extract::<PyReadwriteArray1<'py, f32>>() .ok()
      .zip(category.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(input.extract::<PyReadonlyArray1<'py, f32>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let category = category.as_array();
      let category = category.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let input = input.as_array();
      let input = input.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_group_zscore(&ctx, r, category, input).map_err(|e| e.into())
    } else if let Some(((r, category), input)) = r.cast::<PyList>().ok().zip(category.cast::<PyList>().ok()).zip(input.cast::<PyList>().ok()) {
      if r.len() != category.len() || input.len() != category.len() { return Err(PyValueError::new_err("length mismatch")); }
      if let Some(((mut r, category), input)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f64>>>().ok()
        .zip(category.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok())
        .zip(input.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok()) {
        // ... list iter logic ...
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let category = category.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let input = input.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter().zip(category.into_par_iter()).zip(input.into_par_iter())
          .map(|((mut out, a), b)| {
            let r = out.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
            let category = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            let input = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            ta_group_zscore(&ctx, r, category, input).map_err(|e| e.into())
          }).collect_into_vec(&mut _r);
        match _r.into_iter().find(|x| x.is_err()) { Some(e) => e, None => Ok(()) }
      } else if let Some(((mut r, category), input)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f32>>>().ok()
        .zip(category.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok())
        .zip(input.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok()) {
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let category = category.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let input = input.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter().zip(category.into_par_iter()).zip(input.into_par_iter())
          .map(|((mut out, a), b)| {
            let r = out.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
            let category = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            let input = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            ta_group_zscore(&ctx, r, category, input).map_err(|e| e.into())
          }).collect_into_vec(&mut _r);
        match _r.into_iter().find(|x| x.is_err()) { Some(e) => e, None => Ok(()) }
      } else { Err(PyValueError::new_err("invalid input list")) }
    } else { Err(PyValueError::new_err("invalid input")) }
  }
  /// Find highest value in a preceding `periods` window
  #[pyfunction]
  fn hhv<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_hhv(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_hhv(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_hhv(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_hhv(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// The number of periods that have passed since the array reached its `periods` period high
  #[pyfunction]
  fn hhvbars<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_hhvbars(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_hhvbars(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_hhvbars(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_hhvbars(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Linear Regression Intercept
  #[pyfunction]
  fn intercept<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_intercept(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_intercept(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_intercept(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_intercept(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Calculate rolling sample excess Kurtosis over a moving window
  #[pyfunction]
  fn kurtosis<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_kurtosis(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_kurtosis(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_kurtosis(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_kurtosis(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Find lowest value in a preceding `periods` window
  #[pyfunction]
  fn llv<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_llv(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_llv(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_llv(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_llv(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// The number of periods that have passed since the array reached its periods period low
  #[pyfunction]
  fn llvbars<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_llvbars(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_llvbars(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_llvbars(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_llvbars(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// For 2 arrays A and B, return true if previous N periods A < B, Current A >= B
  #[pyfunction]
  fn longcross<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    a: &'py Bound<'_, PyAny>,
    b: &'py Bound<'_, PyAny>,
    n: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some(((mut r, a), b)) = r
      .extract::<PyReadwriteArray1<'py, bool>>() .ok()
      .zip(a.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(b.extract::<PyReadonlyArray1<'py, f64>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let a = a.as_array();
      let a = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let b = b.as_array();
      let b = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_longcross(&ctx, r, a, b, n).map_err(|e| e.into())
    } else { Err(PyValueError::new_err("invalid input (expected bool, float, float)")) }
  }
  /// Linear Weighted Moving Average
  #[pyfunction]
  fn lwma<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_lwma(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_lwma(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_lwma(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_lwma(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Simple Moving Average, also known as arithmetic moving average
  #[pyfunction]
  fn ma<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_ma(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_ma(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_ma(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_ma(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Rolling Maximum Drawdown.
  #[pyfunction]
  fn max_drawdown<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_max_drawdown(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_max_drawdown(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_max_drawdown(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_max_drawdown(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Calculate rolling min-max difference (range) over a moving window
  #[pyfunction]
  fn min_max_diff<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_min_max_diff(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_min_max_diff(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_min_max_diff(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_min_max_diff(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Calculate rolling k-th central moment over a moving window
  #[pyfunction]
  fn moment<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
    k: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_moment(&ctx, r, input, periods, k).map_err(|e| e.into())
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
      ta_moment(&ctx, r, input, periods, k).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_moment(&ctx, r, input, periods, k).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_moment(&ctx, r, input, periods, k).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Neutralize the effect of a categorical variable on a numeric variable
  #[pyfunction]
  fn neutralize<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    category: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some(((mut r, category), input)) = r
      .extract::<PyReadwriteArray1<'py, f64>>() .ok()
      .zip(category.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(input.extract::<PyReadonlyArray1<'py, f64>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let category = category.as_array();
      let category = category.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let input = input.as_array();
      let input = input.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_neutralize(&ctx, r, category, input).map_err(|e| e.into())
    } else if let Some(((mut r, category), input)) = r
      .extract::<PyReadwriteArray1<'py, f32>>() .ok()
      .zip(category.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(input.extract::<PyReadonlyArray1<'py, f32>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let category = category.as_array();
      let category = category.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let input = input.as_array();
      let input = input.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_neutralize(&ctx, r, category, input).map_err(|e| e.into())
    } else if let Some(((r, category), input)) = r.cast::<PyList>().ok().zip(category.cast::<PyList>().ok()).zip(input.cast::<PyList>().ok()) {
      if r.len() != category.len() || input.len() != category.len() { return Err(PyValueError::new_err("length mismatch")); }
      if let Some(((mut r, category), input)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f64>>>().ok()
        .zip(category.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok())
        .zip(input.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok()) {
        // ... list iter logic ...
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let category = category.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let input = input.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter().zip(category.into_par_iter()).zip(input.into_par_iter())
          .map(|((mut out, a), b)| {
            let r = out.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
            let category = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            let input = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            ta_neutralize(&ctx, r, category, input).map_err(|e| e.into())
          }).collect_into_vec(&mut _r);
        match _r.into_iter().find(|x| x.is_err()) { Some(e) => e, None => Ok(()) }
      } else if let Some(((mut r, category), input)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f32>>>().ok()
        .zip(category.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok())
        .zip(input.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok()) {
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let category = category.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let input = input.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter().zip(category.into_par_iter()).zip(input.into_par_iter())
          .map(|((mut out, a), b)| {
            let r = out.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
            let category = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            let input = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            ta_neutralize(&ctx, r, category, input).map_err(|e| e.into())
          }).collect_into_vec(&mut _r);
        match _r.into_iter().find(|x| x.is_err()) { Some(e) => e, None => Ok(()) }
      } else { Err(PyValueError::new_err("invalid input list")) }
    } else { Err(PyValueError::new_err("invalid input")) }
  }
  /// Calculate product of values in preceding `periods` window
  #[pyfunction]
  fn product<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_product(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_product(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_product(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_product(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Calculate rolling quantile over a moving window
  #[pyfunction]
  fn quantile<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
    q: f64,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_quantile(&ctx, r, input, periods, q).map_err(|e| e.into())
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
      ta_quantile(&ctx, r, input, periods, q as f32).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_quantile(&ctx, r, input, periods, q).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_quantile(&ctx, r, input, periods, q as f32).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Calculate rank in a sliding window with size `periods`
  #[pyfunction]
  fn rank<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_rank(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_rank(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_rank(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_rank(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// For 2 arrays A and B, return true if A[i-1] > B[i-1] and A[i] <= B[i]
  #[pyfunction]
  fn rcross<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    a: &'py Bound<'_, PyAny>,
    b: &'py Bound<'_, PyAny>,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some(((mut r, a), b)) = r
      .extract::<PyReadwriteArray1<'py, bool>>() .ok()
      .zip(a.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(b.extract::<PyReadonlyArray1<'py, f64>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let a = a.as_array();
      let a = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let b = b.as_array();
      let b = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_rcross(&ctx, r, a, b).map_err(|e| e.into())
    } else { Err(PyValueError::new_err("invalid input (expected bool, float, float)")) }
  }
  /// Right shift input array by `periods`, r[i] = input[i - periods]
  #[pyfunction]
  fn r#ref<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_ref(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_ref(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_ref(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_ref(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Calculate Regression Coefficient (Beta) of Y on X over a moving window
  #[pyfunction]
  fn regbeta<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    y: &'py Bound<'_, PyAny>,
    x: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some(((mut r, y), x)) = r
      .extract::<PyReadwriteArray1<'py, f64>>() .ok()
      .zip(y.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(x.extract::<PyReadonlyArray1<'py, f64>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let y = y.as_array();
      let y = y.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let x = x.as_array();
      let x = x.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_regbeta(&ctx, r, y, x, periods).map_err(|e| e.into())
    } else if let Some(((mut r, y), x)) = r
      .extract::<PyReadwriteArray1<'py, f32>>() .ok()
      .zip(y.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(x.extract::<PyReadonlyArray1<'py, f32>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let y = y.as_array();
      let y = y.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let x = x.as_array();
      let x = x.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_regbeta(&ctx, r, y, x, periods).map_err(|e| e.into())
    } else if let Some(((r, y), x)) = r.cast::<PyList>().ok().zip(y.cast::<PyList>().ok()).zip(x.cast::<PyList>().ok()) {
      if r.len() != y.len() || x.len() != y.len() { return Err(PyValueError::new_err("length mismatch")); }
      if let Some(((mut r, y), x)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f64>>>().ok()
        .zip(y.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok())
        .zip(x.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok()) {
        // ... list iter logic ...
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let y = y.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let x = x.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter().zip(y.into_par_iter()).zip(x.into_par_iter())
          .map(|((mut out, a), b)| {
            let r = out.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
            let y = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            let x = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            ta_regbeta(&ctx, r, y, x, periods).map_err(|e| e.into())
          }).collect_into_vec(&mut _r);
        match _r.into_iter().find(|x| x.is_err()) { Some(e) => e, None => Ok(()) }
      } else if let Some(((mut r, y), x)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f32>>>().ok()
        .zip(y.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok())
        .zip(x.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok()) {
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let y = y.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let x = x.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter().zip(y.into_par_iter()).zip(x.into_par_iter())
          .map(|((mut out, a), b)| {
            let r = out.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
            let y = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            let x = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            ta_regbeta(&ctx, r, y, x, periods).map_err(|e| e.into())
          }).collect_into_vec(&mut _r);
        match _r.into_iter().find(|x| x.is_err()) { Some(e) => e, None => Ok(()) }
      } else { Err(PyValueError::new_err("invalid input list")) }
    } else { Err(PyValueError::new_err("invalid input")) }
  }
  /// Calculate Regression Residual of Y on X over a moving window
  #[pyfunction]
  fn regresi<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    y: &'py Bound<'_, PyAny>,
    x: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some(((mut r, y), x)) = r
      .extract::<PyReadwriteArray1<'py, f64>>() .ok()
      .zip(y.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(x.extract::<PyReadonlyArray1<'py, f64>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let y = y.as_array();
      let y = y.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let x = x.as_array();
      let x = x.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_regresi(&ctx, r, y, x, periods).map_err(|e| e.into())
    } else if let Some(((mut r, y), x)) = r
      .extract::<PyReadwriteArray1<'py, f32>>() .ok()
      .zip(y.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(x.extract::<PyReadonlyArray1<'py, f32>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let y = y.as_array();
      let y = y.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let x = x.as_array();
      let x = x.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_regresi(&ctx, r, y, x, periods).map_err(|e| e.into())
    } else if let Some(((r, y), x)) = r.cast::<PyList>().ok().zip(y.cast::<PyList>().ok()).zip(x.cast::<PyList>().ok()) {
      if r.len() != y.len() || x.len() != y.len() { return Err(PyValueError::new_err("length mismatch")); }
      if let Some(((mut r, y), x)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f64>>>().ok()
        .zip(y.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok())
        .zip(x.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok()) {
        // ... list iter logic ...
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let y = y.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let x = x.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter().zip(y.into_par_iter()).zip(x.into_par_iter())
          .map(|((mut out, a), b)| {
            let r = out.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
            let y = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            let x = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            ta_regresi(&ctx, r, y, x, periods).map_err(|e| e.into())
          }).collect_into_vec(&mut _r);
        match _r.into_iter().find(|x| x.is_err()) { Some(e) => e, None => Ok(()) }
      } else if let Some(((mut r, y), x)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f32>>>().ok()
        .zip(y.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok())
        .zip(x.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok()) {
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let y = y.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let x = x.iter().map(|x| x.as_array()).collect::<Vec<_>>();
        let mut _r = vec![];
        r.into_par_iter().zip(y.into_par_iter()).zip(x.into_par_iter())
          .map(|((mut out, a), b)| {
            let r = out.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
            let y = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            let x = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
            ta_regresi(&ctx, r, y, x, periods).map_err(|e| e.into())
          }).collect_into_vec(&mut _r);
        match _r.into_iter().find(|x| x.is_err()) { Some(e) => e, None => Ok(()) }
      } else { Err(PyValueError::new_err("invalid input list")) }
    } else { Err(PyValueError::new_err("invalid input")) }
  }
  /// For 2 arrays A and B, return true if previous N periods A > B, Current A <= B
  #[pyfunction]
  fn rlongcross<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    a: &'py Bound<'_, PyAny>,
    b: &'py Bound<'_, PyAny>,
    n: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some(((mut r, a), b)) = r
      .extract::<PyReadwriteArray1<'py, bool>>() .ok()
      .zip(a.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(b.extract::<PyReadonlyArray1<'py, f64>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let a = a.as_array();
      let a = a.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let b = b.as_array();
      let b = b.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_rlongcross(&ctx, r, a, b, n).map_err(|e| e.into())
    } else { Err(PyValueError::new_err("invalid input (expected bool, float, float)")) }
  }
  /// Conditional cumulative add: r[t] = r[t-1] + (cond[t] ? input[t] : 0)
  #[pyfunction]
  fn scan_add<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    condition: &'py Bound<'_, PyAny>,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some(((mut r, input), condition)) = r
      .extract::<PyReadwriteArray1<'py, f64>>() .ok()
      .zip(input.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(condition.extract::<PyReadonlyArray1<'py, bool>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let input = input.as_array();
      let input = input.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let condition = condition.as_array();
      let condition = condition.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_scan_add(&ctx, r, input, condition).map_err(|e| e.into())
    } else if let Some(((mut r, input), condition)) = r
      .extract::<PyReadwriteArray1<'py, f32>>() .ok()
      .zip(input.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(condition.extract::<PyReadonlyArray1<'py, bool>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let input = input.as_array();
      let input = input.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let condition = condition.as_array();
      let condition = condition.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_scan_add(&ctx, r, input, condition).map_err(|e| e.into())
    } else { Err(PyValueError::new_err("invalid input (expected float, float, bool)")) }
  }
  /// Conditional cumulative multiply: r[t] = r[t-1] * (cond[t] ? input[t] : 1)
  #[pyfunction]
  fn scan_mul<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    condition: &'py Bound<'_, PyAny>,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some(((mut r, input), condition)) = r
      .extract::<PyReadwriteArray1<'py, f64>>() .ok()
      .zip(input.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(condition.extract::<PyReadonlyArray1<'py, bool>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let input = input.as_array();
      let input = input.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let condition = condition.as_array();
      let condition = condition.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_scan_mul(&ctx, r, input, condition).map_err(|e| e.into())
    } else if let Some(((mut r, input), condition)) = r
      .extract::<PyReadwriteArray1<'py, f32>>() .ok()
      .zip(input.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(condition.extract::<PyReadonlyArray1<'py, bool>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let input = input.as_array();
      let input = input.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let condition = condition.as_array();
      let condition = condition.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_scan_mul(&ctx, r, input, condition).map_err(|e| e.into())
    } else { Err(PyValueError::new_err("invalid input (expected float, float, bool)")) }
  }
  /// Rolling Sharpe Ratio of returns.
  #[pyfunction]
  fn sharpe<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_sharpe(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_sharpe(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_sharpe(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_sharpe(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Calculate rolling sample Skewness over a moving window
  #[pyfunction]
  fn skewness<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_skewness(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_skewness(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_skewness(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_skewness(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Linear Regression Slope
  #[pyfunction]
  fn slope<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_slope(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_slope(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_slope(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_slope(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Exponential Moving Average (variant of well-known EMA) weight = m / n
  #[pyfunction]
  fn sma<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    n: usize,
    m: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_sma(&ctx, r, input, n, m).map_err(|e| e.into())
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
      ta_sma(&ctx, r, input, n, m).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_sma(&ctx, r, input, n, m).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_sma(&ctx, r, input, n, m).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Calculate Standard Deviation over a moving window
  #[pyfunction]
  fn stddev<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_stddev(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_stddev(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_stddev(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_stddev(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Calculate sum of values in preceding `periods` window
  #[pyfunction]
  fn sum<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_sum(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_sum(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_sum(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_sum(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Calculate number of periods (bars) backwards until the sum of values is greater than or equal to `amount`
  #[pyfunction]
  fn sumbars<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    amount: f64,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_sumbars(&ctx, r, input, amount).map_err(|e| e.into())
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
      ta_sumbars(&ctx, r, input, amount as f32).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_sumbars(&ctx, r, input, amount).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_sumbars(&ctx, r, input, amount as f32).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Calculate sum of values in preceding `periods` window where `condition` is true
  #[pyfunction]
  fn sumif<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    condition: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
    let mut ctx = ctx(py);
    // 2. check input type and do dispatch
    if let Some(((mut r, input), condition)) = r
      .extract::<PyReadwriteArray1<'py, f64>>() .ok()
      .zip(input.extract::<PyReadonlyArray1<'py, f64>>().ok())
      .zip(condition.extract::<PyReadonlyArray1<'py, bool>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let input = input.as_array();
      let input = input.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let condition = condition.as_array();
      let condition = condition.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_sumif(&ctx, r, input, condition, periods).map_err(|e| e.into())
    } else if let Some(((mut r, input), condition)) = r
      .extract::<PyReadwriteArray1<'py, f32>>() .ok()
      .zip(input.extract::<PyReadonlyArray1<'py, f32>>().ok())
      .zip(condition.extract::<PyReadonlyArray1<'py, bool>>().ok()) {
      let mut r = r.as_array_mut();
      let r = r.as_slice_mut().ok_or(PyValueError::new_err("failed to get mutable slice"))?;
      let input = input.as_array();
      let input = input.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      let condition = condition.as_array();
      let condition = condition.as_slice().ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_sumif(&ctx, r, input, condition, periods).map_err(|e| e.into())
    } else { Err(PyValueError::new_err("invalid input (expected float, float, bool)")) }
  }
  /// Calculate Variance over a moving window
  #[pyfunction]
  fn var<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_var(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_var(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_var(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_var(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Calculate weighted delay (exponentially weighted lag)
  #[pyfunction]
  fn weighted_delay<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_weighted_delay(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_weighted_delay(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_weighted_delay(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_weighted_delay(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }
  /// Calculate rolling Z-Score over a moving window
  #[pyfunction]
  fn zscore<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    #[allow(unused_mut)]
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
      ta_zscore(&ctx, r, input, periods).map_err(|e| e.into())
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
      ta_zscore(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_zscore(&ctx, r, input, periods).map_err(|e| e.into())
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
            let r = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((r, input)) = r.zip(input) {
              ta_zscore(&ctx, r, input, periods).map_err(|e| e.into())
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
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      Err(PyValueError::new_err("invalid input"))
    }
  }

pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
  m.add_function(wrap_pyfunction!(alpha, m)?)?;
  m.add_function(wrap_pyfunction!(backfill, m)?)?;
  m.add_function(wrap_pyfunction!(barslast, m)?)?;
  m.add_function(wrap_pyfunction!(barssince, m)?)?;
  m.add_function(wrap_pyfunction!(beta, m)?)?;
  m.add_function(wrap_pyfunction!(bins, m)?)?;
  m.add_function(wrap_pyfunction!(bw_split, m)?)?;
  m.add_function(wrap_pyfunction!(cc_rank, m)?)?;
  m.add_function(wrap_pyfunction!(cc_zscore, m)?)?;
  m.add_function(wrap_pyfunction!(corr, m)?)?;
  m.add_function(wrap_pyfunction!(corr2, m)?)?;
  m.add_function(wrap_pyfunction!(count, m)?)?;
  m.add_function(wrap_pyfunction!(count_nans, m)?)?;
  m.add_function(wrap_pyfunction!(cov, m)?)?;
  m.add_function(wrap_pyfunction!(cross, m)?)?;
  m.add_function(wrap_pyfunction!(dma, m)?)?;
  m.add_function(wrap_pyfunction!(entropy, m)?)?;
  m.add_function(wrap_pyfunction!(fret, m)?)?;
  m.add_function(wrap_pyfunction!(fw_split, m)?)?;
  m.add_function(wrap_pyfunction!(group_rank, m)?)?;
  m.add_function(wrap_pyfunction!(group_zscore, m)?)?;
  m.add_function(wrap_pyfunction!(hhv, m)?)?;
  m.add_function(wrap_pyfunction!(hhvbars, m)?)?;
  m.add_function(wrap_pyfunction!(intercept, m)?)?;
  m.add_function(wrap_pyfunction!(kurtosis, m)?)?;
  m.add_function(wrap_pyfunction!(llv, m)?)?;
  m.add_function(wrap_pyfunction!(llvbars, m)?)?;
  m.add_function(wrap_pyfunction!(longcross, m)?)?;
  m.add_function(wrap_pyfunction!(lwma, m)?)?;
  m.add_function(wrap_pyfunction!(ma, m)?)?;
  m.add_function(wrap_pyfunction!(max_drawdown, m)?)?;
  m.add_function(wrap_pyfunction!(min_max_diff, m)?)?;
  m.add_function(wrap_pyfunction!(moment, m)?)?;
  m.add_function(wrap_pyfunction!(neutralize, m)?)?;
  m.add_function(wrap_pyfunction!(product, m)?)?;
  m.add_function(wrap_pyfunction!(quantile, m)?)?;
  m.add_function(wrap_pyfunction!(rank, m)?)?;
  m.add_function(wrap_pyfunction!(rcross, m)?)?;
  m.add_function(wrap_pyfunction!(r#ref, m)?)?;
  m.add_function(wrap_pyfunction!(regbeta, m)?)?;
  m.add_function(wrap_pyfunction!(regresi, m)?)?;
  m.add_function(wrap_pyfunction!(rlongcross, m)?)?;
  m.add_function(wrap_pyfunction!(scan_add, m)?)?;
  m.add_function(wrap_pyfunction!(scan_mul, m)?)?;
  m.add_function(wrap_pyfunction!(sharpe, m)?)?;
  m.add_function(wrap_pyfunction!(skewness, m)?)?;
  m.add_function(wrap_pyfunction!(slope, m)?)?;
  m.add_function(wrap_pyfunction!(sma, m)?)?;
  m.add_function(wrap_pyfunction!(stddev, m)?)?;
  m.add_function(wrap_pyfunction!(sum, m)?)?;
  m.add_function(wrap_pyfunction!(sumbars, m)?)?;
  m.add_function(wrap_pyfunction!(sumif, m)?)?;
  m.add_function(wrap_pyfunction!(var, m)?)?;
  m.add_function(wrap_pyfunction!(weighted_delay, m)?)?;
  m.add_function(wrap_pyfunction!(zscore, m)?)?;
  Ok(())
}
