use anyhow::{Result, anyhow, bail};
use std::fmt::Write as FmtWrite;
use std::io::Write;
use std::{env, fs, path::Path};

/// Type of ta function parameter with name
#[derive(Debug, Default)]
enum TaType {
  #[default]
  None,
  Num(String),
  Int(String),
  NumArray(String),
  BoolArray(String),
  #[allow(dead_code)]
  IntArray(String),
  #[allow(dead_code)]
  Bool(String),
  #[allow(dead_code)]
  Context(String),
}

impl TaType {
  #[allow(dead_code)]
  fn name(&self) -> &str {
    match self {
      TaType::None => "",
      TaType::Num(n) => n,
      TaType::Int(n) => n,
      TaType::NumArray(n) => n,
      TaType::BoolArray(n) => n,
      TaType::IntArray(n) => n,
      TaType::Bool(n) => n,
      TaType::Context(n) => n,
    }
  }

  fn to_py_type_str(&self) -> String {
    match self {
      TaType::None => "None".to_string(),
      TaType::Num(n) => format!("{}: float", n),
      TaType::Int(n) => format!("{}: int", n),
      TaType::NumArray(n) => format!("{}: np.ndarray[float]", n),
      TaType::BoolArray(n) => format!("{}: np.ndarray[bool]", n),
      TaType::IntArray(n) => format!("{}: np.ndarray[int]", n),
      TaType::Bool(n) => format!("{}: bool", n),
      TaType::Context(n) => format!("{}: Context", n),
    }
  }
}

impl TryFrom<&str> for TaType {
  type Error = anyhow::Error;

  fn try_from(value: &str) -> Result<Self> {
    // format: name: type
    let (name, ty) = value
      .split_once(':')
      .map(|(name, ty)| (name.trim(), ty.trim()))
      .ok_or(anyhow!("invalid ta function parameter format: {}", value))?;

    match ty {
      "NumT" | "&NumT" => Ok(TaType::Num(name.to_string())),
      "bool" | "&bool" => Ok(TaType::Bool(name.to_string())),
      "usize" | "&usize" => Ok(TaType::Int(name.to_string())),
      "&[NumT]" | "&mut [NumT]" => Ok(TaType::NumArray(name.to_string())),
      "&[bool]" | "&mut [bool]" => Ok(TaType::BoolArray(name.to_string())),
      "&[usize]" | "&mut [usize]" => Ok(TaType::IntArray(name.to_string())),
      "Context" | "&Context" => Ok(TaType::Context(name.to_string())),
      _ => bail!("invalid ta function parameter type: {}", value),
    }
  }
}

/// A ta function
///
/// ta function is function that name start with `ta_`
#[derive(Debug, Default)]
struct TaFunc {
  /// function name, without `ta_` prefix
  name: String,
  /// function parameters
  params: Vec<TaType>,
  /// function documentation
  doc: String,
}

fn parse_ta_file<P: AsRef<Path>>(file_name: P) -> Result<Vec<TaFunc>> {
  let content = fs::read_to_string(file_name)?;
  let mut functions = Vec::new();
  let mut current_doc = String::new();
  let mut lines = content.lines().peekable();

  while let Some(line) = lines.next() {
    let trimmed = line.trim();
    if trimmed.starts_with("///") {
      let doc_line = trimmed.strip_prefix("///").unwrap().trim();
      if !current_doc.is_empty() {
        current_doc.push('\n');
      }
      current_doc.push_str(doc_line);
      continue;
    }

    if trimmed.is_empty() {
      continue;
    }

    let code_line = if let Some(idx) = line.find("//") {
      &line[..idx]
    } else {
      line
    };

    if let Some(idx) = code_line.find("pub fn ta_") {
      let after_fn = &code_line[idx + "pub fn ta_".len()..];
      let name_end = after_fn
        .find(|c| c == '<' || c == '(')
        .ok_or(anyhow!("invalid function definition: {}", line))?;
      let name = after_fn[..name_end].trim().to_string();

      let mut full_decl = code_line.to_string();
      while !full_decl.contains(')') {
        if let Some(next_line) = lines.next() {
          let next_trimmed = next_line.trim();
          if next_trimmed.starts_with("///") {
            continue;
          }
          let next_code = if let Some(c_idx) = next_line.find("//") {
            &next_line[..c_idx]
          } else {
            next_line
          };
          full_decl.push_str(" ");
          full_decl.push_str(next_code.trim());
        } else {
          break;
        }
      }

      let args_start = full_decl
        .find('(')
        .ok_or(anyhow!("function parameter start not found"))?;

      // Find the closing parenthesis corresponding to the arguments
      // We look after args_start
      let args_end = full_decl[args_start + 1..]
        .find(')')
        .map(|i| args_start + 1 + i)
        .ok_or(anyhow!("function parameter end not found"))?;

      let args_str = &full_decl[args_start + 1..args_end];
      let mut params = Vec::new();
      if !args_str.trim().is_empty() {
        for arg in args_str.split(',') {
          let arg = arg.trim();
          if arg.is_empty() {
            continue;
          }
          params.push(TaType::try_from(arg)?);
        }
      }

      functions.push(TaFunc {
        name,
        params,
        doc: std::mem::take(&mut current_doc),
      });
    } else {
      current_doc.clear();
    }
  }
  Ok(functions)
}

fn build_py_bindings(functions: &[TaFunc]) -> Result<()> {
  let out_dir = env::var("OUT_DIR")?;
  let mut file = fs::File::create(out_dir + "/algo_bindings.rs")?;

  let mut code = String::new();

  for func in functions {
    // We only support functions with array outputs (NumArray) as the first argument for now,
    // tailored to the ema example.
    // In strict sense we should make this more generic, but sticking to requested features first.

    // Find arguments that are NumArray or specific types
    // The first argument is output array (r), second is input array (input) for ema.
    // Let's assume standard TA signature: (ctx, r, input, params...)

    // We need to map arguments for the python signature.
    // Python signature: (py, r, input, ...)

    let py_func_name = &func.name; // e.g. "ema"
    let rust_func_name = format!("ta_{}", func.name);

    // Build python function signature
    let mut py_args = String::new();

    // Iterating params to build signature
    // We skip Context as it is internal
    for param in &func.params {
      match param {
        TaType::Context(_) => continue,
        TaType::BoolArray(n) => {
          writeln!(py_args, "    {}: &'py Bound<'_, PyAny>,", n)?;
        }
        TaType::NumArray(n) => {
          // For python args, we just list them.
          // However based on "ema" example:
          // r: &'py Bound<'_, PyAny>, input: &'py Bound<'_, PyAny>
          // are the first two array args.
          writeln!(py_args, "    {}: &'py Bound<'_, PyAny>,", n)?;
        }
        TaType::Num(n) => {
          writeln!(py_args, "    {}: f64,", n)?;
          // For dispatch call, we need to cast if necessary or just pass
        }
        TaType::Int(n) => {
          // usize in rust, int in python
          writeln!(py_args, "    {}: usize,", n)?;
        }
        _ => {} // Handle others as needed
      }
    }

    writeln!(
      code,
      "  /// {}",
      func.doc.lines().next().unwrap_or("").trim()
    )?;
    writeln!(code, "  #[pyfunction]")?;
    if py_func_name == "ref" {
      writeln!(code, "  fn r#{}<'py>(", py_func_name)?;
    } else {
      writeln!(code, "  fn {}<'py>(", py_func_name)?;
    }
    writeln!(code, "    py: Python<'py>,")?;
    write!(code, "{}", py_args)?;
    writeln!(code, "  ) -> PyResult<()> {{")?;

    writeln!(code, "    // 1. get context")?;
    writeln!(code, "    #[allow(unused_mut)]")?;
    writeln!(code, "    let mut ctx = ctx(py);")?;

    writeln!(code, "    // 2. check input type and do dispatch")?;

    // We assume the first two NumArrays are 'r' and 'input' based on the template
    // This is a bit rigid but fits the `ema` template requirement.
    // Finding the names of the first two NumArrays
    // Finding the names of the first two Arrays (NumArray or BoolArray)
    let arrays: Vec<(&String, &TaType)> = func
      .params
      .iter()
      .filter_map(|p| match p {
        TaType::NumArray(n) => Some((n, p)),
        TaType::BoolArray(n) => Some((n, p)),
        _ => None,
      })
      .collect();

    // Support 2 arrays (r, input) or 3 arrays (r, a, b)
    if arrays.len() < 2 {
      // Fallback or skip if not matching pattern
      writeln!(code, "    Ok(())")?;
      writeln!(code, "  }}")?;
      continue;
    }

    if arrays.len() == 4 {
      let r_name = arrays[0].0;
      let a_name = arrays[1].0;
      let b_name = arrays[2].0;
      let c_name = arrays[3].0;

      let gen_args = |type_name: &str| -> String {
        let mut args = String::new();
        for param in &func.params {
          match param {
            TaType::NumArray(_) | TaType::BoolArray(_) | TaType::Context(_) => {}
            TaType::Num(n) => {
              if type_name == "f32" {
                let _ = write!(args, ", {} as f32", n);
              } else {
                let _ = write!(args, ", {}", n);
              }
            }
            TaType::Int(n) => {
              let _ = write!(args, ", {}", n);
            }
            _ => {}
          }
        }
        args
      };

      let args_f64 = gen_args("f64");
      let args_f32 = gen_args("f32");

      writeln!(
        code,
        "    if let Some((((mut {}, {}), {}), {})) = {}",
        r_name, a_name, b_name, c_name, r_name
      )?;
      writeln!(
        code,
        "      .extract::<PyReadwriteArray1<'py, f64>>().ok()"
      )?;
      writeln!(
        code,
        "      .zip({}.extract::<PyReadonlyArray1<'py, f64>>().ok())",
        a_name
      )?;
      writeln!(
        code,
        "      .zip({}.extract::<PyReadonlyArray1<'py, f64>>().ok())",
        b_name
      )?;
      writeln!(
        code,
        "      .zip({}.extract::<PyReadonlyArray1<'py, f64>>().ok()) {{",
        c_name
      )?;

      writeln!(code, "      let mut {} = {}.as_array_mut();", r_name, r_name)?;
      writeln!(code, "      let {} = {}.as_slice_mut().ok_or(PyValueError::new_err(\"failed to get mutable slice\"))?;", r_name, r_name)?;

      writeln!(code, "      let {} = {}.as_array();", a_name, a_name)?;
      writeln!(code, "      let {} = {}.as_slice().ok_or(PyValueError::new_err(\"failed to get slice\"))?;", a_name, a_name)?;
      writeln!(code, "      let {} = {}.as_array();", b_name, b_name)?;
      writeln!(code, "      let {} = {}.as_slice().ok_or(PyValueError::new_err(\"failed to get slice\"))?;", b_name, b_name)?;
      writeln!(code, "      let {} = {}.as_array();", c_name, c_name)?;
      writeln!(code, "      let {} = {}.as_slice().ok_or(PyValueError::new_err(\"failed to get slice\"))?;", c_name, c_name)?;

      writeln!(
        code,
        "      {}(&ctx, {}, {}, {}, {}{}).map_err(|e| e.into())",
        rust_func_name, r_name, a_name, b_name, c_name, args_f64
      )?;

      writeln!(
        code,
        "    }} else if let Some((((mut {}, {}), {}), {})) = {}",
        r_name, a_name, b_name, c_name, r_name
      )?;
      writeln!(
        code,
        "      .extract::<PyReadwriteArray1<'py, f32>>().ok()"
      )?;
      writeln!(
        code,
        "      .zip({}.extract::<PyReadonlyArray1<'py, f32>>().ok())",
        a_name
      )?;
      writeln!(
        code,
        "      .zip({}.extract::<PyReadonlyArray1<'py, f32>>().ok())",
        b_name
      )?;
      writeln!(
        code,
        "      .zip({}.extract::<PyReadonlyArray1<'py, f32>>().ok()) {{",
        c_name
      )?;

      writeln!(code, "      let mut {} = {}.as_array_mut();", r_name, r_name)?;
      writeln!(code, "      let {} = {}.as_slice_mut().ok_or(PyValueError::new_err(\"failed to get mutable slice\"))?;", r_name, r_name)?;

      writeln!(code, "      let {} = {}.as_array();", a_name, a_name)?;
      writeln!(code, "      let {} = {}.as_slice().ok_or(PyValueError::new_err(\"failed to get slice\"))?;", a_name, a_name)?;
      writeln!(code, "      let {} = {}.as_array();", b_name, b_name)?;
      writeln!(code, "      let {} = {}.as_slice().ok_or(PyValueError::new_err(\"failed to get slice\"))?;", b_name, b_name)?;
      writeln!(code, "      let {} = {}.as_array();", c_name, c_name)?;
      writeln!(code, "      let {} = {}.as_slice().ok_or(PyValueError::new_err(\"failed to get slice\"))?;", c_name, c_name)?;

      writeln!(
        code,
        "      {}(&ctx, {}, {}, {}, {}{}).map_err(|e| e.into())",
        rust_func_name, r_name, a_name, b_name, c_name, args_f32
      )?;

      writeln!(code, "    }} else {{ Err(PyValueError::new_err(\"invalid input\")) }}")?;
      writeln!(code, "  }}")?;
      continue;
    }

    if arrays.len() == 3 {
      // 3 arrays: r (output), a (input), b (input)
      let r_name = arrays[0].0;
      let a_name = arrays[1].0;
      let b_name = arrays[2].0;

      // Assume output is BoolArray (for CROSS) or NumArray depending on usage.
      // Actually for CROSS, r is defined as &mut [bool] in rust, so it is BoolArray in TaType?
      // Let's check TaType parsing.
      // &mut [bool] -> BoolArray.
      // But `CROSS` output is `&mut [bool]`.
      // My implementation in `cross.rs` uses `&mut [bool]`.
      // In python, we probably want to return `bool` array or `float` array (AmiBroker uses float 0.0/1.0 usually).
      // Standard AmiBroker logic uses 1.0/0.0 for True/False conditions.
      // But some return actual bools?
      // Looking at `algo_gen.py` for COMPARAISON funcs like logic, wait.
      // `BARSLAST` takes `&[bool]` but returns `&mut [NumT]`.
      // `CROSS` returns boolean condition (True/False).
      // If I use `&mut [bool]`, I need to extract `PyReadwriteArray1<bool>`.
      // This is fine. Python users can filter with it.

      // However, I should check if `NumArray` vs `BoolArray` affects generation logic significantly.
      // `r_name` extraction depends on its type.

      let r_is_bool = matches!(arrays[0].1, TaType::BoolArray(_));
      // let a_is_bool = matches!(arrays[1].1, TaType::BoolArray(_));
      // let b_is_bool = matches!(arrays[2].1, TaType::BoolArray(_));

      // We only implement typical numeric CROSS(Num, Num) -> Bool for now.
      // r: BoolArray, a: NumArray, b: NumArray.

      let gen_args = |type_name: &str| -> String {
        let mut args = String::new();
        for param in &func.params {
          match param {
            TaType::NumArray(_) | TaType::BoolArray(_) | TaType::Context(_) => {}
            TaType::Num(n) => {
              if type_name == "f32" {
                let _ = write!(args, ", {} as f32", n);
              } else {
                let _ = write!(args, ", {}", n);
              }
            }
            TaType::Int(n) => {
              let _ = write!(args, ", {}", n);
            }
            _ => {}
          }
        }
        args
      };

      let args_f64 = gen_args("f64");
      let args_f32 = gen_args("f32");
      // Actually we should support f32 inputs if possible.
      // But for "lazy" implementation, let's stick to f64 compatibility or just handle f64.

      // Generate code for (r: Bool, a: Num, b: Num)
      // Also handling List inputs.

      // 1. Array inputs
      if r_is_bool {
        // Expect r to be bool array
        writeln!(
          code,
          "    if let Some(((mut {}, {}), {})) = {}",
          r_name, a_name, b_name, r_name
        )?;
        writeln!(
          code,
          "      .extract::<PyReadwriteArray1<'py, bool>>() .ok()"
        )?;
        writeln!(
          code,
          "      .zip({}.extract::<PyReadonlyArray1<'py, f64>>().ok())",
          a_name
        )?;
        writeln!(
          code,
          "      .zip({}.extract::<PyReadonlyArray1<'py, f64>>().ok()) {{",
          b_name
        )?;

        writeln!(
          code,
          "      let mut {} = {}.as_array_mut();",
          r_name, r_name
        )?;
        writeln!(
          code,
          "      let {} = {}.as_slice_mut().ok_or(PyValueError::new_err(\"failed to get mutable slice\"))?;",
          r_name, r_name
        )?;

        writeln!(code, "      let {} = {}.as_array();", a_name, a_name)?;
        writeln!(
          code,
          "      let {} = {}.as_slice().ok_or(PyValueError::new_err(\"failed to get slice\"))?;",
          a_name, a_name
        )?;

        writeln!(code, "      let {} = {}.as_array();", b_name, b_name)?;
        writeln!(
          code,
          "      let {} = {}.as_slice().ok_or(PyValueError::new_err(\"failed to get slice\"))?;",
          b_name, b_name
        )?;

        writeln!(
          code,
          "      {}(&ctx, {}, {}, {}{}).map_err(|e| e.into())",
          rust_func_name, r_name, a_name, b_name, args_f64
        )?;

        writeln!(
          code,
          "    }} else if let Some((({}, {}), {})) = {}.cast::<PyList>().ok().zip({}.cast::<PyList>().ok()).zip({}.cast::<PyList>().ok()) {{",
          r_name, a_name, b_name, r_name, a_name, b_name
        )?;
        // List logic
        writeln!(
          code,
          "      if {}.len() != {}.len() || {}.len() != {}.len() {{ return Err(PyValueError::new_err(\"length mismatch\")); }}",
          r_name, a_name, b_name, a_name
        )?;

        writeln!(
          code,
          "      if let Some(((mut {}, {}), {})) = {}",
          r_name, a_name, b_name, r_name
        )?;
        writeln!(
          code,
          "        .extract::<Vec<PyReadwriteArray1<'py, bool>>>().ok()"
        )?;
        writeln!(
          code,
          "        .zip({}.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok())",
          a_name
        )?;
        writeln!(
          code,
          "        .zip({}.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok()) {{",
          b_name
        )?;
        writeln!(
          code,
          "        let {} = {}.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();",
          r_name, r_name
        )?;
        writeln!(
          code,
          "        let {} = {}.iter().map(|x| x.as_array()).collect::<Vec<_>>();",
          a_name, a_name
        )?;
        writeln!(
          code,
          "        let {} = {}.iter().map(|x| x.as_array()).collect::<Vec<_>>();",
          b_name, b_name
        )?;

        writeln!(code, "        let mut _r = vec![];")?;
        writeln!(
          code,
          "        {}.into_par_iter().zip({}.into_par_iter()).zip({}.into_par_iter())",
          r_name, a_name, b_name
        )?;
        writeln!(code, "          .map(|((mut out, a), b)| {{")?;
        writeln!(
          code,
          "            let {} = out.as_slice_mut().ok_or(PyValueError::new_err(\"failed to get mutable slice\"))?;",
          r_name
        )?;
        writeln!(
          code,
          "            let {} = a.as_slice().ok_or(PyValueError::new_err(\"failed to get slice\"))?;",
          a_name
        )?;
        writeln!(
          code,
          "            let {} = b.as_slice().ok_or(PyValueError::new_err(\"failed to get slice\"))?;",
          b_name
        )?;
        writeln!(
          code,
          "            {}(&ctx, {}, {}, {}{}).map_err(|e| e.into())",
          rust_func_name, r_name, a_name, b_name, args_f64
        )?;
        writeln!(code, "          }}).collect_into_vec(&mut _r);")?;
        writeln!(
          code,
          "        match _r.into_iter().find(|x| x.is_err()) {{ Some(e) => e, None => Ok(()) }}"
        )?;

        writeln!(
          code,
          "      }} else {{ Err(PyValueError::new_err(\"invalid input lists\")) }}"
        )?;

        writeln!(
          code,
          "    }} else {{ Err(PyValueError::new_err(\"invalid input\")) }}"
        )?;

        writeln!(code, "  }}")?;
        continue;
      } else {
        // Assume numeric output (NumT) if not bool
        // f64
        writeln!(
          code,
          "    if let Some(((mut {}, {}), {})) = {}",
          r_name, a_name, b_name, r_name
        )?;
        writeln!(
          code,
          "      .extract::<PyReadwriteArray1<'py, f64>>() .ok()"
        )?;
        writeln!(
          code,
          "      .zip({}.extract::<PyReadonlyArray1<'py, f64>>().ok())",
          a_name
        )?;
        writeln!(
          code,
          "      .zip({}.extract::<PyReadonlyArray1<'py, f64>>().ok()) {{",
          b_name
        )?;

        writeln!(
          code,
          "      let mut {} = {}.as_array_mut();",
          r_name, r_name
        )?;
        writeln!(
          code,
          "      let {} = {}.as_slice_mut().ok_or(PyValueError::new_err(\"failed to get mutable slice\"))?;",
          r_name, r_name
        )?;

        writeln!(code, "      let {} = {}.as_array();", a_name, a_name)?;
        writeln!(
          code,
          "      let {} = {}.as_slice().ok_or(PyValueError::new_err(\"failed to get slice\"))?;",
          a_name, a_name
        )?;

        writeln!(code, "      let {} = {}.as_array();", b_name, b_name)?;
        writeln!(
          code,
          "      let {} = {}.as_slice().ok_or(PyValueError::new_err(\"failed to get slice\"))?;",
          b_name, b_name
        )?;

        writeln!(
          code,
          "      {}(&ctx, {}, {}, {}{}).map_err(|e| e.into())",
          rust_func_name, r_name, a_name, b_name, args_f64
        )?;

        // f32
        writeln!(
          code,
          "    }} else if let Some(((mut {}, {}), {})) = {}",
          r_name, a_name, b_name, r_name
        )?;
        writeln!(
          code,
          "      .extract::<PyReadwriteArray1<'py, f32>>() .ok()"
        )?;
        writeln!(
          code,
          "      .zip({}.extract::<PyReadonlyArray1<'py, f32>>().ok())",
          a_name
        )?;
        writeln!(
          code,
          "      .zip({}.extract::<PyReadonlyArray1<'py, f32>>().ok()) {{",
          b_name
        )?;

        writeln!(
          code,
          "      let mut {} = {}.as_array_mut();",
          r_name, r_name
        )?;
        writeln!(
          code,
          "      let {} = {}.as_slice_mut().ok_or(PyValueError::new_err(\"failed to get mutable slice\"))?;",
          r_name, r_name
        )?;

        writeln!(code, "      let {} = {}.as_array();", a_name, a_name)?;
        writeln!(
          code,
          "      let {} = {}.as_slice().ok_or(PyValueError::new_err(\"failed to get slice\"))?;",
          a_name, a_name
        )?;

        writeln!(code, "      let {} = {}.as_array();", b_name, b_name)?;
        writeln!(
          code,
          "      let {} = {}.as_slice().ok_or(PyValueError::new_err(\"failed to get slice\"))?;",
          b_name, b_name
        )?;

        writeln!(
          code,
          "      {}(&ctx, {}, {}, {}{}).map_err(|e| e.into())",
          rust_func_name, r_name, a_name, b_name, args_f32
        )?;

        // List
        writeln!(
          code,
          "    }} else if let Some((({}, {}), {})) = {}.cast::<PyList>().ok().zip({}.cast::<PyList>().ok()).zip({}.cast::<PyList>().ok()) {{",
          r_name, a_name, b_name, r_name, a_name, b_name
        )?;
        writeln!(
          code,
          "      if {}.len() != {}.len() || {}.len() != {}.len() {{ return Err(PyValueError::new_err(\"length mismatch\")); }}",
          r_name, a_name, b_name, a_name
        )?;

        // List f64
        writeln!(
          code,
          "      if let Some(((mut {}, {}), {})) = {}",
          r_name, a_name, b_name, r_name
        )?;
        writeln!(
          code,
          "        .extract::<Vec<PyReadwriteArray1<'py, f64>>>().ok()"
        )?;
        writeln!(
          code,
          "        .zip({}.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok())",
          a_name
        )?;
        writeln!(
          code,
          "        .zip({}.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok()) {{",
          b_name
        )?;
        writeln!(
          code,
          "        let {} = {}.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();",
          r_name, r_name
        )?;
        writeln!(
          code,
          "        let {} = {}.iter().map(|x| x.as_array()).collect::<Vec<_>>();",
          a_name, a_name
        )?;
        writeln!(
          code,
          "        let {} = {}.iter().map(|x| x.as_array()).collect::<Vec<_>>();",
          b_name, b_name
        )?;

        writeln!(code, "        let mut _r = vec![];")?;
        writeln!(
          code,
          "        {}.into_par_iter().zip({}.into_par_iter()).zip({}.into_par_iter())",
          r_name, a_name, b_name
        )?;
        writeln!(code, "          .map(|((mut out, a), b)| {{")?;
        writeln!(
          code,
          "            let {} = out.as_slice_mut().ok_or(PyValueError::new_err(\"failed to get mutable slice\"))?;",
          r_name
        )?;
        writeln!(
          code,
          "            let {} = a.as_slice().ok_or(PyValueError::new_err(\"failed to get slice\"))?;",
          a_name
        )?;
        writeln!(
          code,
          "            let {} = b.as_slice().ok_or(PyValueError::new_err(\"failed to get slice\"))?;",
          b_name
        )?;
        writeln!(
          code,
          "            {}(&ctx, {}, {}, {}{}).map_err(|e| e.into())",
          rust_func_name, r_name, a_name, b_name, args_f64
        )?;
        writeln!(code, "          }}).collect_into_vec(&mut _r);")?;
        writeln!(
          code,
          "        match _r.into_iter().find(|x| x.is_err()) {{ Some(e) => e, None => Ok(()) }}"
        )?;

        // List f32
        writeln!(
          code,
          "      }} else if let Some(((mut {}, {}), {})) = {}",
          r_name, a_name, b_name, r_name
        )?;
        writeln!(
          code,
          "        .extract::<Vec<PyReadwriteArray1<'py, f32>>>().ok()"
        )?;
        writeln!(
          code,
          "        .zip({}.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok())",
          a_name
        )?;
        writeln!(
          code,
          "        .zip({}.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok()) {{",
          b_name
        )?;
        writeln!(
          code,
          "        let {} = {}.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();",
          r_name, r_name
        )?;
        writeln!(
          code,
          "        let {} = {}.iter().map(|x| x.as_array()).collect::<Vec<_>>();",
          a_name, a_name
        )?;
        writeln!(
          code,
          "        let {} = {}.iter().map(|x| x.as_array()).collect::<Vec<_>>();",
          b_name, b_name
        )?;

        writeln!(code, "        let mut _r = vec![];")?;
        writeln!(
          code,
          "        {}.into_par_iter().zip({}.into_par_iter()).zip({}.into_par_iter())",
          r_name, a_name, b_name
        )?;
        writeln!(code, "          .map(|((mut out, a), b)| {{")?;
        writeln!(
          code,
          "            let {} = out.as_slice_mut().ok_or(PyValueError::new_err(\"failed to get mutable slice\"))?;",
          r_name
        )?;
        writeln!(
          code,
          "            let {} = a.as_slice().ok_or(PyValueError::new_err(\"failed to get slice\"))?;",
          a_name
        )?;
        writeln!(
          code,
          "            let {} = b.as_slice().ok_or(PyValueError::new_err(\"failed to get slice\"))?;",
          b_name
        )?;
        writeln!(
          code,
          "            {}(&ctx, {}, {}, {}{}).map_err(|e| e.into())",
          rust_func_name, r_name, a_name, b_name, args_f32
        )?;
        writeln!(code, "          }}).collect_into_vec(&mut _r);")?;
        writeln!(
          code,
          "        match _r.into_iter().find(|x| x.is_err()) {{ Some(e) => e, None => Ok(()) }}"
        )?;
        writeln!(
          code,
          "      }} else {{ Err(PyValueError::new_err(\"invalid input lists\")) }}"
        )?;

        writeln!(
          code,
          "    }} else {{ Err(PyValueError::new_err(\"invalid input\")) }}"
        )?;

        writeln!(code, "  }}")?;
        continue;
      }
    }

    let r_name = arrays[0].0;
    let input_name = arrays[1].0;
    let input_is_bool = matches!(arrays[1].1, TaType::BoolArray(_));

    // Additional params for the function call
    // We need to pass them to the rust function.
    // context is first.
    // r and input are handled in the dispatch blocks.
    // other params come after.

    // Construct the argument list for the rust call `ta_xxx`
    // We need to know the order in rust function.
    // (&ctx, r, input, periods...)

    // We'll generate the call args string dynamically inside the loops
    // But non-array params are just passed by name.

    // We'll generate the call args string dynamically inside the loops
    // Helper closure to generate args with specific type
    let gen_args = |type_name: &str| -> String {
      let mut args = String::new();
      for param in &func.params {
        match param {
          TaType::NumArray(_) | TaType::Context(_) => {} // Handled separately
          TaType::Num(n) => {
            if type_name == "f32" {
              let _ = write!(args, ", {} as f32", n);
            } else {
              let _ = write!(args, ", {}", n);
            }
          }
          TaType::Int(n) => {
            let _ = write!(args, ", {}", n);
          }
          _ => {}
        }
      }
      args
    };

    let args_f64 = gen_args("f64");
    let args_f32 = gen_args("f32");

    if input_is_bool {
      // f64 block with bool input
      writeln!(
        code,
        "    if let Some((mut {}, {})) = {}",
        r_name, input_name, r_name
      )?;
      writeln!(code, "      .extract::<PyReadwriteArray1<'py, f64>>()")?;
      writeln!(code, "      .ok()")?;
      writeln!(
        code,
        "      .zip({}.extract::<PyReadonlyArray1<'py, bool>>().ok())",
        input_name
      )?;
      writeln!(code, "    {{")?;
      writeln!(
        code,
        "      let mut {} = {}.as_array_mut();",
        r_name, r_name
      )?;
      writeln!(code, "      let {} = {}", r_name, r_name)?;
      writeln!(code, "        .as_slice_mut()")?;
      writeln!(
        code,
        "        .ok_or(PyValueError::new_err(\"failed to get mutable slice\"))?;"
      )?;
      writeln!(
        code,
        "      let {} = {}.as_array();",
        input_name, input_name
      )?;
      writeln!(code, "      let {} = {}", input_name, input_name)?;
      writeln!(code, "        .as_slice()")?;
      writeln!(
        code,
        "        .ok_or(PyValueError::new_err(\"failed to get slice\"))?;"
      )?;
      writeln!(
        code,
        "      {}(&ctx, {}, {}{}).map_err(|e| e.into())",
        rust_func_name, r_name, input_name, args_f64
      )?;

      // f32 block with bool input
      writeln!(
        code,
        "    }} else if let Some((mut {}, {})) = {}",
        r_name, input_name, r_name
      )?;
      writeln!(code, "      .extract::<PyReadwriteArray1<'py, f32>>()")?;
      writeln!(code, "      .ok()")?;
      writeln!(
        code,
        "      .zip({}.extract::<PyReadonlyArray1<'py, bool>>().ok())",
        input_name
      )?;
      writeln!(code, "    {{")?;
      writeln!(
        code,
        "      let mut {} = {}.as_array_mut();",
        r_name, r_name
      )?;
      writeln!(code, "      let {} = {}", r_name, r_name)?;
      writeln!(code, "        .as_slice_mut()")?;
      writeln!(
        code,
        "        .ok_or(PyValueError::new_err(\"invalid input\"))?;"
      )?;
      writeln!(
        code,
        "      let {} = {}.as_array();",
        input_name, input_name
      )?;
      writeln!(code, "      let {} = {}", input_name, input_name)?;
      writeln!(code, "        .as_slice()")?;
      writeln!(
        code,
        "        .ok_or(PyValueError::new_err(\"invalid input\"))?;"
      )?;
      writeln!(
        code,
        "      {}(&ctx, {}, {}{}).map_err(|e| e.into())",
        rust_func_name, r_name, input_name, args_f32
      )?;
    } else {
      // Standard NumArray input (f64/f32 matching)
      // f64 block
      writeln!(
        code,
        "    if let Some((mut {}, {})) = {}",
        r_name, input_name, r_name
      )?;
      writeln!(code, "      .extract::<PyReadwriteArray1<'py, f64>>()")?;
      writeln!(code, "      .ok()")?;
      writeln!(
        code,
        "      .zip({}.extract::<PyReadonlyArray1<'py, f64>>().ok())",
        input_name
      )?;
      writeln!(code, "    {{")?;
      writeln!(code, "      // input is f64 array")?;
      writeln!(
        code,
        "      let mut {} = {}.as_array_mut();",
        r_name, r_name
      )?;
      writeln!(code, "      let {} = {}", r_name, r_name)?;
      writeln!(code, "        .as_slice_mut()")?;
      writeln!(
        code,
        "        .ok_or(PyValueError::new_err(\"failed to get mutable slice\"))?;"
      )?;
      writeln!(
        code,
        "      let {} = {}.as_array();",
        input_name, input_name
      )?;
      writeln!(code, "      let {} = {}", input_name, input_name)?;
      writeln!(code, "        .as_slice()")?;
      writeln!(
        code,
        "        .ok_or(PyValueError::new_err(\"failed to get slice\"))?;"
      )?;
      writeln!(
        code,
        "      {}(&ctx, {}, {}{}).map_err(|e| e.into())",
        rust_func_name, r_name, input_name, args_f64
      )?;

      // f32 block
      writeln!(
        code,
        "    }} else if let Some((mut {}, {})) = {}",
        r_name, input_name, r_name
      )?;
      writeln!(code, "      .extract::<PyReadwriteArray1<'py, f32>>()")?;
      writeln!(code, "      .ok()")?;
      writeln!(
        code,
        "      .zip({}.extract::<PyReadonlyArray1<'py, f32>>().ok())",
        input_name
      )?;
      writeln!(code, "    {{")?;
      writeln!(code, "      // input is f32 array")?;
      writeln!(
        code,
        "      let mut {} = {}.as_array_mut();",
        r_name, r_name
      )?;
      writeln!(code, "      let {} = {}", r_name, r_name)?;
      writeln!(code, "        .as_slice_mut()")?;
      writeln!(
        code,
        "        .ok_or(PyValueError::new_err(\"invalid input\"))?;"
      )?;
      writeln!(
        code,
        "      let {} = {}.as_array();",
        input_name, input_name
      )?;
      writeln!(code, "      let {} = {}", input_name, input_name)?;
      writeln!(code, "        .as_slice()")?;
      writeln!(
        code,
        "        .ok_or(PyValueError::new_err(\"invalid input\"))?;"
      )?;
      writeln!(
        code,
        "      {}(&ctx, {}, {}{}).map_err(|e| e.into())",
        rust_func_name, r_name, input_name, args_f32
      )?;
    }

    // List block
    writeln!(
      code,
      "    }} else if let Some(({}, {})) = {}.cast::<PyList>().ok().zip({}.cast::<PyList>().ok()) {{",
      r_name, input_name, r_name, input_name
    )?;
    writeln!(code, "      // input is list of arrays")?;
    writeln!(
      code,
      "      // each array is a group, ensure groups is set to 1"
    )?;
    writeln!(code, "      ctx._groups = 1;")?;
    writeln!(code, "      if {}.len() != {}.len() {{", r_name, input_name)?;
    writeln!(
      code,
      "        return Err(PyValueError::new_err(\"length mismatch\"));"
    )?;
    writeln!(code, "      }}")?;

    if input_is_bool {
      // List f64 with bool input
      writeln!(
        code,
        "      // check if each array is f64 array output and bool input"
      )?;
      writeln!(
        code,
        "      if let Some((mut {}, {})) = {}",
        r_name, input_name, r_name
      )?;
      writeln!(
        code,
        "        .extract::<Vec<PyReadwriteArray1<'py, f64>>>()"
      )?;
      writeln!(code, "        .ok()")?;
      writeln!(
        code,
        "        .zip({}.extract::<Vec<PyReadonlyArray1<'py, bool>>>().ok())",
        input_name
      )?;
      writeln!(code, "      {{")?;
      writeln!(
        code,
        "        let {} = {}.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();",
        r_name, r_name
      )?;
      writeln!(
        code,
        "        let {} = {}.iter().map(|x| x.as_array()).collect::<Vec<_>>();",
        input_name, input_name
      )?;
      writeln!(code, "        let mut _r = vec![];")?;
      writeln!(code, "        {}.into_par_iter()", r_name)?;
      writeln!(code, "          .zip({}.into_par_iter())", input_name)?;
      writeln!(code, "          .map(|(mut out, input)| {{")?;
      writeln!(code, "            let {} = out.as_slice_mut();", r_name)?;
      writeln!(code, "            let {} = input.as_slice();", input_name)?;
      writeln!(
        code,
        "            if let Some(({}, {})) = {}.zip({}) {{",
        r_name, input_name, r_name, input_name
      )?;
      writeln!(
        code,
        "              {}(&ctx, {}, {}{}).map_err(|e| e.into())",
        rust_func_name, r_name, input_name, args_f64
      )?;
      writeln!(code, "            }} else {{")?;
      writeln!(
        code,
        "              Err(PyValueError::new_err(\"invalid input\"))"
      )?;
      writeln!(code, "            }}")?;
      writeln!(code, "          }})")?;
      writeln!(code, "          .collect_into_vec(&mut _r);")?;
      writeln!(code, "        match _r.into_iter().find(|x| x.is_err()) {{")?;
      writeln!(code, "          Some(e) => e,")?;
      writeln!(code, "          None => Ok(()),")?;
      writeln!(code, "        }}")?;

      // List f32 with bool input
      writeln!(code, "      // check if each array is f32 array output")?;
      writeln!(
        code,
        "      }} else if let Some((mut {}, {})) = {}",
        r_name, input_name, r_name
      )?;
      writeln!(
        code,
        "        .extract::<Vec<PyReadwriteArray1<'py, f32>>>()"
      )?;
      writeln!(code, "        .ok()")?;
      writeln!(
        code,
        "        .zip({}.extract::<Vec<PyReadonlyArray1<'py, bool>>>().ok())",
        input_name
      )?;
      writeln!(code, "      {{")?;
      writeln!(
        code,
        "        let {} = {}.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();",
        r_name, r_name
      )?;
      writeln!(
        code,
        "        let {} = {}.iter().map(|x| x.as_array()).collect::<Vec<_>>();",
        input_name, input_name
      )?;
      writeln!(code, "        let mut _r = vec![];")?;
      writeln!(code, "        {}.into_par_iter()", r_name)?;
      writeln!(code, "          .zip({}.into_par_iter())", input_name)?;
      writeln!(code, "          .map(|(mut out, input)| {{")?;
      writeln!(code, "            let {} = out.as_slice_mut();", r_name)?;
      writeln!(code, "            let {} = input.as_slice();", input_name)?;
      writeln!(
        code,
        "            if let Some(({}, {})) = {}.zip({}) {{",
        r_name, input_name, r_name, input_name
      )?;
      writeln!(
        code,
        "              {}(&ctx, {}, {}{}).map_err(|e| e.into())",
        rust_func_name, r_name, input_name, args_f32
      )?;
      writeln!(code, "            }} else {{")?;
      writeln!(
        code,
        "              Err(PyValueError::new_err(\"invalid input\"))"
      )?;
      writeln!(code, "            }}")?;
      writeln!(code, "          }})")?;
      writeln!(code, "          .collect_into_vec(&mut _r);")?;
      writeln!(code, "        match _r.into_iter().find(|x| x.is_err()) {{")?;
      writeln!(code, "          Some(e) => e,")?;
      writeln!(code, "          None => Ok(()),")?;
      writeln!(code, "        }}")?;
      writeln!(code, "      }} else {{")?;
      writeln!(
        code,
        "        Err(PyValueError::new_err(\"invalid input\"))"
      )?;
      writeln!(code, "      }}")?;
      writeln!(code, "    }} else {{")?;
      writeln!(code, "      Err(PyValueError::new_err(\"invalid input\"))")?;
      writeln!(code, "    }}")?;
    } else {
      // Standard NumArray List
      // List f64
      writeln!(code, "      // check if each array is f64 array")?;
      writeln!(
        code,
        "      if let Some((mut {}, {})) = {}",
        r_name, input_name, r_name
      )?;
      writeln!(
        code,
        "        .extract::<Vec<PyReadwriteArray1<'py, f64>>>()"
      )?;
      writeln!(code, "        .ok()")?;
      writeln!(
        code,
        "        .zip({}.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok())",
        input_name
      )?;
      writeln!(code, "      {{")?;
      writeln!(
        code,
        "        let {} = {}.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();",
        r_name, r_name
      )?;
      writeln!(
        code,
        "        let {} = {}.iter().map(|x| x.as_array()).collect::<Vec<_>>();",
        input_name, input_name
      )?;
      writeln!(code, "        let mut _r = vec![];")?;
      writeln!(code, "        {}.into_par_iter()", r_name)?;
      writeln!(code, "          .zip({}.into_par_iter())", input_name)?;
      writeln!(code, "          .map(|(mut out, input)| {{")?; // Renaming locally to out/input to match template roughly or just use names
      writeln!(code, "            let {} = out.as_slice_mut();", r_name)?;
      writeln!(code, "            let {} = input.as_slice();", input_name)?;
      writeln!(
        code,
        "            if let Some(({}, {})) = {}.zip({}) {{",
        r_name, input_name, r_name, input_name
      )?;
      writeln!(
        code,
        "              {}(&ctx, {}, {}{}).map_err(|e| e.into())",
        rust_func_name, r_name, input_name, args_f64
      )?;
      writeln!(code, "            }} else {{")?;
      writeln!(
        code,
        "              Err(PyValueError::new_err(\"invalid input\"))"
      )?;
      writeln!(code, "            }}")?;
      writeln!(code, "          }})")?;
      writeln!(code, "          .collect_into_vec(&mut _r);")?;
      writeln!(code, "        match _r.into_iter().find(|x| x.is_err()) {{")?;
      writeln!(code, "          Some(e) => e,")?;
      writeln!(code, "          None => Ok(()),")?;
      writeln!(code, "        }}")?;

      // List f32
      writeln!(code, "      // check if each array is f32 array")?;
      writeln!(
        code,
        "      }} else if let Some((mut {}, {})) = {}",
        r_name, input_name, r_name
      )?;
      writeln!(
        code,
        "        .extract::<Vec<PyReadwriteArray1<'py, f32>>>()"
      )?;
      writeln!(code, "        .ok()")?;
      writeln!(
        code,
        "        .zip({}.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok())",
        input_name
      )?;
      writeln!(code, "      {{")?;
      writeln!(
        code,
        "        let {} = {}.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();",
        r_name, r_name
      )?;
      writeln!(
        code,
        "        let {} = {}.iter().map(|x| x.as_array()).collect::<Vec<_>>();",
        input_name, input_name
      )?;
      writeln!(code, "        let mut _r = vec![];")?;
      writeln!(code, "        {}.into_par_iter()", r_name)?;
      writeln!(code, "          .zip({}.into_par_iter())", input_name)?;
      writeln!(code, "          .map(|(mut out, input)| {{")?;
      writeln!(code, "            let {} = out.as_slice_mut();", r_name)?;
      writeln!(code, "            let {} = input.as_slice();", input_name)?;
      writeln!(
        code,
        "            if let Some(({}, {})) = {}.zip({}) {{",
        r_name, input_name, r_name, input_name
      )?;
      writeln!(
        code,
        "              {}(&ctx, {}, {}{}).map_err(|e| e.into())",
        rust_func_name, r_name, input_name, args_f32
      )?;
      writeln!(code, "            }} else {{")?;
      writeln!(
        code,
        "              Err(PyValueError::new_err(\"invalid input\"))"
      )?;
      writeln!(code, "            }}")?;
      writeln!(code, "          }})")?;
      writeln!(code, "          .collect_into_vec(&mut _r);")?;
      writeln!(code, "        match _r.into_iter().find(|x| x.is_err()) {{")?;
      writeln!(code, "          Some(e) => e,")?;
      writeln!(code, "          None => Ok(()),")?;
      writeln!(code, "        }}")?;

      writeln!(code, "      }} else {{")?;
      writeln!(
        code,
        "        Err(PyValueError::new_err(\"invalid input\"))"
      )?;
      writeln!(code, "      }}")?;

      writeln!(code, "    }} else {{")?;
      writeln!(code, "      Err(PyValueError::new_err(\"invalid input\"))")?;
      writeln!(code, "    }}")?;
    }

    writeln!(code, "  }}")?;
  }

  // Generate register_functions
  writeln!(code, "")?;
  writeln!(
    code,
    "pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {{"
  )?;
  for func in functions {
    if func.name == "ref" {
      writeln!(
        code,
        "  m.add_function(wrap_pyfunction!(r#{}, m)?)?;",
        func.name
      )?;
    } else {
      writeln!(
        code,
        "  m.add_function(wrap_pyfunction!({}, m)?)?;",
        func.name
      )?;
    }
  }
  writeln!(code, "  Ok(())")?;
  writeln!(code, "}}")?;

  file.write_all(code.as_bytes())?;

  Ok(())
}

fn build_algo_py(functions: &[TaFunc]) -> Result<()> {
  let out_file = "python/alpha/algo/algo_gen.py";
  let mut file = fs::File::create(out_file)?;

  writeln!(file, "# THIS FILE IS AUTO-GENERATED, DO NOT EDIT")?;
  writeln!(file, "")?;
  writeln!(file, "import numpy as np")?;
  writeln!(file, "from . import _algo")?;
  writeln!(file, "")?;

  for func in functions {
    // Generate docstring
    let doc = func
      .doc
      .lines()
      .map(|l| format!("  {}", l))
      .collect::<Vec<_>>()
      .join("\n");

    let py_func_name = func.name.to_uppercase();
    let rust_func_name = &func.name;

    // Filter params
    let arrays: Vec<(&String, &TaType)> = func
      .params
      .iter()
      .filter_map(|p| match p {
        TaType::NumArray(n) => Some((n, p)),
        TaType::BoolArray(n) => Some((n, p)),
        _ => None,
      })
      .collect();

    if arrays.len() < 2 {
      // Need at least 2 arrays (output and input)
      continue;
    }


    if arrays.len() == 4 {
      let r_name = arrays[0].0;
      let a_name = arrays[1].0;
      let b_name = arrays[2].0;
      let c_name = arrays[3].0;

      let mut py_params = vec![
        format!("{}: np.ndarray | list[np.ndarray]", a_name),
        format!("{}: np.ndarray | list[np.ndarray]", b_name),
        format!("{}: np.ndarray | list[np.ndarray]", c_name),
      ];
      let mut call_params = vec![r_name.as_str(), a_name.as_str(), b_name.as_str(), c_name.as_str()];

      for param in &func.params {
        match param {
          TaType::Num(n) => {
            py_params.push(format!("{}: float", n));
            call_params.push(n);
          }
          TaType::Int(n) => {
            py_params.push(format!("{}: int", n));
            call_params.push(n);
          }
          _ => {}
        }
      }

      writeln!(file, "def {}(", py_func_name)?;
      writeln!(file, "  {}", py_params.join(", "))?;
      writeln!(file, ") -> np.ndarray | list[np.ndarray]:")?;
      if !doc.trim().is_empty() {
        writeln!(file, "  \"\"\"")?;
        writeln!(file, "{}", doc)?;
        writeln!(file, "  \"\"\"")?;
      }

      writeln!(
        file,
        "  if isinstance({}, list) and isinstance({}, list) and isinstance({}, list):",
        a_name, b_name, c_name
      )?;

      writeln!(
        file,
        "    {} = [np.empty_like(x) for x in {}]",
        r_name, a_name
      )?;

      for name in [a_name, b_name, c_name] {
        writeln!(
          file,
          "    {} = [x.astype(float) for x in {}]",
          name, name
        )?;
      }

      writeln!(
        file,
        "    _algo.{}({})",
        rust_func_name,
        call_params.join(", ")
      )?;
      writeln!(file, "    return {}", r_name)?;

      writeln!(file, "  else:")?;
      writeln!(file, "    {} = np.empty_like({})", r_name, a_name)?;
      for name in [a_name, b_name, c_name] {
        writeln!(file, "    {} = {}.astype(float)", name, name)?;
      }

      writeln!(
        file,
        "    _algo.{}({})",
        rust_func_name,
        call_params.join(", ")
      )?;
      writeln!(file, "    return {}", r_name)?;
      writeln!(file, "")?;
      continue;
    }

    if arrays.len() == 3 {
      // Handle 3 arrays: r, a, b
      let r_name = arrays[0].0;
      let a_name = arrays[1].0;
      let b_name = arrays[2].0;

      let r_is_bool = matches!(arrays[0].1, TaType::BoolArray(_));
      #[allow(unused)]
      let a_is_bool = matches!(arrays[1].1, TaType::BoolArray(_));
      #[allow(unused)]
      let b_is_bool = matches!(arrays[2].1, TaType::BoolArray(_));

      let mut py_params = vec![
        format!("{}: np.ndarray | list[np.ndarray]", a_name),
        format!("{}: np.ndarray | list[np.ndarray]", b_name),
      ];
      let mut call_params = vec![r_name.as_str(), a_name.as_str(), b_name.as_str()];

      for param in &func.params {
        match param {
          TaType::Num(n) => {
            py_params.push(format!("{}: float", n));
            call_params.push(n);
          }
          TaType::Int(n) => {
            py_params.push(format!("{}: int", n));
            call_params.push(n);
          }
          _ => {}
        }
      }

      writeln!(file, "def {}(", py_func_name)?;
      writeln!(file, "  {}", py_params.join(", "))?;
      writeln!(file, ") -> np.ndarray | list[np.ndarray]:")?;
      if !doc.trim().is_empty() {
        writeln!(file, "  \"\"\"")?;
        writeln!(file, "{}", doc)?;
        writeln!(file, "  \"\"\"")?;
      }

      writeln!(
        file,
        "  if isinstance({}, list) and isinstance({}, list):",
        a_name, b_name
      )?;

      // Output initialization
      let dtype = if r_is_bool { ", dtype=bool" } else { "" }; // Note: numpy default float usually, but strict bool output?
      // Wait, for CROSS, rust returns bool. Python numpy bool array.
      // If we want float 0/1, we should convert?
      // Let's stick to bool for now, user can cast.

      writeln!(
        file,
        "    {} = [np.empty_like(x{}) for x in {}]",
        r_name, dtype, a_name
      )?;

      // Inputs casting not strictly needed if already f64, but good for safety.
      // Assuming inputs are numeric.
      writeln!(
        file,
        "    {} = [x.astype(float) for x in {}]",
        a_name, a_name
      )?;
      writeln!(
        file,
        "    {} = [x.astype(float) for x in {}]",
        b_name, b_name
      )?;

      writeln!(
        file,
        "    _algo.{}({})",
        rust_func_name,
        call_params.join(", ")
      )?;
      writeln!(file, "    return {}", r_name)?;

      writeln!(file, "  else:")?;
      writeln!(file, "    {} = np.empty_like({}{})", r_name, a_name, dtype)?;
      writeln!(file, "    {} = {}.astype(float)", a_name, a_name)?;
      writeln!(file, "    {} = {}.astype(float)", b_name, b_name)?;

      writeln!(
        file,
        "    _algo.{}({})",
        rust_func_name,
        call_params.join(", ")
      )?;
      writeln!(file, "    return {}", r_name)?;
      writeln!(file, "")?;
      continue;
    }

    let r_name = arrays[0].0;
    let input_name = arrays[1].0;
    let input_is_bool = matches!(arrays[1].1, TaType::BoolArray(_));

    let mut py_params = vec![format!("{}: np.ndarray | list[np.ndarray]", input_name)];
    let mut call_params = vec![r_name.as_str(), input_name.as_str()];

    for param in &func.params {
      match param {
        TaType::Num(n) => {
          py_params.push(format!("{}: float", n));
          call_params.push(n);
        }
        TaType::Int(n) => {
          py_params.push(format!("{}: int", n));
          call_params.push(n);
        }
        _ => {}
      }
    }

    let py_params_str = py_params.join(", ");
    let call_params_str = call_params.join(", ");

    writeln!(file, "def {}(", py_func_name)?;
    writeln!(file, "  {}", py_params_str)?;
    writeln!(file, ") -> np.ndarray | list[np.ndarray]:")?;
    if !doc.trim().is_empty() {
      writeln!(file, "  \"\"\"")?;
      writeln!(file, "{}", doc)?;
      writeln!(file, "  \"\"\"")?;
    }
    writeln!(file, "  if isinstance({}, list):", input_name)?;
    if input_is_bool {
      writeln!(
        file,
        "    {} = [np.empty_like(x, dtype=float) for x in {}]",
        r_name, input_name
      )?;
      writeln!(
        file,
        "    {} = [x.astype(bool) for x in {}]",
        input_name, input_name
      )?;
    } else {
      writeln!(
        file,
        "    {} = [np.empty_like(x) for x in {}]",
        r_name, input_name
      )?;
    }
    writeln!(file, "    _algo.{}({})", rust_func_name, call_params_str)?;
    writeln!(file, "    return {}", r_name)?;
    writeln!(file, "  else:")?;
    if input_is_bool {
      writeln!(
        file,
        "    {} = np.empty_like({}, dtype=float)",
        r_name, input_name
      )?;
      writeln!(file, "    {} = {}.astype(bool)", input_name, input_name)?;
    } else {
      writeln!(file, "    {} = np.empty_like({})", r_name, input_name)?;
    }
    writeln!(file, "    _algo.{}({})", rust_func_name, call_params_str)?;
    writeln!(file, "    return {}", r_name)?;
    writeln!(file, "")?;
  }

  Ok(())
}

fn build_algo_md(functions: &Vec<TaFunc>) -> Result<()> {
  let mut file = fs::File::create("python/alpha/algo.md")?;
  writeln!(file, "List of available functions with python type hints:")?;
  writeln!(file, "")?;
  writeln!(
    file,
    "the `np.ndarray` is `ndarray` type in `numpy` package"
  )?;
  writeln!(file, "")?;
  for func in functions {
    writeln!(
      file,
      "- {}({}): {}",
      func.name.to_uppercase(),
      func
        .params
        .iter()
        .skip(2)
        .map(|p| p.to_py_type_str())
        .collect::<Vec<_>>()
        .join(", "),
      func
        .doc
        .split('\n')
        .take_while(|line| !line.starts_with("Ref:"))
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
    )?;
  }
  Ok(())
}

fn main() -> Result<()> {
  let src_dir = "src/algo";
  let dir = fs::read_dir(src_dir)?;
  let mut functions = Vec::new();
  for entry in dir {
    let entry = entry?;
    if entry.file_type()?.is_dir() {
      continue;
    }
    let path = entry.path();
    if let Ok(a) = parse_ta_file(&path) {
      functions.extend(a.into_iter());
    }
  }
  functions.sort_by_key(|a| a.name.clone());

  build_algo_md(&functions)?;

  // skip ema, we will write it as template by hand
  functions.retain(|f| f.name != "ema");
  build_py_bindings(&functions)?;
  build_algo_py(&functions)?;

  Ok(())
}
