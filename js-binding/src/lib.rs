mod join;
mod utils;

use std::collections::HashMap;

use alpha_algo::{ta_bw_split, ta_bw_split_factor, ta_fw_split, ta_fw_split_factor};
//use lua_alpha_lib::{Line, LuaExecutor, NumArray};
use mlang::{Context, Line, MRuntime, NumArray};
use wasm_bindgen::prelude::*;

use crate::utils::set_panic_hook;
use join::join_asof_impl;

/// the line result from `execute`
#[wasm_bindgen(getter_with_clone)]
#[derive(Debug, Clone)]
pub struct JSLine {
  /// which type of the line
  pub kind: String,
  /// name of the line
  pub name: String,
  /// data of the line
  pub data: Box<[f64]>,
  /// color of the line
  pub color: Option<String>,
  /// for some `kind`, which only draw when `when` is true
  pub when: Option<Box<[u8]>>,
  /// extra data for some `kind`
  pub ext_data: Option<JsValue>,
}

impl From<Line> for JSLine {
  fn from(line: Line) -> Self {
    let kind = line.kind.as_str();
    let ext_data = match kind {
      "icon" => line.ext_data.and_then(|v| {
        if v.len() == 4 {
          let mut b = [0u8; 4];
          b.copy_from_slice(&v);
          Some(JsValue::from(u32::from_le_bytes(b)))
        } else {
          None
        }
      }),
      "text" => line
        .ext_data
        .map(|v| JsValue::from(String::from_utf8(v).unwrap())),
      _ => None,
    };
    Self {
      kind: line.kind,
      name: line.name,
      data: Box::from(line.data),
      color: line.color,
      when: line
        .when
        .map(|v| v.into_iter().map(|b| if b { 1u8 } else { 0u8 }).collect()),
      ext_data,
    }
  }
}

/// a named array used in the script.
#[wasm_bindgen(getter_with_clone)]
#[derive(Debug, Clone)]
pub struct NamedArray {
  pub name: String,
  pub data: Box<[f64]>,
}

#[wasm_bindgen]
impl NamedArray {
  #[wasm_bindgen(constructor)]
  pub fn new(name: String, data: Box<[f64]>) -> Self {
    Self { name, data }
  }
}

#[wasm_bindgen(getter_with_clone)]
pub struct NamedValue {
  pub name: String,
  pub value: f64,
}

#[wasm_bindgen]
impl NamedValue {
  #[wasm_bindgen(constructor)]
  pub fn new(name: String, value: f64) -> Self {
    Self { name, value }
  }
}

/// execute a mlang script.
///
/// data contains named array in the script.
/// params contains named values in the script.
/// the function returns lines which will be drawn on the canvas.
#[wasm_bindgen]
pub fn execute(
  code: &str,
  data: Vec<NamedArray>,
  params: Vec<NamedValue>,
) -> Result<Vec<JSLine>, JsError> {
  set_panic_hook();
  let data = data
    .into_iter()
    .map(|NamedArray { name, data }| (name, NumArray::from(data)))
    .collect::<HashMap<String, NumArray>>();
  let params = params
    .into_iter()
    .map(|NamedValue { name, value }| (name, value))
    .collect::<HashMap<String, f64>>();
  let lines = MRuntime::new(Context::default())
    .execute(code, &data, &params)
    .map_err(|e| JsError::new(&e.to_string()))?;

  Ok(lines.into_iter().map(JSLine::from).collect())
}

/// align d2 to d1, fill not found values by `method`.
/// `method` is one of: 0 zero, 1 nan, -2 backward fill, 2 forward fill
/// d2, v2 should have same size
/// returns v2 aligned to d1, have same length as d1
#[wasm_bindgen]
pub fn join_asof(d1: &[f64], d2: &[f64], v2: &[f64], method: i32) -> Result<Box<[f64]>, JsError> {
  join_asof_impl(d1, d2, v2, method).map_err(|e| JsError::new(&e))
}

/// do split for price with `method`
/// method can be
/// - 1 forward split
/// - 2 forward split factor
/// - -1 backward split
/// - -2 backward split factor
///
/// `dividend`, `transfer_shares`, `right_shares`, `right_price` should have same length as `price`, so call
/// `join_asof` with method 0 first to align it
#[wasm_bindgen]
pub fn apply_split(
  method: i32,
  price: &[f64],
  dividend: &[f64],
  transfer_shares: &[f64],
  right_shares: &[f64],
  right_price: &[f64],
) -> Result<Box<[f64]>, JsError> {
  let ctx = Context::default();
  let mut r = vec![0.0; price.len()];

  let e = match method {
    1 => ta_fw_split(
      &ctx,
      r.as_mut_slice(),
      price,
      dividend,
      transfer_shares,
      right_shares,
      right_price,
    ),
    2 => ta_fw_split_factor(
      &ctx,
      &mut r,
      price,
      dividend,
      transfer_shares,
      right_shares,
      right_price,
    ),
    -1 => ta_bw_split(
      &ctx,
      &mut r,
      price,
      dividend,
      transfer_shares,
      right_shares,
      right_price,
    ),
    -2 => ta_bw_split_factor(
      &ctx,
      &mut r,
      price,
      dividend,
      transfer_shares,
      right_shares,
      right_price,
    ),
    _ => Err(alpha_algo::Error::InvalidParameter(format!(
      "Invalid method {}",
      method
    ))),
  };

  if let Err(e) = e {
    return Err(JsError::new(&e.to_string()));
  }
  Ok(r.into_boxed_slice())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_join_asof_forward_fill() {
    let d1 = vec![10.0, 20.0, 30.0, 40.0];
    let d2 = vec![15.0, 30.0, 35.0];
    let v2 = vec![100.0, 200.0, 300.0];

    let res = join_asof_impl(&d1, &d2, &v2, 2).unwrap();
    assert!(res[0].is_nan());
    assert_eq!(res[1], 100.0);
    assert_eq!(res[2], 200.0);
    assert_eq!(res[3], 300.0);
  }

  #[test]
  fn test_join_asof_backward_fill() {
    let d1 = vec![10.0, 20.0, 30.0, 40.0];
    let d2 = vec![15.0, 30.0, 35.0];
    let v2 = vec![100.0, 200.0, 300.0];

    let res = join_asof_impl(&d1, &d2, &v2, -2).unwrap();
    assert_eq!(res[0], 100.0);
    assert_eq!(res[1], 200.0);
    assert_eq!(res[2], 200.0);
    assert!(res[3].is_nan());
  }

  #[test]
  fn test_join_asof_zero_fill() {
    let d1 = vec![10.0, 20.0, 30.0, 40.0];
    let d2 = vec![15.0, 30.0, 35.0];
    let v2 = vec![100.0, 200.0, 300.0];

    let res = join_asof_impl(&d1, &d2, &v2, 0).unwrap();
    assert_eq!(res[0], 0.0);
    assert_eq!(res[1], 0.0);
    assert_eq!(res[2], 200.0);
    assert_eq!(res[3], 0.0);
  }

  #[test]
  fn test_join_asof_nan_fill() {
    let d1 = vec![10.0, 20.0, 30.0, 40.0];
    let d2 = vec![15.0, 30.0, 35.0];
    let v2 = vec![100.0, 200.0, 300.0];

    let res = join_asof_impl(&d1, &d2, &v2, 1).unwrap();
    assert!(res[0].is_nan());
    assert!(res[1].is_nan());
    assert_eq!(res[2], 200.0);
    assert!(res[3].is_nan());
  }

  #[test]
  fn test_join_asof_validation() {
    let d1 = vec![10.0, 20.0];
    let d2 = vec![10.0, 20.0];
    let v2 = vec![100.0];

    assert!(join_asof_impl(&d1, &d2, &v2, 0).is_err());
    assert!(join_asof_impl(&d1, &d2[..1], &v2, 999).is_err());
  }
}
