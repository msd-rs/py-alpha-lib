mod utils;

use std::{collections::HashMap, rc::Rc};

//use lua_alpha_lib::{Line, LuaExecutor, NumArray};
use mlang::{Context, Line, MRuntime, NumArray};
use wasm_bindgen::prelude::*;

use crate::utils::set_panic_hook;

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
