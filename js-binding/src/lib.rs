mod utils;

use std::{collections::HashMap, rc::Rc};

//use lua_alpha_lib::{Line, LuaExecutor, NumArray};
use wasm_bindgen::prelude::*;

/*
#[wasm_bindgen]
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
      when: line.when.map(|v| Box::from(v)),
      ext_data,
    }
  }
}

#[wasm_bindgen]
pub struct MLang {
  executor: LuaExecutor,
}

#[wasm_bindgen]
impl MLang {
  #[wasm_bindgen(constructor)]
  pub fn new() -> Result<Self, JsError> {
    Ok(Self {
      executor: LuaExecutor::new().map_err(|e| JsError::new(&e.to_string()))?,
    })
  }

  #[wasm_bindgen]
  pub fn execute(
    &self,
    code: &str,
    data: HashMap<String, Box<[f64]>>,
    params: HashMap<String, f64>,
  ) -> Result<Vec<LineInfo>, JsError> {
    let data = data
      .into_iter()
      .map(|(k, v)| (k, NumArray { data: Rc::from(v) }))
      .collect::<HashMap<String, NumArray>>();

    let lines = self
      .executor
      .execute_mlang(code, datas, params)
      .map_err(|e| JsError::new(&e.to_string()))?;

    Ok(lines.into_iter().map(JSLine::from).collect())
  }
}
*/
