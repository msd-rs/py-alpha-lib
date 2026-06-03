use anyhow::Result;
use std::collections::HashMap;

use crate::{
  numarray::{BoolArray, NumArray, register_num_array},
  to_lua,
};
use mlua::prelude::*;

/// represents an line after executed
#[derive(Debug, Clone, Default)]
pub struct Line {
  /// which type of the line
  pub kind: String,
  /// name of the line
  pub name: String,
  /// data of the line
  pub data: Vec<f64>,
  /// color of the line
  pub color: Option<String>,
  /// for some `kind`, which only draw when `when` is true
  pub when: Option<Vec<bool>>,
  /// extra data for some `kind`
  pub ext_data: Option<Vec<u8>>,
}

impl FromLua for Line {
  fn from_lua(value: LuaValue, _: &Lua) -> LuaResult<Self> {
    match value {
      LuaValue::Table(t) => {
        let kind = t.get::<String>("kind")?;
        let name = t.get::<String>("name")?;
        let data = t.get::<LuaAnyUserData>("data")?;
        match kind.as_str() {
          "icon" => {
            let when = t.get::<LuaAnyUserData>("when")?;
            let ext_data = t.get::<u32>("ext_data")?;
            Ok(Line {
              kind,
              name,
              data: data.take::<NumArray>()?.into(),
              when: Some(when.take::<BoolArray>()?.into()),
              ext_data: Some(ext_data.to_le_bytes().to_vec()),
              ..Default::default()
            })
          }
          "text" => {
            let when = t.get::<LuaAnyUserData>("when")?;
            let ext_data = t.get::<LuaValue>("ext_data")?;
            let ext_data = match ext_data {
              LuaValue::Number(num) => num.to_le_bytes().to_vec(),
              LuaValue::String(s) => s.as_bytes().to_vec(),
              _ => vec![],
            };
            Ok(Line {
              kind,
              name,
              data: data.take::<NumArray>()?.into(),
              when: Some(when.take::<BoolArray>()?.into()),
              ext_data: Some(ext_data),
              ..Default::default()
            })
          }
          _ => {
            let color = t.get("color").unwrap_or_default();
            Ok(Line {
              kind,
              name,
              data: data.take::<NumArray>()?.into(),
              color,
              ..Default::default()
            })
          }
        }
      }
      _ => Err(LuaError::runtime("Expected a table for Line")),
    }
  }
}

#[derive(Debug)]
pub struct LuaExecutor {
  lua: Lua,
}

impl LuaExecutor {
  pub fn new() -> Result<Self> {
    let executor = Self { lua: Lua::new() };
    executor.init_runtime()?;
    Ok(executor)
  }

  fn init_runtime(&self) -> Result<()> {
    register_num_array(&self.lua)?;
    self
      .lua
      .globals()
      .set("ctx", alpha_algo::Context::default())?;
    crate::ta_binding::register_ta_funcs(&self.lua)?;
    crate::ta_binding::register_draw_functions(&self.lua)?;
    Ok(())
  }

  pub fn execute_mlang(
    &self,
    code: &str,
    datas: HashMap<String, NumArray>,
    params: HashMap<String, f64>,
  ) -> Result<Vec<Line>> {
    let code = to_lua(code)?;
    self.execute(&code, datas, params)
  }

  pub fn execute(
    &self,
    code: &str,
    datas: HashMap<String, NumArray>,
    params: HashMap<String, f64>,
  ) -> Result<Vec<Line>> {
    self.lua.load(code).exec()?;

    let compute_fn: LuaFunction = self.lua.globals().get("compute")?;
    compute_fn
      .call::<Vec<Line>>((datas, params))
      .map_err(|e| e.into())
  }
}
