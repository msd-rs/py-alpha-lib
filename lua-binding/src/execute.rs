use anyhow::Result;
use std::collections::HashMap;

use crate::{
  numarray::{BoolArray, NumArray, register_num_array},
  to_lua,
};
use mlua::prelude::*;

/// represents an line after executed
#[derive(Debug, Clone)]
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
        let kind = t.get("kind")?;
        let name = t.get("name")?;
        let data = t.get::<LuaAnyUserData>("data")?;
        let color = t.get("color")?;
        let when = t.get::<Option<LuaAnyUserData>>("when")?;
        let ext_data = t.get("ext_data")?;
        Ok(Line {
          kind,
          name,
          data: data.take::<NumArray>()?.into(),
          color,
          when: when
            .and_then(|w| w.take::<BoolArray>().ok())
            .map(|w| w.into()),
          ext_data,
        })
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
    Ok(())
  }

  pub fn execute_mlang(
    &self,
    code: &str,
    datas: HashMap<String, Vec<f64>>,
    params: HashMap<String, f64>,
  ) -> Result<Vec<Line>> {
    let code = to_lua(code)?;
    self.execute(&code, datas, params)
  }

  pub fn execute(
    &self,
    code: &str,
    datas: HashMap<String, Vec<f64>>,
    params: HashMap<String, f64>,
  ) -> Result<Vec<Line>> {
    self.lua.load(code).exec()?;

    let compute_fn: LuaFunction = self.lua.globals().get("compute")?;
    let datas = datas
      .into_iter()
      .map(|(k, v)| (k, NumArray::from(v)))
      .collect::<HashMap<_, _>>();
    compute_fn
      .call::<Vec<Line>>((datas, params))
      .map_err(|e| e.into())
  }
}
