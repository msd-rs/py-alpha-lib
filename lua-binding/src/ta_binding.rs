use crate::numarray::{BoolArray, NumArray};
use alpha_algo::*;
use mlua::prelude::*;

fn ctx(lua: &Lua) -> LuaUserDataRef<Context> {
  lua
    .globals()
    .get::<LuaAnyUserData>("ctx")
    .and_then(|ctx| ctx.borrow())
    .unwrap()
}

include!("algo_bindings.rs");

pub fn register_draw_functions(lua: &Lua) -> LuaResult<()> {
  lua.globals().set(
    "DRAWICON",
    lua.create_function(|lua, (when, pos, icon): (BoolArray, NumArray, u32)| {
      if when.len() != pos.len() {
        return Err(mlua::Error::runtime(
          "when and pos must have the same length",
        ));
      }
      let t = lua.create_table()?;
      t.set("kind", "icon")?;
      t.set("name", "icon")?;
      t.set("data", pos)?;
      t.set("when", when)?;
      t.set("ext_data", icon)?;
      Ok(t)
    })?,
  )?;
  lua.globals().set(
    "DRAWTEXT",
    lua.create_function(|lua, (when, pos, text): (BoolArray, NumArray, String)| {
      if when.len() != pos.len() {
        return Err(mlua::Error::runtime(
          "when and pos must have the same length",
        ));
      }
      let t = lua.create_table()?;
      t.set("kind", "text")?;
      t.set("name", "text")?;
      t.set("data", pos)?;
      t.set("when", when)?;
      t.set("ext_data", text)?;
      Ok(t)
    })?,
  )?;
  Ok(())
}
