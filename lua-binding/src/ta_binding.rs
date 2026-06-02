use std::ops::Deref;

use crate::numarray::NumArray;
use alpha_algo::*;
use mlua::prelude::*;

fn ctx(lua: &Lua) -> LuaUserDataRef<Context> {
  lua
    .globals()
    .get::<LuaAnyUserData>("ctx")
    .and_then(|ctx| ctx.borrow())
    .unwrap()
}


/**
 * template of register all ta functions
pub fn register_ta_funcs(lua: &Lua) -> LuaResult<()> {
  lua.globals().set(
    "EMA", // upper case ta function name without `ta_` prefix
    lua.create_function(|lua, (data, period): (NumArray, usize)| {
      let mut r = vec![0.0; data.len()];
      ta_ema::<f64>(&ctx(lua), &mut r, &data, period)?;
      Ok(NumArray::from(r))
    })?,
  )
}
*/