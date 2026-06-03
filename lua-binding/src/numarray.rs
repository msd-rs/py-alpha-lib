use mlua::prelude::*;
use std::ops::{Add, BitAnd, BitOr, Deref, Div, Mul, Rem, Sub};
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct NumArray {
  pub data: Rc<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct BoolArray {
  pub data: Rc<Vec<bool>>,
}

impl Into<Vec<f64>> for NumArray {
  fn into(self) -> Vec<f64> {
    match Rc::try_unwrap(self.data) {
      Ok(data) => data,
      Err(data) => data.to_vec(),
    }
  }
}

impl Into<Vec<bool>> for BoolArray {
  fn into(self) -> Vec<bool> {
    match Rc::try_unwrap(self.data) {
      Ok(data) => data,
      Err(data) => data.to_vec(),
    }
  }
}

impl From<Vec<f64>> for NumArray {
  fn from(data: Vec<f64>) -> Self {
    NumArray {
      data: Rc::new(data),
    }
  }
}

impl From<Vec<bool>> for BoolArray {
  fn from(data: Vec<bool>) -> Self {
    BoolArray {
      data: Rc::new(data),
    }
  }
}

impl Deref for NumArray {
  type Target = [f64];

  fn deref(&self) -> &Self::Target {
    self.data.as_slice()
  }
}

impl Deref for BoolArray {
  type Target = [bool];

  fn deref(&self) -> &Self::Target {
    self.data.as_slice()
  }
}

impl FromLua for NumArray {
  fn from_lua(value: mlua::Value, _lua: &Lua) -> LuaResult<Self> {
    match value {
      LuaValue::UserData(ud) => ud.borrow::<NumArray>().map(|arr| arr.clone()),
      _ => Err(LuaError::runtime("Unsupported operand type")),
    }
  }
}

impl FromLua for BoolArray {
  fn from_lua(value: mlua::Value, _lua: &Lua) -> LuaResult<Self> {
    match value {
      LuaValue::UserData(ud) => ud.borrow::<BoolArray>().map(|arr| arr.clone()),
      _ => Err(LuaError::runtime("Unsupported operand type")),
    }
  }
}

macro_rules! binary_op {
  ($name:ident, $op:ident, $input_type:ident, $return_type:ident, $el_type:ident) => {
    fn $name(lua: &Lua, a: &$input_type, b: LuaValue) -> LuaResult<$return_type> {
      match (b) {
        LuaValue::Table(t) => {
          let tlen = t.len()? as usize;
          if tlen != a.data.len() {
            return Err(LuaError::runtime(format!(
              "Mismatched array length, left: {}, right: {}",
              a.data.len(),
              tlen
            )));
          }

          let mut data = Vec::with_capacity(tlen);
          for ((d, l), r) in data
            .iter_mut()
            .zip(a.data.iter())
            .zip(t.sequence_values::<$el_type>())
          {
            *d = l.$op(&(r?));
          }
          Ok($return_type::from(data))
        }
        LuaValue::UserData(other) => {
          if let Ok(other) = other.borrow::<$input_type>() {
            if a.data.len() != other.data.len() {
              return Err(LuaError::runtime(format!(
                "Mismatched array length, left: {}, right: {}",
                a.data.len(),
                other.data.len()
              )));
            }
            Ok($return_type::from(
              a.data
                .iter()
                .zip(other.data.iter())
                .map(|(x, y)| x.$op(y))
                .collect::<Vec<_>>(),
            ))
          } else {
            Err(LuaError::runtime("Unsupported operand type"))
          }
        }
        _ => {
          if let Ok(x) = lua.convert::<$el_type>(b) {
            return Ok($return_type::from(
              a.data.iter().map(|y| y.$op(&x)).collect::<Vec<_>>(),
            ));
          } else {
            return Err(LuaError::runtime("Unsupported operand type"));
          }
        }
      }
    }
  };
}

trait FloatExt {
  fn powf_ref(self, n: &f64) -> f64;
  fn eq_f(self, other: &f64) -> bool;
  fn ne_f(self, other: &f64) -> bool;
  fn lt_f(self, other: &f64) -> bool;
  fn le_f(self, other: &f64) -> bool;
  fn gt_f(self, other: &f64) -> bool;
  fn ge_f(self, other: &f64) -> bool;
}

const FLOAT_EPSILON: f64 = 1e-9;
impl FloatExt for f64 {
  fn powf_ref(self, n: &f64) -> f64 {
    self.powf(*n)
  }
  fn eq_f(self, other: &f64) -> bool {
    (self.sub(other)).abs() < FLOAT_EPSILON
  }
  fn ne_f(self, other: &f64) -> bool {
    (self.sub(other)).abs() >= FLOAT_EPSILON
  }
  fn lt_f(self, other: &f64) -> bool {
    self.sub(other) < -FLOAT_EPSILON
  }
  fn le_f(self, other: &f64) -> bool {
    self.sub(other) <= FLOAT_EPSILON
  }
  fn gt_f(self, other: &f64) -> bool {
    self.sub(other) > FLOAT_EPSILON
  }
  fn ge_f(self, other: &f64) -> bool {
    self.sub(other) >= -FLOAT_EPSILON
  }
}

binary_op!(numarray_add, add, NumArray, NumArray, f64);
binary_op!(numarray_sub, sub, NumArray, NumArray, f64);
binary_op!(numarray_mul, mul, NumArray, NumArray, f64);
binary_op!(numarray_div, div, NumArray, NumArray, f64);
binary_op!(numarray_pow, powf_ref, NumArray, NumArray, f64);
binary_op!(numarray_rem, rem, NumArray, NumArray, f64);
binary_op!(numarray_lt, lt_f, NumArray, BoolArray, f64);
binary_op!(numarray_le, le_f, NumArray, BoolArray, f64);
binary_op!(numarray_eq, eq_f, NumArray, BoolArray, f64);
binary_op!(numarray_ne, ne_f, NumArray, BoolArray, f64);
binary_op!(numarray_gt, gt_f, NumArray, BoolArray, f64);
binary_op!(numarray_ge, ge_f, NumArray, BoolArray, f64);
binary_op!(boolarray_and, bitand, BoolArray, BoolArray, bool);
binary_op!(boolarray_or, bitor, BoolArray, BoolArray, bool);

impl LuaUserData for NumArray {
  fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
    methods.add_meta_method(LuaMetaMethod::ToString, |_: &Lua, a: &Self, _: ()| {
      Ok(format!("NumArray {:?}", a.data))
    });

    methods.add_meta_method(LuaMetaMethod::Add, numarray_add);
    methods.add_meta_method(LuaMetaMethod::Sub, numarray_sub);
    methods.add_meta_method(LuaMetaMethod::Mul, numarray_mul);
    methods.add_meta_method(LuaMetaMethod::Div, numarray_div);
    methods.add_meta_method(LuaMetaMethod::Pow, numarray_pow);
    methods.add_meta_method(LuaMetaMethod::Mod, numarray_rem);
    methods.add_method("le", numarray_le);
    methods.add_method("lt", numarray_lt);
    methods.add_method("eq", numarray_eq);
    methods.add_method("ne", numarray_ne);
    methods.add_method("gt", numarray_gt);
    methods.add_method("ge", numarray_ge);

    methods.add_meta_method(LuaMetaMethod::Unm, |_: &Lua, a: &Self, _: ()| {
      Ok(NumArray::from(
        a.data.iter().map(|x| -x).collect::<Vec<f64>>(),
      ))
    });
  }
}

impl LuaUserData for BoolArray {
  fn add_methods<M: LuaUserDataMethods<Self>>(methods: &mut M) {
    methods.add_meta_method(LuaMetaMethod::ToString, |_: &Lua, a: &Self, _: ()| {
      Ok(format!("BoolArray {:?}", a.data))
    });

    methods.add_meta_method(LuaMetaMethod::BAnd, boolarray_and);
    methods.add_meta_method(LuaMetaMethod::BOr, boolarray_or);
    methods.add_meta_method(LuaMetaMethod::Unm, |_: &Lua, a: &Self, _: ()| {
      Ok(BoolArray::from(
        a.data.iter().map(|x| !x).collect::<Vec<bool>>(),
      ))
    });
  }
}

pub fn register_num_array(lua: &Lua) -> LuaResult<()> {
  lua.globals().set(
    "numarray_new",
    lua.create_function(|lua, (v, v2): (LuaValue, LuaValue)| match v {
      LuaValue::Table(t) => {
        let mut data = Vec::with_capacity(t.len()? as usize);
        for v in t.sequence_values::<f64>() {
          data.push(v?);
        }
        Ok(lua.create_userdata(NumArray::from(data))?)
      }
      _ => {
        let n = lua.convert::<usize>(v)?;
        let num = lua.convert::<f64>(v2).unwrap_or(0.0);
        Ok(lua.create_userdata(NumArray::from(vec![num; n]))?)
      }
    })?,
  )?;

  lua.globals().set(
    "numarray_range",
    lua.create_function(|lua, (stop, start, step): (LuaValue, LuaValue, LuaValue)| {
      let stop = lua.convert::<f64>(stop)?;
      let start = lua.convert::<f64>(start).unwrap_or(0.0);
      let step = lua.convert::<f64>(step).unwrap_or(1.0);
      let mut data = Vec::new();
      let mut cur = start;
      while cur < stop {
        data.push(cur);
        cur += step;
      }
      Ok(lua.create_userdata(NumArray::from(data))?)
    })?,
  )?;

  Ok(())
}
