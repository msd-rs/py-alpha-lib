use crate::numarray::register_num_array;
use mlua::prelude::*;

#[test]
fn test_lua_load() {
  let lua = Lua::new();
  let chunk = lua.load(
    r#"
    function test()
      return 1
    end
  "#,
  );
  chunk.exec().unwrap();
  let test = lua.globals().get::<LuaFunction>("test").unwrap();
  let result = test.call::<i32>(()).unwrap();
  assert_eq!(result, 1);
}

#[test]
fn test_numarray() {
  let lua = Lua::new();
  register_num_array(&lua).unwrap();
  let chunk = lua.load(
    r#"
    local a = numarray_new('10', 1)
    local b = numarray_range(10)
    local c = numarray_new({1,2,3,4,5})
    print(a)
    print(b)
    print(a+b)
    print(a+1)
    print(a+2.0)
    print(a-0.5)
    print(a* 2)
    print((a*2)^2)
    print(c)
    print(a:le(b))
    print(a:lt(b))
    print(a:eq(b))
    print(a:ne(b))
    print(a:gt(b))
    print(a:ge(b))
    print(b:ge(5) & b:lt(3))
  "#,
  );
  chunk.exec().unwrap();
}
