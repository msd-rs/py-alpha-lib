use anyhow::Result;
use std::collections::HashMap;

use crate::{LuaExecutor, numarray::NumArray};

#[test]
fn test_execute() -> Result<()> {
  let code = r#"
  function compute(datas, params)
    local lines = {}
    for k, v in pairs(datas) do
      table.insert(lines, {
        kind = "line",
        name = k,
        data = v,
        color = "red",
      })
    end
    return lines
  end
"#;

  let executor = LuaExecutor::new()?;
  let mut datas = HashMap::new();
  datas.insert("data1".to_string(), NumArray::from(vec![1.0, 2.0, 3.0]));
  let params = HashMap::new();
  let lines = executor.execute(code, datas, params)?;

  assert_eq!(lines.len(), 1);
  assert_eq!(lines[0].name, "data1");
  assert_eq!(lines[0].data, vec![1.0, 2.0, 3.0]);
  assert_eq!(lines[0].color, Some("red".to_string()));
  assert_eq!(lines[0].when, None);
  assert_eq!(lines[0].ext_data, None);
  Ok(())
}

#[test]
fn test_ta_execute() -> Result<()> {
  let code = r#"
    MA5 : MA(C, 3);
  "#;
  let lua_code = crate::to_lua(code)?;
  let executor = LuaExecutor::new()?;
  let mut datas = HashMap::new();
  datas.insert(
    "C".to_string(),
    NumArray::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
  );
  let params = HashMap::new();
  let lines = executor.execute(&lua_code, datas, params)?;
  assert_eq!(lines.len(), 1);
  assert_eq!(lines[0].name, "ma5");
  assert_eq!(lines[0].data.len(), 5);
  Ok(())
}

#[test]
fn test_execute_draw() -> Result<()> {
  let code = r#"
    DRAWICON(C>MA(C,5), C, 1);
  "#;
  let executor = LuaExecutor::new()?;
  let mut datas = HashMap::new();
  datas.insert(
    "C".to_string(),
    NumArray::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
  );
  let params = HashMap::new();
  let lines = executor.execute_mlang(code, datas, params)?;
  assert_eq!(lines.len(), 1);
  assert_eq!(lines[0].name, "icon");
  assert_eq!(lines[0].data.len(), 5);
  assert_eq!(lines[0].kind, "icon");
  let icon = match lines[0].ext_data.as_ref() {
    Some(v) => u32::from_le_bytes(v.as_slice().try_into()?),
    None => 0,
  };
  assert_eq!(icon, 1);

  Ok(())
}
