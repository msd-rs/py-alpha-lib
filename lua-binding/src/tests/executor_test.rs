use anyhow::Result;
use std::collections::HashMap;

use crate::{Line, LuaExecutor};

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
  datas.insert("data1".to_string(), vec![1.0, 2.0, 3.0]);
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
