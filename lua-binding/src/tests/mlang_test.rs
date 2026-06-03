use crate::to_lua;

#[test]
fn test_to_lua_compiler() {
  let code = r#"
    MA5 := MA(C,5);
    MA10 : MA(C,P1), RED;
    UP:=CROSS(MA5,MA10);
    DRAWICON(UP,LOW,1);
  "#;
  let lua_code = to_lua(code).unwrap();
  assert!(lua_code.contains("local ma5 = MA(D.C, 5)"));
  assert!(lua_code.contains("local ma10 = MA(D.C, P.P1)"));
  assert!(
    lua_code
      .contains(r#"table.insert(lines, {kind="line", name="ma10", data = ma10, color = "red"})"#)
  );
  assert!(lua_code.contains("local up = CROSS(ma5, ma10)"));
  assert!(lua_code.contains("table.insert(lines, DRAWICON(up, D.L, 1))"));

  // Check comparison operations
  let cmp_code = "UP := A == B;";
  let cmp_lua = to_lua(cmp_code).unwrap();
  assert!(cmp_lua.contains("local up = D.A:eq(D.B)"));

  let cmp_code_rev = "UP := 1 == A;";
  let cmp_lua_rev = to_lua(cmp_code_rev).unwrap();
  assert!(cmp_lua_rev.contains("local up = D.A:eq(1)"));

  let cmp_code_lt = "UP := 1 < A;";
  let cmp_lua_lt = to_lua(cmp_code_lt).unwrap();
  assert!(cmp_lua_lt.contains("local up = D.A:gt(1)"));

  // Check arithmetic swapping
  let arith_swap = "UP := 1 + A;";
  let arith_swap_lua = to_lua(arith_swap).unwrap();
  assert!(arith_swap_lua.contains("local up = (D.A + 1)"));

  let arith_noswap = "UP := A + 1;";
  let arith_noswap_lua = to_lua(arith_noswap).unwrap();
  assert!(arith_noswap_lua.contains("local up = (D.A + 1)"));

  let arith_const = "UP := 1 + 2;";
  let arith_const_lua = to_lua(arith_const).unwrap();
  assert!(arith_const_lua.contains("local up = (1 + 2)"));

  // Check logical operators
  let log_arr = "UP := A && B;";
  let log_arr_lua = to_lua(log_arr).unwrap();
  assert!(log_arr_lua.contains("local up = (D.A & D.B)"));

  let log_const = "UP := 1 && 0;";
  let log_const_lua = to_lua(log_const).unwrap();
  assert!(log_const_lua.contains("local up = (1 and 0)"));

  // Check defined variables (local variables)
  let local_code = r#"
    A := 1;
    B := 2;
    UP := A == B;
  "#;
  let local_lua = to_lua(local_code).unwrap();
  assert!(local_lua.contains("local a = 1"));
  assert!(local_lua.contains("local b = 2"));
  assert!(local_lua.contains("local up = a:eq(b)"));
}
