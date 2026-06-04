use anyhow::Result;
use std::collections::HashMap;

use crate::{MRuntime, MValue, NumArray};
use alpha_algo::Context;

#[test]
fn test_mvalue_arithmetic() -> Result<()> {
  let v1 = MValue::Num(5.0);
  let v2 = MValue::Num(3.0);

  // Scalar Arithmetic
  assert_eq!(
    match v1.add(&v2)? {
      MValue::Num(n) => n,
      _ => panic!(),
    },
    8.0
  );
  assert_eq!(
    match v1.sub(&v2)? {
      MValue::Num(n) => n,
      _ => panic!(),
    },
    2.0
  );
  assert_eq!(
    match v1.mul(&v2)? {
      MValue::Num(n) => n,
      _ => panic!(),
    },
    15.0
  );
  assert_eq!(
    match v1.div(&v2)? {
      MValue::Num(n) => n,
      _ => panic!(),
    },
    5.0 / 3.0
  );
  assert_eq!(
    match v1.rem(&v2)? {
      MValue::Num(n) => n,
      _ => panic!(),
    },
    2.0
  );
  assert_eq!(
    match v1.pow(&v2)? {
      MValue::Num(n) => n,
      _ => panic!(),
    },
    125.0
  );

  // Array Arithmetic
  let arr1 = MValue::NumArray(NumArray::from(vec![1.0, 2.0, 3.0]));
  let arr2 = MValue::NumArray(NumArray::from(vec![10.0, 20.0, 30.0]));

  let add_arr = arr1.add(&arr2)?;
  match &add_arr {
    MValue::NumArray(arr) => {
      assert_eq!(arr.len(), 3);
      assert_eq!(arr[0], 11.0);
      assert_eq!(arr[1], 22.0);
      assert_eq!(arr[2], 33.0);
    }
    _ => panic!(),
  }

  // Scalar-Array Promotion Arithmetic
  let promoted = arr1.add(&MValue::Num(10.0))?;
  match &promoted {
    MValue::NumArray(arr) => {
      assert_eq!(arr.len(), 3);
      assert_eq!(arr[0], 11.0);
      assert_eq!(arr[1], 12.0);
      assert_eq!(arr[2], 13.0);
    }
    _ => panic!(),
  }

  Ok(())
}

#[test]
fn test_mvalue_comparisons() -> Result<()> {
  let v1 = MValue::Num(5.0);
  let v2 = MValue::Num(3.0);

  // Scalar comparison
  assert_eq!(
    match v1.gt_op(&v2)? {
      MValue::Bool(b) => b,
      _ => panic!(),
    },
    true
  );
  assert_eq!(
    match v1.lt_op(&v2)? {
      MValue::Bool(b) => b,
      _ => panic!(),
    },
    false
  );
  assert_eq!(
    match v1.eq_op(&v2)? {
      MValue::Bool(b) => b,
      _ => panic!(),
    },
    false
  );

  // Array comparison
  let arr = MValue::NumArray(NumArray::from(vec![1.0, 4.0, 2.0]));
  let cmp_res = arr.gt_op(&MValue::Num(2.0))?;
  match &cmp_res {
    MValue::BoolArray(barr) => {
      assert_eq!(barr.len(), 3);
      assert_eq!(barr[0], false);
      assert_eq!(barr[1], true);
      assert_eq!(barr[2], false);
    }
    _ => panic!(),
  }

  Ok(())
}

#[test]
fn test_mvalue_logicals() -> Result<()> {
  let t = MValue::Bool(true);
  let f = MValue::Bool(false);

  assert_eq!(
    match t.and_op(&f)? {
      MValue::Bool(b) => b,
      _ => panic!(),
    },
    false
  );
  assert_eq!(
    match t.or_op(&f)? {
      MValue::Bool(b) => b,
      _ => panic!(),
    },
    true
  );

  // Logical with numbers (truthiness)
  let n1 = MValue::Num(1.0);
  let n0 = MValue::Num(0.0);
  assert_eq!(
    match n1.and_op(&n0)? {
      MValue::Bool(b) => b,
      _ => panic!(),
    },
    false
  );
  assert_eq!(
    match n1.or_op(&n0)? {
      MValue::Bool(b) => b,
      _ => panic!(),
    },
    true
  );

  // Array logicals
  let arr1 = MValue::BoolArray(crate::numarray::BoolArray::from(vec![true, false, true]));
  let arr2 = MValue::BoolArray(crate::numarray::BoolArray::from(vec![false, false, true]));
  let and_res = arr1.and_op(&arr2)?;
  match &and_res {
    MValue::BoolArray(barr) => {
      assert_eq!(barr.len(), 3);
      assert_eq!(barr[0], false);
      assert_eq!(barr[1], false);
      assert_eq!(barr[2], true);
    }
    _ => panic!(),
  }

  Ok(())
}

#[test]
fn test_runtime_execute_simple() -> Result<()> {
  let code = r#"
    A := 1;
    B := 2;
    UP := A == B;
  "#;

  let mut datas = HashMap::new();
  datas.insert("C".to_string(), NumArray::from(vec![1.0, 2.0, 3.0]));
  let params = HashMap::new();

  let mut rt = MRuntime::new(Context::default());
  let lines = rt.execute(code, &datas, &params)?;

  // A, B, UP are local variables, no LineDef or DRAW, so lines is empty.
  assert_eq!(lines.len(), 0);

  let vars = rt.variables();
  assert!(vars.contains_key("a"));
  assert!(vars.contains_key("b"));
  assert!(vars.contains_key("up"));

  match &vars["up"] {
    MValue::Bool(b) => assert_eq!(*b, false),
    _ => panic!(),
  }

  Ok(())
}

#[test]
fn test_runtime_ta_and_lines() -> Result<()> {
  let code = r#"
    MA3 : MA(C, 3);
  "#;

  let mut datas = HashMap::new();
  datas.insert(
    "C".to_string(),
    NumArray::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
  );
  let params = HashMap::new();

  let mut rt = MRuntime::new(Context::default());
  let lines = rt.execute(code, &datas, &params)?;

  assert_eq!(lines.len(), 1);
  assert_eq!(lines[0].kind, "line");
  assert_eq!(lines[0].name, "ma3");
  assert_eq!(lines[0].data.len(), 5);

  Ok(())
}

#[test]
fn test_runtime_draw_and_ternary() -> Result<()> {
  let code = r#"
    UP := C > 2.0;
    DRAWICON(UP, C, 1);
  "#;

  let mut datas = HashMap::new();
  datas.insert(
    "C".to_string(),
    NumArray::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
  );
  let params = HashMap::new();

  let mut rt = MRuntime::new(Context::default());
  let lines = rt.execute(code, &datas, &params)?;

  assert_eq!(lines.len(), 1);
  assert_eq!(lines[0].kind, "icon");
  assert_eq!(lines[0].name, "icon");
  assert_eq!(lines[0].data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

  let when = lines[0].when.as_ref().unwrap();
  assert_eq!(when, &vec![false, false, true, true, true]);

  let icon = u32::from_le_bytes(lines[0].ext_data.as_ref().unwrap().as_slice().try_into()?);
  assert_eq!(icon, 1);

  Ok(())
}

#[test]
fn test_runtime_self_patterns() -> Result<()> {
  // MA5 : C > O ? C * SELF : SELF;
  // This should compile to ScanMul
  let code = r#"
    MA5 : C > 2.0 ? 2.0 * SELF : SELF;
  "#;

  let mut datas = HashMap::new();
  datas.insert(
    "C".to_string(),
    NumArray::from(vec![1.0, 2.0, 3.0, 4.0, 2.0]),
  );
  let params = HashMap::new();

  let mut rt = MRuntime::new(Context::default());
  let lines = rt.execute(code, &datas, &params)?;

  assert_eq!(lines.len(), 1);
  assert_eq!(lines[0].name, "ma5");
  // Expected values of ta_scan_mul for input=[2.0, 2.0, 2.0, 2.0, 2.0], cond=[F, F, T, T, F]
  // t0: cond=F, acc=1, r=1
  // t1: cond=F, acc=1, r=1
  // t2: cond=T, acc=1*2=2, r=2
  // t3: cond=T, acc=2*2=4, r=4
  // t4: cond=F, acc=4, r=4
  assert_eq!(lines[0].data, vec![1.0, 1.0, 2.0, 4.0, 4.0]);

  Ok(())
}

#[test]
fn test_empty_data_input() -> Result<()> {
  let code = r#"
  MA5 : MA(C, 5);
"#;
  let mut datas = HashMap::new();
  datas.insert("C".to_string(), NumArray::from(vec![]));
  let params = HashMap::new();

  let mut rt = MRuntime::new(Context::default());
  let lines = rt.execute(code, &datas, &params)?;

  assert_eq!(lines.len(), 1);
  assert_eq!(lines[0].name, "ma5");
  assert_eq!(lines[0].data.len(), 0);

  Ok(())
}
