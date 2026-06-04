use anyhow::{Result, anyhow};
use crate::numarray::{NumArray, BoolArray};

const FLOAT_EPSILON: f64 = 1e-9;

trait FloatExt {
  fn eq_f(self, other: &f64) -> bool;
  fn ne_f(self, other: &f64) -> bool;
  fn lt_f(self, other: &f64) -> bool;
  fn le_f(self, other: &f64) -> bool;
  fn gt_f(self, other: &f64) -> bool;
  fn ge_f(self, other: &f64) -> bool;
}

impl FloatExt for f64 {
  fn eq_f(self, other: &f64) -> bool {
    (self - other).abs() < FLOAT_EPSILON
  }
  fn ne_f(self, other: &f64) -> bool {
    (self - other).abs() >= FLOAT_EPSILON
  }
  fn lt_f(self, other: &f64) -> bool {
    self - other < -FLOAT_EPSILON
  }
  fn le_f(self, other: &f64) -> bool {
    self - other <= FLOAT_EPSILON
  }
  fn gt_f(self, other: &f64) -> bool {
    self - other > FLOAT_EPSILON
  }
  fn ge_f(self, other: &f64) -> bool {
    self - other >= -FLOAT_EPSILON
  }
}

#[derive(Debug, Clone)]
pub enum MValue {
  Num(f64),
  Bool(bool),
  Str(String),
  NumArray(NumArray),
  BoolArray(BoolArray),
}

impl MValue {
  pub fn to_num_array(&self, len: usize) -> Result<NumArray> {
    match self {
      MValue::NumArray(arr) => Ok(arr.clone()),
      MValue::Num(n) => Ok(NumArray::from(vec![*n; len])),
      _ => Err(anyhow!("Cannot convert/promote to NumArray: {:?}", self)),
    }
  }

  pub fn to_bool_array(&self, len: usize) -> Result<BoolArray> {
    match self {
      MValue::BoolArray(arr) => Ok(arr.clone()),
      MValue::Bool(b) => Ok(BoolArray::from(vec![*b; len])),
      MValue::Num(n) => Ok(BoolArray::from(vec![*n != 0.0; len])),
      MValue::NumArray(arr) => {
        let res = arr.iter().map(|&x| x != 0.0).collect::<Vec<_>>();
        Ok(BoolArray::from(res))
      }
      _ => Err(anyhow!("Cannot convert/promote to BoolArray: {:?}", self)),
    }
  }

  pub fn to_bool_val(&self) -> Result<MValue> {
    match self {
      MValue::Bool(b) => Ok(MValue::Bool(*b)),
      MValue::Num(n) => Ok(MValue::Bool(*n != 0.0)),
      MValue::BoolArray(arr) => Ok(MValue::BoolArray(arr.clone())),
      MValue::NumArray(arr) => {
        let res = arr.iter().map(|&x| x != 0.0).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      _ => Err(anyhow!("Cannot convert to boolean value: {:?}", self)),
    }
  }

  pub fn add(&self, other: &Self) -> Result<MValue> {
    match (self, other) {
      (MValue::Num(a), MValue::Num(b)) => Ok(MValue::Num(a + b)),
      (MValue::NumArray(a), MValue::NumArray(b)) => {
        if a.len() != b.len() {
          return Err(anyhow!("Mismatched array length: {} and {}", a.len(), b.len()));
        }
        let res = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect::<Vec<_>>();
        Ok(MValue::NumArray(NumArray::from(res)))
      }
      (MValue::NumArray(a), MValue::Num(b)) => {
        let res = a.iter().map(|x| x + b).collect::<Vec<_>>();
        Ok(MValue::NumArray(NumArray::from(res)))
      }
      (MValue::Num(a), MValue::NumArray(b)) => {
        let res = b.iter().map(|x| a + x).collect::<Vec<_>>();
        Ok(MValue::NumArray(NumArray::from(res)))
      }
      _ => Err(anyhow!("Invalid types for Add: {:?} and {:?}", self, other)),
    }
  }

  pub fn sub(&self, other: &Self) -> Result<MValue> {
    match (self, other) {
      (MValue::Num(a), MValue::Num(b)) => Ok(MValue::Num(a - b)),
      (MValue::NumArray(a), MValue::NumArray(b)) => {
        if a.len() != b.len() {
          return Err(anyhow!("Mismatched array length: {} and {}", a.len(), b.len()));
        }
        let res = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect::<Vec<_>>();
        Ok(MValue::NumArray(NumArray::from(res)))
      }
      (MValue::NumArray(a), MValue::Num(b)) => {
        let res = a.iter().map(|x| x - b).collect::<Vec<_>>();
        Ok(MValue::NumArray(NumArray::from(res)))
      }
      (MValue::Num(a), MValue::NumArray(b)) => {
        let res = b.iter().map(|x| a - x).collect::<Vec<_>>();
        Ok(MValue::NumArray(NumArray::from(res)))
      }
      _ => Err(anyhow!("Invalid types for Sub: {:?} and {:?}", self, other)),
    }
  }

  pub fn mul(&self, other: &Self) -> Result<MValue> {
    match (self, other) {
      (MValue::Num(a), MValue::Num(b)) => Ok(MValue::Num(a * b)),
      (MValue::NumArray(a), MValue::NumArray(b)) => {
        if a.len() != b.len() {
          return Err(anyhow!("Mismatched array length: {} and {}", a.len(), b.len()));
        }
        let res = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect::<Vec<_>>();
        Ok(MValue::NumArray(NumArray::from(res)))
      }
      (MValue::NumArray(a), MValue::Num(b)) => {
        let res = a.iter().map(|x| x * b).collect::<Vec<_>>();
        Ok(MValue::NumArray(NumArray::from(res)))
      }
      (MValue::Num(a), MValue::NumArray(b)) => {
        let res = b.iter().map(|x| a * x).collect::<Vec<_>>();
        Ok(MValue::NumArray(NumArray::from(res)))
      }
      _ => Err(anyhow!("Invalid types for Mul: {:?} and {:?}", self, other)),
    }
  }

  pub fn div(&self, other: &Self) -> Result<MValue> {
    match (self, other) {
      (MValue::Num(a), MValue::Num(b)) => Ok(MValue::Num(a / b)),
      (MValue::NumArray(a), MValue::NumArray(b)) => {
        if a.len() != b.len() {
          return Err(anyhow!("Mismatched array length: {} and {}", a.len(), b.len()));
        }
        let res = a.iter().zip(b.iter()).map(|(x, y)| x / y).collect::<Vec<_>>();
        Ok(MValue::NumArray(NumArray::from(res)))
      }
      (MValue::NumArray(a), MValue::Num(b)) => {
        let res = a.iter().map(|x| x / b).collect::<Vec<_>>();
        Ok(MValue::NumArray(NumArray::from(res)))
      }
      (MValue::Num(a), MValue::NumArray(b)) => {
        let res = b.iter().map(|x| a / x).collect::<Vec<_>>();
        Ok(MValue::NumArray(NumArray::from(res)))
      }
      _ => Err(anyhow!("Invalid types for Div: {:?} and {:?}", self, other)),
    }
  }

  pub fn rem(&self, other: &Self) -> Result<MValue> {
    match (self, other) {
      (MValue::Num(a), MValue::Num(b)) => Ok(MValue::Num(a % b)),
      (MValue::NumArray(a), MValue::NumArray(b)) => {
        if a.len() != b.len() {
          return Err(anyhow!("Mismatched array length: {} and {}", a.len(), b.len()));
        }
        let res = a.iter().zip(b.iter()).map(|(x, y)| x % y).collect::<Vec<_>>();
        Ok(MValue::NumArray(NumArray::from(res)))
      }
      (MValue::NumArray(a), MValue::Num(b)) => {
        let res = a.iter().map(|x| x % b).collect::<Vec<_>>();
        Ok(MValue::NumArray(NumArray::from(res)))
      }
      (MValue::Num(a), MValue::NumArray(b)) => {
        let res = b.iter().map(|x| a % x).collect::<Vec<_>>();
        Ok(MValue::NumArray(NumArray::from(res)))
      }
      _ => Err(anyhow!("Invalid types for Rem: {:?} and {:?}", self, other)),
    }
  }

  pub fn pow(&self, other: &Self) -> Result<MValue> {
    match (self, other) {
      (MValue::Num(a), MValue::Num(b)) => Ok(MValue::Num(a.powf(*b))),
      (MValue::NumArray(a), MValue::NumArray(b)) => {
        if a.len() != b.len() {
          return Err(anyhow!("Mismatched array length: {} and {}", a.len(), b.len()));
        }
        let res = a.iter().zip(b.iter()).map(|(x, y)| x.powf(*y)).collect::<Vec<_>>();
        Ok(MValue::NumArray(NumArray::from(res)))
      }
      (MValue::NumArray(a), MValue::Num(b)) => {
        let res = a.iter().map(|x| x.powf(*b)).collect::<Vec<_>>();
        Ok(MValue::NumArray(NumArray::from(res)))
      }
      (MValue::Num(a), MValue::NumArray(b)) => {
        let res = b.iter().map(|x| a.powf(*x)).collect::<Vec<_>>();
        Ok(MValue::NumArray(NumArray::from(res)))
      }
      _ => Err(anyhow!("Invalid types for Pow: {:?} and {:?}", self, other)),
    }
  }

  pub fn neg_op(&self) -> Result<MValue> {
    match self {
      MValue::Num(n) => Ok(MValue::Num(-n)),
      MValue::Bool(b) => Ok(MValue::Bool(!b)),
      MValue::NumArray(arr) => {
        let res = arr.iter().map(|&x| -x).collect::<Vec<_>>();
        Ok(MValue::NumArray(NumArray::from(res)))
      }
      MValue::BoolArray(arr) => {
        let res = arr.iter().map(|&x| !x).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      _ => Err(anyhow!("Cannot negate type: {:?}", self)),
    }
  }

  pub fn eq_op(&self, other: &Self) -> Result<MValue> {
    match (self, other) {
      (MValue::Num(a), MValue::Num(b)) => Ok(MValue::Bool(a.eq_f(b))),
      (MValue::NumArray(a), MValue::NumArray(b)) => {
        if a.len() != b.len() { return Err(anyhow!("Length mismatch")); }
        let res = a.iter().zip(b.iter()).map(|(x, y)| x.eq_f(y)).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      (MValue::NumArray(a), MValue::Num(b)) => {
        let res = a.iter().map(|x| x.eq_f(b)).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      (MValue::Num(a), MValue::NumArray(b)) => {
        let res = b.iter().map(|x| a.eq_f(x)).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      (MValue::Bool(a), MValue::Bool(b)) => Ok(MValue::Bool(a == b)),
      (MValue::BoolArray(a), MValue::BoolArray(b)) => {
        if a.len() != b.len() { return Err(anyhow!("Length mismatch")); }
        let res = a.iter().zip(b.iter()).map(|(x, y)| x == y).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      (MValue::Str(a), MValue::Str(b)) => Ok(MValue::Bool(a == b)),
      _ => Ok(MValue::Bool(false)),
    }
  }

  pub fn ne_op(&self, other: &Self) -> Result<MValue> {
    match (self, other) {
      (MValue::Num(a), MValue::Num(b)) => Ok(MValue::Bool(a.ne_f(b))),
      (MValue::NumArray(a), MValue::NumArray(b)) => {
        if a.len() != b.len() { return Err(anyhow!("Length mismatch")); }
        let res = a.iter().zip(b.iter()).map(|(x, y)| x.ne_f(y)).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      (MValue::NumArray(a), MValue::Num(b)) => {
        let res = a.iter().map(|x| x.ne_f(b)).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      (MValue::Num(a), MValue::NumArray(b)) => {
        let res = b.iter().map(|x| a.ne_f(x)).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      (MValue::Bool(a), MValue::Bool(b)) => Ok(MValue::Bool(a != b)),
      (MValue::BoolArray(a), MValue::BoolArray(b)) => {
        if a.len() != b.len() { return Err(anyhow!("Length mismatch")); }
        let res = a.iter().zip(b.iter()).map(|(x, y)| x != y).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      (MValue::Str(a), MValue::Str(b)) => Ok(MValue::Bool(a != b)),
      _ => Ok(MValue::Bool(true)),
    }
  }

  pub fn lt_op(&self, other: &Self) -> Result<MValue> {
    match (self, other) {
      (MValue::Num(a), MValue::Num(b)) => Ok(MValue::Bool(a.lt_f(b))),
      (MValue::NumArray(a), MValue::NumArray(b)) => {
        if a.len() != b.len() { return Err(anyhow!("Length mismatch")); }
        let res = a.iter().zip(b.iter()).map(|(x, y)| x.lt_f(y)).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      (MValue::NumArray(a), MValue::Num(b)) => {
        let res = a.iter().map(|x| x.lt_f(b)).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      (MValue::Num(a), MValue::NumArray(b)) => {
        let res = b.iter().map(|x| a.lt_f(x)).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      _ => Err(anyhow!("Comparison (<) not supported for these types")),
    }
  }

  pub fn le_op(&self, other: &Self) -> Result<MValue> {
    match (self, other) {
      (MValue::Num(a), MValue::Num(b)) => Ok(MValue::Bool(a.le_f(b))),
      (MValue::NumArray(a), MValue::NumArray(b)) => {
        if a.len() != b.len() { return Err(anyhow!("Length mismatch")); }
        let res = a.iter().zip(b.iter()).map(|(x, y)| x.le_f(y)).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      (MValue::NumArray(a), MValue::Num(b)) => {
        let res = a.iter().map(|x| x.le_f(b)).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      (MValue::Num(a), MValue::NumArray(b)) => {
        let res = b.iter().map(|x| a.le_f(x)).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      _ => Err(anyhow!("Comparison (<=) not supported for these types")),
    }
  }

  pub fn gt_op(&self, other: &Self) -> Result<MValue> {
    match (self, other) {
      (MValue::Num(a), MValue::Num(b)) => Ok(MValue::Bool(a.gt_f(b))),
      (MValue::NumArray(a), MValue::NumArray(b)) => {
        if a.len() != b.len() { return Err(anyhow!("Length mismatch")); }
        let res = a.iter().zip(b.iter()).map(|(x, y)| x.gt_f(y)).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      (MValue::NumArray(a), MValue::Num(b)) => {
        let res = a.iter().map(|x| x.gt_f(b)).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      (MValue::Num(a), MValue::NumArray(b)) => {
        let res = b.iter().map(|x| a.gt_f(x)).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      _ => Err(anyhow!("Comparison (>) not supported for these types")),
    }
  }

  pub fn ge_op(&self, other: &Self) -> Result<MValue> {
    match (self, other) {
      (MValue::Num(a), MValue::Num(b)) => Ok(MValue::Bool(a.ge_f(b))),
      (MValue::NumArray(a), MValue::NumArray(b)) => {
        if a.len() != b.len() { return Err(anyhow!("Length mismatch")); }
        let res = a.iter().zip(b.iter()).map(|(x, y)| x.ge_f(y)).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      (MValue::NumArray(a), MValue::Num(b)) => {
        let res = a.iter().map(|x| x.ge_f(b)).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      (MValue::Num(a), MValue::NumArray(b)) => {
        let res = b.iter().map(|x| a.ge_f(x)).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      _ => Err(anyhow!("Comparison (>=) not supported for these types")),
    }
  }

  pub fn and_op(&self, other: &Self) -> Result<MValue> {
    let lhs = self.to_bool_val()?;
    let rhs = other.to_bool_val()?;
    match (lhs, rhs) {
      (MValue::Bool(a), MValue::Bool(b)) => Ok(MValue::Bool(a && b)),
      (MValue::BoolArray(a), MValue::BoolArray(b)) => {
        if a.len() != b.len() { return Err(anyhow!("Length mismatch")); }
        let res = a.iter().zip(b.iter()).map(|(&x, &y)| x && y).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      (MValue::BoolArray(a), MValue::Bool(b)) => {
        let res = a.iter().map(|&x| x && b).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      (MValue::Bool(a), MValue::BoolArray(b)) => {
        let res = b.iter().map(|&x| a && x).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      _ => unreachable!(),
    }
  }

  pub fn or_op(&self, other: &Self) -> Result<MValue> {
    let lhs = self.to_bool_val()?;
    let rhs = other.to_bool_val()?;
    match (lhs, rhs) {
      (MValue::Bool(a), MValue::Bool(b)) => Ok(MValue::Bool(a || b)),
      (MValue::BoolArray(a), MValue::BoolArray(b)) => {
        if a.len() != b.len() { return Err(anyhow!("Length mismatch")); }
        let res = a.iter().zip(b.iter()).map(|(&x, &y)| x || y).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      (MValue::BoolArray(a), MValue::Bool(b)) => {
        let res = a.iter().map(|&x| x || b).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      (MValue::Bool(a), MValue::BoolArray(b)) => {
        let res = b.iter().map(|&x| a || x).collect::<Vec<_>>();
        Ok(MValue::BoolArray(BoolArray::from(res)))
      }
      _ => unreachable!(),
    }
  }
}

// Implement standard operators for MValue
impl std::ops::Add for MValue {
  type Output = MValue;
  fn add(self, other: Self) -> Self::Output {
    MValue::add(&self, &other).unwrap()
  }
}
impl<'a, 'b> std::ops::Add<&'b MValue> for &'a MValue {
  type Output = MValue;
  fn add(self, other: &'b MValue) -> Self::Output {
    MValue::add(self, other).unwrap()
  }
}

impl std::ops::Sub for MValue {
  type Output = MValue;
  fn sub(self, other: Self) -> Self::Output {
    MValue::sub(&self, &other).unwrap()
  }
}
impl<'a, 'b> std::ops::Sub<&'b MValue> for &'a MValue {
  type Output = MValue;
  fn sub(self, other: &'b MValue) -> Self::Output {
    MValue::sub(self, other).unwrap()
  }
}

impl std::ops::Mul for MValue {
  type Output = MValue;
  fn mul(self, other: Self) -> Self::Output {
    MValue::mul(&self, &other).unwrap()
  }
}
impl<'a, 'b> std::ops::Mul<&'b MValue> for &'a MValue {
  type Output = MValue;
  fn mul(self, other: &'b MValue) -> Self::Output {
    MValue::mul(self, other).unwrap()
  }
}

impl std::ops::Div for MValue {
  type Output = MValue;
  fn div(self, other: Self) -> Self::Output {
    MValue::div(&self, &other).unwrap()
  }
}
impl<'a, 'b> std::ops::Div<&'b MValue> for &'a MValue {
  type Output = MValue;
  fn div(self, other: &'b MValue) -> Self::Output {
    MValue::div(self, other).unwrap()
  }
}

impl std::ops::Rem for MValue {
  type Output = MValue;
  fn rem(self, other: Self) -> Self::Output {
    MValue::rem(&self, &other).unwrap()
  }
}
impl<'a, 'b> std::ops::Rem<&'b MValue> for &'a MValue {
  type Output = MValue;
  fn rem(self, other: &'b MValue) -> Self::Output {
    MValue::rem(self, other).unwrap()
  }
}
