use std::ops::Deref;
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

impl From<Box<[f64]>> for NumArray {
  fn from(data: Box<[f64]>) -> Self {
    NumArray {
      data: Rc::new(Vec::from(data)),
    }
  }
}

impl From<Box<[bool]>> for BoolArray {
  fn from(data: Box<[bool]>) -> Self {
    BoolArray {
      data: Rc::new(Vec::from(data)),
    }
  }
}

impl From<BoolArray> for Box<[u8]> {
  fn from(data: BoolArray) -> Self {
    data
      .data
      .iter()
      .map(|&x| if x { 1u8 } else { 0u8 })
      .collect()
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
