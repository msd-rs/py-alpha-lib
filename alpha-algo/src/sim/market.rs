// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use crate::algo::Error;

#[derive(Debug, Clone, Copy)]
pub struct MarketSlices<'a> {
  pub groups: usize,
  pub bars: usize,
  pub ts: &'a [i64],
  pub price: &'a [f64],
  pub deal_price: &'a [f64],
  pub fw_price: Option<&'a [f64]>,
  pub limited: Option<&'a [i32]>,
  pub dividend: Option<&'a [f64]>,
  pub transfer_shares: Option<&'a [f64]>,
  pub right_shares: Option<&'a [f64]>,
  pub right_price: Option<&'a [f64]>,
}

impl<'a> MarketSlices<'a> {
  pub fn validate(&self) -> Result<(), Error> {
    if self.groups == 0 || self.bars == 0 {
      return Err(Error::InvalidParameter(
        "groups and bars must be positive".to_string(),
      ));
    }
    let expected = self.groups.checked_mul(self.bars).ok_or_else(|| {
      Error::InvalidParameter("groups * bars overflows usize".to_string())
    })?;

    let check = |name: &str, len: usize| -> Result<(), Error> {
      if len != expected {
        return Err(Error::InvalidParameter(format!(
          "{name} length ({len}) does not match groups ({}) * bars ({}) = {expected}",
          self.groups, self.bars
        )));
      }
      Ok(())
    };

    check("ts", self.ts.len())?;
    check("price", self.price.len())?;
    check("deal_price", self.deal_price.len())?;
    if let Some(s) = self.fw_price {
      check("fw_price", s.len())?;
    }
    if let Some(s) = self.limited {
      check("limited", s.len())?;
    }
    if let Some(s) = self.dividend {
      check("dividend", s.len())?;
    }
    if let Some(s) = self.transfer_shares {
      check("transfer_shares", s.len())?;
    }
    if let Some(s) = self.right_shares {
      check("right_shares", s.len())?;
    }
    if let Some(s) = self.right_price {
      check("right_price", s.len())?;
    }
    Ok(())
  }
}
