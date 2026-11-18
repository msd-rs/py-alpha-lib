// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

pub const ACTION_KEEP: i32 = 0;
pub const ACTION_BUY: i32 = 1;
pub const ACTION_SELL: i32 = 2;
pub const ACTION_BUY_PERCENT: i32 = 3;
pub const ACTION_SELL_PERCENT: i32 = 4;
pub const ACTION_BUY_NO_CASH: i32 = 5;
pub const ACTION_SELL_NO_HOLD: i32 = 6;

#[derive(Debug, Clone, Copy, Default)]
pub struct FeeRule {
  pub ts: i64,
  pub commission_buy: f64,
  pub min_commission_buy: f64,
  pub tax_buy: f64,
  pub transaction_fee_buy: f64,
  pub commission_sell: f64,
  pub min_commission_sell: f64,
  pub tax_sell: f64,
  pub transaction_fee_sell: f64,
}

impl FeeRule {
  #[inline]
  pub fn buy_fee(&self, value: f64) -> f64 {
    if value == 0.0 {
      return 0.0;
    }
    let commission = value * self.commission_buy;
    commission.max(self.min_commission_buy) + value * self.tax_buy + self.transaction_fee_buy
  }

  #[inline]
  pub fn sell_fee(&self, value: f64) -> f64 {
    if value == 0.0 {
      return 0.0;
    }
    let commission = value * self.commission_sell;
    commission.max(self.min_commission_sell) + value * self.tax_sell + self.transaction_fee_sell
  }
}

#[derive(Debug, Clone)]
pub struct FeeSchedule {
  rules: Vec<FeeRule>,
}

static DEFAULT_FEE_RULE: FeeRule = FeeRule {
  ts: 0,
  commission_buy: 0.0,
  min_commission_buy: 0.0,
  tax_buy: 0.0,
  transaction_fee_buy: 0.0,
  commission_sell: 0.0,
  min_commission_sell: 0.0,
  tax_sell: 0.0,
  transaction_fee_sell: 0.0,
};

impl Default for FeeSchedule {
  fn default() -> Self {
    Self {
      rules: vec![FeeRule::default()],
    }
  }
}

impl FeeSchedule {
  pub fn new(mut rules: Vec<FeeRule>) -> Self {
    rules.sort_by_key(|r| r.ts);
    if rules.is_empty() {
      rules.push(FeeRule::default());
    }
    Self { rules }
  }

  #[inline]
  pub fn get(&self, ts: i64) -> &FeeRule {
    if self.rules.is_empty() {
      return &DEFAULT_FEE_RULE;
    }
    if self.rules.len() == 1 {
      return &self.rules[0];
    }
    for i in 0..self.rules.len() {
      if self.rules[i].ts > ts {
        if i == 0 {
          return &self.rules[0];
        }
        return &self.rules[i - 1];
      }
    }
    &self.rules[self.rules.len() - 1]
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecMode {
  Close,
  NextOpen,
}

impl ExecMode {
  pub fn signal_lag(&self) -> usize {
    match self {
      ExecMode::Close => 0,
      ExecMode::NextOpen => 1,
    }
  }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RiskConfig {
  /// Stop loss threshold as positive decimal (e.g. 0.06 triggers sell when holding_ror <= -0.06).
  pub stop_loss: Option<f64>,
  /// Take profit threshold as positive decimal (e.g. 0.20 triggers sell when holding_ror >= 0.20).
  pub take_profit: Option<f64>,
  /// Trailing drawdown from highest price since entry (e.g. 0.10 triggers sell when price <= peak * 0.90).
  pub trailing_stop: Option<f64>,
  /// Maximum holding bars (e.g. Some(20) triggers sell when holding_days >= 20).
  pub max_holding_days: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct SimConfig {
  pub initial_cash: f64,
  pub unit: u64,
  pub slippage: f64,
  pub warmup_bars: usize,
  pub start_bar: usize,
  pub exec_mode: ExecMode,
  pub check_limit: bool,
  pub fees: FeeSchedule,
  pub risk: RiskConfig,
}

impl Default for SimConfig {
  fn default() -> Self {
    Self {
      initial_cash: 100_000.0,
      unit: 100,
      slippage: 0.0,
      warmup_bars: 0,
      start_bar: 1,
      exec_mode: ExecMode::Close,
      check_limit: true,
      fees: FeeSchedule::default(),
      risk: RiskConfig::default(),
    }
  }
}
