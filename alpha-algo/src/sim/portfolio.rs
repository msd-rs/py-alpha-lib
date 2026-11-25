// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use crate::algo::Error;
use crate::sim::account::{TradeEvents, resolve_adj_factors, update_adj_cost_for_paid_rights};
use crate::sim::config::{ACTION_BUY, ACTION_SELL, FeeSchedule};

#[inline]
fn is_valid_price(p: f64) -> bool {
  !p.is_nan() && p > 0.0
}

#[inline]
fn from_cash(budget: f64, price: f64, unit: u64) -> f64 {
  if price <= 0.0 || price.is_nan() || budget <= 0.0 {
    return 0.0;
  }
  let raw = (budget / price).floor() as u64;
  let u = unit.max(1);
  (raw - (raw % u)) as f64
}

#[derive(Debug, Clone)]
pub struct PortfolioEngine {
  pub groups: usize,
  pub bars: usize,
  pub initial_cash: f64,
  pub cash: f64,
  pub unit: u64,
  pub slippage: f64,
  pub tolerance: f64,
  pub deducted: f64,
  pub fees: FeeSchedule,
  pub tradable_mask: Vec<bool>,
  pub shares: Vec<f64>,
  pub raw_cost_price: Vec<f64>,
  pub raw_mark_price: Vec<f64>,
  pub adj_cost_price: Vec<f64>,
  pub dividends: Vec<f64>,
  pub entry_bars: Vec<i64>,
  /// Pre-indexed corporate action group indices per bar for O(1) sparse updates
  corp_groups_by_bar: Vec<Vec<usize>>,
  pub trades: TradeEvents,
}

impl PortfolioEngine {
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    groups: usize,
    bars: usize,
    initial_cash: f64,
    unit: u64,
    slippage: f64,
    tolerance: f64,
    fees: FeeSchedule,
    tradable_mask: Vec<bool>,
    dividend: Option<&[f64]>,
    transfer_shares: Option<&[f64]>,
    right_shares: Option<&[f64]>,
  ) -> Result<Self, Error> {
    if groups == 0 || bars == 0 {
      return Err(Error::InvalidParameter(
        "groups and bars must be positive".to_string(),
      ));
    }
    if tradable_mask.len() != groups {
      return Err(Error::LengthMismatch(tradable_mask.len(), groups));
    }

    let expected = groups * bars;
    let mut corp_groups_by_bar = vec![Vec::new(); bars];
    let has_div = dividend.filter(|s| s.len() == expected);
    let has_trans = transfer_shares.filter(|s| s.len() == expected);
    let has_right = right_shares.filter(|s| s.len() == expected);

    if has_div.is_some() || has_trans.is_some() || has_right.is_some() {
      for g in 0..groups {
        let off = g * bars;
        for t in 0..bars {
          let idx = off + t;
          let d = has_div.map_or(0.0, |s| s[idx]);
          let tr = has_trans.map_or(0.0, |s| s[idx]);
          let r = has_right.map_or(0.0, |s| s[idx]);
          if (d > 0.0 && !d.is_nan()) || (tr > 0.0 && !tr.is_nan()) || (r > 0.0 && !r.is_nan()) {
            corp_groups_by_bar[t].push(g);
          }
        }
      }
    }

    Ok(Self {
      groups,
      bars,
      initial_cash,
      cash: initial_cash,
      unit: unit.max(1),
      slippage,
      tolerance,
      deducted: 0.0,
      fees,
      tradable_mask,
      shares: vec![0.0; groups],
      raw_cost_price: vec![0.0; groups],
      raw_mark_price: vec![0.0; groups],
      adj_cost_price: vec![0.0; groups],
      dividends: vec![0.0; groups],
      entry_bars: vec![-1; groups],
      corp_groups_by_bar,
      trades: TradeEvents::default(),
    })
  }

  #[inline]
  pub fn set_tradable_mask(&mut self, mask: Vec<bool>) -> Result<(), Error> {
    if mask.len() != self.groups {
      return Err(Error::LengthMismatch(mask.len(), self.groups));
    }
    self.tradable_mask = mask;
    Ok(())
  }

  #[inline]
  pub fn state_of(&self, g: usize) -> Result<(f64, f64, f64, f64, f64, i64), Error> {
    if g >= self.groups {
      return Err(Error::InvalidParameter(format!(
        "group index {g} out of range [0, {})",
        self.groups
      )));
    }
    Ok((
      self.shares[g],
      self.raw_cost_price[g],
      self.raw_mark_price[g],
      self.adj_cost_price[g],
      self.dividends[g],
      self.entry_bars[g],
    ))
  }

  pub fn holding_rors(&self, price: &[f64], bar: usize, out: &mut [f64]) -> Result<(), Error> {
    if out.len() != self.groups {
      return Err(Error::LengthMismatch(out.len(), self.groups));
    }
    let expected = self.groups * self.bars;
    if price.len() < expected || bar >= self.bars {
      return Err(Error::LengthMismatch(price.len(), expected));
    }
    for g in 0..self.groups {
      let sh = self.shares[g];
      let cost_basis = sh * self.raw_cost_price[g];
      if sh <= 0.0 || cost_basis <= 0.0 {
        out[g] = 0.0;
        continue;
      }
      let p = price[g * self.bars + bar];
      let mark_p = if is_valid_price(p) {
        p
      } else if is_valid_price(self.raw_mark_price[g]) {
        self.raw_mark_price[g]
      } else {
        self.raw_cost_price[g]
      };
      out[g] = (sh * mark_p + self.dividends[g] - cost_basis) / cost_basis;
    }
    Ok(())
  }

  #[inline]
  pub fn market_value(&self) -> f64 {
    let mut total = 0.0;
    for g in 0..self.groups {
      if self.tradable_mask[g] && self.shares[g] > 0.0 {
        let p = if self.raw_mark_price[g] > 0.0 {
          self.raw_mark_price[g]
        } else {
          self.raw_cost_price[g]
        };
        total += self.shares[g] * p;
      }
    }
    total
  }

  #[inline]
  pub fn asset(&self) -> f64 {
    self.cash + self.market_value()
  }

  /// Apply corporate actions ONLY on groups that actually have corporate action events on bar `t`
  pub fn apply_corporate_actions(
    &mut self,
    t: usize,
    dividend: Option<&[f64]>,
    transfer_shares: Option<&[f64]>,
    right_shares: Option<&[f64]>,
    right_price: Option<&[f64]>,
    raw_prices: Option<&[f64]>,
    fw_prices: Option<&[f64]>,
  ) {
    if t >= self.bars {
      return;
    }
    let groups_on_bar = &self.corp_groups_by_bar[t];
    if groups_on_bar.is_empty() {
      return;
    }
    for &g in groups_on_bar {
      let sh = self.shares[g];
      if sh <= 0.0 {
        continue;
      }
      let off = g * self.bars;
      let idx = off + t;
      let div = dividend.map_or(0.0, |s| s[idx]);
      let trans = transfer_shares.map_or(0.0, |s| s[idx]);
      let r_share = right_shares.map_or(0.0, |s| s[idx]);
      let r_price = right_price.map_or(0.0, |s| s[idx]);

      if div > 0.0 && !div.is_nan() {
        let amt = self.shares[g] * div;
        self.cash += amt;
        self.dividends[g] += amt;
      }
      if trans > 0.0 && !trans.is_nan() {
        let old_sh = self.shares[g];
        self.shares[g] += old_sh * trans;
        if self.shares[g] > 0.0 {
          let ratio = old_sh / self.shares[g];
          self.raw_cost_price[g] *= ratio;
        }
      }
      if r_share > 0.0 && !r_share.is_nan() {
        let max_r = self.shares[g] * r_share;
        let (actual_r, right_cost) = if r_price > 0.0 && !r_price.is_nan() {
          let affordable = if self.cash > 0.0 {
            (self.cash / r_price).floor()
          } else {
            0.0
          };
          let actual = max_r.min(affordable);
          let cost = actual * r_price;
          self.cash -= cost;
          (actual, cost)
        } else {
          (max_r, 0.0)
        };
        if actual_r > 0.0 {
          let old_sh = self.shares[g];
          let old_cost = old_sh * self.raw_cost_price[g];
          self.shares[g] += actual_r;
          if self.shares[g] > 0.0 {
            self.raw_cost_price[g] = (old_cost + right_cost) / self.shares[g];
            if right_cost > 0.0 {
              let adj_info = raw_prices
                .zip(fw_prices)
                .and_then(|(rp, fp)| resolve_adj_factors(rp, fp, off, t));
              self.adj_cost_price[g] = update_adj_cost_for_paid_rights(
                self.adj_cost_price[g],
                old_sh,
                self.shares[g],
                actual_r,
                r_price,
                div,
                trans,
                adj_info,
              );
            }
          }
        }
      }
    }
  }

  #[inline]
  pub fn update_mark_to_market(
    &mut self,
    t: usize,
    mark_prices: &[f64],
    out_cash: &mut [f64],
    out_assets: &mut [f64],
    out_ror: &mut [f64],
  ) {
    let mut mv = 0.0;
    for g in 0..self.groups {
      let sh = self.shares[g];
      if sh > 0.0 {
        let p = mark_prices[g];
        if is_valid_price(p) {
          self.raw_mark_price[g] = p;
        }
        if self.tradable_mask[g] {
          let eff_p = if self.raw_mark_price[g] > 0.0 {
            self.raw_mark_price[g]
          } else {
            self.raw_cost_price[g]
          };
          mv += sh * eff_p;
        }
      }
    }
    let asset = self.cash + mv;
    out_cash[t] = self.cash;
    out_assets[t] = asset;
    out_ror[t] = if self.initial_cash > 0.0 {
      asset / self.initial_cash - 1.0
    } else {
      0.0
    };
  }

  #[inline]
  pub fn update_mark_to_market_strided(
    &mut self,
    t: usize,
    mark_prices_all: &[f64],
    out_cash: &mut [f64],
    out_assets: &mut [f64],
    out_ror: &mut [f64],
  ) {
    let mut mv = 0.0;
    for g in 0..self.groups {
      let sh = self.shares[g];
      if sh > 0.0 {
        let p = mark_prices_all[g * self.bars + t];
        if is_valid_price(p) {
          self.raw_mark_price[g] = p;
        }
        if self.tradable_mask[g] {
          let eff_p = if self.raw_mark_price[g] > 0.0 {
            self.raw_mark_price[g]
          } else {
            self.raw_cost_price[g]
          };
          mv += sh * eff_p;
        }
      }
    }
    let asset = self.cash + mv;
    out_cash[t] = self.cash;
    out_assets[t] = asset;
    out_ror[t] = if self.initial_cash > 0.0 {
      asset / self.initial_cash - 1.0
    } else {
      0.0
    };
  }

  #[allow(clippy::too_many_arguments)]
  pub fn sell_group(
    &mut self,
    g: usize,
    t: usize,
    ts_val: i64,
    shares_to_sell: f64,
    deal_price: f64,
    out_actions: &mut [i32],
    out_action_shares: &mut [f64],
  ) {
    let sh = self.shares[g];
    if sh <= 0.0 || shares_to_sell <= 0.0 || !is_valid_price(deal_price) {
      return;
    }
    let actual_shares = sh.min(shares_to_sell);
    if actual_shares <= 0.0 {
      return;
    }
    let exec_price = deal_price * (1.0 - self.slippage);
    if !is_valid_price(exec_price) {
      return;
    }
    let val = actual_shares * exec_price;
    let fee = self.fees.get(ts_val).sell_fee(val);
    self.cash += val - fee;

    let rem = sh - actual_shares;
    if rem <= 0.0 {
      self.shares[g] = 0.0;
      self.raw_cost_price[g] = 0.0;
      self.raw_mark_price[g] = 0.0;
      self.adj_cost_price[g] = 0.0;
      self.dividends[g] = 0.0;
      self.entry_bars[g] = -1;
    } else {
      self.dividends[g] *= rem / sh;
      self.shares[g] = rem;
      self.raw_mark_price[g] = exec_price;
    }
    self.deducted += fee;
    let j = g * self.bars + t;
    out_actions[j] = ACTION_SELL;
    out_action_shares[j] = actual_shares;

    self.trades.push(
      g as i32,
      t as i32,
      ts_val,
      ACTION_SELL,
      actual_shares,
      deal_price,
      fee,
      self.cash,
      self.shares[g],
      exec_price,
    );
  }

  #[allow(clippy::too_many_arguments)]
  pub fn buy_group(
    &mut self,
    g: usize,
    t: usize,
    ts_val: i64,
    cash_budget: f64,
    deal_price: f64,
    raw_p: Option<f64>,
    fw_p: Option<f64>,
    out_actions: &mut [i32],
    out_action_shares: &mut [f64],
  ) {
    if self.cash <= 0.0 || cash_budget <= 0.0 || !is_valid_price(deal_price) {
      return;
    }
    let budget = self.cash.min(cash_budget);
    let exec_price = deal_price * (1.0 + self.slippage);
    if !is_valid_price(exec_price) {
      return;
    }
    let fee_rule = self.fees.get(ts_val);
    let mut buy_shares = from_cash(budget, exec_price, self.unit);
    let mut fee = fee_rule.buy_fee(buy_shares * exec_price);
    let u = self.unit as f64;
    while buy_shares > 0.0 && buy_shares * exec_price + fee > budget {
      buy_shares = (buy_shares - u).max(0.0);
      fee = fee_rule.buy_fee(buy_shares * exec_price);
    }
    if buy_shares <= 0.0 {
      return;
    }

    let buy_adj_cost = match (raw_p, fw_p) {
      (Some(rp), Some(fw)) if is_valid_price(rp) && !fw.is_nan() => exec_price * (fw / rp),
      _ => exec_price,
    };

    if self.shares[g] <= 0.0 {
      self.entry_bars[g] = t as i64;
    }

    let val = buy_shares * exec_price;
    self.cash -= val + fee;
    let new_shares = self.shares[g] + buy_shares;
    let total_cost = self.shares[g] * self.raw_cost_price[g] + val;
    let total_adj = self.shares[g] * self.adj_cost_price[g] + buy_shares * buy_adj_cost;
    self.shares[g] = new_shares;
    self.raw_cost_price[g] = total_cost / new_shares;
    self.adj_cost_price[g] = total_adj / new_shares;
    self.raw_mark_price[g] = exec_price;
    self.deducted += fee;

    let j = g * self.bars + t;
    out_actions[j] = ACTION_BUY;
    out_action_shares[j] = buy_shares;

    self.trades.push(
      g as i32,
      t as i32,
      ts_val,
      ACTION_BUY,
      buy_shares,
      deal_price,
      fee,
      self.cash,
      self.shares[g],
      exec_price,
    );
  }

  #[allow(clippy::too_many_arguments)]
  pub fn rebalance_step(
    &mut self,
    t: usize,
    ts_val: i64,
    target_weights: &[f64],
    deal_prices: &[f64],
    mark_prices: &[f64],
    limited: Option<&[i32]>,
    tolerance: Option<f64>,
    universe_mask: Option<&[bool]>,
    raw_prices_all: Option<&[f64]>,
    fw_prices_all: Option<&[f64]>,
    out_actions: &mut [i32],
    out_action_shares: &mut [f64],
    out_cash: &mut [f64],
    out_assets: &mut [f64],
    out_ror: &mut [f64],
  ) -> Result<(), Error> {
    if target_weights.len() != self.groups
      || deal_prices.len() != self.groups
      || mark_prices.len() != self.groups
    {
      return Err(Error::LengthMismatch(target_weights.len(), self.groups));
    }

    let tol = tolerance.unwrap_or(self.tolerance);
    let mut weights = vec![0.0; self.groups];
    let mut w_sum = 0.0;
    for g in 0..self.groups {
      if !self.tradable_mask[g] {
        continue;
      }
      let w = target_weights[g];
      let clean_w = if w.is_nan() || w <= 0.0 { 0.0 } else { w };
      weights[g] = clean_w;
      w_sum += clean_w;
    }
    if w_sum > 1.0 {
      for w in &mut weights {
        *w /= w_sum;
      }
    }

    // Update current valuation with deal_prices where valid before computing targets
    for g in 0..self.groups {
      if !self.tradable_mask[g] {
        continue;
      }
      let dp_g = deal_prices[g];
      if is_valid_price(dp_g) && self.shares[g] > 0.0 {
        self.raw_mark_price[g] = dp_g;
      }
    }

    let total_equity = self.asset();
    let mut target_values: Vec<f64> = weights.iter().map(|&w| total_equity * w).collect();

    if tol > 0.0 && total_equity > 0.0 {
      for g in 0..self.groups {
        if !self.tradable_mask[g] || weights[g] == 0.0 {
          continue;
        }
        let dp_g = deal_prices[g];
        if !is_valid_price(dp_g) {
          continue;
        }
        let curr_val = self.shares[g] * dp_g;
        let curr_w = curr_val / total_equity;
        if (weights[g] - curr_w).abs() <= tol {
          target_values[g] = curr_val;
        }
      }
    }

    // Pass 1: Sells first to free up cash
    for g in 0..self.groups {
      if !self.tradable_mask[g] {
        continue;
      }
      if limited.map_or(false, |l| l[g] == -1) {
        continue;
      }
      let dp_g = deal_prices[g];
      if !is_valid_price(dp_g) {
        continue;
      }
      let sh = self.shares[g];
      let curr_val = sh * dp_g;
      let tgt_val = target_values[g];
      if tgt_val == 0.0 && sh > 0.0 {
        self.sell_group(g, t, ts_val, sh, dp_g, out_actions, out_action_shares);
      } else if curr_val > tgt_val {
        let excess_val = curr_val - tgt_val;
        let raw_sh = (excess_val / dp_g).floor() as u64;
        let sh_sell = (raw_sh - (raw_sh % self.unit)) as f64;
        if sh_sell > 0.0 {
          self.sell_group(g, t, ts_val, sh_sell, dp_g, out_actions, out_action_shares);
        }
      }
    }

    // Pass 2: Buys using available cash
    let mut buy_deficits: Vec<(usize, f64, f64)> = Vec::new();
    let mut total_deficit = 0.0;
    for g in 0..self.groups {
      if !self.tradable_mask[g] {
        continue;
      }
      if universe_mask.map_or(false, |u| !u[g]) {
        continue;
      }
      if limited.map_or(false, |l| l[g] == 1) {
        continue;
      }
      let dp_g = deal_prices[g];
      if !is_valid_price(dp_g) {
        continue;
      }
      let curr_val = self.shares[g] * dp_g;
      let tgt_val = target_values[g];
      if tgt_val > curr_val {
        let def = tgt_val - curr_val;
        total_deficit += def;
        buy_deficits.push((g, def, dp_g));
      }
    }

    if total_deficit > 0.0 && self.cash > 0.0 {
      let scale = (self.cash / total_deficit).min(1.0);
      for (g, def, dp_g) in buy_deficits {
        let j = g * self.bars + t;
        let rp = raw_prices_all.map(|s| s[j]);
        let fp = fw_prices_all.map(|s| s[j]);
        self.buy_group(
          g,
          t,
          ts_val,
          def * scale,
          dp_g,
          rp,
          fp,
          out_actions,
          out_action_shares,
        );
      }
    }

    self.update_mark_to_market(t, mark_prices, out_cash, out_assets, out_ror);
    Ok(())
  }

  #[allow(clippy::too_many_arguments)]
  pub fn step_bar(
    &mut self,
    t: usize,
    sig_i: usize,
    ts_val: i64,
    skip_trade: bool,
    sell_percents_all: &[f64],
    buy_percents_all: &[f64],
    target_weights_bar: Option<&[f64]>,
    tolerance: Option<f64>,
    deal_prices_all: &[f64],
    mark_prices_all: &[f64],
    limited_all: Option<&[i32]>,
    universe_mask: Option<&[bool]>,
    raw_prices_all: Option<&[f64]>,
    fw_prices_all: Option<&[f64]>,
    out_actions: &mut [i32],
    out_action_shares: &mut [f64],
    out_cash: &mut [f64],
    out_assets: &mut [f64],
    out_ror: &mut [f64],
  ) -> Result<(), Error> {
    let expected = self.groups * self.bars;
    if t >= self.bars
      || mark_prices_all.len() < expected
      || deal_prices_all.len() < expected
      || out_cash.len() < self.bars
      || out_assets.len() < self.bars
      || out_ror.len() < self.bars
    {
      return Err(Error::LengthMismatch(mark_prices_all.len(), expected));
    }

    if skip_trade {
      self.update_mark_to_market_strided(t, mark_prices_all, out_cash, out_assets, out_ror);
      return Ok(());
    }

    if sig_i >= self.bars
      || sell_percents_all.len() < expected
      || buy_percents_all.len() < expected
      || out_actions.len() < expected
      || out_action_shares.len() < expected
    {
      return Err(Error::LengthMismatch(sell_percents_all.len(), expected));
    }

    // 1. Execute explicit sell signals first (risk control / stop-loss priority)
    let mut has_explicit_sells = false;
    let mut sold_p: Vec<f64> = Vec::new();
    for g in 0..self.groups {
      if !self.tradable_mask[g] {
        continue;
      }
      let sp = sell_percents_all[g * self.bars + sig_i];
      if sp <= 0.0 {
        continue;
      }
      let idx_now = g * self.bars + t;
      if limited_all.map_or(false, |l| l[idx_now] == -1) {
        continue;
      }
      let sh = self.shares[g];
      if sh <= 0.0 {
        continue;
      }
      let mut shares_to_sell = sh * sp;
      if sp < 1.0 {
        let floored = shares_to_sell.floor() as u64;
        shares_to_sell = (floored - (floored % self.unit)) as f64;
      }
      if shares_to_sell > 0.0 {
        let dp_g = deal_prices_all[idx_now];
        self.sell_group(g, t, ts_val, shares_to_sell, dp_g, out_actions, out_action_shares);
        if !has_explicit_sells {
          sold_p = vec![0.0; self.groups];
          has_explicit_sells = true;
        }
        sold_p[g] = sp;
      }
    }

    // 2. Check if target_weights were provided at sig_i
    if let Some(tw) = target_weights_bar {
      if tw.len() != self.groups {
        return Err(Error::LengthMismatch(tw.len(), self.groups));
      }
      let mut adj_tw;
      let tw_slice = if has_explicit_sells {
        adj_tw = tw.to_vec();
        for g in 0..self.groups {
          if sold_p[g] > 0.0 {
            adj_tw[g] = (adj_tw[g] * (1.0 - sold_p[g])).max(0.0);
          }
        }
        &adj_tw[..]
      } else {
        tw
      };

      let mut deal_bar = Vec::with_capacity(self.groups);
      let mut mark_bar = Vec::with_capacity(self.groups);
      let mut lim_bar_vec = if limited_all.is_some() {
        Some(Vec::with_capacity(self.groups))
      } else {
        None
      };
      for g in 0..self.groups {
        let j = g * self.bars + t;
        deal_bar.push(deal_prices_all[j]);
        mark_bar.push(mark_prices_all[j]);
        if let (Some(l_all), Some(lb)) = (limited_all, lim_bar_vec.as_mut()) {
          lb.push(l_all[j]);
        }
      }

      return self.rebalance_step(
        t,
        ts_val,
        tw_slice,
        &deal_bar,
        &mark_bar,
        lim_bar_vec.as_deref(),
        tolerance,
        universe_mask,
        raw_prices_all,
        fw_prices_all,
        out_actions,
        out_action_shares,
        out_cash,
        out_assets,
        out_ror,
      );
    }

    // 3. Otherwise process buy signals sharing available cash across active signals
    if self.cash > 0.0 {
      let mut buy_groups: Vec<(usize, f64)> = Vec::new();
      let mut total_req = 0.0;
      for g in 0..self.groups {
        if !self.tradable_mask[g] {
          continue;
        }
        if universe_mask.map_or(false, |u| !u[g]) {
          continue;
        }
        let idx_now = g * self.bars + t;
        if limited_all.map_or(false, |l| l[idx_now] == 1) {
          continue;
        }
        let bp = buy_percents_all[g * self.bars + sig_i];
        if bp > 0.0 {
          total_req += bp;
          buy_groups.push((g, bp));
        }
      }
      if !buy_groups.is_empty() && total_req > 0.0 {
        let alloc_cash = self.cash * total_req.min(1.0);
        for (g, bp) in buy_groups {
          let j = g * self.bars + t;
          let budget = alloc_cash * (bp / total_req);
          let dp_g = deal_prices_all[j];
          let rp = raw_prices_all.map(|s| s[j]);
          let fp = fw_prices_all.map(|s| s[j]);
          self.buy_group(
            g,
            t,
            ts_val,
            budget,
            dp_g,
            rp,
            fp,
            out_actions,
            out_action_shares,
          );
        }
      }
    }

    self.update_mark_to_market_strided(t, mark_prices_all, out_cash, out_assets, out_ror);
    Ok(())
  }
}
