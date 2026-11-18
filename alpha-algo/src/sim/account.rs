// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use rayon::prelude::*;

use crate::algo::Error;
use crate::sim::config::{
  ACTION_BUY, ACTION_BUY_NO_CASH, ACTION_KEEP, ACTION_SELL, ACTION_SELL_NO_HOLD, ExecMode,
  FeeRule, RiskConfig, SimConfig,
};
use crate::sim::market::MarketSlices;

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

#[derive(Debug, Clone, Default)]
pub struct TradeEvents {
  pub group: Vec<i32>,
  pub bar: Vec<i32>,
  pub ts: Vec<i64>,
  pub action: Vec<i32>,
  pub shares: Vec<f64>,
  pub deal_price: Vec<f64>,
  pub fee: Vec<f64>,
  pub cash_after: Vec<f64>,
  pub pos_shares: Vec<f64>,
  pub raw_cost_price: Vec<f64>,
}

impl TradeEvents {
  #[inline]
  pub fn push(
    &mut self,
    group: i32,
    bar: i32,
    ts: i64,
    action: i32,
    shares: f64,
    deal_price: f64,
    fee: f64,
    cash_after: f64,
    pos_shares: f64,
    raw_cost_price: f64,
  ) {
    self.group.push(group);
    self.bar.push(bar);
    self.ts.push(ts);
    self.action.push(action);
    self.shares.push(shares);
    self.deal_price.push(deal_price);
    self.fee.push(fee);
    self.cash_after.push(cash_after);
    self.pos_shares.push(pos_shares);
    self.raw_cost_price.push(raw_cost_price);
  }

  pub fn extend_from(&mut self, other: TradeEvents) {
    self.group.extend(other.group);
    self.bar.extend(other.bar);
    self.ts.extend(other.ts);
    self.action.extend(other.action);
    self.shares.extend(other.shares);
    self.deal_price.extend(other.deal_price);
    self.fee.extend(other.fee);
    self.cash_after.extend(other.cash_after);
    self.pos_shares.extend(other.pos_shares);
    self.raw_cost_price.extend(other.raw_cost_price);
  }

  pub fn len(&self) -> usize {
    self.group.len()
  }

  pub fn is_empty(&self) -> bool {
    self.group.is_empty()
  }
}

#[derive(Debug, Clone, Default)]
pub struct AccountSnapshot {
  pub cash: Vec<f64>,
  pub shares: Vec<f64>,
  pub raw_cost_price: Vec<f64>,
  pub raw_mark_price: Vec<f64>,
  pub adj_cost_price: Vec<f64>,
  pub dividends: Vec<f64>,
  pub per_assets: Vec<f64>,
  pub deducted: Vec<f64>,
  pub holding_bars: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct AccountState {
  pub cash: f64,
  pub shares: f64,
  pub raw_cost_price: f64,
  pub raw_mark_price: f64,
  pub adj_cost_price: f64,
  pub dividends: f64,
  pub per_assets: f64,
  pub deducted: f64,
  pub holding_bars: i32,
  pub peak_price: f64,
  pub pending_risk_sell: bool,
}

impl AccountState {
  pub fn new(initial_cash: f64) -> Self {
    Self {
      cash: initial_cash,
      shares: 0.0,
      raw_cost_price: 0.0,
      raw_mark_price: 0.0,
      adj_cost_price: 0.0,
      dividends: 0.0,
      per_assets: 0.0,
      deducted: 0.0,
      holding_bars: 0,
      peak_price: 0.0,
      pending_risk_sell: false,
    }
  }

  #[inline]
  pub fn effective_mark_price(&self) -> f64 {
    if self.raw_mark_price > 0.0 {
      self.raw_mark_price
    } else {
      self.raw_cost_price
    }
  }

  #[inline]
  pub fn market_value(&self) -> f64 {
    if self.shares <= 0.0 {
      0.0
    } else {
      self.shares * self.effective_mark_price()
    }
  }

  #[inline]
  pub fn asset(&self) -> f64 {
    self.cash + self.market_value()
  }

  #[inline]
  pub fn holding_ror(&self) -> f64 {
    let cost_basis = self.shares * self.raw_cost_price;
    if self.shares <= 0.0 || cost_basis <= 0.0 {
      return 0.0;
    }
    (self.market_value() + self.dividends - cost_basis) / cost_basis
  }

  #[inline]
  pub fn apply_corporate_actions(
    &mut self,
    div: f64,
    trans: f64,
    r_share: f64,
    r_price: f64,
  ) {
    if self.shares <= 0.0 {
      return;
    }
    if div > 0.0 && !div.is_nan() {
      let amount = self.shares * div;
      self.cash += amount;
      self.dividends += amount;
    }
    if trans > 0.0 && !trans.is_nan() {
      let old_shares = self.shares;
      self.shares += self.shares * trans;
      if self.shares > 0.0 {
        let ratio = old_shares / self.shares;
        self.raw_cost_price *= ratio;
        self.adj_cost_price *= ratio;
      }
    }
    if r_share > 0.0 && !r_share.is_nan() {
      let max_r = self.shares * r_share;
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
        let old_cost = self.shares * self.raw_cost_price;
        let old_raw_cost = self.raw_cost_price;
        self.shares += actual_r;
        if self.shares > 0.0 {
          self.raw_cost_price = (old_cost + right_cost) / self.shares;
          if old_raw_cost > 0.0 {
            self.adj_cost_price *= self.raw_cost_price / old_raw_cost;
          }
        }
      }
    }
  }

  #[inline]
  pub fn do_buy(
    &mut self,
    pct: f64,
    deal_p: f64,
    raw_p: f64,
    fw_p: Option<f64>,
    unit: u64,
    slippage: f64,
    fee_rule: &FeeRule,
  ) -> (i32, f64, f64, f64) {
    if !is_valid_price(deal_p) || self.cash <= 0.0 {
      return (ACTION_BUY_NO_CASH, 0.0, 0.0, 0.0);
    }
    let exec_price = deal_p * (1.0 + slippage);
    if !is_valid_price(exec_price) {
      return (ACTION_BUY_NO_CASH, 0.0, 0.0, 0.0);
    }

    let mut buy_shares = from_cash(self.cash * pct, exec_price, unit);
    let mut fee = fee_rule.buy_fee(buy_shares * exec_price);
    if buy_shares * exec_price + fee > self.cash {
      buy_shares = from_cash((self.cash - fee).max(0.0), exec_price, unit);
      fee = fee_rule.buy_fee(buy_shares * exec_price);
      let u = unit.max(1) as f64;
      while buy_shares > 0.0 && buy_shares * exec_price + fee > self.cash {
        buy_shares = (buy_shares - u).max(0.0);
        fee = fee_rule.buy_fee(buy_shares * exec_price);
      }
    }

    if buy_shares <= 0.0 {
      return (ACTION_BUY_NO_CASH, 0.0, 0.0, 0.0);
    }

    let buy_adj_cost = match fw_p {
      Some(fw) if is_valid_price(raw_p) && !fw.is_nan() => exec_price * (fw / raw_p),
      _ => exec_price,
    };

    let val = buy_shares * exec_price;
    self.cash -= val + fee;
    let new_shares = self.shares + buy_shares;
    let total_cost_basis = self.shares * self.raw_cost_price + val;
    let total_adj_basis = self.shares * self.adj_cost_price + buy_shares * buy_adj_cost;
    self.shares = new_shares;
    self.raw_cost_price = total_cost_basis / new_shares;
    self.adj_cost_price = total_adj_basis / new_shares;
    self.raw_mark_price = exec_price;
    self.deducted += fee;
    if self.per_assets == 0.0 {
      self.per_assets = self.asset();
    }
    (ACTION_BUY, buy_shares, exec_price, fee)
  }

  #[inline]
  pub fn do_sell(
    &mut self,
    pct: f64,
    deal_p: f64,
    slippage: f64,
    fee_rule: &FeeRule,
  ) -> (i32, f64, f64, f64) {
    if !is_valid_price(deal_p) || self.shares <= 0.0 || pct <= 0.0 {
      return (ACTION_SELL_NO_HOLD, 0.0, 0.0, 0.0);
    }
    let exec_price = deal_p * (1.0 - slippage);
    if !is_valid_price(exec_price) {
      return (ACTION_SELL_NO_HOLD, 0.0, 0.0, 0.0);
    }
    let sell_shares = (self.shares * pct).min(self.shares);
    if sell_shares <= 0.0 {
      return (ACTION_SELL_NO_HOLD, 0.0, 0.0, 0.0);
    }

    let val = sell_shares * exec_price;
    let fee = fee_rule.sell_fee(val);
    self.cash += val - fee;
    let rem_shares = self.shares - sell_shares;
    if rem_shares <= 0.0 {
      self.shares = 0.0;
      self.raw_cost_price = 0.0;
      self.raw_mark_price = 0.0;
      self.adj_cost_price = 0.0;
      self.dividends = 0.0;
      self.per_assets = 0.0;
    } else {
      let ratio = rem_shares / self.shares;
      self.dividends *= ratio;
      self.shares = rem_shares;
      self.raw_mark_price = exec_price;
    }
    self.deducted += fee;
    (ACTION_SELL, sell_shares, exec_price, fee)
  }

  #[inline]
  pub fn check_risk(&mut self, mark_p: f64, risk: &RiskConfig) -> bool {
    if self.shares <= 0.0 {
      return false;
    }
    if is_valid_price(mark_p) {
      self.raw_mark_price = mark_p;
      if mark_p > self.peak_price {
        self.peak_price = mark_p;
      }
    }
    let ror = self.holding_ror();
    if let Some(sl) = risk.stop_loss {
      if ror <= -sl {
        return true;
      }
    }
    if let Some(tp) = risk.take_profit {
      if ror >= tp {
        return true;
      }
    }
    if let Some(tr) = risk.trailing_stop {
      if self.peak_price > 0.0 && is_valid_price(mark_p) && mark_p <= self.peak_price * (1.0 - tr) {
        return true;
      }
    }
    if let Some(max_d) = risk.max_holding_days {
      if self.holding_bars >= max_d as i32 {
        return true;
      }
    }
    false
  }

  #[inline]
  #[allow(clippy::too_many_arguments)]
  pub fn step_one_bar_with_corp_flag(
    &mut self,
    g: i32,
    t: usize,
    ts_val: i64,
    raw_p: f64,
    deal_p: f64,
    fw_p: Option<f64>,
    lim: i32,
    div: f64,
    trans: f64,
    r_share: f64,
    r_price: f64,
    apply_corp: bool,
    buy_p: f64,
    sell_p: f64,
    allow_trade: bool,
    cfg: &SimConfig,
    base_asset: f64,
    out_assets: &mut [f64],
    out_ror: &mut [f64],
    out_ror_hold: &mut [f64],
    out_actions: &mut [i32],
    out_action_shares: &mut [f64],
    trades: &mut TradeEvents,
  ) {
    // 1. Corporate actions at bar start if holding shares (unless already pre-applied)
    if apply_corp {
      self.apply_corporate_actions(div, trans, r_share, r_price);
    }

    let has_risk = cfg.risk.stop_loss.is_some()
      || cfg.risk.take_profit.is_some()
      || cfg.risk.trailing_stop.is_some()
      || cfg.risk.max_holding_days.is_some();

    let mut eff_sell_p = sell_p;
    if allow_trade && has_risk && cfg.exec_mode == ExecMode::Close {
      if self.check_risk(raw_p, &cfg.risk) {
        eff_sell_p = eff_sell_p.max(1.0);
      }
    } else if allow_trade && self.pending_risk_sell {
      eff_sell_p = eff_sell_p.max(1.0);
      self.pending_risk_sell = false;
    }

    let can_sell = allow_trade && eff_sell_p > 0.0 && (!cfg.check_limit || lim != -1);
    let can_buy = allow_trade && buy_p > 0.0 && (!cfg.check_limit || lim != 1);

    let fee_rule = cfg.fees.get(ts_val);
    let mut last_action = ACTION_KEEP;
    let mut last_shares = 0.0;

    if can_sell && can_buy {
      let (act_s, sh_s, ep_s, fee_s) = self.do_sell(eff_sell_p, deal_p, cfg.slippage, fee_rule);
      if act_s == ACTION_SELL {
        trades.push(
          g,
          t as i32,
          ts_val,
          act_s,
          sh_s,
          ep_s,
          fee_s,
          self.cash,
          self.shares,
          self.raw_cost_price,
        );
      }
      let (act_b, sh_b, ep_b, fee_b) =
        self.do_buy(buy_p, deal_p, raw_p, fw_p, cfg.unit, cfg.slippage, fee_rule);
      if act_b == ACTION_BUY {
        trades.push(
          g,
          t as i32,
          ts_val,
          act_b,
          sh_b,
          ep_b,
          fee_b,
          self.cash,
          self.shares,
          self.raw_cost_price,
        );
      }
      last_action = act_b;
      last_shares = sh_b;
    } else if can_sell {
      let (act_s, sh_s, ep_s, fee_s) = self.do_sell(eff_sell_p, deal_p, cfg.slippage, fee_rule);
      if act_s == ACTION_SELL {
        trades.push(
          g,
          t as i32,
          ts_val,
          act_s,
          sh_s,
          ep_s,
          fee_s,
          self.cash,
          self.shares,
          self.raw_cost_price,
        );
      }
      last_action = act_s;
      last_shares = sh_s;
    } else if can_buy {
      let (act_b, sh_b, ep_b, fee_b) =
        self.do_buy(buy_p, deal_p, raw_p, fw_p, cfg.unit, cfg.slippage, fee_rule);
      if act_b == ACTION_BUY {
        trades.push(
          g,
          t as i32,
          ts_val,
          act_b,
          sh_b,
          ep_b,
          fee_b,
          self.cash,
          self.shares,
          self.raw_cost_price,
        );
      }
      last_action = act_b;
      last_shares = sh_b;
    } else if is_valid_price(raw_p) {
      self.raw_mark_price = raw_p;
    }

    // Ensure mark-to-market uses closing price when executed at next_open
    if cfg.exec_mode == ExecMode::NextOpen && self.shares > 0.0 && is_valid_price(raw_p) {
      self.raw_mark_price = raw_p;
    }

    let asset = self.asset();
    let hold_profit = if self.per_assets > 0.0 {
      (asset - self.per_assets) / self.per_assets
    } else {
      0.0
    };

    out_assets[t] = asset;
    out_ror[t] = if base_asset > 0.0 {
      asset / base_asset - 1.0
    } else {
      0.0
    };
    out_ror_hold[t] = hold_profit;
    out_actions[t] = last_action;
    out_action_shares[t] = last_shares;

    if self.shares > 0.0 {
      self.holding_bars += 1;
      if is_valid_price(raw_p) && raw_p > self.peak_price {
        self.peak_price = raw_p;
      } else if self.peak_price == 0.0 {
        self.peak_price = self.effective_mark_price();
      }
      if has_risk && cfg.exec_mode == ExecMode::NextOpen && t >= cfg.warmup_bars {
        if self.check_risk(raw_p, &cfg.risk) {
          self.pending_risk_sell = true;
        }
      }
    } else {
      self.holding_bars = 0;
      self.peak_price = 0.0;
      self.pending_risk_sell = false;
    }
  }
}

#[allow(clippy::too_many_arguments)]
pub fn sim_accounts(
  market: &MarketSlices<'_>,
  cfg: &SimConfig,
  buy_percent: &[f64],
  sell_percent: &[f64],
  out_assets: &mut [f64],
  out_ror: &mut [f64],
  out_ror_hold: &mut [f64],
  out_actions: &mut [i32],
  out_action_shares: &mut [f64],
) -> Result<(TradeEvents, AccountSnapshot), Error> {
  market.validate()?;
  let expected = market.groups * market.bars;
  if buy_percent.len() != expected
    || sell_percent.len() != expected
    || out_assets.len() != expected
    || out_ror.len() != expected
    || out_ror_hold.len() != expected
    || out_actions.len() != expected
    || out_action_shares.len() != expected
  {
    return Err(Error::LengthMismatch(buy_percent.len(), expected));
  }

  let bars = market.bars;
  let groups = market.groups;
  let lag = cfg.exec_mode.signal_lag();

  let results: Vec<(TradeEvents, AccountState)> = (0..groups)
    .into_par_iter()
    .zip(out_assets.par_chunks_mut(bars))
    .zip(out_ror.par_chunks_mut(bars))
    .zip(out_ror_hold.par_chunks_mut(bars))
    .zip(out_actions.par_chunks_mut(bars))
    .zip(out_action_shares.par_chunks_mut(bars))
    .map(|(((((g, assets_g), ror_g), ror_hold_g), act_g), sh_g)| {
      let off = g * bars;
      let ts_g = &market.ts[off..off + bars];
      let price_g = &market.price[off..off + bars];
      let deal_g = &market.deal_price[off..off + bars];
      let buy_g = &buy_percent[off..off + bars];
      let sell_g = &sell_percent[off..off + bars];

      let fw_g = market.fw_price.map(|s| &s[off..off + bars]);
      let lim_g = market.limited.map(|s| &s[off..off + bars]);
      let div_g = market.dividend.map(|s| &s[off..off + bars]);
      let trans_g = market.transfer_shares.map(|s| &s[off..off + bars]);
      let rs_g = market.right_shares.map(|s| &s[off..off + bars]);
      let rp_g = market.right_price.map(|s| &s[off..off + bars]);

      assets_g.fill(cfg.initial_cash);
      ror_g.fill(0.0);
      ror_hold_g.fill(0.0);
      act_g.fill(ACTION_KEEP);
      sh_g.fill(0.0);

      let mut state = AccountState::new(cfg.initial_cash);
      let mut local_trades = TradeEvents::default();

      let start_t = cfg.start_bar.min(bars);
      for t in start_t..bars {
        let sig_t = t as isize - lag as isize;
        let allow_trade = sig_t >= 0 && t >= cfg.warmup_bars;
        let (buy_p, sell_p) = if allow_trade {
          let st = sig_t as usize;
          (buy_g[st], sell_g[st])
        } else {
          (0.0, 0.0)
        };

        let lim = lim_g.map_or(0, |s| s[t]);
        let div = div_g.map_or(0.0, |s| s[t]);
        let trans = trans_g.map_or(0.0, |s| s[t]);
        let r_share = rs_g.map_or(0.0, |s| s[t]);
        let r_price = rp_g.map_or(0.0, |s| s[t]);
        let fw_p = fw_g.map(|s| s[t]);

        state.step_one_bar_with_corp_flag(
          g as i32,
          t,
          ts_g[t],
          price_g[t],
          deal_g[t],
          fw_p,
          lim,
          div,
          trans,
          r_share,
          r_price,
          true,
          buy_p,
          sell_p,
          allow_trade,
          cfg,
          cfg.initial_cash,
          assets_g,
          ror_g,
          ror_hold_g,
          act_g,
          sh_g,
          &mut local_trades,
        );
      }

      (local_trades, state)
    })
    .collect();

  let mut all_trades = TradeEvents::default();
  let mut snap = AccountSnapshot {
    cash: Vec::with_capacity(groups),
    shares: Vec::with_capacity(groups),
    raw_cost_price: Vec::with_capacity(groups),
    raw_mark_price: Vec::with_capacity(groups),
    adj_cost_price: Vec::with_capacity(groups),
    dividends: Vec::with_capacity(groups),
    per_assets: Vec::with_capacity(groups),
    deducted: Vec::with_capacity(groups),
    holding_bars: Vec::with_capacity(groups),
  };

  for (tr, st) in results {
    all_trades.extend_from(tr);
    snap.cash.push(st.cash);
    snap.shares.push(st.shares);
    snap.raw_cost_price.push(st.raw_cost_price);
    snap.raw_mark_price.push(st.raw_mark_price);
    snap.adj_cost_price.push(st.adj_cost_price);
    snap.dividends.push(st.dividends);
    snap.per_assets.push(st.per_assets);
    snap.deducted.push(st.deducted);
    snap.holding_bars.push(st.holding_bars);
  }

  Ok((all_trades, snap))
}

/// Stateful SoA multi-account batch for strategies that execute Python bar-by-bar.
#[derive(Debug, Clone)]
pub struct AccountBatch {
  pub groups: usize,
  pub bars: usize,
  pub cfg: SimConfig,
  pub states: Vec<AccountState>,
  pub trades: TradeEvents,
  corp_applied_bar: Option<usize>,
}

impl AccountBatch {
  pub fn new(groups: usize, bars: usize, cfg: SimConfig) -> Result<Self, Error> {
    if groups == 0 || bars == 0 {
      return Err(Error::InvalidParameter(
        "groups and bars must be positive".to_string(),
      ));
    }
    let states = vec![AccountState::new(cfg.initial_cash); groups];
    Ok(Self {
      groups,
      bars,
      cfg,
      states,
      trades: TradeEvents::default(),
      corp_applied_bar: None,
    })
  }

  /// Apply corporate actions for bar `t` ahead of strategy `execute()` so
  /// `holding_ror()` and `position()` reflect ex-dividend/ex-split adjustments.
  pub fn apply_corporate_actions_bar(
    &mut self,
    t: usize,
    dividend: Option<&[f64]>,
    transfer_shares: Option<&[f64]>,
    right_shares: Option<&[f64]>,
    right_price: Option<&[f64]>,
  ) -> Result<(), Error> {
    if t >= self.bars {
      return Err(Error::InvalidParameter(format!(
        "bar {t} out of bounds [0, {})",
        self.bars
      )));
    }
    if self.corp_applied_bar == Some(t) {
      return Ok(());
    }
    if dividend.is_some() || transfer_shares.is_some() || right_shares.is_some() {
      let bars = self.bars;
      for (g, state) in self.states.iter_mut().enumerate() {
        if state.shares <= 0.0 {
          continue;
        }
        let idx = g * bars + t;
        let div = dividend.map_or(0.0, |s| s[idx]);
        let trans = transfer_shares.map_or(0.0, |s| s[idx]);
        let r_share = right_shares.map_or(0.0, |s| s[idx]);
        let r_price = right_price.map_or(0.0, |s| s[idx]);
        state.apply_corporate_actions(div, trans, r_share, r_price);
      }
    }
    self.corp_applied_bar = Some(t);
    Ok(())
  }

  #[allow(clippy::too_many_arguments)]
  pub fn step_bar(
    &mut self,
    t: usize,
    market: &MarketSlices<'_>,
    buy_percent: &[f64],
    sell_percent: &[f64],
    out_assets: &mut [f64],
    out_ror: &mut [f64],
    out_ror_hold: &mut [f64],
    out_actions: &mut [i32],
    out_action_shares: &mut [f64],
  ) -> Result<(), Error> {
    if t >= self.bars {
      return Err(Error::InvalidParameter(format!(
        "bar {t} out of bounds [0, {})",
        self.bars
      )));
    }
    let expected = self.groups * self.bars;
    if buy_percent.len() != expected
      || sell_percent.len() != expected
      || out_assets.len() != expected
      || out_ror.len() != expected
      || out_ror_hold.len() != expected
      || out_actions.len() != expected
      || out_action_shares.len() != expected
    {
      return Err(Error::LengthMismatch(buy_percent.len(), expected));
    }

    let lag = self.cfg.exec_mode.signal_lag();
    let sig_t = t as isize - lag as isize;
    let allow_trade = sig_t >= 0 && t >= self.cfg.warmup_bars;
    let bars = self.bars;
    let apply_corp = self.corp_applied_bar != Some(t);
    self.corp_applied_bar = None;

    for (g, state) in self.states.iter_mut().enumerate() {
      let off = g * bars;
      let idx = off + t;
      let ts_val = market.ts[idx];
      let raw_p = market.price[idx];
      let deal_p = market.deal_price[idx];
      let fw_p = market.fw_price.map(|s| s[idx]);
      let lim = market.limited.map_or(0, |s| s[idx]);
      let div = market.dividend.map_or(0.0, |s| s[idx]);
      let trans = market.transfer_shares.map_or(0.0, |s| s[idx]);
      let r_share = market.right_shares.map_or(0.0, |s| s[idx]);
      let r_price = market.right_price.map_or(0.0, |s| s[idx]);

      let (buy_p, sell_p) = if allow_trade {
        let sig_idx = off + sig_t as usize;
        (buy_percent[sig_idx], sell_percent[sig_idx])
      } else {
        (0.0, 0.0)
      };

      state.step_one_bar_with_corp_flag(
        g as i32,
        t,
        ts_val,
        raw_p,
        deal_p,
        fw_p,
        lim,
        div,
        trans,
        r_share,
        r_price,
        apply_corp,
        buy_p,
        sell_p,
        allow_trade,
        &self.cfg,
        self.cfg.initial_cash,
        &mut out_assets[off..off + bars],
        &mut out_ror[off..off + bars],
        &mut out_ror_hold[off..off + bars],
        &mut out_actions[off..off + bars],
        &mut out_action_shares[off..off + bars],
        &mut self.trades,
      );
    }
    Ok(())
  }

  pub fn holding_rors(&self, price: &[f64], bar: usize, out: &mut [f64]) -> Result<(), Error> {
    if out.len() != self.groups {
      return Err(Error::LengthMismatch(out.len(), self.groups));
    }
    for (g, st) in self.states.iter().enumerate() {
      let cost_basis = st.shares * st.raw_cost_price;
      if st.shares <= 0.0 || cost_basis <= 0.0 {
        out[g] = 0.0;
        continue;
      }
      let p = price[g * self.bars + bar];
      let mark_p = if is_valid_price(p) {
        p
      } else {
        st.effective_mark_price()
      };
      out[g] = (st.shares * mark_p + st.dividends - cost_basis) / cost_basis;
    }
    Ok(())
  }

  pub fn snapshot(&self) -> AccountSnapshot {
    let mut snap = AccountSnapshot {
      cash: Vec::with_capacity(self.groups),
      shares: Vec::with_capacity(self.groups),
      raw_cost_price: Vec::with_capacity(self.groups),
      raw_mark_price: Vec::with_capacity(self.groups),
      adj_cost_price: Vec::with_capacity(self.groups),
      dividends: Vec::with_capacity(self.groups),
      per_assets: Vec::with_capacity(self.groups),
      deducted: Vec::with_capacity(self.groups),
      holding_bars: Vec::with_capacity(self.groups),
    };
    for st in &self.states {
      snap.cash.push(st.cash);
      snap.shares.push(st.shares);
      snap.raw_cost_price.push(st.raw_cost_price);
      snap.raw_mark_price.push(st.raw_mark_price);
      snap.adj_cost_price.push(st.adj_cost_price);
      snap.dividends.push(st.dividends);
      snap.per_assets.push(st.per_assets);
      snap.deducted.push(st.deducted);
      snap.holding_bars.push(st.holding_bars);
    }
    snap
  }
}
