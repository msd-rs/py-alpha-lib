// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

pub mod account;
pub mod config;
pub mod market;
pub mod portfolio;

pub use account::{AccountBatch, AccountSnapshot, AccountState, TradeEvents, sim_accounts};
pub use config::{
  ACTION_BUY, ACTION_BUY_NO_CASH, ACTION_BUY_PERCENT, ACTION_KEEP, ACTION_SELL,
  ACTION_SELL_NO_HOLD, ACTION_SELL_PERCENT, ExecMode, FeeRule, FeeSchedule, RiskConfig, SimConfig,
};
pub use market::MarketSlices;
pub use portfolio::PortfolioEngine;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_sim_accounts_basic_buy_sell_and_unrounded_prices() {
    let prices = [10.33, 10.33, 10.33, 20.45, 20.45, 10.33];
    let ts = [0i64, 1, 2, 3, 4, 5];
    let buy_sig = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    let sell_sig = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0];

    let market = MarketSlices {
      groups: 1,
      bars: 6,
      ts: &ts,
      price: &prices,
      deal_price: &prices,
      fw_price: None,
      limited: None,
      dividend: None,
      transfer_shares: None,
      right_shares: None,
      right_price: None,
    };
    let mut cfg = SimConfig::default();
    cfg.initial_cash = 1500.0;
    cfg.start_bar = 1;

    let mut out_assets = vec![0.0; 6];
    let mut out_ror = vec![0.0; 6];
    let mut out_ror_hold = vec![0.0; 6];
    let mut out_actions = vec![0; 6];
    let mut out_action_shares = vec![0.0; 6];

    let (trades, snap) = sim_accounts(
      &market,
      &cfg,
      &buy_sig,
      &sell_sig,
      &mut out_assets,
      &mut out_ror,
      &mut out_ror_hold,
      &mut out_actions,
      &mut out_action_shares,
    )
    .unwrap();

    assert_eq!(
      out_actions,
      vec![
        ACTION_KEEP,
        ACTION_BUY,
        ACTION_BUY_NO_CASH,
        ACTION_SELL,
        ACTION_SELL_NO_HOLD,
        ACTION_BUY,
      ]
    );
    let expected_assets = [1500.0, 1500.0, 1500.0, 2512.0, 2512.0, 2512.0];
    for (a, e) in out_assets.iter().zip(expected_assets.iter()) {
      assert!((a - e).abs() < 1e-9, "got {a}, expected {e}");
    }
    assert_eq!(snap.shares[0], 200.0);
    assert!((snap.cash[0] - 446.0).abs() < 1e-9);
    assert_eq!(trades.len(), 3); // only actual BUY/SELL fills (bars 1, 3, 5)
  }

  #[test]
  fn test_sim_accounts_corporate_actions_and_fee_and_nan() {
    // Matches test_account_zero_cash_and_fee_consistency semantics:
    // Bar 0: skipped by start_bar=1, Bar 1: buy 100@10 + 5.0 min fee = 1005 -> cash=0
    // Bar 2: right_shares=0.2 @ 8.0 with cash=0 -> affordable=0, cash stays >= 0
    // Bar 3: NaN price (suspended) -> keeps 10.0 mark price, assets stays 1000
    // Bar 4: dividend=0.5, transfer_shares=1.0 (10送10) -> cash += 50, shares = 200, price=5.0 -> assets=1050
    let ts = [0i64, 1, 2, 3, 4];
    let prices = [10.0, 10.0, 10.0, f64::NAN, 5.0];
    let div = [0.0, 0.0, 0.0, 0.0, 0.5];
    let trans = [0.0, 0.0, 0.0, 0.0, 1.0];
    let rs = [0.0, 0.0, 0.2, 0.0, 0.0];
    let rp = [0.0, 0.0, 8.0, 0.0, 0.0];
    let buy_sig = [0.0, 1.0, 1.0, 0.0, 0.0];
    let sell_sig = [0.0; 5];

    let market = MarketSlices {
      groups: 1,
      bars: 5,
      ts: &ts,
      price: &prices,
      deal_price: &prices,
      fw_price: None,
      limited: None,
      dividend: Some(&div),
      transfer_shares: Some(&trans),
      right_shares: Some(&rs),
      right_price: Some(&rp),
    };
    let mut cfg = SimConfig::default();
    cfg.initial_cash = 1005.0;
    cfg.fees = FeeSchedule::new(vec![FeeRule {
      ts: 0,
      commission_buy: 0.001,
      min_commission_buy: 5.0,
      ..Default::default()
    }]);

    let mut out_assets = vec![0.0; 5];
    let mut out_ror = vec![0.0; 5];
    let mut out_ror_hold = vec![0.0; 5];
    let mut out_actions = vec![0; 5];
    let mut out_action_shares = vec![0.0; 5];

    let (_, snap) = sim_accounts(
      &market,
      &cfg,
      &buy_sig,
      &sell_sig,
      &mut out_assets,
      &mut out_ror,
      &mut out_ror_hold,
      &mut out_actions,
      &mut out_action_shares,
    )
    .unwrap();

    assert_eq!(out_actions[1], ACTION_BUY);
    assert_eq!(out_actions[2], ACTION_BUY_NO_CASH);
    assert!((out_assets[3] - 1000.0).abs() < 1e-9);
    assert!((out_assets[4] - 1050.0).abs() < 1e-9);
    assert_eq!(snap.shares[0], 200.0);
    assert!((snap.cash[0] - 50.0).abs() < 1e-9);
    assert!((snap.raw_cost_price[0] - 5.0).abs() < 1e-9);
  }

  #[test]
  fn test_sim_accounts_next_open_and_limit() {
    let ts = [0i64, 1, 2, 3];
    let deal_p = [10.0, 10.0, 10.0, 20.0];
    let close_p = [10.0, 12.0, 15.0, 20.0];
    let buy_sig = [0.0, 1.0, 0.0, 0.0];
    let sell_sig = [0.0, 0.0, 1.0, 0.0];

    let market = MarketSlices {
      groups: 1,
      bars: 4,
      ts: &ts,
      price: &close_p,
      deal_price: &deal_p,
      fw_price: None,
      limited: None,
      dividend: None,
      transfer_shares: None,
      right_shares: None,
      right_price: None,
    };
    let mut cfg = SimConfig::default();
    cfg.initial_cash = 10_000.0;
    cfg.exec_mode = ExecMode::NextOpen;

    let mut out_assets = vec![0.0; 4];
    let mut out_ror = vec![0.0; 4];
    let mut out_ror_hold = vec![0.0; 4];
    let mut out_actions = vec![0; 4];
    let mut out_action_shares = vec![0.0; 4];

    let (_, snap) = sim_accounts(
      &market,
      &cfg,
      &buy_sig,
      &sell_sig,
      &mut out_assets,
      &mut out_ror,
      &mut out_ror_hold,
      &mut out_actions,
      &mut out_action_shares,
    )
    .unwrap();

    // Signal at bar 1 bought at bar 2 open (10.0 -> 1000 shares), marked at bar 2 close (15.0 -> 15000)
    assert!((out_assets[2] - 15_000.0).abs() < 1e-9);
    // Signal at bar 2 sold at bar 3 open (20.0 -> 20000 cash)
    assert!((snap.cash[0] - 20_000.0).abs() < 1e-9);
    assert_eq!(snap.shares[0], 0.0);
  }

  #[test]
  fn test_portfolio_engine_rebalance_and_sparse_corp_action() {
    let dividend = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]; // bar 2 on group 0 has div=1.0
    let mut engine = PortfolioEngine::new(
      2,
      3,
      10_000.0,
      100,
      0.0,
      0.0,
      FeeSchedule::default(),
      vec![true, true],
      Some(&dividend),
      None,
      None,
    )
    .unwrap();

    let mut out_actions = vec![0; 6];
    let mut out_sh = vec![0.0; 6];
    let mut out_cash = vec![0.0; 3];
    let mut out_assets = vec![0.0; 3];
    let mut out_ror = vec![0.0; 3];

    // Bar 1: rebalance 50%/50% at price 10.0 -> 500 shares each, cash=0
    engine
      .rebalance_step(
        1,
        1,
        &[0.5, 0.5],
        &[10.0, 10.0],
        &[10.0, 10.0],
        None,
        None,
        None,
        None,
        None,
        &mut out_actions,
        &mut out_sh,
        &mut out_cash,
        &mut out_assets,
        &mut out_ror,
      )
      .unwrap();
    assert_eq!(engine.shares, vec![500.0, 500.0]);
    assert!((engine.cash - 0.0).abs() < 1e-9);

    // Bar 2: apply corporate action -> group 0 receives 500 * 1.0 = 500 cash
    engine.apply_corporate_actions(2, Some(&dividend), None, None, None);
    assert!((engine.cash - 500.0).abs() < 1e-9);
    engine.update_mark_to_market(2, &[9.0, 10.0], &mut out_cash, &mut out_assets, &mut out_ror);
    assert!((out_assets[2] - 10_000.0).abs() < 1e-9);
  }

  #[test]
  fn test_portfolio_engine_holding_rors_and_step_bar() {
    let prices_all = [10.0, 10.0, 12.0, 10.0, 10.0, 10.0];
    let mut engine = PortfolioEngine::new(
      2,
      3,
      10_000.0,
      100,
      0.0,
      0.0,
      FeeSchedule::default(),
      vec![true, true],
      None,
      None,
      None,
    )
    .unwrap();

    let mut out_actions = vec![0; 6];
    let mut out_sh = vec![0.0; 6];
    let mut out_cash = vec![0.0; 3];
    let mut out_assets = vec![0.0; 3];
    let mut out_ror = vec![0.0; 3];

    let sell_all = [0.0; 6];
    let buy_all = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0];

    engine
      .step_bar(
        1,
        1,
        1,
        false,
        &sell_all,
        &buy_all,
        None,
        None,
        &prices_all,
        &prices_all,
        None,
        None,
        None,
        None,
        &mut out_actions,
        &mut out_sh,
        &mut out_cash,
        &mut out_assets,
        &mut out_ror,
      )
      .unwrap();
    assert_eq!(engine.shares, vec![500.0, 0.0]);

    let mut rors = vec![0.0; 2];
    engine.holding_rors(&prices_all, 2, &mut rors).unwrap();
    assert!((rors[0] - 0.2).abs() < 1e-9);
    assert_eq!(rors[1], 0.0);

    // Explicit sell + rebalance conflict on bar 2
    let sell_conflict = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    engine
      .step_bar(
        2,
        2,
        2,
        false,
        &sell_conflict,
        &[0.0; 6],
        Some(&[0.5, 0.5]),
        None,
        &prices_all,
        &prices_all,
        None,
        None,
        None,
        None,
        &mut out_actions,
        &mut out_sh,
        &mut out_cash,
        &mut out_assets,
        &mut out_ror,
      )
      .unwrap();
    assert_eq!(engine.shares[0], 0.0);
    assert!(engine.shares[1] > 0.0);
  }
}
