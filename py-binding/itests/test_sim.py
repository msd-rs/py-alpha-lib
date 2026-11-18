# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import pytest
import alpha as al
from alpha.sim import AccountBatch, MarketData, PortfolioEngine, SimConfig, Simulator


def test_simulator_basic_and_bool_signals():
  prices = np.array([10.33, 10.33, 10.33, 20.45, 20.45, 10.33])
  ts = np.arange(6, dtype=np.int64)
  buy = np.array([False, True, True, False, False, True])
  sell = np.array([False, False, False, True, True, False])

  with al.Context(groups=1):
    md = MarketData(ts=ts, price=prices)
    cfg = SimConfig(initial_cash=1500.0, start_bar=1)
    res = Simulator(md, cfg).run(buy, sell)

  assert res.actions.tolist() == [0, 1, 5, 2, 6, 1]
  assert np.allclose(res.assets, [1500.0, 1500.0, 1500.0, 2512.0, 2512.0, 2512.0])
  assert res.final_state["shares"][0] == 200.0
  assert np.isclose(res.final_state["cash"][0], 446.0)
  assert len(res.trades["bar"]) == 3


def test_simulator_corporate_actions_and_next_open():
  # 2 groups x 5 bars
  ts = np.tile(np.arange(5, dtype=np.int64), 2)
  close_p = np.tile(np.array([10.0, 12.0, 15.0, 7.5, 8.0]), 2)
  open_p = np.tile(np.array([10.0, 10.0, 10.0, 7.5, 8.0]), 2)
  div = np.tile(np.array([0.0, 0.0, 0.0, 0.5, 0.0]), 2)
  trans = np.tile(np.array([0.0, 0.0, 0.0, 1.0, 0.0]), 2)
  zeros = np.zeros(10, dtype=np.float64)

  buy = np.tile(np.array([False, True, False, False, False]), 2)
  sell = np.tile(np.array([False, False, False, False, False]), 2)

  md = MarketData(
    ts=ts,
    price=close_p,
    deal_price=open_p,
    corp_actions=(div, trans, zeros, zeros),
    groups=2,
  )
  cfg = SimConfig(initial_cash=10_000.0, exec_mode="next_open")
  res = Simulator(md, cfg).run(buy, sell)

  # Bought at bar 2 open=10.0 -> 1000 shares, bar 2 close=15.0 -> 15000
  assert np.isclose(res.assets[2], 15_000.0)
  # Bar 3: div=0.5 (+500 cash), trans=1.0 (2000 shares), price=7.5 -> 2000*7.5 + 500 = 15500
  assert np.isclose(res.assets[3], 15_500.0)
  assert np.isclose(res.assets[8], 15_500.0)


def test_simulator_fail_fast_validation():
  with pytest.raises(ValueError, match="divisible by groups"):
    MarketData(ts=np.arange(5), price=np.ones(5), groups=2)


def test_account_batch_step_by_step():
  ts = np.arange(5, dtype=np.int64)
  prices = np.array([10.0, 10.0, 12.0, 9.0, 11.0])
  md = MarketData(ts=ts, price=prices, groups=1)
  cfg = SimConfig(initial_cash=1000.0)
  batch = AccountBatch(md, cfg)

  buy = np.zeros(5, dtype=np.float64)
  sell = np.zeros(5, dtype=np.float64)
  out_assets = np.full(5, 1000.0, dtype=np.float64)
  out_ror = np.zeros(5, dtype=np.float64)
  out_rh = np.zeros(5, dtype=np.float64)
  out_act = np.zeros(5, dtype=np.int32)
  out_sh = np.zeros(5, dtype=np.float64)

  buy[1] = 1.0
  batch.step_bar(
    1, buy, sell,
    out_assets=out_assets, out_ror=out_ror, out_ror_hold=out_rh,
    out_actions=out_act, out_action_shares=out_sh,
  )
  assert batch.shares()[0] == 100.0
  assert np.isclose(batch.holding_rors(2)[0], 0.2)


def test_portfolio_engine_basic():
  eng = PortfolioEngine(2, 3, np.array([True, True]), initial_cash=10_000.0)
  out_act = np.zeros(6, dtype=np.int32)
  out_sh = np.zeros(6, dtype=np.float64)
  out_cash = np.zeros(3, dtype=np.float64)
  out_assets = np.zeros(3, dtype=np.float64)
  out_ror = np.zeros(3, dtype=np.float64)

  eng.rebalance_step(
    1,
    1,
    np.array([0.5, 0.5]),
    np.array([10.0, 10.0]),
    np.array([10.0, 10.0]),
    out_act,
    out_sh,
    out_cash,
    out_assets,
    out_ror,
  )
  assert eng.shares().tolist() == [500.0, 500.0]
  assert np.isclose(out_assets[1], 10_000.0)
