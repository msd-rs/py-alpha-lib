// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use alpha_algo::sim::{
  AccountBatch, AccountSnapshot, ExecMode, FeeRule, FeeSchedule, MarketSlices, PortfolioEngine,
  RiskConfig, SimConfig, TradeEvents, sim_accounts,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn parse_exec_mode(mode: &str) -> PyResult<ExecMode> {
  match mode {
    "close" => Ok(ExecMode::Close),
    "next_open" => Ok(ExecMode::NextOpen),
    _ => Err(PyValueError::new_err(format!(
      "unsupported exec_mode: '{mode}', expected 'close' or 'next_open'"
    ))),
  }
}

fn get_f64_field(obj: &Bound<'_, PyAny>, name: &str, default: f64) -> PyResult<f64> {
  if let Ok(dict) = obj.cast::<PyDict>() {
    match dict.get_item(name)? {
      Some(v) => v.extract::<f64>(),
      None => Ok(default),
    }
  } else {
    match obj.getattr(name) {
      Ok(v) => v.extract::<f64>(),
      Err(_) => Ok(default),
    }
  }
}

fn get_i64_field(obj: &Bound<'_, PyAny>, name: &str, default: i64) -> PyResult<i64> {
  if let Ok(dict) = obj.cast::<PyDict>() {
    match dict.get_item(name)? {
      Some(v) => v.extract::<i64>(),
      None => Ok(default),
    }
  } else {
    match obj.getattr(name) {
      Ok(v) => v.extract::<i64>(),
      Err(_) => Ok(default),
    }
  }
}

fn parse_fee_schedule(fees: Option<&Bound<'_, PyAny>>) -> PyResult<FeeSchedule> {
  let Some(fees_obj) = fees else {
    return Ok(FeeSchedule::default());
  };
  let mut rules = Vec::new();
  for item in fees_obj.try_iter()? {
    let item = item?;
    rules.push(FeeRule {
      ts: get_i64_field(&item, "ts", 0)?,
      commission_buy: get_f64_field(&item, "commission_buy", 0.0)?,
      min_commission_buy: get_f64_field(&item, "min_commission_buy", 0.0)?,
      tax_buy: get_f64_field(&item, "tax_buy", 0.0)?,
      transaction_fee_buy: get_f64_field(&item, "transaction_fee_buy", 0.0)?,
      commission_sell: get_f64_field(&item, "commission_sell", 0.0)?,
      min_commission_sell: get_f64_field(&item, "min_commission_sell", 0.0)?,
      tax_sell: get_f64_field(&item, "tax_sell", 0.0)?,
      transaction_fee_sell: get_f64_field(&item, "transaction_fee_sell", 0.0)?,
    });
  }
  Ok(FeeSchedule::new(rules))
}

fn trades_to_pydict<'py>(py: Python<'py>, trades: TradeEvents) -> PyResult<Bound<'py, PyDict>> {
  let d = PyDict::new(py);
  d.set_item("group", trades.group.into_pyarray(py))?;
  d.set_item("bar", trades.bar.into_pyarray(py))?;
  d.set_item("ts", trades.ts.into_pyarray(py))?;
  d.set_item("action", trades.action.into_pyarray(py))?;
  d.set_item("shares", trades.shares.into_pyarray(py))?;
  d.set_item("deal_price", trades.deal_price.into_pyarray(py))?;
  d.set_item("fee", trades.fee.into_pyarray(py))?;
  d.set_item("cash_after", trades.cash_after.into_pyarray(py))?;
  d.set_item("pos_shares", trades.pos_shares.into_pyarray(py))?;
  d.set_item("raw_cost_price", trades.raw_cost_price.into_pyarray(py))?;
  Ok(d)
}

fn snapshot_to_pydict<'py>(py: Python<'py>, snap: AccountSnapshot) -> PyResult<Bound<'py, PyDict>> {
  let d = PyDict::new(py);
  d.set_item("cash", snap.cash.into_pyarray(py))?;
  d.set_item("shares", snap.shares.into_pyarray(py))?;
  d.set_item("raw_cost_price", snap.raw_cost_price.into_pyarray(py))?;
  d.set_item("raw_mark_price", snap.raw_mark_price.into_pyarray(py))?;
  d.set_item("adj_cost_price", snap.adj_cost_price.into_pyarray(py))?;
  d.set_item("dividends", snap.dividends.into_pyarray(py))?;
  d.set_item("per_assets", snap.per_assets.into_pyarray(py))?;
  d.set_item("deducted", snap.deducted.into_pyarray(py))?;
  d.set_item("holding_bars", snap.holding_bars.into_pyarray(py))?;
  Ok(d)
}

#[allow(clippy::too_many_arguments)]
fn build_sim_config(
  initial_cash: f64,
  unit: u64,
  slippage: f64,
  warmup_bars: usize,
  start_bar: usize,
  exec_mode: &str,
  check_limit: bool,
  fees: Option<&Bound<'_, PyAny>>,
  stop_loss: Option<f64>,
  take_profit: Option<f64>,
  trailing_stop: Option<f64>,
  max_holding_days: Option<u32>,
) -> PyResult<SimConfig> {
  Ok(SimConfig {
    initial_cash,
    unit: unit.max(1),
    slippage,
    warmup_bars,
    start_bar,
    exec_mode: parse_exec_mode(exec_mode)?,
    check_limit,
    fees: parse_fee_schedule(fees)?,
    risk: RiskConfig {
      stop_loss,
      take_profit,
      trailing_stop,
      max_holding_days,
    },
  })
}

#[pyfunction]
#[pyo3(signature = (
  groups,
  bars,
  ts,
  price,
  deal_price,
  buy_percent,
  sell_percent,
  out_assets,
  out_ror,
  out_ror_hold,
  out_actions,
  out_action_shares,
  *,
  fw_price = None,
  limited = None,
  dividend = None,
  transfer_shares = None,
  right_shares = None,
  right_price = None,
  initial_cash = 100000.0,
  unit = 100,
  slippage = 0.0,
  warmup_bars = 0,
  start_bar = 1,
  exec_mode = "close",
  check_limit = true,
  fees = None,
  stop_loss = None,
  take_profit = None,
  trailing_stop = None,
  max_holding_days = None,
))]
#[allow(clippy::too_many_arguments)]
pub fn sim_accounts_into<'py>(
  py: Python<'py>,
  groups: usize,
  bars: usize,
  ts: PyReadonlyArray1<'py, i64>,
  price: PyReadonlyArray1<'py, f64>,
  deal_price: PyReadonlyArray1<'py, f64>,
  buy_percent: PyReadonlyArray1<'py, f64>,
  sell_percent: PyReadonlyArray1<'py, f64>,
  mut out_assets: PyReadwriteArray1<'py, f64>,
  mut out_ror: PyReadwriteArray1<'py, f64>,
  mut out_ror_hold: PyReadwriteArray1<'py, f64>,
  mut out_actions: PyReadwriteArray1<'py, i32>,
  mut out_action_shares: PyReadwriteArray1<'py, f64>,
  fw_price: Option<PyReadonlyArray1<'py, f64>>,
  limited: Option<PyReadonlyArray1<'py, i32>>,
  dividend: Option<PyReadonlyArray1<'py, f64>>,
  transfer_shares: Option<PyReadonlyArray1<'py, f64>>,
  right_shares: Option<PyReadonlyArray1<'py, f64>>,
  right_price: Option<PyReadonlyArray1<'py, f64>>,
  initial_cash: f64,
  unit: u64,
  slippage: f64,
  warmup_bars: usize,
  start_bar: usize,
  exec_mode: &str,
  check_limit: bool,
  fees: Option<&Bound<'py, PyAny>>,
  stop_loss: Option<f64>,
  take_profit: Option<f64>,
  trailing_stop: Option<f64>,
  max_holding_days: Option<u32>,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
  let cfg = build_sim_config(
    initial_cash,
    unit,
    slippage,
    warmup_bars,
    start_bar,
    exec_mode,
    check_limit,
    fees,
    stop_loss,
    take_profit,
    trailing_stop,
    max_holding_days,
  )?;

  let ts_s = ts.as_slice()?;
  let price_s = price.as_slice()?;
  let deal_s = deal_price.as_slice()?;
  let buy_s = buy_percent.as_slice()?;
  let sell_s = sell_percent.as_slice()?;

  let fw_s = fw_price.as_ref().map(|a| a.as_slice()).transpose()?;
  let lim_s = limited.as_ref().map(|a| a.as_slice()).transpose()?;
  let div_s = dividend.as_ref().map(|a| a.as_slice()).transpose()?;
  let trans_s = transfer_shares.as_ref().map(|a| a.as_slice()).transpose()?;
  let rs_s = right_shares.as_ref().map(|a| a.as_slice()).transpose()?;
  let rp_s = right_price.as_ref().map(|a| a.as_slice()).transpose()?;

  let oa = out_assets.as_slice_mut()?;
  let oror = out_ror.as_slice_mut()?;
  let orh = out_ror_hold.as_slice_mut()?;
  let oact = out_actions.as_slice_mut()?;
  let osh = out_action_shares.as_slice_mut()?;

  let market = MarketSlices {
    groups,
    bars,
    ts: ts_s,
    price: price_s,
    deal_price: deal_s,
    fw_price: fw_s,
    limited: lim_s,
    dividend: div_s,
    transfer_shares: trans_s,
    right_shares: rs_s,
    right_price: rp_s,
  };

  let (trades, snap) = py
    .detach(|| sim_accounts(&market, &cfg, buy_s, sell_s, oa, oror, orh, oact, osh))
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

  Ok((trades_to_pydict(py, trades)?, snapshot_to_pydict(py, snap)?))
}

#[pyclass(name = "_AccountBatch")]
pub struct PyAccountBatch {
  inner: AccountBatch,
}

#[pymethods]
impl PyAccountBatch {
  #[new]
  #[pyo3(signature = (
    groups,
    bars,
    *,
    initial_cash = 100000.0,
    unit = 100,
    slippage = 0.0,
    warmup_bars = 0,
    start_bar = 1,
    exec_mode = "close",
    check_limit = true,
    fees = None,
    stop_loss = None,
    take_profit = None,
    trailing_stop = None,
    max_holding_days = None,
  ))]
  #[allow(clippy::too_many_arguments)]
  fn new(
    groups: usize,
    bars: usize,
    initial_cash: f64,
    unit: u64,
    slippage: f64,
    warmup_bars: usize,
    start_bar: usize,
    exec_mode: &str,
    check_limit: bool,
    fees: Option<&Bound<'_, PyAny>>,
    stop_loss: Option<f64>,
    take_profit: Option<f64>,
    trailing_stop: Option<f64>,
    max_holding_days: Option<u32>,
  ) -> PyResult<Self> {
    let cfg = build_sim_config(
      initial_cash,
      unit,
      slippage,
      warmup_bars,
      start_bar,
      exec_mode,
      check_limit,
      fees,
      stop_loss,
      take_profit,
      trailing_stop,
      max_holding_days,
    )?;
    let inner = AccountBatch::new(groups, bars, cfg)
      .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(Self { inner })
  }

  #[pyo3(signature = (
    bar,
    ts,
    price,
    deal_price,
    buy_percent,
    sell_percent,
    out_assets,
    out_ror,
    out_ror_hold,
    out_actions,
    out_action_shares,
    *,
    fw_price = None,
    limited = None,
    dividend = None,
    transfer_shares = None,
    right_shares = None,
    right_price = None,
  ))]
  #[allow(clippy::too_many_arguments)]
  fn step_bar<'py>(
    &mut self,
    py: Python<'py>,
    bar: usize,
    ts: PyReadonlyArray1<'py, i64>,
    price: PyReadonlyArray1<'py, f64>,
    deal_price: PyReadonlyArray1<'py, f64>,
    buy_percent: PyReadonlyArray1<'py, f64>,
    sell_percent: PyReadonlyArray1<'py, f64>,
    mut out_assets: PyReadwriteArray1<'py, f64>,
    mut out_ror: PyReadwriteArray1<'py, f64>,
    mut out_ror_hold: PyReadwriteArray1<'py, f64>,
    mut out_actions: PyReadwriteArray1<'py, i32>,
    mut out_action_shares: PyReadwriteArray1<'py, f64>,
    fw_price: Option<PyReadonlyArray1<'py, f64>>,
    limited: Option<PyReadonlyArray1<'py, i32>>,
    dividend: Option<PyReadonlyArray1<'py, f64>>,
    transfer_shares: Option<PyReadonlyArray1<'py, f64>>,
    right_shares: Option<PyReadonlyArray1<'py, f64>>,
    right_price: Option<PyReadonlyArray1<'py, f64>>,
  ) -> PyResult<()> {
    let ts_s = ts.as_slice()?;
    let price_s = price.as_slice()?;
    let deal_s = deal_price.as_slice()?;
    let buy_s = buy_percent.as_slice()?;
    let sell_s = sell_percent.as_slice()?;

    let fw_s = fw_price.as_ref().map(|a| a.as_slice()).transpose()?;
    let lim_s = limited.as_ref().map(|a| a.as_slice()).transpose()?;
    let div_s = dividend.as_ref().map(|a| a.as_slice()).transpose()?;
    let trans_s = transfer_shares.as_ref().map(|a| a.as_slice()).transpose()?;
    let rs_s = right_shares.as_ref().map(|a| a.as_slice()).transpose()?;
    let rp_s = right_price.as_ref().map(|a| a.as_slice()).transpose()?;

    let oa = out_assets.as_slice_mut()?;
    let oror = out_ror.as_slice_mut()?;
    let orh = out_ror_hold.as_slice_mut()?;
    let oact = out_actions.as_slice_mut()?;
    let osh = out_action_shares.as_slice_mut()?;

    let market = MarketSlices {
      groups: self.inner.groups,
      bars: self.inner.bars,
      ts: ts_s,
      price: price_s,
      deal_price: deal_s,
      fw_price: fw_s,
      limited: lim_s,
      dividend: div_s,
      transfer_shares: trans_s,
      right_shares: rs_s,
      right_price: rp_s,
    };

    py.detach(|| {
      self
        .inner
        .step_bar(bar, &market, buy_s, sell_s, oa, oror, orh, oact, osh)
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
  }

  #[pyo3(signature = (
    bar,
    *,
    dividend = None,
    transfer_shares = None,
    right_shares = None,
    right_price = None,
    raw_prices = None,
    fw_prices = None,
  ))]
  fn apply_corporate_actions_bar<'py>(
    &mut self,
    bar: usize,
    dividend: Option<PyReadonlyArray1<'py, f64>>,
    transfer_shares: Option<PyReadonlyArray1<'py, f64>>,
    right_shares: Option<PyReadonlyArray1<'py, f64>>,
    right_price: Option<PyReadonlyArray1<'py, f64>>,
    raw_prices: Option<PyReadonlyArray1<'py, f64>>,
    fw_prices: Option<PyReadonlyArray1<'py, f64>>,
  ) -> PyResult<()> {
    let div_s = dividend.as_ref().map(|a| a.as_slice()).transpose()?;
    let trans_s = transfer_shares.as_ref().map(|a| a.as_slice()).transpose()?;
    let rs_s = right_shares.as_ref().map(|a| a.as_slice()).transpose()?;
    let rp_s = right_price.as_ref().map(|a| a.as_slice()).transpose()?;
    let raw_s = raw_prices.as_ref().map(|a| a.as_slice()).transpose()?;
    let fw_s = fw_prices.as_ref().map(|a| a.as_slice()).transpose()?;
    self
      .inner
      .apply_corporate_actions_bar(bar, div_s, trans_s, rs_s, rp_s, raw_s, fw_s)
      .map_err(|e| PyValueError::new_err(e.to_string()))
  }

  fn holding_rors<'py>(
    &self,
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    bar: usize,
  ) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mut out = vec![0.0; self.inner.groups];
    let p = price.as_slice()?;
    self
      .inner
      .holding_rors(p, bar, &mut out)
      .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray(py))
  }

  fn shares<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    let v: Vec<f64> = self.inner.states.iter().map(|s| s.shares).collect();
    v.into_pyarray(py)
  }

  fn holding_bars<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
    let v: Vec<i32> = self.inner.states.iter().map(|s| s.holding_bars).collect();
    v.into_pyarray(py)
  }

  fn state_of(&self, g: usize) -> PyResult<(f64, f64, f64, f64, f64, f64, i32)> {
    let st = self.inner.states.get(g).ok_or_else(|| {
      PyValueError::new_err(format!(
        "group index {g} out of range [0, {})",
        self.inner.groups
      ))
    })?;
    Ok((
      st.cash,
      st.shares,
      st.raw_cost_price,
      st.raw_mark_price,
      st.adj_cost_price,
      st.dividends,
      st.holding_bars,
    ))
  }

  fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    snapshot_to_pydict(py, self.inner.snapshot())
  }

  fn trades<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    trades_to_pydict(py, self.inner.trades.clone())
  }
}

#[pyclass(name = "_PortfolioEngine")]
pub struct PyPortfolioEngine {
  inner: PortfolioEngine,
}

#[pymethods]
impl PyPortfolioEngine {
  #[new]
  #[pyo3(signature = (
    groups,
    bars,
    tradable_mask,
    *,
    initial_cash = 100000.0,
    unit = 100,
    slippage = 0.0,
    tolerance = 0.0,
    fees = None,
    dividend = None,
    transfer_shares = None,
    right_shares = None,
  ))]
  #[allow(clippy::too_many_arguments)]
  fn new(
    groups: usize,
    bars: usize,
    tradable_mask: PyReadonlyArray1<'_, bool>,
    initial_cash: f64,
    unit: u64,
    slippage: f64,
    tolerance: f64,
    fees: Option<&Bound<'_, PyAny>>,
    dividend: Option<PyReadonlyArray1<'_, f64>>,
    transfer_shares: Option<PyReadonlyArray1<'_, f64>>,
    right_shares: Option<PyReadonlyArray1<'_, f64>>,
  ) -> PyResult<Self> {
    let mask = tradable_mask.as_slice()?.to_vec();
    let fee_sched = parse_fee_schedule(fees)?;
    let div_s = dividend.as_ref().map(|a| a.as_slice()).transpose()?;
    let trans_s = transfer_shares.as_ref().map(|a| a.as_slice()).transpose()?;
    let rs_s = right_shares.as_ref().map(|a| a.as_slice()).transpose()?;

    let inner = PortfolioEngine::new(
      groups,
      bars,
      initial_cash,
      unit,
      slippage,
      tolerance,
      fee_sched,
      mask,
      div_s,
      trans_s,
      rs_s,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(Self { inner })
  }

  #[getter]
  fn cash(&self) -> f64 {
    self.inner.cash
  }

  #[getter]
  fn deducted(&self) -> f64 {
    self.inner.deducted
  }

  #[getter]
  fn market_value(&self) -> f64 {
    self.inner.market_value()
  }

  #[getter]
  fn asset(&self) -> f64 {
    self.inner.asset()
  }

  fn set_tradable_mask(&mut self, tradable_mask: PyReadonlyArray1<'_, bool>) -> PyResult<()> {
    let mask = tradable_mask.as_slice()?.to_vec();
    self
      .inner
      .set_tradable_mask(mask)
      .map_err(|e| PyValueError::new_err(e.to_string()))
  }

  fn state_of(&self, g: usize) -> PyResult<(f64, f64, f64, f64, f64, i64)> {
    self
      .inner
      .state_of(g)
      .map_err(|e| PyValueError::new_err(e.to_string()))
  }

  fn holding_rors<'py>(
    &self,
    py: Python<'py>,
    price: PyReadonlyArray1<'py, f64>,
    bar: usize,
  ) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mut out = vec![0.0; self.inner.groups];
    let p = price.as_slice()?;
    py.detach(|| self.inner.holding_rors(p, bar, &mut out))
      .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray(py))
  }

  fn shares<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    self.inner.shares.clone().into_pyarray(py)
  }

  fn raw_cost_price<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    self.inner.raw_cost_price.clone().into_pyarray(py)
  }

  fn adj_cost_price<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    self.inner.adj_cost_price.clone().into_pyarray(py)
  }

  fn raw_mark_price<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    self.inner.raw_mark_price.clone().into_pyarray(py)
  }

  fn dividends<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
    self.inner.dividends.clone().into_pyarray(py)
  }

  fn entry_bars<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
    self.inner.entry_bars.clone().into_pyarray(py)
  }

  #[pyo3(signature = (
    bar,
    dividend = None,
    transfer_shares = None,
    right_shares = None,
    right_price = None,
    raw_prices = None,
    fw_prices = None,
  ))]
  fn apply_corporate_actions<'py>(
    &mut self,
    bar: usize,
    dividend: Option<PyReadonlyArray1<'py, f64>>,
    transfer_shares: Option<PyReadonlyArray1<'py, f64>>,
    right_shares: Option<PyReadonlyArray1<'py, f64>>,
    right_price: Option<PyReadonlyArray1<'py, f64>>,
    raw_prices: Option<PyReadonlyArray1<'py, f64>>,
    fw_prices: Option<PyReadonlyArray1<'py, f64>>,
  ) -> PyResult<()> {
    let div_s = dividend.as_ref().map(|a| a.as_slice()).transpose()?;
    let trans_s = transfer_shares.as_ref().map(|a| a.as_slice()).transpose()?;
    let rs_s = right_shares.as_ref().map(|a| a.as_slice()).transpose()?;
    let rp_s = right_price.as_ref().map(|a| a.as_slice()).transpose()?;
    let raw_s = raw_prices.as_ref().map(|a| a.as_slice()).transpose()?;
    let fw_s = fw_prices.as_ref().map(|a| a.as_slice()).transpose()?;
    self
      .inner
      .apply_corporate_actions(bar, div_s, trans_s, rs_s, rp_s, raw_s, fw_s);
    Ok(())
  }

  #[allow(clippy::too_many_arguments)]
  fn update_mark_to_market<'py>(
    &mut self,
    bar: usize,
    mark_prices: PyReadonlyArray1<'py, f64>,
    mut out_cash: PyReadwriteArray1<'py, f64>,
    mut out_assets: PyReadwriteArray1<'py, f64>,
    mut out_ror: PyReadwriteArray1<'py, f64>,
  ) -> PyResult<()> {
    let mp = mark_prices.as_slice()?;
    let oc = out_cash.as_slice_mut()?;
    let oa = out_assets.as_slice_mut()?;
    let oror = out_ror.as_slice_mut()?;
    self.inner.update_mark_to_market(bar, mp, oc, oa, oror);
    Ok(())
  }

  #[allow(clippy::too_many_arguments)]
  fn update_mark_to_market_strided<'py>(
    &mut self,
    bar: usize,
    mark_prices_all: PyReadonlyArray1<'py, f64>,
    mut out_cash: PyReadwriteArray1<'py, f64>,
    mut out_assets: PyReadwriteArray1<'py, f64>,
    mut out_ror: PyReadwriteArray1<'py, f64>,
  ) -> PyResult<()> {
    let mp = mark_prices_all.as_slice()?;
    let oc = out_cash.as_slice_mut()?;
    let oa = out_assets.as_slice_mut()?;
    let oror = out_ror.as_slice_mut()?;
    self
      .inner
      .update_mark_to_market_strided(bar, mp, oc, oa, oror);
    Ok(())
  }

  #[pyo3(signature = (
    bar,
    sig_i,
    ts_val,
    skip_trade,
    sell_percents_all,
    buy_percents_all,
    deal_prices_all,
    mark_prices_all,
    out_actions,
    out_action_shares,
    out_cash,
    out_assets,
    out_ror,
    *,
    target_weights_bar = None,
    tolerance = None,
    limited_all = None,
    universe_mask = None,
    raw_prices_all = None,
    fw_prices_all = None,
  ))]
  #[allow(clippy::too_many_arguments)]
  fn step_bar<'py>(
    &mut self,
    py: Python<'py>,
    bar: usize,
    sig_i: usize,
    ts_val: i64,
    skip_trade: bool,
    sell_percents_all: PyReadonlyArray1<'py, f64>,
    buy_percents_all: PyReadonlyArray1<'py, f64>,
    deal_prices_all: PyReadonlyArray1<'py, f64>,
    mark_prices_all: PyReadonlyArray1<'py, f64>,
    mut out_actions: PyReadwriteArray1<'py, i32>,
    mut out_action_shares: PyReadwriteArray1<'py, f64>,
    mut out_cash: PyReadwriteArray1<'py, f64>,
    mut out_assets: PyReadwriteArray1<'py, f64>,
    mut out_ror: PyReadwriteArray1<'py, f64>,
    target_weights_bar: Option<PyReadonlyArray1<'py, f64>>,
    tolerance: Option<f64>,
    limited_all: Option<PyReadonlyArray1<'py, i32>>,
    universe_mask: Option<PyReadonlyArray1<'py, bool>>,
    raw_prices_all: Option<PyReadonlyArray1<'py, f64>>,
    fw_prices_all: Option<PyReadonlyArray1<'py, f64>>,
  ) -> PyResult<()> {
    let sp = sell_percents_all.as_slice()?;
    let bp = buy_percents_all.as_slice()?;
    let dp = deal_prices_all.as_slice()?;
    let mp = mark_prices_all.as_slice()?;
    let tw = target_weights_bar.as_ref().map(|a| a.as_slice()).transpose()?;
    let lim = limited_all.as_ref().map(|a| a.as_slice()).transpose()?;
    let umask = universe_mask.as_ref().map(|a| a.as_slice()).transpose()?;
    let rpa = raw_prices_all.as_ref().map(|a| a.as_slice()).transpose()?;
    let fpa = fw_prices_all.as_ref().map(|a| a.as_slice()).transpose()?;

    let oact = out_actions.as_slice_mut()?;
    let osh = out_action_shares.as_slice_mut()?;
    let oc = out_cash.as_slice_mut()?;
    let oa = out_assets.as_slice_mut()?;
    let oror = out_ror.as_slice_mut()?;

    py.detach(|| {
      self.inner.step_bar(
        bar,
        sig_i,
        ts_val,
        skip_trade,
        sp,
        bp,
        tw,
        tolerance,
        dp,
        mp,
        lim,
        umask,
        rpa,
        fpa,
        oact,
        osh,
        oc,
        oa,
        oror,
      )
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
  }

  #[pyo3(signature = (
    bar,
    ts_val,
    target_weights,
    deal_prices,
    mark_prices,
    out_actions,
    out_action_shares,
    out_cash,
    out_assets,
    out_ror,
    *,
    limited = None,
    tolerance = None,
    universe_mask = None,
    raw_prices_all = None,
    fw_prices_all = None,
  ))]
  #[allow(clippy::too_many_arguments)]
  fn rebalance_step<'py>(
    &mut self,
    bar: usize,
    ts_val: i64,
    target_weights: PyReadonlyArray1<'py, f64>,
    deal_prices: PyReadonlyArray1<'py, f64>,
    mark_prices: PyReadonlyArray1<'py, f64>,
    mut out_actions: PyReadwriteArray1<'py, i32>,
    mut out_action_shares: PyReadwriteArray1<'py, f64>,
    mut out_cash: PyReadwriteArray1<'py, f64>,
    mut out_assets: PyReadwriteArray1<'py, f64>,
    mut out_ror: PyReadwriteArray1<'py, f64>,
    limited: Option<PyReadonlyArray1<'py, i32>>,
    tolerance: Option<f64>,
    universe_mask: Option<PyReadonlyArray1<'py, bool>>,
    raw_prices_all: Option<PyReadonlyArray1<'py, f64>>,
    fw_prices_all: Option<PyReadonlyArray1<'py, f64>>,
  ) -> PyResult<()> {
    let tw = target_weights.as_slice()?;
    let dp = deal_prices.as_slice()?;
    let mp = mark_prices.as_slice()?;
    let lim = limited.as_ref().map(|a| a.as_slice()).transpose()?;
    let umask = universe_mask.as_ref().map(|a| a.as_slice()).transpose()?;
    let rpa = raw_prices_all.as_ref().map(|a| a.as_slice()).transpose()?;
    let fpa = fw_prices_all.as_ref().map(|a| a.as_slice()).transpose()?;

    let oact = out_actions.as_slice_mut()?;
    let osh = out_action_shares.as_slice_mut()?;
    let oc = out_cash.as_slice_mut()?;
    let oa = out_assets.as_slice_mut()?;
    let oror = out_ror.as_slice_mut()?;

    self
      .inner
      .rebalance_step(
        bar, ts_val, tw, dp, mp, lim, tolerance, umask, rpa, fpa, oact, osh, oc, oa, oror,
      )
      .map_err(|e| PyValueError::new_err(e.to_string()))
  }

  fn sell_group<'py>(
    &mut self,
    g: usize,
    bar: usize,
    ts_val: i64,
    shares_to_sell: f64,
    deal_price: f64,
    mut out_actions: PyReadwriteArray1<'py, i32>,
    mut out_action_shares: PyReadwriteArray1<'py, f64>,
  ) -> PyResult<()> {
    let oact = out_actions.as_slice_mut()?;
    let osh = out_action_shares.as_slice_mut()?;
    self
      .inner
      .sell_group(g, bar, ts_val, shares_to_sell, deal_price, oact, osh);
    Ok(())
  }

  #[pyo3(signature = (
    g,
    bar,
    ts_val,
    cash_budget,
    deal_price,
    out_actions,
    out_action_shares,
    *,
    raw_p = None,
    fw_p = None,
  ))]
  #[allow(clippy::too_many_arguments)]
  fn buy_group<'py>(
    &mut self,
    g: usize,
    bar: usize,
    ts_val: i64,
    cash_budget: f64,
    deal_price: f64,
    mut out_actions: PyReadwriteArray1<'py, i32>,
    mut out_action_shares: PyReadwriteArray1<'py, f64>,
    raw_p: Option<f64>,
    fw_p: Option<f64>,
  ) -> PyResult<()> {
    let oact = out_actions.as_slice_mut()?;
    let osh = out_action_shares.as_slice_mut()?;
    self.inner.buy_group(
      g, bar, ts_val, cash_budget, deal_price, raw_p, fw_p, oact, osh,
    );
    Ok(())
  }

  fn trades<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    trades_to_pydict(py, self.inner.trades.clone())
  }
}

pub fn register_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
  m.add_function(wrap_pyfunction!(sim_accounts_into, m)?)?;
  m.add_class::<PyAccountBatch>()?;
  m.add_class::<PyPortfolioEngine>()?;
  Ok(())
}
