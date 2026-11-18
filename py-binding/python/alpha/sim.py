# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""High-performance Rust simulation engine for vectorized and bar-by-bar backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from .algo import _algo
from .algo._algo import get_ctx, _AccountBatch, _PortfolioEngine

__all__ = [
  "MarketData",
  "SimConfig",
  "SimResult",
  "Simulator",
  "AccountBatch",
  "PortfolioEngine",
]


def _to_contiguous(arr: Any, dtype: np.dtype | type) -> np.ndarray:
  a = np.asarray(arr, dtype=dtype)
  if not a.flags.c_contiguous:
    a = np.ascontiguousarray(a, dtype=dtype)
  return a


def _to_signal_f64(arr: Any, expected_len: int, name: str) -> np.ndarray:
  a = np.asarray(arr)
  if a.ndim != 1 or len(a) != expected_len:
    raise ValueError(
      f"{name} must be a 1D array of length {expected_len}, got shape {a.shape}"
    )
  if a.dtype == np.bool_:
    return np.where(a, 1.0, 0.0)
  return _to_contiguous(a, np.float64)


def _to_ts_i64(arr: Any) -> np.ndarray:
  a = np.asarray(arr)
  if np.issubdtype(a.dtype, np.datetime64):
    a = a.astype("datetime64[s]").astype(np.int64)
  elif a.dtype != np.int64:
    a = a.astype(np.int64)
  if not a.flags.c_contiguous:
    a = np.ascontiguousarray(a, dtype=np.int64)
  return a


@dataclass
class MarketData:
  """Validated flat market and corporate-action data container for Rust simulation.

  All arrays use the standard `groups * bars` row-major block layout.
  """

  ts: np.ndarray
  price: np.ndarray
  deal_price: np.ndarray | None = None
  fw_price: np.ndarray | None = None
  limited: np.ndarray | None = None
  corp_actions: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
  groups: int | None = None

  def __post_init__(self):
    if self.groups is None:
      self.groups = max(1, int(get_ctx()[2]))
    else:
      self.groups = int(self.groups)
    if self.groups <= 0:
      raise ValueError(f"groups must be positive, got {self.groups}")

    self.ts = _to_ts_i64(self.ts)
    self.price = _to_contiguous(self.price, np.float64)
    total = len(self.price)
    if total == 0 or total % self.groups != 0:
      raise ValueError(
        f"price length ({total}) must be positive and divisible by groups ({self.groups})"
      )
    self.bars = total // self.groups

    if len(self.ts) != total:
      raise ValueError(f"ts length ({len(self.ts)}) != price length ({total})")

    if self.deal_price is None:
      self.deal_price = self.price
    else:
      self.deal_price = _to_contiguous(self.deal_price, np.float64)
      if len(self.deal_price) != total:
        raise ValueError(
          f"deal_price length ({len(self.deal_price)}) != expected ({total})"
        )

    if self.fw_price is not None:
      self.fw_price = _to_contiguous(self.fw_price, np.float64)
      if len(self.fw_price) != total:
        raise ValueError(
          f"fw_price length ({len(self.fw_price)}) != expected ({total})"
        )

    if self.limited is not None:
      lim_arr = np.asarray(self.limited)
      if np.issubdtype(lim_arr.dtype, np.floating):
        lim_arr = np.nan_to_num(lim_arr, nan=0.0).astype(np.int32)
      self.limited = _to_contiguous(lim_arr, np.int32)
      if len(self.limited) != total:
        raise ValueError(
          f"limited length ({len(self.limited)}) != expected ({total})"
        )

    if self.corp_actions is not None:
      if len(self.corp_actions) != 4:
        raise ValueError(
          "corp_actions must be a 4-tuple of (dividend, transfer_shares, right_shares, right_price)"
        )
      div, trans, rs, rp = (
        _to_contiguous(x, np.float64) for x in self.corp_actions
      )
      for name, arr in (
        ("dividend", div),
        ("transfer_shares", trans),
        ("right_shares", rs),
        ("right_price", rp),
      ):
        if len(arr) != total:
          raise ValueError(f"{name} length ({len(arr)}) != expected ({total})")
      self.corp_actions = (div, trans, rs, rp)

  @classmethod
  def from_dict(
    cls,
    data: dict[str, np.ndarray],
    *,
    groups: int | None = None,
    exec_mode: str = "close",
    check_limit: bool = True,
  ) -> "MarketData":
    """Build MarketData directly from a DataProvider-style dict."""
    if "ts" not in data or "price" not in data:
      raise ValueError("data must contain 'ts' and 'price'")
    if exec_mode == "next_open":
      deal_p = data.get("raw_open", data.get("open", data["price"]))
    else:
      deal_p = data["price"]
    fw_p = data.get("close")
    lim = data.get("limited") if check_limit else None

    corp_keys = ("dividend", "transfer_shares", "right_shares", "right_price")
    if all(k in data for k in corp_keys):
      corp = (
        data["dividend"],
        data["transfer_shares"],
        data["right_shares"],
        data["right_price"],
      )
    elif any(k in data for k in corp_keys):
      zeros = np.zeros(len(data["price"]), dtype=np.float64)
      corp = tuple(data.get(k, zeros) for k in corp_keys)  # type: ignore[assignment]
    else:
      corp = None

    return cls(
      ts=data["ts"],
      price=data["price"],
      deal_price=deal_p,
      fw_price=fw_p,
      limited=lim,
      corp_actions=corp,
      groups=groups,
    )


@dataclass
class SimConfig:
  """Execution, fee, and optional risk-management configuration."""

  initial_cash: float = 100_000.0
  unit: int = 100
  slippage: float = 0.0
  exec_mode: str = "close"
  warmup_bars: int = 0
  start_bar: int = 1
  check_limit: bool = True
  fees: list[Any] | None = None
  stop_loss: float | None = None
  take_profit: float | None = None
  trailing_stop: float | None = None
  max_holding_days: int | None = None


@dataclass
class SimResult:
  """Structured simulation output."""

  assets: np.ndarray
  ror: np.ndarray
  ror_hold: np.ndarray
  actions: np.ndarray
  action_shares: np.ndarray
  trades: dict[str, np.ndarray] = field(default_factory=dict)
  final_state: dict[str, np.ndarray] = field(default_factory=dict)


class Simulator:
  """High-speed parallel account simulator backed by alpha-algo Rust kernel."""

  def __init__(self, market: MarketData, config: SimConfig | None = None):
    self.market = market
    self.config = config if config is not None else SimConfig()

  def run_into(
    self,
    buy: np.ndarray,
    sell: np.ndarray,
    *,
    out_assets: np.ndarray,
    out_ror: np.ndarray,
    out_ror_hold: np.ndarray,
    out_actions: np.ndarray,
    out_action_shares: np.ndarray,
  ) -> SimResult:
    """Execute simulation writing bar-level outputs directly into pre-allocated buffers."""
    m = self.market
    c = self.config
    total = m.groups * m.bars

    buy_f64 = _to_signal_f64(buy, total, "buy")
    sell_f64 = _to_signal_f64(sell, total, "sell")

    div, trans, rs, rp = m.corp_actions if m.corp_actions is not None else (None, None, None, None)

    trades, snap = _algo.sim_accounts_into(
      m.groups,
      m.bars,
      m.ts,
      m.price,
      m.deal_price,
      buy_f64,
      sell_f64,
      out_assets,
      out_ror,
      out_ror_hold,
      out_actions,
      out_action_shares,
      fw_price=m.fw_price,
      limited=m.limited,
      dividend=div,
      transfer_shares=trans,
      right_shares=rs,
      right_price=rp,
      initial_cash=float(c.initial_cash),
      unit=int(c.unit),
      slippage=float(c.slippage),
      warmup_bars=int(c.warmup_bars),
      start_bar=int(c.start_bar),
      exec_mode=c.exec_mode,
      check_limit=bool(c.check_limit),
      fees=c.fees,
      stop_loss=c.stop_loss,
      take_profit=c.take_profit,
      trailing_stop=c.trailing_stop,
      max_holding_days=c.max_holding_days,
    )
    return SimResult(
      assets=out_assets,
      ror=out_ror,
      ror_hold=out_ror_hold,
      actions=out_actions,
      action_shares=out_action_shares,
      trades=trades,
      final_state=snap,
    )

  def run(self, buy: np.ndarray, sell: np.ndarray) -> SimResult:
    """Execute simulation and allocate new output arrays."""
    total = self.market.groups * self.market.bars
    out_assets = np.empty(total, dtype=np.float64)
    out_ror = np.empty(total, dtype=np.float64)
    out_ror_hold = np.empty(total, dtype=np.float64)
    out_actions = np.empty(total, dtype=np.int32)
    out_action_shares = np.empty(total, dtype=np.float64)
    return self.run_into(
      buy,
      sell,
      out_assets=out_assets,
      out_ror=out_ror,
      out_ror_hold=out_ror_hold,
      out_actions=out_actions,
      out_action_shares=out_action_shares,
    )


class AccountBatch:
  """Bar-by-bar multi-account SoA state engine for strategies with custom execute()."""

  def __init__(self, market: MarketData, config: SimConfig | None = None):
    self.market = market
    self.config = config if config is not None else SimConfig()
    c = self.config
    self._inner = _AccountBatch(
      market.groups,
      market.bars,
      initial_cash=float(c.initial_cash),
      unit=int(c.unit),
      slippage=float(c.slippage),
      warmup_bars=int(c.warmup_bars),
      start_bar=int(c.start_bar),
      exec_mode=c.exec_mode,
      check_limit=bool(c.check_limit),
      fees=c.fees,
      stop_loss=c.stop_loss,
      take_profit=c.take_profit,
      trailing_stop=c.trailing_stop,
      max_holding_days=c.max_holding_days,
    )

  def apply_corporate_actions_bar(self, bar: int) -> None:
    m = self.market
    if m.corp_actions is None:
      self._inner.apply_corporate_actions_bar(bar)
      return
    div, trans, rs, rp = m.corp_actions
    self._inner.apply_corporate_actions_bar(
      bar,
      dividend=div,
      transfer_shares=trans,
      right_shares=rs,
      right_price=rp,
    )

  def step_bar(
    self,
    bar: int,
    buy_percent: np.ndarray,
    sell_percent: np.ndarray,
    *,
    out_assets: np.ndarray,
    out_ror: np.ndarray,
    out_ror_hold: np.ndarray,
    out_actions: np.ndarray,
    out_action_shares: np.ndarray,
  ) -> None:
    m = self.market
    div, trans, rs, rp = m.corp_actions if m.corp_actions is not None else (None, None, None, None)
    self._inner.step_bar(
      bar,
      m.ts,
      m.price,
      m.deal_price,
      buy_percent,
      sell_percent,
      out_assets,
      out_ror,
      out_ror_hold,
      out_actions,
      out_action_shares,
      fw_price=m.fw_price,
      limited=m.limited,
      dividend=div,
      transfer_shares=trans,
      right_shares=rs,
      right_price=rp,
    )

  def holding_rors(self, bar: int) -> np.ndarray:
    return self._inner.holding_rors(self.market.price, bar)

  def shares(self) -> np.ndarray:
    return self._inner.shares()

  def holding_bars(self) -> np.ndarray:
    return self._inner.holding_bars()

  def state_of(self, group: int) -> tuple[float, float, float, float, float, float, int]:
    return self._inner.state_of(group)

  def snapshot(self) -> dict[str, np.ndarray]:
    return self._inner.snapshot()

  def trades(self) -> dict[str, np.ndarray]:
    return self._inner.trades()


class PortfolioEngine:
  """Shared-cash portfolio engine with sparse corporate action tracking in Rust."""

  def __init__(
    self,
    groups: int,
    bars: int,
    tradable_mask: np.ndarray,
    *,
    initial_cash: float = 100_000.0,
    unit: int = 100,
    slippage: float = 0.0,
    tolerance: float = 0.0,
    fees: list[Any] | None = None,
    dividend: np.ndarray | None = None,
    transfer_shares: np.ndarray | None = None,
    right_shares: np.ndarray | None = None,
  ):
    mask_b = _to_contiguous(tradable_mask, np.bool_)
    div = _to_contiguous(dividend, np.float64) if dividend is not None else None
    trans = _to_contiguous(transfer_shares, np.float64) if transfer_shares is not None else None
    rs = _to_contiguous(right_shares, np.float64) if right_shares is not None else None
    self._inner = _PortfolioEngine(
      groups,
      bars,
      mask_b,
      initial_cash=float(initial_cash),
      unit=int(unit),
      slippage=float(slippage),
      tolerance=float(tolerance),
      fees=fees,
      dividend=div,
      transfer_shares=trans,
      right_shares=rs,
    )

  @property
  def cash(self) -> float:
    return self._inner.cash

  @property
  def deducted(self) -> float:
    return self._inner.deducted

  def shares(self) -> np.ndarray:
    return self._inner.shares()

  def raw_cost_price(self) -> np.ndarray:
    return self._inner.raw_cost_price()

  def adj_cost_price(self) -> np.ndarray:
    return self._inner.adj_cost_price()

  def raw_mark_price(self) -> np.ndarray:
    return self._inner.raw_mark_price()

  def dividends(self) -> np.ndarray:
    return self._inner.dividends()

  def entry_bars(self) -> np.ndarray:
    return self._inner.entry_bars()

  def apply_corporate_actions(
    self,
    bar: int,
    dividend: np.ndarray | None = None,
    transfer_shares: np.ndarray | None = None,
    right_shares: np.ndarray | None = None,
    right_price: np.ndarray | None = None,
  ) -> None:
    self._inner.apply_corporate_actions(bar, dividend, transfer_shares, right_shares, right_price)

  def update_mark_to_market(
    self,
    bar: int,
    mark_prices: np.ndarray,
    out_cash: np.ndarray,
    out_assets: np.ndarray,
    out_ror: np.ndarray,
  ) -> None:
    self._inner.update_mark_to_market(bar, mark_prices, out_cash, out_assets, out_ror)

  def rebalance_step(
    self,
    bar: int,
    ts_val: int,
    target_weights: np.ndarray,
    deal_prices: np.ndarray,
    mark_prices: np.ndarray,
    out_actions: np.ndarray,
    out_action_shares: np.ndarray,
    out_cash: np.ndarray,
    out_assets: np.ndarray,
    out_ror: np.ndarray,
    *,
    limited: np.ndarray | None = None,
    tolerance: float | None = None,
    universe_mask: np.ndarray | None = None,
    raw_prices_all: np.ndarray | None = None,
    fw_prices_all: np.ndarray | None = None,
  ) -> None:
    self._inner.rebalance_step(
      bar,
      int(ts_val),
      target_weights,
      deal_prices,
      mark_prices,
      out_actions,
      out_action_shares,
      out_cash,
      out_assets,
      out_ror,
      limited=limited,
      tolerance=tolerance,
      universe_mask=universe_mask,
      raw_prices_all=raw_prices_all,
      fw_prices_all=fw_prices_all,
    )

  def sell_group(
    self,
    g: int,
    bar: int,
    ts_val: int,
    shares_to_sell: float,
    deal_price: float,
    out_actions: np.ndarray,
    out_action_shares: np.ndarray,
  ) -> None:
    self._inner.sell_group(
      g, bar, int(ts_val), float(shares_to_sell), float(deal_price), out_actions, out_action_shares
    )

  def buy_group(
    self,
    g: int,
    bar: int,
    ts_val: int,
    cash_budget: float,
    deal_price: float,
    out_actions: np.ndarray,
    out_action_shares: np.ndarray,
    *,
    raw_p: float | None = None,
    fw_p: float | None = None,
  ) -> None:
    self._inner.buy_group(
      g, bar, int(ts_val), float(cash_budget), float(deal_price), out_actions, out_action_shares, raw_p=raw_p, fw_p=fw_p
    )

  def trades(self) -> dict[str, np.ndarray]:
    return self._inner.trades()
