# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""
Generic ExecContext for alpha factor computation.

Naming convention: WQ BRAIN is the canonical naming standard.
  - TS_XXX(x, d)    — time-series / rolling window operators
  - XXX(x)          — cross-sectional operators (across groups)
  - Element-wise    — MAX, MIN, ABS, SIGN, LOG, etc.

Aliases from other systems are provided for compatibility:
  - WQ Alpha 101:  CORRELATION, COVARIANCE, DECAY_LINEAR, SIGNEDPOWER, ...
  - GTJA Alpha 191: SUMAC, STD, HIGHDAY, LOWDAY, DECAYLINEAR, REGBETA, ...
  - AmiBroker/通达信: HHV, LLV, HHVBARS, LLVBARS, REF, DMA, ...

Usage:
  import alpha
  from alpha.context import ExecContext

  data = pl.read_csv("data.csv").sort(["securityid", "tradetime"])
  ctx = ExecContext(data)
  # securities/trades auto-inferred from securityid/tradetime columns
  # alpha.set_ctx(groups=...) called automatically
"""

import logging
import numpy as np
import alpha

logger = logging.getLogger(__name__)


def _returns(a: np.ndarray) -> np.ndarray:
  return a / alpha.REF(a, 1) - 1


def _extract_cols(data, cols):
  """Extract columns as numpy arrays (polars or pandas)."""
  out = {}
  for c in cols:
    out[c] = data[c].to_numpy()
  return out


def _fill_panel(data, securities, trades, cols):
  """Extract columns, padding incomplete panels with NaN via numpy scatter.

  Returns a dict {col_name: np.ndarray} of length securities*trades.
  Zero-copy when the panel is already complete.
  """
  expected = securities * trades
  actual = len(data)
  if actual == expected:
    return _extract_cols(data, cols)

  # Build flat index: row i → security_idx * trades + trade_idx
  # Use the DataFrame engine for fast categorical mapping, then numpy
  try:
    flat_idx = (
      data.select(
        (data["securityid"].rank("dense") - 1).cast(int) * trades
        + (data["tradetime"].rank("dense") - 1).cast(int)
      )
      .to_series()
      .to_numpy()
    )
  except Exception:
    import pandas as pd

    sid = data["securityid"]
    tid = data["tradetime"]
    flat_idx = (
      sid.astype("category").cat.codes.values * trades
      + tid.astype("category").cat.codes.values
    )

  out = {}
  for c in cols:
    target = np.full(expected, np.nan, dtype=np.float64)
    target[flat_idx] = data[c].to_numpy()
    out[c] = target

  n_missing = expected - actual
  logger.warning(
    "Filled %d missing rows to complete %d×%d panel", n_missing, securities, trades
  )
  return out


class ExecContext:
  """
  Unified execution context for alpha factor expressions.

  Provides:
    - OHLCV data fields (OPEN, HIGH, LOW, CLOSE, VOLUME, VWAP)
    - Derived fields (RETURNS/RET, DTM, DBM, TR, HD, LD, SEQUENCE)
    - All operator methods used by wq101, gtja191, and WQ BRAIN transpiler output
    - Dynamic ADV{n} access (average daily dollar volume over n days, per paper definition)
    - Variable aliases (AMOUNT=VOLUME, VOL=VOLUME)

  Args:
    data: DataFrame (polars or pandas) with columns: open, high, low, close, vol, vwap
    securities: Number of securities (stocks). Auto-inferred if 0.
    trades: Number of trading days. Auto-inferred if 0.
    fill: If True (default), pad missing rows with NaN to complete the panel.
  """

  def __init__(self, data, securities: int = 0, trades: int = 0, fill: bool = True):
    # Auto-infer securities and trades from data columns
    if securities == 0 and trades == 0:
      try:
        securities = data["securityid"].n_unique()
        trades = data["tradetime"].n_unique()
      except Exception:
        try:
          securities = data["securityid"].nunique()
          trades = data["tradetime"].nunique()
        except Exception:
          pass

    if securities > 0:
      alpha.set_ctx(groups=securities)

    # Extract OHLCV arrays, filling incomplete panels with NaN
    _cols = ["open", "high", "low", "close", "vol", "vwap"]
    # Include indclass if available in data
    _has_indclass = False
    _has_cap = False
    try:
      _ = data["indclass"]
      _has_indclass = True
      _cols.append("indclass")
    except Exception:
      pass
    try:
      _ = data["cap"]
      _has_cap = True
      _cols.append("cap")
    except Exception:
      pass

    if fill and securities > 0 and trades > 0:
      d = _fill_panel(data, securities, trades, _cols)
    else:
      d = _extract_cols(data, _cols)

    self.OPEN = d["open"]
    self.HIGH = d["high"]
    self.LOW = d["low"]
    self.CLOSE = d["close"]
    self.VOLUME = d["vol"].astype(np.float64)
    self.VWAP = d["vwap"]

    if _has_indclass:
      indclass = d["indclass"]
      self.INDCLASS_SUBINDUSTRY = indclass
      self.INDCLASS_INDUSTRY = np.floor(indclass / 10000)
      self.INDCLASS_SECTOR = np.floor(indclass / 1000000)
    if _has_cap:
      self.CAP = d["cap"].astype(np.float64)
    self.RETURNS = _returns(self.CLOSE)
    self.RET = self.RETURNS

    if securities > 0 and trades > 0:
      self.BANCHMARKINDEXCLOSE = np.repeat(self.CLOSE[0:trades], securities)
      self.BANCHMARKINDEXOPEN = np.repeat(self.OPEN[0:trades], securities)
      self.DTM = self._calc_DTM()
      self.DBM = self._calc_DBM()
      self.TR = self._calc_TR()
      self.HD = self.HIGH - alpha.REF(self.HIGH, 1)
      self.LD = self.LOW - alpha.REF(self.LOW, 1)
      self._SEQUENCE = np.tile(np.arange(1, trades + 1, dtype=np.float64), securities)

  def __call__(self, name: str) -> np.ndarray:
    if name.startswith("ADV"):
      n = name[3:]
      if len(n) == 0:
        return self.VOLUME
      return alpha.MA(self.VOLUME, int(n))
    if name.startswith("INDCLASS."):
      return getattr(self, name.replace(".", "_"))
    if name == "AMOUNT" or name == "VOL":
      return self.VOLUME
    if name == "SEQUENCE":
      return self._SEQUENCE
    return getattr(self, name)

  # ── Derived field helpers ──────────────────────────────────────────

  def _calc_DTM(self):
    return np.where(
      self.OPEN <= alpha.REF(self.OPEN, 1),
      0,
      np.maximum(self.HIGH - self.OPEN, self.OPEN - alpha.REF(self.OPEN, 1)),
    )

  def _calc_DBM(self):
    return np.where(
      self.OPEN >= alpha.REF(self.OPEN, 1),
      0,
      np.maximum(self.OPEN - self.LOW, self.OPEN - alpha.REF(self.OPEN, 1)),
    )

  def _calc_TR(self):
    return np.maximum(
      np.maximum(
        self.HIGH - self.LOW,
        np.abs(self.HIGH - alpha.REF(self.CLOSE, 1)),
      ),
      np.abs(self.LOW - alpha.REF(self.CLOSE, 1)),
    )

  # ====================================================================
  #  TS_ — Time-Series / Rolling Window Operators
  #
  #  Canonical form: TS_XXX(x, d)
  #  All rolling-window operators use the TS_ prefix.
  # ====================================================================

  # ── TS: Sum ─────────────────────────────────────────────────────────
  #   BRAIN: ts_sum        GTJA: SUM, SUMAC        AmiBroker: SUM

  def TS_SUM(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.SUM(a, int(w))

  SUM = TS_SUM  # wq101 / gtja191 / AmiBroker
  SUMAC = TS_SUM  # gtja191

  # ── TS: Mean ────────────────────────────────────────────────────────
  #   BRAIN: ts_mean       GTJA: MA, MEAN          AmiBroker: MA

  def TS_MEAN(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.MA(a, int(w))

  MA = TS_MEAN  # universal
  MEAN = TS_MEAN  # wq101

  # ── TS: EMA family ──────────────────────────────────────────────────
  #   Not in BRAIN canonical set, but widely used in GTJA/AmiBroker

  def EMA(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.EMA(a, int(w))

  def SMA(self, a, *args) -> np.ndarray:
    """SMA with 2 args = simple MA, with 3 args = EMA variant (weight=m/n)."""
    if len(args) == 1:
      return alpha.MA(a, int(args[0]))
    else:
      return alpha.SMA(a, int(args[0]), int(args[1]))

  def DMA(self, a: np.ndarray, weight: float) -> np.ndarray:
    return alpha.DMA(a, float(weight))

  # ── TS: Std Dev / Variance ─────────────────────────────────────────
  #   BRAIN: ts_std_dev    GTJA: STD               wq101: STDDEV

  def TS_STD_DEV(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.STDDEV(a, int(w))

  STDDEV = TS_STD_DEV  # wq101
  STD = TS_STD_DEV  # gtja191

  def TS_VARIANCE(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.VAR(a, int(w))

  VAR = TS_VARIANCE  # gtja191

  # ── TS: Correlation / Covariance (two-input) ───────────────────────
  #   BRAIN: ts_correlation, ts_covariance
  #   wq101: CORRELATION, COVARIANCE
  #   GTJA:  CORR, COV

  def TS_CORRELATION(self, a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    return alpha.CORR2(a, b, int(w))

  CORR = TS_CORRELATION  # gtja191
  CORRELATION = TS_CORRELATION  # wq101

  def TS_COVARIANCE(self, a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    return alpha.COV(a, b, int(w))

  COV = TS_COVARIANCE  # gtja191
  COVARIANCE = TS_COVARIANCE  # wq101

  # ── TS: Extremes ───────────────────────────────────────────────────
  #   BRAIN: ts_max, ts_min, ts_argmax, ts_argmin
  #   GTJA:  TSMAX, TSMIN, HIGHDAY, LOWDAY
  #   AmiBroker: HHV, LLV, HHVBARS, LLVBARS

  def TS_MAX(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.HHV(a, int(w))

  TSMAX = TS_MAX  # gtja191

  def TS_MIN(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.LLV(a, int(w))

  TSMIN = TS_MIN  # gtja191

  def TS_ARGMAX(self, a: np.ndarray, w: int) -> np.ndarray:
    return int(w) - alpha.HHVBARS(a, int(w))

  def TS_ARGMIN(self, a: np.ndarray, w: int) -> np.ndarray:
    return int(w) - alpha.LLVBARS(a, int(w))

  def HIGHDAY(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.HHVBARS(a, int(w))

  def LOWDAY(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.LLVBARS(a, int(w))

  # ── TS: Rank ───────────────────────────────────────────────────────
  #   BRAIN: ts_rank       GTJA: TSRANK

  def TS_RANK(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.RANK(a, int(w))

  TSRANK = TS_RANK  # gtja191

  # ── TS: Quantile ────────────────────────────────────────────────────

  def TS_QUANTILE(self, a: np.ndarray, w: int, q: float) -> np.ndarray:
    return alpha.QUANTILE(a, int(w), q)

  # ── TS: Delay / Delta ──────────────────────────────────────────────
  #   BRAIN: ts_delay, ts_delta
  #   wq101: DELAY, DELTA
  #   AmiBroker: REF

  def TS_DELAY(self, a: np.ndarray, p: int) -> np.ndarray:
    return alpha.REF(a, int(p))

  DELAY = TS_DELAY  # wq101

  def TS_DELTA(self, a: np.ndarray, p: int) -> np.ndarray:
    return a - alpha.REF(a, int(p))

  DELTA = TS_DELTA  # wq101

  # ── TS: Decay Linear / Weighted Mean ───────────────────────────────
  #   BRAIN: ts_decay_linear
  #   wq101: DECAY_LINEAR
  #   GTJA:  DECAYLINEAR, WMA

  def TS_DECAY_LINEAR(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.LWMA(a, int(w))

  DECAY_LINEAR = TS_DECAY_LINEAR  # wq101
  DECAYLINEAR = TS_DECAY_LINEAR  # gtja191
  WMA = TS_DECAY_LINEAR  # gtja191

  # ── TS: Product ────────────────────────────────────────────────────
  #   BRAIN: ts_product    wq101: PRODUCT    GTJA: PROD

  def TS_PRODUCT(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.PRODUCT(a, int(w))

  PRODUCT = TS_PRODUCT  # wq101
  PROD = TS_PRODUCT  # gtja191

  # ── TS: Regression ─────────────────────────────────────────────────
  #   GTJA: REGBETA, REGRESI

  def TS_REGBETA(self, a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    return alpha.REGBETA(a, b, int(w))

  REGBETA = TS_REGBETA  # gtja191

  def TS_REGRESI(self, a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    return alpha.REGRESI(a, b, int(w))

  REGRESI = TS_REGRESI  # gtja191

  # ── TS: Slope / Intercept ───────────────────────────────────────

  def SLOPE(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.SLOPE(a, int(w))

  def INTERCEPT(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.INTERCEPT(a, int(w))

  # ── TS: Counting / Conditional ─────────────────────────────────────
  #   GTJA: COUNT, SUMIF

  def TS_COUNT(self, cond: np.ndarray, w: int) -> np.ndarray:
    return alpha.COUNT(np.asarray(cond, dtype=bool), int(w))

  COUNT = TS_COUNT  # gtja191

  def TS_SUMIF(self, a: np.ndarray, w: int, cond) -> np.ndarray:
    return alpha.SUMIF(
      np.asarray(a, dtype=np.float64), np.asarray(cond, dtype=bool), int(w)
    )

  SUMIF = TS_SUMIF  # gtja191

  def SUMBARS(self, a: np.ndarray, amount: float) -> np.ndarray:
    return alpha.SUMBARS(a, amount)

  # ── TS: Bar Counting ──────────────────────────────────────────

  def BARSLAST(self, cond: np.ndarray) -> np.ndarray:
    return alpha.BARSLAST(np.asarray(cond, dtype=bool))

  def BARSSINCE(self, cond: np.ndarray) -> np.ndarray:
    return alpha.BARSSINCE(np.asarray(cond, dtype=bool))

  # ── TS: Conditional Scan (SELF recursion) ───────────────────────────
  #   GTJA: SELF-referencing patterns

  def SCAN_MUL(self, operand: np.ndarray, cond: np.ndarray) -> np.ndarray:
    return alpha.SCAN_MUL(
      np.asarray(operand, dtype=np.float64), np.asarray(cond, dtype=bool)
    )

  def SCAN_ADD(self, operand: np.ndarray, cond: np.ndarray) -> np.ndarray:
    return alpha.SCAN_ADD(
      np.asarray(operand, dtype=np.float64), np.asarray(cond, dtype=bool)
    )

  # ── TS: Z-Score ────────────────────────────────────────────────────
  #   BRAIN: ts_zscore

  def TS_ZSCORE(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.ZSCORE(a, int(w))

  # ── TS: Higher Moments ─────────────────────────────────────────────
  #   BRAIN: ts_skewness, ts_kurtosis

  def TS_SKEWNESS(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.SKEWNESS(a, int(w))

  def TS_KURTOSIS(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.KURTOSIS(a, int(w))

  # ── TS: Data Utilities ─────────────────────────────────────────────
  #   BRAIN: ts_backfill, ts_count_nans

  def TS_BACKFILL(self, a: np.ndarray) -> np.ndarray:
    return alpha.BACKFILL(a)

  def TS_COUNT_NANS(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.COUNT_NANS(a, int(w))

  # ── TS: Entropy ────────────────────────────────────────────────────
  #   BRAIN: ts_entropy

  def TS_ENTROPY(self, a: np.ndarray, w: int, bins: int = 10) -> np.ndarray:
    return alpha.ENTROPY(a, int(w), int(bins))

  # ── TS: Min-Max Diff (Range) ─────────────────────────────────────
  #   BRAIN: ts_min_max_diff

  def TS_MIN_MAX_DIFF(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.MIN_MAX_DIFF(a, int(w))

  # ── TS: Weighted Delay ───────────────────────────────────────────
  #   BRAIN: ts_weighted_delay

  def TS_WEIGHTED_DELAY(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.WEIGHTED_DELAY(a, int(w))

  # ── TS: Central Moment ───────────────────────────────────────────
  #   BRAIN: ts_moment

  def TS_MOMENT(self, a: np.ndarray, w: int, k: int = 2) -> np.ndarray:
    return alpha.MOMENT(a, int(w), int(k))

  # ── TS: Binning ────────────────────────────────────────────────

  def BINS(self, a: np.ndarray, n: int) -> np.ndarray:
    return alpha.BINS(a, int(n))

  # ── TS: Future Return ──────────────────────────────────────────

  def FRET(self, open: np.ndarray, close: np.ndarray, is_calc: np.ndarray, delay: int, periods: int) -> np.ndarray:
    return alpha.FRET(open, close, is_calc, int(delay), int(periods))

  # ── TS: Cross Detection ────────────────────────────────────────────
  #   AmiBroker/GTJA: CROSS, LONGCROSS

  def CROSS(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return alpha.CROSS(a, b)

  def LONGCROSS(self, a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    return alpha.LONGCROSS(a, b, int(n))

  def RCROSS(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return alpha.RCROSS(a, b)

  def RLONGCROSS(self, a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    return alpha.RLONGCROSS(a, b, int(n))

  # ====================================================================
  #  Cross-Sectional Operators (no prefix)
  #
  #  Operate across all groups at each time step.
  # ====================================================================

  # ── Rank ────────────────────────────────────────────────────────────
  #   BRAIN: rank          wq101/GTJA: RANK

  def RANK(self, a: np.ndarray) -> np.ndarray:
    return alpha.CC_RANK(a)

  # ── Z-Score ─────────────────────────────────────────────────────────
  #   BRAIN: zscore

  def ZSCORE(self, a: np.ndarray) -> np.ndarray:
    return alpha.CC_ZSCORE(a)

  # ── Scale ───────────────────────────────────────────────────────────
  #   BRAIN: scale         wq101: SCALE

  def SCALE(self, a: np.ndarray, k: int = 1) -> np.ndarray:
    return a * k / np.abs(a).sum()

  # ── IndNeutralize ───────────────────────────────────────────────────
  #   wq101: INDNEUTRALIZE(value, IndClass.xxx)

  def INDNEUTRALIZE(self, value: np.ndarray, category: np.ndarray) -> np.ndarray:
    return alpha.NEUTRALIZE(category, value)

  # ====================================================================
  #  Grouped Cross-Sectional Operators (GROUP_ prefix)
  #
  #  Operate across items within each category group at each time step.
  # ====================================================================

  # ── Group Rank ────────────────────────────────────────────────────
  #   BRAIN: group_rank

  def GROUP_RANK(self, a: np.ndarray, group: np.ndarray) -> np.ndarray:
    return alpha.GROUP_RANK(group, a)

  # ── Group Z-Score ─────────────────────────────────────────────────
  #   BRAIN: group_zscore

  def GROUP_ZSCORE(self, a: np.ndarray, group: np.ndarray) -> np.ndarray:
    return alpha.GROUP_ZSCORE(group, a)

  # ====================================================================
  #  Element-wise Operators
  # ====================================================================

  def MAX(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)

  def MIN(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a, b)

  def ABS(self, a: np.ndarray) -> np.ndarray:
    return np.abs(a)

  def LOG(self, a: np.ndarray) -> np.ndarray:
    return np.log(a)

  def SIGN(self, a: np.ndarray) -> np.ndarray:
    return np.sign(a)

  def SIGNEDPOWER(self, a: np.ndarray, p) -> np.ndarray:
    return np.sign(a) * np.power(np.abs(a), p)

  def FILTER(self, a: np.ndarray, cond: np.ndarray) -> np.ndarray:
    return np.where(cond, a, np.nan)

  # ── Sanitization ───────────────────────────────────────────────────
  #   BRAIN: pasteurize / purify, truncate / tail

  def PASTEURIZE(self, a: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(a), a, 0.0)

  PURIFY = PASTEURIZE  # BRAIN alias

  def TAIL(self, a: np.ndarray, limit: float = 3.0) -> np.ndarray:
    return np.clip(a, -limit, limit)

  TRUNCATE = TAIL  # BRAIN alias
