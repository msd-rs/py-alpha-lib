# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

# THIS FILE IS AUTO-GENERATED, DO NOT EDIT

import numpy as np
from . import _algo

def _to_f64(a):
  """Ensure array is float64. Zero-copy if already float64."""
  if a.dtype == np.float64:
    return a
  return a.astype(np.float64)

def _to_bool(a):
  """Ensure array is bool. Zero-copy if already bool."""
  if a.dtype == np.bool_:
    return a
  return a.astype(bool)

def _to_usize(a):
  """Ensure array is usize (uint64). Zero-copy if already uint64."""
  if a.dtype == np.uint64:
    return a
  return a.astype(np.uint64)

def ALPHA[A: np.ndarray | list[np.ndarray]](
  input: A, benchmark: A, periods: int
) -> A:
  """
  Rolling Jensen's Alpha of asset returns against benchmark returns.
  
  Alpha = mean(input) - Beta * mean(benchmark)
  Measures excess return of an asset relative to its expected return given beta.
  
  Ref: https://en.wikipedia.org/wiki/Jensen%27s_alpha
  """
  if isinstance(input, list) and isinstance(benchmark, list):
    input = [_to_f64(x) for x in input]
    benchmark = [_to_f64(x) for x in benchmark]
    r = [np.empty_like(x) for x in input]
    _algo.alpha(r, input, benchmark, periods)
    return r
  else:
    input = _to_f64(input)
    benchmark = _to_f64(benchmark)
    r = np.empty_like(input)
    _algo.alpha(r, input, benchmark, periods)
    return r

def BACKFILL[A: np.ndarray | list[np.ndarray]](
  input: A
) -> A:
  """
  Forward-fill NaN values with the last valid observation
  
  Iterates forward through each group; if x[i] is NaN, copies the last valid value.
  Leading NaNs (before any valid value) remain NaN.
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.backfill(r, input)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.backfill(r, input)
    return r

def BARSLAST[A: np.ndarray | list[np.ndarray]](
  input: A
) -> A:
  """
  Calculate number of bars since last condition true
  
  Ref: https://www.amibroker.com/guide/afl/barslast.html
  """
  if isinstance(input, list):
    r = [np.empty_like(x, dtype=float) for x in input]
    input = [x.astype(bool) for x in input]
    _algo.barslast(r, input)
    return r
  else:
    r = np.empty_like(input, dtype=float)
    input = input.astype(bool)
    _algo.barslast(r, input)
    return r

def BARSSINCE[A: np.ndarray | list[np.ndarray]](
  input: A
) -> A:
  """
  Calculate number of bars since first condition true
  
  Ref: https://www.amibroker.com/guide/afl/barssince.html
  """
  if isinstance(input, list):
    r = [np.empty_like(x, dtype=float) for x in input]
    input = [x.astype(bool) for x in input]
    _algo.barssince(r, input)
    return r
  else:
    r = np.empty_like(input, dtype=float)
    input = input.astype(bool)
    _algo.barssince(r, input)
    return r

def BETA[A: np.ndarray | list[np.ndarray]](
  input: A, benchmark: A, periods: int
) -> A:
  """
  Rolling Beta coefficient of asset returns against benchmark returns.
  
  Beta = Covariance(input, benchmark) / Variance(benchmark)
  Measures systematic risk of an asset relative to the market.
  
  Ref: https://en.wikipedia.org/wiki/Beta_(finance)
  """
  if isinstance(input, list) and isinstance(benchmark, list):
    input = [_to_f64(x) for x in input]
    benchmark = [_to_f64(x) for x in benchmark]
    r = [np.empty_like(x) for x in input]
    _algo.beta(r, input, benchmark, periods)
    return r
  else:
    input = _to_f64(input)
    benchmark = _to_f64(benchmark)
    r = np.empty_like(input)
    _algo.beta(r, input, benchmark, periods)
    return r

def BINS[A: np.ndarray | list[np.ndarray]](
  input: A, bins: int
) -> A:
  """
  Discretize the input into n bins, the ctx.groups() is the number of groups
  
  Bins are 0-based index.
  Same value are assigned to the same bin.
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.bins(r, input, bins)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.bins(r, input, bins)
    return r

def BW_SPLIT[A: np.ndarray | list[np.ndarray]](
  price: A, dividend: A, transfer_shares: A, right_shares: A, right_price: A
) -> A:
  """
  Backward split and dividend adjustment
  
  Adjusts prices backward (from latest to earliest event) using a loop for precise calculation.
  """
  if isinstance(price, list) and isinstance(dividend, list) and isinstance(transfer_shares, list) and isinstance(right_shares, list) and isinstance(right_price, list):
    price = [_to_f64(x) for x in price]
    dividend = [_to_f64(x) for x in dividend]
    transfer_shares = [_to_f64(x) for x in transfer_shares]
    right_shares = [_to_f64(x) for x in right_shares]
    right_price = [_to_f64(x) for x in right_price]
    r = [np.empty_like(x) for x in price]
    _algo.bw_split(r, price, dividend, transfer_shares, right_shares, right_price)
    return r
  else:
    price = _to_f64(price)
    dividend = _to_f64(dividend)
    transfer_shares = _to_f64(transfer_shares)
    right_shares = _to_f64(right_shares)
    right_price = _to_f64(right_price)
    r = np.empty_like(price)
    _algo.bw_split(r, price, dividend, transfer_shares, right_shares, right_price)
    return r

def CC_RANK[A: np.ndarray | list[np.ndarray]](
  input: A
) -> A:
  """
  Calculate rank percentage cross group dimension, the ctx.groups() is the number of groups
  Same value are averaged
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.cc_rank(r, input)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.cc_rank(r, input)
    return r

def CC_ZSCORE[A: np.ndarray | list[np.ndarray]](
  input: A
) -> A:
  """
  Calculate cross-sectional Z-Score across groups at each time step
  
  Z-Score = (x - mean) / stddev, computed across all groups for each time position.
  NaN values are excluded from mean/stddev computation. NaN input produces NaN output.
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.cc_zscore(r, input)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.cc_zscore(r, input)
    return r

def CORR[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Time Series Correlation in moving window on self
  
  Calculates the correlation coefficient between the input series and the time index.
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.corr(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.corr(r, input, periods)
    return r

def CORR2[A: np.ndarray | list[np.ndarray]](
  x: A, y: A, periods: int
) -> A:
  """
  Calculate two series correlation over a moving window
  
  Correlation = Cov(X, Y) / (StdDev(X) * StdDev(Y))
  """
  if isinstance(x, list) and isinstance(y, list):
    x = [_to_f64(x) for x in x]
    y = [_to_f64(x) for x in y]
    r = [np.empty_like(x) for x in x]
    _algo.corr2(r, x, y, periods)
    return r
  else:
    x = _to_f64(x)
    y = _to_f64(y)
    r = np.empty_like(x)
    _algo.corr2(r, x, y, periods)
    return r

def COUNT[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Calculate number of periods where condition is true in passed `periods` window
  
  Ref: https://www.amibroker.com/guide/afl/count.html
  """
  if isinstance(input, list):
    r = [np.empty_like(x, dtype=float) for x in input]
    input = [x.astype(bool) for x in input]
    _algo.count(r, input, periods)
    return r
  else:
    r = np.empty_like(input, dtype=float)
    input = input.astype(bool)
    _algo.count(r, input, periods)
    return r

def COUNT_NANS[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Count number of NaN values in a rolling window
  
  For each position, counts the number of NaN values in the preceding `periods` elements.
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.count_nans(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.count_nans(r, input, periods)
    return r

def COUNT_V[A: np.ndarray | list[np.ndarray]](
  input: A, periods: A
) -> A:
  """
  Calculate number of periods where condition is true in passed variable `periods` window
  
  Ref: https://www.amibroker.com/guide/afl/count.html
  """
  if isinstance(input, list) and isinstance(periods, list):
    input = [_to_bool(x) for x in input]
    periods = [_to_usize(x) for x in periods]
    r = [np.empty_like(x, dtype=float) for x in periods]
    _algo.count_v(r, input, periods)
    return r
  else:
    input = _to_bool(input)
    periods = _to_usize(periods)
    r = np.empty_like(periods, dtype=float)
    _algo.count_v(r, input, periods)
    return r

def COV[A: np.ndarray | list[np.ndarray]](
  x: A, y: A, periods: int
) -> A:
  """
  Calculate Covariance over a moving window
  
  Covariance = (SumXY - (SumX * SumY) / N) / (N - 1)
  """
  if isinstance(x, list) and isinstance(y, list):
    x = [_to_f64(x) for x in x]
    y = [_to_f64(x) for x in y]
    r = [np.empty_like(x) for x in x]
    _algo.cov(r, x, y, periods)
    return r
  else:
    x = _to_f64(x)
    y = _to_f64(y)
    r = np.empty_like(x)
    _algo.cov(r, x, y, periods)
    return r

def CROSS[A: np.ndarray | list[np.ndarray]](
  a: A, b: A
) -> A:
  """
  For 2 arrays A and B, return true if A[i-1] < B[i-1] and A[i] >= B[i]
  alias: golden_cross, cross_ge
  """
  if isinstance(a, list) and isinstance(b, list):
    a = [_to_f64(x) for x in a]
    b = [_to_f64(x) for x in b]
    r = [np.empty_like(x, dtype=bool) for x in a]
    _algo.cross(r, a, b)
    return r
  else:
    a = _to_f64(a)
    b = _to_f64(b)
    r = np.empty_like(a, dtype=bool)
    _algo.cross(r, a, b)
    return r

def DMA[A: np.ndarray | list[np.ndarray]](
  input: A, weight: float
) -> A:
  """
  Exponential Moving Average
  current = weight * current + (1 - weight) * previous
  
  Ref: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.dma(r, input, weight)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.dma(r, input, weight)
    return r

def DMA_V[A: np.ndarray | list[np.ndarray]](
  input: A, weight: A
) -> A:
  """
  Exponential Moving Average with variable weight
  
  Ref: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
  """
  if isinstance(input, list) and isinstance(weight, list):
    input = [_to_f64(x) for x in input]
    weight = [_to_f64(x) for x in weight]
    r = [np.empty_like(x) for x in input]
    _algo.dma_v(r, input, weight)
    return r
  else:
    input = _to_f64(input)
    weight = _to_f64(weight)
    r = np.empty_like(input)
    _algo.dma_v(r, input, weight)
    return r

def EMA_V[A: np.ndarray | list[np.ndarray]](
  input: A, periods: A
) -> A:
  """
  Exponential Moving Average with variable periods
  
  Ref: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
  """
  if isinstance(input, list) and isinstance(periods, list):
    input = [_to_f64(x) for x in input]
    periods = [_to_usize(x) for x in periods]
    r = [np.empty_like(x) for x in input]
    _algo.ema_v(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    periods = _to_usize(periods)
    r = np.empty_like(input)
    _algo.ema_v(r, input, periods)
    return r

def ENTROPY[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int, bins: int
) -> A:
  """
  Calculate rolling Shannon entropy over a moving window
  
  Discretizes values into `bins` equal-width buckets within the window's
  [min, max] range, then computes -sum(p * ln(p)) where p is the frequency
  of each occupied bin. Uses natural log (base e).
  Requires at least 2 valid values. Single-value windows return 0.
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.entropy(r, input, periods, bins)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.entropy(r, input, periods, bins)
    return r

def FRET[A: np.ndarray | list[np.ndarray]](
  open: A, close: A, is_calc: A, delay: int, periods: int
) -> A:
  """
  Future Return
  
  Calculates the return from the open price of the delayed day (t+delay) to the close price of the future day (t+delay+periods-1).
  Return = (Close[t+delay+periods-1] - Open[t+delay]) / Open[t+delay]
  
  If n=1, delay=1, it calculates (Close[t+1] - Open[t+1]) / Open[t+1].
  If `is_calc[t+delay]` is 0, returns NaN.
  """
  if isinstance(open, list) and isinstance(close, list) and isinstance(is_calc, list):
    open = [_to_f64(x) for x in open]
    close = [_to_f64(x) for x in close]
    is_calc = [_to_f64(x) for x in is_calc]
    r = [np.empty_like(x) for x in open]
    _algo.fret(r, open, close, is_calc, delay, periods)
    return r
  else:
    open = _to_f64(open)
    close = _to_f64(close)
    is_calc = _to_f64(is_calc)
    r = np.empty_like(open)
    _algo.fret(r, open, close, is_calc, delay, periods)
    return r

def FW_SPLIT[A: np.ndarray | list[np.ndarray]](
  price: A, dividend: A, transfer_shares: A, right_shares: A, right_price: A
) -> A:
  """
  Forward split and dividend adjustment
  
  Adjusts prices forward (from earliest to latest event) using a loop for precise calculation.
  """
  if isinstance(price, list) and isinstance(dividend, list) and isinstance(transfer_shares, list) and isinstance(right_shares, list) and isinstance(right_price, list):
    price = [_to_f64(x) for x in price]
    dividend = [_to_f64(x) for x in dividend]
    transfer_shares = [_to_f64(x) for x in transfer_shares]
    right_shares = [_to_f64(x) for x in right_shares]
    right_price = [_to_f64(x) for x in right_price]
    r = [np.empty_like(x) for x in price]
    _algo.fw_split(r, price, dividend, transfer_shares, right_shares, right_price)
    return r
  else:
    price = _to_f64(price)
    dividend = _to_f64(dividend)
    transfer_shares = _to_f64(transfer_shares)
    right_shares = _to_f64(right_shares)
    right_price = _to_f64(right_price)
    r = np.empty_like(price)
    _algo.fw_split(r, price, dividend, transfer_shares, right_shares, right_price)
    return r

def GROUP_RANK[A: np.ndarray | list[np.ndarray]](
  category: A, input: A
) -> A:
  """
  Calculate rank percentage within each category group at each time step
  
  For each time position, groups items by `category` value, then computes
  rank percentage within each group. Same value gets averaged rank.
  NaN in category or input produces NaN output.
  """
  if isinstance(category, list) and isinstance(input, list):
    category = [_to_f64(x) for x in category]
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in category]
    _algo.group_rank(r, category, input)
    return r
  else:
    category = _to_f64(category)
    input = _to_f64(input)
    r = np.empty_like(category)
    _algo.group_rank(r, category, input)
    return r

def GROUP_ZSCORE[A: np.ndarray | list[np.ndarray]](
  category: A, input: A
) -> A:
  """
  Calculate Z-Score within each category group at each time step
  
  For each time position, groups items by `category` value, then computes
  (x - group_mean) / group_std within each group.
  NaN in category or input produces NaN output.
  Groups with fewer than 2 valid values produce NaN.
  """
  if isinstance(category, list) and isinstance(input, list):
    category = [_to_f64(x) for x in category]
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in category]
    _algo.group_zscore(r, category, input)
    return r
  else:
    category = _to_f64(category)
    input = _to_f64(input)
    r = np.empty_like(category)
    _algo.group_zscore(r, category, input)
    return r

def HHV[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Find highest value in a preceding `periods` window
  
  Ref: https://www.amibroker.com/guide/afl/hhv.html
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.hhv(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.hhv(r, input, periods)
    return r

def HHV_V[A: np.ndarray | list[np.ndarray]](
  input: A, periods: A
) -> A:
  """
  Find highest value in a preceding variable `periods` window
  
  Ref: https://www.amibroker.com/guide/afl/hhv.html
  """
  if isinstance(input, list) and isinstance(periods, list):
    input = [_to_f64(x) for x in input]
    periods = [_to_usize(x) for x in periods]
    r = [np.empty_like(x) for x in input]
    _algo.hhv_v(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    periods = _to_usize(periods)
    r = np.empty_like(input)
    _algo.hhv_v(r, input, periods)
    return r

def HHVBARS[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  The number of periods that have passed since the array reached its `periods` period high
  
  Ref: https://www.amibroker.com/guide/afl/hhvbars.html
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.hhvbars(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.hhvbars(r, input, periods)
    return r

def HHVBARS_V[A: np.ndarray | list[np.ndarray]](
  input: A, periods: A
) -> A:
  """
  The number of periods that have passed since the array reached its variable `periods` period high
  
  Ref: https://www.amibroker.com/guide/afl/hhvbars.html
  """
  if isinstance(input, list) and isinstance(periods, list):
    input = [_to_f64(x) for x in input]
    periods = [_to_usize(x) for x in periods]
    r = [np.empty_like(x) for x in input]
    _algo.hhvbars_v(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    periods = _to_usize(periods)
    r = np.empty_like(input)
    _algo.hhvbars_v(r, input, periods)
    return r

def INTERCEPT[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Linear Regression Intercept
  
  Calculates the intercept of the linear regression line for a moving window.
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.intercept(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.intercept(r, input, periods)
    return r

def KURTOSIS[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Calculate rolling sample excess Kurtosis over a moving window
  
  Uses adjusted Fisher formula (matches pandas):
  kurt = n(n+1)/((n-1)(n-2)(n-3)) * sum(((x-mean)/std)^4) - 3(n-1)^2/((n-2)(n-3))
  Requires at least 4 valid values.
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.kurtosis(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.kurtosis(r, input, periods)
    return r

def LLV[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Find lowest value in a preceding `periods` window
  
  Ref: https://www.amibroker.com/guide/afl/llv.html
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.llv(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.llv(r, input, periods)
    return r

def LLV_V[A: np.ndarray | list[np.ndarray]](
  input: A, periods: A
) -> A:
  """
  Find lowest value in a preceding variable `periods` window
  
  Ref: https://www.amibroker.com/guide/afl/llv.html
  """
  if isinstance(input, list) and isinstance(periods, list):
    input = [_to_f64(x) for x in input]
    periods = [_to_usize(x) for x in periods]
    r = [np.empty_like(x) for x in input]
    _algo.llv_v(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    periods = _to_usize(periods)
    r = np.empty_like(input)
    _algo.llv_v(r, input, periods)
    return r

def LLVBARS[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  The number of periods that have passed since the array reached its periods period low
  
  Ref: https://www.amibroker.com/guide/afl/llvbars.html
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.llvbars(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.llvbars(r, input, periods)
    return r

def LLVBARS_V[A: np.ndarray | list[np.ndarray]](
  input: A, periods: A
) -> A:
  """
  The number of periods that have passed since the array reached its variable `periods` period low
  
  Ref: https://www.amibroker.com/guide/afl/llvbars.html
  """
  if isinstance(input, list) and isinstance(periods, list):
    input = [_to_f64(x) for x in input]
    periods = [_to_usize(x) for x in periods]
    r = [np.empty_like(x) for x in input]
    _algo.llvbars_v(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    periods = _to_usize(periods)
    r = np.empty_like(input)
    _algo.llvbars_v(r, input, periods)
    return r

def LONGCROSS[A: np.ndarray | list[np.ndarray]](
  a: A, b: A, n: int
) -> A:
  """
  For 2 arrays A and B, return true if previous N periods A < B, Current A >= B
  """
  if isinstance(a, list) and isinstance(b, list):
    a = [_to_f64(x) for x in a]
    b = [_to_f64(x) for x in b]
    r = [np.empty_like(x, dtype=bool) for x in a]
    _algo.longcross(r, a, b, n)
    return r
  else:
    a = _to_f64(a)
    b = _to_f64(b)
    r = np.empty_like(a, dtype=bool)
    _algo.longcross(r, a, b, n)
    return r

def LWMA[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Linear Weighted Moving Average
  
  LWMA = SUM(Price * Weight) / SUM(Weight)
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.lwma(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.lwma(r, input, periods)
    return r

def LWMA_V[A: np.ndarray | list[np.ndarray]](
  input: A, periods: A
) -> A:
  """
  Linear Weighted Moving Average with variable periods
  
  LWMA = SUM(Price * Weight) / SUM(Weight)
  """
  if isinstance(input, list) and isinstance(periods, list):
    input = [_to_f64(x) for x in input]
    periods = [_to_usize(x) for x in periods]
    r = [np.empty_like(x) for x in input]
    _algo.lwma_v(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    periods = _to_usize(periods)
    r = np.empty_like(input)
    _algo.lwma_v(r, input, periods)
    return r

def MA[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Simple Moving Average, also known as arithmetic moving average
  
  Ref: https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.ma(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.ma(r, input, periods)
    return r

def MA_V[A: np.ndarray | list[np.ndarray]](
  input: A, periods: A
) -> A:
  """
  Simple Moving Average with variable periods
  
  Ref: https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average
  """
  if isinstance(input, list) and isinstance(periods, list):
    input = [_to_f64(x) for x in input]
    periods = [_to_usize(x) for x in periods]
    r = [np.empty_like(x) for x in input]
    _algo.ma_v(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    periods = _to_usize(periods)
    r = np.empty_like(input)
    _algo.ma_v(r, input, periods)
    return r

def MAX_DRAWDOWN[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Rolling Maximum Drawdown.
  
  MaxDrawdown = minimum peak-to-trough decline within the rolling window.
  Result is expressed as a negative return (e.g. -0.2 means 20% drawdown from peak).
  Input should be a price or equity curve series.
  
  Ref: https://en.wikipedia.org/wiki/Drawdown_(economics)
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.max_drawdown(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.max_drawdown(r, input, periods)
    return r

def MIN_MAX_DIFF[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Calculate rolling min-max difference (range) over a moving window
  
  TS_MIN_MAX_DIFF = TS_MAX(x, d) - TS_MIN(x, d)
  Single-pass using two monotonic deques for efficiency.
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.min_max_diff(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.min_max_diff(r, input, periods)
    return r

def MOMENT[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int, k: int
) -> A:
  """
  Calculate rolling k-th central moment over a moving window
  
  MOMENT(x, d, k) = mean((x - mean)^k) over window of d periods.
  This is the raw (non-adjusted) sample moment.
  k=2 gives variance (population), k=3 gives raw third moment, etc.
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.moment(r, input, periods, k)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.moment(r, input, periods, k)
    return r

def NEUTRALIZE[A: np.ndarray | list[np.ndarray]](
  category: A, input: A
) -> A:
  """
  Neutralize the effect of a categorical variable on a numeric variable
  """
  if isinstance(category, list) and isinstance(input, list):
    category = [_to_f64(x) for x in category]
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in category]
    _algo.neutralize(r, category, input)
    return r
  else:
    category = _to_f64(category)
    input = _to_f64(input)
    r = np.empty_like(category)
    _algo.neutralize(r, category, input)
    return r

def PRODUCT[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Calculate product of values in preceding `periods` window
  
  If periods is 0, it calculates the cumulative product from the first valid value.
  
  Ref: https://www.amibroker.com/guide/afl/product.html
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.product(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.product(r, input, periods)
    return r

def QUANTILE[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int, q: float
) -> A:
  """
  Calculate rolling quantile over a moving window
  
  QUANTILE(x, d, q) returns the q-th quantile (0 <= q <= 1) of values
  in the preceding d periods. Uses linear interpolation between data points
  (matching numpy/pandas percentile with interpolation='linear').
  NaN values are excluded from the computation. Requires at least 1 valid value.
  
  Ref: https://numpy.org/doc/stable/reference/generated/numpy.quantile.html
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.quantile(r, input, periods, q)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.quantile(r, input, periods, q)
    return r

def RANK[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Calculate rank in a sliding window with size `periods`
  
  Uses min-rank method for ties (same as pandas rankdata method='min').
  NaN values are treated as larger than all non-NaN values.
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.rank(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.rank(r, input, periods)
    return r

def RCROSS[A: np.ndarray | list[np.ndarray]](
  a: A, b: A
) -> A:
  """
  For 2 arrays A and B, return true if A[i-1] > B[i-1] and A[i] <= B[i]
  alias: death_cross, cross_le
  """
  if isinstance(a, list) and isinstance(b, list):
    a = [_to_f64(x) for x in a]
    b = [_to_f64(x) for x in b]
    r = [np.empty_like(x, dtype=bool) for x in a]
    _algo.rcross(r, a, b)
    return r
  else:
    a = _to_f64(a)
    b = _to_f64(b)
    r = np.empty_like(a, dtype=bool)
    _algo.rcross(r, a, b)
    return r

def REF[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Right shift input array by `periods`, r[i] = input[i - periods]
  
  Ref: https://www.amibroker.com/guide/afl/ref.html
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.ref(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.ref(r, input, periods)
    return r

def REF_V[A: np.ndarray | list[np.ndarray]](
  input: A, periods: A
) -> A:
  """
  Right shift input array by variable `periods`, r[i] = input[i - periods[i]]
  
  Ref: https://www.amibroker.com/guide/afl/ref.html
  """
  if isinstance(input, list) and isinstance(periods, list):
    input = [_to_f64(x) for x in input]
    periods = [_to_usize(x) for x in periods]
    r = [np.empty_like(x) for x in input]
    _algo.ref_v(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    periods = _to_usize(periods)
    r = np.empty_like(input)
    _algo.ref_v(r, input, periods)
    return r

def REGBETA[A: np.ndarray | list[np.ndarray]](
  y: A, x: A, periods: int
) -> A:
  """
  Calculate Regression Coefficient (Beta) of Y on X over a moving window
  
  Beta = Cov(X, Y) / Var(X)
  """
  if isinstance(y, list) and isinstance(x, list):
    y = [_to_f64(x) for x in y]
    x = [_to_f64(x) for x in x]
    r = [np.empty_like(x) for x in y]
    _algo.regbeta(r, y, x, periods)
    return r
  else:
    y = _to_f64(y)
    x = _to_f64(x)
    r = np.empty_like(y)
    _algo.regbeta(r, y, x, periods)
    return r

def REGRESI[A: np.ndarray | list[np.ndarray]](
  y: A, x: A, periods: int
) -> A:
  """
  Calculate Regression Residual of Y on X over a moving window
  
  Returns the residual of the last point: epsilon = Y - (alpha + beta * X)
  """
  if isinstance(y, list) and isinstance(x, list):
    y = [_to_f64(x) for x in y]
    x = [_to_f64(x) for x in x]
    r = [np.empty_like(x) for x in y]
    _algo.regresi(r, y, x, periods)
    return r
  else:
    y = _to_f64(y)
    x = _to_f64(x)
    r = np.empty_like(y)
    _algo.regresi(r, y, x, periods)
    return r

def RLONGCROSS[A: np.ndarray | list[np.ndarray]](
  a: A, b: A, n: int
) -> A:
  """
  For 2 arrays A and B, return true if previous N periods A > B, Current A <= B
  """
  if isinstance(a, list) and isinstance(b, list):
    a = [_to_f64(x) for x in a]
    b = [_to_f64(x) for x in b]
    r = [np.empty_like(x, dtype=bool) for x in a]
    _algo.rlongcross(r, a, b, n)
    return r
  else:
    a = _to_f64(a)
    b = _to_f64(b)
    r = np.empty_like(a, dtype=bool)
    _algo.rlongcross(r, a, b, n)
    return r

def SCAN_ADD[A: np.ndarray | list[np.ndarray]](
  input: A, condition: A
) -> A:
  """
  Conditional cumulative add: r[t] = r[t-1] + (cond[t] ? input[t] : 0)
  
  Used for SELF-referencing alpha expressions with additive accumulation.
  Serial within each stock, parallel across stocks via rayon.
  """
  if isinstance(input, list) and isinstance(condition, list):
    input = [_to_f64(x) for x in input]
    condition = [_to_bool(x) for x in condition]
    r = [np.empty_like(x) for x in input]
    _algo.scan_add(r, input, condition)
    return r
  else:
    input = _to_f64(input)
    condition = _to_bool(condition)
    r = np.empty_like(input)
    _algo.scan_add(r, input, condition)
    return r

def SCAN_MUL[A: np.ndarray | list[np.ndarray]](
  input: A, condition: A
) -> A:
  """
  Conditional cumulative multiply: r[t] = r[t-1] * (cond[t] ? input[t] : 1)
  
  Used for SELF-referencing alpha expressions like GTJA #143.
  Serial within each stock, parallel across stocks via rayon.
  """
  if isinstance(input, list) and isinstance(condition, list):
    input = [_to_f64(x) for x in input]
    condition = [_to_bool(x) for x in condition]
    r = [np.empty_like(x) for x in input]
    _algo.scan_mul(r, input, condition)
    return r
  else:
    input = _to_f64(input)
    condition = _to_bool(condition)
    r = np.empty_like(input)
    _algo.scan_mul(r, input, condition)
    return r

def SHARPE[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Rolling Sharpe Ratio of returns.
  
  Sharpe = mean(returns) / stddev(returns)
  Measures risk-adjusted return over a rolling window.
  
  Ref: https://en.wikipedia.org/wiki/Sharpe_ratio
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.sharpe(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.sharpe(r, input, periods)
    return r

def SKEWNESS[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Calculate rolling sample Skewness over a moving window
  
  Uses adjusted Fisher-Pearson formula (matches pandas):
  skew = n / ((n-1)(n-2)) * sum(((x-mean)/std)^3)
  Requires at least 3 valid values.
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.skewness(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.skewness(r, input, periods)
    return r

def SLOPE[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Linear Regression Slope
  
  Calculates the slope of the linear regression line for a moving window.
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.slope(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.slope(r, input, periods)
    return r

def SMA[A: np.ndarray | list[np.ndarray]](
  input: A, n: int, m: int
) -> A:
  """
  Exponential Moving Average (variant of well-known EMA) weight = m / n
  
  Ref: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.sma(r, input, n, m)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.sma(r, input, n, m)
    return r

def SMA_V[A: np.ndarray | list[np.ndarray]](
  input: A, ns: A, m: int
) -> A:
  """
  Exponential Moving Average (variant of well-known EMA) weight = m / n with variable periods
  
  Ref: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
  """
  if isinstance(input, list) and isinstance(ns, list):
    input = [_to_f64(x) for x in input]
    ns = [_to_usize(x) for x in ns]
    r = [np.empty_like(x) for x in input]
    _algo.sma_v(r, input, ns, m)
    return r
  else:
    input = _to_f64(input)
    ns = _to_usize(ns)
    r = np.empty_like(input)
    _algo.sma_v(r, input, ns, m)
    return r

def STDDEV[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Calculate Standard Deviation over a moving window
  
  Ref: https://en.wikipedia.org/wiki/Standard_deviation
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.stddev(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.stddev(r, input, periods)
    return r

def SUM[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Calculate sum of values in preceding `periods` window
  
  If periods is 0, it calculates the cumulative sum from the first valid value.
  
  Ref: https://www.amibroker.com/guide/afl/sum.html
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.sum(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.sum(r, input, periods)
    return r

def SUMBARS[A: np.ndarray | list[np.ndarray]](
  input: A, amount: float
) -> A:
  """
  Calculate number of periods (bars) backwards until the sum of values is greater than or equal to `amount`
  
  Ref: https://www.amibroker.com/guide/afl/sumbars.html
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.sumbars(r, input, amount)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.sumbars(r, input, amount)
    return r

def SUMIF[A: np.ndarray | list[np.ndarray]](
  input: A, condition: A, periods: int
) -> A:
  """
  Calculate sum of values in preceding `periods` window where `condition` is true
  
  Ref: Custom extension
  """
  if isinstance(input, list) and isinstance(condition, list):
    input = [_to_f64(x) for x in input]
    condition = [_to_bool(x) for x in condition]
    r = [np.empty_like(x) for x in input]
    _algo.sumif(r, input, condition, periods)
    return r
  else:
    input = _to_f64(input)
    condition = _to_bool(condition)
    r = np.empty_like(input)
    _algo.sumif(r, input, condition, periods)
    return r

def VAR[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Calculate Variance over a moving window
  
  Variance = (SumSq - (Sum^2)/N) / (N - 1)
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.var(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.var(r, input, periods)
    return r

def WEIGHTED_DELAY[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Calculate weighted delay (exponentially weighted lag)
  
  WEIGHTED_DELAY(x, k) = (k * x[t-1] + (k-1) * x[t-2] + ... + 1 * x[t-k]) / (k*(k+1)/2)
  This is essentially LWMA applied to the lagged (shifted by 1) series over k periods.
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.weighted_delay(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.weighted_delay(r, input, periods)
    return r

def ZSCORE[A: np.ndarray | list[np.ndarray]](
  input: A, periods: int
) -> A:
  """
  Calculate rolling Z-Score over a moving window
  
  Z-Score = (x - mean) / stddev, computed over a rolling window of `periods`.
  Uses sample stddev (ddof=1) to match pandas.
  """
  if isinstance(input, list):
    input = [_to_f64(x) for x in input]
    r = [np.empty_like(x) for x in input]
    _algo.zscore(r, input, periods)
    return r
  else:
    input = _to_f64(input)
    r = np.empty_like(input)
    _algo.zscore(r, input, periods)
    return r

