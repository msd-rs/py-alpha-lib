# THIS FILE IS AUTO-GENERATED, DO NOT EDIT

import numpy as np
from . import _algo

def BARSLAST(
  input: np.ndarray | list[np.ndarray]
) -> np.ndarray | list[np.ndarray]:
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

def BARSSINCE(
  input: np.ndarray | list[np.ndarray]
) -> np.ndarray | list[np.ndarray]:
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

def CORR(
  x: np.ndarray | list[np.ndarray], y: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Calculate Correlation over a moving window
  
  Correlation = Cov(X, Y) / (StdDev(X) * StdDev(Y))
  """
  if isinstance(x, list) and isinstance(y, list):
    r = [np.empty_like(x) for x in x]
    x = [x.astype(float) for x in x]
    y = [x.astype(float) for x in y]
    _algo.corr(r, x, y, periods)
    return r
  else:
    r = np.empty_like(x)
    x = x.astype(float)
    y = y.astype(float)
    _algo.corr(r, x, y, periods)
    return r

def COUNT(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
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

def COV(
  x: np.ndarray | list[np.ndarray], y: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Calculate Covariance over a moving window
  
  Covariance = (SumXY - (SumX * SumY) / N) / (N - 1)
  """
  if isinstance(x, list) and isinstance(y, list):
    r = [np.empty_like(x) for x in x]
    x = [x.astype(float) for x in x]
    y = [x.astype(float) for x in y]
    _algo.cov(r, x, y, periods)
    return r
  else:
    r = np.empty_like(x)
    x = x.astype(float)
    y = y.astype(float)
    _algo.cov(r, x, y, periods)
    return r

def CROSS(
  a: np.ndarray | list[np.ndarray], b: np.ndarray | list[np.ndarray]
) -> np.ndarray | list[np.ndarray]:
  """
  For 2 arrays A and B, return true if A[i-1] < B[i-1] and A[i] >= B[i]
  alias: golden_cross, cross_ge
  """
  if isinstance(a, list) and isinstance(b, list):
    r = [np.empty_like(x, dtype=bool) for x in a]
    a = [x.astype(float) for x in a]
    b = [x.astype(float) for x in b]
    _algo.cross(r, a, b)
    return r
  else:
    r = np.empty_like(a, dtype=bool)
    a = a.astype(float)
    b = b.astype(float)
    _algo.cross(r, a, b)
    return r

def DMA(
  input: np.ndarray | list[np.ndarray], weight: float
) -> np.ndarray | list[np.ndarray]:
  """
  Exponential Moving Average
  current = weight * current + (1 - weight) * previous
  
  Ref: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.dma(r, input, weight)
    return r
  else:
    r = np.empty_like(input)
    _algo.dma(r, input, weight)
    return r

def FRET(
  open: np.ndarray | list[np.ndarray], high: np.ndarray | list[np.ndarray], low: np.ndarray | list[np.ndarray], close: np.ndarray | list[np.ndarray], delay: int, periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Future Return
  
  Calculates the return from the open price of the delayed day (t+delay) to the close price of the future day (t+delay+periods-1).
  Return = (Close[t+delay+periods-1] - Open[t+delay]) / Open[t+delay]
  
  If n=1, delay=1, it calculates (Close[t+1] - Open[t+1]) / Open[t+1].
  If High[t+delay] == Open[t+delay] == Low[t+delay] == Close[t+delay], returns NaN.
  """
  if isinstance(open, list) and isinstance(high, list) and isinstance(low, list) and isinstance(close, list):
    r = [np.empty_like(x) for x in open]
    open = [x.astype(float) for x in open]
    high = [x.astype(float) for x in high]
    low = [x.astype(float) for x in low]
    close = [x.astype(float) for x in close]
    _algo.fret(r, open, high, low, close, delay, periods)
    return r
  else:
    r = np.empty_like(open)
    open = open.astype(float)
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    _algo.fret(r, open, high, low, close, delay, periods)
    return r

def HHV(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Find highest value in a preceding `periods` window
  
  Ref: https://www.amibroker.com/guide/afl/hhv.html
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.hhv(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.hhv(r, input, periods)
    return r

def HHVBARS(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  The number of periods that have passed since the array reached its `periods` period high
  
  Ref: https://www.amibroker.com/guide/afl/hhvbars.html
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.hhvbars(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.hhvbars(r, input, periods)
    return r

def INTERCEPT(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Linear Regression Intercept
  
  Calculates the intercept of the linear regression line for a moving window.
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.intercept(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.intercept(r, input, periods)
    return r

def LLV(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Find lowest value in a preceding `periods` window
  
  Ref: https://www.amibroker.com/guide/afl/llv.html
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.llv(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.llv(r, input, periods)
    return r

def LLVBARS(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  The number of periods that have passed since the array reached its periods period low
  
  Ref: https://www.amibroker.com/guide/afl/llvbars.html
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.llvbars(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.llvbars(r, input, periods)
    return r

def LONGCROSS(
  a: np.ndarray | list[np.ndarray], b: np.ndarray | list[np.ndarray], n: int
) -> np.ndarray | list[np.ndarray]:
  """
  For 2 arrays A and B, return true if previous N periods A < B, Current A >= B
  """
  if isinstance(a, list) and isinstance(b, list):
    r = [np.empty_like(x, dtype=bool) for x in a]
    a = [x.astype(float) for x in a]
    b = [x.astype(float) for x in b]
    _algo.longcross(r, a, b, n)
    return r
  else:
    r = np.empty_like(a, dtype=bool)
    a = a.astype(float)
    b = b.astype(float)
    _algo.longcross(r, a, b, n)
    return r

def LWMA(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Linear Weighted Moving Average
  
  LWMA = SUM(Price * Weight) / SUM(Weight)
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.lwma(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.lwma(r, input, periods)
    return r

def MA(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Simple Moving Average, also known as arithmetic moving average
  
  Ref: https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.ma(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.ma(r, input, periods)
    return r

def PRODUCT(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Calculate product of values in preceding `periods` window
  
  If periods is 0, it calculates the cumulative product from the first valid value.
  
  Ref: https://www.amibroker.com/guide/afl/product.html
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.product(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.product(r, input, periods)
    return r

def RANK(
  input: np.ndarray | list[np.ndarray]
) -> np.ndarray | list[np.ndarray]:
  """
  Calculate rank percentage cross group dimension, the ctx.groups() is the number of groups
  Same value are averaged
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.rank(r, input)
    return r
  else:
    r = np.empty_like(input)
    _algo.rank(r, input)
    return r

def RCROSS(
  a: np.ndarray | list[np.ndarray], b: np.ndarray | list[np.ndarray]
) -> np.ndarray | list[np.ndarray]:
  """
  For 2 arrays A and B, return true if A[i-1] > B[i-1] and A[i] <= B[i]
  alias: death_cross, cross_le
  """
  if isinstance(a, list) and isinstance(b, list):
    r = [np.empty_like(x, dtype=bool) for x in a]
    a = [x.astype(float) for x in a]
    b = [x.astype(float) for x in b]
    _algo.rcross(r, a, b)
    return r
  else:
    r = np.empty_like(a, dtype=bool)
    a = a.astype(float)
    b = b.astype(float)
    _algo.rcross(r, a, b)
    return r

def REF(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Right shift input array by `periods`, r[i] = input[i - periods]
  
  Ref: https://www.amibroker.com/guide/afl/ref.html
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.ref(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.ref(r, input, periods)
    return r

def RLONGCROSS(
  a: np.ndarray | list[np.ndarray], b: np.ndarray | list[np.ndarray], n: int
) -> np.ndarray | list[np.ndarray]:
  """
  For 2 arrays A and B, return true if previous N periods A > B, Current A <= B
  """
  if isinstance(a, list) and isinstance(b, list):
    r = [np.empty_like(x, dtype=bool) for x in a]
    a = [x.astype(float) for x in a]
    b = [x.astype(float) for x in b]
    _algo.rlongcross(r, a, b, n)
    return r
  else:
    r = np.empty_like(a, dtype=bool)
    a = a.astype(float)
    b = b.astype(float)
    _algo.rlongcross(r, a, b, n)
    return r

def SLOPE(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Linear Regression Slope
  
  Calculates the slope of the linear regression line for a moving window.
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.slope(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.slope(r, input, periods)
    return r

def SMA(
  input: np.ndarray | list[np.ndarray], n: int, m: int
) -> np.ndarray | list[np.ndarray]:
  """
  Exponential Moving Average (variant of well-known EMA) weight = m / n
  
  Ref: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.sma(r, input, n, m)
    return r
  else:
    r = np.empty_like(input)
    _algo.sma(r, input, n, m)
    return r

def STDDEV(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Calculate Standard Deviation over a moving window
  
  Ref: https://en.wikipedia.org/wiki/Standard_deviation
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.stddev(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.stddev(r, input, periods)
    return r

def SUM(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Calculate sum of values in preceding `periods` window
  
  If periods is 0, it calculates the cumulative sum from the first valid value.
  
  Ref: https://www.amibroker.com/guide/afl/sum.html
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.sum(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.sum(r, input, periods)
    return r

def SUMBARS(
  input: np.ndarray | list[np.ndarray], amount: float
) -> np.ndarray | list[np.ndarray]:
  """
  Calculate number of periods (bars) backwards until the sum of values is greater than or equal to `amount`
  
  Ref: https://www.amibroker.com/guide/afl/sumbars.html
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.sumbars(r, input, amount)
    return r
  else:
    r = np.empty_like(input)
    _algo.sumbars(r, input, amount)
    return r

def TS_RANK(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Calculate rank in a sliding window with size `periods`
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.ts_rank(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.ts_rank(r, input, periods)
    return r

def VAR(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Calculate Variance over a moving window
  
  Variance = (SumSq - (Sum^2)/N) / (N - 1)
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.var(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.var(r, input, periods)
    return r

