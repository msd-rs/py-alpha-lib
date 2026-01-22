# THIS FILE IS AUTO-GENERATED, DO NOT EDIT

import numpy as np
from . import _algo

def BARSLAST(
  input: np.ndarray | list[np.ndarray]
) -> np.ndarray | list[np.ndarray]:
  """
  Bars since last condition true
  
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
  Bars since first condition true
  
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

def COUNT(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Count periods where condition is true
  
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

def CROSS(
  a: np.ndarray | list[np.ndarray], b: np.ndarray | list[np.ndarray]
) -> np.ndarray | list[np.ndarray]:
  """
  CROSS(A, B): Previous A < B, Current A >= B
  
  Ref: https://www.amibroker.com/guide/afl/cross.html
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
  input: np.ndarray | list[np.ndarray], alpha: float
) -> np.ndarray | list[np.ndarray]:
  """
  Exponential Moving Average
  
  https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
  
  current = alpha * current + (1 - alpha) * previous
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.dma(r, input, alpha)
    return r
  else:
    r = np.empty_like(input)
    _algo.dma(r, input, alpha)
    return r

def HHV(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Highest High Value
  
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
  Bars since Highest High Value
  
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

def LLV(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Lowest Low Value
  
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
  Bars since Lowest Low Value
  
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
  LONGCROSS(A,B,N): Previous N A < B, Current A >= B
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

def MA(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Moving Average
  
  https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average
  
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.ma(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.ma(r, input, periods)
    return r

def RANK(
  input: np.ndarray | list[np.ndarray]
) -> np.ndarray | list[np.ndarray]:
  """
  rank by group dim
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
  RCROSE(A, B): Previous A > B, Current A <= B
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
  Reference to value N periods ago
  
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
  RLONGCROSS(A,B,N): Previous N A > B, Current A <= B
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

def SMA(
  input: np.ndarray | list[np.ndarray], n: int, m: int
) -> np.ndarray | list[np.ndarray]:
  """
  Exponential Moving Average (variant of EMA)
  
  alpha = m / n
  
  https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.sma(r, input, n, m)
    return r
  else:
    r = np.empty_like(input)
    _algo.sma(r, input, n, m)
    return r

def SUM(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Sum of value N periods ago
  
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
  Sums X backwards until the sum is greater than or equal to A
  
  Returns the number of periods (bars) passed.
  
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
  rank by ts dim
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.ts_rank(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.ts_rank(r, input, periods)
    return r

