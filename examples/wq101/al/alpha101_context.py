import pandas as pd
import numpy as np
import alpha


def returns(a: np.ndarray):
  return a / alpha.REF(a, 1) - 1


class ExecContext:
  def __init__(self, data: pd.DataFrame):
    self.OPEN = data["open"].to_numpy()
    self.HIGH = data["high"].to_numpy()
    self.LOW = data["low"].to_numpy()
    self.CLOSE = data["close"].to_numpy()
    self.VOLUME = data["vol"].to_numpy().astype(np.float64)
    self.RETURNS = returns(data["close"].to_numpy())
    self.VWAP = data["vwap"].to_numpy()

  def __call__(self, name: str) -> np.ndarray:
    if name.startswith("ADV"):
      n = name[3:]
      if len(n) == 0:
        return self.VOLUME
      else:
        w = int(n)
        return self.SMA(self.VOLUME, w)
    return getattr(self, name)

  def SUM(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.SUM(a, int(w))

  def SMA(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.MA(a, int(w))

  def STDDEV(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.STDDEV(a, int(w))

  def CORRELATION(self, a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    return alpha.CORR(a, b, int(w))

  def COVARIANCE(self, a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    return alpha.COV(a, b, int(w))

  def TS_RANK(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.TS_RANK(a, int(w))

  def PRODUCT(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.PRODUCT(a, int(w))

  def TS_MIN(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.LLV(a, int(w))

  def MIN(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a, b)

  def TS_MAX(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.HHV(a, int(w))

  def MAX(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)

  def DELTA(self, a: np.ndarray, p: int) -> np.ndarray:
    return a - alpha.REF(a, int(p))

  def DELAY(self, a: np.ndarray, p: int) -> np.ndarray:
    return alpha.REF(a, int(p))

  def SCALE(self, a: np.ndarray, k: int = 1) -> np.ndarray:
    sum = np.abs(a).sum()
    return a * k / sum

  def RANK(self, a: np.ndarray) -> np.ndarray:
    return alpha.RANK(a)

  def TS_ARGMAX(self, a: np.ndarray, w: int) -> np.ndarray:
    return w - alpha.HHVBARS(a, int(w))

  def TS_ARGMIN(self, a: np.ndarray, w: int) -> np.ndarray:
    return w - alpha.LLVBARS(a, int(w))

  def DECAY_LINEAR(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.LWMA(a, int(w))

  def SIGNEDPOWER(self, a: np.ndarray, p: float | np.ndarray) -> np.ndarray:
    return np.power(a, p)

  def LOG(self, a: np.ndarray) -> np.ndarray:
    return np.log(a)

  def ABS(self, a: np.ndarray) -> np.ndarray:
    return np.abs(a)

  def SIGN(self, a: np.ndarray) -> np.ndarray:
    return np.sign(a)
