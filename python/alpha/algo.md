List of available functions with python type hints:

the `np.ndarray` is `ndarray` type in `numpy` package

- BARSLAST(input: np.ndarray[bool]): Calculate number of bars since last condition true
- BARSSINCE(input: np.ndarray[bool]): Calculate number of bars since first condition true
- BINS(input: np.ndarray[float], bins: int): Discretize the input into n bins, the ctx.groups() is the number of groups  Bins are 0-based index. Same value are assigned to the same bin.
- CORR(x: np.ndarray[float], y: np.ndarray[float], periods: int): Calculate Correlation over a moving window  Correlation = Cov(X, Y) / (StdDev(X) * StdDev(Y))
- COUNT(input: np.ndarray[bool], periods: int): Calculate number of periods where condition is true in passed `periods` window
- COV(x: np.ndarray[float], y: np.ndarray[float], periods: int): Calculate Covariance over a moving window  Covariance = (SumXY - (SumX * SumY) / N) / (N - 1)
- CROSS(a: np.ndarray[float], b: np.ndarray[float]): For 2 arrays A and B, return true if A[i-1] < B[i-1] and A[i] >= B[i] alias: golden_cross, cross_ge
- DMA(input: np.ndarray[float], weight: float): Exponential Moving Average current = weight * current + (1 - weight) * previous
- EMA(input: np.ndarray[float], periods: int): Exponential Moving Average (variant of well-known EMA) weight = 2 / (n + 1)
- FRET(open: np.ndarray[float], close: np.ndarray[float], is_calc: np.ndarray[float], delay: int, periods: int): Future Return  Calculates the return from the open price of the delayed day (t+delay) to the close price of the future day (t+delay+periods-1). Return = (Close[t+delay+periods-1] - Open[t+delay]) / Open[t+delay]  If n=1, delay=1, it calculates (Close[t+1] - Open[t+1]) / Open[t+1]. If `is_calc[t+delay]` is 0, returns NaN.
- HHV(input: np.ndarray[float], periods: int): Find highest value in a preceding `periods` window
- HHVBARS(input: np.ndarray[float], periods: int): The number of periods that have passed since the array reached its `periods` period high
- INTERCEPT(input: np.ndarray[float], periods: int): Linear Regression Intercept  Calculates the intercept of the linear regression line for a moving window.
- LLV(input: np.ndarray[float], periods: int): Find lowest value in a preceding `periods` window
- LLVBARS(input: np.ndarray[float], periods: int): The number of periods that have passed since the array reached its periods period low
- LONGCROSS(a: np.ndarray[float], b: np.ndarray[float], n: int): For 2 arrays A and B, return true if previous N periods A < B, Current A >= B
- LWMA(input: np.ndarray[float], periods: int): Linear Weighted Moving Average  LWMA = SUM(Price * Weight) / SUM(Weight)
- MA(input: np.ndarray[float], periods: int): Simple Moving Average, also known as arithmetic moving average
- NEUTRALIZE(category: np.ndarray[float], input: np.ndarray[float]): Neutralize the effect of a categorical variable on a numeric variable
- PRODUCT(input: np.ndarray[float], periods: int): Calculate product of values in preceding `periods` window  If periods is 0, it calculates the cumulative product from the first valid value.
- RANK(input: np.ndarray[float]): Calculate rank percentage cross group dimension, the ctx.groups() is the number of groups Same value are averaged
- RCROSS(a: np.ndarray[float], b: np.ndarray[float]): For 2 arrays A and B, return true if A[i-1] > B[i-1] and A[i] <= B[i] alias: death_cross, cross_le
- REF(input: np.ndarray[float], periods: int): Right shift input array by `periods`, r[i] = input[i - periods]
- REGBETA(y: np.ndarray[float], x: np.ndarray[float], periods: int): Calculate Regression Coefficient (Beta) of Y on X over a moving window  Beta = Cov(X, Y) / Var(X)
- REGRESI(y: np.ndarray[float], x: np.ndarray[float], periods: int): Calculate Regression Residual of Y on X over a moving window  Returns the residual of the last point: epsilon = Y - (alpha + beta * X)
- RLONGCROSS(a: np.ndarray[float], b: np.ndarray[float], n: int): For 2 arrays A and B, return true if previous N periods A > B, Current A <= B
- SLOPE(input: np.ndarray[float], periods: int): Linear Regression Slope  Calculates the slope of the linear regression line for a moving window.
- SMA(input: np.ndarray[float], n: int, m: int): Exponential Moving Average (variant of well-known EMA) weight = m / n
- STDDEV(input: np.ndarray[float], periods: int): Calculate Standard Deviation over a moving window
- SUM(input: np.ndarray[float], periods: int): Calculate sum of values in preceding `periods` window  If periods is 0, it calculates the cumulative sum from the first valid value.
- SUMBARS(input: np.ndarray[float], amount: float): Calculate number of periods (bars) backwards until the sum of values is greater than or equal to `amount`
- SUMIF(input: np.ndarray[float], condition: np.ndarray[bool], periods: int): Calculate sum of values in preceding `periods` window where `condition` is true
- TS_RANK(input: np.ndarray[float], periods: int): Calculate rank in a sliding window with size `periods`
- VAR(input: np.ndarray[float], periods: int): Calculate Variance over a moving window  Variance = (SumSq - (Sum^2)/N) / (N - 1)
