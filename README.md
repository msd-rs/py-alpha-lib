# Introduction

`alpha-lib` is a Python library that implements various algorithms and functions commonly used in quantitative finance and algorithmic trading.

For financial data analysis, there are many algorithms required a rolling window calculation. This library provides efficient implementations of these algorithms.

## Algorithms

| Name       | Description                                                  | Ref Link                                                                |
| Name       | Description                                                  | Ref Link                                                                |
| ---------- | ------------------------------------------------------------ | ----------------------------------------------------------------------- |
| BARSLAST   | Calculate number of bars since last condition true           | https://www.amibroker.com/guide/afl/barslast.html                       |
| BARSSINCE  | Calculate number of bars since first condition true          | https://www.amibroker.com/guide/afl/barssince.html                      |
| BINS       | Discretize the input into n bins, the ctx.groups() is the number of groups  Bins are 0-based index. Same value are assigned to the same bin. |                                                                         |
| CORR       | Calculate Correlation over a moving window  Correlation = Cov(X, Y) / (StdDev(X) * StdDev(Y)) |                                                                         |
| COUNT      | Calculate number of periods where condition is true in passed `periods` window | https://www.amibroker.com/guide/afl/count.html                          |
| COV        | Calculate Covariance over a moving window  Covariance = (SumXY - (SumX * SumY) / N) / (N - 1) |                                                                         |
| CROSS      | For 2 arrays A and B, return true if A[i-1] < B[i-1] and A[i] >= B[i] alias: golden_cross, cross_ge | https://www.amibroker.com/guide/afl/cross.html                          |
| DMA        | Exponential Moving Average current = weight * current + (1 - weight) * previous | https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average |
| EMA        | Exponential Moving Average (variant of well-known EMA) weight = 2 / (n + 1) | https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average |
| FRET       | Future Return  Calculates the return from the open price of the delayed day (t+delay) to the close price of the future day (t+delay+periods-1). Return = (Close[t+delay+periods-1] - Open[t+delay]) / Open[t+delay]  If n=1, delay=1, it calculates (Close[t+1] - Open[t+1]) / Open[t+1]. If `is_calc[t+delay]` is 0, returns NaN. |                                                                         |
| HHV        | Find highest value in a preceding `periods` window           | https://www.amibroker.com/guide/afl/hhv.html                            |
| HHVBARS    | The number of periods that have passed since the array reached its `periods` period high | https://www.amibroker.com/guide/afl/hhvbars.html                        |
| INTERCEPT  | Linear Regression Intercept  Calculates the intercept of the linear regression line for a moving window. |                                                                         |
| LLV        | Find lowest value in a preceding `periods` window            | https://www.amibroker.com/guide/afl/llv.html                            |
| LLVBARS    | The number of periods that have passed since the array reached its periods period low | https://www.amibroker.com/guide/afl/llvbars.html                        |
| LONGCROSS  | For 2 arrays A and B, return true if previous N periods A < B, Current A >= B |                                                                         |
| LWMA       | Linear Weighted Moving Average  LWMA = SUM(Price * Weight) / SUM(Weight) |                                                                         |
| MA         | Simple Moving Average, also known as arithmetic moving average | https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average      |
| NEUTRALIZE | Neutralize the effect of a categorical variable on a numeric variable |                                                                         |
| PRODUCT    | Calculate product of values in preceding `periods` window  If periods is 0, it calculates the cumulative product from the first valid value. |                                                                         |
| RANK       | Calculate rank percentage cross group dimension, the ctx.groups() is the number of groups Same value are averaged |                                                                         |
| RCROSS     | For 2 arrays A and B, return true if A[i-1] > B[i-1] and A[i] <= B[i] alias: death_cross, cross_le |                                                                         |
| REF        | Right shift input array by `periods`, r[i] = input[i - periods] | https://www.amibroker.com/guide/afl/ref.html                            |
| REGBETA    | Calculate Regression Coefficient (Beta) of Y on X over a moving window  Beta = Cov(X, Y) / Var(X) |                                                                         |
| REGRESI    | Calculate Regression Residual of Y on X over a moving window  Returns the residual of the last point: epsilon = Y - (alpha + beta * X) |                                                                         |
| RLONGCROSS | For 2 arrays A and B, return true if previous N periods A > B, Current A <= B |                                                                         |
| SLOPE      | Linear Regression Slope  Calculates the slope of the linear regression line for a moving window. |                                                                         |
| SMA        | Exponential Moving Average (variant of well-known EMA) weight = m / n | https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average |
| STDDEV     | Calculate Standard Deviation over a moving window            |                                                                         |
| SUM        | Calculate sum of values in preceding `periods` window  If periods is 0, it calculates the cumulative sum from the first valid value. | https://www.amibroker.com/guide/afl/sum.html                            |
| SUMBARS    | Calculate number of periods (bars) backwards until the sum of values is greater than or equal to `amount` | https://www.amibroker.com/guide/afl/sumbars.html                        |
| SUMIF      | Calculate sum of values in preceding `periods` window where `condition` is true |                                                                         |
| TS_RANK    | Calculate rank in a sliding window with size `periods`       |                                                                         |
| VAR        | Calculate Variance over a moving window  Variance = (SumSq - (Sum^2)/N) / (N - 1) |                                                                         |

# Usage

## Installation

You can install the library using pip:

```bash
pip install py-alpha-lib
```

## Simple Example

```python
import alpha as al
import numpy as np

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)

# Calculate 3-period moving average, note that first 2 values are average of available values
result = al.MA(data, 3)
print(result)
# Output: [1.  1.5 2.  3.  4.  5.  6.  7.  8.  9. ]

# Calculate 3-period exponential moving average, first 2 values are NaN
al.set_ctx(flags=al.FLAG_STRICTLY_CYCLE)
result = al.EMA(data, 3)
print(result)
# Output: [ nan  nan 2.  3.  4.  5.  6.  7.  8.  9. ]

# Calculate 3-period exponential moving average, skipping NaN values
al.set_ctx(flags=al.FLAG_SKIP_NAN)
data_with_nan = np.array([1, 2, None, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
result = al.MA(data_with_nan, 3)
print(result)
# Output: [1.  1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5]
```

## Environment Context

You may notice that some functions have different behaviors based on the context settings. You can set the context using `al.set_ctx()` function. The context includes:

- `groups`: Number of groups to divide the data into for group-wise operations. `groups` used calculations multiple stocks(for example) in a single array.
  - Each group is assumed to be of equal size and contiguous in the input array.
  - Each group is processed paralleled and independently. This is why the performance is very good.
  - For `rank` function, groups is required to be set greater than 1. Because rank is a group-wise operation.
- `start`: The starting index for calculations.
  - For some case, this may reduce unnecessary computations.
  - Default is 0.
- `flags`: Additional flags to modify function behaviors.
  - `FLAG_SKIP_NAN`: When this flag is set, functions will skip NaN values during computations.
  - `FLAG_STRICTLY_CYCLE`: When this flag is set, functions will strictly cycle over the data, meaning that initial periods that do not have enough data will be filled with NaN.
  - You can combine multiple flags using bitwise OR operation, e.g., `flags=FLAG_SKIP_NAN | FLAG_STRICTLY_CYCLE`.

## Vibe Coding

When you need LLM to help you implement new factor in python, you can let LLM known which functions are available in `alpha-lib` by providing [the list of supported functions](python/alpha/algo.md) as context.

## Factor expression to Python code

You can convert factor expressions to Python code using the `lang` module. For example:

```bash
python -m alpha.lang examples/wq101/alpha101.txt
```

This will read the factor expressions from [`examples/wq101/alpha101.txt`](examples/wq101/alpha101.txt) and generate corresponding Python code using `alpha-lib` functions.

After generating the code, you may need to adjust the code

- Fix type conversions between `float` and `bool`.
- Add context settings if needed.

# Full Example

## WorldQuant 101 famous alpha 101

[The WorldQuant 101 alpha factors](https://arxiv.org/pdf/1601.009913.pdf) are a set of quantitative trading signals developed by WorldQuant. There are some implementations of these alpha factors, for example:
[DolphinDB implementation: ](https://github.com/dolphindb/DolphinDBModules/blob/master/wq101alpha/README.md), it provides 101 alpha factors implemented in DolphinDB language also with comparative `pandas` based Python implementation. It's a good starting point for comparing with our `alpha-lib`.

The full implementation of these 101 alpha factors using `alpha-lib` can be found in the [wq101](examples/wq101) folder of this repository. This implementation leverages the efficient algorithms provided by `alpha-lib` to compute the alpha factors.

- `al`: is the factor implemented using `alpha-lib`.
- `pd_`: is the factor implemented using `pandas` for comparison.
- Because we can not setup the full featured DolphinDB environment here, we just use it's results.

### Run the example

Show help message:

```
$ examples/wq101/main.py --help
usage: main.py [-h] [-s START] [-e END] [-v] [-d DATA] [-o OUTPUT] [--with-pd] [--with-al] [no ...]

positional arguments:
  no                    alpha numbers to run, e.g., 1 2 3

options:
  -h, --help            show this help message and exit
  -s, --start START     start alpha number
  -e, --end END         end alpha number
  -v, --verbose         enable verbose logging
  -d, --data DATA       data file path
  -o, --output OUTPUT   save output to file
  --with-pd             run pandas implementation
  --with-al             run alpha-lib implementation
```

```bash
# Run specific alpha factors both pandas and alpha-lib implementations
examples/wq101/main.py --with-pd --with-al 1 2 3 4

# Run a range of alpha factors using alpha-lib implementation
examples/wq101/main.py --with-al -s 1 -e 102

```

Because the `pandas` implementation is too slow for some factors, below is a 1~14 factors (`examples/wq101/main.py --with-al -s 1 -e 15`) run time comparison on a sample dataset with 4000 stocks and 261 trading days, total 1,044,000 factors to compute for each factor.

The _pandas/DolphinDB_ is copied from the [DolphinDB implementation result](https://github.com/dolphindb/DolphinDBModules/blob/master/wq101alpha/README.md#31-dolphindb-vs-python-pandas)

The `Value` columns are used to verify the correctness of the implementations, they should be the same or very close.

The hardware/soft environment is:

- CPU: Intel 13th Gen Core i7-13700K (16 cores, 24 threads)
- RAM: 32GB
- OS: Ubuntu 22.04 LTS
- Python: 3.14 without free-threading
- pandas: 3.0
- numpy: 2.4

| no   | pandasTime(ms) | alphaLibTime(ms) | SpeedUp<br/>(pandas/alphaLib) | SpeedUp<br/>(pandas/DolphinDB) | pandasValue | alphaLibValue |
| ---- | -------------- | ---------------- | ----------------------------- | ------------------------------ | ----------- | ------------- |
| data | 11396          | 718              | 15                            |                                |             |               |
| #001 | 14231          | 7                | 2033                          | 800                            | 0.182125    | 0.182125      |
| #002 | 465            | 14               | 33                            | 9                              | -0.64422    | -0.326332     |
| #003 | 430            | 8                | 53                            | 14                             | 0.236184    | 0.236184      |
| #004 | 55107          | 6                | 9184                          | 1193                           | -8          | -8            |
| #005 | 105            | 9                | 11                            | 5                              | -0.331333   | -0.331333     |
| #006 | 351            | 2                | 175                           | 84                             | 0.234518    | 0.234518      |
| #007 | 43816          | 17               | 2577                          | 486                            | -1          | -1            |
| #008 | 222            | 9                | 24                            | 14                             | -0.6435     | -0.6435       |
| #009 | 97             | 9                | 10                            | 14                             | 17.012321   | 17.012321     |
| #010 | 145            | 11               | 13                            | 6                              | 0.662       | 0.662         |
| #011 | 158            | 10               | 15                            | 6                              | 0.785196    | 0.892723      |
| #012 | 4              | 4                | 1                             | 0.7                            | -17.012321  | -17.012321    |
| #013 | 446            | 9                | 49                            | 8                              | -0.58       | -0.58         |
| #014 | 398            | 8                | 49                            | 18                             | 0.095449    | 0.095449      |

# Development

To contribute to the development of `alpha-lib`, you can clone the repository and set up a development environment.

Toolchain requirements:

- Rust (latest stable)
- Python (3.11+)
- [maturin](https://github.com/PyO3/maturin) (for building Python bindings)

## Vibe Coding

This project is co-created with `Gemini-3.0-Pro` , when you want add new algo, use skill [add_algo.md](.agent/skills/add_algo.md) let AI to do correct code change for you.
