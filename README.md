# Introduction

`alpha-lib` is a Python library that implements various algorithms and functions commonly used in quantitative finance and algorithmic trading.

For financial data analysis, there are many algorithms required a rolling window calculation. This library provides efficient implementations of these algorithms.

## Algorithms

| Name       | Description                                                  | Ref Link                                                                |
| ---------- | ------------------------------------------------------------ | ----------------------------------------------------------------------- |
| BARSLAST   | Bars since last condition true                               | https://www.amibroker.com/guide/afl/barslast.html                       |
| BARSSINCE  | Bars since first condition true                              | https://www.amibroker.com/guide/afl/barssince.html                      |
| COUNT      | Count periods where condition is true                        | https://www.amibroker.com/guide/afl/count.html                          |
| CROSS      | CROSS(A, B): Previous A < B, Current A >= B                  | https://www.amibroker.com/guide/afl/cross.html                          |
| DMA        | Exponential Moving Average                                   | https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average |
| EMA        | Exponential Moving Average(weight = 2 / (n + 1))             | https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average |
| HHV        | Highest High Value                                           | https://www.amibroker.com/guide/afl/hhv.html                            |
| HHVBARS    | Bars since Highest High Value                                | https://www.amibroker.com/guide/afl/hhvbars.html                        |
| LLV        | Lowest Low Value                                             | https://www.amibroker.com/guide/afl/llv.html                            |
| LLVBARS    | Bars since Lowest Low Value                                  | https://www.amibroker.com/guide/afl/llvbars.html                        |
| LONGCROSS  | LONGCROSS(A,B,N): Previous N A < B, Current A >= B           |                                                                         |
| MA         | Moving Average                                               | https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average      |
| RANK       | rank by group dim                                            |                                                                         |
| RCROSS     | RCROSE(A, B): Previous A > B, Current A <= B                 |                                                                         |
| REF        | Reference to value N periods ago                             | https://www.amibroker.com/guide/afl/ref.html                            |
| RLONGCROSS | RLONGCROSS(A,B,N): Previous N A > B, Current A <= B          |                                                                         |
| SMA        | Exponential Moving Average (weight = m / n)                  | https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average |
| SUM        | Sum of value N periods ago                                   | https://www.amibroker.com/guide/afl/sum.html                            |
| SUMBARS    | Sums X backwards until the sum is greater than or equal to A | https://www.amibroker.com/guide/afl/sumbars.html                        |
| TS_RANK    | rank by ts dim                                               |

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
