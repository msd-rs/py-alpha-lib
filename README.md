# alpha-lib

High-performance quantitative finance algorithm library, implemented in Rust with Python bindings ([PyO3](https://pyo3.rs/)).

Provides efficient rolling-window calculations commonly used in factor-based quantitative trading.

## Performance

Benchmarked on Alpha 101, 4000 stocks x 261 trading days (1,044,000 data points per factor):

| Implementation | Factors | Data Load | Compute | Total | Speedup |
|---|---|---|---|---|---|
| pandas | 75 | 31.2s | 2,643s | 2,675s (44min) | 1x |
| polars_ta | 81 | 0.3s | 58s | 58s | 46x |
| **alpha-lib** | **101** | **0.3s** | **3.6s** | **3.9s** | **729x** |

See [COMPARISON.md](articles/COMPARISON.md) for per-factor timing and correctness analysis.

## Installation

```bash
pip install py-alpha-lib
```


## Usage

### Context Settings

Control computation behavior via `alpha.set_ctx()`:

- **`groups`** — Number of securities in the data array. Each group is processed independently and in parallel. Required for cross-sectional operations like `RANK`.
- **`start`** — Starting index for calculation (default: 0).
- **`end`** — Ending index for calculation (default: `len(data)`). `end` can be used when you want to calculate only a part of the data. for example, when back test iteratively.
- **`flags`** — Bitwise flags:
  - `FLAG_SKIP_NAN` (1): Skip NaN values in rolling windows.
  - `FLAG_STRICTLY_CYCLE` (2): Return NaN until window is full (matches pandas `rolling()` default).
  - Combine with `|`: `flags=FLAG_SKIP_NAN | FLAG_STRICTLY_CYCLE`

  ```python
  import alpha
  import numpy as np

  data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)

  # 3-period moving average (partial results during warm-up)
  result = alpha.MA(data, 3)
  # [1.  1.5 2.  3.  4.  5.  6.  7.  8.  9.]

  # Strict mode: NaN until window is full
  alpha.set_ctx(flags=alpha.FLAG_STRICTLY_CYCLE)
  result = alpha.MA(data, 3)
  # [nan nan 2.  3.  4.  5.  6.  7.  8.  9.]

  # Skip NaN values
  alpha.set_ctx(flags=alpha.FLAG_SKIP_NAN)
  data_nan = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
  result = alpha.MA(data_nan, 3)
  #[1.    1.5     nan 2.333 3.667 5.    6.    7.    8.    9.   ]
  ```


### Example 1: Plug and Play

```python
import alpha
from alpha.context import ExecContext

# ExecContext auto-infers groups from securityid/tradetime columns
# and calls alpha.set_ctx(groups=...) automatically
data = pl.read_csv("data.csv").sort(["securityid", "tradetime"])
ctx = ExecContext(data)

# Call operators directly on numpy arrays
close = data["close"].to_numpy()
ma20 = alpha.MA(close, 20)
rank = alpha.RANK(close)       # cross-sectional rank (groups auto-configured)
corr = alpha.CORR(close, data["vol"].to_numpy().astype(float), 10)
```

Data layout: flat 1D array `[stock1_day1, stock1_day2, ..., stockN_dayM]`, sorted by security then time. The `groups` parameter tells the library where each stock's data begins.


### Example 2: Factor Expression Transpiler

Convert factor expressions to Python code, then run:

```bash
python -m alpha.lang examples/wq101/alpha101.txt
```

```python
# 3. Use generated code
from alpha.context import ExecContext
from factors import alpha_001

data = pl.read_csv("data.csv").sort(["securityid", "tradetime"])
ctx = ExecContext(data)  # auto-infers groups
result = alpha_001(ctx)
```

## Factor expression to Python code

You can convert factor expressions to Python code using the `lang` module. For example:

```bash
python -m alpha.lang examples/wq101/alpha101.txt
```

This will read the factor expressions from [`examples/wq101/alpha101.txt`](examples/wq101/alpha101.txt) and generate corresponding Python code using `alpha-lib` functions.

After generating the code, you may need to adjust the code

- Fix type conversions between `float` and `bool`.
- Add context settings if needed.

## Benchmarking and Full Examples

### GTJA Alpha 191

Implementation of 190/191 factors from the GTJA (国泰君安) Alpha 191 factor set in [`examples/gtja191/`](examples/gtja191/):

| Metric | Value |
|---|---|
| Computable | 190 / 191 |
| Compute time | ~4.5s (4000 stocks × 261 days) |
| Avg per factor | 24ms |

```bash
python -m examples.gtja191.al 143     # run specific factor
python -m examples.gtja191.al          # run all factors
```

### WorldQuant Alpha 101

Full implementation of [101 Formulaic Alphas](https://arxiv.org/pdf/1601.00991.pdf) in [`examples/wq101/`](examples/wq101/):

- `al/` — alpha-lib implementation (Rust backend)
- `pd_/` — pandas reference (DolphinDB port)
- `pl_/` — polars_ta reference

```bash
examples/wq101/main.py --with-al 1 2 3 4 # Run specific factors
examples/wq101/main.py --with-al -s 1 -e 102 # Run all factors
examples/wq101/main.py --with-pd --with-al -s 1 -e 15 # Compare with pandas
```

Benchmark scripts in [`benchmarks/`](benchmarks/).

### Supported Algorithms

Naming Rules:

- Function starts with `CC_` means it is a cross-commodity/cross-security/cross-group operation.
- Function without prefix means it is a rolling window operation.


| Name | Description |
|---|---|
| BACKFILL | Forward-fill NaN values with the last valid observation |
| BARSLAST | Calculate number of bars since last condition true |
| BARSSINCE | Calculate number of bars since first condition true |
| BINS | Discretize the input into n bins, the ctx.groups() is the number of groups |
| CC_RANK | Calculate rank percentage cross group dimension, the ctx.groups() is the number of groups Same value are averaged |
| CC_ZSCORE | Calculate cross-sectional Z-Score across groups at each time step |
| CORR | Time Series Correlation in moving window on self |
| CORR2 | Calculate two series correlation over a moving window |
| COUNT | Calculate number of periods where condition is true in passed `periods` window |
| COUNT_NANS | Count number of NaN values in a rolling window |
| COV | Calculate Covariance over a moving window |
| CROSS | For 2 arrays A and B, return true if A[i-1] < B[i-1] and A[i] >= B[i] alias: golden_cross, cross_ge |
| DMA | Exponential Moving Average current = weight * current + (1 - weight) * previous |
| EMA | Exponential Moving Average (variant of well-known EMA) weight = 2 / (n + 1) |
| ENTROPY | Calculate rolling Shannon entropy over a moving window |
| FRET | Future Return |
| GROUP_RANK | Calculate rank percentage within each category group at each time step |
| GROUP_ZSCORE | Calculate Z-Score within each category group at each time step |
| HHV | Find highest value in a preceding `periods` window |
| HHVBARS | The number of periods that have passed since the array reached its `periods` period high |
| INTERCEPT | Linear Regression Intercept |
| KURTOSIS | Calculate rolling sample excess Kurtosis over a moving window |
| LLV | Find lowest value in a preceding `periods` window |
| LLVBARS | The number of periods that have passed since the array reached its periods period low |
| LONGCROSS | For 2 arrays A and B, return true if previous N periods A < B, Current A >= B |
| LWMA | Linear Weighted Moving Average |
| MA | Simple Moving Average, also known as arithmetic moving average |
| MIN_MAX_DIFF | Calculate rolling min-max difference (range) over a moving window |
| MOMENT | Calculate rolling k-th central moment over a moving window |
| NEUTRALIZE | Neutralize the effect of a categorical variable on a numeric variable |
| PRODUCT | Calculate product of values in preceding `periods` window |
| RANK | Calculate rank in a sliding window with size `periods` |
| RCROSS | For 2 arrays A and B, return true if A[i-1] > B[i-1] and A[i] <= B[i] alias: death_cross, cross_le |
| REF | Right shift input array by `periods`, r[i] = input[i - periods] |
| REGBETA | Calculate Regression Coefficient (Beta) of Y on X over a moving window |
| REGRESI | Calculate Regression Residual of Y on X over a moving window |
| RLONGCROSS | For 2 arrays A and B, return true if previous N periods A > B, Current A <= B |
| SCAN_ADD | Conditional cumulative add: r[t] = r[t-1] + (cond[t] ? input[t] : 0) |
| SCAN_MUL | Conditional cumulative multiply: r[t] = r[t-1] * (cond[t] ? input[t] : 1) |
| SKEWNESS | Calculate rolling sample Skewness over a moving window |
| SLOPE | Linear Regression Slope |
| SMA | Exponential Moving Average (variant of well-known EMA) weight = m / n |
| STDDEV | Calculate Standard Deviation over a moving window |
| SUM | Calculate sum of values in preceding `periods` window |
| SUMBARS | Calculate number of periods (bars) backwards until the sum of values is greater than or equal to `amount` |
| SUMIF | Calculate sum of values in preceding `periods` window where `condition` is true |
| VAR | Calculate Variance over a moving window |
| WEIGHTED_DELAY | Calculate weighted delay (exponentially weighted lag) |
| ZSCORE | Calculate rolling Z-Score over a moving window |

Full function signatures: [python/alpha/algo.md](python/alpha/algo.md)


## Development

Requirements:
- Rust (latest stable)
- Python 3.11+
- [maturin](https://github.com/PyO3/maturin)

```bash
# Build and install in development mode
maturin develop --release

# Run tests
cargo test
```

### Vibe Coding

When adding new algorithms with LLM assistance, provide [the function list](python/alpha/algo.md) as context. Use the skill [add_algo.md](.agent/skills/add_algo.md) for guided implementation.

This project is a co-created by `Gemini` (through [Antigravity](https://antigravity.google/)) and `Claude` (from [tic-top](https://github.com/tic-top/py-alpha-lib)).