import alpha as al
import numpy as np
import pandas as pd

np.set_printoptions(precision=3, suppress=True)

# Calculate 3-period moving average, note that first 2 values are average of available values
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float64)
result = al.MA(data, 3)
print(result)
# Output: [1.  1.5 2.  3.  4.  5.  6.  7.  8.  9. ]

# Calculate 3-period exponential moving average, first 2 values are NaN
al.set_ctx(flags=al.FLAG_STRICTLY_CYCLE)
result = al.MA(data, 3)
print(result)
# Output: [nan nan  2.  3.  4.  5.  6.  7.  8.  9.]


# Calculate 3-period exponential moving average, skipping NaN values
al.set_ctx(flags=al.FLAG_SKIP_NAN)
data_with_nan = np.array([1, 2, None, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
result = al.MA(data_with_nan, 3)
print(result)
# Output: [1.    1.5     nan 2.333 3.667 5.    6.    7.    8.    9.   ]



# Calculate Slope
al.set_ctx(flags=0,groups=2)
data_slope = np.array([1, 3, 5, 7, 9,1, 3, 5, 7, 9], dtype=np.float64)
result_slope = al.SLOPE(data_slope, 3)
print("SLOPE(3):", result_slope)

data_slope = np.array([1, 3, 5, 7, 9,1, 3, 5, 7, 9], dtype=np.float64)
result_slope = al.INTERCEPT(data_slope, 3)
print("INTERCEPT(3):", result_slope)
# Output should show 2.0 for full windows


# Calculate Future Return (FRET)
# Reset groups to 1 to treat data as single series
al.set_ctx(flags=al.FLAG_STRICTLY_CYCLE, groups=2)
open_p = np.array([10, 11, 12, 13, 14, 15] * 2, dtype=np.float64)
high_p = np.array([11, 12, 12, 14, 15, 16] * 2, dtype=np.float64)
low_p = np.array([9, 10, 12, 12, 13, 14] * 2, dtype=np.float64)
close_p = np.array([10.5, 11.5, 12, 13.5, 14.5, 15.5] * 2, dtype=np.float64)
df_ohlc = pd.DataFrame({"open": open_p, "high": high_p, "low": low_p, "close": close_p})
df_ohlc["is_calc"] = (
    ~(
        (df_ohlc["high"] == df_ohlc["open"])
        & (df_ohlc["open"] == df_ohlc["low"])
        & (df_ohlc["low"] == df_ohlc["close"])
    )
).astype(np.float64)
is_calc = df_ohlc["is_calc"].to_numpy(dtype=np.float64)

# FRET(delay=1, periods=3):  Return = (Close[i+delay+periods-1] - Open[i+delay]) / Open[i+delay]
result_fret1 = al.FRET(open_p, close_p, is_calc, 1, 3)
print("FRET(1, 3):", result_fret1)

# FRET(delay=2, periods=1):  Return = (Close[i+delay+periods-1] - Open[i+delay]) / Open[i+delay]
# Shifted by 1 day relative to default.
result_fret2 = al.FRET(open_p, close_p, is_calc, 2, 1)
print("FRET(2, 1):", result_fret2)

# Calculate Bins
# Group data into 2 bins
al.set_ctx(flags=al.FLAG_STRICTLY_CYCLE, groups=2)
data_bins = np.array([1, 2, 3, 4, 5,11, 22, 33, 44, 0], dtype=np.float64)
result_bins = al.BINS(data_bins, 2)
print("BINS(5, 2):", result_bins)
# Expected: [0. 0. 0. 1. 1.] (based on 0-based index logic)

df = pd.DataFrame(
    {
        "time":  ["t0","t1","t2","t3"] * 3,
        "stock": ["A"]*4 + ["B"]*4 + ["C"]*4,
        "category": [1,1,1,1,   1,2,1,2,   1,3,3,1],
        "value":    [1,2,3,4,   5,6,7,8,   9,10,11,np.nan],
    }
)
# 1) 先排序，保证每个stock内部按time排列；同时保证stock顺序固定
df = df.sort_values(["stock", "time"], kind="mergesort").reset_index(drop=True)

# 2) 转成库需要的一维布局：group-major（每个stock一组，拼起来）
category = df["category"].to_numpy(dtype=np.float64)
value = df["value"].to_numpy(dtype=np.float64)

# 3) groups=股票数
groups = df["stock"].nunique()
al.set_ctx(flags=0, groups=groups)

result_neutralize = al.NEUTRALIZE(category, value)
print("NEUTRALIZE:", result_neutralize)

df = df.assign(neutralize=result_neutralize)
print(df.pivot_table(index="time", columns="stock", values="neutralize"))

