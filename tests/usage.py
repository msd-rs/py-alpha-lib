import alpha as al
import numpy as np

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
al.set_ctx(flags=0, groups=1)
open_p = np.array([10, 11, 12, 13, 14, 15], dtype=np.float64)
high_p = np.array([11, 12, 12, 14, 15, 16], dtype=np.float64)
low_p = np.array([9, 10, 12, 12, 13, 14], dtype=np.float64)
close_p = np.array([10.5, 11.5, 12, 13.5, 14.5, 15.5], dtype=np.float64)

# FRET(delay=1, periods=3):  Return = (Close[i+delay+periods-1] - Open[i+delay]) / Open[i+delay]
result_fret1 = al.FRET(open_p, high_p, low_p, close_p, 1, 3)
print("FRET(1, 3):", result_fret1)

# FRET(delay=2, periods=1):  Return = (Close[i+delay+periods-1] - Open[i+delay]) / Open[i+delay]
# Shifted by 1 day relative to default.
result_fret2 = al.FRET(open_p, high_p, low_p, close_p, 2, 1)
print("FRET(2, 1):", result_fret2)
