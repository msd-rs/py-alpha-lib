import numpy as np
import sys
import os

# Ensure we can import alpha.
# If alpha is installed in site-packages, fine.
# If not, we might need to add path.
# Assuming maturin develop was run or we run this with `maturin run` or similar.
# Or if we just build, we might need to point to target/debug?

try:
  import alpha
  from alpha.algo import SUMIF

  print("Successfully imported alpha.algo.SUMIF")
except ImportError:
  # Try adding local python dir if valid? Or check target.
  # But usually in this env we should run maturin develop
  print("Could not import alpha.algo.SUMIF directly.")
  sys.exit(1)


def test_sumif():
  a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
  c = np.array([True, False, True, False, True])
  # periods = 3
  # 0: [0] -> 1 (T) -> 1
  # 1: [0,1] -> 1(T), 2(F) -> 1
  # 2: [0,1,2] -> 1(T), 2(F), 3(T) -> 1+3=4
  # 3: [1,2,3] -> 2(F), 3(T), 4(F) -> 3
  # 4: [2,3,4] -> 3(T), 4(F), 5(T) -> 3+5=8

  expected = np.array([1.0, 1.0, 4.0, 3.0, 8.0])

  res = SUMIF(a, c, 3)
  print("Input A:", a)
  print("Input C:", c)
  print("Result:", res)

  np.testing.assert_allclose(res, expected, rtol=1e-5, equal_nan=True)
  print("Test Passed!")


if __name__ == "__main__":
  test_sumif()
