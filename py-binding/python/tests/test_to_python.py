# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

from pathlib import Path
from alpha.lang.to_python import to_python, to_python_file
import io


def test_basic_arithmetic():
  code = "1 + 2 * 3"
  py_code = to_python("test_arith", code, as_function=False)
  assert "1 + 2 * 3" in py_code


def test_power():
  code = "2 ^ 3"
  py_code = to_python("test_power", code, as_function=False)
  assert "np.power(2, 3)" in py_code


def test_ternary():
  code = "a > b ? 1 : 0"
  py_code = to_python("test_ternary", code, as_function=False)
  assert "np.where(ctx('a') > ctx('b'), 1, 0)" in py_code


def test_logic():
  code = "a && b"
  py_code = to_python("test_logic", code, as_function=False)
  assert "np.bitwise_and(ctx('a'), ctx('b'))" in py_code

  code = "a || b"
  py_code = to_python("test_logic_or", code, as_function=False)
  assert "np.bitwise_or(ctx('a'), ctx('b'))" in py_code


def test_function_mode():
  code = "close / open"
  py_code = to_python("alpha1", code, as_function=True)
  expected_def = "def alpha1(ctx: ExecContext):"
  expected_return = "return ctx('close') / ctx('open')"
  assert expected_def in py_code
  assert expected_return in py_code


def test_optimize():
  code = "close + close"
  py_code = to_python("alpha_opt", code, as_function=True, optimize=True)
  assert "_close = ctx('close')" in py_code
  assert "return _close + _close" in py_code


def test_name_convertor():
  code = "CLOSE / OPEN"
  py_code = to_python(
    "alpha_conv", code, as_function=False, name_convertor=lambda s: s.lower()
  )
  assert "ctx('close') / ctx('open')" in py_code


def test_to_python_file():
  code_file = "contrib/alpha101.txt"
  py_code = io.StringIO()
  codes = Path(code_file).read_text().splitlines()
  to_python_file(codes, fp=py_code, name_convertor=lambda s: s.upper())
  code = py_code.getvalue()
  exec(code)


def test_alpha_001():
  code = "(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) -0.5)"
  py_code = to_python("alpha_001", code, as_function=True)
  assert (
    "ctx.rank(ctx.Ts_ArgMax(ctx.SignedPower(np.where(ctx('returns') < 0, ctx.stddev(ctx('returns'), 20), ctx('close')), 2.), 5)) - 0.5"
    in py_code
  )
