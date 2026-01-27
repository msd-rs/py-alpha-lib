import sys
from typing import Callable
from .parser import Lark_StandAlone, Transformer, v_args
import numpy as np
import re
import math
import io

parser = Lark_StandAlone()


class ExecContext:
  def __call__(self, name: str) -> np.ndarray:
    pass


@v_args(inline=True)
class AlphaTransformer(Transformer):
  def __init__(
    self,
    name_convertor: Callable[[str], str] | None = None,
  ):
    self.name_convertor = name_convertor
    self.variables = set()

  def start(self, expr):
    return expr

  def ternary_expr(self, cond, true_case, false_case):
    return f"np.where({cond}, {true_case}, {false_case})"

  def logical_or_expr(self, left, *rights):
    result = left
    for right in rights:
      result = f"np.bitwise_or({result}, {right})"
    return result

  def logical_and_expr(self, left, *rights):
    result = left
    for right in rights:
      result = f"np.bitwise_and({result}, {right})"
    return result

  def eq(self, left, right):
    return f"{left} == {right}"

  def ne(self, left, right):
    return f"{left} != {right}"

  def lt(self, left, right):
    return f"{left} < {right}"

  def gt(self, left, right):
    return f"{left} > {right}"

  def le(self, left, right):
    return f"{left} <= {right}"

  def ge(self, left, right):
    return f"{left} >= {right}"

  def sum(self, first, *rest):
    result = first
    it = iter(rest)
    for op, val in zip(it, it):
      result = f"{result} {op} {val}"
    return result

  def product(self, first, *rest):
    result = first
    it = iter(rest)
    for op, val in zip(it, it):
      result = f"{result} {op} {val}"
    return result

  def power(self, base, *rest):
    result = base
    it = iter(rest)
    for op, val in zip(it, it):
      result = f"np.power({result}, {val})"
    return result

  def neg(self, minus, item):
    return f"-{item}"

  def func_call(self, name, args=""):
    # Unwrap ctx('...') if present, because function names shouldn't be wrapped
    if name.startswith("ctx('") and name.endswith("')"):
      name = name[5:-2]
    return f"ctx.{name}({args})"

  def arguments(self, *args):
    return ", ".join(args)

  def NAME(self, name):
    name = str(name)
    if self.name_convertor:
      name_key = self.name_convertor(name)
    else:
      name_key = name
    self.variables.add(name_key)
    return f"ctx('{name_key}')"

  def NUMBER(self, name):
    return str(name)

  def dotted_name(self, *names):
    real_names = []
    for n in names:
      if n.startswith("ctx('") and n.endswith("')"):
        real_names.append(n[5:-2])
      else:
        real_names.append(n)

    full_name = ".".join(real_names)

    # Treating dotted name as a variable access string too, similar to NAME
    # Assuming dotted names are also data fields provided by ctx
    if self.name_convertor:
      key = self.name_convertor(full_name)
    else:
      key = full_name
    self.variables.add(key)
    return f"ctx('{key}')"

  def add_op(self, op):
    return str(op)

  def mul_op(self, op):
    return str(op)


def to_python(
  name: str,
  code: str,
  /,
  indent: int = 0,
  indent_by: str = "  ",
  as_function: bool = False,
  name_convertor: Callable[[str], str] | None = None,
  optimize: bool = False,
) -> str:
  """
  Convert a parse tree to Python code.

  There are two modes:
    1. Function mode: Convert the code as a function.
      - All function arguments are (ctx: ExecContext)
      - In generated function, convert each variable name to ctx('VARIABLE_NAME') to get the data.
      - Return the result of the code.
    2. Variable mode: Convert the code as a variable.
      - assume there is a global ExecContext variable named 'ctx'
      - In generated variable, convert each variable name to ctx('VARIABLE_NAME') to get the data.

  Args:
    name: The name of the target function or variable.
    code: The code to convert.
    indent: The init number of spaces to indent the code.
    indent_by: The string to use for indentation.
    as_function: Whether to convert the code as a function or a variable.
    name_convertor: A optional function to convert the identifier name in the code. For example, 'to_lower_case' or 'to_snake_case'.
    optimize: In function mode, optimize the code by declare variables when multiple times used.

  Returns:
    The converted code.
  """
  if not code.strip():
    return ""

  try:
    tree = parser.parse(code)
  except Exception as e:
    raise ValueError(f"Failed to parse code: {code}") from e

  transformer = AlphaTransformer(name_convertor=name_convertor)
  converted_expr = transformer.transform(tree)

  indent_str = indent_by * indent

  if as_function:
    lines = []
    lines.append(f"{indent_str}def {name}(ctx):")

    body_indent = indent_str + indent_by

    if optimize:
      # Count occurrences
      var_usage = {}
      for var in transformer.variables:
        pattern = re.escape(f"ctx('{var}')")
        count = len(re.findall(pattern, converted_expr))
        var_usage[var] = count

      # Sort variables to ensure consistent output
      sorted_vars = sorted([v for v, c in var_usage.items() if c > 1])

      for var in sorted_vars:
        safe_var_name = "_" + var.replace(".", "_")  # simple safe name
        lines.append(f"{body_indent}{safe_var_name} = ctx('{var}')")
        # Replace in expression
        converted_expr = converted_expr.replace(f"ctx('{var}')", safe_var_name)

    lines.append(f"{body_indent}return {converted_expr}")
    return "\n".join(lines)
  else:
    return f"{indent_str}{name} = {converted_expr}"


def to_python_file(
  codes: list[str],
  names: list[str] | str = "alpha_",
  /,
  fp: io.StringIO | None = None,
  imports: list[str] | None = [],
  name_convertor: Callable[[str], str] | None = None,
):
  if isinstance(names, str):
    n = len(codes)
    w = math.ceil(math.log10(n))
    names = [f"{names}{i + 1:0{w}d}" for i in range(n)]

  assert len(names) == len(codes)

  if fp is None:
    fp = sys.stdout

  for i in imports:
    print(f"{i}", file=fp)

  if "import numpy as np" not in imports:
    print("import numpy as np", file=fp)

  for name, code in zip(names, codes):
    print(f"# {code}", file=fp)
    py_code = to_python(
      name, code, as_function=True, optimize=True, name_convertor=name_convertor
    )
    print(py_code, file=fp)
    print("\n\n", file=fp)
