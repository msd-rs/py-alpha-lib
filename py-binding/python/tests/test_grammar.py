# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

from lark import Lark


def test_grammar_lark():
  with open("python/alpha/lang/alpha.lark", "r") as f:
    grammar = f.read()

  parser = Lark(grammar, start="start", parser="lalr")

  with open("contrib/alpha101.txt", "r") as f:
    lines = f.readlines()

  success_count = 0
  fail_count = 0

  for i, line in enumerate(lines):
    line = line.strip()
    if not line:
      continue
    try:
      parser.parse(line)
      success_count += 1
    except Exception as e:
      print(f"Failed to parse line {i + 1}: {line}")
      print(e)
      fail_count += 1

  assert success_count == 101
