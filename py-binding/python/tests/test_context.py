# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

import concurrent.futures
import pytest
import alpha as al


def test_context_basic():
  """Test basic properties of Context, get_ctx, reset_ctx and set_ctx."""
  al.reset_ctx(start=0, end=0, groups=1, flags=0)
  assert al.get_ctx() == (0, 0, 1, 0)

  ctx = al.Context(start=5, end=10, groups=4, flags=al.FLAG_SKIP_NAN)
  assert ctx.start == 5
  assert ctx.end == 10
  assert ctx.groups == 4
  assert ctx.flags == al.FLAG_SKIP_NAN

  ctx.apply()
  assert al.get_ctx() == (5, 10, 4, al.FLAG_SKIP_NAN)

  current = al.Context.current()
  assert current == ctx

  # set_ctx partial update compatibility
  al.set_ctx(groups=8)
  assert al.get_ctx() == (5, 10, 8, al.FLAG_SKIP_NAN)

  al.reset_ctx()
  assert al.get_ctx() == (0, 0, 1, 0)


def test_context_with_statement():
  """Test that 'with ctx:' automatically applies state and restores on exit."""
  al.reset_ctx(start=0, end=0, groups=1, flags=0)

  ctx = al.Context(groups=10, flags=al.FLAG_STRICTLY_CYCLE)
  assert al.get_ctx() == (0, 0, 1, 0)

  with ctx:
    assert al.get_ctx() == (0, 0, 10, al.FLAG_STRICTLY_CYCLE)

  # Check that state was cleanly restored
  assert al.get_ctx() == (0, 0, 1, 0)


def test_context_with_exception():
  """Test that context is cleanly restored even when an exception is raised."""
  al.reset_ctx(start=0, end=0, groups=1, flags=0)
  ctx = al.Context(groups=99)

  with pytest.raises(RuntimeError):
    with ctx:
      assert al.get_ctx()[2] == 99
      raise RuntimeError("test error")

  assert al.get_ctx() == (0, 0, 1, 0)


def test_context_nested():
  """Test nested context managers restore outer states correctly."""
  al.reset_ctx(start=0, end=0, groups=1, flags=0)

  outer_ctx = al.Context(groups=10)
  inner_ctx = al.Context(groups=20)

  with outer_ctx:
    assert al.get_ctx()[2] == 10
    with inner_ctx:
      assert al.get_ctx()[2] == 20
    # Restored to outer
    assert al.get_ctx()[2] == 10

  # Restored to initial
  assert al.get_ctx() == (0, 0, 1, 0)


def test_context_manual_activate_token():
  """Test token-based activation for workflows spanning multiple functions."""
  al.reset_ctx(start=0, end=0, groups=1, flags=0)

  ctx = al.Context(groups=77)
  token = ctx.activate()
  try:
    assert al.get_ctx()[2] == 77
  finally:
    token.restore()

  assert al.get_ctx() == (0, 0, 1, 0)


def test_thread_local_isolation():
  """Test that each thread has its own independent context (TLS)."""
  al.reset_ctx(start=0, end=0, groups=1, flags=0)

  def worker(target_groups: int):
    # Before applying, thread starts with default context
    ctx = al.Context(groups=target_groups)
    with ctx:
      # Verify thread has its own groups setting
      current = al.get_ctx()
      if current[2] != target_groups:
        return False
    # Verify restored to default in this thread
    return al.get_ctx()[2] == 1

  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(worker, g) for g in [10, 20, 30, 40, 50]]
    results = [f.result() for f in futures]

  assert all(results)
  # Main thread should be untouched
  assert al.get_ctx() == (0, 0, 1, 0)


def test_context_with_algo_calculation():
  """Test that algorithm calculations strictly follow Context in with-block."""
  import numpy as np

  # 3 securities, 4 timestamps each
  x = np.arange(1.0, 13.0)

  ctx_g3 = al.Context(groups=3)
  ctx_g2 = al.Context(groups=2)

  with ctx_g3:
    res3 = al.CC_RANK(x)
    # With groups=3, values at same timestamp are normalized over 3 securities (max is 1.0, min is 1/3)
    assert np.isclose(res3[-1], 1.0)
    assert np.isclose(res3[0], 1.0 / 3.0)

    # Nested with groups=2
    with ctx_g2:
      res2 = al.CC_RANK(x)
      # With groups=2, first timestamp only has 2 securities (max is 1.0, min is 0.5)
      assert np.isclose(res2[0], 0.5)

    # Automatically back to groups=3
    res3_again = al.CC_RANK(x)
    assert np.isclose(res3_again[0], 1.0 / 3.0)

  # Restored to default
  assert al.get_ctx() == (0, 0, 1, 0)
