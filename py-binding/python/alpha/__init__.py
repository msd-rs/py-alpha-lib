# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""
Alpha Library: A high-performance rolling window calculation library implemented in Rust with Python bindings.

Terms:
  - The `A` type in functions below is `numpy.ndarray`

"""

from .algo import *
from .algo import __all__ as _algo_all
from .lang import *
from .lang import __all__ as _lang_all
from .context import Context, ContextToken, ExecContext
from .context import __all__ as _context_all

# skip nan values during computations
FLAG_SKIP_NAN: int = 1
# strictly cycle over the data
FLAG_STRICTLY_CYCLE: int = 2

__all__ = (
  _algo_all
  + _lang_all
  + [
    "FLAG_SKIP_NAN",
    "FLAG_STRICTLY_CYCLE",
  ]
)
