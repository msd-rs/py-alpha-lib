# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

from .algo import *
from .lang import *
from .context import ExecContext


# skip nan values during computations
FLAG_SKIP_NAN: int = 1
# strictly cycle over the data
FLAG_STRICTLY_CYCLE: int = 2
