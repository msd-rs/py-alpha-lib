# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

from .to_python import to_python_file

if __name__ == "__main__":
  import sys
  import pathlib

  if len(sys.argv) < 2:
    print("Usage: python -m alpha.lang <file | code>")
    sys.exit(1)

  file_path = pathlib.Path(sys.argv[1])
  if file_path.exists():
    codes = file_path.read_text().splitlines()
    to_python_file(codes, name_convertor=lambda x: x.upper())
  else:
    code = "\n".join(sys.argv[1:])
    to_python_file([code], name_convertor=lambda x: x.upper())
