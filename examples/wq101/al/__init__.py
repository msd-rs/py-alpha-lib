import alpha
import time
import pandas as pd
import logging
import numpy as np
from .alpha101_context import ExecContext
from . import alpha101
import argparse


logger = logging.getLogger("alpha_backend")


def setup_context(data_path: str) -> (ExecContext, int, int):
  logger.info("Loading data")
  t1 = time.time()
  data = pd.read_csv(data_path)
  df = data.set_index(["securityid", "tradetime"])
  security_count = df.index.get_level_values("securityid").value_counts().shape[0]
  trade_count = df.index.get_level_values("tradetime").value_counts().shape[0]
  alpha.set_ctx(groups=security_count)
  ctx = ExecContext(df)
  t2 = time.time()
  logger.info("Data loaded in %f seconds", t2 - t1)
  return ctx, trade_count, security_count, int((t2 - t1) * 1000)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("no", nargs="*", type=int)
  parser.add_argument("-s", "--start", type=int, required=False)
  parser.add_argument("-e", "--end", type=int, required=False)
  parser.add_argument("-v", "--verbose", action="store_true", required=False)
  parser.add_argument(
    "-d", "--data", type=str, required=False, default="dataPerformance.csv"
  )
  return parser.parse_args()


nofunc = set(
  [48, 56, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100]
)


def main(args):
  ctx, trade_count, security_count, load_time = setup_context(args.data)

  if len(args.no) == 0:
    start = args.start or 1
    end = args.end or 102
    args.no = [i for i in filter(lambda x: x not in nofunc, range(start, end))]

  results = [("data", load_time)]
  for no in args.no:
    if no in nofunc:
      logger.info("unsupported alpha %d", no)
    else:
      fn_name = f"alpha_{no:03d}"
      t1 = time.time()
      logger.info("Computing alpha %s", fn_name)
      fn = getattr(alpha101, fn_name)
      v = fn(ctx)
      t2 = time.time()
      logger.info("Alpha %s computed in %f seconds", fn_name, t2 - t1)
      if args.verbose:
        print(v[0:trade_count])
      results.append((f"#{no:03d}", int((t2 - t1) * 1000), v[trade_count - 1]))

  df = pd.DataFrame(results, columns=["no", "alphaLibTime", "alphaLibValue"])
  df.set_index("no", inplace=True)
  return df


if __name__ == "__main__":
  np.set_printoptions(precision=3, suppress=True)
  FORMAT = "%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
  logging.basicConfig(level=logging.INFO, format=FORMAT)
  args = parse_args()
  df = main(args)
  print(df.to_string())
