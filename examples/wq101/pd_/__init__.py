# !/usr/bin/env python3
# DolphinDB Inc.
# @Author: DolphinDB
# @Last modification time: 2022.09.01
# @FileName: wq101alphaPyTime.py

# This script is to test the performance of the implementation of WorldQuant 101 alpha in python.
# You will need to put alpha101_adjusted.py and this wq101alphaPyTime.py in the same folder.
# You will need to use dataPerformance.csv. Don't forget to change your directory.
# The overall time cost is about 42 minutes.

import time
import pandas as pd
from .alpha101_adjusted import Alphas
import argparse

import logging

logger = logging.getLogger("pandas_backend")


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


def setup_context(data_path: str) -> (Alphas, int):
  logger.info("Loading data")
  t1 = time.time()
  data = pd.read_csv(data_path)
  df = data.pivot(index="tradetime", columns="securityid")
  ctx = Alphas(df)
  t2 = time.time()
  logger.info("Data loaded in %f seconds", t2 - t1)
  return ctx, int((t2 - t1) * 1000)


nofunc = set(
  [48, 56, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100]
)


def main(args):
  ctx, load_time = setup_context(args.data)

  if len(args.no) == 0:
    start = args.start or 1
    end = args.end or 102
    args.no = [i for i in filter(lambda x: x not in nofunc, range(start, end))]

  results = [("data", load_time)]
  for no in args.no:
    t1 = time.time()
    fn_name = f"alpha{no:03d}"
    logger.info("Computing alpha %s", fn_name)
    fn = getattr(Alphas, fn_name)
    res = fn(ctx)
    t2 = time.time()
    logger.info("Alpha %s computed in %f seconds", fn_name, t2 - t1)
    if args.verbose:
      print(res["sz000001"])
    results.append((f"#{no:03d}", int((t2 - t1) * 1000), res["sz000001"].iloc[-1]))

  df = pd.DataFrame(results, columns=["no", "pandasTime", "pandasValue"])
  df.set_index("no", inplace=True)
  return df


if __name__ == "__main__":
  FORMAT = "%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
  logging.basicConfig(level=logging.INFO, format=FORMAT)
  args = parse_args()
  df = main(args)
  print(df.to_string())
