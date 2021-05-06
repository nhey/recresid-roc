import os
from datetime import timedelta
from timeit import default_timer as timer
import numpy as np
from glob import glob
import futhark_data
from load_dataset import load_fut_data
from roc_validate import validate

print("\nMAP-DISTRIBUTED PROCEDURE: mroc")

print("Validating random data sets in ./data.")
for path in glob("./data/*.in"):
  print("... ", path)
  Xt, image = load_fut_data(path)
  name = os.path.basename(path).replace(".in", "")
  validate(name, 1, Xt.T, image)

print("Validating real world data sets.")
datasets = [
  ("data/real/sahara.in", "sahara", 4),
  ("data/real/peru.in", "peru", 4),
  ("data/real/africa.in", "africa", 32),
]
for (path, name, chunks) in datasets:
  print("\n== Validating {} data set ({}).".format(name, path))
  with open(path, "rb") as f:
    Xt, image = futhark_data.load(f)
  validate(name, chunks, Xt.T, image)
