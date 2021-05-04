import os
from datetime import timedelta
from timeit import default_timer as timer
import numpy as np
import futhark_data
from roc_validate import validate

print("\nMAP-DISTRIBUTED PROCEDURE: mrecresid")
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
