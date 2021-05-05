import statsmodels.api as sm
import numpy as np
from load_dataset import load_fut_data
from python.roc import history_roc, compute_confidence_brownian

alpha = 0.05
conf = compute_confidence_brownian(alpha)

# R SETUP
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri

R = robjects.r
rpy2.robjects.numpy2ri.activate()

# History ROC from bfast package, only it returns index not `time[index]`.
R('''
  suppressMessages(library("bfast"))
  history_roc.matrix <- function(X, y, level = 0.05) {
    n <- nrow(X)
    #data_rev <- data[n:1,]
    #data_rev$response <- ts(data_rev$response)
    X_rev  <- X[n:1,]
    y_rev <-  y[n:1]
    # time_rev <-  time[n:1]
    
    if (!is.ts(y)) y <- ts(y) # needed?
    y_rcus <- efp.matrix(X_rev,y_rev, type = "Rec-CUSUM")
    
    pval = sctest(y_rcus)$p.value
    y_start <- if(!is.na(pval) && pval < level) {
      length(y_rcus$process) - min(which(abs(y_rcus$process)[-1] > boundary(y_rcus)[-1])) + 1
    } else {
      1    
    }
    y_start
  }
''')

from glob import glob
print("Validating data sets in ./data.")
for fname in glob("./data/*.in"):
  print("... ", fname)
  Xt, image = load_fut_data(fname)
  X = Xt.T
  ok = True
  i = 0
  for y in image:
    i += 1
    nan_inds = np.isnan(y)
    y = y[~nan_inds]
    Xnn = X[~nan_inds]
    robjects.r.assign("y", y)
    robjects.r.assign("X", Xnn)
    py_res = history_roc(Xnn.T, y, alpha, conf)
    R_res = R("history_roc.matrix(X, y, level={})".format(alpha))[0]
    diff = py_res - int(R_res -1)
    if diff != 0:
      print("python:", py_res)
      print("R:     ", int(R_res - 1))
      print("at data set index", i)
      ok = False
  print(ok)

import futhark_data
import os
from datetime import timedelta
from timeit import default_timer as timer
print("Validating data sets in ./data/real.")
datasets = [
  # invalidate cache if you change number of chunks
  ("data/real/sahara.in", "sahara", 4),
  ("data/real/peru.in", "peru", 4),
  ("data/real/africa.in", "africa", 32),
]
cache_dir = ".cache/roc"
for (path, name, chunks) in datasets:
  print("\n== Validating {} data set ({}).".format(name, path))
  with open(path, "rb") as f:
    Xt, image = futhark_data.load(f)
  X = Xt.T
  print("image size", image.shape)
  print("regressor matrix", X.shape)
  k = X.shape[1]
  m, N = image.shape
  for i, image_chunk in enumerate(np.array_split(image, chunks)):
    print("~~ chunk {} ({}/{})".format(image_chunk.shape, i+1, chunks))
    print("Computing python results...", end="")
    if not os.path.isdir(cache_dir):
      os.makedirs(cache_dir)
    py_res_file = "{}/{}.chunk{}.npy".format(cache_dir, name, i)
    if os.path.exists(py_res_file):
      with open(py_res_file, "rb") as f:
        py_res = np.load(f)
      print("loaded from", py_res_file)
    else:
      py_res = np.zeros(image_chunk.shape[0])
      t_start = timer()
      for j, y in enumerate(image_chunk):
        nan_inds = np.isnan(y)
        ynn = y[~nan_inds]
        Xnn = X[~nan_inds]
        res = history_roc(Xnn.T, ynn, alpha, conf)
        py_res[j] = int(res)
      t_stop = timer()
      print(timedelta(seconds=t_stop-t_start))
      with open(py_res_file, "wb") as f:
        np.save(f, py_res)
    print("Computing R results...", end="")
    if not os.path.isdir(cache_dir):
      os.makedirs(cache_dir)
    R_res_file = "{}/{}.chunk{}.R.npy".format(cache_dir, name, i)
    if os.path.exists(R_res_file):
      with open(R_res_file, "rb") as f:
        R_res = np.load(f)
      print("loaded from", R_res_file)
    else:
      R_res = np.zeros(image_chunk.shape[0])
      t_start = timer()
      for j, y in enumerate(image_chunk):
        nan_inds = np.isnan(y)
        ynn = y[~nan_inds]
        Xnn = X[~nan_inds]
        if ynn.size == 0:
          R_res[j] = 1 # R is 1-indexed!
        else:
          robjects.r.assign("y", ynn)
          robjects.r.assign("X", Xnn)
          res = R("history_roc.matrix(X, y, level={})".format(alpha))[0]
          R_res[j] = int(res)
      t_stop = timer()
      print(timedelta(seconds=t_stop-t_start))
      with open(R_res_file, "wb") as f:
        np.save(f, R_res)

    R_res = R_res - 1
    check = np.all(py_res == R_res)
    print("all equal:", end="")
    if check:
      print("\033[92m PASSED \033[0m")
    else:
      print("\033[91m FAILED \033[0m")
      inds = np.where(py_res != R_res)
      print("(chunk pixel index in image, failed time series index)")
      print(inds)
      # print("First offending pixel")
      # print(image_chunk[inds[0][0]])
      print("Start of stable history indices that differ")
      print("Python", py_res[inds])
      print("R     ", R_res[inds])
