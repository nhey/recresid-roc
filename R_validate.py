import statsmodels.api as sm
import numpy as np
from load_dataset import load_fut_data
from python.roc import history_roc, compute_confidence_brownian, efp, sctest, history_roc_debug

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
  for y in image:
    nan_inds = np.isnan(y)
    y = y[~nan_inds]
    Xnn = X[~nan_inds]
    robjects.r.assign("y", y)
    robjects.r.assign("X", Xnn)
    py_res = history_roc_debug(Xnn.T, y, alpha, conf)
    R_res = R("history_roc.matrix(X, y, level={})".format(alpha))
    print("python:", py_res[0])
    print("R:     ", int(R_res[0] - 1))
  break

