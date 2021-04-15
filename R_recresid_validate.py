import statsmodels.api as sm
import numpy as np
from load_dataset import load_fut_data
from python.recresid import recresid

alpha = 0.05

# R SETUP
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri

R = robjects.r
rpy2.robjects.numpy2ri.activate()

R('''
  suppressMessages(library("strucchangeRcpp"))
''')

from glob import glob
print("Validating data sets in ./data.")
for fname in glob("./data/*.in"):
  print("... ", fname)
  Xt, image = load_fut_data(fname)
  X = Xt.T
  i = 0
  for y in image:
    i += 1
    nan_inds = np.isnan(y)
    y = y[~nan_inds]
    Xnn = X[~nan_inds]
    robjects.r.assign("y", y)
    robjects.r.assign("X", Xnn)
    py_res = recresid(Xnn, y)
    R_res = np.array(R("recresid(X, y)"))
    if not np.allclose(py_res, R_res):
        print("python:", py_res)
        print("R:     ", R_res)
        print("at data set index", i)
        py_save = py_res
        R_save = R_res
