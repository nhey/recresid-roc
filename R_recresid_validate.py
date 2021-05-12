import statsmodels.api as sm
import numpy as np
from load_dataset import load_fut_data
from python.recresid import recresid

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
  ok = True
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
      print("python[:4], python[-1]:", py_res[:4], py_res[-1])
      print("R[:4],           R[-1]:", R_res[:4], R_res[-1])
      print("at data set index", i)
      print("Relative absolute error")
      rel_err = np.abs((R_res - py_res)/R_res)
      per_err = rel_err * 100
      np.set_printoptions(formatter={"float": "{0:0.4e}".format})
      print("Max error  {:10.5e} ({:.4f}%)".format(np.max(rel_err),
                                                   np.max(per_err)))
      print("Min error  {:10.5e} ({:.4f}%)".format(np.min(rel_err),
                                                   np.min(per_err)))
      print("Mean error {:10.5e} ({:.4f}%)".format(np.mean(rel_err),
                                                   np.mean(per_err)))
      py_save = py_res
      R_save = R_res
      ok = False
  print(ok)
