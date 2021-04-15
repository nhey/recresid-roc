from datetime import timedelta
from timeit import default_timer as timer
import statsmodels.api as sm
import numpy as np
from load_dataset import load_fut_data
from python.recresid import recresid
from recresid_pyopencl import recresid_pyopencl

recresid_fut = recresid_pyopencl()

duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")
Y = duncan_prestige.data['income']
X = duncan_prestige.data['education']
X = sm.add_constant(X)
# Note: the copy is really important here.
# GPU results are garbage without it.
X, y = np.array(X, dtype=np.float64).copy(), np.array(Y, dtype=np.float64).copy()

py_res = recresid(X, y)
ocl_res = recresid_fut.recresid(2, X, y)

print("PROCEDURE: recresid")
print("Validating statsmodels example")
print("python:", py_res)
print("opencl:", ocl_res.get())
print("allclose:", np.allclose(py_res, ocl_res.get()))

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
    ynn = y[~nan_inds]
    Xnn = X[~nan_inds]
    py_res = recresid(Xnn, ynn)
    ocl_res = recresid_fut.recresid(1, Xnn, ynn)

    allclose = np.allclose(py_res, ocl_res.get())
    if not allclose:
      print("python", py_res)
      print("opencl", ocl_res)
      print("at pixel", i)
      py_single = py_res
      ocl_single = ocl_res.get()
      # break
    ok = ok and allclose
  print(ok)

print("\nMAP-DISTRIBUTED PROCEDURE: mrecresid")
print("Validating data sets in ./data.")
for fname in glob("./data/*.in"):
  print("... ", fname)
  Xt, image = load_fut_data(fname)
  X = Xt.T
  print("Computing python results...")
  py_res = []
  py_mask = []
  for y in image:
    nan_inds = np.isnan(y)
    py_mask.append(nan_inds)
    ynn = y[~nan_inds]
    Xnn = X[~nan_inds]
    py_res.append(recresid(Xnn, ynn))

  py_res = np.concatenate(py_res) # flat
  print("Computing opencl results...")
  ocl_resT, num_checks = recresid_fut.mrecresid(1, X, image)
  ocl_res = ocl_resT.get().T
  ocl_res = ocl_res[~np.isnan(ocl_res)] # flat
  print(np.allclose(py_res, ocl_res), num_checks)
