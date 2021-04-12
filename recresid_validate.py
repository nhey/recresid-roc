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

print("Validating statsmodels example")
print("python:", py_res)
print("opencl:", ocl_res.get())
print("allclose:", np.allclose(py_res, ocl_res.get()))

def validate(num_tests, bsz):
  ok = True
  for i in range(num_tests):
    X = np.random.rand(100,8).astype(np.float64)
    y = np.random.rand(100,1).astype(np.float64)
    py_res = recresid(X, y)
    ocl_res = recresid_fut.recresid(2, X, y)

    ok = ok and np.allclose(py_res, ocl_res.get())
    if not ok:
      print("python", py_res)
      print("opencl", ocl_res)
      break
  return ok

# runs = 100
# print("\nValidating {} random runs block size 1...".format(runs))
# print(validate(runs, 1))
# print("Validating {} random runs block size 2...".format(runs))
# print(validate(runs, 2))
# print("Validating {} random runs block size 4...".format(runs))
# print(validate(runs, 4))

from glob import glob
print("Validating data sets in ./data.")
for fname in glob("./data/*.in"):
  print("... ", fname)
  Xt, image = load_fut_data(fname)
  X = Xt.T
  ok = True
  for y in image:
    nan_inds = np.isnan(y)
    ynn = y[~nan_inds]
    Xnn = X[~nan_inds]
    py_res = recresid(Xnn, ynn)
    ocl_res = recresid_fut.recresid(1, Xnn, ynn)

    ok = ok and np.allclose(py_res, ocl_res.get())
    if not ok:
      print("python", py_res)
      print("opencl", ocl_res)
      break
  print(ok)
