import statsmodels.api as sm
import numpy as np
from python.roc import history_roc, compute_confidence_brownian, efp, sctest, history_roc_debug
from roc_pyopencl import roc_pyopencl

roc_fut = roc_pyopencl()
alpha = 0.05
conf = compute_confidence_brownian(alpha)

print("Validating statsmodels example")
duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")
Y = duncan_prestige.data['income']
X = duncan_prestige.data['education']
X = sm.add_constant(X)
# Note: the copy is really important here.
# GPU results are garbage without it.
X, y = np.array(X, dtype=np.float64).copy(), np.array(Y, dtype=np.float64).copy()

set_nan_inds = [2,5,10,13,15,20,22,25,27,30,33,37,39]
y[set_nan_inds] = np.nan
nan_inds = np.isnan(y)

y = y[~nan_inds]
X = X[~nan_inds]
py_res = history_roc(X.T, y, alpha, conf)
ocl_res = roc_fut.history_roc(1, alpha, conf, X.T, y)

print("python:", py_res)
print("opencl:", ocl_res)


def load_fut_data(filename, np_dtype=np.float64, f_dtype="f64"):
  parse = lambda line: line.replace(f_dtype,"").replace(".nan", "float('nan')")
  with open(filename, "r") as f:
    Xt = np.array(eval(parse(f.readline())), dtype=np_dtype)
    image = np.array(eval(parse(f.readline())), dtype=np_dtype)
  return (Xt, image)

from glob import glob
print("Validating data sets in ./data.")
for fname in glob("./data/*.in"):
  print("... ", fname)
  Xt, image = load_fut_data(fname)
  X = Xt.T
  for y in image:
    nan_inds = np.isnan(y)
    y = y[~nan_inds]
    Xtnn = (X[~nan_inds]).T
    py_res = history_roc_debug(Xtnn, y, alpha, conf)
    ocl_res = roc_fut.history_roc(1, alpha, conf, Xtnn, y)
    print("python:", py_res)
    print("opencl:", ocl_res)

# def validate(num_tests, bsz):
#   ok = True
#   num_significant = num_tests
#   for i in range(num_tests):
#     X = np.random.rand(100,6).astype(np.float64)
#     y = np.random.rand(100,1).astype(np.float64)
#     py_res, pval_pass, _ = history_roc_debug(X.T, y, alpha, conf)
#     ocl_res, ocl_pass = roc_fut.history_roc(1, alpha, conf, X.T, y)
#     if not pval_pass:
#       num_significant -= 1

#     passed = py_res == ocl_res
#     ok = ok and passed
#     if not passed:
#       print("python", py_res)
#       print("opencl", ocl_res)
#   return ok, num_significant

# runs = 10
# print("\nValidating {} random runs block size 1...".format(runs))
# print(validate(runs, 1))
# print("Validating {} random runs block size 2...".format(runs))
# print(validate(runs, 2))
# print("Validating {} random runs block size 3...".format(runs))
# print(validate(runs, 3))
