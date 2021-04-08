import statsmodels.api as sm
import numpy as np
from python.roc import history_roc, compute_confidence_brownian, efp, sctest
from roc_pyopencl import roc_pyopencl

roc_fut = roc_pyopencl()

duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")
Y = duncan_prestige.data['income']
X = duncan_prestige.data['education']
X = sm.add_constant(X)
# Note: the copy is really important here.
# GPU results are garbage without it.
X, y = np.array(X, dtype=np.float64).copy(), np.array(Y, dtype=np.float64).copy()

alpha = 0.05
conf = compute_confidence_brownian(alpha)

py_res = history_roc(X.T, y, alpha, conf)
ocl_res = roc_fut.history_roc(1, X.T, y, alpha, conf)

print("Validating statsmodels example")
print("python:", py_res)
print("opencl:", ocl_res)

def validate(num_tests, bsz):
  ok = True
  num_significant = num_tests
  for i in range(num_tests):
    X = np.random.rand(100,6).astype(np.float64)
    y = np.random.rand(100,1).astype(np.float64)
    py_res, pval_pass = history_roc(X.T, y, alpha, conf)
    ocl_res = roc_fut.history_roc(1, X.T, y, alpha, conf)
    if not pval_pass:
      num_significant -= 1

    passed = py_res == ocl_res
    ok = ok and passed
    if not passed:
      print("python", py_res)
      print("opencl", ocl_res)
  return ok, num_significant

runs = 50
print("\nValidating {} random runs block size 1...".format(runs))
print(validate(runs, 1))
print("Validating {} random runs block size 2...".format(runs))
print(validate(runs, 2))
print("Validating {} random runs block size 3...".format(runs))
print(validate(runs, 3))
