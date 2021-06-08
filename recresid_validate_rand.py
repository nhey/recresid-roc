import os
from datetime import timedelta
from timeit import default_timer as timer
from glob import glob
import numpy as np
from load_dataset import load_fut_data
from python.recresid import recresid
from recresid_pyopencl import recresid_pyopencl

recresid_fut = recresid_pyopencl()

print("\nSINGLE PIXEL PROCEDURE: recresid")
print("Validating data sets in ./data.")
for fname in glob("./data/*.in"):
  print("... ", fname)
  Xt, image = load_fut_data(fname)
  X = Xt.T
  if image.shape[0] > 500:
    print("Large image. Skipping because it would be very slow.")
    continue
  ok = True
  max_num_checks = 0
  total = len(image)
  for i, y in enumerate(image):
    if (i+1) % 100 == 0:
      print("{}%, max num stability checks: {}".format((i+1)/total*100,
                                                       max_num_checks))
    nan_inds = np.isnan(y)
    ynn = y[~nan_inds]
    Xnn = X[~nan_inds]
    py_res = recresid(Xnn, ynn)
    ocl_res, num_checks = recresid_fut.recresid(Xnn, ynn)

    max_num_checks = max(max_num_checks, num_checks)

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

from recresid_validate import validate
print("\nMAP-DISTRIBUTED PROCEDURE: mrecresid")
print("Validating data sets in ./data.")
for path in glob("./data/*.in"):
  print("... ", path)
  Xt, image = load_fut_data(path)
  name = os.path.basename(path).replace(".in", "")
  validate(name, 1, Xt.T, image)
