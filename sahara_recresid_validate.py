import os
from datetime import timedelta
from timeit import default_timer as timer
import numpy as np
import futhark_data
from python.recresid import recresid
from recresid_pyopencl import recresid_pyopencl

recresid_fut = recresid_pyopencl()

def validate(name, X, image):
  chunks = 4
  print("image size", image.shape)
  print("regressor matrix", X.shape)
  k = X.shape[1]
  m, N = image.shape
  for i, image_chunk in enumerate(np.array_split(image, chunks)):
    print("== chunk", image_chunk.shape)
    print("Computing python results...", end="")
    py_res_file = "{}.chunk{}.npy".format(name, i)
    if os.path.exists(py_res_file):
      with open(py_res_file, "rb") as f:
        py_res = np.load(f)
      print("loaded from", py_res_file)
    else:
      num_recresids_padded = N-k
      py_res = np.empty((image_chunk.shape[0],num_recresids_padded))
      py_res.fill(np.nan)
      py_len = []
      t_start = timer()
      for i, y in enumerate(image_chunk):
        nan_inds = np.isnan(y)
        ynn = y[~nan_inds]
        Xnn = X[~nan_inds]
        py_len.append(len(ynn))
        # with np.errstate(invalid='raise'):
        #   try:
        #     res = recresid(Xnn, ynn)
        #   except:
        #     print("at chunk pixel", i)
        res = recresid(Xnn, ynn)
        if res.size > 0:
          py_res[i,:res.size] = res
      t_stop = timer()
      print(timedelta(seconds=t_stop-t_start))
      with open(py_res_file, "wb") as f:
        np.save(f, py_res)

    print("Computing opencl results...", end="")
    t_start = timer()
    ocl_resT, num_checks = recresid_fut.mrecresid(X, image_chunk)
    t_stop = timer()
    ocl_res = ocl_resT.get().T
    print(timedelta(seconds=t_stop-t_start))
    check = np.allclose(py_res, ocl_res, equal_nan=True)
    print("All close?", check)
    if not check:
      inds = np.where(~np.isclose(py_res, ocl_res, equal_nan=True))
      print(inds)
      print("First offending pixel")
      print(image_chunk[inds[0][0]])
      print("Python recresid")
      print(py_res[inds])
      print("Futhark recresid")
      print(ocl_res[inds])
      # np.set_printoptions(formatter={"float": "{0:0.15e}".format}, threshold=np.inf)
      # print(X)
      # print(image_chunk[inds[0][0]])
    print("Number of stability checks:", num_checks)

    print("Relative absolute error")
    rel_err = np.abs((py_res - ocl_res)/py_res)
    rel_err = rel_err[~np.isnan(rel_err)]
    per_err = rel_err * 100
    print("Max error  {:10.5e} ({:.4f}%)".format(np.max(rel_err),
                                                 np.max(per_err)))
    print("Min error  {:10.5e} ({:.4f}%)".format(np.min(rel_err),
                                                 np.min(per_err)))
    print("Mean error {:10.5e} ({:.4f}%)".format(np.mean(rel_err),
                                                 np.mean(per_err)))

print("\nMAP-DISTRIBUTED PROCEDURE: mrecresid")
for (path, name) in [("data/sahara.in", "sahara")]:
  print("Validating {} data set ({}).".format(name, path))
  with open(path, "rb") as f:
    Xt, image = futhark_data.load(f)
  validate(name, Xt.T, image)
