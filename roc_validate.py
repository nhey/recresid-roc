import os
from datetime import timedelta
from timeit import default_timer as timer
import numpy as np
from python.roc import history_roc, compute_confidence_brownian
from mroc_pyopencl import mroc_pyopencl

alpha = 0.05
conf = compute_confidence_brownian(alpha)

ocl = mroc_pyopencl()

def validate(name, chunks, X, image, cache_dir=".cache/roc"):
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
      for i, y in enumerate(image_chunk):
        nan_inds = np.isnan(y)
        ynn = y[~nan_inds]
        Xnn = X[~nan_inds]
        res = history_roc(Xnn.T, ynn, alpha, conf)
        py_res[i] = res
      t_stop = timer()
      print(timedelta(seconds=t_stop-t_start))
      with open(py_res_file, "wb") as f:
        np.save(f, py_res)

    print("Computing opencl results...", end="")
    t_start = timer()
    ocl_res = ocl.mhistory_roc(alpha, conf, X, image_chunk)
    t_stop = timer()
    ocl_res = ocl_res.get()
    print(timedelta(seconds=t_stop-t_start))

    check = np.all(py_res == ocl_res)
    print("all equal:", end="")
    if check:
      print("\033[92m PASSED \033[0m")
    else:
      print("\033[91m FAILED \033[0m")
      inds = np.where(py_res != ocl_res)
      print("(chunk pixel index in image, failed time series index)")
      print(inds)
      print("First offending pixel")
      print(image_chunk[inds[0][0]])
      print("Start of stable history indices that differ")
      print("Python", py_res[inds])
      print("Futhark", ocl_res[inds])

