import os
from urllib.request import urlretrieve
from datetime import timedelta
from timeit import default_timer as timer
import numpy as np
from python.recresid import recresid
from recresid_pyopencl import recresid_pyopencl

# Peru image from diku-dk/bfast.
peru_dir = "data/peru_small"
image_path = peru_dir + "/image.npy"

if not os.path.isdir(peru_dir):
  os.makedirs(peru_dir)

if not os.path.exists(image_path):
  print("Downloading data...")
  url = 'https://sid.erda.dk/share_redirect/fcwjD77gUY/data.npy'
  urlretrieve(url, image_path)

recresid_fut = recresid_pyopencl()

print("\nMAP-DISTRIBUTED PROCEDURE: mrecresid")
print("Validating peru small")
Xt = np.load(peru_dir + "/X.npy")
X = Xt.T
N, k = X.shape
image = np.load(image_path)
image = np.transpose(image, (1,2,0)).reshape(-1, image.shape[0])
image = np.array(image, dtype=np.float64)
image[image == -32768] = np.nan
# subset image time series
image = image[:, :N]
# filter pixels with less than k valid values
nans = np.isnan(image)
enough_data = N - np.sum(nans, axis=1) > k
image = image[enough_data]
print("image size", image.shape)
print("regressor matrix", X.shape)
# chunk image to fit on my GPU (about 4GiB of mem)
chunks = 4
print("num chunks", chunks)
for i, image_chunk in enumerate(np.array_split(image, chunks)): 
  print("Computing python results...", end="")
  py_save = "peru_chunk{}.npy".format(i)
  if os.path.exists(py_save):
    with open(py_save, "rb") as f:
      py_res = np.load(f)
    print("loaded from", py_save)
  else:
    t_start = timer()
    py_res = []
    py_mask = []
    for y in image_chunk:
      nan_inds = np.isnan(y)
      py_mask.append(nan_inds)
      ynn = y[~nan_inds]
      Xnn = X[~nan_inds]
      py_res.append(recresid(Xnn, ynn))
    print(timedelta(seconds=timer()-t_start))
    py_res = np.concatenate(py_res) # flat
    with open(py_save, "wb") as f:
      np.save(f, py_res)

  print("Computing opencl results...", end="")
  t_start = timer()
  ocl_resT, num_checks = recresid_fut.mrecresid(X, image_chunk)
  t_stop = timer()
  ocl_res = ocl_resT.get().T
  ocl_res = ocl_res[~np.isnan(ocl_res)] # flat
  print(timedelta(seconds=t_stop-t_start))
  print("Number of stability checks:", num_checks)

  print("Relative absolute error")
  rel_err = np.abs((py_res - ocl_res)/py_res)
  per_err = rel_err * 100
  np.set_printoptions(formatter={"float": "{0:0.1e}".format})
  print("Max error  {:10.5e} ({:.4f}%)".format(np.max(rel_err),
                                               np.max(per_err)))
  print("Min error  {:10.5e} ({:.4f}%)".format(np.min(rel_err),
                                               np.min(per_err)))
  print("Mean error {:10.5e} ({:.4f}%)".format(np.mean(rel_err),
                                               np.mean(per_err)))

