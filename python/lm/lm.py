import numpy as np
from scipy.linalg import cho_solve
from .dqrls import dqrls

def lm(X, y):
  n,p  = X.shape
  # Note dqrls overwrites its inputs
  qr, b, rsd, qty, rank, jpvt, qraux = dqrls(X.copy(), n, p, y.copy())
  r = np.triu(qr)[:p, :p]
  # Inverting r with cholesky gives (X.T X)^{-1}
  cov_params = cho_solve((r[:rank, :rank], False), np.identity(rank))
  # Pivot fitted parameters to match original order of Xs columns
  b[jpvt] = b[range(p)]
  scratch = np.zeros((p,p))
  scratch[:rank, :rank] = cov_params
  # swap rows
  scratch[jpvt, :] = scratch[range(p), :]
  # swap columns
  scratch[:, jpvt] = scratch[:, range(p)]
  cov_params = scratch
  return b, cov_params, rank, r, qraux
