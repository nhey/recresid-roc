import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from .dqrdc2 import dqrdc2
from .dqrqty import dqrqty

def lm(X, y):
  n,p  = X.shape
  y = y.reshape(n)
  A, rank, qraux, jpvt = dqrdc2(X.copy(), n, n, p)
  r = np.triu(A)[:p, :p]
  # Inverting r with cholesky gives (X.T X)^{-1}
  cov_params = cho_solve((r[:rank, :rank], False), np.identity(rank))
  # compute parameters
  qty = dqrqty(A, qraux, rank, y.copy())
  beta = solve_triangular(r[:rank, :rank], qty[:rank])
  # Pivot fitted parameters to match original order of Xs columns
  b = np.zeros(p)
  b[:rank] = beta
  b[jpvt] = b[range(p)]
  scratch = np.zeros((p,p))
  scratch[:rank, :rank] = cov_params
  # swap rows
  scratch[jpvt, :] = scratch[range(p), :]
  # swap columns
  scratch[:, jpvt] = scratch[:, range(p)]
  cov_params = scratch
  return b, cov_params, rank, r, qraux
