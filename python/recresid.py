from python.lm.lm import lm
import numpy as np

def _nonans(xs):
  return not np.any(np.isnan(xs))

# Mimics the behaviour of R's `all.equal`.
def all_equal(target, current, tol):
  xy = np.mean(np.abs(target - current))
  xn = np.mean(np.abs(target))
  if xn > tol:
    xy = xy/xn
  return xy <= tol

# NOTE
# BFAST R-package makes extensive use of the
# strucchange R-package. However, if armadillo
# bindings is available it will use their own
# "strucchangeRcpp" version. In this, recursive
# residuals differ significantly in that
# only an absolute check is done on the equality
# of parameters (R mostly does relative check)
# AND it will use a simple solver rather than QR
# whenever the condition number is less than some
# tolerance.
#
# This is the armadillo version, which is simply
# an absolute difference |x - y| <= tol.
def approx_equal(x, y, tol):
  return np.mean(np.abs(x - y)) <= tol

def recresid(X, y, tol=None):
    n, k = X.shape
    assert(n == y.shape[0])
    if n == 0:
      return np.array([])

    if tol is None:
        tol = np.sqrt(np.finfo(np.float64).eps) / k

    y = y.reshape(n, 1)
    ret = np.zeros(n - k)

    # initialize recursion
    yh = y[:k] # k x 1
    Xh = X[:k] # k x k
    b, cov_params, rank, _, _ = lm(Xh, yh)

    X1 = cov_params # (X'X)^(-1), k x k
    inds = np.isnan(X1)
    X1[inds] = 0.0
    bhat = np.nan_to_num(b, 0.0) # k x 1

    check = True
    for r in range(k, n):
        prev_rank = rank
        # Compute recursive residual
        x = X[r]
        d = X1 @ x
        fr = 1 + (x @ d)
        resid = y[r] - np.nansum(x * bhat) # dotprod ignoring nans
        ret[r-k] = resid / np.sqrt(fr)

        # Update formulas
        X1 = X1 - (np.outer(d, d))/fr
        bhat += X1 @ x * resid

        # Check numerical stability (rectify if unstable).
        if check:
            # We check update formula value against full OLS fit
            b, cov_params, rank, _, _ = lm(X[:r+1], y[:r+1])
            # R checks nans in fitted parameters; same as rank.
            # Also check on latest recresidual, because fr may
            # be nan.
            nona = (rank == k and prev_rank == k
                              and not np.isnan(ret[r-k]))
            check = not (nona and approx_equal(b, bhat, tol))
            X1 = cov_params
            bhat = np.nan_to_num(b, 0.0)

    return ret
