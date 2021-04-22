from python.lm.lm import lm
import numpy as np

def _nonans(xs):
  return not np.any(np.isnan(xs))

def recresid(X, y, tol=None):
    n, k = X.shape
    assert(n == y.shape[0])

    if tol is None:
        tol = np.sqrt(np.finfo(np.float64).eps) / k

    y = y.reshape(n, 1)
    ret = np.zeros(n - k)

    # initialize recursion
    yh = y[:k] # k x 1
    Xh = X[:k] # k x k
    b, cov_params, rank, _, _ = lm(Xh, yh)

    X1 = np.zeros((k,k))
    X1[:rank, :rank] = cov_params # (X'X)^(-1), k x k
    bhat = np.nan_to_num(b, 0.0) # k x 1

    check = True
    for r in range(k, n):
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

            nona = _nonans(bhat) and _nonans(b)
            check = not (nona and np.allclose(b, bhat, atol=tol))
            X1 = np.zeros((k,k))
            X1[:rank, :rank] = cov_params
            bhat = np.nan_to_num(b, 0.0)

    return ret
