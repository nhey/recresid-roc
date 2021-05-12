import numpy as np
from scipy import stats, optimize

from .recresid import recresid

# From Brown, Durbin, Evans (1975).
def _pval_brownian_motion_max(x):
    Q = lambda x: 1 - stats.norm.cdf(x, loc=0, scale=1)
    p = 2 * (Q(3*x) + np.exp(-4*x**2) - np.exp(-4*x**2)*Q(x))
    return p

def compute_confidence_brownian(alpha):
  return optimize.brentq(lambda x: _pval_brownian_motion_max(x) - alpha, 0, 20)

def efp(X, y):
  # Recursive CUSUM process
  k, n = X.shape
  w  = recresid(X.T, y)
  sigma = np.std(w, ddof=1) # division by `N-1` to match R's `sd` function.
  process = np.cumsum(np.append([0],w))/(sigma*np.sqrt(n-k))
  return process

# Linear boundary for Brownian motion (limiting process of rec.resid. CUSUM).
def boundary(process, confidence):
    n = process.shape[0]
    t = np.linspace(0, 1, num=n)
    bounds = confidence + (2*confidence*t) # from Zeileis' strucchange description.
    return bounds

# Structural change test for Brownian motion.
def sctest(process):
    x = process[1:]
    n = x.shape[0]
    j = np.linspace(1/n, 1, num=n)
    x = x * 1/(1 + 2*j)
    stat = np.max(np.abs(x))
    return _pval_brownian_motion_max(stat)

def history_roc(X, y, alpha, confidence):
  if y.shape[0] == 0: return 0
  X_rev = np.flip(X, axis=1)
  y_rev = y[::-1]
  rcus = efp(X_rev, y_rev)

  pval = sctest(rcus)
  y_start = 0
  if not np.isnan(pval) and pval < alpha:
      bounds = boundary(rcus, confidence)
      inds = (np.abs(rcus[1:]) > bounds[1:]).nonzero()[0]
      y_start = rcus.size - np.min(inds) - 1 if inds.size > 0 else 0
  return y_start

def history_roc_debug(X, y, alpha, confidence):
  X_rev = np.flip(X, axis=1)
  y_rev = y[::-1]
  rcus = efp(X_rev, y_rev)

  pval = sctest(rcus)
  y_start = 0
  pval_pass = False
  if not np.isnan(pval) and pval < alpha:
      bounds = boundary(rcus, confidence)
      inds = (np.abs(rcus[1:]) > bounds[1:]).nonzero()[0]
      y_start = rcus.size - np.min(inds) - 1 if inds.size > 0 else 0
      pval_pass = True
  return y_start, pval_pass, rcus
