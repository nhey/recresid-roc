import numpy as np
from scipy import linalg

def dqrqty(a, qraux, k, y):
  n, _ = a.shape
  ju = min(k, n-1)

  qty = np.zeros(n)
  qty[0:n] = y[0:n]
  for j in range ( 0, ju ):
    if ( qraux[j] != 0.0 ):
      # put qraux on diagonal
      ajj = a[j,j]
      a[j,j] = qraux[j]
      t = - (np.sum(a[j:n,j] * qty[j:n]))/a[j,j]
      qty[j:n] = qty[j:n] + t*a[j:n,j]
      # revert back to original diagonal
      a[j,j] = ajj
  return qty

# TODO --- implement "qty" in python;
#          check against dqrsl example/this loop code
#      --- implement "qty" in futhark; check against python qty
#      --- still remember the difference in qraux last val;
#          will be interesting if this is not visible in qty
#      --- at this point ols can be implemented in futhark by
#          just replacing qr and qty in existing module

if __name__ == "__main__":
  from dqrdc2 import dqrdc2
  # R SETUP
  from rpy2.robjects.packages import importr
  import rpy2.robjects as robjects
  import rpy2.robjects.numpy2ri

  R = robjects.r
  rpy2.robjects.numpy2ri.activate()

  A = [
    np.array([[ 1.0, 1.0, 0.0 ],
              [ 1.0, 0.0, 1.0 ],
              [ 0.0, 1.0, 1.0 ]]),
    np.array([[-1.88817327e-05, 4.45881295e-05],
              [2.95671876e-04, -6.98212188e-04]]),
    np.array(
      [[6.5246915563830465e-01, 2.6898186620336068e-01, 3.9603252438218946e-02,
        4.4373528252401873e-03, 8.2447286588856050e-01, 8.9140731054505462e-01],
       [5.1313213654246059e-01, 7.3434945988349465e-01, 5.5188716354720868e-01,
        1.0079718428879749e+00, 7.4830633475416231e-01, 1.3145974530565441e-01],
       [7.2547014534819743e-01, 4.0433041410592792e-02, 7.4131341509083071e-01,
        2.7851708788433721e-01, 7.6786943729924229e-01, 7.2308758035458709e-01],
       [3.5624070954474513e-01, 4.1975333920306263e-01, 2.3324669516740348e-02,
        8.0211001856291528e-02, 7.1146094864811960e-01, 5.6936943923268668e-01],
       [7.7833436273445999e-01, 3.7522038351818859e-01, 9.8281140792853039e-01,
        8.5379316311691156e-01, 8.0325685538209579e-01, 4.2613613024778435e-01],
       [6.7698669031975445e-01, 3.8951620742479959e-01, 1.3610913752960105e-01,
        7.3042307561297437e-01, 5.1707556606341842e-01, 3.2601827381111798e-01],
       [4.7395074251717978e-01, 8.9010465356494051e-01, 9.3509366486950529e-01,
        3.6706094970806979e-01, 5.6294435970471779e-01, 3.3823027951470763e-01]]
    ),
    np.array(
      [[1.0, 1.0,   0.5,       0.866025,  0.866025,  0.5,
        1.0,  0.0],
       [1.0, 2.0,   0.866025,  0.5,       0.866025, -0.5,
        0.0, -1.0],
       [1.0, 4.0,   0.866025, -0.5,      -0.866025, -0.5,
        -0.0,  1.0],
       [1.0, 5.0,   0.5,      -0.866025, -0.866025,  0.5,
        1.0,  0.0],
       [1.0, 7.0,  -0.5,      -0.866025,  0.866025,  0.5,
        -1.0, -0.0],
       [1.0, 9.0,  -1.0,      -0.0,       0.0,      -1.0,
        1.0,  0.0],
       [1.0, 14.0,  0.866025,  0.5,       0.866025, -0.5,
        0.0, -1.0],
       [1.0, 17.0,  0.5,      -0.866025, -0.866025,  0.5,
        1.0,  0.0]]
    )
  ]

  for a in A:
    robjects.r.assign("a", a)
    print("===")
    n, p = a.shape
    tol = 1e-7
    np_q, np_r = np.linalg.qr(a)
    x, rank, qraux, jpvt = dqrdc2(a.copy(), n, n, p, tol)
    print(x)
    for i in range(n):
      y = np.zeros(n)
      y[i] = 1.0
      qty = dqrqty(x, qraux, rank, y)
      print(qty[:p])
    print(rank)
    print(R("qr.Q(qr(a))"))
    print(R("qr(a)$rank"))
