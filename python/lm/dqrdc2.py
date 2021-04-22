# All comments (besides this) are from the original dqrdc2 file.

#     Dqrdc2 is a *modification* of Linpack's dqrdc ('DQRDC') for R
#
#     dqrdc2 uses householder transformations to compute the qr
#     factorization of an n by p matrix x.  a limited column
#     pivoting strategy based on the 2-norms of the reduced columns
#     moves columns with near-zero norm to the right-hand edge of
#     the x matrix.  this strategy means that sequential one
#     degree-of-freedom effects can be computed in a natural way.
#
#     i am very nervous about modifying linpack code in this way.
#     if you are a computational linear algebra guru and you really
#     understand how to solve this problem please feel free to
#     suggest improvements to this code.
#
#     Another change was to compute the rank.
#
#     on entry
#
#        x       double precision(ldx,p), where ldx .ge. n.
#                x contains the matrix whose decomposition is to be
#                computed.
#
#        ldx     integer.
#                ldx is the leading dimension of the array x.
#
#        n       integer.
#                n is the number of rows of the matrix x.
#
#        p       integer.
#                p is the number of columns of the matrix x.
#
#        tol     double precision
#                tol is the nonnegative tolerance used to
#                determine the subset of the columns of x
#                included in the solution.
#
#        jpvt    integer(p).
#                integers which are swapped in the same way as the
#                the columns of x during pivoting.  on entry these
#                should be set equal to the column indices of the
#                columns of the x matrix (typically 1 to p).
#
#        work    double precision(p,2).
#                work is a work array.
#
#     on return
#
#        x       x contains in its upper triangle the upper
#                triangular matrix r of the qr factorization.
#                below its diagonal x contains information from
#                which the orthogonal part of the decomposition
#                can be recovered.  note that if pivoting has
#                been requested, the decomposition is not that
#                of the original matrix x but that of x
#                with its columns permuted as described by jpvt.
#
#        k       integer.
#                k contains the number of columns of x judged
#                to be linearly independent, i.e., "the rank"
#
#        qraux   double precision(p).
#                qraux contains further information required to recover
#                the orthogonal part of the decomposition.
#
#        jpvt    jpvt(j) contains the index of the column of the
#                original matrix that has been interchanged into
#                the j-th column.  Consequently, jpvt[] codes a
#		 permutation of 1:p; it is called 'pivot' in R
#

#
#     original (dqrdc.f) linpack version dated 08/14/78 .
#     g.w. stewart, university of maryland, argonne national lab.
#
#     This version dated 22 August 1995
#     Ross Ihaka
#
#     bug fixes 29 September 1999 BDR (p > n case, inaccurate ranks)
#
#
#     dqrdc2 uses the following functions and subprograms.
#
#     blas daxpy,ddot,dscal,dnrm2
#     fortran dabs,dmax1,min0,dsqrt
#
import numpy as np

def dqrdc2(x, ldx, n, p, tol=1e-7):
  qraux = np.zeros(p)
  work = np.zeros((p,2))
  jpvt = np.arange(p)
#
#
# compute the norms of the columns of x.
#
  if n > 0:
    for j in range(0,p):
      qraux[j] = np.linalg.norm(x[:,j], 2)
      work[j,0] = qraux[j]
      work[j,1] = qraux[j]
      if work[j,1] == 0.0:
        work[j,1] = 1.0
#
# perform the householder reduction of x.
#
  lup = min(n,p)
  k = p + 1
#
#   previous version only cycled l to lup
#
#   cycle the columns from l to p left-to-right until one
#   with non-negligible norm is located.  a column is considered
#   to have become negligible if its norm has fallen below
#   tol times its original norm.  the check for l .le. k
#   avoids infinite cycling.
#
  for l in range(0, lup):
    while l+1 < k and qraux[l] < work[l,1]*tol:
      lp1 = l+1
      for i in range(n):
        t = x[i,l]
        for j in range(lp1, p):
          x[i, j-1] = x[i,j]
        x[i,p-1] = t
      i = jpvt[l]
      t = qraux[l]
      tt = work[l,0]
      ttt = work[l,1]
      for j in range(lp1, p):
        jpvt[j-1] = jpvt[j]
        qraux[j-1] = qraux[j]
        work[j-1,:] = work[j,:]
      jpvt[p-1] = i
      qraux[p-1] = t
      work[p-1,0] = tt
      work[p-1,1] = ttt
      k = k -1
#
#     compute the householder transformation for column l.
#
    if l+1 != n:
      nrmxl = np.linalg.norm(x[l:, l], 2)
      if nrmxl != 0.0:
        if x[l,l] != 0.0: nrmxl = np.sign(x[l,l]) * abs(nrmxl)
        x[l:, l] = x[l:, l] / nrmxl
        x[l,l] = 1.0 + x[l,l]
        #
        #       apply the transformation to the remaining columns,
        #       updating the norms.
        #
        for j in range(l+1, p):
          t = -1*np.dot(x[l:,l], x[l:,j])/x[l,l]
          x[l:,j] = x[l:,j] + t*x[l:,l]
          if qraux[j] != 0.0:
            tt = 1.0 - (abs(x[l,j])/qraux[j])**2
            tt = max(tt, 0.0)
            t = tt
        #
        # modified 9/99 by BDR. Re-compute norms if there is large reduction
        # The tolerance here is on the squared norm
        # In this version we need accurate norms, so re-compute often.
        #  work(j,1) is only updated in one case: looks like a bug -- no longer used
        #
        #           tt = 1.0 + 0.05*tt*(qraux[j]/work[j,0])**2
        #           if tt != 1.0:
            if abs(t) >= 1e-6:
              qraux[j] = qraux[j]*np.sqrt(t)
            else:
              qraux[j] = np.linalg.norm(x[l+1:,j], 2)
              work[j,0] = qraux[j]
        #
        #     save the transformation.
        #
        qraux[l] = x[l,l]
        x[l,l] = -1*nrmxl

  k = min(k-1, n)
  return x, k, qraux, jpvt
