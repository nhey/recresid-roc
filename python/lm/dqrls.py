#
#     dqrsl applies the output of dqrdc to compute coordinate
#     transformations, projections, and least squares solutions.
#     for k .le. min(n,p), let xk be the matrix
#
#            xk = (x(jpvt(1)),x(jpvt(2)), ... ,x(jpvt(k)))
#
#     formed from columns jpvt(1), ... ,jpvt(k) of the original
#     n x p matrix x that was input to dqrdc (if no pivoting was
#     done, xk consists of the first k columns of x in their
#     original order).  dqrdc produces a factored orthogonal matrix q
#     and an upper triangular matrix r such that
#
#              xk = q * (r)
#                       (0)
#
#     this information is contained in coded form in the arrays
#     x and qraux.
#
#     on entry
#
#        x      double precision(ldx,p).
#               x contains the output of dqrdc.
#
#        ldx    integer.
#               ldx is the leading dimension of the array x.
#
#        n      integer.
#               n is the number of rows of the matrix xk.  it must
#               have the same value as n in dqrdc.
#
#        k      integer.
#               k is the number of columns of the matrix xk.  k
#               must not be greater than min(n,p), where p is the
#               same as in the calling sequence to dqrdc.
#
#        qraux  double precision(p).
#               qraux contains the auxiliary output from dqrdc.
#
#        y      double precision(n)
#               y contains an n-vector that is to be manipulated
#               by dqrsl.
#
#        job    integer.
#               job specifies what is to be computed.  job has
#               the decimal expansion abcde, with the following
#               meaning.
#
#                    if a.ne.0, compute qy.
#                    if b,c,d, or e .ne. 0, compute qty.
#                    if c.ne.0, compute b.
#                    if d.ne.0, compute rsd.
#                    if e.ne.0, compute xb.
#
#               note that a request to compute b, rsd, or xb
#               automatically triggers the computation of qty, for
#               which an array must be provided in the calling
#               sequence.
#
#     on return
#
#        qy     double precision(n).
#               qy contains q*y, if its computation has been
#               requested.
#
#        qty    double precision(n).
#               qty contains trans(q)*y, if its computation has
#               been requested.  here trans(q) is the
#               transpose of the matrix q.
#
#        b      double precision(k)
#               b contains the solution of the least squares problem
#
#                    minimize norm2(y - xk*b),
#
#               if its computation has been requested.  (note that
#               if pivoting was requested in dqrdc, the j-th
#               component of b will be associated with column jpvt(j)
#               of the original matrix x that was input into dqrdc.)
#
#        rsd    double precision(n).
#               rsd contains the least squares residual y - xk*b,
#               if its computation has been requested.  rsd is
#               also the orthogonal projection of y onto the
#               orthogonal complement of the column space of xk.
#
#        xb     double precision(n).
#               xb contains the least squares approximation xk*b,
#               if its computation has been requested.  xb is also
#               the orthogonal projection of y onto the column space
#               of x.
#
#        info   integer.
#               info is zero unless the computation of b has
#               been requested and r is exactly singular.  in
#               this case, info is the index of the first zero
#               diagonal element of r and b is left unaltered.
#
#     the parameters qy, qty, b, rsd, and xb are not referenced
#     if their computation is not requested and in this case
#     can be replaced by dummy variables in the calling program.
#     to save storage, the user may in some cases use the same
#     array for different parameters in the calling sequence.  a
#     frequently occuring example is when one wishes to compute
#     any of b, rsd, or xb and does not need y or qty.  in this
#     case one may identify y, qty, and one of b, rsd, or xb, while
#     providing separate arrays for anything else that is to be
#     computed.  thus the calling sequence
#
#          call dqrsl(x,ldx,n,k,qraux,y,dum,y,b,y,dum,110,info)
#
#     will result in the computation of b and rsd, with rsd
#     overwriting y.  more generally, each item in the following
#     list contains groups of permissible identifications for
#     a single calling sequence.
#
#          1. (y,qty,b) (rsd) (xb) (qy)
#
#          2. (y,qty,rsd) (b) (xb) (qy)
#
#          3. (y,qty,xb) (b) (rsd) (qy)
#
#          4. (y,qy) (qty,b) (rsd) (xb)
#
#          5. (y,qy) (qty,rsd) (b) (xb)
#
#          6. (y,qy) (qty,xb) (b) (rsd)
#
#     in any group the value returned in the array allocated to
#     the group corresponds to the last member of the group.
#
#     linpack. this version dated 08/14/78 .
#     g.w. stewart, university of maryland, argonne national lab.
#
#     dqrsl uses the following functions and subprograms.
#
#     BLAS      daxpy,dcopy,ddot
#     Fortran   dabs,min0,mod
#
import numpy as np
from .dqrsl import dqrsl
from .dqrdc2 import dqrdc2

def dqrls(x, n, p, y, tol=1e-7):
#
#     reduce x.
#
  x, k, qraux, jpvt = dqrdc2(x, n, n, p, tol)
#
#     solve the truncated least squares problem for rhs.
#
  if k > 0:
    qy, qty, b, rsd, xb, info = dqrsl(x, n, n, k, qraux, y, 1110)
  else:
    raise Exception("Rank is zero. No parameters fitted.")
#
#     set the unused components of b to zero.
#
  beta = np.empty(p)
  beta.fill(np.nan)
  beta[:k] = b[:k]
  return x, beta, rsd, qty, k, jpvt, qraux
