#! /usr/bin/env python
#
def dqrsl ( a, lda, n, k, qraux, y, job ):

#*****************************************************************************80
#
## DQRSL computes transformations, projections, and least squares solutions.
#
#  Discussion:
#
#    DQRSL requires the output of DQRDC.
#
#    For K <= min(N,P), let AK be the matrix
#
#      AK = ( A(JPVT(1)), A(JPVT(2)), ..., A(JPVT(K)) )
#
#    formed from columns JPVT(1), ..., JPVT(K) of the original
#    N by P matrix A that was input to DQRDC.  If no pivoting was
#    done, AK consists of the first K columns of A in their
#    original order.  DQRDC produces a factored orthogonal matrix Q
#    and an upper triangular matrix R such that
#
#      AK = Q * (R)
#               (0)
#
#    This information is contained in coded form in the arrays
#    A and QRAUX.
#
#    The parameters QY, QTY, B, RSD, and AB are not referenced
#    if their computation is not requested and in this case
#    can be replaced by dummy variables in the calling program.
#    To save storage, the user may in some cases use the same
#    array for different parameters in the calling sequence.  A
#    frequently occuring example is when one wishes to compute
#    any of B, RSD, or AB and does not need Y or QTY.  In this
#    case one may identify Y, QTY, and one of B, RSD, or AB, while
#    providing separate arrays for anything else that is to be
#    computed.
#
#    Thus the calling sequence
#
#      call dqrsl ( a, lda, n, k, qraux, y, dum, y, b, y, dum, 110, info )
#
#    will result in the computation of B and RSD, with RSD
#    overwriting Y.  More generally, each item in the following
#    list contains groups of permissible identifications for
#    a single calling sequence.
#
#      1. (Y,QTY,B) (RSD) (AB) (QY)
#
#      2. (Y,QTY,RSD) (B) (AB) (QY)
#
#      3. (Y,QTY,AB) (B) (RSD) (QY)
#
#      4. (Y,QY) (QTY,B) (RSD) (AB)
#
#      5. (Y,QY) (QTY,RSD) (B) (AB)
#
#      6. (Y,QY) (QTY,AB) (B) (RSD)
#
#    In any group the value returned in the array allocated to
#    the group corresponds to the last member of the group.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    30 August 2016
#
#  Author:
#
#    Original FORTRAN77 version by Dongarra, Moler, Bunch, Stewart.
#    Python version by John Burkardt.
#
#  Reference:
#
#    Dongarra, Moler, Bunch, Stewart,
#    LINPACK User's Guide,
#    SIAM, (Society for Industrial and Applied Mathematics),
#    3600 University City Science Center,
#    Philadelphia, PA, 19104-2688.
#    ISBN 0-89871-172-X
#
#  Parameters:
#
#    Input, real A(LDA,P), contains the output of DQRDC.
#
#    Input, integer LDA, the leading dimension of the array A.
#
#    Input, integer N, the number of rows of the matrix AK.  It must
#    have the same value as N in DQRDC.
#
#    Input, integer K, the number of columns of the matrix AK.  K
#    must not be greater than min(N,P), where P is the same as in the
#    calling sequence to DQRDC.
#
#    Input, real QRAUX(P), the auxiliary output from DQRDC.
#
#    Input, real Y(N), a vector to be manipulated by DQRSL.
#
#    Input, integer JOB, specifies what is to be computed.  JOB has
#    the decimal expansion ABCDE, with the following meaning:
#      if A /= 0, compute QY.
#      if B /= 0, compute QTY.
#      if C /= 0, compute QTY and B.
#      if D /= 0, compute QTY and RSD.
#      if E /= 0, compute QTY and AB.
#    Note that a request to compute B, RSD, or AB automatically triggers
#    the computation of QTY, for which an array must be provided in the
#    calling sequence.
#
#    Output, real QY(N), contains Q * Y, if requested.
#
#    Output, real QTY(N), contains Q' * Y, if requested.
#
#    Output, real B(K), the solution of the least squares problem
#      minimize norm2 ( Y - AK * B),
#    if its computation has been requested.  Note that if pivoting was
#    requested in DQRDC, the J-th component of B will be associated with
#    column JPVT(J) of the original matrix A that was input into DQRDC.
#
#    Output, real RSD(N), the least squares residual Y - AK * B,
#    if its computation has been requested.  RSD is also the orthogonal
#    projection of Y onto the orthogonal complement of the column space
#    of AK.
#
#    Output, real AB(N), the least squares approximation Ak * B,
#    if its computation has been requested.  AB is also the orthogonal
#    projection of Y onto the column space of A.
#
#    Output, integer INFO, is zero unless the computation of B has
#    been requested and R is exactly singular.  In this case, INFO is the
#    index of the first zero diagonal element of R, and B is left unaltered.
#
  import numpy as np

  qy = np.zeros ( n )
  qty = np.zeros ( n )
  b = np.zeros ( k )
  rsd = np.zeros ( n )
  ab = np.zeros ( n )
#
#  Set info flag.
#
  info = 0
#
#  Determine what is to be computed.
#
  cqy =  int   ( job / 10000         ) != 0
  cqty =       ( job % 10000 )         != 0
  cb =   int ( ( job %  1000 ) / 100 ) != 0
  cr =   int ( ( job %   100 ) /  10 ) != 0
  cab =        ( job %    10 )         != 0

  ju = min ( k, n - 1 )
#
#  Special action when N = 1.
#
  if ( ju == 0 ):

    qy[0] = y[0]
    qty[0] = y[0]
    ab[0] = y[0]

    if ( a[0,0] == 0.0 ):
      info = 1
    else:
      b[0] = y[0] / a[0]

    rsd[0] = 0.0

    return qy, qty, b, rsd, ab, info
#
#  Set up to compute QY or QTY.
#
  for i in range ( 0, n ):
    qy[i] = y[i]
    qty[i] = y[i]
#
#  Compute QY.
#
  if ( cqy ):

    for j in range ( ju - 1, -1, -1 ):

      if ( qraux[j] != 0.0 ):
        ajj = a[j,j]
        a[j,j] = qraux[j]
        t = 0.0
        for i in range ( j, n ):
          t = t - a[i,j] * qy[i]
        t = t / a[j,j]
        for i in range ( j, n ):
          qy[i] = qy[i] + t * a[i,j]
        a[j,j] = ajj
#
#  Compute Q'*Y.
#
  if ( cqty ):

    for j in range ( 0, ju ):
      if ( qraux[j] != 0.0 ):
        ajj = a[j,j]
        a[j,j] = qraux[j]
        t = 0.0
        for i in range ( j, n ):
          t = t - a[i,j] * qty[i]
        t = t / a[j,j]
        for i in range ( j, n ):
          qty[i] = qty[i] + t * a[i,j]
        a[j,j] = ajj
#
#  Set up to compute B, RSD, or AB.
#
  for i in range ( 0, k ):
    b[i] = qty[i]
    ab[i] = qty[i]

  for i in range ( k, n ):
    rsd[i] = qty[i]
#
#  Compute B.
#
  if ( cb ):

    for j in range ( k - 1, -1, -1 ):

      if ( a[j,j] == 0.0 ):
        info = j + 1
        break

      b[j] = b[j] / a[j,j]

      if ( 0 < j ):
        t = - b[j]
        for i in range ( 0, j ):
          b[i] = b[i] + t * a[i,j]
#
#  Compute RSD or AB as required.
#
  if ( cr or cab ):

    for j in range ( ju - 1, -1, -1 ):

      if ( qraux[j] != 0.0 ):

        ajj = a[j,j]
        a[j,j] = qraux[j]

        if ( cr ):
          t = 0.0
          for i in range ( j, n ):
            t = t - a[i,j] * rsd[i]
          t = t / a[j,j]
          for i in range ( j, n ):
            rsd[i] = rsd[i] + t * a[i,j]

        if ( cab ):
          t = 0.0
          for i in range ( j, n ):
            t = t - a[i,j] * ab[i]
          t = t / a[j,j]
          for i in range ( j, n ):
            ab[i] = ab[i] + t * a[i,j]

        a[j,j] = ajj

  return qy, qty, b, rsd, ab, info

def dqrsl_test ( ):

#*****************************************************************************80
#
## DQRSL_TEST tests DQRSL.
#
#  Discussion:
#
#    DQRSL can solve a linear system that was factored by DQRDC.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    30 August 2016
#
#  Author:
#
#    John Burkardt
#
  import numpy as np
  import platform
  from dqrdc import dqrdc

  n = 5
  p = 3
  lda = n

  print ( '' )
  print ( 'DQRSL_TEST' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  DQRSL solves a rectangular linear system A*x=b in the' )
  print ( '  least squares sense after A has been factored by DQRDC.' )
#
#  Set the matrix A.
#
  a = np.array \
  ( [ \
    [ 1.0,  1.0,  1.0 ], \
    [ 1.0,  2.0,  4.0 ], \
    [ 1.0,  3.0,  9.0 ], \
    [ 1.0,  4.0, 16.0 ], \
    [ 1.0,  5.0, 25.0 ]  \
  ] )

  print ( '' )
  print ( '  The matrix A:' )
  print ( '' )

  for i in range ( 0, n ):
    for j in range ( 0, p ):
      print ( '  %14.6g' % ( a[i,j] ), end = '' )
    print ( '' )
#
#  Decompose the matrix.
#
  print ( '' )
  print ( '  Decompose the matrix.' )

  job = 0
  ipvt = np.zeros ( n, dtype = np.int32 )

  a, qraux, ipvt = dqrdc ( a, lda, n, p, ipvt, job )
#
#  Call DQRSL to compute the least squares solution A*x=b.
#
  job = 110

  y = np.array \
  ( [ \
    1.0, \
    2.3, \
    4.6, \
    3.1, \
    1.2  \
  ] )  

  b2 = np.array \
  ( [ \
    -3.02, \
     4.4914286, \
    -0.72857143 
  ] )

  qy, qty, b, r, xb, info = dqrsl ( a, lda, n, p, qraux, y, job )

  if ( info != 0 ):
    print ( '' )
    print ( 'DQRSL_TEST - Warning!' )
    print ( '  DQRSL returns INFO = %d' % ( info ) )
    return

  print ( '' )
  print ( '      X          X(expected):' )
  print ( '' )

  for i in range ( 0, p ):
    print ( '  %14.6g  %14.6g' % ( b[i], b2[i] ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'DQRSL_TEST' )
  print ( '  Normal end of execution.' )
  return

if ( __name__ == '__main__' ):
  from timestamp import timestamp
  timestamp ( )
  dqrsl_test ( )
  timestamp ( )
 
