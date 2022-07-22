from   .ActiveSet    import ActiveSet
from   .ClosurePool  import ClosurePool, DummyPool
from    math         import isinf, sqrt
import  numpy        as     np
from   .PathSearch   import FindDistAbs, FindDistAbsCg, FindDist, FindDistCg
from    random       import random, gauss, randint
from    scipy.sparse import issparse
from   .Util         import Add, ChunkedColNorm, ToColOrder, MakeWeights, GetCol, WDot

EPS      = 1e-12   # Errs < EPS are considered negligible and ignored in grad calculation
DERR_MAX = 1e-5    # If rate of change of error exceeds DERR_MAX at least DEL from
DEL      = 1e-8    # the starting position, then further line search is abandoned

# Error modes
HINGE = 0          # Hinge loss error
LSTSQ = 1          # Least squares
ABSER = 2          # Absolute error

#--------------------------------------------------------------------------------
#    Desc: Feature index and direction to column number
#--------------------------------------------------------------------------------
#       f: Feature index
#       d: Direction (-1/+1)
#--------------------------------------------------------------------------------
#     RET: Column index
#--------------------------------------------------------------------------------
def ColFromFD(f, d):
   return (f << 1) | (d > 0)

#--------------------------------------------------------------------------------
#    Desc: date column traversal order indices
#--------------------------------------------------------------------------------
#     PLV: Path loss values
#       o: Column order method
#--------------------------------------------------------------------------------
#     RET: Indices specifying order to traverse columns
#--------------------------------------------------------------------------------
def ColToFD(fd):
   return fd >> 1, ((fd % 2) << 1) - 1

#--------------------------------------------------------------------------------
#   Desc: Compute hinge-loss error roc for columns of data matrix
#--------------------------------------------------------------------------------
#     Ai: Data matrix
#      Y: Targets
#      W: Sample weights
#      S: Target signs
#      B: Boundary array
#      d: Direction (+/-)
#--------------------------------------------------------------------------------
#    RET: Error rate of change in signed column direction
#--------------------------------------------------------------------------------
def ColumnGradients(Ai, Y, W, S, B, d):
   DSi      = np.int8(d) * SignInt8(Ai)
   Ai       = np.abs(Ai)
   # These increase error
   grad     = W @ ((Ai * (S != DSi) * (B >= 0)) if not issparse(Ai) else
                    Ai.multiply((DSi.multiply(S) < 0).multiply(B >= 0)))
   # These decrease error
   grad    -= W @ ((Ai * (S == DSi) * (B >  0)) if not issparse(Ai) else
                    Ai.multiply((DSi.multiply(S) > 0).multiply(B > 0)))
   return grad   # Error change in direction from this point

#--------------------------------------------------------------------------------
#    Desc: Path constraint function
#--------------------------------------------------------------------------------
#      CV: Coefficient vector
#      CN: Column norms
#       b: Intercept
#     err: Current error
#    fMin: Coefficient lower bounds
#    fMax: Coefficient upper bounds
#     bAr: Below coefficient index constraints
#     aAr: Above coefficient index constraints
#       f: Feature index
#       d: Direction
#    dMax: Maximum distance to move along path before retrying direction
#--------------------------------------------------------------------------------
#     RET: Return maximum feasible distance along path
#--------------------------------------------------------------------------------
def CnstPath(CV, CN, b, err, fMin, fMax, bAr, aAr, f, d, dMax):
   if CN[f] == 0:
      return 0
   # Order constraint; preserves relative order of coefficients
   oBnd = np.inf
   if (aAr is not None) and (d > 0) and (aAr[f] != -1):
      oBnd = CV[aAr[f]] - CV[f]        # Coef approaching above boundary
   if (bAr is not None) and (d < 0) and (bAr[f] != -1):
      oBnd = CV[f] - CV[bAr[f]]        # Coef approaching below boundary
   # Feasible region boundary
   fBnd = (fMax[f] - CV[f]) if d > 0 else (CV[f] - fMin[f])
   # Min of order boundary, feasibility boundary, and normed dMax retry distance
   return min(oBnd, fBnd, dMax / CN[f])

#--------------------------------------------------------------------------------
#     Desc: Error Function for Absolute Loss
#--------------------------------------------------------------------------------
#        X: Solution vector
#        Y: Target values
#        W: Sample weights
#--------------------------------------------------------------------------------
#      RET: Hinge error
#--------------------------------------------------------------------------------
def ErrAbs(X, Y, W):
   return np.dot(np.abs(Y - X), W)

#--------------------------------------------------------------------------------
#     Desc: Error Function for Hinge Loss
#--------------------------------------------------------------------------------
#        X: Solution vector
#        Y: Target values (boolean, 0/1, or -1/+1)
#        W: Sample weights
#--------------------------------------------------------------------------------
#      RET: Hinge error
#--------------------------------------------------------------------------------
def ErrHinge(X, Y, W):
   return np.dot(np.maximum(SignInt8(Y) * (Y - X) , 0), W)

#--------------------------------------------------------------------------------
#     Desc: Error Function for Least-Squares Loss
#--------------------------------------------------------------------------------
#        X: Solution vector
#        Y: Target values
#        W: Sample weights
#--------------------------------------------------------------------------------
#      RET: Least-squares error
#--------------------------------------------------------------------------------
def ErrLstSq(X, Y, W):
   return np.dot(np.square(X - Y), W)

#--------------------------------------------------------------------------------
#     Desc: Error Function for Least-Squares Loss
#--------------------------------------------------------------------------------
#        X: Solution vector
#--------------------------------------------------------------------------------
#      RET: Search function, error function
#--------------------------------------------------------------------------------
def GetObjFx(obj):
   if   obj == HINGE:
      return SearchHinge, ErrHinge
   elif obj == LSTSQ:
      return SearchLstSq, ErrLstSq
   elif obj == ABSER:
      return SearchAbsEr, ErrAbs

#--------------------------------------------------------------------------------
#     Desc: Iterative Constrained Pathways Optimizer
#--------------------------------------------------------------------------------
#        A: Data matrix
#        Y: Target values (boolean, 0/1, or -1/+1)
#        W: Sample weights
#     fMin: Coefficient lower bounds
#     fMax: Coefficient upper bounds
#  maxIter: Maximum number of solver iterations
#      mrg: Classifier margin for hinge-loss
#        b: Initial guess (calculated log-odds of class 1 if None)
#        c: Constant value (0 to disable)
#     dMax: Maximum distance to move along path before retrying direction
#     norm: Normalize dMax by column norm (T/F)
#       CO: Columns sorted by error rate of change
#       fg: Feature groups (for constraining maximum number of allowed groups)
# maxGroup: Maximum number of allowed feature groups. No limit if <= 0. Once
#           num groups is exhausted, algorithm is constrained to only use
#           groups that have been already used. Negative groups are ignored
#      CFx: Criteria function for abandoning path (if true, path is abandoned)
#      tol: Moving average error reduction tolerance
#      obj: Error objective (HINGE, LSTSQ, etc.)
#     mOrd: Convert matrix to this order (C or F) if not None
#     clip: Clip coefs with magnitude less than this to exactly 0
#      bAr: Below coefficient index constraints
#      aAr: Above coefficient index constraints
#    nJump: Number of times to try and jump out of stall
#    nThrd: Number of threads to search for paths
#        v: Verbose mode (0 off; 1 low; 2 high)
#--------------------------------------------------------------------------------
#      RET: Coefficients, intercept
#--------------------------------------------------------------------------------
def ICPSolve(A, Y, W, fMin=None, fMax=None, maxIter=200, mrg=1.0, b=None, c=0, dMax=1.234,
             norm=True, CO=None, fg=None, maxGroup=0, CFx=CnstPath, tol=-1e-5, obj=HINGE,
             mOrd='F', clip=1e-9, bAr=None, aAr=None, nJump=0, nThrd=1, v=1):
   if mOrd is not None:
      A = ToColOrder(A)

   m, n = A.shape

   if obj == HINGE:
      Y  = np.where(Y > 0, mrg, -mrg)           # Signed margin values as targets

   W = MakeWeights(W, m)                        # Sample weights

   hc = int(c != 0)
   CV = np.zeros(n + hc)                        # Coefficient vector
   if fMin is not None:                         # Max value feature constraints
      CV   = np.maximum(CV, fMin)
   else:
      fMin = np.full(n + hc, -np.inf)           # Unconstrained below

   if fMax is not None:                         # Min value feature constraints
      CV   = np.minimum(CV, fMax)
   else:
      fMax = np.full(n + hc,  np.inf)           # Unconstrained above

   if b is None:
      b = np.dot(Y, W)                          # Initial guess
   if clip:
      b  = 0. if (-clip < b < clip) else b      # Clip small values to 0.

   X = np.full(m, b)                            # Current solution n.b. type depends on b
   for i in CV.nonzero()[0]:
      X += CV[i] * GetCol(A, i)

   feaSet = {}                                  # Used feature groups set
   for fgi in fg[CV != 0]:
      feaSet[fgi] = feaSet.get(fgi, 0) + 1

   bFea = -1                                    # Pre-declare some variables here
   u    =  0.0                                  # Current update
   err  =  np.inf                               # Current error
   mar  = -np.inf                               # Moving average of error reduction
   j    =  False                                # Jump flag

   nd  = (n + hc) << 1                          # 2 directions (+/-) for each column
   pch = 0.1
   if CO is None:
      pch = 0                                   # Default to hot set empty
      CO  = np.arange(nd)                       # Default column order

   # Column norms; use to adjust vMax by column length for a more fair policy
   CN = ChunkedColNorm(A, c=c) if norm else np.ones(n + hc)

   srchFx, errFx = GetObjFx(obj)

   # Create search closures
   CP = (DummyPool if nThrd == 1 else ClosurePool)(
      [srchFx(A=A, Y=Y, W=W, c=c, eps0=EPS, eps1=DERR_MAX, eps2=DEL)
       for _ in range(nThrd)])

   sRes = np.empty((nd, 3))                     # Keeps track of dir performance

   hotSet = ActiveSet(nd)                       # Initialize active "hot" set with
   hotSet.update(CO[:round(pch * nd)])          # highest scoring columns

   notSet = ActiveSet(nd)                       # Cool set initialized with rest
   notSet.update(CO[round(pch * nd):])
   runSet = ActiveSet(nd)                       # Temporary set

   err = errFx(X, Y, W)                         # Initialize error value

   i = 0                                        # Iteration count
   while True:                                  # Loop for each algorithm iteration

      if v > 0:
         print('{:5d} {:10.7f} {:10.7f} [{:5d}] {:+8.5f} {:+8.5f} {:4d} {:s} {:s}'.format(
                i, err, mar, bFea, CV[bFea], u, len(hotSet), '*'*(CV[bFea] == 0), 'j'*j))

      # Check if error within tolerance
      if (mar > tol) or (i > maxIter):
         if v > 1: print('Reached stopping criteria')
         break
      i += 1

      for nl in range(nd):                      # Loop until a direction is found
         key, rv = CP.Get(wait=CP.Full())       # Try to get a finished search result
         if rv is not None:
            sRes[key, :] = rv
            if rv[0] < -EPS:                    # Check if error reduction down this
               break                            # path is good enough; break

         # Try another column
         if (len(notSet) > 0) and (random() > (1 - 1 / (0.25 + sqrt(len(hotSet))))):
            cp = notSet.popr()
         else:
            cp = hotSet.popr()
         runSet.add(cp)                         # Record col; tbd if goes to hot/cold set
         f, d  = ColToFD(cp)

         # Find constraint along path
         vMax = CFx(CV, CN, b, err, fMin, fMax, bAr, aAr, f, d, dMax)
         if vMax <= 0:
            sRes[cp, :] = np.nan                # Mark this col for cold set
            continue                            # This direction is constrained; skip

         CP.Start(key=cp, args=(X, vMax, f, d))

      for key, rv in CP.GetAll():               # Join any running threads and get results
         sRes[key, :] = rv

      bErr = np.inf
      bCp  = 0
      while len(runSet) > 0:                    # Find best dir & refill hot/cold sets
         cp = runSet.pop()
         cErr, _, cSla = sRes[cp]               # Err, distance, slack to constraint
         if cErr < bErr:
            bErr = cErr
            bCp  = cp

         if (cErr < -EPS) and (cSla < EPS):
            hotSet.add(cp)                      # Add to hot set; column looks promising
         else:
            notSet.add(cp)                      # Cool set; column isn't helping now

      j             = False                     # Jump flag
      bFea, bDir    = ColToFD(bCp)              # Coefficient index and direction
      # Stall when err > EPS or (-EPS <= err <= EPS) and coef magnitude increases
      if (bErr >= -EPS) and ((bErr > EPS) or not RedMag(CV[bFea], bDir)):
         if nJump > 0:                          # Try to jump out of stall a
            nJump -= 1                          # limited number of times
            bFea   = f = randint(0, n - 1 + hc)
            u      = gauss(0, 10 * abs(tol))    # Random normal preturbation
            CV[f] += u
            if f < n:                           # Preturb coef
               X   = Add(X, GetCol(A, f) * u)
            else:                               # Preturb bias
               X  += u * c
            j      = True                       # This update is a jump
            continue
         if v > 1: print('Algorithm Stalled: ({0}, {1}, {2})'.format(bErr, f, u))
         break                                  # Algorithm stalled; break

      _, bDst, bSla = sRes[bCp]                 # Distance and slack; error from above
      u             = bDir * bDst               # The current proposed coefficient update

      if maxGroup > 0:
         if -clip < (CV[bFea] + u) < clip:
            if fg[bFea] > 0:
               feaSet[fg[bFea]] = cc = feaSet.get(fg[bFea], 0) - 1
               if cc <= 0:                      # Coef is now 0; remove from feature set
                  del feaSet[fg[bFea]]          # All coef in group 0; delete group
         elif CV[bFea] == 0:                    # This is a new non-zero coef
            feaSet[fg[bFea]] = feaSet.get(fg[bFea], 0) + 1

      if bFea < n:                              # Update via coef
         Cf = u * GetCol(A, bFea)               # Get column bFea
      else:                                     # Update via bias
         Cf = u * c

      X         = Add(X, Cf)                    # Column update; handle sparse
      CV[bFea] += u                             # Update coefficient vector
      # Moving average error reduction for stopping criteria
      mar  = bErr if isinf(mar) else (0.05 * bErr + 0.95 * mar)
      err += bErr

   # Clip very small coef to exactly zero
   for i in range(len(CV)):
      if -clip < CV[i] < clip:
         CV[i] = 0.

   return CV, b, err, i

#--------------------------------------------------------------------------------
#   Desc: Iterative Constrained Pathways Solver with constant columns
#--------------------------------------------------------------------------------
#      A: Data matrix
#      T: Target values (boolean, 0/1, or -1/+1)
#      W: Sample weights
#      c: Constant column
# kwargs: See description of ICPSolve
#--------------------------------------------------------------------------------
#    RET: Coefficients, intercept
#--------------------------------------------------------------------------------
def ICPSolveConst(A, Y, W, c=1, **kwargs):
   _, n = A.shape
   CV, b, err, i = ICPSolve(A, Y, W, c=c, **kwargs)

   if c == 0:
      return CV, b, err, i

   cCol = [n]
   # Combine bias from original solution with all const columns
   b = b + CV[cCol].sum()

   # Remove constant columns from coefficient vector
   cIdx = np.ones(n + 1, np.bool)
   cIdx[cCol] = False
   CV = CV[cIdx].copy()

   return CV, b, err, i

#--------------------------------------------------------------------------------
#   Desc: Check if a coefficient update reduces its magnitude
#--------------------------------------------------------------------------------
#      c: Coefficient
#      u: Update
#--------------------------------------------------------------------------------
#    RET: True if update u reduces magnitude of coefficient c
#--------------------------------------------------------------------------------
def RedMag(c, u):
   return ((c < 0) and (u > 0)) or ((c > 0) and (u < 0))

#--------------------------------------------------------------------------------
#   Desc: Get univariate rule error
#--------------------------------------------------------------------------------
#      A: The rule matrix
#      Y: The target margin values
#      W: Sample weights
#      d: The direction to move in
#     bs: Block size (larger values increase memory usage)
#      c: Constant column (0/1)
#--------------------------------------------------------------------------------
#    RET: Change in error in signed column direction
#--------------------------------------------------------------------------------
def RuleErr(A, Y, W=None, b=None, d=1, bs=1.6e7, c=1):
   m, n = A.shape
   W    = MakeWeights(W, m)
   if b is None:
      b = np.dot(Y, W)
   Y = Y.reshape(-1, 1)
   S = SignInt8(Y)
   B = S * (Y - b)
   # Split up for low-memory
   err = np.empty(n + (c != 0))
   cs  = max(1, round(bs / m))
   s   = e = 0
   while s < n:
      e        = min(e + cs, n)
      err[s:e] = ColumnGradients(GetCol(A, slice(s, e)), Y, W, S, B, d)
      s        = e
   # Handle constant column separately
   if c != 0:
      err[n] = ColumnGradients(np.full((m, 1), c), Y, W, S, B, d)
   return err

#--------------------------------------------------------------------------------
#    Desc: Closure wrapping path search functions along with temp buffers
#--------------------------------------------------------------------------------
#       A: Data matrix
#       Y: Target values (boolean, 0/1, or -1/+1)
#       W: Sample weights
#       c: Constant value
#--------------------------------------------------------------------------------
#     RET: Function that provides (best error along path, best distance to move)
#--------------------------------------------------------------------------------
def SearchAbsEr(A, Y, W, c, eps0, *args, **kwargs):
   m, n = A.shape
   # Pre-allocate temporary vectors for search
   Ac   = np.empty(m, dtype=A.dtype)
   CW   = np.empty(m + 1)
   BV   = np.empty(m + 1)
   BVsi = np.empty(m + 1, np.intp)

   rvs  = np.empty(3)

   # Search closure
   def Fx(X, vMax, f, d):
      if f < n:
         Af = GetCol(A, f)
      else:                                  # Constant column == n
         Ac.fill(c)
         Af = Ac

      if issparse(Af):
         r  = Af.indices         # Only non-zero elements impact update
         v  = Af.data            # N.b. that "data" may contain 0s if e.g. the
         nf = v.shape[0]         # sparse-vector is multiplied by 0. If .data is used
         Ac[:nf] = v             # then .indices should be used instead of .nonzero()

         FindDistAbs(Ac, nf, Y[r], W[r], X[r], d, vMax, eps0, rvs, CW, BV, BVsi)
      elif isinstance(Af, np.ndarray) and Af.data.contiguous:
         FindDistAbsCg(Af, m, Y, W, X, d, vMax, eps0, rvs, CW, BV, BVsi)
      else:    # Dense non-contiguous; copy to contiguous array
         Ac[:] = Af
         FindDistAbsCg(Ac, m, Y, W, X, d, vMax, eps0, rvs, CW, BV, BVsi)
      return rvs

   return Fx


#--------------------------------------------------------------------------------
#    Desc: Closure wrapping path search functions along with temp buffers
#--------------------------------------------------------------------------------
#       A: Data matrix
#       Y: Target values (boolean, 0/1, or -1/+1)
#       W: Sample weights
#       c: Constant value
#    eps0: Movement must improve error by at least this much to count
#    eps1: If gradient ever becomes larger than this value, stop searching
#    eps2: If gradient is larger than eps1 and have moved at least eps2 from start; break
#--------------------------------------------------------------------------------
#     RET: Function that provides (best error along path, best distance to move)
#--------------------------------------------------------------------------------
def SearchHinge(A, Y, W, c, eps0, eps1, eps2, *args, **kwargs):
   m, n = A.shape
   # Pre-allocate temporary vectors for search
   BVae = np.empty(m + 1)
   AWae = np.empty(m)
   AEsi = np.empty(m + 1, np.intp)
   BVse = np.empty(m + 1)
   AWse = np.empty(m)
   SEsi = np.empty(m + 1, np.intp)

   Ac  = np.empty(m, dtype=A.dtype)
   rvs = np.empty(3)

   # Search closure
   def Fx(X, vMax, f, d):
      if f < n:
         Af = GetCol(A, f)
      else:                      # Constant column == n
         Ac.fill(c)
         Af = Ac

      if issparse(Af):
         r  = Af.indices         # Only non-zero elements impact update
         v  = Af.data            # N.b. that "data" may contain 0s if e.g. the
         nf = v.shape[0]         # sparse-vector is multiplied by 0. If .data is used
         Ac[:nf] = v             # then .indices should be used instead of .nonzero()

         FindDist(Ac, nf, Y[r], W[r], X[r], d, vMax, eps0, eps1, eps2,
                  rvs, BVae, AWae, AEsi, BVse, AWse, SEsi)
      elif isinstance(Af, np.ndarray) and Af.data.contiguous:
         FindDistCg(Af, m, Y, W, X, d, vMax, eps0, eps1, eps2, rvs,
                    BVae, AWae, AEsi, BVse, AWse, SEsi)
      else:    # Dense non-contiguous; copy to contiguous array
         Ac[:] = Af
         FindDistCg(Ac, m, Y, W, X, d, vMax, eps0, eps1, eps2, rvs,
                    BVae, AWae, AEsi, BVse, AWse, SEsi)
      return rvs

   return Fx

#--------------------------------------------------------------------------------
#    Desc: Closure wrapping path search functions along with temp buffers
#--------------------------------------------------------------------------------
#       A: Data matrix
#       Y: Target values (boolean, 0/1, or -1/+1)
#       W: Sample weights
#       c: Constant value
#--------------------------------------------------------------------------------
#     RET: Function that provides (best error along path, best distance to move)
#--------------------------------------------------------------------------------
def SearchLstSq(A, Y, W, c, *args, **kwargs):
   m, n = A.shape
   # Pre-allocate temporary vectors for search
   Ac = np.empty(m, dtype=A.dtype)

   # Search closure
   def Fx(X, vMax, f, d):
      if f < n:
         C = GetCol(A, f)
      else:                                  # Constant column == n
         Ac.fill(c)
         C = Ac

      R   = (Y - X)                          # Residual vector
      CdC = WDot(C, C, W)                    # Weighted dot product
      CdR = WDot(R, C, W) * d                # Signed & weighted dot product
      u   = CdR / CdC                        # Proj of col onto residual vec
      if u < 0:
         return 0., 0., vMax                 # Wrong direction; abort
      u  = min(u, vMax)                      # Only move up to contraint
      e  = u * u * CdC - 2 * u * CdR         # Update to quadratic loss
      return e, u, vMax - u

   return Fx

#--------------------------------------------------------------------------------
#   Desc: Sign function with buffer
#--------------------------------------------------------------------------------
#      X: The target array
#    eps: np.abs(x) > eps in order to have non-zero sign result
#--------------------------------------------------------------------------------
#    RET: Sign as a 8 bit int
#--------------------------------------------------------------------------------
def SignInt8(X, eps=EPS):
   R  = (X > EPS).astype(np.int8)
   R -= (X < -EPS).astype(np.int8)
   return R
