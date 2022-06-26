import  numpy        as     np
from    .PathSearch  import FindDist
from    random       import randint
from    .Util        import ChunkedDotCol, ChunkedDotRow, GroupBy, ToColOrder
from    .ClosurePool import ClosurePool

EPS      = 1e-12   # Errs < EPS are considered negligible and ignored in grad calculation
DERR_MAX = 1e-5    # If rate of change of error exceeds DERR_MAX at least DEL from
DEL      = 1e-8    # the starting position, then further line search is abandoned

#--------------------------------------------------------------------------------
#    Desc: date column traversal order indices
#--------------------------------------------------------------------------------
#      CO: Column traversal order
#      ep: Element traversal index
#      cp: Current traversal index
#     nsd: Minimum swap distance
#     xsd: Maximum swap distance
#--------------------------------------------------------------------------------
#   Return: Indices specifying order to traverse columns
#--------------------------------------------------------------------------------
def ColOrder(CO, ep, cp, nsd, xsd):
   np = randint(nsd, xsd) if (xsd > nsd) else nsd
   np = (cp + np) % CO.shape[0]
   CO[ep], CO[np] = CO[np], CO[ep]
   return CO

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
#    Desc: Closure wrapping path search functions along with temp buffers
#--------------------------------------------------------------------------------
#       A: Data matrix
#    eps0: Movement must improve error by at least this much to count
#    eps1: If gradient ever becomes larger than this value, stop searching
#    eps2: If gradient is larger than eps1 and have moved at least eps2 from start; break
#--------------------------------------------------------------------------------
#     RET: Function that provides (best error along path, best distance to move)
#--------------------------------------------------------------------------------
def DistSearch(A, Y, W, S, eps0, eps1, eps2):
   n, _ = A.shape
   # Pre-allocate temporary vectors for search
   BVae = np.empty(n + 1)
   AWae = np.empty(n)
   AEsi = np.empty(n, np.intp)

   BVse = np.empty(n + 1)
   AWse = np.empty(n)
   SEsi = np.empty(n, np.intp)

   tmp = np.empty(n)


   def Fx(B, X, vMax, f, d):
      nonlocal n, eps0, eps1, eps2, BVae, AWae, AEsi, BVse, AWse, SEsi, tmp

      rvs = np.empty(3)
      FindDist(A, n, Y, W, S, B, X, f, d, vMax, eps0, eps1, eps2, rvs,
               BVae, AWae, AEsi, BVse, AWse, SEsi, tmp)

      return rvs

   return Fx

#--------------------------------------------------------------------------------
#    Desc: Path constraint function
#--------------------------------------------------------------------------------
#      CV: Coefficient vector
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
def ErrInc(CV, b, err, fMin, fMax, bAr, aAr, f, d, dMax):
   # Order constraint; preserves relative order of coefficients
   oBnd = np.inf
   if (aAr is not None) and (d > 0) and (aAr[f] != -1):
      oBnd = CV[aAr[f]] - CV[f]        # Coef approaching above boundary
   if (bAr is not None) and (d < 0) and (bAr[f] != -1):
      oBnd = CV[f] - CV[bAr[f]]        # Coef approaching below boundary
   # Feasible region boundary
   fBnd = (fMax[f] - CV[f]) if d > 0 else (CV[f] - fMin[f])
   # Minimum satisfies all constraints
   return min(oBnd, fBnd, dMax)

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
#     dMax: Maximum distance to move along path before retrying direction
#       bs: Block size for calculating initial dot product (lower values use less memory)
#      nsd: Min swap distance
#      xsd: Max swap distance
#       CO: Column traversal order
#       fg: Feature groups (for constraining maximum number of allowed groups)
# maxGroup: Maximum number of allowed feature groups. No limit if <= 0. Once
#           num groups is exhausted, algorithm is constrained to only use
#           groups that have been already used. Negative groups are ignored
#      CFx: Criteria function for abandoning path (if true, path is abandoned)
#      tol: Moving average error reduction tolerance
#     mOrd: Convert matrix to this order (C or F) if not None
#     clip: Clip coefs with magnitude less than this to exactly 0
#      bAr: Below coefficient index constraints
#      aAr: Above coefficient index constraints
#    nThrd: Number of threads to search for paths
#     gThr: Number of successful iterations required to grow eps0
#     eps0: Initial error reduction tolerance
#     eps1: Minimum error reduction tolerance
#        v: Verbose mode (0 off; 1 low; 2 high)
#--------------------------------------------------------------------------------
#      RET: Coefficients, intercept
#--------------------------------------------------------------------------------
def ICPSolve(A, Y, W, fMin=None, fMax=None, maxIter=200, mrg=1.0, b=None, dMax=0.1,
             bs=16000000, nsd=0, xsd=0.5, CO=None, fg=None, maxGroup=0, CFx=None,
             tol=-1e-5, mOrd='F', clip=1e-9, bAr=None, aAr=None, nThrd=1, gThr=8,
             eps0=-1e-5, eps1=-EPS, v=1):
   if mOrd is not None:
      A = ToColOrder(A)
   if np.issubdtype(A.dtype, np.bool):
      A = A.view(np.int8)
   n, m  = A.shape

   Y  = np.where(Y > 0, mrg, -mrg)            # Clipped log odds target
   S  = SignInt8(Y)                           # Sample sign

   if W is None:
      W = np.full(Y.shape[0], 1 / Y.shape[0]) # Default to equal weights
   else:
      W = W / W.sum()                         # Force sum==1 for weighted avg

   CV = np.zeros(m)                           # Coefficient vector
   if fMin is not None:                       # Max value feature constraints
      CV   = np.maximum(CV, fMin)
   else:
      fMin = np.full(m, -np.inf)

   if fMax is not None:                       # Min value feature constraints
      CV   = np.minimum(CV, fMax)
   else:
      fMax = np.full(m,  np.inf)

   if b is None:
      b = np.dot(Y, W)                        # Initial guess
   if clip:
      b  = b if (np.abs(b) >= clip) else 0    # Clip small values to 0

   X = ChunkedDotRow(A, CV, bs=bs) + b        # Current solution

   feaSet = set()                             # Used feature groups set

   if CFx is None:
      CFx = ErrInc                            # Column constrain function

   bFea = -1
   c    =  0                                  # Iteration count
   u    =  0.0                                # Current update
   err  =  np.inf                             # Current error
   mar  = -np.inf                             # Moving average of error reduction

   nd  = CV.shape[0] << 1                     # 2 directions (+/-) for each column
   sp  = 0                                    # Column order start pointer
   if CO is None:
      CO = np.arange(nd)                      # Default column order

   # Minimum and max swap distances
   nsd = round(nsd * nd) if isinstance(nsd, float) else nsd
   xsd = round(xsd * nd) if isinstance(xsd, float) else xsd
   nsd = min(nsd, xsd)
   xsd = max(nsd, xsd)

   # Create search closures
   CP = ClosurePool([DistSearch(A, Y, W, S, EPS, DERR_MAX, DEL) for _ in range(nThrd)])

   sCol = np.empty(nd, dtype=np.int)            # Keeps track of thread -> dir info
   sRes = np.empty((nd, 3))                     # Keeps track of dir performance

   nPass  = 0
   epsMax = eps0 * 10

   while True:                                  # Loop for each algorithm iteration
      B   = S * (Y - X)                         # Distance to margin; <0 if correct
      err = (np.maximum(B, 0) * W).sum()        # Mean hinge error

      if v > 0:
         print('{:5d} {:10.7f} {:10.7f} [{:5d}] {:+8.5f} {:+8.5f}'.format(
                                                      c, err, mar, bFea, CV[bFea], u))
      # Check if error within tolerance
      if (mar > tol) or (c > maxIter):
         if v > 1: print('Reached stopping criteria')
         break
      c   += 1

      sRes.fill(np.nan)
      for nl in range(nd):                      # Loop until a direction is found
         cp    = (sp + nl) % nd
         f, d  = ColToFD(CO[cp])

         key, rv = CP.Get(wait=CP.Full())       # Try to get a finished search result
         if rv is not None:
            cErr, cDst, cSla = rv               # Error, distance moved, slack to constraint
            sRes[key, :] = rv
            sCol[key] = CO[key]
            if cErr <= eps0:
               break                            # Error reduction down this path is good enough; break

         # Find constraint along path
         vMax = CFx(CV, b, err, fMin, fMax, bAr, aAr, f, d, dMax)
         if vMax <= 0:
            continue

         CP.Start(key=cp, args=(B, X, vMax, f, d))

      # Join all remaining threads and get results
      for key, rv in CP.GetAll():
         sRes[key, :] = rv
         sCol[key]    = CO[key]

      bErr = np.inf
      for i in range(nl):
         cp = (sp + i) % nd

         cErr, cDst, cSla = sRes[cp]
         if cErr < bErr:
            bErr = cErr
            bDst = cDst
            bFea, bDir = ColToFD(sCol[cp])

         # Update traversal plan by moving columns that reduce error and have slack ahead
         if (cErr < 0.0) and (cSla > 0.0):
            ColOrder(CO, cp, sp + nl, nsd, xsd)

      u = bDir * bDst            # The current proposed coefficient update
      if (bErr >= EPS) or ((bErr > -EPS) and not RedMag(CV[f], u)):
         if v > 1: print('Algorithm Stalled: ({0}, {1}, {2})'.format(bErr, f, u))
         break                   # Cannot reduce error more than eps1; break

      if bErr <= eps0:
         nPass += 1
         if (nPass > gThr) and ((eps0 * 10) >= epsMax):
            if v > 1: print('Growing eps0: {:g} -> {:g}'.format(eps0, eps0 * 10.0))
            nPass  =  0
            eps0  *= 10
      else:
         if v > 1: print('Shrinking eps0: {:g} -> {:g}'.format(eps0, eps0 / 10.0))
         eps0  /= 10.0              # Adjust eps0 as necessary
         nPass  = 0

      # TODO: handle columns set to 0
      if (maxGroup > 0) and (fg[bFea] > 0):
         feaSet.add(fg[bFea])       # This col may use a new group; count towards limit

      if np.abs(CV[bFea] + u) < clip:
         if v > 1: print('Clip: {:g} -> 0.0'.format(CV[bFea] + u))
         u = -CV[bFea]              # Quantize coefficients that are very close to 0

      CV[bFea] += u                 # Update coefficient vector
      X        += u * A[:, bFea]    # Update current solution

      sp        = (sp + nl) % nd    # Update starting position
      mar = bErr if np.isinf(mar) else (0.05 * bErr + 0.95 * mar)

   return CV, b, err, c

#--------------------------------------------------------------------------------
#   Desc: Iterative Constrained Pathways Solver with constant columns
#--------------------------------------------------------------------------------
#      A: Data matrix
#      T: Target values (boolean, 0/1, or -1/+1)
#      W: Sample weights
# kwargs: See description of ICPSolve
#--------------------------------------------------------------------------------
#    RET: Coefficients, intercept
#--------------------------------------------------------------------------------
def ICPSolveConst(A, Y, W, cCol=None, **kwargs):
   CV, b, err, c = ICPSolve(A, Y, W, **kwargs)

   if cCol is None:
      return CV, b, err, c

   if not hasattr(cCol, '__len__'):
      cCol = [cCol]

   # Combine bias from original solution with all const columns
   b = b + CV[cCol].sum()

   # Remove constant columns from coefficient vector
   cIdx = np.ones(A.shape[1], np.bool)
   cIdx[cCol] = False
   CV = CV[cIdx].copy()

   return CV, b, err, c

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
#     cs: Chunk size (larger values increase memory usage)
#--------------------------------------------------------------------------------
#    RET: Change in error in signed column direction
#--------------------------------------------------------------------------------
def RuleErr(A, Y, W=None, b=None, d=1, bs=16000000):
   if W is None:
      W = np.full(A.shape[0], 1 / A.shape[0])
   if b is None:
      b = np.dot(Y, W)
   Y = Y.reshape(-1, 1)
   S = SignInt8(Y)
   B = S * (Y - b)
   # Split up for low-memory
   n, m = A.shape
   err = np.empty(m)
   cs = bs // n
   s = e = 0
   while s < m:
      e       += cs
      Ai       = A[:, s:e]
      DSi      = np.int8(d) * SignInt8(Ai)
      Ai       = np.abs(Ai)
      aea      = W @ (Ai * (S != DSi) * (B >= 0))    # These increase error
      sea      = W @ (Ai * (S == DSi) * (B >  0))    # These decrease error
      err[s:e] = aea - sea                           # Error change in direction
      s        = e
   return err

#--------------------------------------------------------------------------------
#   Desc: Get univariate rule orientation
#--------------------------------------------------------------------------------
#      A: The rule matrix
#      Y: The target values {-1, 1}
#      W: Sample weights
#      b: The base rate (mean of target if None)
#     bs: Block size (larger values increase memory usage)
#--------------------------------------------------------------------------------
#    RET: Sign indicating the direction of each rule
#--------------------------------------------------------------------------------
def RuleSign(A, Y, W=None, b=None, bs=16000000):
   if W is None:
      W = np.full(A.shape[0], 1 / A.shape[0])
   Y = Y * W
   if b is None:
      b = Y.sum()
   # Split up for low-memory
   nht = ChunkedDotCol(A, W, bs=bs)    # Weighted pct hits total
   nhp = ChunkedDotCol(A, Y, bs=bs)    # Weight Y sum for samples with rule hits
   # Amount each rule changes average target versus the base rate
   return (nhp / nht) - b

#--------------------------------------------------------------------------------
#   Desc: Create rule order constraints
#--------------------------------------------------------------------------------
#     fg: Feature groups
#     cs: Column score for ordering
#      m: Order mode:
#           a: Absolute; Order constraints irrespective of group
#           r: Relative; Order constraint only within same group
#--------------------------------------------------------------------------------
#    RET: Below constraints, Above constraints
#--------------------------------------------------------------------------------
def RuleOrder(fg, cs, m='r'):
   if m == 'a':   # Use absolute ordering
      fg = np.ones_like(fg)
   BA = np.empty_like(fg)
   AA = np.empty_like(fg)
   for fi, sfi in GroupBy(range(len(fg)), key=fg.__getitem__):
      sfi = np.array(sfi)
      rsi = sfi[cs[sfi].argsort()]
      for i, si in enumerate(rsi):
         BA[si] = rsi[i - 1] if i > 0                    else -1
         AA[si] = rsi[i + 1] if ((i + 1) < rsi.shape[0]) else -1
   return BA, AA

#--------------------------------------------------------------------------------
#   Desc: Sign function with buffer
#--------------------------------------------------------------------------------
#      X: The target array
#    eps: np.abs(x) > eps in order to have non-zero sign result
#--------------------------------------------------------------------------------
#    RET: Sign as a 8 bit int
#--------------------------------------------------------------------------------
def SignInt8(X, eps=EPS):
   return (X > EPS).astype(np.int8) - (X < -EPS).astype(np.int8)

