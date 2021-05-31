from    joblib      import Parallel, delayed
from    random      import randint
import  numpy       as     np
from    .PathSearch import FindDist
from    .Util       import GroupBy

EPS      = 1e-12   # Errs < EPS are considered negligible and ignored in grad calculation
DERR_MAX = 1e-5    # If rate of change of error exceeds DERR_MAX at least DEL from
DEL      = 1e-8    # the starting position, then further line search is abandoned

#--------------------------------------------------------------------------------
#    Desc: date column traversal order indices
#--------------------------------------------------------------------------------
#      CO: Column traversal order
#      cp: Current traversal index
#     nsd: Minimum swap distance
#     xsd: Maximum swap distance
#--------------------------------------------------------------------------------
#   Return: Indices specifying order to traverse columns
#--------------------------------------------------------------------------------
def ColOrder(CO, cp, nsd, xsd):
   np = randint(nsd, xsd) if (xsd > nsd) else nsd
   np = (cp + np) % CO.shape[0]
   CO[cp], CO[np] = CO[np], CO[cp]
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
def DistSearch(A, eps0, eps1, eps2):
   n, _ = A.shape
   # Pre-allocate temporary vectors for search
   BVae = np.empty(n + 1)
   AWae = np.empty(n + 1)
   AEsi = np.empty(n, np.intp)

   BVse = np.empty(n + 1)
   AWse = np.empty(n + 1)
   SEsi = np.empty(n, np.intp)

   tmp = np.empty(n)
   rvs = np.empty(2)

   def Fx(Af, Y, W, S, B, X, vMax, d):
      nonlocal n, eps0, eps1, eps2, rvs, BVae, AWae, AEsi, BVse, AWse, SEsi, tmp
      FindDist(Af, n, Y, W, S, B, X, d, vMax, eps0, eps1, eps2, rvs,
             BVae, AWae, AEsi, BVse, AWse, SEsi, tmp)
      return tuple(rvs)

   return Fx

#--------------------------------------------------------------------------------
#    Desc: Path constraint function
#--------------------------------------------------------------------------------
#      CV: Coefficient vector
#       b: Intercept
#     err: Current error
#    fMin: Coefficient lower bounds
#    fMax: Coefficient upper bounds
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
#      mrg: Margin log-odds
#        b: Initial guess (calculated log-odds of class 1 if None)
#     dMax: Maximum distance to move along path before retrying direction
#      nsd: Min swap distance
#      xsd: Max swap distance
#       CO: Column traversal order
#       fg: Feature groups (for constraining maximum number of allowed groups)
# maxGroup: Maximum number of allowed feature groups. No limit if <= 0. Once
#           num groups is exhausted, algorithm is constrained to only use
#           groups that have been already used. Negative groups are ignored
#      CFx: Criteria function for abandoning path (if true, path is abandoned)
#      tol: Moving average error reduction tolerance
#     clip: Clip coefs with magnitude less than this to exactly 0
#      bAr: Below coefficient index constraints
#      aAr: Above coefficient index constraints
#    nPath: Number of paths to explore at each step (best found is used)
#    nThrd: Number of threads to search for paths (should be <= nPath)
#     eps0: Initial error reduction tolerance
#     eps1: Minimum error reduction tolerance
#        v: Verbose mode (0 off; 1 low; 2 high)
#--------------------------------------------------------------------------------
#      RET: Coefficients, intercept
#--------------------------------------------------------------------------------
def ICPSolve(A, Y, W, fMin=None, fMax=None, maxIter=200, mrg=1.0, b=None, dMax=0.1,
          nsd=1, xsd=0.5, CO=None, fg=None, maxGroup=0, CFx=None, tol=-1e-5,
          clip=1e-9, bAr=None, aAr=None, nPath=1, nThrd=1, eps0=-1e-5, eps1=-EPS, v=1):
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

   X  = b + A @ CV                            # Current solution

   feaSet = set()                             # Used feature groups set

   if CFx is None:
      CFx = ErrInc                            # Column constrain function

   f   = -1
   c   =  0                                   # Iteration count
   u   =  0.0                                 # Current update
   err =  np.inf                              # Current error
   mar = -np.inf                              # Moving average of error reduction

   nd  = CV.shape[0] << 1
   cp  = -1                                   # Column order pointer
   if CO is None:
      CO = np.arange(nd)                      # Column order

   # Minimum and max swap distances
   nsd = round(nsd * nd) if isinstance(nsd, float) else nsd
   xsd = round(xsd * nd) if isinstance(xsd, float) else xsd
   nsd = min(nsd, xsd)
   xsd = max(nsd, xsd)

   # Create search closures
   DSFx = [DistSearch(A, EPS, DERR_MAX, DEL) for _ in range(nThrd)]
   sArg = np.empty(nPath, dtype='object')

   with Parallel(n_jobs=nThrd, prefer='threads') as TP:
      while True:                                  # Loop for each algorithm iteration
         B   = S * (Y - X)                         # Distance to margin; <0 if correct
         err = (np.maximum(B, 0) * W).sum()        # Mean hinge error

         if v > 0:
            print('{:5d} {:10.7f} {:10.7f} [{:5d}] {:+8.4f}'.format(c, err, mar, f, u))
         # Check if error within tolerance
         if (mar > tol) or (c > maxIter):
            if v > 1: print('Reached stopping criteria')
            break
         c   += 1

         bErr = np.inf
         nl   = 0
         while nl < nd:                            # Loop until a direction is found
            npf    = 0                             # Number of paths found
            sArg[:] = None
            while (nl < nd) and (npf < nPath):     # Loop to find paths with slack
               cp   = (cp + 1) % nd
               f, d = ColToFD(CO[cp])

               # Find constraint along path
               vMax = CFx(CV, b, err, fMin, fMax, bAr, aAr, f, d, dMax)
               if vMax <= 0:
                  continue
               # Manually round-robin search closures due to memory re-use
               sArg[npf] = (A[:, f].astype(np.double), f, d, cp, vMax, DSFx[npf % nThrd])
               npf += 1
               nl  += 1

            if npf == 0:                           # No paths have slack
               break

            # Try all paths in parallel
            sRes = TP(delayed(SFxi)(Afi, Y, W, S, B, X, vMaxi, di)
                       for Afi, _, di, _, vMaxi, SFxi in sArg[:npf])
            # Take path with lowest (pathErr, pathDist)
            cIdx = min(range(npf), key=lambda i : sRes[i])
            cErr, cDst = sRes[cIdx]

            if cErr < bErr:         # Record best column seen so far
               Af, f, d, _, _, _ = sArg[cIdx]
               bErr = cErr
               bDst = cDst
               bCol = Af
               bFea = f
               bDir = d

            if (bDst >= EPS) and (bErr <= eps0):
               break                # Found a dir and dist that reduces error >= eps0

         if bErr > eps1:
            if v > 1: print('Algorithm Stalled')
            break                   # Cannot reduce error more than eps1; break

         while bErr > eps0:
            if v > 1: print('Lowering eps0: {:g} -> {:g}'.format(eps0, eps0 / 10.0))
            eps0 /= 10.0            # Adjust eps0 if necessary

         if (maxGroup > 0) and (fg[bFea] > 0):
            feaSet.add(fg[bFea])    # This col may use a new group; count towards limit

         u = bDir * bDst
         if np.abs(CV[bFea] + u) < clip:
            if v > 1: print('Clip: {:g} -> 0.0'.format(CV[bFea] + u))
            u = -CV[bFea]           # Quantize coefficients that are very close to 0

         CV[bFea] += u              # Update coeficient vector
         X        += u * bCol       # Update current solution

         mar = bErr if np.isinf(mar) else (0.01 * bErr + 0.99 * mar)

         # Update traversal plan by moving columns that reduce error ahead
         for (_, _, _, cpi, _, _), (pErr, _) in zip(sArg[:npf], sRes[:npf]):
            if pErr <= eps0:
               ColOrder(CO, cpi, nsd, xsd)

   return CV, b, err

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
   CV, b, err = ICPSolve(A, Y, W, **kwargs)

   if cCol is None:
      return CV, b, err

   if not hasattr(cCol, '__len__'):
      cCol = [cCol]

   b = b + CV[cCol].sum()

   cIdx = np.ones(A.shape[1], np.bool)
   cIdx[cCol] = False
   cIdx = cIdx.nonzero()[0]
   CV = CV[cIdx].copy()

   return CV, b, err

#--------------------------------------------------------------------------------
#   Desc: Get univariate rule error
#--------------------------------------------------------------------------------
#      A: The rule matrix
#      Y: The target margin values
#      W: Sample weights
#      d: The direction to move in
#--------------------------------------------------------------------------------
#    RET: Change in error in signed column direction
#--------------------------------------------------------------------------------
def RuleErr(A, Y, W=None, b=None, d=1):
   if W is None:
      W = np.full(A.shape[0], 1 / A.shape[0])
   if b is None:
      b = np.dot(Y, W)
   S = SignInt8(Y)
   B = S * (Y - b)
   # Split up for low-memory
   err = np.empty(A.shape[1])
   for i in range(A.shape[1]):
      Ai     = d * A[:, i]
      DSi    = SignInt8(Ai)
      aea    = np.abs(Ai * W) @ ((S != DSi) & (B >= 0))   # These increase error
      sea    = np.abs(Ai * W) @ ((S == DSi) & (B >  0))   # These decrease error
      err[i] = aea - sea
   # Amount each rule changes average target versus the base rate
   return err

#--------------------------------------------------------------------------------
#   Desc: Get univariate rule orientation
#--------------------------------------------------------------------------------
#      A: The rule matrix
#      Y: The target values {-1, 1}
#      W: Sample weights
#      b: The base rate (mean of target if None)
#--------------------------------------------------------------------------------
#    RET: Sign indicating the direction of each rule
#--------------------------------------------------------------------------------
def RuleSign(A, Y, W=None, b=None):
   if W is None:
      W = np.full(A.shape[0], 1 / A.shape[0])
   Y = Y * W
   if b is None:
      b = Y.sum()
   # Split up for low-memory
   nht = np.empty(A.shape[1])
   nhp = np.empty(A.shape[1])
   for i in range(A.shape[1]):
      Ai     = A[:, i]
      nht[i] = W @ Ai                 # Weighted pct hits total
      nhp[i] = Y @ Ai                 # Weight Y sum for samples with rule hits
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