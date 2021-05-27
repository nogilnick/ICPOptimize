from   .Consolidate     import ConsolidateRules, RuleStr, OP_LE, OP_GT
import numpy            as     np
from   scipy.special    import expit
from   .Solver          import ICPSolveConst, RuleErr, RuleOrder, RuleSign

# Default tree model parameters
DEF_PAR = dict(n_estimators=50, min_impurity_decrease=1e-7, n_iter_no_change=3,
               tol=1e-5, max_depth=1)

#--------------------------------------------------------------------------------
#   Desc: Construct rule matrix
#--------------------------------------------------------------------------------
#      A: Data matrix
#     FA: Feature array of indices into A
#     TA: Threshold array with value paired with features in FA
#      c: Include constant column (0/1)
#--------------------------------------------------------------------------------
def ConstructRuleMatrix(A, FA, TA, c=0):
   c  = int(c != 0)
   nf = FA.shape[0]
   RM = np.empty((A.shape[0], c + 2 * nf), dtype=np.bool)
   
   RM[:,   :   nf]  = A[:, FA] <= TA      # Rules
   RM[:, nf:2 * nf] = 1 - RM[:, :nf]      # Inverted rules  

   if c != 0:
      RM[:, 2 * nf] = np.bool(c)

   return RM      

#--------------------------------------------------------------------------------
#   Desc: Extract splits from tree model
#--------------------------------------------------------------------------------
#     tm: Tree model
#--------------------------------------------------------------------------------
def ExtractSplits(tm):
   EL  = tm.estimators_
   if hasattr(EL, 'ravel'):
      EL = EL.ravel()

   fSet = set()
   FTA  = []
   for Ei in EL:
      for FAi, TAi in zip(Ei.tree_.feature, Ei.tree_.threshold):
         if FAi < 0:
            continue
         ti = (int(FAi), float(TAi))
         if ti in fSet:
            continue
         FTA.append(ti)
         fSet.add(ti)
   return FTA

#--------------------------------------------------------------------------------
#    Iterative Constrained Pathways Rule Ensemble (ICPRE)
#    Desc: The approach of this algorithm is:
#           1. Extract rules from the original data using tree methods
#           2. Identify the "intuitive" direction of each rule
#           3. Solve a sign constrained hinge loss problem
#           4. Split potentially overlapping rules into non-overlapping rules
#--------------------------------------------------------------------------------
#      lr: Learning rate
# maxIter: Maximum number of solver iterations
#     mrg: Target classification margin
#      ig: Initial guess (calculated log-odds of class 1 if None)
#      tm: Tree model constructor
#   tmPar: Extra parameters for tree model
#     nsd: Minimum column swap distance
#     xsd: Maximum column swap distance
#    ESFx: Extract split functions
#     tol: Solver error tolerance
# maxFeat: Maximum number of original features that can be included in model
#     CFx: Criteria function for abandoning path (Path abandoned if CFx true)
#       c: Use constant (0/1)
#   nPath: Number of paths to explore at each step (best found is used)
#   nThrd: Number of threads to search for paths (should be <= nPath)
#    cOrd: Column order constraints mode (n: None, r: Relative, a: Absolute)
#       v: Verbosity (0: off; 1: low; 2: high)
#--------------------------------------------------------------------------------
class ICPRuleEnsemble:
   
   def __repr__(self):
      if not hasattr(self, 'CV'):
         return 'ICPRE()'
      return 'ICPRE({:d})'.format(self.CV.shape[0])
   
   def __str__(self):
      return self.__repr__()
   
   def __init__(self, lr=0.1, maxIter=200, mrg=1.0, ig=None, tm='gbc', tmPar=None,
                nsd=1, xsd=0.5, ESFx=ExtractSplits, tol=1e-2, maxFeat=0, CFx=None, c=1,
                nPath=1, nThrd=1, cOrd='n', v=0):
      self.lr      = lr
      self.maxIter = maxIter
      self.mrg     = mrg
      self.ig      = ig
      self.nsd     = nsd
      self.xsd     = xsd
      self.ESFx    = ESFx
      self.tol     = tol
      self.maxFeat = maxFeat
      self.CFx     = CFx
      self.c       = c
      self.nPath   = nPath
      self.nThrd   = nThrd
      self.cOrd    = cOrd
      self.v       = v
      
      self.tm = tm
      if tm == 'gbc':
         from sklearn.ensemble import GradientBoostingClassifier
         self.tm = GradientBoostingClassifier
      
      self.tmPar = DEF_PAR.copy()
      if tmPar is not None:
         self.tmPar.update(tmPar)

   def decision_function(self, A):
      return self.transform(A) @ self.CV + self.b

   # Fit the model on data matrix A, target values Y, and sample weights W
   def fit(self, A, Y, W=None):
      self.classes_, Y = np.unique(Y, return_inverse=True)
      Y = np.where(Y > 0, self.mrg, -self.mrg)
      W = np.full(Y.shape[0], 1 / Y.shape[0]) if W is None else (W.ravel() / W.sum())
      
      # Initial value
      b = np.dot(Y, W)
      
      # Fit tree method (tm) on original data
      GBC = self.tm(**self.tmPar)
      GBC.fit(A, Y, W)
      
      # Extract unique rules using from the tree model
      FA, TA = zip(*self.ESFx(GBC))
      FA     = np.array(FA)
      TA     = np.array(TA)
      
      # Construct rule matrix
      RM   = ConstructRuleMatrix(A, FA, TA, c=self.c)
      
      # Identify sign of rules and constraints
      rs   = RuleSign(RM, Y, W, b=b)
      fMin = np.where(rs > 0,    0., -np.inf)         # Rule sign constraints
      fMax = np.where(rs > 0, np.inf,     0.)                    
      
      # Order columns by amount of error change in that direction
      ren  = RuleErr(RM, Y, W, b=b, d=-1)
      rep  = RuleErr(RM, Y, W, b=b, d=+1)
      CO   = np.c_[ren, rep].argsort(axis=None)
      
      if self.c != 0:                                 # Use an intercept                         
         fMin[-1] = -np.inf                           # Intercept is unconstrained
         fMax[-1] =  np.inf
         fg       = np.concatenate((FA, FA, [-1]))    # Last column is intercept
         cCol     = [RM.shape[1] - 1]
      else:
         fg       = np.concatenate((FA, FA))
         cCol     = None
      
      # Make order constraints on coefficients by univariate rule peformance
      bAr = aAr = None
      if self.cOrd in ('a', 'r'):
         bAr, aAr = RuleOrder(fg, rs)
      
      CV, b, _ = ICPSolveConst(RM, Y, W, cCol=cCol, fMin=fMin, fMax=fMax, fg=fg,
                               maxIter=self.maxIter, mrg=self.mrg, b=self.ig, CO=CO,
                               tol=self.tol, maxGroup=self.maxFeat, CFx=self.CFx,
                               nsd=self.nsd, xsd=self.xsd, dMax=self.lr, bAr=bAr, aAr=aAr,
                               nPath=self.nPath, nThrd=self.nThrd, v=self.v)
      
      nzi     = CV.nonzero()[0]                       # Identify non-zero coefs
      self.FA = fg[nzi].copy()                        # Rule feature index array
      self.TA = np.concatenate((TA, TA))[nzi].copy()  # Rule threshold value array
      self.OP = np.repeat([OP_LE, OP_GT], TA.shape[0])[nzi].copy() # Operator (<=|>)
      self.CV = CV[nzi].copy()                        # Reduced coefficient vector size
      self.b  = b                                     # Intercept term
      
      return self
   
   def GetRules(self, FN=None, c=True):
      rl = []
      if self.b != 0:
         rl.append(('True', self.b))
         
      n  = len(self.CV)
      if n == 0:
         return rl
      
      FN = ['F{0}'.format(i) for i in range(self.FA.max() + 1)] if FN is None else FN
      
      if not c:
         rl.extend((RuleStr(FN[self.FA[i]], self.OP[i], self.TA[i]), self.CV[i])
                    for i in range(n))
      else:
         rl.extend(ConsolidateRules(self.CV, self.FA, self.TA, self.OP, FN))
         
      return rl
   
   def transform(self, A):
      RM = (A[:, self.FA] <= self.TA).astype(np.uint8)
      RM[:, self.OP > 0] = 1 - RM[:, self.OP > 0]
      return RM
   
   def predict_proba(self, A):
      Y       = np.zeros((A.shape[0], 2))
      Y[:, 1] = expit(self.decision_function(A))
      Y[:, 0] = 1 - Y[:, 1]
      return Y
   
   def predict(self, A):
      return self.classes_[(self.decision_function(A) >= 0).astype(np.int)]
   
   def score(self, A, Y):
      return (Y == self.predict(A)).mean()