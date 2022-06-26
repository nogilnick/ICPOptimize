from   .Rules        import (ConsolidateRules, EvaluateRules, ExtractCurve, RuleStr,
                             OP_LE, OP_GT, RuleStrings)
import numpy         as     np
from   scipy.special import expit
from   .Solver       import EPS, ICPSolveConst, RuleErr, RuleOrder, RuleSign

# Default tree model parameters
DEF_PAR = {
   'gbc': dict(n_estimators=50, min_impurity_decrease=1e-8, max_depth=1),
   'xgc': dict(n_estimators=50, gamma=1e-7, max_depth=1),
   'rfc': dict(n_estimators=50, min_impurity_decrease=1e-8, max_depth=4),
   'xrt': dict(n_estimators=50, min_impurity_decrease=1e-8, max_depth=4)}

#--------------------------------------------------------------------------------
#   Desc: Construct problem constraints
#--------------------------------------------------------------------------------
#     RM: Data matrix
#      Y: Target labels (0/1) or (-1, 1)
#      W: Sample weights (must sum to 1)
#     fg: Feature groups
#      c: Include constant column (0/1)
#   cOrd: Column order constraints mode (n: None, r: Relative, a: Absolute)
#      b: Initial guess (weighted average margin target if None)
#     bs: Block size for calculating dot products (lower values use less memory)
#--------------------------------------------------------------------------------
#    RET: Problem constraints
#--------------------------------------------------------------------------------
def Constrain(RM, Y, W=None, fg=None, c=1, mrg=1.0, cOrd='r', b=None, bs=16000000):
   if W is None:
      W = np.full(Y.shape[0], 1 / Y.shape[0])      # Default to equal weights
   else:
      W = W / W.sum()                              # Force sum==1 for weighted avg

   YS = np.where(Y > 0,  np.int8(+1),  np.int8(-1))
   M  = np.where(Y > 0,          mrg,         -mrg)

   if b is None:
      b = np.dot(W, M)                             # Initial value

   rs   = RuleSign(RM, YS, W, b=b, bs=bs)          # Identify sign of rules
   fMin = np.where(rs > 0,    0., -np.inf)         # Rule sign constraints lower bounds
   fMax = np.where(rs > 0, np.inf,     0.)         # Rule sign constraints upper bounds

   ren  = RuleErr(RM, M, W, b=b, d=-1, bs=bs)      # Order columns by error change
   rep  = RuleErr(RM, M, W, b=b, d=+1, bs=bs)
   CO   = np.c_[ren, rep].argsort(axis=None)       # Even indices: d=-1, Odd: d=+1

   if c != 0:                                      # Use an intercept
      fMin[-1] = -np.inf                           # Intercept is unconstrained
      fMax[-1] =  np.inf
      cCol     = [RM.shape[1] - 1]
   else:
      cCol     = None

   # Make order constraints on coefficients by univariate rule performance
   bAr = aAr = None
   if cOrd in ('a', 'r'):
      bAr, aAr = RuleOrder(fg, rs, m=cOrd)

   return fMin, fMax, cCol, CO, bAr, aAr

#--------------------------------------------------------------------------------
#   Desc: Construct rule matrix
#--------------------------------------------------------------------------------
#      A: Data matrix
#     FA: Feature array of indices into A
#     TA: Threshold array with value paired with features in FA
#      c: Include constant column (0/1)
#--------------------------------------------------------------------------------
#    RET: Rule matrix
#--------------------------------------------------------------------------------
def ConstructRuleMatrix(A, FA=None, TA=None, c=0, o='F'):
   c  = int(c != 0)
   nf = A.shape[1] if FA is None else len(FA)
   RM = np.empty((A.shape[0], c + 2 * nf), dtype=np.bool, order=o)

   if (FA is not None) and (TA is not None):
      RM[:,   :    nf] = A[:, FA] <= TA      # Rules
   else:
      RM[:,   :    nf] = A                   # Rules

   RM[:, nf:2 * nf] = 1 - RM[:, :nf]         # Inverted rules

   if FA is None:
      FA = np.arange(nf)

   if c != 0:
      RM[:, 2 * nf] = np.bool(c)
      fg = np.concatenate((FA, FA, [-1]))    # Last column is intercept
   else:
      fg = np.concatenate((FA, FA))

   return RM, fg

#--------------------------------------------------------------------------------
#   Desc: Extract splits from a scikit-learn tree model
#--------------------------------------------------------------------------------
#     tm: Tree model
#--------------------------------------------------------------------------------
#    YLD: Iterable of (feature, split)
#--------------------------------------------------------------------------------
def ExtractSplits(tm):
   EL  = tm.estimators_
   if hasattr(EL, 'ravel'):
      EL = EL.ravel()

   fSet = set()
   for Ei in EL:
      for FAi, TAi in zip(Ei.tree_.feature, Ei.tree_.threshold):
         if FAi < 0:
            continue
         ti = (int(FAi), float(TAi))
         if ti in fSet:
            continue
         yield ti
         fSet.add(ti)

#--------------------------------------------------------------------------------
#   Desc: Extract splits from an XGBoost tree model
#--------------------------------------------------------------------------------
#     tm: Tree model
#--------------------------------------------------------------------------------
#    YLD: Iterable of (feature, split)
#--------------------------------------------------------------------------------
def ExtractSplitsXG(tm):
   DF = tm.get_booster().trees_to_dataframe()

   fSet = set()
   for FAi, TAi in zip(DF.Feature, DF.Split):
      if TAi != TAi:
         continue
      ti = (int(FAi[1:]), float(TAi))
      if ti in fSet:
         continue
      yield ti
      fSet.add(ti)

#--------------------------------------------------------------------------------
#   Desc: Sets tree model parameters given input
#--------------------------------------------------------------------------------
#     tm: Tree model string
#  tmPar: Tree model parameters
#--------------------------------------------------------------------------------
#    RET: Tree model constructor, split extractor function, tree model parameters
#--------------------------------------------------------------------------------
def GetModelParams(tm, tmPar):
      ESFx = ExtractSplits
      if   tm == 'gbc':
         from sklearn.ensemble import GradientBoostingClassifier
         cf   = GradientBoostingClassifier
      elif tm == 'rfc':
         from sklearn.ensemble import RandomForestClassifier
         cf   = RandomForestClassifier
      elif tm == 'xgc':
         from xgboost import XGBClassifier
         cf   = XGBClassifier
         ESFx = ExtractSplitsXG
      elif tm == 'xtc':
         from sklearn.ensemble import ExtraTreesClassifier
         cf   = ExtraTreesClassifier

      if tmPar is None:
         tmPar = DEF_PAR.get(tm, {})

      return cf, ESFx, tmPar

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
#      ig: Initial guess (weighted average margin target if None)
#      tm: Tree model constructor
#   tmPar: Extra parameters for tree model
#      bs: Block size for calculating dot products (lower values use less memory)
#     nsd: Minimum column swap distance
#     xsd: Maximum column swap distance
#    ESFx: Extract split functions
#     tol: Moving average error reduction tolerance
# maxFeat: Maximum number of original features that can be included in model
#     CFx: Criteria function for abandoning path (Path abandoned if CFx true)
#       c: Use constant (0/1)
#    clip: Clip coefs with magnitude less than this to exactly 0
#   nThrd: Number of threads to search for paths (should be <= nPath)
#    cOrd: Column order constraints mode (n: None, r: Relative, a: Absolute)
#    gThr: Number of succesful iterations required to grow eps0
#    eps0: Initial error reduction tolerance
#    eps1: Minimum error reduction tolerance
#       v: Verbosity (0: off; 1: low; 2: high)
#--------------------------------------------------------------------------------
class ICPRuleEnsemble:

   def __repr__(self):
      if not hasattr(self, 'CV'):
         return 'ICPRE()'
      return 'ICPRE({:d})'.format(self.CV.shape[0])

   def __str__(self):
      return self.__repr__()

   def __init__(self, lr=0.05, maxIter=1250, mrg=1.0, ig=None, tm='gbc', tmPar=None,
                bs=16000000, nsd=0, xsd=0.25, ESFx=ExtractSplits, tol=-5e-7, maxFeat=0,
                CFx=None, c=1, clip=1e-9, nThrd=1, cOrd='n', gThr=10, eps0=-1e-5,
                eps1=-EPS, v=0):
      self.lr      = lr
      self.maxIter = maxIter
      self.mrg     = mrg
      self.ig      = ig
      self.nsd     = nsd
      self.xsd     = xsd
      self.ESFx    = ESFx
      self.tm      = tm
      self.tmPar   = tmPar
      self.bs      = bs
      self.tol     = tol
      self.maxFeat = maxFeat
      self.CFx     = CFx
      self.c       = c
      self.clip    = clip
      self.nThrd   = nThrd
      self.cOrd    = cOrd
      self.gThr    = gThr
      self.eps0    = eps0
      self.eps1    = eps1
      self.v       = v

   def decision_function(self, A):
      return self.transform(A) @ self.CV + self.b

   # Given a data matrix and list of feature names, explain prediction for each sample
   def Explain(self, A, FN=None, ffmt='{:.5f}'):
      # Get consolidated rule strings without bias term; these align with self.CR
      rStr = self.GetRules(FN=FN, c=True, ffmt=ffmt, bias=False)

      il = [('bias', self.b)] if (self.b != 0.0) else []
      ex = [list(il) for _ in range(A.shape[0])]
      ER = EvaluateRules(A, self.CR)
      for j, (Rj, _) in enumerate(ER):
         nzi = Rj.nonzero()[0]
         for i in nzi:
            ex[i].append(rStr[j])   # Append rule string tuple for each hit sample
      return ex

   # Gets influence of each input feature on the final classification
   def feature_activation(self, A):
      B  = np.zeros_like(A)
      TM = self.transform(A)
      for f in set(self.FA):
         ci = (self.FA == f).nonzero()[0]
         B[:, f] = TM[:, ci] @ self.CV[ci]
      return B

   # Fit the model on data matrix A, target values Y, and sample weights W
   def fit(self, A, Y, W=None):
      self.classes_, Y = np.unique(Y, return_inverse=True)
      Y = Y.astype(np.int8)

      # Fit tree method (tm) on original data
      tm, ESFx, tmPar = GetModelParams(self.tm, self.tmPar)
      GBC = tm(**tmPar)
      GBC.fit(A, Y, W)

      # Extract unique rules using from the tree model
      FA, TA = zip(*ESFx(GBC))
      FA     = np.array(FA)
      TA     = np.array(TA)

      if self.v > 1:
         print('Extracted {:d} Rules.'.format(FA.shape[0]))

      # Construct rule matrix
      RM, fg = ConstructRuleMatrix(A, FA=FA, TA=TA, c=self.c)

      # Obtain problem constraints
      fMin, fMax, cCol, CO, bAr, aAr = Constrain(RM, Y, W=W, fg=fg, c=self.c,
                                     mrg=self.mrg, cOrd=self.cOrd, b=self.ig, bs=self.bs)

      # Obtain solution
      CV, b, _, self.nIter = \
        ICPSolveConst(RM, Y, W, cCol=cCol, fMin=fMin, fMax=fMax, fg=fg, mrg=self.mrg,
                      maxIter=self.maxIter, b=self.ig, CO=CO, tol=self.tol, CFx=self.CFx,
                      maxGroup=self.maxFeat, nsd=self.nsd, xsd=self.xsd, dMax=self.lr,
                      bAr=bAr, aAr=aAr, gThr=self.gThr, nThrd=self.nThrd, bs=self.bs,
                      clip=self.clip, eps0=self.eps0, eps1=self.eps1, v=self.v)

      nzi     = CV.nonzero()[0]                       # Identify non-zero coefs
      self.FA = fg[nzi].copy()                        # Rule feature index array
      self.TA = np.concatenate((TA, TA))[nzi].copy()  # Rule threshold value array
      self.OP = np.repeat([OP_LE, OP_GT], TA.shape[0])[nzi].copy() # Operator (<=|>)
      self.CV = CV[nzi].copy()                        # Reduced coefficient vector size
      self.b  = b                                     # Intercept term

      self.CR = ConsolidateRules(self.CV, self.FA, self.TA, self.OP, eps=EPS)

      # Feature importance as sum of magnitude of rule coefficients using feature
      self.feature_importances_ = np.zeros(A.shape[1])
      for fi, cvi in zip(self.FA, self.CV):
         self.feature_importances_[fi] += np.abs(cvi)

      return self

   # List of (x, y, isOriginal) where y is the change in response when feature f is x
   def GetResponseCurve(self, f):
      return ExtractCurve(self.CV, self.FA, self.TA, self.OP, f)

   def GetRules(self, FN=None, c=True, ffmt='{:.5f}', bias=True):
      rl = []
      if bias and (self.b != 0):
         rl.append(('True', self.b))

      n  = len(self.CV)
      if n == 0:
         return rl

      if FN is None:
         FN = ['F{0}'.format(i) for i in range(self.FA.max() + 1)]

      if not c:
         rl.extend((RuleStr(FN[self.FA[i]], self.OP[i], self.TA[i]), self.CV[i])
                    for i in range(n))
      else:
         rl.extend(RuleStrings(self.CR, FN, ffmt=ffmt))

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