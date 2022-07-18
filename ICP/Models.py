from   .Binning      import KBDisc
from   .Rules        import (ConsolidateRules, EvaluateRules, ExtractCurve, RuleStr,
                             OP_LE, OP_GT, RuleStrings, GetRegionPoints, RuleSign,
                             RuleOrder)
import numpy         as     np
from   scipy.special import expit
from   .Solver       import EPS, CnstPath, GetObjFx, HINGE, LSTSQ, ICPSolveConst, RuleErr
from   .Util         import ColCorr, GetCol, MakeWeights, ToArray

# Default tree model parameters
DEF_PAR = {
   'gbc': dict(n_estimators=50, min_impurity_decrease=1e-8, max_depth=2),
   'xgc': dict(n_estimators=50, gamma=1e-7, max_depth=2),
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
def Constrain(RM, Y, W=None, fg=None, c=1, mrg=1.0, cOrd='r', b=None, bs=1.6e7, t=None):
   m, n = RM.shape
   W    = MakeWeights(W, m)

   # Evaluate columns for contraining and ordering
   rs = ColCorr(RM, Y, W, c=c) if t == 'corr' else RuleSign(RM, Y, W, bs=bs, c=c)

   fMin = np.where(rs > 0,    0., -np.inf)         # Rule sign constraints lower bounds
   fMax = np.where(rs > 0, np.inf,     0.)         # Rule sign constraints upper bounds

   M    = np.where(Y > 0, mrg, -mrg)               # Sort rules by their gradient
   b    = np.dot(W, M) if b is None else b         # Initial guess / bias
   ren  = RuleErr(RM, M, W, b=b, d=-1, bs=bs, c=c)
   rep  = RuleErr(RM, M, W, b=b, d=+1, bs=bs, c=c)
   CO   = np.c_[ren, rep].argsort(axis=None)       # Even indices: d=-1, Odd: d=+1

   if c != 0:                                      # Use an intercept
      fMin[-1] = -np.inf                           # Intercept is unconstrained
      fMax[-1] =  np.inf

   # Make order constraints on coefficients by univariate rule performance
   fg  = np.arange(n + (c != 0)) if fg is None else fg  # Default each col to unique group
   bAr = aAr = None
   if cOrd in ('a', 'r'):
      bAr, aAr = RuleOrder(fg, rs, m=cOrd)

   return fMin, fMax, CO, bAr, aAr

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
def ConstructRuleMatrix(A, FA=None, TA=None, c=0, o='F', dtype='int8'):
   c  = int(c != 0)
   nf = A.shape[1] if FA is None else len(FA)
   RM = np.empty((A.shape[0], 2 * nf), dtype=dtype, order=o)

   if (FA is not None) and (TA is not None):
      RM[:,   :    nf] = GetCol(A, FA) <= TA  # Compute rule matrix
   else:
      RM[:,   :    nf] = A                    # Use A as rule-matrix directly

   RM[:, nf:2 * nf] = 1 - RM[:, :nf]          # Compliment rules

   # Feature index array
   FA = np.arange(nf) if FA is None else FA

   # Last column is intercept and is handled internally by algorithm
   fg = np.concatenate((FA, FA, [-1] * (c != 0)))

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
         fSet.add(ti)
         yield ti

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
      fSet.add(ti)
      yield ti

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
#    Iterative Constrained Pathways Base Estimator
#--------------------------------------------------------------------------------
class ICPBase:

   def __repr__(self):
      return '{}({})'.format(self.__class__.__name__, len(self))

   def __len__(self):
      return len(getattr(self, 'CV', []))

   def __str__(self):
      return self.__repr__()

   def decision_function(self, A):
      return ToArray(self.transform(A) @ self.CV + self.b)

   # Gets influence of each input feature on the final classification
   def feature_activation(self, A):
      return self.transform(A) * self.CV

   def predict(self, A):
      return ToArray(self.transform(A) @ self.CV + self.b)

   def score(self, A, Y, W=None):  # Weighted error computation
      Y = self.to_targets(Y)
      return self.errFx(self.decision_function(A), Y, MakeWeights(W, A.shape[0]))

   def transform(self, A):
      return A

   def to_targets(self, Y):
      return Y

#--------------------------------------------------------------------------------
#    Iterative Constrained Pathways Base Classifier
#--------------------------------------------------------------------------------
class ICPBaseClassifier(ICPBase):

   def predict(self, A):
      return self.classes_[(self.decision_function(A) >= 0).astype(np.int)]

   def predict_proba(self, A):
      return expit(self.decision_function(A))

   def to_targets(self, Y):
      return np.where(Y > 0, self.mrg, -self.mrg)

#--------------------------------------------------------------------------------
#    Iterative Constrained Pathways Rule Ensemble (ICPRE)
#    Desc: The approach of this algorithm is:
#           1. Extract rules from the original data using tree methods
#           2. Identify the "intuitive" direction of each rule
#           3. Solve a sign constrained hinge loss problem
#           4. Split potentially overlapping rules into non-overlapping rules
#--------------------------------------------------------------------------------
#      lr: Learning rate
#    norm: Normalize lr by column norms (T/F)
# maxIter: Maximum number of solver iterations
#     mrg: Target classification margin
#      ig: Initial guess (weighted average margin target if None)
#      tm: Tree model constructor
#   tmPar: Extra parameters for tree model
#      bs: Block size for calculating dot products (lower values use less memory)
#    ESFx: Extract split functions
#     tol: Moving average error reduction tolerance
# maxFeat: Maximum number of original features that can be included in model
#    cnst: Tuple for explicit specification of problem constraints
#     CFx: Criteria function for abandoning path (Path abandoned if CFx true)
#       c: Use constant (0/1)
#    clip: Clip coefs with magnitude less than this to exactly 0
#   nJump: Number of times to try and jump out of stall
#   nThrd: Number of threads to search for paths (should be <= nPath)
#    cOrd: Column order constraints mode (n: None, r: Relative, a: Absolute)
#       v: Verbosity (0: off; 1: low; 2: high)
#--------------------------------------------------------------------------------
class ICPRuleEnsemble(ICPBaseClassifier):

   def __init__(self, lr=0.5, norm=True, maxIter=3000, mrg=1.0, ig=None, tm='gbc',
                tmPar=None, bs=1.6e7, ESFx=ExtractSplits, tol=-5e-7, maxFeat=0, cnst=None,
                CFx=CnstPath, c=1, clip=1e-10, nJump=0, nThrd=1, cOrd='n', v=0):
      self.lr      = lr
      self.norm    = norm
      self.maxIter = maxIter
      self.mrg     = mrg
      self.ig      = ig
      self.ESFx    = ESFx
      self.tm      = tm
      self.tmPar   = tmPar
      self.bs      = bs
      self.tol     = tol
      self.maxFeat = maxFeat
      self.cnst    = cnst
      self.CFx     = CFx
      self.c       = c
      self.clip    = clip
      self.nJump   = nJump
      self.nThrd   = nThrd
      self.cOrd    = cOrd
      self.v       = v

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
      W = MakeWeights(W, A.shape[0])

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
      fMin, fMax, CO, bAr, aAr = Constrain(RM, Y, W=W, fg=fg, c=self.c, mrg=self.mrg,
         cOrd=self.cOrd, b=self.ig, bs=self.bs) if self.cnst is None else self.cnst

      # Obtain solution
      CV, b, self.err, self.nIter = ICPSolveConst(
         RM, Y, W, fMin=fMin, fMax=fMax, fg=fg, mrg=self.mrg, maxIter=self.maxIter,
         b=self.ig, c=self.c, CO=CO, obj=HINGE, tol=self.tol, CFx=self.CFx,
         maxGroup=self.maxFeat, dMax=self.lr, norm=self.norm, bAr=bAr, aAr=aAr,
         nJump=self.nJump, nThrd=self.nThrd, clip=self.clip, v=self.v)

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

      # Error function
      _, self.errFx = GetObjFx(HINGE)

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
      RM = (GetCol(A, self.FA) <= self.TA).astype(np.uint8)
      RM[:, self.OP > 0] = 1 - RM[:, self.OP > 0]
      return RM

#--------------------------------------------------------------------------------
#    Iterative Constrained Pathways Linear Classifier (ICPLC)
#    Desc: The approach of this algorithm is:
#           1. Identify the "intuitive" direction of each column
#           2. Solve a sign constrained hinge loss problem
#--------------------------------------------------------------------------------
#      lr: Learning rate
#    norm: Normalize lr by column norms (T/F)
# maxIter: Maximum number of solver iterations
#     mrg: Target classification margin
#      ig: Initial guess (weighted average margin target if None)
#      bs: Block size for calculating dot products (lower values use less memory)
#     tol: Moving average error reduction tolerance
# maxFeat: Maximum number of original features that can be included in model
#     CFx: Criteria function for abandoning path (Path abandoned if CFx true)
#       c: Use constant (0/1)
#    clip: Clip coefs with magnitude less than this to exactly 0
#    cnst: Tuple for explicit specification of problem constraints
#   nJump: Number of times to try and jump out of stall
#   nThrd: Number of threads to search for paths (should be <= nPath)
#    cOrd: Column order constraints mode (n: None, r: Relative, a: Absolute)
#       v: Verbosity (0: off; 1: low; 2: high)
#--------------------------------------------------------------------------------
class ICPLinearClassifier(ICPBaseClassifier):

   def __init__(self, lr=0.5, norm=True, maxIter=3000, mrg=1.0, ig=None, bs=1.6e7,
                tol=-5e-7, maxFeat=0, CFx=CnstPath, c=1, clip=1e-10, cnst=None, nJump=0,
                nThrd=1, cOrd='n', v=0):
      self.lr      = lr
      self.norm    = norm
      self.maxIter = maxIter
      self.mrg     = mrg
      self.ig      = ig
      self.bs      = bs
      self.tol     = tol
      self.maxFeat = maxFeat
      self.cnst    = cnst
      self.CFx     = CFx
      self.c       = c
      self.clip    = clip
      self.nJump   = nJump
      self.nThrd   = nThrd
      self.cOrd    = cOrd
      self.v       = v

   # Fit the model on data matrix A, target values Y, and sample weights W
   def fit(self, A, Y, W=None):
      self.classes_, Y = np.unique(Y, return_inverse=True)
      Y    = Y.astype(np.int8)
      m, n = A.shape
      W    = MakeWeights(W, m)

      # Each column in unique group
      fg = np.arange(n + (self.c != 0))

      # Obtain problem constraints
      fMin, fMax, CO, bAr, aAr = Constrain(A, Y, W=W, fg=fg, c=self.c, mrg=self.mrg,
       cOrd=self.cOrd, b=self.ig, bs=self.bs, t='corr') if self.cnst is None else self.cnst

      # Obtain solution
      self.CV, self.b, self.err, self.nIter = ICPSolveConst(
         A, Y, W, fMin=fMin, fMax=fMax, fg=fg, mrg=self.mrg, maxIter=self.maxIter,
         b=self.ig, c=self.c, CO=CO, tol=self.tol, CFx=self.CFx, maxGroup=self.maxFeat,
         obj=HINGE, dMax=self.lr, norm=self.norm, bAr=bAr, aAr=aAr, nJump=self.nJump,
         nThrd=self.nThrd, clip=self.clip, v=self.v)

      # Feature importance as sum of magnitude of rule coefficients using feature
      self.feature_importances_ = np.abs(self.CV)

      # Error function
      _, self.errFx = GetObjFx(HINGE)

      return self

#--------------------------------------------------------------------------------
#    Iterative Constrained Pathways Linear Regressor (ICPLR)
#    Desc: The approach of this algorithm is:
#           1. Identify the "intuitive" direction of each column
#           2. Solve a sign constrained least-squares problem
#--------------------------------------------------------------------------------
#      lr: Learning rate
#    norm: Normalize lr by column norms (T/F)
# maxIter: Maximum number of solver iterations
#      ig: Initial guess (weighted average margin target if None)
#      bs: Block size for calculating dot products (lower values use less memory)
#     tol: Moving average error reduction tolerance
# maxFeat: Maximum number of original features that can be included in model
#     CFx: Criteria function for abandoning path (Path abandoned if CFx true)
#       c: Use constant (0/1)
#    clip: Clip coefs with magnitude less than this to exactly 0
#    cnst: Tuple for explicit specification of problem constraints
#   nJump: Number of times to try and jump out of stall
#   nThrd: Number of threads to search for paths (should be <= nPath)
#    cOrd: Column order constraints mode (n: None, r: Relative, a: Absolute)
#       v: Verbosity (0: off; 1: low; 2: high)
#--------------------------------------------------------------------------------
class ICPLinearRegressor(ICPBase):

   def __init__(self, lr=0.5, norm=True, maxIter=3000, ig=None, bs=1.6e7, tol=-5e-7,
                maxFeat=0, CFx=CnstPath, c=1, clip=1e-10, cnst=None, nJump=0, nThrd=1,
                cOrd='n', v=0):
      self.lr      = lr
      self.norm    = norm
      self.maxIter = maxIter
      self.ig      = ig
      self.bs      = bs
      self.tol     = tol
      self.maxFeat = maxFeat
      self.cnst    = cnst
      self.CFx     = CFx
      self.c       = c
      self.clip    = clip
      self.nJump   = nJump
      self.nThrd   = nThrd
      self.cOrd    = cOrd
      self.v       = v

   # Fit the model on data matrix A, target values Y, and sample weights W
   def fit(self, A, Y, W=None):
      m, n = A.shape
      W    = MakeWeights(W, m)

      # Each column in unique group
      fg = np.arange(n + (self.c != 0))

      # Obtain problem constraints
      fMin, fMax, CO, bAr, aAr = Constrain(A, Y, W=W, fg=fg, c=self.c, cOrd=self.cOrd,
          b=self.ig, bs=self.bs, t='corr') if self.cnst is None else self.cnst

      # Obtain solution
      self.CV, self.b, self.err, self.nIter = ICPSolveConst(
         A, Y, W, fMin=fMin, fMax=fMax, fg=fg, maxIter=self.maxIter,
         b=self.ig, c=self.c, CO=CO, tol=self.tol, CFx=self.CFx, maxGroup=self.maxFeat,
         obj=LSTSQ, dMax=self.lr, norm=self.norm, bAr=bAr, aAr=aAr, nJump=self.nJump,
         nThrd=self.nThrd, clip=self.clip, v=self.v)

      # Feature importance as sum of magnitude of rule coefficients using feature
      self.feature_importances_ = np.abs(self.CV)

      # Error function
      _, self.errFx = GetObjFx(LSTSQ)

      return self

#--------------------------------------------------------------------------------
#    ICP Binning Classifier (ICPBC)
#    Desc: The approach of this algorithm is:
#           1. Bin and 1-hot encode original features
#           2. Identify the "intuitive" direction of each feature
#           3. Solve a sign constrained hinge loss problem
#--------------------------------------------------------------------------------
#      lr: Learning rate
#    norm: Normalize lr by column norms (T/F)
# maxIter: Maximum number of solver iterations
#     mrg: Target classification margin
#      ig: Initial guess (weighted average margin target if None)
#   kbPar: Extra parameters for the binning transformer
#      bs: Block size for calculating dot products (lower values use less memory)
#    ESFx: Extract split functions
#     tol: Moving average error reduction tolerance
# maxFeat: Maximum number of original features that can be included in model
#     CFx: Criteria function for abandoning path (Path abandoned if CFx true)
#       c: Use constant (0/1)
#    cnst: Tuple for explicit specification of problem constraints
#    clip: Clip coefs with magnitude less than this to exactly 0
#   nJump: Number of times to try and jump out of stall
#   nThrd: Number of threads to search for paths (should be <= nPath)
#    cOrd: Column order constraints mode (n: None, r: Relative, a: Absolute)
#       v: Verbosity (0: off; 1: low; 2: high)
#--------------------------------------------------------------------------------
class ICPBinningClassifier(ICPBaseClassifier):

   def __init__(self, lr=0.5, norm=True, maxIter=3000, mrg=1.0, ig=None, kbPar=None,
                bs=1.6e7, tol=-5e-7, maxFeat=0, CFx=CnstPath, c=1, cnst=None, clip=1e-10,
                nThrd=1, cOrd='n', nJump=0, v=0):
      self.lr      = lr
      self.norm    = norm
      self.maxIter = maxIter
      self.mrg     = mrg
      self.ig      = ig
      self.kbPar   = dict(n_bins=12, const=True) if kbPar is None else kbPar
      self.bs      = bs
      self.tol     = tol
      self.maxFeat = maxFeat
      self.cnst    = cnst
      self.CFx     = CFx
      self.c       = c
      self.clip    = clip
      self.nJump   = nJump
      self.nThrd   = nThrd
      self.cOrd    = cOrd
      self.v       = v

   # Given a data matrix and list of feature names, explain prediction for each sample
   def Explain(self, A, FN=None, ffmt='{:.5f}'):
      TA   = self.KBD.transform(A, ohe=False)
      m    = A.shape[0]

      fStr = '({:s} < {{}} <= {:s})'.format(ffmt, ffmt)

      il = [('bias', self.b)] if (self.b != 0.0) else []
      ex = [list(il) for _ in range(m)]
      for j in range(m):
         for f, bj in enumerate(TA[j]):
            if self.LA[f][bj] == 0:
               continue
            rStr = fStr.format(self.TA[f][bj], FN[f], self.TA[f][bj + 1])
            # Append rule string tuple for each hit sample
            ex[j].append((rStr, self.LA[f][bj]))
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
      Y    = Y.astype(np.int8)
      m, n = A.shape
      W    = MakeWeights(W, m)

      # Fit tree method (tm) on original data
      self.KBD = KBDisc(**self.kbPar)
      self.KBD.const = hc = self.c != 0

      self.KBD.fit(A)
      fg      = np.repeat(np.arange(n + hc), self.KBD.nBins)
      self.FA = fg[:-1 if hc else None]

      # Handle const column by solver algorithm internally
      self.KBD.RemoveConst()
      RM = self.KBD.transform(A)

      if self.v > 1:
         print('Extracted {:d} Rules.'.format(RM.shape[1]))

      # List of bin boundaries for each feature
      self.TA = []
      for f in range(n):
         self.TA.append(np.array([-np.inf, *self.KBD.bEdge[f], np.inf]))

      # Obtain problem constraints
      fMin, fMax, CO, bAr, aAr = Constrain(RM, Y, W=W, fg=fg, c=self.c, mrg=self.mrg,
         cOrd=self.cOrd, b=self.ig, bs=self.bs) if self.cnst is None else self.cnst

      # Obtain solution
      self.CV, self.b, self.err, self.nIter = ICPSolveConst(
         RM, Y, W, fMin=fMin, fMax=fMax, fg=fg, mrg=self.mrg, maxIter=self.maxIter,
         b=self.ig, c=self.c, CO=CO, tol=self.tol, CFx=self.CFx, maxGroup=self.maxFeat,
         obj=HINGE, dMax=self.lr, norm=self.norm, bAr=bAr, aAr=aAr, nJump=self.nJump,
         nThrd=self.nThrd, clip=self.clip, v=self.v)

      # Feature importance as sum of magnitude of rule coefficients using feature
      self.feature_importances_ = np.zeros(A.shape[1])
      for fi, cvi in zip(self.FA, self.CV):
         self.feature_importances_[fi] += np.abs(cvi)

      # List of levels for each bin
      self.LA = []
      for f in range(n):
         self.LA.append(self.CV[self.FA == f])

      # Error function
      _, self.errFx = GetObjFx(HINGE)

      return self

   # List of (x, y, isOriginal) where y is the change in response when feature f is x
   def GetResponseCurve(self, f):
      rp = GetRegionPoints(self.TA[f])
      cf = self.CV[self.FA == f]
      return zip(rp, np.repeat(cf, len(rp) // len(cf)), [True for _ in rp])

   def GetRules(self, FN=None, ffmt='{:.5f}', bias=True, sz=True):
      fStr = '({} < {{}} <= {})'.format(ffmt, ffmt)
      rl = []
      for f in range(len(self.LA)):
         for j in range(len(self.LA[f])):
            if sz and (self.LA[f][j] == 0):
               continue
            rStr = fStr.format(self.TA[f][j], FN[f], self.TA[f][j + 1])
            # Append rule string tuple for each hit sample
            rl.append((rStr, self.LA[f][j]))
      return rl

   def transform(self, A):
      return self.KBD.transform(A)
