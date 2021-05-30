import numpy     as     np
from   .Util     import GroupBy

# Operator codes
OP_LE = 0
OP_GT = 1
OP_EQ = 2
OP_LT = 3
OP_GE = 4

OP_STR = ['<=', '>', '==', '<', '>=']

#--------------------------------------------------------------------------------
# Evaluate rule
#--------------------------------------------------------------------------------
#   a: Feature value
#   b: Threshold value
#  op: Operator code
#   v: Coefficient value
#--------------------------------------------------------------------------------
# RET: Value of rule for given feature value
#--------------------------------------------------------------------------------
def CheckRule(a, op, b, v):
   if op == OP_LE:
      return v * (a <= b)
   if op == OP_GT:
      return v * (a >  b)
   if op == OP_EQ:
      return v * (a == b)
   if op == OP_LT:
      return v * (a <  b)
   if op == OP_GE:
      return v * (a >= b)
   return np.nan

#--------------------------------------------------------------------------------
# Apply all rules for a feature to a given value
#--------------------------------------------------------------------------------
#  TV: Threshold vector
#  OP: Operator vector
#  CV: Coefficient vectors
#   v: Coefficient value
#--------------------------------------------------------------------------------
# RET: Value of rule for given feature value
#--------------------------------------------------------------------------------
def CheckRules(TV, OP, CV, v):
   return sum(CheckRule(v, OPi, TVi, CVi) for TVi, OPi, CVi in zip(TV, OP, CV))

#--------------------------------------------------------------------------------
# Consolidate into non-overlapping rules
#--------------------------------------------------------------------------------
#  CV: Coefficient vectors
#  CI: Column indices
#  TV: Threshold vector
#  OP: Operator vector
#   g: Group rules on same feature with same coefficients
# eps: Rules with coefficient smaller than this are filtered; negative disables
#--------------------------------------------------------------------------------
# RET: Consolidated rules
#--------------------------------------------------------------------------------
def ConsolidateRules(CV, CI, TV, OP, g=True, eps=0.0):
   rl = []
   for fi in set(CI):  # Loop over unique column indices
      bndA, valA, orgA = zip(*ExtractCurve(CV, CI, TV, OP, fi))
      rli = []         # Criteria for current rule
      si  = 0          # Starting index
      cv  = valA[si]   # Current value
      i   = 1          # Current index
      while i < len(valA):
         if (i < (len(valA) - 1)) and (valA[i] == cv):
            i += 1
            continue
         ei = i

         lo = uo = OP_LT    # Try exclusive bounds by default
         if not (np.isinf(bndA[ei]) or orgA[ei]):
            ei -= 1        # Endpoint not original, go backwards and use inclusive
            uo  = OP_LE
         if not (np.isinf(bndA[si]) or orgA[si]):
            si += 1        # Start point not original, go forwards and use inclusive
            lo  = OP_LE

         if np.abs(cv) <= eps:        # Don't report ranges with coefficient 0
            pass
         elif bndA[si] == bndA[ei]:   # Start and end are same; use equality
            rli.append(((fi, OP_EQ, bndA[si]), cv))
         else:                        # Use an interval
            rli.append(((bndA[si], lo, fi, uo, bndA[ei]), cv))

         si  = ei
         cv  = valA[i]
         i  += 1

      # If g is set, group rules with same pred val and OR together
      rli, coef = zip(*rli)
      key = coef if g else range(len(rli))
      grp = GroupBy(range(len(rli)), key=key.__getitem__)

      rl.extend((tuple(rli[j] for j in g), k) for k, g in grp)

   return rl

#--------------------------------------------------------------------------------
# Checks consolidated rules
#--------------------------------------------------------------------------------
#    A: Data matrix
#   rl: List of rule tuples provided by ConsolidateRules
#--------------------------------------------------------------------------------
#  RET: List of (boolean vector, coefficient) tuples
#--------------------------------------------------------------------------------
def EvaluateRules(A, rl):
   rv = []
   for ri in rl:  # Loop over unique column indices
      cond, coef = ri

      ri = np.zeros(A.shape[0], np.int)
      for ci in cond:

         if  len(ci) == 3:
            fi, oi, ti = ci
            ri += CheckRule(A[:, fi], oi, ti, True)
         else:
            lt, lo, fi, uo, ut = ci

            Af = A[:, fi]
            c1 = CheckRule(lt, lo, Af, True)
            c2 = CheckRule(Af, uo, ut, True)

            ri += (c1 & c2)
      rv.append((ri > 0, coef))
   return rv

#--------------------------------------------------------------------------------
# Get data defining 1-d feature-response curve
#--------------------------------------------------------------------------------
#  CV: Coefficient vectors
#  CI: Column indices
#  TV: Threshold vector
#  OP: Operator vector
#  fi: Feature index
#--------------------------------------------------------------------------------
# RET: String representation of rule
#--------------------------------------------------------------------------------
def ExtractCurve(CV, CI, TV, OP, f):
   nzi = (CI == f).nonzero()[0]
   CVi = CV[nzi]
   TVi = TV[nzi]
   OPi = OP[nzi]

   # Unique also sorts thesholds. Add points at infinity
   intA = [-np.inf] + list(np.unique(TVi)) + [np.inf]
   crvA = []   # Array of (x, y, isOriginal)
   for i in range(len(intA) - 1):
      a = intA[i]
      b = intA[i + 1]
      # Obtain response at boundaries of and inside this interval
      crvA.append((a, CheckRules(TVi, OPi, CVi, a), not np.isinf(a)))
      # Check value inside of region; this is not an original point
      a1 = (b - 2) if np.isinf(a) else a
      b1 = (a + 2) if np.isinf(b) else b
      m  = a1 + (b1 - a1) * 0.01
      crvA.append((m, CheckRules(TVi, OPi, CVi, m), False))
   crvA.append((b, CheckRules(TVi, OPi, CVi, b), False))
   return crvA

#--------------------------------------------------------------------------------
# Rule to string
#--------------------------------------------------------------------------------
#   f: Feature name
#   o: Operator code
#   v: Threshold value
#--------------------------------------------------------------------------------
# RET: String representation of rule
#--------------------------------------------------------------------------------
def RuleStr(f, o, v):
   return '{0} {1} {2}'.format(f, OP_STR[o], v)

#--------------------------------------------------------------------------------
# Creates strings from rule tuples
#--------------------------------------------------------------------------------
#   rl: List of rule tuples provided by ConsolidateRules
#   FN: List of feature names
# ffmt: Float format string
#--------------------------------------------------------------------------------
#  RET: List of (rule string, coefficient) tuples
#--------------------------------------------------------------------------------
def RuleStrings(rl, FN, ffmt='{:.5f}'):
   rStr = []
   for ri in rl:  # Loop over unique column indices
      cond, coef = ri

      rli = []
      for ci in cond:
         rlij = []
         if  len(ci) == 3:
            fi, oi, ti = ci
            ti = ffmt.format(ti)
            rlij.append('({0} {1} {2})'.format(FN[fi], OP_STR[oi], ti))
         else:
            lt, lo, fi, uo, ut = ci

            if not np.isinf(lt):
               lt = ffmt.format(lt)
               rlij.append('({0} {1} {2})'.format(lt, OP_STR[lo], FN[fi]))

            if not np.isinf(ut):
               ut = ffmt.format(ut)
               rlij.append('({0} {1} {2})'.format(FN[fi], OP_STR[uo], ut))

         rli.append(' & '.join(rlij))
      cfmt = '({0})' if len(rli) > 1 else '{0}'
      rStr.append((' | '.join(cfmt.format(ci) for ci in rli), coef))
   return rStr