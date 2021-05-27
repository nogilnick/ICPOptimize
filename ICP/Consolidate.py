from   itertools import groupby
import numpy     as     np

# Operator codes
OP_LE = 0
OP_GT = 1

#--------------------------------------------------------------------------------
# Evaluate rule
#--------------------------------------------------------------------------------
#   a: Feature value
#   b: Threshold value
#  op: Operator code
#   b: Coefficient value
#--------------------------------------------------------------------------------
# RET: String representation of rule
#--------------------------------------------------------------------------------
def CheckRule(a, b, op, c):
    if op == OP_LE:
        return c * (a <= b)
    if op == OP_GT:
        return c * (a > b)
    return np.nan

#--------------------------------------------------------------------------------
# Consolidate into non-overlaping rules
#--------------------------------------------------------------------------------
#  CV: Coefficient vectors
#  CI: Column indices
#  TV: Threshold vector
#  OP: Operator
#  FN: Feature names
#--------------------------------------------------------------------------------
# RET: Consolidated rules
#--------------------------------------------------------------------------------
def ConsolidateRules(CV, CI, TV, OP, FN, g=True):
    rl = []
    uf = set(CI)
    for fi in uf:   # Loop over unique column indices
        nzi = (CI == fi).nonzero()[0]
        CVi = CV[nzi]
        TVi = TV[nzi]
        OPi = OP[nzi]
        
        # Identify intervals in which prediction value stays the same
        inta = [-np.inf] + list(np.unique(TVi)) + [np.inf]
        bndA = []       # Boundary array
        orgA = []       # Array marking original vs added values
        for i in range(len(inta) - 1):
            a = inta[i]
            b = inta[i + 1]
            
            a1 = (b - 2) if np.isinf(a) else a
            b1 = (a + 2) if np.isinf(b) else b
            
            m = (a1 + b1) / 2               # Check value in middle of region
            bndA.append(a)
            orgA.append(not np.isinf(a))    # Add point at infinity at ends
            bndA.append(m)
            orgA.append(False)              # Midpoint is not an original point
        bndA.append(b)
        orgA.append(False)
        # Value of predictions for each boundary point
        valA = [
         sum(CheckRule(vi, TVij, OPij, CVij) for TVij, OPij, CVij in zip(TVi, OPi, CVi))
                for vi in bndA]
        
        rli = []            # Criteria for current rule
        si  = 0             # Starting index
        cv  = valA[si]      # Current value
        i   = 1             # Current index
        while i < len(valA):
            if (i < (len(valA) - 1)) and (valA[i] == cv):
                i += 1
                continue
            ei = i

            # If lower/upper boundary is infinite; then don't show it
            shl = not np.isinf(bndA[si])
            shu = not np.isinf(bndA[ei])
    
            lo = uo = '<'       # Try exclusive bounds by default
            if not orgA[ei]:    # Endpoint not original, go backwards and use inclusive
                ei -= 1
                uo  = '<='
            if not orgA[si]:    # Start point not original, go forwards and use inclusive
                si += 1
                lo  = '<='
    
            if bndA[si] == bndA[ei]:    # Start and end are same; use equality
                rli.append(('({0} == {1})'.format(FN[fi], bndA[si]), cv))
            else:                       # Use an interval
                fmt = '{:s}'
                rls = []
                if shl:                 # Show lower condition
                    rls.append('({0} {1} {2})'.format(bndA[si], lo, FN[fi]))
                if shu:                 # Show upper condition
                    rls.append('({0} {1} {2})'.format(FN[fi], uo, bndA[ei]))
                if len(rls) > 1:        # AND conditions together in ()
                    fmt = '({:s})'
                rli.append((fmt.format(' & '.join(rls)), cv))
            
            si  = ei
            cv  = valA[i]
            i  += 1

        if g:   # Group rules with same pred val and OR together
            for cvi, rli in GroupBy(rli, lambda x : x[1]):
                rl.append((' | '.join(j[0] for j in rli), cvi))
        else:   # Keep rules separate
            rl.extend(rli)

    return rl

#--------------------------------------------------------------------------------
# Group an iterator
#--------------------------------------------------------------------------------
#   X: The iterator
# key: Callable to produce group-by key
#  rg: Return group key
#--------------------------------------------------------------------------------
# YLD: Yields groups
#--------------------------------------------------------------------------------
def GroupBy(X, key=lambda x : x):
    if not hasattr(X, '__getitem__'):
        X = list(X)
    M  = list(map(key, X))
    si = sorted(range(len(X)), key=lambda i : M[i])
    for k, g in groupby(si, key=lambda i : M[i]):
        yield (k, tuple(X[i] for i in g))

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
    return '{0} {1} {2}'.format(f, '<=' if (o == OP_LE) else '>', v)                

