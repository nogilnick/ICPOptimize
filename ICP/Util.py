from itertools import groupby

#--------------------------------------------------------------------------------
# Group an iterator
#--------------------------------------------------------------------------------
#   X: The iterator
# key: Callable to produce group-by key
#  rg: Return group key
#--------------------------------------------------------------------------------
# YLD: Yields groups
#--------------------------------------------------------------------------------
def GroupBy(X, key=lambda x : x, rg=True):
    if not hasattr(X, '__getitem__'):
        X = list(X)
    M  = list(map(key, X))
    si = sorted(range(len(X)), key=lambda i : M[i])
    for k, g in groupby(si, key=lambda i : M[i]):
        if rg:
            yield (k, tuple(X[i] for i in g))
        else:
            yield tuple(X[i] for i in g)