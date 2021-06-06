from   itertools    import groupby
import numpy        as     np
from   scipy.sparse import issparse

#--------------------------------------------------------------------------------
# Desc: Compute dot product (x @ A) in chunks to reduce memory usage
#--------------------------------------------------------------------------------
#    A: Matrix A
#    x: Vector x
#   bs: Block size
#--------------------------------------------------------------------------------
#  RET: x @ A
#--------------------------------------------------------------------------------
def ChunkedDotCol(A, x, bs=16000000):
   n, m = A.shape
   p = np.empty(m)
   cs = bs // n
   s = e = 0
   while s < m:
      e     += cs
      Ai     = A[:, s:e]
      p[s:e] = x @ Ai
      s      = e
   return p

#--------------------------------------------------------------------------------
# Desc: Compute dot product (A @ x) in chunks to reduce memory usage
#--------------------------------------------------------------------------------
#    A: Matrix A
#    x: Vector x
#   bs: Block size
#--------------------------------------------------------------------------------
#  RET: A @ x
#--------------------------------------------------------------------------------
def ChunkedDotRow(A, x, bs=16000000):
   n, m = A.shape
   p = np.empty(n)
   cs = bs // m
   s = e = 0
   while s < n:
      e     += cs
      Ai     = A[s:e]
      p[s:e] = Ai @ x
      s      = e
   return p

#--------------------------------------------------------------------------------
# Desc: Group an iterator
#--------------------------------------------------------------------------------
#    X: The iterator
#  key: Callable to produce group-by key
#   rg: Return group key
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

#--------------------------------------------------------------------------------
# Desc: Converts matrix to appropriate column order
#--------------------------------------------------------------------------------
#    A: Matrix A
#--------------------------------------------------------------------------------
#  RET: A in column order
#--------------------------------------------------------------------------------
def ToColOrder(A):
   if issparse(A):
      A = A.tocsc()
   elif not A.flags['F_CONTIGUOUS']:
      A = np.asfortranarray(A)
   return A