from   itertools    import groupby
import numpy        as     np
from   scipy.sparse import issparse, linalg as slinalg

#--------------------------------------------------------------------------------
# Desc: Adds x and y handling sparse matrices
#--------------------------------------------------------------------------------
#    x: Input array
#    y: Input vector/array
#--------------------------------------------------------------------------------
#  RET: x (updated as per y)
#--------------------------------------------------------------------------------
def Add(x, y):
   if issparse(y):   # N.B. this assumes y is csc
      x[y.indices] += y.data
   else:
      x += y
   return x

#--------------------------------------------------------------------------------
# Desc: Correlation coef of each column of A with Y
#--------------------------------------------------------------------------------
#    A: Matrix A
#    Y: Target vector
#    W: Sample weights
#    c: Include constant (0 to disable)
#--------------------------------------------------------------------------------
#  RET: x @ A
#--------------------------------------------------------------------------------
def ColCorr(A, Y, W=None, c=1):
   m, n    = A.shape
   W       = MakeWeights(W, m)
   Yc      = (Y - Y.mean()) * W
   Yd      = Yc.std()
   ccf     = np.empty(n + (c != 0))
   for i in range(A.shape[1]):
      Ai     = ToArray(GetCol(A, i))
      ccf[i] = np.dot(Ai - Ai.mean(), Yc) / (Yd * Ai.std())
   if c != 0:                              # Handle constant separately
      ccf[n] = c * np.dot(2. * (Y > 0) - 1., W)
   return ccf

#--------------------------------------------------------------------------------
# Desc: Copy a list of properties from 1 object to another
#--------------------------------------------------------------------------------
#   O1: Object 1
#   O2: Object 2
# prop: The properties to copy
#--------------------------------------------------------------------------------
#  RET: Object 2
#--------------------------------------------------------------------------------
def Copy(O1, O2, prop):
   for i in prop:
      if hasattr(O1, i):
         setattr(O2, i, getattr(O1, i))
   return O2

#--------------------------------------------------------------------------------
# Desc: Compute dot product (x @ A) in chunks to reduce memory usage
#--------------------------------------------------------------------------------
#    A: Matrix A
#    x: Vector x
#   bs: Block size
#--------------------------------------------------------------------------------
#  RET: x @ A
#--------------------------------------------------------------------------------
def ChunkedDotCol(A, x, bs=1.6e7):
   n, m = A.shape
   p    = np.empty(m)
   cs   = max(1, round(bs / n))
   s    = e = 0
   while s < m:
      e     += cs
      Ai     = GetCol(A, slice(s, e))
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
def ChunkedDotRow(A, x, bs=1.6e7):
   n, m = A.shape
   p    = np.empty(n)
   cs   = max(1, round(bs / n))
   s    = e = 0
   while s < n:
      e     += cs
      Ai     = A[s:e]
      p[s:e] = Ai @ x
      s      = e
   return p

#--------------------------------------------------------------------------------
# Desc: Compute column norms with low memory footprint
#--------------------------------------------------------------------------------
#    X: Data matrix
#    c: Constant (0 to disable)
#  ord: Norm ord
#--------------------------------------------------------------------------------
#  RET: Column norms
#--------------------------------------------------------------------------------
def ChunkedColNorm(A, c, ord=2):
   m, n = A.shape
   p    = np.empty(n + (c != 0))
   NFx  = np.linalg.norm if not issparse(A) else slinalg.norm
   for i in range(n):
      p[i] = NFx(A[:, i], ord=ord, axis=0)
   if c != 0:
      p[n] = np.sqrt(m) * c
   return p

#--------------------------------------------------------------------------------
# Desc: Handle getting a column from an ndarray, data frame, sparse matrix, etc
#--------------------------------------------------------------------------------
#    X: The vector/array
#    i: The column to return
#--------------------------------------------------------------------------------
#  RET: The i-th column of the input
#--------------------------------------------------------------------------------
def GetCol(X, i):
   if isinstance(X, np.ndarray):
      return X[:, i]
   if issparse(X):
      if isinstance(i, int):
         return X.getcol(i)
      return X[:, i]
   if hasattr(X, 'iloc'):
      return X.iloc[:, i].values
   return X[i]

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
# Desc: Ensure input is a valid weight array
#--------------------------------------------------------------------------------
#    W: The vector/array
#    m: Number of input samples
#--------------------------------------------------------------------------------
#  RET: A valid weight array (defaults to uniform)
#--------------------------------------------------------------------------------
def MakeWeights(W, m):
   return np.full(m, 1 / m) if W is None else (W / W.sum())

#--------------------------------------------------------------------------------
# Desc: Multiplies x and y handling sparse matrices
#--------------------------------------------------------------------------------
#    x: Input array
#    y: Input vector/array
#--------------------------------------------------------------------------------
#  RET: x (updated as per y)
#--------------------------------------------------------------------------------
def Mult(x, y):
   if issparse(y):
      return y.multiply(x.reshape(y.shape))
   return np.multiply(x, y)

#--------------------------------------------------------------------------------
# Desc: Handle sparse arrays where flat arrays are expected
#--------------------------------------------------------------------------------
#    X: The vector/array
#--------------------------------------------------------------------------------
#  RET: An array w/ contents of X
#--------------------------------------------------------------------------------
def ToArray(X):
   if isinstance(X, np.matrix) or issparse(X):
      return X.A.ravel()
   return X

#--------------------------------------------------------------------------------
# Desc: Converts matrix to appropriate column order
#--------------------------------------------------------------------------------
#    A: Matrix A
#--------------------------------------------------------------------------------
#  RET: A in column order
#--------------------------------------------------------------------------------
def ToColOrder(A):
   if issparse(A) and (A.getformat() != 'csc'):
      A = A.tocsc()
   elif isinstance(A, np.ndarray) and (not A.data.f_contiguous):
      A = np.asfortranarray(A)
   return A

#--------------------------------------------------------------------------------
# Desc: Weighted dot product
#--------------------------------------------------------------------------------
#    X: Vector
#    Y: Vector
#    W: Weight vector
#--------------------------------------------------------------------------------
#  RET: X^T @ diag(W) @ Y
#--------------------------------------------------------------------------------
def WDot(X, Y, W):
   v = W @ Mult(X, Y)
   if isinstance(v, float):
      return v
   return v[0]