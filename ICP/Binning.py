from   itertools import count
from   sklearn.preprocessing import OneHotEncoder
import numpy as np

#--------------------------------------------------------------------------------
# Desc: Shrink bin array by deleting tiny bins
#--------------------------------------------------------------------------------
#    x: Bin edges
#--------------------------------------------------------------------------------
#  RET: x (updated as per y)
#--------------------------------------------------------------------------------
def Shrink(x, eps=1e-8):
   s = 1
   for j in range(1, len(x)):
      if (x[j] - x[s - 1]) < eps:
         continue
      x[s] = x[j]
      s += 1
   return x[:s]

#--------------------------------------------------------------------------------
# Desc: Test helper function; 1-hot matrix to dense ordinal
#--------------------------------------------------------------------------------
#    X: One hot matrix
# nBin: Number of bins per original column
#--------------------------------------------------------------------------------
#  RET: x (updated as per y)
#--------------------------------------------------------------------------------
def OHToDense(X, nBin):
   nBin = np.cumsum(nBin)
   # This assume there no 0s (invalid bin?) in the 1-hot (non-dense) array
   _, B = X.nonzero()
   B = B.reshape((X.shape[0], -1))
   B -= (nBin - nBin[0])
   return B

#--------------------------------------------------------------------------------
# Desc: Get a column with support for data frame string columns
#--------------------------------------------------------------------------------
#    X: Data matrix
#    i: Column
#--------------------------------------------------------------------------------
#  RET: Column i from X
#--------------------------------------------------------------------------------
def GetCol(X, i):
   if isinstance(X, np.ndarray):
      return X[:, i]
   if hasattr(X, 'iloc'):
      if isinstance(i, str):
         return X[i]
      return X.iloc[:, i]
   return X[:, i]

# Lightweight binning transformer w/ low-memory footprint
class KBDisc:

   prop = ['nb', 'dt', 'enc', 'OHE', 'cols', 'bEdge', 'nBins', 'bTot', 'nFeat']

   def __str__(self):
      return '{}({})'.format(type(self).__name__, self.nb)

   def __repr__(self):
      return self.__str__()

   def __init__(self, n_bins=12, dtype='int8', encode='onehot', use_col=None, const=False):
      self.nb    = n_bins
      self.dt    = dtype
      self.enc   = encode
      self.OHE   = None
      self.cols  = use_col
      self.const = const

   def AddConst(self):
      if self.const:
         return
      self.nBins.append(1)
      self.bTot  += 1
      self.bEdge.append([0])
      self.nFeat += 1
      self.const = True
      return self.MakeEncoder()

   def fit(self, A):
      m, n = A.shape

      nb = min(m, self.nb)
      qt = np.linspace(0, 1, nb - 1)
      self.bEdge = []

      self.cols = range(n) if self.cols is None else self.cols
      for i in self.cols:
         bei = np.quantile(GetCol(A, i), qt)
         # Remove small bins as necessary
         self.bEdge.append(Shrink(bei).copy())

      if self.const: # add in constant column
         self.bEdge.append([])

      self.nBins = [len(i) + 1 for i in self.bEdge]
      self.bTot  = sum(self.nBins)

      if 'onehot' in self.enc:
         self.MakeEncoder()

      self.nFeat = len(self.bEdge)
      return self

   def fit_transform(self, A):
      return self.fit(A).transform(A)

   def MakeEncoder(self):
      self.OHE = (OneHotEncoder(
                   categories=list(map(np.arange, self.nBins)),
                   sparse=('dense' not in self.enc), dtype=self.dt)
                  .fit(np.zeros((1, len(self.nBins)))))
      return self

   def RemoveConst(self):
      if not self.const:
         return
      self.nBins  = self.nBins[:-1]
      self.bTot  -= 1
      self.bEdge  = self.bEdge[:-1]
      self.nFeat -= 1
      self.const = False
      return self.MakeEncoder()

   def transform(self, A, ohe=True):
      B = np.empty((A.shape[0], self.nFeat), dtype=int)

      for i, ci, be in zip(count(), self.cols, self.bEdge):
         B[:, i] = np.searchsorted(be, GetCol(A, ci), side='left')

      if self.const:
         B[:, -1] = 0

      if (self.OHE is not None) and ohe:
         return self.OHE.transform(B)

      return B
