# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
cimport cython
import  numpy as np
cimport numpy as np
np.import_array()

from libc.math   cimport fabs, fmin, isinf, INFINITY
from libc.stdlib cimport free, malloc

cdef extern from "stdlib.h":
   ctypedef void const_void "const void"
   void qsort(void *base, int nmemb, int size,
              int(*compar)(const_void *, const_void *)) nogil

#--------------------------------------------------------------------------------
# Sorts boundaries
#--------------------------------------------------------------------------------
# B: The boundaries to sort
# I: The array of indices to sort
# n: The length of the array
#--------------------------------------------------------------------------------
cdef void SortBounds(Bound* B, INT_t n) nogil:
   if n <= 1:
      return

   qsort(<void*> B, n, sizeof(Bound), CmpBound)

   return

# Helper comparison function
cdef inline int CmpBound(const_void* a, const_void* b) nogil:
   cdef DOUBLE_t v = (<Bound*> a).b - (<Bound*> b).b
   return (<int> (v > 0)) - (<int> (v < 0))

# Clipped sign function
cdef inline int SignInt(DAT_TYPE_t x, DOUBLE_t eps) nogil:
   return (<int> (x > eps)) - (<int> (x < -eps))

@cython.final
cdef class PatherAbs:
   #--------------------------------------------------------------------------------
   # Constructor
   #--------------------------------------------------------------------------------
   #    m: The length of the vector
   # eps0: Movement must improve error by at least this much to count
   #--------------------------------------------------------------------------------
   def __cinit__(self, INT_t m, DOUBLE_t eps0):
      # Structure array for sorting with index; 1 extra for sentinel
      self.B    = <Bound*> malloc((m + 1) * sizeof(Bound))
      self.bErr = 0.0
      self.bDst = 0.0
      self.bSlk = 0.0
      self.eps0 = eps0
      self.m    = m

   def __dealloc__(self):
      free(self.B)

   #--------------------------------------------------------------------------------
   # Search for optimal distance to move along a direction up to some max (Abs)
   #--------------------------------------------------------------------------------
   #    A: Vector specifying the direction
   #    m: The length of the vector
   #    Y: The target vector
   #    W: Sample weights
   #    X: Current solution
   #  mrg: Hinge margin
   #    d: Sign of current direction
   # vMax: Maximum distance to move along direction
   # eps0: Movement must improve error by at least this much to count
   # eps1: If gradient ever becomes larger than this value, stop searching
   # eps2: If gradient is larger than eps1 and have moved at least eps2 from start; break
   #  out: Output vector
   #--------------------------------------------------------------------------------
   cpdef void FindDist(self, DAT_TYPE_t[::1] A, INT_t m, DOUBLE_t[:] Y, DOUBLE_t[:] W,
                       DOUBLE_t[:] X, int d, DOUBLE_t vMax) nogil:
      cdef Bound* B = self.B
      cdef INT_t  c = 0
      cdef INT_t  i

      cdef DOUBLE_t eps0  = self.eps0
      cdef DOUBLE_t  mae  = 0.0               # Magnitude adding error
      cdef DOUBLE_t  mse  = 0.0               # Magnitude subtracting error
      cdef DOUBLE_t  roc
      cdef DOUBLE_t  u
      cdef DOUBLE_t  pu
      cdef DOUBLE_t  er

      for i in range(m):
         if SignInt(A[i], eps0) == 0:
             continue                         # This sample does not affect error

         B[c].b = d * (Y[i] - X[i]) / A[i]    # Boundary
         B[c].w = fabs(A[i]) * W[i]           # Weight

         if B[c].b <= 0:
            mae += B[c].w                     # Already passed zero; increasing
         else:
            mse += B[c].w                     # Error is decreasing
            c   += (B[c].b <= vMax)           # Direction changes within vMax; take note

      roc = mae - mse
      if roc >= 0.0:
         self.bErr = self.bDst = 0.0          # Error already increases in this dir
         self.bSlk = vMax
         return

      SortBounds(B, c)                        # Sort boundaries markers
      B[c].b = INFINITY                       # Sentinel values at end of array
      c     += 1
      er     = 0.0
      pu     = 0.0
      for i in range(c):
         u   = fmin(B[i].b, vMax)             # Travel up to vMax
         er += (u - pu) * roc                 # ROC is constant over this distance

         if u == vMax:
            break

         roc += 2 * B[i].w                    # Change from - to + (hence 2x)
         if roc >= 0.0:
            break                             # Error now increasing; break
         pu = u

      self.bErr = er
      self.bDst = u
      self.bSlk = vMax - u
      return

   #--------------------------------------------------------------------------------
   # Same as above but all arrays are contiguous
   #--------------------------------------------------------------------------------
   cpdef void FindDistCg(self, DAT_TYPE_t[::1] A, DOUBLE_t[::1] Y, DOUBLE_t[::1] W,
                         DOUBLE_t[::1] X, int d, DOUBLE_t vMax) nogil:
      cdef Bound* B = self.B
      cdef INT_t  m = self.m
      cdef INT_t  c = 0
      cdef INT_t  i

      cdef DOUBLE_t eps0  = self.eps0
      cdef DOUBLE_t  mae  = 0.0               # Magnitude adding error
      cdef DOUBLE_t  mse  = 0.0               # Magnitude subtracting error
      cdef DOUBLE_t  roc
      cdef DOUBLE_t  u
      cdef DOUBLE_t  pu
      cdef DOUBLE_t  er

      for i in range(m):
         if SignInt(A[i], eps0) == 0:
             continue                         # This sample does not affect error

         B[c].b = d * (Y[i] - X[i]) / A[i]    # Boundary
         B[c].w = fabs(A[i]) * W[i]           # Weight

         if B[c].b <= 0:
            mae += B[c].w                     # Already passed zero; increasing
         else:
            mse += B[c].w                     # Error is decreasing
            c   += (B[c].b <= vMax)           # Direction changes within vMax; take note

      roc = mae - mse
      if roc >= 0.0:
         self.bErr = self.bDst = 0.0          # Error already increases in this dir
         self.bSlk = vMax
         return

      SortBounds(B, c)                        # Sort boundaries markers
      B[c].b = INFINITY                       # Sentinel values at end of array
      c     += 1
      er     = 0.0
      pu     = 0.0
      for i in range(c):
         u   = fmin(B[i].b, vMax)             # Travel up to vMax
         er += (u - pu) * roc                 # ROC is constant over this distance

         if u == vMax:
            break

         roc += 2 * B[i].w                    # Change from - to + (hence 2x)
         if roc >= 0.0:
            break                             # Error now increasing; break
         pu = u

      self.bErr = er
      self.bDst = u
      self.bSlk = vMax - u
      return

   # Get search results
   cpdef GetResults(self):
      return self.bErr, self.bDst, self.bSlk

@cython.final
cdef class PatherHinge:
   #--------------------------------------------------------------------------------
   # Constructor
   #--------------------------------------------------------------------------------
   #    m: The length of the vector
   # eps0: Movement must improve error by at least this much to count
   # eps1: If gradient ever becomes larger than this value, stop searching
   # eps2: If gradient is larger than eps1 and have moved at least eps2 from start; break
   #--------------------------------------------------------------------------------
   def __cinit__(self, INT_t m, DOUBLE_t eps0, DOUBLE_t eps1, DOUBLE_t eps2):
      # Structure array for sorting with index; 2 extra for left/right sentinels
      self.B    = <Bound*> malloc((m + 2) * sizeof(Bound))

      self.bErr = 0.0
      self.bDst = 0.0
      self.bSlk = 0.0
      self.eps0 = eps0
      self.eps1 = eps1
      self.eps2 = eps2
      self.m    = m

   def __dealloc__(self):
      free(self.B)

   #--------------------------------------------------------------------------------
   # Search for optimal distance to move along a direction up to some max (Hinge)
   #--------------------------------------------------------------------------------
   #    A: Vector specifying the direction
   #    Y: The target vector
   #    W: Sample weights
   #    X: Current solution
   #  mrg: Hinge margin
   #    d: Sign of current direction
   # vMax: Maximum distance to move along direction

   #--------------------------------------------------------------------------------
   cpdef void FindDist(self, DAT_TYPE_t[::1] A, INT_t m, DOUBLE_t[:] Y, DOUBLE_t[:] W,
                       DOUBLE_t[:] X, INT_t d, DOUBLE_t vMax) nogil:
      cdef DOUBLE_t  eps0 = self.eps0
      cdef DOUBLE_t  eps1 = self.eps1
      cdef DOUBLE_t  eps2 = self.eps2
      cdef Bound*    B    = self.B

      cdef INT_t i
      cdef INT_t j

      cdef INT_t     ai   = m                  # Start at right side of bound array
      cdef DOUBLE_t  mae  = 0.0                # Magnitude adding error

      cdef INT_t     si   = 0                  # Start at left side of bound array
      cdef DOUBLE_t  mse  = 0.0                # Magnitude subtracting error

      cdef DOUBLE_t rErr
      cdef DOUBLE_t bErr
      cdef DOUBLE_t bDst
      cdef DOUBLE_t cp
      cdef DOUBLE_t Bi
      cdef DOUBLE_t Wi
      cdef int      Si
      cdef int      ASi

      for i in range(m):
         ASi = SignInt(A[i], eps0)
         if ASi == 0:
            continue                           # This sample does not affect error

         Wi = fabs(A[i]) * W[i]                # Weight
         Bi = d * (Y[i] - X[i]) / A[i]         # Bi>0 --> boundary ahead in direction d*Ai

         Si = SignInt(Y[i], eps0)
         if (d > 0) ^ (ASi > 0) ^ (Si > 0):    # Direction correct if prod (d*a*y) > 0
            if Bi > 0:                         # Boundary ahead; sample must be wrong
               if Bi <= vMax:                  # Reachable within vMax
                  B[si].b = Bi
                  B[si].w = Wi                 # keep track to adjust gradient
                  si     += 1                  # Next open position
               mse       += Wi                 # This sample subtracts from error
         else:                                 # Else; direction is wrong
            if (0 < Bi) and (Bi <= vMax):      # Reachable within vMax; sample is right
               B[ai].b = Bi
               B[ai].w = Wi                    # keep track to adjust gradient
               ai     -= 1                     # Next open position
            elif Bi <= 0:                      # Sample already reached
               mae    += Wi                    # Adds to error in this direction

      if (ai == m) and (si == 0):              # No boundaries
         if mae >= mse:                        # Error only increases
            self.bErr = self.bDst = 0.0
            self.bSlk = vMax
         else:                                 # Unbounded within vMax
            self.bErr = vMax * (mae - mse)     # Move max distance allowed
            self.bDst = vMax
            self.bSlk = 0.0
         return

      # m total (+2 sentinel); m - ai in upper, thus ai in lower
      # B + ai + 1 == move past ai lower + 1 sentinel to reach the upper array
      SortBounds(B + ai + 1, m - ai)           # Sort the m-ai boundaries that add error
      B[m + 1].b = INFINITY                    # Sentinel value at end of array

      SortBounds(B, si)                        # Sort the si boundaries that sub error
      B[si].b    = INFINITY                    # Sentinel value at end of array

      # Find optimal distance to move along direction vector up to vMax
      rErr = 0.0                               # Cur change in error
      bErr = INFINITY                          # Min error delta seen so far
      bDst = 0.0                               # Best distance
      cp   = 0                                 # Current position
      i    = ai + 1                            # Index to next boundary adding err
      j    = 0                                 # Index to next boundary subing err
      while True:
         Bi   = fmin(B[i].b, B[j].b)
         Bi   = fmin(Bi, vMax)                 # Move up to vMax; no further
         rErr = rErr + (Bi - cp) * (mae - mse) # Error change for this step
         cp   = Bi

         if rErr < bErr:
            bErr = rErr                        # Lowest error seen so far
            bDst = Bi

         if Bi >= vMax:                        # Path is constrained by interval
            break

         while B[i].b == cp:                   # Update error roc
            mae += B[i].w
            i   += 1

         while B[j].b == cp:
            mse -= B[j].w
            j   += 1

         if ((mae - mse) > eps1) and (cp > eps2):
            break                              # Err increasing error; stop

      self.bErr = bErr
      self.bDst = bDst
      self.bSlk = vMax - bDst
      return

   #--------------------------------------------------------------------------------
   # Same as above but all arrays are contiguous
   #--------------------------------------------------------------------------------
   cpdef void FindDistCg(self, DAT_TYPE_t[::1] A, DOUBLE_t[::1] Y, DOUBLE_t[::1] W,
                         DOUBLE_t[::1] X, int d, DOUBLE_t vMax) nogil:
      cdef DOUBLE_t  eps0 = self.eps0
      cdef DOUBLE_t  eps1 = self.eps1
      cdef DOUBLE_t  eps2 = self.eps2
      cdef Bound*    B    = self.B
      cdef INT_t     m    = self.m

      cdef INT_t i
      cdef INT_t j

      cdef INT_t     ai   = m                  # Start at right side of bound array
      cdef DOUBLE_t  mae  = 0.0                # Magnitude adding error

      cdef INT_t     si   = 0                  # Start at left side of bound array
      cdef DOUBLE_t  mse  = 0.0                # Magnitude subtracting error

      cdef DOUBLE_t rErr
      cdef DOUBLE_t bErr
      cdef DOUBLE_t bDst
      cdef DOUBLE_t cp
      cdef DOUBLE_t Bi
      cdef DOUBLE_t Wi
      cdef int      Si
      cdef int      ASi

      for i in range(m):
         ASi = SignInt(A[i], eps0)
         if ASi == 0:
            continue                           # This sample does not affect error

         Wi = fabs(A[i]) * W[i]                # Weight
         Bi = d * (Y[i] - X[i]) / A[i]         # Bi>0 --> boundary ahead in direction d*Ai

         Si = SignInt(Y[i], eps0)
         if (d > 0) ^ (ASi > 0) ^ (Si > 0):    # Direction correct if prod (d*a*y) > 0
            if Bi > 0:                         # Boundary ahead; sample must be wrong
               if Bi <= vMax:                  # Reachable within vMax
                  B[si].b = Bi
                  B[si].w = Wi                 # keep track to adjust gradient
                  si     += 1                  # Next open position
               mse       += Wi                 # This sample subtracts from error
         else:                                 # Else; direction is wrong
            if (0 < Bi) and (Bi <= vMax):      # Reachable within vMax; sample is right
               B[ai].b = Bi
               B[ai].w = Wi                    # keep track to adjust gradient
               ai     -= 1                     # Next open position
            elif Bi <= 0:                      # Sample already reached
               mae    += Wi                    # Adds to error in this direction

      if (ai == m) and (si == 0):              # No boundaries
         if mae >= mse:                        # Error only increases
            self.bErr = self.bDst = 0.0
            self.bSlk = vMax
         else:                                 # Unbounded within vMax
            self.bErr = vMax * (mae - mse)     # Move max distance allowed
            self.bDst = vMax
            self.bSlk = 0.0
         return

      # m total (+2 sentinel); m - ai in upper, thus ai in lower
      # B + ai + 1 == move past ai lower + 1 sentinel to reach the upper array
      SortBounds(B + ai + 1, m - ai)           # Sort the m-ai boundaries that add error
      B[m + 1].b = INFINITY                    # Sentinel value at end of array

      SortBounds(B, si)                        # Sort the si boundaries that sub error
      B[si].b    = INFINITY                    # Sentinel value at end of array

      # Find optimal distance to move along direction vector up to vMax
      rErr = 0.0                               # Cur change in error
      bErr = INFINITY                          # Min error delta seen so far
      bDst = 0.0                               # Best distance
      cp   = 0                                 # Current position
      i    = ai + 1                            # Index to next boundary adding err
      j    = 0                                 # Index to next boundary subing err
      while True:
         Bi   = fmin(B[i].b, B[j].b)
         Bi   = fmin(Bi, vMax)                 # Move up to vMax; no further
         rErr = rErr + (Bi - cp) * (mae - mse) # Error change for this step
         cp   = Bi

         if rErr < bErr:
            bErr = rErr                        # Lowest error seen so far
            bDst = Bi

         if Bi >= vMax:                        # Path is constrained by interval
            break

         while B[i].b == cp:                   # Update error roc
            mae += B[i].w
            i   += 1

         while B[j].b == cp:
            mse -= B[j].w
            j   += 1

         if ((mae - mse) > eps1) and (cp > eps2):
            break                              # Err increasing error; stop

      self.bErr = bErr
      self.bDst = bDst
      self.bSlk = vMax - bDst
      return

   # Get search results
   cpdef GetResults(self):
      return self.bErr, self.bDst, self.bSlk
