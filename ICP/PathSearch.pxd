cimport cython
import numpy as np
cimport numpy as np
np.import_array()

ctypedef np.npy_bool      BOOL_t
ctypedef np.int8_t        CHAR_t
ctypedef np.npy_float64 DOUBLE_t
ctypedef np.npy_intp       INT_t
ctypedef np.npy_long      LONG_t

# Allow multiple datatypes
ctypedef fused DAT_TYPE_t:
    BOOL_t
    CHAR_t
    DOUBLE_t
    INT_t

# Helper structure for performing argsort
cdef struct Bound:
   DOUBLE_t w                        # Weight (err change) associated with boundary
   DOUBLE_t b                        # Boundary position (distance ahead)

@cython.final
cdef class PatherAbs:
   cdef Bound*     B                 # Structure array for sorting boundaries
   cdef DOUBLE_t   bErr              # Best error observed
   cdef DOUBLE_t   bDst              # Best distance observed
   cdef DOUBLE_t   bSlk              # Slack remaining after moving bDst
   cdef DOUBLE_t   eps0              # Magnitude threshold for marking values 0
   cdef INT_t      m                 # Number of samples

   # Some arrays are non-contiguous
   cpdef void FindDist(self, DAT_TYPE_t[::1] A, INT_t m, DOUBLE_t[:] Y, DOUBLE_t[:] W,
                            DOUBLE_t[:] X, int d, DOUBLE_t vMax) nogil

   # All arrays are contiguous
   cpdef void FindDistCg(self, DAT_TYPE_t[::1] A, DOUBLE_t[::1] Y, DOUBLE_t[::1] W,
                              DOUBLE_t[::1] X, int d, DOUBLE_t vMax) nogil

   # Get search results
   cpdef GetResults(self)

@cython.final
cdef class PatherHinge:
   cdef Bound*     B                 # Structure array for sorting boundaries
   cdef DOUBLE_t   bErr              # Best error observed
   cdef DOUBLE_t   bDst              # Best distance observed
   cdef DOUBLE_t   bSlk              # Slack remaining after moving bDst
   cdef DOUBLE_t   eps0              # Magnitude threshold for marking values 0
   cdef DOUBLE_t   eps1
   cdef DOUBLE_t   eps2
   cdef INT_t      m                 # Number of samples

   # Some arrays are non-contiguous
   cpdef void FindDist(self, DAT_TYPE_t[::1] A, INT_t m, DOUBLE_t[:] Y, DOUBLE_t[:] W,
                              DOUBLE_t[:] X, INT_t d, DOUBLE_t vMax) nogil

   # All arrays are contiguous
   cpdef void FindDistCg(self, DAT_TYPE_t[::1] A, DOUBLE_t[::1] Y, DOUBLE_t[::1] W,
                                DOUBLE_t[::1] X, int d, DOUBLE_t vMax) nogil

   # Get search results
   cpdef GetResults(self)
