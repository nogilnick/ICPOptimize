cimport cython
ctypedef size_t       UINT64_t
ctypedef unsigned int UINT32_t

cdef UINT32_t Hash32(UINT32_t x) nogil

cdef UINT32_t RandRange(UINT32_t a, UINT32_t b, UINT32_t* x) nogil

# Fixed size data structure with O(1) add/remove/choice
@cython.final
cdef class ActiveSet:
   cdef UINT64_t* L
   cdef UINT64_t  n
   cdef UINT32_t  r

   cpdef void add(self, UINT64_t x) nogil

   cpdef UINT64_t choice(self) nogil

   cpdef UINT64_t get(self, UINT64_t i) nogil

   cpdef UINT64_t pop(self) nogil

   cpdef UINT64_t popi(self, UINT64_t i) nogil

   cpdef UINT64_t popr(self) nogil

   cpdef ActiveSet update(self, I)
