# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
cimport cython
from libc.stdlib cimport free, malloc

cdef inline UINT32_t Hash32(UINT32_t x) nogil:
    x ^= x << 13
    x ^= x >> 17
    x ^= x <<  5
    return x

cdef inline UINT32_t RandRange(UINT32_t a, UINT32_t b, UINT32_t* x) nogil:
   x[0] = Hash32(x[0])
   return a + x[0] % (b - a)

# Fixed size data structure with O(1) add/remove/choice
@cython.final
cdef class ActiveSet:

   def __cinit__(self, UINT64_t c, UINT32_t r):
      self.L = <UINT64_t*> malloc(c * sizeof(UINT64_t))
      self.n = 0
      self.r = r + 0x4f6cdd1d

   def __dealloc__(self):
      free(self.L)

   def __getitem__(self, i):
      return self.get(i)

   def __len__(self):
      return self.n

   def __repr__(self):
      return self.__str__()

   def __str__(self):
      return 'ActiveSet({})'.format(self.n)

   cpdef void add(self, UINT64_t x) nogil:
      self.L[self.n] = x
      self.n += 1

   cpdef UINT64_t choice(self) nogil:
      return self.L[RandRange(0, self.n, &self.r)]

   cpdef UINT64_t get(self, UINT64_t i) nogil:
      return self.L[i]

   cpdef UINT64_t pop(self) nogil:
      self.n -= 1
      return self.L[self.n]

   cpdef UINT64_t popi(self, UINT64_t i) nogil:
      self.L[i], self.L[self.n - 1] = self.L[self.n - 1], self.L[i]
      self.n -= 1
      return self.L[self.n]

   cpdef UINT64_t popr(self) nogil:
      cdef UINT32_t i = RandRange(0, self.n, &self.r)
      self.L[i], self.L[self.n - 1] = self.L[self.n - 1], self.L[i]
      self.n -= 1
      return self.L[self.n]

   cpdef ActiveSet update(self, I):
      for i in I:
         self.add(i)
      return self
