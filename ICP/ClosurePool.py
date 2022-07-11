from   math      import inf
from   threading import Thread
from   queue     import Empty, Queue

def RunAndEnqueue(Q, Fx, slot, args=(), kwargs={}):
   rv = Fx(*args, **kwargs)
   Q.put((slot, rv))

# Class for running a thread pool in which each worker has state associated with it
# in the form of a closure
class ClosurePool:

   def __str__(self):
      return '{}({})'.format(type(self).__name__, self.nThrd)

   def __repr__(self):
      return self.__str__()

   # Init the ClosurePool. Expects an iterable of closures that will be used as workers
   def __init__(self, WL):
      self.WL    = list(WL)
      self.kArr  = [None for _ in self.WL]   # Result key
      self.nRun  = 0
      self.nThrd = len(WL)
      self.Q     = Queue()
      self.slot  = 0
      self.thrd  = [None for _ in self.WL]

   def Full(self):
      return self.nRun >= self.nThrd

   def Get(self, default=None, wait=False):
      if self.nRun <= 0:
         return None, default
      try:
         self.slot, rv = self.Q.get(wait)
         self.thrd[self.slot] = None
         self.nRun -= 1
         return self.kArr[self.slot], rv
      except Empty:
         pass
      return None, default

   def GetAll(self, n=inf, wait=True):
      while (self.nRun > 0) and (n > 0):
         try:
            self.slot, rv = self.Q.get(wait)
            self.thrd[self.slot] = None
            self.nRun -= 1
            yield self.kArr[self.slot], rv
         except Empty:
            break
         n -= 1

   # Launch another thread with return value associated with a specified key
   def Start(self, key=None, args=(), kwargs={}):
      if self.nRun >= self.nThrd:
         return False

      # Find empty slot
      while self.thrd[self.slot] is not None:
         self.slot = (self.slot + 1) % self.nThrd

      self.thrd[self.slot] = Thread(target=RunAndEnqueue, args=(
                                    self.Q, self.WL[self.slot], self.slot, args, kwargs))
      self.thrd[self.slot].start()
      self.kArr[self.slot] = key
      self.nRun += 1
      return True

# Class with the same interface as ClosurePool but for single threaded behavior
class DummyPool:

   def __str__(self):
      return '{}()'.format(type(self).__name__)

   def __repr__(self):
      return self.__str__()

   # Expects a list of exactly 1 closure
   def __init__(self, WL):
      self.WL    = WL[0]
      self.res   = None
      self.key   = None

   def Full(self):
      return False

   def Get(self, default=None, wait=False):
      k, r = self.key, self.res
      self.key = self.res = None
      return k, r

   def GetAll(self, n=inf, wait=True):
      if False:   # Provide same unpacking behavior but never return anything
         yield None, None

   def Start(self, key=None, args=(), kwargs={}):
      self.res = self.WL(*args, **kwargs)    # Run on same thread
      self.key = key
      return True
