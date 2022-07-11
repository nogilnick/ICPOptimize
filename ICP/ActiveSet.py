from random import randint

# Fixed size data structure with O(1) add/remove/choice
class ActiveSet:

   def __init__(self, n):
      self.L = [None] * n
      self.n = 0

   def __getitem__(self, i):
      return self.L[i]

   def __len__(self):
      return self.n

   def __repr__(self):
      return self.__str__()

   def __str__(self):
      return 'ActiveSet({})'.format(self.n)

   def add(self, x):
      self.L[self.n] = x
      self.n += 1
      return self

   def choice(self):
      return self.L[randint(0, self.n - 1)]

   def pop(self):
      self.n -= 1
      return self.L[self.n]

   def popi(self, i=0):
      self.L[i], self.L[self.n - 1] = self.L[self.n - 1], self.L[i]
      self.n -= 1
      return self.L[self.n]

   def popr(self):
      i = randint(0, self.n - 1)
      self.L[i], self.L[self.n - 1] = self.L[self.n - 1], self.L[i]
      self.n -= 1
      return self.L[self.n]

   def update(self, I):
      for i in I:
         self.L[self.n] = i
         self.n += 1
      return self
