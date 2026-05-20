from numbers import Number
from abc import abstractmethod

import random
import copy


class Node:
  __fintess: Number
  ret_type: str = 'float'

  def __init__(self, value):
    self.value = value

  @abstractmethod
  def eval(self, **kwargs):
    pass

  @property
  def fitness(self):
    return self.__fintess

  @fitness.setter
  def fitness(self, fitness: Number):
    self.__fintess = fitness

  @property
  def node_count(self):
    return 1

  @property
  def depth(self):
    return 0

  def get_subtree(self, index):
    if index == 0:
      return self
    return None

  def replace_subtree(self, index, new_node):
    if index == 0:
      return new_node
    return self

  def get_relative_depth(self, index, current_depth=0):
    if index == 0:
      return current_depth
    return None

  def copy(self):
    return copy.deepcopy(self)

##################################################################
# Terminals
##################################################################

class Terminal(Node):
  def __init__(self, value):
    super().__init__(value)

  def eval(self, **kwargs):
    return self.value

  def __str__(self):
    return str(self.value)

class Variable(Terminal):
  def __init__(self, value):
    super().__init__(value)

  def eval(self, **kwargs):
    return kwargs.get(self.value, 0)

class Constant(Terminal):
  def __init__(self):
    const = random.random() * 0.4 - 0.2
    super().__init__(round(const, 2))

ConstantValue = 'C'


##################################################################
# Functions
##################################################################

class Function(Node):
  arity: int
  arg_types: tuple = ('float', 'float')

  def __init__(self, value, left, right):
    self.value = value
    self.left = left
    self.right = right
    self.arity = 2

  @abstractmethod
  def eval(self, **kwargs):
    pass

  @property
  def node_count(self):
    return 1 + self.left.node_count + self.right.node_count

  @property
  def depth(self):
    return 1 + max(self.left.depth, self.right.depth)

  def get_subtree(self, index):
    if index == 0:
      return self
    
    left_count = self.left.node_count
    if index <= left_count:
      return self.left.get_subtree(index - 1)
    else:
      return self.right.get_subtree(index - 1 - left_count)

  def replace_subtree(self, index, new_node):
    if index == 0:
      return new_node
    
    left_count = self.left.node_count
    if index <= left_count:
      self.left = self.left.replace_subtree(index - 1, new_node)
    else:
      self.right = self.right.replace_subtree(index - 1 - left_count, new_node)
    return self

  def get_relative_depth(self, index, current_depth=0):
    if index == 0:
      return current_depth
    
    left_count = self.left.node_count
    if index <= left_count:
      return self.left.get_relative_depth(index - 1, current_depth + 1)
    else:
      return self.right.get_relative_depth(index - 1 - left_count, current_depth + 1)

# Functions: BasisArithmetic 
class BasisArithmetic(Function):
  def __init__(self, value, left, right):
    super().__init__(value, left, right)
    self.arity = 2

class Add(BasisArithmetic):
  def __init__(self, left, right):
    super().__init__("+", left, right)

  def eval(self, **kwargs):
    return self.left.eval(**kwargs) + self.right.eval(**kwargs)

  def __str__(self):
    return f"({self.left} + {self.right})"

class Sub(BasisArithmetic):
  def __init__(self, left, right):
    super().__init__("-", left, right)

  def eval(self, **kwargs):
    return self.left.eval(**kwargs) - self.right.eval(**kwargs)

  def __str__(self):
    return f"({self.left} - {self.right})"

class Mul(BasisArithmetic):
  def __init__(self, left, right):
    super().__init__("*", left, right)

  def eval(self, **kwargs):
    return self.left.eval(**kwargs) * self.right.eval(**kwargs)

  def __str__(self):
    return f"({self.left} * {self.right})"


class Div(BasisArithmetic):
  def __init__(self, left, right):
    super().__init__("/", left, right)

  def eval(self, **kwargs):
    rightValue = self.right.eval(**kwargs)
    if rightValue == 0:
      return 1

    return self.left.eval(**kwargs) / rightValue

  def __str__(self):
    return f"({self.left} / {self.right})"

# Functions: ThresholdArithmetic

class Max(BasisArithmetic):
  def __init__(self, left, right):
    super().__init__("max", left, right)

  def eval(self, **kwargs):
    return max(self.left.eval(**kwargs), self.right.eval(**kwargs))

  def __str__(self):
    return f"max({self.left}, {self.right})"
  
class Min(BasisArithmetic):
  def __init__(self, left, right):
    super().__init__("min", left, right)

  def eval(self, **kwargs):
    return min(self.left.eval(**kwargs), self.right.eval(**kwargs))

  def __str__(self):
    return f"min({self.left}, {self.right})"

# Functions: StructuredRoot
class StructuredRoot(Function):
  def __init__(self, left, right):
    super().__init__("StructuredNode", left, right)
    self.ret_type = 'prediction'
    self.arg_types = ('float', 'float')

  def eval(self, **kwargs):
    return self.left.eval(**kwargs) + self.right.eval(**kwargs)

  def __str__(self):
    return f"StructuredRoot({self.left} + {self.right})"