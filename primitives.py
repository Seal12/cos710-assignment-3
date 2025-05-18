from typing import List, Dict
import math

############################################
# Base class

class Node:
  arity = 0

  def evaluate(self, fitnessCase: Dict):
    """Evaluate the expression given come fitness case."""
    raise NotImplementedError
  
  def getEval(self):
    if self.eval == None:
      raise ValueError
    
    return self.eval

  def setIndex(self, index = 0):
    self.index = index
    return index

  def __str__(self):
    """Return string of node"""
    raise NotImplementedError
  

############################################
# Terminals

class Terminal(Node):

  def setIndex(self, index = 0):
    self.index = index
    return index

  def __str__(self):
    return f'{self.value}'

class Variable(Terminal):
  def __init__(self, value: str):
    self.value = value

  def evaluate(self, fitnessCase):
    self.eval = fitnessCase[self.value]
    return self.eval

class Const(Terminal):
  def __init__(self, value):
    self.value = value

  def evaluate(self, _ = None, __ = None):
    return self.value

############################################
# Functions: Conditional Statement Functions

class IF_ELSE_THEN(Node):
  arity = 3

  def __init__(self, condition: Node, conditionTrue: Node, conditionFalse: Node):
    self.arg1 = condition
    self.arg2 = conditionTrue
    self.arg3 = conditionFalse

  def setIndex(self, index = 0):
    self.index = index
    nextIndex = self.arg1.setIndex(index + 1)
    nextIndex = self.arg2.setIndex(nextIndex + 1)
    nextIndex = self.arg3.setIndex(nextIndex + 1)

    return nextIndex

  def evaluate(self, fitnessCase):
    condition = self.arg1.evaluate(fitnessCase)
    ifTrue = self.arg2.evaluate(fitnessCase)
    ifFalse = self.arg3.evaluate(fitnessCase)

    if condition:
      return ifTrue
    else:
      return ifFalse
    
  def __str__(self, depth=0):
    condition = str(self.arg1)
    ifTrue = str(self.arg2)
    ifFalse = str(self.arg3)

    return "IF({}) \n{}THEN({}) \n{}ELSE({})".format(condition, '\t' * depth, ifTrue, '\t' * depth, ifFalse)

############################################
# Functions: Logic Operator Functions

class LogicOperator(Node):
  arity = 2

  def __init__(self, arg1: Node, arg2: Node):
    self.arg1 = arg1
    self.arg2 = arg2

  def setIndex(self, index = 0):
    self.index = index
    nextIndex = self.arg1.setIndex(index + 1)
    nextIndex = self.arg2.setIndex(nextIndex + 1)

    return nextIndex

class AndOperator(LogicOperator):
  def evaluate(self, fitnessCase):
    self.eval = self.arg1.evaluate(fitnessCase) and self.arg2.evaluate(fitnessCase)
    return self.eval
  
  def __str__(self):
    return f'({self.arg1}&&{self.arg2})'

class OROperator(LogicOperator):
  def evaluate(self, fitnessCase):
    self.eval = self.arg1.evaluate(fitnessCase) or self.arg2.evaluate(fitnessCase)
    return self.eval
  
  def __str__(self):
    return f'({self.arg1}||{self.arg2})'

class EqualTo(LogicOperator):
  def evaluate(self, fitnessCase):
    self.eval = self.arg1.evaluate(fitnessCase) == self.arg2.evaluate(fitnessCase)
    return self.eval
  
  def __str__(self):
    return f'({self.arg1}=={self.arg2})'

class LessThan(LogicOperator):
  def evaluate(self, fitnessCase):
    self.eval = self.arg1.evaluate(fitnessCase) < self.arg2.evaluate(fitnessCase)
    return self.eval
  
  def __str__(self):
    return f'({self.arg1}<{self.arg2})'

class LessOrEqualThan(LogicOperator):
  def evaluate(self, fitnessCase):
    self.eval = self.arg1.evaluate(fitnessCase) <= self.arg2.evaluate(fitnessCase)
    return self.eval
  
  def __str__(self):
    return f'({self.arg1}<={self.arg2})'

class GreaterThan(LogicOperator):
  def evaluate(self, fitnessCase):
    self.eval = self.arg1.evaluate(fitnessCase) > self.arg2.evaluate(fitnessCase)
    return self.eval
  
  def __str__(self):
    return f'({self.arg1}>{self.arg2})'

class GreaterOrEqalThan(LogicOperator):
  def evaluate(self, fitnessCase):
    self.eval = self.arg1.evaluate(fitnessCase) >= self.arg2.evaluate(fitnessCase)
    return self.eval
  
  def __str__(self):
    return f'(({self.arg1}>={self.arg2})'
