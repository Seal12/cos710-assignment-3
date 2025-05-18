from typing import List, Dict
import math

from structures import ConditionType

class Condition:
  def __init__(self, variable, threshold, condition: ConditionType):
    self.variable = variable
    self.threshold = threshold
    self.condition = condition

  def evaluate(self, data):
    if self.condition == ConditionType.Equal:
      return data[self.variable] == self.threshold
    elif self.condition == ConditionType.GreaterThan:
      return data[self.variable] > self.threshold
    else:
      return data[self.variable] < self.threshold

  def __str__(self):
    return f'({self.variable} {self.condition} {self.threshold})'

class Node:
  def __init__(self, condition: Condition = None, if_true = None, if_false=None, label=None):
    """Initialise tree"""
    raise NotImplementedError
    
  def evaluate(self, fitnessCase: Dict):
    """Evaluate the expression given come fitness case."""
    raise NotImplementedError
  
  def setIndex(self, index = 0):
    self.index = index
    return index

  def __str__(self):
    """Return string of node"""
    raise NotImplementedError
  
class RuleNode(Node):
  def __init__(self, condition: Condition = None, if_true: Node = None, if_false: Node =None, label: str =None):
    self.condition = condition  # Boolean condition for decision-making
    self.if_true = if_true  # Sub-rule if condition is true
    self.if_false = if_false  # Sub-rule if condition is false
    self.label = label  # Terminal classification label

  def evaluate(self, data):
    """Evaluate the rule-based tree on the given data."""
    if self.label is not None:  # Terminal rule
      return self.label
    
    if self.condition.evaluate(data):
      return self.if_true.evaluate(data)
    else:
      return self.if_false.evaluate(data)

  def __str__(self):
    """Display rule in a readable format."""
    if self.label is not None:
      return f"Label({self.label})"

    return f"IF({self.condition}) THEN({self.if_true}) ELSE({self.if_false})"
