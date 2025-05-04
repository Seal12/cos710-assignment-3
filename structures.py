from enum import Enum

class GP_Params:
  def __init__(self, terminalSet, functionSet, populationSize, generations, maxTreeDepth = 2, crossoverRate = 0.7, mutationRate = 0.2, reproductionRate = 0.1, minTreeDepth=2):
    self.TerminalSet = terminalSet
    self.FunctionSet = functionSet
    self.PopulationSize = populationSize
    self.Generations = generations
    self.MaxTreeDepth = maxTreeDepth
    self.MinTreeDepth = minTreeDepth
    self.MutationRate = mutationRate
    self.CrossoverRate = crossoverRate
    self.ReproductionRate = reproductionRate

class FitnessCases:
  def __init__(self, inputs, outputs, features):
    self.inputs = inputs
    self.outputs = outputs
    self.features = features

class SelectionMethod(Enum):
  FitnessProportianateSelection = 'fitness-proportianate-selection'
  TournamentSelection = 'tournament-selection'

class GenerationMetods(Enum):
  Grow = 'grow'
  Full = 'full'
  Ramped = 'ramped-half-and-half'

class FitnessMethod(Enum):
  RawFitness = 'raw-fitness'
  AdjustedFitness = 'adjusted-fitness'
  NormalisedFitness = 'normalised-fitness'
