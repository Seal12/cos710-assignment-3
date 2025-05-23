from collections import Counter
from functools import cache
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from sklearn.model_selection import train_test_split
from typing import List

import argparse
import copy
import math
import numpy as np
import os
import pandas
import random
import time
import matplotlib.pyplot as plt

import primitives as Primitives
from structures import FitnessCases, FitnessMethod, GenerationMetods, GP_Params, SelectionMethod

###############################################################
# Data

def isFolderExist(dir: str):
  try:
    if os.path.exists(dir):
      return True
    else:
      os.mkdir(dir)
      return True
  except:
    print(Exception(f'Could not create director "{dir}"'))
    return False

def getDataFrame(filePath):
  df = pandas.read_csv(filePath, delimiter='\t')

  df.dropna(inplace=True)

  X = df.iloc[:, :-1]
  Y = df.iloc[:, -1] 
  
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

  print(df.info())
  print()

  print("Training Inputs Shape:", X_train.shape)
  print("Training Target Shape:", Y_train.shape)
  print("Test Inputs Shape:", X_test.shape)
  print("Test Target Shape:", Y_test.shape)
  print()

  return [
    (X_train, Y_train),
    (X_test, Y_test)
  ]

def plotFitnessPerGeneration(generationFitnesses: List, title, prefix = ''):
  print(f'Plot Fitness Per Generation: {title}')
  numDataPoints = len(generationFitnesses)
  fig, ax = plt.subplots(figsize=(16, 8))

  x = np.arange(numDataPoints)

  # Line
  y_line = generationFitnesses
  ax.plot(x, y_line, marker='o', linestyle='solid', color='r', label='Fitness')

  majorTicksSteps = max(round(numDataPoints/10), 1)
  ax.xaxis.set_major_locator(MultipleLocator(majorTicksSteps))

  # Minor axis steps
  ax.xaxis.set_minor_locator(AutoMinorLocator(5))
  ax.grid(which="both", linestyle="--", linewidth=0.5)

  plt.xlabel('Generation')
  plt.ylabel('Population Fitness')
  plt.title(title)
  plt.legend()

  if not isFolderExist('./out'):
    return

  # Display the graph
  fileName = title.replace(' ', '_') +  '.png'
  plt.savefig(f'./out/{prefix}_{fileName}')

def plotBarGraph(x_values: List, y_values: List, title: str, xLabel: str, yLabel: str, prefix = ''):
  fig, ax = plt.subplots(figsize=(16, 8))
  
  ax.bar(x_values, y_values, color='skyblue', edgecolor='black')

  # ax.yaxis.set_minor_locator(AutoMinorLocator())
  ax.grid(which="both", linestyle="--", linewidth=0.5)

  # Add labels and title
  plt.xlabel(xLabel)
  plt.ylabel(yLabel)
  plt.title(title)

  if not isFolderExist('./out'):
    return

  fileName = title.replace(' ', '_') +  '.png'
  fig.savefig(f'./out/{prefix}_{fileName}')

############################################
# Population Helpers

def createTerminalPair(value):
  if value == 'AGE':
    newValue = random.randint(7, 78)
    return Primitives.Variable(value), Primitives.Const(newValue)
  elif value in ['SEX', 'ANTIVIRALS', 'HISTOLOGY']:
    newValue = random.randint(1, 2)
    return Primitives.Variable(value), Primitives.Const(newValue)
  elif value == 'BILIRUBIN':
    newValue = random.randint(0, 34)
    return Primitives.Variable(value), Primitives.Const(newValue)
  elif value == 'ALK PHOSPHATE':
    newValue = random.randint(0, 83)
    return Primitives.Variable(value), Primitives.Const(newValue)
  elif value == 'SGOT':
    newValue = random.randint(0, 84)
    return Primitives.Variable(value), Primitives.Const(newValue)
  elif value == 'ALBUMIN':
    newValue = random.randint(0, 29)
    return Primitives.Variable(value), Primitives.Const(newValue)
  elif value == 'PROTIME':
    newValue = random.randint(0, 44)
    return Primitives.Variable(value), Primitives.Const(newValue)
  else:
    newValue = random.randint(0, 1)
    return Primitives.Variable(value), Primitives.Const(newValue)

def countTreeComplexity(tree: Primitives.Node):
  if tree == None:
    return 0
  
  if isinstance(tree, Primitives.Terminal):
    return 1
  
  complexity = 0

  for i in range(tree.arity):
    argName = f'arg{i+1}'
    complexity += countTreeComplexity(getattr(tree, argName))    
  
  return complexity

def calculateComplexityFrequency(population: List[Primitives.Node]):
  complexity = [countTreeComplexity(indiv) for indiv in population]
  frequency = Counter(complexity)

  return frequency

def getDepth(tree: Primitives.Node | None):
  if not tree:
    return -1
  
  if isinstance(tree, Primitives.Terminal):
    return 0

  depths = []

  for i in range(tree.arity):
    argName = f'arg{i+1}'
    depths.append(getDepth(getattr(tree, argName)))

  return max(depths)

def getDepthByIndex(index: int, tree: Primitives.Node):
  if not isinstance(tree, Primitives.Node):
    raise AssertionError('Tree param not a primitive node')

  if tree.index == None:
    raise AssertionError('Missing tree index')
  
  if tree.index == index:
    return 0
  
  if isinstance(tree, Primitives.Terminal):
    return None
  
  for i in range(tree.arity):
    argName = f'arg{i+1}'
    treeDepth = getDepthByIndex(index, getattr(tree, argName))

    if treeDepth == None:
      continue

    return treeDepth + 1

  return None

def printPopulation(population: List[Primitives.Node]):
  """
  Print the expression each indivudual represents
  """

  for i, individual in enumerate(population):
    print(f'ind {i}: {individual}')

  print()


###############################################################
# Population Methods

ComparisonFunctions = [
  Primitives.EqualTo,
  Primitives.GreaterOrEqalThan,
  Primitives.GreaterThan,
  Primitives.LessOrEqualThan,
  Primitives.LessThan
]

def generateFullTree(terminalSet: List[Primitives.Node], functionSet: List[Primitives.Node], depth, maxDepth = 1):
  if depth == 1:
    value = random.choice(terminalSet)
    treeNode, thresholdNode = createTerminalPair(value)

    function: Primitives.Node = random.choice(ComparisonFunctions)
    root = function(treeNode, thresholdNode)
    
    return root

  function: Primitives.Node = random.choice(functionSet)
  childNodes = []

  for _ in range(function.arity):
    childNodes.append(generateFullTree(terminalSet, functionSet, depth-1))

  root = function(*childNodes)

  return root

def generateGrowTree(terminalSet: List[Primitives.Node], functionSet: List[Primitives.Node], depth = 0, maxDepth = 1):
  if depth == 1:
    value = random.choice(terminalSet)
    treeNode, thresholdNode = createTerminalPair(value)

    function: Primitives.Node = random.choice(ComparisonFunctions)
    root = function(treeNode, thresholdNode)
    
    return root

  function: Primitives.Node = random.choice(functionSet)

  childNodes = []
  for _ in range(function.arity):
    randomDepth = random.randint(1, max(depth - 1, 1))
    childNodes.append(generateGrowTree(terminalSet, functionSet, randomDepth))

  root = function(*childNodes)

  return root

def generatePopulation(terminalSet: List[Primitives.Node], functionSet: List[Primitives.Node], populationSize, method = GenerationMetods.Full, maxDepth = 2, minDepth = 1):
  population = []

  if method == GenerationMetods.Full:
    population.extend([generateFullTree(terminalSet, functionSet, maxDepth, maxDepth) for _ in range(populationSize)])
  elif method == GenerationMetods.Grow:
    population.extend([generateGrowTree(terminalSet, functionSet, maxDepth, maxDepth) for _ in range(populationSize)])
  else:
    treesPerDepth = math.floor(populationSize / (maxDepth - 1))

    for depth in range(minDepth, maxDepth + 1):
      population.extend([generateFullTree(terminalSet, functionSet, depth) for _ in range(math.floor(treesPerDepth / 2))])
      population.extend([generateGrowTree(terminalSet, functionSet, depth) for _ in range(math.ceil(treesPerDepth / 2))])

  while len(population) < populationSize:
    population.append(generateFullTree(terminalSet, functionSet, minDepth))

  random.shuffle(population)

  print(f'Generated initial population. Size = {len(population)}')

  return population

###############################################################
# Fitness

def createFitnessCaseDict(features, inputs):
  return dict(zip(features, inputs))

def getClass(result):
  return 2 if result else 1

@cache
def calculateHitsRatio(individual: Primitives.Node, fitnessCases: FitnessCases):
  predictions = np.array([
    getClass(individual.evaluate(createFitnessCaseDict(fitnessCases.features.values, case))) for case in fitnessCases.inputs
  ])
  targetValues = np.array(fitnessCases.outputs)

  hitCount = sum([a == b for a,b in zip(predictions, targetValues)])

  return hitCount / len(fitnessCases.outputs)

def countTrivialComparisons(node):
  """
  Count trivial comparisons
  """
  trivialComparisons = 0

  if isinstance(node, Primitives.Terminal):
    return 0

  if isinstance(node, Primitives.LogicOperator) or isinstance(node, Primitives.ComparisonOperator):
    if isinstance(node.arg1, Primitives.Const) and isinstance(node.arg2, Primitives.Const):
      return 1

  if hasattr(node, "arg1"):
    trivialComparisons += countTrivialComparisons(node.arg1)

  if hasattr(node, "arg2"):
    trivialComparisons += countTrivialComparisons(node.arg2)

  if hasattr(node, "arg3"):
    trivialComparisons += countTrivialComparisons(node.arg3)
  
  return trivialComparisons 

def calculateFitness(population: List[Primitives.Node], fitnessCases: FitnessCases):
  penaltyWeight = 0.01
  treeFitnessPairs = [(calculateHitsRatio(indiv, fitnessCases), countTrivialComparisons(indiv)) for indiv in population]

  return [accuracy - (trivComp*penaltyWeight) for accuracy, trivComp in treeFitnessPairs]

def printPopulationFitness(population: List[Primitives.Node], fitness = []):
  """
  Print the expression each indivudual represents
  """

  i = 0
  for individual, fit in zip(population, fitness):
    print(f'ind {i}: {fit} => {individual}')
    i += 1
    print()

  print('===================================')
  print()

###############################################################
# Selection Methods

def fitnessProportianateSelection(population: List[Primitives.Node], fitness: List):
  populationSize = len(population)
  selectionOccurances = [round(nf * populationSize) for nf in fitness]

  matingPool = []

  for i, occurances in enumerate(selectionOccurances):
    matingPool.extend([population[i] for _ in range(occurances)])
  
  parentIndex = random.randint(0, len(matingPool) - 1)
  return matingPool[parentIndex]

def tournamentSelection(population: List[Primitives.Node], fitness: List, tournamentSize = 2):

  indivIndexes = [random.randint(0, len(population) - 1) for _ in range(tournamentSize)]

  tournament = [population[index] for index in indivIndexes]
  tournamentFitness = [fitness[index] for index in indivIndexes]

  parentIndex = tournamentFitness.index(max(tournamentFitness))

  return tournament[parentIndex]

def selectParent(population: List[Primitives.Node], fitness: List, selectionMethod: SelectionMethod, tournamentSize = 2):
  if selectionMethod == SelectionMethod.FitnessProportianateSelection:
    return fitnessProportianateSelection(population, fitness)

  return tournamentSelection(population, fitness, tournamentSize)

############################################
# Genetic Operator Helpers

def replaceSubtree(tree: Primitives.Node, replacement: Primitives.Node, targetInddex):
  if tree.index == targetInddex:
    return replacement

  if isinstance(tree, Primitives.Terminal):
    return tree
    
  for i in range(tree.arity):
    argName = f'arg{i+1}'
    subTree = getattr(tree, argName)
    setattr(tree, argName, replaceSubtree(subTree, replacement, targetInddex))

  return tree

def getTreeNodeByIndex(tree: Primitives.Node, nodeIndex):
  if not isinstance(tree, Primitives.Node):
    return None

  if tree.index == nodeIndex:
    return tree
  
  for i in range(tree.arity):
    argName = f'arg{i+1}'
    node = getTreeNodeByIndex(getattr(tree, argName), nodeIndex)

    if not node:
      continue

    return node

  return None

############################################
# Genetic Operators

def crossover(parent1: Primitives.Node, parent2: Primitives.Node, crossoverRate: float, maxDepth):
  parent1Copy = copy.deepcopy(parent1)
  parent2Copy = copy.deepcopy(parent2)
  
  if random.random() > crossoverRate:
    return parent1Copy, parent2Copy

  maxIndex1 = parent1Copy.setIndex()
  maxIndex2 = parent2Copy.setIndex()

  crossoverPoint1 = random.randint(1, maxIndex1)
  crossoverPoint2 = random.randint(1, maxIndex2)

  temp1 = getTreeNodeByIndex(parent1Copy, crossoverPoint1)
  temp2 = getTreeNodeByIndex(parent2Copy, crossoverPoint2)

  offspring1 = replaceSubtree(parent1Copy, temp2, crossoverPoint1)
  offspring2 = replaceSubtree(parent2Copy, temp1, crossoverPoint2)


  if (getDepth(offspring1) > maxDepth or getDepth(offspring2) > maxDepth):
    return copy.deepcopy(parent1), copy.deepcopy(parent2)

  return offspring1, offspring2

def mutate(functionSet: List, terminalSet: List, parent: Primitives.Node, mutationRate: float, maxDepth: int):
  parentCopy = copy.deepcopy(parent)

  if random.random() > mutationRate:
    return parentCopy

  maxIndex = parentCopy.setIndex()
  cutoffIndex = random.randint(1, max(maxIndex, 1))
  cutoffDepth = getDepthByIndex(cutoffIndex, parentCopy)

  replacementDepth = max(maxDepth - cutoffDepth, 0)
  replacementTree = generateGrowTree(terminalSet, functionSet, replacementDepth)
  mutatedParent = replaceSubtree(parentCopy, replacementTree, cutoffIndex)

  return mutatedParent

############################################
# Analytics Helpers

def countTreeNodes(tree: Primitives.Node): 
  if isinstance(tree, Primitives.Terminal):
    return 1

  count = 1

  for i in range(tree.arity):
    argName = f'arg{i+1}'
    count += countTreeNodes(getattr(tree,argName))

  return count

############################################
# Analytics

def calculateAverageFitness(fitness: List[float]):
  return sum(fitness) / len(fitness)

def calculateAverageTreeComplexity(population: List[Primitives.Node]):
  complexity = [countTreeNodes(indiv) for indiv in population]

  return sum(complexity) / len(complexity)
  
def calculatePopulationVariety(population: List[Primitives.Node]):
  populationSignatures = [str(indiv) for indiv in population]

  individualCounts = {}

  for sig in populationSignatures:
    if sig in individualCounts:
      individualCounts[sig] += 1
    else:
      individualCounts[sig] = 1

  duplicates = 0
  for value in individualCounts.values():
    if value > 1:
      duplicates += value - 1 

  variety = (len(population) - duplicates) / len(population)

  return variety

###############################################################
# Genetic Algorithm: Regression
###############################################################

def evolveRegressor(gpParams: GP_Params, dataset = './hepatitis.tsv', seed = 0):

  plotPrefix = dataset.split('.')[0] + '_' + str(seed)

  train, test  = getDataFrame(dataset)

  x_train, y_train = train
  x_test, y_test = test

  fitnessCases_Train = FitnessCases(x_train.values, y_train.values, x_train.columns)
  fitnessCases_Test = FitnessCases(x_test.values, y_test.values, x_test.columns)

  population = generatePopulation(
    terminalSet=gpParams.TerminalSet,
    functionSet=gpParams.FunctionSet,
    populationSize=gpParams.PopulationSize,
    minDepth=gpParams.MinTreeDepth,
    maxDepth=gpParams.MaxTreeDepth,
    method=GenerationMetods.Ramped
  )

  fitness = calculateFitness(population, fitnessCases_Train)

  printPopulationFitness(population, fitness)

  # variety = calculatePopulationVariety(population)
  populationComplexity = []
  populationFitness = []
  populationVariety = []
  bestFitness = []

  # complexity = [countTreeComplexity(indiv) for indiv in population]
  frequency = calculateComplexityFrequency(population)

  plotBarGraph(frequency.keys(), frequency.values(), 'Initial Population Tree Complexity Distribution ', 'Complexity (NO. nodes)', 'Frequency (NO. trees)', prefix=plotPrefix)

  # plotHistogram(complexity, "Initial population complexity Histogram", "Individual", 'Complexity', prefix=plotPrefix)

  tournamentSize = 3

  fitness = calculateFitness(population, fitnessCases_Train)

  for gen in range(gpParams.Generations):
    print(f'Gen {gen}')

    newPopulation = []

    if gpParams.ReproductionRate < random.random():
      reproducedParent = selectParent(population, fitness, SelectionMethod.TournamentSelection, tournamentSize)
      newPopulation.append(copy.deepcopy(reproducedParent))

    for _ in range(gpParams.PopulationSize // 2):
      parent1 = selectParent(population, fitness, SelectionMethod.TournamentSelection, tournamentSize)
      parent2 = selectParent(population, fitness, SelectionMethod.TournamentSelection, tournamentSize)

      offspring1, offspring2 = crossover(parent1, parent2, gpParams.CrossoverRate, gpParams.MaxTreeDepth)

      newPopulation.extend([
        mutate(gpParams.FunctionSet, gpParams.TerminalSet, offspring1, gpParams.MutationRate, gpParams.MaxTreeDepth), 
        mutate(gpParams.FunctionSet, gpParams.TerminalSet, offspring2, gpParams.MutationRate, gpParams.MaxTreeDepth)
      ])

    fitness = calculateFitness(newPopulation, fitnessCases_Train)
    totalFitness = sum(fitness)
    print(f'population size = {len(newPopulation)}')
    print(f'totalFitness = {totalFitness}')
    populationFitness.append(totalFitness)

    bestIndex = fitness.index(max(fitness))
    bestIndiv: Primitives.Node = newPopulation[bestIndex]
    bestIndivFitness = fitness[bestIndex]
    
    # bestFitness.append(fitness[bestIndex])
    bestFitness.append(bestIndivFitness)

    # variety = calculatePopulationVariety(newPopulation)
    # complexity = calculateAverageTreeComplexity(newPopulation)
    averageFitness = calculateAverageFitness(fitness)

    # populationComplexity.append(complexity)
    # populationVariety.append(variety)

    print(f'AverageFitness = {averageFitness}')
    # print(f'Population Complexity = {complexity}')
    # print(f'Population Variety = {variety}')
    print(f'Best Individual = {bestIndiv} ;')
    print(f'Best Individual Fitness = {bestIndivFitness};')
    print()

    population = newPopulation

  plotFitnessPerGeneration(populationFitness, title='Population fitness for each Generation', prefix=plotPrefix)
  plotFitnessPerGeneration(bestFitness, title='Fitness of the best individual in each Generation', prefix=plotPrefix)

  # plotHistogram(populationComplexity, 'Structural Complexity Histogram', 'Generations', 'Structural Complexity', prefix=plotPrefix)
  # plotHistogram(populationFitness, 'Standardized Fitness Histogram', 'Generations', 'Standardized Fitness', prefix=plotPrefix)
  # plotHistogram(populationVariety, 'Variety Histogram', 'Generations', 'Variety Fitness', prefix=plotPrefix)

  finalFitness = calculateFitness(population, fitnessCases_Test)

  return population, finalFitness, bestIndivFitness

###############################################################
# Program

FUNCTION_SET = [
  Primitives.IF_ELSE_THEN,
  Primitives.AndOperator,
  Primitives.EqualTo,
  Primitives.GreaterThan,
  Primitives.GreaterOrEqalThan,
  Primitives.LessThan,
  Primitives.LessOrEqualThan,
  Primitives.OROperator
]

TERMINAL_SET = ['AGE','SEX','STEROID','ANTIVIRALS','FATIGUE','MALAISE','ANOREXIA','LIVER BIG','LIVER FIRM','SPLEEN PALPABLE','SPIDERS','ASCITES', 'VARICES', 'BILIRUBIN', 'ALK PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME', 'HISTOLOGY']

parser = argparse.ArgumentParser(description='Argument parser for a structure-based GP')
parser.add_argument('--seed', type=int, default=random.randint(100, 200), help='Psuedo-number generator seed.')

if __name__ == '__main__':
  args = parser.parse_args()

  seed = args.seed
  # seed = 3
  random.seed(seed)

  print(f'Random Seed = {seed}')

  print()
  startTime = time.time()

  gpParams = GP_Params(
    terminalSet=TERMINAL_SET,
    functionSet=FUNCTION_SET,
    populationSize=100,
    maxTreeDepth=7,
    minTreeDepth=3,
    crossoverRate=0.6,
    mutationRate=0.2,
    generations=100,
    reproductionRate=0.2
  )

  topIndividuals, populationFitness, bestIndivFitness = evolveRegressor(gpParams, dataset='hepatitis.tsv', seed=seed)

  endTime = time.time()
  duration = round(endTime - startTime, 2)
  # printPopulationFitness(topIndividuals, topFitness)

  print()
  print(f'Random Seed = {seed}')
  print(f'duration = {duration}s')
  print(f'bestIndivFitness = {bestIndivFitness}')
