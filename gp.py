import argparse
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
import pandas as pd
import csv
from concurrent.futures import ProcessPoolExecutor

import primitives as Primitives
from utils import Logger

##########################################
# Data processing
##########################################

def load_data(filename: str, lag_hours=1):
  """Loads CSV data and parses dates."""
  df = pd.read_csv(filename, parse_dates=['utc_timestamp'], dayfirst=True)
  df = df.sort_values('utc_timestamp')
  
  Logger.debug(df.info)

  expected_delta = pd.Timedelta(minutes=15)
  is_regular = (df['utc_timestamp'].diff().iloc[1:] == expected_delta).all()

  if not is_regular:
    print("Warning: Gaps detected. Use df.resample('15T').interpolate() first.")
    raise Exception("Data is not regular")

  target_col = 'Electricity_load'

  data = df[[target_col]].copy()

  for i in range(1, (lag_hours * 4) + 1):
    data[f'load_t-{i}'] = data[target_col].shift(i)

  # data['lag_24h'] = data[target_col].shift(96)
  # data['lag_7d'] = data[target_col].shift(672)
  # Remove rows with NaN values created by shifting
  data = data.dropna()
  Logger.debug(data.info)

  return data

def normalize_window(window_data):
  """Applies Z-Score standardization to numeric columns in the window."""
  df_norm = window_data.copy()
  numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
  # Avoid division by zero if a column is constant
  df_norm[numeric_cols] = (df_norm[numeric_cols] - df_norm[numeric_cols].mean()) / df_norm[numeric_cols].std().replace(0, 1)
  return df_norm

##########################################
# Visualizations
##########################################

def plot_metrics(history, seed: int):
  """Plots the fitness metrics across generations and saves the figure."""
  gens = [h['gen'] for h in history]
  best = [h['best'] for h in history]
  worst = [h['worst'] for h in history]
  avg = [h['avg'] for h in history]
  std = [h['std'] for h in history]

  plt.figure(figsize=(10, 6))
  
  # Plot Average with Std Dev shading
  plt.fill_between(gens, np.array(avg) - np.array(std), np.array(avg) + np.array(std), alpha=0.2, color='gray', label='Std Dev')
  plt.plot(gens, avg, label='Average Fitness', color='black', linewidth=2)
  
  # Plot Best and Worst
  plt.plot(gens, best, label='Best Fitness', color='green', marker='o')
  plt.plot(gens, worst, label='Worst Fitness', color='red', linestyle='--', alpha=0.5)

  plt.title('Evolution of Population Fitness (MAPE)')
  plt.xlabel('Generation')
  plt.ylabel('Fitness (Lower is Better)')
  plt.yscale('log')
  
  if len(gens) > 1:
    plt.xticks(np.linspace(min(gens), max(gens), 10, dtype=int))
  
  plt.legend()
  plt.grid(True, which='major', linestyle='--', alpha=0.5)

  # Save the plot
  if not os.path.exists('out'):
    os.makedirs('out')
  
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  filename = f'out/{timestamp}_gp_fitness_{seed}.png'
  plt.savefig(filename)
  plt.close()
  Logger.info(f'Fitness evolution plot saved to {filename}')

def plot_best_fitness(history, seed: int):
  """Plots the best fitness across generations and saves the figure."""
  gens = [h['gen'] for h in history]
  best = [h['best'] for h in history]

  plt.figure(figsize=(10, 6))
  
  # Plot Best
  plt.plot(gens, best, label='Best Fitness', color='green', marker='o')

  plt.title('Evolution of Best Fitness (MAPE)')
  plt.xlabel('Generation')
  plt.ylabel('Fitness (Lower is Better)')
  plt.yscale('log')
  
  if len(gens) > 1:
    plt.xticks(np.linspace(min(gens), max(gens), 10, dtype=int))
  
  plt.legend()
  plt.grid(True, which='major', linestyle='--', alpha=0.5)

  # Save the plot
  if not os.path.exists('out'):
    os.makedirs('out')
  
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  filename = f'out/{timestamp}_gp_best_fitness_{seed}.png'
  plt.savefig(filename)
  plt.close()
  Logger.info(f'Best fitness plot saved to {filename}')

def plot_structural_metrics(history, seed: int):
  """Plots the structural metrics (size and depth) and saves the figure."""
  gens = [h['gen'] for h in history]
  avg_size = [h['avg_tree_size'] for h in history]
  max_size = [h['max_tree_size'] for h in history]
  min_size = [h['min_tree_size'] for h in history]
  avg_depth = [h['avg_tree_depth'] for h in history]

  fig, ax1 = plt.subplots(figsize=(10, 6))

  color = 'tab:blue'
  ax1.set_xlabel('Generation')
  ax1.set_ylabel('Tree Size (Nodes)', color=color)
  ax1.plot(gens, avg_size, color=color, label='Avg Size', linewidth=2)
  ax1.fill_between(gens, min_size, max_size, color=color, alpha=0.1, label='Size Range')
  ax1.tick_params(axis='y', labelcolor=color)
  ax1.grid(True, which='both', linestyle='--', alpha=0.5)

  ax2 = ax1.twinx()
  color = 'tab:red'
  ax2.set_ylabel('Avg Tree Depth', color=color)
  ax2.plot(gens, avg_depth, color=color, label='Avg Depth', linestyle='--')
  ax2.tick_params(axis='y', labelcolor=color)

  h1, l1 = ax1.get_legend_handles_labels()
  h2, l2 = ax2.get_legend_handles_labels()

  if len(gens) > 1:
    plt.xticks(np.linspace(min(gens), max(gens), 10, dtype=int))
  
  plt.legend(h1 + h2, l1 + l2)
  plt.title('Evolution of Population Structure')
  fig.tight_layout()
  
  # Save the plot
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  filename = f'out/{timestamp}_gp_structure_{seed}.png'
  plt.savefig(filename)
  plt.close()
  Logger.info(f'Structural evolution plot saved to {filename}')

def create_matric(generation: int, fitness: list[float], pop: list[Primitives.Node]):
  sizes = [ind.node_count for ind in pop]
  depths = [ind.depth for ind in pop]

  metrics = {
    'gen': generation,
    'size': len(pop),
    'best': np.min(fitness),
    'worst': np.max(fitness),
    'avg': np.mean(fitness),
    'std': np.std(fitness),
    'avg_tree_size': np.mean(sizes),
    'max_tree_size': np.max(sizes),
    'min_tree_size': np.min(sizes),
    'avg_tree_depth': np.mean(depths)
  }

  return metrics

def plot_tree(tree: Primitives.Node, seed: int):
  """
  Plots an individual tree and saves it as a PNG.
  """
  depth = tree.depth
  width = min(100, max(10, (2 ** depth) * 0.8))
  height = min(50, max(6, (depth + 1) * 1.5))
  
  fig, ax = plt.subplots(figsize=(width, height))
  ax.axis('off')
  
  def draw_node(node, x, y, dx, dy):
    if hasattr(node, 'left') and hasattr(node, 'right'):
      ax.plot([x, x - dx], [y, y - dy], 'k-', lw=1.5, zorder=1)
      ax.plot([x, x + dx], [y, y - dy], 'k-', lw=1.5, zorder=1)

      draw_node(node.left, x - dx, y - dy, dx / 2, dy)
      draw_node(node.right, x + dx, y - dy, dx / 2, dy)
      
    label = str(node.value) if hasattr(node, 'value') else str(node)
    
    ax.text(x, y, label, ha='center', va='center',
      bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=1'),
      fontsize=12,
      zorder=2
    )

  draw_node(tree, 0, 0, width / 2, 1)
  
  out_dir = 'out/trees'
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  
  filename = f'{out_dir}/{seed}_individual_tree.png'

  plt.savefig(filename, bbox_inches='tight')
  plt.close()
  Logger.info(f'Individual tree plot saved to {filename}')

  return filename
  
##########################################
# Population methods
##########################################

def get_type_of_primitive(p):
  if hasattr(p, 'ret_type'):
    return p.ret_type
  return 'float'

def generate_random_tree_grow(max_dep, expected_type='float'):
  valid_terms = [t for t in Terminals if get_type_of_primitive(t) == expected_type]
  if max_dep <= 0:
    term = random.choice(valid_terms)
    if term == Primitives.ConstantValue:
      return Primitives.Constant()
    else:
      return Primitives.Variable(term)
  
  valid_funcs = [f for f in Functions if get_type_of_primitive(f) == expected_type]
  choice = random.choice(valid_funcs + valid_terms)

  if choice in valid_funcs:
    arg_types = getattr(choice, 'arg_types', ('float', 'float'))
    left = generate_random_tree_grow(max_dep - 1, expected_type=arg_types[0])
    right = generate_random_tree_grow(max_dep - 1, expected_type=arg_types[1])
    return choice(left, right)
  else:
    if choice == Primitives.ConstantValue:
      return Primitives.Constant()
    else:
      return Primitives.Variable(choice)

def generate_random_tree_full(max_depth, expected_type='float'):
  valid_terms = [t for t in Terminals if get_type_of_primitive(t) == expected_type]
  if max_depth <= 0:
    term = random.choice(valid_terms)
    if term == Primitives.ConstantValue:
      return Primitives.Constant()
    else:
      return Primitives.Variable(term)
  else:
    valid_funcs = [f for f in Functions if get_type_of_primitive(f) == expected_type]
    func = random.choice(valid_funcs)
    arg_types = getattr(func, 'arg_types', ('float', 'float'))
    left = generate_random_tree_full(max_depth - 1, expected_type=arg_types[0])
    right = generate_random_tree_full(max_depth - 1, expected_type=arg_types[1])
    return func(left, right)

def generate_random_tree(max_depth, gen_method: str = 'full', expected_type='float'):
  if max_depth < 1:
      max_depth = 1

  valid_funcs = [f for f in Functions if get_type_of_primitive(f) == expected_type]
  func = random.choice(valid_funcs)
  arg_types = getattr(func, 'arg_types', ('float', 'float'))
  
  if gen_method == 'full':
    left = generate_random_tree_full(max_depth - 1, expected_type=arg_types[0])
    right = generate_random_tree_full(max_depth - 1, expected_type=arg_types[1])
  elif gen_method == 'grow':
    left = generate_random_tree_grow(max_depth - 1, expected_type=arg_types[0])
    right = generate_random_tree_grow(max_depth - 1, expected_type=arg_types[1])
  elif gen_method == 'half-and-half':
    if random.random() < 0.5:
      left = generate_random_tree_full(max_depth - 1, expected_type=arg_types[0])
      right = generate_random_tree_full(max_depth - 1, expected_type=arg_types[1])
    else:
      left = generate_random_tree_grow(max_depth - 1, expected_type=arg_types[0])
      right = generate_random_tree_grow(max_depth - 1, expected_type=arg_types[1])
  else:
    raise ValueError("Invalid generation method")
    
  return func(left, right)

def init_pop(pop_size, max_depth, gen_method: str = 'ramped-half-and-half'):
  if max_depth < 1:
    raise ValueError("max_depth must be greater than 0")

  pop = []
  if gen_method == 'ramped-half-and-half':
    depths = list(range(1, max_depth + 1))
    trees_per_depth = pop_size // len(depths)
    
    for depth in depths:
      for i in range(trees_per_depth):
        method = 'full' if i < trees_per_depth // 2 else 'grow'
        left = generate_random_tree(depth - 1, method, expected_type='float')
        right = generate_random_tree(depth - 1, method, expected_type='float')
        pop.append(Primitives.StructuredRoot(left, right))
        
    while len(pop) < pop_size:
      method = random.choice(['full', 'grow'])
      left = generate_random_tree(max_depth - 1, method, expected_type='float')
      right = generate_random_tree(max_depth - 1, method, expected_type='float')
      pop.append(Primitives.StructuredRoot(left, right))
  else:
    for _ in range(pop_size):
      left = generate_random_tree(max_depth - 1, gen_method, expected_type='float')
      right = generate_random_tree(max_depth - 1, gen_method, expected_type='float')
      pop.append(Primitives.StructuredRoot(left, right))
      
  return pop

def print_population(pop, fitness=None):
  Logger.info('='*40)
  Logger.info(f'Population Size: {len(pop)}')

  for i, tree in enumerate(pop):
    individual = ''
    exp = str(tree)
    
    if fitness is not None:
      individual += f'{exp}: {fitness[i]} '
    else:
      individual = exp

    Logger.info(f'{i}: {individual}')
  
  Logger.info('')

##########################################
# Fitness
##########################################

def raw_fitness(tree, data):
  """Calculates the Mean Absolute Percentage Error (MAPE) of the tree on the given data."""
  sum_error = 0
  target_col = 'Electricity_load'
  total_rows = len(data)

  for i, (idx, row) in enumerate(data.iterrows()):

    target = row[target_col]
    # Prepare kwargs for evaluation from the row, excluding the target column
    inputs = row.drop(labels=[target_col]).to_dict()
    
    try:
      result = tree.eval(**inputs)
      sum_error += abs((target - result) / max(abs(target), 1e-8))
    except (ZeroDivisionError, OverflowError, ValueError):
      sum_error += 10
    except Exception:
      sum_error += 10

  return sum_error / total_rows

def _eval_worker(args):
  idx, individual, cases = args
  indivError = raw_fitness(individual, cases)
  return idx, indivError

def calculate_fitness(pop, cases):
  fitness_results = [0] * len(pop)
  Logger.info("Evaluating Population (Multiprocessing)...")
  
  num_cpu = max(1, os.cpu_count() - 1)
  with ProcessPoolExecutor(max_workers=num_cpu) as executor:
    args_list = [(i, ind, cases) for i, ind in enumerate(pop)]
    total_evals = len(pop)
    for count, (idx, indivError) in enumerate(executor.map(_eval_worker, args_list), 1):
      pop[idx].fitness = indivError
      fitness_results[idx] = indivError
      Logger.progress_log(f'Evaluating Population Progress: {(count / total_evals) * 100:.1f}% ({count}/{total_evals})')
      Logger.debug(f'{idx}: {pop[idx]} Fitness: {pop[idx].fitness}')
  
  print('') # Clear progress bar newline
  return fitness_results


##########################################
# Selection methods
##########################################
def tournament_selection(population, fitness, tournament_size: int = 2, selection_count: int = 2):
  winners = []

  pop_indices = list(range(len(population)))
  Logger.debug(f'pop_indices: {pop_indices}')

  for _ in range(selection_count):
    participants = random.sample(pop_indices, min(tournament_size, len(population)))
    Logger.debug(f'participants: {participants}')

    best_idx = participants[0]
    for idx in participants[1:]:
      if fitness[idx] < fitness[best_idx]:
        best_idx = idx
    
    winners.append(population[best_idx])

  return winners

##########################################
# Structural Similarity
##########################################

def nodes_match(n1, n2):
  if type(n1) != type(n2):
    return False
  if isinstance(n1, Primitives.Variable):
    return n1.value == n2.value
  return True

def calculate_similarity(node1, node2, cutoff_depth, current_depth=0):
  """
  Calculates the Global Index (GI) and Local Index (LI) for structural similarity.
  GI: function nodes in common between root and cutoff_depth.
  LI: function and terminal nodes in common in subtrees after cutoff_depth.
  """
  gi = 0
  li = 0
  
  if node1 is None or node2 is None:
    return gi, li
    
  if nodes_match(node1, node2):
    is_func = isinstance(node1, Primitives.Function)
    is_term = isinstance(node1, Primitives.Terminal)
    
    if current_depth <= cutoff_depth:
      if is_func:
        gi += 1
    else:
      if is_func or is_term:
        li += 1
        
    if is_func:
      l_gi, l_li = calculate_similarity(node1.left, node2.left, cutoff_depth, current_depth + 1)
      r_gi, r_li = calculate_similarity(node1.right, node2.right, cutoff_depth, current_depth + 1)
      gi += l_gi + r_gi
      li += l_li + r_li
      
  return gi, li

##########################################
# Genetic Operators
##########################################

def mutation(parent: Primitives.Node, max_depth: int, mutation_rate: float = 0.5):
  """Performs size-fair subtree mutation with up to 3 retries to stay within max_depth."""
  if random.random() > mutation_rate:
    return parent

  for _ in range(3):
    child = parent.copy()
    
    if child.node_count <= 1:
      return parent 
      
    idx = random.randint(1, child.node_count - 1)
    target_type = child.get_subtree(idx).ret_type
    
    mutant_subtree = generate_random_tree(max_depth=max(1, max_depth // 2), gen_method='grow', expected_type=target_type)
    
    child = child.replace_subtree(idx, mutant_subtree)
    
    if child.depth <= max_depth:
      return child
  
  return parent

def crossover(
  parent1: Primitives.Node,
  parent2: Primitives.Node,
  max_depth: int,
  crossover_rate: float = 0.5
):
  """Performs robust size-fair subtree crossover ensuring children stay within max_depth."""
  if random.random() > crossover_rate:
    return parent1, parent2
  
  child1 = parent1.copy()
  child2 = parent2.copy()

  # Select a random node in child1 (skipping root index 0)
  if child1.node_count <= 1:
    raise Exception(f"Child1 has no nodes to swap: {child1}")
  
  idx1 = random.randint(1, child1.node_count - 1)
  d1 = child1.get_relative_depth(idx1)
  capacity1 = max_depth - d1
  
  subtree1 = child1.get_subtree(idx1)
  subtree1_depth = subtree1.depth
  s1_ret_type = subtree1.ret_type

  valid_indices2 = []
  
  for idx in range(1, child2.node_count):
    subtree2 = child2.get_subtree(idx)
    s2_depth = subtree2.depth
    d2 = child2.get_relative_depth(idx)
    s2_ret_type = subtree2.ret_type
    
    if s2_depth <= capacity1 and subtree1_depth <= (max_depth - d2) and s1_ret_type == s2_ret_type:
      valid_indices2.append(idx)
          
  if not valid_indices2:
    return parent1, parent2
  
  idx2 = random.choice(valid_indices2)

  # Get copies of the nodes to swap
  node1 = child1.get_subtree(idx1).copy()
  node2 = child2.get_subtree(idx2).copy()

  # Perform the swap
  child1 = child1.replace_subtree(idx1, node2)
  child2 = child2.replace_subtree(idx2, node1)

  return child1, child2

##########################################
# GP: Params
##########################################

# Terminals = [ 'x', 'y', Primitives.ConstantValue ]
Terminals = [f'load_t-{i}' for i in range(1, 9)] + [Primitives.ConstantValue]
# Terminals.extend(['lag_24h', 'lag_7d'])

Functions = [
  Primitives.Add,
  Primitives.Sub,
  Primitives.Mul,
  Primitives.Div,
  Primitives.Max,
  Primitives.Min
]

FitnessCases = [
  (1, 1, 2),
  (1, 2, 3),
  (2, 1, 3),
  (2, 2, 4)
]

##########################################
# GP
##########################################

parser = argparse.ArgumentParser()
parser.add_argument("--pop-size", type=int, default=100)
parser.add_argument("--max-depth", type=int, default=5)
parser.add_argument("--seed", type=int, default=random.randint(0, 100))
parser.add_argument("--gens", type=int, default=70)
parser.add_argument("--crossover-rate", type=float, default=0.8)
parser.add_argument("--mutation-rate", type=float, default=0.19)
parser.add_argument("--reproduction-rate", type=float, default=0.01)
parser.add_argument("--tournament-size", type=int, default=2)
parser.add_argument("--window-size", type=int, default=30)

args = parser.parse_args()

Window_Size = 4 * 24 * args.window_size # 14 Days

if __name__ == "__main__":
  print(args)
  random.seed(args.seed)
  history = []

  data = load_data('data/Residential_Energy_Dataset_UK- 2014-2020.csv', lag_hours=2)

  pop = init_pop(args.pop_size, args.max_depth, gen_method='ramped-half-and-half')
  start_indx = 0
  end_idx = start_indx + Window_Size
  sub_data = data.iloc[start_indx:end_idx]
  fitness = calculate_fitness(pop, sub_data)

  Logger.info('Init population')
  print_population(pop, fitness)

  # Collect metrics for current generation
  metrics = create_matric(0, fitness, pop)
  history.append(metrics)

  start_time = time.time()
  for i in range(args.gens-1):
    
    Logger.info(f'Generation {i+1}:')
    
    new_population = []
  
    num_reproduction = max(1, int(args.pop_size * args.reproduction_rate))
    num_crossover = int(args.pop_size * args.crossover_rate)
    if num_crossover % 2 != 0:
      num_crossover -= 1 # Keep even to avoid overshooting
      
    # 1. Reproduction Phase (True Elitism)
    elite_indices = sorted(range(len(fitness)), key=lambda k: fitness[k])[:num_reproduction]
    for idx in elite_indices:
      if len(new_population) < args.pop_size:
        new_population.append(pop[idx].copy())
      
    # 2. Crossover Phase
    target_crossover = len(new_population) + num_crossover
    while len(new_population) < target_crossover and len(new_population) < args.pop_size - 1:
      parent1 = tournament_selection(pop, fitness, tournament_size=args.tournament_size, selection_count=1)[0]
      parent2 = tournament_selection(pop, fitness, tournament_size=args.tournament_size, selection_count=1)[0]
      
      # Use Structural Similarity to prevent redundant crossover
      cutoff = max(1, args.max_depth // 2)
      gi, li = calculate_similarity(parent1, parent2, cutoff_depth=cutoff)
      
      attempts = 0
      while gi + li >= min(parent1.node_count, parent2.node_count) * 0.8 and attempts < 3:
        parent2 = tournament_selection(pop, fitness, tournament_size=args.tournament_size, selection_count=1)[0]
        gi, li = calculate_similarity(parent1, parent2, cutoff_depth=cutoff)
        attempts += 1

      child1, child2 = crossover(parent1, parent2, max_depth=args.max_depth, crossover_rate=1.0) # Forced execution
      new_population.append(child1)
      if len(new_population) < args.pop_size:
        new_population.append(child2)

    # 3. Mutation Phase (Fills all remaining slots to ensure exact pop_size)
    while len(new_population) < args.pop_size:
      mut_parent = tournament_selection(
        pop, fitness, tournament_size=args.tournament_size, selection_count=1
      )[0]
      mut_child = mutation(mut_parent, args.max_depth, mutation_rate=1.0) # Forced execution
      new_population.append(mut_child)

    # Fitness update for the generation
    start_indx = end_idx
    end_idx = start_indx + Window_Size
    sub_data = data.iloc[start_indx:end_idx]
    fitness = calculate_fitness(new_population, sub_data)

    pop = new_population

    best_idx = fitness.index(min(fitness))

    # Collect metrics for the final generation
    metrics = create_matric(i + 1, fitness, pop)
    history.append(metrics)
    Logger.info(f'Test cases: Count = {end_idx - start_indx}, Start = {start_indx}, End = {end_idx}')
    Logger.info(f'Results: Size={metrics["size"]:.6f}, Best={metrics["best"]:.6f}, Avg={metrics["avg"]:.6f}, Std={metrics["std"]:.6f}')
    Logger.info('')

  end_time = time.time()
  duration = end_time - start_time
  Logger.info(f'Total execution time: {duration:.2f} seconds')

  print_population(pop, fitness)
  Logger.info('')

  best_idx = fitness.index(min(fitness))
  best_indiv = pop[best_idx]
  Logger.info(f'Best individual: {best_indiv}')
  Logger.info(f'Best fitness: {fitness[best_idx]}')
  Logger.info(f'Total execution time: {duration:.2f} seconds')
  Logger.info('')
  plot_metrics(history, seed=args.seed)
  plot_best_fitness(history, seed=args.seed)
  plot_structural_metrics(history, seed=args.seed)
  plot_tree(best_indiv, seed=args.seed)

  results_dir = 'out/results'
  os.makedirs(results_dir, exist_ok=True)
  results_file = os.path.join(results_dir, 'results.txt')
  with open(results_file, 'a') as f:
    f.write(f"--- Run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    f.write(f"Configuration: {vars(args)}\n")
    f.write(f"Execution Time: {duration:.2f} seconds\n")
    f.write(f"Final Avg Fitness: {sum(fitness) / len(fitness):.4f}\n")
    f.write(f"Best Fitness (MAPE): {fitness[best_idx]:.4f}\n")
    f.write(f"Best Individual Topology: {best_indiv}\n")
    f.write(f"Best Individual Depth: {best_indiv.depth}\n")
    f.write(f"Best Individual Node Count: {best_indiv.node_count}\n")
    f.write("-" * 50 + "\n\n")

  if history:
    csv_file = os.path.join(results_dir, f"history_seed_{args.seed}.csv")
    with open(csv_file, 'w', newline='') as f:
      writer = csv.DictWriter(f, fieldnames=history[0].keys())
      writer.writeheader()
      writer.writerows(history)
    Logger.info(f'Metrics history saved to {csv_file}')
