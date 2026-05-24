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
from primitives import Individual, GRAMMAR
from utils import Logger

##########################################
# Data processing
##########################################


def load_data(filename: str, lag_hours=1):
    """Loads CSV data and parses dates."""
    df = pd.read_csv(filename, parse_dates=["utc_timestamp"], dayfirst=True)
    df = df.sort_values("utc_timestamp")

    Logger.debug(df.info)

    expected_delta = pd.Timedelta(minutes=15)
    is_regular = (df["utc_timestamp"].diff().iloc[1:] == expected_delta).all()

    if not is_regular:
        print("Warning: Gaps detected. Use df.resample('15T').interpolate() first.")
        raise Exception("Data is not regular")

    target_col = "Electricity_load"

    data = df[[target_col]].copy()

    for i in range(1, (lag_hours * 4) + 1):
        data[f"load_t-{i}"] = data[target_col].shift(i)

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
    df_norm[numeric_cols] = (
        df_norm[numeric_cols] - df_norm[numeric_cols].mean()
    ) / df_norm[numeric_cols].std().replace(0, 1)
    return df_norm


##########################################
# Visualizations
##########################################


def plot_metrics(history, seed: int):
    """Plots the fitness metrics across generations and saves the figure."""
    gens = [h["gen"] for h in history]
    best = [h["best"] for h in history]
    worst = [h["worst"] for h in history]
    avg = [h["avg"] for h in history]
    std = [h["std"] for h in history]

    plt.figure(figsize=(10, 6))

    # Plot Average with Std Dev shading
    plt.fill_between(
        gens,
        np.array(avg) - np.array(std),
        np.array(avg) + np.array(std),
        alpha=0.2,
        color="gray",
        label="Std Dev",
    )
    plt.plot(gens, avg, label="Average Fitness", color="black", linewidth=2)

    # Plot Best and Worst
    plt.plot(gens, best, label="Best Fitness", color="green", marker="o")
    plt.plot(gens, worst, label="Worst Fitness", color="red", linestyle="--", alpha=0.5)

    plt.title("Evolution of Population Fitness (MAPE)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Lower is Better)")
    plt.yscale("log")

    if len(gens) > 1:
        plt.xticks(np.linspace(min(gens), max(gens), 10, dtype=int))

    plt.legend()
    plt.grid(True, which="major", linestyle="--", alpha=0.5)

    # Save the plot
    if not os.path.exists("out"):
        os.makedirs("out")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"out/{timestamp}_gp_fitness_{seed}.png"
    plt.savefig(filename)
    plt.close()
    Logger.info(f"Fitness evolution plot saved to {filename}")


def plot_best_fitness(history, seed: int):
    """Plots the best fitness across generations and saves the figure."""
    gens = [h["gen"] for h in history]
    best = [h["best"] for h in history]

    plt.figure(figsize=(10, 6))

    # Plot Best
    plt.plot(gens, best, label="Best Fitness", color="green", marker="o")

    plt.title("Evolution of Best Fitness (MAPE)")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Lower is Better)")
    plt.yscale("log")

    if len(gens) > 1:
        plt.xticks(np.linspace(min(gens), max(gens), 10, dtype=int))

    plt.legend()
    plt.grid(True, which="major", linestyle="--", alpha=0.5)

    # Save the plot
    if not os.path.exists("out"):
        os.makedirs("out")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"out/{timestamp}_gp_best_fitness_{seed}.png"
    plt.savefig(filename)
    plt.close()
    Logger.info(f"Best fitness plot saved to {filename}")


def plot_structural_metrics(history, seed: int):
    """Plots the structural metrics (size and depth) and saves the figure."""
    gens = [h["gen"] for h in history]
    avg_size = [h["avg_tree_size"] for h in history]
    max_size = [h["max_tree_size"] for h in history]
    min_size = [h["min_tree_size"] for h in history]
    avg_depth = [h["avg_tree_depth"] for h in history]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = "tab:blue"
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Tree Size (Nodes)", color=color)
    ax1.plot(gens, avg_size, color=color, label="Avg Size", linewidth=2)
    ax1.fill_between(
        gens, min_size, max_size, color=color, alpha=0.1, label="Size Range"
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Avg Tree Depth", color=color)
    ax2.plot(gens, avg_depth, color=color, label="Avg Depth", linestyle="--")
    ax2.tick_params(axis="y", labelcolor=color)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    if len(gens) > 1:
        plt.xticks(np.linspace(min(gens), max(gens), 10, dtype=int))

    plt.legend(h1 + h2, l1 + l2)
    plt.title("Evolution of Population Structure")
    fig.tight_layout()

    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"out/{timestamp}_gp_structure_{seed}.png"
    plt.savefig(filename)
    plt.close()
    Logger.info(f"Structural evolution plot saved to {filename}")


def create_matric(generation: int, fitness: list[float], pop: list[Primitives.Node]):
    sizes = [ind.node_count for ind in pop]
    depths = [ind.depth for ind in pop]

    metrics = {
        "gen": generation,
        "size": len(pop),
        "best": np.min(fitness),
        "worst": np.max(fitness),
        "avg": np.mean(fitness),
        "std": np.std(fitness),
        "avg_tree_size": np.mean(sizes),
        "max_tree_size": np.max(sizes),
        "min_tree_size": np.min(sizes),
        "avg_tree_depth": np.mean(depths),
    }

    return metrics


def plot_individual(individual: Primitives.Individual, seed: int):
    """
    Plots an individual as a single figure:
      - The phenotype expression tree.
      - Genotype codon array printed as a text annotation below the tree.
    """
    tree = individual.phenotype
    genotype = individual.genotype
    n = len(genotype)

    depth = tree.depth
    tree_width = min(100, max(10, (2**depth) * 0.8))
    tree_height = min(50, max(6, (depth + 1) * 1.5))

    fig, ax = plt.subplots(figsize=(tree_width, tree_height))
    ax.axis("off")
    ax.set_title(f"Phenotype: {tree}", fontsize=11, pad=10, wrap=True)

    def draw_node(node, x, y, dx, dy):
        if hasattr(node, "left") and hasattr(node, "right"):
            ax.plot([x, x - dx], [y, y - dy], "k-", lw=1.5, zorder=1)
            ax.plot([x, x + dx], [y, y - dy], "k-", lw=1.5, zorder=1)
            draw_node(node.left, x - dx, y - dy, dx / 2, dy)
            draw_node(node.right, x + dx, y - dy, dx / 2, dy)

        label = str(node.value) if hasattr(node, "value") else str(node)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            bbox=dict(facecolor="lightblue", edgecolor="black", boxstyle="round,pad=1"),
            fontsize=12,
            zorder=2,
        )

    draw_node(tree, 0, 0, tree_width / 2, 1)

    # ── Genotype annotation ─────────────────────────────────────────────
    gt_label = f"Genotype ({n} codons): {genotype}"
    fig.text(
        0.5,
        0.01,
        gt_label,
        ha="center",
        va="bottom",
        fontsize=9,
        wrap=True,
        bbox=dict(facecolor="lightyellow", edgecolor="grey", boxstyle="round,pad=0.5"),
    )

    # ── Save ────────────────────────────────────────────────────────────
    out_dir = "out/trees"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    filename = f"{out_dir}/{seed}_individual_tree.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    Logger.info(f"Individual tree plot saved to {filename}")

    return filename


##########################################
# Population methods
##########################################


def generate_random_individual(target_depth, method):
    genotype = []

    def derive(non_terminal, current_depth):
        if non_terminal not in GRAMMAR:
            return

        rules = GRAMMAR[non_terminal]

        if non_terminal == "<expr>":
            if current_depth < target_depth:
                if method == "full":
                    r_idx = 0
                else:  # grow
                    r_idx = random.randint(0, len(rules) - 1)
            else:
                r_idx = random.choice([1, 2])
        else:
            r_idx = random.randint(0, len(rules) - 1)

        multiples = [m for m in range(0, 50) if (r_idx + len(rules) * m) < 256]
        if multiples:
            codon = r_idx + len(rules) * random.choice(multiples)
        else:
            codon = r_idx
        genotype.append(codon)

        rule = rules[r_idx]
        rule_type = rule[0]

        if non_terminal == "<root>":
            derive("<expr>", current_depth + 1)
            derive("<expr>", current_depth + 1)
        elif non_terminal == "<expr>":
            if rule_type == "Function":
                derive("<op>", current_depth)
                derive("<expr>", current_depth + 1)
                derive("<expr>", current_depth + 1)
            elif rule_type == "Variable":
                derive("<var>", current_depth)
            elif rule_type == "Constant":
                pass

    derive("<root>", 0)
    ind = Individual(genotype)
    ind.decode()

    return ind


def init_pop(pop_size, max_depth, gen_method: str = "ramped-half-and-half"):
    pop = []
    if gen_method == "ramped-half-and-half":
        depths = list(range(1, max_depth + 1))
        trees_per_depth = pop_size // len(depths)

        for depth in depths:
            for i in range(trees_per_depth):
                method = "full" if i < trees_per_depth // 2 else "grow"
                pop.append(generate_random_individual(depth, method))

        while len(pop) < pop_size:
            method = random.choice(["full", "grow"])
            depth = random.choice(depths)
            pop.append(generate_random_individual(depth, method))
    else:
        for _ in range(pop_size):
            pop.append(generate_random_individual(max_depth, gen_method))

    return pop


def print_population(pop, fitness=None):
    Logger.info("=" * 40)
    Logger.info(f"Population Size: {len(pop)}")

    for i, tree in enumerate(pop):
        individual = ""
        exp = str(tree)

        if fitness is not None:
            individual += f"{exp}: {fitness[i]} "
        else:
            individual = exp

        Logger.info(f"{i}: {individual}")

    Logger.info("")


##########################################
# Fitness
##########################################


def raw_fitness(tree, data):
    """Calculates the Mean Absolute Percentage Error (MAPE) of the tree on the given data."""
    if tree.phenotype is None:
        tree.decode()
    if tree.phenotype is None:
        return 1e9

    sum_error = 0
    target_col = "Electricity_load"
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


def eval_worker(args):
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
        for count, (idx, indivError) in enumerate(
            executor.map(eval_worker, args_list), 1
        ):
            pop[idx].fitness = indivError
            fitness_results[idx] = indivError
            Logger.progress_log(
                f"Evaluating Population Progress: {(count / total_evals) * 100:.1f}% ({count}/{total_evals})"
            )
            Logger.debug(f"{idx}: {pop[idx]} Fitness: {pop[idx].fitness}")

    print("")  # Clear progress bar newline
    return fitness_results


##########################################
# Selection methods
##########################################
def tournament_selection(
    population, fitness, tournament_size: int = 2, selection_count: int = 2
):
    winners = []

    pop_indices = list(range(len(population)))
    Logger.debug(f"pop_indices: {pop_indices}")

    for _ in range(selection_count):
        participants = random.sample(pop_indices, min(tournament_size, len(population)))
        Logger.debug(f"participants: {participants}")

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
            l_gi, l_li = calculate_similarity(
                node1.left, node2.left, cutoff_depth, current_depth + 1
            )
            r_gi, r_li = calculate_similarity(
                node1.right, node2.right, cutoff_depth, current_depth + 1
            )
            gi += l_gi + r_gi
            li += l_li + r_li

    return gi, li


##########################################
# Genetic Operators
##########################################


def crossover(
    parent1: Individual,
    parent2: Individual,
    max_depth: int,
    crossover_rate: float = 0.5,
):
    """Performs genotype-level single-point crossover for standard GE on flat codon lists."""
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()

    gt1 = list(parent1.genotype)
    gt2 = list(parent2.genotype)

    if len(gt1) > 1 and len(gt2) > 1:
        cut_point = random.randint(1, min(len(gt1), len(gt2)) - 1)
        c1_gt = gt1[:cut_point] + gt2[cut_point:]
        c2_gt = gt2[:cut_point] + gt1[cut_point:]
    else:
        c1_gt = list(gt1)
        c2_gt = list(gt2)

    child1 = Individual(c1_gt)
    child2 = Individual(c2_gt)

    child1.decode()
    child2.decode()

    if (
        child1.phenotype is not None
        and child2.phenotype is not None
        and child1.depth <= max_depth
        and child2.depth <= max_depth
    ):
        return child1, child2
    else:
        return parent1.copy(), parent2.copy()


def mutation(parent: Individual, max_depth: int, mutation_rate: float = 0.5):
    """Performs standard codon flip mutation on a flat list."""
    if random.random() > mutation_rate:
        return parent.copy()

    for _ in range(3):
        child_gt = list(parent.genotype)
        for j in range(len(child_gt)):
            if random.random() < 0.15:
                child_gt[j] = random.randint(0, 255)

        child = Individual(child_gt)
        child.decode()
        if child.phenotype is not None and child.depth <= max_depth:
            return child

    return parent.copy()


##########################################
# GP: Params
##########################################

# Terminals = [ 'x', 'y', Primitives.ConstantValue ]
Terminals = [f"load_t-{i}" for i in range(1, 9)] + [Primitives.ConstantValue]
# Terminals.extend(['lag_24h', 'lag_7d'])

Functions = [
    Primitives.Add,
    Primitives.Sub,
    Primitives.Mul,
    Primitives.Div,
    Primitives.Max,
    Primitives.Min,
]

FitnessCases = [(1, 1, 2), (1, 2, 3), (2, 1, 3), (2, 2, 4)]

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

Window_Size = 4 * 24 * args.window_size  # 14 Days

if __name__ == "__main__":
    print(args)
    random.seed(args.seed)
    history = []
    start_time = time.time()

    data = load_data("data/Residential_Energy_Dataset_UK- 2014-2020.csv", lag_hours=2)

    pop = init_pop(args.pop_size, args.max_depth, gen_method="ramped-half-and-half")
    start_indx = 0
    end_idx = start_indx + Window_Size
    sub_data = data.iloc[start_indx:end_idx]
    fitness = calculate_fitness(pop, sub_data)

    Logger.info("Init population")
    print_population(pop, fitness)

    # Collect metrics for current generation
    metrics = create_matric(0, fitness, pop)
    history.append(metrics)

    for i in range(args.gens - 1):
        Logger.info(f"Generation {i + 1}:")

        new_population = []

        num_reproduction = max(1, int(args.pop_size * args.reproduction_rate))
        num_crossover = int(args.pop_size * args.crossover_rate)
        if num_crossover % 2 != 0:
            num_crossover -= 1  # Keep even to avoid overshooting

        # 1. Reproduction Phase (True Elitism)
        elite_indices = sorted(range(len(fitness)), key=lambda k: fitness[k])[
            :num_reproduction
        ]
        for idx in elite_indices:
            if len(new_population) < args.pop_size:
                new_population.append(pop[idx].copy())

        # 2. Crossover Phase
        target_crossover = len(new_population) + num_crossover
        while (
            len(new_population) < target_crossover
            and len(new_population) < args.pop_size - 1
        ):
            parent1 = tournament_selection(
                pop, fitness, tournament_size=args.tournament_size, selection_count=1
            )[0]
            parent2 = tournament_selection(
                pop, fitness, tournament_size=args.tournament_size, selection_count=1
            )[0]

            # Use Structural Similarity to prevent redundant crossover
            cutoff = max(1, args.max_depth // 2)
            gi, li = calculate_similarity(
                parent1.phenotype, parent2.phenotype, cutoff_depth=cutoff
            )

            attempts = 0
            while (
                gi + li >= min(parent1.node_count, parent2.node_count) * 0.8
                and attempts < 3
            ):
                parent2 = tournament_selection(
                    pop,
                    fitness,
                    tournament_size=args.tournament_size,
                    selection_count=1,
                )[0]
                gi, li = calculate_similarity(
                    parent1.phenotype, parent2.phenotype, cutoff_depth=cutoff
                )
                attempts += 1

            child1, child2 = crossover(
                parent1, parent2, max_depth=args.max_depth, crossover_rate=1.0
            )  # Forced execution
            new_population.append(child1)
            if len(new_population) < args.pop_size:
                new_population.append(child2)

        # 3. Mutation Phase (Fills all remaining slots to ensure exact pop_size)
        while len(new_population) < args.pop_size:
            mut_parent = tournament_selection(
                pop, fitness, tournament_size=args.tournament_size, selection_count=1
            )[0]
            mut_child = mutation(
                mut_parent, args.max_depth, mutation_rate=1.0
            )  # Forced execution
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
        Logger.info(
            f"Test cases: Count = {end_idx - start_indx}, Start = {start_indx}, End = {end_idx}"
        )
        Logger.info(
            f"Results: Size={metrics['size']:.6f}, Best={metrics['best']:.6f}, Avg={metrics['avg']:.6f}, Std={metrics['std']:.6f}"
        )
        Logger.info("")

    end_time = time.time()
    duration = end_time - start_time
    Logger.info(f"Total execution time: {duration:.2f} seconds")

    print_population(pop, fitness)
    Logger.info("")

    best_idx = fitness.index(min(fitness))
    best_indiv = pop[best_idx]
    Logger.info(f"Best individual: {best_indiv}")
    Logger.info(f"Best fitness: {fitness[best_idx]}")
    Logger.info(f"Total execution time: {duration:.2f} seconds")
    Logger.info("")
    plot_metrics(history, seed=args.seed)
    plot_best_fitness(history, seed=args.seed)
    plot_structural_metrics(history, seed=args.seed)
    plot_individual(best_indiv, seed=args.seed)

    results_dir = "out/results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "results.txt")
    with open(results_file, "a") as f:
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
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        Logger.info(f"Metrics history saved to {csv_file}")
