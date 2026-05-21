# Skill File: COS710 Assignment 3 — Structure-Based Grammatical Evolution (SBGE)

## Project Overview

A **Structure-Based Grammatical Evolution (SBGE)** system for time-series regression on residential electricity load data (UK, 2014–2020). The algorithm evolves symbolic regression equations to predict `Electricity_load` using the 8 most recent 15-minute lag features (`load_t-1` to `load_t-8`). The fitness metric is **MAPE** (Mean Absolute Percentage Error); lower is better.

This is Assignment 3 of COS710, ported from a tree-based Genetic Programming system (Assignment 2). The key algorithmic change is replacing tree-level operators with **Dynamic Structured Grammatical Evolution (DSGE)**: a grammar-constrained genotype represented as dictionaries of integer codon lists, one list per non-terminal symbol.

---

## File Structure

```
Assignment 3/
├── sbge.py            # Main evolution loop, operators, fitness, visualizations
├── primitives.py      # Node classes, GRAMMAR definition, Individual class
├── utils/
│   └── logger.py      # Progress-bar-friendly Logger (Logger.info/debug/progress_log)
├── data/
│   └── Residential_Energy_Dataset_UK- 2014-2020.csv
└── out/               # Generated outputs (plots, results, tree PNGs)
    ├── results/
    │   ├── results.txt
    │   └── history_seed_<seed>.csv
    └── trees/
        └── <seed>_individual_tree.png
```

---

## Running the Algorithm

```bash
# From the Assignment 3 directory, using the venv in Assignment 3:
../Assignment\ 3/.venv/bin/python3 sbge.py [args]

# Key CLI arguments (all have defaults):
--pop-size         int    default=100     Population size
--max-depth        int    default=5       Max tree depth
--gens             int    default=70      Number of generations
--seed             int    default=random  RNG seed
--crossover-rate   float  default=0.8     Fraction of pop produced by crossover
--mutation-rate    float  default=0.19    Fraction of pop produced by mutation
--reproduction-rate float default=0.01   Fraction of pop kept via elitism
--tournament-size  int    default=2       Tournament selection size
--window-size      int    default=30      Sliding window in days (× 4 × 24 = rows)
```

---

## Architecture: `primitives.py`

### Node Hierarchy
All tree nodes inherit from `Node`. Key properties on every node:
- `.eval(**kwargs)` — evaluates the node given a dict of feature values
- `.node_count` — total nodes in subtree (recursive)
- `.depth` — max depth of subtree (recursive)
- `.ret_type` — `'float'` (default) or `'prediction'` (only `StructuredRoot`)
- `.copy()` — `copy.deepcopy`
- `.get_subtree(idx)`, `.replace_subtree(idx, node)`, `.get_relative_depth(idx)` — tree surgery by DFS index

```
Node
├── Terminal
│   ├── Variable(value)       # looks up kwargs[value]
│   └── Constant(value=None)  # random in [-0.2, 0.2] if value not given
└── Function(value, left, right)
    └── BasisArithmetic
        ├── Add, Sub, Mul
        ├── Div               # protected: returns 1 if divisor == 0
        ├── Max, Min
        └── StructuredRoot    # always the root; eval = left + right; ret_type='prediction'
```

`ConstantValue = 'C'` is a sentinel string used as a grammar terminal tag (not a node class).

### GRAMMAR (DSGE)
Defined in `primitives.py` as a dict of production rules:

```python
GRAMMAR = {
    '<root>':  [('StructuredRoot', ['<expr>', '<expr>'])],
    '<expr>':  [('Function', ['<op>', '<expr>', '<expr>']),   # index 0
                ('Variable', ['<var>']),                      # index 1
                ('Constant', ['<const>'])],                   # index 2
    '<op>':    [('+',), ('-',), ('*',), ('/',), ('max',), ('min',)],
    '<var>':   [('load_t-1',), ..., ('load_t-8',)],
    '<const>': [('C',)]
}
```

> **Important**: `<expr>` rule index 0 → Function node (recursive), indices 1/2 → terminals. During `generate_random_individual`, when `current_depth >= target_depth`, only indices `[1, 2]` are picked to force a terminal and avoid infinite recursion.

### Individual class
Also defined in `primitives.py`. Imported in `sbge.py` via `from primitives import Individual, GRAMMAR`.

```python
class Individual:
    genotype:  dict[str, list[int]]  # per-NT codon lists; key = non-terminal string
    constants: list[float]           # ephemeral constants in order of use
    phenotype: Node | None           # decoded tree; None until decode() is called
    fitness:   float | None

    def decode()    # DSGE mapping: genotype → phenotype tree (dynamic: appends codons if list runs out)
    def copy()      # deep copy of genotype/constants; preserves fitness/phenotype refs
    def eval(**kwargs)           # delegates to phenotype.eval(); calls decode() if needed
    def __str__()                # delegates to str(phenotype)
    @property node_count, depth  # delegates to phenotype; auto-decodes if needed
```

**After `decode()`**: codon lists and constants are trimmed to exactly the number of codons consumed. This keeps genotype aligned with phenotype structure.

---

## Architecture: `sbge.py`

### Data Pipeline (lines 20–55)
- `load_data(filename, lag_hours=2)` — loads CSV, validates 15-min regularity, creates lag columns `load_t-1..load_t-8`, drops NaN rows.
- `normalize_window(window_data)` — Z-score normalization per window (currently unused in main loop but available).
- The evolution loop uses a **sliding window**: `Window_Size = 4 * 24 * args.window_size` rows per generation, advancing `start_indx` each generation.

### Visualization (lines 61–230)
| Function | Output |
|---|---|
| `plot_metrics(history, seed)` | `out/<timestamp>_gp_fitness_<seed>.png` — best/avg/worst + std shading, log-y |
| `plot_best_fitness(history, seed)` | `out/<timestamp>_gp_best_fitness_<seed>.png` — best only |
| `plot_structural_metrics(history, seed)` | `out/<timestamp>_gp_structure_<seed>.png` — avg/min/max size + avg depth |
| `plot_tree(tree: Node, seed)` | `out/trees/<seed>_individual_tree.png` — matplotlib tree diagram |
| `create_matric(gen, fitness, pop)` | returns metrics dict: `gen, size, best, worst, avg, std, avg/max/min_tree_size, avg_tree_depth` |

> Note: `plot_tree` takes a **`Node`** (the phenotype), not an `Individual`. Call it as `plot_tree(best_indiv.phenotype, seed=args.seed)`.

### Population Initialization (lines 237–301)
- `generate_random_individual(target_depth, method)` — recursive grammar derivation; `method` is `'full'` or `'grow'`.
- `init_pop(pop_size, max_depth, gen_method='ramped-half-and-half')` — default Ramped Half-and-Half across depths 1..max_depth, first half of each depth bucket uses `full`, second half uses `grow`.

### Fitness (lines 324–366)
- `raw_fitness(tree, data)` — MAPE over all rows; on exception adds penalty `10`.
- `calculate_fitness(pop, cases)` — multiprocessing via `ProcessPoolExecutor(max_workers=cpu_count-1)`; returns `list[float]`. Also sets `individual.fitness` in-place.
- `_eval_worker(args)` — picklable top-level worker for multiprocessing.

> **Multiprocessing gotcha**: `Individual` objects must be picklable. `primitives.py` has no unpicklable state.

### Selection (lines 372–389)
- `tournament_selection(population, fitness, tournament_size=2, selection_count=2)` — standard k-tournament; returns list of winners (references into population).

### Structural Similarity (lines 395–431)
- `calculate_similarity(node1, node2, cutoff_depth)` — returns `(gi, li)`.
  - `gi`: matching function nodes at depth ≤ cutoff
  - `li`: matching nodes at depth > cutoff
- Used in the crossover phase to avoid mating structurally near-identical parents (retries up to 3 times if `gi + li >= 0.8 * min_node_count`).
- Takes **`Node`** objects — call as `calculate_similarity(parent1.phenotype, parent2.phenotype, ...)`.

### Genetic Operators (lines 437–509)
#### `crossover(parent1, parent2, max_depth, crossover_rate=0.5)`
- Genotype-level single-point crossover, **per non-terminal**: cuts each NT's codon list at a random point and swaps suffixes.
- Also crosses constants lists.
- Rejects children exceeding `max_depth` (returns parent copies instead).

#### `mutation(parent, max_depth, mutation_rate=0.5)`
- Codon flip: each codon independently mutated with probability `0.15` → random valid rule index.
- Constant perturbation: each constant independently perturbed with prob `0.15` → `gauss(0, 0.05)`.
- Up to 3 retry attempts; falls back to `parent.copy()` if depth always exceeds limit.

### Evolution Loop (lines 554–677)
Three-phase per generation:
1. **Elitism/Reproduction** — top `reproduction_rate × pop_size` individuals copied directly.
2. **Crossover** — tournament selection pairs, similarity check, forced crossover (`crossover_rate=1.0`).
3. **Mutation** — fills remaining slots; forced mutation (`mutation_rate=1.0`).

Outputs per run: `results.txt` (appended), `history_seed_<seed>.csv`, 4 PNG plots.

---

## Key Conventions & Gotchas

1. **`Individual` vs `Node`**: The population is a `list[Individual]`. Functions expecting a bare tree node (e.g. `calculate_similarity`, `plot_tree`, `raw_fitness`) receive `individual.phenotype`.
2. **Decode is lazy**: `phenotype` is `None` until `decode()` is called. `Individual.eval()`, `__str__()`, `node_count`, `depth` all auto-call `decode()`.
3. **Codon trimming**: After `decode()`, `individual.genotype[nt]` is truncated to exactly the number of codons consumed. Extra codons from crossover tails may be trimmed on next decode. This is by design (DSGE).
4. **Constants are positional**: `individual.constants[i]` corresponds to the i-th `<const>` expansion in DFS order. Crossover on constants uses the same single-point scheme as codons.
5. **`StructuredRoot` is always the root**: The grammar only allows one production for `<root>`. Its `ret_type` is `'prediction'`; all other nodes return `'float'`.
6. **Protected division**: `Div.eval()` returns `1` (not `inf`) when the denominator is zero.
7. **Sliding window fitness**: Each generation evaluates on a *different* window of the dataset (non-overlapping, advancing forward). If the dataset is exhausted the slice will be empty — no guard currently exists for this.
8. **`Terminals` and `Functions` lists** (lines 516–526) are legacy from the old GP system. They are no longer used in the DSGE code but remain in the file.
