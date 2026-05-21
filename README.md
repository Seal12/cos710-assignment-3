# COS710 Assignment 2: Structure-Based Genetic Programming

This repository contains a high-performance **Structure-Based Genetic Programming (SBGP)** system designed for time-series forecasting of residential electricity load. The system evolves mathematical expression trees to predict load values based on historical lagged data.

## Key Features

- **Structure-Based GP:** Implements morphological similarity measures using Global Index (GI) and Local Index (LI) to enforce structural diversity and mitigate bloat.
- **Asynchronous Multiprocessing:** Utilizes `concurrent.futures.ProcessPoolExecutor` to parallelize fitness evaluations across all available CPU cores, enabling rapid traversal of large datasets.
- **Robust Metrics:** Uses **Mean Absolute Percentage Error (MAPE)** for scale-invariant fitness evaluation.
- **Tumbling Sliding Window:** Evaluates individuals across shifting 30-day temporal windows to ensure generalization across the full 5.7-year dataset.
- **True Elitism:** Guarantees the preservation of the absolute best individuals across generations.

## Prerequisites

Ensure you have Python 3.8+ installed. You can install the required dependencies using pip:

```bash
pip install pandas numpy matplotlib
```

## Usage

### Running with Python
You can run the simulation using the following command:

```bash
python3 sbge.py --pop-size 100 --gens 70 --window-size 30
```

### Running with Docker
You can also run the simulation within an isolated container:

1. **Build the image:**
   ```bash
   docker build -t gp-simulation .
   ```

2. **Run the simulation:**
   (Mapping the `out/` volume ensures you can access the results on your host machine)
   ```bash
   docker run -v $(pwd)/out:/app/out gp-simulation
   ```

3. **Override default arguments:**
   ```bash
   docker run -v $(pwd)/out:/app/out gp-simulation --pop-size 50 --gens 20
   ```

### Command Line Arguments
| Argument | Description | Default |
| :--- | :--- | :--- |
| `--pop-size` | Number of individuals in the population | 100 |
| `--gens` | Number of generations to run | 70 |
| `--max-depth` | Maximum allowed depth for any tree | 5 |
| `--window-size` | Size of the sliding window in days | 30 |
| `--tournament-size` | Number of individuals in tournament selection | 2 |
| `--crossover-rate` | Proportion of pop generated via crossover | 0.8 |
| `--mutation-rate` | Proportion of pop generated via mutation | 0.19 |
| `--reproduction-rate`| Proportion of pop generated via elitism | 0.01 |
| `--seed` | Random seed for reproducibility | Random |

### Building a Standalone Executable
If you have `PyInstaller` installed in your virtual environment, use the following command to build a standalone executable:

```bash
python3 -m PyInstaller --onefile sbge.py
```
The resulting binary will be located in the `dist/` directory.

## Output and Results

All experimental outputs are stored in the `out/` directory:
- **`results.txt`:** A consolidated log of every run, including configuration, best MAPE, and the evolved mathematical equation.
- **Convergence Plots:** Visualization of population fitness (Best, Average, Worst) over generations.
- **Tree Visualizations:** PNG renders of the absolute best evolved mathematical structures.

## Project Structure

- `sbge.py`: Main entry point and evolutionary logic.
- `primitives.py`: Definition of function sets (+, -, *, /, min, max) and terminal nodes.
- `utils/logger.py`: Custom logging and progress monitoring.
- `data/`: Directory containing the UK Residential Energy Dataset.
- `report/`: LaTeX source and finalized academic report.
