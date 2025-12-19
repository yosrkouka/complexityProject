# Optimization Algorithms Comparison Dashboard

Interactive Streamlit dashboard for comparing **Simulated Annealing (SA)**, **Genetic Algorithm (GA)**, and **Tabu Search (TS)** on the same task scheduling problem.

## Features

- 📊 **Side-by-side comparison** of three optimization algorithms
- 📈 **Convergence analysis** with interactive plots
- 🔍 **Constraint diagnostics** with Gantt charts and capacity analysis
- ⚙️ **Complexity analysis** (theoretical Big-O and empirical scaling)
- 💾 **Export results** to CSV or JSON
- 🎛️ **Fully configurable** algorithm parameters

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your dataset CSV has the following columns:
   - `Task_ID`
   - `Priority`
   - `Target (Optimal Scheduling)` (1 = urgent, 0 = non-urgent)
   - `Execution_Time (s)`
   - `CPU_Usage (%)`
   - `Network_IO (MB/s)`

## Usage

Run the dashboard:
```bash
streamlit run app.py
```

The dashboard will open in your browser. Use the sidebar to:
1. **Load dataset**: Upload CSV or specify local file path
2. **Configure parameters**: Set common constraints and algorithm-specific parameters
3. **Run algorithms**: Click "Run All" to execute SA, GA, and TS
4. **Explore results**: Navigate through 4 tabs:
   - **Overview**: KPI cards and summary metrics
   - **Convergence**: Fitness evolution and acceptance rates
   - **Constraints**: Gantt charts, capacity usage, position distributions
   - **Complexity**: Theoretical analysis and empirical scaling experiments

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── algorithms/
│   ├── __init__.py
│   ├── sa.py             # Simulated Annealing wrapper
│   ├── ga.py             # Genetic Algorithm wrapper
│   └── tabu.py           # Tabu Search wrapper
└── utils/
    ├── __init__.py
    ├── data.py           # Data loading and constraint generation
    ├── metrics.py        # Fitness and constraint evaluation
    └── plots.py          # Visualization utilities
```

## Algorithm Parameters

### Common Parameters
- **Random seed**: For reproducibility
- **Number of tasks to sample**: Optional subsampling
- **Capacity per slot**: System capacity constraint
- **CPU threshold**: Context availability threshold (%)
- **Network threshold**: Context availability threshold (MB/s)

### Simulated Annealing
- **Max iterations**: Maximum number of iterations
- **Initial temperature (T0)**: Starting temperature
- **Cooling rate (alpha)**: Temperature decay factor
- **Smart mutation probability**: Probability of using smart mutation

### Genetic Algorithm
- **Population size**: Number of individuals
- **Generations**: Number of generations
- **Crossover rate**: Probability of crossover
- **Mutation rate**: Probability of mutation
- **Elitism**: Number of best individuals preserved

### Tabu Search
- **Max iterations**: Maximum number of iterations
- **Tabu list size (tenure)**: Memory length
- **Neighborhood size**: Swaps evaluated per iteration
- **Max iterations without improvement**: Early stopping criterion

## Standardized Output Format

All algorithms return a dictionary with:
- `best_solution`: List of Task_ID in optimal order
- `best_fitness`: Best fitness value found
- `history_best`: List of best fitness per iteration
- `history_current`: List of current fitness per iteration
- `accepted_hist`: Acceptance history (SA) or move history (TS/GA)
- `penalty_hist`: Constraint penalty evolution
- `runtime_sec`: Execution time in seconds
- `n_iter`: Number of iterations completed
- `meta`: Dictionary with algorithm parameters used

## Constraints

All algorithms respect the same constraints:
1. **Unicity**: Each task appears exactly once (permutation encoding)
2. **Time windows**: Each task has [r_i, d_i] release/deadline window
3. **Capacity**: Aggregated slots cannot exceed capacity limit
4. **Context**: Tasks prefer scheduling when CPU < threshold and Network > threshold

## Notes

- The dashboard uses Streamlit caching to avoid recomputation
- Results persist in session state until cleared
- Export functionality supports both CSV (detailed schedule) and JSON (summary)
- Empirical scaling experiments may take several minutes for large datasets

## License

This project is provided as-is for educational and research purposes.

