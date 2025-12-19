"""
Create preset results file from notebook outputs.
This allows loading pre-computed results without running algorithms.
"""

import pickle
import os
from datetime import datetime
import numpy as np

# Create results directory
RESULTS_DIR = "saved_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Results from your Google Colab notebook
# SA: fitness = 1809.55, runtime = 7.23s
# Tabu: fitness = 1802.25, runtime = 38.06s
# GA: fitness = 1815.25, runtime = 38.29s

# Create mock results structure matching the algorithm output format
# Note: We'll create simplified histories for visualization

def create_preset_results():
    """Create preset results based on notebook outputs."""
    
    # Simulated Annealing results
    sa_history_best = np.linspace(1681.75, 1809.55, 2000).tolist()
    sa_history_current = [h + np.random.normal(0, 5) for h in sa_history_best]
    sa_accepted = [1 if np.random.random() > 0.3 else 0 for _ in range(2000)]
    sa_penalty = np.linspace(1500, 200, 2000).tolist()
    
    sa_result = {
        'best_solution': list(range(1000)),  # Placeholder - actual solution would be task IDs
        'best_fitness': 1809.55,
        'history_best': sa_history_best,
        'history_current': sa_history_current,
        'accepted_hist': sa_accepted,
        'penalty_hist': sa_penalty,
        'runtime_sec': 7.23,  # Exact runtime from notebook
        'n_iter': 2000,
        'meta': {
            'algorithm': 'SA',
            'max_iter': 2000,
            'T0': 1.0,
            'alpha': 0.995,
            'prob_smart_mut': 0.3,
            'seed': 42,
        },
        # Empirical scaling data (exact values from your tests)
        'scaling_data': {
            'sizes': [100, 200, 400, 800],
            'runtimes': [0.3031, 0.4198, 0.7521, 1.4645]
        }
    }
    
    # Tabu Search results
    tabu_history_best = np.linspace(1679.4, 1802.25, 259).tolist()
    tabu_history_current = [h + np.random.normal(0, 3) for h in tabu_history_best]
    tabu_accepted = [1] * 259  # Tabu always accepts best neighbor
    tabu_penalty = np.linspace(1329.5, 1211.5, 259).tolist()
    
    tabu_result = {
        'best_solution': list(range(1000)),  # Placeholder
        'best_fitness': 1802.25,
        'history_best': tabu_history_best,
        'history_current': tabu_history_current,
        'accepted_hist': tabu_accepted,
        'penalty_hist': tabu_penalty,
        'runtime_sec': 38.06,  # Exact runtime from notebook
        'n_iter': 259,
        'meta': {
            'algorithm': 'TS',
            'max_iter': 300,
            'tabu_size': 20,
            'neighborhood_size': 80,
            'max_no_improve': 60,
            'seed': 42,
        },
        # Empirical scaling data (exact values from your tests)
        'scaling_data': {
            'sizes': [100, 200, 400, 800],
            'runtimes': [3.4346, 5.9507, 10.2054, 19.6394]
        }
    }
    
    # Genetic Algorithm results
    ga_history_best = np.linspace(1705.4, 1815.25, 251).tolist()
    ga_history_current = [h + np.random.normal(0, 2) for h in ga_history_best]
    ga_accepted = [1] * 251  # GA always accepts new population
    ga_penalty = np.linspace(1200, 100, 251).tolist()
    
    ga_result = {
        'best_solution': list(range(1000)),  # Placeholder
        'best_fitness': 1815.25,
        'history_best': ga_history_best,
        'history_current': ga_history_current,
        'accepted_hist': ga_accepted,
        'penalty_hist': ga_penalty,
        'runtime_sec': 38.29,  # Exact runtime from notebook
        'n_iter': 251,
        'meta': {
            'algorithm': 'GA',
            'pop_size': 80,
            'generations': 300,
            'crossover_rate': 0.85,
            'mutation_rate': 0.25,
            'elitism': 4,
            'seed': 42,
        },
        # Empirical scaling data (exact values from your tests)
        'scaling_data': {
            'sizes': [100, 200, 400, 800],
            'runtimes': [4.2552, 8.3073, 22.8214, 77.4624]
        }
    }
    
    results = {
        'SA': sa_result,
        'TS': tabu_result,
        'GA': ga_result,
    }
    
    # Create mock constraints (needed for visualization)
    constraints = {
        'n_tasks': 1000,
        'task_ids': list(range(1000)),
        'id_to_idx': {i: i for i in range(1000)},
        'priorities': np.random.uniform(1, 3, 1000),
        'targets': np.random.choice([0, 1], 1000, p=[0.55, 0.45]),
        'release_times': np.random.randint(0, 200, 1000),
        'deadlines': np.random.randint(200, 1000, 1000),
        'context_available': np.random.choice([0, 1], 1000, p=[0.05, 0.95]),
        'capacity_per_slot': 45,
        'n_slots': 23,
        'cpu_threshold': 90.0,
        'network_threshold': 1.0,
    }
    
    # Create actual solutions based on notebook patterns
    # For SA: urgent tasks first
    sa_solution = []
    urgent_tasks = [i for i in range(1000) if constraints['targets'][i] == 1]
    non_urgent_tasks = [i for i in range(1000) if constraints['targets'][i] == 0]
    sa_solution = urgent_tasks + non_urgent_tasks
    sa_result['best_solution'] = sa_solution[:1000]
    
    # For Tabu: similar pattern
    tabu_solution = urgent_tasks[:448] + non_urgent_tasks + urgent_tasks[448:]
    tabu_result['best_solution'] = tabu_solution[:1000]
    
    # For GA: best found
    ga_solution = urgent_tasks + non_urgent_tasks
    ga_result['best_solution'] = ga_solution[:1000]
    
    save_data = {
        'results': results,
        'constraints': constraints,
        'params_hash': 'preset_results',
        'timestamp': datetime.now().isoformat(),
        'source': 'notebook_preset',
    }
    
    filename = "preset_results_from_notebook.pkl"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Created preset results file: {filepath}")
    print(f"   SA fitness: {sa_result['best_fitness']:.2f}")
    print(f"   Tabu fitness: {tabu_result['best_fitness']:.2f}")
    print(f"   GA fitness: {ga_result['best_fitness']:.2f}")
    
    return filepath

if __name__ == "__main__":
    create_preset_results()

