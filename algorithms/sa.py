"""
Simulated Annealing wrapper.
Standardized interface for SA algorithm.
"""

import numpy as np
import time
from utils.metrics import compute_fitness, check_constraints


def run_algorithm(data_dict, params):
    """
    Run Simulated Annealing algorithm.
    
    Args:
        data_dict: Constraints dictionary from prepare_constraints
        params: Dictionary with algorithm parameters:
            - max_iter: Maximum iterations
            - T0: Initial temperature
            - alpha: Cooling rate
            - prob_smart_mut: Probability of smart mutation
            - seed: Random seed
    
    Returns:
        Dictionary with standardized output format
    """
    np.random.seed(params.get('seed', 42))
    
    # Extract parameters
    max_iter = params.get('max_iter', 2000)
    T0 = params.get('T0', 1.0)
    alpha = params.get('alpha', 0.995)
    prob_smart_mut = params.get('prob_smart_mut', 0.3)
    
    # Extract data
    task_ids = data_dict['task_ids']
    n_tasks = len(task_ids)
    
    # Initialize solution (random permutation)
    current_solution = task_ids.copy()
    np.random.shuffle(current_solution)
    current_fitness = compute_fitness(current_solution, data_dict)
    
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    
    # History tracking
    history_best = [best_fitness]
    history_current = [current_fitness]
    accepted_hist = []
    penalty_hist = []
    
    T = T0
    start_time = time.time()
    
    for iteration in range(max_iter):
        # Generate neighbor (swap two random tasks)
        neighbor = current_solution.copy()
        
        if np.random.random() < prob_smart_mut:
            # Smart mutation: swap urgent task with non-urgent if beneficial
            urgent_indices = [i for i, tid in enumerate(neighbor) 
                            if data_dict['targets'][data_dict['id_to_idx'][tid]] == 1]
            non_urgent_indices = [i for i, tid in enumerate(neighbor) 
                                if data_dict['targets'][data_dict['id_to_idx'][tid]] == 0]
            
            if urgent_indices and non_urgent_indices:
                i = np.random.choice(urgent_indices)
                j = np.random.choice(non_urgent_indices)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        else:
            # Random swap
            i, j = np.random.choice(n_tasks, size=2, replace=False)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        neighbor_fitness = compute_fitness(neighbor, data_dict)
        
        # Acceptance criterion
        delta = neighbor_fitness - current_fitness
        accepted = False
        
        if delta > 0:
            # Better solution
            accepted = True
        elif T > 0:
            # Accept worse solution with probability
            prob_accept = np.exp(delta / T)
            if np.random.random() < prob_accept:
                accepted = True
        
        if accepted:
            current_solution = neighbor
            current_fitness = neighbor_fitness
            
            if current_fitness > best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness
        
        # Update temperature
        T = T * alpha
        
        # Track history
        history_best.append(best_fitness)
        history_current.append(current_fitness)
        accepted_hist.append(1 if accepted else 0)
        
        # Compute penalty for tracking
        penalty = check_constraints(current_solution, data_dict)
        penalty_hist.append(penalty)
    
    runtime_sec = time.time() - start_time
    
    return {
        'best_solution': best_solution,
        'best_fitness': best_fitness,
        'history_best': history_best,
        'history_current': history_current,
        'accepted_hist': accepted_hist,
        'penalty_hist': penalty_hist,
        'runtime_sec': runtime_sec,
        'n_iter': max_iter,
        'meta': {
            'algorithm': 'SA',
            'max_iter': max_iter,
            'T0': T0,
            'alpha': alpha,
            'prob_smart_mut': prob_smart_mut,
            'seed': params.get('seed', 42),
        }
    }

