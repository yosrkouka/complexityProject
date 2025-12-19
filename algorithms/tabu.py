"""
Tabu Search wrapper.
Standardized interface for Tabu Search algorithm.
"""

import numpy as np
import time
from utils.metrics import compute_fitness, check_constraints


def generate_neighbors(solution, n_neighbors):
    """Generate n_neighbors neighbors by swapping."""
    neighbors = []
    n = len(solution)
    
    for _ in range(n_neighbors):
        neighbor = solution.copy()
        i, j = np.random.choice(n, size=2, replace=False)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        neighbors.append(neighbor)
    
    return neighbors


def run_algorithm(data_dict, params):
    """
    Run Tabu Search algorithm.
    
    Args:
        data_dict: Constraints dictionary from prepare_constraints
        params: Dictionary with algorithm parameters:
            - max_iter: Maximum iterations
            - tabu_size: Tabu list size (tenure)
            - neighborhood_size: Number of neighbors to explore per iteration
            - max_no_improve: Maximum iterations without improvement
            - seed: Random seed
    
    Returns:
        Dictionary with standardized output format
    """
    np.random.seed(params.get('seed', 42))
    
    # Extract parameters
    max_iter = params.get('max_iter', 300)
    tabu_size = params.get('tabu_size', 20)
    neighborhood_size = params.get('neighborhood_size', 80)
    max_no_improve = params.get('max_no_improve', 60)
    
    # Extract data
    task_ids = data_dict['task_ids']
    n_tasks = len(task_ids)
    
    # Initialize solution
    current_solution = task_ids.copy()
    np.random.shuffle(current_solution)
    current_fitness = compute_fitness(current_solution, data_dict)
    
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    
    # Tabu list: store recent moves (as tuple of swapped positions)
    tabu_list = []
    
    # History tracking
    history_best = [best_fitness]
    history_current = [current_fitness]
    accepted_hist = []
    penalty_hist = []
    
    no_improve_count = 0
    start_time = time.time()
    
    for iteration in range(max_iter):
        # Generate neighbors
        neighbors = generate_neighbors(current_solution, neighborhood_size)
        
        # Evaluate neighbors
        neighbor_fitness = [compute_fitness(n, data_dict) for n in neighbors]
        
        # Find best non-tabu neighbor (or best overall if all tabu)
        best_neighbor_idx = None
        best_neighbor_fitness = float('-inf')
        
        for idx, (neighbor, fitness) in enumerate(zip(neighbors, neighbor_fitness)):
            # Check if move is tabu
            # Simple check: compare solution similarity
            is_tabu = False
            for tabu_move in tabu_list:
                # Check if this neighbor matches a tabu solution
                if neighbor == tabu_move:
                    is_tabu = True
                    break
            
            # Aspiration criterion: accept tabu if better than best
            if not is_tabu or fitness > best_fitness:
                if fitness > best_neighbor_fitness:
                    best_neighbor_idx = idx
                    best_neighbor_fitness = fitness
        
        # If all neighbors are tabu and worse, pick best anyway
        if best_neighbor_idx is None:
            best_neighbor_idx = np.argmax(neighbor_fitness)
            best_neighbor_fitness = neighbor_fitness[best_neighbor_idx]
        
        # Update current solution
        current_solution = neighbors[best_neighbor_idx]
        current_fitness = best_neighbor_fitness
        
        # Update tabu list
        # Store the solution (or a hash) in tabu list
        tabu_list.append(tuple(current_solution))
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
        
        # Update best
        if current_fitness > best_fitness:
            best_solution = current_solution.copy()
            best_fitness = current_fitness
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Early stopping
        if no_improve_count >= max_no_improve:
            break
        
        # Track history
        history_best.append(best_fitness)
        history_current.append(current_fitness)
        accepted_hist.append(1)  # Tabu always accepts best neighbor
        
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
        'n_iter': iteration + 1,
        'meta': {
            'algorithm': 'TS',
            'max_iter': max_iter,
            'tabu_size': tabu_size,
            'neighborhood_size': neighborhood_size,
            'max_no_improve': max_no_improve,
            'seed': params.get('seed', 42),
        }
    }

