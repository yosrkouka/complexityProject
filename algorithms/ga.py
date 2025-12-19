"""
Genetic Algorithm wrapper.
Standardized interface for GA algorithm.
"""

import numpy as np
import time
from utils.metrics import compute_fitness, check_constraints


def crossover(parent1, parent2):
    """Order crossover (OX) for permutation."""
    n = len(parent1)
    start, end = sorted(np.random.choice(n, size=2, replace=False))
    
    child = [None] * n
    child[start:end] = parent1[start:end]
    
    pos = end
    for item in parent2:
        if item not in child:
            if pos >= n:
                pos = 0
            child[pos] = item
            pos += 1
    
    return child


def mutate(individual, mutation_rate):
    """Swap mutation."""
    if np.random.random() < mutation_rate:
        n = len(individual)
        i, j = np.random.choice(n, size=2, replace=False)
        individual[i], individual[j] = individual[j], individual[i]
    return individual


def run_algorithm(data_dict, params):
    """
    Run Genetic Algorithm.
    
    Args:
        data_dict: Constraints dictionary from prepare_constraints
        params: Dictionary with algorithm parameters:
            - pop_size: Population size
            - generations: Number of generations
            - crossover_rate: Crossover probability
            - mutation_rate: Mutation probability
            - elitism: Number of elite individuals to preserve
            - seed: Random seed
    
    Returns:
        Dictionary with standardized output format
    """
    np.random.seed(params.get('seed', 42))
    
    # Extract parameters
    pop_size = params.get('pop_size', 80)
    generations = params.get('generations', 300)
    crossover_rate = params.get('crossover_rate', 0.85)
    mutation_rate = params.get('mutation_rate', 0.25)
    elitism = params.get('elitism', 4)
    
    # Extract data
    task_ids = data_dict['task_ids']
    n_tasks = len(task_ids)
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        individual = task_ids.copy()
        np.random.shuffle(individual)
        population.append(individual)
    
    # Evaluate initial population
    fitness_scores = [compute_fitness(ind, data_dict) for ind in population]
    
    # Find best
    best_idx = np.argmax(fitness_scores)
    best_solution = population[best_idx].copy()
    best_fitness = fitness_scores[best_idx]
    
    # History tracking
    history_best = [best_fitness]
    history_current = [np.mean(fitness_scores)]
    accepted_hist = []
    penalty_hist = []
    
    start_time = time.time()
    
    for generation in range(generations):
        # Selection: tournament selection
        new_population = []
        
        # Elitism: keep best individuals
        elite_indices = np.argsort(fitness_scores)[-elitism:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            parent1_idx = tournament_indices[np.argmax(tournament_fitness)]
            
            tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            parent2_idx = tournament_indices[np.argmax(tournament_fitness)]
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            if np.random.random() < crossover_rate:
                child = crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            child = mutate(child, mutation_rate)
            
            new_population.append(child)
        
        # Update population
        population = new_population[:pop_size]
        fitness_scores = [compute_fitness(ind, data_dict) for ind in population]
        
        # Track best
        gen_best_idx = np.argmax(fitness_scores)
        gen_best_fitness = fitness_scores[gen_best_idx]
        
        if gen_best_fitness > best_fitness:
            best_solution = population[gen_best_idx].copy()
            best_fitness = gen_best_fitness
        
        # Track history
        history_best.append(best_fitness)
        history_current.append(np.mean(fitness_scores))
        accepted_hist.append(1)  # GA always accepts new population
        
        # Compute penalty for tracking
        penalty = check_constraints(best_solution, data_dict)
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
        'n_iter': generations,
        'meta': {
            'algorithm': 'GA',
            'pop_size': pop_size,
            'generations': generations,
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate,
            'elitism': elitism,
            'seed': params.get('seed', 42),
        }
    }

