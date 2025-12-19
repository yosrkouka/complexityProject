"""
Metrics and solution analysis utilities.
"""

import numpy as np


def analyze_solution(solution, constraints):
    """
    Analyze a solution and compute constraint compliance metrics.
    
    Args:
        solution: List of Task_ID in scheduled order
        constraints: Constraints dictionary from prepare_constraints
        
    Returns:
        Dictionary with metrics
    """
    n_tasks = len(solution)
    id_to_idx = constraints['id_to_idx']
    priorities = constraints['priorities']
    targets = constraints['targets']
    release_times = constraints['release_times']
    deadlines = constraints['deadlines']
    context_available = constraints['context_available']
    capacity_per_slot = constraints['capacity_per_slot']
    
    # Check window compliance
    in_window = 0
    for pos, task_id in enumerate(solution):
        idx = id_to_idx[task_id]
        if release_times[idx] <= pos <= deadlines[idx]:
            in_window += 1
    
    # Check context compliance
    context_ok = 0
    for task_id in solution:
        idx = id_to_idx[task_id]
        if context_available[idx] == 1:
            context_ok += 1
    
    # Check capacity violations
    n_slots = constraints['n_slots']
    slot_size = max(1, n_tasks // n_slots)
    capacity_violations = 0
    capacity_violation_magnitude = 0.0
    
    for slot_idx in range(n_slots):
        start_pos = slot_idx * slot_size
        end_pos = min((slot_idx + 1) * slot_size, n_tasks)
        slot_tasks = end_pos - start_pos
        
        if slot_tasks > capacity_per_slot:
            capacity_violations += 1
            capacity_violation_magnitude += (slot_tasks - capacity_per_slot)
    
    return {
        'n_tasks': n_tasks,
        'in_window': in_window,
        'context_ok': context_ok,
        'capacity_violations': capacity_violations,
        'capacity_violation_magnitude': capacity_violation_magnitude,
    }


def check_constraints(solution, constraints):
    """
    Check constraints and return penalty score.
    Returns penalty (0 if all constraints are satisfied).
    Matches notebook implementation.
    """
    penalty = 0
    n = len(solution)
    id_to_idx = constraints['id_to_idx']
    priorities = constraints['priorities']
    release_times = constraints['release_times']
    deadlines = constraints['deadlines']
    context_available = constraints['context_available']
    capacity_per_slot = constraints['capacity_per_slot']
    
    slot_usage = {}
    
    for pos, task_id in enumerate(solution):
        idx = id_to_idx[task_id]
        t = pos
        
        # Window constraint violation
        r_i = release_times[idx]
        d_i = deadlines[idx]
        if t < r_i or t > d_i:
            penalty += priorities[idx]
        
        # Capacity (aggregated slots ~20)
        slot = t // (n // 20 + 1)
        slot_usage[slot] = slot_usage.get(slot, 0) + 1
        
        # Bad context
        if context_available[idx] == 0:
            penalty += priorities[idx] * 0.5
    
    # Capacity violations
    for slot, count in slot_usage.items():
        if count > capacity_per_slot:
            penalty += (count - capacity_per_slot) * 10
    
    return penalty


def compute_fitness(solution, constraints):
    """
    Compute fitness value for a solution.
    Higher is better.
    Matches notebook implementation: score - penalty * 0.1
    
    Args:
        solution: List of Task_ID in scheduled order
        constraints: Constraints dictionary
        
    Returns:
        Fitness value (float)
    """
    id_to_idx = constraints['id_to_idx']
    priorities = constraints['priorities']
    release_times = constraints['release_times']
    deadlines = constraints['deadlines']
    context_available = constraints['context_available']
    
    score = 0.0
    n = len(solution)
    
    # Positive score: sum priorities for tasks in their window
    for pos, task_id in enumerate(solution):
        idx = id_to_idx[task_id]
        r_i = release_times[idx]
        d_i = deadlines[idx]
        
        if r_i <= pos <= d_i:  # Task is in its window
            if context_available[idx] == 1:  # Context OK
                score += priorities[idx]
            else:  # Bad context
                score += priorities[idx] * 0.3
    
    # Calculate penalty
    penalty = check_constraints(solution, constraints)
    
    # Final fitness: score - penalty * 0.1
    return score - penalty * 0.1

