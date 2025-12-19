"""
Data loading and constraint preparation utilities.
"""

import pandas as pd
import numpy as np


def load_data(file_path):
    """
    Load dataset from CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with required columns
    """
    df = pd.read_csv(file_path)
    
    # Validate required columns
    required_cols = [
        'Task_ID', 
        'Priority', 
        'Target (Optimal Scheduling)', 
        'Execution_Time (s)', 
        'CPU_Usage (%)', 
        'Network_IO (MB/s)'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def prepare_constraints(data, capacity_per_slot=45, cpu_threshold=90.0, 
                        network_threshold=1.0, seed=42):
    """
    Prepare constraints dictionary from dataset.
    
    Args:
        data: DataFrame with task data
        capacity_per_slot: Capacity limit per aggregated slot
        cpu_threshold: CPU usage threshold for context availability
        network_threshold: Network IO threshold for context availability
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with constraints and task data
    """
    np.random.seed(seed)
    
    n_tasks = len(data)
    task_ids = data['Task_ID'].tolist()
    
    # Create mapping from Task_ID to index
    id_to_idx = {task_id: idx for idx, task_id in enumerate(task_ids)}
    
    # Extract task attributes
    priorities = data['Priority'].values
    targets = data['Target (Optimal Scheduling)'].values  # 1 = urgent, 0 = non-urgent
    execution_times = data['Execution_Time (s)'].values
    cpu_usage = data['CPU_Usage (%)'].values
    network_io = data['Network_IO (MB/s)'].values
    
    # Generate time windows [r_i, d_i]
    # Release time: random between 0 and some max
    # Deadline: release_time + execution_time + slack
    max_release = max(100, n_tasks // 2)
    release_times = np.random.randint(0, max_release, size=n_tasks)
    
    # Deadline: release + execution + slack (urgent tasks have tighter windows)
    slack_multiplier = np.where(targets == 1, 
                                np.random.uniform(1.5, 3.0, n_tasks),
                                np.random.uniform(3.0, 6.0, n_tasks))
    deadlines = release_times + (execution_times * slack_multiplier).astype(int)
    deadlines = np.maximum(deadlines, release_times + execution_times.astype(int))
    
    # Context availability: CPU < threshold AND Network > threshold
    context_available = ((cpu_usage < cpu_threshold) & (network_io > network_threshold)).astype(int)
    
    # Calculate capacity per slot (aggregate tasks into slots)
    # Approximate: each slot can hold ~capacity_per_slot tasks
    n_slots = max(1, (n_tasks + capacity_per_slot - 1) // capacity_per_slot)
    
    return {
        'n_tasks': n_tasks,
        'task_ids': task_ids,
        'id_to_idx': id_to_idx,
        'priorities': priorities,
        'targets': targets,
        'execution_times': execution_times,
        'cpu_usage': cpu_usage,
        'network_io': network_io,
        'release_times': release_times,
        'deadlines': deadlines,
        'context_available': context_available,
        'capacity_per_slot': capacity_per_slot,
        'n_slots': n_slots,
        'cpu_threshold': cpu_threshold,
        'network_threshold': network_threshold,
    }


def sample_data(data, n_sample=None, seed=42):
    """
    Sample n_sample tasks from dataset (or return all if n_sample is None).
    
    Args:
        data: DataFrame
        n_sample: Number of tasks to sample (None = all)
        seed: Random seed
        
    Returns:
        Sampled DataFrame
    """
    if n_sample is None or n_sample >= len(data):
        return data.copy()
    
    np.random.seed(seed)
    sampled = data.sample(n=min(n_sample, len(data)), random_state=seed)
    return sampled.reset_index(drop=True)

