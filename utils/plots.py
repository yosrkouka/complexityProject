"""
Plotting utilities for visualization.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def plot_convergence_comparison(results_dict, use_plotly=True):
    """
    Plot best fitness convergence for all algorithms.
    
    Args:
        results_dict: Dictionary with algorithm names as keys and result dicts as values
        use_plotly: If True, use Plotly; else matplotlib
        
    Returns:
        Figure object
    """
    if use_plotly:
        fig = go.Figure()
        
        colors = {'SA': '#1f77b4', 'GA': '#ff7f0e', 'TS': '#2ca02c'}
        
        for alg_name, result in results_dict.items():
            if result and 'history_best' in result and result['history_best']:
                history = result['history_best']
                fig.add_trace(go.Scatter(
                    x=list(range(len(history))),
                    y=history,
                    mode='lines',
                    name=alg_name,
                    line=dict(color=colors.get(alg_name, '#000000'), width=2)
                ))
        
        fig.update_layout(
            title='Best Fitness Convergence Comparison',
            xaxis_title='Iteration',
            yaxis_title='Best Fitness',
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {'SA': '#1f77b4', 'GA': '#ff7f0e', 'TS': '#2ca02c'}
        
        for alg_name, result in results_dict.items():
            if result and 'history_best' in result and result['history_best']:
                history = result['history_best']
                ax.plot(history, label=alg_name, color=colors.get(alg_name, '#000000'), linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Fitness')
        ax.set_title('Best Fitness Convergence Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


def plot_current_fitness(results_dict, use_plotly=True):
    """
    Plot current fitness evolution.
    
    Args:
        results_dict: Dictionary with algorithm results
        use_plotly: If True, use Plotly
        
    Returns:
        Figure object
    """
    if use_plotly:
        fig = go.Figure()
        colors = {'SA': '#1f77b4', 'GA': '#ff7f0e', 'TS': '#2ca02c'}
        
        for alg_name, result in results_dict.items():
            if result and 'history_current' in result and result['history_current']:
                history = result['history_current']
                fig.add_trace(go.Scatter(
                    x=list(range(len(history))),
                    y=history,
                    mode='lines',
                    name=alg_name,
                    line=dict(color=colors.get(alg_name, '#000000'), width=1, dash='dash'),
                    opacity=0.7
                ))
        
        fig.update_layout(
            title='Current Fitness Evolution',
            xaxis_title='Iteration',
            yaxis_title='Current Fitness',
            template='plotly_white',
            height=500
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {'SA': '#1f77b4', 'GA': '#ff7f0e', 'TS': '#2ca02c'}
        
        for alg_name, result in results_dict.items():
            if result and 'history_current' in result and result['history_current']:
                history = result['history_current']
                ax.plot(history, label=alg_name, color=colors.get(alg_name, '#000000'), 
                       linewidth=1, linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Current Fitness')
        ax.set_title('Current Fitness Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


def plot_acceptance_rate(results_dict, use_plotly=True):
    """
    Plot acceptance rate (for SA) or move acceptance.
    
    Args:
        results_dict: Dictionary with algorithm results
        use_plotly: If True, use Plotly
        
    Returns:
        Figure object
    """
    if use_plotly:
        fig = go.Figure()
        colors = {'SA': '#1f77b4', 'GA': '#ff7f0e', 'TS': '#2ca02c'}
        
        for alg_name, result in results_dict.items():
            if result and 'accepted_hist' in result and result['accepted_hist']:
                accepted = result['accepted_hist']
                # Compute rolling mean if needed
                window = max(1, len(accepted) // 20)
                if len(accepted) > window:
                    rolling_mean = np.convolve(accepted, np.ones(window)/window, mode='valid')
                    x_vals = list(range(window-1, len(accepted)))
                else:
                    rolling_mean = accepted
                    x_vals = list(range(len(accepted)))
                
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=rolling_mean,
                    mode='lines',
                    name=f'{alg_name} Acceptance Rate',
                    line=dict(color=colors.get(alg_name, '#000000'), width=2)
                ))
        
        fig.update_layout(
            title='Acceptance Rate Evolution',
            xaxis_title='Iteration',
            yaxis_title='Acceptance Rate (rolling mean)',
            template='plotly_white',
            height=500
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {'SA': '#1f77b4', 'GA': '#ff7f0e', 'TS': '#2ca02c'}
        
        for alg_name, result in results_dict.items():
            if result and 'accepted_hist' in result and result['accepted_hist']:
                accepted = result['accepted_hist']
                window = max(1, len(accepted) // 20)
                if len(accepted) > window:
                    rolling_mean = np.convolve(accepted, np.ones(window)/window, mode='valid')
                    x_vals = list(range(window-1, len(accepted)))
                else:
                    rolling_mean = accepted
                    x_vals = list(range(len(accepted)))
                
                ax.plot(x_vals, rolling_mean, label=f'{alg_name}', 
                       color=colors.get(alg_name, '#000000'), linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Acceptance Rate (rolling mean)')
        ax.set_title('Acceptance Rate Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


def plot_penalty_evolution(results_dict, use_plotly=True):
    """
    Plot constraint penalty evolution.
    
    Args:
        results_dict: Dictionary with algorithm results
        use_plotly: If True, use Plotly
        
    Returns:
        Figure object
    """
    if use_plotly:
        fig = go.Figure()
        colors = {'SA': '#1f77b4', 'GA': '#ff7f0e', 'TS': '#2ca02c'}
        
        for alg_name, result in results_dict.items():
            if result and 'penalty_hist' in result and result['penalty_hist']:
                penalties = result['penalty_hist']
                fig.add_trace(go.Scatter(
                    x=list(range(len(penalties))),
                    y=penalties,
                    mode='lines',
                    name=alg_name,
                    line=dict(color=colors.get(alg_name, '#000000'), width=2)
                ))
        
        fig.update_layout(
            title='Constraint Penalty Evolution',
            xaxis_title='Iteration',
            yaxis_title='Penalty',
            template='plotly_white',
            height=500
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {'SA': '#1f77b4', 'GA': '#ff7f0e', 'TS': '#2ca02c'}
        
        for alg_name, result in results_dict.items():
            if result and 'penalty_hist' in result and result['penalty_hist']:
                penalties = result['penalty_hist']
                ax.plot(penalties, label=alg_name, color=colors.get(alg_name, '#000000'), linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Penalty')
        ax.set_title('Constraint Penalty Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


def plot_window_compliance_gantt(solution, constraints, N=60, use_plotly=True):
    """
    Plot Gantt chart showing window compliance for top N tasks.
    
    Args:
        solution: List of Task_ID
        constraints: Constraints dictionary
        N: Number of tasks to display
        use_plotly: If True, use Plotly
        
    Returns:
        Figure object
    """
    id_to_idx = constraints['id_to_idx']
    release_times = constraints['release_times']
    deadlines = constraints['deadlines']
    targets = constraints['targets']
    
    # Take first N tasks
    display_tasks = solution[:min(N, len(solution))]
    
    if use_plotly:
        fig = go.Figure()
        
        for i, task_id in enumerate(display_tasks):
            idx = id_to_idx[task_id]
            r_i = release_times[idx]
            d_i = deadlines[idx]
            is_urgent = targets[idx]
            
            # Check if scheduled position is in window
            in_window = r_i <= i <= d_i
            color = 'green' if in_window else 'red'
            
            # Bar for window
            fig.add_trace(go.Scatter(
                x=[r_i, d_i, d_i, r_i, r_i],
                y=[i-0.3, i-0.3, i+0.3, i+0.3, i-0.3],
                fill='toself',
                fillcolor=color,
                line=dict(color=color, width=1),
                mode='lines',
                showlegend=False,
                opacity=0.3,
                hoverinfo='skip'
            ))
            
            # Scheduled position marker
            fig.add_trace(go.Scatter(
                x=[i],
                y=[i],
                mode='markers',
                marker=dict(size=10, color='blue' if is_urgent else 'orange', symbol='circle'),
                name='Urgent' if is_urgent else 'Normal',
                showlegend=(i == 0),
                text=f'Task {task_id}<br>Window: [{r_i}, {d_i}]<br>Scheduled: {i}',
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title=f'Window Compliance Gantt (Top {len(display_tasks)} tasks)',
            xaxis_title='Position',
            yaxis_title='Task Index',
            template='plotly_white',
            height=max(400, len(display_tasks) * 8),
            showlegend=True
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(12, max(6, len(display_tasks) * 0.3)))
        
        for i, task_id in enumerate(display_tasks):
            idx = id_to_idx[task_id]
            r_i = release_times[idx]
            d_i = deadlines[idx]
            is_urgent = targets[idx]
            
            in_window = r_i <= i <= d_i
            color = 'green' if in_window else 'red'
            
            # Window bar
            ax.barh(i, d_i - r_i, left=r_i, color=color, alpha=0.3, height=0.6)
            
            # Scheduled position
            marker_color = 'blue' if is_urgent else 'orange'
            ax.scatter(i, i, s=100, c=marker_color, marker='o', zorder=5)
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Task Index')
        ax.set_title(f'Window Compliance Gantt (Top {len(display_tasks)} tasks)')
        ax.grid(True, alpha=0.3)
        
        return fig


def plot_capacity_usage(solution, constraints, use_plotly=True):
    """
    Plot capacity usage per slot with capacity line.
    
    Args:
        solution: List of Task_ID
        constraints: Constraints dictionary
        use_plotly: If True, use Plotly
        
    Returns:
        Figure object
    """
    n_tasks = len(solution)
    capacity_per_slot = constraints['capacity_per_slot']
    n_slots = constraints['n_slots']
    slot_size = max(1, n_tasks // n_slots)
    
    slot_usage = []
    for slot_idx in range(n_slots):
        start_pos = slot_idx * slot_size
        end_pos = min((slot_idx + 1) * slot_size, n_tasks)
        slot_tasks = end_pos - start_pos
        slot_usage.append(slot_tasks)
    
    if use_plotly:
        fig = go.Figure()
        
        # Capacity usage bars
        fig.add_trace(go.Bar(
            x=list(range(len(slot_usage))),
            y=slot_usage,
            name='Tasks per Slot',
            marker_color='lightblue'
        ))
        
        # Capacity limit line
        fig.add_trace(go.Scatter(
            x=list(range(len(slot_usage))),
            y=[capacity_per_slot] * len(slot_usage),
            mode='lines',
            name=f'Capacity Limit ({capacity_per_slot})',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Capacity Usage by Slot',
            xaxis_title='Slot Index',
            yaxis_title='Number of Tasks',
            template='plotly_white',
            height=500
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(range(len(slot_usage)), slot_usage, color='lightblue', label='Tasks per Slot')
        ax.axhline(y=capacity_per_slot, color='red', linestyle='--', linewidth=2, 
                  label=f'Capacity Limit ({capacity_per_slot})')
        
        ax.set_xlabel('Slot Index')
        ax.set_ylabel('Number of Tasks')
        ax.set_title('Capacity Usage by Slot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


def plot_position_distribution(solution, constraints, use_plotly=True):
    """
    Plot distribution of scheduled positions for urgent vs non-urgent tasks.
    
    Args:
        solution: List of Task_ID
        constraints: Constraints dictionary
        use_plotly: If True, use Plotly
        
    Returns:
        Figure object
    """
    id_to_idx = constraints['id_to_idx']
    targets = constraints['targets']
    
    urgent_positions = []
    non_urgent_positions = []
    
    for pos, task_id in enumerate(solution):
        idx = id_to_idx[task_id]
        if targets[idx] == 1:
            urgent_positions.append(pos)
        else:
            non_urgent_positions.append(pos)
    
    if use_plotly:
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=urgent_positions,
            name='Urgent Tasks',
            opacity=0.7,
            nbinsx=20,
            marker_color='red'
        ))
        
        fig.add_trace(go.Histogram(
            x=non_urgent_positions,
            name='Non-Urgent Tasks',
            opacity=0.7,
            nbinsx=20,
            marker_color='blue'
        ))
        
        fig.update_layout(
            title='Position Distribution: Urgent vs Non-Urgent',
            xaxis_title='Scheduled Position',
            yaxis_title='Frequency',
            template='plotly_white',
            height=500,
            barmode='overlay'
        )
        
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(urgent_positions, bins=20, alpha=0.7, label='Urgent Tasks', color='red')
        ax.hist(non_urgent_positions, bins=20, alpha=0.7, label='Non-Urgent Tasks', color='blue')
        
        ax.set_xlabel('Scheduled Position')
        ax.set_ylabel('Frequency')
        ax.set_title('Position Distribution: Urgent vs Non-Urgent')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig

