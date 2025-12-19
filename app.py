"""
Interactive Dashboard for Comparing Optimization Algorithms
SA, GA, and Tabu Search for Task Scheduling Problem
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import pickle
from datetime import datetime
import plotly.graph_objects as go

from utils.data import load_data, prepare_constraints, sample_data
from utils.metrics import analyze_solution
from utils.plots import (
    plot_convergence_comparison,
    plot_current_fitness,
    plot_acceptance_rate,
    plot_penalty_evolution,
    plot_window_compliance_gantt,
    plot_capacity_usage,
    plot_position_distribution,
)

from algorithms.sa import run_algorithm as run_sa
from algorithms.ga import run_algorithm as run_ga
from algorithms.tabu import run_algorithm as run_tabu

# Page config
st.set_page_config(
    page_title="Optimization Algorithms Comparison",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'constraints' not in st.session_state:
    st.session_state.constraints = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'results_loaded' not in st.session_state:
    st.session_state.results_loaded = False

# Results cache directory
RESULTS_DIR = "saved_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_results(results, constraints, params_hash, filename=None):
    """Save results to file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.pkl"
    
    filepath = os.path.join(RESULTS_DIR, filename)
    
    save_data = {
        'results': results,
        'constraints': constraints,
        'params_hash': params_hash,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)
    
    return filepath


def load_results(filename):
    """Load results from file."""
    filepath = os.path.join(RESULTS_DIR, filename)
    
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'rb') as f:
        save_data = pickle.load(f)
    
    return save_data


def list_saved_results():
    """List all saved result files."""
    if not os.path.exists(RESULTS_DIR):
        return []
    
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.pkl')]
    files.sort(reverse=True)  # Most recent first
    return files


def get_params_hash(common_params, sa_params, ga_params, tabu_params):
    """Create a hash of parameters to identify unique configurations."""
    import hashlib
    params_str = json.dumps({
        'common': common_params,
        'sa': sa_params,
        'ga': ga_params,
        'tabu': tabu_params
    }, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()[:8]


def load_dataset():
    """Load dataset from upload or file path."""
    st.sidebar.header("📁 Dataset")
    
    upload_method = st.sidebar.radio(
        "Data source",
        ["Upload CSV", "Local file path"],
        key="upload_method"
    )
    
    if upload_method == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Choose CSV file",
            type=['csv'],
            key="csv_upload"
        )
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.sidebar.success(f"✅ Loaded {len(data)} tasks")
                return data
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")
                return None
    else:
        file_path = st.sidebar.text_input(
            "Enter file path",
            value="cloud_task_scheduling_dataset.csv",
            key="file_path"
        )
        if file_path:
            try:
                data = load_data(file_path)
                st.session_state.data = data
                st.sidebar.success(f"✅ Loaded {len(data)} tasks")
                return data
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")
                return None
    
    return None


def get_common_params():
    """Get common parameters from sidebar."""
    st.sidebar.header("⚙️ Common Parameters")
    
    seed = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=42,
        key="seed"
    )
    
    n_sample = st.sidebar.number_input(
        "Number of tasks to sample (leave 0 for all)",
        min_value=0,
        value=0,
        key="n_sample"
    )
    
    capacity_per_slot = st.sidebar.number_input(
        "Capacity per slot",
        min_value=1,
        value=45,
        key="capacity"
    )
    
    cpu_threshold = st.sidebar.number_input(
        "CPU threshold (%)",
        min_value=0.0,
        max_value=100.0,
        value=90.0,
        key="cpu_threshold"
    )
    
    network_threshold = st.sidebar.number_input(
        "Network threshold (MB/s)",
        min_value=0.0,
        value=1.0,
        key="network_threshold"
    )
    
    return {
        'seed': int(seed),
        'n_sample': int(n_sample) if n_sample > 0 else None,
        'capacity_per_slot': int(capacity_per_slot),
        'cpu_threshold': cpu_threshold,
        'network_threshold': network_threshold,
    }


def get_sa_params():
    """Get SA parameters."""
    st.sidebar.header("🔥 Simulated Annealing")
    
    max_iter = st.sidebar.number_input(
        "Max iterations",
        min_value=100,
        value=2000,
        step=100,
        key="sa_max_iter"
    )
    
    T0 = st.sidebar.number_input(
        "Initial temperature (T0)",
        min_value=0.01,
        value=1.0,
        step=0.1,
        key="sa_T0"
    )
    
    alpha = st.sidebar.slider(
        "Cooling rate (alpha)",
        min_value=0.9,
        max_value=0.999,
        value=0.995,
        step=0.001,
        key="sa_alpha"
    )
    
    prob_smart_mut = st.sidebar.slider(
        "Smart mutation probability",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        key="sa_prob_smart"
    )
    
    return {
        'max_iter': int(max_iter),
        'T0': T0,
        'alpha': alpha,
        'prob_smart_mut': prob_smart_mut,
    }


def get_ga_params():
    """Get GA parameters."""
    st.sidebar.header("🧬 Genetic Algorithm")
    
    pop_size = st.sidebar.number_input(
        "Population size",
        min_value=10,
        value=80,
        step=10,
        key="ga_pop_size"
    )
    
    generations = st.sidebar.number_input(
        "Generations",
        min_value=50,
        value=300,
        step=50,
        key="ga_generations"
    )
    
    crossover_rate = st.sidebar.slider(
        "Crossover rate",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.05,
        key="ga_crossover"
    )
    
    mutation_rate = st.sidebar.slider(
        "Mutation rate",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        key="ga_mutation"
    )
    
    elitism = st.sidebar.number_input(
        "Elitism",
        min_value=1,
        value=4,
        key="ga_elitism"
    )
    
    return {
        'pop_size': int(pop_size),
        'generations': int(generations),
        'crossover_rate': crossover_rate,
        'mutation_rate': mutation_rate,
        'elitism': int(elitism),
    }


def get_tabu_params():
    """Get Tabu Search parameters."""
    st.sidebar.header("🚫 Tabu Search")
    
    max_iter = st.sidebar.number_input(
        "Max iterations",
        min_value=50,
        value=300,
        step=50,
        key="tabu_max_iter"
    )
    
    tabu_size = st.sidebar.number_input(
        "Tabu list size (tenure)",
        min_value=5,
        value=20,
        step=5,
        key="tabu_size"
    )
    
    neighborhood_size = st.sidebar.number_input(
        "Neighborhood size",
        min_value=10,
        value=80,
        step=10,
        key="tabu_neighborhood"
    )
    
    max_no_improve = st.sidebar.number_input(
        "Max iterations without improvement",
        min_value=10,
        value=60,
        step=10,
        key="tabu_no_improve"
    )
    
    return {
        'max_iter': int(max_iter),
        'tabu_size': int(tabu_size),
        'neighborhood_size': int(neighborhood_size),
        'max_no_improve': int(max_no_improve),
    }


def run_all_algorithms(data_dict, sa_params, ga_params, tabu_params, use_cache=True):
    """Run all algorithms with optional caching."""
    # Check for cached results first
    if use_cache:
        params_hash = get_params_hash(
            st.session_state.common_params,
            sa_params, ga_params, tabu_params
        )
        cached_files = list_saved_results()
        
        # Try to find matching cached result
        for filename in cached_files:
            try:
                cached_data = load_results(filename)
                if cached_data and cached_data.get('params_hash') == params_hash:
                    st.info(f"📦 Loaded cached results from {filename}")
                    return cached_data['results'], cached_data['constraints']
            except:
                continue
    
    # Run algorithms if no cache found
    results = {}
    
    # Run SA
    try:
        sa_result = run_sa(data_dict, sa_params)
        results['SA'] = sa_result
    except Exception as e:
        st.error(f"SA error: {e}")
        results['SA'] = None
    
    # Run GA
    try:
        ga_result = run_ga(data_dict, ga_params)
        results['GA'] = ga_result
    except Exception as e:
        st.error(f"GA error: {e}")
        results['GA'] = None
    
    # Run Tabu
    try:
        tabu_result = run_tabu(data_dict, tabu_params)
        results['TS'] = tabu_result
    except Exception as e:
        st.error(f"Tabu error: {e}")
        results['TS'] = None
    
    return results, data_dict


def tab_overview():
    """Tab 1: Overview with KPIs and summary table."""
    st.header("📊 Overview")
    
    if not st.session_state.results:
        st.info("👆 Please run algorithms first using the sidebar controls.")
        return
    
    # KPI Cards
    cols = st.columns(3)
    algorithms = ['SA', 'GA', 'TS']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, alg in enumerate(algorithms):
        with cols[i]:
            if st.session_state.results.get(alg):
                result = st.session_state.results[alg]
                best_fit = result['best_fitness']
                runtime = result['runtime_sec']
                
                st.metric(
                    label=f"{alg} - Best Fitness",
                    value=f"{best_fit:.2f}",
                    delta=f"{runtime:.2f}s"
                )
            else:
                st.metric(label=alg, value="N/A")
    
    st.divider()
    
    # Detailed metrics table
    st.subheader("📈 Detailed Metrics")
    
    metrics_data = []
    for alg in algorithms:
        if st.session_state.results.get(alg):
            result = st.session_state.results[alg]
            solution = result['best_solution']
            analysis = analyze_solution(solution, st.session_state.constraints)
            
            metrics_data.append({
                'Algorithm': alg,
                'Best Fitness': f"{result['best_fitness']:.2f}",
                'Runtime (s)': f"{result['runtime_sec']:.2f}",
                '% In Window': f"{100 * analysis['in_window'] / analysis['n_tasks']:.1f}%",
                '% Context OK': f"{100 * analysis['context_ok'] / analysis['n_tasks']:.1f}%",
                'Capacity Violations': analysis['capacity_violations'],
                'Violation Magnitude': f"{analysis['capacity_violation_magnitude']:.1f}",
            })
    
    if metrics_data:
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)
        
        # Ranking
        st.subheader("🏆 Ranking")
        df_rank = pd.DataFrame(metrics_data)
        df_rank['Best Fitness'] = df_rank['Best Fitness'].astype(float)
        df_rank['Runtime (s)'] = df_rank['Runtime (s)'].astype(float)
        df_rank = df_rank.sort_values('Best Fitness', ascending=False)
        st.dataframe(df_rank[['Algorithm', 'Best Fitness', 'Runtime (s)']], 
                    use_container_width=True, hide_index=True)
    else:
        st.warning("No results available.")


def tab_convergence():
    """Tab 2: Convergence and search behavior."""
    st.header("📈 Convergence & Search Behavior")
    
    if not st.session_state.results:
        st.info("👆 Please run algorithms first.")
        return
    
    # Filter available results
    available_results = {k: v for k, v in st.session_state.results.items() if v is not None}
    
    if not available_results:
        st.warning("No results available.")
        return
    
    # Best fitness comparison
    st.subheader("Best Fitness Convergence")
    fig_best = plot_convergence_comparison(available_results, use_plotly=True)
    st.plotly_chart(fig_best, use_container_width=True)
    
    # Current fitness
    st.subheader("Current Fitness Evolution")
    fig_current = plot_current_fitness(available_results, use_plotly=True)
    st.plotly_chart(fig_current, use_container_width=True)
    
    # Acceptance rate
    st.subheader("Acceptance Rate")
    fig_accept = plot_acceptance_rate(available_results, use_plotly=True)
    st.plotly_chart(fig_accept, use_container_width=True)
    
    # Penalty evolution
    st.subheader("Constraint Penalty Evolution")
    fig_penalty = plot_penalty_evolution(available_results, use_plotly=True)
    st.plotly_chart(fig_penalty, use_container_width=True)


def tab_constraints():
    """Tab 3: Constraint diagnostics."""
    st.header("🔍 Constraint Diagnostics")
    
    if not st.session_state.results:
        st.info("👆 Please run algorithms first.")
        return
    
    # Select algorithm to visualize
    alg_options = [k for k, v in st.session_state.results.items() if v is not None]
    if not alg_options:
        st.warning("No results available.")
        return
    
    selected_alg = st.selectbox("Select algorithm to visualize", alg_options)
    result = st.session_state.results[selected_alg]
    solution = result['best_solution']
    
    # Window compliance Gantt
    st.subheader("Window Compliance (Gantt Chart)")
    N = st.slider("Number of tasks to display", 20, min(200, len(solution)), 60)
    fig_gantt = plot_window_compliance_gantt(solution, st.session_state.constraints, N=N, use_plotly=True)
    st.plotly_chart(fig_gantt, use_container_width=True)
    
    # Capacity usage
    st.subheader("Capacity Usage by Slot")
    fig_capacity = plot_capacity_usage(solution, st.session_state.constraints, use_plotly=True)
    st.plotly_chart(fig_capacity, use_container_width=True)
    
    # Position distribution
    st.subheader("Position Distribution: Urgent vs Non-urgent")
    fig_dist = plot_position_distribution(solution, st.session_state.constraints, use_plotly=True)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Top 10 tasks table
    st.subheader("Top 10 Scheduled Tasks")
    id_to_idx = st.session_state.constraints["id_to_idx"]
    priorities = st.session_state.constraints["priorities"]
    targets = st.session_state.constraints["targets"]
    release_times = st.session_state.constraints["release_times"]
    deadlines = st.session_state.constraints["deadlines"]
    context_available = st.session_state.constraints["context_available"]
    
    top_tasks_data = []
    for pos in range(min(10, len(solution))):
        task_id = solution[pos]
        idx = id_to_idx[task_id]
        top_tasks_data.append({
            'Position': pos,
            'Task_ID': task_id,
            'Priority': f"{priorities[idx]:.2f}",
            'Urgent': 'Yes' if targets[idx] == 1 else 'No',
            'Window': f"[{release_times[idx]}, {deadlines[idx]}]",
            'Context': 'OK' if context_available[idx] == 1 else 'Bad',
        })
    
    df_top = pd.DataFrame(top_tasks_data)
    st.dataframe(df_top, use_container_width=True, hide_index=True)


def tab_complexity():
    """Tab 4: Complexity and scalability analysis."""
    st.header("⚙️ Complexity & Scalability")
    
    # Theoretical complexity
    st.subheader("📚 Theoretical Complexity Analysis")
    
    # Complexity summary table
    st.markdown("### 📊 Complexité Temporelle (Big-O)")
    complexity_data = {
        'Algorithme': ['Simulated Annealing (SA)', 'Genetic Algorithm (GA)', 'Tabu Search (TS)'],
        'Complexité Temporelle': ['O(I × n)', 'O(G × P × n)', 'O(I × m × n)']
    }
    df_complexity = pd.DataFrame(complexity_data)
    st.dataframe(df_complexity, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Légende :**
    - **I** = nombre d'itérations
    - **G** = nombre de générations
    - **P** = taille de la population
    - **m** = taille du voisinage
    - **n** = nombre de tâches
    """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Simulated Annealing (SA)**
        - **Time Complexity**: O(I × n)
          - I = iterations
          - n = number of tasks
          - Dominated by: fitness evaluation per iteration
        - **Space Complexity**: O(n)
          - Store current and best solution
        
        **Genetic Algorithm (GA)**
        - **Time Complexity**: O(G × P × n)
          - G = generations
          - P = population size
          - n = number of tasks
          - Dominated by: fitness evaluation for entire population each generation
        - **Space Complexity**: O(P × n)
          - Store entire population
        """)
    
    with col2:
        st.markdown("""
        **Tabu Search (TS)**
        - **Time Complexity**: O(I × m × n)
          - I = iterations
          - m = neighborhood size
          - n = number of tasks
          - Dominated by: exploring neighborhood each iteration
        - **Space Complexity**: O(n + tabu_size)
          - Store current solution + tabu list
        
        **Runtime Dominance:**
        - SA: Fitness evaluation (O(n)) × iterations
        - GA: Fitness evaluation (O(n)) × population × generations
        - TS: Neighborhood exploration (O(m × n)) × iterations
        """)
    
    st.divider()
    
    # Empirical scaling
    st.subheader("📊 Empirical Scaling Experiment")
    
    # Check if we have preset scaling data
    has_preset_scaling = False
    scaling_results = {'size': [100, 200, 400, 800], 'SA': [], 'GA': [], 'TS': []}
    
    if st.session_state.results:
        # Try to get scaling data from preset results
        for alg in ['SA', 'GA', 'TS']:
            if alg in st.session_state.results and st.session_state.results[alg]:
                result = st.session_state.results[alg]
                if 'scaling_data' in result:
                    scaling_results[alg] = result['scaling_data']['runtimes']
                    has_preset_scaling = True
    
    if has_preset_scaling:
        # Use preset scaling data (exact values)
        st.info("📊 Using exact scaling data from preset results")
        
        # Plot results
        fig = go.Figure()
        
        colors = {'SA': '#1f77b4', 'GA': '#ff7f0e', 'TS': '#2ca02c'}
        
        for alg in ['SA', 'GA', 'TS']:
            if scaling_results[alg]:
                fig.add_trace(go.Scatter(
                    x=scaling_results['size'],
                    y=scaling_results[alg],
                    mode='lines+markers',
                    name=alg,
                    line=dict(width=2, color=colors.get(alg)),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title='Empirical Runtime Scaling (Exact Values)',
            xaxis_title='Number of Tasks (n)',
            yaxis_title='Runtime (seconds)',
            yaxis_type='log',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display table with exact values
        df_scaling = pd.DataFrame(scaling_results)
        
        # Format values for better readability
        df_display = df_scaling.copy()
        for col in ['SA', 'GA', 'TS']:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Add explanation
        with st.expander("📖 Explication du tableau"):
            st.markdown("""
            **Ce tableau montre les temps d'exécution réels (en secondes) pour chaque algorithme :**
            
            - **SA (Simulated Annealing)** : Le plus rapide, scaling presque linéaire
            - **GA (Genetic Algorithm)** : Le plus lent, scaling super-linéaire  
            - **TS (Tabu Search)** : Intermédiaire, scaling quasi-linéaire
            
            **Observations :**
            - SA reste rapide même sur 800 tâches (1.46s)
            - GA devient très lent sur grands problèmes (77.46s pour 800 tâches)
            - TS offre un bon compromis vitesse/qualité
            
            Ces valeurs confirment l'analyse théorique de complexité.
            """)
        
        st.caption("💡 These are the exact runtime values from your empirical tests.")
        
        # Evolution graphs over time
        st.divider()
        st.subheader("📈 Evolution de la Fitness en Fonction du Temps")
        st.markdown("Graphiques montrant l'évolution de la fitness de chaque algorithme au cours du temps d'exécution.")
        
        # Create three columns for three graphs
        col1, col2, col3 = st.columns(3)
        
        # SA Evolution
        with col1:
            if 'SA' in st.session_state.results and st.session_state.results['SA']:
                sa_result = st.session_state.results['SA']
                if 'history_best' in sa_result and 'runtime_sec' in sa_result:
                    # Calculate time points
                    n_iter = len(sa_result['history_best'])
                    total_time = sa_result['runtime_sec']
                    time_points = [total_time * (i / n_iter) for i in range(n_iter)]
                    
                    fig_sa = go.Figure()
                    fig_sa.add_trace(go.Scatter(
                        x=time_points,
                        y=sa_result['history_best'],
                        mode='lines',
                        name='Best Fitness',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    fig_sa.update_layout(
                        title='SA - Evolution',
                        xaxis_title='Temps (secondes)',
                        yaxis_title='Fitness',
                        template='plotly_white',
                        height=350,
                        showlegend=True
                    )
                    st.plotly_chart(fig_sa, use_container_width=True)
        
        # GA Evolution
        with col2:
            if 'GA' in st.session_state.results and st.session_state.results['GA']:
                ga_result = st.session_state.results['GA']
                if 'history_best' in ga_result and 'runtime_sec' in ga_result:
                    # Calculate time points
                    n_iter = len(ga_result['history_best'])
                    total_time = ga_result['runtime_sec']
                    time_points = [total_time * (i / n_iter) for i in range(n_iter)]
                    
                    fig_ga = go.Figure()
                    fig_ga.add_trace(go.Scatter(
                        x=time_points,
                        y=ga_result['history_best'],
                        mode='lines',
                        name='Best Fitness',
                        line=dict(color='#ff7f0e', width=2)
                    ))
                    fig_ga.update_layout(
                        title='GA - Evolution',
                        xaxis_title='Temps (secondes)',
                        yaxis_title='Fitness',
                        template='plotly_white',
                        height=350,
                        showlegend=True
                    )
                    st.plotly_chart(fig_ga, use_container_width=True)
        
        # TS Evolution
        with col3:
            if 'TS' in st.session_state.results and st.session_state.results['TS']:
                ts_result = st.session_state.results['TS']
                if 'history_best' in ts_result and 'runtime_sec' in ts_result:
                    # Calculate time points
                    n_iter = len(ts_result['history_best'])
                    total_time = ts_result['runtime_sec']
                    time_points = [total_time * (i / n_iter) for i in range(n_iter)]
                    
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(
                        x=time_points,
                        y=ts_result['history_best'],
                        mode='lines',
                        name='Best Fitness',
                        line=dict(color='#2ca02c', width=2)
                    ))
                    fig_ts.update_layout(
                        title='TS - Evolution',
                        xaxis_title='Temps (secondes)',
                        yaxis_title='Fitness',
                        template='plotly_white',
                        height=350,
                        showlegend=True
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)
        
        # Combined comparison graph
        st.divider()
        st.subheader("📊 Comparaison: Evolution de Tous les Algorithmes")
        
        fig_combined = go.Figure()
        colors = {'SA': '#1f77b4', 'GA': '#ff7f0e', 'TS': '#2ca02c'}
        
        for alg in ['SA', 'GA', 'TS']:
            if alg in st.session_state.results and st.session_state.results[alg]:
                result = st.session_state.results[alg]
                if 'history_best' in result and 'runtime_sec' in result:
                    n_iter = len(result['history_best'])
                    total_time = result['runtime_sec']
                    time_points = [total_time * (i / n_iter) for i in range(n_iter)]
                    
                    fig_combined.add_trace(go.Scatter(
                        x=time_points,
                        y=result['history_best'],
                        mode='lines',
                        name=alg,
                        line=dict(color=colors.get(alg), width=2)
                    ))
        
        fig_combined.update_layout(
            title='Evolution Comparée de la Fitness',
            xaxis_title='Temps (secondes)',
            yaxis_title='Fitness',
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_combined, use_container_width=True)
    
    else:
        # Original scaling experiment (if no preset data)
        if st.session_state.data is None:
            st.info("👆 Please load dataset first or load preset results to see exact scaling data.")
            return
        
        if st.button("Run Scaling Experiment", type="primary", key="run_scaling"):
            with st.spinner("Running scaling experiment... This may take a while."):
                sizes = [100, 200, 400, 800]
                max_size = len(st.session_state.data)
                sizes = [s for s in sizes if s <= max_size]
                
                scaling_results = {'size': [], 'SA': [], 'GA': [], 'TS': []}
                
                # Use parameters from session state (set in main sidebar)
                if 'common_params' not in st.session_state:
                    st.warning("⚠️ Please configure parameters in the sidebar first, then run algorithms.")
                    return
                
                common_params = st.session_state.common_params.copy()
                sa_params = st.session_state.sa_params.copy()
                ga_params = st.session_state.ga_params.copy()
                tabu_params = st.session_state.tabu_params.copy()
                
                # Reduce iterations for scaling test
                sa_params['max_iter'] = min(500, sa_params['max_iter'])
                ga_params['generations'] = min(100, ga_params['generations'])
                tabu_params['max_iter'] = min(150, tabu_params['max_iter'])
                
                for size in sizes:
                    st.write(f"Testing with {size} tasks...")
                    
                    # Sample data
                    sampled_data = sample_data(st.session_state.data, n_sample=size, seed=common_params['seed'])
                    constraints = prepare_constraints(
                        sampled_data,
                        capacity_per_slot=common_params['capacity_per_slot'],
                        cpu_threshold=common_params['cpu_threshold'],
                        network_threshold=common_params['network_threshold'],
                        seed=common_params['seed']
                    )
                    
                    scaling_results['size'].append(size)
                    
                    # Run SA
                    try:
                        sa_result = run_sa(constraints, sa_params)
                        scaling_results['SA'].append(sa_result['runtime_sec'])
                    except:
                        scaling_results['SA'].append(None)
                    
                    # Run GA
                    try:
                        ga_result = run_ga(constraints, ga_params)
                        scaling_results['GA'].append(ga_result['runtime_sec'])
                    except:
                        scaling_results['GA'].append(None)
                    
                    # Run TS
                    try:
                        tabu_result = run_tabu(constraints, tabu_params)
                        scaling_results['TS'].append(tabu_result['runtime_sec'])
                    except:
                        scaling_results['TS'].append(None)
                
                # Plot results
                fig = go.Figure()
                
                for alg in ['SA', 'GA', 'TS']:
                    times = scaling_results[alg]
                    valid_indices = [i for i, t in enumerate(times) if t is not None]
                    valid_sizes = [scaling_results['size'][i] for i in valid_indices]
                    valid_times = [times[i] for i in valid_indices]
                    
                    if valid_times:
                        fig.add_trace(go.Scatter(
                            x=valid_sizes,
                            y=valid_times,
                            mode='lines+markers',
                            name=alg,
                            line=dict(width=2)
                        ))
                
                fig.update_layout(
                    title='Empirical Runtime Scaling',
                    xaxis_title='Number of Tasks (n)',
                    yaxis_title='Runtime (seconds)',
                    yaxis_type='log',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display table
                df_scaling = pd.DataFrame(scaling_results)
                st.dataframe(df_scaling, use_container_width=True, hide_index=True)




# Main app
def main():
    st.title("📊 Optimization Algorithms Comparison Dashboard")
    st.markdown("Compare **Simulated Annealing**, **Genetic Algorithm**, and **Tabu Search** for task scheduling")
    
    # Sidebar
    data = load_dataset()
    
    if data is not None:
        # Auto-load preset results on first load if available
        preset_file = "preset_results_from_notebook.pkl"
        if not st.session_state.results_loaded and preset_file in list_saved_results():
            try:
                cached_data = load_results(preset_file)
                if cached_data:
                    st.session_state.results = cached_data['results']
                    st.session_state.constraints = cached_data['constraints']
                    st.session_state.results_loaded = True
                    st.info("⚡ Loaded preset results automatically! (You can still run algorithms if needed)")
            except:
                pass
        
        # Get parameters
        common_params = get_common_params()
        sa_params = get_sa_params()
        ga_params = get_ga_params()
        tabu_params = get_tabu_params()
        
        # Merge seed into algorithm params
        sa_params['seed'] = common_params['seed']
        ga_params['seed'] = common_params['seed']
        tabu_params['seed'] = common_params['seed']
        
        # Store params in session state for reuse in other tabs
        st.session_state.common_params = common_params
        st.session_state.sa_params = sa_params
        st.session_state.ga_params = ga_params
        st.session_state.tabu_params = tabu_params
        
        st.sidebar.divider()
        
        # Run buttons
        st.sidebar.header("🚀 Run Algorithms")
        
        # Quick load preset results
        if preset_file in list_saved_results():
            if st.sidebar.button("⚡ Load Preset Results (Fast)", type="primary", use_container_width=True, key="load_preset"):
                try:
                    cached_data = load_results(preset_file)
                    if cached_data:
                        st.session_state.results = cached_data['results']
                        st.session_state.constraints = cached_data['constraints']
                        st.session_state.results_loaded = True
                        st.sidebar.success("✅ Loaded preset results!")
                        st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Error loading preset: {e}")
        
        st.sidebar.divider()
        
        # Cache option
        use_cache = st.sidebar.checkbox("Use cached results (if available)", value=True, key="use_cache")
        auto_save = st.sidebar.checkbox("Auto-save results", value=True, key="auto_save")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            run_all = st.button("▶️ Run All", type="primary", use_container_width=True)
        with col2:
            clear_results = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_results:
            st.session_state.results = {}
            st.session_state.results_loaded = False
            st.rerun()
        
        if run_all:
            with st.spinner("Running algorithms... This may take a while."):
                # Prepare data
                sampled_data = sample_data(data, n_sample=common_params['n_sample'], seed=common_params['seed'])
                constraints = prepare_constraints(
                    sampled_data,
                    capacity_per_slot=common_params['capacity_per_slot'],
                    cpu_threshold=common_params['cpu_threshold'],
                    network_threshold=common_params['network_threshold'],
                    seed=common_params['seed']
                )
                
                st.session_state.constraints = constraints
                
                # Run algorithms (with caching)
                results, _ = run_all_algorithms(constraints, sa_params, ga_params, tabu_params, use_cache=use_cache)
                st.session_state.results = results
                
                # Auto-save results
                if auto_save and results and any(results.values()):
                    params_hash = get_params_hash(common_params, sa_params, ga_params, tabu_params)
                    save_path = save_results(results, constraints, params_hash)
                    st.session_state.results_loaded = True
                    st.success(f"✅ Algorithms completed! Results saved to {os.path.basename(save_path)}")
                else:
                    st.success("✅ Algorithms completed!")
                
                st.rerun()
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview",
            "📈 Convergence",
            "🔍 Constraints",
            "⚙️ Complexity"
        ])
        
        with tab1:
            tab_overview()
        
        with tab2:
            tab_convergence()
        
        with tab3:
            tab_constraints()
        
        with tab4:
            tab_complexity()
        
        # Load/Save Results section
        st.sidebar.divider()
        st.sidebar.header("💾 Results Management")
        
        # Load saved results
        saved_files = list_saved_results()
        if saved_files:
            st.sidebar.subheader("📂 Load Saved Results")
            
            # Highlight preset file
            preset_file = "preset_results_from_notebook.pkl"
            if preset_file in saved_files:
                st.sidebar.info("💡 Tip: Use 'Load Preset Results' button above for instant loading!")
            
            selected_file = st.sidebar.selectbox(
                "Select saved results file",
                saved_files,
                key="load_results_file",
                index=0 if preset_file in saved_files else None
            )
            
            if st.sidebar.button("📥 Load Results", key="load_results_btn"):
                try:
                    cached_data = load_results(selected_file)
                    if cached_data:
                        st.session_state.results = cached_data['results']
                        st.session_state.constraints = cached_data['constraints']
                        st.session_state.results_loaded = True
                        st.sidebar.success(f"✅ Loaded {selected_file}")
                        st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Error loading: {e}")
        
        # Save current results
        if st.session_state.results:
            st.sidebar.subheader("💾 Save Current Results")
            save_filename = st.sidebar.text_input(
                "Filename (optional)",
                value="",
                key="save_filename",
                help="Leave empty for auto-generated name"
            )
            
            if st.sidebar.button("💾 Save Results", key="save_results_btn"):
                params_hash = get_params_hash(common_params, sa_params, ga_params, tabu_params)
                filename = save_filename if save_filename else None
                save_path = save_results(
                    st.session_state.results,
                    st.session_state.constraints,
                    params_hash,
                    filename
                )
                st.sidebar.success(f"✅ Saved to {os.path.basename(save_path)}")
        
        # Export section
        st.sidebar.divider()
        st.sidebar.header("📤 Export Results")
        
        if st.session_state.results:
            export_format = st.sidebar.selectbox("Export format", ["CSV", "JSON"], key="export_format")
            
            # Create comprehensive CSV
            rows = []
            for alg, result in st.session_state.results.items():
                if result:
                    solution = result['best_solution']
                    id_to_idx = st.session_state.constraints["id_to_idx"]
                    priorities = st.session_state.constraints["priorities"]
                    targets = st.session_state.constraints["targets"]
                    release_times = st.session_state.constraints["release_times"]
                    deadlines = st.session_state.constraints["deadlines"]
                    context_available = st.session_state.constraints["context_available"]
                    
                    for pos, task_id in enumerate(solution):
                        idx = id_to_idx[task_id]
                        rows.append({
                            'Algorithm': alg,
                            'Position': pos,
                            'Task_ID': task_id,
                            'Priority': priorities[idx],
                            'Urgent': targets[idx],
                            'Release_Time': release_times[idx],
                            'Deadline': deadlines[idx],
                            'Context_OK': context_available[idx],
                        })
            
            if export_format == "CSV":
                df_export = pd.DataFrame(rows)
                csv = df_export.to_csv(index=False)
                csv_bytes = csv.encode('utf-8')
                st.sidebar.download_button(
                    label="Download CSV",
                    data=csv_bytes,
                    file_name="results.csv",
                    mime="text/csv"
                )
            else:  # JSON
                export_data = {}
                for alg, result in st.session_state.results.items():
                    if result:
                        export_data[alg] = {
                            'best_fitness': result['best_fitness'],
                            'runtime_sec': result['runtime_sec'],
                            'n_iter': result['n_iter'],
                            'meta': result['meta'],
                            'best_solution': result['best_solution'],
                        }
                
                json_str = json.dumps(export_data, indent=2)
                json_bytes = json_str.encode('utf-8')
                st.sidebar.download_button(
                    label="Download JSON",
                    data=json_bytes,
                    file_name="results.json",
                    mime="application/json"
                )
    else:
        st.info("👈 Please load a dataset from the sidebar to begin.")


if __name__ == "__main__":
    main()

