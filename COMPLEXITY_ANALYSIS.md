# Complexity Analysis Documentation

This document explains how the time and space complexity are calculated for each optimization algorithm (Simulated Annealing, Genetic Algorithm, and Tabu Search) used in the task scheduling problem.

---

## Table of Contents

1. [General Approach](#general-approach)
2. [Simulated Annealing (SA)](#simulated-annealing-sa)
3. [Genetic Algorithm (GA)](#genetic-algorithm-ga)
4. [Tabu Search (TS)](#tabu-search-ts)
5. [Empirical Scaling](#empirical-scaling)
6. [Comparison Summary](#comparison-summary)

---

## General Approach

### Complexity Analysis Method

For each algorithm, we analyze:

1. **Time Complexity**: How the runtime scales with input size
2. **Space Complexity**: How memory usage scales with input size
3. **Dominant Operations**: What operations consume the most time

### Key Variables

- **n** = number of tasks (problem size)
- **I** = number of iterations (for SA and TS)
- **G** = number of generations (for GA)
- **P** = population size (for GA)
- **m** = neighborhood size per iteration (for TS)
- **tabu_size** = tabu list length (for TS)

---

## Simulated Annealing (SA)

### Algorithm Overview

Simulated Annealing is a single-solution metaheuristic that:
1. Starts with a random solution
2. Generates neighbors by swapping tasks
3. Accepts better solutions or worse solutions with decreasing probability
4. Cools down the temperature over time

### Time Complexity Analysis

#### Main Loop Structure
```python
for iteration in range(max_iter):  # I iterations
    # Generate neighbor: O(1) - single swap
    # Evaluate fitness: O(n) - must check all tasks
    # Update solution: O(1)
```

#### Step-by-Step Breakdown

1. **Initialization**: O(n)
   - Create random permutation: O(n)
   - Evaluate initial fitness: O(n)

2. **Per Iteration**: O(n)
   - **Neighbor Generation**: O(1)
     - Random swap: O(1)
     - Smart mutation (if used): O(1) to O(n) worst case, but typically O(1)
   - **Fitness Evaluation**: O(n)
     - Must iterate through all n tasks to compute:
       - Position rewards (priority-weighted)
       - Window constraint violations
       - Context availability checks
       - Capacity violations per slot
   - **Acceptance Decision**: O(1)
     - Compare fitness values: O(1)
     - Calculate acceptance probability: O(1)
   - **Solution Update**: O(1)
     - Copy solution if accepted: O(n) worst case, but typically O(1) for references

3. **Total Time Complexity**: **O(I × n)**
   - I iterations × O(n) fitness evaluation per iteration
   - Dominated by: **Fitness evaluation** (O(n)) repeated I times

#### Space Complexity

- **Current Solution**: O(n) - stores permutation of n task IDs
- **Best Solution**: O(n) - stores best permutation found
- **History Arrays**: O(I) - stores fitness history
  - `history_best`: O(I)
  - `history_current`: O(I)
  - `accepted_hist`: O(I)
  - `penalty_hist`: O(I)

**Total Space Complexity**: **O(n + I)**
- Dominated by: **Solution storage** (O(n)) when I << n, or **History** (O(I)) when I >> n
- Typically: **O(n)** since I is usually much larger than n

---

## Genetic Algorithm (GA)

### Algorithm Overview

Genetic Algorithm is a population-based metaheuristic that:
1. Maintains a population of solutions
2. Selects parents using tournament selection
3. Creates offspring via crossover and mutation
4. Replaces population with new generation
5. Preserves elite individuals

### Time Complexity Analysis

#### Main Loop Structure
```python
for generation in range(generations):  # G generations
    # Evaluate entire population: O(P × n)
    # Selection: O(P × tournament_size)
    # Crossover: O(P × n)
    # Mutation: O(P × n)
    # Update population: O(P × n)
```

#### Step-by-Step Breakdown

1. **Initialization**: O(P × n)
   - Create P random permutations: O(P × n)
   - Evaluate initial population fitness: O(P × n)

2. **Per Generation**: O(P × n)
   - **Fitness Evaluation**: O(P × n)
     - Evaluate all P individuals: P × O(n) = O(P × n)
     - Each fitness evaluation: O(n) (same as SA)
   
   - **Selection**: O(P × k)
     - Tournament selection with k=3: O(P × 3) = O(P)
     - Per individual: O(k) comparisons
   
   - **Crossover**: O(P × n)
     - Order crossover (OX): O(n) per child
     - Create P children: O(P × n)
   
   - **Mutation**: O(P × n)
     - Swap mutation: O(1) per individual
     - Applied to P individuals: O(P)
     - But worst case checking: O(P × n)
   
   - **Elitism**: O(elitism × n)
     - Copy best individuals: O(elitism × n) = O(P × n) worst case
     - Typically elitism << P, so O(n)

3. **Total Time Complexity**: **O(G × P × n)**
   - G generations × O(P × n) operations per generation
   - Dominated by: **Fitness evaluation** (O(P × n)) repeated G times

#### Space Complexity

- **Population**: O(P × n) - stores P permutations of n tasks
- **Fitness Scores**: O(P) - stores fitness for each individual
- **Temporary Arrays**: O(n) - for crossover/mutation operations
- **History Arrays**: O(G) - stores generation-level history
  - `history_best`: O(G)
  - `history_current`: O(G)
  - `accepted_hist`: O(G)
  - `penalty_hist`: O(G)

**Total Space Complexity**: **O(P × n + G)**
- Dominated by: **Population storage** (O(P × n))
- Typically: **O(P × n)** since P × n >> G in most cases

---

## Tabu Search (TS)

### Algorithm Overview

Tabu Search is a memory-based metaheuristic that:
1. Maintains a current solution
2. Explores a neighborhood of solutions
3. Selects best non-tabu neighbor
4. Updates tabu list to avoid revisiting solutions
5. Uses aspiration criterion to override tabu status

### Time Complexity Analysis

#### Main Loop Structure
```python
for iteration in range(max_iter):  # I iterations
    # Generate m neighbors: O(m × n)
    # Evaluate neighbors: O(m × n)
    # Check tabu list: O(m × tabu_size)
    # Update solution: O(1)
    # Update tabu list: O(1)
```

#### Step-by-Step Breakdown

1. **Initialization**: O(n)
   - Create random permutation: O(n)
   - Evaluate initial fitness: O(n)
   - Initialize tabu list: O(1)

2. **Per Iteration**: O(m × n)
   - **Neighbor Generation**: O(m × n)
     - Generate m neighbors: O(m)
     - Each neighbor requires swap: O(1)
     - But creating m copies: O(m × n)
   
   - **Fitness Evaluation**: O(m × n)
     - Evaluate all m neighbors: m × O(n) = O(m × n)
     - Each fitness evaluation: O(n) (same as SA)
   
   - **Tabu List Check**: O(m × tabu_size)
     - For each neighbor: check against tabu list
     - Tabu list contains up to `tabu_size` entries
     - Each check: O(n) comparison (comparing solutions)
     - Total: O(m × tabu_size × n) worst case
     - But typically: O(m × tabu_size) if using hash comparison
   
   - **Best Neighbor Selection**: O(m)
     - Find maximum fitness: O(m) comparisons
   
   - **Tabu List Update**: O(1) to O(n)
     - Add solution to list: O(1) if using hash
     - Remove oldest if full: O(1)
     - But storing solution: O(n) space

3. **Total Time Complexity**: **O(I × m × n)**
   - I iterations × O(m × n) operations per iteration
   - Dominated by: **Neighborhood exploration** (O(m × n)) repeated I times
   - Note: If tabu check is O(m × tabu_size × n), then: **O(I × m × n × tabu_size)** worst case
   - Typically: **O(I × m × n)** with efficient tabu checking

#### Space Complexity

- **Current Solution**: O(n) - stores permutation of n tasks
- **Best Solution**: O(n) - stores best permutation found
- **Tabu List**: O(tabu_size × n) - stores tabu_size solutions
- **Neighborhood**: O(m × n) - temporary storage for m neighbors
- **History Arrays**: O(I) - stores iteration-level history
  - `history_best`: O(I)
  - `history_current`: O(I)
  - `accepted_hist`: O(I)
  - `penalty_hist`: O(I)

**Total Space Complexity**: **O(n + tabu_size × n + m × n + I)**
- Simplified: **O((tabu_size + m) × n + I)**
- Dominated by: **Tabu list** (O(tabu_size × n)) or **Neighborhood** (O(m × n))
- Typically: **O(tabu_size × n)** if tabu_size ≥ m

---

## Empirical Scaling

### Methodology

To validate theoretical complexity, we perform empirical scaling experiments:

1. **Test Sizes**: Run algorithms on different problem sizes (n = 100, 200, 400, 800 tasks)

2. **Fixed Parameters**: Keep algorithm parameters constant:
   - SA: max_iter = 500, T0 = 1.0, alpha = 0.995
   - GA: generations = 100, pop_size = 80
   - TS: max_iter = 150, neighborhood_size = 80, tabu_size = 20

3. **Measure Runtime**: Record execution time for each algorithm on each problem size

4. **Plot Scaling**: Create log-scale plot of runtime vs. problem size

### Expected Scaling Behavior

- **SA**: Should show **linear scaling** with n (O(n) per iteration × I iterations)
- **GA**: Should show **linear scaling** with n (O(n) per individual × P individuals × G generations)
- **TS**: Should show **linear scaling** with n (O(n) per neighbor × m neighbors × I iterations)

### Interpretation

- **Slope ≈ 1**: Confirms O(n) scaling (linear)
- **Slope > 1**: Suggests super-linear complexity (e.g., O(n log n) or O(n²))
- **Slope < 1**: Suggests sub-linear complexity (rare, possibly due to caching)

---

## Comparison Summary

| Algorithm | Time Complexity | Space Complexity | Dominant Operation |
|-----------|----------------|-------------------|---------------------|
| **SA** | O(I × n) | O(n + I) | Fitness evaluation per iteration |
| **GA** | O(G × P × n) | O(P × n + G) | Fitness evaluation for entire population |
| **TS** | O(I × m × n) | O((tabu_size + m) × n + I) | Neighborhood exploration and evaluation |

### Relative Performance (Typical Parameters)

Assuming:
- I = 2000 (SA iterations)
- G = 300, P = 80 (GA: generations × population)
- I = 300, m = 80 (TS: iterations × neighborhood)

**Time Complexity Comparison**:
- **SA**: O(2000 × n) = O(2000n)
- **GA**: O(300 × 80 × n) = O(24,000n)
- **TS**: O(300 × 80 × n) = O(24,000n)

**Conclusion**: 
- SA is typically **fastest** (fewer operations per iteration)
- GA and TS are similar (both explore multiple solutions per iteration)
- GA may be slower due to larger population size

### Space Complexity Comparison

- **SA**: Most memory-efficient (O(n) typically)
- **GA**: Most memory-intensive (O(P × n) with P = 80)
- **TS**: Moderate (O(tabu_size × n) with tabu_size = 20)

---

## Notes

1. **Fitness Function**: All algorithms use the same fitness function (`compute_fitness`), which has O(n) complexity:
   - Iterates through all n tasks: O(n)
   - Checks constraints: O(n)
   - Computes penalties: O(n)
   - Total: O(n)

2. **Practical Considerations**:
   - Actual runtime depends on implementation details
   - Python overhead may affect constant factors
   - Caching and optimizations can improve performance
   - Problem-specific characteristics may affect scaling

3. **Parameter Impact**:
   - Increasing iterations (I, G) increases time linearly
   - Increasing population (P) or neighborhood (m) increases time linearly
   - These parameters trade off solution quality vs. runtime

---

## References

- Complexity analysis based on standard algorithm analysis techniques
- Big-O notation: upper bound on growth rate
- Empirical validation through scaling experiments
- Analysis assumes worst-case or average-case scenarios

---

**Last Updated**: December 2024

