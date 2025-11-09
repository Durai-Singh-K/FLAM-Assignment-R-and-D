# Methodology: Parametric Curve Fitting

## Problem

Given parametric equations:
- x(t) = t·cos(θ) - e^(M|t|)·sin(0.3t)·sin(θ) + X
- y(t) = 42 + t·sin(θ) + e^(M|t|)·sin(0.3t)·cos(θ)

Find: θ ∈ (0°, 50°), M ∈ (-0.05, 0.05), X ∈ (0, 100)

Minimize: L1 distance = Σ |x_pred - x_data| + |y_pred - y_data|

## Approach

The challenge is that both parameters (θ, M, X) and t-values are unknown. Direct optimization over all 1503 variables (3 parameters + 1500 t-values) is infeasible.

Solution: Iterative alternating optimization (similar to EM algorithm)

### Phase 1: Initialization
- Assume uniform t spacing: t_i = 6 + (i-1)·54/1499
- Global optimization using Differential Evolution
- Result: Initial parameter estimates

### Phase 2: Iterative Refinement
Repeat until convergence:
1. Fix parameters, estimate t-values via closest-point matching on dense curve sampling
2. Ensure t-values are monotonic and within bounds [6, 60]
3. Fix t-values, optimize parameters using Nelder-Mead
4. Check convergence: stop if improvement < 1.0

### Phase 3: Fine-tuning
- Grid search in local region around best solution
- Test 9,261 parameter combinations
- Select best result

### Phase 4: Final Polishing
- High-resolution t-value estimation (5000 samples)
- BFGS optimization for final precision

## Convergence Analysis

Each iteration produces monotonically decreasing L1 distance:
- E-step (t estimation) maintains or decreases L1
- M-step (parameter optimization) maintains or decreases L1
- L1 is bounded below by 0
- Therefore L1 converges by monotone convergence theorem

May converge to local optimum; global initialization (DE) reduces this risk.

## Error Sources

1. Discretization error: O(0.01) from 5000-sample curve
2. Numerical error: O(1e-8) from optimizer tolerance
3. Model error: Data noise and measurement errors

## Complexity

- Differential Evolution: ~22M operations
- Iterative refinement (K=8): ~72M operations
- Grid search: ~14M operations
- Total: ~126M operations ≈ 2-3 minutes

## Validation

Tested multiple methods:
- Differential Evolution
- Dual Annealing
- Nelder-Mead
- Joint Optimization
- Grid Search

All converged to similar solutions, confirming robustness.
