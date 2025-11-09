# Parametric Curve Fitting - Final Answer

## Parameters (Optimized)

```
theta = 0.484965 rad (27.786456 degrees)
M = 0.019486
X = 54.427319
```

## Desmos Format

```latex
\left(t*\cos(0.484965)-e^{0.019486\left|t\right|}\cdot\sin(0.3t)\sin(0.484965)\ +54.427319,42+\ t*\sin(0.484965)+e^{0.019486\left|t\right|}\cdot\sin(0.3t)\cos(0.484965)\right)
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| L1 Distance | 37637.57 |
| Mean Error | 18.80 |
| Median Error | 16.35 |
| Max Error | 52.11 |
| Std Dev | 12.57 |

## Algorithm

Four-phase advanced optimization:

### Phase 1: Global Search

- Differential Evolution with 2000 iterations, popsize 30
- Uniform t-spacing initialization
- L1 Distance: 37,865.09

### Phase 2: Iterative Refinement

- Alternating optimization (12 iterations)
- High-resolution t-value estimation (5000 samples)
- Nelder-Mead + BFGS hybrid optimization
- L1 Distance: 37,638.59 (improvement: 226.50)

### Phase 3: Fine Grid Search

- Ultra-fine local search (132,651 combinations)
- Grid resolution: 51 x 51 x 51 around best point
- L1 Distance: 37,638.59

### Phase 4: Final Polish

- Multi-pass BFGS with 10,000 max iterations
- Ultra-high resolution t estimation (10,000 samples)
- Final L1 Distance: 37,638.55

## Convergence Summary

Starting L1: 37,865.09
Final L1: 37,637.57
Total Improvement: 227.52 units (0.60%)

Key insight: Allowing iterations to continue despite temporary negative improvements revealed better optima.
Iteration 13 found L1=37,637.75 (after temporary setback at Iter 12)
Iteration 14 found L1=37,637.73 (further improvement)

Total iterations: 29 refinement cycles (patience-based stopping) + 3 polish passes

## Experiments Conducted

See EXPERIMENTS_AND_RESULTS.md for detailed documentation of:

1. **Experiment 1:** Baseline Differential Evolution optimization
2. **Experiment 2:** Iterative refinement with early stopping (discovered limitation)
3. **Experiment 3:** Patience-based convergence mechanism (0.85 unit improvement)
4. **Experiment 4:** Two-level grid search refinement (verification)
5. **Experiment 5:** Multi-pass BFGS final polish (0.16 unit improvement)
6. **Experiment 6:** Parametric curve validation (error metrics)
7. **Experiment 7:** Reproducibility and sensitivity analysis (100% reproducible)

## Key Learning

The patience mechanism (Experiment 3) discovered that temporary negative improvements can precede better solutions. This led to a 0.85 unit improvement over early stopping strategy.

## References

All experiments include proper academic references:

- Storn & Price (1997): Differential Evolution
- Nocedal & Wright (2006): Numerical Optimization / BFGS
- Boyd & Vandenberghe (2004): Convex Optimization
- Saltelli et al. (2008): Sensitivity Analysis
- Press et al. (2007): Numerical Recipes
