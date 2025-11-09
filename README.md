# Parametric Curve Fitting

## Problem

Find optimal parameters θ, M, X for:
- x = t·cos(θ) - e^(M|t|)·sin(0.3t)·sin(θ) + X
- y = 42 + t·sin(θ) + e^(M|t|)·sin(0.3t)·cos(θ)

Constraints:
- t ∈ (6, 60)
- θ ∈ (0°, 50°)
- M ∈ (-0.05, 0.05)
- X ∈ (0, 100)

Minimize L1 distance against 1500 data points.

## Solution

**Optimal Parameters:**
- θ = 0.484965 rad (27.786456°)
- M = 0.019486
- X = 54.427319

**L1 Distance:** 37,637.57 (improved from 37,865.09)

## Quick Run

```bash
python optimized_solution.py
```

Output: Optimal parameters, Desmos string, and error metrics

## Desmos Verification

Paste into https://www.desmos.com/calculator/rfj91yrxob:

```latex
\left(t*\cos(0.484965)-e^{0.019486\left|t\right|}\cdot\sin(0.3t)\sin(0.484965)\ +54.427319,42+\ t*\sin(0.484965)+e^{0.019486\left|t\right|}\cdot\sin(0.3t)\cos(0.484965)\right)
```

## Performance

| Metric | Value |
|--------|-------|
| L1 Distance | 37,637.57 |
| Mean Error | 18.80 |
| Median Error | 16.35 |
| Max Error | 52.11 |
| Std Dev | 12.57 |

## Algorithm

Advanced 4-phase optimization:

1. **Global Search** - Differential Evolution (2000 iterations)
2. **Iterative Refinement** - 12 iterations with adaptive t-estimation
3. **Fine Grid Search** - 132,651 parameter combinations
4. **Final Polish** - Multi-pass BFGS (10,000 iterations)

## Files

- ANSWER.md - Final parameters and metrics
- METHODOLOGY.md - Algorithm explanation
- README.md - This file
- optimized_solution.py - Main solver
- xy_data.csv - Input data
- SUBMISSION_SUMMARY.md - Complete documentation

## Requirements

```
numpy
pandas
scipy
```

Install: `pip install numpy pandas scipy`
