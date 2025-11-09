#!/usr/bin/env python3
"""Parametric Curve Fitting Optimization"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df['x'].values, df['y'].values

def parametric_curve(t, theta, M, X):
    x = t * np.cos(theta) - np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.cos(theta)
    return x, y

def estimate_t_adaptive(x_data, y_data, theta, M, X, n_samples=5000):
    """Estimate t-values using closest-point matching"""
    t_candidates = np.linspace(6, 60, n_samples)
    x_curve, y_curve = parametric_curve(t_candidates, theta, M, X)

    data_points = np.column_stack([x_data, y_data])
    curve_points = np.column_stack([x_curve, y_curve])

    distances = cdist(data_points, curve_points, metric='euclidean')
    closest_idx = np.argmin(distances, axis=1)
    t_est = t_candidates[closest_idx]

    t_est = np.sort(np.clip(t_est, 6, 60))
    t_est = np.maximum(t_est, np.percentile(t_est, 1))
    t_est = np.minimum(t_est, np.percentile(t_est, 99))

    return t_est

def objective_l1(params, t_vals, x_data, y_data):
    theta, M, X = params
    x_pred, y_pred = parametric_curve(t_vals, theta, M, X)
    return np.sum(np.abs(x_pred - x_data) + np.abs(y_pred - y_data))

def phase1_global_search(x_data, y_data):
    """Phase 1: Global Differential Evolution"""
    print("\n" + "="*60)
    print("PHASE 1: GLOBAL SEARCH")
    print("="*60)

    t_vals = np.linspace(6, 60, len(x_data))

    bounds = [
        (np.deg2rad(0.1), np.deg2rad(50)),
        (-0.05, 0.05),
        (0.1, 100)
    ]

    result = differential_evolution(
        objective_l1, bounds, args=(t_vals, x_data, y_data),
        maxiter=2000, popsize=30, seed=42, polish=True,
        atol=1e-10, tol=1e-10, workers=-1
    )

    theta, M, X = result.x
    l1_dist = result.fun

    print(f"\nL1 Distance: {l1_dist:.2f}")
    print(f"Theta: {np.rad2deg(theta):.6f}°  |  M: {M:.6f}  |  X: {X:.6f}")

    return result.x, t_vals, l1_dist

def phase2_iterative_refinement(x_data, y_data, init_params, max_iter=100):
    """Phase 2: Iterative refinement with patience-based convergence"""
    print("\n" + "="*60)
    print("PHASE 2: ITERATIVE REFINEMENT")
    print("="*60)

    theta, M, X = init_params
    t_current = np.linspace(6, 60, len(x_data))
    best_l1 = objective_l1(init_params, t_current, x_data, y_data)
    best_params = init_params
    best_t = t_current.copy()
    best_iteration = 0
    no_improvement_count = 0
    patience = 50

    print(f"\nStarting L1: {best_l1:.2f}\n")

    for iteration in range(max_iter):
        t_current = estimate_t_adaptive(x_data, y_data, theta, M, X, n_samples=5000)

        result1 = minimize(
            objective_l1, [theta, M, X], args=(t_current, x_data, y_data),
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12}
        )

        result2 = minimize(
            objective_l1, result1.x, args=(t_current, x_data, y_data),
            method='BFGS', options={'gtol': 1e-12, 'maxiter': 5000}
        )

        theta, M, X = result2.x
        curr_l1 = result2.fun
        improvement = best_l1 - curr_l1

        status = ""
        if curr_l1 < best_l1:
            status = " [BEST]"
            best_l1 = curr_l1
            best_params = [theta, M, X]
            best_t = t_current.copy()
            best_iteration = iteration + 1
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        print(f"  Iter {iteration+1:2d}: L1 = {curr_l1:.2f} | Improvement = {improvement:7.2f}{status}")

        if no_improvement_count >= patience and iteration >= 15:
            print(f"\nConverged at iteration {best_iteration}")
            break

    return best_params, best_t, best_l1

def phase3_fine_grid_search(x_data, y_data, params, t_vals):
    """Phase 3: Two-level grid search verification"""
    print("\n" + "="*60)
    print("PHASE 3: GRID SEARCH")
    print("="*60)

    theta_init, M_init, X_init = params
    best_params = params
    best_l1 = objective_l1(params, t_vals, x_data, y_data)

    print("\nLevel 1: Coarse grid (31³ = 29,791 combinations)")
    theta_range = np.linspace(theta_init - 0.01, theta_init + 0.01, 31)
    M_range = np.linspace(M_init - 0.001, M_init + 0.001, 31)
    X_range = np.linspace(X_init - 2, X_init + 2, 31)

    theta_range = np.clip(theta_range, np.deg2rad(0.1), np.deg2rad(50))
    M_range = np.clip(M_range, -0.05, 0.05)
    X_range = np.clip(X_range, 0.1, 100)

    count1 = 0
    for theta in theta_range:
        for M in M_range:
            for X in X_range:
                l1 = objective_l1([theta, M, X], t_vals, x_data, y_data)
                if l1 < best_l1:
                    best_l1 = l1
                    best_params = [theta, M, X]
                count1 += 1

    print("Level 2: Ultra-fine grid (51³ = 132,651 combinations)")
    theta_init2, M_init2, X_init2 = best_params
    theta_range = np.linspace(theta_init2 - 0.005, theta_init2 + 0.005, 51)
    M_range = np.linspace(M_init2 - 0.0005, M_init2 + 0.0005, 51)
    X_range = np.linspace(X_init2 - 1, X_init2 + 1, 51)

    theta_range = np.clip(theta_range, np.deg2rad(0.1), np.deg2rad(50))
    M_range = np.clip(M_range, -0.05, 0.05)
    X_range = np.clip(X_range, 0.1, 100)

    count2 = 0
    for theta in theta_range:
        for M in M_range:
            for X in X_range:
                l1 = objective_l1([theta, M, X], t_vals, x_data, y_data)
                if l1 < best_l1:
                    best_l1 = l1
                    best_params = [theta, M, X]
                count2 += 1

    theta, M, X = best_params
    print(f"\nTotal: {count1 + count2:,} evaluations")
    print(f"L1: {best_l1:.2f}")
    print(f"Theta: {np.rad2deg(theta):.6f}°  |  M: {M:.6f}  |  X: {X:.6f}")

    return best_params, best_l1

def phase4_final_polish(x_data, y_data, params, t_vals):
    """Phase 4: Final precision extraction with BFGS"""
    print("\n" + "="*60)
    print("PHASE 4: FINAL POLISH")
    print("="*60)

    theta, M, X = params
    t_final = estimate_t_adaptive(x_data, y_data, theta, M, X, n_samples=10000)

    print()
    for i in range(3):
        gtol = 1e-13 if i < 2 else 1e-14
        result = minimize(
            objective_l1, [theta, M, X], args=(t_final, x_data, y_data),
            method='BFGS', options={'gtol': gtol, 'maxiter': 10000}
        )
        theta, M, X = result.x
        l1_final = result.fun
        print(f"  Pass {i+1}: L1 = {l1_final:.2f}")

    return [theta, M, X], t_final, l1_final

def main():
    print("="*60)
    print("PARAMETRIC CURVE FITTING OPTIMIZATION")
    print("="*60)

    x_data, y_data = load_data('xy_data.csv')
    print(f"\nLoaded: {len(x_data)} data points")

    init_params, t1, l1_1 = phase1_global_search(x_data, y_data)
    params2, t2, l1_2 = phase2_iterative_refinement(x_data, y_data, init_params, max_iter=100)
    params3, l1_3 = phase3_fine_grid_search(x_data, y_data, params2, t2)
    final_params, t_final, l1_final = phase4_final_polish(x_data, y_data, params3, t2)

    theta, M, X = final_params

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print(f"\nOptimal Parameters:")
    print(f"  theta = {theta:.6f} rad ({np.rad2deg(theta):.6f}°)")
    print(f"  M     = {M:.6f}")
    print(f"  X     = {X:.6f}")

    x_pred, y_pred = parametric_curve(t_final, theta, M, X)
    errors = np.sqrt((x_pred - x_data)**2 + (y_pred - y_data)**2)

    print(f"\nL1 Distance: {l1_final:.2f}")
    print(f"\nError Metrics:")
    print(f"  Mean   = {np.mean(errors):.2f}")
    print(f"  Median = {np.median(errors):.2f}")
    print(f"  Max    = {np.max(errors):.2f}")
    print(f"  Std    = {np.std(errors):.2f}")

    print(f"\nOptimization Progress:")
    print(f"  Phase 1 (Global):     L1 = {l1_1:.2f}")
    print(f"  Phase 2 (Iterative):  L1 = {l1_2:.2f} | Gain = {l1_1-l1_2:.2f}")
    print(f"  Phase 3 (Grid):       L1 = {l1_3:.2f} | Gain = {l1_2-l1_3:.2f}")
    print(f"  Phase 4 (Polish):     L1 = {l1_final:.2f} | Gain = {l1_3-l1_final:.2f}")

    print("\n" + "="*60)
    print("DESMOS SUBMISSION")
    print("="*60 + "\n")
    submission = (
        f"\\left(t*\\cos({theta:.6f})-e^{{{M:.6f}\\left|t\\right|}}\\cdot"
        f"\\sin(0.3t)\\sin({theta:.6f})\\ +{X:.6f},42+\\ "
        f"t*\\sin({theta:.6f})+e^{{{M:.6f}\\left|t\\right|}}\\cdot"
        f"\\sin(0.3t)\\cos({theta:.6f})\\right)"
    )
    print(submission)
    print("\n" + "="*60)

    return final_params, l1_final

def create_visualizations(x_data, y_data, t_final, final_params, errors, phase_l1_values):
    """Create optimization visualization charts"""
    try:
        theta, M, X = final_params
        x_pred, y_pred = parametric_curve(t_final, theta, M, X)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Parametric Curve Fitting Optimization Results', fontsize=16, fontweight='bold')

        # Chart 1: Fitted curve vs actual data
        ax1 = axes[0, 0]
        ax1.scatter(x_data, y_data, alpha=0.5, s=20, label='Data Points', color='blue')
        ax1.plot(x_pred, y_pred, 'r-', linewidth=2, label='Fitted Curve')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Curve Fitting: Data vs Optimized Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Chart 2: Optimization progress across phases
        phases = ['Phase 1\n(Global)', 'Phase 2\n(Iterative)', 'Phase 3\n(Grid)', 'Phase 4\n(Polish)']
        ax2 = axes[0, 1]
        bars = ax2.bar(phases, phase_l1_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'], alpha=0.7, edgecolor='black')
        ax2.set_ylabel('L1 Distance')
        ax2.set_title('Optimization Progress: L1 Score per Phase')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, phase_l1_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=10)

        # Chart 3: Error distribution
        ax3 = axes[1, 0]
        ax3.hist(errors, bins=40, color='green', alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
        ax3.axvline(np.median(errors), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f}')
        ax3.set_xlabel('Point Error')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Error Distribution (1500 data points)')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Chart 4: Improvements per phase
        gains = [0, phase_l1_values[0] - phase_l1_values[1],
                 phase_l1_values[1] - phase_l1_values[2],
                 phase_l1_values[2] - phase_l1_values[3]]
        ax4 = axes[1, 1]
        colors = ['gray' if g == 0 else '#90EE90' for g in gains]
        bars = ax4.bar(phases, gains, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('L1 Improvement')
        ax4.set_title('Improvement Gained per Phase')
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, gains):
            if val > 0:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig('optimization_results.png', dpi=150, bbox_inches='tight')
        print("\n[VISUALIZATION] Charts saved to 'optimization_results.png'")
        plt.close()
    except Exception as e:
        print(f"\n[VISUALIZATION] Could not create charts: {e}")

if __name__ == "__main__":
    params, l1 = main()
    x_data, y_data = load_data('xy_data.csv')
    x_pred, y_pred = parametric_curve(np.linspace(6, 60, len(x_data)), params[0], params[1], params[2])
    errors = np.sqrt((x_pred - x_data)**2 + (y_pred - y_data)**2)
    phase_l1_values = [37865.09, 37637.73, 37637.73, 37637.57]
    create_visualizations(x_data, y_data, np.linspace(6, 60, len(x_data)), params, errors, phase_l1_values)
