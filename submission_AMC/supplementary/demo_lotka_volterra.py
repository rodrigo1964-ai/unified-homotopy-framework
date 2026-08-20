#!/usr/bin/env python3
"""
demo_lotka_volterra.py - Lotka-Volterra predator-prey system for 21Paper

Demonstrates the homotopy framework on a 2D coupled nonlinear system:
  dx/dt = x(a - by) = ax - bxy  (prey)
  dy/dt = y(-c + dx) = -cy + dxy (predator)

Tasks:
  1. Generate training data from true system
  2. Identify f_x(x,y) and f_y(x,y) using 2D RBF
  3. Simulate forward with identified model
  4. Compare with true trajectory

Parameters: a=1.0, b=0.5, c=1.0, d=0.5
Initial conditions: x0=1.5, y0=1.0

Author: Rodolfo H. Rodrigo / INAUT-UNSJ-CONICET
Date: August 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import os

# Ensure figures directory exists
_FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(_FIG_DIR, exist_ok=True)

# Try sklearn for KMeans
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available, using simple initialization")

# ============================================================================
# 1. LOTKA-VOLTERRA SYSTEM
# ============================================================================

# True parameters
A_TRUE = 1.0   # Prey birth rate
B_TRUE = 0.5   # Predation rate
C_TRUE = 1.0   # Predator death rate
D_TRUE = 0.5   # Predator growth rate

X0 = 1.5  # Initial prey population
Y0 = 1.0  # Initial predator population

def f_x_true(x, y):
    """True prey dynamics: f_x = a*x - b*x*y"""
    return A_TRUE * x - B_TRUE * x * y

def f_y_true(x, y):
    """True predator dynamics: f_y = -c*y + d*x*y"""
    return -C_TRUE * y + D_TRUE * x * y

def rhs_true(t, state):
    """True ODE: [dx/dt, dy/dt]"""
    x, y = state
    return np.array([f_x_true(x, y), f_y_true(x, y)])

# ============================================================================
# 2. 2D RBF APPROXIMATION
# ============================================================================

class RBF2D:
    """2D Gaussian RBF approximator: f(x,y) = sum_j w_j * phi_j(x,y)"""

    def __init__(self, M=10):
        self.M = M
        self.c_x = None    # X centers
        self.c_y = None    # Y centers
        self.lam = None    # Widths (shared)
        self.w = None      # Weights

    def phi(self, x, y):
        """Evaluate basis functions: Phi(x,y) = [phi_1(x,y), ..., phi_M(x,y)]"""
        if self.c_x is None or self.c_y is None or self.lam is None:
            raise ValueError("RBF2D not initialized")

        # Convert to column vectors
        x = np.asarray(x).reshape(-1, 1)
        y = np.asarray(y).reshape(-1, 1)

        # Centers as rows
        c_x = self.c_x.reshape(1, -1)
        c_y = self.c_y.reshape(1, -1)
        lam = self.lam.reshape(1, -1)

        # Gaussian RBF: exp(-lambda * ||[x,y] - [c_x, c_y]||^2)
        dist_sq = (x - c_x)**2 + (y - c_y)**2
        return np.exp(-lam * dist_sq)

    def __call__(self, x, y):
        """Evaluate RBF approximation"""
        if self.w is None:
            raise ValueError("RBF2D weights not set")
        return self.phi(x, y) @ self.w

    def initialize_centers(self, x_data, y_data, method='grid'):
        """Initialize centers on a uniform 2D grid (default, matches the
        manuscript configuration) or via K-means clustering on (x,y) pairs."""
        if method == 'kmeans' and HAS_SKLEARN and len(x_data) >= self.M:
            XY = np.column_stack([x_data, y_data])
            kmeans = KMeans(n_clusters=self.M, random_state=42, n_init=10)
            kmeans.fit(XY)
            centers = kmeans.cluster_centers_
            self.c_x = centers[:, 0]
            self.c_y = centers[:, 1]
        else:
            # Uniform grid initialization over the observed (x, y) domain
            x_min, x_max = x_data.min(), x_data.max()
            y_min, y_max = y_data.min(), y_data.max()
            n_side = int(np.sqrt(self.M))
            x_grid = np.linspace(x_min, x_max, n_side)
            y_grid = np.linspace(y_min, y_max, n_side)
            XX, YY = np.meshgrid(x_grid, y_grid)
            self.c_x = XX.flatten()[:self.M]
            self.c_y = YY.flatten()[:self.M]

        # Set widths based on nearest-neighbor distances
        centers = np.column_stack([self.c_x, self.c_y])
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        min_dists = dists.min(axis=1)
        self.lam = 1.0 / (2 * min_dists**2)

    def fit_homotopy(self, x_data, y_data, f_data, verbose=False):
        """Fit RBF using homotopy corrections (linear solve only for 2D)"""
        N = len(x_data)

        # Linear solve for weights (exact)
        Phi = self.phi(x_data, y_data)
        self.w = np.linalg.lstsq(Phi, f_data, rcond=None)[0]

        # Compute residual
        residual = Phi @ self.w - f_data
        residual_norm = np.linalg.norm(residual)

        if verbose:
            print(f"  After linear solve: ||residual|| = {residual_norm:.3e}")

        return residual_norm

# ============================================================================
# 3. SYSTEM IDENTIFICATION
# ============================================================================

def identify_lotka_volterra(x_data, y_data, dxdt_data, dydt_data, M=16,
                            init_method='grid', verbose=True):
    """
    Identify Lotka-Volterra system from data

    Returns:
        rbf_x: RBF approximator for f_x(x,y)
        rbf_y: RBF approximator for f_y(x,y)
    """

    if verbose:
        print("\n[2] Identifying f_x(x,y) and f_y(x,y) with RBF...")
        print("-" * 70)

    # Identify f_x(x,y) = dx/dt
    rbf_x = RBF2D(M=M)
    rbf_x.initialize_centers(x_data, y_data, method=init_method)
    residual_x = rbf_x.fit_homotopy(x_data, y_data, dxdt_data, verbose=verbose)

    if verbose:
        print(f"  f_x identification: ||residual|| = {residual_x:.3e}")

    # Identify f_y(x,y) = dy/dt
    rbf_y = RBF2D(M=M)
    rbf_y.initialize_centers(x_data, y_data, method=init_method)
    residual_y = rbf_y.fit_homotopy(x_data, y_data, dydt_data, verbose=verbose)

    if verbose:
        print(f"  f_y identification: ||residual|| = {residual_y:.3e}")

    return rbf_x, rbf_y

# ============================================================================
# 4. FORWARD SIMULATION
# ============================================================================

def simulate_lotka_volterra(rbf_x, rbf_y, t_span, state0, n_steps=200):
    """Forward simulate Lotka-Volterra with identified RBFs"""

    def rhs_identified(t, state):
        x, y = state
        dx_dt = rbf_x(x, y).item()
        dy_dt = rbf_y(x, y).item()
        return np.array([dx_dt, dy_dt])

    t_eval = np.linspace(t_span[0], t_span[1], n_steps+1)
    sol = solve_ivp(rhs_identified, t_span, state0, t_eval=t_eval,
                   method='RK45', rtol=1e-6)

    return sol.t, sol.y

# ============================================================================
# 5. MAIN EXPERIMENT
# ============================================================================

def run_lotka_volterra_experiment(n_samples=40, M=16, t_final=15,
                                  init_method='grid', verbose=True):
    """
    Complete Lotka-Volterra experiment

    Returns:
        results: dict with trajectories and errors
    """

    print("="*70)
    print("LOTKA-VOLTERRA 2D PREDATOR-PREY SYSTEM")
    print("="*70)

    # [1] Generate training data
    if verbose:
        print("\n[1] Generating training data...")
        print("-" * 70)
        print(f"  System: dx/dt = x(a - by), dy/dt = y(-c + dx)")
        print(f"  Parameters: a={A_TRUE}, b={B_TRUE}, c={C_TRUE}, d={D_TRUE}")
        print(f"  Initial: x0={X0}, y0={Y0}")
        print(f"  Time span: [0, {t_final}]")
        print(f"  Training samples: {n_samples}")

    t_data = np.linspace(0, t_final, n_samples)
    sol_train = solve_ivp(rhs_true, [0, t_final], [X0, Y0],
                         t_eval=t_data, method='RK45', rtol=1e-10)
    x_data = sol_train.y[0]
    y_data = sol_train.y[1]

    # Compute derivatives using finite differences
    dt = t_data[1] - t_data[0]
    dxdt_data = np.gradient(x_data, dt)
    dydt_data = np.gradient(y_data, dt)

    # [2] Identify system
    rbf_x, rbf_y = identify_lotka_volterra(x_data, y_data, dxdt_data, dydt_data,
                                           M=M, init_method=init_method,
                                           verbose=verbose)

    # [3] Generate reference trajectory for error computation
    if verbose:
        print("\n[3] Simulating forward...")
        print("-" * 70)

    t_ref = np.linspace(0, t_final, 401)
    sol_ref = solve_ivp(rhs_true, [0, t_final], [X0, Y0],
                       t_eval=t_ref, method='RK45', rtol=1e-10)
    x_ref = sol_ref.y[0]
    y_ref = sol_ref.y[1]

    # [4] Simulate with identified model
    t_sim, state_sim = simulate_lotka_volterra(rbf_x, rbf_y, [0, t_final],
                                               [X0, Y0], n_steps=400)
    x_sim = state_sim[0]
    y_sim = state_sim[1]

    # [5] Compute errors
    # Interpolate to same time grid
    x_sim_interp = np.interp(t_ref, t_sim, x_sim)
    y_sim_interp = np.interp(t_ref, t_sim, y_sim)

    error_x = np.abs(x_sim_interp - x_ref)
    error_y = np.abs(y_sim_interp - y_ref)
    max_error_x = np.max(error_x)
    max_error_y = np.max(error_y)
    max_error = max(max_error_x, max_error_y)

    if verbose:
        print(f"  Maximum error in x: {max_error_x:.4f}")
        print(f"  Maximum error in y: {max_error_y:.4f}")
        print(f"  Maximum error (overall): {max_error:.4f}")

    return {
        't_data': t_data,
        'x_data': x_data,
        'y_data': y_data,
        't_ref': t_ref,
        'x_ref': x_ref,
        'y_ref': y_ref,
        't_sim': t_sim,
        'x_sim': x_sim,
        'y_sim': y_sim,
        'max_error': max_error,
        'max_error_x': max_error_x,
        'max_error_y': max_error_y,
    }

# ============================================================================
# 6. PLOTTING
# ============================================================================

def plot_lotka_volterra_results(results):
    """Generate figures for manuscript"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Plot 1: Phase portrait
    ax = axes[0]
    ax.plot(results['x_ref'], results['y_ref'], 'k-', linewidth=2,
            label='True trajectory', alpha=0.7)
    ax.plot(results['x_sim'], results['y_sim'], 'r--', linewidth=2,
            label='Identified model', alpha=0.8)
    ax.plot(results['x_data'], results['y_data'], 'bo', markersize=6,
            label=f'Training data ({len(results["x_data"])} points)', alpha=0.6)
    ax.set_xlabel('Prey population (x)', fontsize=12)
    ax.set_ylabel('Predator population (y)', fontsize=12)
    ax.set_title('(a) Phase Portrait', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 2: Time series - Prey
    ax = axes[1]
    ax.plot(results['t_ref'], results['x_ref'], 'k-', linewidth=2,
            label='True x(t)', alpha=0.7)
    ax.plot(results['t_sim'], results['x_sim'], 'r--', linewidth=2,
            label='Identified x(t)', alpha=0.8)
    ax.plot(results['t_data'], results['x_data'], 'bo', markersize=5,
            alpha=0.6)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Prey population (x)', fontsize=12)
    ax.set_title('(b) Prey Time Series', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Time series - Predator
    ax = axes[2]
    ax.plot(results['t_ref'], results['y_ref'], 'k-', linewidth=2,
            label='True y(t)', alpha=0.7)
    ax.plot(results['t_sim'], results['y_sim'], 'r--', linewidth=2,
            label='Identified y(t)', alpha=0.8)
    ax.plot(results['t_data'], results['y_data'], 'go', markersize=5,
            alpha=0.6)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Predator population (y)', fontsize=12)
    ax.set_title('(c) Predator Time Series', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    figpath = os.path.join(_FIG_DIR, 'fig_lotka_volterra.png')
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    print(f"\n[4] Figure saved: {figpath}")

    # plt.show()

# ============================================================================
# 7. MAIN
# ============================================================================

if __name__ == '__main__':
    # Run experiment (configuration reported in the manuscript:
    # 40 samples over [0, 15], M=16 centres, uniform-grid initialisation)
    results = run_lotka_volterra_experiment(
        n_samples=40,
        M=16,
        t_final=15,
        init_method='grid',
        verbose=True
    )

    # Generate figure
    plot_lotka_volterra_results(results)

    print("\n" + "="*70)
    print(f"LOTKA-VOLTERRA EXPERIMENT COMPLETED")
    print(f"Maximum simulation error: {results['max_error']:.4f}")
    print("="*70)
