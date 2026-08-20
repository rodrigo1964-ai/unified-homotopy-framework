#!/usr/bin/env python3
"""
demo_noise_robustness.py - Noise robustness experiments for 21Paper

Evaluates the homotopy framework's robustness to measurement noise:
  - Adds Gaussian noise to training data at levels: 1%, 5%, 10%
  - Compares homotopy vs L-BFGS-B (SciPy) under noise
  - Generates figure showing degradation of performance vs SNR

System: Tank with orifice (same as main demo)
Architecture: 5-center RBF (same as main demo)

Author: Rodolfo H. Rodrigo / INAUT-UNSJ-CONICET
Date: August 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
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
# 1. PHYSICAL SYSTEM
# ============================================================================

ALPHA_TRUE = 0.5
H0 = 1.0

def u(t):
    """Input flow rate (step-ramp excitation)"""
    if t < 2.0:
        return 0.0
    elif t < 6.0:
        return 0.3 * (t - 2.0) / 4.0
    else:
        return 0.3

def f_true(h):
    """True nonlinearity: f(h) = alpha * sqrt(h)"""
    return ALPHA_TRUE * np.sqrt(np.maximum(h, 0))

def rhs_true(t, y):
    """True ODE: dh/dt = u(t) - f(h)"""
    return u(t) - f_true(y[0])

# ============================================================================
# 2. RBF APPROXIMATION
# ============================================================================

class RBF:
    """Gaussian RBF approximator: f(h) = sum_j w_j * phi_j(h)"""

    def __init__(self, M=5):
        self.M = M
        self.c = None      # Centers
        self.lam = None    # Widths
        self.w = None      # Weights

    def phi(self, h):
        """Evaluate basis functions: Phi(h) = [phi_1(h), ..., phi_M(h)]"""
        if self.c is None or self.lam is None:
            raise ValueError("RBF not initialized")
        h = np.asarray(h).reshape(-1, 1)
        c = self.c.reshape(1, -1)
        lam = self.lam.reshape(1, -1)
        return np.exp(-lam * (h - c)**2)

    def phi_derivative(self, h, order=1):
        """Evaluate derivatives of basis functions"""
        Phi = self.phi(h)
        h = np.asarray(h).reshape(-1, 1)
        c = self.c.reshape(1, -1)
        lam = self.lam.reshape(1, -1)

        if order == 1:
            return -2 * lam * (h - c) * Phi
        elif order == 2:
            return (4 * lam**2 * (h - c)**2 - 2 * lam) * Phi
        elif order == 3:
            return (-8 * lam**3 * (h - c)**3 + 12 * lam**2 * (h - c)) * Phi
        else:
            raise NotImplementedError(f"Derivative order {order} not implemented")

    def __call__(self, h):
        """Evaluate RBF approximation"""
        if self.w is None:
            raise ValueError("RBF weights not set")
        return self.phi(h) @ self.w

    def initialize_kmeans(self, h_data):
        """Initialize centers and widths using K-means clustering"""
        if HAS_SKLEARN:
            kmeans = KMeans(n_clusters=self.M, random_state=42, n_init=10)
            kmeans.fit(h_data.reshape(-1, 1))
            self.c = kmeans.cluster_centers_.flatten()
        else:
            # Simple uniform initialization
            self.c = np.linspace(h_data.min(), h_data.max(), self.M)

        # Set widths based on nearest-neighbor distances
        self.lam = np.ones(self.M)
        for j in range(self.M):
            dists = np.abs(self.c - self.c[j])
            dists = dists[dists > 0]
            if len(dists) > 0:
                self.lam[j] = 1.0 / (2 * np.min(dists)**2)

    def _solve_linear(self, h_data, f_data):
        """Exact linear-stage solve for the weights."""
        Phi = self.phi(h_data)
        self.w = np.linalg.lstsq(Phi, f_data, rcond=None)[0]
        return Phi @ self.w - f_data

    def _jacobian_nl(self, h_data):
        """Jacobian of the residual w.r.t. the nonlinear parameters (c, lam).

        With phi_j = exp(-lam_j*(h - c_j)^2):
          dN_i/dc_j   =  w_j * 2*lam_j*(h_i - c_j) * phi_ij   (= -w_j * dphi/dh)
          dN_i/dlam_j = -w_j * (h_i - c_j)^2 * phi_ij
        """
        Phi = self.phi(h_data)
        h_col = np.asarray(h_data).reshape(-1, 1)
        c_row = self.c.reshape(1, -1)
        lam_row = self.lam.reshape(1, -1)
        J_c = 2 * lam_row * (h_col - c_row) * Phi * self.w.reshape(1, -1)
        J_lam = -(h_col - c_row)**2 * Phi * self.w.reshape(1, -1)
        return np.hstack([J_c, J_lam])

    def _hessian_diag_nl(self, h_data):
        """Column sums of the diagonal parameter Hessian of the residual.

        d2N_i/dc_j2   = w_j * (4*lam_j^2*(h_i - c_j)^2 - 2*lam_j) * phi_ij
        d2N_i/dlam_j2 = w_j * (h_i - c_j)^4 * phi_ij
        """
        Phi = self.phi(h_data)
        h_col = np.asarray(h_data).reshape(-1, 1)
        c_row = self.c.reshape(1, -1)
        lam_row = self.lam.reshape(1, -1)
        H_c = ((4 * lam_row**2 * (h_col - c_row)**2 - 2 * lam_row) * Phi
               * self.w.reshape(1, -1)).sum(axis=0)
        H_lam = ((h_col - c_row)**4 * Phi * self.w.reshape(1, -1)).sum(axis=0)
        return np.concatenate([H_c, H_lam])

    def fit_homotopy(self, h_data, f_data, max_outer=3, tol=1e-9, verbose=False):
        """Fit RBF with the bilinear homotopy corrections.

        Per outer iteration (Algorithm 1 of the manuscript):
          1. z1 exact linear solve for the weights.
          2. z1 Newton step on the nonlinear parameters (centres, widths).
          3. z2 diagonal-Hessian Halley step on the same parameters.
        A step that increases the residual is reverted, so the nominal
        path has no tuning parameters (no learning rate).
        """
        M = self.M
        residual = self._solve_linear(h_data, f_data)
        residual_norm = np.linalg.norm(residual)

        if verbose:
            print(f"  After linear solve: ||residual|| = {residual_norm:.3e}")

        for _ in range(max_outer):
            if residual_norm < tol:
                break

            # z1 Newton on (c, lam)
            J = self._jacobian_nl(h_data)
            delta1 = -np.linalg.lstsq(J, residual, rcond=None)[0]
            c_prev, lam_prev, w_prev = self.c.copy(), self.lam.copy(), self.w.copy()
            self.c = self.c + delta1[:M]
            self.lam = np.maximum(self.lam + delta1[M:], 1e-6)
            residual_new = self._solve_linear(h_data, f_data)
            norm_new = np.linalg.norm(residual_new)
            if norm_new > residual_norm:
                self.c, self.lam, self.w = c_prev, lam_prev, w_prev
                break
            residual, residual_norm = residual_new, norm_new
            if verbose:
                print(f"  After z1 Newton:  ||residual|| = {residual_norm:.3e}")

            # z2 Halley (diagonal Hessian) on (c, lam)
            J = self._jacobian_nl(h_data)
            H_diag = self._hessian_diag_nl(h_data)
            delta2 = np.zeros(2 * M)
            for j in range(2 * M):
                Jcol = J[:, j]
                gp = np.dot(Jcol, Jcol)
                if gp > 1e-12 and abs(H_diag[j]) > 1e-15:
                    delta2[j] = -0.5 * np.dot(residual, Jcol)**2 \
                                * H_diag[j] / gp**1.5
            c_prev, lam_prev, w_prev = self.c.copy(), self.lam.copy(), self.w.copy()
            self.c = self.c + delta2[:M]
            self.lam = np.maximum(self.lam + delta2[M:], 1e-6)
            residual_new = self._solve_linear(h_data, f_data)
            norm_new = np.linalg.norm(residual_new)
            if norm_new > residual_norm:
                self.c, self.lam, self.w = c_prev, lam_prev, w_prev
            else:
                residual, residual_norm = residual_new, norm_new
                if verbose:
                    print(f"  After z2 Halley:  ||residual|| = {residual_norm:.3e}")

        return residual_norm

    def fit_lbfgsb(self, h_data, f_data, max_iter=5000, verbose=False):
        """Fit RBF using L-BFGS-B (SciPy) on all parameters"""
        N = len(h_data)

        # Initialize weights to zero
        self.w = np.zeros(self.M)

        # Pack parameters: [c1, ..., cM, lam1, ..., lamM, w1, ..., wM]
        theta0 = np.concatenate([self.c, self.lam, self.w])

        def loss(theta):
            c = theta[:self.M]
            lam = theta[self.M:2*self.M]
            w = theta[2*self.M:]

            # Compute RBF output
            h_col = h_data.reshape(-1, 1)
            c_row = c.reshape(1, -1)
            lam_row = lam.reshape(1, -1)
            Phi = np.exp(-lam_row * (h_col - c_row)**2)
            f_pred = Phi @ w

            # MSE loss
            return 0.5 * np.mean((f_pred - f_data)**2)

        result = minimize(loss, theta0, method='L-BFGS-B',
                         options={'maxiter': max_iter, 'disp': False})

        # Unpack result
        self.c = result.x[:self.M]
        self.lam = result.x[self.M:2*self.M]
        self.w = result.x[2*self.M:]

        # Compute final residual
        Phi = self.phi(h_data)
        residual = Phi @ self.w - f_data
        residual_norm = np.linalg.norm(residual)

        if verbose:
            print(f"  L-BFGS-B: ||residual|| = {residual_norm:.3e} ({result.nit} iter)")

        return residual_norm

# ============================================================================
# 3. FORWARD SIMULATION
# ============================================================================

def simulate_forward(rbf, h0, t_span, n_steps=200):
    """Forward simulate ODE using identified RBF"""
    t_sim = np.linspace(t_span[0], t_span[1], n_steps+1)
    h_sim = np.zeros(n_steps+1)
    h_sim[0] = h0

    dt = (t_span[1] - t_span[0]) / n_steps

    for i in range(n_steps):
        t_i = t_sim[i]
        h_i = h_sim[i]

        # Simple forward Euler (could use integral homotopy here)
        f_i = rbf(h_i).item()
        dhdt = u(t_i) - f_i
        h_sim[i+1] = h_i + dt * dhdt

    return t_sim, h_sim

# ============================================================================
# 4. NOISE EXPERIMENTS
# ============================================================================

def add_gaussian_noise(data, noise_level):
    """Add Gaussian noise: data_noisy = data + noise_level * std(data) * N(0,1)"""
    std = np.std(data)
    noise = np.random.randn(len(data)) * noise_level * std
    return data + noise

def run_noise_experiment(noise_levels=[0.0, 0.01, 0.05, 0.10], n_trials=5, verbose=True):
    """
    Run noise robustness experiments

    Returns:
        results: dict with keys 'noise_levels', 'homotopy_*_errors', 'lbfgsb_*_errors'
    """

    print("="*70)
    print("NOISE ROBUSTNESS EXPERIMENTS")
    print("="*70)

    # Generate clean data
    if verbose:
        print("\n[1] Generating clean data (20 points)...")

    t_data = np.linspace(0, 10, 20)
    sol = solve_ivp(rhs_true, [0, 10], [H0], t_eval=t_data, method='RK45', rtol=1e-10)
    h_data_clean = sol.y[0]
    u_data = np.array([u(ti) for ti in t_data])

    # Compute f from clean data
    T = t_data[1] - t_data[0]
    dhdt = np.gradient(h_data_clean, T)
    f_data_clean = u_data - dhdt

    # Generate reference trajectory for error computation
    t_ref = np.linspace(0, 10, 201)
    sol_ref = solve_ivp(rhs_true, [0, 10], [H0], t_eval=t_ref, method='RK45', rtol=1e-10)
    h_ref = sol_ref.y[0]

    # Storage for results
    results = {
        'noise_levels': noise_levels,
        'homotopy_id_errors': [],
        'homotopy_sim_errors': [],
        'lbfgsb_id_errors': [],
        'lbfgsb_sim_errors': []
    }

    for noise_level in noise_levels:
        if verbose:
            print(f"\n[2] Noise level: {noise_level*100:.1f}%")
            print("-" * 70)

        homotopy_id_errors_trials = []
        homotopy_sim_errors_trials = []
        lbfgsb_id_errors_trials = []
        lbfgsb_sim_errors_trials = []

        for trial in range(n_trials):
            np.random.seed(42 + trial)  # Reproducible

            # Add noise to h_data
            h_data_noisy = add_gaussian_noise(h_data_clean, noise_level)

            # Recompute f_data from noisy h
            dhdt_noisy = np.gradient(h_data_noisy, T)
            f_data_noisy = u_data - dhdt_noisy

            # === HOMOTOPY METHOD ===
            rbf_hom = RBF(M=5)
            rbf_hom.initialize_kmeans(h_data_noisy)
            id_error_hom = rbf_hom.fit_homotopy(h_data_noisy, f_data_noisy, verbose=False)

            # Simulate forward
            _, h_sim_hom = simulate_forward(rbf_hom, H0, [0, 10], n_steps=200)
            sim_error_hom = np.max(np.abs(h_sim_hom - h_ref))

            homotopy_id_errors_trials.append(id_error_hom)
            homotopy_sim_errors_trials.append(sim_error_hom)

            # === L-BFGS-B METHOD ===
            rbf_lb = RBF(M=5)
            rbf_lb.initialize_kmeans(h_data_noisy)
            id_error_lb = rbf_lb.fit_lbfgsb(h_data_noisy, f_data_noisy,
                                            max_iter=5000, verbose=False)

            # Simulate forward
            _, h_sim_lb = simulate_forward(rbf_lb, H0, [0, 10], n_steps=200)
            sim_error_lb = np.max(np.abs(h_sim_lb - h_ref))

            lbfgsb_id_errors_trials.append(id_error_lb)
            lbfgsb_sim_errors_trials.append(sim_error_lb)

        # Average over trials
        results['homotopy_id_errors'].append(np.mean(homotopy_id_errors_trials))
        results['homotopy_sim_errors'].append(np.mean(homotopy_sim_errors_trials))
        results['lbfgsb_id_errors'].append(np.mean(lbfgsb_id_errors_trials))
        results['lbfgsb_sim_errors'].append(np.mean(lbfgsb_sim_errors_trials))

        if verbose:
            print(f"  Homotopy:  ID error = {results['homotopy_id_errors'][-1]:.3e}, "
                  f"Sim error = {results['homotopy_sim_errors'][-1]:.4f}")
            print(f"  L-BFGS-B:  ID error = {results['lbfgsb_id_errors'][-1]:.3e}, "
                  f"Sim error = {results['lbfgsb_sim_errors'][-1]:.4f}")

    return results

# ============================================================================
# 5. PLOTTING
# ============================================================================

def plot_noise_robustness(results):
    """Generate figure showing noise robustness"""

    noise_levels = np.array(results['noise_levels']) * 100  # Convert to percentage

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Plot 1: Identification Error vs Noise
    ax1.semilogy(noise_levels, results['homotopy_id_errors'],
                 'o-', linewidth=2, markersize=8, label='Homotopy', color='C0')
    ax1.semilogy(noise_levels, results['lbfgsb_id_errors'],
                 's--', linewidth=2, markersize=8, label='L-BFGS-B', color='C1')
    ax1.set_xlabel('Noise Level (%)', fontsize=12)
    ax1.set_ylabel('Identification Residual (L2 norm)', fontsize=12)
    ax1.set_title('(a) Identification Error vs Noise', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.5, 10.5])

    # Plot 2: Simulation Error vs Noise
    ax2.plot(noise_levels, results['homotopy_sim_errors'],
             'o-', linewidth=2, markersize=8, label='Homotopy', color='C0')
    ax2.plot(noise_levels, results['lbfgsb_sim_errors'],
             's--', linewidth=2, markersize=8, label='L-BFGS-B', color='C1')
    ax2.set_xlabel('Noise Level (%)', fontsize=12)
    ax2.set_ylabel('Max Forward Simulation Error', fontsize=12)
    ax2.set_title('(b) Forward Simulation Error vs Noise', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-0.5, 10.5])

    plt.tight_layout()

    # Save figure
    figpath = os.path.join(_FIG_DIR, 'fig_noise_robustness.png')
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    print(f"\n[3] Figure saved: {figpath}")

    # plt.show()

# ============================================================================
# 6. MAIN
# ============================================================================

if __name__ == '__main__':
    # Run noise experiments
    results = run_noise_experiment(
        noise_levels=[0.0, 0.01, 0.05, 0.10],
        n_trials=5,
        verbose=True
    )

    # Generate figure
    plot_noise_robustness(results)

    print("\n" + "="*70)
    print("NOISE ROBUSTNESS EXPERIMENTS COMPLETED")
    print("="*70)
