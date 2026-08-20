#!/usr/bin/env python3
"""
demo_optimizer_comparison.py - Extended optimizer comparison for 21Paper

Compares the homotopy framework with modern gradient-based optimizers:
  - Vanilla Gradient Descent (baseline)
  - Adam optimizer (adaptive learning rate)
  - L-BFGS-B (quasi-Newton method)
  - Trust-Region Reflective (SciPy default)
  - Homotopy (Newton + Halley corrections)

System: Tank with orifice, MLP with 8 sigmoid units
Task: Identify f(h) = 0.5*sqrt(h) from 20 samples along the trajectory
Metric: Wall-clock time, final residual, iterations

Author: Rodolfo H. Rodrigo / INAUT-UNSJ-CONICET
Date: August 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
import os

# Ensure figures directory exists
_FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(_FIG_DIR, exist_ok=True)

# ============================================================================
# 1. PHYSICAL SYSTEM
# ============================================================================

ALPHA_TRUE = 0.5
H0 = 1.0

def u(t):
    """Input flow rate (step-ramp excitation of the MLP benchmark,
    same as demo_21paper_mlp.py: ramp on t in [1, 3])"""
    if t < 1.0:
        return 0.0
    elif t < 3.0:
        return 0.15 * (t - 1.0)
    else:
        return 0.3

def f_true(h):
    """True nonlinearity: f(h) = alpha * sqrt(h)"""
    return ALPHA_TRUE * np.sqrt(np.maximum(h, 0))

def rhs_true(t, y):
    """True ODE: dh/dt = u(t) - f(h)"""
    return u(t) - f_true(y[0])

# ============================================================================
# 2. MLP APPROXIMATION
# ============================================================================

class MLP:
    """Sigmoid MLP approximator (same architecture as demo_21paper_mlp.py):

        f(h) = sum_j w_j * sigma(W1_j*h + b1_j) + b2

    2M nonlinear parameters (W1, b1) and M+1 linear parameters (w, b2).
    """

    def __init__(self, M=8):
        self.M = M
        self.W1 = None     # Hidden slopes
        self.b1 = None     # Hidden biases
        self.w = None      # Output weights
        self.b2 = 0.0      # Output bias

    def sigma(self, x):
        """Sigmoid activation (numerically safe for large |x|)"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))

    def sigma_derivative(self, x, order=1):
        """Derivatives of sigmoid"""
        sig = self.sigma(x)
        if order == 1:
            return sig * (1 - sig)
        elif order == 2:
            return sig * (1 - sig) * (1 - 2*sig)
        elif order == 3:
            sig_prime = sig * (1 - sig)
            return sig_prime * (1 - 6*sig + 6*sig**2)
        else:
            raise NotImplementedError(f"Derivative order {order} not implemented")

    def phi(self, h):
        """Evaluate basis functions: Phi(h) = [sigma_1(h), ..., sigma_M(h)]"""
        if self.W1 is None or self.b1 is None:
            raise ValueError("MLP not initialized")
        h = np.asarray(h).reshape(-1, 1)
        x = h * self.W1.reshape(1, -1) + self.b1.reshape(1, -1)
        return self.sigma(x)

    def __call__(self, h):
        """Evaluate MLP approximation"""
        if self.w is None:
            raise ValueError("MLP weights not set")
        return self.phi(h) @ self.w + self.b2

    def initialize_uniform(self, h_data):
        """Sigmoid centres uniform on [h_min, h_max], initial slope 5,
        with the same seeded perturbations as demo_21paper_mlp.py so that
        every method starts from the identical Section 9.2 initialisation."""
        rng = np.random.RandomState(42)
        centers = np.linspace(h_data.min(), h_data.max(), self.M)
        self.W1 = np.full(self.M, 5.0) + 0.5 * rng.randn(self.M)
        self.b1 = -5.0 * centers + 0.3 * rng.randn(self.M)
        self.w = np.zeros(self.M)
        self.b2 = 0.0

    def pack_params(self):
        """Pack all parameters into a single vector"""
        return np.concatenate([self.W1, self.b1, self.w, [self.b2]])

    def unpack_params(self, theta):
        """Unpack parameters from a vector"""
        M = self.M
        self.W1 = theta[:M]
        self.b1 = theta[M:2*M]
        self.w = theta[2*M:3*M]
        self.b2 = theta[3*M]

    def _solve_linear(self, h_data, f_data):
        """Exact linear-stage solve for the last-layer weights and bias."""
        Phi = self.phi(h_data)
        Phi_ext = np.column_stack([Phi, np.ones(len(h_data))])
        wb = np.linalg.lstsq(Phi_ext, f_data, rcond=None)[0]
        self.w = wb[:-1]
        self.b2 = wb[-1]
        return Phi @ self.w + self.b2 - f_data

    def _jacobian_nl(self, h_data):
        """Jacobian of the residual w.r.t. the nonlinear parameters (W1, b1).

        With x_ij = W1_j*h_i + b1_j:
          dN_i/dW1_j = w_j * sigma'(x_ij) * h_i
          dN_i/db1_j = w_j * sigma'(x_ij)
        """
        h_col = np.asarray(h_data).reshape(-1, 1)
        x = h_col * self.W1.reshape(1, -1) + self.b1.reshape(1, -1)
        sig_p = self.sigma_derivative(x, order=1)
        J_W = sig_p * self.w.reshape(1, -1) * h_col
        J_b = sig_p * self.w.reshape(1, -1)
        return np.hstack([J_W, J_b])

    def _hessian_diag_nl(self, h_data):
        """Column sums of the diagonal parameter Hessian of the residual.

        d2N_i/dW1_j2 = w_j * sigma''(x_ij) * h_i^2
        d2N_i/db1_j2 = w_j * sigma''(x_ij)
        """
        h_col = np.asarray(h_data).reshape(-1, 1)
        x = h_col * self.W1.reshape(1, -1) + self.b1.reshape(1, -1)
        sig_pp = self.sigma_derivative(x, order=2)
        H_W = (sig_pp * self.w.reshape(1, -1) * h_col**2).sum(axis=0)
        H_b = (sig_pp * self.w.reshape(1, -1)).sum(axis=0)
        return np.concatenate([H_W, H_b])

    def fit_homotopy(self, h_data, f_data, max_outer=3, tol=1e-9, verbose=False):
        """Fit MLP with the bilinear homotopy corrections.

        Per outer iteration (Algorithm 1 of the manuscript):
          1. z1 exact linear solve for the last-layer weights and bias.
          2. z1 Newton step on the nonlinear parameters (W1, b1).
          3. z2 diagonal-Hessian Halley step on the same parameters.
        A step that increases the residual is reverted, so the nominal
        path has no tuning parameters (no learning rate).
        """
        M = self.M
        residual = self._solve_linear(h_data, f_data)
        residual_norm = np.linalg.norm(residual)
        n_outer = 0

        if verbose:
            print(f"  After linear solve: ||residual|| = {residual_norm:.3e}")

        for _ in range(max_outer):
            if residual_norm < tol:
                break
            n_outer += 1

            # z1 Newton on (W1, b1)
            J = self._jacobian_nl(h_data)
            delta1 = -np.linalg.lstsq(J, residual, rcond=None)[0]
            state_prev = (self.W1.copy(), self.b1.copy(), self.w.copy(), self.b2)
            self.W1 = self.W1 + delta1[:M]
            self.b1 = self.b1 + delta1[M:]
            residual_new = self._solve_linear(h_data, f_data)
            norm_new = np.linalg.norm(residual_new)
            if norm_new > residual_norm:
                self.W1, self.b1, self.w, self.b2 = state_prev
                break
            residual, residual_norm = residual_new, norm_new
            if verbose:
                print(f"  After z1 Newton:  ||residual|| = {residual_norm:.3e}")

            # z2 Halley (diagonal Hessian) on (W1, b1)
            J = self._jacobian_nl(h_data)
            H_diag = self._hessian_diag_nl(h_data)
            delta2 = np.zeros(2 * M)
            for j in range(2 * M):
                Jcol = J[:, j]
                gp = np.dot(Jcol, Jcol)
                if gp > 1e-12 and abs(H_diag[j]) > 1e-15:
                    delta2[j] = -0.5 * np.dot(residual, Jcol)**2 \
                                * H_diag[j] / gp**1.5
            state_prev = (self.W1.copy(), self.b1.copy(), self.w.copy(), self.b2)
            self.W1 = self.W1 + delta2[:M]
            self.b1 = self.b1 + delta2[M:]
            residual_new = self._solve_linear(h_data, f_data)
            norm_new = np.linalg.norm(residual_new)
            if norm_new > residual_norm:
                self.W1, self.b1, self.w, self.b2 = state_prev
            else:
                residual, residual_norm = residual_new, norm_new
                if verbose:
                    print(f"  After z2 Halley:  ||residual|| = {residual_norm:.3e}")

        return residual_norm, n_outer

    def fit_optimizer(self, h_data, f_data, method='GD', max_iter=5000,
                     lr=0.01, verbose=False):
        """
        Fit MLP using gradient-based optimizer

        Methods:
          - 'GD': Vanilla gradient descent
          - 'Adam': Adam optimizer
          - 'L-BFGS-B': L-BFGS-B quasi-Newton
          - 'TRF': Trust-Region Reflective
        """
        N = len(h_data)

        # Initialize weights to zero
        # Pack parameters (initialize_uniform already set w = 0, b2 = 0)
        theta0 = self.pack_params()

        def loss(theta):
            """MSE loss"""
            self.unpack_params(theta)
            f_pred = self.phi(h_data) @ self.w + self.b2
            return 0.5 * np.mean((f_pred - f_data)**2)

        def gradient(theta):
            """Gradient of MSE loss over all 3M+1 parameters"""
            self.unpack_params(theta)
            Phi = self.phi(h_data)
            residual = Phi @ self.w + self.b2 - f_data

            # Gradient w.r.t nonlinear parameters (W1, b1)
            grad_nl = self._jacobian_nl(h_data).T @ residual / N

            # Gradient w.r.t linear parameters (w, b2)
            grad_w = Phi.T @ residual / N
            grad_b2 = residual.sum() / N

            return np.concatenate([grad_nl, grad_w, [grad_b2]])

        # Select optimizer
        if method == 'GD':
            # Vanilla gradient descent with fixed learning rate
            theta = theta0.copy()
            for i in range(max_iter):
                grad = gradient(theta)
                theta -= lr * grad
            result_nit = max_iter
            result_nfev = max_iter

        elif method in ('Adam', 'Adam-decay'):
            # Adam optimizer.  With a constant step Adam normalises the update to
            # magnitude ~lr in every coordinate, so it stalls in a ball of that
            # radius instead of converging; the 'Adam-decay' variant applies a
            # cosine schedule, which is the standard remedy.
            theta = theta0.copy()
            m = np.zeros_like(theta)
            v = np.zeros_like(theta)
            beta1, beta2 = 0.9, 0.999
            eps = 1e-8
            use_decay = (method == 'Adam-decay')

            for i in range(max_iter):
                lr_i = lr * 0.5 * (1 + np.cos(np.pi * i / max_iter)) if use_decay else lr
                grad = gradient(theta)
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                m_hat = m / (1 - beta1**(i+1))
                v_hat = v / (1 - beta2**(i+1))
                theta -= lr_i * m_hat / (np.sqrt(v_hat) + eps)

            result_nit = max_iter
            result_nfev = max_iter

        elif method == 'L-BFGS-B':
            # L-BFGS-B quasi-Newton
            result = minimize(loss, theta0, method='L-BFGS-B', jac=gradient,
                            options={'maxiter': max_iter, 'disp': False})
            theta = result.x
            result_nit = result.nit
            result_nfev = result.nfev

        elif method == 'TRF':
            # Trust-Region Reflective (least-squares)
            from scipy.optimize import least_squares

            def residuals(theta):
                self.unpack_params(theta)
                f_pred = self.phi(h_data) @ self.w + self.b2
                return f_pred - f_data

            result = least_squares(residuals, theta0, method='trf',
                                  max_nfev=2500, verbose=0)
            theta = result.x
            result_nit = result.nfev  # least_squares doesn't have .nit
            result_nfev = result.nfev

        else:
            raise ValueError(f"Unknown method: {method}")

        # Unpack final parameters
        self.unpack_params(theta)

        # Compute final residual
        f_pred = self.phi(h_data) @ self.w + self.b2
        residual = f_pred - f_data
        residual_norm = np.linalg.norm(residual)

        if verbose:
            print(f"  {method}: ||residual|| = {residual_norm:.3e} "
                  f"({result_nit} iter, {result_nfev} nfev)")

        return residual_norm, result_nit, result_nfev

# ============================================================================
# 3. OPTIMIZER COMPARISON
# ============================================================================

def run_optimizer_comparison(n_runs=5, verbose=True):
    """
    Compare wall-clock performance of different optimizers

    Returns:
        results: dict with keys for each method
    """

    print("="*70)
    print("OPTIMIZER COMPARISON: MLP WITH 8 SIGMOID UNITS")
    print("="*70)

    # Generate data
    if verbose:
        print("\n[1] Generating data (20 points)...")

    # Same identification problem as Section 9 / benchmark_wallclock.py:
    # samples (h_i, f(h_i)) of the unknown nonlinearity along the trajectory.
    t_data = np.linspace(0, 10, 20)
    sol = solve_ivp(rhs_true, [0, 10], [H0], t_eval=t_data, method='RK45',
                    rtol=1e-10, atol=1e-12)
    h_data = sol.y[0]
    f_data = f_true(h_data)

    # Methods to compare
    # Learning rates for the first-order baselines are the best found by
    # tune_optimizer_baselines.py (grid sweep, identical 5000-iter budget).
    methods = [
        ('GD', 'Gradient Descent (tuned lr)',
         {'method': 'GD', 'max_iter': 5000, 'lr': 0.1}),
        ('Adam', 'Adam (cosine decay, tuned lr)',
         {'method': 'Adam-decay', 'max_iter': 5000, 'lr': 0.1}),
        ('L-BFGS-B', 'L-BFGS-B', {'method': 'L-BFGS-B', 'max_iter': 5000, 'lr': None}),
        ('TRF', 'Trust-Region Reflective', {'method': 'TRF', 'max_iter': 2500, 'lr': None}),
        ('Homotopy', 'Homotopy (z1+z2)', {'method': 'homotopy'}),
    ]

    results = {}

    for method_key, method_name, params in methods:
        if verbose:
            print(f"\n[2] Running {method_name}...")

        residuals = []
        times = []
        iters = []
        nfevs = []

        for run in range(n_runs):
            np.random.seed(42 + run)

            mlp = MLP(M=8)
            mlp.initialize_uniform(h_data)

            t_start = time.time()

            if params['method'] == 'homotopy':
                residual_norm, nit = mlp.fit_homotopy(h_data, f_data, verbose=False)
                nfev = nit
            else:
                residual_norm, nit, nfev = mlp.fit_optimizer(
                    h_data, f_data,
                    method=params['method'],
                    max_iter=params['max_iter'],
                    lr=params.get('lr'),
                    verbose=False
                )

            t_elapsed = time.time() - t_start

            residuals.append(residual_norm)
            times.append(t_elapsed)
            iters.append(nit)
            nfevs.append(nfev)

        # Compute medians
        results[method_key] = {
            'name': method_name,
            'residual': np.median(residuals),
            'time': np.median(times),
            'iter': int(np.median(iters)),
            'nfev': int(np.median(nfevs)),
        }

        if verbose:
            print(f"  Median residual: {results[method_key]['residual']:.3e}")
            print(f"  Median time: {results[method_key]['time']:.4f} s")
            print(f"  Median iter/nfev: {results[method_key]['iter']} / {results[method_key]['nfev']}")

    return results

# ============================================================================
# 4. GENERATE TABLE
# ============================================================================

def print_latex_table(results):
    """Generate LaTeX table for manuscript"""

    print("\n" + "="*70)
    print("LATEX TABLE FOR MANUSCRIPT")
    print("="*70)
    print()

    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Wall-clock comparison on the MLP identification problem (8")
    print(r"sigmoids, 20 data points, identical initialisation, single CPU core,")
    print(r"no warm start). Median over 5 runs.}")
    print(r"\label{tab:wallclock}")
    print(r"\rowcolors{2}{lightgray}{white}")
    print(r"\begin{tabular}{L{5cm} C{3cm} C{2.5cm} C{2.5cm}}")
    print(r"\toprule")
    print(r"\textbf{Method} & \textbf{Final residual} & \textbf{Iter./nfev} & \textbf{Wall time} \\")
    print(r"\midrule")

    for key in ['GD', 'Adam', 'L-BFGS-B', 'TRF', 'Homotopy']:
        r = results[key]
        residual_str = f"{r['residual']:.1e}".replace('e-0', r'\cdot 10^{-').replace('e-', r'\cdot 10^{-') + '}'
        if r['residual'] >= 1e-2:
            residual_str = f"{r['residual']:.1e}".replace('e-0', r'\cdot 10^{-').replace('e-', r'\cdot 10^{-') + '}'

        time_str = f"{r['time']:.4f}" if r['time'] >= 0.001 else f"{r['time']:.1e}"

        print(f"{r['name']} & ${residual_str}$ & {r['iter']:,} / {r['nfev']:,} & {time_str}\\,s \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()

# ============================================================================
# 5. MAIN
# ============================================================================

if __name__ == '__main__':
    # Run comparison
    results = run_optimizer_comparison(n_runs=5, verbose=True)

    # Generate LaTeX table
    print_latex_table(results)

    print("\n" + "="*70)
    print("OPTIMIZER COMPARISON COMPLETED")
    print("="*70)
