#!/usr/bin/env python3
"""
benchmark_wallclock.py - Wall-clock comparison for MLP identification

Runs three methods on the SAME identification problem (orifice-tank, 20 data
points, MLP with 8 sigmoids, same initialization):

    1. Gradient descent on all 25 params (eta = 1e-2, 5000 iter)
    2. SciPy Trust-Region Reflective (least_squares, method='trf')
       NOTE: 'lm' is not applicable here because the problem is
       underdetermined (N=20 residuals, n=25 parameters); 'lm' requires
       N >= n. TRF is the standard SciPy choice for this regime and is
       conceptually similar (trust-region nonlinear LS).
    3. Homotopy (z1 LS + z1 Newton + z2 Halley) -- this work

Each method is run REPEATS times. Reports final residual ||N||_2 and the
median wall-clock time. The numbers feed Section 10.5 ("Wall-clock
comparison") of the manuscript.

Usage:
    python3 benchmark_wallclock.py [--repeats N]

Requires:
    numpy, scipy
    demo_21paper_mlp.py in the same directory (provides shared primitives)

Author: R. H. Rodrigo, H. D. Patino  /  INAUT-UNSJ-CONICET
"""

from __future__ import annotations

import argparse
import sys
import time
from statistics import median

import numpy as np
from scipy.optimize import least_squares

# Reuse the demo primitives so the benchmark is identical to the published
# identification problem. We import after silencing stdout because the demo
# functions print as a side effect when called via main().
import demo_21paper_mlp as demo


# ---------------------------------------------------------------------------
# Common problem setup
# ---------------------------------------------------------------------------

def build_problem():
    """Same data as demo: RK45 on orifice-tank, 20 points."""
    from scipy.integrate import solve_ivp
    h0 = 1.0
    t_eval = np.linspace(0.0, 10.0, 20)
    sol = solve_ivp(demo.ode_rhs, (0.0, 10.0), [h0],
                    t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12)
    h_data = sol.y[0]
    f_data = np.array([demo.f_true(h) for h in h_data])
    return h_data, f_data


def initial_params(h_data, M=8):
    """
    Identical initialization to demo.initialize_mlp. We delegate to the
    demo function rather than re-implementing it: this guarantees the
    benchmark uses the same initial parameter vector that the demo and
    the manuscript report.
    """
    return demo.initialize_mlp(h_data, M=M)


def residual_vector(theta, h_data, f_data, M=8):
    """Residual N(theta) = f_MLP(h; theta) - f_data, theta in R^25."""
    W1 = theta[:M]
    b1 = theta[M:2*M]
    W2 = theta[2*M:3*M]
    b2 = theta[3*M]
    return demo.mlp_predict(h_data, W1, b1, W2, b2) - f_data


def residual_norm(theta, h_data, f_data):
    return float(np.linalg.norm(residual_vector(theta, h_data, f_data)))


# ---------------------------------------------------------------------------
# Method 1: Gradient descent on all parameters
# ---------------------------------------------------------------------------

def run_gradient_descent(h_data, f_data, eta=1e-2, n_iter=5000, M=8):
    """
    Vanilla GD on the MSE loss J(theta) = 0.5 * ||N(theta)||^2.
    All 25 parameters updated jointly. No bilinear separation.
    Gradient computed analytically (chain rule, matches backprop).
    """
    W1, b1, W2, b2 = initial_params(h_data, M=M)
    theta = np.concatenate([W1, b1, W2, [b2]])

    N = len(h_data)

    def jacobian_full(W1, b1, W2):
        # J shape (N, 25): columns = (W1, b1, W2, b2)
        z = np.outer(h_data, W1) + b1            # (N, M)
        s = demo.sigmoid(z)                       # (N, M)
        sp = demo.sigmoid_prime(z)                # (N, M)
        # dN/dW1[j] = W2[j] * sp_ij * h_i
        dN_dW1 = sp * W2 * h_data[:, None]        # (N, M)
        # dN/db1[j] = W2[j] * sp_ij
        dN_db1 = sp * W2                          # (N, M)
        # dN/dW2[j] = s_ij
        dN_dW2 = s                                # (N, M)
        # dN/db2 = 1
        dN_db2 = np.ones((N, 1))
        return np.hstack([dN_dW1, dN_db1, dN_dW2, dN_db2])

    for _ in range(n_iter):
        W1 = theta[:M]
        b1 = theta[M:2*M]
        W2 = theta[2*M:3*M]
        b2 = theta[3*M]
        N0 = demo.mlp_predict(h_data, W1, b1, W2, b2) - f_data
        J = jacobian_full(W1, b1, W2)
        grad = J.T @ N0       # gradient of 0.5 * ||N||^2
        theta = theta - eta * grad

    return theta


# ---------------------------------------------------------------------------
# Method 2: SciPy Trust-Region Reflective (TRF)
# ---------------------------------------------------------------------------

def run_trust_region(h_data, f_data, M=8):
    """
    SciPy least_squares with the TRF algorithm. Default tolerances; finite
    difference Jacobian. All 25 parameters jointly. TRF (rather than LM) is
    used because the problem has 20 residuals and 25 parameters, which makes
    LM inapplicable (it requires N >= n).
    """
    W1, b1, W2, b2 = initial_params(h_data, M=M)
    theta0 = np.concatenate([W1, b1, W2, [b2]])

    res = least_squares(
        residual_vector,
        theta0,
        args=(h_data, f_data, M),
        method='trf',
    )
    return res.x, res.nfev


# ---------------------------------------------------------------------------
# Method 3: Homotopy (this work)
# ---------------------------------------------------------------------------

def run_homotopy(h_data, f_data, M=8):
    """
    z1 LS on (W2, b2) -- exact one step.
    z1 Newton on (W1, b1).
    z2 Halley on (W1, b1).
    z1 LS on (W2, b2) again.
    Single outer iteration (the demo converges in iteration 0).

    Reuses primitives from demo_21paper_mlp.
    """
    W1, b1, W2, b2 = initial_params(h_data, M=M)

    # Step 1: z1 LS on linear part
    W2, b2 = demo.solve_linear_weights(h_data, f_data, W1, b1)
    N0 = demo.mlp_predict(h_data, W1, b1, W2, b2) - f_data

    # If already converged, stop (this happens in the demo)
    if np.linalg.norm(N0) < 1e-6:
        theta = np.concatenate([W1, b1, W2, [b2]])
        return theta

    # Step 2: z1 Newton on (W1, b1)
    J_nl = demo.compute_jacobian_nl(h_data, W1, b1, W2)
    delta1 = -np.linalg.lstsq(J_nl, N0, rcond=None)[0]
    W1 = W1 + delta1[:M]
    b1 = b1 + delta1[M:]

    # Re-solve linear part
    W2, b2 = demo.solve_linear_weights(h_data, f_data, W1, b1)
    N1 = demo.mlp_predict(h_data, W1, b1, W2, b2) - f_data

    # Step 3: z2 Halley on (W1, b1)
    J_nl = demo.compute_jacobian_nl(h_data, W1, b1, W2)
    delta2 = demo.compute_halley_correction(h_data, W1, b1, W2, N1, J_nl)
    W1 = W1 + delta2[:M]
    b1 = b1 + delta2[M:]

    # Final linear solve
    W2, b2 = demo.solve_linear_weights(h_data, f_data, W1, b1)

    theta = np.concatenate([W1, b1, W2, [b2]])
    return theta


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def time_method(name, fn, h_data, f_data, repeats):
    """Run fn(h_data, f_data) `repeats` times, return (median_time, residual)."""
    times = []
    last_theta = None
    last_extra = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(h_data, f_data)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if isinstance(result, tuple):
            last_theta, last_extra = result
        else:
            last_theta = result
    res = residual_norm(last_theta, h_data, f_data)
    return median(times), res, last_extra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=5,
                        help='Number of timing repetitions (median reported)')
    args = parser.parse_args()

    print()
    print('=' * 78)
    print('WALL-CLOCK BENCHMARK FOR MLP IDENTIFICATION')
    print('=' * 78)
    print(f'Problem    : orifice-tank, 20 data points, MLP with 8 sigmoids')
    print(f'Repeats    : {args.repeats}  (median time reported)')
    print(f'Hardware   : single CPU, no warm start, identical initialization')
    print()

    # Suppress demo stdout when we import-call its primitives only
    h_data, f_data = build_problem()

    print('-' * 78)
    print(f'{"Method":<40} {"Final ||N||":>15} {"Time (s)":>12} {"Iters":>8}')
    print('-' * 78)

    # 1. Gradient descent
    t_gd, r_gd, _ = time_method(
        'Gradient descent (eta=1e-2)',
        lambda h, f: run_gradient_descent(h, f, eta=1e-2, n_iter=5000),
        h_data, f_data, args.repeats)
    print(f'{"Gradient descent (eta=1e-2, 5000 it)":<40} '
          f'{r_gd:>15.4e} {t_gd:>12.4f} {"5000":>8}')

    # 2. Trust-Region Reflective
    t_tr, r_tr, nfev_tr = time_method(
        'Trust-Region Reflective (SciPy)',
        lambda h, f: run_trust_region(h, f),
        h_data, f_data, args.repeats)
    print(f'{"Trust-Region Reflective (SciPy)":<40} '
          f'{r_tr:>15.4e} {t_tr:>12.4f} {str(nfev_tr):>8}')

    # 3. Homotopy
    t_h, r_h, _ = time_method(
        'Homotopy z1+z2 (this work)',
        lambda h, f: run_homotopy(h, f),
        h_data, f_data, args.repeats)
    print(f'{"Homotopy z1+z2 (this work)":<40} '
          f'{r_h:>15.4e} {t_h:>12.4f} {"1 outer":>8}')

    print('-' * 78)
    print()
    print('Notes:')
    print('  * Gradient descent and LM are run on all 25 parameters jointly.')
    print('  * Homotopy exploits the bilinear structure: W2,b2 by LS;')
    print('    W1,b1 by Newton (z1) + Halley (z2).')
    print('  * Wall-clock excludes data generation, ODE simulation, plotting.')
    print()


if __name__ == '__main__':
    main()
