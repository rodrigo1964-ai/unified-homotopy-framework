#!/usr/bin/env python3
"""
tune_optimizer_baselines.py - Learning-rate tuning for the gradient-based baselines.

Rationale: reporting a hand-picked learning rate makes a baseline look weak for
reasons that have nothing to do with the method under study.  Reviewers read that
as an unfair comparison.  This script sweeps the learning rate for the first-order
baselines (GD, Adam) under an identical iteration budget and reports the BEST
configuration found for each, so the wall-clock table compares tuned baselines.

Author: Rodolfo H. Rodrigo / INAUT-UNSJ-CONICET
Date: August 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
import time

from demo_optimizer_comparison import MLP, rhs_true, f_true, H0

BUDGET = 5000          # identical iteration budget for every first-order method
LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def make_data(n=20, t_final=10.0):
    # Same target as Section 9 / benchmark_wallclock.py: exact samples f(h_i)
    t = np.linspace(0, t_final, n)
    sol = solve_ivp(rhs_true, [0, t_final], [H0], t_eval=t, method='RK45',
                    rtol=1e-10, atol=1e-12)
    h = sol.y[0]
    return h, f_true(h)


def run(method, lr, h_data, f_data):
    mlp = MLP(M=8)
    mlp.initialize_uniform(h_data)
    t0 = time.time()
    res, nit, nfev = mlp.fit_optimizer(h_data, f_data, method=method,
                                       max_iter=BUDGET, lr=lr)
    return res, time.time() - t0, nit, nfev


def sweep(method, h_data, f_data):
    print(f"\n{method}  (budget = {BUDGET} iterations)")
    print(f"  {'lr':>8} | {'residual':>12} | {'time':>8}")
    print("  " + "-" * 34)
    best = None
    for lr in LR_GRID:
        try:
            res, t, nit, nfev = run(method, lr, h_data, f_data)
            if not np.isfinite(res):
                print(f"  {lr:>8.0e} | {'diverged':>12} | {'--':>8}")
                continue
            print(f"  {lr:>8.0e} | {res:>12.3e} | {t:>7.3f}s")
            if best is None or res < best[1]:
                best = (lr, res, t, nit, nfev)
        except (FloatingPointError, OverflowError, ValueError):
            print(f"  {lr:>8.0e} | {'failed':>12} | {'--':>8}")
    return best


if __name__ == '__main__':
    np.seterr(over='raise', invalid='raise')
    h_data, f_data = make_data()

    print("=" * 60)
    print("LEARNING-RATE SWEEP FOR FIRST-ORDER BASELINES")
    print("=" * 60)

    results = {}
    for method in ('GD', 'Adam', 'Adam-decay'):
        best = sweep(method, h_data, f_data)
        results[method] = best
        if best:
            print(f"  -> best lr = {best[0]:.0e}, residual = {best[1]:.3e}")

    # Methods with no learning rate to tune
    print("\nMethods without a learning-rate hyperparameter:")
    for method in ('L-BFGS-B', 'TRF'):
        res, t, nit, nfev = run(method, None, h_data, f_data)
        results[method] = (None, res, t, nit, nfev)
        print(f"  {method:<10} residual = {res:.3e}  time = {t:.4f}s  ({nit} it / {nfev} nfev)")

    # Homotopy
    mlp = MLP(M=8)
    mlp.initialize_uniform(h_data)
    t0 = time.time()
    res_h, n_outer = mlp.fit_homotopy(h_data, f_data)
    t_h = time.time() - t0
    results['Homotopy'] = (None, res_h, t_h, n_outer, n_outer)
    print(f"  {'Homotopy':<10} residual = {res_h:.3e}  time = {t_h:.6f}s  ({n_outer} outer)")

    print("\n" + "=" * 60)
    print("SUMMARY (tuned baselines)")
    print("=" * 60)
    for k, v in results.items():
        if v is None:
            continue
        lr, res, t = v[0], v[1], v[2]
        lr_s = f"lr={lr:.0e}" if lr is not None else "--"
        print(f"  {k:<10} {lr_s:>10}  residual={res:.3e}  time={t:.6f}s")
