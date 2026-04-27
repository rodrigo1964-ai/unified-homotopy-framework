#!/usr/bin/env python3
"""
demo_21paper.py - Complete demo for unified homotopy framework

Demonstrates:
  Level 1: RBF identification via parametric homotopy (Newton + Halley)
  Level 2: ODE simulation via integral homotopy with exponential kernel

System: Tank with orifice, dh/dt + alpha*sqrt(h) = u(t)
Data: 20 points generated with RK45
Goal: Identify RBF with 5 centers, simulate with integral solver

Author: Rodolfo H. Rodrigo / INAUT-UNSJ-CONICET
Date: April 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

# Ensure figures directory exists (relative to script location)
_FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(_FIG_DIR, exist_ok=True)

# Try sklearn for KMeans, fallback to simple implementation if not available
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available, using simple KMeans implementation")

# ============================================================================
# 1. PHYSICAL SYSTEM: TANK WITH ORIFICE
# ============================================================================

ALPHA_TRUE = 0.5  # True parameter: Cd * A_o * sqrt(2g) / A_t
H0 = 1.0          # Initial water level

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
# 2. GENERATE "EXPERIMENTAL" DATA (20 POINTS)
# ============================================================================

print("="*70)
print("21PAPER DEMO: Unified Homotopy Framework")
print("="*70)
print("\n[1] GENERATING EXPERIMENTAL DATA")
print("-" * 70)

t_data = np.linspace(0, 10, 20, dtype=np.float64)
sol = solve_ivp(rhs_true, [0, 10], [H0], t_eval=t_data, method='RK45', rtol=1e-10)
h_data = sol.y[0]
u_data = np.array([u(ti) for ti in t_data], dtype=np.float64)

# Estimate f from data using numerical derivatives
T = t_data[1] - t_data[0]
dhdt = np.zeros(20, dtype=np.float64)
dhdt[0] = (h_data[1] - h_data[0]) / T
dhdt[-1] = (h_data[-1] - h_data[-2]) / T
for i in range(1, 19):
    dhdt[i] = (h_data[i+1] - h_data[i-1]) / (2*T)

f_data = u_data - dhdt  # f(h) values at 20 points

print(f"Data points: {len(t_data)}")
print(f"Time range: [{t_data[0]:.1f}, {t_data[-1]:.1f}]")
print(f"h range: [{h_data.min():.4f}, {h_data.max():.4f}]")
print(f"f range: [{f_data.min():.4f}, {f_data.max():.4f}]")

# ============================================================================
# 3. SIMPLE KMEANS IMPLEMENTATION (FALLBACK)
# ============================================================================

def simple_kmeans(data, n_clusters, max_iter=10, random_state=42):
    """Simple KMeans for 1D data"""
    np.random.seed(random_state)
    data = data.reshape(-1, 1)

    # Initialize centers randomly
    indices = np.random.choice(len(data), n_clusters, replace=False)
    centers = data[indices].copy()

    for _ in range(max_iter):
        # Assign points to nearest center
        distances = np.abs(data - centers.T)
        labels = np.argmin(distances, axis=1)

        # Update centers
        new_centers = np.array([data[labels == k].mean() for k in range(n_clusters)])
        new_centers = new_centers.reshape(-1, 1)

        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    return centers.flatten()

# ============================================================================
# 4. LEVEL 1: RBF IDENTIFICATION VIA HOMOTOPY
# ============================================================================

print("\n[2] LEVEL 1: RBF IDENTIFICATION (Parametric Homotopy)")
print("-" * 70)

M = 5  # Number of RBF centers

# Step 0: K-means initialization
if HAS_SKLEARN:
    kmeans = KMeans(n_clusters=M, random_state=42, n_init=10).fit(h_data.reshape(-1, 1))
    c0 = kmeans.cluster_centers_.flatten()
else:
    c0 = simple_kmeans(h_data, M, random_state=42)

c0 = np.sort(c0)

# Initialize widths: distance to nearest neighbor
lambda0 = np.zeros(M, dtype=np.float64)
for j in range(M):
    dists = np.abs(c0 - c0[j])
    dists[j] = np.inf
    lambda0[j] = np.min(dists) * 0.8

print(f"\nK-means initialization:")
print(f"  c₀ = {c0}")
print(f"  λ₀ = {lambda0}")

# Store residuals for plotting
residual_norms = []

# Step 1: Linear least squares for weights (z₁ linear, exact)
Phi = np.zeros((20, M), dtype=np.float64)
for i in range(20):
    for j in range(M):
        r = np.abs(h_data[i] - c0[j]) / lambda0[j]
        Phi[i, j] = np.exp(-r**2)

w0 = np.linalg.lstsq(Phi, f_data, rcond=None)[0]
N0 = Phi @ w0 - f_data
norm_N0 = np.linalg.norm(N0)
residual_norms.append(norm_N0)

print(f"\nStep 1: LS for weights")
print(f"  ||N₀|| = {norm_N0:.6e}")

# Step 2: Newton correction on centers (z₁ nonlinear)
Jc = np.zeros((20, M), dtype=np.float64)
for i in range(20):
    for j in range(M):
        r_ij = (h_data[i] - c0[j]) / lambda0[j]
        phi_prime = -2 * r_ij * np.exp(-r_ij**2)
        Jc[i, j] = w0[j] * phi_prime * (-1.0 / lambda0[j])

delta_c1 = -np.linalg.lstsq(Jc, N0, rcond=None)[0]
c1 = c0 + delta_c1

# Recalculate Phi and weights
Phi1 = np.zeros((20, M), dtype=np.float64)
for i in range(20):
    for j in range(M):
        r = np.abs(h_data[i] - c1[j]) / lambda0[j]
        Phi1[i, j] = np.exp(-r**2)

w1 = np.linalg.lstsq(Phi1, f_data, rcond=None)[0]
N1 = Phi1 @ w1 - f_data
norm_N1 = np.linalg.norm(N1)
residual_norms.append(norm_N1)

print(f"Step 2: Newton on centers")
print(f"  ||N₁|| = {norm_N1:.6e}  (reduction: {norm_N0/norm_N1:.2f}x)")

# Step 3: Halley correction on centers (z₂ quadratic)
Hc_diag = np.zeros(M, dtype=np.float64)
for j in range(M):
    hess_sum = 0.0
    for i in range(20):
        r_ij = (h_data[i] - c1[j]) / lambda0[j]
        phi_pp = (4*r_ij**2 - 2) * np.exp(-r_ij**2)
        hess_sum += w1[j] * phi_pp / lambda0[j]**2
    Hc_diag[j] = hess_sum

# Recalculate Jacobian at c1
Jc1 = np.zeros((20, M), dtype=np.float64)
for i in range(20):
    for j in range(M):
        r_ij = (h_data[i] - c1[j]) / lambda0[j]
        phi_prime = -2 * r_ij * np.exp(-r_ij**2)
        Jc1[i, j] = w1[j] * phi_prime * (-1.0 / lambda0[j])

# Halley correction (component-wise)
delta_c2 = np.zeros(M, dtype=np.float64)
for j in range(M):
    Jcol = Jc1[:, j]
    gp = np.dot(Jcol, Jcol)  # (J^T J)_jj
    if abs(gp) > 1e-12 and abs(Hc_diag[j]) > 1e-15:
        delta_c2[j] = -0.5 * np.dot(N1, Jcol)**2 * Hc_diag[j] / gp**1.5

c2 = c1 + delta_c2

# Recalculate
Phi2 = np.zeros((20, M), dtype=np.float64)
for i in range(20):
    for j in range(M):
        r = np.abs(h_data[i] - c2[j]) / lambda0[j]
        Phi2[i, j] = np.exp(-r**2)

w2 = np.linalg.lstsq(Phi2, f_data, rcond=None)[0]
N2 = Phi2 @ w2 - f_data
norm_N2 = np.linalg.norm(N2)
residual_norms.append(norm_N2)

print(f"Step 3: Halley on centers")
print(f"  ||N₂|| = {norm_N2:.6e}  (reduction: {norm_N1/norm_N2:.2f}x)")

# Step 4: Newton correction on widths (z₁ on λ)
Jlam = np.zeros((20, M), dtype=np.float64)
for i in range(20):
    for j in range(M):
        r_ij = np.abs(h_data[i] - c2[j]) / lambda0[j]
        phi_prime = -2 * r_ij * np.exp(-r_ij**2)
        Jlam[i, j] = -w2[j] * phi_prime * r_ij / lambda0[j]

delta_lam1 = -np.linalg.lstsq(Jlam, N2, rcond=None)[0]
lambda1 = lambda0 + delta_lam1
lambda1 = np.maximum(lambda1, 0.01)  # Avoid negative widths

# Step 5: Final weights with updated parameters
Phi_final = np.zeros((20, M), dtype=np.float64)
for i in range(20):
    for j in range(M):
        r = np.abs(h_data[i] - c2[j]) / lambda1[j]
        Phi_final[i, j] = np.exp(-r**2)

w_final = np.linalg.lstsq(Phi_final, f_data, rcond=None)[0]
N_final = Phi_final @ w_final - f_data
norm_Nf = np.linalg.norm(N_final)
residual_norms.append(norm_Nf)

print(f"Step 4: Newton on widths + final LS")
print(f"  ||N_final|| = {norm_Nf:.6e}  (total reduction: {norm_N0/norm_Nf:.2f}x)")

print(f"\nFinal RBF parameters:")
print(f"  j      w_j         c_j        λ_j")
for j in range(M):
    print(f"  {j+1}    {w_final[j]:8.5f}    {c2[j]:8.5f}   {lambda1[j]:8.5f}")

# ============================================================================
# 5. LEVEL 2: ODE SIMULATION WITH IDENTIFIED RBF
# ============================================================================

print("\n[3] LEVEL 2: ODE SIMULATION (Integral Homotopy)")
print("-" * 70)

# Define RBF and derivatives
def rbf_f(h):
    """Identified RBF: f_hat(h)"""
    result = 0.0
    for j in range(M):
        r = (h - c2[j]) / lambda1[j]
        result += w_final[j] * np.exp(-r**2)
    return result

def rbf_df(h):
    """First derivative: f_hat'(h)"""
    result = 0.0
    for j in range(M):
        r = (h - c2[j]) / lambda1[j]
        result += w_final[j] * (-2*r) * np.exp(-r**2) / lambda1[j]
    return result

def rbf_d2f(h):
    """Second derivative: f_hat''(h)"""
    result = 0.0
    for j in range(M):
        r = (h - c2[j]) / lambda1[j]
        result += w_final[j] * (4*r**2 - 2) * np.exp(-r**2) / lambda1[j]**2
    return result

# Linearization for initial solution
h_eq = np.mean(h_data)
k = rbf_df(h_eq)

print(f"Linearization: k = f'(h_eq) = {k:.6f} at h_eq = {h_eq:.4f}")
print(f"Exponential kernel: K(t,s) = exp(-k(t-s))")

# Reduced nonlinearity
def f_tilde(h):
    return rbf_f(h) - k * h

def df_tilde(h):
    return rbf_df(h) - k

def d2f_tilde(h):
    return rbf_d2f(h)

# Integral solver with IIR filter (200 points, 10x finer than data)
n_sim = 200
t_sim = np.linspace(0, 10, n_sim, dtype=np.float64)
T_sim = t_sim[1] - t_sim[0]
u_sim = np.array([u(ti) for ti in t_sim], dtype=np.float64)

alpha_iir = np.exp(-k * T_sim)
h_sim = np.zeros(n_sim, dtype=np.float64)
h_sim[0] = H0
S = 0.0

for step in range(1, n_sim):
    h_sim[step] = h_sim[step-1]  # Initial estimate

    w_k = 1.0  # Trapezoidal weight (interior)
    y0_term = H0 * np.exp(-k * t_sim[step])

    # z₁: Newton correction
    ft = f_tilde(h_sim[step])
    g = h_sim[step] + T_sim*w_k*ft - T_sim*w_k*u_sim[step] - alpha_iir*S - y0_term
    gp = 1.0 + T_sim*w_k*df_tilde(h_sim[step])

    if abs(gp) > 1e-12:
        z1 = -g / gp
        h_sim[step] += z1

    # z₂: Halley correction
    ft = f_tilde(h_sim[step])
    g = h_sim[step] + T_sim*w_k*ft - T_sim*w_k*u_sim[step] - alpha_iir*S - y0_term
    gp = 1.0 + T_sim*w_k*df_tilde(h_sim[step])
    gpp = T_sim*w_k*d2f_tilde(h_sim[step])

    if abs(gp) > 1e-12:
        z2 = -0.5 * g**2 * gpp / gp**3
        h_sim[step] += z2

    # Update IIR state
    h_k = u_sim[step] - f_tilde(h_sim[step])
    S = alpha_iir * S + T_sim * w_k * h_k

# Linear solution for comparison
h_linear = np.zeros(n_sim, dtype=np.float64)
h_linear[0] = H0
S_lin = 0.0

for step in range(1, n_sim):
    y0_term = H0 * np.exp(-k * t_sim[step])
    h_linear[step] = -alpha_iir * S_lin + y0_term

    h_k_lin = u_sim[step]
    S_lin = alpha_iir * S_lin + T_sim * h_k_lin

# High-resolution reference solution
sol_ref = solve_ivp(rhs_true, [0, 10], [H0], t_eval=t_sim, method='RK45', rtol=1e-10)
h_ref = sol_ref.y[0]

# Compute errors
error_rbf = np.abs(h_sim - h_ref)
error_linear = np.abs(h_linear - h_ref)
max_error_rbf = np.max(error_rbf)
max_error_linear = np.max(error_linear)

print(f"\nSimulation results (200 time points):")
print(f"  Max error RBF vs true:    {max_error_rbf:.6e}")
print(f"  Max error linear vs true: {max_error_linear:.6e}")
print(f"  Improvement factor:       {max_error_linear/max_error_rbf:.2f}x")

# ============================================================================
# 6. GENERATE FIGURES
# ============================================================================

print("\n[4] GENERATING FIGURES")
print("-" * 70)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 9

# Figure 1: Identification quality (f_true vs f_RBF vs data)
fig1, ax1 = plt.subplots(figsize=(8, 5))

h_plot = np.linspace(h_data.min()*0.9, h_data.max()*1.1, 200)
f_true_plot = ALPHA_TRUE * np.sqrt(h_plot)
f_rbf_plot = np.array([rbf_f(h) for h in h_plot])

ax1.plot(h_plot, f_true_plot, 'k-', linewidth=2, label='f(h) = 0.5√h (true)')
ax1.plot(h_plot, f_rbf_plot, 'r--', linewidth=1.5, label='f_RBF(h) (identified)')
ax1.plot(h_data, f_data, 'bo', markersize=5, label='Data (20 points)')
ax1.set_xlabel('Water level h')
ax1.set_ylabel('Nonlinearity f(h)')
ax1.set_title('Level 1: RBF Identification Quality')
ax1.legend()
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(os.path.join(_FIG_DIR, 'fig_identification.png'), dpi=300)
print("  Saved: fig_identification.png")

# Figure 2: Simulation comparison
fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax2a.plot(t_sim, h_ref, 'k-', linewidth=2, label='True (RK45 reference)')
ax2a.plot(t_sim, h_sim, 'r--', linewidth=1.5, label='RBF homotopy solver')
ax2a.plot(t_sim, h_linear, 'b:', linewidth=1.5, label='Linear solution')
ax2a.plot(t_data, h_data, 'ko', markersize=3, alpha=0.5, label='Data points')
ax2a.set_ylabel('Water level h(t)')
ax2a.set_title('Level 2: ODE Simulation Comparison')
ax2a.legend(loc='best')
ax2a.grid(True, alpha=0.3)

ax2b.plot(t_sim, u_sim, 'g-', linewidth=2, label='Input u(t)')
ax2b.set_xlabel('Time t')
ax2b.set_ylabel('Input flow u(t)')
ax2b.legend(loc='best')
ax2b.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig(os.path.join(_FIG_DIR, 'fig_simulation.png'), dpi=300)
print("  Saved: fig_simulation.png")

# Figure 3: Residual convergence
fig3, ax3 = plt.subplots(figsize=(8, 5))

steps = ['Step 1\n(LS)', 'Step 2\n(Newton c)', 'Step 3\n(Halley c)', 'Step 4\n(Final)']
ax3.semilogy(steps, residual_norms, 'ro-', linewidth=2, markersize=8)
ax3.set_ylabel('Residual norm ||N||')
ax3.set_title('Parametric Homotopy Convergence')
ax3.grid(True, alpha=0.3, which='both')
fig3.tight_layout()
fig3.savefig(os.path.join(_FIG_DIR, 'fig_residuals.png'), dpi=300)
print("  Saved: fig_residuals.png")

# Figure 4: RBF components
fig4, ax4 = plt.subplots(figsize=(8, 5))

for j in range(M):
    comp = np.array([w_final[j] * np.exp(-((h - c2[j])/lambda1[j])**2) for h in h_plot])
    ax4.plot(h_plot, comp, '--', alpha=0.5, linewidth=1, label=f'φ_{j+1}')

ax4.plot(h_plot, f_rbf_plot, 'k-', linewidth=2, label='Sum (f_RBF)')
ax4.axhline(0, color='gray', linestyle='-', linewidth=0.5)
ax4.set_xlabel('Water level h')
ax4.set_ylabel('RBF components')
ax4.set_title(f'RBF Decomposition ({M} Gaussian centers)')
ax4.legend(loc='best', ncol=2, fontsize=8)
ax4.grid(True, alpha=0.3)
fig4.tight_layout()
fig4.savefig(os.path.join(_FIG_DIR, 'fig_rbf_components.png'), dpi=300)
print("  Saved: fig_rbf_components.png")

# ============================================================================
# 7. SUMMARY TABLE
# ============================================================================

print("\n[5] SUCCESS CRITERIA")
print("-" * 70)

reduction_ratio = norm_N0 / norm_Nf
improvement_factor = max_error_linear / max_error_rbf
h_range = h_data.max() - h_data.min()
relative_error_rbf = max_error_rbf / h_range

print(f"  ||N_final|| / ||N_0||:      {reduction_ratio:.4f}  {'✓ PASS' if reduction_ratio > 10 else '✗ FAIL'} (target: < 0.1)")
print(f"  Max RBF error / h_range:    {relative_error_rbf:.4f}  {'✓ PASS' if relative_error_rbf < 0.05 else '✗ FAIL'} (target: < 0.05)")
print(f"  Max linear error / h_range: {max_error_linear/h_range:.4f}  {'✓ PASS' if max_error_linear/h_range > 0.1 else '✗ FAIL'} (target: > 0.1)")
print(f"  Improvement RBF/linear:     {improvement_factor:.2f}x   {'✓ PASS' if improvement_factor > 3 else '✗ FAIL'} (target: > 3x)")

print("\n" + "="*70)
print("DEMO COMPLETE")
print("="*70)
print("\nKey results:")
print(f"  • RBF identified with {M} centers from {len(t_data)} data points")
print(f"  • Residual reduced {reduction_ratio:.1f}x via parametric homotopy")
print(f"  • Simulation accuracy improved {improvement_factor:.1f}x over linear solution")
print(f"  • All figures saved to {_FIG_DIR}/")
print("\nThis demonstrates:")
print("  1. Parametric homotopy (Level 1) identifies nonlinear RBF from sparse data")
print("  2. Integral homotopy (Level 2) solves ODE without numerical derivatives")
print("  3. Unified framework: same Newton+Halley corrections for both levels")
print("="*70)
