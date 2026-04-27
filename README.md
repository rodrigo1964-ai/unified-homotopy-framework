# Unified Homotopy Framework

**Architecture-agnostic system identification and ODE simulation via homotopy series**

Companion code for the paper *"A Unified Homotopy Framework for Nonlinear System Identification and Simulation: Constructive Dimensioning of Bilinear Neural Architectures"* by R. H. Rodrigo and H. D. Patiño (Instituto de Automática, INAUT, UNSJ–CONICET).

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.31955865-blue.svg)](https://doi.org/10.6084/m9.figshare.31955865)
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--8787--0038-green.svg)](https://orcid.org/0000-0002-8787-0038)

---

## What this is

A reference implementation of a unified algorithmic framework, derived from Liao's homotopy deformation equation, that solves two problems traditionally treated by separate methods:

1. **Identification** of a neural function approximator from data (RBF, MLP, ...).
2. **Simulation** of the resulting nonlinear ODE.

Both problems are reduced to closed-form Newton (z₁) and Halley (z₂) corrections derived from the same deformation equation. The bilinear parameter structure shared by common architectures—a last layer that is linear in its weights and internal parameters that are nonlinear—is exploited to solve the linear sub-problem exactly in one step and to apply curvature corrections to the nonlinear sub-problem.

The convergence conditions induce a **constructive dimensioning chain**

```
ε_sim → ε_id → M_min → N_min → T_max
```

that maps a target simulation accuracy to the minimum number of neurons, minimum number of training samples, and maximum admissible integration step. The choice of the auxiliary linear operator acts as an explicit, quantifiable inductive bias.

---

## Demos

Both demos identify the unknown nonlinearity of the discharge-flow ODE

```
dh/dt + 0.5·sqrt(h) = u(t)
```

from 20 synthetic data points and then forward-simulate the identified model.

### `demo_21paper.py` — RBF (5 Gaussian centers)

```bash
python demo_21paper.py
```

Reported result: maximum simulation error 2.3% over a 200-step horizon, identification residual 3.1·10⁻², 1 outer iteration, 35× improvement over the linearized baseline.

### `demo_21paper_mlp.py` — MLP (8 sigmoid units)

```bash
python demo_21paper_mlp.py
```

Reported result: maximum simulation error 0.96%, identification residual 1.7·10⁻⁸ (machine precision), 1 outer iteration, 8.1× improvement over the linearized baseline.

Both demos run on a single CPU and require only NumPy, SciPy, and Matplotlib.

---

## Installation

```bash
git clone https://github.com/rodrigo1964-ai/unified-homotopy-framework.git
cd unified-homotopy-framework
pip install -r requirements.txt
```

Tested with Python 3.8, 3.10, and 3.12.

---

## Method overview

### Bilinear residual

For both architectures the identification residual has the form

```
N(θ) = Φ(η) · w − f
```

where `w` are linear parameters (last-layer weights) and `η` are nonlinear parameters (RBF centers and widths, or MLP hidden weights and biases).

### Homotopy corrections

* **z₁ (Newton)** — first-order correction using the Jacobian of the residual.
* **z₂ (Halley)** — second-order correction using the diagonal of the parameter Hessian.
* **z₃ (optional)** — third-order correction; not used in the demos.

The linear sub-problem is solved exactly by least squares (z₁ exact, z₂ = z₃ = 0). The nonlinear sub-problem is corrected by z₁ + z₂. There is no scalar learning rate and no stochastic gradient descent step; the magnitude of each correction is determined by the local geometry of the residual.

### Wall-clock comparison (MLP, 20 data points, 1 CPU core)

Median over 5 runs, identical initialization, no warm start.

| Method                                  | Final residual         | Iter./nfev  | Wall time  |
|-----------------------------------------|------------------------|-------------|------------|
| Gradient descent (η = 10⁻²)             | 9.9·10⁻³               | 5,000       | 0.26 s     |
| SciPy Trust-Region Reflective           | 3.1·10⁻⁶               | 2,500 (max) | 1.53 s     |
| Homotopy (z₁ + z₂, this work)           | 1.7·10⁻⁸               | 1 outer     | 0.0002 s   |

Gradient descent does not converge to a useful tolerance within 5,000 iterations and stalls near 10⁻². TRF reaches 3·10⁻⁶ but exhausts its default function-evaluation budget (100·n = 2,500) before meeting its tolerance. Levenberg–Marquardt (`method='lm'`) is not applicable: the problem is underdetermined (20 residuals, 25 parameters), which the LM solver does not support. The homotopy corrections, by exploiting the bilinear separation, are five orders of magnitude more accurate than gradient descent and two orders more accurate than TRF, in roughly three orders of magnitude less wall time.

Reproduce with `python3 benchmark_wallclock.py`.

---

## Repository structure

```
unified-homotopy-framework/
├── demo_21paper.py            RBF demo
├── demo_21paper_mlp.py        MLP demo
├── benchmark_wallclock.py     Reproducible wall-clock benchmark vs. GD and TRF
├── figures/                   PNG figures generated by the demos
├── tests/                     Smoke tests for the demos
├── requirements.txt           Pinned dependencies
├── CITATION.cff               Citation metadata (GitHub-renderable)
├── LICENSE                    MIT
└── README.md                  This file
```

---

## Limitations

The numerical demonstration in this repository covers a one-dimensional first-order nonlinear ODE with two architectures (RBF, MLP). Generalization to multi-dimensional, stiff, and chaotic systems and to architectures with convolutional, recurrent, or attention-based hidden representations is consistent with the bilinear structure but has not been validated experimentally here. See the *Limitations and future work* section of the paper.

---

## Citation

If you use this code or framework in your research, please cite:

```bibtex
@software{rodrigo2026homotopy,
  author       = {Rodrigo, Rodolfo H. and Pati\~{n}o, H. Daniel},
  title        = {Unified Homotopy Framework for Nonlinear System
                  Identification and Simulation},
  year         = {2026},
  publisher    = {GitHub / Figshare},
  url          = {https://github.com/rodrigo1964-ai/unified-homotopy-framework},
  doi          = {10.6084/m9.figshare.31955865},
  note         = {Companion code for the paper of the same name.}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Authors

**Rodolfo H. Rodrigo** (corresponding) — `rrodrigo@inaut.unsj.edu.ar`  
**H. Daniel Patiño** — `dpatino@inaut.unsj.edu.ar`  
Instituto de Automática (INAUT), Universidad Nacional de San Juan – CONICET, Argentina.

---

## Acknowledgments

This work was carried out at INAUT–UNSJ–CONICET. We thank Shijun Liao for the foundational work on the Homotopy Analysis Method on which this paper builds.
