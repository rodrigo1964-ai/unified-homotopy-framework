# Unified Homotopy Framework

**Architecture-agnostic system identification and ODE simulation using homotopy series**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.31955865-blue.svg)](https://doi.org/10.6084/m9.figshare.31955865)

---

## Overview

This repository implements a **unified homotopy framework** for nonlinear system identification and ODE simulation. The framework is architecture-agnostic, working seamlessly with different function approximators (RBF networks, MLPs, etc.) without requiring gradient descent, backpropagation, or learning rate tuning.

**Key Innovation:** Direct analytical solution via homotopy series truncation (z₁, z₂, z₃) exploiting bilinear parameter structure.

---

## Features

- ✅ **Architecture-independent**: Works with RBF, MLP, and potentially other architectures
- ✅ **No backpropagation**: Analytical derivatives and direct solvers
- ✅ **Fast convergence**: 1-5 iterations to machine precision
- ✅ **Bilinear structure**: Linear parameters solved exactly via least squares
- ✅ **ODE simulation**: Integrated framework for system identification + forward simulation
- ✅ **Self-contained demos**: Complete examples with visualization

---

## Demos

### 1. RBF Demo (`demo_21paper.py`)

System identification using **Radial Basis Functions** (5 Gaussian centers):

```bash
python demo_21paper.py
```

**Results:**
- Max simulation error: 2.3%
- 35× improvement over linear baseline
- 1 iteration to convergence

### 2. MLP Demo (`demo_21paper_mlp.py`)

System identification using **Multi-Layer Perceptron** (8 sigmoids):

```bash
python demo_21paper_mlp.py
```

**Results:**
- Max simulation error: 0.96%
- 8.1× improvement over linear baseline
- 1 iteration to convergence

Both demos solve the same problem:
```
dh/dt + f(h) = u(t)
f(h) = 0.5√h  (unknown, to be identified)
```

---

## Method Overview

### Bilinear Structure

For both RBF and MLP, the residual has the form:

```
N(θ) = w^T · φ(h; θ_nl) - f_data
```

where:
- **w** (linear params): Solved exactly by least squares
- **θ_nl** (nonlinear params): Solved by Newton (z₁) + Halley (z₂)

### Homotopy Corrections

1. **z₁ (Newton)**: First-order correction using Jacobian
2. **z₂ (Halley)**: Second-order correction using Hessian diagonal
3. **z₃ (optional)**: Third-order correction for extreme accuracy

No learning rate, no epochs, no gradient descent.

---

## Installation

```bash
# Clone repository
git clone https://github.com/rodrigo1964-ai/unified-homotopy-framework.git
cd unified-homotopy-framework

# Install dependencies (minimal)
pip install numpy scipy matplotlib
```

**Requirements:**
- Python 3.8+
- NumPy
- SciPy (only for reference ODE solver)
- Matplotlib (for visualization)

---

## Usage Example

```python
import numpy as np
from demo_21paper_mlp import identify_mlp, simulate_with_mlp

# Your training data
h_data = np.array([...])  # states
f_data = np.array([...])  # function values

# Level 1: Identification
W1, b1, W2, b2, history = identify_mlp(h_data, f_data)

# Level 2: Simulation
t_eval = np.linspace(0, 10, 200)
h_simulated = simulate_with_mlp(W1, b1, W2, b2, t_eval)
```

---

## Project Structure

```
21Paper/
├── demo_21paper.py              # RBF demo (main reference)
├── demo_21paper_mlp.py          # MLP demo (architecture independence)
├── CLAUDE_mlp_demo.md           # MLP demo contract/specification
├── RESULTS_mlp_demo.md          # Detailed results and analysis
├── figures/                     # Generated plots
│   ├── fig_mlp_identification.png
│   ├── fig_mlp_simulation.png
│   ├── fig_mlp_sigmoids.png
│   └── ...
└── README.md                    # This file
```

---

## Results Summary

| Metric | RBF (5 centers) | MLP (8 sigmoids) |
|--------|-----------------|-------------------|
| Parameters | 15 | 25 |
| Max sim error | 2.3% | 0.96% |
| Improvement vs linear | 35× | 8.1× |
| Convergence | 1 iteration | 1 iteration |

Both architectures achieve excellent accuracy with minimal iterations, demonstrating the framework's effectiveness and generality.

---

## Mathematical Background

The framework is based on:

1. **Universal approximation**: Both RBF and MLP can approximate continuous functions
2. **Barron's theorem**: Provides convergence rates for neural networks
3. **Homotopy analysis**: Systematic series expansion for nonlinear problems
4. **Bilinear structure**: Exploits parameter separability for efficiency

See [RESULTS_mlp_demo.md](RESULTS_mlp_demo.md) for detailed mathematical derivations.

---

## Key Differences from Traditional Deep Learning

| Aspect | Traditional DL | This Framework |
|--------|---------------|----------------|
| Training | Gradient descent, backprop | Analytical Jacobian/Hessian |
| Hyperparameters | Learning rate, batch size, etc. | None required |
| Convergence | 100s-1000s epochs | 1-5 iterations |
| Derivatives | Autodiff | Closed-form expressions |
| Guarantees | Heuristic | Mathematical (homotopy series) |

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{rodrigo2026homotopy,
  author = {Rodrigo, Rodolfo H.},
  title = {Unified Homotopy Framework for System Identification},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/rodrigo1964-ai/unified-homotopy-framework},
  doi = {10.6084/m9.figshare.31955865},
  note = {ORCID: 0000-0002-8787-0038}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Author

**Rodolfo H. Rodrigo**  
INAUT-UNSJ-CONICET  
Argentina  
[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--8787--0038-green.svg)](https://orcid.org/0000-0002-8787-0038)

---

## Future Work

- [ ] Polynomial basis functions
- [ ] Wavelet basis functions
- [ ] Multi-output systems
- [ ] Partial differential equations (PDEs)
- [ ] Attention mechanisms
- [ ] Benchmark against traditional methods

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Acknowledgments

This work builds upon classical homotopy analysis methods and modern universal approximation theory, bridging numerical analysis and machine learning.
