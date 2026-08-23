# Universal Law of Noise in Complex Systems: Stochastic Regularization

This repository presents the core theoretical foundations and implementation for the **Stochastic Fokker-Planck Regularization** of complex systems, based on the **SCCI-2026 Framework** by **Mohamed Nour Kayad (MNK)** (ORCID: 0009-0002-1443-7599).

## 📌 Abstract / Core Concept

When complex optimization routines encounter flat landscapes or bad generalization plateaus, the underlying geometry deforms, resulting in a **degenerate Hessian**:

\[\det H \rightarrow 0\]

This framework introduces an intentional **Anisotropic Informational Noise** combined with a **Fokker-Planck formulation with mass reinjection** to guarantee global convergence where classical determinism fails.

## 💻 Usage Example

To protect your neural networks against Gradient Flattening, implement the optimizer as follows:

```python
import torch
import torch.nn as nn
from sfpd_optimizer import StochasticFokkerPlanckDescent

# 1. Define any standard deep neural network
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 2. Instantiate the SCCI-2026 customized optimizer
optimizer = StochasticFokkerPlanckDescent(
    model.parameters(), 
    lr=1e-3, 
    sigma=1e-4,          # Controls the diffusion matrix
    det_threshold=1e-5   # Triggers the reinjection source r(q)
)
```

## 📄 Citation & Attribution

```bibtex
@article{kayad2026stochastic,
  author    = {Nour Kayad, Mohamed},
  title     = {Stabilité spectrale, variations de connexions et régularisation stochastique des surfaces extrémales quantiques},
  institution = {Ministry of National Education and Vocational Training (MENFOP), Djibouti},
  year      = {2026},
  month     = {August},
  note      = {ORCID iD: 0009-0002-1443-7599}
}
```

**License**: Distributed under the GNU GPL v3.0 License.
