import torch
import torch.nn as nn
from sfpd_optimizer import StochasticFokkerPlanckDescent

# 1. Define any standard deep neural network
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 2. Instantiate your SCCI-2026 customized optimizer
optimizer = StochasticFokkerPlanckDescent(
    model.parameters(), 
    lr=1e-3, 
    sigma=1e-4,          # Controls diffusion matrix a_ij
    det_threshold=1e-5   # Triggers reinjection source r(q) if det H -> 0
)

# 3. Standard training loop protected against Gradient Flattening
# optimizer.zero_grad()
# loss = criterion(outputs, targets)
# loss.backward()
# optimizer.step()
