# TorchEggroll

[![PyPI version](https://badge.fury.io/py/torcheggroll.svg)](https://pypi.org/project/torcheggroll/)
[![Tests](https://github.com/ai4flab/torcheggroll/actions/workflows/test.yml/badge.svg)](https://github.com/ai4flab/torcheggroll/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

EggRoll-style Evolution Strategies with low-rank noise for PyTorch.

TorchEggroll provides a simple, efficient Evolution Strategies (ES) optimizer for PyTorch models. It uses low-rank noise for matrix parameters, reducing variance in gradient estimates while maintaining computational efficiency.

## Installation

```bash
pip install torcheggroll
```

Or with uv:

```bash
uv add torcheggroll
```

## Quick Start

```python
import torch
import torch.nn as nn
from torcheggroll import TorchEggrollES

# Define your model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# Create the ES optimizer
es = TorchEggrollES(
    model=model,
    pop_size=32,      # Population size (must be even for antithetic sampling)
    sigma=0.1,        # Noise scale
    lr=0.1,           # Learning rate
    rank=4,           # Rank for low-rank noise on matrices
    antithetic=True,  # Use antithetic sampling for lower variance
)

# Define your fitness function (higher is better)
def fitness(m):
    pred = m(x_train)
    loss = ((pred - y_train) ** 2).mean()
    return -float(loss)  # Negative loss = higher fitness

# Optimize!
for step in range(100):
    mean_fitness = es.step(fitness)
    print(f"Step {step}: fitness = {mean_fitness:.4f}")
```

## Features

- **Low-rank noise for matrices**: Uses EggRoll-style A @ B.T noise for 2D parameters, reducing variance in gradient estimates
- **Antithetic sampling**: Half the population uses +noise, half uses -noise, further reducing variance
- **Parameter filtering**: Optimize only specific parameters using `param_filter`
- **Works with any nn.Module**: No modifications needed to your model

## API Reference

### TorchEggrollES

```python
TorchEggrollES(
    model: nn.Module,           # Model to optimize
    pop_size: int = 32,         # Population size per step
    sigma: float = 0.02,        # Noise scale
    lr: float = 0.05,           # Learning rate
    rank: int = 4,              # Rank for low-rank noise
    device: torch.device = None,# Device (inferred from model if None)
    param_filter: Callable = None,  # Filter which params to optimize
    normalize_fitness: bool = True, # Z-score normalize fitness
    antithetic: bool = True,    # Use antithetic sampling
)
```

**Methods:**

- `step(eval_fn) -> float`: Run one ES step. `eval_fn(model) -> float` should return fitness (higher is better). Returns mean fitness across population.

### Utility Functions

- `generate_lora_noise(param, rank, sigma, seed, device)`: Generate low-rank A @ B.T noise for a 2D tensor
- `generate_standard_noise(param, sigma, seed, device)`: Generate standard Gaussian noise

## Examples

See the [examples/](examples/) directory for complete examples:

- `nano_classifier.py`: Train a factorized classifier using ES

Run the example:

```bash
python examples/nano_classifier.py --steps 50 --pop-size 64
```

## How It Works

TorchEggroll implements Evolution Strategies with two key optimizations:

1. **Low-rank noise**: For matrix parameters (2D tensors), instead of generating full Gaussian noise, we generate low-rank noise as `A @ B.T` where A and B are small random matrices. This reduces the effective dimensionality of the search space.

2. **Antithetic sampling**: For each random perturbation, we evaluate both +noise and -noise. This creates correlated pairs that reduce variance in the gradient estimate.

The ES gradient is estimated as:
```
grad ≈ (1/N) * Σ fitness_i * noise_i
```

Where `fitness_i` is the normalized fitness of the i-th population member.

## Related Projects

- [hyperfunc](https://github.com/ai4flab/hyperfunc): Higher-level ES optimization framework that uses TorchEggroll internally

## License

MIT
