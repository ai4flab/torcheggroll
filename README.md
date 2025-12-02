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

# Training data
x_train = torch.randn(64, 10)
y_train = torch.randn(64, 5)

# Define your loss function
def mse_loss(outputs, targets):
    return ((outputs - targets) ** 2).mean()

# Optimize!
for step in range(100):
    mean_fitness = es.step(x_train, mse_loss, y_train)
    print(f"Step {step}: fitness = {mean_fitness:.4f}")
```

## Features

- **Low-rank noise for matrices**: Uses EggRoll-style A @ B.T noise for 2D parameters, reducing variance in gradient estimates
- **Antithetic sampling**: Half the population uses +noise, half uses -noise, further reducing variance
- **Vectorized evaluation**: Uses torch.vmap for parallel population evaluation on GPU/MPS
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

- `step(inputs, loss_fn, targets) -> float`: Run one ES step using vmap for parallel evaluation.
  - `inputs`: Input tensor (batch_size, ...) broadcast to all population members
  - `loss_fn(outputs, targets) -> scalar`: Loss function (lower is better)
  - `targets`: Target tensor for supervised learning
  - Returns mean fitness across population (negated loss, so higher is better)

**Async-Friendly API** (for custom evaluation, e.g., with LLM calls):

- `prepare_population() -> List[Dict[str, Tensor]]`: Generate perturbed parameters without evaluation
- `get_stacked_params() -> Dict[str, Tensor]`: Get params stacked as (pop_size, *shape) for vmap
- `apply_fitness_scores(fitnesses: Tensor) -> float`: Apply fitness scores and update parameters

**Checkpointing:**

- `state_dict() -> Dict`: Return optimizer state for checkpointing
- `load_state_dict(state: Dict)`: Load optimizer state from checkpoint

### Utility Functions

**Low-level noise generation:**

- `generate_lora_noise(param, rank, sigma, seed, device)`: Generate low-rank A @ B.T noise for a 2D tensor
- `generate_standard_noise(param, sigma, seed, device)`: Generate standard Gaussian noise

**Higher-level utilities for custom ES implementations:**

- `generate_noise_for_shapes(shapes, ranks, pop_size, sigma, epoch, device, ...)`: Generate noise for multiple tensors at once
- `compute_es_gradient(noise, rewards, normalize_fitness=True)`: Compute ES gradient from noise and rewards

These utilities are useful when building custom ES optimizers that don't use `nn.Module`:

```python
from torcheggroll import generate_noise_for_shapes, compute_es_gradient
import torch

# Define parameter shapes and ranks
shapes = {"W1": (20, 10), "b1": (20,), "W2": (5, 20)}
ranks = {"W1": 4, "b1": None, "W2": 4}  # None = standard noise

# Generate noise for population
noise = generate_noise_for_shapes(
    shapes, ranks,
    pop_size=32,
    sigma=0.1,
    epoch=0,
    device=torch.device("cpu"),
)
# noise["W1"] shape: (32, 20, 10)
# noise["b1"] shape: (32, 20)

# After evaluating fitness...
rewards = torch.randn(32)  # fitness per population member

# Compute gradients
grads = compute_es_gradient(noise, rewards)
# grads["W1"] shape: (20, 10) - same as original param
```

## Async-Friendly API

For custom evaluation (e.g., with async LLM calls or external fitness functions), use the two-step API:

```python
import torch
from torcheggroll import TorchEggrollES

model = nn.Linear(10, 5)
es = TorchEggrollES(model, pop_size=32, sigma=0.1, lr=0.1)

# Step 1: Generate perturbed parameters
population = es.prepare_population()
# population is List[Dict[str, Tensor]], one dict per population member

# Step 2: Evaluate externally (your custom logic)
fitnesses = []
for params in population:
    # Custom evaluation - could be async LLM calls, etc.
    output = torch.func.functional_call(model, params, (inputs,))
    fitness = your_custom_fitness_fn(output)
    fitnesses.append(fitness)
fitnesses = torch.tensor(fitnesses)

# Step 3: Apply fitness scores and update
mean_fitness = es.apply_fitness_scores(fitnesses)
```

For vmap-style parallel evaluation:

```python
# Get stacked params for vmap
population = es.prepare_population()
stacked = es.get_stacked_params()  # Dict[name -> (pop_size, *shape)]

# Use vmap for parallel evaluation
from torch.func import vmap, functional_call

batched_forward = vmap(
    lambda *p: functional_call(model, dict(zip(stacked.keys(), p)), (inputs,)),
    in_dims=tuple(0 for _ in stacked)
)
all_outputs = batched_forward(*stacked.values())  # (pop_size, batch, ...)

# Compute fitness and apply
fitnesses = compute_fitness(all_outputs, targets)
mean_fitness = es.apply_fitness_scores(fitnesses)
```

## Checkpointing

Save and restore optimizer state:

```python
# Save
state = es.state_dict()
torch.save(state, "checkpoint.pt")

# Load
state = torch.load("checkpoint.pt")
es.load_state_dict(state)
```

## Examples

See the [examples/](examples/) directory for complete examples:

- `nano_classifier.py`: Train a factorized classifier using ES
- `nano_egg/`: Train a byte-level language model (minGRU) using ES

Run the classifier example:

```bash
python examples/nano_classifier.py --steps 50 --pop-size 64
```

Run the language model example:

```bash
# Quick test (~1 min)
pip install torcheggroll[nano-egg]
python examples/nano_egg/train.py --mode float --epochs 50 \
    --hidden-dim 32 --n-layers 1 --pop-size 512 --max-docs 1000
```

## How It Works

TorchEggroll implements Evolution Strategies with three key optimizations:

1. **Low-rank noise**: For matrix parameters (2D tensors), instead of generating full Gaussian noise, we generate low-rank noise as `A @ B.T` where A and B are small random matrices. This reduces the effective dimensionality of the search space.

2. **Antithetic sampling**: For each random perturbation, we evaluate both +noise and -noise. This creates correlated pairs that reduce variance in the gradient estimate.

3. **Vectorized evaluation**: Uses `torch.vmap` and `torch.func.functional_call` to evaluate the entire population in parallel, enabling efficient GPU/MPS acceleration.

The ES gradient is estimated as:
```
grad ≈ (1/N) * Σ fitness_i * noise_i
```

Where `fitness_i` is the normalized fitness of the i-th population member.

## Related Projects

- [hyperfunc](https://github.com/ai4flab/hyperfunc): Higher-level ES optimization framework that uses TorchEggroll internally

## License

MIT
