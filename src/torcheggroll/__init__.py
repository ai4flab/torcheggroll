"""TorchEggroll - EggRoll-style Evolution Strategies for PyTorch.

This library provides Evolution Strategies (ES) optimization with EggRoll-style
low-rank noise for efficient gradient estimation on neural network parameters.

Key features:
- Low-rank noise for matrices: reduces variance in gradient estimates
- Antithetic sampling: half population uses +noise, half uses -noise
- Works with any nn.Module via param_filter

Example:
    >>> from torcheggroll import TorchEggrollES
    >>> import torch.nn as nn
    >>>
    >>> model = nn.Linear(10, 5)
    >>> es = TorchEggrollES(model, pop_size=32, sigma=0.1, lr=0.1)
    >>>
    >>> def fitness(m):
    ...     # Your evaluation function
    ...     return float(m(torch.randn(1, 10)).sum())
    >>>
    >>> for step in range(100):
    ...     mean_fitness = es.step(fitness)
"""

from .es import (
    TorchEggrollES,
    generate_lora_noise,
    generate_standard_noise,
)

__version__ = "0.1.0"

__all__ = [
    "TorchEggrollES",
    "generate_lora_noise",
    "generate_standard_noise",
]
