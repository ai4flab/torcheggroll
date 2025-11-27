"""TorchEggroll - EggRoll-style Evolution Strategies for PyTorch.

This library provides Evolution Strategies (ES) optimization with EggRoll-style
low-rank noise for efficient gradient estimation on neural network parameters.

Key features:
- Low-rank noise for matrices: reduces variance in gradient estimates
- Antithetic sampling: half population uses +noise, half uses -noise
- Vectorized evaluation using torch.vmap for parallel processing
- Works with any nn.Module via param_filter

Example:
    >>> import torch
    >>> import torch.nn as nn
    >>> from torcheggroll import TorchEggrollES
    >>>
    >>> model = nn.Linear(10, 5)
    >>> es = TorchEggrollES(model, pop_size=32, sigma=0.1, lr=0.1)
    >>>
    >>> # Training data
    >>> x = torch.randn(32, 10)
    >>> y = torch.randn(32, 5)
    >>>
    >>> def mse_loss(outputs, targets):
    ...     return ((outputs - targets) ** 2).mean()
    >>>
    >>> for step in range(100):
    ...     mean_fitness = es.step(x, mse_loss, y)
"""

from .es import (
    TorchEggrollES,
    generate_lora_noise,
    generate_standard_noise,
    generate_noise_for_shapes,
    compute_es_gradient,
)

__version__ = "0.1.0"

__all__ = [
    "TorchEggrollES",
    "generate_lora_noise",
    "generate_standard_noise",
    "generate_noise_for_shapes",
    "compute_es_gradient",
]
