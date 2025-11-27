"""EggRoll-style Evolution Strategies with low-rank noise for PyTorch.

This module provides TorchEggrollES, an Evolution Strategies optimizer that uses
low-rank noise for 2D parameters (matrices) and standard Gaussian noise for 1D
parameters. Key features include:

- Low-rank noise for matrices: reduces variance in gradient estimates
- Antithetic sampling: half population uses +noise, half uses -noise
- Proper ES gradient estimate: weighted average of noise by fitness
"""

from typing import Callable, List, Optional

import torch
from torch import nn


def generate_lora_noise(
    param: torch.Tensor,
    rank: int,
    sigma: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate low-rank noise for a 2D parameter (matrix).

    For a matrix of shape (out_dim, in_dim), generates:
    - A: (out_dim, rank)
    - B: (in_dim, rank)
    - noise = A @ B.T * sigma / sqrt(rank)

    The division by sqrt(rank) normalizes the variance.

    Args:
        param: The parameter tensor to generate noise for (must be 2D)
        rank: The rank of the low-rank noise
        sigma: The noise scale
        seed: Random seed for reproducibility
        device: Device to generate noise on

    Returns:
        Low-rank noise tensor of same shape as param
    """
    out_dim, in_dim = param.shape
    r = min(rank, out_dim, in_dim)

    # Use deterministic seeding for reproducibility (needed for antithetic sampling)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # Generate A and B factors
    total_elements = out_dim + in_dim
    lora_params = torch.randn(
        total_elements, r, generator=gen, device=device, dtype=param.dtype
    )
    B = lora_params[:in_dim]  # (in_dim, r)
    A = lora_params[in_dim:]  # (out_dim, r)

    # noise = A @ B.T, scaled
    noise = (A @ B.t()) * (sigma / (r**0.5))
    return noise


def generate_standard_noise(
    param: torch.Tensor,
    sigma: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate standard Gaussian noise for non-matrix parameters.

    Args:
        param: The parameter tensor to generate noise for
        sigma: The noise scale
        seed: Random seed for reproducibility
        device: Device to generate noise on

    Returns:
        Gaussian noise tensor of same shape as param
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    noise = torch.randn(param.shape, generator=gen, device=device, dtype=param.dtype)
    return noise * sigma


class TorchEggrollES:
    """
    Evolution Strategies trainer with EggRoll-style low-rank noise
    for 2D parameters (matrices).

    Key features:
    - Low-rank noise for matrices: reduces variance in gradient estimates
    - Antithetic sampling: half population uses +noise, half uses -noise
    - Proper ES gradient estimate: weighted average of noise by fitness

    Works with any nn.Module. Use param_filter to select which parameters
    to optimize (e.g., only LoRA weights).

    Example:
        >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        >>> es = TorchEggrollES(model, pop_size=32, sigma=0.1, lr=0.1)
        >>>
        >>> def fitness(m):
        ...     # Evaluate model fitness (higher is better)
        ...     return float(-loss_fn(m(x), y))
        >>>
        >>> for step in range(100):
        ...     mean_fitness = es.step(fitness)
        ...     print(f"Step {step}: {mean_fitness:.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        pop_size: int = 32,
        sigma: float = 0.02,
        lr: float = 0.05,
        rank: int = 4,
        device: Optional[torch.device] = None,
        param_filter: Optional[Callable[[nn.Parameter, str], bool]] = None,
        normalize_fitness: bool = True,
        antithetic: bool = True,
    ) -> None:
        """
        Initialize the ES optimizer.

        Args:
            model: nn.Module to optimise.
            pop_size: Population size per ES step. If antithetic=True, must be even.
            sigma: Noise scale.
            lr: Learning rate for ES update.
            rank: Rank of low-rank noise for 2D params.
            device: Device to run on; inferred from model if None.
            param_filter: Optional (param, name) -> bool to choose which params
                          to include in ES updates.
            normalize_fitness: If True, z-score fitnesses; else only mean-center.
            antithetic: If True, use antithetic (mirrored) sampling for lower variance.
        """
        self.model = model
        self.pop_size = pop_size
        self.sigma = sigma
        self.lr = lr
        self.rank = rank
        self.normalize_fitness = normalize_fitness
        self.antithetic = antithetic

        if antithetic and pop_size % 2 != 0:
            raise ValueError("pop_size must be even when using antithetic sampling")

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = torch.device(device)
        self.model.to(self.device)

        # Gather parameters (and names) to evolve
        self.params: List[nn.Parameter] = []
        self.names: List[str] = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if param_filter is not None and not param_filter(p, name):
                continue
            self.params.append(p)
            self.names.append(name)

        if not self.params:
            raise ValueError("TorchEggrollES: no parameters selected for ES.")

        # Base (mean) parameters - the center of our search distribution
        self.base_params: List[torch.Tensor] = [
            p.detach().clone().to(self.device) for p in self.params
        ]

        # Epoch counter for seeding
        self.epoch = 0

    def _refresh_base(self) -> None:
        """Copy current model params to base_params."""
        for base, p in zip(self.base_params, self.params):
            base.copy_(p.detach())

    def _generate_noise(self, param_idx: int, pop_idx: int) -> torch.Tensor:
        """
        Generate noise for a specific parameter and population member.

        With antithetic sampling:
        - pop_idx 0,1 share the same base noise (1 is negated)
        - pop_idx 2,3 share the same base noise (3 is negated)
        - etc.
        """
        base = self.base_params[param_idx]
        device = self.device

        if self.antithetic:
            # Pairs share noise seeds
            noise_idx = pop_idx // 2
            sign = 1.0 if pop_idx % 2 == 0 else -1.0
        else:
            noise_idx = pop_idx
            sign = 1.0

        # Create deterministic seed from epoch, noise_idx, and param_idx
        seed = hash((self.epoch, noise_idx, param_idx)) & 0x7FFFFFFF

        if base.ndim == 2:
            noise = generate_lora_noise(base, self.rank, self.sigma, seed, device)
        else:
            noise = generate_standard_noise(base, self.sigma, seed, device)

        return noise * sign

    def step(self, eval_fn: Callable[[nn.Module], float]) -> float:
        """
        Execute one ES optimization step.

        Args:
            eval_fn: Function that takes the model and returns a scalar fitness
                     (higher is better).

        Returns:
            Mean fitness across the population (for logging).
        """
        self._refresh_base()
        pop_size = self.pop_size
        device = self.device

        # Store noises for gradient computation
        noises: List[List[torch.Tensor]] = [
            [torch.zeros_like(base, device=device) for base in self.base_params]
            for _ in range(pop_size)
        ]
        fitnesses = torch.empty(pop_size, device=device, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            for i in range(pop_size):
                # 1) Apply noise to each parameter
                for j, (p, base) in enumerate(zip(self.params, self.base_params)):
                    noise = self._generate_noise(j, i)
                    p.data = base + noise
                    noises[i][j] = noise

                # 2) Evaluate fitness
                fitness = eval_fn(self.model)
                fitnesses[i] = float(fitness)

        # 3) Normalize fitnesses (z-score or mean-center)
        rewards = fitnesses
        if self.normalize_fitness:
            std = rewards.std()
            if std > 1e-8:
                rewards = (rewards - rewards.mean()) / std
            else:
                rewards = rewards - rewards.mean()
        else:
            rewards = rewards - rewards.mean()

        # 4) ES gradient estimate and update
        # grad ≈ (1/N) * Σ fitness_i * noise_i
        # Then we scale by sqrt(N) as in HyperscaleES
        with torch.no_grad():
            for j, base in enumerate(self.base_params):
                grad = torch.zeros_like(base, device=device)
                for i in range(pop_size):
                    grad.add_(rewards[i] * noises[i][j])
                grad /= pop_size
                # Scale by sqrt(pop_size) as in HyperscaleES
                grad *= pop_size**0.5

                # Apply update
                base.add_(self.lr * grad)
                self.params[j].data.copy_(base)

        self.epoch += 1
        return float(fitnesses.mean().item())
