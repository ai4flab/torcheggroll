"""EggRoll-style Evolution Strategies with low-rank noise for PyTorch.

This module provides TorchEggrollES, an Evolution Strategies optimizer that uses
low-rank noise for 2D parameters (matrices) and standard Gaussian noise for 1D
parameters. Key features include:

- Low-rank noise for matrices: reduces variance in gradient estimates
- Antithetic sampling: half population uses +noise, half uses -noise
- Vectorized evaluation using torch.vmap for parallel population processing
- Proper ES gradient estimate: weighted average of noise by fitness
"""

import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.func import functional_call, vmap


def generate_lora_noise_batched(
    shape: Tuple[int, int],
    rank: int,
    sigma: float,
    pop_size: int,
    seeds: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Generate batched low-rank noise for a 2D parameter (matrix).

    For a matrix of shape (out_dim, in_dim), generates for each population member:
    - A: (out_dim, rank)
    - B: (in_dim, rank)
    - noise = A @ B.T * sigma / sqrt(rank)

    Args:
        shape: (out_dim, in_dim) shape of the parameter
        rank: The rank of the low-rank noise
        sigma: The noise scale
        pop_size: Number of population members
        seeds: (pop_size,) tensor of seeds for each member
        device: Device to generate noise on
        dtype: Data type for noise

    Returns:
        (pop_size, out_dim, in_dim) batched noise tensor
    """
    out_dim, in_dim = shape
    r = min(rank, out_dim, in_dim)

    # Generate all A and B factors at once
    # We use the seeds to create deterministic but different noise per member
    total_elements = (out_dim + in_dim) * r

    # Create batched random noise using seeds
    # For reproducibility, we generate sequentially but store batched
    all_noise = []
    for i in range(pop_size):
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seeds[i].item()))
        lora_params = torch.randn(out_dim + in_dim, r, generator=gen, device=device, dtype=dtype)
        B = lora_params[:in_dim]   # (in_dim, r)
        A = lora_params[in_dim:]   # (out_dim, r)
        noise = (A @ B.t()) * (sigma / (r ** 0.5))
        all_noise.append(noise)

    return torch.stack(all_noise, dim=0)  # (pop_size, out_dim, in_dim)


def generate_standard_noise_batched(
    shape: Tuple[int, ...],
    sigma: float,
    pop_size: int,
    seeds: Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Generate batched standard Gaussian noise for non-matrix parameters.

    Args:
        shape: Shape of the parameter
        sigma: The noise scale
        pop_size: Number of population members
        seeds: (pop_size,) tensor of seeds
        device: Device to generate noise on
        dtype: Data type for noise

    Returns:
        (pop_size, *shape) batched noise tensor
    """
    all_noise = []
    for i in range(pop_size):
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seeds[i].item()))
        noise = torch.randn(shape, generator=gen, device=device, dtype=dtype) * sigma
        all_noise.append(noise)

    return torch.stack(all_noise, dim=0)  # (pop_size, *shape)


# Keep old functions for backwards compatibility
def generate_lora_noise(
    param: Tensor,
    rank: int,
    sigma: float,
    seed: int,
    device: torch.device,
) -> Tensor:
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

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    total_elements = out_dim + in_dim
    lora_params = torch.randn(
        total_elements, r, generator=gen, device=device, dtype=param.dtype
    )
    B = lora_params[:in_dim]
    A = lora_params[in_dim:]

    noise = (A @ B.t()) * (sigma / (r ** 0.5))
    return noise


def generate_standard_noise(
    param: Tensor,
    sigma: float,
    seed: int,
    device: torch.device,
) -> Tensor:
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


# ============================================================
# Stateless utility functions for use without nn.Module
# ============================================================


def generate_noise_for_shapes(
    shapes: Dict[str, Tuple[int, ...]],
    ranks: Dict[str, Optional[int]],
    pop_size: int,
    sigma: float,
    epoch: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    antithetic: bool = True,
) -> Dict[str, Tensor]:
    """
    Generate low-rank/standard noise for arbitrary tensor shapes.

    This is the low-level API for use without nn.Module. It generates
    EGGROLL-style low-rank noise for 2D tensors and standard Gaussian
    noise for other shapes.

    Args:
        shapes: Dict mapping param name -> shape tuple
        ranks: Dict mapping param name -> noise rank (None = standard Gaussian)
        pop_size: Population size
        sigma: Noise scale
        epoch: Epoch number for seeding (ensures different noise each epoch)
        device: Target device
        dtype: Target dtype
        antithetic: Use mirrored sampling (pop_size must be even)

    Returns:
        Dict mapping param name -> (pop_size, *shape) noise tensor

    Example:
        >>> shapes = {"layer1": (64, 128), "bias": (64,)}
        >>> ranks = {"layer1": 4, "bias": None}
        >>> noise = generate_noise_for_shapes(shapes, ranks, 32, 0.1, 0, device)
        >>> noise["layer1"].shape  # (32, 64, 128)
    """
    if antithetic and pop_size % 2 != 0:
        raise ValueError("pop_size must be even when using antithetic sampling")

    n_seeds = pop_size // 2 if antithetic else pop_size
    all_noise = {}

    for idx, (name, shape) in enumerate(shapes.items()):
        rank = ranks.get(name)

        # Generate seeds for this parameter
        seeds = torch.tensor(
            [hash((epoch, i, idx)) & 0x7FFFFFFF for i in range(n_seeds)],
            device=device,
            dtype=torch.long
        )

        # Generate noise based on shape and rank
        if len(shape) == 2 and rank is not None:
            # 2D tensor with rank specified -> low-rank noise
            noise = generate_lora_noise_batched(
                shape, rank, sigma, n_seeds, seeds, device, dtype
            )
        else:
            # Non-2D or no rank -> standard Gaussian
            noise = generate_standard_noise_batched(
                shape, sigma, n_seeds, seeds, device, dtype
            )

        # For antithetic: duplicate and negate
        if antithetic:
            noise = torch.cat([noise, -noise], dim=0)

        all_noise[name] = noise

    return all_noise


def compute_es_gradient(
    noise: Dict[str, Tensor],
    rewards: Tensor,
    normalize_fitness: bool = True,
) -> Dict[str, Tensor]:
    """
    Compute ES gradient from noise and rewards.

    This implements the standard ES gradient estimate:
        grad = E[reward * noise] ≈ (1/N) * sum(reward_i * noise_i)

    With normalization and scaling for stable updates.

    Args:
        noise: Dict of noise tensors, each (pop_size, *shape)
        rewards: Fitness scores (pop_size,) - higher is better
        normalize_fitness: Z-score normalize rewards before gradient computation

    Returns:
        Dict of gradient tensors (same shapes as base params, without pop_size dim)

    Example:
        >>> noise = {"layer1": torch.randn(32, 64, 128)}
        >>> rewards = torch.randn(32)
        >>> grads = compute_es_gradient(noise, rewards)
        >>> grads["layer1"].shape  # (64, 128)
    """
    pop_size = rewards.shape[0]

    # Normalize rewards
    if normalize_fitness:
        std = rewards.std()
        if std > 1e-8:
            rewards = (rewards - rewards.mean()) / std
        else:
            rewards = rewards - rewards.mean()
    else:
        rewards = rewards - rewards.mean()

    # Compute gradient for each parameter
    gradients = {}
    for name, n in noise.items():
        # n shape: (pop_size, *param_shape)
        expanded_rewards = rewards.view(pop_size, *([1] * (n.ndim - 1)))
        grad = (expanded_rewards * n).mean(dim=0) * (pop_size ** 0.5)
        gradients[name] = grad

    return gradients


class TorchEggrollES:
    """
    Evolution Strategies trainer with EggRoll-style low-rank noise
    for 2D parameters (matrices).

    Key features:
    - Low-rank noise for matrices: reduces variance in gradient estimates
    - Antithetic sampling: half population uses +noise, half uses -noise
    - Vectorized evaluation using torch.vmap for parallel processing
    - Proper ES gradient estimate: weighted average of noise by fitness

    Works with any nn.Module. Use param_filter to select which parameters
    to optimize (e.g., only LoRA weights).

    Example:
        >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        >>> es = TorchEggrollES(model, pop_size=32, sigma=0.1, lr=0.1)
        >>>
        >>> # Define loss function and optimize
        >>> def mse_loss(outputs, targets):
        ...     return ((outputs - targets) ** 2).mean()
        >>>
        >>> for step in range(100):
        ...     mean_fitness = es.step(x_batch, mse_loss, y_batch)
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
        self.param_names: List[str] = []
        self.param_shapes: List[Tuple[int, ...]] = []
        self.param_dtypes: List[torch.dtype] = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if param_filter is not None and not param_filter(p, name):
                continue
            self.param_names.append(name)
            self.param_shapes.append(tuple(p.shape))
            self.param_dtypes.append(p.dtype)

        if not self.param_names:
            raise ValueError("TorchEggrollES: no parameters selected for ES.")

        # Store base parameters as dict for functional_call
        self.base_params: Dict[str, Tensor] = {}
        for name in self.param_names:
            # Navigate to get the actual parameter
            parts = name.split('.')
            obj = self.model
            for part in parts:
                obj = getattr(obj, part)
            self.base_params[name] = obj.detach().clone()

        # Epoch counter for seeding
        self.epoch = 0

    def _refresh_base(self) -> None:
        """Copy current model params to base_params."""
        for name in self.param_names:
            parts = name.split('.')
            obj = self.model
            for part in parts:
                obj = getattr(obj, part)
            self.base_params[name] = obj.detach().clone()

    def _generate_all_noise(self) -> Dict[str, Tensor]:
        """
        Generate noise for all parameters and all population members.

        Returns:
            Dict mapping param name -> (pop_size, *param_shape) noise tensor
        """
        pop_size = self.pop_size
        device = self.device

        # For antithetic sampling, we only need half the seeds
        if self.antithetic:
            n_seeds = pop_size // 2
        else:
            n_seeds = pop_size

        all_noise = {}

        for idx, name in enumerate(self.param_names):
            shape = self.param_shapes[idx]
            dtype = self.param_dtypes[idx]

            # Generate seeds for this parameter
            seeds = torch.tensor(
                [hash((self.epoch, i, idx)) & 0x7FFFFFFF for i in range(n_seeds)],
                device=device,
                dtype=torch.long
            )

            # Generate noise
            if len(shape) == 2:
                noise = generate_lora_noise_batched(
                    shape, self.rank, self.sigma, n_seeds, seeds, device, dtype
                )
            else:
                noise = generate_standard_noise_batched(
                    shape, self.sigma, n_seeds, seeds, device, dtype
                )

            # For antithetic: duplicate and negate
            if self.antithetic:
                noise = torch.cat([noise, -noise], dim=0)  # (pop_size, *shape)

            all_noise[name] = noise

        return all_noise

    def _get_full_params_dict(self) -> Dict[str, Tensor]:
        """Get full parameter dict including non-evolved params."""
        params = {}
        for name, p in self.model.named_parameters():
            if name in self.base_params:
                params[name] = self.base_params[name]
            else:
                params[name] = p.detach()
        # Also include buffers
        for name, b in self.model.named_buffers():
            params[name] = b
        return params

    def step(
        self,
        inputs: Tensor,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        targets: Tensor,
    ) -> float:
        """
        Execute one ES optimization step using vmap for parallel evaluation.

        This method uses torch.vmap to evaluate the entire population in parallel,
        which can be significantly faster on GPU/MPS.

        Args:
            inputs: Input tensor (batch_size, ...) - will be broadcast to all population
            loss_fn: Function(outputs, targets) -> scalar loss (lower is better)
            targets: Target tensor for supervised learning

        Returns:
            Mean fitness across the population (for logging).

        Example:
            >>> def mse_loss(outputs, targets):
            ...     return ((outputs - targets) ** 2).mean()
            >>> mean_fitness = es.step(x_batch, mse_loss, y_batch)
        """
        self._refresh_base()
        pop_size = self.pop_size
        device = self.device

        # Generate all noise: Dict[name -> (pop_size, *param_shape)]
        all_noise = self._generate_all_noise()

        # Stack base params and noise for vmap
        # We need params as: Dict[name -> (pop_size, *shape)]
        stacked_params = {}
        for name in self.param_names:
            base = self.base_params[name]
            noise = all_noise[name]
            stacked_params[name] = base.unsqueeze(0) + noise  # (pop_size, *shape)

        # For non-evolved params, just repeat
        for name, p in self.model.named_parameters():
            if name not in stacked_params:
                stacked_params[name] = p.detach().unsqueeze(0).expand(pop_size, *p.shape)

        # Include buffers
        for name, b in self.model.named_buffers():
            stacked_params[name] = b.unsqueeze(0).expand(pop_size, *b.shape)

        # Define single evaluation function
        def single_forward(params_dict: Dict[str, Tensor]) -> Tensor:
            return functional_call(self.model, params_dict, (inputs,))

        # Use vmap to evaluate all population members in parallel
        self.model.eval()
        with torch.no_grad():
            # vmap over the first dimension of each param tensor
            batched_forward = vmap(
                lambda *params: functional_call(
                    self.model,
                    dict(zip(stacked_params.keys(), params)),
                    (inputs,)
                ),
                in_dims=tuple(0 for _ in stacked_params),
            )

            # Get outputs for all population members: (pop_size, batch_size, ...)
            all_outputs = batched_forward(*stacked_params.values())

            # Compute losses for each population member
            # Need to vmap the loss function too
            def compute_loss(outputs: Tensor) -> Tensor:
                return loss_fn(outputs, targets)

            batched_loss = vmap(compute_loss)
            losses = batched_loss(all_outputs)  # (pop_size,)

            # ES maximizes fitness, so negate loss
            fitnesses = -losses

        # Normalize fitnesses
        rewards = fitnesses
        if self.normalize_fitness:
            std = rewards.std()
            if std > 1e-8:
                rewards = (rewards - rewards.mean()) / std
            else:
                rewards = rewards - rewards.mean()
        else:
            rewards = rewards - rewards.mean()

        # ES gradient estimate and update
        with torch.no_grad():
            for name in self.param_names:
                noise = all_noise[name]  # (pop_size, *shape)
                expanded_rewards = rewards.view(pop_size, *([1] * (noise.ndim - 1)))
                grad = (expanded_rewards * noise).mean(dim=0) * (pop_size ** 0.5)
                self.base_params[name] = self.base_params[name] + self.lr * grad

        # Copy updated params back to model
        self._update_model_params()

        self.epoch += 1
        return float(fitnesses.mean().item())

    def _update_model_params(self) -> None:
        """Copy base_params back to the model."""
        for name in self.param_names:
            parts = name.split('.')
            obj = self.model
            for part in parts[:-1]:
                obj = getattr(obj, part)
            param = getattr(obj, parts[-1])
            param.data.copy_(self.base_params[name])
