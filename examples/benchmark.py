#!/usr/bin/env python3
"""Benchmark comparing step() vs step_vmap() performance."""

import argparse
import time
import torch
import torch.nn as nn

from torcheggroll import TorchEggrollES


class MLP(nn.Module):
    """Simple MLP for benchmarking."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def benchmark_step(model, es, x, y, n_steps: int) -> float:
    """Benchmark the legacy step() method."""
    def eval_fn(m):
        with torch.no_grad():
            loss = ((m(x) - y) ** 2).mean()
            return -loss.item()

    start = time.time()
    for _ in range(n_steps):
        es.step(eval_fn)
    elapsed = time.time() - start
    return elapsed


def benchmark_step_vmap(model, es, x, y, n_steps: int) -> float:
    """Benchmark the vmap step_vmap() method."""
    def mse_loss(outputs, targets):
        return ((outputs - targets) ** 2).mean()

    start = time.time()
    for _ in range(n_steps):
        es.step_vmap(x, mse_loss, y)
    elapsed = time.time() - start
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark ES methods")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--pop-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Pop size: {args.pop_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"N layers: {args.n_layers}")
    print(f"N steps: {args.n_steps}")
    print()

    # Create model and data
    model = MLP(64, args.hidden_dim, 10, args.n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    x = torch.randn(args.batch_size, 64, device=device)
    y = torch.randn(args.batch_size, 10, device=device)

    # Benchmark step()
    print("\n--- Benchmarking step() ---")
    model_step = MLP(64, args.hidden_dim, 10, args.n_layers).to(device)
    es_step = TorchEggrollES(model_step, pop_size=args.pop_size, sigma=0.1, lr=0.1, device=device)

    # Warmup
    for _ in range(args.warmup):
        def eval_fn(m):
            with torch.no_grad():
                return -((m(x) - y) ** 2).mean().item()
        es_step.step(eval_fn)

    time_step = benchmark_step(model_step, es_step, x, y, args.n_steps)
    print(f"step() time: {time_step:.3f}s ({args.n_steps} steps)")
    print(f"step() per step: {time_step/args.n_steps*1000:.1f}ms")

    # Benchmark step_vmap()
    print("\n--- Benchmarking step_vmap() ---")
    model_vmap = MLP(64, args.hidden_dim, 10, args.n_layers).to(device)
    es_vmap = TorchEggrollES(model_vmap, pop_size=args.pop_size, sigma=0.1, lr=0.1, device=device)

    # Warmup
    def mse_loss(outputs, targets):
        return ((outputs - targets) ** 2).mean()
    for _ in range(args.warmup):
        es_vmap.step_vmap(x, mse_loss, y)

    time_vmap = benchmark_step_vmap(model_vmap, es_vmap, x, y, args.n_steps)
    print(f"step_vmap() time: {time_vmap:.3f}s ({args.n_steps} steps)")
    print(f"step_vmap() per step: {time_vmap/args.n_steps*1000:.1f}ms")

    # Comparison
    print("\n--- Comparison ---")
    speedup = time_step / time_vmap
    print(f"Speedup: {speedup:.2f}x")
    if speedup > 1:
        print(f"step_vmap() is {speedup:.2f}x faster")
    else:
        print(f"step() is {1/speedup:.2f}x faster")


if __name__ == "__main__":
    main()
