#!/usr/bin/env python3
"""
Simple test to verify ES works on a character-level task.

This creates a tiny dataset with very predictable patterns
to verify that the ES optimizer can learn something.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torcheggroll import TorchEggrollES


class TinyRNN(nn.Module):
    """Minimal RNN for pattern prediction."""

    def __init__(self, vocab_size: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

        # Small init for ES stability
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) -> (batch, seq_len, vocab_size)"""
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        return self.head(out)


def create_simple_dataset(n_samples: int = 1000, seq_len: int = 16):
    """Create a dataset with simple patterns.

    Pattern: alternating 0,1,0,1,... or 2,3,2,3,...
    The task is to predict the next token.
    """
    data = []
    for _ in range(n_samples):
        # Choose pattern type
        if np.random.rand() < 0.5:
            # Pattern: 0,1,0,1,...
            seq = np.array([i % 2 for i in range(seq_len + 1)], dtype=np.int64)
        else:
            # Pattern: 2,3,2,3,...
            seq = np.array([2 + i % 2 for i in range(seq_len + 1)], dtype=np.int64)
        data.append(seq)

    data = np.array(data)
    inputs = data[:, :-1]  # (n, seq_len)
    targets = data[:, 1:]  # (n, seq_len)
    return torch.from_numpy(inputs), torch.from_numpy(targets)


def compute_loss(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute negative cross-entropy (higher = better) for ES fitness."""
    logits = model(inputs)
    loss = F.cross_entropy(logits.view(-1, model.vocab_size), targets.view(-1))
    return -loss.item()  # Negative because ES maximizes


def compute_accuracy(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute prediction accuracy."""
    with torch.no_grad():
        logits = model(inputs)
        preds = logits.argmax(dim=-1)
        return (preds == targets).float().mean().item()


def main():
    print("=" * 60)
    print("Simple ES Test: Pattern Prediction")
    print("=" * 60)

    # Create dataset
    print("\nCreating dataset...")
    train_inputs, train_targets = create_simple_dataset(1000, seq_len=16)
    test_inputs, test_targets = create_simple_dataset(200, seq_len=16)

    print(f"Train: {train_inputs.shape}")
    print(f"Test: {test_inputs.shape}")
    print(f"Sample sequence: {train_inputs[0].tolist()}")
    print(f"Sample targets:  {train_targets[0].tolist()}")

    # Create model
    model = TinyRNN(vocab_size=4, hidden_dim=32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # Initial accuracy
    init_acc = compute_accuracy(model, test_inputs, test_targets)
    print(f"Initial accuracy: {init_acc:.1%}")

    # Create ES optimizer
    es = TorchEggrollES(
        model=model,
        pop_size=128,
        sigma=0.1,
        lr=0.05,
        rank=4,
        antithetic=True,
        normalize_fitness=True,
    )

    # Training
    print("\nTraining...")
    print("-" * 60)

    n_epochs = 100
    best_acc = 0.0

    for epoch in range(n_epochs):
        start = time.time()

        # Sample batch
        idx = np.random.choice(len(train_inputs), size=64, replace=False)
        batch_in = train_inputs[idx]
        batch_tgt = train_targets[idx]

        # ES step
        def fitness_fn(m):
            return compute_loss(m, batch_in, batch_tgt)

        mean_fitness = es.step(fitness_fn)

        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc = compute_accuracy(model, test_inputs, test_targets)
            best_acc = max(best_acc, acc)
            marker = " *" if acc == best_acc else ""
            elapsed = time.time() - start
            print(f"Epoch {epoch+1:3d} | loss={-mean_fitness:.3f} | acc={acc:.1%}{marker} | {elapsed:.2f}s")
        else:
            elapsed = time.time() - start
            print(f"Epoch {epoch+1:3d} | loss={-mean_fitness:.3f} | {elapsed:.2f}s")

    print("-" * 60)
    print(f"Best accuracy: {best_acc:.1%}")

    # Show sample predictions
    print("\nSample predictions:")
    with torch.no_grad():
        logits = model(test_inputs[:5])
        preds = logits.argmax(dim=-1)
        for i in range(5):
            inp = test_inputs[i].tolist()
            tgt = test_targets[i].tolist()
            pred = preds[i].tolist()
            correct = sum(p == t for p, t in zip(pred, tgt))
            print(f"  Input:  {inp}")
            print(f"  Target: {tgt}")
            print(f"  Pred:   {pred} ({correct}/{len(tgt)} correct)")
            print()


if __name__ == "__main__":
    main()
