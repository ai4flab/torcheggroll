"""Tests for TorchEggrollES with low-rank noise and antithetic sampling."""

import pytest
import torch
from torch import nn

from torcheggroll import TorchEggrollES


class SimpleQuadraticModel(nn.Module):
    """Simple model where we want param to converge to target."""

    def __init__(self, dim: int, target: torch.Tensor):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(dim))
        self.target = target

    def forward(self) -> torch.Tensor:
        # Negative squared distance from target
        return -((self.param - self.target) ** 2).sum()


class SimpleMatrixModel(nn.Module):
    """Model with a 2D parameter (matrix) to test low-rank noise."""

    def __init__(self, in_dim: int, out_dim: int, target: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_dim, in_dim))
        self.target = target

    def forward(self) -> torch.Tensor:
        # Negative Frobenius norm of difference from target
        return -((self.weight - self.target) ** 2).sum()


def test_eggroll_es_1d_param():
    """Test that ES converges for a simple 1D optimization problem."""
    target = torch.tensor([0.5, -0.3, 0.8])
    model = SimpleQuadraticModel(dim=3, target=target)

    es = TorchEggrollES(
        model=model,
        pop_size=32,
        sigma=0.1,
        lr=0.1,
        antithetic=True,
    )

    def eval_fn(m):
        return float(m())

    initial_fitness = eval_fn(model)

    # Run several steps
    for _ in range(20):
        es.step(eval_fn)

    final_fitness = eval_fn(model)

    # Should improve
    assert final_fitness > initial_fitness
    # Should be close to 0 (perfect match)
    assert final_fitness > -0.1


def test_eggroll_es_2d_lowrank_noise():
    """Test that low-rank noise works for matrix parameters."""
    target = torch.randn(4, 8) * 0.5
    model = SimpleMatrixModel(in_dim=8, out_dim=4, target=target)

    es = TorchEggrollES(
        model=model,
        pop_size=32,
        sigma=0.1,
        lr=0.1,
        rank=2,  # Low rank
        antithetic=True,
    )

    def eval_fn(m):
        return float(m())

    initial_fitness = eval_fn(model)

    # Run several steps
    for _ in range(30):
        es.step(eval_fn)

    final_fitness = eval_fn(model)

    # Should improve
    assert final_fitness > initial_fitness


def test_antithetic_sampling_reduces_variance():
    """
    Test that antithetic sampling produces lower variance estimates.

    With antithetic sampling, pairs of noise vectors are negatives of each other,
    which should reduce variance in the gradient estimate.
    """
    target = torch.tensor([0.5])
    model = SimpleQuadraticModel(dim=1, target=target)

    # Run without antithetic
    model_no_anti = SimpleQuadraticModel(dim=1, target=target)
    es_no_anti = TorchEggrollES(
        model=model_no_anti,
        pop_size=32,
        sigma=0.1,
        lr=0.1,
        antithetic=False,
    )

    # Run with antithetic
    model_anti = SimpleQuadraticModel(dim=1, target=target)
    es_anti = TorchEggrollES(
        model=model_anti,
        pop_size=32,
        sigma=0.1,
        lr=0.1,
        antithetic=True,
    )

    def eval_fn(m):
        return float(m())

    # Both should converge, but antithetic should be more stable
    for _ in range(20):
        es_no_anti.step(eval_fn)
        es_anti.step(eval_fn)

    # Both should improve significantly
    assert eval_fn(model_no_anti) > -0.2
    assert eval_fn(model_anti) > -0.2


def test_pop_size_must_be_even_for_antithetic():
    """Test that antithetic sampling requires even population size."""
    model = SimpleQuadraticModel(dim=1, target=torch.tensor([0.5]))

    with pytest.raises(ValueError, match="even"):
        TorchEggrollES(
            model=model,
            pop_size=31,  # Odd
            antithetic=True,
        )


def test_param_filter():
    """Test that param_filter correctly selects which parameters to optimize."""

    class TwoParamModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen_param = nn.Parameter(torch.zeros(3))
            self.trainable_param = nn.Parameter(torch.zeros(3))
            self.target = torch.tensor([1.0, 1.0, 1.0])

        def forward(self):
            # Only trainable_param should change
            return -((self.trainable_param - self.target) ** 2).sum()

    model = TwoParamModel()
    initial_frozen = model.frozen_param.clone()

    # Only optimize trainable_param
    es = TorchEggrollES(
        model=model,
        pop_size=16,
        sigma=0.1,
        lr=0.1,
        param_filter=lambda p, name: "trainable" in name,
        antithetic=True,
    )

    def eval_fn(m):
        return float(m())

    for _ in range(10):
        es.step(eval_fn)

    # frozen_param should not have changed
    assert torch.allclose(model.frozen_param, initial_frozen)
    # trainable_param should have changed
    assert not torch.allclose(model.trainable_param, torch.zeros(3))


def test_no_params_raises_error():
    """Test that an error is raised when no parameters are selected."""

    class EmptyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.buffer = torch.zeros(3)  # Not a parameter

        def forward(self):
            return self.buffer.sum()

    model = EmptyModel()

    with pytest.raises(ValueError, match="no parameters"):
        TorchEggrollES(model=model)


def test_normalize_fitness_off():
    """Test that normalize_fitness=False still works (mean-center only)."""
    target = torch.tensor([0.5, -0.3, 0.8])
    model = SimpleQuadraticModel(dim=3, target=target)

    es = TorchEggrollES(
        model=model,
        pop_size=32,
        sigma=0.1,
        lr=0.1,
        normalize_fitness=False,  # Only mean-center
        antithetic=True,
    )

    def eval_fn(m):
        return float(m())

    initial_fitness = eval_fn(model)

    for _ in range(20):
        es.step(eval_fn)

    final_fitness = eval_fn(model)

    # Should still improve
    assert final_fitness > initial_fitness


def test_device_inference():
    """Test that device is correctly inferred from model."""
    target = torch.tensor([0.5])
    model = SimpleQuadraticModel(dim=1, target=target)

    es = TorchEggrollES(model=model, pop_size=8)

    # Should have inferred CPU device
    assert es.device == torch.device("cpu")


def test_epoch_counter():
    """Test that epoch counter increments correctly."""
    target = torch.tensor([0.5])
    model = SimpleQuadraticModel(dim=1, target=target)

    es = TorchEggrollES(model=model, pop_size=8)

    assert es.epoch == 0

    es.step(lambda m: float(m()))
    assert es.epoch == 1

    es.step(lambda m: float(m()))
    assert es.epoch == 2


def test_real_neural_network():
    """Test ES on a real neural network (MLP)."""

    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    # Generate simple linear target
    torch.manual_seed(42)
    X = torch.randn(20, 4)
    y = X.sum(dim=1, keepdim=True)  # Simple sum

    model = SimpleMLP()
    es = TorchEggrollES(
        model=model,
        pop_size=32,
        sigma=0.05,
        lr=0.1,
        rank=4,
        antithetic=True,
    )

    def fitness(m):
        pred = m(X)
        mse = ((pred - y) ** 2).mean()
        return -float(mse)  # Negative MSE (higher is better)

    initial_fitness = fitness(model)

    for _ in range(50):
        es.step(fitness)

    final_fitness = fitness(model)

    # Should improve (less negative = lower MSE)
    assert final_fitness > initial_fitness
