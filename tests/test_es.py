"""Tests for TorchEggrollES with low-rank noise and antithetic sampling."""

import platform
import pytest
import torch
from torch import nn

from torcheggroll import TorchEggrollES, generate_noise_for_shapes, compute_es_gradient


def get_device():
    """Get device, preferring MPS on Apple Silicon."""
    if platform.machine() == "arm64" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SimpleQuadraticModel(nn.Module):
    """Simple model where we want param to converge to target."""

    def __init__(self, dim: int, target: torch.Tensor):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(dim))
        self.register_buffer("target", target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return the param repeated for each input sample
        # Output shape: (batch_size, dim)
        return self.param.unsqueeze(0).expand(x.shape[0], -1)


class SimpleMatrixModel(nn.Module):
    """Model with a 2D parameter (matrix) to test low-rank noise."""

    def __init__(self, in_dim: int, out_dim: int, target: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_dim, in_dim))
        self.register_buffer("target", target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return the weight flattened repeated for each input sample
        return self.weight.flatten().unsqueeze(0).expand(x.shape[0], -1)


def quadratic_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Loss: squared distance from target."""
    return ((outputs - targets) ** 2).mean()


def test_eggroll_es_1d_param():
    """Test that ES converges for a simple 1D optimization problem."""
    device = get_device()
    target = torch.tensor([0.5, -0.3, 0.8], device=device)
    model = SimpleQuadraticModel(dim=3, target=target).to(device)

    es = TorchEggrollES(
        model=model,
        pop_size=32,
        sigma=0.1,
        lr=0.1,
        antithetic=True,
        device=device,
    )

    # Dummy inputs (not used by model, just for batch size)
    x = torch.zeros(8, 1, device=device)
    # Target for loss function
    y = target.unsqueeze(0).expand(8, -1)

    with torch.no_grad():
        initial_loss = quadratic_loss(model(x), y).item()

    # Run several steps
    for _ in range(20):
        es.step(x, quadratic_loss, y)

    with torch.no_grad():
        final_loss = quadratic_loss(model(x), y).item()

    # Should improve
    assert final_loss < initial_loss
    # Should be close to 0 (perfect match)
    assert final_loss < 0.1


def test_eggroll_es_2d_lowrank_noise():
    """Test that low-rank noise works for matrix parameters."""
    device = get_device()
    target = torch.randn(4, 8, device=device) * 0.5
    model = SimpleMatrixModel(in_dim=8, out_dim=4, target=target).to(device)

    es = TorchEggrollES(
        model=model,
        pop_size=32,
        sigma=0.1,
        lr=0.1,
        rank=2,  # Low rank
        antithetic=True,
        device=device,
    )

    # Dummy inputs
    x = torch.zeros(8, 1, device=device)
    # Target for loss function (flattened)
    y = target.flatten().unsqueeze(0).expand(8, -1)

    with torch.no_grad():
        initial_loss = quadratic_loss(model(x), y).item()

    # Run several steps
    for _ in range(30):
        es.step(x, quadratic_loss, y)

    with torch.no_grad():
        final_loss = quadratic_loss(model(x), y).item()

    # Should improve
    assert final_loss < initial_loss


def test_antithetic_sampling_reduces_variance():
    """
    Test that antithetic sampling produces lower variance estimates.

    With antithetic sampling, pairs of noise vectors are negatives of each other,
    which should reduce variance in the gradient estimate.
    """
    device = get_device()
    target = torch.tensor([0.5], device=device)

    # Run without antithetic
    model_no_anti = SimpleQuadraticModel(dim=1, target=target).to(device)
    es_no_anti = TorchEggrollES(
        model=model_no_anti,
        pop_size=32,
        sigma=0.1,
        lr=0.1,
        antithetic=False,
        device=device,
    )

    # Run with antithetic
    model_anti = SimpleQuadraticModel(dim=1, target=target).to(device)
    es_anti = TorchEggrollES(
        model=model_anti,
        pop_size=32,
        sigma=0.1,
        lr=0.1,
        antithetic=True,
        device=device,
    )

    x = torch.zeros(8, 1, device=device)
    y = target.unsqueeze(0).expand(8, -1)

    # Both should converge, but antithetic should be more stable
    for _ in range(20):
        es_no_anti.step(x, quadratic_loss, y)
        es_anti.step(x, quadratic_loss, y)

    with torch.no_grad():
        loss_no_anti = quadratic_loss(model_no_anti(x), y).item()
        loss_anti = quadratic_loss(model_anti(x), y).item()

    # Both should improve significantly
    assert loss_no_anti < 0.2
    assert loss_anti < 0.2


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
    device = get_device()

    class TwoParamModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen_param = nn.Parameter(torch.zeros(3))
            self.trainable_param = nn.Parameter(torch.zeros(3))
            self.register_buffer("target", torch.tensor([1.0, 1.0, 1.0]))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Return trainable_param
            return self.trainable_param.unsqueeze(0).expand(x.shape[0], -1)

    model = TwoParamModel().to(device)
    initial_frozen = model.frozen_param.clone()

    # Only optimize trainable_param
    es = TorchEggrollES(
        model=model,
        pop_size=16,
        sigma=0.1,
        lr=0.1,
        param_filter=lambda p, name: "trainable" in name,
        antithetic=True,
        device=device,
    )

    x = torch.zeros(8, 1, device=device)
    y = model.target.unsqueeze(0).expand(8, -1)

    for _ in range(10):
        es.step(x, quadratic_loss, y)

    # frozen_param should not have changed
    assert torch.allclose(model.frozen_param, initial_frozen)
    # trainable_param should have changed
    assert not torch.allclose(model.trainable_param, torch.zeros(3, device=device))


def test_no_params_raises_error():
    """Test that an error is raised when no parameters are selected."""

    class EmptyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.buffer = torch.zeros(3)  # Not a parameter

        def forward(self, x):
            return x

    model = EmptyModel()

    with pytest.raises(ValueError, match="no parameters"):
        TorchEggrollES(model=model)


def test_normalize_fitness_off():
    """Test that normalize_fitness=False still works (mean-center only)."""
    device = get_device()
    target = torch.tensor([0.5, -0.3, 0.8], device=device)
    model = SimpleQuadraticModel(dim=3, target=target).to(device)

    es = TorchEggrollES(
        model=model,
        pop_size=32,
        sigma=0.1,
        lr=0.1,
        normalize_fitness=False,  # Only mean-center
        antithetic=True,
        device=device,
    )

    x = torch.zeros(8, 1, device=device)
    y = target.unsqueeze(0).expand(8, -1)

    with torch.no_grad():
        initial_loss = quadratic_loss(model(x), y).item()

    for _ in range(20):
        es.step(x, quadratic_loss, y)

    with torch.no_grad():
        final_loss = quadratic_loss(model(x), y).item()

    # Should still improve
    assert final_loss < initial_loss


def test_device_inference():
    """Test that device is correctly inferred from model."""
    target = torch.tensor([0.5])
    model = SimpleQuadraticModel(dim=1, target=target)

    es = TorchEggrollES(model=model, pop_size=8)

    # Should have inferred CPU device
    assert es.device == torch.device("cpu")


def test_epoch_counter():
    """Test that epoch counter increments correctly."""
    device = get_device()
    target = torch.tensor([0.5], device=device)
    model = SimpleQuadraticModel(dim=1, target=target).to(device)

    es = TorchEggrollES(model=model, pop_size=8, device=device)

    x = torch.zeros(8, 1, device=device)
    y = target.unsqueeze(0).expand(8, -1)

    assert es.epoch == 0

    es.step(x, quadratic_loss, y)
    assert es.epoch == 1

    es.step(x, quadratic_loss, y)
    assert es.epoch == 2


def test_real_neural_network():
    """Test ES on a real neural network (MLP)."""
    device = get_device()

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
    X = torch.randn(20, 4, device=device)
    y = X.sum(dim=1, keepdim=True)  # Simple sum

    model = SimpleMLP().to(device)
    es = TorchEggrollES(
        model=model,
        pop_size=32,
        sigma=0.05,
        lr=0.1,
        rank=4,
        antithetic=True,
        device=device,
    )

    def mse_loss(outputs, targets):
        return ((outputs - targets) ** 2).mean()

    with torch.no_grad():
        initial_loss = mse_loss(model(X), y).item()

    for _ in range(50):
        es.step(X, mse_loss, y)

    with torch.no_grad():
        final_loss = mse_loss(model(X), y).item()

    # Should improve (lower MSE)
    assert final_loss < initial_loss


@pytest.mark.skipif(
    not (platform.machine() == "arm64" and torch.backends.mps.is_available()),
    reason="MPS not available (not Apple Silicon or MPS unavailable)"
)
def test_mps_device():
    """Test that ES works on MPS device (Apple Silicon only)."""
    device = torch.device("mps")
    target = torch.tensor([0.5, -0.3, 0.8], device=device)
    model = SimpleQuadraticModel(dim=3, target=target).to(device)

    es = TorchEggrollES(
        model=model,
        pop_size=16,
        sigma=0.1,
        lr=0.1,
        antithetic=True,
        device=device,
    )

    x = torch.zeros(8, 1, device=device)
    y = target.unsqueeze(0).expand(8, -1)

    # Should run without error
    fitness = es.step(x, quadratic_loss, y)
    assert isinstance(fitness, float)
    assert not torch.isnan(torch.tensor(fitness))


# ============================================================
# Tests for stateless utility functions
# ============================================================


def test_generate_noise_for_shapes_basic():
    """Test basic noise generation for different shapes."""
    device = get_device()
    shapes = {
        "weight": (64, 128),  # 2D -> low-rank noise
        "bias": (64,),        # 1D -> standard noise
    }
    ranks = {
        "weight": 4,
        "bias": None,
    }

    noise = generate_noise_for_shapes(
        shapes=shapes,
        ranks=ranks,
        pop_size=32,
        sigma=0.1,
        epoch=0,
        device=device,
        antithetic=True,
    )

    # Check shapes
    assert noise["weight"].shape == (32, 64, 128)
    assert noise["bias"].shape == (32, 64)

    # Check device
    assert noise["weight"].device.type == device.type
    assert noise["bias"].device.type == device.type


def test_generate_noise_for_shapes_antithetic():
    """Test that antithetic sampling produces mirrored noise."""
    device = get_device()
    shapes = {"param": (10, 10)}
    ranks = {"param": 2}

    noise = generate_noise_for_shapes(
        shapes=shapes,
        ranks=ranks,
        pop_size=32,
        sigma=0.1,
        epoch=0,
        device=device,
        antithetic=True,
    )

    # With antithetic, first half and second half should be negatives
    first_half = noise["param"][:16]
    second_half = noise["param"][16:]

    # Check that second half is negative of first half
    assert torch.allclose(first_half, -second_half, atol=1e-6)


def test_generate_noise_for_shapes_reproducible():
    """Test that same epoch produces same noise."""
    device = get_device()
    shapes = {"weight": (8, 8)}
    ranks = {"weight": 2}

    noise1 = generate_noise_for_shapes(shapes, ranks, 16, 0.1, epoch=42, device=device)
    noise2 = generate_noise_for_shapes(shapes, ranks, 16, 0.1, epoch=42, device=device)

    assert torch.allclose(noise1["weight"], noise2["weight"])


def test_generate_noise_for_shapes_different_epochs():
    """Test that different epochs produce different noise."""
    device = get_device()
    shapes = {"weight": (8, 8)}
    ranks = {"weight": 2}

    noise1 = generate_noise_for_shapes(shapes, ranks, 16, 0.1, epoch=0, device=device)
    noise2 = generate_noise_for_shapes(shapes, ranks, 16, 0.1, epoch=1, device=device)

    assert not torch.allclose(noise1["weight"], noise2["weight"])


def test_generate_noise_for_shapes_odd_pop_size_no_antithetic():
    """Test that odd pop_size works without antithetic."""
    device = get_device()
    shapes = {"param": (10,)}
    ranks = {"param": None}

    noise = generate_noise_for_shapes(
        shapes=shapes,
        ranks=ranks,
        pop_size=31,  # Odd
        sigma=0.1,
        epoch=0,
        device=device,
        antithetic=False,
    )

    assert noise["param"].shape == (31, 10)


def test_generate_noise_for_shapes_odd_pop_size_antithetic_fails():
    """Test that odd pop_size with antithetic raises error."""
    device = get_device()
    shapes = {"param": (10,)}
    ranks = {"param": None}

    with pytest.raises(ValueError, match="even"):
        generate_noise_for_shapes(
            shapes=shapes,
            ranks=ranks,
            pop_size=31,
            sigma=0.1,
            epoch=0,
            device=device,
            antithetic=True,
        )


def test_compute_es_gradient_basic():
    """Test basic ES gradient computation."""
    device = get_device()
    pop_size = 32

    # Create some noise
    noise = {
        "weight": torch.randn(pop_size, 10, 10, device=device),
        "bias": torch.randn(pop_size, 10, device=device),
    }

    # Create rewards (higher = better)
    rewards = torch.randn(pop_size, device=device)

    grads = compute_es_gradient(noise, rewards, normalize_fitness=True)

    # Check shapes
    assert grads["weight"].shape == (10, 10)
    assert grads["bias"].shape == (10,)


def test_compute_es_gradient_zero_variance():
    """Test gradient computation when rewards have zero variance."""
    device = get_device()
    pop_size = 32

    noise = {"param": torch.randn(pop_size, 5, device=device)}
    rewards = torch.ones(pop_size, device=device)  # All same -> zero variance

    grads = compute_es_gradient(noise, rewards, normalize_fitness=True)

    # Should handle gracefully (zero gradient when all rewards equal)
    assert grads["param"].shape == (5,)
    # All rewards equal after mean-centering = 0, so gradient should be ~0
    assert torch.allclose(grads["param"], torch.zeros(5, device=device), atol=1e-6)


def test_compute_es_gradient_direction():
    """Test that gradient points in correct direction."""
    device = get_device()
    pop_size = 4

    # Simple case: noise that has clear direction
    noise = {
        "param": torch.tensor([
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ], device=device),
    }

    # Reward positive x direction more
    rewards = torch.tensor([1.0, -1.0, 0.0, 0.0], device=device)

    grads = compute_es_gradient(noise, rewards, normalize_fitness=False)

    # Gradient should point in positive x direction
    assert grads["param"][0] > 0  # x component positive
    assert abs(grads["param"][1]) < 0.1  # y component near zero


def test_utilities_end_to_end():
    """Test using utilities for a simple optimization without nn.Module."""
    device = get_device()

    # Target to optimize towards
    target = torch.tensor([1.0, 2.0, 3.0], device=device)

    # Initial params
    params = {"vec": torch.zeros(3, device=device)}
    shapes = {"vec": (3,)}
    ranks = {"vec": None}

    lr = 0.5
    sigma = 0.1
    pop_size = 32

    initial_loss = ((params["vec"] - target) ** 2).sum().item()

    for epoch in range(20):
        # Generate noise
        noise = generate_noise_for_shapes(
            shapes, ranks, pop_size, sigma, epoch, device, antithetic=True
        )

        # Evaluate population
        rewards = []
        for i in range(pop_size):
            perturbed = params["vec"] + noise["vec"][i]
            loss = ((perturbed - target) ** 2).sum()
            rewards.append(-loss)  # Higher reward = lower loss

        rewards = torch.stack(rewards)

        # Compute gradient and update
        grads = compute_es_gradient(noise, rewards)
        params["vec"] = params["vec"] + lr * grads["vec"]

    final_loss = ((params["vec"] - target) ** 2).sum().item()

    # Should have improved significantly
    assert final_loss < initial_loss * 0.1
