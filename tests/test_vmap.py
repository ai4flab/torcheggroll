"""Tests for vmap-based ES evaluation."""

import platform
import pytest
import torch
import torch.nn as nn

from torcheggroll import TorchEggrollES


def get_device():
    """Get device, preferring MPS on Apple Silicon."""
    if platform.machine() == "arm64" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def test_step_basic():
    """Test that step runs without errors."""
    device = get_device()
    model = SimpleModel().to(device)
    es = TorchEggrollES(model, pop_size=8, sigma=0.1, lr=0.1, device=device)

    x = torch.randn(16, 10, device=device)  # batch of 16 samples
    y = torch.randn(16, 5, device=device)   # targets

    def mse_loss(outputs, targets):
        return ((outputs - targets) ** 2).mean()

    # Should run without error
    fitness = es.step(x, mse_loss, y)
    assert isinstance(fitness, float)


def test_step_improves_loss():
    """Test that step actually improves the model."""
    torch.manual_seed(42)
    device = get_device()

    model = SimpleModel().to(device)
    es = TorchEggrollES(model, pop_size=32, sigma=0.1, lr=0.1, device=device)

    # Create simple regression data
    x = torch.randn(32, 10, device=device)
    y = x[:, :5] * 0.5  # Simple linear relationship

    def mse_loss(outputs, targets):
        return ((outputs - targets) ** 2).mean()

    # Get initial loss
    with torch.no_grad():
        initial_loss = mse_loss(model(x), y).item()

    # Run several steps
    for _ in range(20):
        es.step(x, mse_loss, y)

    # Get final loss
    with torch.no_grad():
        final_loss = mse_loss(model(x), y).item()

    # Loss should decrease
    assert final_loss < initial_loss, f"Loss should decrease: {initial_loss} -> {final_loss}"


def test_step_with_different_batch_sizes():
    """Test step with various batch sizes."""
    device = get_device()
    model = SimpleModel().to(device)
    es = TorchEggrollES(model, pop_size=8, sigma=0.1, lr=0.1, device=device)

    def mse_loss(outputs, targets):
        return ((outputs - targets) ** 2).mean()

    for batch_size in [1, 4, 16, 64]:
        x = torch.randn(batch_size, 10, device=device)
        y = torch.randn(batch_size, 5, device=device)

        fitness = es.step(x, mse_loss, y)
        assert isinstance(fitness, float), f"Failed for batch_size={batch_size}"


def test_step_epoch_increments():
    """Test that epoch counter increments with step."""
    device = get_device()
    model = SimpleModel().to(device)
    es = TorchEggrollES(model, pop_size=8, sigma=0.1, lr=0.1, device=device)

    x = torch.randn(8, 10, device=device)
    y = torch.randn(8, 5, device=device)

    def mse_loss(outputs, targets):
        return ((outputs - targets) ** 2).mean()

    assert es.epoch == 0
    es.step(x, mse_loss, y)
    assert es.epoch == 1
    es.step(x, mse_loss, y)
    assert es.epoch == 2


def test_step_antithetic_sampling():
    """Test that antithetic sampling is being used correctly."""
    device = get_device()
    model = SimpleModel().to(device)

    # Test with antithetic=True (default)
    es = TorchEggrollES(model, pop_size=8, sigma=0.1, lr=0.1, antithetic=True, device=device)

    x = torch.randn(8, 10, device=device)
    y = torch.randn(8, 5, device=device)

    def mse_loss(outputs, targets):
        return ((outputs - targets) ** 2).mean()

    fitness = es.step(x, mse_loss, y)
    assert isinstance(fitness, float)


def test_step_no_antithetic():
    """Test that non-antithetic mode works."""
    device = get_device()
    model = SimpleModel().to(device)

    # Test with antithetic=False
    es = TorchEggrollES(model, pop_size=8, sigma=0.1, lr=0.1, antithetic=False, device=device)

    x = torch.randn(8, 10, device=device)
    y = torch.randn(8, 5, device=device)

    def mse_loss(outputs, targets):
        return ((outputs - targets) ** 2).mean()

    fitness = es.step(x, mse_loss, y)
    assert isinstance(fitness, float)


@pytest.mark.skipif(
    not (platform.machine() == "arm64" and torch.backends.mps.is_available()),
    reason="MPS not available (not Apple Silicon or MPS unavailable)"
)
def test_step_on_mps():
    """Test that step works on MPS device (Apple Silicon only)."""
    device = torch.device("mps")
    model = SimpleModel().to(device)
    es = TorchEggrollES(model, pop_size=8, sigma=0.1, lr=0.1, device=device)

    x = torch.randn(16, 10, device=device)
    y = torch.randn(16, 5, device=device)

    def mse_loss(outputs, targets):
        return ((outputs - targets) ** 2).mean()

    # Should run without error on MPS
    fitness = es.step(x, mse_loss, y)
    assert isinstance(fitness, float)
    assert not torch.isnan(torch.tensor(fitness))
