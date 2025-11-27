# Nano EGG - Language Model Training with Evolution Strategies

This example demonstrates training a byte-level language model using
Evolution Strategies optimization with TorchEggroll.

## Overview

Inspired by the [nano-egg](https://github.com/ESHyperscale/nano-egg) project,
this example trains a minGRU-based model to predict the next byte in text.
Unlike traditional gradient-based training, we use Evolution Strategies (ES)
which only requires forward passes through the model.

## Quick Start

Install dependencies:
```bash
pip install torcheggroll[nano-egg]
# or
uv pip install torcheggroll[nano-egg]
```

Run a quick test (~1-2 minutes):
```bash
python examples/nano_egg/train.py --mode float --epochs 50 \
    --hidden-dim 32 --n-layers 1 --pop-size 512 \
    --max-docs 1000 --lr 0.02 --sigma 0.03
```

Expected output:
```
Epoch    1 | train=8.048 bpb | valid=8.012 bpb
...
Epoch   50 | train=7.335 bpb | valid=7.242 bpb
```

## Baselines

The loss is measured in bits-per-byte (bpb). Lower is better:

| Method | bpb |
|--------|-----|
| Random (log2(256)) | 8.0 |
| Unigram frequency | 5.0 |
| Bigram model | 4.0 |
| gzip compression | 2.77 |

## Models

Two model modes are available:

### Float Mode (`--mode float`)
Standard PyTorch float32 model with:
- minGRU layers (simplified GRU without reset gate)
- Pre-norm architecture with layer normalization
- GELU activation in MLP

This mode works well with TorchEggroll's standard ES optimizer.

### Integer Mode (`--mode int`)
Integer-quantized model with:
- int8 weights and activations
- Fixed-point arithmetic (FIXED_POINT=4)
- Integer approximations for layer norm and activations

Note: Integer mode is experimental. The original nano-egg uses a
specialized Q-EGGROLL algorithm with discrete updates that isn't
fully replicated here.

## Hyperparameter Tuning

Key hyperparameters for ES training:

| Parameter | Description | Suggested Range |
|-----------|-------------|-----------------|
| `--pop-size` | Population size | 256-1024 |
| `--sigma` | Noise scale | 0.01-0.1 |
| `--lr` | Learning rate | 0.01-0.05 |
| `--hidden-dim` | Hidden dimension | 32-128 |
| `--n-layers` | Number of layers | 1-4 |

Tips:
- Larger population sizes give more stable gradients
- Smaller sigma for larger models (avoid destructive noise)
- Lower learning rate for longer training

## Files

- `train.py` - Main training script
- `simple_test.py` - Simple pattern prediction test to verify ES works

## References

- [EGGROLL Paper](https://arxiv.org/abs/2407.05896)
- [nano-egg Implementation](https://github.com/ESHyperscale/nano-egg)
- [minGRU Paper](https://arxiv.org/abs/2410.01201)
