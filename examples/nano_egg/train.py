#!/usr/bin/env python3
"""
Nano EGG - minGRU Language Model trained with Evolution Strategies.

This is a PyTorch implementation inspired by the nano-egg project
(https://github.com/ESHyperscale/nano-egg) which trains a byte-level
language model using Evolution Strategies optimization.

Key features:
- Float or integer-quantized minGRU architecture
- Low-rank noise for ES gradient estimation via TorchEggroll
- Byte-level language modeling on minipile dataset

Quick test (float mode, ~1 min):
    python examples/nano_egg/train.py --mode float --epochs 50 \\
        --hidden-dim 32 --n-layers 1 --pop-size 512 \\
        --max-docs 1000 --lr 0.02 --sigma 0.03

Longer training (float mode, ~10 min):
    python examples/nano_egg/train.py --mode float --epochs 200 \\
        --hidden-dim 64 --n-layers 2 --pop-size 512 \\
        --max-docs 1000 --lr 0.01 --sigma 0.02

Note: The original nano-egg uses a custom Q-EGGROLL algorithm with
discrete int8 updates. This implementation uses standard ES which
works better in float mode. For very long training on full dataset,
use the original JAX implementation.

Baselines for bits-per-byte (bpb):
    - Random (log2(256)): 8.0 bpb
    - Unigram frequency:  5.0 bpb
    - Bigram model:       4.0 bpb
    - gzip compression:   2.77 bpb
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torcheggroll import TorchEggrollES


# ============================================================
# Constants (matching nano-egg)
# ============================================================

FIXED_POINT = 4  # Values are stored as int8 * 2^FIXED_POINT
MAX_INT8 = 127
LOGMAX = 7  # log2(128) - 1


# ============================================================
# Integer Operations
# ============================================================

def clipped_add(*tensors: torch.Tensor) -> torch.Tensor:
    """Add tensors with int32 intermediate and clip back to int8 range."""
    result = sum(t.to(torch.int32) for t in tensors)
    return result.clamp(-MAX_INT8, MAX_INT8).to(torch.int8)


def int_matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Integer matrix multiplication with fixed-point scaling.

    x: (..., in_dim) int8
    weight: (out_dim, in_dim) int8

    Returns: (..., out_dim) int8, scaled by 1/sqrt(in_dim) * 1/16
    """
    # Use int32 for accumulation
    result = torch.matmul(x.to(torch.int32), weight.T.to(torch.int32))
    # Scale down: divide by 2^FIXED_POINT * sqrt(in_dim)
    scale = (2 ** FIXED_POINT) * int(np.sqrt(weight.shape[-1]))
    result = result // scale
    return result.clamp(-MAX_INT8, MAX_INT8).to(torch.int8)


def int_embedding(indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Integer embedding lookup.

    indices: (...) long
    weight: (vocab_size, hidden_dim) int8

    Returns: (..., hidden_dim) int8
    """
    return weight[indices]


# ============================================================
# Integer Layer Norm (EGG-style)
# ============================================================

class IntLayerNorm(nn.Module):
    """
    Integer-only layer normalization.

    Approximates: x * weight / mean(|x|)
    Using integer division lookup table for efficiency.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Weight initialized to 2^FIXED_POINT (= 1.0 in fixed point)
        self.weight = nn.Parameter(
            torch.ones(hidden_dim, dtype=torch.int8) * (2 ** FIXED_POINT)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., hidden_dim) int8
        Returns: (..., hidden_dim) int8
        """
        # Compute mean absolute value
        abs_sum = torch.abs(x.to(torch.int32)).sum(dim=-1, keepdim=True)
        abs_mean = (abs_sum // x.shape[-1]).clamp(min=1)  # Avoid division by zero

        # Scale: x * weight / abs_mean
        # weight is 16x scale, so result is in correct fixed-point
        numerator = x.to(torch.int32) * self.weight.to(torch.int32)
        result = numerator // abs_mean

        return result.clamp(-MAX_INT8, MAX_INT8).to(torch.int8)


# ============================================================
# Integer minGRU
# ============================================================

class IntMinGRU(nn.Module):
    """
    Integer-only minimal GRU (minGRU).

    minGRU is a simplified GRU without the reset gate:
        f_t = sigmoid(W_f @ x + U_f @ h + b_f)
        h_t = f_t * h_{t-1} + (1 - f_t) * tanh(W_h @ x + U_h @ h + b_h)

    In integer mode, sigmoid and tanh are approximated as identity
    (the output is already in [-127, 127] range).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Forget gate parameters
        self.W_f = nn.Parameter(self._init_weight(hidden_dim, hidden_dim))
        self.U_f = nn.Parameter(self._init_weight(hidden_dim, hidden_dim))
        self.b_f = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.int8))

        # Hidden gate parameters
        self.W_h = nn.Parameter(self._init_weight(hidden_dim, hidden_dim))
        self.U_h = nn.Parameter(self._init_weight(hidden_dim, hidden_dim))
        self.b_h = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.int8))

    def _init_weight(self, out_dim: int, in_dim: int) -> torch.Tensor:
        """Initialize weight with scaled normal distribution."""
        w = torch.randn(out_dim, in_dim) * (2 ** FIXED_POINT)
        return w.round().clamp(-MAX_INT8, MAX_INT8).to(torch.int8)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, hidden_dim) int8 - input (in fixed-point)
        state: (batch, hidden_dim) int8 - previous hidden state

        Returns: (output, new_state), both (batch, hidden_dim) int8
        """
        # Forget gate: f_t = Wf @ x + Uf @ state + bf
        f_t = clipped_add(
            int_matmul(x, self.W_f),
            int_matmul(state, self.U_f),
            self.b_f
        )

        # Gate computation in int32
        # f_t is in [-127, 127], we map to [0, 254] for gating
        f_gate = f_t.to(torch.int32) + MAX_INT8  # [0, 254]

        # Gated past: (f + 127) * state / 254 ≈ sigmoid(f) * state
        gated_past = (f_gate * state.to(torch.int32)) >> (LOGMAX + 1)
        gated_past = gated_past.clamp(-MAX_INT8, MAX_INT8).to(torch.int8)

        # Candidate hidden: h_tilde = Wh @ x + Uh @ gated_past + bh
        h_tilde = clipped_add(
            int_matmul(x, self.W_h),
            int_matmul(gated_past, self.U_h),
            self.b_h
        )

        # New state: h_t = state + f_gate * (h_tilde - state) / 254
        # This is: state + (1 - sigmoid(-f)) * (h_tilde - state) ≈ lerp
        diff = h_tilde.to(torch.int32) - state.to(torch.int32)
        new_state = state.to(torch.int32) + ((f_gate * diff) >> (LOGMAX + 1))
        new_state = new_state.clamp(-MAX_INT8, MAX_INT8).to(torch.int8)

        return new_state, new_state


# ============================================================
# Integer MLP
# ============================================================

class IntMLP(nn.Module):
    """
    Integer-only MLP with one hidden layer.

    Architecture: hidden_dim -> hidden_dim*4 -> hidden_dim
    Note: ReLU is applied but commented out in nano-egg
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        mlp_dim = hidden_dim * 4

        self.w1 = nn.Parameter(self._init_weight(mlp_dim, hidden_dim))
        self.w2 = nn.Parameter(self._init_weight(hidden_dim, mlp_dim))

    def _init_weight(self, out_dim: int, in_dim: int) -> torch.Tensor:
        w = torch.randn(out_dim, in_dim) * (2 ** FIXED_POINT)
        return w.round().clamp(-MAX_INT8, MAX_INT8).to(torch.int8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., hidden_dim) int8 -> (..., hidden_dim) int8"""
        h = int_matmul(x, self.w1)
        # ReLU (optional, nano-egg comments it out)
        # h = h.clamp(min=0)
        out = int_matmul(h, self.w2)
        return out


# ============================================================
# Transformer Block (EGG Layer)
# ============================================================

class IntEGGLayer(nn.Module):
    """
    Single EGG layer: LayerNorm -> minGRU -> residual -> LayerNorm -> MLP -> residual
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.ln1 = IntLayerNorm(hidden_dim)
        self.gru = IntMinGRU(hidden_dim)
        self.ln2 = IntLayerNorm(hidden_dim)
        self.mlp = IntMLP(hidden_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, hidden_dim) int8
        state: (batch, hidden_dim) int8

        Returns: (output, new_state)
        """
        # Pre-norm attention/GRU
        residual = x
        x = self.ln1(x)
        x, state = self.gru(x, state)
        x = clipped_add(x, residual)

        # Pre-norm MLP
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = clipped_add(x, residual)

        return x, state


# ============================================================
# Full Model (EGG)
# ============================================================

class IntEGG(nn.Module):
    """
    Integer-only EGG model for byte-level language modeling.

    Architecture:
    - Byte embedding (256 vocab)
    - N layers of EGG blocks
    - Output projection to vocab
    """

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 256,
        n_layers: int = 6,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Embedding
        emb_weight = torch.randn(vocab_size, hidden_dim) * (2 ** FIXED_POINT)
        self.embedding = nn.Parameter(
            emb_weight.round().clamp(-MAX_INT8, MAX_INT8).to(torch.int8)
        )

        # Layers
        self.layers = nn.ModuleList([
            IntEGGLayer(hidden_dim) for _ in range(n_layers)
        ])

        # Output
        self.ln_out = IntLayerNorm(hidden_dim)
        head_weight = torch.randn(vocab_size, hidden_dim) * (2 ** FIXED_POINT)
        self.head = nn.Parameter(
            head_weight.round().clamp(-MAX_INT8, MAX_INT8).to(torch.int8)
        )

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden states for all layers."""
        return torch.zeros(
            self.n_layers, batch_size, self.hidden_dim,
            dtype=torch.int8, device=device
        )

    def forward(
        self,
        tokens: torch.Tensor,
        states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single timestep.

        tokens: (batch,) long - input token indices
        states: (n_layers, batch, hidden_dim) int8 - hidden states

        Returns:
            logits: (batch, vocab_size) float32 - output logits
            new_states: (n_layers, batch, hidden_dim) int8
        """
        batch_size = tokens.shape[0]
        device = tokens.device

        if states is None:
            states = self.init_state(batch_size, device)

        # Embedding lookup
        x = int_embedding(tokens, self.embedding)  # (batch, hidden_dim)

        # Process through layers
        new_states = []
        for i, layer in enumerate(self.layers):
            x, new_state = layer(x, states[i])
            new_states.append(new_state)

        new_states = torch.stack(new_states, dim=0)

        # Output projection
        x = self.ln_out(x)
        logits = int_matmul(x, self.head)

        # Convert to float for softmax/loss
        logits = logits.to(torch.float32)

        return logits, new_states

    def forward_sequence(
        self,
        tokens: torch.Tensor,
        states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a sequence.

        tokens: (batch, seq_len) long
        states: (n_layers, batch, hidden_dim) int8

        Returns:
            all_logits: (batch, seq_len, vocab_size) float32
            final_states: (n_layers, batch, hidden_dim) int8
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device

        if states is None:
            states = self.init_state(batch_size, device)

        all_logits = []
        for t in range(seq_len):
            logits, states = self.forward(tokens[:, t], states)
            all_logits.append(logits)

        all_logits = torch.stack(all_logits, dim=1)  # (batch, seq_len, vocab)
        return all_logits, states


# ============================================================
# Simple Float minGRU (for verification)
# ============================================================

class FloatMinGRU(nn.Module):
    """
    Standard float minGRU for verification that training works.

    This is a simple float implementation without integer quantization.
    Used to verify that ES training converges before debugging integer math.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 256,
        n_layers: int = 6,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(hidden_dim, elementwise_affine=False),
                'W_f': nn.Linear(hidden_dim, hidden_dim, bias=False),
                'U_f': nn.Linear(hidden_dim, hidden_dim, bias=True),
                'W_h': nn.Linear(hidden_dim, hidden_dim, bias=False),
                'U_h': nn.Linear(hidden_dim, hidden_dim, bias=True),
                'ln2': nn.LayerNorm(hidden_dim, elementwise_affine=False),
                'mlp_up': nn.Linear(hidden_dim, hidden_dim * 4, bias=False),
                'mlp_down': nn.Linear(hidden_dim * 4, hidden_dim, bias=False),
            }))

        # Output
        self.ln_out = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Initialize with small weights for stability
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=0.02)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)

    def forward_sequence(
        self,
        tokens: torch.Tensor,
        states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        tokens: (batch, seq_len) long
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device

        if states is None:
            states = self.init_state(batch_size, device)

        all_logits = []

        for t in range(seq_len):
            # Embedding
            x = self.embedding(tokens[:, t])  # (batch, hidden_dim)

            # Process through layers
            new_states = []
            for i, layer in enumerate(self.layers):
                state = states[i]

                # Pre-norm GRU
                residual = x
                x_norm = layer['ln1'](x)

                # minGRU step
                f = torch.sigmoid(layer['W_f'](x_norm) + layer['U_f'](state))
                h_tilde = torch.tanh(layer['W_h'](x_norm) + layer['U_h'](state * f))
                new_state = f * state + (1 - f) * h_tilde

                x = residual + new_state
                new_states.append(new_state)

                # Pre-norm MLP
                residual = x
                x_norm = layer['ln2'](x)
                x = residual + layer['mlp_down'](F.gelu(layer['mlp_up'](x_norm)))

            states = torch.stack(new_states, dim=0)

            # Output
            logits = self.head(self.ln_out(x))
            all_logits.append(logits)

        all_logits = torch.stack(all_logits, dim=1)
        return all_logits, states


# ============================================================
# Float Wrapper for ES (converts int8 params to float for perturbation)
# ============================================================

class EGGFloat(nn.Module):
    """
    Float wrapper around IntEGG for use with TorchEggrollES.

    TorchEggrollES works with float parameters, so we maintain float
    parameters and convert to int8 during forward pass.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 256,
        n_layers: int = 6,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Float parameters (will be quantized to int8 during forward)
        scale = 2 ** FIXED_POINT

        # Embedding
        self.embedding = nn.Parameter(torch.randn(vocab_size, hidden_dim) * scale)

        # Layers (flattened for ES)
        self.ln1_weights = nn.ParameterList([
            nn.Parameter(torch.ones(hidden_dim) * scale) for _ in range(n_layers)
        ])
        self.ln2_weights = nn.ParameterList([
            nn.Parameter(torch.ones(hidden_dim) * scale) for _ in range(n_layers)
        ])

        # GRU parameters per layer
        self.W_f = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim, hidden_dim) * scale) for _ in range(n_layers)
        ])
        self.U_f = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim, hidden_dim) * scale) for _ in range(n_layers)
        ])
        self.b_f = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_dim)) for _ in range(n_layers)
        ])
        self.W_h = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim, hidden_dim) * scale) for _ in range(n_layers)
        ])
        self.U_h = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim, hidden_dim) * scale) for _ in range(n_layers)
        ])
        self.b_h = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_dim)) for _ in range(n_layers)
        ])

        # MLP parameters per layer
        self.mlp_w1 = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim * 4, hidden_dim) * scale) for _ in range(n_layers)
        ])
        self.mlp_w2 = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim, hidden_dim * 4) * scale) for _ in range(n_layers)
        ])

        # Output
        self.ln_out_weight = nn.Parameter(torch.ones(hidden_dim) * scale)
        self.head = nn.Parameter(torch.randn(vocab_size, hidden_dim) * scale)

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize float to int8."""
        return x.round().clamp(-MAX_INT8, MAX_INT8).to(torch.int8)

    def _int_ln(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Integer layer norm with float weight."""
        w = self._quantize(weight)
        abs_sum = torch.abs(x.to(torch.int32)).sum(dim=-1, keepdim=True)
        abs_mean = (abs_sum // x.shape[-1]).clamp(min=1)
        numerator = x.to(torch.int32) * w.to(torch.int32)
        result = numerator // abs_mean
        return result.clamp(-MAX_INT8, MAX_INT8).to(torch.int8)

    def _int_matmul(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Integer matmul with float weight."""
        w = self._quantize(weight)
        result = torch.matmul(x.to(torch.int32), w.T.to(torch.int32))
        scale = (2 ** FIXED_POINT) * int(np.sqrt(w.shape[-1]))
        result = result // scale
        return result.clamp(-MAX_INT8, MAX_INT8).to(torch.int8)

    def _gru_step(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        W_f: torch.Tensor,
        U_f: torch.Tensor,
        b_f: torch.Tensor,
        W_h: torch.Tensor,
        U_h: torch.Tensor,
        b_h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One GRU step with float parameters."""
        # Quantize parameters
        Wf = self._quantize(W_f)
        Uf = self._quantize(U_f)
        bf = self._quantize(b_f)
        Wh = self._quantize(W_h)
        Uh = self._quantize(U_h)
        bh = self._quantize(b_h)

        # Forget gate
        f_t = clipped_add(
            self._int_matmul(x, W_f),
            self._int_matmul(state, U_f),
            bf
        )

        f_gate = f_t.to(torch.int32) + MAX_INT8
        gated_past = (f_gate * state.to(torch.int32)) >> (LOGMAX + 1)
        gated_past = gated_past.clamp(-MAX_INT8, MAX_INT8).to(torch.int8)

        h_tilde = clipped_add(
            self._int_matmul(x, W_h),
            self._int_matmul(gated_past, U_h),
            bh
        )

        diff = h_tilde.to(torch.int32) - state.to(torch.int32)
        new_state = state.to(torch.int32) + ((f_gate * diff) >> (LOGMAX + 1))
        new_state = new_state.clamp(-MAX_INT8, MAX_INT8).to(torch.int8)

        return new_state, new_state

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            self.n_layers, batch_size, self.hidden_dim,
            dtype=torch.int8, device=device
        )

    def forward_sequence(
        self,
        tokens: torch.Tensor,
        states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training (returns loss-ready logits).

        tokens: (batch, seq_len) long
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device

        if states is None:
            states = self.init_state(batch_size, device)

        # Quantize embedding
        emb = self._quantize(self.embedding)

        all_logits = []
        for t in range(seq_len):
            # Embedding
            x = emb[tokens[:, t]]  # (batch, hidden_dim) int8

            # Reset state on padding/start token (token 0)
            mask = (tokens[:, t] == 0).unsqueeze(0).unsqueeze(-1)
            states = torch.where(mask, torch.zeros_like(states), states)

            # Process through layers
            new_states = []
            for i in range(self.n_layers):
                # Pre-norm GRU
                residual = x
                x = self._int_ln(x, self.ln1_weights[i])
                x, new_state = self._gru_step(
                    x, states[i],
                    self.W_f[i], self.U_f[i], self.b_f[i],
                    self.W_h[i], self.U_h[i], self.b_h[i]
                )
                x = clipped_add(x, residual)

                # Pre-norm MLP
                residual = x
                x = self._int_ln(x, self.ln2_weights[i])
                h = self._int_matmul(x, self.mlp_w1[i])
                x = self._int_matmul(h, self.mlp_w2[i])
                x = clipped_add(x, residual)

                new_states.append(new_state)

            states = torch.stack(new_states, dim=0)

            # Output
            x = self._int_ln(x, self.ln_out_weight)
            logits = self._int_matmul(x, self.head)
            all_logits.append(logits.to(torch.float32))

        all_logits = torch.stack(all_logits, dim=1)
        return all_logits, states


# ============================================================
# Dataset Loading
# ============================================================

def download_minipile(data_dir: Path, max_train_docs: int = 0, max_valid_docs: int = 0) -> None:
    """Download and preprocess minipile dataset.

    Args:
        data_dir: Directory to save the data
        max_train_docs: Max training documents (0 = all)
        max_valid_docs: Max validation documents (0 = all)
    """
    from datasets import load_dataset
    from tqdm import tqdm

    data_dir.mkdir(parents=True, exist_ok=True)

    # Use suffix if limited
    suffix = ""
    if max_train_docs > 0:
        suffix = f"_{max_train_docs}"

    train_path = data_dir / f"minipile_train{suffix}.npy"
    valid_path = data_dir / f"minipile_valid{suffix}.npy"

    if train_path.exists() and valid_path.exists():
        print("Dataset already downloaded")
        return train_path, valid_path

    print("Downloading minipile dataset...")
    ds = load_dataset("JeanKaddour/minipile")

    splits_to_process = []
    max_docs = {"train": max_train_docs, "validation": max_valid_docs}
    if not train_path.exists():
        splits_to_process.append(("train", train_path))
    if not valid_path.exists():
        splits_to_process.append(("validation", valid_path))

    for split, path in splits_to_process:
        print(f"Processing {split}...")
        arrays = []
        texts = ds[split]["text"]
        limit = max_docs[split] if max_docs[split] > 0 else len(texts)
        for i, text in enumerate(tqdm(texts, desc=f"Encoding {split}", total=min(limit, len(texts)))):
            if i >= limit:
                break
            # Prepend 0 (start token) to each document
            arrays.append(np.array([0] + list(text.encode("utf-8")), dtype=np.uint8))

        print(f"Concatenating {len(arrays)} documents...")
        out_array = np.concatenate(arrays)
        print(f"  {split}: {len(out_array):,} bytes")
        np.save(path, out_array)

    print("Dataset ready!")
    return train_path, valid_path


def load_dataset_np(data_dir: Path, split: str = "train", max_docs: int = 0) -> np.ndarray:
    """Load preprocessed dataset."""
    suffix = f"_{max_docs}" if max_docs > 0 else ""
    # Map split name to file name (validation -> valid)
    file_split = "valid" if split == "validation" else split
    path = data_dir / f"minipile_{file_split}{suffix}.npy"
    if not path.exists():
        download_minipile(data_dir, max_train_docs=max_docs, max_valid_docs=max_docs)
    return np.load(path)


# ============================================================
# Training
# ============================================================

def compute_loss(
    model: nn.Module,
    input_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
) -> float:
    """
    Compute cross-entropy loss in bits per byte.

    Returns negative loss (higher = better) for ES fitness.
    """
    logits, _ = model.forward_sequence(input_tokens)

    # Cross-entropy loss
    vocab_size = logits.shape[-1]
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = target_tokens.view(-1).long()

    # Use log_softmax for numerical stability
    log_probs = F.log_softmax(logits_flat, dim=-1)
    loss = F.nll_loss(log_probs, targets_flat, reduction="mean")

    # Convert to bits (log base 2)
    bits_per_byte = loss.item() / np.log(2)

    return -bits_per_byte  # Negative because ES maximizes fitness


def main():
    parser = argparse.ArgumentParser(description="Nano EGG - Integer ES training")

    # Model args
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--mode", type=str, default="int", choices=["int", "float"],
                        help="Model mode: int (integer quantized) or float")

    # Training args
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--pop-size", type=int, default=512, help="ES population size")
    parser.add_argument("--sigma", type=float, default=0.5, help="ES noise scale")
    parser.add_argument("--lr", type=float, default=0.1, help="ES learning rate")
    parser.add_argument("--rank", type=int, default=4, help="Low-rank noise rank")

    # Data args
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per fitness eval")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--max-docs", type=int, default=0, help="Max docs to load (0 = all, 10000 for quick test)")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--validate-every", type=int, default=10, help="Validate every N epochs")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("Nano EGG - Integer-only minGRU with Evolution Strategies")
    print("=" * 60)

    # Load data
    print("\nLoading dataset...")
    train_data = load_dataset_np(data_dir, "train", max_docs=args.max_docs)
    valid_data = load_dataset_np(data_dir, "validation", max_docs=args.max_docs)
    print(f"Train: {len(train_data):,} bytes")
    print(f"Valid: {len(valid_data):,} bytes")

    # Create model
    print(f"\nCreating model (hidden={args.hidden_dim}, layers={args.n_layers}, mode={args.mode})...")
    if args.mode == "float":
        model = FloatMinGRU(
            vocab_size=256,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
        ).to(device)
    else:
        model = EGGFloat(
            vocab_size=256,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Create ES optimizer
    print(f"\nCreating TorchEggrollES (pop={args.pop_size}, σ={args.sigma}, lr={args.lr})...")
    es = TorchEggrollES(
        model=model,
        pop_size=args.pop_size,
        sigma=args.sigma,
        lr=args.lr,
        rank=args.rank,
        antithetic=True,
        normalize_fitness=True,
    )

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)
    print("Baselines: unigram=5.0 bits, bigram=4.0 bits, gzip=2.77 bits")
    print("-" * 60)

    # Precompute number of training sequences
    n_train_seqs = (len(train_data) - 1) // args.seq_len

    best_valid_bpb = float("inf")

    for epoch in range(args.epochs):
        start_time = time.time()

        # Sample batch indices for this epoch
        batch_starts = np.random.randint(0, len(train_data) - args.seq_len - 1, size=args.batch_size)

        # Create batch tensors
        input_batch = np.stack([train_data[i:i+args.seq_len] for i in batch_starts])
        target_batch = np.stack([train_data[i+1:i+args.seq_len+1] for i in batch_starts])

        input_tokens = torch.from_numpy(input_batch).long().to(device)
        target_tokens = torch.from_numpy(target_batch).long().to(device)

        # Define fitness function
        def fitness_fn(m: nn.Module) -> float:
            return compute_loss(m, input_tokens, target_tokens)

        # ES step
        mean_fitness = es.step(fitness_fn)
        train_bpb = -mean_fitness

        elapsed = time.time() - start_time
        throughput = args.batch_size * args.seq_len * args.pop_size / elapsed

        # Validation
        if (epoch + 1) % args.validate_every == 0 or epoch == 0:
            # Sample validation batch
            valid_starts = np.random.randint(0, len(valid_data) - args.seq_len - 1, size=args.batch_size)
            valid_input = np.stack([valid_data[i:i+args.seq_len] for i in valid_starts])
            valid_target = np.stack([valid_data[i+1:i+args.seq_len+1] for i in valid_starts])

            valid_input_t = torch.from_numpy(valid_input).long().to(device)
            valid_target_t = torch.from_numpy(valid_target).long().to(device)

            with torch.no_grad():
                valid_bpb = -compute_loss(model, valid_input_t, valid_target_t)

            if valid_bpb < best_valid_bpb:
                best_valid_bpb = valid_bpb
                marker = " *"
            else:
                marker = ""

            print(
                f"Epoch {epoch+1:4d} | "
                f"train={train_bpb:.3f} bpb | "
                f"valid={valid_bpb:.3f} bpb{marker} | "
                f"{throughput/1e6:.2f}M tok/s | "
                f"{elapsed:.1f}s"
            )
        else:
            print(
                f"Epoch {epoch+1:4d} | "
                f"train={train_bpb:.3f} bpb | "
                f"{throughput/1e6:.2f}M tok/s | "
                f"{elapsed:.1f}s"
            )

    print("-" * 60)
    print(f"Best validation: {best_valid_bpb:.3f} bits per byte")
    print("Done!")


if __name__ == "__main__":
    main()
