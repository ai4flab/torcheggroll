# TorchEggroll - Full HyperscaleES Implementation Plan

Goal: Make TorchEggroll a complete, performant PyTorch port of HyperscaleES with MPS support.

## Current State

- ✅ Basic EGGROLL algorithm (low-rank noise, antithetic sampling)
- ✅ Works on CPU
- ✅ Vectorized population evaluation with torch.vmap (Phase 1 complete)
- ✅ MPS/GPU support via vmap (tested on Apple Silicon)
- ❌ Missing pre-generated noise matrix (BIG_RAND_MATRIX)
- ❌ Missing Q-EGGROLL (int8 discrete)
- ❌ Missing other noiser variants

---

## Phase 1: Vectorize with torch.vmap ✅ COMPLETED

**Goal**: Parallel population evaluation instead of sequential loop.

**Status**: Implemented and tested. The `step()` method now uses `torch.vmap` and `functional_call` for parallel evaluation.

### Current (slow):
```python
for i in range(pop_size):
    # Apply noise to each parameter
    for j, (p, base) in enumerate(zip(self.params, self.base_params)):
        noise = self._generate_noise(j, i)
        p.data = base + noise
    fitness = eval_fn(self.model)
```

### Target (fast):
```python
# Use torch.func.vmap to evaluate all population members in parallel
from torch.func import vmap, functional_call

def single_eval(params_dict, noise_dict):
    perturbed = {k: v + noise_dict[k] for k, v in params_dict.items()}
    return functional_call(model, perturbed, inputs)

# Vectorize over population dimension
batched_eval = vmap(single_eval, in_dims=(None, 0))
all_outputs = batched_eval(base_params, all_noises)  # (pop_size, ...)
```

### Files to modify:
- `src/torcheggroll/es.py` - New `VmapEggrollES` class

### Key challenges:
- `torch.vmap` requires functional style (no in-place ops)
- Need to use `torch.func.functional_call` for model evaluation
- Noise generation must be batched

---

## Phase 2: Pre-generated Noise Matrix

**Goal**: Generate noise once, slice into it (like HyperscaleES BIG_RAND_MATRIX).

### Current:
```python
# Generate fresh noise every time
noise = torch.randn(param.shape, generator=gen, ...)
```

### Target:
```python
class NoiseBuffer:
    def __init__(self, size: int = 2**26, seed: int = 0):
        # ~256MB of float32 noise, generated once
        gen = torch.Generator().manual_seed(seed)
        self.buffer = torch.randn(size, generator=gen)

    def get_noise(self, start_idx: int, shape: tuple) -> torch.Tensor:
        numel = math.prod(shape)
        return self.buffer[start_idx:start_idx + numel].view(shape)

    def get_lora_noise(self, start_idx: int, out_dim: int, in_dim: int, rank: int):
        chunk = self.buffer[start_idx:start_idx + (out_dim + in_dim) * rank]
        chunk = chunk.view(out_dim + in_dim, rank)
        return chunk[:in_dim], chunk[in_dim:]  # B, A
```

### Benefits:
- Deterministic noise from index (reproducible)
- No per-step random generation overhead
- Works well with unified memory (MPS)

---

## Phase 3: MPS Device Support

**Goal**: Full MPS (Metal) support on Apple Silicon.

### Tasks:
1. Test all ops on MPS device
2. Handle MPS-unsupported ops with fallbacks
3. Benchmark CPU vs MPS

### Code changes:
```python
# Device selection
if device is None:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
```

### Known MPS issues to handle:
- Some random ops may not be supported
- int8 operations limited
- Need to test vmap compatibility

---

## Phase 4: Multiple Noiser Types

**Goal**: Support different ES algorithms like HyperscaleES.

### Noiser abstraction:
```python
class Noiser(Protocol):
    def perturb(self, params: Dict[str, Tensor], pop_idx: int) -> Dict[str, Tensor]:
        """Apply noise to parameters for one population member."""
        ...

    def update(self, params: Dict[str, Tensor], fitnesses: Tensor,
               noises: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """Compute parameter updates from fitness-weighted noise."""
        ...

# Registry
NOISERS = {
    "eggroll": EggrollNoiser,      # Low-rank A @ B.T
    "open_es": OpenESNoiser,        # Standard Gaussian
    "eggroll_bs": EggrollBSNoiser,  # With baseline subtraction
    "sparse": SparseNoiser,         # Sparse updates
}
```

### Files:
- `src/torcheggroll/noisers/__init__.py`
- `src/torcheggroll/noisers/base.py`
- `src/torcheggroll/noisers/eggroll.py`
- `src/torcheggroll/noisers/open_es.py`

---

## Phase 5: Q-EGGROLL (Quantized/Discrete)

**Goal**: Int8 weights with discrete +1/-1 updates (from nano-egg).

### Key components:

1. **Sign-based fitness**:
```python
def convert_fitnesses(raw_scores: Tensor) -> Tensor:
    # Antithetic pairs: compare +noise vs -noise
    paired = raw_scores.view(-1, 2)
    return torch.sign(paired[:, 0] - paired[:, 1]).to(torch.int8)
```

2. **Discrete updates**:
```python
def discrete_update(param: Tensor, Z: Tensor, threshold: float) -> Tensor:
    # Z = sum(fitness * noise) aggregated signal
    param_int = param.to(torch.int32)
    # Only update if signal exceeds threshold
    should_update = Z.abs() > threshold * math.sqrt(pop_size)
    delta = torch.sign(Z).to(torch.int32)
    return torch.where(should_update, param_int + delta, param_int).clamp(-127, 127).to(torch.int8)
```

3. **Integer forward pass**:
```python
def int_matmul(x: Tensor, weight: Tensor) -> Tensor:
    # x: int8, weight: int8 -> int8
    result = torch.matmul(x.int(), weight.T.int())
    scale = 16 * int(math.sqrt(weight.shape[-1]))
    return (result // scale).clamp(-127, 127).to(torch.int8)
```

### Files:
- `src/torcheggroll/noisers/qeggroll.py`
- `src/torcheggroll/int_ops.py` (integer arithmetic utilities)

---

## Phase 6: Pluggable Optimizers

**Goal**: Support different optimizers for the ES gradient (like optax in JAX).

### Interface:
```python
class ESOptimizer(Protocol):
    def step(self, params: List[Tensor], grads: List[Tensor]) -> List[Tensor]:
        ...

class SGDOptimizer:
    def __init__(self, lr: float):
        self.lr = lr

    def step(self, params, grads):
        return [p + self.lr * g for p, g in zip(params, grads)]

class AdamOptimizer:
    def __init__(self, lr: float, betas=(0.9, 0.999)):
        self.lr = lr
        self.betas = betas
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0

    def step(self, params, grads):
        # Standard Adam update on ES gradients
        ...
```

### Usage:
```python
es = TorchEggrollES(
    model,
    noiser="eggroll",
    optimizer=AdamOptimizer(lr=0.001),
)
```

---

## Implementation Order

1. **Phase 1 (vmap)** - Biggest performance win, enables GPU parallelism
2. **Phase 2 (noise buffer)** - Memory optimization, deterministic noise
3. **Phase 4 (noisers)** - Modular architecture
4. **Phase 5 (Q-EGGROLL)** - Int8 discrete updates
5. **Phase 6 (optimizers)** - Pluggable optimizer support
6. **Phase 3 (MPS)** - Test everything on Metal at the end

---

## Testing Plan

Each phase needs:
1. Unit tests for new functionality
2. Benchmark vs previous implementation
3. Test on CPU, CUDA (if available), MPS
4. Regression test on nano_egg example

```bash
# Run tests
pytest tests/ -v

# Benchmark
python examples/benchmark.py --device cpu
python examples/benchmark.py --device mps
```

---

## Success Metrics

- [ ] 10x+ speedup from vmap on GPU/MPS
- [ ] All HyperscaleES noiser types implemented
- [ ] Q-EGGROLL matches nano-egg loss curves
- [ ] MPS works without fallbacks
- [ ] Clean API matching HyperscaleES patterns
