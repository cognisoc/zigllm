# neural_primitives.normalization

## Module Path

```
zigllama.neural_primitives.normalization
```

**Source file:** `src/neural_primitives/normalization.zig`

---

## Public Functions

### `layerNorm`

```zig
pub fn layerNorm(
    input: Tensor(f32),
    gamma: Tensor(f32),
    beta: Tensor(f32),
    eps: f32,
    allocator: std.mem.Allocator,
) !Tensor(f32)
```

Layer Normalization (Ba et al., 2016). Normalizes across the feature dimension
of each token independently:

```
output[i] = gamma * (input[i] - mean) / sqrt(variance + eps) + beta
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `input` | `Tensor(f32)` | Input tensor, shape `[seq_len, d_model]` |
| `gamma` | `Tensor(f32)` | Learned scale, shape `[d_model]` |
| `beta` | `Tensor(f32)` | Learned bias, shape `[d_model]` |
| `eps` | `f32` | Small constant for numerical stability (typically 1e-5) |
| `allocator` | `Allocator` | Memory allocator for the result |

**Returns:** normalized tensor with the same shape as `input`.

### `rmsNorm`

```zig
pub fn rmsNorm(
    input: Tensor(f32),
    gamma: Tensor(f32),
    eps: f32,
    allocator: std.mem.Allocator,
) !Tensor(f32)
```

Root Mean Square Normalization (Zhang & Sennrich, 2019). The normalization
used by LLaMA. Simpler and faster than LayerNorm because it skips the mean
subtraction and bias term:

```
output[i] = gamma * input[i] / sqrt(mean(input^2) + eps)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `input` | `Tensor(f32)` | Input tensor, shape `[seq_len, d_model]` |
| `gamma` | `Tensor(f32)` | Learned scale, shape `[d_model]` |
| `eps` | `f32` | Stability constant (LLaMA default: 1e-5) |
| `allocator` | `Allocator` | Memory allocator |

**Returns:** normalized tensor with the same shape as `input`.

### `batchNorm`

```zig
pub fn batchNorm(
    input: Tensor(f32),
    gamma: Tensor(f32),
    beta: Tensor(f32),
    running_mean: Tensor(f32),
    running_var: Tensor(f32),
    eps: f32,
    training: bool,
    allocator: std.mem.Allocator,
) !Tensor(f32)
```

Batch Normalization (Ioffe & Szegedy, 2015). Normalizes across the batch
dimension. Primarily included for completeness; LLaMA does not use batch
normalization.

### `groupNorm`

```zig
pub fn groupNorm(
    input: Tensor(f32),
    gamma: Tensor(f32),
    beta: Tensor(f32),
    num_groups: usize,
    eps: f32,
    allocator: std.mem.Allocator,
) !Tensor(f32)
```

Group Normalization (Wu & He, 2018). Divides channels into groups and
normalizes within each group. Useful for small-batch scenarios.

---

## Error Types

- `TensorError.IncompatibleShapes` -- `gamma`/`beta` size does not match the
  feature dimension of `input`.
- `TensorError.OutOfMemory`

---

## Usage Example

```zig
const norm = @import("zigllama").neural_primitives.normalization;
const Tensor = @import("zigllama").foundation.tensor.Tensor;

// Prepare inputs
var input = try Tensor(f32).init(allocator, &[_]usize{ 16, 4096 });
defer input.deinit();

var gamma = try Tensor(f32).init(allocator, &[_]usize{4096});
defer gamma.deinit();
// Fill gamma with 1.0 (identity scale)
for (gamma.data) |*v| v.* = 1.0;

// RMSNorm -- the normalization used by LLaMA
var output = try norm.rmsNorm(input, gamma, 1e-5, allocator);
defer output.deinit();
```

---

## Related Modules

- [`transformers.transformer_block`](transformer-block.md) -- Applies
  normalization before or after attention/FFN sub-layers.
- [`models.llama`](llama.md) -- Uses `rmsNorm` with the model's `norm_eps`
  parameter.
