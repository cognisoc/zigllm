# models.config

## Module Path

```
zigllama.models.config
```

**Source file:** `src/models/config.zig`

---

## Public Types

### `ActivationType`

```zig
pub const ActivationType = enum {
    ReLU,
    GELU,
    SiLU,
    SwiGLU,
    GeGLU,
};
```

### `NormalizationType`

```zig
pub const NormalizationType = enum {
    LayerNorm,
    RMSNorm,
    BatchNorm,
    GroupNorm,
};
```

### `PositionEncodingType`

```zig
pub const PositionEncodingType = enum {
    Sinusoidal,
    Rotary,        // RoPE -- used by LLaMA
    ALiBi,         // Attention with Linear Biases
    Learned,       // Learned absolute positions
    None,
};
```

### `ModelSize`

```zig
pub const ModelSize = enum {
    LLaMA_7B,
    LLaMA_13B,
    LLaMA_30B,
    LLaMA_65B,
    LLaMA2_7B,
    LLaMA2_13B,
    LLaMA2_70B,
    CodeLlama_7B,
    CodeLlama_13B,
    CodeLlama_34B,
    TinyLlama,
};
```

### `ModelConfig`

```zig
pub const ModelConfig = struct {
    // Architecture
    d_model: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: ?usize,       // null = same as num_heads (MHA)
    d_ff: usize,
    vocab_size: usize,
    max_seq_len: usize,

    // Normalization
    norm_type: NormalizationType,
    norm_eps: f32,

    // Activation
    activation: ActivationType,

    // Position encoding
    position_encoding: PositionEncodingType,
    rope_theta: f32,

    // Regularization
    dropout: f32,
    attention_dropout: f32,

    // Quantization
    weight_quant: ?QuantType,
};
```

Unified configuration that describes any supported model architecture.

---

## Public Functions

### `ModelConfig.llama`

```zig
pub fn llama(size: ModelSize) ModelConfig
```

Return a `ModelConfig` pre-filled for the given LLaMA variant. Sets
`norm_type = .RMSNorm`, `activation = .SwiGLU`, `position_encoding = .Rotary`,
and variant-specific dimensions.

### `ModelConfig.validate`

```zig
pub fn validate(self: ModelConfig) !void
```

Check internal consistency:

- `d_model` must be divisible by `num_heads`.
- `num_kv_heads` (if set) must divide `num_heads` evenly.
- `d_ff` must be positive.
- `max_seq_len` must be positive.

Returns `error{InvalidConfig}` on failure.

### `ModelConfig.parameterCount`

```zig
pub fn parameterCount(self: ModelConfig) u64
```

Estimate the total number of trainable parameters:

```
embedding + num_layers * (4 * d_model^2 + 2 * d_model * d_ff) + vocab_size * d_model
```

### `ModelConfig.memoryRequirements`

```zig
pub fn memoryRequirements(self: ModelConfig) u64
```

Estimate peak memory usage in bytes assuming f32 weights:

```
parameterCount() * 4
```

For quantized models, divide by the compression ratio of the target format.

---

## Error Types

- `error{InvalidConfig}` -- returned by `validate`.

---

## Usage Example

```zig
const cfg = @import("zigllama").models.config;

// Get a pre-built LLaMA-7B config
var config = cfg.ModelConfig.llama(.LLaMA_7B);
try config.validate();

const params = config.parameterCount();
const mem = config.memoryRequirements();

std.debug.print("LLaMA-7B: {} parameters, {} bytes\n", .{ params, mem });
// ~6.7B parameters, ~26.8 GB in f32
```

---

## Related Modules

- [`models.llama`](llama.md) -- Uses `LLaMAConfig` (a subset of `ModelConfig`).
- [`linear_algebra.quantization`](quantization.md) -- Quantization types
  referenced by `weight_quant`.
