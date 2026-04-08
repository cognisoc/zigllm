# Model Configuration

## Overview

Every model architecture in ZigLLM is instantiated from a configuration struct that
fully specifies its dimensions, component choices, and numerical parameters. The
`ModelConfig` struct in `src/models/config.zig` serves as the shared configuration
type, while each architecture also defines its own specialized config (e.g.,
`LLaMAConfig`, `MistralConfig`) that converts to `ModelConfig` when needed.

Understanding model configuration is prerequisite to working with any architecture.
The configuration determines parameter count, memory footprint, and the specific
algorithmic choices (activation functions, normalization, position encoding) that
define the model's behavior.

---

## ModelConfig Struct

The central configuration type captures all parameters needed to instantiate a
transformer-based language model.

```zig
pub const ModelConfig = struct {
    // === Model Architecture ===
    d_model: usize,          // Hidden dimension (embedding size)
    num_layers: usize,       // Number of transformer layers
    num_heads: usize,        // Number of attention heads
    d_ff: usize,             // Feed-forward intermediate dimension

    // === Vocabulary and Sequence ===
    vocab_size: usize,       // Number of tokens in vocabulary
    max_seq_len: usize,      // Maximum supported sequence length

    // === Component Choices ===
    activation: ActivationType,           // FFN activation function
    normalization: NormalizationType,     // Normalization layer type
    position_encoding: PositionEncodingType, // Position encoding strategy

    // === Numerical Parameters ===
    norm_eps: f32,           // Epsilon for numerical stability in norms
    dropout_rate: f32,       // Dropout rate (training only, 0.0 for inference)

    // === RoPE Parameters ===
    rope_theta: f32,         // RoPE base frequency (typically 10000.0)
    rope_scaling: f32,       // RoPE scaling factor (1.0 = no scaling)

    // === Attention Options ===
    attention_bias: bool,    // Use bias in attention output projection
    qkv_bias: bool,          // Use bias in Q, K, V projections

    // === Memory Optimization ===
    gradient_checkpointing: bool,  // Trade compute for memory
    use_flash_attention: bool,     // Use memory-efficient attention kernel
};
```

!!! definition "Key Dimensions"
    - \( d_\text{model} \): The hidden dimension that flows through the entire network.
      All residual connections operate at this dimension.
    - \( d_\text{ff} \): The intermediate dimension of the feed-forward network.
      For standard FFN, \( d_\text{ff} = 4 \cdot d_\text{model} \). For gated
      activations (SwiGLU), \( d_\text{ff} \approx \frac{8}{3} \cdot d_\text{model} \).
    - \( d_\text{head} = d_\text{model} / n_\text{heads} \): Per-head dimension,
      typically 64 or 128.

---

## Enumerated Types

### ActivationType

Controls the nonlinearity in the feed-forward network.

```zig
pub const ActivationType = enum {
    ReLU,    // max(0, x) -- original transformer
    GELU,    // Gaussian Error Linear Unit -- GPT-2, BERT
    SwiGLU,  // Swish-gated -- LLaMA, Mistral, Qwen
    GeGLU,   // GELU-gated -- Gemma
    GLU,     // Standard gated linear unit
};
```

The `parameterMultiplier()` method returns the number of weight matrices needed:

| Activation | Multiplier | Matrices | Used By |
|:-----------|:----------:|:---------|:--------|
| ReLU | 2.0 | up + down | Original Transformer |
| GELU | 2.0 | up + down | GPT-2, BERT, Falcon |
| SwiGLU | 3.0 | gate + up + down | LLaMA, Mistral, Qwen |
| GeGLU | 3.0 | gate + up + down | Gemma |
| GLU | 3.0 | gate + up + down | -- |

### NormalizationType

```zig
pub const NormalizationType = enum {
    LayerNorm,  // Original transformer, GPT-2, BERT
    RMSNorm,    // LLaMA, Mistral, Qwen (more efficient)
    None,       // No normalization
};
```

!!! info "RMSNorm vs LayerNorm"
    RMSNorm omits the mean-centering step of LayerNorm, computing only
    \( \hat{x} = x / \text{RMS}(x) \) where \( \text{RMS}(x) = \sqrt{\frac{1}{d} \sum x_i^2 + \epsilon} \).
    This saves one reduction operation per layer with negligible accuracy loss.

### PositionEncodingType

```zig
pub const PositionEncodingType = enum {
    None,        // No positional information
    Sinusoidal,  // Fixed sinusoidal (original Transformer)
    Learned,     // Trainable position embeddings (GPT-2)
    RoPE,        // Rotary Position Embeddings (LLaMA, Mistral)
    ALiBi,       // Attention with Linear Biases (BLOOM, Falcon-1B)
};
```

---

## Preset Configurations

ZigLLM provides factory methods for canonical model sizes. The `ModelConfig.llama()`
method creates configurations for the LLaMA family.

### LLaMA Family Presets

| Parameter | 7B | 13B | 30B | 65B |
|:----------|---:|----:|----:|----:|
| `d_model` | 4096 | 5120 | 6656 | 8192 |
| `num_layers` | 32 | 40 | 60 | 80 |
| `num_heads` | 32 | 40 | 52 | 64 |
| `d_ff` | 11008 | 13824 | 17920 | 22016 |
| `vocab_size` | 32000 | 32000 | 32000 | 32000 |
| `max_seq_len` | 2048 | 2048 | 2048 | 2048 |
| `activation` | SwiGLU | SwiGLU | SwiGLU | SwiGLU |
| `normalization` | RMSNorm | RMSNorm | RMSNorm | RMSNorm |
| `position_encoding` | RoPE | RoPE | RoPE | RoPE |
| `norm_eps` | 1e-6 | 1e-6 | 1e-6 | 1e-6 |
| `rope_theta` | 10000.0 | 10000.0 | 10000.0 | 10000.0 |
| `d_head` | 128 | 128 | 128 | 128 |
| `gradient_checkpointing` | false | false | true | true |

!!! tip "Scaling Pattern"
    Observe that \( d_\text{ff} \approx \frac{8}{3} \cdot d_\text{model} \) for all
    LLaMA variants. This ratio is characteristic of SwiGLU activations, which use three
    weight matrices instead of two, so the intermediate dimension is reduced from
    \( 4 \cdot d_\text{model} \) to compensate.

### CodeLlama Variants

CodeLlama extends LLaMA with longer context support:

```zig
.CodeLlama_7B => blk: {
    var config = ModelConfig.llama(.LLaMA_7B);
    config.max_seq_len = 16384;  // 8x longer than base LLaMA
    config.rope_scaling = 1.0;
    break :blk config;
},
```

---

## Configuration Validation

The `validate()` method checks for common configuration errors before model
instantiation.

```zig
pub fn validate(self: ModelConfig) !void {
    // Head dimension must divide evenly
    if (self.d_model % self.num_heads != 0)
        return error.IncompatibleHeadDimension;

    // Reasonable dimension ranges
    if (self.d_model < 64 or self.d_model > 32768)
        return error.UnreasonableModelDimension;

    if (self.num_layers < 1 or self.num_layers > 1000)
        return error.UnreasonableLayerCount;

    if (self.num_heads < 1 or self.num_heads > 256)
        return error.UnreasonableHeadCount;

    if (self.vocab_size < 1000 or self.vocab_size > 1000000)
        return error.UnreasonableVocabSize;

    // Numerical stability
    if (self.norm_eps < 1e-12 or self.norm_eps > 1e-3)
        return error.UnreasonableEpsilon;
}
```

!!! algorithm "Validation Checks"
    1. **Head dimension compatibility**: \( d_\text{model} \bmod n_\text{heads} = 0 \)
    2. **Dimension bounds**: \( 64 \le d_\text{model} \le 32768 \)
    3. **Layer count bounds**: \( 1 \le n_\text{layers} \le 1000 \)
    4. **Head count bounds**: \( 1 \le n_\text{heads} \le 256 \)
    5. **Vocabulary bounds**: \( 1000 \le V \le 1{,}000{,}000 \)
    6. **Epsilon bounds**: \( 10^{-12} \le \epsilon \le 10^{-3} \)

---

## Parameter Counting

The `parameterCount()` method computes the total number of trainable parameters.

\[
P = \underbrace{V \cdot d}_{\text{embeddings}} + L \cdot \Big(
\underbrace{3d^2 + d^2}_{\text{attention}} +
\underbrace{m \cdot d \cdot d_\text{ff}}_{\text{FFN}} +
\underbrace{2d}_{\text{norms}}
\Big) + \underbrace{d}_{\text{final norm}} + \underbrace{d_\text{pos}}_{\text{position}}
\]

where:

- \( V \) = `vocab_size`, \( d \) = `d_model`, \( L \) = `num_layers`
- \( m \) = activation parameter multiplier (2.0 for standard, 3.0 for gated)
- \( d_\text{pos} \) = position encoding parameters (if any)

```zig
pub fn parameterCount(self: ModelConfig) usize {
    const embedding_params = self.vocab_size * self.d_model;

    var layer_params: usize = 0;
    const qkv_params = 3 * self.d_model * self.d_model;
    const attention_output_params = self.d_model * self.d_model;
    layer_params += qkv_params + attention_output_params;

    const ffn_multiplier = self.activation.parameterMultiplier();
    const ffn_params = @as(usize, @intFromFloat(
        ffn_multiplier * @as(f32, @floatFromInt(self.d_model * self.d_ff))
    ));
    layer_params += ffn_params;

    if (self.normalization.hasParameters()) {
        layer_params += 2 * self.d_model;
    }

    return embedding_params + (layer_params * self.num_layers) + ...;
}
```

---

## Memory Requirements

The `memoryRequirements()` method estimates the total memory needed for inference at a
given batch size and sequence length.

!!! complexity "Memory Breakdown"
    | Component | Formula | LLaMA-7B (B=1, S=2048) |
    |:----------|:--------|:-----------------------|
    | Parameters | \( P \times 4 \) bytes | ~26.8 GB |
    | Activations | \( B \times S \times d \times L \times 4 \) | ~1.0 GB |
    | KV Cache | \( 2 \times B \times H \times S \times d_h \times L \times 4 \) | ~2.0 GB |
    | **Total** | Sum | **~29.8 GB** |

```zig
pub fn memoryRequirements(self: ModelConfig, batch_size: usize, sequence_length: usize)
    struct { parameters: usize, activations: usize, kv_cache: usize, total: usize }
{
    const param_memory = self.parameterCount() * @sizeOf(f32);
    const activation_memory = batch_size * sequence_length *
        self.d_model * self.num_layers * @sizeOf(f32);
    const kv_cache_memory = 2 * batch_size * self.num_heads *
        sequence_length * self.headDim() * self.num_layers * @sizeOf(f32);
    return .{
        .parameters = param_memory,
        .activations = activation_memory,
        .kv_cache = kv_cache_memory,
        .total = param_memory + activation_memory + kv_cache_memory,
    };
}
```

---

## Custom Configurations

To create a non-standard configuration, use the `custom()` factory method or
construct the struct directly.

```zig
// Factory method with sensible defaults
const config = ModelConfig.custom(
    512,    // d_model
    6,      // num_layers
    8,      // num_heads
    1000,   // vocab_size
);
// d_ff defaults to 4 * d_model = 2048
// activation defaults to SwiGLU
// normalization defaults to RMSNorm
// position_encoding defaults to RoPE

// Validate before use
try config.validate();
```

For full control, construct the struct directly:

```zig
const config = ModelConfig{
    .d_model = 256,
    .num_layers = 4,
    .num_heads = 4,
    .d_ff = 688,              // ~8/3 * 256
    .vocab_size = 500,
    .max_seq_len = 512,
    .activation = .GELU,      // Override: use GELU instead of SwiGLU
    .normalization = .LayerNorm,
    .position_encoding = .Learned,
    .norm_eps = 1e-5,
    .dropout_rate = 0.0,
    .rope_theta = 10000.0,
    .rope_scaling = 1.0,
    .attention_bias = true,
    .qkv_bias = true,
    .gradient_checkpointing = false,
    .use_flash_attention = false,
};
```

!!! tip "Testing Tip"
    When writing tests, use very small configurations (e.g., `d_model=16`, `num_layers=1`)
    to keep test execution fast while still exercising the full forward pass.

---

## Architecture-Specific Configs

Each model architecture provides its own configuration struct that captures
architecture-specific parameters not present in the generic `ModelConfig`.

| Architecture | Config Struct | Extra Fields |
|:-------------|:-------------|:-------------|
| LLaMA | `LLaMAConfig` | `use_rope`, `rope_theta` |
| Mistral | `MistralConfig` | `n_kv_heads`, `sliding_window`, `num_experts` |
| GPT-2 | `GPT2Config` | `dropout` (learned positions implicit) |
| Falcon | `FalconConfig` | `parallel_attn`, `multi_query`, `alibi`, `new_decoder_architecture` |
| Qwen | `QwenConfig` | `rope_scaling`, `use_dynamic_ntk`, `use_logn_attn`, `use_sliding_window` |

These specialized configs typically provide a `fromVariant()` or `create()` factory
and, where applicable, a `toModelConfig()` conversion method for interoperability.

---

## References

[^1]: Touvron, H. et al. "LLaMA: Open and Efficient Foundation Language Models." arXiv:2302.13971, 2023.
[^2]: Zhang, S. et al. "OPT: Open Pre-trained Transformer Language Models." arXiv:2205.01068, 2022.
