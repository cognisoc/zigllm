# models.llama

## Module Path

```
zigllama.models.llama
```

**Source file:** `src/models/llama.zig`

---

## Public Types

### `ModelSize`

```zig
pub const ModelSize = enum {
    LLaMA_7B,
    LLaMA_13B,
    LLaMA_30B,
    LLaMA_65B,
    CodeLlama_7B,
    CodeLlama_13B,
    CodeLlama_34B,
    LLaMA2_7B,
    LLaMA2_13B,
    LLaMA2_70B,
    TinyLlama,
};
```

Predefined model sizes with known configurations.

### `LLaMAConfig`

```zig
pub const LLaMAConfig = struct {
    d_model: usize,
    num_layers: usize,
    num_heads: usize,
    d_ff: usize,
    vocab_size: usize,
    max_seq_len: usize,
    norm_eps: f32,
    use_rope: bool,
    rope_theta: f32,
};
```

| Field | Description | LLaMA-7B Default |
|-------|-------------|-----------------|
| `d_model` | Embedding / hidden dimension | 4096 |
| `num_layers` | Number of transformer layers | 32 |
| `num_heads` | Number of attention heads | 32 |
| `d_ff` | Feed-forward inner dimension | 11008 |
| `vocab_size` | Vocabulary size | 32000 |
| `max_seq_len` | Maximum context length | 2048 |
| `norm_eps` | RMSNorm epsilon | 1e-5 |
| `use_rope` | Enable RoPE (always `true`) | true |
| `rope_theta` | RoPE base frequency | 10000.0 |

### `LLaMAModel`

```zig
pub const LLaMAModel = struct {
    config: LLaMAConfig,
    embeddings: TokenEmbedding,
    layers: []TransformerBlock,
    output_norm: Tensor(f32),
    lm_head: Tensor(f32),
    allocator: std.mem.Allocator,
};
```

Complete LLaMA language model. Composes embeddings, a stack of transformer
blocks, a final RMSNorm, and a linear output projection (lm_head) that maps
hidden states to vocabulary logits.

---

## Public Functions

### `LLaMAConfig.init`

```zig
pub fn init(model_size: ModelSize) LLaMAConfig
```

Return a pre-filled configuration for the given model size.

### `LLaMAModel.init`

```zig
pub fn init(
    config: LLaMAConfig,
    allocator: std.mem.Allocator,
) !LLaMAModel
```

Allocate all model parameters according to `config`. Weights are
zero-initialized; load from a GGUF file to populate them.

### `LLaMAModel.deinit`

```zig
pub fn deinit(self: *LLaMAModel) void
```

Free all parameter tensors and layer structures.

### `LLaMAModel.forward`

```zig
pub fn forward(
    self: LLaMAModel,
    tokens: []const u32,
    allocator: std.mem.Allocator,
) !Tensor(f32)
```

Run a full forward pass:

1. Token embedding lookup.
2. Sequential pass through all transformer blocks with causal mask.
3. RMSNorm on the final hidden states.
4. Linear projection to vocabulary logits.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `tokens` | `[]const u32` | Input token ID sequence |
| `allocator` | `Allocator` | Memory allocator for intermediates |

**Returns:** `!Tensor(f32)` with shape `[seq_len, vocab_size]` containing logits.

### `LLaMAModel.generate`

```zig
pub fn generate(
    self: LLaMAModel,
    tokens: []const u32,
    max_tokens: usize,
    allocator: std.mem.Allocator,
) ![]u32
```

Autoregressive text generation. Repeatedly calls `forward` on the last token,
samples from the output distribution, and appends the result until
`max_tokens` is reached or an EOS token is produced.

---

## Error Types

- `TensorError.OutOfMemory`
- `TensorError.InvalidIndex` -- token ID exceeds vocabulary size.

---

## Usage Example

```zig
const llama = @import("zigllama").models.llama;

// Create a 7B configuration
const config = llama.LLaMAConfig.init(.LLaMA_7B);

// Allocate the model
var model = try llama.LLaMAModel.init(config, allocator);
defer model.deinit();

// Forward pass
const prompt_tokens = &[_]u32{ 1, 15043, 29892, 920 };
var logits = try model.forward(prompt_tokens, allocator);
defer logits.deinit();

// Or generate tokens autoregressively
const output_tokens = try model.generate(prompt_tokens, 128, allocator);
defer allocator.free(output_tokens);
```

---

## Related Modules

- [`models.config`](config.md) -- Extended model configuration and presets.
- [`models.gguf`](gguf.md) -- Load weights from GGUF files into a
  `LLaMAModel`.
- [`models.tokenizer`](tokenizer.md) -- Encode text to token IDs and decode
  back.
- [`inference.generation`](generation.md) -- Full-featured text generation with
  sampling strategies.
