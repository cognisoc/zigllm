# transformers.transformer_block

## Module Path

```
zigllama.transformers.transformer_block
```

**Source file:** `src/transformers/transformer_block.zig`

---

## Public Types

### `NormPlacement`

```zig
pub const NormPlacement = enum {
    PreNorm,
    PostNorm,
};
```

| Variant | Description |
|---------|-------------|
| `PreNorm` | Normalize before attention/FFN (LLaMA, GPT-2 style) |
| `PostNorm` | Normalize after attention/FFN (original Transformer) |

Pre-normalization provides better gradient flow in deep networks and is used by
all LLaMA variants.

### `TransformerBlock`

```zig
pub const TransformerBlock = struct {
    attention: MultiHeadAttention,
    ffn: FeedForward,
    norm1: Tensor(f32),         // attention sub-layer norm weights
    norm2: Tensor(f32),         // FFN sub-layer norm weights
    norm_placement: NormPlacement,
    norm_eps: f32,
    allocator: std.mem.Allocator,
};
```

A single transformer layer containing attention, feed-forward, and
normalization sub-layers with residual connections.

### `Transformer`

```zig
pub const Transformer = struct {
    blocks: []TransformerBlock,
    d_model: usize,
    num_layers: usize,
    allocator: std.mem.Allocator,
};
```

Stack of `TransformerBlock` layers. The full encoder or decoder body of a
transformer model.

---

## Public Functions

### `TransformerBlock.init`

```zig
pub fn init(
    num_heads: usize,
    d_model: usize,
    d_ff: usize,
    ffn_type: FFNType,
    norm_placement: NormPlacement,
    norm_eps: f32,
    allocator: std.mem.Allocator,
) !TransformerBlock
```

Allocate a single transformer block with all sub-layer weights.

### `TransformerBlock.deinit`

```zig
pub fn deinit(self: *TransformerBlock) void
```

Free all owned tensors.

### `TransformerBlock.forward`

```zig
pub fn forward(
    self: TransformerBlock,
    input: Tensor(f32),
    mask: ?Tensor(f32),
    allocator: std.mem.Allocator,
) !Tensor(f32)
```

Forward pass through one transformer layer.

For `PreNorm` (LLaMA):

```
h = input + attention(rmsNorm(input))
output = h + ffn(rmsNorm(h))
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `input` | `Tensor(f32)` | Shape `[seq_len, d_model]` |
| `mask` | `?Tensor(f32)` | Optional causal or padding mask |
| `allocator` | `Allocator` | Memory allocator |

**Returns:** `!Tensor(f32)` with shape `[seq_len, d_model]`.

### `Transformer.init`

```zig
pub fn init(
    num_layers: usize,
    num_heads: usize,
    d_model: usize,
    d_ff: usize,
    ffn_type: FFNType,
    norm_placement: NormPlacement,
    norm_eps: f32,
    allocator: std.mem.Allocator,
) !Transformer
```

Allocate the full stack of transformer blocks.

### `Transformer.deinit`

```zig
pub fn deinit(self: *Transformer) void
```

Free all blocks.

### `Transformer.forward`

```zig
pub fn forward(
    self: Transformer,
    input: Tensor(f32),
    mask: ?Tensor(f32),
    allocator: std.mem.Allocator,
) !Tensor(f32)
```

Run the input through every layer sequentially. Each layer receives the
output of the previous one.

---

## Error Types

- `TensorError.IncompatibleShapes`
- `TensorError.OutOfMemory`

---

## Usage Example

```zig
const tb = @import("zigllama").transformers.transformer_block;
const Tensor = @import("zigllama").foundation.tensor.Tensor;

// Build a 32-layer LLaMA-style transformer body
var transformer = try tb.Transformer.init(
    32,     // num_layers
    32,     // num_heads
    4096,   // d_model
    11008,  // d_ff
    .SwiGLU,
    .PreNorm,
    1e-5,
    allocator,
);
defer transformer.deinit();

var input = try Tensor(f32).init(allocator, &[_]usize{ 128, 4096 });
defer input.deinit();

var output = try transformer.forward(input, null, allocator);
defer output.deinit();
```

---

## Related Modules

- [`transformers.attention`](attention.md) -- The attention sub-layer.
- [`transformers.feed_forward`](feed-forward.md) -- The FFN sub-layer.
- [`neural_primitives.normalization`](normalization.md) -- Normalization applied
  within each block.
- [`models.llama`](llama.md) -- Wraps `Transformer` with embeddings and an
  output head.
