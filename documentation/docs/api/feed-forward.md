# transformers.feed_forward

## Module Path

```
zigllama.transformers.feed_forward
```

**Source file:** `src/transformers/feed_forward.zig`

---

## Public Types

### `FFNType`

```zig
pub const FFNType = enum {
    Standard,
    SwiGLU,
    GeGLU,
    GLU,
};
```

Feed-forward network variant.

| Variant | Formula | Used By |
|---------|---------|---------|
| `Standard` | `W2 * activation(W1 * x)` | Original Transformer |
| `SwiGLU` | `W2 * (silu(W1 * x) * (W3 * x))` | LLaMA |
| `GeGLU` | `W2 * (gelu(W1 * x) * (W3 * x))` | PaLM |
| `GLU` | `W2 * (sigmoid(W1 * x) * (W3 * x))` | GLU variants |

### `FeedForward`

```zig
pub const FeedForward = struct {
    w1: Tensor(f32),            // [d_model, d_ff]
    w2: Tensor(f32),            // [d_ff, d_model]
    w3: ?Tensor(f32),           // [d_model, d_ff] (gated variants only)
    activation: FFNType,
    ffn_type: FFNType,
    d_model: usize,
    d_ff: usize,
    allocator: std.mem.Allocator,
};
```

| Field | Description |
|-------|-------------|
| `w1` | Up-projection (or gate projection for gated variants) |
| `w2` | Down-projection |
| `w3` | Up-projection for the non-gated branch (null for `Standard`) |
| `activation` | Which activation to apply |

### `ExpertFFN`

```zig
pub const ExpertFFN = struct {
    experts: []FeedForward,
    gate: Tensor(f32),
    num_experts: usize,
    top_k: usize,
};
```

Mixture-of-Experts feed-forward network. Routes each token to the `top_k`
highest-scoring experts via a learned gating function.

---

## Public Functions

### `FeedForward.init`

```zig
pub fn init(
    d_model: usize,
    d_ff: usize,
    ffn_type: FFNType,
    allocator: std.mem.Allocator,
) !FeedForward
```

Allocate weight matrices. For gated variants (`SwiGLU`, `GeGLU`, `GLU`), three
matrices are created; for `Standard`, only two.

### `FeedForward.deinit`

```zig
pub fn deinit(self: *FeedForward) void
```

Free all weight tensors.

### `FeedForward.forward`

```zig
pub fn forward(
    self: FeedForward,
    input: Tensor(f32),
    allocator: std.mem.Allocator,
) !Tensor(f32)
```

Apply the feed-forward sub-layer. For SwiGLU (the LLaMA default):

```
output = W2 * (silu(W1 * input) * (W3 * input))
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `input` | `Tensor(f32)` | Shape `[seq_len, d_model]` |
| `allocator` | `Allocator` | Memory allocator for intermediates and result |

**Returns:** `!Tensor(f32)` with shape `[seq_len, d_model]`.

---

## Error Types

- `TensorError.IncompatibleShapes` -- input feature dimension does not match
  `d_model`.
- `TensorError.OutOfMemory`

---

## Usage Example

```zig
const ff = @import("zigllama").transformers.feed_forward;
const Tensor = @import("zigllama").foundation.tensor.Tensor;

// LLaMA-style SwiGLU FFN
var ffn = try ff.FeedForward.init(4096, 11008, .SwiGLU, allocator);
defer ffn.deinit();

var input = try Tensor(f32).init(allocator, &[_]usize{ 128, 4096 });
defer input.deinit();

var output = try ffn.forward(input, allocator);
defer output.deinit();
// output.shape == [128, 4096]
```

---

## Related Modules

- [`neural_primitives.activations`](activations.md) -- Activation functions
  used inside the FFN.
- [`transformers.transformer_block`](transformer-block.md) -- Combines FFN with
  attention and normalization.
