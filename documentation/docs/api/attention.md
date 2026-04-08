# transformers.attention

## Module Path

```
zigllama.transformers.attention
```

**Source file:** `src/transformers/attention.zig`

---

## Public Types

### `AttentionType`

```zig
pub const AttentionType = enum {
    SelfAttention,
    CrossAttention,
    CausalAttention,
    SparseAttention,
};
```

| Variant | Description |
|---------|-------------|
| `SelfAttention` | Q, K, V all come from the same input |
| `CrossAttention` | Q from one sequence, K/V from another |
| `CausalAttention` | Self-attention with a causal (lower-triangular) mask |
| `SparseAttention` | Attention with a sparse pattern for long sequences |

### `MultiHeadAttention`

```zig
pub const MultiHeadAttention = struct {
    num_heads: usize,
    d_model: usize,
    d_k: usize,
    d_v: usize,
    w_q: Tensor(f32),   // [d_model, d_model]
    w_k: Tensor(f32),   // [d_model, d_model]
    w_v: Tensor(f32),   // [d_model, d_model]
    w_o: Tensor(f32),   // [d_model, d_model]
    allocator: std.mem.Allocator,
};
```

Multi-head attention mechanism. Splits the model dimension into `num_heads`
parallel attention heads, each operating on `d_k = d_model / num_heads`
dimensions.

---

## Public Functions

### `scaledDotProductAttention`

```zig
pub fn scaledDotProductAttention(
    Q: Tensor(f32),
    K: Tensor(f32),
    V: Tensor(f32),
    mask: ?Tensor(f32),
    allocator: std.mem.Allocator,
) !Tensor(f32)
```

Core attention computation:

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `Q` | `Tensor(f32)` | Queries, shape `[seq_len, d_k]` |
| `K` | `Tensor(f32)` | Keys, shape `[seq_len, d_k]` |
| `V` | `Tensor(f32)` | Values, shape `[seq_len, d_v]` |
| `mask` | `?Tensor(f32)` | Optional mask (0 = attend, -inf = mask out) |
| `allocator` | `Allocator` | Memory allocator |

**Returns:** `!Tensor(f32)` with shape `[seq_len, d_v]`.

### `MultiHeadAttention.init`

```zig
pub fn init(
    num_heads: usize,
    d_model: usize,
    allocator: std.mem.Allocator,
) !MultiHeadAttention
```

Allocate weight matrices for all heads. Sets `d_k = d_v = d_model / num_heads`.

### `MultiHeadAttention.deinit`

```zig
pub fn deinit(self: *MultiHeadAttention) void
```

Free all weight tensors.

### `MultiHeadAttention.forward`

```zig
pub fn forward(
    self: MultiHeadAttention,
    Q: Tensor(f32),
    K: Tensor(f32),
    V: Tensor(f32),
    mask: ?Tensor(f32),
    allocator: std.mem.Allocator,
) !Tensor(f32)
```

Full multi-head attention forward pass:

1. Project Q, K, V through learned weight matrices.
2. Split into `num_heads` parallel heads.
3. Apply `scaledDotProductAttention` to each head.
4. Concatenate heads and project through `w_o`.

**Returns:** `!Tensor(f32)` with shape `[seq_len, d_model]`.

### `createCausalMask`

```zig
pub fn createCausalMask(
    seq_len: usize,
    allocator: std.mem.Allocator,
) !Tensor(f32)
```

Create a lower-triangular causal mask of shape `[seq_len, seq_len]`. Positions
above the diagonal are filled with `-inf` to prevent attending to future
tokens during autoregressive generation.

---

## Error Types

- `TensorError.IncompatibleShapes` -- Q/K/V dimension mismatch.
- `TensorError.OutOfMemory`

---

## Usage Example

```zig
const attn = @import("zigllama").transformers.attention;
const Tensor = @import("zigllama").foundation.tensor.Tensor;

var mha = try attn.MultiHeadAttention.init(32, 4096, allocator);
defer mha.deinit();

var input = try Tensor(f32).init(allocator, &[_]usize{ 128, 4096 });
defer input.deinit();

var mask = try attn.createCausalMask(128, allocator);
defer mask.deinit();

// Self-attention: Q = K = V = input
var output = try mha.forward(input, input, input, mask, allocator);
defer output.deinit();
```

---

## Related Modules

- [`inference.kv_cache`](kv-cache.md) -- Caches K/V projections across
  generation steps.
- [`transformers.transformer_block`](transformer-block.md) -- Wraps attention
  with normalization and FFN.
- [`neural_primitives.embeddings`](embeddings.md) -- RoPE applied to Q/K before
  attention.
