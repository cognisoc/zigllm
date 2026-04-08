# neural_primitives.embeddings

## Module Path

```
zigllama.neural_primitives.embeddings
```

**Source file:** `src/neural_primitives/embeddings.zig`

---

## Public Types

### `TokenEmbedding`

```zig
pub const TokenEmbedding = struct {
    weights: Tensor(f32),
    vocab_size: usize,
    d_model: usize,
};
```

Learned embedding table that maps token IDs to dense vectors.

| Field | Type | Description |
|-------|------|-------------|
| `weights` | `Tensor(f32)` | Shape `[vocab_size, d_model]` |
| `vocab_size` | `usize` | Number of tokens in vocabulary |
| `d_model` | `usize` | Embedding dimension |

### `SegmentEmbedding`

```zig
pub const SegmentEmbedding = struct {
    weights: Tensor(f32),
    num_segments: usize,
    d_model: usize,
};
```

Optional segment embeddings (used by BERT-style models, not LLaMA).

---

## Public Functions

### `TokenEmbedding.init`

```zig
pub fn init(
    vocab_size: usize,
    d_model: usize,
    allocator: std.mem.Allocator,
) !TokenEmbedding
```

Allocate and zero-initialize an embedding table.

### `TokenEmbedding.deinit`

```zig
pub fn deinit(self: *TokenEmbedding) void
```

Free the weight tensor.

### `TokenEmbedding.forward`

```zig
pub fn forward(
    self: TokenEmbedding,
    token_ids: []const u32,
    allocator: std.mem.Allocator,
) !Tensor(f32)
```

Look up embeddings for a sequence of token IDs. Returns a tensor of shape
`[seq_len, d_model]`.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `token_ids` | `[]const u32` | Input token IDs |
| `allocator` | `Allocator` | Memory allocator for the result |

### `sinusoidalPositionalEncoding`

```zig
pub fn sinusoidalPositionalEncoding(
    seq_len: usize,
    d_model: usize,
    allocator: std.mem.Allocator,
) !Tensor(f32)
```

Generate sinusoidal positional encodings as described in "Attention Is All You
Need" (Vaswani et al., 2017). Returns a tensor of shape `[seq_len, d_model]`.

Even dimensions use `sin(pos / 10000^(2i/d))`, odd dimensions use the cosine.

### `rotaryPositionalEmbedding`

```zig
pub fn rotaryPositionalEmbedding(
    tensor: Tensor(f32),
    positions: []const usize,
    theta: f32,
    allocator: std.mem.Allocator,
) !Tensor(f32)
```

Apply Rotary Position Embedding (RoPE) in-place. This is the position encoding
used by LLaMA. It rotates pairs of dimensions by position-dependent angles,
encoding relative position information directly into query and key vectors.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `tensor` | `Tensor(f32)` | Input Q or K tensor, shape `[seq_len, d_head]` |
| `positions` | `[]const usize` | Absolute positions for each token |
| `theta` | `f32` | Base frequency (LLaMA default: 10000.0) |
| `allocator` | `Allocator` | Memory allocator |

**Returns:** tensor with RoPE applied.

---

## Error Types

- `TensorError.InvalidIndex` -- a token ID exceeds `vocab_size`.
- `TensorError.OutOfMemory`

---

## Usage Example

```zig
const emb = @import("zigllama").neural_primitives.embeddings;

// Create embedding table
var tok_emb = try emb.TokenEmbedding.init(32000, 4096, allocator);
defer tok_emb.deinit();

// Look up embeddings for a token sequence
const tokens = &[_]u32{ 1, 15043, 29892, 920, 526, 366 };
var hidden = try tok_emb.forward(tokens, allocator);
defer hidden.deinit();

// Apply RoPE
const positions = &[_]usize{ 0, 1, 2, 3, 4, 5 };
var with_rope = try emb.rotaryPositionalEmbedding(
    hidden, positions, 10000.0, allocator,
);
defer with_rope.deinit();
```

---

## Related Modules

- [`transformers.attention`](attention.md) -- Applies RoPE to Q/K before
  computing attention scores.
- [`models.llama`](llama.md) -- Configures `rope_theta` via `LLaMAConfig`.
- [`models.tokenizer`](tokenizer.md) -- Produces the token IDs fed to
  `TokenEmbedding.forward`.
