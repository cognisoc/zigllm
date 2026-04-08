# inference.advanced_sampling

## Module Path

```
zigllama.inference.advanced_sampling
```

**Source file:** `src/inference/advanced_sampling.zig`

---

## Public Types

### `AdvancedSamplingType`

```zig
pub const AdvancedSamplingType = enum {
    Mirostat,
    Typical,
    TailFree,
    LocallyTypical,
    Classifier,
    Contrastive,
};
```

| Variant | Description |
|---------|-------------|
| `Mirostat` | Entropy-targeting sampler (Basu et al., 2021) |
| `Typical` | Typical sampling based on information content |
| `TailFree` | Removes low-probability tail tokens |
| `LocallyTypical` | Position-aware typical sampling |
| `Classifier` | Classifier-free guidance |
| `Contrastive` | Contrastive search (Su et al., 2022) |

### `MirostatConfig`

```zig
pub const MirostatConfig = struct {
    version: enum { V1, V2 },
    tau: f32,
    eta: f32,
    epsilon: f32,
    max_iterations: u32,
};
```

| Field | Description | Typical Value |
|-------|-------------|---------------|
| `version` | V1 (classic) or V2 (simplified) | `.V2` |
| `tau` | Target entropy / perplexity | 5.0 |
| `eta` | Learning rate for tau adaptation | 0.1 |
| `epsilon` | Convergence threshold | 1e-4 |
| `max_iterations` | Max adaptation steps per token | 100 |

### `TypicalConfig`

```zig
pub const TypicalConfig = struct {
    mass: f32,
    min_tokens: u32,
};
```

| Field | Description | Typical Value |
|-------|-------------|---------------|
| `mass` | Probability mass threshold (0.0--1.0) | 0.95 |
| `min_tokens` | Minimum tokens to keep regardless of mass | 1 |

### `TailFreeConfig`

```zig
pub const TailFreeConfig = struct {
    z: f32,
    min_tokens: u32,
};
```

| Field | Description | Typical Value |
|-------|-------------|---------------|
| `z` | Tail-free threshold (0.0--1.0); lower = more aggressive pruning | 0.95 |
| `min_tokens` | Minimum tokens to retain | 1 |

### `AdvancedSampler`

```zig
pub const AdvancedSampler = struct {
    allocator: std.mem.Allocator,
    rng: std.rand.DefaultPrng,
};
```

Stateful sampler that maintains an RNG and allocator for all advanced sampling
methods.

---

## Public Functions

### `AdvancedSampler.init`

```zig
pub fn init(allocator: std.mem.Allocator, seed: ?u64) AdvancedSampler
```

Create a sampler. If `seed` is `null`, the current timestamp is used.

### `AdvancedSampler.sampleMirostat`

```zig
pub fn sampleMirostat(
    self: *AdvancedSampler,
    logits: Tensor(f32),
    config: MirostatConfig,
) !u32
```

Mirostat sampling. Adaptively adjusts the effective temperature to maintain a
target perplexity, producing text with consistent quality regardless of prompt.

### `AdvancedSampler.sampleTypical`

```zig
pub fn sampleTypical(
    self: *AdvancedSampler,
    logits: Tensor(f32),
    config: TypicalConfig,
) !u32
```

Typical sampling. Selects tokens whose information content is close to the
expected information, filtering out both very common and very rare tokens.

### `AdvancedSampler.sampleTailFree`

```zig
pub fn sampleTailFree(
    self: *AdvancedSampler,
    logits: Tensor(f32),
    config: TailFreeConfig,
) !u32
```

Tail-free sampling. Computes the second derivative of the sorted probability
distribution and removes tokens in the "tail" where the second derivative is
near zero.

### `AdvancedSampler.sampleContrastive`

```zig
pub fn sampleContrastive(
    self: *AdvancedSampler,
    logits: Tensor(f32),
    hidden_states: Tensor(f32),
    config: ContrastiveConfig,
) !u32
```

Contrastive search. Balances token probability with a degeneration penalty
based on cosine similarity between the candidate token's hidden state and
previous hidden states.

---

## Error Types

- `TensorError.InvalidShape` -- logits tensor has wrong dimensions.
- `error{OutOfMemory}`

---

## Usage Example

```zig
const adv = @import("zigllama").inference.advanced_sampling;

var sampler = adv.AdvancedSampler.init(allocator, 42);

// Mirostat v2 -- maintain target perplexity of 5.0
const token = try sampler.sampleMirostat(logits, .{
    .version = .V2,
    .tau = 5.0,
    .eta = 0.1,
    .epsilon = 1e-4,
    .max_iterations = 100,
});

// Typical sampling
const typical_token = try sampler.sampleTypical(logits, .{
    .mass = 0.95,
    .min_tokens = 1,
});
```

---

## Related Modules

- [`inference.generation`](generation.md) -- Standard sampling strategies
  (top-k, top-p, temperature).
- [`inference.grammar_constraints`](grammar-constraints.md) -- Can be combined
  with advanced sampling to constrain output format.
