# inference.kv_cache

## Module Path

```
zigllama.inference.kv_cache
```

**Source file:** `src/inference/kv_cache.zig`

---

## Public Types

### `CacheStrategy`

```zig
pub const CacheStrategy = enum {
    Always,
    LongSequenceOnly,
    Adaptive,
    Disabled,
};
```

| Variant | Behavior |
|---------|----------|
| `Always` | Cache K/V for every token at every layer |
| `LongSequenceOnly` | Enable caching only when sequence length exceeds a threshold |
| `Adaptive` | Dynamically enable/disable based on available memory |
| `Disabled` | No caching; recompute K/V each step (saves memory) |

### `KVCacheEntry`

```zig
pub const KVCacheEntry = struct {
    key: Tensor(f32),
    value: Tensor(f32),
    position: usize,
};
```

Cached key and value tensors for a single position in a single layer.

### `LayerKVCache`

```zig
pub const LayerKVCache = struct {
    entries: std.ArrayList(KVCacheEntry),
    max_seq_len: usize,
    head_dim: usize,
    num_heads: usize,
};
```

K/V cache for one transformer layer.

### `ModelKVCache`

```zig
pub const ModelKVCache = struct {
    layers: []LayerKVCache,
    num_layers: usize,
    strategy: CacheStrategy,
    allocator: std.mem.Allocator,
};
```

Aggregate cache spanning all layers of the model.

### `MultiSequenceKVCache`

```zig
pub const MultiSequenceKVCache = struct {
    sequences: std.AutoHashMap(u64, ModelKVCache),
    max_sequences: usize,
};
```

Manages independent caches for multiple concurrent sequences (e.g., different
requests in a batch).

### `SlidingWindowKVCache`

```zig
pub const SlidingWindowKVCache = struct {
    cache: LayerKVCache,
    window_size: usize,
};
```

Fixed-size sliding window that evicts the oldest entries when the window is
full. Used by Mistral-style attention.

---

## Public Functions

### `ModelKVCache.init`

```zig
pub fn init(
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    strategy: CacheStrategy,
    allocator: std.mem.Allocator,
) !ModelKVCache
```

Allocate the full cache structure. Memory for individual entries is allocated
lazily as tokens are generated.

### `ModelKVCache.deinit`

```zig
pub fn deinit(self: *ModelKVCache) void
```

Free all cached tensors and layer structures.

### `LayerKVCache.append`

```zig
pub fn append(self: *LayerKVCache, key: Tensor(f32), value: Tensor(f32)) !void
```

Add a new K/V pair at the next position.

### `LayerKVCache.get`

```zig
pub fn get(self: LayerKVCache, position: usize) ?KVCacheEntry
```

Retrieve the cached K/V pair at the given position. Returns `null` if the
position has not been cached.

### `ModelKVCache.clear`

```zig
pub fn clear(self: *ModelKVCache) void
```

Discard all cached entries across all layers. The cache structure itself is
retained and can be reused.

### `ModelKVCache.compact`

```zig
pub fn compact(self: *ModelKVCache) !void
```

Reclaim memory by removing evicted entries and shrinking internal arrays.

---

## Error Types

- `error{CacheFull}` -- the cache has reached `max_seq_len` and the strategy
  does not allow eviction.
- `error{OutOfMemory}`

---

## Usage Example

```zig
const kvc = @import("zigllama").inference.kv_cache;

var cache = try kvc.ModelKVCache.init(
    32,     // num_layers
    32,     // num_heads
    128,    // head_dim (4096 / 32)
    2048,   // max_seq_len
    .Always,
    allocator,
);
defer cache.deinit();

// During generation, append K/V after each attention computation
try cache.layers[layer_idx].append(key_tensor, value_tensor);

// Retrieve cached K/V for attention
if (cache.layers[layer_idx].get(position)) |entry| {
    // Use entry.key and entry.value
}

// Reset between prompts
cache.clear();
```

---

## Related Modules

- [`transformers.attention`](attention.md) -- Produces the K/V tensors that are
  cached.
- [`inference.generation`](generation.md) -- Uses the cache during
  autoregressive generation.
- [`inference.batching`](batching.md) -- `MultiSequenceKVCache` supports
  batched inference.
