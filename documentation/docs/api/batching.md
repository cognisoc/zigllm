# inference.batching

## Module Path

```
zigllama.inference.batching
```

**Source file:** `src/inference/batching.zig`

---

## Public Types

### `BatchingStrategy`

```zig
pub const BatchingStrategy = enum {
    FixedSize,
    DynamicTimeout,
    Adaptive,
    Continuous,
};
```

| Variant | Behavior |
|---------|----------|
| `FixedSize` | Wait for exactly N requests before processing |
| `DynamicTimeout` | Process when batch is full or a timeout expires |
| `Adaptive` | Adjust batch size based on queue pressure and latency targets |
| `Continuous` | Process requests as they arrive, inserting into running batches |

### `BatchRequest`

```zig
pub const BatchRequest = struct {
    id: u64,
    prompt: []const u8,
    config: GenerationConfig,
    priority: u8,
    callback: ?StreamCallback,
};
```

A single generation request submitted to the batch processor.

### `BatchResult`

```zig
pub const BatchResult = struct {
    id: u64,
    result: GenerationResult,
    latency_ms: u64,
    queue_time_ms: u64,
};
```

Result for one request within a batch, including timing metadata.

### `BatchProcessor`

```zig
pub const BatchProcessor = struct {
    queue: std.ArrayList(BatchRequest),
    workers: []std.Thread,
    config: BatchConfig,
    stats: BatchStats,
    model: *LLaMAModel,
    tokenizer: *SimpleTokenizer,
    allocator: std.mem.Allocator,
};
```

Manages a request queue and a pool of worker threads that process batches of
requests against a shared model.

### `BatchConfig`

```zig
pub const BatchConfig = struct {
    max_batch_size: usize = 32,
    strategy: BatchingStrategy = .DynamicTimeout,
    timeout_ms: u64 = 100,
    num_workers: usize = 1,
    max_queue_size: usize = 1024,
};
```

### `BatchStats`

```zig
pub const BatchStats = struct {
    total_requests: u64,
    total_tokens: u64,
    avg_latency_ms: f64,
    avg_throughput_tps: f64,
};
```

---

## Public Functions

### `BatchProcessor.init`

```zig
pub fn init(
    model: *LLaMAModel,
    tokenizer: *SimpleTokenizer,
    config: BatchConfig,
    allocator: std.mem.Allocator,
) !BatchProcessor
```

Create the batch processor and start worker threads.

### `BatchProcessor.deinit`

```zig
pub fn deinit(self: *BatchProcessor) void
```

Drain the queue, join worker threads, and free resources.

### `BatchProcessor.submit`

```zig
pub fn submit(self: *BatchProcessor, request: BatchRequest) !BatchResult
```

Submit a request and block until the result is ready. The request may be
batched with others for higher throughput.

### `BatchProcessor.submitAsync`

```zig
pub fn submitAsync(self: *BatchProcessor, request: BatchRequest) !u64
```

Submit a request without blocking. Returns a request ID that can be used to
poll for the result later.

### `BatchProcessor.processQueue`

```zig
pub fn processQueue(self: *BatchProcessor) !void
```

Internal: called by worker threads to dequeue and process a batch of requests.

---

## Error Types

- `error{QueueFull}` -- the request queue has reached `max_queue_size`.
- `error{ProcessorStopped}` -- the batch processor has been shut down.
- Inherits generation errors from `TextGenerator`.

---

## Usage Example

```zig
const batch = @import("zigllama").inference.batching;

var processor = try batch.BatchProcessor.init(
    &model,
    &tokenizer,
    .{ .max_batch_size = 8, .strategy = .DynamicTimeout },
    allocator,
);
defer processor.deinit();

const result = try processor.submit(.{
    .id = 1,
    .prompt = "Explain quantum computing",
    .config = gen.GenerationConfig.balanced(),
    .priority = 0,
    .callback = null,
});

std.debug.print("Response: {s}\n", .{result.result.text});
std.debug.print("Latency: {} ms\n", .{result.latency_ms});
```

---

## Related Modules

- [`inference.generation`](generation.md) -- Underlying generation engine used
  by each worker.
- [`inference.kv_cache`](kv-cache.md) -- `MultiSequenceKVCache` manages
  per-request caches.
- [`inference.streaming`](streaming.md) -- Per-request streaming via the
  `callback` field.
