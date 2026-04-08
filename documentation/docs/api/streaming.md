# inference.streaming

## Module Path

```
zigllama.inference.streaming
```

**Source file:** `src/inference/streaming.zig`

---

## Public Types

### `StreamStatus`

```zig
pub const StreamStatus = enum {
    Active,
    Paused,
    Complete,
    Error,
};
```

Current state of a streaming generation session.

### `TokenChunk`

```zig
pub const TokenChunk = struct {
    token_id: TokenId,
    text: []const u8,
    log_prob: ?f32,
    position: usize,
    is_special: bool,
};
```

A single token emitted by the streaming generator. Includes the decoded text
fragment and optional log probability.

### `StreamCallback`

```zig
pub const StreamCallback = *const fn (chunk: TokenChunk, context: ?*anyopaque) StreamAction;

pub const StreamAction = enum {
    Continue,
    Pause,
    Stop,
};
```

User-provided function invoked for each generated token. Return `Stop` to end
generation early or `Pause` to suspend.

### `StreamingConfig`

```zig
pub const StreamingConfig = struct {
    generation_config: GenerationConfig,
    buffer_size: usize = 16,
    flush_interval_ms: u64 = 50,
    emit_special_tokens: bool = false,
};
```

| Field | Description |
|-------|-------------|
| `generation_config` | Underlying generation parameters |
| `buffer_size` | Number of tokens buffered before flushing |
| `flush_interval_ms` | Maximum time between flushes |
| `emit_special_tokens` | Whether to emit BOS/EOS via the callback |

### `TokenBuffer`

```zig
pub const TokenBuffer = struct {
    buffer: []TokenChunk,
    head: usize,
    tail: usize,
    capacity: usize,
};
```

Ring buffer for token chunks, used internally to decouple generation speed from
callback throughput.

---

## Public Functions

### `streamWithCallback`

```zig
pub fn streamWithCallback(
    generator: *TextGenerator,
    prompt: []const u8,
    callback: StreamCallback,
    context: ?*anyopaque,
    config: StreamingConfig,
) !StreamStatus
```

Generate tokens from `prompt` and invoke `callback` for each one. Generation
continues until a stop condition is met, the callback returns `Stop`, or an
error occurs.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `generator` | `*TextGenerator` | Configured text generator |
| `prompt` | `[]const u8` | Input prompt |
| `callback` | `StreamCallback` | Per-token callback |
| `context` | `?*anyopaque` | User data passed to callback |
| `config` | `StreamingConfig` | Streaming parameters |

**Returns:** final `StreamStatus`.

### `TokenBuffer.init`

```zig
pub fn init(capacity: usize, allocator: std.mem.Allocator) !TokenBuffer
```

Allocate a ring buffer with the given capacity.

### `TokenBuffer.push`

```zig
pub fn push(self: *TokenBuffer, chunk: TokenChunk) !void
```

Append a chunk to the buffer. Returns `error{BufferFull}` if at capacity.

### `TokenBuffer.pop`

```zig
pub fn pop(self: *TokenBuffer) ?TokenChunk
```

Remove and return the oldest chunk, or `null` if the buffer is empty.

---

## Error Types

- `error{BufferFull}` -- token buffer capacity exceeded.
- `error{StreamInterrupted}` -- callback returned an error.
- Inherits all errors from `TextGenerator.generate`.

---

## Usage Example

```zig
const streaming = @import("zigllama").inference.streaming;

fn onToken(chunk: streaming.TokenChunk, _: ?*anyopaque) streaming.StreamAction {
    std.io.getStdOut().writer().print("{s}", .{chunk.text}) catch {};
    return .Continue;
}

const config = streaming.StreamingConfig{
    .generation_config = gen.GenerationConfig.balanced(),
    .buffer_size = 8,
};

const status = try streaming.streamWithCallback(
    &generator, "Tell me a story", onToken, null, config,
);
// status == .Complete
```

---

## Related Modules

- [`inference.generation`](generation.md) -- The `TextGenerator` that drives
  token production.
- [`inference.batching`](batching.md) -- Batch processing with per-request
  streaming.
