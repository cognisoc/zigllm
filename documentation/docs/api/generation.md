# inference.generation

## Module Path

```
zigllama.inference.generation
```

**Source file:** `src/inference/generation.zig`

---

## Public Types

### `SamplingStrategy`

```zig
pub const SamplingStrategy = enum {
    Greedy,
    TopK,
    TopP,
    Temperature,
    Combined,
};
```

| Variant | Behavior |
|---------|----------|
| `Greedy` | Always pick the highest-probability token |
| `TopK` | Sample from the `k` most probable tokens |
| `TopP` | Sample from the smallest set whose cumulative probability exceeds `p` |
| `Temperature` | Scale logits by `1/temperature` before sampling |
| `Combined` | Apply top-k, then top-p, then temperature (the default) |

### `GenerationConfig`

```zig
pub const GenerationConfig = struct {
    strategy: SamplingStrategy = .Combined,
    temperature: f32 = 0.7,
    top_k: u32 = 40,
    top_p: f32 = 0.9,
    max_tokens: u32 = 512,
    min_tokens: u32 = 1,
    stop_tokens: []const TokenId = &[_]TokenId{SpecialTokens.EOS},
    stop_strings: []const []const u8 = &[_][]const u8{},
    repetition_penalty: f32 = 1.1,
    length_penalty: f32 = 1.0,
    seed: ?u64 = null,
};
```

Controls every aspect of the generation loop.

### `StopReason`

```zig
pub const StopReason = enum {
    MaxTokens,
    StopToken,
    StopString,
    EndOfSequence,
};
```

### `GenerationResult`

```zig
pub const GenerationResult = struct {
    tokens: []TokenId,
    text: []const u8,
    log_probs: ?[]f32,
    stop_reason: StopReason,
    stats: GenerationStats,
};
```

Returned by `generate`. Contains the produced tokens, decoded text, optional
per-token log probabilities, the reason generation stopped, and timing
statistics.

### `TextGenerator`

```zig
pub const TextGenerator = struct {
    model: *LLaMAModel,
    tokenizer: *SimpleTokenizer,
    config: GenerationConfig,
    allocator: std.mem.Allocator,
};
```

High-level generation engine that ties together a model, tokenizer, and
sampling configuration.

---

## Public Functions

### `TextGenerator.init`

```zig
pub fn init(
    model: *LLaMAModel,
    tokenizer: *SimpleTokenizer,
    allocator: std.mem.Allocator,
    config: GenerationConfig,
) TextGenerator
```

Construct a generator. Does not allocate; the returned struct is ready to use
immediately.

### `TextGenerator.generate`

```zig
pub fn generate(
    self: *TextGenerator,
    prompt: []const u8,
) !GenerationResult
```

End-to-end text generation:

1. Encode `prompt` to token IDs.
2. Run the autoregressive generation loop with the configured sampling strategy.
3. Decode output tokens to text.
4. Return a `GenerationResult`.

### `GenerationConfig.creative`

```zig
pub fn creative() GenerationConfig
```

Preset: `temperature=1.0, top_k=0, top_p=0.95`. Good for creative writing.

### `GenerationConfig.balanced`

```zig
pub fn balanced() GenerationConfig
```

Preset: `temperature=0.7, top_k=40, top_p=0.9`. The default -- good balance
between coherence and variety.

### `GenerationConfig.deterministic`

```zig
pub fn deterministic() GenerationConfig
```

Preset: `strategy=.Greedy, temperature=0.0`. Always produces the same output
for the same input.

---

## Error Types

- `error{EmptyPrompt}` -- prompt string is empty.
- `error{ModelError}` -- forward pass failed.
- `error{OutOfMemory}`

---

## Usage Example

```zig
const gen = @import("zigllama").inference.generation;

var config = gen.GenerationConfig.balanced();
config.max_tokens = 256;

var generator = gen.TextGenerator.init(&model, &tokenizer, allocator, config);

const result = try generator.generate("Once upon a time");
defer allocator.free(result.tokens);
defer allocator.free(result.text);

std.debug.print("{s}\n", .{result.text});
std.debug.print("Stop reason: {}\n", .{result.stop_reason});
```

---

## Related Modules

- [`models.llama`](llama.md) -- The `LLaMAModel` driven by the generator.
- [`models.tokenizer`](tokenizer.md) -- Encodes prompts and decodes outputs.
- [`inference.kv_cache`](kv-cache.md) -- Speeds up autoregressive generation.
- [`inference.streaming`](streaming.md) -- Stream tokens as they are generated.
- [`inference.advanced_sampling`](advanced-sampling.md) -- Mirostat, typical,
  tail-free sampling strategies.
