---
title: "Tutorial: Your First Inference"
description: "Step-by-step guide to loading a model, tokenising a prompt, generating text, and decoding output in ZigLlama."
---

# Tutorial: Your First Inference

This tutorial walks through the minimum code required to generate text with
ZigLlama.  By the end you will have a working program that loads a LLaMA 7B
configuration, tokenises a prompt, runs autoregressive generation, and prints
the decoded output.

**Prerequisites:** Zig 0.13+, ZigLlama cloned and building.

**Estimated time:** 15 minutes.

---

## Step 1: Create an Allocator

Zig requires explicit memory management.  ZigLlama uses the standard
`GeneralPurposeAllocator` (GPA), which provides leak detection in debug builds.

```zig
const std = @import("std");
const Allocator = std.mem.Allocator;

pub fn main() !void {
    // The GPA tracks every allocation and reports leaks on deinit.
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const status = gpa.deinit();
        if (status == .leak) {
            std.log.err("Memory leak detected!", .{});
        }
    }
    const allocator = gpa.allocator();

    // ... next steps use `allocator` everywhere ...
}
```

!!! info "Why not `page_allocator`?"
    `page_allocator` requests memory directly from the OS in page-sized
    chunks and never tracks individual allocations.  GPA is slower but
    catches leaks, double-frees, and use-after-free -- invaluable during
    development.

---

## Step 2: Configure the Model

ZigLlama provides preset configurations for every supported LLaMA size.
We select 7B, which defines $d_\text{model}=4096$, 32 layers, 32 attention
heads, and a vocabulary of 32 000 tokens.

```zig
const models = @import("models/llama.zig");
const config_mod = @import("models/config.zig");

const config = models.LLaMAConfig.init(.LLaMA_7B);
std.log.info("Model: {s}", .{config_mod.ModelSize.LLaMA_7B.name()});
std.log.info("Parameters: {d:.1f}B", .{config_mod.ModelSize.LLaMA_7B.parameterCount()});
std.log.info("d_model={d}, layers={d}, heads={d}", .{
    config.d_model, config.num_layers, config.num_heads,
});
```

!!! tip "Smaller models for experimentation"
    During development you can shrink the config by overriding fields:
    ```zig
    var small_config = models.LLaMAConfig.init(.LLaMA_7B);
    small_config.num_layers = 2;     // only 2 transformer blocks
    small_config.d_model = 256;      // narrow embedding
    small_config.num_heads = 4;
    small_config.d_ff = 512;
    ```
    This creates a toy model that initialises in milliseconds.

---

## Step 3: Initialise the Model

`LLaMAModel.init` allocates embedding tables, transformer blocks, and the
final output projection:

```zig
var model = try models.LLaMAModel.init(allocator, config);
defer model.deinit();
```

At this point the model weights are **randomly initialised**.  In a production
setting you would load weights from a GGUF file (see
[GGUF Model Loading](../models/gguf-loading.md)).

---

## Step 4: Create a Tokenizer

The `SimpleTokenizer` maps between text and integer token IDs:

```zig
const tokenizer_mod = @import("models/tokenizer.zig");

var tokenizer = try tokenizer_mod.SimpleTokenizer.init(allocator, config.vocab_size);
defer tokenizer.deinit();
```

`SimpleTokenizer` provides a word-level tokeniser suitable for educational
demos.  For production accuracy, load a SentencePiece vocabulary from the model
file.

---

## Step 5: Create a TextGenerator

The `TextGenerator` ties the model, tokeniser, and sampling configuration
together:

```zig
const generation = @import("inference/generation.zig");

var generator = generation.TextGenerator.init(&model, &tokenizer, allocator, null);

// Use the balanced preset (temperature 0.7, top-k 40, top-p 0.9)
try generator.setConfig(generation.GenerationConfig.balanced());
```

Four presets are available:

| Preset | Temperature | Top-k | Top-p | Use Case |
|--------|------------|-------|-------|----------|
| `creative()` | 0.9 | 50 | 0.95 | Poetry, fiction |
| `balanced()` | 0.7 | 40 | 0.9 | General Q&A |
| `focused()` | 0.3 | 20 | 0.8 | Factual, code |
| `deterministic()` | 0.0 | 1 | 1.0 | Reproducible output |

---

## Step 6: Generate Text

Call `generate` with a prompt string.  The engine performs autoregressive
decoding: tokenise the prompt, run a forward pass, sample a token, append it,
and repeat until a stop condition is met.

```zig
const prompt = "The transformer architecture";
std.log.info("Prompt: \"{s}\"", .{prompt});

const result = try generator.generate(prompt);
defer result.deinit(allocator);
```

`GenerationResult` contains:

| Field | Type | Description |
|-------|------|-------------|
| `tokens` | `[]TokenId` | All generated token IDs (including the prompt). |
| `text` | `?[]u8` | Decoded string, if a tokeniser was provided. |
| `log_probs` | `[]f32` | Per-token log-probability. |
| `num_tokens` | `u32` | Number of newly generated tokens. |
| `stop_reason` | `StopReason` | Why generation stopped. |
| `stats` | `GenerationStats` | Timing: tokens/sec, ms/token. |

---

## Step 7: Decode and Display

```zig
if (result.text) |text| {
    std.log.info("Generated text: {s}", .{text});
}

std.log.info("Tokens generated: {d}", .{result.num_tokens});
std.log.info("Stop reason: {s}", .{result.stop_reason.description()});
std.log.info("Tokens/sec: {d:.1f}", .{result.stats.tokens_per_second});
std.log.info("Time/token: {d:.1f} ms", .{result.stats.time_per_token_ms});
```

---

## Complete Program

Putting it all together:

```zig
const std = @import("std");
const models = @import("models/llama.zig");
const tokenizer_mod = @import("models/tokenizer.zig");
const generation = @import("inference/generation.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Configure and initialise
    const config = models.LLaMAConfig.init(.LLaMA_7B);
    var model = try models.LLaMAModel.init(allocator, config);
    defer model.deinit();

    var tokenizer = try tokenizer_mod.SimpleTokenizer.init(allocator, config.vocab_size);
    defer tokenizer.deinit();

    // Create generator with balanced sampling
    var generator = generation.TextGenerator.init(&model, &tokenizer, allocator, null);
    try generator.setConfig(generation.GenerationConfig.balanced());

    // Generate
    const result = try generator.generate("The transformer architecture");
    defer result.deinit(allocator);

    // Output
    if (result.text) |text| {
        std.debug.print("Output: {s}\n", .{text});
    }
    std.debug.print("Tokens: {d}, Speed: {d:.1f} tok/s\n", .{
        result.num_tokens, result.stats.tokens_per_second,
    });
}
```

!!! warning "Random weights"
    Because the model is randomly initialised, the generated text will be
    incoherent.  This tutorial demonstrates the **plumbing**; to get
    meaningful output, load real weights from a GGUF file as described in
    [GGUF Model Loading](../models/gguf-loading.md).

---

## What to Try Next

- **Change the sampling preset** to `creative()` or `deterministic()` and
  observe how the output distribution shifts.
- **Reduce `max_tokens`** to 10 and inspect the per-token log-probabilities
  in `result.log_probs`.
- **Load a real model** from a GGUF file and generate coherent text.
- **Move on to** [Understanding Attention](understanding-attention.md) to see
  what happens *inside* each forward pass.
