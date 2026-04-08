---
title: "Demo Walkthroughs"
description: "Annotated walkthrough of all twelve ZigLlama example programs, with instructions for running each and explanations of the key concepts they illustrate."
---

# Demo Walkthroughs

This page provides a narrated tour of every example program in the `examples/`
directory.  For each demo we cover: what it demonstrates, how to run it, key
concepts illustrated, and expected output.

---

## 1. simple_demo.zig -- Architecture Summary

**What it demonstrates:** A birds-eye view of ZigLlama's six-layer
architecture, printing capabilities and statistics for each layer.

**How to run:**

```bash
zig run examples/simple_demo.zig
```

**Key concepts:**

- The progressive layer model (Foundation through Inference).
- Parameter counts and test coverage.
- Parity analysis against llama.cpp (~90% educational, ~40% production).

**Expected output (excerpt):**

```
Layer 1: Foundation
   Multi-dimensional tensors with efficient memory layout
Layer 2: Linear Algebra
   SIMD acceleration (AVX, AVX2, NEON auto-detection)
...
Performance: ~5ms/token with all optimizations
```

---

## 2. educational_demo.zig -- Progressive Walkthrough

**What it demonstrates:** An interactive tour through every layer, creating
objects at each level and showing their relationships.

**How to run:**

```bash
zig run examples/educational_demo.zig
```

**Key concepts:**

- `Tensor` creation and element access (Layer 1).
- SIMD-accelerated matrix multiplication (Layer 2).
- Activation functions: SwiGLU, ReLU, GELU (Layer 3).
- Multi-head attention forward pass (Layer 4).
- `LLaMAModel` instantiation and configuration (Layer 5).
- `TextGenerator` with different sampling strategies (Layer 6).

!!! tip "Best starting point"
    If you read only one example, read this one.  It covers the full stack
    in 50 lines of `main` plus per-layer helper functions.

---

## 3. benchmark_demo.zig -- Matrix Multiplication Benchmarks

**What it demonstrates:** Performance comparison between naive $O(n^3)$
matrix multiplication and SIMD-accelerated kernels across varying matrix
sizes.

**How to run:**

```bash
zig run examples/benchmark_demo.zig -OReleaseFast
```

!!! warning "Optimisation level matters"
    Run with `-OReleaseFast` to get meaningful timings.  Debug builds
    include bounds checking on every element access.

**Key concepts:**

- `Matrix` struct with flat `[]f32` storage and row-major layout.
- `matmulNaive`: triple loop baseline.
- `matmulSIMD`: uses Zig's `@Vector` for architecture-portable SIMD
  (4-wide on both x86_64 and aarch64).
- Timer-based benchmarking with multiple sizes (64, 256, 1024, 4096).

**Expected output (example on x86_64 with AVX2):**

```
Size   Naive (ms)  SIMD (ms)  Speedup
  64       0.02       0.01     2.0x
 256       2.10       0.70     3.0x
1024     540.00     125.00     4.3x
4096       ---     8000.00      ---
```

---

## 4. model_architectures_demo.zig -- Multi-Architecture Support

**What it demonstrates:** Instantiation of multiple transformer architectures
(LLaMA, GPT-2, Mistral, Falcon, etc.) and comparison of their configurations.

**How to run:**

```bash
zig run examples/model_architectures_demo.zig
```

**Key concepts:**

- `ModelSize` enum and `LLaMAConfig.init` for standard sizes.
- Architectural differences: number of heads, FFN multiplier, normalization type.
- ZigLlama's support for 18 distinct architectures.

---

## 5. gguf_demo.zig -- GGUF Format Exploration

**What it demonstrates:** Parsing a GGUF file header, reading metadata
key-value pairs, and listing tensor descriptors.

**How to run:**

```bash
zig run examples/gguf_demo.zig -- path/to/model.gguf
```

**Key concepts:**

- GGUF magic number (`0x46475547`) and version field.
- Metadata types: string, uint32, float32, array.
- Tensor descriptor: name, shape, data type, offset.
- Memory-mapped I/O for zero-copy tensor access.

---

## 6. parity_demo.zig -- llama.cpp Comparison

**What it demonstrates:** Side-by-side comparison of ZigLlama outputs against
llama.cpp reference values for attention, softmax, RMSNorm, and generation.

**How to run:**

```bash
zig run examples/parity_demo.zig
```

**Key concepts:**

- Numerical tolerance thresholds for floating-point comparison.
- Component-level parity: which operations match llama.cpp exactly and which
  diverge.
- The educational-vs-production parity gap and where it comes from.

---

## 7. multi_modal_demo.zig -- Vision-Language Pipeline

**What it demonstrates:** Encoding an image through a vision transformer,
projecting the visual tokens into the text embedding space, and generating a
caption.

**How to run:**

```bash
zig run examples/multi_modal_demo.zig
```

**Key concepts:**

- Image patch embedding (splitting an image into 16x16 patches).
- Cross-modal projection layer.
- Interleaved visual and text tokens in the decoder.

---

## 8. multi_modal_concepts_demo.zig -- Fusion Strategies

**What it demonstrates:** Conceptual overview of different multi-modal fusion
approaches: early fusion, late fusion, and cross-attention fusion.

**How to run:**

```bash
zig run examples/multi_modal_concepts_demo.zig
```

**Key concepts:**

- Early fusion: concatenate modalities before the first transformer block.
- Late fusion: process modalities independently and combine outputs.
- Cross-attention fusion: use visual tokens as keys/values in text-decoder
  cross-attention layers.

---

## 9. threading_demo.zig -- Parallel Computation

**What it demonstrates:** Thread-pool creation, parallel matrix multiplication,
and NUMA-aware memory allocation.

**How to run:**

```bash
zig run examples/threading_demo.zig
```

**Key concepts:**

- `std.Thread` and `std.Thread.Pool` usage in Zig.
- Work partitioning: splitting rows across threads.
- Synchronisation via atomics.
- NUMA topology detection and thread pinning.

---

## 10. chat_templates_demo.zig -- Template System

**What it demonstrates:** Applying six different chat-template formats to the
same multi-turn conversation and printing the formatted prompts.

**How to run:**

```bash
zig run examples/chat_templates_demo.zig
```

**Key concepts:**

- `ChatTemplateManager.init`, `loadTemplate`, `applyTemplate`.
- `ChatMessage` struct with `role` and `content`.
- Template auto-detection from model name.
- Comparison of formatting across LLaMA 2, LLaMA 3, ChatML, Mistral,
  Alpaca, and Claude styles.

**Expected output (excerpt):**

```
Template: ChatML
==================================================
<|im_start|>system
You are a helpful AI assistant that explains complex topics clearly.<|im_end|>
<|im_start|>user
Can you explain how transformers work in neural networks?<|im_end|>
...
```

---

## 11. perplexity_demo.zig -- Evaluation and Benchmarking

**What it demonstrates:** Configuring the perplexity evaluator with different
presets, running evaluations, and interpreting the results.

**How to run:**

```bash
zig run examples/perplexity_demo.zig
```

**Key concepts:**

- `PerplexityConfig` presets: default, high-precision, fast, temperature-scaled.
- `BenchmarkSuite.generateSyntheticDataset` for reproducible testing.
- `BenchmarkResults.printReport` for formatted output.
- Saving results to JSON for offline analysis.

---

## 12. main.zig -- Master Demo

**What it demonstrates:** A curated selection of highlights from every other
example, designed to run in under 5 minutes and touch every major subsystem.

**How to run:**

```bash
zig build run
# or
zig run examples/main.zig
```

**Key concepts:**

- End-to-end execution from tensor creation to text generation.
- Summary statistics: test count, parity percentages, performance numbers.
- Serves as a quick smoke test after code changes.

---

## Running All Examples

To compile and verify that every example builds:

```bash
for f in examples/*.zig; do
    echo "Building $f ..."
    zig build-exe "$f" --check 2>&1 | head -1
done
```

!!! info "Build system integration"
    The `build.zig` file defines named build steps for each example.
    Use `zig build --help` to see available targets.
