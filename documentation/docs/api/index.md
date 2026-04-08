# API Reference

## Overview

ZigLlama exposes its functionality through a layered module hierarchy defined in
`src/main.zig`. The architecture follows a bottom-up design where each layer
builds on the one below it:

1. **Foundation** -- Tensors, memory management, file formats, threading.
2. **Linear Algebra** -- SIMD matrix operations and quantization.
3. **Neural Primitives** -- Activations, normalization, embeddings.
4. **Transformers** -- Attention, feed-forward networks, transformer blocks.
5. **Models** -- Complete LLaMA architecture, configuration, tokenization, GGUF loading.
6. **Inference** -- Generation engine, caching, streaming, batching, profiling.

All public modules are reachable from the root import:

```zig
const zigllama = @import("zigllama");
const Tensor = zigllama.foundation.tensor.Tensor;
```

---

## Module Index

### Foundation

| Module | Source | Description |
|--------|--------|-------------|
| [`foundation.tensor`](tensor.md) | `src/foundation/tensor.zig` | Generic tensor type with multi-dimensional operations |
| [`foundation.memory_mapping`](memory-mapping.md) | `src/foundation/memory_mapping.zig` | Memory-mapped I/O (internal) |
| [`foundation.gguf_format`](gguf-format.md) | `src/foundation/gguf_format.zig` | Low-level GGUF format constants and helpers (internal) |
| [`foundation.blas_integration`](blas-integration.md) | `src/foundation/blas_integration.zig` | BLAS library interface (internal) |
| [`foundation.threading`](threading.md) | `src/foundation/threading.zig` | Thread pool, work stealing, and NUMA support (internal) |

### Linear Algebra

| Module | Source | Description |
|--------|--------|-------------|
| [`linear_algebra.matrix_ops`](matrix-ops.md) | `src/linear_algebra/matrix_ops.zig` | SIMD-accelerated matrix operations |
| [`linear_algebra.quantization`](quantization.md) | `src/linear_algebra/quantization.zig` | Quantization and dequantization framework |
| [`linear_algebra.k_quantization`](k-quantization.md) | `src/linear_algebra/k_quantization.zig` | K-quantization formats (internal) |
| [`linear_algebra.iq_quantization`](iq-quantization.md) | `src/linear_algebra/iq_quantization.zig` | Importance quantization formats (internal) |

### Neural Primitives

| Module | Source | Description |
|--------|--------|-------------|
| [`neural_primitives.activations`](activations.md) | `src/neural_primitives/activations.zig` | Activation functions (ReLU, GELU, SiLU, SwiGLU) |
| [`neural_primitives.normalization`](normalization.md) | `src/neural_primitives/normalization.zig` | Normalization layers (LayerNorm, RMSNorm) |
| [`neural_primitives.embeddings`](embeddings.md) | `src/neural_primitives/embeddings.zig` | Token and positional embeddings |

### Transformers

| Module | Source | Description |
|--------|--------|-------------|
| [`transformers.attention`](attention.md) | `src/transformers/attention.zig` | Multi-head attention mechanisms |
| [`transformers.feed_forward`](feed-forward.md) | `src/transformers/feed_forward.zig` | Feed-forward network variants |
| [`transformers.transformer_block`](transformer-block.md) | `src/transformers/transformer_block.zig` | Complete transformer blocks |

### Models

| Module | Source | Description |
|--------|--------|-------------|
| [`models.llama`](llama.md) | `src/models/llama.zig` | LLaMA model architecture |
| [`models.config`](config.md) | `src/models/config.zig` | Model configuration and presets |
| [`models.tokenizer`](tokenizer.md) | `src/models/tokenizer.zig` | Tokenization (Simple, BPE) |
| [`models.gguf`](gguf.md) | `src/models/gguf.zig` | GGUF file reader |

### Inference

| Module | Source | Description |
|--------|--------|-------------|
| [`inference.generation`](generation.md) | `src/inference/generation.zig` | Text generation engine |
| [`inference.kv_cache`](kv-cache.md) | `src/inference/kv_cache.zig` | Key-value cache for inference |
| [`inference.streaming`](streaming.md) | `src/inference/streaming.zig` | Streaming token generation |
| [`inference.batching`](batching.md) | `src/inference/batching.zig` | Batch request processing |
| [`inference.profiling`](profiling.md) | `src/inference/profiling.zig` | Performance profiling |
| [`inference.advanced_sampling`](advanced-sampling.md) | `src/inference/advanced_sampling.zig` | Advanced sampling methods (Mirostat, Typical, Tail-Free) |
| [`inference.grammar_constraints`](grammar-constraints.md) | `src/inference/grammar_constraints.zig` | Grammar-constrained generation |

---

## Conventions

- All allocating functions accept a `std.mem.Allocator` and return errors via
  Zig's error union syntax (`!T`).
- Types suffixed with `Error` are error sets specific to that module.
- Functions prefixed with `deinit` free resources owned by a struct instance.
- Modules marked **(internal)** are implementation details; their APIs may change
  between releases.

## Version

```zig
pub const version = std.SemanticVersion{
    .major = 0,
    .minor = 1,
    .patch = 0,
};
```
