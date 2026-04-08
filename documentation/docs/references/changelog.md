---
title: "Changelog"
description: "Milestone-based changelog for the ZigLlama project, from the initial foundation layer through the current release."
---

# Changelog

This changelog documents ZigLlama's development milestones.  Each version
represents a complete architectural layer, following the project's progressive
bottom-up design.  Within each milestone, components are listed in dependency
order -- foundational pieces first, higher-level abstractions second.

---

## v0.6.0 -- Inference Engine

*Layer 6: Text generation, sampling, caching, streaming, and batch processing.*

This milestone completes the full inference stack, enabling end-to-end text
generation from a loaded model.

### Added

- **Text Generation** (`text_generation.zig`).  Autoregressive token-by-token
  generation with configurable stopping criteria (max tokens, EOS token,
  stop sequences).
- **Sampling Strategies** (`sampling.zig`).  Eight decoding strategies:
    - Greedy (argmax).
    - Top-k sampling with configurable \( k \).
    - Top-p (nucleus) sampling with cumulative probability threshold.
    - Temperature scaling.
    - Mirostat v1 and v2 adaptive sampling.
    - Locally typical sampling.
    - Tail-free sampling.
- **Repetition Penalty.**  Logit-based penalty for previously generated tokens,
  with configurable penalty factor and lookback window.
- **KV Cache** (`kv_cache.zig`).  Pre-allocated key-value cache supporting
  incremental updates.  Per-token cost drops from \( O(T) \) to \( O(1) \)
  for previously computed positions.
- **Streaming** (`streaming.zig`).  Callback-based token emission for
  real-time output.  Tokens are delivered to the caller as soon as they are
  decoded, before the full sequence is complete.
- **Batch Processing** (`batch_processing.zig`).  Concurrent processing of
  multiple independent sequences with length-based grouping for efficient
  padding.
- **Grammar-Constrained Decoding.**  BNF-based grammar constraints that mask
  invalid tokens at each generation step, ensuring output conforms to a
  specified grammar (e.g., JSON, code).
- **Profiling and Diagnostics.**  Per-layer timing, memory-usage tracking,
  and token-throughput reporting for performance analysis.

---

## v0.5.0 -- Model Architectures

*Layer 5: Eighteen model families, GGUF loading, tokenisation, and chat
templates.*

This milestone adds support for loading and running real pre-trained models.

### Added

- **LLaMA / LLaMA 2** (`llama.zig`).  The primary reference architecture:
  RMSNorm, SwiGLU, RoPE, pre-norm blocks.  Supports 7B, 13B, 33B, and 65B
  configurations.  Llama 2 adds grouped-query attention.
- **Mistral** (`mistral.zig`).  Sliding window attention with configurable
  window size, GQA, and SwiGLU.
- **GPT-2** (`gpt2.zig`).  Post-norm transformer with GELU activation and
  learned positional embeddings.
- **Falcon** (`falcon.zig`).  Multi-query attention, ALiBi positional encoding,
  and parallel attention-FFN computation.
- **Qwen** (`qwen.zig`).  RoPE with NTK-aware scaling, SwiGLU, and RMSNorm.
- **Phi** (`phi.zig`).  Partial RoPE application, dense attention, and
  GELU activation.
- **GPT-J** (`gptj.zig`).  Rotary embeddings applied to a subset of head
  dimensions, parallel attention-FFN layout.
- **GPT-NeoX** (`gpt_neox.zig`).  Full rotary embeddings with parallel
  attention-FFN and post-norm configuration.
- **BLOOM** (`bloom.zig`).  ALiBi positional encoding, LayerNorm, and GELU
  activation.
- **Mamba** (`mamba.zig`).  Selective state-space model with linear-time
  sequence processing, selective scan mechanism.
- **BERT** (`bert.zig`).  Bidirectional encoder with masked language modelling,
  segment embeddings, and `[CLS]`/`[SEP]` tokens.
- **Gemma** (`gemma.zig`).  GeGLU activation, RoPE, and RMSNorm with
  Google's weight initialisation conventions.
- **StarCoder** (`starcoder.zig`).  Multi-query attention, fill-in-the-middle
  support, and code-specific tokenisation.
- **Mixture of Experts** (`mixture_of_experts.zig`).  Top-k expert routing
  with load-balancing, configurable number of experts and active expert count.
- **Multi-Modal** (`multi_modal.zig`).  Vision-language architecture with
  ViT image encoder, projection layer, and cross-attention to the language
  model.
- **Model Converter** (`model_converter.zig`).  Conversion between
  quantisation formats for existing GGUF files.
- **Perplexity Evaluation** (`perplexity.zig`).  Token-level log-likelihood
  computation for model quality assessment.
- **GGUF Loader** (`gguf_format.zig`).  Version 3 format reader with typed
  metadata extraction, tensor descriptor parsing, and alignment validation.
- **Tokenizer** (`tokenizer.zig`).  SentencePiece (LLaMA), byte-level BPE
  (GPT-2), and WordPiece (BERT) support with encode/decode round-trip
  fidelity.
- **Model Configuration** (`model_config.zig`).  Unified configuration struct
  covering all 18 architectures with architecture-specific parameter
  validation.
- **Chat Templates.**  Jinja-style template rendering for instruction-tuned
  models (Llama 2 Chat, Mistral Instruct, ChatML).

---

## v0.4.0 -- Transformer Components

*Layer 4: Multi-head attention, feed-forward networks, and complete
transformer blocks.*

### Added

- **Multi-Head Attention** (`multi_head_attention.zig`).  Full MHA with
  support for MHA, GQA, and MQA configurations.  Includes:
    - Q/K/V linear projections.
    - Scaled dot-product attention with causal masking.
    - Output projection and concatenation.
    - Sliding window attention (Mistral).
    - ALiBi bias computation (BLOOM, Falcon).
- **Feed-Forward Networks** (`feed_forward.zig`).  Standard two-layer FFN
  and gated variants:
    - Standard: \( W_2 \cdot \sigma(W_1 x + b_1) + b_2 \).
    - SwiGLU: \( W_2 \cdot (\text{SiLU}(W_1 x) \odot W_3 x) \).
    - GeGLU: \( W_2 \cdot (\text{GELU}(W_1 x) \odot W_3 x) \).
- **Transformer Blocks** (`transformer_block.zig`).  Composable block
  abstraction supporting:
    - Pre-norm and post-norm ordering.
    - Parallel and sequential attention-FFN layout.
    - Configurable normalization (RMSNorm, LayerNorm).
    - Configurable attention type (MHA, GQA, MQA).

---

## v0.3.0 -- Neural Primitives

*Layer 3: Activation functions, normalization layers, embeddings, and
positional encodings.*

### Added

- **Activation Functions** (`activation_functions.zig`).
    - ReLU, Leaky ReLU, GELU (exact and fast approximation).
    - Sigmoid, Tanh, SiLU (Swish).
    - SwiGLU and GeGLU gated activations.
    - Softmax with log-sum-exp numerical stabilisation.
- **Normalization** (`normalization.zig`).
    - Layer Normalization (Ba et al., 2016).
    - RMS Normalization (Zhang & Sennrich, 2019).
    - Configurable epsilon, learnable scale and bias parameters.
- **Embeddings** (`embeddings.zig`).
    - Token embedding lookup.
    - Learned positional embeddings (GPT-2).
    - Segment embeddings (BERT).
    - Tied embedding / un-embedding weight sharing.
- **Rotary Position Embeddings** (`rope.zig`).
    - Standard RoPE with configurable base frequency.
    - NTK-aware frequency scaling for context extension.
    - Partial application (Phi-style, applied to a subset of dimensions).
    - YaRN interpolation support.

---

## v0.2.0 -- Linear Algebra

*Layer 2: SIMD-accelerated matrix operations and quantisation formats.*

### Added

- **SIMD Operations** (`simd_operations.zig`).
    - Vectorised dot product, matrix-vector multiply, matrix-matrix multiply.
    - Cache-blocking for L1/L2 residency.
    - Auto-vectorisation targeting AVX2, AVX-512, and NEON via Zig `@Vector`.
- **Basic Quantization** (`quantization.zig`).
    - Q4_0: 4-bit quantisation with per-block scale.
    - Q4_1: 4-bit with per-block scale and minimum.
    - Q5_0, Q5_1: 5-bit variants.
    - Q8_0, Q8_1: 8-bit quantisation.
    - Quantise and dequantise routines for all formats.
- **K-Quantization** (`k_quantization.zig`).
    - Q2_K, Q3_K, Q4_K, Q5_K, Q6_K formats.
    - Super-block structure with nested scale quantisation.
    - SIMD-accelerated dequantisation kernels.
- **IQ-Quantization** (`iq_quantization.zig`).
    - IQ1_S, IQ1_M: ultra-low 1-bit formats.
    - IQ2_XXS, IQ2_XS, IQ2_S: 2-bit formats.
    - IQ3_XXS, IQ3_S: 3-bit formats.
    - IQ4_NL, IQ4_XS: 4-bit non-linear formats.
    - Importance-based weight selection.
    - Lookup-table dequantisation.

---

## v0.1.0 -- Foundation Layer

*Layer 1: Core data structures, memory management, model file I/O, and
threading.*

### Added

- **Tensor Operations** (`tensor.zig`).
    - Generic `Tensor(T)` struct with compile-time type specialisation.
    - Row-major memory layout with explicit shape and stride.
    - Element-wise arithmetic, slicing, reshaping, transposition.
    - Bounds-checked access with optional runtime safety.
- **Memory Management.**
    - Allocator-explicit design throughout the codebase.
    - `defer` / `errdefer` resource cleanup patterns.
    - Arena allocator for per-inference scratch memory.
- **Memory-Mapped I/O** (`memory_mapping.zig`).
    - `MemoryMap` abstraction over POSIX `mmap`.
    - `ModelFileMapper` for zero-copy model weight access.
    - Automatic page alignment and size calculation.
- **GGUF Binary Format** (`gguf_format.zig`).
    - Version 3 header parsing (magic, version, tensor count, metadata count).
    - Typed metadata key-value store (uint8 through string arrays).
    - Tensor descriptor parsing (name, dimensions, type, offset).
    - Alignment-aware data section access.
- **BLAS Integration** (`blas_integration.zig`).
    - Vtable-based interface supporting OpenBLAS, MKL, and Accelerate.
    - Pure-Zig SIMD fallback when no external BLAS is available.
    - GEMM, GEMV, and batch operations.
- **CPU Threading** (`threading.zig`).
    - Work-stealing thread pool with configurable worker count.
    - NUMA-aware task distribution.
    - Parallel matrix and attention operations.
- **Basic Math** (`math.zig`).
    - Fused multiply-add, fast reciprocal square root.
    - Log-sum-exp for numerical stability.
    - Half-precision (f16) conversion utilities.

---

## Current Status

| Metric | Value |
|--------|-------|
| Test cases | 285+ (all passing) |
| Examples | 12 |
| Model architectures | 18 families |
| Quantisation formats | 18+ |
| Sampling strategies | 8 |
| Source lines (approx.) | ~30,000 |
| llama.cpp production parity | ~90% |

---

## Roadmap

The following items are under consideration for future milestones.  No
timeline is committed.

- **Speculative decoding** for accelerated generation.
- **Beam search** as an additional decoding strategy.
- **C API** header generation for cross-language integration.
- **WebAssembly** compilation target for browser-based inference.
- **Additional model architectures** as they are released in GGUF format.
- **Expanded documentation** including video walkthroughs and interactive
  notebooks.
