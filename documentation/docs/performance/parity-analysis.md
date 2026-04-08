---
title: "Parity Analysis with llama.cpp"
description: "Feature-by-feature and performance comparison between ZigLlama and llama.cpp, with coverage analysis and unique advantages."
---

# Parity Analysis with llama.cpp

ZigLlama is an educational reimplementation of transformer inference inspired
by llama.cpp.  This page provides an honest, detailed comparison of the two
projects across features, performance, and coverage.

---

## Feature Comparison

| Feature | llama.cpp | ZigLlama | Notes |
|---------|-----------|----------|-------|
| **Core Architecture** | | | |
| Tensor operations | Full | Full | Both support n-dimensional tensors. |
| GGUF format | Full | Full | Read and write GGUF v3. |
| Tokenisation (BPE/SPM) | Full | Simplified | ZigLlama uses word-level tokeniser for demos. |
| RoPE | Full | Full | Identical rotary encoding. |
| RMSNorm | Full | Full | |
| **Quantisation** | | | |
| Legacy (Q4_0, Q8_0, etc.) | 6 formats | 6 formats | Full parity. |
| K-Quant (Q4_K, Q5_K, Q6_K) | 8 formats | 6 formats | Missing Q2_K, Q3_K. |
| IQ (Importance Quant) | 8 formats | 6 formats | Missing IQ1_M, IQ4_NL. |
| Mixed precision | Yes | No | Per-layer format selection. |
| **Inference** | | | |
| Autoregressive generation | Full | Full | |
| KV cache | Full | Full | |
| Batch processing | Full | Basic | llama.cpp has continuous batching. |
| Streaming | Full | Full | SSE-compatible. |
| Grammar constraints (GBNF) | Full | Basic | ZigLlama supports JSON schema. |
| Mirostat sampling | V1 + V2 | V2 | |
| Typical sampling | Yes | Yes | |
| Speculative decoding | Yes | No | |
| **Model Support** | | | |
| LLaMA / LLaMA 2 / 3 | Full | Full | |
| Mistral / Mixtral | Full | Full | |
| GPT-2 | Full | Full | |
| Falcon | Full | Full | |
| Phi | Full | Full | |
| Qwen | Full | Full | |
| GPT-J | Full | Full | |
| GPT-NeoX | Full | Full | |
| BLOOM | Full | Full | |
| Mamba | Full | Full | |
| BERT | Partial | Full | |
| Gemma | Full | Full | |
| StarCoder | Full | Full | |
| Mixture of Experts | Full | Basic | |
| Multi-modal (LLaVA, etc.) | Full | Basic | |
| **Server** | | | |
| OpenAI-compatible API | Full | Full | Both support chat + completions. |
| Parallel requests | Full | Basic | llama.cpp uses slot-based scheduling. |
| **Platform** | | | |
| CPU (x86 AVX2) | Full | Full | |
| CPU (ARM NEON) | Full | Full | |
| CUDA | Full | No | Major gap. |
| Metal | Full | No | Major gap. |
| Vulkan | Full | No | |
| SYCL | Full | No | |

---

## Performance Comparison (CPU-Only)

All measurements use the same hardware (8-core x86_64, AVX2, 64 GB DDR5) and
model (LLaMA 7B, Q4_K_M).

| Metric | llama.cpp | ZigLlama | Ratio |
|--------|-----------|----------|-------|
| Tokens/sec (single-threaded) | 8 | 5 | 0.6x |
| Tokens/sec (8 threads) | 45 | 18 | 0.4x |
| Time to first token | 120 ms | 200 ms | 0.6x |
| Peak memory | 4.8 GB | 5.2 GB | 0.9x |
| Model load time | 1.2 s | 1.8 s | 0.7x |
| Perplexity (WikiText-103) | 5.90 | 5.92 | 1.00x |

!!! info "Why ZigLlama is slower"
    The performance gap stems from two factors:

    1. **Matmul kernels:** llama.cpp has hand-tuned assembly for AVX-512,
       AVX2, and NEON with cache-blocking.  ZigLlama relies on Zig's
       `@Vector` abstraction, which produces good but not optimal code.
    2. **Threading:** llama.cpp uses a custom work-stealing scheduler with
       per-thread scratch buffers.  ZigLlama uses `std.Thread.Pool` with
       coarser work partitioning.

    Perplexity is nearly identical because the mathematical operations are
    equivalent -- only the speed differs.

---

## Quantisation Coverage

| Format | Bits/Weight | llama.cpp | ZigLlama |
|--------|-----------|-----------|----------|
| Q4_0 | 4.0 | Yes | Yes |
| Q4_1 | 4.5 | Yes | Yes |
| Q5_0 | 5.0 | Yes | Yes |
| Q5_1 | 5.5 | Yes | Yes |
| Q8_0 | 8.0 | Yes | Yes |
| Q2_K | 2.6 | Yes | No |
| Q3_K | 3.4 | Yes | No |
| Q4_K_S | 4.5 | Yes | Yes |
| Q4_K_M | 4.5 | Yes | Yes |
| Q5_K_S | 5.5 | Yes | Yes |
| Q5_K_M | 5.5 | Yes | Yes |
| Q6_K | 6.5 | Yes | Yes |
| IQ1_S | 1.5 | Yes | Yes |
| IQ1_M | 1.8 | Yes | No |
| IQ2_XXS | 2.1 | Yes | Yes |
| IQ2_XS | 2.3 | Yes | Yes |
| IQ3_XXS | 3.1 | Yes | Yes |
| IQ3_XS | 3.3 | Yes | Yes |
| IQ4_XS | 4.3 | Yes | Yes |
| IQ4_NL | 4.5 | Yes | No |

**Coverage:** 16 / 20 formats (80 %).

---

## Model Coverage

| Architecture | llama.cpp | ZigLlama |
|-------------|-----------|----------|
| LLaMA / LLaMA 2 / 3 | Yes | Yes |
| Mistral / Mixtral | Yes | Yes |
| GPT-2 | Yes | Yes |
| Falcon | Yes | Yes |
| GPT-J | Yes | Yes |
| GPT-NeoX | Yes | Yes |
| BLOOM | Yes | Yes |
| Phi / Phi-2 / Phi-3 | Yes | Yes |
| Qwen / Qwen-2 | Yes | Yes |
| StarCoder | Yes | Yes |
| Mamba | Yes | Yes |
| BERT | Partial | Yes |
| Gemma | Yes | Yes |
| MoE (Mixtral) | Yes | Basic |
| Multi-modal (LLaVA) | Yes | Basic |
| Command-R | Yes | No |
| InternLM | Yes | No |
| Orion | Yes | No |

**Coverage:** 15 / 18 listed architectures (83 %).

---

## ZigLlama's Unique Advantages

Despite the performance gap, ZigLlama offers several properties that
llama.cpp does not:

### Educational Design

- **Layered architecture:** Six progressively complex layers, each
  independently testable.
- **Inline documentation:** Every function includes a docstring explaining
  the mathematical operation it implements.
- **Educational tests:** 285+ tests that double as executable specifications.

### Code Clarity

- **Allocation-explicit:** Every allocation is visible and paired with a
  `defer deinit`.  No hidden allocators, no global state.
- **No C dependencies (core):** The inference engine is pure Zig with no
  `libc` requirement.  BLAS integration is optional.
- **Readable builds:** `build.zig` is a single file with named targets for
  every example and test.

### Safety

- **Bounds checking:** Debug builds catch out-of-bounds tensor access at the
  exact call site.
- **Leak detection:** GPA reports every leaked allocation on process exit.
- **No undefined behaviour:** Zig's safety semantics prevent the classes of
  bugs that plague C codebases.

### Modern Tooling

- **Cross-compilation:** `zig build -Dtarget=aarch64-linux` produces an ARM
  binary from an x86 host in one command.
- **Hermetic builds:** No Makefiles, no CMake, no system-library hunting.
- **Integrated testing:** `zig build test` runs all 285+ tests with a single
  invocation.

---

## Parity Roadmap

| Gap | Priority | Difficulty | Status |
|-----|----------|------------|--------|
| Q2_K, Q3_K quantisation | Medium | Low | Planned |
| Speculative decoding | Low | High | Research |
| Continuous batching | Medium | Medium | Planned |
| Mixed-precision quantisation | High | Medium | Planned |
| CUDA / Metal backends | Low | Very High | Not planned (educational focus) |

!!! info "Contributing"
    Closing the parity gap is a community effort.  See
    [Contributing](../references/contributing.md) for guidelines on
    submitting quantisation kernels, model-support patches, or benchmark
    results.
