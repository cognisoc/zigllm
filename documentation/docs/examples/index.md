---
title: "Examples and Tutorials"
description: "Runnable examples and step-by-step tutorials covering ZigLlama from first inference to production chatbots."
---

# Examples and Tutorials

ZigLlama ships twelve self-contained example programs in the `examples/`
directory.  Each can be compiled and executed independently; together they form
a progressive curriculum that mirrors the six architectural layers.

---

## Example Programs

| # | File | Layer | Description | Est. Time |
|---|------|-------|-------------|-----------|
| 1 | `simple_demo.zig` | All | End-to-end summary of every layer; prints architecture stats. | 2 min |
| 2 | `educational_demo.zig` | All | Progressive walkthrough from tensors to generation. | 5 min |
| 3 | `benchmark_demo.zig` | 2 | Matrix-multiplication benchmarks: naive vs SIMD, varying sizes. | 3 min |
| 4 | `model_architectures_demo.zig` | 5 | Instantiate and inspect LLaMA, Mistral, GPT-2, Falcon, etc. | 3 min |
| 5 | `gguf_demo.zig` | 1, 5 | Parse a GGUF file header, list tensors, read metadata. | 2 min |
| 6 | `parity_demo.zig` | All | Compare ZigLlama outputs against llama.cpp reference values. | 5 min |
| 7 | `multi_modal_demo.zig` | 5 | Vision-language pipeline: image encoder + text decoder. | 4 min |
| 8 | `multi_modal_concepts_demo.zig` | 5 | Conceptual overview of multi-modal fusion strategies. | 3 min |
| 9 | `threading_demo.zig` | 1 | Thread-pool creation, parallel matmul, NUMA awareness. | 3 min |
| 10 | `chat_templates_demo.zig` | 5, 6 | Apply LLaMA 2, ChatML, Mistral templates to a conversation. | 2 min |
| 11 | `perplexity_demo.zig` | Eval | Configure evaluator, run benchmarks, compare quantisations. | 4 min |
| 12 | `main.zig` | All | Master demo that invokes highlights from every other example. | 5 min |

!!! tip "Running an example"
    ```bash
    zig build run-example -- educational_demo
    # or, directly:
    zig run examples/educational_demo.zig
    ```

---

## Tutorials

The tutorials below provide annotated, step-by-step walkthroughs that go
deeper than the standalone examples.

| Tutorial | What you will build |
|----------|---------------------|
| [Your First Inference](first-inference.md) | Load a model, tokenise a prompt, generate text, and decode the output -- all in ~40 lines of Zig. |
| [Understanding Attention](understanding-attention.md) | Construct Q, K, V tensors from scratch, compute scaled dot-product attention, and visualise multi-head splits. |
| [Quantization in Practice](quantization-practice.md) | Take an FP32 model, quantise it to Q4_K, compare outputs, and measure the perplexity delta. |
| [Building a Chatbot](building-chatbot.md) | Wire the HTTP server, chat templates, and streaming together into an interactive chatbot you can test with `curl`. |

---

## Demo Walkthroughs

For a narrated, line-by-line walkthrough of every example program, see
[Demo Walkthroughs](demo-walkthroughs.md).  Each section explains what the
example demonstrates, how to run it, the key concepts it illustrates, and the
expected terminal output.

---

## Prerequisites

All examples assume you have:

1. A working Zig toolchain (0.13+).  See [Installation](../getting-started/installation.md).
2. The ZigLlama repository cloned and the build system verified (`zig build test`).
3. For model-loading examples: a GGUF model file.  The tutorials indicate where
   to download one when needed.

!!! info "No GPU required"
    Every example runs on CPU.  SIMD acceleration (AVX2 / NEON) is used
    automatically when available but is not mandatory.
