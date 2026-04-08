# Architecture

This section provides a comprehensive treatment of ZigLlama's software
architecture -- the design decisions, structural invariants, and module
boundaries that make the project both an effective educational resource and a
capable inference engine.

---

## Section Map

| Page | Focus |
|------|-------|
| [Design Principles](design-principles.md) | The philosophical and engineering tenets that guided every line of code -- educational clarity, progressive complexity, test-driven development, and documentation-as-code. |
| [The 6-Layer Progressive Architecture](six-layer-overview.md) | A detailed walkthrough of each architectural layer, from low-level tensor storage through linear algebra, neural primitives, transformer blocks, model definitions, and finally the inference engine.  Includes dependency diagrams, data-flow sequences, and per-layer component inventories. |
| [Module Dependencies](module-dependencies.md) | The full import graph, public API surface (26 re-exported modules), internal modules, and the strict layering rule that keeps the dependency DAG acyclic. |
| [Comparison with llama.cpp](llama-cpp-comparison.md) | A side-by-side feature, quantization, model-coverage, and performance analysis against the industry-standard C++ implementation, together with ZigLlama's unique value proposition. |

---

## Quick Orientation

ZigLlama is structured as a **six-layer progressive stack**.  Each layer
depends only on layers below it, producing a clean directed acyclic graph (DAG)
of imports:

```
Layer 6  Inference        -- generation, caching, streaming, batching
Layer 5  Models           -- LLaMA + 17 other architectures, GGUF, tokenizers
Layer 4  Transformers     -- attention, feed-forward, transformer blocks
Layer 3  Neural Primitives-- activations, normalization, embeddings
Layer 2  Linear Algebra   -- SIMD matmul, quantization (Q/K/IQ)
Layer 1  Foundation       -- Tensor(T), MemoryMap, GGUF reader, BLAS, threading
```

!!! info "Reading Order"
    If you are new to the project, start with **Design Principles** to
    understand *why* the architecture looks the way it does, then proceed
    through the **6-Layer Overview** for the *what*, and finally consult
    **Module Dependencies** for the precise *how* of the import graph.

---

## Key Metrics at a Glance

| Metric | Value |
|--------|-------|
| Architectural layers | 6 |
| Public modules (re-exported) | 26 |
| Internal modules | 12 |
| Model architectures supported | 18 |
| Quantization formats | 18+ (Q4_0 through IQ4_NL) |
| Test count | 285+ |
| Lines of Zig (approx.) | 15 000+ |
