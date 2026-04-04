# zigllm

**Learn how LLMs work by building one in Zig -- from tensors to text generation.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Zig](https://img.shields.io/badge/Zig-0.14+-f7a41d?logo=zig&logoColor=white)](https://ziglang.org)
[![Tests](https://img.shields.io/badge/tests-285%2B_passing-brightgreen)](#testing)
[![Models](https://img.shields.io/badge/architectures-18_families-blueviolet)](#model-architectures)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Skelf-Research/zigllm/pulls)

zigllm is an educational implementation of transformer architectures in Zig. It implements 18 model families (LLaMA, Mistral, GPT-2, Falcon, Mamba, BERT, and more) across 6 progressive layers, each building on the last. Every component is documented to teach *why* it works, not just *how*.

It is also a demonstration that Zig is a viable language for ML/AI workloads -- offering manual memory control, comptime generics, and first-class SIMD without a runtime or garbage collector.

## Why zigllm?

- **Learn transformers by building one.** Progressive architecture takes you from raw tensor ops to full text generation. No magic -- every layer is explicit.
- **Zig for ML/AI.** Comptime, SIMD intrinsics, and deterministic memory management make Zig uniquely suited for high-performance inference. This project proves it.
- **Read real code, not slides.** 285+ tests serve as executable documentation. Each test demonstrates a concept and validates the math.

## Quick Start

```bash
git clone https://github.com/Skelf-Research/zigllm.git
cd zigllm
zig build test
```

### Prerequisites

- [Zig 0.14+](https://ziglang.org/download/)
- A modern CPU (AVX/AVX2 recommended but not required)

## Architecture

zigllm builds understanding through 6 progressive layers:

```
 6. Inference         Text generation, sampling, KV caching, streaming
 5. Models            LLaMA, GPT-2, Mistral, Falcon, GGUF loading, tokenization
 4. Transformers      Multi-head attention, feed-forward networks, full blocks
 3. Neural Primitives Activations (SwiGLU, GELU), normalization (RMSNorm), RoPE
 2. Linear Algebra    SIMD matrix ops, K-quantization, IQ-quantization (18+ formats)
 1. Foundation        Tensors, memory management, memory mapping
```

Each layer only depends on the layers below it. Start at the bottom and work up.

## Model Architectures

18 architecture families implemented, covering ~80% of real-world LLM usage:

| Category | Architectures |
|---|---|
| **Core LLMs** | LLaMA/LLaMA2, Mistral, GPT-2, Falcon, Qwen, Phi, GPT-J, GPT-NeoX, BLOOM |
| **Specialized** | Mamba (state-space), BERT (bidirectional), Gemma, StarCoder (code) |
| **Advanced** | Mixture of Experts (MoE), Multi-modal (vision-language), BLAS integration |

## Features

**Optimizations** -- KV caching (20x speedup), SIMD acceleration (3-5x), 18+ quantization formats (up to 95% memory reduction), memory-mapped model loading, batch processing.

**Sampling** -- Greedy, top-k, top-p, temperature, Mirostat, typical, tail-free, and contrastive decoding. Grammar-constrained generation (JSON, regex, CFG).

**Format support** -- GGUF model loading compatible with the llama.cpp ecosystem. Models from 1B to 70B+ parameters.

## Documentation

| Path | What you'll learn |
|---|---|
| [Quick tour](docs/README.md) | The big picture |
| [Layer 1: Foundations](docs/01-foundations/) | Tensors and memory |
| [Layer 2: Linear Algebra](docs/02-linear-algebra/) | SIMD and quantization |
| [Layer 3: Neural Primitives](docs/03-neural-primitives/) | Activations and normalization |
| [Layer 4: Transformers](docs/04-transformers/) | Attention and FFN |
| [Layer 5: Models](docs/05-models/) | LLaMA architecture and GGUF |
| [Layer 6: Inference](docs/06-inference/) | Generation and optimization |

## Testing

```bash
zig build test                    # All 285+ tests
zig build test-foundation         # Foundation layer only
zig build test-linear-algebra     # Linear algebra layer only
```

## Examples

| Example | Description |
|---|---|
| `examples/simple_demo.zig` | End-to-end overview |
| `examples/educational_demo.zig` | Layer-by-layer walkthrough |
| `examples/benchmark_demo.zig` | Performance analysis |
| `examples/gguf_demo.zig` | Loading pre-trained models |
| `examples/model_architectures_demo.zig` | Comparing 18 architectures |

## Contributing

Contributions that improve educational value are especially welcome:

- Clearer explanations and documentation
- Additional tests and edge cases
- New model architecture implementations
- Visualization tools for attention patterns and tensor operations

Please keep code readable -- educational clarity takes priority over micro-optimizations.

## License

[MIT](LICENSE)

## Acknowledgments

- [Meta AI](https://ai.meta.com/) -- LLaMA architecture
- [Georgi Gerganov / llama.cpp](https://github.com/ggerganov/llama.cpp) -- production reference
- [Zig](https://ziglang.org/) -- the language
