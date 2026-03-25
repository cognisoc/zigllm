# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

ZigLlama is an educational LLaMA (Large Language Model) implementation in Zig, designed to teach transformer architectures through progressive implementation. It follows a 6-layer progressive architecture from foundations to inference, with the goal of achieving production parity with llama.cpp while maintaining educational clarity.

**Key Facts:**
- **Primary language**: Zig 0.13+
- **Architecture**: 6 progressive layers from tensors to inference
- **Educational focus**: 285+ comprehensive tests, extensive documentation
- **Coverage**: 18 model architectures (LLaMA, GPT-2, Mistral, Falcon, Qwen, Phi, etc.)
- **Performance**: 400x speedup with optimizations (KV caching, SIMD, quantization)

## Build and Test Commands

### Basic Build and Test
```bash
# Build the project (Note: build.zig has issues, tests work independently)
zig build

# Run all 285+ comprehensive tests
zig test src/main.zig

# Run layer-specific tests
zig test src/foundation/tensor.zig          # Foundation layer
zig test src/linear_algebra/matrix_ops.zig  # Linear algebra layer
zig test src/neural_primitives/             # Neural primitives
zig test src/transformers/                  # Transformers layer
zig test src/models/llama.zig              # Models layer
zig test src/inference/generation.zig       # Inference layer
```

### Running Examples
```bash
# Educational demos (run directly with zig)
zig run examples/simple_demo.zig           # Complete journey overview
zig run examples/educational_demo.zig      # Progressive concepts
zig run examples/benchmark_demo.zig        # Performance analysis
zig run examples/parity_demo.zig          # llama.cpp parity demo
zig run examples/model_architectures_demo.zig # 18 model architectures

# Multi-modal and specialized examples
zig run examples/multi_modal_demo.zig      # Vision-language models
zig run examples/threading_demo.zig        # Threading optimization
zig run examples/gguf_demo.zig            # GGUF format support
```

### Test Categories and Coverage
- **Foundation**: 8 tests (tensors, memory management, memory mapping)
- **Linear Algebra**: 25 tests (SIMD, K-quantization, IQ-quantization)
- **Neural Primitives**: 12 tests (activations, normalization, embeddings)
- **Transformers**: 15 tests (attention, feed-forward, sliding window)
- **Models**: 120 tests (18 architectures, GGUF loading, tokenization)
- **Inference**: 80 tests (generation, sampling, caching, streaming)
- **Production Parity**: 25 tests (integration and advanced features)

## Project Architecture

### Progressive 6-Layer Architecture
```
┌─ 6. Inference Layer ────────────────────────────────────┐
│  🚀 Text Generation • Sampling • KV Caching • Streaming │
├─ 5. Models Layer ──────────────────────────────────────┤
│  🦙 LLaMA Architecture • GGUF Support • Tokenization   │
├─ 4. Transformers Layer ────────────────────────────────┤
│  🎯 Multi-Head Attention • Feed-Forward • Full Blocks  │
├─ 3. Neural Primitives Layer ───────────────────────────┤
│  🧠 Activations • Normalization • Embeddings • RoPE    │
├─ 2. Linear Algebra Layer ──────────────────────────────┤
│  ⚡ SIMD Operations • Quantization • Cache Optimization │
└─ 1. Foundation Layer ──────────────────────────────────┘
   🧮 Tensors • Memory Management • Basic Operations
```

### Source Code Structure
```
src/
├── main.zig                    # Main library entry point
├── foundation/                 # Layer 1: Tensors and memory
│   ├── tensor.zig             # Multi-dimensional tensors
│   ├── memory_mapping.zig     # mmap/mlock optimization
│   ├── threading.zig          # Thread-safe operations
│   ├── gguf_format.zig        # GGUF format support
│   └── blas_integration.zig   # BLAS acceleration
├── linear_algebra/            # Layer 2: Optimized math operations
│   ├── matrix_ops.zig         # SIMD-optimized operations
│   ├── quantization.zig       # Standard quantization
│   ├── k_quantization.zig     # K-quantization schemes
│   └── iq_quantization.zig    # Importance quantization
├── neural_primitives/         # Layer 3: Neural network components
│   ├── activations.zig        # ReLU, GELU, SwiGLU, etc.
│   ├── normalization.zig      # LayerNorm, RMSNorm
│   └── embeddings.zig         # Token/position embeddings
├── transformers/              # Layer 4: Transformer components
│   ├── attention.zig          # Multi-head attention + RoPE
│   ├── feed_forward.zig       # FFN with modern activations
│   └── transformer_block.zig  # Complete transformer blocks
├── models/                    # Layer 5: Complete model architectures (18 families)
│   ├── llama.zig             # LLaMA/LLaMA2 architecture
│   ├── gpt2.zig              # GPT-2 implementation
│   ├── mistral.zig           # Mistral architecture
│   ├── config.zig            # Model configurations
│   ├── tokenizer.zig         # Tokenization systems
│   ├── gguf.zig              # GGUF format loading
│   ├── chat_templates.zig    # Chat template system
│   ├── mixture_of_experts.zig # MoE architectures
│   ├── multi_modal.zig       # Vision-language models
│   ├── mamba.zig             # State-space models
│   └── [12 other architectures]
├── inference/                 # Layer 6: Text generation and optimization
│   ├── generation.zig         # Core text generation
│   ├── kv_cache.zig          # Key-value caching
│   ├── streaming.zig          # Real-time streaming
│   ├── batching.zig          # Batch processing
│   ├── advanced_sampling.zig  # 8 sampling strategies
│   ├── grammar_constraints.zig # Grammar-guided generation
│   └── profiling.zig         # Performance analysis
├── server/                   # HTTP server implementation
│   ├── cli.zig              # Command-line interface
│   └── http_server.zig      # OpenAI-compatible server
├── tools/                    # Model conversion tools
│   ├── model_converter.zig   # Model format conversion
│   └── converter_cli.zig     # CLI for conversion
└── evaluation/               # Model evaluation
    └── perplexity.zig       # Perplexity calculation
```

### Key Implementation Features

**18 Model Architectures Supported:**
- **Core Language Models**: LLaMA/LLaMA2, Mistral, GPT-2, Falcon, Qwen, Phi, GPT-J, GPT-NeoX, BLOOM
- **Specialized Architectures**: Mamba (state-space), BERT (bidirectional), Gemma, StarCoder
- **Advanced Components**: Mixture of Experts (MoE), Multi-modal vision-language models

**Production-Ready Optimizations:**
- **Quantization**: 18+ formats including K-quantization and importance quantization
- **SIMD Acceleration**: AVX/AVX2/NEON with auto-detection
- **Memory Optimization**: mmap/mlock for efficient model loading
- **KV Caching**: 95%+ inference speedup
- **Advanced Sampling**: 8 strategies including Mirostat, Typical, Tail-free
- **Grammar Constraints**: JSON, RegEx, CFG support
- **Threading**: Thread-safe concurrent inference

## Development Guidelines

### Educational Focus
- Every component includes extensive documentation explaining the "why" behind implementation decisions
- Code connects implementation to transformer theory and mathematical foundations
- Progressive complexity building from simple concepts to advanced systems
- Comprehensive test coverage demonstrating correct usage and edge cases

### Code Quality Standards
- **Clarity over cleverness**: Readable code that explains itself
- **Mathematical foundations**: Connect code to underlying mathematics in comments
- **Performance context**: Explain computational complexity and optimization opportunities
- **Educational testing**: Tests serve as examples and validate educational claims

### Testing Approach
- **Reference validation**: Test against known-good outputs
- **Edge case coverage**: Test boundary conditions and error paths
- **Performance regression**: Benchmark critical paths
- **Layer isolation**: Each layer can be tested independently

### Project Principles
1. **Educational first, performance second** - but achieve both
2. **Progressive component architecture** - build understanding step by step
3. **Test-driven development** - 285+ comprehensive tests
4. **Documentation as code** - self-contained learning resource
5. **Feature parity and beyond** - match llama.cpp while maintaining clarity

## Usage Patterns

### Basic Library Usage
```zig
const zigllama = @import("src/main.zig");

// Initialize model configuration
const config = zigllama.models.config.ModelConfig.llama(.LLaMA_7B);

// Create model and tokenizer
var model = try zigllama.models.llama.LLaMAModel.init(config, allocator);
var tokenizer = try zigllama.models.tokenizer.SimpleTokenizer.init(allocator, config.vocab_size);

// Generate text
var generator = zigllama.inference.generation.TextGenerator.init(&model, &tokenizer, allocator, null);
const result = try generator.generate("The future of AI is");
```

### Layer-by-Layer Learning
Start with foundation and work up through the layers:
1. **Foundation**: Understand tensors and memory management
2. **Linear Algebra**: Learn SIMD optimization and quantization
3. **Neural Primitives**: Implement activations and normalization
4. **Transformers**: Build attention mechanisms and feed-forward networks
5. **Models**: Assemble complete architectures
6. **Inference**: Add generation and optimization

## Performance Characteristics

### Optimization Results
- **KV Caching**: 20x speedup, 50% memory reduction
- **SIMD Operations**: 3-5x speedup
- **K-Quantization**: 87% memory reduction (Q4_K)
- **IQ-Quantization**: 95% memory reduction (IQ1_S)
- **Combined optimizations**: 400x total speedup

### Model Support Status
- **Implemented**: 18/94 architectures (19% numeric, ~80% real-world usage)
- **Educational parity**: 100% (complete transformer understanding)
- **Production parity**: ~90% (remarkable achievement for educational codebase)

This codebase serves as both a comprehensive learning resource for transformer architectures and a production-capable inference engine, demonstrating that educational clarity and high performance can coexist.