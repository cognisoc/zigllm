# ZigLlama Documentation

> **Educational documentation for learning transformers through implementation**

Welcome to the ZigLlama documentation! This is a comprehensive learning resource that will take you from basic tensor operations to a complete LLaMA implementation.

## 🚀 **Start Learning Now**

| Quick Path | Description | Time |
|------------|-------------|------|
| 🏃 [**Quick Tour**](../README.md#quick-start) | See the big picture first | 30 min |
| 📚 [**Complete Journey**](01-foundations/) | Step-by-step learning path | 1-2 weeks |
| 🔍 [**Browse Code**](../src/) | Explore implementation directly | Variable |

## 📍 **Learning Navigation**

| Step | Topic | Documentation | Key Source Files | Try It |
|------|-------|---------------|------------------|--------|
| **1** | Tensors & Memory | [📖 Foundations](01-foundations/) | [`tensor.zig`](../src/foundation/tensor.zig) | `zig test ../src/foundation/` |
| **2** | SIMD & Quantization | [📖 Linear Algebra](02-linear-algebra/) | [`matrix_ops.zig`](../src/linear_algebra/matrix_ops.zig) | `zig test ../src/linear_algebra/` |
| **3** | Activations & Norms | [📖 Neural Primitives](03-neural-primitives/) | [`activations.zig`](../src/neural_primitives/activations.zig) | `zig test ../src/neural_primitives/` |
| **4** | Attention & FFN | [📖 Transformers](04-transformers/) | [`attention.zig`](../src/transformers/attention.zig) | `zig test ../src/transformers/` |
| **5** | LLaMA Architecture | [📖 Models](05-models/) | [`llama.zig`](../src/models/llama.zig) | `zig test ../src/models/` |
| **6** | Text Generation | [📖 Inference](06-inference/) | [`generation.zig`](../src/inference/generation.zig) | `zig test ../src/inference/` |

## 🎯 How to Use This Documentation

This documentation is designed as a **progressive learning experience**:

1. **Start at the foundation** and work your way up through the layers
2. **Read the code alongside the docs** - they're designed to complement each other
3. **Run the examples and tests** as you learn each component
4. **Build your intuition** before moving to the next layer

## 📚 Learning Path

### [01 - Foundations](01-foundations/)
**Start here!** Learn about tensors, memory layout, and basic operations.
- Tensor fundamentals and multi-dimensional arrays
- Memory management and efficient data access
- Basic mathematical operations (addition, matrix multiplication)
- Connection to transformer architecture

### [02 - Linear Algebra](02-linear-algebra/)
**SIMD optimizations and quantization techniques** - [`src/linear_algebra/`](../src/linear_algebra/)
- Optimized matrix operations using SIMD instructions
- K-quantization and IQ-quantization (18+ formats)
- Memory alignment and cache optimization
- Performance analysis and benchmarking

### [03 - Neural Primitives](03-neural-primitives/)
**Activation functions and normalization layers** - [`src/neural_primitives/`](../src/neural_primitives/)
- Modern activations: ReLU, GELU, SwiGLU, GeGLU
- Normalization: LayerNorm, RMSNorm, BatchNorm
- Dropout and regularization techniques
- Embedding layers and RoPE positional encodings

### [04 - Transformers](04-transformers/)
**The heart of modern LLMs** - [`src/transformers/`](../src/transformers/)
- Multi-head attention mechanism with RoPE
- Scaled dot-product attention mathematics
- Feed-forward networks with SwiGLU/GeGLU
- Complete transformer block implementation

### [05 - Models](05-models/)
**Complete model architectures (18 families)** - [`src/models/`](../src/models/)
- LLaMA, GPT-2, Mistral, Falcon, Qwen, Phi architectures
- GGUF format support and model loading
- Advanced components: MoE, Multi-modal, BERT, Mamba
- Tokenization and chat templates

### [06 - Inference](06-inference/)
**Text generation and optimization** - [`src/inference/`](../src/inference/)
- Advanced sampling strategies (8 methods)
- KV-cache optimization for >95% speedup
- Grammar constraints and streaming generation
- Batch processing and performance profiling

## 🎓 Learning Features

### Progressive Complexity
Each layer builds upon the previous, ensuring you understand the fundamentals before tackling advanced concepts.

### Theory + Practice
Every concept includes:
- **Mathematical foundation** - the underlying mathematics
- **Code implementation** - how it translates to Zig
- **Transformer connection** - how it fits into the bigger picture
- **Performance notes** - optimization opportunities and trade-offs

### Comprehensive Examples
- **Unit tests** demonstrate correct usage
- **Integration examples** show components working together
- **Performance benchmarks** reveal computational characteristics
- **Educational demos** connect concepts to real transformer operations

## 📖 Documentation Types

### 🏗️ Architecture Guides
High-level explanations of how components fit together, with visual diagrams and mathematical foundations.

### 📋 API Reference
Detailed documentation of every public function, with examples and performance characteristics.

### 🎯 Learning Tutorials
Step-by-step guides that build understanding progressively, with hands-on exercises.

### ⚡ Performance Analysis
Deep dives into computational complexity, memory usage, and optimization opportunities.

## 🔧 Hands-On Learning

### 🚀 Try It Now
```bash
# Quick demo - see the full system in action
zig run examples/simple_demo.zig

# Test your understanding as you learn
zig test src/foundation/tensor.zig        # Layer 1
zig test src/linear_algebra/matrix_ops.zig # Layer 2
zig test src/neural_primitives/            # Layer 3
zig test src/transformers/                 # Layer 4
zig test src/models/llama.zig             # Layer 5
zig test src/inference/generation.zig     # Layer 6

# Run all 285+ tests
zig build test

# Explore live examples
zig run examples/educational_demo.zig     # Progressive concepts
zig run examples/model_architectures_demo.zig # 18 architectures
```

### 🎯 Interactive Learning
Each layer includes **hands-on exercises**:
- **Foundation**: Create and manipulate tensors
- **Linear Algebra**: Benchmark SIMD optimizations
- **Neural Primitives**: Test activation functions
- **Transformers**: Implement attention from scratch
- **Models**: Load and configure LLaMA models
- **Inference**: Generate text with different strategies

### Code Navigation
```
zigllm/src/
├── foundation/      # Start here - tensors and basic operations
├── linear_algebra/  # SIMD and optimization techniques
├── neural_primitives/ # Activations and normalization
├── transformers/    # Attention and feed-forward networks
├── models/          # Complete LLaMA implementation
└── inference/       # Generation and sampling
```

## 🎯 Learning Objectives

By the end of this documentation, you'll understand:

- **How transformers work** from first principles
- **Implementation details** that affect performance
- **Mathematical foundations** behind each operation
- **Optimization techniques** used in production systems
- **Design trade-offs** in neural network implementations

## 🤝 Contributing to Documentation

We welcome improvements to make learning even better:

- **Clarify confusing explanations** - if something isn't clear, let us know!
- **Add visual diagrams** - ASCII art and mathematical notation help understanding
- **Improve examples** - better demonstrations make concepts stick
- **Fix errors** - accuracy is crucial for learning

## 📚 External Resources

While ZigLlama documentation is designed to be self-contained, these resources provide additional context:

- **Papers**: Original transformer and LLaMA papers for mathematical foundations
- **References**: Links to relevant research and implementation details
- **Tutorials**: Complementary learning materials from the broader community

## 🚀 Quick Start for Impatient Learners

If you want to jump in immediately:

1. Read [Project Principles](PROJECT_PRINCIPLES.md) for our educational philosophy
2. Start with [Foundation Layer](01-foundations/) to understand tensors
3. Run `zig build test` to see comprehensive examples
4. Explore the source code in `src/foundation/tensor.zig`

But remember: **understanding builds progressively**. Don't skip the fundamentals!

---

## 💡 Learning Tips

- **Code along** - implement concepts as you learn them
- **Ask questions** - use GitHub issues for clarification
- **Experiment** - modify examples to test your understanding
- **Teach others** - explaining concepts solidifies learning

**Remember**: The goal isn't just to build a working transformer, but to deeply understand how and why it works. Take your time, and enjoy the journey of discovery!

---

*"The best way to understand transformers is to build one yourself, with full understanding of every component."*