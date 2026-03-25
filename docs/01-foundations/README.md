# Foundation Layer - Learning Guide

**🏠 [Home](../../README.md) → 📚 [Docs](../README.md) → 🧮 Foundation Layer**

---

The foundation layer provides the fundamental building blocks for all transformer operations. This is where we learn about tensors, memory management, and basic mathematical operations that form the backbone of neural networks.

## 🚀 Quick Start
```bash
# Try tensors in action
zig test ../../src/foundation/tensor.zig

# See practical examples
zig run ../../examples/simple_demo.zig
```

## Learning Objectives

By the end of this section, you'll understand:

- **Tensor Fundamentals**: What tensors are and why they're central to neural networks
- **Memory Layout**: How multi-dimensional data is stored and accessed efficiently
- **Basic Operations**: Addition, multiplication, and other operations on tensors
- **Performance Considerations**: Memory allocation patterns and computational complexity

## Components

### 📊 Tensor Operations (`src/foundation/tensor.zig`)

The heart of our foundation layer - a comprehensive tensor implementation that prioritizes educational clarity.

**Key Features:**
- Multi-dimensional array support with arbitrary shapes
- Row-major memory layout (matching most ML frameworks)
- Educational documentation connecting operations to transformer concepts
- Comprehensive error handling and bounds checking

**Mathematical Foundation:**
```
Tensor Hierarchy:
0D: Scalar     → Single number (bias values)
1D: Vector     → Token embeddings [d_model]
2D: Matrix     → Attention weights [seq_len, seq_len]
3D: Batch      → Multiple sequences [batch, seq_len, d_model]
4D: Multi-head → [batch, heads, seq_len, seq_len]
```

## Progressive Learning Path

### Step 1: Understanding Tensors
```zig
// Create a 2×3 matrix (2 rows, 3 columns)
var matrix = try Tensor(f32).init(allocator, &[_]usize{2, 3});
defer matrix.deinit();

// This represents 6 floating-point numbers in memory:
// [a, b, c, d, e, f] arranged as:
// [[a, b, c],
//  [d, e, f]]
```

### Step 2: Indexing and Memory Layout
```zig
// Set first row: [1.0, 2.0, 3.0]
try matrix.set(&[_]usize{0, 0}, 1.0);
try matrix.set(&[_]usize{0, 1}, 2.0);
try matrix.set(&[_]usize{0, 2}, 3.0);

// Memory layout: strides help convert [i,j] to flat index
// For shape [2,3]: strides = [3, 1]
// Index [0,1] = 0*3 + 1*1 = 1 (second element)
```

### Step 3: Matrix Multiplication
```zig
// Transformer core operation: input @ weights
var input = try Tensor(f32).init(allocator, &[_]usize{seq_len, d_model});
var weights = try Tensor(f32).init(allocator, &[_]usize{d_model, d_output});
var output = try input.matmul(weights, allocator);

// This is exactly what happens in:
// - Query/Key/Value projections: X @ W_q, X @ W_k, X @ W_v
// - Feed-forward layers: X @ W1, hidden @ W2
// - Output projections: hidden @ W_out
```

## Transformer Connection

Every operation in this foundation layer directly relates to transformer architecture:

| Operation | Transformer Usage | Mathematical Form |
|-----------|-------------------|-------------------|
| Matrix Creation | Token embeddings, model weights | `Tensor(seq_len, d_model)` |
| Matrix Multiplication | Linear projections | `Y = X @ W + b` |
| Addition | Residual connections, bias addition | `output = input + residual` |
| Indexing | Token selection, attention masking | `attention[i, j]` |

## Performance Considerations

### Memory Allocation
- **Contiguous Layout**: Enables efficient cache utilization
- **SIMD-Friendly**: Row-major layout works well with vector instructions
- **Memory Pooling**: Future optimization for reducing allocations

### Computational Complexity
- **Matrix Multiplication**: O(n³) - the computational bottleneck
- **Element-wise Operations**: O(n) - highly parallelizable
- **Memory Bandwidth**: Often the limiting factor in practice

## Testing Strategy

Our comprehensive test suite covers:

1. **Correctness Tests**: Verify mathematical operations against known results
2. **Edge Case Tests**: Handle boundary conditions and error cases
3. **Performance Tests**: Ensure operations scale as expected
4. **Educational Tests**: Demonstrate transformer-relevant usage patterns

## 🎯 Next Steps

Once you understand tensors and basic operations:

### ➡️ [**Linear Algebra Layer**](../02-linear-algebra/)
SIMD optimizations and quantization - [`src/linear_algebra/`](../../src/linear_algebra/)

### ➡️ [**Neural Primitives**](../03-neural-primitives/)
Activation functions and normalization - [`src/neural_primitives/`](../../src/neural_primitives/)

### ➡️ [**Full Learning Path**](../README.md#learning-navigation)
See all 6 layers with direct links to source code

## Hands-On Practice

```bash
# Run foundation layer tests
zig build test-foundation

# Run the main demo to see tensors in action
zig build run

# Explore the comprehensive test suite
zig test tests/unit/test_tensor.zig
```

## Key Takeaways

- **Tensors are the fundamental data structure** for all neural network operations
- **Memory layout matters** for both performance and correctness
- **Matrix multiplication is the core operation** that enables learning in transformers
- **Educational clarity doesn't require sacrificing correctness** - we can have both

The foundation layer might seem simple, but it's where transformer magic begins. Every attention head, every feed-forward layer, every residual connection builds on these tensor operations.

---

*"The foundation layer is like learning to read music notation - it seems basic, but it unlocks everything that comes after."*