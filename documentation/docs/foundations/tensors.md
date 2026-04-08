---
title: "Tensor Operations"
description: "Mathematical definition, memory layout, and Zig implementation of the Tensor(T) generic struct used throughout ZigLlama."
---

# Tensor Operations

Tensors are the fundamental data structure of neural networks.  Every weight
matrix, every activation map, every attention score in a transformer is stored
and manipulated as a tensor.  This page gives a rigorous mathematical
definition, derives the row-major memory layout, and walks through ZigLlama's
`Tensor(T)` implementation line by line.

---

## 1. Mathematical Definition

!!! definition "Tensor"

    A **tensor** of order \( k \) (also called *rank* \( k \)) over the reals
    is an element of the space

    \[
        \mathcal{T} \in \mathbb{R}^{n_1 \times n_2 \times \cdots \times n_k}
    \]

    where each \( n_i \in \mathbb{N}_{>0} \) is the size of the \( i \)-th
    **axis** (or *dimension*).  The tuple \( (n_1, n_2, \ldots, n_k) \) is the
    **shape** of the tensor and the total number of scalar entries is
    \( \prod_{i=1}^{k} n_i \).

In the context of deep learning, we almost always work with tensors whose
entries are floating-point numbers -- `f32`, `f16`, or quantised
representations.  ZigLlama's `Tensor(T)` is generic over the element type `T`,
so the same code handles all precisions.

---

## 2. Tensor Hierarchy

| Rank | Name | Shape Example | Transformer Usage |
|---:|---|---|---|
| 0 | Scalar | \( () \) | Loss value, learning rate |
| 1 | Vector | \( (d,) \) | Bias term, single embedding |
| 2 | Matrix | \( (m, n) \) | Weight matrix \( \mathbf{W}_Q \), attention logits |
| 3 | 3-Tensor | \( (b, s, d) \) | Batched token embeddings |
| 4 | 4-Tensor | \( (b, h, s, s) \) | Multi-head attention scores |

!!! info "Why Higher Ranks?"

    Multi-head attention naturally produces a rank-4 tensor: one axis for the
    batch, one for the number of heads, and two for the query/key sequence
    positions.  Keeping this structure explicit avoids error-prone reshaping.

---

## 3. Memory Layout

ZigLlama stores tensors in **row-major** (C-order) layout: the *rightmost*
index changes fastest as you walk linearly through memory.

### 3.1 Stride Formula

!!! theorem "Row-Major Strides"

    For a tensor of shape \( (n_1, n_2, \ldots, n_k) \), the **stride** of
    axis \( i \) is

    \[
        \text{stride}_i = \prod_{j=i+1}^{k} n_j
    \]

    and the flat (linear) index corresponding to multi-index
    \( (i_1, i_2, \ldots, i_k) \) is

    \[
        \text{flat} = \sum_{j=1}^{k} i_j \cdot \text{stride}_j
    \]

### 3.2 Worked Example

Consider a tensor of shape \( (2, 3, 4) \):

| Axis | Size \( n_i \) | Stride |
|---:|---:|---:|
| 0 | 2 | \( 3 \times 4 = 12 \) |
| 1 | 3 | \( 4 \) |
| 2 | 4 | \( 1 \) |

Element \( (1, 2, 3) \) maps to flat index
\( 1 \cdot 12 + 2 \cdot 4 + 3 \cdot 1 = 23 \).

```
Memory:  [ 0  1  2  3 | 4  5  6  7 | 8  9 10 11 | 12 13 14 15 | 16 17 18 19 | 20 21 22 23 ]
                                                                                          ^
                                                                             element (1,2,3)
```

---

## 4. Zig Implementation

### 4.1 Struct Definition

The core data structure lives in `src/foundation/tensor.zig`:

```zig
pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Raw data storage in contiguous memory
        data: []T,

        /// Shape describes tensor dimensions
        shape: []usize,

        /// Strides for efficient indexing
        /// For shape [2, 3, 4], strides are [12, 4, 1]
        strides: []usize,

        /// Total number of elements
        size: usize,

        /// Memory allocator for cleanup
        allocator: Allocator,

        // ... methods follow
    };
}
```

!!! tip "Comptime Generics"

    `Tensor(f32)` and `Tensor(f16)` are *distinct types* resolved entirely at
    compile time.  There is no runtime polymorphism cost -- the compiler
    monomorphizes every method for the concrete element type.

### 4.2 Shape Type

```zig
pub const Shape = []const usize;
```

Shape is a slice of `usize` values.  Passing it as `[]const usize` allows
callers to use stack-allocated arrays:

```zig
var t = try Tensor(f32).init(allocator, &[_]usize{ 2, 3, 4 });
```

### 4.3 Error Set

```zig
pub const TensorError = error{
    InvalidShape,
    IncompatibleShapes,
    InvalidIndex,
    OutOfMemory,
};
```

Each error variant maps to a well-defined precondition violation, keeping error
handling explicit rather than relying on exceptions.

---

## 5. Core Operations

### 5.1 Initialization and Cleanup

```zig
pub fn init(allocator: Allocator, shape: Shape) TensorError!Self {
    if (shape.len == 0) return TensorError.InvalidShape;

    var size: usize = 1;
    for (shape) |dim| {
        if (dim == 0) return TensorError.InvalidShape;
        size *= dim;
    }

    const data = allocator.alloc(T, size) catch return TensorError.OutOfMemory;
    errdefer allocator.free(data);

    const owned_shape = allocator.dupe(usize, shape) catch return TensorError.OutOfMemory;
    errdefer allocator.free(owned_shape);

    const strides = allocator.alloc(usize, shape.len) catch return TensorError.OutOfMemory;
    // ... compute strides (see Section 3.1) ...

    @memset(data, @as(T, 0));

    return Self{ .data = data, .shape = owned_shape, .strides = strides,
                 .size = size, .allocator = allocator };
}

pub fn deinit(self: *Self) void {
    self.allocator.free(self.data);
    self.allocator.free(self.shape);
    self.allocator.free(self.strides);
}
```

!!! warning "Ownership"

    `init` duplicates the incoming `shape` slice because the caller may free it.
    The tensor **owns** its `data`, `shape`, and `strides` arrays and frees all
    three in `deinit`.

### 5.2 Indexing -- `get` and `set`

!!! algorithm "Multi-Index to Flat Index"

    **Input:** indices \( (i_1, \ldots, i_k) \), strides \( (s_1, \ldots, s_k) \)

    **Output:** flat index \( f \)

    1. \( f \leftarrow 0 \)
    2. **for** \( j = 1 \) to \( k \) **do**
        - **if** \( i_j \ge n_j \) **then** return `InvalidIndex`
        - \( f \leftarrow f + i_j \cdot s_j \)
    3. **return** \( f \)

```zig
pub fn get(self: Self, indices: []const usize) TensorError!T {
    if (indices.len != self.shape.len) return TensorError.InvalidIndex;

    var flat_index: usize = 0;
    for (indices, 0..) |idx, dim| {
        if (idx >= self.shape[dim]) return TensorError.InvalidIndex;
        flat_index += idx * self.strides[dim];
    }

    return self.data[flat_index];
}
```

### 5.3 Matrix Multiplication

!!! definition "Matrix Product"

    For \( \mathbf{A} \in \mathbb{R}^{m \times n} \) and
    \( \mathbf{B} \in \mathbb{R}^{n \times p} \), the product
    \( \mathbf{C} = \mathbf{A}\mathbf{B} \in \mathbb{R}^{m \times p} \) is
    defined element-wise as

    \[
        C_{ij} = \sum_{k=1}^{n} A_{ik}\, B_{kj}
    \]

The implementation supports both 2-D and batched 3-D \(\times\) 2-D
multiplication:

```zig
pub fn matmul(self: Self, other: Self, allocator: Allocator) TensorError!Self {
    // 3D @ 2D batched matmul: [batch, m, n] @ [n, p] -> [batch, m, p]
    if (self.ndim() == 3 and other.ndim() == 2) {
        const batch = self.shape[0];
        const m = self.shape[1];
        const n = self.shape[2];
        if (n != other.shape[0]) return TensorError.IncompatibleShapes;
        const p = other.shape[1];
        var result = try Self.init(allocator, &[_]usize{ batch, m, p });
        for (0..batch) |b| {
            for (0..m) |i| {
                for (0..p) |j| {
                    var sum: T = 0;
                    for (0..n) |k| {
                        sum += self.data[b * m * n + i * n + k]
                             * other.data[k * p + j];
                    }
                    result.data[b * m * p + i * p + j] = sum;
                }
            }
        }
        return result;
    }

    // Standard 2D matmul: [m, n] @ [n, p] -> [m, p]
    if (self.ndim() != 2 or other.ndim() != 2)
        return TensorError.IncompatibleShapes;
    const m = self.shape[0];
    const n = self.shape[1];
    if (n != other.shape[0]) return TensorError.IncompatibleShapes;
    const p = other.shape[1];

    var result = try Self.init(allocator, &[_]usize{ m, p });
    for (0..m) |i| {
        for (0..p) |j| {
            var sum: T = 0;
            for (0..n) |k| {
                sum += (try self.get(&[_]usize{ i, k }))
                     * (try other.get(&[_]usize{ k, j }));
            }
            try result.set(&[_]usize{ i, j }, sum);
        }
    }
    return result;
}
```

!!! tip "Production Path"

    The triple-nested loop above is the *educational* implementation.  In
    production, `BlasInterface.gemm()` (see [BLAS Integration](blas-integration.md))
    replaces it with an optimised SIMD kernel, yielding orders-of-magnitude
    speedup.

### 5.4 Element-wise Addition

\[
    \mathbf{C} = \mathbf{A} + \mathbf{B}, \quad C_i = A_i + B_i \quad \forall\, i
\]

```zig
pub fn add(self: Self, other: Self, allocator: Allocator) TensorError!Self {
    if (!std.mem.eql(usize, self.shape, other.shape))
        return TensorError.IncompatibleShapes;

    var result = try Self.init(allocator, self.shape);
    for (0..self.size) |i| {
        result.data[i] = self.data[i] + other.data[i];
    }
    return result;
}
```

### 5.5 Element-wise Multiplication (Hadamard Product)

\[
    \mathbf{C} = \mathbf{A} \odot \mathbf{B}, \quad C_i = A_i \cdot B_i
\]

Used extensively in gated architectures (SwiGLU, GeGLU) where the gate signal
is multiplied element-wise with the content signal.

```zig
pub fn elementWiseMultiply(self: Self, other: Self, allocator: Allocator) TensorError!Self {
    if (!std.mem.eql(usize, self.shape, other.shape))
        return TensorError.IncompatibleShapes;

    var result = try Self.init(allocator, self.shape);
    for (0..self.size) |i| {
        result.data[i] = self.data[i] * other.data[i];
    }
    return result;
}
```

---

## 6. Complexity Analysis

!!! complexity "Time Complexity of Core Operations"

    | Operation | Time | Space (output) |
    |---|---|---|
    | `init` (zero-fill) | \( O(N) \) where \( N = \prod n_i \) | \( O(N) \) |
    | `get` / `set` | \( O(k) \) where \( k \) = rank | \( O(1) \) |
    | `add` | \( O(N) \) | \( O(N) \) |
    | `elementWiseMultiply` | \( O(N) \) | \( O(N) \) |
    | `matmul` (\( m \times n \) by \( n \times p \)) | \( O(mnp) \) | \( O(mp) \) |
    | `fill` | \( O(N) \) | \( O(1) \) (in-place) |

    For square matrices of size \( n \), `matmul` is \( O(n^3) \).
    Strassen's algorithm (\( O(n^{2.807}) \)) is not implemented because the
    practical cross-over point is rarely reached at the matrix sizes encountered
    in single-sequence LLM inference[^1].

---

## 7. Transformer Applications

The following table maps each tensor operation to its role inside a single
transformer block:

| Tensor Operation | Transformer Component | Typical Shapes |
|---|---|---|
| `matmul` | Q/K/V projection \( \mathbf{Q} = \mathbf{X}\mathbf{W}_Q \) | \( (s, d) \times (d, d_h) \) |
| `matmul` | Attention scores \( \mathbf{Q}\mathbf{K}^{\!\top} \) | \( (s, d_h) \times (d_h, s) \) |
| `matmul` | Feed-forward up-projection | \( (s, d) \times (d, 4d) \) |
| `add` | Residual connection \( \mathbf{X} + \text{Attn}(\mathbf{X}) \) | \( (s, d) \) |
| `add` | Bias addition | \( (s, d) + (d,) \) (broadcast) |
| `elementWiseMultiply` | SwiGLU gate \( \sigma(\mathbf{g}) \odot \mathbf{v} \) | \( (s, 4d) \) |
| `fill` | KV cache initialisation | \( (L, s, d_h) \) |
| `get` / `set` | KV cache update at position \( t \) | \( O(1) \) per element |

---

## 8. Code Example: Simulating a Q-Projection

This complete example mirrors the test in `tests/unit/test_tensor.zig`:

```zig
const std = @import("std");
const Tensor = @import("foundation/tensor.zig").Tensor;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Two tokens, three features each
    var input = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer input.deinit();

    // Weight matrix: 3 features -> 2 query dims
    var w_q = try Tensor(f32).init(allocator, &[_]usize{ 3, 2 });
    defer w_q.deinit();

    // Populate input: token 0 = [1, 0, 1], token 1 = [0, 1, 1]
    try input.set(&[_]usize{ 0, 0 }, 1.0);
    try input.set(&[_]usize{ 0, 2 }, 1.0);
    try input.set(&[_]usize{ 1, 1 }, 1.0);
    try input.set(&[_]usize{ 1, 2 }, 1.0);

    // W_q = [[1,0],[0,1],[1,0]]
    try w_q.set(&[_]usize{ 0, 0 }, 1.0);
    try w_q.set(&[_]usize{ 1, 1 }, 1.0);
    try w_q.set(&[_]usize{ 2, 0 }, 1.0);

    // Q = input @ W_q
    var queries = try input.matmul(w_q, allocator);
    defer queries.deinit();

    // queries[0] = [2, 0], queries[1] = [1, 1]
    const writer = std.io.getStdOut().writer();
    try queries.print(writer);
}
```

---

## References

[^1]: Huang, J. et al. "Strassen's Algorithm Reloaded on GPUs." *SC '24*, 2024.
[^2]: Vaswani, A. et al. "Attention Is All You Need." *NeurIPS*, 2017.
[^3]: Dauphin, Y. et al. "Language Modeling with Gated Convolutional Networks." *ICML*, 2017.
[^4]: Shazeer, N. "GLU Variants Improve Transformer." *arXiv:2002.05202*, 2020.
