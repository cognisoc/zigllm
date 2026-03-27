// Foundation Layer: Tensor Operations
//
// Educational implementation of tensor operations for transformer models.
// This is the foundational layer that all higher abstractions build upon.
//
// ## Learning Objectives
// - Understand multi-dimensional arrays and memory layout
// - Learn basic tensor operations used in neural networks
// - Connect mathematical operations to their computational implementation
//
// ## Mathematical Foundation
// A tensor is a generalization of scalars, vectors, and matrices:
// - 0D tensor (scalar): single number
// - 1D tensor (vector): array of numbers [a₁, a₂, ..., aₙ]
// - 2D tensor (matrix): rectangular array of numbers
// - nD tensor: n-dimensional array
//
// In transformers, tensors represent:
// - Token embeddings: [sequence_length, embedding_dim]
// - Attention weights: [batch_size, num_heads, seq_len, seq_len]
// - Model parameters: various shapes depending on layer

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Shape represents the dimensions of a tensor
/// Example: [2, 3, 4] represents a tensor with 2×3×4 = 24 elements
pub const Shape = []const usize;

/// TensorError represents possible tensor operation errors
pub const TensorError = error{
    InvalidShape,
    IncompatibleShapes,
    InvalidIndex,
    OutOfMemory,
};

/// Educational Tensor implementation
///
/// ## Design Philosophy
/// This tensor prioritizes educational clarity over raw performance.
/// Every operation includes documentation connecting it to transformer concepts.
///
/// ## Memory Layout
/// Tensors use row-major (C-style) memory layout where the rightmost dimension
/// changes fastest. For a 2×3 matrix:
/// ```
/// [[a, b, c],     Memory: [a, b, c, d, e, f]
///  [d, e, f]]
/// ```
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

        /// Initialize a new tensor with the given shape
        ///
        /// ## Educational Note: Memory Allocation
        /// Neural networks require careful memory management due to:
        /// - Large model sizes (billions of parameters)
        /// - Intermediate activations during forward/backward passes
        /// - Need for efficient cache utilization
        ///
        /// ## Parameters
        /// - `allocator`: Memory allocator for tensor data
        /// - `shape`: Dimensions of the tensor
        ///
        /// ## Returns
        /// New tensor initialized with zero values
        pub fn init(allocator: Allocator, shape: Shape) TensorError!Self {
            if (shape.len == 0) return TensorError.InvalidShape;

            // Calculate total size
            var size: usize = 1;
            for (shape) |dim| {
                if (dim == 0) return TensorError.InvalidShape;
                size *= dim;
            }

            // Allocate memory
            const data = allocator.alloc(T, size) catch return TensorError.OutOfMemory;
            errdefer allocator.free(data);

            // Copy shape (we need ownership)
            const owned_shape = allocator.dupe(usize, shape) catch return TensorError.OutOfMemory;
            errdefer allocator.free(owned_shape);

            // Calculate strides for efficient indexing
            const strides = allocator.alloc(usize, shape.len) catch return TensorError.OutOfMemory;
            errdefer allocator.free(strides);

            // Compute strides: rightmost dimension has stride 1
            if (shape.len > 0) {
                strides[shape.len - 1] = 1;
                if (shape.len > 1) {
                    var i: i32 = @intCast(shape.len - 2);
                    while (i >= 0) : (i -= 1) {
                        const ui: usize = @intCast(i);
                        strides[ui] = strides[ui + 1] * shape[ui + 1];
                    }
                }
            }

            // Initialize with zeros
            @memset(data, @as(T, 0));

            return Self{
                .data = data,
                .shape = owned_shape,
                .strides = strides,
                .size = size,
                .allocator = allocator,
            };
        }

        /// Clean up tensor memory
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.allocator.free(self.shape);
            self.allocator.free(self.strides);
        }

        /// Get number of dimensions
        pub fn ndim(self: Self) usize {
            return self.shape.len;
        }

        /// Fill tensor with a constant value
        ///
        /// ## Educational Note: Tensor Initialization
        /// Common initialization patterns in neural networks:
        /// - Zero initialization: For biases and some layer outputs
        /// - Random initialization: For weights (Xavier, He, etc.)
        /// - Constant initialization: For debugging and testing
        pub fn fill(self: *Self, value: T) void {
            @memset(self.data, value);
        }

        /// Get element at given indices
        ///
        /// ## Educational Note: Indexing
        /// Multi-dimensional indexing converts to 1D using strides:
        /// flat_index = sum(indices[i] * strides[i])
        ///
        /// This is how tensors map logical coordinates to memory addresses.
        pub fn get(self: Self, indices: []const usize) TensorError!T {
            if (indices.len != self.shape.len) return TensorError.InvalidIndex;

            var flat_index: usize = 0;
            for (indices, 0..) |idx, dim| {
                if (idx >= self.shape[dim]) return TensorError.InvalidIndex;
                flat_index += idx * self.strides[dim];
            }

            return self.data[flat_index];
        }

        /// Set element at given indices
        pub fn set(self: *Self, indices: []const usize, value: T) TensorError!void {
            if (indices.len != self.shape.len) return TensorError.InvalidIndex;

            var flat_index: usize = 0;
            for (indices, 0..) |idx, dim| {
                if (idx >= self.shape[dim]) return TensorError.InvalidIndex;
                flat_index += idx * self.strides[dim];
            }

            self.data[flat_index] = value;
        }

        /// Element-wise addition
        ///
        /// ## Educational Note: Broadcasting
        /// Real implementations support broadcasting (adding tensors of
        /// compatible but different shapes). We require exact shape matching
        /// for educational clarity.
        ///
        /// ## Transformer Usage
        /// - Adding positional embeddings to token embeddings
        /// - Residual connections in transformer blocks
        /// - Bias addition in linear layers
        pub fn add(self: Self, other: Self, allocator: Allocator) TensorError!Self {
            if (!std.mem.eql(usize, self.shape, other.shape)) {
                return TensorError.IncompatibleShapes;
            }

            var result = try Self.init(allocator, self.shape);

            for (0..self.size) |i| {
                result.data[i] = self.data[i] + other.data[i];
            }

            return result;
        }

        /// Element-wise multiplication (Hadamard product)
        ///
        /// ## Educational Note: Element-wise Operations in Transformers
        /// Element-wise multiplication is crucial for gating mechanisms:
        ///
        /// ### Gated Linear Units (GLU, SwiGLU, GeGLU)
        /// ```
        /// gate_values = sigmoid(linear1(x))  or  SiLU(linear1(x))
        /// content_values = linear2(x)
        /// output = gate_values ⊙ content_values  (⊙ = element-wise multiply)
        /// ```
        ///
        /// ### Attention Mechanisms
        /// ```
        /// attention_output = attention_weights ⊙ values
        /// ```
        ///
        /// ### Mathematical Definition
        /// For tensors A and B of same shape: C[i] = A[i] × B[i]
        pub fn elementWiseMultiply(self: Self, other: Self, allocator: Allocator) TensorError!Self {
            // Check that shapes match exactly
            if (!std.mem.eql(usize, self.shape, other.shape)) {
                return TensorError.IncompatibleShapes;
            }

            var result = try Self.init(allocator, self.shape);

            for (0..self.size) |i| {
                result.data[i] = self.data[i] * other.data[i];
            }

            return result;
        }

        /// Matrix multiplication for 2D tensors
        ///
        /// ## Educational Note: Matrix Multiplication in Transformers
        /// This is the most important operation in neural networks:
        ///
        /// ### Query-Key-Value Projections
        /// ```
        /// Q = X @ W_q  (input @ query_weights)
        /// K = X @ W_k  (input @ key_weights)
        /// V = X @ W_v  (input @ value_weights)
        /// ```
        ///
        /// ### Feed-Forward Networks
        /// ```
        /// hidden = X @ W1 + b1
        /// output = hidden @ W2 + b2
        /// ```
        ///
        /// ### Mathematical Definition
        /// For matrices A(m×n) and B(n×p), result C(m×p):
        /// C[i,j] = Σ(k=0 to n-1) A[i,k] × B[k,j]
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
                                sum += self.data[b * m * n + i * n + k] * other.data[k * p + j];
                            }
                            result.data[b * m * p + i * p + j] = sum;
                        }
                    }
                }
                return result;
            }

            // Validate 2D matrices
            if (self.ndim() != 2 or other.ndim() != 2) {
                return TensorError.IncompatibleShapes;
            }

            // Check compatible dimensions: (m×n) @ (n×p) = (m×p)
            const m = self.shape[0];
            const n = self.shape[1];
            const other_n = other.shape[0];
            const p = other.shape[1];

            if (n != other_n) return TensorError.IncompatibleShapes;

            var result = try Self.init(allocator, &[_]usize{ m, p });

            // Triple nested loop: the classic O(n³) matrix multiplication
            // TODO: Add SIMD optimizations in linear_algebra layer
            for (0..m) |i| {
                for (0..p) |j| {
                    var sum: T = 0;
                    for (0..n) |k| {
                        // A[i,k] * B[k,j]
                        const a_val = try self.get(&[_]usize{ i, k });
                        const b_val = try other.get(&[_]usize{ k, j });
                        sum += a_val * b_val;
                    }
                    try result.set(&[_]usize{ i, j }, sum);
                }
            }

            return result;
        }

        /// Print tensor for debugging and education
        ///
        /// ## Educational Note: Tensor Visualization
        /// Understanding tensor contents is crucial for debugging neural networks.
        /// This function provides human-readable tensor representation.
        pub fn print(self: Self, writer: anytype) !void {
            try writer.writeAll("Tensor(shape=[");
            for (self.shape, 0..) |dim, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("{}", .{dim});
            }
            try writer.print("], size={}, type={})\n", .{ self.size, T });

            // Print contents based on dimensionality
            switch (self.ndim()) {
                1 => try self.print1D(writer),
                2 => try self.print2D(writer),
                else => try self.printND(writer),
            }
        }

        fn print1D(self: Self, writer: anytype) !void {
            try writer.writeAll("[");
            for (0..self.shape[0]) |i| {
                if (i > 0) try writer.writeAll(", ");
                const val = try self.get(&[_]usize{i});
                try writer.print("{d:.3}", .{val});
            }
            try writer.writeAll("]\n");
        }

        fn print2D(self: Self, writer: anytype) !void {
            try writer.writeAll("[\n");
            for (0..self.shape[0]) |i| {
                try writer.writeAll("  [");
                for (0..self.shape[1]) |j| {
                    if (j > 0) try writer.writeAll(", ");
                    const val = try self.get(&[_]usize{ i, j });
                    try writer.print("{d:.3}", .{val});
                }
                try writer.writeAll("]");
                if (i < self.shape[0] - 1) try writer.writeAll(",");
                try writer.writeAll("\n");
            }
            try writer.writeAll("]\n");
        }

        fn printND(self: Self, writer: anytype) !void {
            try writer.writeAll("Data: [");
            const max_display = @min(self.size, 20);
            for (0..max_display) |i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("{d:.3}", .{self.data[i]});
            }
            if (self.size > max_display) {
                try writer.print("... ({} more)", .{self.size - max_display});
            }
            try writer.writeAll("]\n");
        }
    };
}

// ============================================================================
// COMPREHENSIVE TESTS
// ============================================================================

test "tensor creation and basic properties" {
    const allocator = testing.allocator;

    // Test 1D tensor
    var tensor1d = try Tensor(f32).init(allocator, &[_]usize{5});
    defer tensor1d.deinit();

    try testing.expectEqual(@as(usize, 1), tensor1d.ndim());
    try testing.expectEqual(@as(usize, 5), tensor1d.size);
    try testing.expectEqual(@as(usize, 5), tensor1d.shape[0]);

    // Test 2D tensor
    var tensor2d = try Tensor(f32).init(allocator, &[_]usize{ 3, 4 });
    defer tensor2d.deinit();

    try testing.expectEqual(@as(usize, 2), tensor2d.ndim());
    try testing.expectEqual(@as(usize, 12), tensor2d.size);
    try testing.expectEqual(@as(usize, 3), tensor2d.shape[0]);
    try testing.expectEqual(@as(usize, 4), tensor2d.shape[1]);
}

test "tensor indexing and data access" {
    const allocator = testing.allocator;

    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor.deinit();

    // Test setting and getting values
    try tensor.set(&[_]usize{ 0, 0 }, 1.5);
    try tensor.set(&[_]usize{ 1, 2 }, 2.7);

    try testing.expectEqual(@as(f32, 1.5), try tensor.get(&[_]usize{ 0, 0 }));
    try testing.expectEqual(@as(f32, 2.7), try tensor.get(&[_]usize{ 1, 2 }));
    try testing.expectEqual(@as(f32, 0.0), try tensor.get(&[_]usize{ 0, 1 }));

    // Test bounds checking
    try testing.expectError(TensorError.InvalidIndex, tensor.get(&[_]usize{ 2, 0 }));
    try testing.expectError(TensorError.InvalidIndex, tensor.get(&[_]usize{ 0, 3 }));
}

test "tensor fill operation" {
    const allocator = testing.allocator;

    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
    defer tensor.deinit();

    tensor.fill(3.14);

    for (0..2) |i| {
        for (0..2) |j| {
            try testing.expectEqual(@as(f32, 3.14), try tensor.get(&[_]usize{ i, j }));
        }
    }
}

test "tensor addition" {
    const allocator = testing.allocator;

    var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
    defer a.deinit();
    var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
    defer b.deinit();

    // Fill tensors with test data
    try a.set(&[_]usize{ 0, 0 }, 1.0);
    try a.set(&[_]usize{ 0, 1 }, 2.0);
    try a.set(&[_]usize{ 1, 0 }, 3.0);
    try a.set(&[_]usize{ 1, 1 }, 4.0);

    try b.set(&[_]usize{ 0, 0 }, 0.5);
    try b.set(&[_]usize{ 0, 1 }, 1.5);
    try b.set(&[_]usize{ 1, 0 }, 2.5);
    try b.set(&[_]usize{ 1, 1 }, 3.5);

    var result = try a.add(b, allocator);
    defer result.deinit();

    try testing.expectEqual(@as(f32, 1.5), try result.get(&[_]usize{ 0, 0 }));
    try testing.expectEqual(@as(f32, 3.5), try result.get(&[_]usize{ 0, 1 }));
    try testing.expectEqual(@as(f32, 5.5), try result.get(&[_]usize{ 1, 0 }));
    try testing.expectEqual(@as(f32, 7.5), try result.get(&[_]usize{ 1, 1 }));
}

test "matrix multiplication - transformer example" {
    const allocator = testing.allocator;

    // Simulate a simple transformer operation: X @ W_q
    // Input: 2 tokens × 3 features
    var input = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer input.deinit();

    // Weight matrix: 3 features → 2 query dimensions
    var query_weights = try Tensor(f32).init(allocator, &[_]usize{ 3, 2 });
    defer query_weights.deinit();

    // Fill with known values for predictable results
    try input.set(&[_]usize{ 0, 0 }, 1.0);
    try input.set(&[_]usize{ 0, 1 }, 0.0);
    try input.set(&[_]usize{ 0, 2 }, 1.0);
    try input.set(&[_]usize{ 1, 0 }, 0.0);
    try input.set(&[_]usize{ 1, 1 }, 1.0);
    try input.set(&[_]usize{ 1, 2 }, 1.0);

    try query_weights.set(&[_]usize{ 0, 0 }, 1.0);
    try query_weights.set(&[_]usize{ 0, 1 }, 0.0);
    try query_weights.set(&[_]usize{ 1, 0 }, 0.0);
    try query_weights.set(&[_]usize{ 1, 1 }, 1.0);
    try query_weights.set(&[_]usize{ 2, 0 }, 1.0);
    try query_weights.set(&[_]usize{ 2, 1 }, 0.0);

    var queries = try input.matmul(query_weights, allocator);
    defer queries.deinit();

    // Verify results
    try testing.expectEqual(@as(usize, 2), queries.shape[0]);
    try testing.expectEqual(@as(usize, 2), queries.shape[1]);

    // Token 0: [1,0,1] @ [[1,0],[0,1],[1,0]] = [2,0]
    try testing.expectEqual(@as(f32, 2.0), try queries.get(&[_]usize{ 0, 0 }));
    try testing.expectEqual(@as(f32, 0.0), try queries.get(&[_]usize{ 0, 1 }));

    // Token 1: [0,1,1] @ [[1,0],[0,1],[1,0]] = [1,1]
    try testing.expectEqual(@as(f32, 1.0), try queries.get(&[_]usize{ 1, 0 }));
    try testing.expectEqual(@as(f32, 1.0), try queries.get(&[_]usize{ 1, 1 }));
}

test "error handling" {
    const allocator = testing.allocator;

    // Invalid shape (empty)
    try testing.expectError(TensorError.InvalidShape, Tensor(f32).init(allocator, &[_]usize{}));

    // Invalid shape (zero dimension)
    try testing.expectError(TensorError.InvalidShape, Tensor(f32).init(allocator, &[_]usize{ 3, 0, 2 }));

    // Incompatible shapes for addition
    var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer a.deinit();
    var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
    defer b.deinit();

    try testing.expectError(TensorError.IncompatibleShapes, a.add(b, allocator));

    // Incompatible shapes for matrix multiplication
    var c = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer c.deinit();
    var d = try Tensor(f32).init(allocator, &[_]usize{ 4, 2 }); // Wrong: 3 ≠ 4
    defer d.deinit();

    try testing.expectError(TensorError.IncompatibleShapes, c.matmul(d, allocator));
}