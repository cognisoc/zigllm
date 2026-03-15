// Unit Tests for Foundation Layer: Tensor Operations
//
// Comprehensive test suite for tensor functionality following our
// educational testing principles:
// - Test correctness against mathematical definitions
// - Test edge cases and error conditions
// - Test performance characteristics
// - Provide educational context for each test

const std = @import("std");
const testing = std.testing;
const Tensor = @import("../../src/foundation/tensor.zig").Tensor;
const TensorError = @import("../../src/foundation/tensor.zig").TensorError;

// Test groups organized by functionality

// ============================================================================
// CREATION AND BASIC PROPERTIES
// ============================================================================

test "tensor creation - various shapes and types" {
    const allocator = testing.allocator;

    // Test different data types
    {
        var f32_tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
        defer f32_tensor.deinit();
        try testing.expectEqual(@as(usize, 6), f32_tensor.size);
    }

    {
        var i32_tensor = try Tensor(i32).init(allocator, &[_]usize{ 4 });
        defer i32_tensor.deinit();
        try testing.expectEqual(@as(usize, 4), i32_tensor.size);
    }

    // Test various dimensionalities
    {
        var tensor_1d = try Tensor(f32).init(allocator, &[_]usize{10});
        defer tensor_1d.deinit();
        try testing.expectEqual(@as(usize, 1), tensor_1d.ndim());
    }

    {
        var tensor_3d = try Tensor(f32).init(allocator, &[_]usize{ 2, 3, 4 });
        defer tensor_3d.deinit();
        try testing.expectEqual(@as(usize, 3), tensor_3d.ndim());
        try testing.expectEqual(@as(usize, 24), tensor_3d.size);
    }

    {
        var tensor_4d = try Tensor(f32).init(allocator, &[_]usize{ 1, 2, 3, 4 });
        defer tensor_4d.deinit();
        try testing.expectEqual(@as(usize, 4), tensor_4d.ndim());
        try testing.expectEqual(@as(usize, 24), tensor_4d.size);
    }
}

test "tensor strides calculation" {
    const allocator = testing.allocator;

    // Test stride calculation for different shapes
    {
        // 1D tensor: [5] -> strides: [1]
        var tensor = try Tensor(f32).init(allocator, &[_]usize{5});
        defer tensor.deinit();
        try testing.expectEqual(@as(usize, 1), tensor.strides[0]);
    }

    {
        // 2D tensor: [3, 4] -> strides: [4, 1]
        var tensor = try Tensor(f32).init(allocator, &[_]usize{ 3, 4 });
        defer tensor.deinit();
        try testing.expectEqual(@as(usize, 4), tensor.strides[0]);
        try testing.expectEqual(@as(usize, 1), tensor.strides[1]);
    }

    {
        // 3D tensor: [2, 3, 4] -> strides: [12, 4, 1]
        var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3, 4 });
        defer tensor.deinit();
        try testing.expectEqual(@as(usize, 12), tensor.strides[0]);
        try testing.expectEqual(@as(usize, 4), tensor.strides[1]);
        try testing.expectEqual(@as(usize, 1), tensor.strides[2]);
    }
}

// ============================================================================
// DATA ACCESS AND MANIPULATION
// ============================================================================

test "tensor indexing - comprehensive bounds checking" {
    const allocator = testing.allocator;
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 3, 4, 2 });
    defer tensor.deinit();

    // Valid indices
    try tensor.set(&[_]usize{ 0, 0, 0 }, 1.0);
    try tensor.set(&[_]usize{ 2, 3, 1 }, 2.0);
    try testing.expectEqual(@as(f32, 1.0), try tensor.get(&[_]usize{ 0, 0, 0 }));
    try testing.expectEqual(@as(f32, 2.0), try tensor.get(&[_]usize{ 2, 3, 1 }));

    // Invalid indices - out of bounds
    try testing.expectError(TensorError.InvalidIndex, tensor.get(&[_]usize{ 3, 0, 0 }));
    try testing.expectError(TensorError.InvalidIndex, tensor.get(&[_]usize{ 0, 4, 0 }));
    try testing.expectError(TensorError.InvalidIndex, tensor.get(&[_]usize{ 0, 0, 2 }));

    // Invalid indices - wrong number of dimensions
    try testing.expectError(TensorError.InvalidIndex, tensor.get(&[_]usize{ 0, 0 }));
    try testing.expectError(TensorError.InvalidIndex, tensor.get(&[_]usize{ 0, 0, 0, 0 }));
}

test "tensor fill operations" {
    const allocator = testing.allocator;
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 3, 3 });
    defer tensor.deinit();

    // Fill with specific value
    tensor.fill(2.5);

    // Verify all elements are filled
    for (0..3) |i| {
        for (0..3) |j| {
            try testing.expectEqual(@as(f32, 2.5), try tensor.get(&[_]usize{ i, j }));
        }
    }

    // Test with different value
    tensor.fill(-1.0);
    try testing.expectEqual(@as(f32, -1.0), try tensor.get(&[_]usize{ 1, 1 }));
}

// ============================================================================
// MATHEMATICAL OPERATIONS
// ============================================================================

test "tensor addition - educational examples" {
    const allocator = testing.allocator;

    // Example: Adding bias to activations (common in neural networks)
    var activations = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer activations.deinit();
    var bias = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer bias.deinit();

    // Set activation values
    try activations.set(&[_]usize{ 0, 0 }, 1.0);
    try activations.set(&[_]usize{ 0, 1 }, 2.0);
    try activations.set(&[_]usize{ 0, 2 }, 3.0);
    try activations.set(&[_]usize{ 1, 0 }, 4.0);
    try activations.set(&[_]usize{ 1, 1 }, 5.0);
    try activations.set(&[_]usize{ 1, 2 }, 6.0);

    // Set bias values
    bias.fill(0.1);

    var result = try activations.add(bias, allocator);
    defer result.deinit();

    try testing.expectEqual(@as(f32, 1.1), try result.get(&[_]usize{ 0, 0 }));
    try testing.expectEqual(@as(f32, 2.1), try result.get(&[_]usize{ 0, 1 }));
    try testing.expectEqual(@as(f32, 6.1), try result.get(&[_]usize{ 1, 2 }));
}

test "matrix multiplication - transformer scenarios" {
    const allocator = testing.allocator;

    // Scenario 1: Query projection in attention mechanism
    // Input embeddings: 3 tokens × 4 features
    // Query weights: 4 features → 2 query dimensions
    {
        var embeddings = try Tensor(f32).init(allocator, &[_]usize{ 3, 4 });
        defer embeddings.deinit();
        var query_weights = try Tensor(f32).init(allocator, &[_]usize{ 4, 2 });
        defer query_weights.deinit();

        // Identity-like embeddings for predictable results
        for (0..3) |i| {
            for (0..4) |j| {
                try embeddings.set(&[_]usize{ i, j }, if (i == j) 1.0 else 0.0);
            }
        }

        // Simple projection weights
        for (0..4) |i| {
            try query_weights.set(&[_]usize{ i, 0 }, @as(f32, @floatFromInt(i + 1)));
            try query_weights.set(&[_]usize{ i, 1 }, @as(f32, @floatFromInt((i + 1) * 2)));
        }

        var queries = try embeddings.matmul(query_weights, allocator);
        defer queries.deinit();

        try testing.expectEqual(@as(usize, 3), queries.shape[0]);
        try testing.expectEqual(@as(usize, 2), queries.shape[1]);

        // Token 0: [1,0,0,0] @ weights = [1, 2]
        try testing.expectEqual(@as(f32, 1.0), try queries.get(&[_]usize{ 0, 0 }));
        try testing.expectEqual(@as(f32, 2.0), try queries.get(&[_]usize{ 0, 1 }));
    }

    // Scenario 2: Batch processing
    // Simulate processing multiple examples simultaneously
    {
        var batch_input = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
        defer batch_input.deinit();
        var weights = try Tensor(f32).init(allocator, &[_]usize{ 3, 3 });
        defer weights.deinit();

        // Set up input batch
        try batch_input.set(&[_]usize{ 0, 0 }, 1.0);
        try batch_input.set(&[_]usize{ 0, 1 }, 2.0);
        try batch_input.set(&[_]usize{ 0, 2 }, 3.0);
        try batch_input.set(&[_]usize{ 1, 0 }, 4.0);
        try batch_input.set(&[_]usize{ 1, 1 }, 5.0);
        try batch_input.set(&[_]usize{ 1, 2 }, 6.0);

        // Identity matrix for simple verification
        for (0..3) |i| {
            for (0..3) |j| {
                try weights.set(&[_]usize{ i, j }, if (i == j) 1.0 else 0.0);
            }
        }

        var output = try batch_input.matmul(weights, allocator);
        defer output.deinit();

        // Output should be same as input (identity transform)
        try testing.expectEqual(@as(f32, 1.0), try output.get(&[_]usize{ 0, 0 }));
        try testing.expectEqual(@as(f32, 6.0), try output.get(&[_]usize{ 1, 2 }));
    }
}

// ============================================================================
// ERROR CONDITIONS AND EDGE CASES
// ============================================================================

test "error handling - comprehensive coverage" {
    const allocator = testing.allocator;

    // Invalid tensor creation
    try testing.expectError(TensorError.InvalidShape, Tensor(f32).init(allocator, &[_]usize{}));
    try testing.expectError(TensorError.InvalidShape, Tensor(f32).init(allocator, &[_]usize{ 0, 5 }));
    try testing.expectError(TensorError.InvalidShape, Tensor(f32).init(allocator, &[_]usize{ 3, 0, 4 }));

    // Incompatible operations
    var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer a.deinit();
    var b = try Tensor(f32).init(allocator, &[_]usize{ 3, 2 });
    defer b.deinit();

    // Addition with different shapes
    try testing.expectError(TensorError.IncompatibleShapes, a.add(b, allocator));

    // Matrix multiplication with wrong dimensions
    var c = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer c.deinit();
    var d = try Tensor(f32).init(allocator, &[_]usize{ 4, 2 }); // 3 ≠ 4
    defer d.deinit();

    try testing.expectError(TensorError.IncompatibleShapes, c.matmul(d, allocator));

    // Non-2D matrices for matmul
    var e = try Tensor(f32).init(allocator, &[_]usize{ 2, 3, 4 });
    defer e.deinit();
    var f = try Tensor(f32).init(allocator, &[_]usize{ 4, 2 });
    defer f.deinit();

    try testing.expectError(TensorError.IncompatibleShapes, e.matmul(f, allocator));
}

// ============================================================================
// PERFORMANCE AND MEMORY TESTS
// ============================================================================

test "memory usage - large tensors" {
    const allocator = testing.allocator;

    // Test that we can create and destroy large tensors without leaks
    const large_size = 1000;
    var large_tensor = try Tensor(f32).init(allocator, &[_]usize{ large_size, large_size });
    large_tensor.fill(1.0);

    // Verify some values
    try testing.expectEqual(@as(f32, 1.0), try large_tensor.get(&[_]usize{ 0, 0 }));
    try testing.expectEqual(@as(f32, 1.0), try large_tensor.get(&[_]usize{ 999, 999 }));

    large_tensor.deinit();
}

test "performance characteristics - matrix multiplication scaling" {
    const allocator = testing.allocator;

    // Test different matrix sizes to understand performance characteristics
    const sizes = [_]usize{ 8, 16, 32 };

    for (sizes) |size| {
        var a = try Tensor(f32).init(allocator, &[_]usize{ size, size });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ size, size });
        defer b.deinit();

        a.fill(1.0);
        b.fill(2.0);

        var result = try a.matmul(b, allocator);
        defer result.deinit();

        // Each element should be size * 1.0 * 2.0 = size * 2.0
        const expected = @as(f32, @floatFromInt(size * 2));
        try testing.expectEqual(expected, try result.get(&[_]usize{ 0, 0 }));
        try testing.expectEqual(expected, try result.get(&[_]usize{ size - 1, size - 1 }));
    }
}