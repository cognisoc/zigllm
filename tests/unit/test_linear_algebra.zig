//! Comprehensive test suite for Linear Algebra layer
//!
//! This test suite validates SIMD-optimized matrix operations and quantization
//! functionality, ensuring both correctness and performance of educational
//! implementations that will form the backbone of transformer computations.
//!
//! ## Educational Focus
//! These tests demonstrate:
//! - How SIMD optimizations maintain mathematical correctness
//! - Quantization accuracy vs compression trade-offs
//! - Performance characteristics of cache-blocking algorithms
//! - Memory alignment requirements for vectorized operations

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const Tensor = @import("../../src/foundation/tensor.zig").Tensor;
const matrix_ops = @import("../../src/linear_algebra/matrix_ops.zig");
const quantization = @import("../../src/linear_algebra/quantization.zig");

// Test allocator for memory leak detection
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const test_allocator = gpa.allocator();

/// Test SIMD matrix multiplication correctness against reference implementation
test "SIMD matrix multiplication correctness" {
    defer _ = gpa.deinit();

    // Test dimensions chosen to exercise both SIMD and scalar fallback paths
    const test_cases = [_]struct { m: usize, n: usize, k: usize }{
        .{ .m = 4, .n = 4, .k = 4 },     // Small, fits in SIMD registers
        .{ .m = 8, .n = 8, .k = 8 },     // Medium, multiple SIMD operations
        .{ .m = 17, .n = 13, .k = 19 },  // Odd dimensions, tests scalar fallback
        .{ .m = 64, .n = 64, .k = 64 },  // Large, tests cache blocking
    };

    for (test_cases) |case| {
        // Create test matrices with predictable values
        var a = try Tensor(f32).init(test_allocator, &[_]usize{ case.m, case.k });
        defer a.deinit();
        var b = try Tensor(f32).init(test_allocator, &[_]usize{ case.k, case.n });
        defer b.deinit();

        // Fill with structured test data
        for (0..case.m) |i| {
            for (0..case.k) |j| {
                const val = @as(f32, @floatFromInt(i * case.k + j + 1)) / 10.0;
                try a.set(&[_]usize{ i, j }, val);
            }
        }

        for (0..case.k) |i| {
            for (0..case.n) |j| {
                const val = @as(f32, @floatFromInt(i * case.n + j + 1)) / 20.0;
                try b.set(&[_]usize{ i, j }, val);
            }
        }

        // Test SIMD implementation
        var result_simd = try matrix_ops.matmulSIMD(f32, a, b, test_allocator);
        defer result_simd.deinit();

        // Test against reference (foundation tensor matmul)
        var result_ref = try a.matmul(b, test_allocator);
        defer result_ref.deinit();

        // Verify shapes match
        try testing.expectEqualSlices(usize, result_simd.shape, result_ref.shape);

        // Verify values match within floating point tolerance
        const tolerance = 1e-5;
        for (0..case.m) |i| {
            for (0..case.n) |j| {
                const simd_val = try result_simd.get(&[_]usize{ i, j });
                const ref_val = try result_ref.get(&[_]usize{ i, j });
                const diff = @abs(simd_val - ref_val);
                try testing.expect(diff < tolerance);
            }
        }
    }
}

/// Test memory alignment for SIMD operations
test "SIMD memory alignment verification" {
    defer _ = gpa.deinit();

    // Create tensor with SIMD-aligned memory
    var tensor = try matrix_ops.createAlignedTensor(f32, &[_]usize{ 16, 16 }, test_allocator);
    defer tensor.deinit();

    // Verify data pointer is properly aligned for AVX operations (32-byte alignment)
    const alignment = @alignOf(@Vector(8, f32)); // AVX requires 32-byte alignment
    const ptr_value = @intFromPtr(tensor.data.ptr);
    try testing.expect(ptr_value % alignment == 0);

    // Fill and verify data integrity
    for (0..16) |i| {
        for (0..16) |j| {
            const val = @as(f32, @floatFromInt(i * 16 + j));
            try tensor.set(&[_]usize{ i, j }, val);
        }
    }

    // Verify all values were stored correctly
    for (0..16) |i| {
        for (0..16) |j| {
            const expected = @as(f32, @floatFromInt(i * 16 + j));
            const actual = try tensor.get(&[_]usize{ i, j });
            try testing.expectEqual(expected, actual);
        }
    }
}

/// Test cache-blocking algorithm performance characteristics
test "cache blocking algorithm correctness" {
    defer _ = gpa.deinit();

    // Create large matrices to test cache blocking
    const size = 128; // Large enough to exceed L1 cache
    var a = try Tensor(f32).init(test_allocator, &[_]usize{ size, size });
    defer a.deinit();
    var b = try Tensor(f32).init(test_allocator, &[_]usize{ size, size });
    defer b.deinit();

    // Initialize with identity-like patterns for predictable results
    for (0..size) |i| {
        for (0..size) |j| {
            try a.set(&[_]usize{ i, j }, if (i == j) 2.0 else 0.1);
            try b.set(&[_]usize{ i, j }, if (i == j) 0.5 else 0.05);
        }
    }

    // Test cache-blocked multiplication
    var result = try matrix_ops.matmulCacheBlocked(f32, a, b, test_allocator, 64);
    defer result.deinit();

    // Verify diagonal elements (should be approximately 2.0 * 0.5 = 1.0)
    const tolerance = 1e-3;
    for (0..size) |i| {
        const diag_val = try result.get(&[_]usize{ i, i });
        try testing.expect(@abs(diag_val - 1.0) < tolerance);
    }
}

/// Test Q4_0 quantization accuracy and compression
test "Q4_0 quantization correctness" {
    defer _ = gpa.deinit();

    // Create test tensor with known value distribution
    var tensor = try Tensor(f32).init(test_allocator, &[_]usize{ 4, 8 });
    defer tensor.deinit();

    // Fill with values that test quantization ranges
    const test_values = [_]f32{
        -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5,
        2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
        -3.0, -2.5, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25,
        -4.0, -3.5, -1.75, -1.25, -0.75, 0.125, 3.25, 4.75
    };

    for (0..4) |i| {
        for (0..8) |j| {
            try tensor.set(&[_]usize{ i, j }, test_values[i * 8 + j]);
        }
    }

    // Quantize to Q4_0 format
    var quantized = try quantization.quantizeTensor(.Q4_0, tensor, test_allocator);
    defer quantized.deinit();

    // Dequantize back to f32
    var dequantized = try quantization.dequantizeTensor(quantized, test_allocator);
    defer dequantized.deinit();

    // Verify compression ratio (Q4_0 uses 4 bits per weight + scale factors)
    const original_bytes = tensor.data.len * @sizeOf(f32);
    const quantized_bytes = quantized.getMemoryUsage();
    const compression_ratio = @as(f32, @floatFromInt(original_bytes)) / @as(f32, @floatFromInt(quantized_bytes));
    try testing.expect(compression_ratio > 4.0); // Should be significantly compressed

    // Verify quantization accuracy (allow for quantization error)
    const max_error = 0.5; // Q4_0 has limited precision
    var total_error: f32 = 0.0;
    var max_observed_error: f32 = 0.0;

    for (0..4) |i| {
        for (0..8) |j| {
            const original = try tensor.get(&[_]usize{ i, j });
            const recovered = try dequantized.get(&[_]usize{ i, j });
            const error = @abs(original - recovered);

            total_error += error;
            max_observed_error = @max(max_observed_error, error);

            try testing.expect(error <= max_error);
        }
    }

    // Verify average error is reasonable
    const avg_error = total_error / @as(f32, @floatFromInt(tensor.data.len));
    try testing.expect(avg_error < 0.2); // Average error should be much lower than max
}

/// Test Q8_0 quantization for higher precision requirements
test "Q8_0 quantization precision" {
    defer _ = gpa.deinit();

    // Create tensor with smooth gradient values
    var tensor = try Tensor(f32).init(test_allocator, &[_]usize{ 8, 8 });
    defer tensor.deinit();

    for (0..8) |i| {
        for (0..8) |j| {
            // Create smooth gradient from -2.0 to +2.0
            const val = -2.0 + 4.0 * (@as(f32, @floatFromInt(i * 8 + j)) / 63.0);
            try tensor.set(&[_]usize{ i, j }, val);
        }
    }

    // Quantize to Q8_0 (higher precision)
    var quantized = try quantization.quantizeTensor(.Q8_0, tensor, test_allocator);
    defer quantized.deinit();

    var dequantized = try quantization.dequantizeTensor(quantized, test_allocator);
    defer dequantized.deinit();

    // Q8_0 should have much better precision than Q4_0
    const max_error = 0.1; // Much tighter tolerance
    for (0..8) |i| {
        for (0..8) |j| {
            const original = try tensor.get(&[_]usize{ i, j });
            const recovered = try dequantized.get(&[_]usize{ i, j });
            const error = @abs(original - recovered);
            try testing.expect(error <= max_error);
        }
    }
}

/// Test INT8 quantization for maximum compatibility
test "INT8 quantization compatibility" {
    defer _ = gpa.deinit();

    // Test tensor with extreme values to test range handling
    var tensor = try Tensor(f32).init(test_allocator, &[_]usize{ 4, 4 });
    defer tensor.deinit();

    const extreme_values = [_]f32{
        -10.0, -5.0, -1.0, -0.1,
        0.0, 0.1, 1.0, 5.0,
        10.0, 15.0, -15.0, -20.0,
        25.0, -25.0, 100.0, -100.0
    };

    for (0..4) |i| {
        for (0..4) |j| {
            try tensor.set(&[_]usize{ i, j }, extreme_values[i * 4 + j]);
        }
    }

    // Quantize to INT8
    var quantized = try quantization.quantizeTensor(.INT8, tensor, test_allocator);
    defer quantized.deinit();

    var dequantized = try quantization.dequantizeTensor(quantized, test_allocator);
    defer dequantized.dequantized.deinit();

    // Verify all values are within INT8 representable range after scaling
    for (0..4) |i| {
        for (0..4) |j| {
            const recovered = try dequantized.get(&[_]usize{ i, j });

            // Values should be finite and reasonable
            try testing.expect(std.math.isFinite(recovered));
            try testing.expect(@abs(recovered) < 1000.0); // Sanity check
        }
    }
}

/// Performance benchmark for SIMD vs scalar operations
test "SIMD performance characteristics" {
    defer _ = gpa.deinit();

    const sizes = [_]usize{ 32, 64, 128 };

    for (sizes) |size| {
        var a = try matrix_ops.createAlignedTensor(f32, &[_]usize{ size, size }, test_allocator);
        defer a.deinit();
        var b = try matrix_ops.createAlignedTensor(f32, &[_]usize{ size, size }, test_allocator);
        defer b.deinit();

        // Fill with random-like values
        for (0..size) |i| {
            for (0..size) |j| {
                const val = @sin(@as(f32, @floatFromInt(i + j)));
                try a.set(&[_]usize{ i, j }, val);
                try b.set(&[_]usize{ i, j }, val * 0.5);
            }
        }

        // Time SIMD implementation
        const simd_start = std.time.nanoTimestamp();
        var simd_result = try matrix_ops.matmulSIMD(f32, a, b, test_allocator);
        const simd_end = std.time.nanoTimestamp();
        defer simd_result.deinit();

        // Time reference implementation
        const ref_start = std.time.nanoTimestamp();
        var ref_result = try a.matmul(b, test_allocator);
        const ref_end = std.time.nanoTimestamp();
        defer ref_result.deinit();

        const simd_time = simd_end - simd_start;
        const ref_time = ref_end - ref_start;

        // SIMD should be faster (though not guaranteed in all cases due to overhead)
        // This is more of a performance characteristic observation than a strict test
        std.debug.print("Size {}: SIMD={}ns, Ref={}ns, Ratio={:.2f}\n", .{
            size, simd_time, ref_time,
            @as(f64, @floatFromInt(ref_time)) / @as(f64, @floatFromInt(simd_time))
        });

        // Verify results are equivalent
        const tolerance = 1e-5;
        for (0..size) |i| {
            for (0..size) |j| {
                const simd_val = try simd_result.get(&[_]usize{ i, j });
                const ref_val = try ref_result.get(&[_]usize{ i, j });
                try testing.expect(@abs(simd_val - ref_val) < tolerance);
            }
        }
    }
}

/// Test comprehensive quantization round-trip accuracy
test "quantization round-trip fidelity" {
    defer _ = gpa.deinit();

    // Test with tensor representing typical neural network weights
    var weights = try Tensor(f32).init(test_allocator, &[_]usize{ 16, 16 });
    defer weights.deinit();

    // Fill with normally distributed values (typical for neural network weights)
    var rng = std.rand.DefaultPrng.init(42);
    const random = rng.random();

    for (0..16) |i| {
        for (0..16) |j| {
            // Approximate normal distribution using Box-Muller transform
            const u1 = random.float(f32);
            const u2 = random.float(f32);
            const normal = @sqrt(-2.0 * @log(u1)) * @cos(2.0 * std.math.pi * u2);
            try weights.set(&[_]usize{ i, j }, normal * 0.5); // Scale for typical weight range
        }
    }

    const quant_types = [_]quantization.QuantType{ .Q4_0, .Q8_0, .INT8 };
    const expected_errors = [_]f32{ 0.3, 0.05, 0.1 }; // Expected max errors for each type

    for (quant_types, expected_errors) |quant_type, max_error| {
        // Test quantization round-trip
        var quantized = try quantization.quantizeTensor(quant_type, weights, test_allocator);
        defer quantized.deinit();

        var recovered = try quantization.dequantizeTensor(quantized, test_allocator);
        defer recovered.deinit();

        // Compute statistics
        var max_abs_error: f32 = 0.0;
        var total_squared_error: f32 = 0.0;

        for (0..16) |i| {
            for (0..16) |j| {
                const original = try weights.get(&[_]usize{ i, j });
                const reconstructed = try recovered.get(&[_]usize{ i, j });
                const error = @abs(original - reconstructed);
                const squared_error = error * error;

                max_abs_error = @max(max_abs_error, error);
                total_squared_error += squared_error;
            }
        }

        const mse = total_squared_error / @as(f32, @floatFromInt(weights.data.len));
        const rmse = @sqrt(mse);

        // Verify error bounds
        try testing.expect(max_abs_error <= max_error);
        try testing.expect(rmse <= max_error * 0.5); // RMSE should be much lower than max error

        std.debug.print("Quantization {}: Max Error={:.4f}, RMSE={:.4f}\n", .{
            quant_type, max_abs_error, rmse
        });
    }
}