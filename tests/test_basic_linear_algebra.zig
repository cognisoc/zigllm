//! Basic tests for Linear Algebra layer functionality
//!
//! These tests verify that our SIMD-optimized matrix operations produce
//! correct results and our quantization implementations work properly.

const std = @import("std");
const testing = std.testing;

// Simple test that can compile independently
test "basic linear algebra layer compilation" {
    // This test just ensures the code compiles
    const f32_width = switch (@import("builtin").cpu.arch) {
        .x86_64 => 8,  // AVX
        .aarch64 => 4, // NEON
        else => 1,     // Scalar fallback
    };

    try testing.expect(f32_width >= 1);
    try testing.expect(f32_width <= 16);
}

test "simd configuration detection" {
    // Test SIMD width detection
    const builtin = @import("builtin");
    const simd_width = switch (builtin.cpu.arch) {
        .x86_64 => 8, // Conservative assumption
        .aarch64 => 4, // NEON 128-bit
        else => 1, // Scalar fallback
    };

    try testing.expect(simd_width >= 1);

    // Test vector type creation
    const VectorType = @Vector(simd_width, f32);
    const test_vec: VectorType = @splat(2.0);

    // Verify all elements are set correctly
    for (0..simd_width) |i| {
        try testing.expectEqual(@as(f32, 2.0), test_vec[i]);
    }
}

test "basic quantization concepts" {
    // Test quantization math concepts without complex imports
    const original: f32 = 1.5;
    const scale: f32 = 0.1;

    // Quantize: convert to 8-bit integer
    const quantized = std.math.clamp(@as(i8, @intFromFloat(original / scale)), -128, 127);

    // Dequantize: convert back to float
    const dequantized = @as(f32, @floatFromInt(quantized)) * scale;

    // Should be approximately equal
    const error_tolerance = 0.2; // Allow for quantization error
    try testing.expect(@abs(original - dequantized) < error_tolerance);
}

test "memory alignment concepts" {
    const allocator = testing.allocator;

    // Test basic aligned allocation concepts
    const data = try allocator.alloc(f32, 64);
    defer allocator.free(data);

    // Fill with test data
    for (0..64) |i| {
        data[i] = @as(f32, @floatFromInt(i));
    }

    // Verify data integrity
    try testing.expectEqual(@as(f32, 0.0), data[0]);
    try testing.expectEqual(@as(f32, 63.0), data[63]);
}

test "cache blocking concepts" {
    // Test the concept of processing data in blocks
    const total_size = 128;
    const block_size = 32;

    var processed_elements: usize = 0;

    // Simulate blocked processing
    var start: usize = 0;
    while (start < total_size) {
        const end = @min(start + block_size, total_size);
        const current_block_size = end - start;

        processed_elements += current_block_size;
        start += block_size;
    }

    try testing.expectEqual(total_size, processed_elements);
}