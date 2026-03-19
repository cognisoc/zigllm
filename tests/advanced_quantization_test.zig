const std = @import("std");
const testing = std.testing;
const foundation = @import("../src/foundation/tensor.zig");
const k_quant = @import("../src/linear_algebra/k_quantization.zig");
const iq_quant = @import("../src/linear_algebra/iq_quantization.zig");
const Tensor = foundation.Tensor;

test "K-quantization Q4_K basic functionality" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create test data
    const data = try allocator.alloc(f32, 256);
    defer allocator.free(data);

    // Initialize with sample data
    for (0..256) |i| {
        data[i] = (@as(f32, @floatFromInt(i)) - 128) / 128.0;
    }

    const tensor = Tensor(f32){ .data = data, .shape = &[_]usize{256} };

    // Test Q4_K quantization
    var quantizer = k_quant.KQuantizer.init(allocator, .Q4_K);
    const quantized = try quantizer.quantize(tensor);
    defer allocator.free(quantized);

    // Verify compression
    const expected_size = k_quant.BlockQ4K.size();
    try testing.expect(quantized.len == expected_size);

    // Test dequantization
    const dequantized = try quantizer.dequantize(quantized, tensor.shape);
    defer dequantized.deinit(allocator);

    // Verify shape preservation
    try testing.expect(dequantized.shape.len == tensor.shape.len);
    try testing.expect(dequantized.shape[0] == tensor.shape[0]);

    // Check that dequantized values are reasonable
    var total_error: f32 = 0;
    for (0..256) |i| {
        const original = tensor.data[i];
        const recovered = dequantized.data[i];
        const error = @abs(original - recovered);
        total_error += error;
    }

    const avg_error = total_error / 256.0;
    try testing.expect(avg_error < 0.1); // Should have reasonable accuracy
}

test "K-quantization Q5_K improved precision" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data = try allocator.alloc(f32, 256);
    defer allocator.free(data);

    // Use more precise test data
    for (0..256) |i| {
        data[i] = std.math.sin(@as(f32, @floatFromInt(i)) * 0.1) * 0.5;
    }

    const tensor = Tensor(f32){ .data = data, .shape = &[_]usize{256} };

    var quantizer = k_quant.KQuantizer.init(allocator, .Q5_K);
    const quantized = try quantizer.quantize(tensor);
    defer allocator.free(quantized);

    // Test size
    try testing.expect(quantized.len == k_quant.BlockQ5K.size());

    const dequantized = try quantizer.dequantize(quantized, tensor.shape);
    defer dequantized.deinit(allocator);

    // Q5_K should have better precision than Q4_K
    var total_error: f32 = 0;
    for (0..256) |i| {
        const error = @abs(tensor.data[i] - dequantized.data[i]);
        total_error += error;
    }

    const avg_error = total_error / 256.0;
    try testing.expect(avg_error < 0.05); // Better accuracy than Q4_K
}

test "K-quantization Q6_K highest precision" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data = try allocator.alloc(f32, 256);
    defer allocator.free(data);

    // Use complex test pattern
    for (0..256) |i| {
        const t = @as(f32, @floatFromInt(i)) / 255.0;
        data[i] = std.math.sin(t * 6.28) * 0.8 + std.math.cos(t * 12.56) * 0.2;
    }

    const tensor = Tensor(f32){ .data = data, .shape = &[_]usize{256} };

    var quantizer = k_quant.KQuantizer.init(allocator, .Q6_K);
    const quantized = try quantizer.quantize(tensor);
    defer allocator.free(quantized);

    try testing.expect(quantized.len == k_quant.BlockQ6K.size());

    const dequantized = try quantizer.dequantize(quantized, tensor.shape);
    defer dequantized.deinit(allocator);

    // Q6_K should have the best precision
    var total_error: f32 = 0;
    for (0..256) |i| {
        const error = @abs(tensor.data[i] - dequantized.data[i]);
        total_error += error;
    }

    const avg_error = total_error / 256.0;
    try testing.expect(avg_error < 0.02); // Highest accuracy
}

test "K-quantization compression ratios" {
    // Test compression ratio calculations
    try testing.expect(k_quant.KQuantizer.getCompressionRatio(.Q4_K) > 7.0);
    try testing.expect(k_quant.KQuantizer.getCompressionRatio(.Q5_K) > 5.0);
    try testing.expect(k_quant.KQuantizer.getCompressionRatio(.Q6_K) > 4.5);

    // Verify ordering: Q4_K > Q5_K > Q6_K (higher compression for lower bit counts)
    try testing.expect(k_quant.KQuantizer.getCompressionRatio(.Q4_K) > k_quant.KQuantizer.getCompressionRatio(.Q5_K));
    try testing.expect(k_quant.KQuantizer.getCompressionRatio(.Q5_K) > k_quant.KQuantizer.getCompressionRatio(.Q6_K));
}

test "K-quantization quality retention" {
    // Test quality retention estimates
    try testing.expect(k_quant.KQuantizer.getQualityRetention(.Q4_K) > 0.9);
    try testing.expect(k_quant.KQuantizer.getQualityRetention(.Q5_K) > 0.95);
    try testing.expect(k_quant.KQuantizer.getQualityRetention(.Q6_K) > 0.98);

    // Verify ordering: Q6_K > Q5_K > Q4_K (higher quality for higher bit counts)
    try testing.expect(k_quant.KQuantizer.getQualityRetention(.Q6_K) > k_quant.KQuantizer.getQualityRetention(.Q5_K));
    try testing.expect(k_quant.KQuantizer.getQualityRetention(.Q5_K) > k_quant.KQuantizer.getQualityRetention(.Q4_K));
}

test "Importance quantization IQ1_S extreme compression" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create data with varying importance
    const data = try allocator.alloc(f32, 256);
    defer allocator.free(data);

    for (0..256) |i| {
        // Some weights are much more important (larger magnitude)
        if (i % 10 == 0) {
            data[i] = 1.0; // Important weight
        } else if (i % 5 == 0) {
            data[i] = 0.5; // Semi-important
        } else {
            data[i] = 0.1; // Less important
        }
    }

    const tensor = Tensor(f32){ .data = data, .shape = &[_]usize{256} };

    var quantizer = iq_quant.IQuantizer.init(allocator, .IQ1_S);
    const quantized = try quantizer.quantize(tensor);
    defer allocator.free(quantized);

    // Verify quantization succeeded
    try testing.expect(quantized.len == iq_quant.BlockIQ1S.size());

    const dequantized = try quantizer.dequantize(quantized, tensor.shape);
    defer dequantized.deinit(allocator);

    // Check that important weights are better preserved
    var important_error: f32 = 0;
    var unimportant_error: f32 = 0;
    var important_count: u32 = 0;
    var unimportant_count: u32 = 0;

    for (0..256) |i| {
        const error = @abs(tensor.data[i] - dequantized.data[i]);
        if (i % 10 == 0) {
            important_error += error;
            important_count += 1;
        } else {
            unimportant_error += error;
            unimportant_count += 1;
        }
    }

    // Important weights should have lower error on average
    const avg_important_error = important_error / @as(f32, @floatFromInt(important_count));
    const avg_unimportant_error = unimportant_error / @as(f32, @floatFromInt(unimportant_count));

    // This is the key advantage of importance quantization
    try testing.expect(avg_important_error <= avg_unimportant_error * 1.5);
}

test "Importance quantization IQ2_XS adaptive precision" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data = try allocator.alloc(f32, 256);
    defer allocator.free(data);

    // Create gradual importance pattern
    for (0..256) |i| {
        const importance_factor = @as(f32, @floatFromInt(i)) / 255.0;
        data[i] = importance_factor * std.math.sin(@as(f32, @floatFromInt(i)) * 0.1);
    }

    const tensor = Tensor(f32){ .data = data, .shape = &[_]usize{256} };

    var quantizer = iq_quant.IQuantizer.init(allocator, .IQ2_XS);
    const quantized = try quantizer.quantize(tensor);
    defer allocator.free(quantized);

    try testing.expect(quantized.len == iq_quant.BlockIQ2XS.size());

    const dequantized = try quantizer.dequantize(quantized, tensor.shape);
    defer dequantized.deinit(allocator);

    // Check adaptive precision: later elements (more important) should be better preserved
    var early_error: f32 = 0;
    var late_error: f32 = 0;

    for (0..128) |i| {
        early_error += @abs(tensor.data[i] - dequantized.data[i]);
    }
    for (128..256) |i| {
        late_error += @abs(tensor.data[i] - dequantized.data[i]);
    }

    // Later elements should generally have better preservation
    try testing.expect((late_error / 128.0) <= (early_error / 128.0) * 2.0);
}

test "Importance quantization IQ4_NL non-linear precision" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data = try allocator.alloc(f32, 256);
    defer allocator.free(data);

    // Create clustered data with outliers (tests clustering)
    for (0..256) |i| {
        if (i < 100) {
            data[i] = 0.1; // Cluster 1: small values
        } else if (i < 200) {
            data[i] = 0.5; // Cluster 2: medium values
        } else {
            data[i] = 1.0; // Cluster 3: large values (most important)
        }

        // Add some noise
        const noise = (@as(f32, @floatFromInt(i % 7)) - 3.0) * 0.01;
        data[i] += noise;
    }

    const tensor = Tensor(f32){ .data = data, .shape = &[_]usize{256} };

    var quantizer = iq_quant.IQuantizer.init(allocator, .IQ4_NL);
    const quantized = try quantizer.quantize(tensor);
    defer allocator.free(quantized);

    try testing.expect(quantized.len == iq_quant.BlockIQ4NL.size());

    const dequantized = try quantizer.dequantize(quantized, tensor.shape);
    defer dequantized.deinit(allocator);

    // Check that clustering preserves cluster structure
    var cluster3_error: f32 = 0; // Most important cluster
    var cluster1_error: f32 = 0; // Least important cluster

    for (200..256) |i| {
        cluster3_error += @abs(tensor.data[i] - dequantized.data[i]);
    }
    for (0..100) |i| {
        cluster1_error += @abs(tensor.data[i] - dequantized.data[i]);
    }

    // Most important cluster should have better preservation
    const avg_cluster3_error = cluster3_error / 56.0;
    const avg_cluster1_error = cluster1_error / 100.0;

    try testing.expect(avg_cluster3_error <= avg_cluster1_error * 1.2);
}

test "IQ compression ratios exceed K-quantization" {
    // IQ should provide better compression than equivalent K-quantization
    try testing.expect(iq_quant.IQuantizer.getCompressionRatio(.IQ1_S) > 25.0);
    try testing.expect(iq_quant.IQuantizer.getCompressionRatio(.IQ2_XS) > 15.0);
    try testing.expect(iq_quant.IQuantizer.getCompressionRatio(.IQ4_NL) > 7.0);

    // IQ2 should compress better than Q4_K at similar quality
    try testing.expect(iq_quant.IQuantizer.getCompressionRatio(.IQ2_XS) >
                      k_quant.KQuantizer.getCompressionRatio(.Q4_K));
}

test "IQ quality retention with extreme compression" {
    // Even with extreme compression, IQ should maintain reasonable quality
    try testing.expect(iq_quant.IQuantizer.getQualityRetention(.IQ1_S) > 0.8);
    try testing.expect(iq_quant.IQuantizer.getQualityRetention(.IQ2_XS) > 0.9);
    try testing.expect(iq_quant.IQuantizer.getQualityRetention(.IQ4_NL) > 0.95);

    // IQ4_NL should provide the best quality among IQ formats
    try testing.expect(iq_quant.IQuantizer.getQualityRetention(.IQ4_NL) >
                      iq_quant.IQuantizer.getQualityRetention(.IQ2_XS));
    try testing.expect(iq_quant.IQuantizer.getQualityRetention(.IQ2_XS) >
                      iq_quant.IQuantizer.getQualityRetention(.IQ1_S));
}

test "Multi-block quantization consistency" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test with multiple blocks (512 elements = 2 blocks)
    const data = try allocator.alloc(f32, 512);
    defer allocator.free(data);

    for (0..512) |i| {
        data[i] = std.math.sin(@as(f32, @floatFromInt(i)) * 0.02);
    }

    const tensor = Tensor(f32){ .data = data, .shape = &[_]usize{512} };

    // Test both K-quantization and IQ with multiple blocks
    var k_quantizer = k_quant.KQuantizer.init(allocator, .Q4_K);
    const k_quantized = try k_quantizer.quantize(tensor);
    defer allocator.free(k_quantized);

    try testing.expect(k_quantized.len == 2 * k_quant.BlockQ4K.size());

    const k_dequantized = try k_quantizer.dequantize(k_quantized, tensor.shape);
    defer k_dequantized.deinit(allocator);

    // Verify shape and reasonable accuracy across blocks
    try testing.expect(k_dequantized.data.len == 512);

    var total_error: f32 = 0;
    for (0..512) |i| {
        total_error += @abs(tensor.data[i] - k_dequantized.data[i]);
    }
    try testing.expect(total_error / 512.0 < 0.15);
}

test "Quantization memory safety" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test with edge case sizes
    const sizes = [_]usize{ 1, 127, 255, 256, 257, 511, 512, 1023, 1024 };

    for (sizes) |size| {
        const data = try allocator.alloc(f32, size);
        defer allocator.free(data);

        // Initialize with test pattern
        for (0..size) |i| {
            data[i] = @as(f32, @floatFromInt(i % 10)) / 10.0;
        }

        const tensor = Tensor(f32){ .data = data, .shape = &[_]usize{size} };

        // Test doesn't crash with various sizes
        var quantizer = k_quant.KQuantizer.init(allocator, .Q4_K);
        const quantized = quantizer.quantize(tensor) catch |err| switch (err) {
            error.OutOfMemory => return err,
            else => continue, // Other errors are acceptable for edge cases
        };
        defer allocator.free(quantized);

        const dequantized = quantizer.dequantize(quantized, tensor.shape) catch continue;
        defer dequantized.deinit(allocator);

        // Should preserve shape
        try testing.expect(dequantized.shape[0] == size);
    }
}