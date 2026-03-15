// Linear Algebra Layer: Quantization Support
//
// Quantization is crucial for efficient LLM inference, reducing memory usage
// and computation while maintaining acceptable accuracy. This module provides
// educational implementations of common quantization schemes used in modern
// transformer models.
//
// ## Educational Objectives
// - Understand how quantization reduces model size and computation
// - Learn different quantization schemes and their trade-offs
// - Connect quantization to memory bandwidth and performance
// - Implement GGUF-compatible quantization formats
//
// ## Quantization in Transformers
// Modern LLMs use quantization to:
// - Reduce memory usage from 16GB (FP32) to 4GB (INT8) or 2GB (INT4)
// - Increase inference speed through integer arithmetic
// - Enable deployment on consumer hardware
// - Maintain acceptable accuracy through careful calibration

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const foundation = @import("../foundation/tensor.zig");
const Tensor = foundation.Tensor;
const TensorError = foundation.TensorError;

/// Quantization data types supported by ZigLlama
///
/// ## Educational Note: Quantization Hierarchy
/// Different quantization schemes trade off between:
/// - **Memory Usage**: Lower bits = less memory
/// - **Accuracy**: Higher bits = better precision
/// - **Computation Speed**: Integer ops often faster than float
/// - **Hardware Support**: Not all formats supported on all devices
pub const QuantType = enum {
    /// 32-bit floating point (no quantization)
    F32,

    /// 16-bit floating point (half precision)
    /// Memory: 50% of F32, Good accuracy, Hardware accelerated on modern GPUs
    F16,

    /// 8-bit integers with per-channel scaling
    /// Memory: 25% of F32, Good accuracy with calibration
    INT8,

    /// 4-bit integers with group scaling (GGUF Q4_0)
    /// Memory: 12.5% of F32, Acceptable accuracy for most uses
    Q4_0,

    /// 4-bit integers with improved accuracy (GGUF Q4_1)
    /// Memory: 12.5% of F32, Better accuracy than Q4_0
    Q4_1,

    /// 8-bit integers with better accuracy (GGUF Q8_0)
    /// Memory: 25% of F32, High accuracy
    Q8_0,
};

/// Quantization parameters for different schemes
pub const QuantParams = union(QuantType) {
    F32: void,
    F16: void,
    INT8: struct {
        scale: f32,
        zero_point: i32,
    },
    Q4_0: struct {
        scale: f16,
    },
    Q4_1: struct {
        scale: f16,
        min: f16,
    },
    Q8_0: struct {
        scale: f32,
    },
};

/// Quantized tensor that stores data in compressed format
///
/// ## Educational Note: Memory Layout
/// Quantized tensors use different memory layouts:
/// - **Block-based**: Groups of elements share scale factors
/// - **Per-channel**: Different scales for each output channel
/// - **Packed**: Multiple values per byte (e.g., two 4-bit values per byte)
pub fn QuantizedTensor(comptime quant_type: QuantType) type {
    return struct {
        const Self = @This();

        /// Original tensor shape
        shape: []usize,

        /// Quantized data in compressed format
        data: []u8,

        /// Quantization parameters
        params: []QuantParams,

        /// Total number of elements
        size: usize,

        /// Memory allocator
        allocator: Allocator,

        /// Block size for group quantization
        block_size: usize,

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.allocator.free(self.shape);
            self.allocator.free(self.params);
        }

        /// Get the memory usage reduction compared to F32
        pub fn compressionRatio(self: Self) f32 {
            const original_bytes = self.size * @sizeOf(f32);
            const compressed_bytes = self.data.len;
            return @as(f32, @floatFromInt(original_bytes)) / @as(f32, @floatFromInt(compressed_bytes));
        }
    };
}

/// Convert F32 tensor to quantized format
///
/// ## Educational Note: Quantization Process
/// Quantization involves several steps:
/// 1. **Range Analysis**: Find min/max values in data
/// 2. **Scale Calculation**: Determine optimal scale factors
/// 3. **Conversion**: Map float values to integer range
/// 4. **Packing**: Store multiple values per byte when possible
///
/// ## Mathematical Foundation
/// For INT8 quantization:
/// ```
/// scale = (max - min) / 255
/// quantized = round((value - min) / scale)
/// dequantized = quantized * scale + min
/// ```
pub fn quantizeTensor(
    comptime target_type: QuantType,
    tensor: Tensor(f32),
    allocator: Allocator
) !QuantizedTensor(target_type) {
    switch (target_type) {
        .Q4_0 => return quantizeQ4_0(tensor, allocator),
        .Q8_0 => return quantizeQ8_0(tensor, allocator),
        .INT8 => return quantizeINT8(tensor, allocator),
        else => @panic("Quantization type not yet implemented"),
    }
}

/// Quantize tensor to Q4_0 format (GGUF compatible)
///
/// ## Educational Note: Q4_0 Format
/// Q4_0 is a popular 4-bit quantization format:
/// - **Block Size**: 32 elements per block
/// - **Storage**: 4 bits per weight + 16-bit scale per block
/// - **Memory**: ~4.5 bits per weight (including scale overhead)
/// - **Accuracy**: Good for most transformer weights
///
/// ## Block Structure
/// ```
/// Block = {
///   scale: f16,           // 2 bytes
///   weights: [16]u8       // 32 weights, 2 per byte
/// }
/// ```
fn quantizeQ4_0(tensor: Tensor(f32), allocator: Allocator) !QuantizedTensor(.Q4_0) {
    const block_size = 32;
    const num_blocks = (tensor.size + block_size - 1) / block_size;

    // Each block: 2 bytes (scale) + 16 bytes (32 weights, 2 per byte)
    const bytes_per_block = 2 + 16;
    const data_size = num_blocks * bytes_per_block;

    const data = try allocator.alloc(u8, data_size);
    errdefer allocator.free(data);

    const shape = try allocator.dupe(usize, tensor.shape);
    errdefer allocator.free(shape);

    const params = try allocator.alloc(QuantParams, num_blocks);
    errdefer allocator.free(params);

    // Process each block
    for (0..num_blocks) |block_idx| {
        const start_elem = block_idx * block_size;
        const end_elem = @min(start_elem + block_size, tensor.size);
        const actual_block_size = end_elem - start_elem;

        // Find range in this block
        var min_val: f32 = std.math.inf(f32);
        var max_val: f32 = -std.math.inf(f32);

        for (start_elem..end_elem) |i| {
            const val = tensor.data[i];
            min_val = @min(min_val, val);
            max_val = @max(max_val, val);
        }

        // Calculate scale (4-bit signed: -8 to 7)
        const scale = (max_val - min_val) / 15.0;
        const scale_f16: f16 = @floatCast(scale);

        params[block_idx] = QuantParams{ .Q4_0 = .{ .scale = scale_f16 } };

        // Store scale in data
        const block_start = block_idx * bytes_per_block;
        const scale_bytes = std.mem.asBytes(&scale_f16);
        @memcpy(data[block_start..block_start + 2], scale_bytes);

        // Quantize and pack weights
        for (0..actual_block_size / 2) |pair_idx| {
            const elem1_idx = start_elem + pair_idx * 2;
            const elem2_idx = @min(elem1_idx + 1, end_elem - 1);

            // Quantize to 4-bit signed (-8 to 7)
            const val1 = tensor.data[elem1_idx];
            const val2 = tensor.data[elem2_idx];

            const q1 = std.math.clamp(@as(i8, @intFromFloat((val1 - min_val) / scale - 8)), -8, 7);
            const q2 = std.math.clamp(@as(i8, @intFromFloat((val2 - min_val) / scale - 8)), -8, 7);

            // Pack two 4-bit values into one byte
            const packed = @as(u8, @bitCast(@as(i8, q1 & 0xF))) |
                          (@as(u8, @bitCast(@as(i8, q2 & 0xF))) << 4);

            data[block_start + 2 + pair_idx] = packed;
        }
    }

    return QuantizedTensor(.Q4_0){
        .shape = shape,
        .data = data,
        .params = params,
        .size = tensor.size,
        .allocator = allocator,
        .block_size = block_size,
    };
}

/// Quantize tensor to Q8_0 format (GGUF compatible)
///
/// ## Educational Note: Q8_0 Format
/// Q8_0 provides higher accuracy than Q4_0:
/// - **Block Size**: 32 elements per block
/// - **Storage**: 8 bits per weight + 32-bit scale per block
/// - **Memory**: ~8.25 bits per weight
/// - **Accuracy**: Very close to FP32
fn quantizeQ8_0(tensor: Tensor(f32), allocator: Allocator) !QuantizedTensor(.Q8_0) {
    const block_size = 32;
    const num_blocks = (tensor.size + block_size - 1) / block_size;

    // Each block: 4 bytes (scale) + 32 bytes (weights)
    const bytes_per_block = 4 + 32;
    const data_size = num_blocks * bytes_per_block;

    const data = try allocator.alloc(u8, data_size);
    errdefer allocator.free(data);

    const shape = try allocator.dupe(usize, tensor.shape);
    errdefer allocator.free(shape);

    const params = try allocator.alloc(QuantParams, num_blocks);
    errdefer allocator.free(params);

    // Process each block
    for (0..num_blocks) |block_idx| {
        const start_elem = block_idx * block_size;
        const end_elem = @min(start_elem + block_size, tensor.size);

        // Find absolute maximum in this block
        var abs_max: f32 = 0.0;
        for (start_elem..end_elem) |i| {
            abs_max = @max(abs_max, @abs(tensor.data[i]));
        }

        // Calculate scale for signed 8-bit (-127 to 127)
        const scale = abs_max / 127.0;
        params[block_idx] = QuantParams{ .Q8_0 = .{ .scale = scale } };

        // Store scale in data
        const block_start = block_idx * bytes_per_block;
        const scale_bytes = std.mem.asBytes(&scale);
        @memcpy(data[block_start..block_start + 4], scale_bytes);

        // Quantize weights
        for (start_elem..end_elem) |i| {
            const elem_in_block = i - start_elem;
            const val = tensor.data[i];
            const quantized = std.math.clamp(@as(i8, @intFromFloat(val / scale)), -127, 127);
            data[block_start + 4 + elem_in_block] = @bitCast(quantized);
        }

        // Pad remaining elements with zeros
        for ((end_elem - start_elem)..block_size) |elem_in_block| {
            data[block_start + 4 + elem_in_block] = 0;
        }
    }

    return QuantizedTensor(.Q8_0){
        .shape = shape,
        .data = data,
        .params = params,
        .size = tensor.size,
        .allocator = allocator,
        .block_size = block_size,
    };
}

/// Simple INT8 quantization with global scale
fn quantizeINT8(tensor: Tensor(f32), allocator: Allocator) !QuantizedTensor(.INT8) {
    // Find global range
    var min_val: f32 = std.math.inf(f32);
    var max_val: f32 = -std.math.inf(f32);

    for (tensor.data) |val| {
        min_val = @min(min_val, val);
        max_val = @max(max_val, val);
    }

    // Calculate scale and zero point
    const scale = (max_val - min_val) / 255.0;
    const zero_point: i32 = @intFromFloat(-min_val / scale);

    const data = try allocator.alloc(u8, tensor.size);
    errdefer allocator.free(data);

    const shape = try allocator.dupe(usize, tensor.shape);
    errdefer allocator.free(shape);

    const params = try allocator.alloc(QuantParams, 1);
    errdefer allocator.free(params);

    params[0] = QuantParams{ .INT8 = .{ .scale = scale, .zero_point = zero_point } };

    // Quantize all elements
    for (tensor.data, 0..) |val, i| {
        const quantized = std.math.clamp(
            @as(i32, @intFromFloat(val / scale)) + zero_point,
            0, 255
        );
        data[i] = @intCast(quantized);
    }

    return QuantizedTensor(.INT8){
        .shape = shape,
        .data = data,
        .params = params,
        .size = tensor.size,
        .allocator = allocator,
        .block_size = tensor.size, // Global quantization
    };
}

/// Dequantize tensor back to F32 format
///
/// ## Educational Note: Dequantization
/// Dequantization reverses the quantization process:
/// 1. **Unpack**: Extract quantized values from compressed format
/// 2. **Scale**: Apply scale factors to restore original range
/// 3. **Convert**: Transform integers back to floating-point
///
/// This process introduces quantization error, but modern techniques
/// minimize the impact through careful calibration.
pub fn dequantizeTensor(
    comptime source_type: QuantType,
    quantized: QuantizedTensor(source_type),
    allocator: Allocator
) !Tensor(f32) {
    var result = try Tensor(f32).init(allocator, quantized.shape);

    switch (source_type) {
        .Q4_0 => try dequantizeQ4_0(quantized, &result),
        .Q8_0 => try dequantizeQ8_0(quantized, &result),
        .INT8 => try dequantizeINT8(quantized, &result),
        else => @panic("Dequantization type not yet implemented"),
    }

    return result;
}

fn dequantizeQ4_0(quantized: QuantizedTensor(.Q4_0), result: *Tensor(f32)) !void {
    const block_size = quantized.block_size;
    const bytes_per_block = 2 + 16;

    for (0..quantized.params.len) |block_idx| {
        const scale = quantized.params[block_idx].Q4_0.scale;
        const block_start = block_idx * bytes_per_block;
        const elem_start = block_idx * block_size;
        const elem_end = @min(elem_start + block_size, quantized.size);

        // Dequantize weights in this block
        for (0..(elem_end - elem_start) / 2) |pair_idx| {
            const packed = quantized.data[block_start + 2 + pair_idx];

            // Unpack two 4-bit values
            const q1 = @as(i8, @bitCast(packed & 0xF));
            const q2 = @as(i8, @bitCast((packed >> 4) & 0xF));

            // Dequantize
            const val1 = (@as(f32, @floatFromInt(q1)) + 8.0) * scale;
            const val2 = (@as(f32, @floatFromInt(q2)) + 8.0) * scale;

            result.data[elem_start + pair_idx * 2] = val1;
            if (elem_start + pair_idx * 2 + 1 < elem_end) {
                result.data[elem_start + pair_idx * 2 + 1] = val2;
            }
        }
    }
}

fn dequantizeQ8_0(quantized: QuantizedTensor(.Q8_0), result: *Tensor(f32)) !void {
    const block_size = quantized.block_size;
    const bytes_per_block = 4 + 32;

    for (0..quantized.params.len) |block_idx| {
        const scale = quantized.params[block_idx].Q8_0.scale;
        const block_start = block_idx * bytes_per_block;
        const elem_start = block_idx * block_size;
        const elem_end = @min(elem_start + block_size, quantized.size);

        // Dequantize weights in this block
        for (elem_start..elem_end) |i| {
            const elem_in_block = i - elem_start;
            const quantized_val = @as(i8, @bitCast(quantized.data[block_start + 4 + elem_in_block]));
            result.data[i] = @as(f32, @floatFromInt(quantized_val)) * scale;
        }
    }
}

fn dequantizeINT8(quantized: QuantizedTensor(.INT8), result: *Tensor(f32)) !void {
    const scale = quantized.params[0].INT8.scale;
    const zero_point = quantized.params[0].INT8.zero_point;

    for (0..quantized.size) |i| {
        const quantized_val = @as(i32, quantized.data[i]) - zero_point;
        result.data[i] = @as(f32, @floatFromInt(quantized_val)) * scale;
    }
}

// ============================================================================
// COMPREHENSIVE TESTS
// ============================================================================

test "Q4_0 quantization round-trip" {
    const allocator = testing.allocator;

    // Create test tensor with known values
    var original = try Tensor(f32).init(allocator, &[_]usize{64});
    defer original.deinit();

    // Fill with test pattern
    for (0..64) |i| {
        original.data[i] = @as(f32, @floatFromInt(i)) / 32.0 - 1.0; // Range: -1 to 1
    }

    // Quantize
    var quantized = try quantizeTensor(.Q4_0, original, allocator);
    defer quantized.deinit();

    // Check compression ratio
    const ratio = quantized.compressionRatio();
    try testing.expect(ratio > 6.0); // Should be ~7-8x compression

    // Dequantize
    var restored = try dequantizeTensor(.Q4_0, quantized, allocator);
    defer restored.deinit();

    // Check that shapes match
    try testing.expectEqualSlices(usize, original.shape, restored.shape);

    // Check reconstruction quality (allow some quantization error)
    var max_error: f32 = 0.0;
    for (0..original.size) |i| {
        const error = @abs(original.data[i] - restored.data[i]);
        max_error = @max(max_error, error);
    }

    // Quantization error should be reasonable for 4-bit
    try testing.expect(max_error < 0.2);
}

test "Q8_0 quantization accuracy" {
    const allocator = testing.allocator;

    // Create test tensor
    var original = try Tensor(f32).init(allocator, &[_]usize{128});
    defer original.deinit();

    // Fill with diverse values
    for (0..128) |i| {
        original.data[i] = std.math.sin(@as(f32, @floatFromInt(i)) * 0.1);
    }

    // Quantize and dequantize
    var quantized = try quantizeTensor(.Q8_0, original, allocator);
    defer quantized.deinit();

    var restored = try dequantizeTensor(.Q8_0, quantized, allocator);
    defer restored.deinit();

    // Q8_0 should have much better accuracy than Q4_0
    var max_error: f32 = 0.0;
    for (0..original.size) |i| {
        const error = @abs(original.data[i] - restored.data[i]);
        max_error = @max(max_error, error);
    }

    // 8-bit quantization should be very accurate
    try testing.expect(max_error < 0.01);

    // Check compression ratio
    const ratio = quantized.compressionRatio();
    try testing.expect(ratio > 3.5 and ratio < 4.5); // Should be ~4x compression
}