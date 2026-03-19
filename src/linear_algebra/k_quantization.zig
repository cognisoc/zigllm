const std = @import("std");
const Allocator = std.mem.Allocator;
const foundation = @import("../foundation/tensor.zig");
const Tensor = foundation.Tensor;

/// K-quantization formats for improved compression with minimal quality loss
/// Based on llama.cpp's k-quants implementation for better performance
pub const KQuantType = enum {
    Q4_K,  // 4-bit K-quantization
    Q5_K,  // 5-bit K-quantization
    Q6_K,  // 6-bit K-quantization
};

/// Block size for K-quantization - matches llama.cpp standard
pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

/// K-quantization block structures following llama.cpp format
pub const BlockQ4K = struct {
    d: f16,                           // Delta (scale factor)
    dmin: f16,                        // Minimum delta for better precision
    scales: [K_SCALE_SIZE]u8,         // Scale values for sub-blocks
    qs: [QK_K / 2]u8,                 // Quantized values (4-bit packed)

    const Self = @This();

    pub fn size() usize {
        return @sizeOf(Self);
    }
};

pub const BlockQ5K = struct {
    d: f16,                           // Delta (scale factor)
    dmin: f16,                        // Minimum delta
    scales: [K_SCALE_SIZE]u8,         // Scale values
    qh: [QK_K / 8]u8,                 // High bits for 5-bit quantization
    qs: [QK_K / 2]u8,                 // Low 4 bits of quantized values

    const Self = @This();

    pub fn size() usize {
        return @sizeOf(Self);
    }
};

pub const BlockQ6K = struct {
    ql: [QK_K / 2]u8,                 // Lower 4 bits of 6-bit values
    qh: [QK_K / 4]u8,                 // Upper 2 bits of 6-bit values
    scales: [QK_K / 16]i8,            // Scale factors (signed)
    d: f16,                           // Global scale factor

    const Self = @This();

    pub fn size() usize {
        return @sizeOf(Self);
    }
};

/// K-quantization implementation with educational clarity
pub const KQuantizer = struct {
    allocator: Allocator,
    quant_type: KQuantType,

    const Self = @This();

    pub fn init(allocator: Allocator, quant_type: KQuantType) Self {
        return Self{
            .allocator = allocator,
            .quant_type = quant_type,
        };
    }

    /// Quantize tensor using K-quantization with improved precision
    pub fn quantize(self: Self, tensor: Tensor(f32)) ![]u8 {
        switch (self.quant_type) {
            .Q4_K => return try self.quantizeQ4K(tensor),
            .Q5_K => return try self.quantizeQ5K(tensor),
            .Q6_K => return try self.quantizeQ6K(tensor),
        }
    }

    /// Dequantize K-quantized data back to f32 tensor
    pub fn dequantize(self: Self, data: []const u8, shape: []const usize) !Tensor(f32) {
        switch (self.quant_type) {
            .Q4_K => return try self.dequantizeQ4K(data, shape),
            .Q5_K => return try self.dequantizeQ5K(data, shape),
            .Q6_K => return try self.dequantizeQ6K(data, shape),
        }
    }

    /// Q4_K quantization - 4.5 bits per weight with sub-block scaling
    fn quantizeQ4K(self: Self, tensor: Tensor(f32)) ![]u8 {
        const num_blocks = (tensor.data.len + QK_K - 1) / QK_K;
        const output_size = num_blocks * BlockQ4K.size();
        const result = try self.allocator.alloc(u8, output_size);

        var block_idx: usize = 0;
        var data_idx: usize = 0;

        while (block_idx < num_blocks) : (block_idx += 1) {
            const block_ptr = @as(*BlockQ4K, @ptrCast(@alignCast(result.ptr + block_idx * BlockQ4K.size())));
            const remaining = @min(QK_K, tensor.data.len - data_idx);

            // Find min/max for this block
            var min_val: f32 = std.math.inf(f32);
            var max_val: f32 = -std.math.inf(f32);

            for (0..remaining) |i| {
                const val = tensor.data[data_idx + i];
                min_val = @min(min_val, val);
                max_val = @max(max_val, val);
            }

            // Calculate scale factors with improved precision
            const scale = (max_val - min_val) / 15.0; // 4-bit range: 0-15
            const inv_scale = if (scale > 0) 1.0 / scale else 0.0;

            block_ptr.d = @as(f16, @floatCast(scale));
            block_ptr.dmin = @as(f16, @floatCast(min_val));

            // Quantize sub-blocks with local scaling
            for (0..K_SCALE_SIZE) |scale_idx| {
                const sub_start = (scale_idx * QK_K) / K_SCALE_SIZE;
                const sub_end = @min(((scale_idx + 1) * QK_K) / K_SCALE_SIZE, remaining);

                // Find local min/max for sub-block
                var local_min: f32 = std.math.inf(f32);
                var local_max: f32 = -std.math.inf(f32);

                for (sub_start..sub_end) |i| {
                    if (data_idx + i < tensor.data.len) {
                        const val = tensor.data[data_idx + i];
                        local_min = @min(local_min, val);
                        local_max = @max(local_max, val);
                    }
                }

                // Local scale factor
                const local_scale = (local_max - local_min) / 15.0;
                const quantized_scale = @min(255, @as(u32, @intFromFloat(@max(0, local_scale * 255.0))));
                block_ptr.scales[scale_idx] = @as(u8, @intCast(quantized_scale));
            }

            // Quantize values with 4-bit precision
            for (0..remaining) |i| {
                if (i % 2 == 0) {
                    // Lower 4 bits
                    const normalized = (tensor.data[data_idx + i] - min_val) * inv_scale;
                    const quantized = @min(15, @as(u8, @intFromFloat(@max(0, normalized + 0.5))));
                    block_ptr.qs[i / 2] = quantized;
                } else {
                    // Upper 4 bits
                    const normalized = (tensor.data[data_idx + i] - min_val) * inv_scale;
                    const quantized = @min(15, @as(u8, @intFromFloat(@max(0, normalized + 0.5))));
                    block_ptr.qs[i / 2] |= quantized << 4;
                }
            }

            data_idx += remaining;
        }

        return result;
    }

    /// Q5_K quantization - 5.5 bits per weight with high bit separation
    fn quantizeQ5K(self: Self, tensor: Tensor(f32)) ![]u8 {
        const num_blocks = (tensor.data.len + QK_K - 1) / QK_K;
        const output_size = num_blocks * BlockQ5K.size();
        const result = try self.allocator.alloc(u8, output_size);

        var block_idx: usize = 0;
        var data_idx: usize = 0;

        while (block_idx < num_blocks) : (block_idx += 1) {
            const block_ptr = @as(*BlockQ5K, @ptrCast(@alignCast(result.ptr + block_idx * BlockQ5K.size())));
            const remaining = @min(QK_K, tensor.data.len - data_idx);

            // Find min/max for this block
            var min_val: f32 = std.math.inf(f32);
            var max_val: f32 = -std.math.inf(f32);

            for (0..remaining) |i| {
                const val = tensor.data[data_idx + i];
                min_val = @min(min_val, val);
                max_val = @max(max_val, val);
            }

            // 5-bit quantization: 0-31 range
            const scale = (max_val - min_val) / 31.0;
            const inv_scale = if (scale > 0) 1.0 / scale else 0.0;

            block_ptr.d = @as(f16, @floatCast(scale));
            block_ptr.dmin = @as(f16, @floatCast(min_val));

            // Initialize high bit array
            @memset(block_ptr.qh[0..], 0);

            // Calculate sub-block scales
            for (0..K_SCALE_SIZE) |scale_idx| {
                const sub_start = (scale_idx * QK_K) / K_SCALE_SIZE;
                const sub_end = @min(((scale_idx + 1) * QK_K) / K_SCALE_SIZE, remaining);

                var local_min: f32 = std.math.inf(f32);
                var local_max: f32 = -std.math.inf(f32);

                for (sub_start..sub_end) |i| {
                    if (data_idx + i < tensor.data.len) {
                        const val = tensor.data[data_idx + i];
                        local_min = @min(local_min, val);
                        local_max = @max(local_max, val);
                    }
                }

                const local_scale = (local_max - local_min) / 31.0;
                const quantized_scale = @min(255, @as(u32, @intFromFloat(@max(0, local_scale * 255.0))));
                block_ptr.scales[scale_idx] = @as(u8, @intCast(quantized_scale));
            }

            // Quantize with 5-bit precision
            for (0..remaining) |i| {
                const normalized = (tensor.data[data_idx + i] - min_val) * inv_scale;
                const quantized = @min(31, @as(u8, @intFromFloat(@max(0, normalized + 0.5))));

                // Store lower 4 bits
                if (i % 2 == 0) {
                    block_ptr.qs[i / 2] = quantized & 0x0F;
                } else {
                    block_ptr.qs[i / 2] |= (quantized & 0x0F) << 4;
                }

                // Store high bit separately
                const high_bit = (quantized >> 4) & 1;
                const bit_idx = i;
                const byte_idx = bit_idx / 8;
                const bit_pos = bit_idx % 8;

                if (byte_idx < block_ptr.qh.len) {
                    block_ptr.qh[byte_idx] |= @as(u8, @intCast(high_bit)) << @as(u3, @intCast(bit_pos));
                }
            }

            data_idx += remaining;
        }

        return result;
    }

    /// Q6_K quantization - 6.5 bits per weight for higher precision
    fn quantizeQ6K(self: Self, tensor: Tensor(f32)) ![]u8 {
        const num_blocks = (tensor.data.len + QK_K - 1) / QK_K;
        const output_size = num_blocks * BlockQ6K.size();
        const result = try self.allocator.alloc(u8, output_size);

        var block_idx: usize = 0;
        var data_idx: usize = 0;

        while (block_idx < num_blocks) : (block_idx += 1) {
            const block_ptr = @as(*BlockQ6K, @ptrCast(@alignCast(result.ptr + block_idx * BlockQ6K.size())));
            const remaining = @min(QK_K, tensor.data.len - data_idx);

            // Find min/max for global scaling
            var min_val: f32 = std.math.inf(f32);
            var max_val: f32 = -std.math.inf(f32);

            for (0..remaining) |i| {
                const val = tensor.data[data_idx + i];
                min_val = @min(min_val, val);
                max_val = @max(max_val, val);
            }

            // 6-bit quantization: 0-63 range
            const scale = (max_val - min_val) / 63.0;
            const inv_scale = if (scale > 0) 1.0 / scale else 0.0;

            block_ptr.d = @as(f16, @floatCast(scale));

            // Calculate sub-block scales (16 elements per scale)
            for (0..QK_K / 16) |scale_idx| {
                const sub_start = scale_idx * 16;
                const sub_end = @min(sub_start + 16, remaining);

                var local_sum: f32 = 0;
                for (sub_start..sub_end) |i| {
                    if (data_idx + i < tensor.data.len) {
                        local_sum += @abs(tensor.data[data_idx + i]);
                    }
                }

                const avg_magnitude = local_sum / 16.0;
                const scale_factor = avg_magnitude * inv_scale;
                const quantized_scale = @max(-128, @min(127, @as(i8, @intFromFloat(scale_factor * 127.0))));
                block_ptr.scales[scale_idx] = quantized_scale;
            }

            // Initialize arrays
            @memset(block_ptr.ql[0..], 0);
            @memset(block_ptr.qh[0..], 0);

            // Quantize with 6-bit precision
            for (0..remaining) |i| {
                const normalized = (tensor.data[data_idx + i] - min_val) * inv_scale;
                const quantized = @min(63, @as(u8, @intFromFloat(@max(0, normalized + 0.5))));

                // Store lower 4 bits
                if (i % 2 == 0) {
                    block_ptr.ql[i / 2] = quantized & 0x0F;
                } else {
                    block_ptr.ql[i / 2] |= (quantized & 0x0F) << 4;
                }

                // Store upper 2 bits (packed 4 per byte)
                const upper_bits = (quantized >> 4) & 0x03;
                const packed_idx = i / 4;
                const bit_offset = (i % 4) * 2;

                if (packed_idx < block_ptr.qh.len) {
                    block_ptr.qh[packed_idx] |= upper_bits << @as(u3, @intCast(bit_offset));
                }
            }

            data_idx += remaining;
        }

        return result;
    }

    /// Dequantize Q4_K format back to f32
    fn dequantizeQ4K(self: Self, data: []const u8, shape: []const usize) !Tensor(f32) {
        const total_elements = blk: {
            var total: usize = 1;
            for (shape) |dim| total *= dim;
            break :blk total;
        };

        const result_data = try self.allocator.alloc(f32, total_elements);
        const result = Tensor(f32){ .data = result_data, .shape = shape };

        const num_blocks = data.len / BlockQ4K.size();
        var output_idx: usize = 0;

        for (0..num_blocks) |block_idx| {
            const block_ptr = @as(*const BlockQ4K, @ptrCast(@alignCast(data.ptr + block_idx * BlockQ4K.size())));
            const scale = @as(f32, @floatCast(block_ptr.d));
            const min_val = @as(f32, @floatCast(block_ptr.dmin));

            // Dequantize each value in the block
            for (0..QK_K) |i| {
                if (output_idx >= total_elements) break;

                // Extract quantized value
                const quantized = if (i % 2 == 0)
                    block_ptr.qs[i / 2] & 0x0F
                else
                    (block_ptr.qs[i / 2] >> 4) & 0x0F;

                // Get local scale factor
                const scale_idx = (i * K_SCALE_SIZE) / QK_K;
                const local_scale = @as(f32, @floatFromInt(block_ptr.scales[scale_idx])) / 255.0;

                // Dequantize to f32
                const dequantized = min_val + (@as(f32, @floatFromInt(quantized)) * scale * local_scale);
                result_data[output_idx] = dequantized;
                output_idx += 1;
            }
        }

        return result;
    }

    /// Dequantize Q5_K format back to f32
    fn dequantizeQ5K(self: Self, data: []const u8, shape: []const usize) !Tensor(f32) {
        const total_elements = blk: {
            var total: usize = 1;
            for (shape) |dim| total *= dim;
            break :blk total;
        };

        const result_data = try self.allocator.alloc(f32, total_elements);
        const result = Tensor(f32){ .data = result_data, .shape = shape };

        const num_blocks = data.len / BlockQ5K.size();
        var output_idx: usize = 0;

        for (0..num_blocks) |block_idx| {
            const block_ptr = @as(*const BlockQ5K, @ptrCast(@alignCast(data.ptr + block_idx * BlockQ5K.size())));
            const scale = @as(f32, @floatCast(block_ptr.d));
            const min_val = @as(f32, @floatCast(block_ptr.dmin));

            for (0..QK_K) |i| {
                if (output_idx >= total_elements) break;

                // Extract lower 4 bits
                const low_bits = if (i % 2 == 0)
                    block_ptr.qs[i / 2] & 0x0F
                else
                    (block_ptr.qs[i / 2] >> 4) & 0x0F;

                // Extract high bit
                const byte_idx = i / 8;
                const bit_pos = i % 8;
                const high_bit = if (byte_idx < block_ptr.qh.len)
                    (block_ptr.qh[byte_idx] >> @as(u3, @intCast(bit_pos))) & 1
                else
                    0;

                // Reconstruct 5-bit value
                const quantized = low_bits | (high_bit << 4);

                // Get local scale
                const scale_idx = (i * K_SCALE_SIZE) / QK_K;
                const local_scale = @as(f32, @floatFromInt(block_ptr.scales[scale_idx])) / 255.0;

                // Dequantize
                const dequantized = min_val + (@as(f32, @floatFromInt(quantized)) * scale * local_scale);
                result_data[output_idx] = dequantized;
                output_idx += 1;
            }
        }

        return result;
    }

    /// Dequantize Q6_K format back to f32
    fn dequantizeQ6K(self: Self, data: []const u8, shape: []const usize) !Tensor(f32) {
        const total_elements = blk: {
            var total: usize = 1;
            for (shape) |dim| total *= dim;
            break :blk total;
        };

        const result_data = try self.allocator.alloc(f32, total_elements);
        const result = Tensor(f32){ .data = result_data, .shape = shape };

        const num_blocks = data.len / BlockQ6K.size();
        var output_idx: usize = 0;

        for (0..num_blocks) |block_idx| {
            const block_ptr = @as(*const BlockQ6K, @ptrCast(@alignCast(data.ptr + block_idx * BlockQ6K.size())));
            const scale = @as(f32, @floatCast(block_ptr.d));

            for (0..QK_K) |i| {
                if (output_idx >= total_elements) break;

                // Extract lower 4 bits
                const low_bits = if (i % 2 == 0)
                    block_ptr.ql[i / 2] & 0x0F
                else
                    (block_ptr.ql[i / 2] >> 4) & 0x0F;

                // Extract upper 2 bits
                const packed_idx = i / 4;
                const bit_offset = (i % 4) * 2;
                const upper_bits = if (packed_idx < block_ptr.qh.len)
                    (block_ptr.qh[packed_idx] >> @as(u3, @intCast(bit_offset))) & 0x03
                else
                    0;

                // Reconstruct 6-bit value
                const quantized = low_bits | (upper_bits << 4);

                // Get local scale (16 elements per scale)
                const scale_idx = i / 16;
                const local_scale = if (scale_idx < block_ptr.scales.len)
                    @as(f32, @floatFromInt(block_ptr.scales[scale_idx])) / 127.0
                else
                    1.0;

                // Dequantize
                const dequantized = @as(f32, @floatFromInt(quantized)) * scale * local_scale;
                result_data[output_idx] = dequantized;
                output_idx += 1;
            }
        }

        return result;
    }

    /// Get compression ratio for the quantization type
    pub fn getCompressionRatio(quant_type: KQuantType) f32 {
        switch (quant_type) {
            .Q4_K => return 32.0 / 4.5,  // ~7.1x compression
            .Q5_K => return 32.0 / 5.5,  // ~5.8x compression
            .Q6_K => return 32.0 / 6.5,  // ~4.9x compression
        }
    }

    /// Get quality retention estimate
    pub fn getQualityRetention(quant_type: KQuantType) f32 {
        switch (quant_type) {
            .Q4_K => return 0.95,  // 95% quality retention
            .Q5_K => return 0.97,  // 97% quality retention
            .Q6_K => return 0.99,  // 99% quality retention
        }
    }
};