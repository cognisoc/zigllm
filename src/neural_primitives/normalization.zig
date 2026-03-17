//! Neural Primitives: Normalization Layers
//!
//! This module implements normalization techniques essential for training
//! stable and effective transformer models.
//!
//! ## Educational Objectives
//! - Understand why normalization is crucial for deep networks
//! - Learn the mathematical differences between normalization techniques
//! - Connect normalization choices to transformer performance
//! - Implement numerically stable normalization algorithms
//!
//! ## Transformer Context
//! Normalization is critical in transformers:
//! - **Layer Normalization**: Stabilizes training in deep networks
//! - **RMS Normalization**: Simpler, faster alternative used in modern models
//! - **Pre/Post-norm**: Placement affects gradient flow and stability
//! - **Numerical Stability**: Prevents overflow/underflow in deep networks

const std = @import("std");
const math = std.math;
const testing = std.testing;
const Allocator = std.mem.Allocator;
const Tensor = @import("../foundation/tensor.zig").Tensor;
const TensorError = @import("../foundation/tensor.zig").TensorError;

/// Normalization types used in transformer architectures
pub const NormalizationType = enum {
    LayerNorm,    // Standard Layer Normalization
    RMSNorm,      // Root Mean Square Normalization
    BatchNorm,    // Batch Normalization (less common in transformers)
    GroupNorm,    // Group Normalization
};

/// Numerical stability epsilon for normalization operations
const NORM_EPSILON: f32 = 1e-6;

/// Layer Normalization
///
/// ## Mathematical Definition
/// For input x with mean μ and variance σ²:
/// ```
/// LayerNorm(x) = γ * ((x - μ) / √(σ² + ε)) + β
/// ```
/// Where:
/// - μ = mean(x) = (1/d) * Σᵢ xᵢ
/// - σ² = var(x) = (1/d) * Σᵢ (xᵢ - μ)²
/// - γ (scale) and β (shift) are learnable parameters
/// - ε is a small constant for numerical stability
///
/// ## Educational Note: Why Layer Normalization?
/// Layer normalization addresses several issues in deep networks:
///
/// 1. **Internal Covariate Shift**: Stabilizes the distribution of inputs
/// 2. **Gradient Flow**: Improves gradient propagation in deep networks
/// 3. **Training Speed**: Enables higher learning rates and faster convergence
/// 4. **Sequence Independence**: Unlike BatchNorm, works with variable-length sequences
///
/// ## Transformer Usage
/// Layer normalization is applied:
/// - **Pre-norm**: Before attention and FFN layers (modern approach)
/// - **Post-norm**: After attention and FFN layers (original Transformer)
/// - **Embedding**: Sometimes applied to input embeddings
///
/// Pre-norm generally provides better gradient flow and training stability.
pub fn layerNorm(comptime T: type, input: Tensor(T), scale: ?Tensor(T), shift: ?Tensor(T), allocator: Allocator) TensorError!Tensor(T) {
    if (T != f32 and T != f64) return TensorError.InvalidShape;
    if (input.ndim() < 1) return TensorError.IncompatibleShapes;

    const last_dim = input.shape[input.shape.len - 1];
    const batch_size = input.size / last_dim;

    var result = try Tensor(T).init(allocator, input.shape);

    // Process each sequence/batch element independently
    for (0..batch_size) |batch_idx| {
        const batch_offset = batch_idx * last_dim;

        // Step 1: Calculate mean
        var sum: T = 0.0;
        for (0..last_dim) |i| {
            sum += input.data[batch_offset + i];
        }
        const mean = sum / @as(T, @floatFromInt(last_dim));

        // Step 2: Calculate variance
        var variance_sum: T = 0.0;
        for (0..last_dim) |i| {
            const diff = input.data[batch_offset + i] - mean;
            variance_sum += diff * diff;
        }
        const variance = variance_sum / @as(T, @floatFromInt(last_dim));

        // Step 3: Normalize with numerical stability
        const std_dev = @sqrt(variance + NORM_EPSILON);

        for (0..last_dim) |i| {
            const normalized = (input.data[batch_offset + i] - mean) / std_dev;

            // Apply scale and shift if provided
            var output_val = normalized;
            if (scale) |s| {
                output_val *= s.data[i];
            }
            if (shift) |b| {
                output_val += b.data[i];
            }

            result.data[batch_offset + i] = output_val;
        }
    }

    return result;
}

/// Root Mean Square Normalization (RMSNorm)
///
/// ## Mathematical Definition
/// ```
/// RMSNorm(x) = γ * (x / √(RMS(x)² + ε))
/// ```
/// Where:
/// - RMS(x) = √((1/d) * Σᵢ xᵢ²)
/// - γ is a learnable scale parameter
/// - ε is numerical stability epsilon
///
/// ## Key Differences from LayerNorm
/// 1. **No Mean Subtraction**: Only normalizes by RMS, not mean-centered
/// 2. **Fewer Parameters**: Only scale parameter, no shift parameter
/// 3. **Faster Computation**: Simpler calculation, fewer operations
/// 4. **Different Invariances**: Scale-invariant but not translation-invariant
///
/// ## Educational Note: Why RMSNorm?
/// RMSNorm offers several advantages:
/// - **Computational Efficiency**: ~15% faster than LayerNorm
/// - **Memory Efficiency**: Fewer parameters to store and update
/// - **Empirical Performance**: Often matches LayerNorm performance
/// - **Simplicity**: Simpler mathematical formulation
///
/// ## Transformer Usage
/// RMSNorm is used in modern efficient transformers:
/// - **LLaMA**: Uses RMSNorm instead of LayerNorm
/// - **Chinchilla**: Employs RMSNorm for efficiency
/// - **PaLM**: Uses RMSNorm in larger models
///
/// The choice between LayerNorm and RMSNorm is often based on
/// computational efficiency requirements.
pub fn rmsNorm(comptime T: type, input: Tensor(T), scale: ?Tensor(T), allocator: Allocator) TensorError!Tensor(T) {
    if (T != f32 and T != f64) return TensorError.InvalidShape;
    if (input.ndim() < 1) return TensorError.IncompatibleShapes;

    const last_dim = input.shape[input.shape.len - 1];
    const batch_size = input.size / last_dim;

    var result = try Tensor(T).init(allocator, input.shape);

    // Process each sequence/batch element independently
    for (0..batch_size) |batch_idx| {
        const batch_offset = batch_idx * last_dim;

        // Step 1: Calculate RMS (Root Mean Square)
        var sum_squares: T = 0.0;
        for (0..last_dim) |i| {
            const x = input.data[batch_offset + i];
            sum_squares += x * x;
        }

        const mean_square = sum_squares / @as(T, @floatFromInt(last_dim));
        const rms = @sqrt(mean_square + NORM_EPSILON);

        // Step 2: Normalize by RMS
        for (0..last_dim) |i| {
            var normalized = input.data[batch_offset + i] / rms;

            // Apply scale if provided
            if (scale) |s| {
                normalized *= s.data[i];
            }

            result.data[batch_offset + i] = normalized;
        }
    }

    return result;
}

/// Batch Normalization
///
/// ## Mathematical Definition
/// For a batch of inputs x with batch statistics:
/// ```
/// BatchNorm(x) = γ * ((x - μ_batch) / √(σ²_batch + ε)) + β
/// ```
/// Where statistics are computed across the batch dimension.
///
/// ## Educational Note: Batch vs Layer Normalization
/// - **BatchNorm**: Normalizes across batch dimension
/// - **LayerNorm**: Normalizes across feature dimension
/// - **Transformers**: Generally prefer LayerNorm for sequence models
///
/// ## Transformer Context
/// BatchNorm is less common in transformers because:
/// - **Variable Sequence Lengths**: Batch statistics are inconsistent
/// - **Small Batch Training**: Unreliable statistics with small batches
/// - **Inference Discrepancy**: Different behavior during training vs inference
///
/// LayerNorm is preferred for these reasons.
pub fn batchNorm(comptime T: type, input: Tensor(T), scale: ?Tensor(T), shift: ?Tensor(T), allocator: Allocator) TensorError!Tensor(T) {
    if (T != f32 and T != f64) return TensorError.InvalidShape;
    if (input.ndim() < 2) return TensorError.IncompatibleShapes;

    const batch_size = input.shape[0];
    const feature_size = input.size / batch_size;

    var result = try Tensor(T).init(allocator, input.shape);

    // Calculate batch statistics for each feature
    for (0..feature_size) |feature_idx| {
        // Step 1: Calculate batch mean for this feature
        var sum: T = 0.0;
        for (0..batch_size) |batch_idx| {
            sum += input.data[batch_idx * feature_size + feature_idx];
        }
        const mean = sum / @as(T, @floatFromInt(batch_size));

        // Step 2: Calculate batch variance for this feature
        var variance_sum: T = 0.0;
        for (0..batch_size) |batch_idx| {
            const diff = input.data[batch_idx * feature_size + feature_idx] - mean;
            variance_sum += diff * diff;
        }
        const variance = variance_sum / @as(T, @floatFromInt(batch_size));
        const std_dev = @sqrt(variance + NORM_EPSILON);

        // Step 3: Normalize all batch elements for this feature
        for (0..batch_size) |batch_idx| {
            const idx = batch_idx * feature_size + feature_idx;
            var normalized = (input.data[idx] - mean) / std_dev;

            // Apply scale and shift
            if (scale) |s| normalized *= s.data[feature_idx];
            if (shift) |b| normalized += b.data[feature_idx];

            result.data[idx] = normalized;
        }
    }

    return result;
}

/// Group Normalization
///
/// ## Mathematical Definition
/// Groups channels into G groups and normalizes within each group:
/// ```
/// GroupNorm(x) = γ * ((x - μ_group) / √(σ²_group + ε)) + β
/// ```
///
/// ## Educational Note: Normalization Spectrum
/// - **InstanceNorm**: G = C (each channel is its own group)
/// - **LayerNorm**: G = 1 (all channels in one group)
/// - **GroupNorm**: 1 < G < C (intermediate grouping)
///
/// ## Transformer Context
/// Less commonly used in transformers, but can be useful for:
/// - **Channel grouping**: When features have natural groupings
/// - **Computational efficiency**: Balance between Layer and Instance norm
/// - **Regularization**: Different normalization can act as regularization
pub fn groupNorm(comptime T: type, input: Tensor(T), groups: usize, scale: ?Tensor(T), shift: ?Tensor(T), allocator: Allocator) TensorError!Tensor(T) {
    if (T != f32 and T != f64) return TensorError.InvalidShape;
    if (input.ndim() < 2) return TensorError.IncompatibleShapes;

    const channels = input.shape[input.shape.len - 1];
    if (channels % groups != 0) return TensorError.IncompatibleShapes;

    const channels_per_group = channels / groups;
    const batch_size = input.size / channels;

    var result = try Tensor(T).init(allocator, input.shape);

    // Process each batch element and group
    for (0..batch_size) |batch_idx| {
        for (0..groups) |group_idx| {
            const group_start = group_idx * channels_per_group;
            const batch_offset = batch_idx * channels;

            // Calculate group statistics
            var sum: T = 0.0;
            for (0..channels_per_group) |i| {
                sum += input.data[batch_offset + group_start + i];
            }
            const mean = sum / @as(T, @floatFromInt(channels_per_group));

            var variance_sum: T = 0.0;
            for (0..channels_per_group) |i| {
                const diff = input.data[batch_offset + group_start + i] - mean;
                variance_sum += diff * diff;
            }
            const variance = variance_sum / @as(T, @floatFromInt(channels_per_group));
            const std_dev = @sqrt(variance + NORM_EPSILON);

            // Apply normalization to group
            for (0..channels_per_group) |i| {
                const idx = batch_offset + group_start + i;
                const channel_idx = group_start + i;

                var normalized = (input.data[idx] - mean) / std_dev;

                if (scale) |s| normalized *= s.data[channel_idx];
                if (shift) |b| normalized += b.data[channel_idx];

                result.data[idx] = normalized;
            }
        }
    }

    return result;
}

/// Generic normalization function dispatcher
///
/// ## Educational Note: Framework Design
/// This dispatcher demonstrates good software engineering practices:
/// - **Single Interface**: Uniform API for different normalization types
/// - **Type Safety**: Compile-time type verification
/// - **Extensibility**: Easy to add new normalization techniques
/// - **Performance**: No runtime overhead from dynamic dispatch
pub fn applyNormalization(comptime T: type, norm_type: NormalizationType, input: Tensor(T),
                         scale: ?Tensor(T), shift: ?Tensor(T), allocator: Allocator) TensorError!Tensor(T) {
    return switch (norm_type) {
        .LayerNorm => layerNorm(T, input, scale, shift, allocator),
        .RMSNorm => rmsNorm(T, input, scale, allocator),
        .BatchNorm => batchNorm(T, input, scale, shift, allocator),
        .GroupNorm => groupNorm(T, input, 8, scale, shift, allocator), // Default 8 groups
    };
}

/// Numerically stable statistics calculation
///
/// ## Educational Note: Numerical Stability
/// Computing variance naively as E[X²] - E[X]² can lead to catastrophic
/// cancellation when the variance is small relative to the mean.
///
/// We use the two-pass algorithm:
/// 1. First pass: Calculate mean
/// 2. Second pass: Calculate variance using the computed mean
///
/// This approach is more numerically stable and prevents precision loss.
pub fn computeStableStats(comptime T: type, data: []const T) struct { mean: T, variance: T } {
    if (data.len == 0) return .{ .mean = 0.0, .variance = 0.0 };

    // First pass: calculate mean
    var sum: T = 0.0;
    for (data) |x| {
        sum += x;
    }
    const mean = sum / @as(T, @floatFromInt(data.len));

    // Second pass: calculate variance
    var variance_sum: T = 0.0;
    for (data) |x| {
        const diff = x - mean;
        variance_sum += diff * diff;
    }
    const variance = variance_sum / @as(T, @floatFromInt(data.len));

    return .{ .mean = mean, .variance = variance };
}

// ============================================================================
// COMPREHENSIVE TESTS
// ============================================================================

test "Layer Normalization correctness" {
    const allocator = testing.allocator;

    // Test with 2D tensor: batch_size=2, features=3
    var input = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer input.deinit();

    // First batch element
    try input.set(&[_]usize{ 0, 0 }, 1.0);
    try input.set(&[_]usize{ 0, 1 }, 2.0);
    try input.set(&[_]usize{ 0, 2 }, 3.0);

    // Second batch element
    try input.set(&[_]usize{ 1, 0 }, 4.0);
    try input.set(&[_]usize{ 1, 1 }, 5.0);
    try input.set(&[_]usize{ 1, 2 }, 6.0);

    // Create scale and shift parameters
    var scale = try Tensor(f32).init(allocator, &[_]usize{3});
    defer scale.deinit();
    scale.fill(1.0); // Identity scale

    var shift = try Tensor(f32).init(allocator, &[_]usize{3});
    defer shift.deinit();
    shift.fill(0.0); // Zero shift

    var result = try layerNorm(f32, input, scale, shift, allocator);
    defer result.deinit();

    // For first batch: mean=2.0, std=sqrt(2/3), should be normalized to ~[-1.22, 0, 1.22]
    const first_batch_mean = try result.get(&[_]usize{ 0, 1 });
    try testing.expectApproxEqAbs(@as(f32, 0.0), first_batch_mean, 1e-6);

    // Test that each row is normalized (mean ≈ 0, std ≈ 1)
    for (0..2) |batch_idx| {
        var batch_sum: f32 = 0.0;
        for (0..3) |feature_idx| {
            batch_sum += try result.get(&[_]usize{ batch_idx, feature_idx });
        }
        const batch_mean = batch_sum / 3.0;
        try testing.expectApproxEqAbs(@as(f32, 0.0), batch_mean, 1e-5);
    }
}

test "RMS Normalization efficiency" {
    const allocator = testing.allocator;

    var input = try Tensor(f32).init(allocator, &[_]usize{ 1, 4 });
    defer input.deinit();

    // Test input: [3, 4, 5, 6]
    try input.set(&[_]usize{ 0, 0 }, 3.0);
    try input.set(&[_]usize{ 0, 1 }, 4.0);
    try input.set(&[_]usize{ 0, 2 }, 5.0);
    try input.set(&[_]usize{ 0, 3 }, 6.0);

    var scale = try Tensor(f32).init(allocator, &[_]usize{4});
    defer scale.deinit();
    scale.fill(2.0); // 2x scale

    var result = try rmsNorm(f32, input, scale, allocator);
    defer result.deinit();

    // Calculate expected RMS: sqrt((9+16+25+36)/4) = sqrt(86/4) = sqrt(21.5) ≈ 4.636
    const expected_rms = @sqrt(21.5);

    // Expected outputs: [3/4.636, 4/4.636, 5/4.636, 6/4.636] * 2
    const expected_0 = (3.0 / expected_rms) * 2.0;

    try testing.expectApproxEqAbs(expected_0, try result.get(&[_]usize{ 0, 0 }), 1e-3);

    // Verify RMS property: no mean subtraction, only RMS normalization
    var sum_squares: f32 = 0.0;
    for (0..4) |i| {
        const val = try result.get(&[_]usize{ 0, i }) / 2.0; // Remove scale
        sum_squares += val * val;
    }
    const output_rms = @sqrt(sum_squares / 4.0);
    try testing.expectApproxEqAbs(@as(f32, 1.0), output_rms, 1e-3);
}

test "Batch Normalization statistics" {
    const allocator = testing.allocator;

    // Test with batch_size=3, features=2
    var input = try Tensor(f32).init(allocator, &[_]usize{ 3, 2 });
    defer input.deinit();

    // Batch 0: [1, 2]
    try input.set(&[_]usize{ 0, 0 }, 1.0);
    try input.set(&[_]usize{ 0, 1 }, 2.0);
    // Batch 1: [3, 4]
    try input.set(&[_]usize{ 1, 0 }, 3.0);
    try input.set(&[_]usize{ 1, 1 }, 4.0);
    // Batch 2: [5, 6]
    try input.set(&[_]usize{ 2, 0 }, 5.0);
    try input.set(&[_]usize{ 2, 1 }, 6.0);

    var result = try batchNorm(f32, input, null, null, allocator);
    defer result.deinit();

    // For feature 0: values [1,3,5], mean=3, std=sqrt(8/3)≈1.633
    // For feature 1: values [2,4,6], mean=4, std=sqrt(8/3)≈1.633

    // Check that batch statistics are normalized correctly
    for (0..2) |feature_idx| {
        var feature_sum: f32 = 0.0;
        for (0..3) |batch_idx| {
            feature_sum += try result.get(&[_]usize{ batch_idx, feature_idx });
        }
        const feature_mean = feature_sum / 3.0;
        try testing.expectApproxEqAbs(@as(f32, 0.0), feature_mean, 1e-5);
    }
}

test "Group Normalization grouping" {
    const allocator = testing.allocator;

    // Test with 1 batch, 6 channels, 2 groups
    var input = try Tensor(f32).init(allocator, &[_]usize{ 1, 6 });
    defer input.deinit();

    for (0..6) |i| {
        try input.set(&[_]usize{ 0, i }, @as(f32, @floatFromInt(i + 1))); // [1,2,3,4,5,6]
    }

    var result = try groupNorm(f32, input, 2, null, null, allocator);
    defer result.deinit();

    // Group 0: channels [1,2,3], should be normalized independently
    // Group 1: channels [4,5,6], should be normalized independently

    // Each group should have approximately zero mean
    var group0_sum: f32 = 0.0;
    var group1_sum: f32 = 0.0;

    for (0..3) |i| {
        group0_sum += try result.get(&[_]usize{ 0, i });
        group1_sum += try result.get(&[_]usize{ 0, i + 3 });
    }

    try testing.expectApproxEqAbs(@as(f32, 0.0), group0_sum / 3.0, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.0), group1_sum / 3.0, 1e-5);
}

test "Numerical stability with small variance" {
    const allocator = testing.allocator;

    var input = try Tensor(f32).init(allocator, &[_]usize{ 1, 3 });
    defer input.deinit();

    // Values very close to each other (small variance)
    try input.set(&[_]usize{ 0, 0 }, 1.000001);
    try input.set(&[_]usize{ 0, 1 }, 1.000002);
    try input.set(&[_]usize{ 0, 2 }, 1.000003);

    var result = try layerNorm(f32, input, null, null, allocator);
    defer result.deinit();

    // Should not produce NaN or infinite values
    for (0..3) |i| {
        const val = try result.get(&[_]usize{ 0, i });
        try testing.expect(math.isFinite(val));
    }

    // Mean should still be approximately zero
    var sum: f32 = 0.0;
    for (0..3) |i| {
        sum += try result.get(&[_]usize{ 0, i });
    }
    try testing.expectApproxEqAbs(@as(f32, 0.0), sum / 3.0, 1e-3);
}

test "Normalization dispatcher" {
    const allocator = testing.allocator;

    var input = try Tensor(f32).init(allocator, &[_]usize{ 1, 4 });
    defer input.deinit();

    for (0..4) |i| {
        try input.set(&[_]usize{ 0, i }, @as(f32, @floatFromInt(i + 1)));
    }

    // Test LayerNorm through dispatcher
    var layer_result = try applyNormalization(f32, .LayerNorm, input, null, null, allocator);
    defer layer_result.deinit();

    // Test RMSNorm through dispatcher
    var rms_result = try applyNormalization(f32, .RMSNorm, input, null, null, allocator);
    defer rms_result.deinit();

    // Results should be different (LayerNorm subtracts mean, RMSNorm doesn't)
    const layer_val = try layer_result.get(&[_]usize{ 0, 0 });
    const rms_val = try rms_result.get(&[_]usize{ 0, 0 });
    try testing.expect(@abs(layer_val - rms_val) > 0.1);
}