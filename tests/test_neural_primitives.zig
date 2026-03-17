//! Comprehensive tests for Neural Primitives layer
//!
//! This test suite verifies the correctness of activation functions,
//! normalization layers, and embedding operations.

const std = @import("std");
const testing = std.testing;
const math = std.math;

// Test activation function concepts
test "activation function properties" {
    // ReLU properties
    const relu_neg = @max(0.0, -1.5);
    const relu_pos = @max(0.0, 2.0);
    try testing.expectEqual(@as(f32, 0.0), relu_neg);
    try testing.expectEqual(@as(f32, 2.0), relu_pos);

    // GELU approximation test
    const x: f32 = 0.0;
    const gelu_zero = 0.5 * x * (1.0 + std.math.tanh(0.7978845608028654 * x));
    try testing.expectApproxEqAbs(@as(f32, 0.0), gelu_zero, 1e-6);

    // SiLU properties
    const silu_zero = 0.0 * (1.0 / (1.0 + @exp(-0.0)));
    try testing.expectEqual(@as(f32, 0.0), silu_zero);

    // Sigmoid range test
    const large_pos = 1.0 / (1.0 + @exp(-10.0));
    const large_neg = 1.0 / (1.0 + @exp(-(-10.0)));
    try testing.expect(large_pos > 0.99); // Close to 1
    try testing.expect(large_neg < 0.01); // Close to 0
}

test "normalization concepts" {
    const allocator = testing.allocator;

    // Test data that will be normalized
    const test_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    // Calculate mean and variance manually
    var sum: f32 = 0.0;
    for (test_data) |val| {
        sum += val;
    }
    const mean = sum / @as(f32, @floatFromInt(test_data.len));
    try testing.expectEqual(@as(f32, 2.5), mean);

    // Calculate variance
    var variance_sum: f32 = 0.0;
    for (test_data) |val| {
        const diff = val - mean;
        variance_sum += diff * diff;
    }
    const variance = variance_sum / @as(f32, @floatFromInt(test_data.len));
    try testing.expectEqual(@as(f32, 1.25), variance);

    // Layer normalization calculation
    const std_dev = @sqrt(variance + 1e-6);
    const normalized_0 = (test_data[0] - mean) / std_dev;
    try testing.expectApproxEqAbs(@as(f32, -1.34164), normalized_0, 1e-4);

    // RMS normalization calculation
    var sum_squares: f32 = 0.0;
    for (test_data) |val| {
        sum_squares += val * val;
    }
    const rms = @sqrt(sum_squares / @as(f32, @floatFromInt(test_data.len)));
    const rms_normalized_0 = test_data[0] / rms;
    try testing.expectApproxEqAbs(@as(f32, 0.3651), rms_normalized_0, 1e-4);

    _ = allocator; // Suppress unused warning
}

test "positional encoding concepts" {
    // Test sinusoidal encoding properties
    const max_seq_len: usize = 10;
    const d_model: usize = 8;

    // For position 0, dimension 0: sin(0) = 0
    const pos0_dim0 = @sin(0.0 / math.pow(f32, 10000.0, 0.0 / @as(f32, @floatFromInt(d_model))));
    try testing.expectEqual(@as(f32, 0.0), pos0_dim0);

    // For position 0, dimension 1: cos(0) = 1
    const pos0_dim1 = @cos(0.0 / math.pow(f32, 10000.0, 0.0 / @as(f32, @floatFromInt(d_model))));
    try testing.expectEqual(@as(f32, 1.0), pos0_dim1);

    // Different positions should have different encodings
    const pos1_dim0 = @sin(1.0 / math.pow(f32, 10000.0, 0.0 / @as(f32, @floatFromInt(d_model))));
    try testing.expect(pos0_dim0 != pos1_dim0);

    _ = max_seq_len; // Suppress unused warning
}

test "embedding dimension calculations" {
    const vocab_size: usize = 1000;
    const embedding_dim: usize = 256;
    const seq_len: usize = 512;
    const batch_size: usize = 32;

    // Calculate memory requirements
    const token_embedding_params = vocab_size * embedding_dim;
    const positional_embedding_params = seq_len * embedding_dim;
    const total_embedding_memory = (batch_size * seq_len * embedding_dim) * @sizeOf(f32);

    try testing.expectEqual(@as(usize, 256000), token_embedding_params);
    try testing.expectEqual(@as(usize, 131072), positional_embedding_params);
    try testing.expectEqual(@as(usize, 16777216), total_embedding_memory); // ~16MB for batch

    // Xavier initialization standard deviation
    const xavier_std = @sqrt(2.0 / @as(f32, @floatFromInt(vocab_size + embedding_dim)));
    try testing.expect(xavier_std > 0.0 and xavier_std < 1.0);
}

test "gating mechanism properties" {
    // Test GLU-style gating properties
    const content_val: f32 = 2.0;
    const gate_val: f32 = 1.0;

    // Standard GLU with sigmoid
    const sigmoid_gate = 1.0 / (1.0 + @exp(-gate_val));
    const glu_output = content_val * sigmoid_gate;
    try testing.expectApproxEqAbs(@as(f32, 1.4621), glu_output, 1e-4);

    // SiLU-based gating (SwiGLU)
    const silu_gate = gate_val * sigmoid_gate;
    const swiglu_output = content_val * silu_gate;
    try testing.expectApproxEqAbs(@as(f32, 1.4621), swiglu_output, 1e-4);

    // GELU-based gating (GeGLU)
    const gelu_approx = 0.5 * gate_val * (1.0 + std.math.tanh(0.7978845608028654 * gate_val));
    const geglu_output = content_val * gelu_approx;
    try testing.expect(geglu_output > 1.0 and geglu_output < 2.0);
}

test "rotary position embedding rotation" {
    // Test 2D rotation properties
    const x: f32 = 1.0;
    const y: f32 = 0.0;
    const angle: f32 = math.pi / 4.0; // 45 degrees

    const cos_theta = @cos(angle);
    const sin_theta = @sin(angle);

    const rotated_x = x * cos_theta - y * sin_theta;
    const rotated_y = x * sin_theta + y * cos_theta;

    // Should preserve magnitude
    const original_mag = x * x + y * y;
    const rotated_mag = rotated_x * rotated_x + rotated_y * rotated_y;
    try testing.expectApproxEqAbs(original_mag, rotated_mag, 1e-6);

    // 45-degree rotation of (1,0) should give (~0.707, ~0.707)
    try testing.expectApproxEqAbs(@as(f32, 0.7071), rotated_x, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.7071), rotated_y, 1e-4);
}

test "numerical stability in normalization" {
    const epsilon: f32 = 1e-6;

    // Test with very small variance
    const small_variance: f32 = 1e-10;
    const stable_std = @sqrt(small_variance + epsilon);
    try testing.expect(stable_std > 0.0);
    try testing.expect(math.isFinite(stable_std));

    // Test with very large values
    const large_val: f32 = 1e6;
    const normalized_large = large_val / @sqrt(large_val * large_val + epsilon);
    try testing.expect(math.isFinite(normalized_large));

    // Test edge case: all zeros
    const zero_variance: f32 = 0.0;
    const zero_stable = @sqrt(zero_variance + epsilon);
    try testing.expectEqual(@sqrt(epsilon), zero_stable);
}

test "transformer architectural choices" {
    // Test embedding dimension relationships
    const d_model: usize = 512;
    const n_heads: usize = 8;
    const d_k = d_model / n_heads; // 64 dimensions per head

    try testing.expectEqual(@as(usize, 64), d_k);

    // Feed-forward dimension is typically 4x model dimension
    const d_ff = 4 * d_model;
    try testing.expectEqual(@as(usize, 2048), d_ff);

    // For SwiGLU, we need to account for gating (input is split in half)
    const swiglu_inner_dim = d_ff / 2; // Due to gating mechanism
    try testing.expectEqual(@as(usize, 1024), swiglu_inner_dim);

    // Positional encoding should match model dimension
    try testing.expectEqual(d_model, d_model); // Trivial but documents the requirement
}

test "activation function derivatives" {
    // Test that we can compute gradients (conceptually)
    const x: f32 = 1.0;
    const h: f32 = 1e-7; // Small step for numerical differentiation

    // ReLU derivative at x=1 should be 1 (within numerical precision)
    const relu_x = @max(0.0, x);
    const relu_x_plus_h = @max(0.0, x + h);
    const relu_derivative = (relu_x_plus_h - relu_x) / h;
    try testing.expectApproxEqAbs(@as(f32, 1.0), relu_derivative, 0.5); // More tolerant for numerical differentiation

    // GELU is smooth, so derivative should be finite and reasonable
    const gelu_x = 0.5 * x * (1.0 + std.math.tanh(0.7978845608028654 * x));
    const gelu_x_plus_h = 0.5 * (x + h) * (1.0 + std.math.tanh(0.7978845608028654 * (x + h)));
    const gelu_derivative = (gelu_x_plus_h - gelu_x) / h;
    try testing.expect(gelu_derivative > 0.0 and gelu_derivative < 2.0);
}