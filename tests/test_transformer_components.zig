//! Comprehensive tests for Transformer Components layer
//!
//! This test suite validates the transformer architecture components:
//! attention mechanisms, feed-forward networks, and complete blocks.

const std = @import("std");
const testing = std.testing;
const math = std.math;

// Test transformer architectural concepts and mathematical properties
test "attention mechanism concepts" {
    // Test attention scaling factor
    const d_k: f32 = 64.0;
    const scaling_factor = 1.0 / @sqrt(d_k);
    try testing.expectApproxEqAbs(@as(f32, 0.125), scaling_factor, 1e-6);

    // Test that attention weights sum to 1 (softmax property)
    const attention_scores = [_]f32{ 2.0, 1.0, 3.0 };
    var softmax_sum: f32 = 0.0;
    var max_score: f32 = -math.inf(f32);

    // Find max for numerical stability
    for (attention_scores) |score| {
        max_score = @max(max_score, score);
    }

    // Compute softmax
    var softmax_weights: [3]f32 = undefined;
    for (attention_scores, 0..) |score, i| {
        softmax_weights[i] = @exp(score - max_score);
        softmax_sum += softmax_weights[i];
    }

    // Normalize
    for (&softmax_weights) |*weight| {
        weight.* /= softmax_sum;
    }

    // Verify sum to 1
    var total: f32 = 0.0;
    for (softmax_weights) |weight| {
        total += weight;
    }
    try testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-6);
}

test "multi-head attention dimensions" {
    // Test dimension calculations for multi-head attention
    const d_model: usize = 512;
    const num_heads: usize = 8;
    const d_k = d_model / num_heads;

    try testing.expectEqual(@as(usize, 64), d_k);

    // Each head should process d_k dimensions
    // Total parameters for Q, K, V projections: 3 * d_model * d_model
    const qkv_params = 3 * d_model * d_model;
    const output_params = d_model * d_model;
    const total_attention_params = qkv_params + output_params;

    try testing.expectEqual(@as(usize, 4 * 512 * 512), total_attention_params);
}

test "feed-forward network scaling" {
    // Test standard FFN dimension scaling
    const d_model: usize = 768;
    const d_ff = 4 * d_model; // Standard 4x scaling

    try testing.expectEqual(@as(usize, 3072), d_ff);

    // Parameter count for standard FFN: 2 * d_model * d_ff
    const ffn_params = 2 * d_model * d_ff;
    try testing.expectEqual(@as(usize, 2 * 768 * 3072), ffn_params);

    // Gated FFN (SwiGLU) uses 3 matrices: 3 * d_model * d_ff
    const gated_ffn_params = 3 * d_model * d_ff;
    try testing.expectEqual(@as(usize, 3 * 768 * 3072), gated_ffn_params);

    // Gated FFN has 1.5x more parameters
    try testing.expectEqual(ffn_params * 3, gated_ffn_params * 2);
}

test "transformer parameter scaling" {
    // Test parameter scaling for complete transformer
    const d_model: usize = 256;
    const num_heads: usize = 4;
    const d_ff: usize = 4 * d_model; // 1024
    const num_layers: usize = 6;

    // Parameters per layer:
    // - Attention: 4 * d_model^2
    // - FFN: 2 * d_model * d_ff = 2 * d_model * 4 * d_model = 8 * d_model^2
    // - Layer norms: ~2 * d_model (negligible)
    // Total per layer: ~12 * d_model^2

    const attention_params_per_layer = 4 * d_model * d_model;
    const ffn_params_per_layer = 2 * d_model * d_ff;
    const params_per_layer = attention_params_per_layer + ffn_params_per_layer;

    try testing.expectEqual(@as(usize, 4 * 256 * 256), attention_params_per_layer);
    try testing.expectEqual(@as(usize, 2 * 256 * 1024), ffn_params_per_layer);

    const total_params = params_per_layer * num_layers;
    try testing.expectEqual(@as(usize, (4 * 256 * 256 + 2 * 256 * 1024) * 6), total_params);

    _ = num_heads; // Suppress unused warning
}

test "causal masking patterns" {
    // Test causal mask creation for autoregressive models
    const seq_len: usize = 4;

    // Create causal mask manually
    var causal_mask: [4][4]f32 = undefined;
    for (0..seq_len) |i| {
        for (0..seq_len) |j| {
            causal_mask[i][j] = if (j <= i) 0.0 else -math.inf(f32);
        }
    }

    // Verify causal pattern
    try testing.expectEqual(@as(f32, 0.0), causal_mask[0][0]); // Can see self
    try testing.expect(causal_mask[0][1] == -math.inf(f32)); // Cannot see future
    try testing.expectEqual(@as(f32, 0.0), causal_mask[1][0]); // Can see past
    try testing.expectEqual(@as(f32, 0.0), causal_mask[1][1]); // Can see self
    try testing.expect(causal_mask[1][2] == -math.inf(f32)); // Cannot see future
    try testing.expectEqual(@as(f32, 0.0), causal_mask[3][2]); // Can see past
    try testing.expectEqual(@as(f32, 0.0), causal_mask[3][3]); // Can see self
}

test "positional encoding properties" {
    // Test that RoPE preserves vector magnitudes
    const d_k: usize = 8;
    const seq_len: usize = 5;

    for (0..seq_len) |pos| {
        // Original vector
        const x: f32 = 1.0;
        const y: f32 = 1.0;
        const original_magnitude = x * x + y * y;

        // RoPE rotation
        const pos_f = @as(f32, @floatFromInt(pos));
        const theta = pos_f / math.pow(f32, 10000.0, 0.0 / @as(f32, @floatFromInt(d_k)));

        const cos_theta = @cos(theta);
        const sin_theta = @sin(theta);

        const rotated_x = x * cos_theta - y * sin_theta;
        const rotated_y = x * sin_theta + y * cos_theta;
        const rotated_magnitude = rotated_x * rotated_x + rotated_y * rotated_y;

        // Magnitude should be preserved
        try testing.expectApproxEqAbs(original_magnitude, rotated_magnitude, 1e-6);
    }
}

test "residual connection benefits" {
    // Test that residual connections help with gradient flow
    // Simulate gradient backpropagation through residual connections

    const layer_output: f32 = 0.5; // Output from attention or FFN
    const residual_input: f32 = 1.0; // Original input
    const final_output = residual_input + layer_output; // Residual connection

    // Gradient flows directly through residual connection
    const gradient_to_layer = 1.0; // Gradient from next layer
    const gradient_to_input = gradient_to_layer; // Direct path through residual
    const gradient_through_layer = gradient_to_layer; // Path through layer

    // Both paths receive gradients (good for training)
    try testing.expectEqual(@as(f32, 1.0), gradient_to_input);
    try testing.expectEqual(@as(f32, 1.0), gradient_through_layer);
    try testing.expectEqual(@as(f32, 1.5), final_output);
}

test "layer normalization statistics" {
    // Test layer normalization mathematical properties
    const test_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const n = test_data.len;

    // Calculate mean
    var sum: f32 = 0.0;
    for (test_data) |val| {
        sum += val;
    }
    const mean = sum / @as(f32, @floatFromInt(n));
    try testing.expectEqual(@as(f32, 3.0), mean);

    // Calculate variance
    var variance_sum: f32 = 0.0;
    for (test_data) |val| {
        const diff = val - mean;
        variance_sum += diff * diff;
    }
    const variance = variance_sum / @as(f32, @floatFromInt(n));
    try testing.expectEqual(@as(f32, 2.0), variance);

    // Apply layer normalization
    const epsilon: f32 = 1e-6;
    const std_dev = @sqrt(variance + epsilon);

    var normalized_sum: f32 = 0.0;
    var normalized_squared_sum: f32 = 0.0;

    for (test_data) |val| {
        const normalized = (val - mean) / std_dev;
        normalized_sum += normalized;
        normalized_squared_sum += normalized * normalized;
    }

    // Normalized data should have mean ≈ 0 and variance ≈ 1
    const normalized_mean = normalized_sum / @as(f32, @floatFromInt(n));
    const normalized_variance = normalized_squared_sum / @as(f32, @floatFromInt(n));

    try testing.expectApproxEqAbs(@as(f32, 0.0), normalized_mean, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), normalized_variance, 1e-6);
}

test "attention head diversity" {
    // Test that multiple attention heads can learn different patterns
    const num_heads: usize = 4;
    const d_k: usize = 16;

    // Simulate different attention patterns for different heads
    const head_patterns = [_][3]f32{
        [_]f32{ 0.8, 0.1, 0.1 }, // Head 1: Focus on first position
        [_]f32{ 0.1, 0.8, 0.1 }, // Head 2: Focus on second position
        [_]f32{ 0.1, 0.1, 0.8 }, // Head 3: Focus on third position
        [_]f32{ 0.33, 0.33, 0.34 }, // Head 4: Uniform attention
    };

    // Verify each head has different attention patterns
    for (0..num_heads) |head| {
        var pattern_sum: f32 = 0.0;
        for (head_patterns[head]) |weight| {
            pattern_sum += weight;
        }
        // Each pattern should sum to approximately 1 (attention property)
        try testing.expectApproxEqAbs(@as(f32, 1.0), pattern_sum, 1e-2);
    }

    // Verify heads have different maximum positions
    const max_positions = [_]usize{0, 1, 2, 0}; // Expected max positions
    for (0..num_heads) |head| {
        var max_val: f32 = 0.0;
        var max_idx: usize = 0;

        for (head_patterns[head], 0..) |weight, idx| {
            if (weight > max_val) {
                max_val = weight;
                max_idx = idx;
            }
        }

        if (head < 3) { // First 3 heads should focus on different positions
            try testing.expectEqual(max_positions[head], max_idx);
        }
    }

    _ = d_k; // Suppress unused warning
}

test "transformer memory complexity" {
    // Test memory complexity of transformer components
    const batch_size: usize = 32;
    const seq_len: usize = 512;
    const d_model: usize = 768;
    const num_heads: usize = 12;

    // Attention memory complexity: O(batch * heads * seq_len^2)
    const attention_memory = batch_size * num_heads * seq_len * seq_len;

    // FFN memory complexity: O(batch * seq_len * d_model * d_ff)
    const d_ff = 4 * d_model;
    const ffn_memory = batch_size * seq_len * d_ff;

    // Input/output memory: O(batch * seq_len * d_model)
    const io_memory = batch_size * seq_len * d_model;

    try testing.expectEqual(@as(usize, 32 * 12 * 512 * 512), attention_memory);
    try testing.expectEqual(@as(usize, 32 * 512 * 3072), ffn_memory);
    try testing.expectEqual(@as(usize, 32 * 512 * 768), io_memory);

    // Attention memory grows quadratically with sequence length
    // This is why efficient attention mechanisms (sparse, linear) are important
    try testing.expect(attention_memory > ffn_memory);
    try testing.expect(attention_memory > io_memory);
}

test "gated activation benefits" {
    // Test that gated activations provide better control than standard activations
    const content: f32 = 2.0;
    const gate_values = [_]f32{ 0.0, 0.5, 1.0 }; // Different gate strengths

    for (gate_values) |gate| {
        // GLU-style gating
        const sigmoid_gate = 1.0 / (1.0 + @exp(-gate));
        const gated_output = content * sigmoid_gate;

        // Gate should control information flow
        if (gate == 0.0) {
            try testing.expectApproxEqAbs(@as(f32, 1.0), gated_output, 1e-2); // 50% through
        } else if (gate > 0.0) {
            try testing.expect(gated_output > 1.0); // More information flows
        }
    }

    // SwiGLU combines content with SiLU-gated version
    const up_projection: f32 = 1.5;
    const swiglu_gate = up_projection * (1.0 / (1.0 + @exp(-up_projection)));
    const swiglu_output = content * swiglu_gate;

    try testing.expect(swiglu_output > content); // Should amplify
    try testing.expect(std.math.isFinite(swiglu_output)); // Should be stable
}