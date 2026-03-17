//! Transformer Components: Multi-Head Attention
//!
//! This module implements the multi-head attention mechanism that is the core
//! innovation of transformer architectures, enabling models to attend to
//! different parts of sequences simultaneously.
//!
//! ## Educational Objectives
//! - Understand the mathematical foundation of attention mechanisms
//! - Learn how multi-head attention enables parallel processing of information
//! - Implement scaled dot-product attention with proper masking
//! - Connect attention to information retrieval and memory systems
//!
//! ## Transformer Context
//! Attention is the key breakthrough that enabled transformers:
//! - **Parallelization**: Unlike RNNs, attention can be computed in parallel
//! - **Long-range Dependencies**: Direct connections between any two positions
//! - **Interpretability**: Attention weights show what the model focuses on
//! - **Flexibility**: Same mechanism works for encoder and decoder architectures

const std = @import("std");
const math = std.math;
const testing = std.testing;
const Allocator = std.mem.Allocator;

// Import our foundation layers
const Tensor = @import("../foundation/tensor.zig").Tensor;
const TensorError = @import("../foundation/tensor.zig").TensorError;
const matrix_ops = @import("../linear_algebra/matrix_ops.zig");
const activations = @import("../neural_primitives/activations.zig");
const normalization = @import("../neural_primitives/normalization.zig");

/// Attention mechanism types used in different transformer variants
pub const AttentionType = enum {
    SelfAttention,      // Attend to same sequence (encoder)
    CrossAttention,     // Attend to different sequence (decoder)
    CausalAttention,    // Masked self-attention (decoder, autoregressive)
    SparseAttention,    // Sparse attention patterns
};

/// Multi-Head Attention Layer
///
/// ## Mathematical Definition
/// For input X ∈ R^(seq_len × d_model), Multi-Head Attention computes:
/// ```
/// MultiHead(Q,K,V) = Concat(head₁, head₂, ..., headₕ)W^O
/// where headᵢ = Attention(QW^Q_i, KW^K_i, VW^V_i)
/// ```
///
/// Each attention head computes:
/// ```
/// Attention(Q,K,V) = softmax(QK^T / √d_k)V
/// ```
///
/// ## Educational Note: Why Multiple Heads?
/// Multiple attention heads allow the model to jointly attend to information
/// from different representation subspaces at different positions:
///
/// 1. **Parallel Processing**: Each head can focus on different patterns
/// 2. **Diverse Representations**: Different heads learn different types of relationships
/// 3. **Computational Efficiency**: Smaller matrices per head (d_k = d_model/num_heads)
/// 4. **Empirical Success**: Consistently improves performance over single-head
///
/// ## Intuitive Understanding
/// Think of attention heads as different "perspectives" or "questions":
/// - Head 1: "Which words are syntactically related?"
/// - Head 2: "Which words are semantically similar?"
/// - Head 3: "Which words indicate temporal relationships?"
/// - etc.
pub const MultiHeadAttention = struct {
    /// Number of attention heads
    num_heads: usize,

    /// Model dimension
    d_model: usize,

    /// Key/query dimension per head
    d_k: usize,

    /// Value dimension per head
    d_v: usize,

    /// Linear projection for queries [d_model × d_model]
    w_q: Tensor(f32),

    /// Linear projection for keys [d_model × d_model]
    w_k: Tensor(f32),

    /// Linear projection for values [d_model × d_model]
    w_v: Tensor(f32),

    /// Output projection [d_model × d_model]
    w_o: Tensor(f32),

    /// Memory allocator
    allocator: Allocator,

    /// Initialize multi-head attention layer
    ///
    /// ## Educational Note: Parameter Initialization
    /// Proper initialization is crucial for transformer training stability:
    /// - **Xavier/Glorot**: Balanced gradient flow
    /// - **Small weights**: Prevent attention saturation
    /// - **Zero bias**: Standard practice for attention layers
    ///
    /// We use Xavier initialization scaled for the specific tensor dimensions.
    pub fn init(allocator: Allocator, d_model: usize, num_heads: usize) !MultiHeadAttention {
        if (d_model % num_heads != 0) return TensorError.IncompatibleShapes;

        const d_k = d_model / num_heads;
        const d_v = d_k; // Standard practice: d_v = d_k

        // Initialize projection matrices
        var w_q = try Tensor(f32).init(allocator, &[_]usize{ d_model, d_model });
        var w_k = try Tensor(f32).init(allocator, &[_]usize{ d_model, d_model });
        var w_v = try Tensor(f32).init(allocator, &[_]usize{ d_model, d_model });
        var w_o = try Tensor(f32).init(allocator, &[_]usize{ d_model, d_model });

        // Xavier initialization
        const xavier_std = @sqrt(2.0 / @as(f32, @floatFromInt(d_model + d_model)));
        initializeWeights(&w_q, xavier_std);
        initializeWeights(&w_k, xavier_std);
        initializeWeights(&w_v, xavier_std);
        initializeWeights(&w_o, xavier_std);

        return MultiHeadAttention{
            .num_heads = num_heads,
            .d_model = d_model,
            .d_k = d_k,
            .d_v = d_v,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .allocator = allocator,
        };
    }

    /// Clean up resources
    pub fn deinit(self: *MultiHeadAttention) void {
        self.w_q.deinit();
        self.w_k.deinit();
        self.w_v.deinit();
        self.w_o.deinit();
    }

    /// Forward pass for multi-head attention
    ///
    /// ## Input Shapes
    /// - query: [batch_size, seq_len, d_model]
    /// - key: [batch_size, seq_len, d_model] (can be different seq_len for cross-attention)
    /// - value: [batch_size, seq_len, d_model]
    /// - mask: Optional [batch_size, seq_len, seq_len] for causal masking
    ///
    /// ## Output Shape
    /// - output: [batch_size, seq_len, d_model]
    ///
    /// ## Educational Note: Attention Flow
    /// 1. **Project**: Transform inputs to Q, K, V using learned matrices
    /// 2. **Reshape**: Split into multiple heads
    /// 3. **Attend**: Compute scaled dot-product attention per head
    /// 4. **Concatenate**: Combine all heads
    /// 5. **Project**: Final linear transformation
    pub fn forward(self: *const MultiHeadAttention, query: Tensor(f32), key: Tensor(f32), value: Tensor(f32), mask: ?Tensor(f32)) !Tensor(f32) {
        if (query.ndim() != 3 or key.ndim() != 3 or value.ndim() != 3) {
            return TensorError.IncompatibleShapes;
        }

        const batch_size = query.shape[0];
        const seq_len_q = query.shape[1];
        const seq_len_k = key.shape[1];

        // Step 1: Linear projections to get Q, K, V
        var Q = try query.matmul(self.w_q, self.allocator);
        defer Q.deinit();
        var K = try key.matmul(self.w_k, self.allocator);
        defer K.deinit();
        var V = try value.matmul(self.w_v, self.allocator);
        defer V.deinit();

        // Step 2: Reshape and transpose for multi-head attention
        // From [batch, seq_len, d_model] to [batch, num_heads, seq_len, d_k]
        var Q_heads = try reshapeForHeads(Q, batch_size, seq_len_q, self.num_heads, self.d_k, self.allocator);
        defer Q_heads.deinit();
        var K_heads = try reshapeForHeads(K, batch_size, seq_len_k, self.num_heads, self.d_k, self.allocator);
        defer K_heads.deinit();
        var V_heads = try reshapeForHeads(V, batch_size, seq_len_k, self.num_heads, self.d_v, self.allocator);
        defer V_heads.deinit();

        // Step 3: Compute scaled dot-product attention for each head
        var attention_output = try scaledDotProductAttention(Q_heads, K_heads, V_heads, mask, self.allocator);
        defer attention_output.deinit();

        // Step 4: Concatenate heads
        // From [batch, num_heads, seq_len, d_v] to [batch, seq_len, d_model]
        var concatenated = try reshapeFromHeads(attention_output, batch_size, seq_len_q, self.num_heads, self.d_v, self.allocator);
        defer concatenated.deinit();

        // Step 5: Final linear projection
        return try concatenated.matmul(self.w_o, self.allocator);
    }
};

/// Scaled Dot-Product Attention
///
/// ## Mathematical Definition
/// ```
/// Attention(Q,K,V) = softmax(QK^T / √d_k)V
/// ```
///
/// ## Educational Note: Why Scaling?
/// The scaling factor 1/√d_k prevents the dot products from growing too large:
/// - **Variance Control**: Dot products have variance d_k, scaling normalizes this
/// - **Softmax Stability**: Large logits cause softmax to saturate (gradients → 0)
/// - **Gradient Flow**: Proper scaling maintains good gradient magnitudes
///
/// ## Algorithm Steps
/// 1. **Compute Scores**: QK^T (attention logits)
/// 2. **Scale**: Divide by √d_k
/// 3. **Mask**: Apply causal/padding masks (optional)
/// 4. **Normalize**: Apply softmax to get attention weights
/// 5. **Apply**: Multiply by values to get output
pub fn scaledDotProductAttention(Q: Tensor(f32), K: Tensor(f32), V: Tensor(f32), mask: ?Tensor(f32), allocator: Allocator) !Tensor(f32) {
    if (Q.ndim() != 4 or K.ndim() != 4 or V.ndim() != 4) return TensorError.IncompatibleShapes;

    const batch_size = Q.shape[0];
    const num_heads = Q.shape[1];
    const seq_len_q = Q.shape[2];
    const seq_len_k = K.shape[2];
    const d_k = Q.shape[3];

    // Step 1: Compute attention scores QK^T
    // We need to perform batched matrix multiplication: [batch, heads, seq_q, d_k] × [batch, heads, d_k, seq_k]
    var scores = try batchedMatMul(Q, K, allocator, true); // transpose K
    defer scores.deinit();

    // Step 2: Scale by sqrt(d_k)
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d_k)));
    for (0..scores.size) |i| {
        scores.data[i] *= scale;
    }

    // Step 3: Apply mask if provided (for causal attention or padding)
    if (mask) |attention_mask| {
        try applyMask(&scores, attention_mask);
    }

    // Step 4: Apply softmax to get attention weights
    var attention_weights = try softmaxLastDim(scores, allocator);
    defer attention_weights.deinit();

    // Step 5: Apply attention weights to values
    return try batchedMatMul(attention_weights, V, allocator, false);
}

/// Causal Attention Mask
///
/// ## Educational Note: Autoregressive Property
/// Causal masking ensures that position i can only attend to positions ≤ i:
/// - **Training**: Prevents model from "cheating" by seeing future tokens
/// - **Inference**: Maintains consistency between training and generation
/// - **Implementation**: Set future positions to -∞ before softmax
///
/// ## Mask Pattern
/// ```
/// 1 0 0 0    (position 0 can only see position 0)
/// 1 1 0 0    (position 1 can see positions 0,1)
/// 1 1 1 0    (position 2 can see positions 0,1,2)
/// 1 1 1 1    (position 3 can see positions 0,1,2,3)
/// ```
pub fn createCausalMask(allocator: Allocator, seq_len: usize) !Tensor(f32) {
    var mask = try Tensor(f32).init(allocator, &[_]usize{ seq_len, seq_len });

    for (0..seq_len) |i| {
        for (0..seq_len) |j| {
            // Allow attention to current and previous positions
            const value: f32 = if (j <= i) 0.0 else -math.inf(f32);
            try mask.set(&[_]usize{ i, j }, value);
        }
    }

    return mask;
}

/// Positional Encoding Integration
///
/// ## Educational Note: Position-Aware Attention
/// Modern transformers integrate positional information directly into attention:
/// - **Additive**: Traditional approach (add to embeddings)
/// - **RoPE**: Rotary embeddings (multiply query/key)
/// - **ALiBi**: Linear biases in attention scores
///
/// This function applies RoPE-style rotary encodings to queries and keys.
pub fn applyRotaryEncoding(queries: Tensor(f32), keys: Tensor(f32), seq_len: usize, allocator: Allocator) !struct { q: Tensor(f32), k: Tensor(f32) } {
    // Apply rotary position embeddings to queries and keys
    const rotated_q = try applyRoPE(queries, seq_len, allocator);
    const rotated_k = try applyRoPE(keys, seq_len, allocator);

    return .{ .q = rotated_q, .k = rotated_k };
}

/// Apply Rotary Position Embeddings (RoPE)
///
/// ## Mathematical Definition
/// For each dimension pair (2i, 2i+1) and position m:
/// ```
/// [q_{2i}, q_{2i+1}] = [cos(mθᵢ), -sin(mθᵢ); sin(mθᵢ), cos(mθᵢ)] [q_{2i}, q_{2i+1}]
/// ```
/// where θᵢ = 1/10000^(2i/d)
fn applyRoPE(tensor: Tensor(f32), seq_len: usize, allocator: Allocator) !Tensor(f32) {
    var result = try Tensor(f32).init(allocator, tensor.shape);
    @memcpy(result.data, tensor.data);

    if (tensor.ndim() != 4) return result; // Skip if not the expected shape

    const batch_size = tensor.shape[0];
    const num_heads = tensor.shape[1];
    const d_k = tensor.shape[3];

    for (0..batch_size) |batch_idx| {
        for (0..num_heads) |head_idx| {
            for (0..seq_len) |pos| {
                const pos_f = @as(f32, @floatFromInt(pos));

                // Apply rotation to pairs of dimensions
                for (0..d_k / 2) |dim_pair| {
                    const i = dim_pair * 2;
                    const j = i + 1;

                    if (j >= d_k) continue;

                    // Calculate rotation angle
                    const dim_f = @as(f32, @floatFromInt(dim_pair));
                    const d_k_f = @as(f32, @floatFromInt(d_k));
                    const theta = pos_f / math.pow(f32, 10000.0, 2.0 * dim_f / d_k_f);

                    // Get values to rotate
                    const x = try tensor.get(&[_]usize{ batch_idx, head_idx, pos, i });
                    const y = try tensor.get(&[_]usize{ batch_idx, head_idx, pos, j });

                    // Apply 2D rotation
                    const cos_theta = @cos(theta);
                    const sin_theta = @sin(theta);

                    const rotated_x = x * cos_theta - y * sin_theta;
                    const rotated_y = x * sin_theta + y * cos_theta;

                    try result.set(&[_]usize{ batch_idx, head_idx, pos, i }, rotated_x);
                    try result.set(&[_]usize{ batch_idx, head_idx, pos, j }, rotated_y);
                }
            }
        }
    }

    return result;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Initialize weights with Xavier/Glorot initialization
fn initializeWeights(tensor: *Tensor(f32), std_dev: f32) void {
    var seed: u32 = 12345;
    for (0..tensor.size) |i| {
        seed = seed *% 1664525 +% 1013904223;
        const rand_f32 = @as(f32, @floatFromInt(seed)) / @as(f32, @floatFromInt(std.math.maxInt(u32)));
        tensor.data[i] = (rand_f32 - 0.5) * 2.0 * std_dev;
    }
}

/// Reshape tensor for multi-head attention
/// From [batch, seq_len, d_model] to [batch, num_heads, seq_len, d_k]
fn reshapeForHeads(tensor: Tensor(f32), batch_size: usize, seq_len: usize, num_heads: usize, d_k: usize, allocator: Allocator) !Tensor(f32) {
    var result = try Tensor(f32).init(allocator, &[_]usize{ batch_size, num_heads, seq_len, d_k });

    for (0..batch_size) |b| {
        for (0..seq_len) |s| {
            for (0..num_heads) |h| {
                for (0..d_k) |d| {
                    const src_idx = &[_]usize{ b, s, h * d_k + d };
                    const dst_idx = &[_]usize{ b, h, s, d };
                    const value = try tensor.get(src_idx);
                    try result.set(dst_idx, value);
                }
            }
        }
    }

    return result;
}

/// Reshape tensor from multi-head format back to standard format
/// From [batch, num_heads, seq_len, d_v] to [batch, seq_len, d_model]
fn reshapeFromHeads(tensor: Tensor(f32), batch_size: usize, seq_len: usize, num_heads: usize, d_v: usize, allocator: Allocator) !Tensor(f32) {
    const d_model = num_heads * d_v;
    var result = try Tensor(f32).init(allocator, &[_]usize{ batch_size, seq_len, d_model });

    for (0..batch_size) |b| {
        for (0..seq_len) |s| {
            for (0..num_heads) |h| {
                for (0..d_v) |d| {
                    const src_idx = &[_]usize{ b, h, s, d };
                    const dst_idx = &[_]usize{ b, s, h * d_v + d };
                    const value = try tensor.get(src_idx);
                    try result.set(dst_idx, value);
                }
            }
        }
    }

    return result;
}

/// Batched matrix multiplication for attention computation
fn batchedMatMul(a: Tensor(f32), b: Tensor(f32), allocator: Allocator, transpose_b: bool) !Tensor(f32) {
    if (a.ndim() != 4 or b.ndim() != 4) return TensorError.IncompatibleShapes;

    const batch_size = a.shape[0];
    const num_heads = a.shape[1];
    const seq_len_a = a.shape[2];
    const d_a = a.shape[3];

    const seq_len_b = if (transpose_b) b.shape[3] else b.shape[2];
    const d_b = if (transpose_b) b.shape[2] else b.shape[3];

    if (d_a != d_b) return TensorError.IncompatibleShapes;

    const result_shape = [_]usize{ batch_size, num_heads, seq_len_a, seq_len_b };
    var result = try Tensor(f32).init(allocator, &result_shape);
    result.fill(0.0);

    // Perform batched matrix multiplication
    for (0..batch_size) |batch_idx| {
        for (0..num_heads) |head_idx| {
            for (0..seq_len_a) |i| {
                for (0..seq_len_b) |j| {
                    var sum: f32 = 0.0;

                    for (0..d_a) |k| {
                        const a_val = try a.get(&[_]usize{ batch_idx, head_idx, i, k });

                        const b_val = if (transpose_b)
                            try b.get(&[_]usize{ batch_idx, head_idx, j, k })
                        else
                            try b.get(&[_]usize{ batch_idx, head_idx, k, j });

                        sum += a_val * b_val;
                    }

                    try result.set(&[_]usize{ batch_idx, head_idx, i, j }, sum);
                }
            }
        }
    }

    return result;
}

/// Apply mask to attention scores (set masked positions to -inf)
fn applyMask(scores: *Tensor(f32), mask: Tensor(f32)) !void {
    if (scores.size != mask.size) return TensorError.IncompatibleShapes;

    for (0..scores.size) |i| {
        if (mask.data[i] < 0) {
            scores.data[i] = -math.inf(f32);
        }
    }
}

/// Softmax operation on the last dimension
fn softmaxLastDim(tensor: Tensor(f32), allocator: Allocator) !Tensor(f32) {
    var result = try Tensor(f32).init(allocator, tensor.shape);

    const batch_size = tensor.shape[0];
    const num_heads = tensor.shape[1];
    const seq_len_q = tensor.shape[2];
    const seq_len_k = tensor.shape[3];

    // Apply softmax to each attention head independently
    for (0..batch_size) |batch_idx| {
        for (0..num_heads) |head_idx| {
            for (0..seq_len_q) |q_idx| {
                // Find maximum for numerical stability
                var max_val: f32 = -math.inf(f32);
                for (0..seq_len_k) |k_idx| {
                    const val = try tensor.get(&[_]usize{ batch_idx, head_idx, q_idx, k_idx });
                    max_val = @max(max_val, val);
                }

                // Compute exponentials and sum
                var exp_sum: f32 = 0.0;
                for (0..seq_len_k) |k_idx| {
                    const val = try tensor.get(&[_]usize{ batch_idx, head_idx, q_idx, k_idx });
                    const exp_val = @exp(val - max_val);
                    try result.set(&[_]usize{ batch_idx, head_idx, q_idx, k_idx }, exp_val);
                    exp_sum += exp_val;
                }

                // Normalize
                for (0..seq_len_k) |k_idx| {
                    const exp_val = try result.get(&[_]usize{ batch_idx, head_idx, q_idx, k_idx });
                    try result.set(&[_]usize{ batch_idx, head_idx, q_idx, k_idx }, exp_val / exp_sum);
                }
            }
        }
    }

    return result;
}

// ============================================================================
// COMPREHENSIVE TESTS
// ============================================================================

test "Multi-head attention initialization" {
    const allocator = testing.allocator;

    var mha = try MultiHeadAttention.init(allocator, 64, 8); // 8 heads, 64 dim
    defer mha.deinit();

    try testing.expectEqual(@as(usize, 8), mha.num_heads);
    try testing.expectEqual(@as(usize, 64), mha.d_model);
    try testing.expectEqual(@as(usize, 8), mha.d_k); // 64/8 = 8

    // Check that weight matrices are initialized
    try testing.expectEqual(@as(usize, 64 * 64), mha.w_q.size);
    try testing.expectEqual(@as(usize, 64 * 64), mha.w_k.size);
    try testing.expectEqual(@as(usize, 64 * 64), mha.w_v.size);
    try testing.expectEqual(@as(usize, 64 * 64), mha.w_o.size);
}

test "Scaled dot-product attention shapes" {
    const allocator = testing.allocator;

    // Create dummy Q, K, V tensors [batch=1, heads=2, seq_len=4, d_k=8]
    var Q = try Tensor(f32).init(allocator, &[_]usize{ 1, 2, 4, 8 });
    defer Q.deinit();
    var K = try Tensor(f32).init(allocator, &[_]usize{ 1, 2, 4, 8 });
    defer K.deinit();
    var V = try Tensor(f32).init(allocator, &[_]usize{ 1, 2, 4, 8 });
    defer V.deinit();

    Q.fill(0.1);
    K.fill(0.2);
    V.fill(0.5);

    var output = try scaledDotProductAttention(Q, K, V, null, allocator);
    defer output.deinit();

    // Output should maintain batch and head dimensions, with values transformed
    try testing.expectEqual(@as(usize, 1), output.shape[0]); // batch
    try testing.expectEqual(@as(usize, 2), output.shape[1]); // heads
    try testing.expectEqual(@as(usize, 4), output.shape[2]); // seq_len
    try testing.expectEqual(@as(usize, 8), output.shape[3]); // d_v
}

test "Causal mask creation" {
    const allocator = testing.allocator;

    var mask = try createCausalMask(allocator, 4);
    defer mask.deinit();

    try testing.expectEqual(@as(usize, 4), mask.shape[0]);
    try testing.expectEqual(@as(usize, 4), mask.shape[1]);

    // Check causal pattern
    try testing.expectEqual(@as(f32, 0.0), try mask.get(&[_]usize{ 0, 0 })); // Can see self
    try testing.expect((try mask.get(&[_]usize{ 0, 1 })) == -math.inf(f32)); // Can't see future
    try testing.expectEqual(@as(f32, 0.0), try mask.get(&[_]usize{ 1, 0 })); // Can see past
    try testing.expectEqual(@as(f32, 0.0), try mask.get(&[_]usize{ 1, 1 })); // Can see self
}

test "Tensor reshaping for multi-head attention" {
    const allocator = testing.allocator;

    // Test reshaping from [batch=1, seq=3, d_model=6] to [batch=1, heads=2, seq=3, d_k=3]
    var input = try Tensor(f32).init(allocator, &[_]usize{ 1, 3, 6 });
    defer input.deinit();

    // Fill with sequential values for easy verification
    for (0..input.size) |i| {
        input.data[i] = @as(f32, @floatFromInt(i));
    }

    var reshaped = try reshapeForHeads(input, 1, 3, 2, 3, allocator);
    defer reshaped.deinit();

    try testing.expectEqual(@as(usize, 1), reshaped.shape[0]); // batch
    try testing.expectEqual(@as(usize, 2), reshaped.shape[1]); // heads
    try testing.expectEqual(@as(usize, 3), reshaped.shape[2]); // seq_len
    try testing.expectEqual(@as(usize, 3), reshaped.shape[3]); // d_k

    // Verify data was reshaped correctly
    try testing.expectEqual(@as(f32, 0.0), try reshaped.get(&[_]usize{ 0, 0, 0, 0 }));
    try testing.expectEqual(@as(f32, 3.0), try reshaped.get(&[_]usize{ 0, 1, 0, 0 }));
}

test "Softmax numerical stability" {
    const allocator = testing.allocator;

    // Create tensor with large values to test numerical stability
    var input = try Tensor(f32).init(allocator, &[_]usize{ 1, 1, 2, 3 });
    defer input.deinit();

    // Set some large values
    try input.set(&[_]usize{ 0, 0, 0, 0 }, 100.0);
    try input.set(&[_]usize{ 0, 0, 0, 1 }, 101.0);
    try input.set(&[_]usize{ 0, 0, 0, 2 }, 99.0);

    var output = try softmaxLastDim(input, allocator);
    defer output.deinit();

    // Check that probabilities sum to 1
    var sum: f32 = 0.0;
    for (0..3) |i| {
        const val = try output.get(&[_]usize{ 0, 0, 0, i });
        sum += val;
        try testing.expect(val >= 0.0 and val <= 1.0); // Valid probability
    }

    try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-6);
}

test "RoPE rotation properties" {
    const allocator = testing.allocator;

    // Test that RoPE preserves magnitude
    var input = try Tensor(f32).init(allocator, &[_]usize{ 1, 1, 2, 4 });
    defer input.deinit();

    // Set known values
    try input.set(&[_]usize{ 0, 0, 0, 0 }, 1.0);
    try input.set(&[_]usize{ 0, 0, 0, 1 }, 0.0);
    try input.set(&[_]usize{ 0, 0, 0, 2 }, 3.0);
    try input.set(&[_]usize{ 0, 0, 0, 3 }, 4.0);

    var rotated = try applyRoPE(input, 2, allocator);
    defer rotated.deinit();

    // Check that magnitude is preserved for first pair
    const orig_mag_1 = 1.0 * 1.0 + 0.0 * 0.0;
    const rot_x1 = try rotated.get(&[_]usize{ 0, 0, 0, 0 });
    const rot_y1 = try rotated.get(&[_]usize{ 0, 0, 0, 1 });
    const rot_mag_1 = rot_x1 * rot_x1 + rot_y1 * rot_y1;

    try testing.expectApproxEqAbs(orig_mag_1, rot_mag_1, 1e-6);

    // Second pair should also preserve magnitude
    const orig_mag_2 = 3.0 * 3.0 + 4.0 * 4.0;
    const rot_x2 = try rotated.get(&[_]usize{ 0, 0, 0, 2 });
    const rot_y2 = try rotated.get(&[_]usize{ 0, 0, 0, 3 });
    const rot_mag_2 = rot_x2 * rot_x2 + rot_y2 * rot_y2;

    try testing.expectApproxEqAbs(orig_mag_2, rot_mag_2, 1e-6);
}