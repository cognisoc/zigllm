//! Neural Primitives: Embedding Operations
//!
//! This module implements embedding layers essential for transformers,
//! converting discrete tokens into continuous vector representations.
//!
//! ## Educational Objectives
//! - Understand how embeddings bridge discrete and continuous spaces
//! - Learn different types of embeddings used in transformers
//! - Implement positional encoding for sequence awareness
//! - Connect embedding dimensions to transformer architecture choices
//!
//! ## Transformer Context
//! Embeddings are the foundation of transformer models:
//! - **Token Embeddings**: Convert vocabulary tokens to vectors
//! - **Positional Embeddings**: Add position information to sequences
//! - **Segment Embeddings**: Distinguish between different input segments
//! - **Learned vs Fixed**: Trade-offs between trainable and fixed embeddings

const std = @import("std");
const math = std.math;
const testing = std.testing;
const Allocator = std.mem.Allocator;
const Tensor = @import("../foundation/tensor.zig").Tensor;
const TensorError = @import("../foundation/tensor.zig").TensorError;

/// Embedding types used in transformer architectures
pub const EmbeddingType = enum {
    Token,       // Vocabulary token embeddings
    Position,    // Positional embeddings
    Segment,     // Segment/sentence embeddings
    Rotary,      // Rotary Position Embeddings (RoPE)
};

/// Positional encoding types
pub const PositionalEncodingType = enum {
    Sinusoidal,  // Fixed sinusoidal encodings (original Transformer)
    Learned,     // Learned positional embeddings
    Rotary,      // Rotary Position Embeddings (RoPE)
    Alibi,       // Attention with Linear Biases (ALiBi)
};

/// Token Embedding Layer
///
/// ## Mathematical Definition
/// For vocabulary size V and embedding dimension d:
/// ```
/// E ∈ R^(V×d)
/// embedding(token_id) = E[token_id, :]
/// ```
///
/// ## Educational Note: Why Embeddings?
/// Embeddings solve several fundamental problems:
///
/// 1. **Discrete to Continuous**: Convert discrete tokens to continuous vectors
/// 2. **Semantic Similarity**: Similar tokens have similar embeddings
/// 3. **Dimensionality**: Map large vocabularies to manageable dimensions
/// 4. **Trainable**: Learn optimal representations during training
///
/// ## Transformer Usage
/// Token embeddings are typically:
/// - **Shared**: Same embedding matrix for input and output
/// - **Scaled**: Often scaled by √d_model for gradient stability
/// - **Normalized**: Sometimes layer-normalized after embedding
///
/// ## Memory Considerations
/// For large vocabularies (e.g., 50K tokens × 768 dims = 38M parameters),
/// embeddings can be a significant portion of model parameters.
pub const TokenEmbedding = struct {
    /// Embedding weight matrix [vocab_size × embedding_dim]
    weights: Tensor(f32),

    /// Vocabulary size
    vocab_size: usize,

    /// Embedding dimension
    embedding_dim: usize,

    /// Memory allocator
    allocator: Allocator,

    /// Initialize token embedding layer
    ///
    /// ## Educational Note: Initialization Strategy
    /// Proper initialization is crucial for transformer training:
    /// - **Xavier/Glorot**: Normal distribution with std = √(2/(fan_in + fan_out))
    /// - **He**: Normal distribution with std = √(2/fan_in)
    /// - **Transformer-specific**: Often use truncated normal with std = 0.02
    ///
    /// We use Xavier initialization for balanced gradients.
    pub fn init(allocator: Allocator, vocab_size: usize, embedding_dim: usize) !TokenEmbedding {
        var weights = try Tensor(f32).init(allocator, &[_]usize{ vocab_size, embedding_dim });

        // Xavier initialization: std = sqrt(2 / (vocab_size + embedding_dim))
        const xavier_std = @sqrt(2.0 / @as(f32, @floatFromInt(vocab_size + embedding_dim)));

        // Simple pseudo-random initialization for demonstration
        // In practice, you'd use a proper random number generator
        var seed: u32 = 12345;
        for (0..weights.size) |i| {
            seed = seed *% 1664525 +% 1013904223; // Linear congruential generator
            const rand_f32 = @as(f32, @floatFromInt(seed)) / @as(f32, @floatFromInt(std.math.maxInt(u32)));
            const gaussian_approx = (rand_f32 - 0.5) * 2.0; // Rough approximation
            weights.data[i] = gaussian_approx * xavier_std;
        }

        return TokenEmbedding{
            .weights = weights,
            .vocab_size = vocab_size,
            .embedding_dim = embedding_dim,
            .allocator = allocator,
        };
    }

    /// Clean up resources
    pub fn deinit(self: *TokenEmbedding) void {
        self.weights.deinit();
    }

    /// Forward pass: convert token IDs to embeddings
    ///
    /// ## Input Shape
    /// - token_ids: [batch_size, sequence_length] or [sequence_length]
    ///
    /// ## Output Shape
    /// - embeddings: [batch_size, sequence_length, embedding_dim] or [sequence_length, embedding_dim]
    ///
    /// ## Educational Note: Embedding Lookup
    /// This operation is essentially:
    /// 1. **One-hot encoding**: Convert token ID to one-hot vector
    /// 2. **Matrix multiplication**: one_hot @ embedding_matrix
    /// 3. **Optimization**: Direct indexing is much more efficient
    pub fn forward(self: *const TokenEmbedding, token_ids: []const u32) !Tensor(f32) {
        const batch_size = token_ids.len;

        var output = try Tensor(f32).init(self.allocator, &[_]usize{ batch_size, self.embedding_dim });

        for (token_ids, 0..) |token_id, batch_idx| {
            if (token_id >= self.vocab_size) return TensorError.InvalidIndex;

            // Copy embedding vector for this token
            for (0..self.embedding_dim) |dim_idx| {
                const embedding_value = try self.weights.get(&[_]usize{ token_id, dim_idx });
                try output.set(&[_]usize{ batch_idx, dim_idx }, embedding_value);
            }
        }

        return output;
    }
};

/// Sinusoidal Positional Encoding
///
/// ## Mathematical Definition
/// For position pos and dimension i:
/// ```
/// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
/// ```
///
/// ## Educational Note: Why Sinusoidal Encoding?
/// The sinusoidal approach has several advantages:
///
/// 1. **Fixed Function**: No parameters to learn, generalizes to any length
/// 2. **Unique Encoding**: Each position has a unique encoding
/// 3. **Relative Position**: PE(pos+k) can be expressed as linear combination of PE(pos)
/// 4. **Frequency Spectrum**: Different dimensions encode different frequencies
///
/// ## Intuitive Understanding
/// - **Low dimensions**: High-frequency patterns (change quickly with position)
/// - **High dimensions**: Low-frequency patterns (change slowly with position)
/// - **Combined**: Create unique "fingerprint" for each position
///
/// ## Transformer Usage
/// Original Transformer paper used fixed sinusoidal encodings,
/// but many modern models use learned positional embeddings.
pub fn sinusoidalPositionalEncoding(allocator: Allocator, max_seq_len: usize, d_model: usize) !Tensor(f32) {
    var pe = try Tensor(f32).init(allocator, &[_]usize{ max_seq_len, d_model });

    for (0..max_seq_len) |pos| {
        for (0..d_model / 2) |i| {
            const pos_f = @as(f32, @floatFromInt(pos));
            const i_f = @as(f32, @floatFromInt(i));
            const d_model_f = @as(f32, @floatFromInt(d_model));

            // Calculate the frequency
            const freq = pos_f / math.pow(f32, 10000.0, 2.0 * i_f / d_model_f);

            // Apply sin to even indices, cos to odd indices
            try pe.set(&[_]usize{ pos, 2 * i }, @sin(freq));

            if (2 * i + 1 < d_model) {
                try pe.set(&[_]usize{ pos, 2 * i + 1 }, @cos(freq));
            }
        }
    }

    return pe;
}

/// Rotary Position Embeddings (RoPE)
///
/// ## Mathematical Definition
/// RoPE applies rotation matrices to query and key vectors:
/// ```
/// q_m = R_m * q    (rotate query by position m)
/// k_n = R_n * k    (rotate key by position n)
/// ```
/// Where R_θ is a rotation matrix with angle θ = pos / 10000^(2i/d)
///
/// ## Educational Note: Why RoPE?
/// RoPE offers several advantages over additive positional encodings:
///
/// 1. **Relative Position**: Attention naturally encodes relative positions
/// 2. **No Length Limit**: Works for any sequence length without retraining
/// 3. **Multiplicative**: Preserves embedding information better than addition
/// 4. **Efficiency**: Can be computed on-the-fly during attention
///
/// ## Implementation Note
/// This is a simplified 2D rotation. Full RoPE applies rotations to
/// pairs of dimensions throughout the embedding vector.
pub fn rotaryPositionalEmbedding(allocator: Allocator, seq_len: usize, d_model: usize, embeddings: Tensor(f32)) !Tensor(f32) {
    if (embeddings.ndim() != 2) return TensorError.IncompatibleShapes;
    if (embeddings.shape[0] != seq_len or embeddings.shape[1] != d_model) return TensorError.IncompatibleShapes;

    var result = try Tensor(f32).init(allocator, embeddings.shape);

    for (0..seq_len) |pos| {
        const pos_f = @as(f32, @floatFromInt(pos));

        for (0..d_model / 2) |i| {
            const i_f = @as(f32, @floatFromInt(i));
            const d_model_f = @as(f32, @floatFromInt(d_model));

            // Calculate rotation angle
            const theta = pos_f / math.pow(f32, 10000.0, 2.0 * i_f / d_model_f);

            // Get the two dimensions to rotate
            const x = try embeddings.get(&[_]usize{ pos, 2 * i });
            const y = if (2 * i + 1 < d_model) try embeddings.get(&[_]usize{ pos, 2 * i + 1 }) else 0.0;

            // Apply 2D rotation
            const cos_theta = @cos(theta);
            const sin_theta = @sin(theta);

            const rotated_x = x * cos_theta - y * sin_theta;
            const rotated_y = x * sin_theta + y * cos_theta;

            try result.set(&[_]usize{ pos, 2 * i }, rotated_x);
            if (2 * i + 1 < d_model) {
                try result.set(&[_]usize{ pos, 2 * i + 1 }, rotated_y);
            }
        }
    }

    return result;
}

/// Add positional encodings to token embeddings
///
/// ## Educational Note: Embedding Combination
/// There are several ways to combine token and positional embeddings:
/// 1. **Addition**: PE + Token (most common, used in original Transformer)
/// 2. **Concatenation**: [PE, Token] (doubles embedding dimension)
/// 3. **Learned Combination**: Trainable mixing weights
///
/// Addition is preferred because:
/// - Maintains embedding dimension
/// - Allows relative position relationships
/// - Simpler and more parameter-efficient
pub fn addPositionalEncoding(token_embeddings: Tensor(f32), positional_encodings: Tensor(f32), allocator: Allocator) !Tensor(f32) {
    if (!std.mem.eql(usize, token_embeddings.shape, positional_encodings.shape)) {
        return TensorError.IncompatibleShapes;
    }

    return try token_embeddings.add(positional_encodings, allocator);
}

/// Segment/Sentence Embeddings
///
/// ## Educational Note: Multi-Segment Input
/// Some transformer tasks require processing multiple text segments:
/// - **BERT**: [CLS] sentence A [SEP] sentence B [SEP]
/// - **Question Answering**: question [SEP] context
/// - **Classification**: text A [SEP] text B
///
/// Segment embeddings help the model distinguish between segments.
pub const SegmentEmbedding = struct {
    weights: Tensor(f32),
    num_segments: usize,
    embedding_dim: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, num_segments: usize, embedding_dim: usize) !SegmentEmbedding {
        var weights = try Tensor(f32).init(allocator, &[_]usize{ num_segments, embedding_dim });

        // Simple initialization
        var seed: u32 = 54321;
        const init_std = 0.02; // Standard transformer initialization

        for (0..weights.size) |i| {
            seed = seed *% 1664525 +% 1013904223;
            const rand_f32 = @as(f32, @floatFromInt(seed)) / @as(f32, @floatFromInt(std.math.maxInt(u32)));
            weights.data[i] = (rand_f32 - 0.5) * 2.0 * init_std;
        }

        return SegmentEmbedding{
            .weights = weights,
            .num_segments = num_segments,
            .embedding_dim = embedding_dim,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SegmentEmbedding) void {
        self.weights.deinit();
    }

    pub fn forward(self: *const SegmentEmbedding, segment_ids: []const u32) !Tensor(f32) {
        const batch_size = segment_ids.len;
        var output = try Tensor(f32).init(self.allocator, &[_]usize{ batch_size, self.embedding_dim });

        for (segment_ids, 0..) |segment_id, batch_idx| {
            if (segment_id >= self.num_segments) return TensorError.InvalidIndex;

            for (0..self.embedding_dim) |dim_idx| {
                const embedding_value = try self.weights.get(&[_]usize{ segment_id, dim_idx });
                try output.set(&[_]usize{ batch_idx, dim_idx }, embedding_value);
            }
        }

        return output;
    }
};

/// Complete embedding layer combining token, positional, and segment embeddings
///
/// ## Educational Note: BERT-style Embeddings
/// This demonstrates the complete embedding approach used in BERT:
/// ```
/// final_embedding = token_embedding + positional_embedding + segment_embedding
/// ```
/// Each component contributes different information:
/// - **Token**: Semantic meaning of words
/// - **Position**: Sequence order information
/// - **Segment**: Multi-sentence structure
pub fn completeEmbedding(token_emb: Tensor(f32), pos_emb: Tensor(f32), seg_emb: ?Tensor(f32), allocator: Allocator) !Tensor(f32) {
    // Add token and positional embeddings
    var result = try token_emb.add(pos_emb, allocator);

    // Add segment embeddings if provided
    if (seg_emb) |segment_embeddings| {
        var temp = result;
        result = try result.add(segment_embeddings, allocator);
        temp.deinit(); // Clean up intermediate result
    }

    return result;
}

// ============================================================================
// COMPREHENSIVE TESTS
// ============================================================================

test "Token embedding forward pass" {
    const allocator = testing.allocator;

    var token_emb = try TokenEmbedding.init(allocator, 100, 64); // 100 vocab, 64 dims
    defer token_emb.deinit();

    // Test token sequence: [1, 5, 10]
    const token_ids = [_]u32{ 1, 5, 10 };
    var embeddings = try token_emb.forward(&token_ids);
    defer embeddings.deinit();

    // Check output shape
    try testing.expectEqual(@as(usize, 3), embeddings.shape[0]);  // 3 tokens
    try testing.expectEqual(@as(usize, 64), embeddings.shape[1]); // 64 dimensions

    // Verify embeddings are different for different tokens
    const emb1_0 = try embeddings.get(&[_]usize{ 0, 0 });
    const emb2_0 = try embeddings.get(&[_]usize{ 1, 0 });
    try testing.expect(emb1_0 != emb2_0);
}

test "Sinusoidal positional encoding properties" {
    const allocator = testing.allocator;

    var pe = try sinusoidalPositionalEncoding(allocator, 10, 8);
    defer pe.deinit();

    // Check shape
    try testing.expectEqual(@as(usize, 10), pe.shape[0]); // 10 positions
    try testing.expectEqual(@as(usize, 8), pe.shape[1]);  // 8 dimensions

    // Check that different positions have different encodings
    const pos0_0 = try pe.get(&[_]usize{ 0, 0 });
    const pos1_0 = try pe.get(&[_]usize{ 1, 0 });
    try testing.expect(pos0_0 != pos1_0);

    // Position 0 should have specific values
    const pos0_dim0 = try pe.get(&[_]usize{ 0, 0 }); // sin(0) = 0
    const pos0_dim1 = try pe.get(&[_]usize{ 0, 1 }); // cos(0) = 1

    try testing.expectApproxEqAbs(@as(f32, 0.0), pos0_dim0, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), pos0_dim1, 1e-6);
}

test "Rotary positional embedding rotation" {
    const allocator = testing.allocator;

    // Create simple test embedding
    var embeddings = try Tensor(f32).init(allocator, &[_]usize{ 2, 4 });
    defer embeddings.deinit();

    // Set known values
    try embeddings.set(&[_]usize{ 0, 0 }, 1.0); // Position 0, dim 0
    try embeddings.set(&[_]usize{ 0, 1 }, 0.0); // Position 0, dim 1
    try embeddings.set(&[_]usize{ 1, 0 }, 1.0); // Position 1, dim 0
    try embeddings.set(&[_]usize{ 1, 1 }, 0.0); // Position 1, dim 1

    var rotated = try rotaryPositionalEmbedding(allocator, 2, 4, embeddings);
    defer rotated.deinit();

    // Position 0 should be unchanged (rotation by 0)
    const pos0_dim0 = try rotated.get(&[_]usize{ 0, 0 });
    try testing.expectApproxEqAbs(@as(f32, 1.0), pos0_dim0, 1e-6);

    // Position 1 should be rotated
    const pos1_dim0 = try rotated.get(&[_]usize{ 1, 0 });
    const pos1_dim1 = try rotated.get(&[_]usize{ 1, 1 });

    // Should satisfy rotation property: x² + y² preserved
    const original_magnitude = 1.0 * 1.0 + 0.0 * 0.0;
    const rotated_magnitude = pos1_dim0 * pos1_dim0 + pos1_dim1 * pos1_dim1;
    try testing.expectApproxEqAbs(original_magnitude, rotated_magnitude, 1e-6);
}

test "Segment embedding functionality" {
    const allocator = testing.allocator;

    var seg_emb = try SegmentEmbedding.init(allocator, 3, 16); // 3 segments, 16 dims
    defer seg_emb.deinit();

    const segment_ids = [_]u32{ 0, 1, 0, 2 }; // Mixed segments
    var embeddings = try seg_emb.forward(&segment_ids);
    defer embeddings.deinit();

    // Check shape
    try testing.expectEqual(@as(usize, 4), embeddings.shape[0]);  // 4 positions
    try testing.expectEqual(@as(usize, 16), embeddings.shape[1]); // 16 dimensions

    // Same segment IDs should produce identical embeddings
    const seg0_pos0 = try embeddings.get(&[_]usize{ 0, 0 });
    const seg0_pos2 = try embeddings.get(&[_]usize{ 2, 0 });
    try testing.expectEqual(seg0_pos0, seg0_pos2);

    // Different segments should produce different embeddings
    const seg0_dim0 = try embeddings.get(&[_]usize{ 0, 0 });
    const seg1_dim0 = try embeddings.get(&[_]usize{ 1, 0 });
    try testing.expect(seg0_dim0 != seg1_dim0);
}

test "Complete embedding combination" {
    const allocator = testing.allocator;

    // Create component embeddings
    var token_emb = try Tensor(f32).init(allocator, &[_]usize{ 3, 4 });
    defer token_emb.deinit();
    token_emb.fill(1.0);

    var pos_emb = try Tensor(f32).init(allocator, &[_]usize{ 3, 4 });
    defer pos_emb.deinit();
    pos_emb.fill(0.5);

    var seg_emb = try Tensor(f32).init(allocator, &[_]usize{ 3, 4 });
    defer seg_emb.deinit();
    seg_emb.fill(0.25);

    // Combine embeddings
    var combined = try completeEmbedding(token_emb, pos_emb, seg_emb, allocator);
    defer combined.deinit();

    // Each element should be sum: 1.0 + 0.5 + 0.25 = 1.75
    const combined_val = try combined.get(&[_]usize{ 0, 0 });
    try testing.expectApproxEqAbs(@as(f32, 1.75), combined_val, 1e-6);
}

test "Embedding dimension consistency" {
    const allocator = testing.allocator;

    var token_emb = try TokenEmbedding.init(allocator, 50, 32);
    defer token_emb.deinit();

    const tokens = [_]u32{ 1, 2, 3, 4, 5 };
    var embeddings = try token_emb.forward(&tokens);
    defer embeddings.deinit();

    var pos_enc = try sinusoidalPositionalEncoding(allocator, 5, 32);
    defer pos_enc.deinit();

    // Should be able to add token and positional embeddings
    var combined = try addPositionalEncoding(embeddings, pos_enc, allocator);
    defer combined.deinit();

    // Shape should be preserved
    try testing.expectEqual(@as(usize, 5), combined.shape[0]);
    try testing.expectEqual(@as(usize, 32), combined.shape[1]);
}