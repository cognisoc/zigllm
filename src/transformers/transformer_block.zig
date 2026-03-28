//! Transformer Components: Complete Transformer Blocks
//!
//! This module implements complete transformer blocks that combine attention
//! and feed-forward layers with normalization and residual connections.
//!
//! ## Educational Objectives
//! - Understand how transformer components combine into complete blocks
//! - Learn the importance of residual connections and normalization placement
//! - Implement both encoder and decoder architectures
//! - Connect block design to training stability and model performance
//!
//! ## Transformer Context
//! Transformer blocks are the fundamental building units:
//! - **Encoder blocks**: Self-attention + FFN (BERT, Vision Transformer)
//! - **Decoder blocks**: Causal self-attention + FFN (GPT, LLaMA)
//! - **Encoder-decoder**: Cross-attention bridges encoder and decoder
//! - **Layer stacking**: Multiple blocks create deep transformer networks

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

// Import our layers
const Tensor = @import("../foundation/tensor.zig").Tensor;
const TensorError = @import("../foundation/tensor.zig").TensorError;
const MultiHeadAttention = @import("attention.zig").MultiHeadAttention;
const FeedForward = @import("feed_forward.zig").FeedForward;
const FFNType = @import("feed_forward.zig").FFNType;
const normalization = @import("../neural_primitives/normalization.zig");

/// Transformer block architectures
pub const TransformerBlockType = enum {
    Encoder,        // Self-attention + FFN (BERT style)
    Decoder,        // Causal self-attention + FFN (GPT style)
    EncoderDecoder, // Self-attention + Cross-attention + FFN (T5 style)
};

/// Normalization placement in transformer blocks
pub const NormPlacement = enum {
    PreNorm,    // Modern: Norm -> Attention -> Add (better gradient flow)
    PostNorm,   // Original: Attention -> Add -> Norm (original paper)
};

/// Complete Transformer Block
///
/// ## Mathematical Definition
///
/// **Pre-Norm Encoder Block (Modern):**
/// ```
/// x₁ = x + Attention(LayerNorm(x))
/// x₂ = x₁ + FFN(LayerNorm(x₁))
/// ```
///
/// **Post-Norm Encoder Block (Original):**
/// ```
/// x₁ = LayerNorm(x + Attention(x))
/// x₂ = LayerNorm(x₁ + FFN(x₁))
/// ```
///
/// ## Educational Note: Why Residual Connections?
/// Residual connections (x + f(x)) are crucial for deep networks:
///
/// 1. **Gradient Flow**: Enable gradients to flow directly to earlier layers
/// 2. **Training Stability**: Prevent vanishing gradients in deep networks
/// 3. **Identity Mapping**: Allow layers to learn identity if needed
/// 4. **Convergence**: Faster and more stable training
///
/// ## Pre-Norm vs Post-Norm
/// - **Pre-Norm**: Better gradient flow, used in modern models
/// - **Post-Norm**: Original design, can be unstable for very deep networks
pub const TransformerBlock = struct {
    /// Block architecture type
    block_type: TransformerBlockType,

    /// Normalization placement
    norm_placement: NormPlacement,

    /// Model dimension
    d_model: usize,

    /// Self-attention layer
    self_attention: MultiHeadAttention,

    /// Cross-attention layer (only for encoder-decoder)
    cross_attention: ?MultiHeadAttention,

    /// Feed-forward network
    ffn: FeedForward,

    /// Layer normalization for attention
    norm1: Tensor(f32), // Scale parameters
    norm1_bias: Tensor(f32), // Bias parameters

    /// Layer normalization for FFN
    norm2: Tensor(f32),
    norm2_bias: Tensor(f32),

    /// Layer normalization for cross-attention (encoder-decoder only)
    norm3: ?Tensor(f32),
    norm3_bias: ?Tensor(f32),

    /// Memory allocator
    allocator: Allocator,

    /// Initialize transformer block
    ///
    /// ## Educational Note: Parameter Count
    /// A typical transformer block contains:
    /// - **Attention**: 4 * d_model² parameters (Q, K, V, O projections)
    /// - **FFN**: 2 * d_model * d_ff parameters (typically d_ff = 4 * d_model)
    /// - **Normalization**: 2 * d_model parameters per norm layer
    ///
    /// Total: ~8 * d_model² parameters per block (attention dominates for large d_model)
    pub fn init(allocator: Allocator, d_model: usize, num_heads: usize, d_ff: usize,
               block_type: TransformerBlockType, norm_placement: NormPlacement, ffn_type: FFNType) !TransformerBlock {

        // Initialize attention layers
        const self_attention = try MultiHeadAttention.init(allocator, d_model, num_heads);
        var cross_attention: ?MultiHeadAttention = null;

        if (block_type == .EncoderDecoder) {
            cross_attention = try MultiHeadAttention.init(allocator, d_model, num_heads);
        }

        // Initialize feed-forward network
        const ffn = try FeedForward.init(allocator, d_model, d_ff, ffn_type);

        // Initialize normalization parameters
        var norm1 = try Tensor(f32).init(allocator, &[_]usize{d_model});
        var norm1_bias = try Tensor(f32).init(allocator, &[_]usize{d_model});
        var norm2 = try Tensor(f32).init(allocator, &[_]usize{d_model});
        var norm2_bias = try Tensor(f32).init(allocator, &[_]usize{d_model});

        // Initialize to identity (scale=1, bias=0)
        norm1.fill(1.0);
        norm1_bias.fill(0.0);
        norm2.fill(1.0);
        norm2_bias.fill(0.0);

        var norm3: ?Tensor(f32) = null;
        var norm3_bias: ?Tensor(f32) = null;

        if (block_type == .EncoderDecoder) {
            norm3 = try Tensor(f32).init(allocator, &[_]usize{d_model});
            norm3_bias = try Tensor(f32).init(allocator, &[_]usize{d_model});
            norm3.?.fill(1.0);
            norm3_bias.?.fill(0.0);
        }

        return TransformerBlock{
            .block_type = block_type,
            .norm_placement = norm_placement,
            .d_model = d_model,
            .self_attention = self_attention,
            .cross_attention = cross_attention,
            .ffn = ffn,
            .norm1 = norm1,
            .norm1_bias = norm1_bias,
            .norm2 = norm2,
            .norm2_bias = norm2_bias,
            .norm3 = norm3,
            .norm3_bias = norm3_bias,
            .allocator = allocator,
        };
    }

    /// Clean up resources
    pub fn deinit(self: *TransformerBlock) void {
        self.self_attention.deinit();
        if (self.cross_attention) |*cross_attn| {
            cross_attn.deinit();
        }
        self.ffn.deinit();
        self.norm1.deinit();
        self.norm1_bias.deinit();
        self.norm2.deinit();
        self.norm2_bias.deinit();
        if (self.norm3) |*norm3| {
            norm3.deinit();
        }
        if (self.norm3_bias) |*norm3_bias| {
            norm3_bias.deinit();
        }
    }

    /// Forward pass through transformer block
    ///
    /// ## Input Shapes
    /// - input: [batch_size, seq_len, d_model]
    /// - encoder_output: [batch_size, encoder_seq_len, d_model] (for cross-attention)
    /// - mask: Optional [batch_size, seq_len, seq_len] for causal masking
    ///
    /// ## Output Shape
    /// - output: [batch_size, seq_len, d_model]
    pub fn forward(self: *const TransformerBlock, input: Tensor(f32), encoder_output: ?Tensor(f32), mask: ?Tensor(f32)) !Tensor(f32) {
        return switch (self.block_type) {
            .Encoder => self.forwardEncoder(input, mask),
            .Decoder => self.forwardDecoder(input, mask),
            .EncoderDecoder => self.forwardEncoderDecoder(input, encoder_output, mask),
        };
    }

    /// Encoder block forward pass
    ///
    /// ## Educational Note: Encoder Architecture
    /// Encoders use bidirectional self-attention:
    /// - Each position can attend to ALL positions
    /// - No causal masking (can see "future" tokens)
    /// - Used for understanding tasks (BERT, sentence classification)
    fn forwardEncoder(self: *const TransformerBlock, input: Tensor(f32), mask: ?Tensor(f32)) !Tensor(f32) {
        return switch (self.norm_placement) {
            .PreNorm => self.forwardEncoderPreNorm(input, mask),
            .PostNorm => self.forwardEncoderPostNorm(input, mask),
        };
    }

    /// Pre-norm encoder: Norm -> Attention -> Add, Norm -> FFN -> Add
    fn forwardEncoderPreNorm(self: *const TransformerBlock, input: Tensor(f32), mask: ?Tensor(f32)) !Tensor(f32) {
        // Self-attention path
        var norm1_out = try normalization.layerNorm(f32, input, self.norm1, self.norm1_bias, self.allocator);
        defer norm1_out.deinit();

        var attn_out = try self.self_attention.forward(norm1_out, norm1_out, norm1_out, mask);
        defer attn_out.deinit();

        var residual1 = try input.add(attn_out, self.allocator);
        defer residual1.deinit();

        // Feed-forward path
        var norm2_out = try normalization.layerNorm(f32, residual1, self.norm2, self.norm2_bias, self.allocator);
        defer norm2_out.deinit();

        var ffn_out = try self.ffn.forward(norm2_out);
        defer ffn_out.deinit();

        return try residual1.add(ffn_out, self.allocator);
    }

    /// Post-norm encoder: Attention -> Add -> Norm, FFN -> Add -> Norm
    fn forwardEncoderPostNorm(self: *const TransformerBlock, input: Tensor(f32), mask: ?Tensor(f32)) !Tensor(f32) {
        // Self-attention path
        var attn_out = try self.self_attention.forward(input, input, input, mask);
        defer attn_out.deinit();

        var residual1 = try input.add(attn_out, self.allocator);
        defer residual1.deinit();

        var norm1_out = try normalization.layerNorm(f32, residual1, self.norm1, self.norm1_bias, self.allocator);
        defer norm1_out.deinit();

        // Feed-forward path
        var ffn_out = try self.ffn.forward(norm1_out);
        defer ffn_out.deinit();

        var residual2 = try norm1_out.add(ffn_out, self.allocator);
        defer residual2.deinit();

        return try normalization.layerNorm(f32, residual2, self.norm2, self.norm2_bias, self.allocator);
    }

    /// Decoder block forward pass
    ///
    /// ## Educational Note: Decoder Architecture
    /// Decoders use causal (masked) self-attention:
    /// - Each position can only attend to current and previous positions
    /// - Causal masking prevents "looking into the future"
    /// - Used for generation tasks (GPT, language modeling)
    fn forwardDecoder(self: *const TransformerBlock, input: Tensor(f32), mask: ?Tensor(f32)) !Tensor(f32) {
        // Decoders typically use causal masking
        // For simplicity, we'll use the same structure as encoder
        // In practice, you'd ensure the mask is causal
        return self.forwardEncoder(input, mask);
    }

    /// Encoder-decoder block forward pass
    ///
    /// ## Educational Note: Encoder-Decoder Architecture
    /// Encoder-decoder blocks have both self and cross-attention:
    /// - **Self-attention**: Process decoder context
    /// - **Cross-attention**: Attend to encoder representations
    /// - **FFN**: Final processing
    /// Used in seq2seq tasks (T5, machine translation)
    fn forwardEncoderDecoder(self: *const TransformerBlock, input: Tensor(f32), encoder_output: ?Tensor(f32), mask: ?Tensor(f32)) !Tensor(f32) {
        if (encoder_output == null or self.cross_attention == null) {
            return TensorError.IncompatibleShapes;
        }

        // Self-attention on decoder input
        var norm1_out = try normalization.layerNorm(f32, input, self.norm1, self.norm1_bias, self.allocator);
        defer norm1_out.deinit();

        var self_attn_out = try self.self_attention.forward(norm1_out, norm1_out, norm1_out, mask);
        defer self_attn_out.deinit();

        var residual1 = try input.add(self_attn_out, self.allocator);
        defer residual1.deinit();

        // Cross-attention: query from decoder, key/value from encoder
        if (self.norm3 == null or self.norm3_bias == null) {
            return TensorError.IncompatibleShapes;
        }

        var norm3_out = try normalization.layerNorm(f32, residual1, self.norm3.?, self.norm3_bias.?, self.allocator);
        defer norm3_out.deinit();

        var cross_attn_out = try self.cross_attention.?.forward(norm3_out, encoder_output.?, encoder_output.?, null);
        defer cross_attn_out.deinit();

        var residual2 = try residual1.add(cross_attn_out, self.allocator);
        defer residual2.deinit();

        // Feed-forward
        var norm2_out = try normalization.layerNorm(f32, residual2, self.norm2, self.norm2_bias, self.allocator);
        defer norm2_out.deinit();

        var ffn_out = try self.ffn.forward(norm2_out);
        defer ffn_out.deinit();

        return try residual2.add(ffn_out, self.allocator);
    }
};

/// Complete Transformer Model
///
/// ## Educational Note: Scaling Transformers
/// Complete transformers stack multiple blocks:
/// - **Depth**: More blocks = more representation power
/// - **Width**: Larger d_model = more capacity per position
/// - **Heads**: More attention heads = more parallel processing
/// - **FFN ratio**: Larger d_ff/d_model = more non-linear capacity
///
/// Modern large models use:
/// - GPT-3: 96 layers, 12,288 dimensions, 96 heads
/// - LLaMA-65B: 80 layers, 8,192 dimensions, 64 heads
pub const Transformer = struct {
    /// Number of transformer blocks
    num_layers: usize,

    /// Model dimension
    d_model: usize,

    /// Individual transformer blocks
    blocks: []TransformerBlock,

    /// Final layer normalization
    final_norm: Tensor(f32),
    final_norm_bias: Tensor(f32),

    allocator: Allocator,

    pub fn init(allocator: Allocator, num_layers: usize, d_model: usize, num_heads: usize,
               d_ff: usize, block_type: TransformerBlockType, norm_placement: NormPlacement, ffn_type: FFNType) !Transformer {

        // Initialize transformer blocks
        var blocks = try allocator.alloc(TransformerBlock, num_layers);
        for (0..num_layers) |i| {
            blocks[i] = try TransformerBlock.init(allocator, d_model, num_heads, d_ff, block_type, norm_placement, ffn_type);
        }

        // Final normalization
        var final_norm = try Tensor(f32).init(allocator, &[_]usize{d_model});
        var final_norm_bias = try Tensor(f32).init(allocator, &[_]usize{d_model});
        final_norm.fill(1.0);
        final_norm_bias.fill(0.0);

        return Transformer{
            .num_layers = num_layers,
            .d_model = d_model,
            .blocks = blocks,
            .final_norm = final_norm,
            .final_norm_bias = final_norm_bias,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Transformer) void {
        for (0..self.num_layers) |i| {
            self.blocks[i].deinit();
        }
        self.allocator.free(self.blocks);
        self.final_norm.deinit();
        self.final_norm_bias.deinit();
    }

    /// Forward pass through complete transformer
    pub fn forward(self: *const Transformer, input: Tensor(f32), encoder_output: ?Tensor(f32), mask: ?Tensor(f32)) !Tensor(f32) {
        var current_input = input;
        var should_free_current = false;

        // Pass through all transformer blocks
        for (0..self.num_layers) |layer_idx| {
            const block_output = try self.blocks[layer_idx].forward(current_input, encoder_output, mask);

            // Free intermediate results (except original input)
            if (should_free_current) {
                current_input.deinit();
            }

            current_input = block_output;
            should_free_current = true;
        }

        // Final layer normalization
        const final_output = try normalization.layerNorm(f32, current_input, self.final_norm, self.final_norm_bias, self.allocator);

        // Clean up final intermediate
        if (should_free_current) {
            current_input.deinit();
        }

        return final_output;
    }
};

// ============================================================================
// COMPREHENSIVE TESTS
// ============================================================================

test "Transformer block initialization" {
    const allocator = testing.allocator;

    var block = try TransformerBlock.init(allocator, 64, 8, 256, .Encoder, .PreNorm, .Standard);
    defer block.deinit();

    try testing.expectEqual(@as(usize, 64), block.d_model);
    try testing.expectEqual(TransformerBlockType.Encoder, block.block_type);
    try testing.expectEqual(NormPlacement.PreNorm, block.norm_placement);

    // Encoder should not have cross-attention
    try testing.expect(block.cross_attention == null);
    try testing.expect(block.norm3 == null);
    try testing.expect(block.norm3_bias == null);
}

test "Encoder-decoder block initialization" {
    const allocator = testing.allocator;

    var block = try TransformerBlock.init(allocator, 32, 4, 128, .EncoderDecoder, .PreNorm, .SwiGLU);
    defer block.deinit();

    try testing.expectEqual(TransformerBlockType.EncoderDecoder, block.block_type);

    // Encoder-decoder should have cross-attention
    try testing.expect(block.cross_attention != null);
    try testing.expect(block.norm3 != null);
    try testing.expect(block.norm3_bias != null);
}

test "Transformer block forward pass shapes" {
    const allocator = testing.allocator;

    var block = try TransformerBlock.init(allocator, 16, 4, 64, .Encoder, .PreNorm, .Standard);
    defer block.deinit();

    // Test input: [batch=2, seq_len=5, d_model=16]
    var input = try Tensor(f32).init(allocator, &[_]usize{ 2, 5, 16 });
    defer input.deinit();
    input.fill(0.1);

    var output = try block.forward(input, null, null);
    defer output.deinit();

    // Output should preserve input shape
    try testing.expectEqualSlices(usize, input.shape, output.shape);
}

test "Complete transformer initialization" {
    const allocator = testing.allocator;

    var transformer = try Transformer.init(allocator, 6, 64, 8, 256, .Encoder, .PreNorm, .GELU);
    defer transformer.deinit();

    try testing.expectEqual(@as(usize, 6), transformer.num_layers);
    try testing.expectEqual(@as(usize, 64), transformer.d_model);
    try testing.expectEqual(@as(usize, 6), transformer.blocks.len);
}

test "Transformer parameter counting" {
    const allocator = testing.allocator;

    // Small transformer for parameter counting
    const num_layers: usize = 2;
    const d_model: usize = 8;
    const num_heads: usize = 2;
    const d_ff: usize = 32;

    var transformer = try Transformer.init(allocator, num_layers, d_model, num_heads, d_ff, .Encoder, .PreNorm, .Standard);
    defer transformer.deinit();

    // Each block should have consistent parameters
    for (transformer.blocks) |block| {
        try testing.expectEqual(d_model, block.d_model);
        try testing.expectEqual(num_heads, block.self_attention.num_heads);
        try testing.expectEqual(d_ff, block.ffn.d_ff);
    }
}

test "Residual connection preservation" {
    const allocator = testing.allocator;

    var block = try TransformerBlock.init(allocator, 8, 2, 32, .Encoder, .PreNorm, .Standard);
    defer block.deinit();

    var input = try Tensor(f32).init(allocator, &[_]usize{ 1, 3, 8 });
    defer input.deinit();

    // Fill with known pattern
    for (0..input.size) |i| {
        input.data[i] = @as(f32, @floatFromInt(i % 8)) / 10.0;
    }

    var output = try block.forward(input, null, null);
    defer output.deinit();

    // Output should be different from input (processing occurred)
    var different = false;
    for (0..input.size) |i| {
        if (@abs(input.data[i] - output.data[i]) > 0.01) {
            different = true;
            break;
        }
    }
    try testing.expect(different);

    // But output should still be reasonable (not NaN/inf)
    for (0..output.size) |i| {
        try testing.expect(std.math.isFinite(output.data[i]));
    }
}