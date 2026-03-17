//! Models: LLaMA Architecture Implementation
//!
//! This module implements the LLaMA (Large Language Model Meta AI) architecture,
//! which represents the state-of-the-art in open-source language models.
//!
//! ## Educational Objectives
//! - Understand how transformer components combine into complete language models
//! - Learn the specific architectural choices that make LLaMA effective
//! - Implement model configuration and scaling for different model sizes
//! - Connect individual components to complete text generation systems
//!
//! ## LLaMA Architecture Key Features
//! - **RMSNorm**: More efficient than LayerNorm
//! - **SwiGLU**: Gated activation in feed-forward networks
//! - **RoPE**: Rotary position embeddings for better length generalization
//! - **Pre-normalization**: Better gradient flow in deep networks
//! - **No bias terms**: Simpler implementation, similar performance

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

// Import our layers
const Tensor = @import("../foundation/tensor.zig").Tensor;
const TensorError = @import("../foundation/tensor.zig").TensorError;
const TokenEmbedding = @import("../neural_primitives/embeddings.zig").TokenEmbedding;
const MultiHeadAttention = @import("../transformers/attention.zig").MultiHeadAttention;
const FeedForward = @import("../transformers/feed_forward.zig").FeedForward;
const FFNType = @import("../transformers/feed_forward.zig").FFNType;
const TransformerBlock = @import("../transformers/transformer_block.zig").TransformerBlock;
const TransformerBlockType = @import("../transformers/transformer_block.zig").TransformerBlockType;
const NormPlacement = @import("../transformers/transformer_block.zig").NormPlacement;
const normalization = @import("../neural_primitives/normalization.zig");

/// LLaMA model configuration
///
/// ## Educational Note: Model Scaling
/// LLaMA models come in different sizes, each with specific parameter counts:
/// - LLaMA-7B: 7 billion parameters
/// - LLaMA-13B: 13 billion parameters
/// - LLaMA-30B: 30 billion parameters
/// - LLaMA-65B: 65 billion parameters
///
/// The scaling follows predictable patterns that we encode in configurations.
pub const LLaMAConfig = struct {
    /// Model dimension (embedding size)
    d_model: usize,

    /// Number of transformer layers
    num_layers: usize,

    /// Number of attention heads
    num_heads: usize,

    /// Feed-forward hidden dimension
    d_ff: usize,

    /// Vocabulary size
    vocab_size: usize,

    /// Maximum sequence length
    max_seq_len: usize,

    /// RMSNorm epsilon for numerical stability
    norm_eps: f32,

    /// Whether to use RoPE (always true for LLaMA)
    use_rope: bool,

    /// RoPE theta (base frequency)
    rope_theta: f32,

    /// Initialize configuration for different LLaMA model sizes
    ///
    /// ## Educational Note: Scaling Laws
    /// These configurations follow empirical scaling laws:
    /// - Layers scale roughly as log(parameters)
    /// - Heads typically scale as d_model / 64 or d_model / 128
    /// - FFN dimension is typically 8/3 * d_model for gated activations
    pub fn init(model_size: ModelSize) LLaMAConfig {
        return switch (model_size) {
            .LLaMA_7B => LLaMAConfig{
                .d_model = 4096,
                .num_layers = 32,
                .num_heads = 32,
                .d_ff = 11008, // ~8/3 * d_model for SwiGLU
                .vocab_size = 32000,
                .max_seq_len = 2048,
                .norm_eps = 1e-6,
                .use_rope = true,
                .rope_theta = 10000.0,
            },
            .LLaMA_13B => LLaMAConfig{
                .d_model = 5120,
                .num_layers = 40,
                .num_heads = 40,
                .d_ff = 13824,
                .vocab_size = 32000,
                .max_seq_len = 2048,
                .norm_eps = 1e-6,
                .use_rope = true,
                .rope_theta = 10000.0,
            },
            .LLaMA_30B => LLaMAConfig{
                .d_model = 6656,
                .num_layers = 60,
                .num_heads = 52,
                .d_ff = 17920,
                .vocab_size = 32000,
                .max_seq_len = 2048,
                .norm_eps = 1e-6,
                .use_rope = true,
                .rope_theta = 10000.0,
            },
            .LLaMA_65B => LLaMAConfig{
                .d_model = 8192,
                .num_layers = 80,
                .num_heads = 64,
                .d_ff = 22016,
                .vocab_size = 32000,
                .max_seq_len = 2048,
                .norm_eps = 1e-6,
                .use_rope = true,
                .rope_theta = 10000.0,
            },
            .Custom => unreachable, // Must be set manually
        };
    }

    /// Calculate total parameter count
    ///
    /// ## Educational Note: Parameter Breakdown
    /// LLaMA parameters are distributed as:
    /// - Token embeddings: vocab_size × d_model
    /// - Transformer layers: num_layers × (attention + FFN + norms)
    /// - Output layer: d_model × vocab_size (often tied with embeddings)
    /// - RMSNorm parameters: negligible compared to linear layers
    pub fn getParameterCount(self: LLaMAConfig) usize {
        // Token embeddings: vocab_size × d_model
        const embedding_params = self.vocab_size * self.d_model;

        // Each layer contains:
        // - Attention: 4 × d_model × d_model (Q, K, V, O projections)
        // - FFN: 3 × d_model × d_ff (gate, up, down for SwiGLU)
        // - RMSNorm: 2 × d_model (attention + FFN norms, negligible)
        const attention_params_per_layer = 4 * self.d_model * self.d_model;
        const ffn_params_per_layer = 3 * self.d_model * self.d_ff;
        const layer_params = attention_params_per_layer + ffn_params_per_layer;

        // Total transformer layers
        const transformer_params = self.num_layers * layer_params;

        // Output projection (often tied with input embeddings)
        const output_params = self.d_model * self.vocab_size;

        // Final RMSNorm
        const final_norm_params = self.d_model;

        return embedding_params + transformer_params + output_params + final_norm_params;
    }
};

/// Standard LLaMA model sizes
pub const ModelSize = enum {
    LLaMA_7B,
    LLaMA_13B,
    LLaMA_30B,
    LLaMA_65B,
    Custom,
};

/// Complete LLaMA model implementation
///
/// ## Architecture Overview
/// ```
/// Input Tokens
///     ↓
/// Token Embeddings (no positional - RoPE handles this)
///     ↓
/// LLaMA Transformer Layers × N
/// │  ├─ RMSNorm
/// │  ├─ Multi-Head Attention (with RoPE)
/// │  ├─ Residual Connection
/// │  ├─ RMSNorm
/// │  ├─ SwiGLU Feed-Forward
/// │  └─ Residual Connection
///     ↓
/// Final RMSNorm
///     ↓
/// Linear Output (to vocab)
///     ↓
/// Logits for Next Token
/// ```
pub const LLaMAModel = struct {
    /// Model configuration
    config: LLaMAConfig,

    /// Token embedding layer
    token_embeddings: TokenEmbedding,

    /// Transformer layers
    transformer_layers: []LLaMATransformerLayer,

    /// Final RMSNorm layer
    final_norm: Tensor(f32),

    /// Output linear layer (logits projection)
    output_projection: Tensor(f32),

    /// Memory allocator
    allocator: Allocator,

    /// Initialize LLaMA model with given configuration
    pub fn init(allocator: Allocator, config: LLaMAConfig) !LLaMAModel {
        // Initialize token embeddings
        var token_embeddings = try TokenEmbedding.init(allocator, config.vocab_size, config.d_model);

        // Initialize transformer layers
        var transformer_layers = try allocator.alloc(LLaMATransformerLayer, config.num_layers);
        for (0..config.num_layers) |i| {
            transformer_layers[i] = try LLaMATransformerLayer.init(allocator, config);
        }

        // Initialize final normalization
        var final_norm = try Tensor(f32).init(allocator, &[_]usize{config.d_model});
        final_norm.fill(1.0); // RMSNorm scale parameters

        // Initialize output projection
        var output_projection = try Tensor(f32).init(allocator, &[_]usize{config.d_model, config.vocab_size});

        // Initialize output weights (often tied with input embeddings in practice)
        const output_std = 1.0 / @sqrt(@as(f32, @floatFromInt(config.d_model)));
        initializeWeights(&output_projection, output_std);

        return LLaMAModel{
            .config = config,
            .token_embeddings = token_embeddings,
            .transformer_layers = transformer_layers,
            .final_norm = final_norm,
            .output_projection = output_projection,
            .allocator = allocator,
        };
    }

    /// Clean up model resources
    pub fn deinit(self: *LLaMAModel) void {
        self.token_embeddings.deinit();

        for (0..self.config.num_layers) |i| {
            self.transformer_layers[i].deinit();
        }
        self.allocator.free(self.transformer_layers);

        self.final_norm.deinit();
        self.output_projection.deinit();
    }

    /// Forward pass through the complete LLaMA model
    ///
    /// ## Input Shape
    /// - token_ids: [batch_size, seq_len] or [seq_len] for single sequence
    ///
    /// ## Output Shape
    /// - logits: [batch_size, seq_len, vocab_size] or [seq_len, vocab_size]
    ///
    /// ## Educational Note: Autoregressive Generation
    /// During generation, we typically:
    /// 1. Feed all available tokens to get logits for next position
    /// 2. Sample from the next token distribution
    /// 3. Append sampled token and repeat
    /// 4. Use KV-cache for efficiency (not implemented here but important)
    pub fn forward(self: *const LLaMAModel, token_ids: []const u32) !Tensor(f32) {
        // 1. Token embeddings
        var embeddings = try self.token_embeddings.forward(token_ids);
        defer embeddings.deinit();

        // 2. Pass through all transformer layers
        var current_hidden = embeddings;
        var should_free = false;

        for (0..self.config.num_layers) |layer_idx| {
            var layer_output = try self.transformer_layers[layer_idx].forward(current_hidden);

            // Free previous layer output (except original embeddings)
            if (should_free) {
                current_hidden.deinit();
            }

            current_hidden = layer_output;
            should_free = true;
        }

        // 3. Final RMSNorm
        var normalized = try normalization.rmsNorm(f32, current_hidden, self.final_norm, self.allocator);

        // Free final hidden state
        if (should_free) {
            current_hidden.deinit();
        }
        defer normalized.deinit();

        // 4. Output projection to vocabulary logits
        return try normalized.matmul(self.output_projection, self.allocator);
    }

    /// Generate text given a prompt
    ///
    /// ## Educational Note: Text Generation Strategies
    /// This is a simple greedy generation. Real implementations use:
    /// - **Temperature sampling**: Control randomness
    /// - **Top-k sampling**: Only consider k most likely tokens
    /// - **Top-p sampling**: Consider tokens up to cumulative probability p
    /// - **Beam search**: Maintain multiple candidate sequences
    pub fn generate(self: *const LLaMAModel, prompt_tokens: []const u32, max_new_tokens: usize, allocator: Allocator) ![]u32 {
        var generated_tokens = try allocator.alloc(u32, prompt_tokens.len + max_new_tokens);

        // Copy prompt
        @memcpy(generated_tokens[0..prompt_tokens.len], prompt_tokens);
        var current_length = prompt_tokens.len;

        // Generate tokens one by one
        for (0..max_new_tokens) |_| {
            // Get current sequence
            const current_sequence = generated_tokens[0..current_length];

            // Forward pass to get logits
            var logits = try self.forward(current_sequence);
            defer logits.deinit();

            // Get logits for the last position (next token prediction)
            const last_pos = current_length - 1;
            var max_logit: f32 = -std.math.inf(f32);
            var best_token: u32 = 0;

            // Simple greedy sampling (choose highest logit)
            for (0..self.config.vocab_size) |token_id| {
                const logit = try logits.get(&[_]usize{last_pos, token_id});
                if (logit > max_logit) {
                    max_logit = logit;
                    best_token = @intCast(token_id);
                }
            }

            // Add generated token
            generated_tokens[current_length] = best_token;
            current_length += 1;

            // Check for end-of-sequence token (typically token id 2)
            if (best_token == 2) break; // EOS token
        }

        // Return only the generated portion (excluding prompt)
        var result = try allocator.alloc(u32, current_length - prompt_tokens.len);
        @memcpy(result, generated_tokens[prompt_tokens.len..current_length]);

        allocator.free(generated_tokens);
        return result;
    }
};

/// Single LLaMA Transformer Layer
///
/// ## Educational Note: LLaMA Layer Structure
/// LLaMA uses pre-normalization with RMSNorm:
/// ```
/// input → RMSNorm → Attention → Add(input) → RMSNorm → SwiGLU → Add → output
/// ```
/// This differs from the original transformer's post-normalization structure.
const LLaMATransformerLayer = struct {
    /// Configuration
    config: LLaMAConfig,

    /// Pre-attention RMSNorm
    attention_norm: Tensor(f32),

    /// Multi-head attention
    attention: MultiHeadAttention,

    /// Pre-FFN RMSNorm
    ffn_norm: Tensor(f32),

    /// SwiGLU feed-forward network
    ffn: FeedForward,

    allocator: Allocator,

    pub fn init(allocator: Allocator, config: LLaMAConfig) !LLaMATransformerLayer {
        // Initialize RMSNorm parameters
        var attention_norm = try Tensor(f32).init(allocator, &[_]usize{config.d_model});
        var ffn_norm = try Tensor(f32).init(allocator, &[_]usize{config.d_model});
        attention_norm.fill(1.0);
        ffn_norm.fill(1.0);

        // Initialize attention (LLaMA uses standard multi-head attention with RoPE)
        var attention = try MultiHeadAttention.init(allocator, config.d_model, config.num_heads);

        // Initialize SwiGLU FFN
        var ffn = try FeedForward.init(allocator, config.d_model, config.d_ff, FFNType.SwiGLU);

        return LLaMATransformerLayer{
            .config = config,
            .attention_norm = attention_norm,
            .attention = attention,
            .ffn_norm = ffn_norm,
            .ffn = ffn,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *LLaMATransformerLayer) void {
        self.attention_norm.deinit();
        self.attention.deinit();
        self.ffn_norm.deinit();
        self.ffn.deinit();
    }

    pub fn forward(self: *const LLaMATransformerLayer, input: Tensor(f32)) !Tensor(f32) {
        // Pre-norm attention path
        var attention_normed = try normalization.rmsNorm(f32, input, self.attention_norm, self.allocator);
        defer attention_normed.deinit();

        // Self-attention (with causal masking for autoregressive generation)
        // Note: In a complete implementation, we'd create causal masks here
        var attention_output = try self.attention.forward(attention_normed, attention_normed, attention_normed, null);
        defer attention_output.deinit();

        // First residual connection
        var after_attention = try input.add(attention_output, self.allocator);
        defer after_attention.deinit();

        // Pre-norm FFN path
        var ffn_normed = try normalization.rmsNorm(f32, after_attention, self.ffn_norm, self.allocator);
        defer ffn_normed.deinit();

        // SwiGLU feed-forward
        var ffn_output = try self.ffn.forward(ffn_normed);
        defer ffn_output.deinit();

        // Second residual connection
        return try after_attention.add(ffn_output, self.allocator);
    }
};

// Helper function for weight initialization
fn initializeWeights(tensor: *Tensor(f32), std_dev: f32) void {
    var seed: u32 = 67890;
    for (0..tensor.size) |i| {
        seed = seed *% 1664525 +% 1013904223;
        const rand_f32 = @as(f32, @floatFromInt(seed)) / @as(f32, @floatFromInt(std.math.maxInt(u32)));
        tensor.data[i] = (rand_f32 - 0.5) * 2.0 * std_dev;
    }
}

// ============================================================================
// COMPREHENSIVE TESTS
// ============================================================================

test "LLaMA configuration parameter counting" {
    const config_7b = LLaMAConfig.init(.LLaMA_7B);
    const config_13b = LLaMAConfig.init(.LLaMA_13B);

    // Test basic configuration values
    try testing.expectEqual(@as(usize, 4096), config_7b.d_model);
    try testing.expectEqual(@as(usize, 32), config_7b.num_layers);
    try testing.expectEqual(@as(usize, 32), config_7b.num_heads);

    // Test parameter counting
    const params_7b = config_7b.getParameterCount();
    const params_13b = config_13b.getParameterCount();

    // 13B should have more parameters than 7B
    try testing.expect(params_13b > params_7b);

    // Check that parameter count is in reasonable range for 7B model
    // Should be around 6.7B parameters (close to 7B)
    try testing.expect(params_7b > 6_000_000_000);
    try testing.expect(params_7b < 8_000_000_000);
}

test "LLaMA model initialization" {
    const allocator = testing.allocator;

    // Create a small custom config for testing
    var config = LLaMAConfig{
        .d_model = 64,
        .num_layers = 2,
        .num_heads = 4,
        .d_ff = 172, // ~8/3 * d_model
        .vocab_size = 100,
        .max_seq_len = 128,
        .norm_eps = 1e-6,
        .use_rope = true,
        .rope_theta = 10000.0,
    };

    var model = try LLaMAModel.init(allocator, config);
    defer model.deinit();

    // Test that model components are initialized correctly
    try testing.expectEqual(config.d_model, model.config.d_model);
    try testing.expectEqual(config.num_layers, model.transformer_layers.len);
    try testing.expectEqual(config.vocab_size, model.token_embeddings.vocab_size);
}

test "LLaMA forward pass shape consistency" {
    const allocator = testing.allocator;

    var config = LLaMAConfig{
        .d_model = 32,
        .num_layers = 1,
        .num_heads = 2,
        .d_ff = 86,
        .vocab_size = 50,
        .max_seq_len = 64,
        .norm_eps = 1e-6,
        .use_rope = true,
        .rope_theta = 10000.0,
    };

    var model = try LLaMAModel.init(allocator, config);
    defer model.deinit();

    // Test forward pass with a sequence of token IDs
    const token_ids = [_]u32{1, 5, 10, 15, 2}; // Including EOS token

    var logits = try model.forward(&token_ids);
    defer logits.deinit();

    // Output should be [seq_len, vocab_size]
    try testing.expectEqual(@as(usize, token_ids.len), logits.shape[0]);
    try testing.expectEqual(config.vocab_size, logits.shape[1]);
}

test "LLaMA generation basic functionality" {
    const allocator = testing.allocator;

    var config = LLaMAConfig{
        .d_model = 16,
        .num_layers = 1,
        .num_heads = 2,
        .d_ff = 43,
        .vocab_size = 20,
        .max_seq_len = 32,
        .norm_eps = 1e-6,
        .use_rope = true,
        .rope_theta = 10000.0,
    };

    var model = try LLaMAModel.init(allocator, config);
    defer model.deinit();

    const prompt = [_]u32{1, 5}; // Simple prompt
    const max_new_tokens = 3;

    var generated = try model.generate(&prompt, max_new_tokens, allocator);
    defer allocator.free(generated);

    // Should generate some tokens (may be less than max if EOS encountered)
    try testing.expect(generated.len <= max_new_tokens);

    // Generated tokens should be valid (within vocab range)
    for (generated) |token| {
        try testing.expect(token < config.vocab_size);
    }
}

test "LLaMA transformer layer residual connections" {
    const allocator = testing.allocator;

    var config = LLaMAConfig{
        .d_model = 8,
        .num_layers = 1,
        .num_heads = 2,
        .d_ff = 22,
        .vocab_size = 10,
        .max_seq_len = 16,
        .norm_eps = 1e-6,
        .use_rope = true,
        .rope_theta = 10000.0,
    };

    var layer = try LLaMATransformerLayer.init(allocator, config);
    defer layer.deinit();

    // Create test input
    var input = try Tensor(f32).init(allocator, &[_]usize{2, 8}); // [seq_len=2, d_model=8]
    defer input.deinit();
    input.fill(0.1);

    var output = try layer.forward(input);
    defer output.deinit();

    // Output should have same shape as input
    try testing.expectEqualSlices(usize, input.shape, output.shape);

    // Due to residual connections, output should be different from input but finite
    var has_differences = false;
    for (0..input.size) |i| {
        try testing.expect(std.math.isFinite(output.data[i]));
        if (@abs(input.data[i] - output.data[i]) > 0.01) {
            has_differences = true;
        }
    }
    try testing.expect(has_differences); // Processing should change values
}

test "LLaMA scaling laws verification" {
    // Test that different model sizes follow expected scaling patterns

    const configs = [_]LLaMAConfig{
        LLaMAConfig.init(.LLaMA_7B),
        LLaMAConfig.init(.LLaMA_13B),
        LLaMAConfig.init(.LLaMA_30B),
        LLaMAConfig.init(.LLaMA_65B),
    };

    // Parameter counts should increase
    var prev_params: usize = 0;
    for (configs) |config| {
        const params = config.getParameterCount();
        try testing.expect(params > prev_params);
        prev_params = params;

        // All models should use same vocabulary and architecture choices
        try testing.expectEqual(@as(usize, 32000), config.vocab_size);
        try testing.expectEqual(@as(usize, 2048), config.max_seq_len);
        try testing.expect(config.use_rope);
        try testing.expectEqual(@as(f32, 10000.0), config.rope_theta);

        // d_ff should be roughly 8/3 * d_model (for SwiGLU gating)
        const expected_ratio = @as(f32, @floatFromInt(config.d_ff)) / @as(f32, @floatFromInt(config.d_model));
        try testing.expect(expected_ratio > 2.5 and expected_ratio < 3.0);
    }
}

test "LLaMA memory management" {
    const allocator = testing.allocator;

    // Test that we can create and destroy models without leaks
    var config = LLaMAConfig{
        .d_model = 4,
        .num_layers = 1,
        .num_heads = 1,
        .d_ff = 11,
        .vocab_size = 5,
        .max_seq_len = 8,
        .norm_eps = 1e-6,
        .use_rope = true,
        .rope_theta = 10000.0,
    };

    // Create and destroy multiple models
    for (0..3) |_| {
        var model = try LLaMAModel.init(allocator, config);
        model.deinit();
    }

    // Test forward pass memory cleanup
    var model = try LLaMAModel.init(allocator, config);
    defer model.deinit();

    const tokens = [_]u32{1, 2};

    // Multiple forward passes should not accumulate memory
    for (0..3) |_| {
        var output = try model.forward(&tokens);
        output.deinit();
    }
}