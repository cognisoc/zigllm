// BLOOM Model Architecture Implementation
// Educational implementation of the BLOOM model family (176B parameter model)
//
// Key architectural features:
// - ALiBi (Attention with Linear Biases) instead of positional embeddings
// - Pre-layer normalization
// - No bias terms in linear layers (except final layer)
// - Causal attention with ALiBi bias
// - Multi-lingual tokenization support

const std = @import("std");
const Tensor = @import("../foundation/tensor.zig").Tensor;
const attention = @import("../transformers/attention.zig");
const feed_forward = @import("../transformers/feed_forward.zig");
const normalization = @import("../neural_primitives/normalization.zig");
const activations = @import("../neural_primitives/activations.zig");
const quantization = @import("../linear_algebra/quantization.zig");

// BLOOM Model Configuration
pub const BloomConfig = struct {
    vocab_size: u32 = 250880,
    n_ctx: u32 = 2048,        // context length
    n_embd: u32 = 14336,      // embedding dimensions
    n_head: u32 = 112,        // attention heads
    n_layer: u32 = 70,        // transformer layers
    use_cache: bool = true,
    layer_norm_epsilon: f32 = 1e-5,
    initializer_range: f32 = 0.02,
    use_alibi: bool = true,   // Use ALiBi attention bias
    attention_dropout: f32 = 0.0,
    hidden_dropout: f32 = 0.0,

    // Model variants
    pub const BLOOM_176B = BloomConfig{
        .vocab_size = 250880,
        .n_ctx = 2048,
        .n_embd = 14336,
        .n_head = 112,
        .n_layer = 70,
    };

    pub const BLOOM_7B1 = BloomConfig{
        .vocab_size = 250880,
        .n_ctx = 2048,
        .n_embd = 4096,
        .n_head = 32,
        .n_layer = 30,
    };

    pub const BLOOM_3B = BloomConfig{
        .vocab_size = 250880,
        .n_ctx = 2048,
        .n_embd = 2560,
        .n_head = 32,
        .n_layer = 30,
    };

    pub const BLOOM_1B7 = BloomConfig{
        .vocab_size = 250880,
        .n_ctx = 2048,
        .n_embd = 2048,
        .n_head = 16,
        .n_layer = 24,
    };

    pub const BLOOM_560M = BloomConfig{
        .vocab_size = 250880,
        .n_ctx = 2048,
        .n_embd = 1024,
        .n_head = 16,
        .n_layer = 24,
    };
};

// ALiBi (Attention with Linear Biases) Implementation
pub const ALiBi = struct {
    slopes: []f32,
    max_seq_len: u32,
    n_heads: u32,
    bias_cache: ?Tensor,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n_heads: u32, max_seq_len: u32) !ALiBi {
        var slopes = try allocator.alloc(f32, n_heads);

        // Compute ALiBi slopes based on the paper
        // slopes[i] = 2^(-8i/n) for head i
        const closest_power_of_2 = blk: {
            var power: u32 = 1;
            while (power < n_heads) {
                power *= 2;
            }
            break :blk power;
        };

        const ratio = @as(f32, @floatFromInt(closest_power_of_2)) / @as(f32, @floatFromInt(n_heads));

        for (0..n_heads) |i| {
            const slope_exp = -8.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n_heads));
            slopes[i] = std.math.pow(f32, 2.0, slope_exp) * ratio;
        }

        return ALiBi{
            .slopes = slopes,
            .max_seq_len = max_seq_len,
            .n_heads = n_heads,
            .bias_cache = null,
            .allocator = allocator,
        };
    }

    pub fn getBias(self: *ALiBi, seq_len: u32) !*Tensor {
        // Cache bias tensor for efficiency
        if (self.bias_cache == null or self.bias_cache.?.shape[2] != seq_len) {
            if (self.bias_cache) |*cache| {
                cache.deinit();
            }

            // Create bias tensor: [n_heads, seq_len, seq_len]
            var bias = try Tensor.zeros(self.allocator, &[_]u32{ self.n_heads, seq_len, seq_len });

            for (0..self.n_heads) |h| {
                for (0..seq_len) |i| {
                    for (0..seq_len) |j| {
                        // ALiBi bias is negative distance * slope
                        var bias_value: f32 = 0.0;
                        if (j > i) {
                            // Future tokens get negative infinity (causal masking)
                            bias_value = -std.math.inf(f32);
                        } else {
                            // Past tokens get linear bias
                            const distance = @as(f32, @floatFromInt(i - j));
                            bias_value = -distance * self.slopes[h];
                        }
                        try bias.set(&[_]u32{ @intCast(h), @intCast(i), @intCast(j) }, bias_value);
                    }
                }
            }

            self.bias_cache = bias;
        }

        return &self.bias_cache.?;
    }

    pub fn deinit(self: *ALiBi) void {
        self.allocator.free(self.slopes);
        if (self.bias_cache) |*cache| {
            cache.deinit();
        }
    }
};

// BLOOM Attention Block with ALiBi
pub const BloomAttention = struct {
    config: BloomConfig,
    query_key_value: Tensor, // fused QKV projection
    dense: Tensor,           // output projection
    alibi: ALiBi,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: BloomConfig) !BloomAttention {
        const qkv_size = 3 * config.n_embd; // Q, K, V combined

        return BloomAttention{
            .config = config,
            .query_key_value = try Tensor.random(allocator, &[_]u32{ config.n_embd, qkv_size }),
            .dense = try Tensor.random(allocator, &[_]u32{ config.n_embd, config.n_embd }),
            .alibi = try ALiBi.init(allocator, config.n_head, config.n_ctx),
            .allocator = allocator,
        };
    }

    pub fn forward(self: *BloomAttention, x: *Tensor) !Tensor {
        const batch_size = x.shape[0];
        const seq_len = x.shape[1];
        const n_embd = x.shape[2];
        const head_dim = n_embd / self.config.n_head;

        // Fused QKV projection
        var qkv = try Tensor.matmul(self.allocator, x, &self.query_key_value);
        defer qkv.deinit();

        // Split QKV and reshape for multi-head attention
        var q = try Tensor.zeros(self.allocator, &[_]u32{ batch_size, seq_len, self.config.n_head, head_dim });
        var k = try Tensor.zeros(self.allocator, &[_]u32{ batch_size, seq_len, self.config.n_head, head_dim });
        var v = try Tensor.zeros(self.allocator, &[_]u32{ batch_size, seq_len, self.config.n_head, head_dim });
        defer q.deinit();
        defer k.deinit();
        defer v.deinit();

        // Split the fused QKV tensor
        for (0..batch_size) |b| {
            for (0..seq_len) |s| {
                for (0..self.config.n_head) |h| {
                    for (0..head_dim) |d| {
                        // Query
                        const q_val = try qkv.get(&[_]u32{ @intCast(b), @intCast(s), @intCast(h * head_dim + d) });
                        try q.set(&[_]u32{ @intCast(b), @intCast(s), @intCast(h), @intCast(d) }, q_val);

                        // Key
                        const k_val = try qkv.get(&[_]u32{ @intCast(b), @intCast(s), @intCast(n_embd + h * head_dim + d) });
                        try k.set(&[_]u32{ @intCast(b), @intCast(s), @intCast(h), @intCast(d) }, k_val);

                        // Value
                        const v_val = try qkv.get(&[_]u32{ @intCast(b), @intCast(s), @intCast(2 * n_embd + h * head_dim + d) });
                        try v.set(&[_]u32{ @intCast(b), @intCast(s), @intCast(h), @intCast(d) }, v_val);
                    }
                }
            }
        }

        // Get ALiBi bias
        var alibi_bias = try self.alibi.getBias(@intCast(seq_len));

        // Perform attention with ALiBi bias
        var attn_output = try self.scaledDotProductAttentionWithBias(
            &q, &k, &v,
            alibi_bias,
            @sqrt(@as(f32, @floatFromInt(head_dim)))
        );
        defer attn_output.deinit();

        // Reshape back to original dimensions
        try attn_output.reshape(&[_]u32{ batch_size, seq_len, n_embd });

        // Apply output projection
        return try Tensor.matmul(self.allocator, &attn_output, &self.dense);
    }

    fn scaledDotProductAttentionWithBias(
        self: *BloomAttention,
        q: *const Tensor,
        k: *const Tensor,
        v: *const Tensor,
        bias: *const Tensor,
        scale: f32,
    ) !Tensor {
        const batch_size = q.shape[0];
        const seq_len = q.shape[1];
        const n_head = q.shape[2];
        const head_dim = q.shape[3];

        // Compute attention scores: Q @ K^T
        var scores = try Tensor.zeros(self.allocator, &[_]u32{ batch_size, n_head, seq_len, seq_len });
        defer scores.deinit();

        for (0..batch_size) |b| {
            for (0..n_head) |h| {
                for (0..seq_len) |i| {
                    for (0..seq_len) |j| {
                        var dot_product: f32 = 0.0;

                        for (0..head_dim) |d| {
                            const q_val = try q.get(&[_]u32{ @intCast(b), @intCast(i), @intCast(h), @intCast(d) });
                            const k_val = try k.get(&[_]u32{ @intCast(b), @intCast(j), @intCast(h), @intCast(d) });
                            dot_product += q_val * k_val;
                        }

                        // Scale and add ALiBi bias
                        const scaled_score = dot_product / scale;
                        const bias_val = try bias.get(&[_]u32{ @intCast(h), @intCast(i), @intCast(j) });

                        try scores.set(&[_]u32{ @intCast(b), @intCast(h), @intCast(i), @intCast(j) }, scaled_score + bias_val);
                    }
                }
            }
        }

        // Apply softmax
        var attn_weights = try Tensor.softmax(self.allocator, &scores, 3);
        defer attn_weights.deinit();

        // Apply attention weights to values
        var output = try Tensor.zeros(self.allocator, &[_]u32{ batch_size, seq_len, n_head, head_dim });

        for (0..batch_size) |b| {
            for (0..seq_len) |i| {
                for (0..n_head) |h| {
                    for (0..head_dim) |d| {
                        var weighted_sum: f32 = 0.0;

                        for (0..seq_len) |j| {
                            const weight = try attn_weights.get(&[_]u32{ @intCast(b), @intCast(h), @intCast(i), @intCast(j) });
                            const value = try v.get(&[_]u32{ @intCast(b), @intCast(j), @intCast(h), @intCast(d) });
                            weighted_sum += weight * value;
                        }

                        try output.set(&[_]u32{ @intCast(b), @intCast(i), @intCast(h), @intCast(d) }, weighted_sum);
                    }
                }
            }
        }

        return output;
    }

    pub fn deinit(self: *BloomAttention) void {
        self.query_key_value.deinit();
        self.dense.deinit();
        self.alibi.deinit();
    }
};

// BLOOM MLP Block
pub const BloomMLP = struct {
    config: BloomConfig,
    dense_h_to_4h: Tensor,   // up projection
    dense_4h_to_h: Tensor,   // down projection
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: BloomConfig) !BloomMLP {
        const intermediate_size = 4 * config.n_embd; // Standard 4x expansion

        return BloomMLP{
            .config = config,
            .dense_h_to_4h = try Tensor.random(allocator, &[_]u32{ config.n_embd, intermediate_size }),
            .dense_4h_to_h = try Tensor.random(allocator, &[_]u32{ intermediate_size, config.n_embd }),
            .allocator = allocator,
        };
    }

    pub fn forward(self: *BloomMLP, x: *Tensor) !Tensor {
        // Up projection
        var hidden = try Tensor.matmul(self.allocator, x, &self.dense_h_to_4h);
        defer hidden.deinit();

        // GELU activation
        try activations.gelu(&hidden);

        // Down projection
        return try Tensor.matmul(self.allocator, &hidden, &self.dense_4h_to_h);
    }

    pub fn deinit(self: *BloomMLP) void {
        self.dense_h_to_4h.deinit();
        self.dense_4h_to_h.deinit();
    }
};

// BLOOM Transformer Block
pub const BloomBlock = struct {
    config: BloomConfig,
    input_layernorm: normalization.LayerNorm,
    self_attention: BloomAttention,
    post_attention_layernorm: normalization.LayerNorm,
    mlp: BloomMLP,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: BloomConfig) !BloomBlock {
        return BloomBlock{
            .config = config,
            .input_layernorm = try normalization.LayerNorm.init(allocator, config.n_embd),
            .self_attention = try BloomAttention.init(allocator, config),
            .post_attention_layernorm = try normalization.LayerNorm.init(allocator, config.n_embd),
            .mlp = try BloomMLP.init(allocator, config),
            .allocator = allocator,
        };
    }

    pub fn forward(self: *BloomBlock, x: *Tensor) !Tensor {
        // Pre-attention layer norm
        var normed1 = try self.input_layernorm.forward(x);
        defer normed1.deinit();

        // Self attention
        var attn_output = try self.self_attention.forward(&normed1);
        defer attn_output.deinit();

        // First residual connection
        var residual1 = try Tensor.add(self.allocator, x, &attn_output);
        defer residual1.deinit();

        // Pre-MLP layer norm
        var normed2 = try self.post_attention_layernorm.forward(&residual1);
        defer normed2.deinit();

        // MLP
        var mlp_output = try self.mlp.forward(&normed2);
        defer mlp_output.deinit();

        // Second residual connection
        return try Tensor.add(self.allocator, &residual1, &mlp_output);
    }

    pub fn deinit(self: *BloomBlock) void {
        self.input_layernorm.deinit();
        self.self_attention.deinit();
        self.post_attention_layernorm.deinit();
        self.mlp.deinit();
    }
};

// Complete BLOOM Model
pub const BloomModel = struct {
    config: BloomConfig,
    word_embeddings: Tensor,                    // token embeddings
    word_embeddings_layernorm: normalization.LayerNorm, // embedding layer norm
    h: []BloomBlock,                           // transformer blocks
    ln_f: normalization.LayerNorm,             // final layer norm
    lm_head: Tensor,                           // language modeling head
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: BloomConfig) !BloomModel {
        var h = try allocator.alloc(BloomBlock, config.n_layer);
        for (0..config.n_layer) |i| {
            h[i] = try BloomBlock.init(allocator, config);
        }

        var word_embeddings = try Tensor.random(allocator, &[_]u32{ config.vocab_size, config.n_embd });
        var lm_head = try Tensor.random(allocator, &[_]u32{ config.n_embd, config.vocab_size });

        return BloomModel{
            .config = config,
            .word_embeddings = word_embeddings,
            .word_embeddings_layernorm = try normalization.LayerNorm.init(allocator, config.n_embd),
            .h = h,
            .ln_f = try normalization.LayerNorm.init(allocator, config.n_embd),
            .lm_head = lm_head,
            .allocator = allocator,
        };
    }

    pub fn forward(self: *BloomModel, input_ids: []const u32) !Tensor {
        const batch_size = 1;
        const seq_len = input_ids.len;

        // Token embedding lookup
        var x = try Tensor.zeros(self.allocator, &[_]u32{ batch_size, @intCast(seq_len), self.config.n_embd });
        for (input_ids, 0..) |token_id, i| {
            const embedding_row = try self.word_embeddings.getRow(token_id);
            defer embedding_row.deinit();

            for (0..self.config.n_embd) |j| {
                const val = try embedding_row.get(&[_]u32{@intCast(j)});
                try x.set(&[_]u32{ 0, @intCast(i), @intCast(j) }, val);
            }
        }

        // Embedding layer normalization (unique to BLOOM)
        var normed_embeddings = try self.word_embeddings_layernorm.forward(&x);
        x.deinit();
        var current = normed_embeddings;

        // Pass through transformer blocks
        for (self.h) |*block| {
            var next = try block.forward(&current);
            current.deinit();
            current = next;
        }

        // Final layer normalization
        var normed = try self.ln_f.forward(&current);
        current.deinit();
        defer normed.deinit();

        // Language modeling head projection
        return try Tensor.matmul(self.allocator, &normed, &self.lm_head);
    }

    pub fn generate(self: *BloomModel, prompt: []const u32, max_length: u32, temperature: f32) ![]u32 {
        var result = try self.allocator.alloc(u32, max_length);
        std.mem.copy(u32, result[0..prompt.len], prompt);

        for (prompt.len..max_length) |i| {
            var logits = try self.forward(result[0..i]);
            defer logits.deinit();

            // Get last token's logits
            const last_token_logits = try logits.getRow(@intCast(i - 1));
            defer last_token_logits.deinit();

            // Apply temperature scaling
            for (0..self.config.vocab_size) |j| {
                const val = try last_token_logits.get(&[_]u32{@intCast(j)});
                try last_token_logits.set(&[_]u32{@intCast(j)}, val / temperature);
            }

            const next_token = try sampleFromLogits(&last_token_logits);
            result[i] = next_token;
        }

        return result;
    }

    pub fn deinit(self: *BloomModel) void {
        self.word_embeddings.deinit();
        self.word_embeddings_layernorm.deinit();
        for (self.h) |*block| {
            block.deinit();
        }
        self.allocator.free(self.h);
        self.ln_f.deinit();
        self.lm_head.deinit();
    }
};

// Helper function for sampling
fn sampleFromLogits(logits: *const Tensor) !u32 {
    var max_val: f32 = -std.math.inf(f32);
    var max_idx: u32 = 0;

    for (0..logits.shape[0]) |i| {
        const val = try logits.get(&[_]u32{@intCast(i)});
        if (val > max_val) {
            max_val = val;
            max_idx = @intCast(i);
        }
    }

    return max_idx;
}

// Educational utilities
pub const BloomUtils = struct {
    pub fn printModelInfo(config: BloomConfig) void {
        std.debug.print("=== BLOOM Model Configuration ===\n");
        std.debug.print("Vocabulary Size: {}\n", .{config.vocab_size});
        std.debug.print("Context Length: {}\n", .{config.n_ctx});
        std.debug.print("Hidden Dimensions: {}\n", .{config.n_embd});
        std.debug.print("Attention Heads: {}\n", .{config.n_head});
        std.debug.print("Transformer Layers: {}\n", .{config.n_layer});
        std.debug.print("Layer Norm Epsilon: {}\n", .{config.layer_norm_epsilon});
        std.debug.print("Uses ALiBi: {}\n", .{config.use_alibi});
        std.debug.print("Attention Dropout: {}\n", .{config.attention_dropout});
        std.debug.print("Hidden Dropout: {}\n", .{config.hidden_dropout});

        const total_params = calculateParameters(config);
        std.debug.print("Total Parameters: ~{:.1}B\n", .{@as(f32, @floatFromInt(total_params)) / 1_000_000_000.0});
        std.debug.print("=================================\n");
    }

    pub fn calculateParameters(config: BloomConfig) u64 {
        const embedding_params = @as(u64, config.vocab_size) * config.n_embd;
        const embedding_ln_params = config.n_embd;
        const attention_params_per_layer = 4 * @as(u64, config.n_embd) * config.n_embd; // QKV + output
        const mlp_params_per_layer = 2 * @as(u64, config.n_embd) * (4 * config.n_embd); // up + down
        const norm_params_per_layer = 2 * config.n_embd; // input + post attention norm

        const layer_params = (attention_params_per_layer + mlp_params_per_layer + norm_params_per_layer) * config.n_layer;
        const final_norm_params = config.n_embd;
        const lm_head_params = @as(u64, config.n_embd) * config.vocab_size;

        return embedding_params + embedding_ln_params + layer_params + final_norm_params + lm_head_params;
    }

    pub fn printALiBiInfo(alibi: *const ALiBi) void {
        std.debug.print("=== ALiBi Configuration ===\n");
        std.debug.print("Number of Heads: {}\n", .{alibi.n_heads});
        std.debug.print("Max Sequence Length: {}\n", .{alibi.max_seq_len});
        std.debug.print("Slopes: ");
        for (alibi.slopes, 0..) |slope, i| {
            std.debug.print("{:.4}", .{slope});
            if (i < alibi.slopes.len - 1) std.debug.print(", ");
        }
        std.debug.print("\n===========================\n");
    }
};