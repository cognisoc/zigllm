// GPT-NeoX Model Architecture Implementation
// Educational implementation of the GPT-NeoX model family (20B parameter model)
//
// Key architectural features:
// - Parallel attention and MLP blocks (like GPT-J)
// - Rotary positional embeddings (RoPE)
// - No bias terms in attention projections
// - Different layer normalization placement (pre-norm)
// - Larger vocabulary and context lengths

const std = @import("std");
const Tensor = @import("../foundation/tensor.zig").Tensor;
const attention = @import("../transformers/attention.zig");
const feed_forward = @import("../transformers/feed_forward.zig");
const normalization = @import("../neural_primitives/normalization.zig");
const activations = @import("../neural_primitives/activations.zig");
const quantization = @import("../linear_algebra/quantization.zig");

// GPT-NeoX Model Configuration
pub const GPTNeoXConfig = struct {
    vocab_size: u32 = 50432,
    n_ctx: u32 = 2048,        // context length
    n_embd: u32 = 6144,       // embedding dimensions
    n_head: u32 = 48,         // attention heads
    n_layer: u32 = 44,        // transformer layers
    rotary_dim: u32 = 32,     // rotary embedding dimensions (partial)
    intermediate_size: u32 = 24576, // MLP intermediate size
    use_parallel_residual: bool = true,
    rope_base: f32 = 10000.0,
    rope_scaling: ?RopeScaling = null,

    // Model variants
    pub const GPT_NEOX_20B = GPTNeoXConfig{
        .vocab_size = 50432,
        .n_ctx = 2048,
        .n_embd = 6144,
        .n_head = 48,
        .n_layer = 44,
        .rotary_dim = 32,
        .intermediate_size = 24576,
    };

    pub const GPT_NEOX_1_3B = GPTNeoXConfig{
        .vocab_size = 50432,
        .n_ctx = 2048,
        .n_embd = 2048,
        .n_head = 16,
        .n_layer = 24,
        .rotary_dim = 32,
        .intermediate_size = 8192,
    };

    pub const GPT_NEOX_410M = GPTNeoXConfig{
        .vocab_size = 50432,
        .n_ctx = 2048,
        .n_embd = 1024,
        .n_head = 16,
        .n_layer = 24,
        .rotary_dim = 32,
        .intermediate_size = 4096,
    };
};

pub const RopeScaling = struct {
    scaling_type: ScalingType,
    scaling_factor: f32,

    pub const ScalingType = enum {
        linear,
        dynamic,
    };
};

// Rotary Positional Embedding (RoPE) with Scaling Support
pub const GPTNeoXRotaryEmbedding = struct {
    dim: u32,
    max_seq_len: u32,
    base: f32,
    scaling: ?RopeScaling,
    cos_cache: Tensor,
    sin_cache: Tensor,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, dim: u32, max_seq_len: u32, base: f32, scaling: ?RopeScaling) !GPTNeoXRotaryEmbedding {
        var rope = GPTNeoXRotaryEmbedding{
            .dim = dim,
            .max_seq_len = max_seq_len,
            .base = base,
            .scaling = scaling,
            .cos_cache = try Tensor.zeros(allocator, &[_]u32{ max_seq_len, dim / 2 }),
            .sin_cache = try Tensor.zeros(allocator, &[_]u32{ max_seq_len, dim / 2 }),
            .allocator = allocator,
        };

        try rope.precompute();
        return rope;
    }

    fn precompute(self: *GPTNeoXRotaryEmbedding) !void {
        var effective_base = self.base;

        // Apply dynamic scaling if specified
        if (self.scaling) |scale| {
            if (scale.scaling_type == .dynamic) {
                effective_base *= std.math.pow(f32, scale.scaling_factor, @as(f32, @floatFromInt(self.dim)) / (@as(f32, @floatFromInt(self.dim)) - 2.0));
            }
        }

        for (0..self.max_seq_len) |pos| {
            var effective_pos = @as(f32, @floatFromInt(pos));

            // Apply linear scaling if specified
            if (self.scaling) |scale| {
                if (scale.scaling_type == .linear) {
                    effective_pos /= scale.scaling_factor;
                }
            }

            for (0..self.dim / 2) |i| {
                const freq = 1.0 / std.math.pow(f32, effective_base, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(self.dim)));
                const angle = effective_pos * freq;

                try self.cos_cache.set(&[_]u32{ @intCast(pos), @intCast(i) }, @cos(angle));
                try self.sin_cache.set(&[_]u32{ @intCast(pos), @intCast(i) }, @sin(angle));
            }
        }
    }

    pub fn apply(self: *GPTNeoXRotaryEmbedding, x: *Tensor, pos: u32) !void {
        const seq_len = x.shape[1];
        const head_dim = x.shape[3];
        const rotary_dim = @min(self.dim, head_dim);

        for (0..seq_len) |t| {
            const actual_pos = pos + t;
            if (actual_pos >= self.max_seq_len) continue;

            for (0..rotary_dim / 2) |i| {
                const cos_val = try self.cos_cache.get(&[_]u32{ @intCast(actual_pos), @intCast(i) });
                const sin_val = try self.sin_cache.get(&[_]u32{ @intCast(actual_pos), @intCast(i) });

                for (0..x.shape[0]) |batch| {
                    for (0..x.shape[2]) |head| {
                        const x0 = try x.get(&[_]u32{ @intCast(batch), @intCast(t), @intCast(head), @intCast(i * 2) });
                        const x1 = try x.get(&[_]u32{ @intCast(batch), @intCast(t), @intCast(head), @intCast(i * 2 + 1) });

                        const new_x0 = x0 * cos_val - x1 * sin_val;
                        const new_x1 = x0 * sin_val + x1 * cos_val;

                        try x.set(&[_]u32{ @intCast(batch), @intCast(t), @intCast(head), @intCast(i * 2) }, new_x0);
                        try x.set(&[_]u32{ @intCast(batch), @intCast(t), @intCast(head), @intCast(i * 2 + 1) }, new_x1);
                    }
                }
            }
        }
    }

    pub fn deinit(self: *GPTNeoXRotaryEmbedding) void {
        self.cos_cache.deinit();
        self.sin_cache.deinit();
    }
};

// GPT-NeoX Attention Block
pub const GPTNeoXAttention = struct {
    config: GPTNeoXConfig,
    query_key_value: Tensor, // fused QKV projection
    dense: Tensor,           // output projection
    rope: GPTNeoXRotaryEmbedding,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: GPTNeoXConfig) !GPTNeoXAttention {
        const head_dim = config.n_embd / config.n_head;
        const qkv_size = 3 * config.n_embd; // Q, K, V combined

        return GPTNeoXAttention{
            .config = config,
            .query_key_value = try Tensor.random(allocator, &[_]u32{ config.n_embd, qkv_size }),
            .dense = try Tensor.random(allocator, &[_]u32{ config.n_embd, config.n_embd }),
            .rope = try GPTNeoXRotaryEmbedding.init(allocator, config.rotary_dim, config.n_ctx, config.rope_base, config.rope_scaling),
            .allocator = allocator,
        };
    }

    pub fn forward(self: *GPTNeoXAttention, x: *Tensor, pos: u32) !Tensor {
        const batch_size = x.shape[0];
        const seq_len = x.shape[1];
        const n_embd = x.shape[2];
        const head_dim = n_embd / self.config.n_head;

        // Fused QKV projection
        var qkv = try Tensor.matmul(self.allocator, x, &self.query_key_value);
        defer qkv.deinit();

        // Split QKV
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

        // Apply rotary embeddings to queries and keys
        try self.rope.apply(&q, pos);
        try self.rope.apply(&k, pos);

        // Perform scaled dot-product attention
        var attn_output = try attention.scaledDotProductAttention(
            self.allocator,
            &q, &k, &v,
            null,
            @sqrt(@as(f32, @floatFromInt(head_dim)))
        );
        defer attn_output.deinit();

        // Reshape back to original dimensions
        try attn_output.reshape(&[_]u32{ batch_size, seq_len, n_embd });

        // Apply output projection
        return try Tensor.matmul(self.allocator, &attn_output, &self.dense);
    }

    pub fn deinit(self: *GPTNeoXAttention) void {
        self.query_key_value.deinit();
        self.dense.deinit();
        self.rope.deinit();
    }
};

// GPT-NeoX MLP Block
pub const GPTNeoXMLP = struct {
    config: GPTNeoXConfig,
    dense_h_to_4h: Tensor,   // up projection
    dense_4h_to_h: Tensor,   // down projection
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: GPTNeoXConfig) !GPTNeoXMLP {
        return GPTNeoXMLP{
            .config = config,
            .dense_h_to_4h = try Tensor.random(allocator, &[_]u32{ config.n_embd, config.intermediate_size }),
            .dense_4h_to_h = try Tensor.random(allocator, &[_]u32{ config.intermediate_size, config.n_embd }),
            .allocator = allocator,
        };
    }

    pub fn forward(self: *GPTNeoXMLP, x: *Tensor) !Tensor {
        // Up projection
        var hidden = try Tensor.matmul(self.allocator, x, &self.dense_h_to_4h);
        defer hidden.deinit();

        // GELU activation
        try activations.gelu(&hidden);

        // Down projection
        return try Tensor.matmul(self.allocator, &hidden, &self.dense_4h_to_h);
    }

    pub fn deinit(self: *GPTNeoXMLP) void {
        self.dense_h_to_4h.deinit();
        self.dense_4h_to_h.deinit();
    }
};

// GPT-NeoX Transformer Layer
pub const GPTNeoXLayer = struct {
    config: GPTNeoXConfig,
    input_layernorm: normalization.LayerNorm,
    post_attention_layernorm: normalization.LayerNorm,
    attention: GPTNeoXAttention,
    mlp: GPTNeoXMLP,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: GPTNeoXConfig) !GPTNeoXLayer {
        return GPTNeoXLayer{
            .config = config,
            .input_layernorm = try normalization.LayerNorm.init(allocator, config.n_embd),
            .post_attention_layernorm = try normalization.LayerNorm.init(allocator, config.n_embd),
            .attention = try GPTNeoXAttention.init(allocator, config),
            .mlp = try GPTNeoXMLP.init(allocator, config),
            .allocator = allocator,
        };
    }

    pub fn forward(self: *GPTNeoXLayer, x: *Tensor, pos: u32) !Tensor {
        if (self.config.use_parallel_residual) {
            // Parallel residual: x + attn(ln1(x)) + mlp(ln2(x))
            var normed_attn = try self.input_layernorm.forward(x);
            defer normed_attn.deinit();

            var normed_mlp = try self.post_attention_layernorm.forward(x);
            defer normed_mlp.deinit();

            var attn_out = try self.attention.forward(&normed_attn, pos);
            defer attn_out.deinit();

            var mlp_out = try self.mlp.forward(&normed_mlp);
            defer mlp_out.deinit();

            var with_attn = try Tensor.add(self.allocator, x, &attn_out);
            defer with_attn.deinit();

            return try Tensor.add(self.allocator, &with_attn, &mlp_out);
        } else {
            // Sequential residual
            var normed1 = try self.input_layernorm.forward(x);
            defer normed1.deinit();

            var attn_out = try self.attention.forward(&normed1, pos);
            defer attn_out.deinit();

            var with_attn = try Tensor.add(self.allocator, x, &attn_out);
            defer with_attn.deinit();

            var normed2 = try self.post_attention_layernorm.forward(&with_attn);
            defer normed2.deinit();

            var mlp_out = try self.mlp.forward(&normed2);
            defer mlp_out.deinit();

            return try Tensor.add(self.allocator, &with_attn, &mlp_out);
        }
    }

    pub fn deinit(self: *GPTNeoXLayer) void {
        self.input_layernorm.deinit();
        self.post_attention_layernorm.deinit();
        self.attention.deinit();
        self.mlp.deinit();
    }
};

// Complete GPT-NeoX Model
pub const GPTNeoXModel = struct {
    config: GPTNeoXConfig,
    embed_in: Tensor,        // input embeddings
    layers: []GPTNeoXLayer,  // transformer layers
    final_layer_norm: normalization.LayerNorm, // final layer norm
    embed_out: Tensor,       // output embeddings (tied with embed_in)
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: GPTNeoXConfig) !GPTNeoXModel {
        var layers = try allocator.alloc(GPTNeoXLayer, config.n_layer);
        for (0..config.n_layer) |i| {
            layers[i] = try GPTNeoXLayer.init(allocator, config);
        }

        var embed_in = try Tensor.random(allocator, &[_]u32{ config.vocab_size, config.n_embd });

        return GPTNeoXModel{
            .config = config,
            .embed_in = embed_in,
            .layers = layers,
            .final_layer_norm = try normalization.LayerNorm.init(allocator, config.n_embd),
            .embed_out = embed_in, // tied weights
            .allocator = allocator,
        };
    }

    pub fn forward(self: *GPTNeoXModel, input_ids: []const u32, position: u32) !Tensor {
        const batch_size = 1;
        const seq_len = input_ids.len;

        // Token embedding lookup
        var x = try Tensor.zeros(self.allocator, &[_]u32{ batch_size, @intCast(seq_len), self.config.n_embd });
        for (input_ids, 0..) |token_id, i| {
            const embedding_row = try self.embed_in.getRow(token_id);
            defer embedding_row.deinit();

            for (0..self.config.n_embd) |j| {
                const val = try embedding_row.get(&[_]u32{@intCast(j)});
                try x.set(&[_]u32{ 0, @intCast(i), @intCast(j) }, val);
            }
        }

        // Pass through transformer layers
        var current = x;
        for (self.layers) |*layer| {
            var next = try layer.forward(&current, position);
            if (current.data.ptr != x.data.ptr) {
                current.deinit();
            }
            current = next;
        }

        // Final layer normalization
        var normed = try self.final_layer_norm.forward(&current);
        if (current.data.ptr != x.data.ptr) {
            current.deinit();
        }
        defer normed.deinit();

        // Output projection
        return try Tensor.matmul(self.allocator, &normed, &self.embed_out);
    }

    pub fn generate(self: *GPTNeoXModel, prompt: []const u32, max_length: u32, temperature: f32) ![]u32 {
        var result = try self.allocator.alloc(u32, max_length);
        std.mem.copy(u32, result[0..prompt.len], prompt);

        var pos: u32 = 0;
        for (prompt.len..max_length) |i| {
            var logits = try self.forward(result[0..i], pos);
            defer logits.deinit();

            const last_token_logits = try logits.getRow(@intCast(i - 1));
            defer last_token_logits.deinit();

            // Apply temperature scaling
            for (0..self.config.vocab_size) |j| {
                const val = try last_token_logits.get(&[_]u32{@intCast(j)});
                try last_token_logits.set(&[_]u32{@intCast(j)}, val / temperature);
            }

            const next_token = try sampleFromLogits(&last_token_logits);
            result[i] = next_token;

            pos += 1;
        }

        return result;
    }

    pub fn deinit(self: *GPTNeoXModel) void {
        self.embed_in.deinit();
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
        self.final_layer_norm.deinit();
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
pub const GPTNeoXUtils = struct {
    pub fn printModelInfo(config: GPTNeoXConfig) void {
        std.debug.print("=== GPT-NeoX Model Configuration ===\n");
        std.debug.print("Vocabulary Size: {}\n", .{config.vocab_size});
        std.debug.print("Context Length: {}\n", .{config.n_ctx});
        std.debug.print("Hidden Dimensions: {}\n", .{config.n_embd});
        std.debug.print("Attention Heads: {}\n", .{config.n_head});
        std.debug.print("Transformer Layers: {}\n", .{config.n_layer});
        std.debug.print("Rotary Dimensions: {}\n", .{config.rotary_dim});
        std.debug.print("Intermediate Size: {}\n", .{config.intermediate_size});
        std.debug.print("RoPE Base: {}\n", .{config.rope_base});
        std.debug.print("Parallel Residual: {}\n", .{config.use_parallel_residual});

        const total_params = calculateParameters(config);
        std.debug.print("Total Parameters: ~{:.1}B\n", .{@as(f32, @floatFromInt(total_params)) / 1_000_000_000.0});
        std.debug.print("====================================\n");
    }

    pub fn calculateParameters(config: GPTNeoXConfig) u64 {
        const embedding_params = @as(u64, config.vocab_size) * config.n_embd;
        const attention_params_per_layer = 4 * @as(u64, config.n_embd) * config.n_embd; // QKV + output
        const mlp_params_per_layer = 2 * @as(u64, config.n_embd) * config.intermediate_size; // up + down
        const norm_params_per_layer = 2 * config.n_embd; // input + post attention norm

        const layer_params = (attention_params_per_layer + mlp_params_per_layer + norm_params_per_layer) * config.n_layer;
        const final_norm_params = config.n_embd;

        return embedding_params + layer_params + final_norm_params;
    }
};