// GPT-J Model Architecture Implementation
// Educational implementation of the GPT-J model family (6B parameter model)
//
// Key architectural features:
// - Parallel attention and MLP blocks (unlike sequential in GPT-2)
// - Rotary positional embeddings (RoPE) instead of learned embeddings
// - No bias terms in linear layers
// - 4096 hidden dimensions, 16 attention heads, 28 layers

const std = @import("std");
const Tensor = @import("../foundation/tensor.zig").Tensor;
const attention = @import("../transformers/attention.zig");
const feed_forward = @import("../transformers/feed_forward.zig");
const normalization = @import("../neural_primitives/normalization.zig");
const activations = @import("../neural_primitives/activations.zig");
const quantization = @import("../linear_algebra/quantization.zig");

// GPT-J Model Configuration
pub const GPTJConfig = struct {
    vocab_size: u32 = 50400,
    n_ctx: u32 = 2048,        // context length
    n_embd: u32 = 4096,       // embedding dimensions
    n_head: u32 = 16,         // attention heads
    n_layer: u32 = 28,        // transformer layers
    rotary_dim: u32 = 64,     // rotary embedding dimensions
    use_parallel_residual: bool = true, // parallel vs sequential residual

    // Model variants
    pub const GPT_J_6B = GPTJConfig{
        .vocab_size = 50400,
        .n_ctx = 2048,
        .n_embd = 4096,
        .n_head = 16,
        .n_layer = 28,
        .rotary_dim = 64,
    };
};

// Rotary Positional Embedding (RoPE) Implementation
pub const RotaryEmbedding = struct {
    dim: u32,
    max_seq_len: u32,
    cos_cache: Tensor,
    sin_cache: Tensor,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, dim: u32, max_seq_len: u32) !RotaryEmbedding {
        var rope = RotaryEmbedding{
            .dim = dim,
            .max_seq_len = max_seq_len,
            .cos_cache = try Tensor.zeros(allocator, &[_]u32{ max_seq_len, dim / 2 }),
            .sin_cache = try Tensor.zeros(allocator, &[_]u32{ max_seq_len, dim / 2 }),
            .allocator = allocator,
        };

        // Precompute cos and sin values for efficiency
        try rope.precompute();
        return rope;
    }

    fn precompute(self: *RotaryEmbedding) !void {
        const base: f32 = 10000.0;

        for (0..self.max_seq_len) |pos| {
            for (0..self.dim / 2) |i| {
                const freq = 1.0 / std.math.pow(f32, base, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(self.dim)));
                const angle = @as(f32, @floatFromInt(pos)) * freq;

                try self.cos_cache.set(&[_]u32{ @intCast(pos), @intCast(i) }, @cos(angle));
                try self.sin_cache.set(&[_]u32{ @intCast(pos), @intCast(i) }, @sin(angle));
            }
        }
    }

    pub fn apply(self: *RotaryEmbedding, x: *Tensor, pos: u32) !void {
        const seq_len = x.shape[1];
        const head_dim = x.shape[3];
        const rotary_dim = @min(self.dim, head_dim);

        for (0..seq_len) |t| {
            const actual_pos = pos + t;
            if (actual_pos >= self.max_seq_len) continue;

            for (0..rotary_dim / 2) |i| {
                const cos_val = try self.cos_cache.get(&[_]u32{ @intCast(actual_pos), @intCast(i) });
                const sin_val = try self.sin_cache.get(&[_]u32{ @intCast(actual_pos), @intCast(i) });

                // Apply rotary transformation: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
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

    pub fn deinit(self: *RotaryEmbedding) void {
        self.cos_cache.deinit();
        self.sin_cache.deinit();
    }
};

// GPT-J Attention Block with RoPE
pub const GPTJAttention = struct {
    config: GPTJConfig,
    q_proj: Tensor,      // query projection (no bias)
    k_proj: Tensor,      // key projection (no bias)
    v_proj: Tensor,      // value projection (no bias)
    out_proj: Tensor,    // output projection (no bias)
    rope: RotaryEmbedding,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: GPTJConfig) !GPTJAttention {
        return GPTJAttention{
            .config = config,
            .q_proj = try Tensor.random(allocator, &[_]u32{ config.n_embd, config.n_embd }),
            .k_proj = try Tensor.random(allocator, &[_]u32{ config.n_embd, config.n_embd }),
            .v_proj = try Tensor.random(allocator, &[_]u32{ config.n_embd, config.n_embd }),
            .out_proj = try Tensor.random(allocator, &[_]u32{ config.n_embd, config.n_embd }),
            .rope = try RotaryEmbedding.init(allocator, config.rotary_dim, config.n_ctx),
            .allocator = allocator,
        };
    }

    pub fn forward(self: *GPTJAttention, x: *Tensor, pos: u32) !Tensor {
        const batch_size = x.shape[0];
        const seq_len = x.shape[1];
        const n_embd = x.shape[2];
        const head_dim = n_embd / self.config.n_head;

        // Compute Q, K, V projections
        var q = try Tensor.matmul(self.allocator, x, &self.q_proj);
        var k = try Tensor.matmul(self.allocator, x, &self.k_proj);
        var v = try Tensor.matmul(self.allocator, x, &self.v_proj);
        defer q.deinit();
        defer k.deinit();
        defer v.deinit();

        // Reshape for multi-head attention
        try q.reshape(&[_]u32{ batch_size, seq_len, self.config.n_head, head_dim });
        try k.reshape(&[_]u32{ batch_size, seq_len, self.config.n_head, head_dim });
        try v.reshape(&[_]u32{ batch_size, seq_len, self.config.n_head, head_dim });

        // Apply rotary embeddings to queries and keys
        try self.rope.apply(&q, pos);
        try self.rope.apply(&k, pos);

        // Perform scaled dot-product attention
        var attn_output = try attention.scaledDotProductAttention(
            self.allocator,
            &q, &k, &v,
            null, // no attention mask for GPT-J
            @sqrt(@as(f32, @floatFromInt(head_dim)))
        );
        defer attn_output.deinit();

        // Reshape back to original dimensions
        try attn_output.reshape(&[_]u32{ batch_size, seq_len, n_embd });

        // Apply output projection
        return try Tensor.matmul(self.allocator, &attn_output, &self.out_proj);
    }

    pub fn deinit(self: *GPTJAttention) void {
        self.q_proj.deinit();
        self.k_proj.deinit();
        self.v_proj.deinit();
        self.out_proj.deinit();
        self.rope.deinit();
    }
};

// GPT-J MLP Block (standard FFN with GELU activation)
pub const GPTJMLP = struct {
    config: GPTJConfig,
    fc_in: Tensor,       // input projection (no bias)
    fc_out: Tensor,      // output projection (no bias)
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: GPTJConfig) !GPTJMLP {
        const intermediate_size = 4 * config.n_embd; // GPT-J uses 4x expansion

        return GPTJMLP{
            .config = config,
            .fc_in = try Tensor.random(allocator, &[_]u32{ config.n_embd, intermediate_size }),
            .fc_out = try Tensor.random(allocator, &[_]u32{ intermediate_size, config.n_embd }),
            .allocator = allocator,
        };
    }

    pub fn forward(self: *GPTJMLP, x: *Tensor) !Tensor {
        // Input projection
        var hidden = try Tensor.matmul(self.allocator, x, &self.fc_in);
        defer hidden.deinit();

        // GELU activation
        try activations.gelu(&hidden);

        // Output projection
        return try Tensor.matmul(self.allocator, &hidden, &self.fc_out);
    }

    pub fn deinit(self: *GPTJMLP) void {
        self.fc_in.deinit();
        self.fc_out.deinit();
    }
};

// GPT-J Transformer Block (parallel residual connections)
pub const GPTJBlock = struct {
    config: GPTJConfig,
    ln_1: normalization.LayerNorm,  // pre-attention layer norm
    attn: GPTJAttention,
    mlp: GPTJMLP,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: GPTJConfig) !GPTJBlock {
        return GPTJBlock{
            .config = config,
            .ln_1 = try normalization.LayerNorm.init(allocator, config.n_embd),
            .attn = try GPTJAttention.init(allocator, config),
            .mlp = try GPTJMLP.init(allocator, config),
            .allocator = allocator,
        };
    }

    pub fn forward(self: *GPTJBlock, x: *Tensor, pos: u32) !Tensor {
        // Layer normalization
        var normed_x = try self.ln_1.forward(x);
        defer normed_x.deinit();

        if (self.config.use_parallel_residual) {
            // Parallel residual: x + attn(ln(x)) + mlp(ln(x))
            var attn_out = try self.attn.forward(&normed_x, pos);
            defer attn_out.deinit();

            var mlp_out = try self.mlp.forward(&normed_x);
            defer mlp_out.deinit();

            var result = try Tensor.add(self.allocator, x, &attn_out);
            var temp = try Tensor.add(self.allocator, &result, &mlp_out);
            result.deinit();
            return temp;
        } else {
            // Sequential residual: x + mlp(ln(x + attn(ln(x))))
            var attn_out = try self.attn.forward(&normed_x, pos);
            defer attn_out.deinit();

            var residual1 = try Tensor.add(self.allocator, x, &attn_out);
            defer residual1.deinit();

            var normed_residual = try self.ln_1.forward(&residual1);
            defer normed_residual.deinit();

            var mlp_out = try self.mlp.forward(&normed_residual);
            defer mlp_out.deinit();

            return try Tensor.add(self.allocator, &residual1, &mlp_out);
        }
    }

    pub fn deinit(self: *GPTJBlock) void {
        self.ln_1.deinit();
        self.attn.deinit();
        self.mlp.deinit();
    }
};

// Complete GPT-J Model
pub const GPTJModel = struct {
    config: GPTJConfig,
    wte: Tensor,         // token embeddings
    blocks: []GPTJBlock, // transformer blocks
    ln_f: normalization.LayerNorm, // final layer norm
    lm_head: Tensor,     // language modeling head (tied with wte)
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: GPTJConfig) !GPTJModel {
        var blocks = try allocator.alloc(GPTJBlock, config.n_layer);
        for (0..config.n_layer) |i| {
            blocks[i] = try GPTJBlock.init(allocator, config);
        }

        var wte = try Tensor.random(allocator, &[_]u32{ config.vocab_size, config.n_embd });

        return GPTJModel{
            .config = config,
            .wte = wte,
            .blocks = blocks,
            .ln_f = try normalization.LayerNorm.init(allocator, config.n_embd),
            .lm_head = wte, // tied weights - same tensor reference
            .allocator = allocator,
        };
    }

    pub fn forward(self: *GPTJModel, input_ids: []const u32, position: u32) !Tensor {
        const batch_size = 1;
        const seq_len = input_ids.len;

        // Token embedding lookup
        var x = try Tensor.zeros(self.allocator, &[_]u32{ batch_size, @intCast(seq_len), self.config.n_embd });
        for (input_ids, 0..) |token_id, i| {
            const embedding_row = try self.wte.getRow(token_id);
            defer embedding_row.deinit();

            for (0..self.config.n_embd) |j| {
                const val = try embedding_row.get(&[_]u32{@intCast(j)});
                try x.set(&[_]u32{ 0, @intCast(i), @intCast(j) }, val);
            }
        }

        // Pass through transformer blocks
        var current = x;
        for (self.blocks) |*block| {
            var next = try block.forward(&current, position);
            if (current.data.ptr != x.data.ptr) {
                current.deinit();
            }
            current = next;
        }

        // Final layer normalization
        var normed = try self.ln_f.forward(&current);
        if (current.data.ptr != x.data.ptr) {
            current.deinit();
        }
        defer normed.deinit();

        // Language modeling head projection
        return try Tensor.matmul(self.allocator, &normed, &self.lm_head);
    }

    pub fn generate(self: *GPTJModel, prompt: []const u32, max_length: u32, temperature: f32) ![]u32 {
        var result = try self.allocator.alloc(u32, max_length);
        std.mem.copy(u32, result[0..prompt.len], prompt);

        var pos: u32 = 0;
        for (prompt.len..max_length) |i| {
            var logits = try self.forward(result[0..i], pos);
            defer logits.deinit();

            // Sample from the last token's logits
            const last_token_logits = try logits.getRow(@intCast(i - 1));
            defer last_token_logits.deinit();

            // Apply temperature scaling
            for (0..self.config.vocab_size) |j| {
                const val = try last_token_logits.get(&[_]u32{@intCast(j)});
                try last_token_logits.set(&[_]u32{@intCast(j)}, val / temperature);
            }

            // Simple sampling (can be enhanced with top-k, top-p)
            const next_token = try sampleFromLogits(&last_token_logits);
            result[i] = next_token;

            pos += 1;
        }

        return result;
    }

    pub fn deinit(self: *GPTJModel) void {
        self.wte.deinit();
        for (self.blocks) |*block| {
            block.deinit();
        }
        self.allocator.free(self.blocks);
        self.ln_f.deinit();
        // Note: lm_head is tied to wte, so no separate deinitialization needed
    }
};

// Helper function for sampling from logits
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
pub const GPTJUtils = struct {
    pub fn printModelInfo(config: GPTJConfig) void {
        std.debug.print("=== GPT-J Model Configuration ===\n");
        std.debug.print("Vocabulary Size: {}\n", .{config.vocab_size});
        std.debug.print("Context Length: {}\n", .{config.n_ctx});
        std.debug.print("Hidden Dimensions: {}\n", .{config.n_embd});
        std.debug.print("Attention Heads: {}\n", .{config.n_head});
        std.debug.print("Transformer Layers: {}\n", .{config.n_layer});
        std.debug.print("Rotary Dimensions: {}\n", .{config.rotary_dim});
        std.debug.print("Parallel Residual: {}\n", .{config.use_parallel_residual});

        const total_params = calculateParameters(config);
        std.debug.print("Total Parameters: ~{:.1}M\n", .{@as(f32, @floatFromInt(total_params)) / 1_000_000.0});
        std.debug.print("================================\n");
    }

    pub fn calculateParameters(config: GPTJConfig) u64 {
        const embedding_params = @as(u64, config.vocab_size) * config.n_embd;
        const attention_params_per_layer = 4 * @as(u64, config.n_embd) * config.n_embd; // Q, K, V, O projections
        const mlp_params_per_layer = 2 * @as(u64, config.n_embd) * (4 * config.n_embd); // up and down projections
        const norm_params_per_layer = config.n_embd; // layer norm parameters

        const layer_params = (attention_params_per_layer + mlp_params_per_layer + norm_params_per_layer) * config.n_layer;
        const final_norm_params = config.n_embd;

        return embedding_params + layer_params + final_norm_params;
    }
};