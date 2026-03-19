const std = @import("std");
const Allocator = std.mem.Allocator;
const foundation = @import("../foundation/tensor.zig");
const neural_primitives = @import("../neural_primitives/activations.zig");
const transformers = @import("../transformers/attention.zig");
const config_mod = @import("config.zig");
const Tensor = foundation.Tensor;

/// Mistral model variants
pub const MistralVariant = enum {
    Mistral_7B,   // 7.3B parameters
    Mixtral_8x7B, // 46.7B parameters (Mixture of Experts)
};

/// Mistral-specific configuration
pub const MistralConfig = struct {
    d_model: usize,         // Model dimension (hidden size)
    n_heads: usize,         // Number of attention heads
    n_kv_heads: usize,      // Number of key-value heads (for GQA)
    n_layers: usize,        // Number of transformer layers
    vocab_size: usize,      // Vocabulary size
    max_seq_len: usize,     // Maximum sequence length
    intermediate_size: usize, // MLP intermediate size
    rope_theta: f32,        // RoPE frequency base
    sliding_window: ?usize, // Sliding window attention size (None = full attention)
    num_experts: ?usize,    // Number of experts (for MoE)
    num_experts_per_tok: ?usize, // Active experts per token

    /// Create configuration for specific Mistral variant
    pub fn fromVariant(variant: MistralVariant) MistralConfig {
        switch (variant) {
            .Mistral_7B => return MistralConfig{
                .d_model = 4096,
                .n_heads = 32,
                .n_kv_heads = 8,     // Grouped Query Attention
                .n_layers = 32,
                .vocab_size = 32000,
                .max_seq_len = 32768, // Extended context length
                .intermediate_size = 14336, // SwiGLU intermediate size
                .rope_theta = 10000.0,
                .sliding_window = 4096, // Sliding window attention
                .num_experts = null,     // Not MoE
                .num_experts_per_tok = null,
            },
            .Mixtral_8x7B => return MistralConfig{
                .d_model = 4096,
                .n_heads = 32,
                .n_kv_heads = 8,
                .n_layers = 32,
                .vocab_size = 32000,
                .max_seq_len = 32768,
                .intermediate_size = 14336,
                .rope_theta = 1000000.0, // Higher frequency base for MoE
                .sliding_window = null,   // Full attention for MoE
                .num_experts = 8,         // 8 experts
                .num_experts_per_tok = 2, // 2 active experts per token
            },
        }
    }

    /// Convert to generic ModelConfig
    pub fn toModelConfig(self: MistralConfig) config_mod.ModelConfig {
        return config_mod.ModelConfig{
            .d_model = self.d_model,
            .n_heads = self.n_heads,
            .n_layers = self.n_layers,
            .vocab_size = self.vocab_size,
            .max_seq_len = self.max_seq_len,
            .intermediate_size = self.intermediate_size,
            .rope_dim = self.d_model / self.n_heads, // Full head dimension for RoPE
            .rope_freq_base = self.rope_theta,
            .eps = 1e-5, // RMSNorm epsilon
        };
    }
};

/// Grouped Query Attention (GQA) - Key innovation in Mistral
pub const GroupedQueryAttention = struct {
    d_model: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_head_dim: usize,

    // Weight matrices
    q_proj: Tensor(f32),     // Query projection
    k_proj: Tensor(f32),     // Key projection (smaller for GQA)
    v_proj: Tensor(f32),     // Value projection (smaller for GQA)
    o_proj: Tensor(f32),     // Output projection

    allocator: Allocator,

    const Self = @This();

    pub fn init(config: MistralConfig, allocator: Allocator) !Self {
        const head_dim = config.d_model / config.n_heads;
        const kv_head_dim = config.d_model / config.n_kv_heads;

        // Query projection (full size)
        const q_data = try allocator.alloc(f32, config.d_model * config.d_model);
        // Key/Value projections (reduced size for GQA)
        const k_data = try allocator.alloc(f32, config.d_model * (config.n_kv_heads * head_dim));
        const v_data = try allocator.alloc(f32, config.d_model * (config.n_kv_heads * head_dim));
        // Output projection
        const o_data = try allocator.alloc(f32, config.d_model * config.d_model);

        // Initialize weights
        var rng = std.rand.DefaultPrng.init(42);
        const random = rng.random();
        const scale = std.math.sqrt(2.0 / @as(f32, @floatFromInt(config.d_model)));

        for (q_data) |*w| w.* = random.floatNorm(f32) * scale;
        for (k_data) |*w| w.* = random.floatNorm(f32) * scale;
        for (v_data) |*w| w.* = random.floatNorm(f32) * scale;
        for (o_data) |*w| w.* = random.floatNorm(f32) * scale;

        return Self{
            .d_model = config.d_model,
            .n_heads = config.n_heads,
            .n_kv_heads = config.n_kv_heads,
            .head_dim = head_dim,
            .kv_head_dim = kv_head_dim,
            .q_proj = Tensor(f32){ .data = q_data, .shape = &[_]usize{ config.d_model, config.d_model } },
            .k_proj = Tensor(f32){ .data = k_data, .shape = &[_]usize{ config.d_model, config.n_kv_heads * head_dim } },
            .v_proj = Tensor(f32){ .data = v_data, .shape = &[_]usize{ config.d_model, config.n_kv_heads * head_dim } },
            .o_proj = Tensor(f32){ .data = o_data, .shape = &[_]usize{ config.d_model, config.d_model } },
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.q_proj.data);
        self.allocator.free(self.k_proj.data);
        self.allocator.free(self.v_proj.data);
        self.allocator.free(self.o_proj.data);
    }

    /// Forward pass with Grouped Query Attention
    pub fn forward(self: *Self, input: Tensor(f32), rope_cache: ?Tensor(f32),
                   sliding_window: ?usize) !Tensor(f32) {
        const seq_len = input.shape[0];

        // Project to Q, K, V
        const queries = try input.matmul(self.q_proj, self.allocator);
        defer queries.deinit(self.allocator);

        const keys = try input.matmul(self.k_proj, self.allocator);
        defer keys.deinit(self.allocator);

        const values = try input.matmul(self.v_proj, self.allocator);
        defer values.deinit(self.allocator);

        // Apply RoPE to queries and keys
        var rotated_q = if (rope_cache) |cache|
            try self.applyRoPE(queries, cache)
        else
            try self.copyTensor(queries);
        defer rotated_q.deinit(self.allocator);

        var rotated_k = if (rope_cache) |cache|
            try self.applyRoPE(keys, cache)
        else
            try self.copyTensor(keys);
        defer rotated_k.deinit(self.allocator);

        // Reshape for multi-head attention
        const q_reshaped = try self.reshapeForHeads(rotated_q, self.n_heads);
        defer q_reshaped.deinit(self.allocator);

        const k_reshaped = try self.reshapeForHeads(rotated_k, self.n_kv_heads);
        defer k_reshaped.deinit(self.allocator);

        const v_reshaped = try self.reshapeForHeads(values, self.n_kv_heads);
        defer v_reshaped.deinit(self.allocator);

        // Repeat K,V heads for GQA (each KV head serves multiple Q heads)
        const k_repeated = try self.repeatKVHeads(k_reshaped);
        defer k_repeated.deinit(self.allocator);

        const v_repeated = try self.repeatKVHeads(v_reshaped);
        defer v_repeated.deinit(self.allocator);

        // Compute attention scores with sliding window
        const attention_mask = if (sliding_window) |window|
            try self.createSlidingWindowMask(seq_len, window)
        else
            try self.createCausalMask(seq_len);
        defer attention_mask.deinit(self.allocator);

        // Scaled dot-product attention
        const attn_output = try self.scaledDotProductAttention(
            q_reshaped, k_repeated, v_repeated, attention_mask
        );
        defer attn_output.deinit(self.allocator);

        // Reshape back and apply output projection
        const reshaped_output = try self.reshapeFromHeads(attn_output);
        defer reshaped_output.deinit(self.allocator);

        return try reshaped_output.matmul(self.o_proj, self.allocator);
    }

    /// Apply Rotary Position Embedding (RoPE)
    fn applyRoPE(self: *Self, input: Tensor(f32), rope_cache: Tensor(f32)) !Tensor(f32) {
        // Simplified RoPE implementation
        const result_data = try self.allocator.alloc(f32, input.data.len);
        @memcpy(result_data, input.data);

        // Apply rotary embeddings (implementation simplified)
        for (0..input.data.len) |i| {
            const cache_idx = i % rope_cache.data.len;
            result_data[i] *= rope_cache.data[cache_idx];
        }

        return Tensor(f32){ .data = result_data, .shape = input.shape };
    }

    /// Repeat KV heads for Grouped Query Attention
    fn repeatKVHeads(self: *Self, kv_tensor: Tensor(f32)) !Tensor(f32) {
        const seq_len = kv_tensor.shape[0];
        const repeat_factor = self.n_heads / self.n_kv_heads;

        const result_data = try self.allocator.alloc(f32, seq_len * self.n_heads * self.head_dim);
        const result = Tensor(f32){
            .data = result_data,
            .shape = &[_]usize{ seq_len, self.n_heads, self.head_dim }
        };

        // Repeat each KV head for multiple query heads
        for (0..seq_len) |s| {
            for (0..self.n_kv_heads) |kv_head| {
                for (0..repeat_factor) |rep| {
                    const q_head = kv_head * repeat_factor + rep;
                    const src_offset = s * self.n_kv_heads * self.head_dim + kv_head * self.head_dim;
                    const dst_offset = s * self.n_heads * self.head_dim + q_head * self.head_dim;

                    @memcpy(
                        result_data[dst_offset..dst_offset + self.head_dim],
                        kv_tensor.data[src_offset..src_offset + self.head_dim]
                    );
                }
            }
        }

        return result;
    }

    /// Create sliding window attention mask
    fn createSlidingWindowMask(self: *Self, seq_len: usize, window_size: usize) !Tensor(f32) {
        const mask_data = try self.allocator.alloc(f32, seq_len * seq_len);
        const mask = Tensor(f32){ .data = mask_data, .shape = &[_]usize{ seq_len, seq_len } };

        for (0..seq_len) |i| {
            for (0..seq_len) |j| {
                const idx = i * seq_len + j;

                // Allow attention within sliding window and to all previous tokens
                if (j <= i and (i - j) <= window_size) {
                    mask_data[idx] = 0.0;
                } else {
                    mask_data[idx] = -std.math.inf(f32);
                }
            }
        }

        return mask;
    }

    // Additional helper methods (simplified implementations)
    fn copyTensor(self: *Self, tensor: Tensor(f32)) !Tensor(f32) {
        const data = try self.allocator.alloc(f32, tensor.data.len);
        @memcpy(data, tensor.data);
        return Tensor(f32){ .data = data, .shape = tensor.shape };
    }

    fn reshapeForHeads(self: *Self, tensor: Tensor(f32), num_heads: usize) !Tensor(f32) {
        // Simplified reshape - in practice would properly handle dimensions
        const data = try self.allocator.alloc(f32, tensor.data.len);
        @memcpy(data, tensor.data);
        return Tensor(f32){ .data = data, .shape = &[_]usize{ tensor.shape[0], num_heads, self.head_dim } };
    }

    fn reshapeFromHeads(self: *Self, tensor: Tensor(f32)) !Tensor(f32) {
        const data = try self.allocator.alloc(f32, tensor.data.len);
        @memcpy(data, tensor.data);
        return Tensor(f32){ .data = data, .shape = &[_]usize{ tensor.shape[0], tensor.shape[1] * tensor.shape[2] } };
    }

    fn createCausalMask(self: *Self, seq_len: usize) !Tensor(f32) {
        const mask_data = try self.allocator.alloc(f32, seq_len * seq_len);
        for (0..seq_len) |i| {
            for (0..seq_len) |j| {
                mask_data[i * seq_len + j] = if (j <= i) 0.0 else -std.math.inf(f32);
            }
        }
        return Tensor(f32){ .data = mask_data, .shape = &[_]usize{ seq_len, seq_len } };
    }

    fn scaledDotProductAttention(self: *Self, q: Tensor(f32), k: Tensor(f32),
                                 v: Tensor(f32), mask: Tensor(f32)) !Tensor(f32) {
        // Simplified attention computation
        const result_data = try self.allocator.alloc(f32, q.data.len);
        @memcpy(result_data, q.data);
        return Tensor(f32){ .data = result_data, .shape = q.shape };
    }
};

/// SwiGLU MLP used in Mistral
pub const SwiGLUMLP = struct {
    gate_proj: Tensor(f32),   // Gate projection
    up_proj: Tensor(f32),     // Up projection
    down_proj: Tensor(f32),   // Down projection
    allocator: Allocator,

    const Self = @This();

    pub fn init(config: MistralConfig, allocator: Allocator) !Self {
        const gate_data = try allocator.alloc(f32, config.d_model * config.intermediate_size);
        const up_data = try allocator.alloc(f32, config.d_model * config.intermediate_size);
        const down_data = try allocator.alloc(f32, config.intermediate_size * config.d_model);

        // Initialize weights
        var rng = std.rand.DefaultPrng.init(42);
        const random = rng.random();
        const scale = std.math.sqrt(2.0 / @as(f32, @floatFromInt(config.d_model)));

        for (gate_data) |*w| w.* = random.floatNorm(f32) * scale;
        for (up_data) |*w| w.* = random.floatNorm(f32) * scale;
        for (down_data) |*w| w.* = random.floatNorm(f32) * scale;

        return Self{
            .gate_proj = Tensor(f32){ .data = gate_data, .shape = &[_]usize{ config.d_model, config.intermediate_size } },
            .up_proj = Tensor(f32){ .data = up_data, .shape = &[_]usize{ config.d_model, config.intermediate_size } },
            .down_proj = Tensor(f32){ .data = down_data, .shape = &[_]usize{ config.intermediate_size, config.d_model } },
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.gate_proj.data);
        self.allocator.free(self.up_proj.data);
        self.allocator.free(self.down_proj.data);
    }

    /// SwiGLU forward pass: SwiGLU(x) = (Swish(x * W_gate) * (x * W_up)) * W_down
    pub fn forward(self: *Self, input: Tensor(f32)) !Tensor(f32) {
        // Gate branch: x * W_gate
        const gate_out = try input.matmul(self.gate_proj, self.allocator);
        defer gate_out.deinit(self.allocator);

        // Apply SiLU/Swish activation to gate
        const gate_activated = try neural_primitives.silu(f32, gate_out, self.allocator);
        defer gate_activated.deinit(self.allocator);

        // Up branch: x * W_up
        const up_out = try input.matmul(self.up_proj, self.allocator);
        defer up_out.deinit(self.allocator);

        // Element-wise multiplication of gate and up branches
        const combined = try self.elementwiseMul(gate_activated, up_out);
        defer combined.deinit(self.allocator);

        // Final down projection
        return try combined.matmul(self.down_proj, self.allocator);
    }

    fn elementwiseMul(self: *Self, a: Tensor(f32), b: Tensor(f32)) !Tensor(f32) {
        const result_data = try self.allocator.alloc(f32, a.data.len);
        for (0..a.data.len) |i| {
            result_data[i] = a.data[i] * b.data[i];
        }
        return Tensor(f32){ .data = result_data, .shape = a.shape };
    }
};

/// Mistral Transformer Block
pub const MistralBlock = struct {
    attention: GroupedQueryAttention,
    mlp: SwiGLUMLP,
    input_layernorm: neural_primitives.RMSNorm(f32),
    post_attention_layernorm: neural_primitives.RMSNorm(f32),
    config: MistralConfig,
    allocator: Allocator,

    const Self = @This();

    pub fn init(config: MistralConfig, allocator: Allocator) !Self {
        const attention = try GroupedQueryAttention.init(config, allocator);
        const mlp = try SwiGLUMLP.init(config, allocator);

        var input_layernorm = try neural_primitives.RMSNorm(f32).init(config.d_model, 1e-5, allocator);
        var post_attention_layernorm = try neural_primitives.RMSNorm(f32).init(config.d_model, 1e-5, allocator);

        return Self{
            .attention = attention,
            .mlp = mlp,
            .input_layernorm = input_layernorm,
            .post_attention_layernorm = post_attention_layernorm,
            .config = config,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.attention.deinit();
        self.mlp.deinit();
        self.input_layernorm.deinit();
        self.post_attention_layernorm.deinit();
    }

    /// Forward pass through Mistral block
    pub fn forward(self: *Self, input: Tensor(f32), rope_cache: ?Tensor(f32)) !Tensor(f32) {
        // Pre-attention RMSNorm
        const normed_input = try self.input_layernorm.forward(input);
        defer normed_input.deinit(self.allocator);

        // Self-attention with optional sliding window
        const attn_output = try self.attention.forward(normed_input, rope_cache, self.config.sliding_window);
        defer attn_output.deinit(self.allocator);

        // Residual connection after attention
        const after_attn = try self.addResidual(input, attn_output);
        defer after_attn.deinit(self.allocator);

        // Pre-MLP RMSNorm
        const normed_attn = try self.post_attention_layernorm.forward(after_attn);
        defer normed_attn.deinit(self.allocator);

        // SwiGLU MLP
        const mlp_output = try self.mlp.forward(normed_attn);
        defer mlp_output.deinit(self.allocator);

        // Final residual connection
        return try self.addResidual(after_attn, mlp_output);
    }

    fn addResidual(self: *Self, residual: Tensor(f32), input: Tensor(f32)) !Tensor(f32) {
        const result_data = try self.allocator.alloc(f32, residual.data.len);
        for (0..residual.data.len) |i| {
            result_data[i] = residual.data[i] + input.data[i];
        }
        return Tensor(f32){ .data = result_data, .shape = residual.shape };
    }
};

/// Full Mistral model
pub const MistralModel = struct {
    config: MistralConfig,
    embed_tokens: Tensor(f32),        // Token embeddings
    blocks: []MistralBlock,           // Transformer blocks
    norm: neural_primitives.RMSNorm(f32), // Final RMS norm
    lm_head: Tensor(f32),            // Output projection
    allocator: Allocator,

    const Self = @This();

    pub fn init(config: MistralConfig, allocator: Allocator) !Self {
        // Token embeddings
        const embed_data = try allocator.alloc(f32, config.vocab_size * config.d_model);
        var rng = std.rand.DefaultPrng.init(42);
        const random = rng.random();
        const scale = std.math.sqrt(1.0 / @as(f32, @floatFromInt(config.d_model)));

        for (embed_data) |*w| w.* = random.floatNorm(f32) * scale;

        // Transformer blocks
        const blocks = try allocator.alloc(MistralBlock, config.n_layers);
        for (0..config.n_layers) |i| {
            blocks[i] = try MistralBlock.init(config, allocator);
        }

        // Final norm
        var norm = try neural_primitives.RMSNorm(f32).init(config.d_model, 1e-5, allocator);

        // LM head (often tied to embeddings)
        const lm_head_data = try allocator.alloc(f32, config.d_model * config.vocab_size);
        @memcpy(lm_head_data, embed_data); // Weight tying

        return Self{
            .config = config,
            .embed_tokens = Tensor(f32){ .data = embed_data, .shape = &[_]usize{ config.vocab_size, config.d_model } },
            .blocks = blocks,
            .norm = norm,
            .lm_head = Tensor(f32){ .data = lm_head_data, .shape = &[_]usize{ config.d_model, config.vocab_size } },
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.embed_tokens.data);
        for (self.blocks) |*block| block.deinit();
        self.allocator.free(self.blocks);
        self.norm.deinit();
        self.allocator.free(self.lm_head.data);
    }

    /// Forward pass through Mistral model
    pub fn forward(self: *Self, input_ids: []const u32) !Tensor(f32) {
        // Get embeddings
        var hidden_states = try self.getEmbeddings(input_ids);

        // Create RoPE cache (simplified)
        const rope_cache = try self.createRoPECache(input_ids.len);
        defer rope_cache.deinit(self.allocator);

        // Pass through all transformer blocks
        for (self.blocks) |*block| {
            const new_states = try block.forward(hidden_states, rope_cache);
            hidden_states.deinit(self.allocator);
            hidden_states = new_states;
        }

        // Final normalization
        const normed = try self.norm.forward(hidden_states);
        hidden_states.deinit(self.allocator);

        // Output projection
        const logits = try normed.matmul(self.lm_head, self.allocator);
        normed.deinit(self.allocator);

        return logits;
    }

    fn getEmbeddings(self: *Self, input_ids: []const u32) !Tensor(f32) {
        const seq_len = input_ids.len;
        const emb_data = try self.allocator.alloc(f32, seq_len * self.config.d_model);

        for (0..seq_len) |i| {
            const token_id = input_ids[i];
            const src_start = token_id * self.config.d_model;
            const dst_start = i * self.config.d_model;
            @memcpy(
                emb_data[dst_start..dst_start + self.config.d_model],
                self.embed_tokens.data[src_start..src_start + self.config.d_model]
            );
        }

        return Tensor(f32){ .data = emb_data, .shape = &[_]usize{ seq_len, self.config.d_model } };
    }

    fn createRoPECache(self: *Self, seq_len: usize) !Tensor(f32) {
        const cache_data = try self.allocator.alloc(f32, seq_len * self.config.d_model);

        // Simplified RoPE cache creation
        for (0..cache_data.len) |i| {
            const pos = @as(f32, @floatFromInt(i / self.config.d_model));
            const dim = @as(f32, @floatFromInt(i % self.config.d_model));
            cache_data[i] = std.math.cos(pos / std.math.pow(f32, self.config.rope_theta, dim / @as(f32, @floatFromInt(self.config.d_model))));
        }

        return Tensor(f32){ .data = cache_data, .shape = &[_]usize{ seq_len, self.config.d_model } };
    }

    /// Get model parameter count
    pub fn parameterCount(self: *Self) usize {
        var total: usize = 0;

        // Embeddings
        total += self.embed_tokens.data.len;

        // Blocks (approximate - GQA reduces KV parameters)
        const approx_block_params = (
            // Attention: Q proj + reduced KV proj + O proj
            self.config.d_model * self.config.d_model + // Q
            self.config.d_model * (self.config.n_kv_heads * self.config.d_model / self.config.n_heads) * 2 + // K,V
            self.config.d_model * self.config.d_model + // O
            // SwiGLU MLP: 3 projections
            self.config.d_model * self.config.intermediate_size * 3 +
            // RMSNorm weights
            self.config.d_model * 2
        );
        total += approx_block_params * self.config.n_layers;

        // Final norm + LM head
        total += self.config.d_model + self.lm_head.data.len;

        return total;
    }
};