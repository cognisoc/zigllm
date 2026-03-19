const std = @import("std");
const Allocator = std.mem.Allocator;
const foundation = @import("../foundation/tensor.zig");
const neural_primitives = @import("../neural_primitives/activations.zig");
const transformers = @import("../transformers/attention.zig");
const config_mod = @import("config.zig");
const Tensor = foundation.Tensor;

/// GPT-2 model configuration variants
pub const GPT2Variant = enum {
    GPT2_124M,  // 124M parameters (small)
    GPT2_355M,  // 355M parameters (medium)
    GPT2_774M,  // 774M parameters (large)
    GPT2_1558M, // 1.5B parameters (xl)
};

/// GPT-2 specific configuration
pub const GPT2Config = struct {
    d_model: usize,         // Model dimension
    n_heads: usize,         // Number of attention heads
    n_layers: usize,        // Number of transformer layers
    vocab_size: usize,      // Vocabulary size
    max_seq_len: usize,     // Maximum sequence length
    dropout: f32,           // Dropout rate (for training)

    /// Create configuration for specific GPT-2 variant
    pub fn fromVariant(variant: GPT2Variant) GPT2Config {
        switch (variant) {
            .GPT2_124M => return GPT2Config{
                .d_model = 768,
                .n_heads = 12,
                .n_layers = 12,
                .vocab_size = 50257,  // GPT-2 vocabulary size
                .max_seq_len = 1024,
                .dropout = 0.1,
            },
            .GPT2_355M => return GPT2Config{
                .d_model = 1024,
                .n_heads = 16,
                .n_layers = 24,
                .vocab_size = 50257,
                .max_seq_len = 1024,
                .dropout = 0.1,
            },
            .GPT2_774M => return GPT2Config{
                .d_model = 1280,
                .n_heads = 20,
                .n_layers = 36,
                .vocab_size = 50257,
                .max_seq_len = 1024,
                .dropout = 0.1,
            },
            .GPT2_1558M => return GPT2Config{
                .d_model = 1600,
                .n_heads = 25,
                .n_layers = 48,
                .vocab_size = 50257,
                .max_seq_len = 1024,
                .dropout = 0.1,
            },
        }
    }

    /// Convert to generic ModelConfig for compatibility
    pub fn toModelConfig(self: GPT2Config) config_mod.ModelConfig {
        return config_mod.ModelConfig{
            .d_model = self.d_model,
            .n_heads = self.n_heads,
            .n_layers = self.n_layers,
            .vocab_size = self.vocab_size,
            .max_seq_len = self.max_seq_len,
            .intermediate_size = self.d_model * 4, // GPT-2 uses 4x expansion
            .rope_dim = 0,        // GPT-2 uses learned positional embeddings
            .rope_freq_base = 0,  // Not applicable
            .eps = 1e-5,          // LayerNorm epsilon
        };
    }
};

/// GPT-2 Transformer Block
pub const GPT2Block = struct {
    ln_1: neural_primitives.LayerNorm(f32),      // Pre-attention layer norm
    attn: transformers.MultiHeadAttention(f32),   // Multi-head self-attention
    ln_2: neural_primitives.LayerNorm(f32),      // Pre-MLP layer norm
    mlp_c_fc: Tensor(f32),                       // MLP projection to intermediate
    mlp_c_proj: Tensor(f32),                     // MLP projection back to d_model
    config: GPT2Config,
    allocator: Allocator,

    const Self = @This();

    pub fn init(config: GPT2Config, allocator: Allocator) !Self {
        const intermediate_size = config.d_model * 4;
        const head_dim = config.d_model / config.n_heads;

        // Initialize layer normalization
        var ln_1 = try neural_primitives.LayerNorm(f32).init(config.d_model, 1e-5, allocator);
        var ln_2 = try neural_primitives.LayerNorm(f32).init(config.d_model, 1e-5, allocator);

        // Initialize attention
        var attn = try transformers.MultiHeadAttention(f32).init(
            config.d_model,
            config.n_heads,
            head_dim,
            allocator
        );

        // Initialize MLP weights
        const mlp_fc_data = try allocator.alloc(f32, config.d_model * intermediate_size);
        const mlp_proj_data = try allocator.alloc(f32, intermediate_size * config.d_model);

        // Initialize with small random weights
        var rng = std.rand.DefaultPrng.init(42);
        const random = rng.random();

        const scale_fc = std.math.sqrt(2.0 / @as(f32, @floatFromInt(config.d_model)));
        const scale_proj = std.math.sqrt(2.0 / @as(f32, @floatFromInt(intermediate_size)));

        for (mlp_fc_data) |*w| {
            w.* = random.floatNorm(f32) * scale_fc;
        }

        for (mlp_proj_data) |*w| {
            w.* = random.floatNorm(f32) * scale_proj;
        }

        const mlp_c_fc = Tensor(f32){
            .data = mlp_fc_data,
            .shape = &[_]usize{ config.d_model, intermediate_size }
        };

        const mlp_c_proj = Tensor(f32){
            .data = mlp_proj_data,
            .shape = &[_]usize{ intermediate_size, config.d_model }
        };

        return Self{
            .ln_1 = ln_1,
            .attn = attn,
            .ln_2 = ln_2,
            .mlp_c_fc = mlp_c_fc,
            .mlp_c_proj = mlp_c_proj,
            .config = config,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.ln_1.deinit();
        self.ln_2.deinit();
        self.attn.deinit();
        self.allocator.free(self.mlp_c_fc.data);
        self.allocator.free(self.mlp_c_proj.data);
    }

    /// Forward pass through GPT-2 transformer block
    pub fn forward(self: *Self, input: Tensor(f32)) !Tensor(f32) {
        // Pre-attention layer norm (GPT-2 uses pre-norm)
        const normed1 = try self.ln_1.forward(input);
        defer normed1.deinit(self.allocator);

        // Multi-head self-attention with causal mask
        const causal_mask = try self.createCausalMask(input.shape[0]);
        defer causal_mask.deinit(self.allocator);

        const attn_output = try self.attn.forward(normed1, normed1, normed1, causal_mask);
        defer attn_output.deinit(self.allocator);

        // Residual connection
        const after_attn = try self.addResidual(input, attn_output);
        defer after_attn.deinit(self.allocator);

        // Pre-MLP layer norm
        const normed2 = try self.ln_2.forward(after_attn);
        defer normed2.deinit(self.allocator);

        // MLP forward pass
        const mlp_output = try self.mlpForward(normed2);
        defer mlp_output.deinit(self.allocator);

        // Final residual connection
        return try self.addResidual(after_attn, mlp_output);
    }

    /// GPT-2 MLP implementation with GELU activation
    fn mlpForward(self: *Self, input: Tensor(f32)) !Tensor(f32) {
        // First linear transformation
        const intermediate = try input.matmul(self.mlp_c_fc, self.allocator);
        defer intermediate.deinit(self.allocator);

        // GELU activation (GPT-2 uses GELU, not ReLU)
        const activated = try neural_primitives.gelu(f32, intermediate, self.allocator);
        defer activated.deinit(self.allocator);

        // Second linear transformation
        return try activated.matmul(self.mlp_c_proj, self.allocator);
    }

    /// Create causal mask for GPT-2 attention
    fn createCausalMask(self: *Self, seq_len: usize) !Tensor(f32) {
        const mask_data = try self.allocator.alloc(f32, seq_len * seq_len);
        const mask = Tensor(f32){ .data = mask_data, .shape = &[_]usize{ seq_len, seq_len } };

        // Initialize causal mask: 0 for allowed positions, -inf for masked
        for (0..seq_len) |i| {
            for (0..seq_len) |j| {
                const idx = i * seq_len + j;
                mask_data[idx] = if (j <= i) 0.0 else -std.math.inf(f32);
            }
        }

        return mask;
    }

    /// Add residual connection
    fn addResidual(self: *Self, residual: Tensor(f32), input: Tensor(f32)) !Tensor(f32) {
        const result_data = try self.allocator.alloc(f32, residual.data.len);
        const result = Tensor(f32){ .data = result_data, .shape = residual.shape };

        for (0..residual.data.len) |i| {
            result_data[i] = residual.data[i] + input.data[i];
        }

        return result;
    }
};

/// Full GPT-2 model implementation
pub const GPT2Model = struct {
    config: GPT2Config,
    token_embeddings: Tensor(f32),        // Token embeddings (vocab_size x d_model)
    position_embeddings: Tensor(f32),     // Learned position embeddings (max_seq_len x d_model)
    blocks: []GPT2Block,                  // Transformer blocks
    ln_f: neural_primitives.LayerNorm(f32), // Final layer norm
    lm_head: Tensor(f32),                 // Language modeling head (d_model x vocab_size)
    allocator: Allocator,

    const Self = @This();

    pub fn init(config: GPT2Config, allocator: Allocator) !Self {
        // Initialize token embeddings
        const token_emb_data = try allocator.alloc(f32, config.vocab_size * config.d_model);
        var rng = std.rand.DefaultPrng.init(42);
        const random = rng.random();

        // Initialize embeddings with small random values
        const emb_scale = std.math.sqrt(1.0 / @as(f32, @floatFromInt(config.d_model)));
        for (token_emb_data) |*emb| {
            emb.* = random.floatNorm(f32) * emb_scale;
        }

        const token_embeddings = Tensor(f32){
            .data = token_emb_data,
            .shape = &[_]usize{ config.vocab_size, config.d_model }
        };

        // Initialize position embeddings
        const pos_emb_data = try allocator.alloc(f32, config.max_seq_len * config.d_model);
        for (pos_emb_data) |*emb| {
            emb.* = random.floatNorm(f32) * emb_scale;
        }

        const position_embeddings = Tensor(f32){
            .data = pos_emb_data,
            .shape = &[_]usize{ config.max_seq_len, config.d_model }
        };

        // Initialize transformer blocks
        const blocks = try allocator.alloc(GPT2Block, config.n_layers);
        for (0..config.n_layers) |i| {
            blocks[i] = try GPT2Block.init(config, allocator);
        }

        // Initialize final layer norm
        var ln_f = try neural_primitives.LayerNorm(f32).init(config.d_model, 1e-5, allocator);

        // Initialize language modeling head (often tied to token embeddings)
        const lm_head_data = try allocator.alloc(f32, config.d_model * config.vocab_size);
        // Initialize LM head with same values as token embeddings (weight tying)
        @memcpy(lm_head_data, token_emb_data);

        const lm_head = Tensor(f32){
            .data = lm_head_data,
            .shape = &[_]usize{ config.d_model, config.vocab_size }
        };

        return Self{
            .config = config,
            .token_embeddings = token_embeddings,
            .position_embeddings = position_embeddings,
            .blocks = blocks,
            .ln_f = ln_f,
            .lm_head = lm_head,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.token_embeddings.data);
        self.allocator.free(self.position_embeddings.data);

        for (self.blocks) |*block| {
            block.deinit();
        }
        self.allocator.free(self.blocks);

        self.ln_f.deinit();
        self.allocator.free(self.lm_head.data);
    }

    /// Forward pass through GPT-2 model
    pub fn forward(self: *Self, input_ids: []const u32) !Tensor(f32) {
        const seq_len = input_ids.len;

        // Get token embeddings
        var token_embeds = try self.getTokenEmbeddings(input_ids);
        defer token_embeds.deinit(self.allocator);

        // Get position embeddings
        var pos_embeds = try self.getPositionEmbeddings(seq_len);
        defer pos_embeds.deinit(self.allocator);

        // Add token and position embeddings
        var hidden_states = try self.addEmbeddings(token_embeds, pos_embeds);
        defer if (hidden_states.data.ptr != token_embeds.data.ptr) hidden_states.deinit(self.allocator);

        // Pass through transformer blocks
        for (self.blocks) |*block| {
            const new_hidden = try block.forward(hidden_states);
            if (hidden_states.data.ptr != token_embeds.data.ptr) {
                hidden_states.deinit(self.allocator);
            }
            hidden_states = new_hidden;
        }

        // Final layer norm
        const normed = try self.ln_f.forward(hidden_states);
        hidden_states.deinit(self.allocator);

        // Language modeling head
        const logits = try normed.matmul(self.lm_head, self.allocator);
        normed.deinit(self.allocator);

        return logits;
    }

    /// Get token embeddings for input token IDs
    fn getTokenEmbeddings(self: *Self, input_ids: []const u32) !Tensor(f32) {
        const seq_len = input_ids.len;
        const emb_data = try self.allocator.alloc(f32, seq_len * self.config.d_model);

        for (0..seq_len) |i| {
            const token_id = input_ids[i];
            const emb_start = token_id * self.config.d_model;
            const emb_end = emb_start + self.config.d_model;

            const dest_start = i * self.config.d_model;
            const dest_end = dest_start + self.config.d_model;

            @memcpy(emb_data[dest_start..dest_end], self.token_embeddings.data[emb_start..emb_end]);
        }

        return Tensor(f32){
            .data = emb_data,
            .shape = &[_]usize{ seq_len, self.config.d_model }
        };
    }

    /// Get position embeddings for sequence
    fn getPositionEmbeddings(self: *Self, seq_len: usize) !Tensor(f32) {
        const pos_data = try self.allocator.alloc(f32, seq_len * self.config.d_model);

        for (0..seq_len) |i| {
            const pos_start = i * self.config.d_model;
            const pos_end = pos_start + self.config.d_model;

            const dest_start = i * self.config.d_model;
            const dest_end = dest_start + self.config.d_model;

            @memcpy(pos_data[dest_start..dest_end], self.position_embeddings.data[pos_start..pos_end]);
        }

        return Tensor(f32){
            .data = pos_data,
            .shape = &[_]usize{ seq_len, self.config.d_model }
        };
    }

    /// Add token and position embeddings
    fn addEmbeddings(self: *Self, token_emb: Tensor(f32), pos_emb: Tensor(f32)) !Tensor(f32) {
        const result_data = try self.allocator.alloc(f32, token_emb.data.len);

        for (0..token_emb.data.len) |i| {
            result_data[i] = token_emb.data[i] + pos_emb.data[i];
        }

        return Tensor(f32){
            .data = result_data,
            .shape = token_emb.shape
        };
    }

    /// Calculate total number of parameters
    pub fn parameterCount(self: *Self) usize {
        var total: usize = 0;

        // Token embeddings
        total += self.token_embeddings.data.len;

        // Position embeddings
        total += self.position_embeddings.data.len;

        // Transformer blocks
        const block_params = (
            // Attention weights (Q, K, V projections + output projection)
            4 * self.config.d_model * self.config.d_model +
            // MLP weights
            self.config.d_model * (self.config.d_model * 4) + // c_fc
            (self.config.d_model * 4) * self.config.d_model +  // c_proj
            // Layer norm parameters (weight + bias) x 2
            2 * 2 * self.config.d_model
        );
        total += block_params * self.config.n_layers;

        // Final layer norm
        total += 2 * self.config.d_model;

        // LM head
        total += self.lm_head.data.len;

        return total;
    }

    /// Get model info string
    pub fn getModelInfo(self: *Self, allocator: Allocator) ![]u8 {
        const param_count = self.parameterCount();
        const param_millions = @as(f32, @floatFromInt(param_count)) / 1_000_000.0;

        return std.fmt.allocPrint(allocator,
            "GPT-2 Model:\n" ++
            "  Parameters: {d:.1f}M\n" ++
            "  Layers: {d}\n" ++
            "  Heads: {d}\n" ++
            "  Model dim: {d}\n" ++
            "  Vocab size: {d}\n" ++
            "  Max sequence length: {d}\n",
            .{ param_millions, self.config.n_layers, self.config.n_heads,
               self.config.d_model, self.config.vocab_size, self.config.max_seq_len }
        );
    }
};