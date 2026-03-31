//! Model Configuration System
//!
//! This module provides comprehensive configuration management for different
//! LLaMA model variants and architectures. It handles the complex parameter
//! relationships that define transformer models.
//!
//! ## Educational Value
//! Understanding model configurations is crucial for transformer architectures:
//! - How parameter scaling affects model capacity
//! - Memory and compute trade-offs between model sizes
//! - Architecture variants and their performance characteristics

const std = @import("std");
const math = std.math;

/// Different LLaMA model sizes with their canonical configurations
pub const ModelSize = enum {
    LLaMA_7B,
    LLaMA_13B,
    LLaMA_30B,
    LLaMA_65B,
    CodeLlama_7B,
    CodeLlama_13B,
    CodeLlama_34B,

    /// Get human-readable model name
    pub fn name(self: ModelSize) []const u8 {
        return switch (self) {
            .LLaMA_7B => "LLaMA-7B",
            .LLaMA_13B => "LLaMA-13B",
            .LLaMA_30B => "LLaMA-30B",
            .LLaMA_65B => "LLaMA-65B",
            .CodeLlama_7B => "CodeLlama-7B",
            .CodeLlama_13B => "CodeLlama-13B",
            .CodeLlama_34B => "CodeLlama-34B",
        };
    }

    /// Get approximate parameter count in billions
    pub fn parameterCount(self: ModelSize) f32 {
        return switch (self) {
            .LLaMA_7B, .CodeLlama_7B => 6.7,
            .LLaMA_13B, .CodeLlama_13B => 13.0,
            .LLaMA_30B => 30.0,
            .CodeLlama_34B => 34.0,
            .LLaMA_65B => 65.2,
        };
    }
};

/// Activation function types used in feed-forward networks
pub const ActivationType = enum {
    /// Standard ReLU activation
    ReLU,
    /// Gaussian Error Linear Unit (used in BERT, GPT)
    GELU,
    /// Swish-Gated Linear Unit (used in LLaMA)
    SwiGLU,
    /// GELU-Gated Linear Unit
    GeGLU,
    /// Standard GLU
    GLU,

    /// Get the parameter multiplier for gated activations
    /// Gated activations need extra parameters for the gate
    pub fn parameterMultiplier(self: ActivationType) f32 {
        return switch (self) {
            .ReLU, .GELU => 2.0,  // Standard FFN: up + down
            .SwiGLU, .GeGLU, .GLU => 3.0,  // Gated FFN: gate + up + down
        };
    }
};

/// Normalization layer types
pub const NormalizationType = enum {
    /// Layer Normalization (original Transformer)
    LayerNorm,
    /// Root Mean Square Normalization (LLaMA)
    RMSNorm,
    /// No normalization
    None,

    /// Whether this normalization type requires learnable parameters
    pub fn hasParameters(self: NormalizationType) bool {
        return switch (self) {
            .LayerNorm, .RMSNorm => true,
            .None => false,
        };
    }
};

/// Position encoding strategy
pub const PositionEncodingType = enum {
    /// No positional encoding
    None,
    /// Sinusoidal position embeddings (original Transformer)
    Sinusoidal,
    /// Learned position embeddings
    Learned,
    /// Rotary Position Embeddings (RoPE)
    RoPE,
    /// Alibi position bias
    ALiBi,

    /// Whether this encoding requires additional parameters
    pub fn requiresParameters(self: PositionEncodingType) bool {
        return switch (self) {
            .None, .Sinusoidal, .ALiBi => false,
            .Learned, .RoPE => true,
        };
    }
};

/// Comprehensive model configuration
pub const ModelConfig = struct {
    // Model Architecture
    d_model: usize,          // Hidden dimension
    num_layers: usize,       // Number of transformer layers
    num_heads: usize,        // Number of attention heads
    d_ff: usize,            // Feed-forward dimension

    // Vocabulary and Sequence
    vocab_size: usize,       // Vocabulary size
    max_seq_len: usize,      // Maximum sequence length

    // Specialized Components
    activation: ActivationType,      // FFN activation function
    normalization: NormalizationType, // Normalization type
    position_encoding: PositionEncodingType, // Position encoding strategy

    // Numerical Parameters
    norm_eps: f32,           // Epsilon for numerical stability
    dropout_rate: f32,       // Dropout rate (if used)

    // RoPE-specific parameters
    rope_theta: f32,         // RoPE base frequency
    rope_scaling: f32,       // RoPE scaling factor

    // Attention-specific
    attention_bias: bool,    // Whether attention uses bias
    qkv_bias: bool,         // Whether QKV projections use bias

    // Memory optimization
    gradient_checkpointing: bool,  // Trade compute for memory
    use_flash_attention: bool,     // Use memory-efficient attention

    /// Create configuration for standard LLaMA model sizes
    pub fn llama(size: ModelSize) ModelConfig {
        return switch (size) {
            .LLaMA_7B => ModelConfig{
                .d_model = 4096,
                .num_layers = 32,
                .num_heads = 32,
                .d_ff = 11008,
                .vocab_size = 32000,
                .max_seq_len = 2048,
                .activation = .SwiGLU,
                .normalization = .RMSNorm,
                .position_encoding = .RoPE,
                .norm_eps = 1e-6,
                .dropout_rate = 0.0,
                .rope_theta = 10000.0,
                .rope_scaling = 1.0,
                .attention_bias = false,
                .qkv_bias = false,
                .gradient_checkpointing = false,
                .use_flash_attention = false,
            },
            .LLaMA_13B => ModelConfig{
                .d_model = 5120,
                .num_layers = 40,
                .num_heads = 40,
                .d_ff = 13824,
                .vocab_size = 32000,
                .max_seq_len = 2048,
                .activation = .SwiGLU,
                .normalization = .RMSNorm,
                .position_encoding = .RoPE,
                .norm_eps = 1e-6,
                .dropout_rate = 0.0,
                .rope_theta = 10000.0,
                .rope_scaling = 1.0,
                .attention_bias = false,
                .qkv_bias = false,
                .gradient_checkpointing = false,
                .use_flash_attention = false,
            },
            .LLaMA_30B => ModelConfig{
                .d_model = 6656,
                .num_layers = 60,
                .num_heads = 52,
                .d_ff = 17920,
                .vocab_size = 32000,
                .max_seq_len = 2048,
                .activation = .SwiGLU,
                .normalization = .RMSNorm,
                .position_encoding = .RoPE,
                .norm_eps = 1e-6,
                .dropout_rate = 0.0,
                .rope_theta = 10000.0,
                .rope_scaling = 1.0,
                .attention_bias = false,
                .qkv_bias = false,
                .gradient_checkpointing = true, // Enable for large models
                .use_flash_attention = true,
            },
            .LLaMA_65B => ModelConfig{
                .d_model = 8192,
                .num_layers = 80,
                .num_heads = 64,
                .d_ff = 22016,
                .vocab_size = 32000,
                .max_seq_len = 2048,
                .activation = .SwiGLU,
                .normalization = .RMSNorm,
                .position_encoding = .RoPE,
                .norm_eps = 1e-6,
                .dropout_rate = 0.0,
                .rope_theta = 10000.0,
                .rope_scaling = 1.0,
                .attention_bias = false,
                .qkv_bias = false,
                .gradient_checkpointing = true,
                .use_flash_attention = true,
            },
            .CodeLlama_7B => blk: {
                var config = ModelConfig.llama(.LLaMA_7B);
                config.max_seq_len = 16384; // Code models support longer sequences
                config.rope_scaling = 1.0;
                break :blk config;
            },
            .CodeLlama_13B => blk: {
                var config = ModelConfig.llama(.LLaMA_13B);
                config.max_seq_len = 16384;
                config.rope_scaling = 1.0;
                break :blk config;
            },
            .CodeLlama_34B => blk: {
                var config = ModelConfig.llama(.LLaMA_30B);
                config.d_model = 8192;
                config.num_layers = 48;
                config.num_heads = 64;
                config.d_ff = 22016;
                config.max_seq_len = 16384;
                config.rope_scaling = 1.0;
                break :blk config;
            },
        };
    }

    /// Create custom configuration
    pub fn custom(
        d_model: usize,
        num_layers: usize,
        num_heads: usize,
        vocab_size: usize
    ) ModelConfig {
        return ModelConfig{
            .d_model = d_model,
            .num_layers = num_layers,
            .num_heads = num_heads,
            .d_ff = 4 * d_model, // Standard 4x scaling
            .vocab_size = vocab_size,
            .max_seq_len = 2048,
            .activation = .SwiGLU,
            .normalization = .RMSNorm,
            .position_encoding = .RoPE,
            .norm_eps = 1e-6,
            .dropout_rate = 0.0,
            .rope_theta = 10000.0,
            .rope_scaling = 1.0,
            .attention_bias = false,
            .qkv_bias = false,
            .gradient_checkpointing = false,
            .use_flash_attention = false,
        };
    }

    /// Validate configuration for common issues
    pub fn validate(self: ModelConfig) !void {
        // Check dimension compatibility
        if (self.d_model % self.num_heads != 0) {
            return error.IncompatibleHeadDimension;
        }

        // Check reasonable ranges
        if (self.d_model < 64 or self.d_model > 32768) {
            return error.UnreasonableModelDimension;
        }

        if (self.num_layers < 1 or self.num_layers > 1000) {
            return error.UnreasonableLayerCount;
        }

        if (self.num_heads < 1 or self.num_heads > 256) {
            return error.UnreasonableHeadCount;
        }

        if (self.vocab_size < 1000 or self.vocab_size > 1000000) {
            return error.UnreasonableVocabSize;
        }

        // Check numerical stability
        if (self.norm_eps < 1e-12 or self.norm_eps > 1e-3) {
            return error.UnreasonableEpsilon;
        }
    }

    /// Calculate head dimension
    pub fn headDim(self: ModelConfig) usize {
        return self.d_model / self.num_heads;
    }

    /// Calculate total parameter count
    pub fn parameterCount(self: ModelConfig) usize {
        // Embedding parameters
        const embedding_params = self.vocab_size * self.d_model;

        // Per-layer parameters
        var layer_params: usize = 0;

        // Attention parameters
        const qkv_params = 3 * self.d_model * self.d_model;
        const attention_output_params = self.d_model * self.d_model;
        layer_params += qkv_params + attention_output_params;

        // FFN parameters
        const ffn_multiplier = self.activation.parameterMultiplier();
        const ffn_params = @as(usize, @intFromFloat(ffn_multiplier * @as(f32, @floatFromInt(self.d_model * self.d_ff))));
        layer_params += ffn_params;

        // Normalization parameters (if applicable)
        if (self.normalization.hasParameters()) {
            layer_params += 2 * self.d_model; // Pre and post norm
        }

        // Position encoding parameters (if applicable)
        var position_params: usize = 0;
        if (self.position_encoding.requiresParameters()) {
            switch (self.position_encoding) {
                .Learned => position_params = self.max_seq_len * self.d_model,
                .RoPE => position_params = self.d_model, // Frequency parameters
                else => {},
            }
        }

        // Final layer norm
        const final_norm_params = if (self.normalization.hasParameters()) self.d_model else 0;

        return embedding_params +
               (layer_params * self.num_layers) +
               position_params +
               final_norm_params;
    }

    /// Estimate memory requirements in bytes for inference
    pub fn memoryRequirements(self: ModelConfig, batch_size: usize, sequence_length: usize) struct {
        parameters: usize,
        activations: usize,
        kv_cache: usize,
        total: usize,
    } {
        const param_count = self.parameterCount();
        const param_memory = param_count * @sizeOf(f32);

        // Activation memory (rough estimate)
        const activation_memory = batch_size * sequence_length * self.d_model * self.num_layers * @sizeOf(f32);

        // KV cache memory
        const kv_cache_memory = 2 * batch_size * self.num_heads * sequence_length * self.headDim() * self.num_layers * @sizeOf(f32);

        const total = param_memory + activation_memory + kv_cache_memory;

        return .{
            .parameters = param_memory,
            .activations = activation_memory,
            .kv_cache = kv_cache_memory,
            .total = total,
        };
    }

    /// Format memory size in human-readable format
    pub fn formatMemorySize(bytes: usize, buffer: []u8) ![]u8 {
        const kb = 1024;
        const mb = kb * 1024;
        const gb = mb * 1024;

        if (bytes >= gb) {
            return try std.fmt.bufPrint(buffer, "{d:.1} GB", .{@as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(gb))});
        } else if (bytes >= mb) {
            return try std.fmt.bufPrint(buffer, "{d:.1} MB", .{@as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(mb))});
        } else if (bytes >= kb) {
            return try std.fmt.bufPrint(buffer, "{d:.1} KB", .{@as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(kb))});
        } else {
            return try std.fmt.bufPrint(buffer, "{d} bytes", .{bytes});
        }
    }

    /// Print detailed model information
    pub fn print(self: ModelConfig, writer: anytype) !void {
        try writer.print("Model Configuration:\n", .{});
        try writer.print("  Architecture: {d} layers, {d} heads, {d} dim, {d} ff\n",
            .{self.num_layers, self.num_heads, self.d_model, self.d_ff});
        try writer.print("  Vocabulary: {d} tokens, max sequence: {d}\n",
            .{self.vocab_size, self.max_seq_len});
        try writer.print("  Activation: {s}, Normalization: {s}\n",
            .{@tagName(self.activation), @tagName(self.normalization)});
        try writer.print("  Position Encoding: {s}\n", .{@tagName(self.position_encoding)});

        const params = self.parameterCount();
        try writer.print("  Parameters: {d} ({d:.1}B)\n", .{params, @as(f32, @floatFromInt(params)) / 1e9});

        const memory = self.memoryRequirements(1, self.max_seq_len);
        var buffer: [64]u8 = undefined;
        const param_str = try self.formatMemorySize(memory.parameters, buffer[0..32]);
        const total_str = try self.formatMemorySize(memory.total, buffer[32..]);
        try writer.print("  Memory (inference): parameters {s}, total ~{s}\n", .{param_str, total_str});
    }
};

// Configuration validation tests
test "model configuration validation" {
    const testing = std.testing;

    // Valid configuration should pass
    const valid_config = ModelConfig.llama(.LLaMA_7B);
    try valid_config.validate();

    // Invalid head dimension should fail
    var invalid_config = valid_config;
    invalid_config.d_model = 100;
    invalid_config.num_heads = 7;
    try testing.expectError(error.IncompatibleHeadDimension, invalid_config.validate());
}

test "parameter counting accuracy" {
    const testing = std.testing;

    const config = ModelConfig.llama(.LLaMA_7B);
    const params = config.parameterCount();

    // LLaMA-7B should have approximately 6.7B parameters
    const expected_range_min: usize = 6_000_000_000; // 6B
    const expected_range_max: usize = 7_500_000_000; // 7.5B

    try testing.expect(params >= expected_range_min);
    try testing.expect(params <= expected_range_max);
}

test "memory requirement estimation" {
    const testing = std.testing;

    const config = ModelConfig.llama(.LLaMA_7B);
    const memory = config.memoryRequirements(1, 512);

    // Parameters should be the largest component
    try testing.expect(memory.parameters > 0);
    try testing.expect(memory.activations > 0);
    try testing.expect(memory.kv_cache > 0);
    try testing.expect(memory.total >= memory.parameters);
    try testing.expect(memory.total >= memory.activations);
    try testing.expect(memory.total >= memory.kv_cache);
}

test "configuration variants" {
    const testing = std.testing;

    // Test different model sizes
    const llama_7b = ModelConfig.llama(.LLaMA_7B);
    const llama_13b = ModelConfig.llama(.LLaMA_13B);
    const code_llama_7b = ModelConfig.llama(.CodeLlama_7B);

    try testing.expect(llama_7b.d_model < llama_13b.d_model);
    try testing.expect(llama_7b.num_layers < llama_13b.num_layers);
    try testing.expect(code_llama_7b.max_seq_len > llama_7b.max_seq_len);

    // All should be valid
    try llama_7b.validate();
    try llama_13b.validate();
    try code_llama_7b.validate();
}