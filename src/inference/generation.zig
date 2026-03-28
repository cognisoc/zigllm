//! Text Generation Engine
//!
//! This module implements the core text generation algorithms for transformer
//! models, including sampling strategies, stop conditions, and optimization
//! techniques used in production inference systems.
//!
//! ## Educational Value
//! Text generation is where language models come to life:
//! - How autoregressive generation works step by step
//! - Sampling strategies and their impact on text quality
//! - Performance optimization techniques for real-time inference
//! - Stop conditions and generation control mechanisms

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const Random = std.Random;
const math = std.math;

const Tensor = @import("../foundation/tensor.zig").Tensor;
const LLaMAModel = @import("../models/llama.zig").LLaMAModel;
const SimpleTokenizer = @import("../models/tokenizer.zig").SimpleTokenizer;
const TokenId = @import("../models/tokenizer.zig").TokenId;
const SpecialTokens = @import("../models/tokenizer.zig").SpecialTokens;

/// Sampling strategy for text generation
pub const SamplingStrategy = enum {
    /// Select token with highest probability (deterministic)
    Greedy,
    /// Sample from top-k most likely tokens
    TopK,
    /// Sample from tokens whose cumulative probability exceeds p
    TopP,
    /// Temperature-scaled sampling from full distribution
    Temperature,
    /// Combined top-k + top-p + temperature sampling
    Combined,

    /// Get human-readable name
    pub fn name(self: SamplingStrategy) []const u8 {
        return switch (self) {
            .Greedy => "Greedy",
            .TopK => "Top-K",
            .TopP => "Top-P (Nucleus)",
            .Temperature => "Temperature",
            .Combined => "Combined",
        };
    }
};

/// Generation configuration parameters
pub const GenerationConfig = struct {
    /// Sampling strategy to use
    strategy: SamplingStrategy = .Combined,

    /// Temperature for probability scaling (0.0 = deterministic, 1.0 = original distribution)
    temperature: f32 = 0.7,

    /// Top-k: only sample from k most likely tokens (0 = disabled)
    top_k: u32 = 40,

    /// Top-p: sample from tokens with cumulative probability p (1.0 = disabled)
    top_p: f32 = 0.9,

    /// Maximum number of tokens to generate
    max_tokens: u32 = 512,

    /// Minimum number of tokens to generate
    min_tokens: u32 = 1,

    /// Stop generation if these token IDs are encountered
    stop_tokens: []const TokenId = &[_]TokenId{SpecialTokens.EOS},

    /// Stop generation if these strings are encountered
    stop_strings: []const []const u8 = &[_][]const u8{},

    /// Repetition penalty to reduce repetitive text (1.0 = no penalty)
    repetition_penalty: f32 = 1.1,

    /// Length penalty for longer sequences (1.0 = no penalty)
    length_penalty: f32 = 1.0,

    /// Random seed for reproducible generation
    seed: ?u64 = null,

    /// Validate configuration parameters
    pub fn validate(self: GenerationConfig) !void {
        if (self.temperature < 0.0) {
            return error.InvalidTemperature;
        }
        if (self.top_p < 0.0 or self.top_p > 1.0) {
            return error.InvalidTopP;
        }
        if (self.max_tokens == 0) {
            return error.InvalidMaxTokens;
        }
        if (self.min_tokens > self.max_tokens) {
            return error.InvalidTokenLimits;
        }
        if (self.repetition_penalty < 0.0) {
            return error.InvalidRepetitionPenalty;
        }
    }

    /// Create configuration for different generation styles
    pub fn creative() GenerationConfig {
        return GenerationConfig{
            .strategy = .Combined,
            .temperature = 0.9,
            .top_k = 50,
            .top_p = 0.95,
            .repetition_penalty = 1.05,
        };
    }

    pub fn balanced() GenerationConfig {
        return GenerationConfig{
            .strategy = .Combined,
            .temperature = 0.7,
            .top_k = 40,
            .top_p = 0.9,
            .repetition_penalty = 1.1,
        };
    }

    pub fn focused() GenerationConfig {
        return GenerationConfig{
            .strategy = .Combined,
            .temperature = 0.3,
            .top_k = 20,
            .top_p = 0.8,
            .repetition_penalty = 1.15,
        };
    }

    pub fn deterministic() GenerationConfig {
        return GenerationConfig{
            .strategy = .Greedy,
            .temperature = 0.0,
            .top_k = 1,
            .top_p = 1.0,
            .repetition_penalty = 1.0,
        };
    }
};

/// Token probability for sampling
pub const TokenProb = struct {
    token_id: TokenId,
    probability: f32,
    log_prob: f32,

    /// Create from logit
    pub fn fromLogit(token_id: TokenId, logit: f32) TokenProb {
        return TokenProb{
            .token_id = token_id,
            .probability = 0.0, // Will be set after softmax
            .log_prob = logit,
        };
    }
};

/// Generation result with metadata
pub const GenerationResult = struct {
    /// Generated token IDs
    tokens: []TokenId,
    /// Generated text (if tokenizer provided)
    text: ?[]u8,
    /// Log probabilities for each token
    log_probs: []f32,
    /// Total log probability of the sequence
    total_log_prob: f32,
    /// Number of tokens generated
    num_tokens: u32,
    /// Reason generation stopped
    stop_reason: StopReason,
    /// Generation statistics
    stats: GenerationStats,

    pub fn deinit(self: GenerationResult, allocator: Allocator) void {
        allocator.free(self.tokens);
        if (self.text) |text| {
            allocator.free(text);
        }
        allocator.free(self.log_probs);
    }
};

/// Reason why generation stopped
pub const StopReason = enum {
    /// Reached maximum token limit
    MaxTokens,
    /// Encountered stop token
    StopToken,
    /// Encountered stop string
    StopString,
    /// Model produced end-of-sequence token
    EndOfSequence,
    /// Error occurred during generation
    Error,

    pub fn description(self: StopReason) []const u8 {
        return switch (self) {
            .MaxTokens => "Maximum token limit reached",
            .StopToken => "Stop token encountered",
            .StopString => "Stop string encountered",
            .EndOfSequence => "End of sequence",
            .Error => "Generation error",
        };
    }
};

/// Generation performance statistics
pub const GenerationStats = struct {
    /// Total generation time in milliseconds
    generation_time_ms: f64,
    /// Tokens per second
    tokens_per_second: f32,
    /// Average time per token in milliseconds
    time_per_token_ms: f32,
    /// Peak memory usage during generation
    peak_memory_bytes: usize,
    /// Number of model forward passes
    num_forward_passes: u32,

    pub fn calculate(num_tokens: u32, generation_time_ms: f64) GenerationStats {
        const tokens_per_second = if (generation_time_ms > 0)
            @as(f32, @floatFromInt(num_tokens)) * 1000.0 / @as(f32, @floatCast(generation_time_ms))
        else
            0.0;

        const time_per_token_ms = if (num_tokens > 0)
            @as(f32, @floatCast(generation_time_ms)) / @as(f32, @floatFromInt(num_tokens))
        else
            0.0;

        return GenerationStats{
            .generation_time_ms = generation_time_ms,
            .tokens_per_second = tokens_per_second,
            .time_per_token_ms = time_per_token_ms,
            .peak_memory_bytes = 0, // TODO: Implement memory tracking
            .num_forward_passes = num_tokens,
        };
    }
};

/// Text generation engine
pub const TextGenerator = struct {
    /// Model for inference
    model: *LLaMAModel,
    /// Tokenizer for text conversion
    tokenizer: *SimpleTokenizer,
    /// Memory allocator
    allocator: Allocator,
    /// Random number generator
    rng: Random,
    /// Current generation configuration
    config: GenerationConfig,

    /// Initialize text generator
    pub fn init(model: *LLaMAModel, tokenizer: *SimpleTokenizer, allocator: Allocator, seed: ?u64) TextGenerator {
        var prng = std.rand.DefaultPrng.init(seed orelse @as(u64, @intCast(std.time.milliTimestamp())));

        return TextGenerator{
            .model = model,
            .tokenizer = tokenizer,
            .allocator = allocator,
            .rng = prng.random(),
            .config = GenerationConfig.balanced(),
        };
    }

    /// Set generation configuration
    pub fn setConfig(self: *TextGenerator, config: GenerationConfig) !void {
        try config.validate();
        self.config = config;

        // Update RNG seed if provided
        if (config.seed) |seed| {
            var prng = std.rand.DefaultPrng.init(seed);
            self.rng = prng.random();
        }
    }

    /// Generate text from prompt
    pub fn generate(self: *TextGenerator, prompt: []const u8) !GenerationResult {
        const start_time = std.time.milliTimestamp();

        // Tokenize prompt
        const prompt_tokens = try self.tokenizer.encode(prompt);
        defer self.allocator.free(prompt_tokens);

        // Generate tokens
        var generated_tokens = ArrayList(TokenId).init(self.allocator);
        defer generated_tokens.deinit();

        var log_probs = ArrayList(f32).init(self.allocator);
        defer log_probs.deinit();

        // Start with prompt tokens (excluding BOS/EOS)
        for (prompt_tokens) |token| {
            if (token != SpecialTokens.BOS and token != SpecialTokens.EOS) {
                try generated_tokens.append(token);
            }
        }

        var total_log_prob: f32 = 0.0;
        var stop_reason = StopReason.MaxTokens;

        // Generate tokens one by one (autoregressive)
        var num_generated: u32 = 0;
        while (num_generated < self.config.max_tokens) {
            // Forward pass through model
            const input_tokens = generated_tokens.items;
            const logits = try self.model.forward(input_tokens, null);
            defer logits.deinit();

            // Get logits for last position (next token prediction)
            const vocab_size = logits.shape[logits.shape.len - 1];
            const last_logits = logits.data[(logits.size - vocab_size)..logits.size];

            // Apply repetition penalty
            const modified_logits = try self.allocator.alloc(f32, vocab_size);
            defer self.allocator.free(modified_logits);
            @memcpy(modified_logits, last_logits);

            try self.applyRepetitionPenalty(modified_logits, generated_tokens.items);

            // Sample next token
            const sample_result = try self.sampleToken(modified_logits);
            const next_token = sample_result.token_id;
            const token_log_prob = sample_result.log_prob;

            // Check stop conditions
            if (self.shouldStop(next_token, generated_tokens.items)) {
                if (next_token == SpecialTokens.EOS) {
                    stop_reason = .EndOfSequence;
                } else {
                    stop_reason = .StopToken;
                }
                break;
            }

            // Add token to sequence
            try generated_tokens.append(next_token);
            try log_probs.append(token_log_prob);
            total_log_prob += token_log_prob;
            num_generated += 1;

            // Check minimum tokens requirement
            if (num_generated >= self.config.min_tokens and next_token == SpecialTokens.EOS) {
                stop_reason = .EndOfSequence;
                break;
            }
        }

        // Decode generated text
        const generated_text = try self.tokenizer.decode(generated_tokens.items);

        // Calculate statistics
        const end_time = std.time.milliTimestamp();
        const generation_time = @as(f64, @floatFromInt(end_time - start_time));
        const stats = GenerationStats.calculate(num_generated, generation_time);

        return GenerationResult{
            .tokens = try generated_tokens.toOwnedSlice(),
            .text = generated_text,
            .log_probs = try log_probs.toOwnedSlice(),
            .total_log_prob = total_log_prob,
            .num_tokens = num_generated,
            .stop_reason = stop_reason,
            .stats = stats,
        };
    }

    /// Sample next token based on configuration
    fn sampleToken(self: *TextGenerator, logits: []f32) !TokenProb {
        return switch (self.config.strategy) {
            .Greedy => try self.sampleGreedy(logits),
            .TopK => try self.sampleTopK(logits, self.config.top_k),
            .TopP => try self.sampleTopP(logits, self.config.top_p),
            .Temperature => try self.sampleTemperature(logits, self.config.temperature),
            .Combined => try self.sampleCombined(logits),
        };
    }

    /// Greedy sampling: select token with highest probability
    fn sampleGreedy(self: *TextGenerator, logits: []f32) !TokenProb {
        _ = self;
        var max_logit: f32 = -math.inf(f32);
        var max_token: TokenId = 0;

        for (logits, 0..) |logit, i| {
            if (logit > max_logit) {
                max_logit = logit;
                max_token = @as(TokenId, @intCast(i));
            }
        }

        return TokenProb{
            .token_id = max_token,
            .probability = 1.0, // Deterministic
            .log_prob = max_logit,
        };
    }

    /// Top-k sampling: sample from k most likely tokens
    fn sampleTopK(self: *TextGenerator, logits: []f32, k: u32) !TokenProb {
        if (k == 0 or k >= logits.len) {
            return self.sampleTemperature(logits, self.config.temperature);
        }

        // Create token-probability pairs
        var token_probs = try self.allocator.alloc(TokenProb, logits.len);
        defer self.allocator.free(token_probs);

        for (logits, 0..) |logit, i| {
            token_probs[i] = TokenProb.fromLogit(@as(TokenId, @intCast(i)), logit);
        }

        // Sort by logit (descending)
        std.mem.sort(TokenProb, token_probs, {}, struct {
            fn lessThan(context: void, a: TokenProb, b: TokenProb) bool {
                _ = context;
                return a.log_prob > b.log_prob;
            }
        }.lessThan);

        // Keep only top-k tokens
        const top_k_slice = token_probs[0..k];

        // Apply softmax and sample
        try self.applySoftmax(top_k_slice);
        return self.sampleFromDistribution(top_k_slice);
    }

    /// Top-p (nucleus) sampling: sample from tokens with cumulative probability p
    fn sampleTopP(self: *TextGenerator, logits: []f32, p: f32) !TokenProb {
        if (p >= 1.0) {
            return self.sampleTemperature(logits, self.config.temperature);
        }

        // Create and sort token-probability pairs
        var token_probs = try self.allocator.alloc(TokenProb, logits.len);
        defer self.allocator.free(token_probs);

        for (logits, 0..) |logit, i| {
            token_probs[i] = TokenProb.fromLogit(@as(TokenId, @intCast(i)), logit);
        }

        std.mem.sort(TokenProb, token_probs, {}, struct {
            fn lessThan(context: void, a: TokenProb, b: TokenProb) bool {
                _ = context;
                return a.log_prob > b.log_prob;
            }
        }.lessThan);

        // Apply softmax to all tokens first
        try self.applySoftmax(token_probs);

        // Find cutoff point where cumulative probability exceeds p
        var cumulative_prob: f32 = 0.0;
        var cutoff: usize = 0;

        for (token_probs, 0..) |token_prob, i| {
            cumulative_prob += token_prob.probability;
            if (cumulative_prob >= p) {
                cutoff = i + 1;
                break;
            }
        }

        if (cutoff == 0) cutoff = 1; // Always include at least one token

        // Renormalize probabilities for selected tokens
        const selected_tokens = token_probs[0..cutoff];
        var sum: f32 = 0.0;
        for (selected_tokens) |token_prob| {
            sum += token_prob.probability;
        }

        for (selected_tokens) |*token_prob| {
            token_prob.probability /= sum;
        }

        return self.sampleFromDistribution(selected_tokens);
    }

    /// Temperature sampling: scale logits and sample from full distribution
    fn sampleTemperature(self: *TextGenerator, logits: []f32, temperature: f32) !TokenProb {
        var token_probs = try self.allocator.alloc(TokenProb, logits.len);
        defer self.allocator.free(token_probs);

        // Scale logits by temperature
        for (logits, 0..) |logit, i| {
            const scaled_logit = if (temperature > 0.0) logit / temperature else logit;
            token_probs[i] = TokenProb.fromLogit(@as(TokenId, @intCast(i)), scaled_logit);
        }

        try self.applySoftmax(token_probs);
        return self.sampleFromDistribution(token_probs);
    }

    /// Combined sampling: top-k + top-p + temperature
    fn sampleCombined(self: *TextGenerator, logits: []f32) !TokenProb {
        // First apply temperature scaling
        var scaled_logits = try self.allocator.alloc(f32, logits.len);
        defer self.allocator.free(scaled_logits);

        const temp = self.config.temperature;
        for (logits, 0..) |logit, i| {
            scaled_logits[i] = if (temp > 0.0) logit / temp else logit;
        }

        // Then apply top-k filtering if enabled
        if (self.config.top_k > 0 and self.config.top_k < logits.len) {
            return self.sampleTopK(scaled_logits, self.config.top_k);
        }

        // Then apply top-p filtering if enabled
        if (self.config.top_p < 1.0) {
            return self.sampleTopP(scaled_logits, self.config.top_p);
        }

        // Otherwise just sample with temperature
        return self.sampleTemperature(logits, self.config.temperature);
    }

    /// Apply softmax to convert logits to probabilities
    fn applySoftmax(self: *TextGenerator, token_probs: []TokenProb) !void {
        _ = self;

        // Find maximum for numerical stability
        var max_logit: f32 = -math.inf(f32);
        for (token_probs) |token_prob| {
            max_logit = @max(max_logit, token_prob.log_prob);
        }

        // Compute exponentials and sum
        var sum: f32 = 0.0;
        for (token_probs) |*token_prob| {
            const exp_logit = @exp(token_prob.log_prob - max_logit);
            token_prob.probability = exp_logit;
            sum += exp_logit;
        }

        // Normalize
        for (token_probs) |*token_prob| {
            token_prob.probability /= sum;
            token_prob.log_prob = @log(token_prob.probability);
        }
    }

    /// Sample from probability distribution
    fn sampleFromDistribution(self: *TextGenerator, token_probs: []const TokenProb) TokenProb {
        const rand_val = self.rng.float(f32);
        var cumulative: f32 = 0.0;

        for (token_probs) |token_prob| {
            cumulative += token_prob.probability;
            if (rand_val <= cumulative) {
                return token_prob;
            }
        }

        // Fallback to last token if rounding errors occur
        return token_probs[token_probs.len - 1];
    }

    /// Apply repetition penalty to reduce repetitive text
    fn applyRepetitionPenalty(self: *TextGenerator, logits: []f32, tokens: []const TokenId) !void {
        if (self.config.repetition_penalty == 1.0) return;

        // Count token frequencies in recent history
        const history_window = @min(tokens.len, 64); // Look at last 64 tokens
        const recent_tokens = if (tokens.len > history_window)
            tokens[tokens.len - history_window..]
        else
            tokens;

        // Apply penalty to recently used tokens
        for (recent_tokens) |token| {
            if (token < logits.len) {
                if (logits[token] > 0) {
                    logits[token] /= self.config.repetition_penalty;
                } else {
                    logits[token] *= self.config.repetition_penalty;
                }
            }
        }
    }

    /// Check if generation should stop
    fn shouldStop(self: *TextGenerator, token: TokenId, tokens: []const TokenId) bool {
        // Check stop tokens
        for (self.config.stop_tokens) |stop_token| {
            if (token == stop_token) {
                return true;
            }
        }

        // Check stop strings (simplified - would need proper string matching)
        if (self.config.stop_strings.len > 0) {
            // TODO: Implement stop string detection
            // This would require decoding recent tokens and checking for stop strings
        }

        _ = tokens;
        return false;
    }
};

// Text generation tests
test "generation configuration validation" {
    const testing = std.testing;

    const valid_config = GenerationConfig.balanced();
    try valid_config.validate();

    var invalid_config = valid_config;
    invalid_config.temperature = -1.0;
    try testing.expectError(error.InvalidTemperature, invalid_config.validate());

    invalid_config = valid_config;
    invalid_config.top_p = 2.0;
    try testing.expectError(error.InvalidTopP, invalid_config.validate());

    invalid_config = valid_config;
    invalid_config.max_tokens = 0;
    try testing.expectError(error.InvalidMaxTokens, invalid_config.validate());
}

test "sampling strategy properties" {
    const testing = std.testing;

    try testing.expectEqualStrings("Greedy", SamplingStrategy.Greedy.name());
    try testing.expectEqualStrings("Top-K", SamplingStrategy.TopK.name());
    try testing.expectEqualStrings("Top-P (Nucleus)", SamplingStrategy.TopP.name());
    try testing.expectEqualStrings("Temperature", SamplingStrategy.Temperature.name());
}

test "generation configuration presets" {
    const testing = std.testing;

    const creative = GenerationConfig.creative();
    const balanced = GenerationConfig.balanced();
    const focused = GenerationConfig.focused();
    const deterministic = GenerationConfig.deterministic();

    try testing.expect(creative.temperature > balanced.temperature);
    try testing.expect(balanced.temperature > focused.temperature);
    try testing.expect(focused.temperature > deterministic.temperature);
    try testing.expectEqual(@as(f32, 0.0), deterministic.temperature);

    try testing.expectEqual(SamplingStrategy.Greedy, deterministic.strategy);
    try testing.expectEqual(SamplingStrategy.Combined, creative.strategy);
}

test "generation statistics calculation" {
    const testing = std.testing;

    const stats = GenerationStats.calculate(100, 2000.0); // 100 tokens in 2 seconds

    try testing.expectEqual(@as(f64, 2000.0), stats.generation_time_ms);
    try testing.expectApproxEqAbs(@as(f32, 50.0), stats.tokens_per_second, 0.1); // 100 tokens / 2 seconds = 50 t/s
    try testing.expectApproxEqAbs(@as(f32, 20.0), stats.time_per_token_ms, 0.1); // 2000ms / 100 tokens = 20ms/token
    try testing.expectEqual(@as(u32, 100), stats.num_forward_passes);
}

test "stop reason descriptions" {
    const testing = std.testing;

    try testing.expectEqualStrings("Maximum token limit reached", StopReason.MaxTokens.description());
    try testing.expectEqualStrings("Stop token encountered", StopReason.StopToken.description());
    try testing.expectEqualStrings("End of sequence", StopReason.EndOfSequence.description());
}