const std = @import("std");
const Allocator = std.mem.Allocator;
const foundation = @import("../foundation/tensor.zig");
const Tensor = foundation.Tensor;

/// Advanced sampling strategies beyond basic top-k/top-p
pub const AdvancedSamplingType = enum {
    Mirostat,        // Mirostat sampling (v1 and v2)
    Typical,         // Typical sampling
    TailFree,        // Tail-free sampling
    LocallyTypical,  // Locally typical sampling
    Classifier,      // Classifier-free guidance
    Contrastive,     // Contrastive search
};

/// Mirostat sampling parameters
pub const MirostatConfig = struct {
    version: enum { V1, V2 },
    tau: f32,           // Target entropy/perplexity
    eta: f32,           // Learning rate for adjusting tau
    epsilon: f32,       // Convergence threshold
    max_iterations: u32, // Maximum adaptation iterations
};

/// Typical sampling configuration
pub const TypicalConfig = struct {
    mass: f32,          // Typical mass threshold (0.0-1.0)
    min_tokens: u32,    // Minimum tokens to keep
};

/// Tail-free sampling configuration
pub const TailFreeConfig = struct {
    z: f32,             // Tail-free parameter (0.0-1.0)
    min_tokens: u32,    // Minimum tokens to keep
};

/// Locally typical sampling configuration
pub const LocallyTypicalConfig = struct {
    mass: f32,          // Local typical mass
    entropy_threshold: f32, // Entropy threshold for locality
};

/// Advanced sampling engine
pub const AdvancedSampler = struct {
    allocator: Allocator,
    rng: std.rand.DefaultPrng,

    const Self = @This();

    pub fn init(allocator: Allocator, seed: ?u64) Self {
        const actual_seed = seed orelse @as(u64, @intCast(std.time.timestamp()));
        return Self{
            .allocator = allocator,
            .rng = std.rand.DefaultPrng.init(actual_seed),
        };
    }

    /// Mirostat sampling - maintains target entropy/perplexity
    pub fn sampleMirostat(self: *Self, logits: Tensor(f32), config: MirostatConfig) !u32 {
        switch (config.version) {
            .V1 => return try self.mirostatV1(logits, config),
            .V2 => return try self.mirostatV2(logits, config),
        }
    }

    /// Mirostat v1: Classic entropy-targeting sampling
    fn mirostatV1(self: *Self, logits: Tensor(f32), config: MirostatConfig) !u32 {
        // Convert logits to probabilities
        const probs = try self.softmax(logits);
        defer probs.deinit(self.allocator);

        // Calculate current entropy
        var entropy: f32 = 0.0;
        for (probs.data) |p| {
            if (p > 1e-10) {
                entropy -= p * std.math.log2(p);
            }
        }

        // Adaptive threshold based on entropy targeting
        var k = config.max_iterations;
        var current_tau = config.tau;
        var iteration: u32 = 0;

        while (iteration < config.max_iterations) {
            // Select top-k tokens based on current threshold
            const selected_indices = try self.selectTopK(probs, k);
            defer self.allocator.free(selected_indices);

            // Calculate entropy of selected distribution
            var selected_entropy: f32 = 0.0;
            var selected_mass: f32 = 0.0;

            for (selected_indices) |idx| {
                const p = probs.data[idx];
                selected_mass += p;
                if (p > 1e-10) {
                    selected_entropy -= p * std.math.log2(p);
                }
            }

            // Normalize entropy by selection mass
            if (selected_mass > 0) {
                selected_entropy /= selected_mass;
            }

            // Check convergence
            const entropy_error = @abs(selected_entropy - current_tau);
            if (entropy_error < config.epsilon) {
                break;
            }

            // Adapt threshold
            if (selected_entropy > current_tau) {
                // Too high entropy, reduce k
                k = @max(1, k - 1);
            } else {
                // Too low entropy, increase k
                k = @min(@as(u32, @intCast(probs.data.len)), k + 1);
            }

            // Update tau with learning rate
            current_tau += config.eta * (selected_entropy - current_tau);
            iteration += 1;
        }

        // Sample from final selection
        const final_selection = try self.selectTopK(probs, k);
        defer self.allocator.free(final_selection);

        return self.sampleFromIndices(probs, final_selection);
    }

    /// Mirostat v2: Improved version with better convergence
    fn mirostatV2(self: *Self, logits: Tensor(f32), config: MirostatConfig) !u32 {
        const probs = try self.softmax(logits);
        defer probs.deinit(self.allocator);

        // Calculate surprisal (negative log probability)
        const surprisal = try self.allocator.alloc(f32, probs.data.len);
        defer self.allocator.free(surprisal);

        for (probs.data, 0..) |p, i| {
            surprisal[i] = if (p > 1e-10) -std.math.log2(p) else std.math.inf(f32);
        }

        // Sort by surprisal (ascending order)
        const sorted_indices = try self.argsort(surprisal);
        defer self.allocator.free(sorted_indices);

        // Adaptive threshold selection
        var selected_count: usize = 1;
        var cumulative_prob: f32 = 0.0;
        var weighted_surprisal: f32 = 0.0;

        for (sorted_indices, 0..) |idx, count| {
            const prob = probs.data[idx];
            const surp = surprisal[idx];

            cumulative_prob += prob;
            weighted_surprisal += prob * surp;

            // Check if we've reached target entropy
            const current_entropy = weighted_surprisal / cumulative_prob;
            if (current_entropy >= config.tau or count >= sorted_indices.len - 1) {
                selected_count = count + 1;
                break;
            }
        }

        // Ensure minimum selection
        selected_count = @max(selected_count, 1);

        // Sample from selected tokens
        const selected = sorted_indices[0..selected_count];
        return self.sampleFromIndices(probs, selected);
    }

    /// Typical sampling - select tokens close to expected information content
    pub fn sampleTypical(self: *Self, logits: Tensor(f32), config: TypicalConfig) !u32 {
        const probs = try self.softmax(logits);
        defer probs.deinit(self.allocator);

        // Calculate entropy of distribution
        var entropy: f32 = 0.0;
        for (probs.data) |p| {
            if (p > 1e-10) {
                entropy -= p * std.math.log2(p);
            }
        }

        // Calculate absolute difference from typical information content
        const typical_info = try self.allocator.alloc(f32, probs.data.len);
        defer self.allocator.free(typical_info);

        for (probs.data, 0..) |p, i| {
            const information = if (p > 1e-10) -std.math.log2(p) else std.math.inf(f32);
            typical_info[i] = @abs(information - entropy);
        }

        // Sort by how close to typical information content
        const sorted_indices = try self.argsort(typical_info);
        defer self.allocator.free(sorted_indices);

        // Select tokens until we reach the desired mass
        var cumulative_mass: f32 = 0.0;
        var selected_count: usize = 0;

        for (sorted_indices, 0..) |idx, count| {
            cumulative_mass += probs.data[idx];
            selected_count = count + 1;

            if (cumulative_mass >= config.mass or selected_count >= config.min_tokens) {
                break;
            }
        }

        // Ensure we have at least min_tokens
        selected_count = @max(selected_count, config.min_tokens);
        selected_count = @min(selected_count, sorted_indices.len);

        const selected = sorted_indices[0..selected_count];
        return self.sampleFromIndices(probs, selected);
    }

    /// Tail-free sampling - remove the "tail" of unlikely tokens
    pub fn sampleTailFree(self: *Self, logits: Tensor(f32), config: TailFreeConfig) !u32 {
        const probs = try self.softmax(logits);
        defer probs.deinit(self.allocator);

        // Sort probabilities in descending order
        const sorted_indices = try self.argsortDescending(probs.data);
        defer self.allocator.free(sorted_indices);

        // Calculate second derivatives to find the tail
        const second_derivatives = try self.allocator.alloc(f32, sorted_indices.len);
        defer self.allocator.free(second_derivatives);

        // Compute second derivatives of the probability distribution
        for (1..sorted_indices.len - 1) |i| {
            const p_prev = probs.data[sorted_indices[i - 1]];
            const p_curr = probs.data[sorted_indices[i]];
            const p_next = probs.data[sorted_indices[i + 1]];

            // Second derivative approximation
            second_derivatives[i] = p_prev - 2.0 * p_curr + p_next;
        }
        // Handle boundaries
        second_derivatives[0] = 0.0;
        second_derivatives[sorted_indices.len - 1] = 0.0;

        // Find cutoff point where second derivative pattern indicates tail
        var cutoff_idx: usize = sorted_indices.len;
        var sum_second_deriv: f32 = 0.0;

        for (1..sorted_indices.len - 1) |i| {
            sum_second_deriv += second_derivatives[i];
            const normalized_sum = sum_second_deriv / @as(f32, @floatFromInt(i));

            // If normalized second derivative sum exceeds threshold, we found the tail
            if (normalized_sum > config.z) {
                cutoff_idx = i;
                break;
            }
        }

        // Ensure minimum tokens
        cutoff_idx = @max(cutoff_idx, config.min_tokens);
        cutoff_idx = @min(cutoff_idx, sorted_indices.len);

        const selected = sorted_indices[0..cutoff_idx];
        return self.sampleFromIndices(probs, selected);
    }

    /// Locally typical sampling - combines typical and local information
    pub fn sampleLocallyTypical(self: *Self, logits: Tensor(f32), config: LocallyTypicalConfig) !u32 {
        const probs = try self.softmax(logits);
        defer probs.deinit(self.allocator);

        // Calculate global entropy
        var global_entropy: f32 = 0.0;
        for (probs.data) |p| {
            if (p > 1e-10) {
                global_entropy -= p * std.math.log2(p);
            }
        }

        // Calculate local entropy contributions
        const local_contributions = try self.allocator.alloc(f32, probs.data.len);
        defer self.allocator.free(local_contributions);

        for (probs.data, 0..) |p, i| {
            const information = if (p > 1e-10) -std.math.log2(p) else std.math.inf(f32);

            // Local contribution considers both information content and local context
            local_contributions[i] = if (global_entropy > config.entropy_threshold)
                @abs(information - global_entropy)  // High entropy: prefer typical
            else
                information;  // Low entropy: prefer high information content
        }

        // Sort by local contribution (ascending for typical-like selection)
        const sorted_indices = try self.argsort(local_contributions);
        defer self.allocator.free(sorted_indices);

        // Select tokens until mass threshold
        var cumulative_mass: f32 = 0.0;
        var selected_count: usize = 0;

        for (sorted_indices, 0..) |idx, count| {
            cumulative_mass += probs.data[idx];
            selected_count = count + 1;

            if (cumulative_mass >= config.mass) {
                break;
            }
        }

        selected_count = @max(selected_count, 1);
        selected_count = @min(selected_count, sorted_indices.len);

        const selected = sorted_indices[0..selected_count];
        return self.sampleFromIndices(probs, selected);
    }

    /// Contrastive search - balance likelihood and diversity
    pub fn sampleContrastive(self: *Self, logits: Tensor(f32), alpha: f32, k: u32) !u32 {
        const probs = try self.softmax(logits);
        defer probs.deinit(self.allocator);

        // Select top-k candidates
        const top_k_indices = try self.selectTopK(probs, k);
        defer self.allocator.free(top_k_indices);

        // For contrastive search, we would typically need context from previous tokens
        // This is a simplified version that balances probability with pseudo-diversity
        const contrastive_scores = try self.allocator.alloc(f32, top_k_indices.len);
        defer self.allocator.free(contrastive_scores);

        for (top_k_indices, 0..) |idx, i| {
            const likelihood_score = std.math.log(probs.data[idx]);

            // Pseudo-diversity: penalize tokens similar to recent high-probability tokens
            var diversity_penalty: f32 = 0.0;
            for (top_k_indices) |other_idx| {
                if (other_idx != idx) {
                    // Simplified similarity based on probability proximity
                    const prob_similarity = 1.0 - @abs(probs.data[idx] - probs.data[other_idx]);
                    diversity_penalty += prob_similarity * probs.data[other_idx];
                }
            }

            contrastive_scores[i] = likelihood_score - alpha * diversity_penalty;
        }

        // Find best scoring token
        var best_idx: usize = 0;
        var best_score: f32 = contrastive_scores[0];

        for (contrastive_scores, 0..) |score, i| {
            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }

        return top_k_indices[best_idx];
    }

    // Helper functions

    /// Convert logits to probabilities using softmax
    fn softmax(self: *Self, logits: Tensor(f32)) !Tensor(f32) {
        const result_data = try self.allocator.alloc(f32, logits.data.len);

        // Find maximum for numerical stability
        var max_val = logits.data[0];
        for (logits.data[1..]) |val| {
            max_val = @max(max_val, val);
        }

        // Compute exp(x - max) and sum
        var sum: f32 = 0.0;
        for (logits.data, 0..) |val, i| {
            result_data[i] = std.math.exp(val - max_val);
            sum += result_data[i];
        }

        // Normalize
        for (result_data) |*val| {
            val.* /= sum;
        }

        return Tensor(f32){ .data = result_data, .shape = logits.shape };
    }

    /// Select top-k indices by probability
    fn selectTopK(self: *Self, probs: Tensor(f32), k: u32) ![]u32 {
        const indices = try self.argsortDescending(probs.data);
        defer self.allocator.free(indices);

        const actual_k = @min(k, @as(u32, @intCast(indices.len)));
        const result = try self.allocator.alloc(u32, actual_k);
        @memcpy(result, indices[0..actual_k]);

        return result;
    }

    /// Sort indices by values in ascending order
    fn argsort(self: *Self, values: []const f32) ![]u32 {
        const indices = try self.allocator.alloc(u32, values.len);
        for (0..values.len) |i| {
            indices[i] = @as(u32, @intCast(i));
        }

        const Context = struct {
            values: []const f32,

            pub fn lessThan(context: @This(), a: u32, b: u32) bool {
                return context.values[a] < context.values[b];
            }
        };

        std.mem.sort(u32, indices, Context{ .values = values }, Context.lessThan);
        return indices;
    }

    /// Sort indices by values in descending order
    fn argsortDescending(self: *Self, values: []const f32) ![]u32 {
        const indices = try self.allocator.alloc(u32, values.len);
        for (0..values.len) |i| {
            indices[i] = @as(u32, @intCast(i));
        }

        const Context = struct {
            values: []const f32,

            pub fn lessThan(context: @This(), a: u32, b: u32) bool {
                return context.values[a] > context.values[b];  // Note: reversed for descending
            }
        };

        std.mem.sort(u32, indices, Context{ .values = values }, Context.lessThan);
        return indices;
    }

    /// Sample from selected indices based on their probabilities
    fn sampleFromIndices(self: *Self, probs: Tensor(f32), indices: []const u32) u32 {
        // Calculate probability mass of selected tokens
        var total_mass: f32 = 0.0;
        for (indices) |idx| {
            total_mass += probs.data[idx];
        }

        // Generate random value
        const random = self.rng.random();
        const rand_val = random.float(f32) * total_mass;

        // Find selected token
        var cumulative: f32 = 0.0;
        for (indices) |idx| {
            cumulative += probs.data[idx];
            if (cumulative >= rand_val) {
                return idx;
            }
        }

        // Fallback (shouldn't happen with proper implementation)
        return indices[indices.len - 1];
    }
};

/// Sampling strategy selector and coordinator
pub const SamplingCoordinator = struct {
    base_sampler: AdvancedSampler,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, seed: ?u64) Self {
        return Self{
            .base_sampler = AdvancedSampler.init(allocator, seed),
            .allocator = allocator,
        };
    }

    /// Adaptive sampling: choose best strategy based on distribution characteristics
    pub fn adaptiveSample(self: *Self, logits: Tensor(f32)) !u32 {
        // Analyze distribution characteristics
        const probs = try self.base_sampler.softmax(logits);
        defer probs.deinit(self.allocator);

        // Calculate entropy
        var entropy: f32 = 0.0;
        for (probs.data) |p| {
            if (p > 1e-10) {
                entropy -= p * std.math.log2(p);
            }
        }

        // Calculate maximum probability
        var max_prob: f32 = 0.0;
        for (probs.data) |p| {
            max_prob = @max(max_prob, p);
        }

        // Adaptive strategy selection
        if (max_prob > 0.8) {
            // Very confident prediction - use simple sampling
            return self.base_sampler.sampleContrastive(logits, 0.1, 5);
        } else if (entropy < 2.0) {
            // Low entropy - use typical sampling
            return try self.base_sampler.sampleTypical(logits, TypicalConfig{
                .mass = 0.9,
                .min_tokens = 3,
            });
        } else if (entropy > 6.0) {
            // High entropy - use mirostat to control
            return try self.base_sampler.sampleMirostat(logits, MirostatConfig{
                .version = .V2,
                .tau = 3.0,
                .eta = 0.1,
                .epsilon = 0.01,
                .max_iterations = 10,
            });
        } else {
            // Moderate entropy - use tail-free sampling
            return try self.base_sampler.sampleTailFree(logits, TailFreeConfig{
                .z = 0.95,
                .min_tokens = 2,
            });
        }
    }

    /// Combined sampling: apply multiple strategies and select best result
    pub fn combinedSample(self: *Self, logits: Tensor(f32), weights: []const f32) !u32 {
        std.debug.assert(weights.len >= 4); // Need at least 4 weights for different strategies

        // Sample with different strategies
        const mirostat_token = try self.base_sampler.sampleMirostat(logits, MirostatConfig{
            .version = .V2,
            .tau = 3.0,
            .eta = 0.1,
            .epsilon = 0.01,
            .max_iterations = 10,
        });

        const typical_token = try self.base_sampler.sampleTypical(logits, TypicalConfig{
            .mass = 0.9,
            .min_tokens = 3,
        });

        const tailfree_token = try self.base_sampler.sampleTailFree(logits, TailFreeConfig{
            .z = 0.95,
            .min_tokens = 2,
        });

        const contrastive_token = try self.base_sampler.sampleContrastive(logits, 0.2, 8);

        // Weighted selection among strategies
        const tokens = [_]u32{ mirostat_token, typical_token, tailfree_token, contrastive_token };
        const total_weight = weights[0] + weights[1] + weights[2] + weights[3];

        const random = self.base_sampler.rng.random();
        const rand_val = random.float(f32) * total_weight;

        var cumulative: f32 = 0.0;
        for (tokens, 0..) |token, i| {
            cumulative += weights[i];
            if (cumulative >= rand_val) {
                return token;
            }
        }

        return tokens[tokens.len - 1];
    }
};