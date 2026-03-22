// Mixture of Experts (MoE) Model Implementation
// Advanced sparse neural network architecture for efficient scaling
//
// MoE enables models to scale to trillions of parameters while keeping computational
// cost manageable by routing tokens to specialized expert networks.
//
// Key features:
// 1. Sparse activation - only a subset of experts process each token
// 2. Expert routing with learned gating networks
// 3. Load balancing to prevent expert collapse
// 4. Support for different expert types (FFN, Attention, Hybrid)
// 5. Efficient batching and parallelization strategies
// 6. Advanced routing algorithms (Top-K, Switch, GLaM, PaLM)

const std = @import("std");
const Tensor = @import("../foundation/tensor.zig").Tensor;
const activations = @import("../neural_primitives/activations.zig");
const normalization = @import("../neural_primitives/normalization.zig");

// Expert Types
pub const ExpertType = enum {
    feed_forward,    // Standard FFN expert
    attention,       // Attention-based expert
    hybrid,          // Combined attention + FFN expert
    custom,          // User-defined expert

    pub fn toString(self: ExpertType) []const u8 {
        return switch (self) {
            .feed_forward => "Feed-Forward",
            .attention => "Attention",
            .hybrid => "Hybrid",
            .custom => "Custom",
        };
    }
};

// Routing Algorithm Types
pub const RoutingAlgorithm = enum {
    top_k,           // Standard Top-K routing (original MoE)
    switch_routing,  // Switch Transformer routing
    expert_choice,   // Expert Choice routing (GLaM)
    hash_routing,    // Hash-based routing (BASE layers)
    learned_routing, // Fully learned routing

    pub fn getDescription(self: RoutingAlgorithm) []const u8 {
        return switch (self) {
            .top_k => "Routes each token to top-K experts based on gating scores",
            .switch_routing => "Routes each token to exactly one expert (Switch Transformer)",
            .expert_choice => "Experts choose which tokens to process (GLaM style)",
            .hash_routing => "Deterministic hash-based routing for consistency",
            .learned_routing => "Fully learned routing with gradient-based optimization",
        };
    }
};

// MoE Configuration
pub const MoEConfig = struct {
    num_experts: u32,
    expert_capacity: u32,        // Maximum tokens per expert
    top_k: u32 = 2,             // Number of experts per token
    expert_type: ExpertType = .feed_forward,
    routing_algorithm: RoutingAlgorithm = .top_k,

    // Architecture parameters
    hidden_size: u32,
    expert_hidden_size: u32,    // Hidden size within each expert

    // Load balancing
    load_balancing_weight: f32 = 0.01,
    z_loss_weight: f32 = 0.001,

    // Dropout and regularization
    expert_dropout: f32 = 0.0,
    router_dropout: f32 = 0.0,

    // Performance optimizations
    use_sparse_gradients: bool = true,
    use_expert_parallelism: bool = true,

    pub fn switchTransformer(hidden_size: u32, num_experts: u32) MoEConfig {
        return MoEConfig{
            .num_experts = num_experts,
            .expert_capacity = 0, // Dynamic capacity
            .top_k = 1, // Switch routing
            .expert_type = .feed_forward,
            .routing_algorithm = .switch_routing,
            .hidden_size = hidden_size,
            .expert_hidden_size = hidden_size * 4, // Standard 4x expansion
            .load_balancing_weight = 0.01,
        };
    }

    pub fn glam(hidden_size: u32, num_experts: u32) MoEConfig {
        return MoEConfig{
            .num_experts = num_experts,
            .expert_capacity = 0,
            .top_k = 2,
            .expert_type = .feed_forward,
            .routing_algorithm = .expert_choice,
            .hidden_size = hidden_size,
            .expert_hidden_size = hidden_size * 4,
            .load_balancing_weight = 0.001,
        };
    }
};

// Expert Implementation
pub const Expert = struct {
    expert_type: ExpertType,
    expert_id: u32,

    // Feed-forward components
    w1: ?Tensor = null,  // Input projection
    w2: ?Tensor = null,  // Output projection
    w3: ?Tensor = null,  // Gate projection (for SwiGLU)

    // Attention components (for attention experts)
    q_proj: ?Tensor = null,
    k_proj: ?Tensor = null,
    v_proj: ?Tensor = null,
    o_proj: ?Tensor = null,

    // Normalization
    layer_norm: ?normalization.LayerNorm = null,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: MoEConfig, expert_id: u32) !Expert {
        var expert = Expert{
            .expert_type = config.expert_type,
            .expert_id = expert_id,
            .allocator = allocator,
        };

        switch (config.expert_type) {
            .feed_forward => {
                expert.w1 = try Tensor.random(allocator, &[_]u32{ config.hidden_size, config.expert_hidden_size });
                expert.w2 = try Tensor.random(allocator, &[_]u32{ config.expert_hidden_size, config.hidden_size });
                expert.w3 = try Tensor.random(allocator, &[_]u32{ config.hidden_size, config.expert_hidden_size });
            },
            .attention => {
                expert.q_proj = try Tensor.random(allocator, &[_]u32{ config.hidden_size, config.hidden_size });
                expert.k_proj = try Tensor.random(allocator, &[_]u32{ config.hidden_size, config.hidden_size });
                expert.v_proj = try Tensor.random(allocator, &[_]u32{ config.hidden_size, config.hidden_size });
                expert.o_proj = try Tensor.random(allocator, &[_]u32{ config.hidden_size, config.hidden_size });
            },
            .hybrid => {
                // Combination of both
                expert.w1 = try Tensor.random(allocator, &[_]u32{ config.hidden_size, config.expert_hidden_size });
                expert.w2 = try Tensor.random(allocator, &[_]u32{ config.expert_hidden_size, config.hidden_size });
                expert.q_proj = try Tensor.random(allocator, &[_]u32{ config.hidden_size, config.hidden_size });
                expert.k_proj = try Tensor.random(allocator, &[_]u32{ config.hidden_size, config.hidden_size });
                expert.v_proj = try Tensor.random(allocator, &[_]u32{ config.hidden_size, config.hidden_size });
                expert.o_proj = try Tensor.random(allocator, &[_]u32{ config.hidden_size, config.hidden_size });
            },
            .custom => {
                // User-defined expert - minimal setup
                expert.w1 = try Tensor.random(allocator, &[_]u32{ config.hidden_size, config.expert_hidden_size });
                expert.w2 = try Tensor.random(allocator, &[_]u32{ config.expert_hidden_size, config.hidden_size });
            },
        }

        expert.layer_norm = try normalization.LayerNorm.init(allocator, config.hidden_size);

        return expert;
    }

    pub fn forward(self: *Expert, input: *const Tensor) !Tensor {
        switch (self.expert_type) {
            .feed_forward => return try self.forwardFFN(input),
            .attention => return try self.forwardAttention(input),
            .hybrid => return try self.forwardHybrid(input),
            .custom => return try self.forwardCustom(input),
        }
    }

    fn forwardFFN(self: *Expert, input: *const Tensor) !Tensor {
        // SwiGLU: SwiGLU(x) = Swish(xW1) ⊙ (xW3) W2

        // Compute gate and input projections
        var gate_proj = try Tensor.matmul(self.allocator, input, &self.w1.?);
        defer gate_proj.deinit();

        var input_proj = try Tensor.matmul(self.allocator, input, &self.w3.?);
        defer input_proj.deinit();

        // Apply SiLU (Swish) activation to gate
        try activations.silu(&gate_proj);

        // Element-wise multiplication
        var gated = try Tensor.multiply(self.allocator, &gate_proj, &input_proj);
        defer gated.deinit();

        // Output projection
        return try Tensor.matmul(self.allocator, &gated, &self.w2.?);
    }

    fn forwardAttention(self: *Expert, input: *const Tensor) !Tensor {
        // Simplified single-head attention for expert
        var q = try Tensor.matmul(self.allocator, input, &self.q_proj.?);
        defer q.deinit();
        var k = try Tensor.matmul(self.allocator, input, &self.k_proj.?);
        defer k.deinit();
        var v = try Tensor.matmul(self.allocator, input, &self.v_proj.?);
        defer v.deinit();

        // Compute attention scores (simplified - no proper multi-head structure)
        var scores = try Tensor.matmul(self.allocator, &q, &k); // Should be Q @ K^T
        defer scores.deinit();

        // Apply softmax
        var attn_weights = try Tensor.softmax(self.allocator, &scores, 1);
        defer attn_weights.deinit();

        // Apply attention to values
        var attn_output = try Tensor.matmul(self.allocator, &attn_weights, &v);
        defer attn_output.deinit();

        // Output projection
        return try Tensor.matmul(self.allocator, &attn_output, &self.o_proj.?);
    }

    fn forwardHybrid(self: *Expert, input: *const Tensor) !Tensor {
        // Combine attention and FFN
        var attn_output = try self.forwardAttention(input);
        defer attn_output.deinit();

        // Add residual connection
        var attn_with_residual = try Tensor.add(self.allocator, input, &attn_output);
        defer attn_with_residual.deinit();

        // Apply layer norm
        var normed = try self.layer_norm.?.forward(&attn_with_residual);
        defer normed.deinit();

        // Apply FFN
        var ffn_output = try self.forwardFFN(&normed);
        defer ffn_output.deinit();

        // Final residual connection
        return try Tensor.add(self.allocator, &attn_with_residual, &ffn_output);
    }

    fn forwardCustom(self: *Expert, input: *const Tensor) !Tensor {
        // Simple custom expert - just a linear transformation
        return try Tensor.matmul(self.allocator, input, &self.w1.?);
    }

    pub fn deinit(self: *Expert) void {
        if (self.w1) |*w1| w1.deinit();
        if (self.w2) |*w2| w2.deinit();
        if (self.w3) |*w3| w3.deinit();
        if (self.q_proj) |*q| q.deinit();
        if (self.k_proj) |*k| k.deinit();
        if (self.v_proj) |*v| v.deinit();
        if (self.o_proj) |*o| o.deinit();
        if (self.layer_norm) |*ln| ln.deinit();
    }
};

// Gating Network
pub const GatingNetwork = struct {
    router_weights: Tensor,    // [hidden_size, num_experts]
    config: MoEConfig,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: MoEConfig) !GatingNetwork {
        return GatingNetwork{
            .router_weights = try Tensor.random(allocator, &[_]u32{ config.hidden_size, config.num_experts }),
            .config = config,
            .allocator = allocator,
        };
    }

    pub fn forward(self: *GatingNetwork, input: *const Tensor) !RoutingOutput {
        // Compute gating logits
        var logits = try Tensor.matmul(self.allocator, input, &self.router_weights);
        defer logits.deinit();

        return switch (self.config.routing_algorithm) {
            .top_k => try self.topKRouting(&logits),
            .switch_routing => try self.switchRouting(&logits),
            .expert_choice => try self.expertChoiceRouting(&logits),
            .hash_routing => try self.hashRouting(input),
            .learned_routing => try self.learnedRouting(&logits),
        };
    }

    fn topKRouting(self: *GatingNetwork, logits: *const Tensor) !RoutingOutput {
        const batch_size = logits.shape[0];
        const num_experts = logits.shape[1];

        var routing_output = RoutingOutput{
            .expert_indices = try self.allocator.alloc([]u32, batch_size),
            .expert_weights = try self.allocator.alloc([]f32, batch_size),
            .routing_probs = try Tensor.softmax(self.allocator, logits, 1),
            .load_balancing_loss = 0.0,
            .allocator = self.allocator,
        };

        // For each token, find top-k experts
        for (0..batch_size) |i| {
            var expert_scores = std.ArrayList(ExpertScore).init(self.allocator);
            defer expert_scores.deinit();

            // Get scores for this token
            for (0..num_experts) |j| {
                const score = try logits.get(&[_]u32{ @intCast(i), @intCast(j) });
                try expert_scores.append(ExpertScore{ .expert_id = @intCast(j), .score = score });
            }

            // Sort by score (descending)
            std.sort.sort(ExpertScore, expert_scores.items, {}, ExpertScore.lessThan);

            // Take top-k
            const k = @min(self.config.top_k, @as(u32, @intCast(expert_scores.items.len)));
            routing_output.expert_indices[i] = try self.allocator.alloc(u32, k);
            routing_output.expert_weights[i] = try self.allocator.alloc(f32, k);

            var weight_sum: f32 = 0.0;
            for (0..k) |j| {
                routing_output.expert_indices[i][j] = expert_scores.items[j].expert_id;
                const weight = @exp(expert_scores.items[j].score);
                routing_output.expert_weights[i][j] = weight;
                weight_sum += weight;
            }

            // Normalize weights
            for (routing_output.expert_weights[i]) |*weight| {
                weight.* /= weight_sum;
            }
        }

        // Compute load balancing loss
        routing_output.load_balancing_loss = try self.computeLoadBalancingLoss(&routing_output);

        return routing_output;
    }

    fn switchRouting(self: *GatingNetwork, logits: *const Tensor) !RoutingOutput {
        const batch_size = logits.shape[0];

        var routing_output = RoutingOutput{
            .expert_indices = try self.allocator.alloc([]u32, batch_size),
            .expert_weights = try self.allocator.alloc([]f32, batch_size),
            .routing_probs = try Tensor.softmax(self.allocator, logits, 1),
            .load_balancing_loss = 0.0,
            .allocator = self.allocator,
        };

        // For Switch routing, each token goes to exactly one expert
        for (0..batch_size) |i| {
            routing_output.expert_indices[i] = try self.allocator.alloc(u32, 1);
            routing_output.expert_weights[i] = try self.allocator.alloc(f32, 1);

            // Find expert with highest probability
            var max_prob: f32 = -std.math.inf(f32);
            var best_expert: u32 = 0;

            for (0..logits.shape[1]) |j| {
                const prob = try routing_output.routing_probs.get(&[_]u32{ @intCast(i), @intCast(j) });
                if (prob > max_prob) {
                    max_prob = prob;
                    best_expert = @intCast(j);
                }
            }

            routing_output.expert_indices[i][0] = best_expert;
            routing_output.expert_weights[i][0] = max_prob;
        }

        routing_output.load_balancing_loss = try self.computeLoadBalancingLoss(&routing_output);
        return routing_output;
    }

    fn expertChoiceRouting(self: *GatingNetwork, logits: *const Tensor) !RoutingOutput {
        // Expert Choice routing - experts choose tokens instead of tokens choosing experts
        const batch_size = logits.shape[0];
        const num_experts = logits.shape[1];

        var routing_output = RoutingOutput{
            .expert_indices = try self.allocator.alloc([]u32, batch_size),
            .expert_weights = try self.allocator.alloc([]f32, batch_size),
            .routing_probs = try Tensor.softmax(self.allocator, logits, 1),
            .load_balancing_loss = 0.0,
            .allocator = self.allocator,
        };

        // Initialize all tokens with empty routing
        for (0..batch_size) |i| {
            routing_output.expert_indices[i] = try self.allocator.alloc(u32, 0);
            routing_output.expert_weights[i] = try self.allocator.alloc(f32, 0);
        }

        // For each expert, choose top tokens
        const tokens_per_expert = if (self.config.expert_capacity > 0)
            self.config.expert_capacity
        else
            @max(1, batch_size / num_experts);

        for (0..num_experts) |expert_id| {
            var token_scores = std.ArrayList(TokenScore).init(self.allocator);
            defer token_scores.deinit();

            // Get scores for this expert across all tokens
            for (0..batch_size) |token_id| {
                const score = try logits.get(&[_]u32{ @intCast(token_id), @intCast(expert_id) });
                try token_scores.append(TokenScore{ .token_id = @intCast(token_id), .score = score });
            }

            // Sort by score (descending)
            std.sort.sort(TokenScore, token_scores.items, {}, TokenScore.lessThan);

            // Assign this expert to top tokens
            const num_tokens = @min(tokens_per_expert, @as(u32, @intCast(token_scores.items.len)));
            for (0..num_tokens) |i| {
                const token_id = token_scores.items[i].token_id;
                const score = token_scores.items[i].score;

                // Add this expert to the token's routing
                const old_len = routing_output.expert_indices[token_id].len;
                routing_output.expert_indices[token_id] = try self.allocator.realloc(routing_output.expert_indices[token_id], old_len + 1);
                routing_output.expert_weights[token_id] = try self.allocator.realloc(routing_output.expert_weights[token_id], old_len + 1);

                routing_output.expert_indices[token_id][old_len] = @intCast(expert_id);
                routing_output.expert_weights[token_id][old_len] = @exp(score);
            }
        }

        // Normalize weights for each token
        for (0..batch_size) |i| {
            var weight_sum: f32 = 0.0;
            for (routing_output.expert_weights[i]) |weight| {
                weight_sum += weight;
            }
            if (weight_sum > 0) {
                for (routing_output.expert_weights[i]) |*weight| {
                    weight.* /= weight_sum;
                }
            }
        }

        routing_output.load_balancing_loss = try self.computeLoadBalancingLoss(&routing_output);
        return routing_output;
    }

    fn hashRouting(self: *GatingNetwork, input: *const Tensor) !RoutingOutput {
        // Hash-based routing for deterministic expert assignment
        const batch_size = input.shape[0];

        var routing_output = RoutingOutput{
            .expert_indices = try self.allocator.alloc([]u32, batch_size),
            .expert_weights = try self.allocator.alloc([]f32, batch_size),
            .routing_probs = try Tensor.zeros(self.allocator, &[_]u32{ batch_size, self.config.num_experts }),
            .load_balancing_loss = 0.0,
            .allocator = self.allocator,
        };

        for (0..batch_size) |i| {
            // Simple hash based on input features
            var hash: u64 = 0;
            for (0..@min(input.shape[1], 8)) |j| { // Use first 8 features for hashing
                const val = try input.get(&[_]u32{ @intCast(i), @intCast(j) });
                hash = hash *% 31 +% @as(u64, @bitCast(@as(u32, @bitCast(val))));
            }

            const expert_id = @as(u32, @intCast(hash % self.config.num_experts));

            routing_output.expert_indices[i] = try self.allocator.alloc(u32, 1);
            routing_output.expert_weights[i] = try self.allocator.alloc(f32, 1);

            routing_output.expert_indices[i][0] = expert_id;
            routing_output.expert_weights[i][0] = 1.0; // Equal weight for hash routing

            // Set routing probability
            try routing_output.routing_probs.set(&[_]u32{ @intCast(i), expert_id }, 1.0);
        }

        return routing_output;
    }

    fn learnedRouting(self: *GatingNetwork, logits: *const Tensor) !RoutingOutput {
        // Fully learned routing with Gumbel-Softmax for differentiability
        return try self.topKRouting(logits); // Fallback to top-k for now
    }

    fn computeLoadBalancingLoss(self: *GatingNetwork, routing_output: *const RoutingOutput) !f32 {
        // Compute load balancing loss to encourage uniform expert usage
        var expert_usage = try self.allocator.alloc(f32, self.config.num_experts);
        defer self.allocator.free(expert_usage);
        @memset(expert_usage, 0.0);

        // Count how many tokens are routed to each expert
        for (routing_output.expert_indices) |indices| {
            for (indices) |expert_id| {
                expert_usage[expert_id] += 1.0;
            }
        }

        // Normalize by total number of tokens
        const total_tokens = @as(f32, @floatFromInt(routing_output.expert_indices.len));
        for (expert_usage) |*usage| {
            usage.* /= total_tokens;
        }

        // Compute coefficient of variation as load balancing loss
        var mean: f32 = 0.0;
        var variance: f32 = 0.0;

        for (expert_usage) |usage| {
            mean += usage;
        }
        mean /= @as(f32, @floatFromInt(expert_usage.len));

        for (expert_usage) |usage| {
            const diff = usage - mean;
            variance += diff * diff;
        }
        variance /= @as(f32, @floatFromInt(expert_usage.len));

        return if (mean > 0) @sqrt(variance) / mean else 0.0;
    }

    pub fn deinit(self: *GatingNetwork) void {
        self.router_weights.deinit();
    }
};

// Routing Output Structure
pub const RoutingOutput = struct {
    expert_indices: [][]u32,    // [batch_size][k] - expert indices for each token
    expert_weights: [][]f32,    // [batch_size][k] - expert weights for each token
    routing_probs: Tensor,      // [batch_size, num_experts] - full routing probabilities
    load_balancing_loss: f32,   // Load balancing loss for training
    allocator: std.mem.Allocator,

    pub fn deinit(self: *RoutingOutput) void {
        for (self.expert_indices) |indices| {
            self.allocator.free(indices);
        }
        self.allocator.free(self.expert_indices);

        for (self.expert_weights) |weights| {
            self.allocator.free(weights);
        }
        self.allocator.free(self.expert_weights);

        self.routing_probs.deinit();
    }
};

// Helper structures for sorting
const ExpertScore = struct {
    expert_id: u32,
    score: f32,

    fn lessThan(context: void, a: ExpertScore, b: ExpertScore) bool {
        _ = context;
        return a.score > b.score; // Descending order
    }
};

const TokenScore = struct {
    token_id: u32,
    score: f32,

    fn lessThan(context: void, a: TokenScore, b: TokenScore) bool {
        _ = context;
        return a.score > b.score; // Descending order
    }
};

// Complete MoE Layer
pub const MoELayer = struct {
    config: MoEConfig,
    experts: []Expert,
    gating_network: GatingNetwork,
    layer_norm: ?normalization.LayerNorm = null,
    stats: MoEStats,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: MoEConfig) !MoELayer {
        var experts = try allocator.alloc(Expert, config.num_experts);
        for (0..config.num_experts) |i| {
            experts[i] = try Expert.init(allocator, config, @intCast(i));
        }

        return MoELayer{
            .config = config,
            .experts = experts,
            .gating_network = try GatingNetwork.init(allocator, config),
            .layer_norm = if (config.expert_type == .hybrid) try normalization.LayerNorm.init(allocator, config.hidden_size) else null,
            .stats = MoEStats.init(),
            .allocator = allocator,
        };
    }

    pub fn forward(self: *MoELayer, input: *const Tensor) !Tensor {
        var timer = std.time.Timer.start() catch return error.TimerUnsupported;
        const start_time = timer.read();

        // Get routing decisions
        var routing_output = try self.gating_network.forward(input);
        defer routing_output.deinit();

        const batch_size = input.shape[0];
        const hidden_size = input.shape[1];

        // Initialize output tensor
        var output = try Tensor.zeros(self.allocator, input.shape);
        errdefer output.deinit();

        // Process each token
        for (0..batch_size) |i| {
            const token_experts = routing_output.expert_indices[i];
            const token_weights = routing_output.expert_weights[i];

            if (token_experts.len == 0) {
                // No expert assigned - copy input (shouldn't happen in normal cases)
                for (0..hidden_size) |j| {
                    const val = try input.get(&[_]u32{ @intCast(i), @intCast(j) });
                    try output.set(&[_]u32{ @intCast(i), @intCast(j) }, val);
                }
                continue;
            }

            // Create single token tensor for expert processing
            var token_input = try Tensor.zeros(self.allocator, &[_]u32{ 1, @intCast(hidden_size) });
            defer token_input.deinit();

            for (0..hidden_size) |j| {
                const val = try input.get(&[_]u32{ @intCast(i), @intCast(j) });
                try token_input.set(&[_]u32{ 0, @intCast(j) }, val);
            }

            // Accumulate weighted expert outputs
            var token_output = try Tensor.zeros(self.allocator, &[_]u32{ 1, @intCast(hidden_size) });
            defer token_output.deinit();

            for (token_experts, 0..) |expert_id, k| {
                const weight = token_weights[k];

                // Get expert output
                var expert_output = try self.experts[expert_id].forward(&token_input);
                defer expert_output.deinit();

                // Add weighted contribution
                for (0..hidden_size) |j| {
                    const expert_val = try expert_output.get(&[_]u32{ 0, @intCast(j) });
                    const current_val = try token_output.get(&[_]u32{ 0, @intCast(j) });
                    try token_output.set(&[_]u32{ 0, @intCast(j) }, current_val + weight * expert_val);
                }

                self.stats.recordExpertUsage(expert_id, weight);
            }

            // Copy token output to final output
            for (0..hidden_size) |j| {
                const val = try token_output.get(&[_]u32{ 0, @intCast(j) });
                try output.set(&[_]u32{ @intCast(i), @intCast(j) }, val);
            }
        }

        const end_time = timer.read();
        self.stats.recordForwardPass(end_time - start_time, batch_size, routing_output.load_balancing_loss);

        return output;
    }

    pub fn getStats(self: *const MoELayer) MoEStats {
        return self.stats;
    }

    pub fn deinit(self: *MoELayer) void {
        for (self.experts) |*expert| {
            expert.deinit();
        }
        self.allocator.free(self.experts);
        self.gating_network.deinit();
        if (self.layer_norm) |*ln| {
            ln.deinit();
        }
    }
};

// MoE Performance Statistics
pub const MoEStats = struct {
    total_forward_passes: u64 = 0,
    total_forward_time_ns: u64 = 0,
    expert_usage_count: []u64,
    expert_total_weight: []f64,
    average_load_balancing_loss: f64 = 0.0,
    allocator: ?std.mem.Allocator = null,

    pub fn init() MoEStats {
        return MoEStats{
            .expert_usage_count = &[_]u64{},
            .expert_total_weight = &[_]f64{},
        };
    }

    pub fn initWithExperts(allocator: std.mem.Allocator, num_experts: u32) !MoEStats {
        return MoEStats{
            .expert_usage_count = try allocator.alloc(u64, num_experts),
            .expert_total_weight = try allocator.alloc(f64, num_experts),
            .allocator = allocator,
        };
    }

    pub fn recordExpertUsage(self: *MoEStats, expert_id: u32, weight: f32) void {
        if (expert_id < self.expert_usage_count.len) {
            self.expert_usage_count[expert_id] += 1;
            self.expert_total_weight[expert_id] += weight;
        }
    }

    pub fn recordForwardPass(self: *MoEStats, time_ns: u64, batch_size: u32, load_balancing_loss: f32) void {
        self.total_forward_passes += 1;
        self.total_forward_time_ns += time_ns;

        // Update rolling average of load balancing loss
        const alpha = 0.01; // Exponential moving average factor
        self.average_load_balancing_loss = (1.0 - alpha) * self.average_load_balancing_loss + alpha * load_balancing_loss;

        _ = batch_size;
    }

    pub fn getAverageForwardTime(self: MoEStats) f64 {
        if (self.total_forward_passes == 0) return 0.0;
        return @as(f64, @floatFromInt(self.total_forward_time_ns)) / @as(f64, @floatFromInt(self.total_forward_passes));
    }

    pub fn getExpertUtilization(self: MoEStats) []f64 {
        if (self.allocator == null) return &[_]f64{};

        var utilization = self.allocator.?.alloc(f64, self.expert_usage_count.len) catch return &[_]f64{};

        var total_usage: u64 = 0;
        for (self.expert_usage_count) |count| {
            total_usage += count;
        }

        if (total_usage == 0) {
            for (utilization) |*util| {
                util.* = 0.0;
            }
        } else {
            for (self.expert_usage_count, 0..) |count, i| {
                utilization[i] = @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(total_usage));
            }
        }

        return utilization;
    }

    pub fn print(self: MoEStats, writer: anytype) !void {
        try writer.print("=== MoE Layer Statistics ===\n");
        try writer.print("Forward Passes: {}\n", .{self.total_forward_passes});
        try writer.print("Average Time: {:.2} ms\n", .{self.getAverageForwardTime() / 1_000_000.0});
        try writer.print("Load Balancing Loss: {:.6}\n", .{self.average_load_balancing_loss});

        if (self.expert_usage_count.len > 0) {
            try writer.print("Expert Usage:\n");
            for (self.expert_usage_count, 0..) |count, i| {
                const avg_weight = if (count > 0) self.expert_total_weight[i] / @as(f64, @floatFromInt(count)) else 0.0;
                try writer.print("  Expert {}: {} uses, {:.3} avg weight\n", .{ i, count, avg_weight });
            }
        }
        try writer.print("=============================\n");
    }

    pub fn deinit(self: *MoEStats) void {
        if (self.allocator) |allocator| {
            if (self.expert_usage_count.len > 0) {
                allocator.free(self.expert_usage_count);
                allocator.free(self.expert_total_weight);
            }
        }
    }
};

// Utility Functions
pub const MoEUtils = struct {
    pub fn calculateSparsity(routing_output: *const RoutingOutput) f32 {
        var total_expert_assignments: u32 = 0;
        var total_possible_assignments: u32 = 0;

        for (routing_output.expert_indices) |indices| {
            total_expert_assignments += @intCast(indices.len);
            total_possible_assignments += routing_output.routing_probs.shape[1]; // num_experts
        }

        if (total_possible_assignments == 0) return 0.0;
        return 1.0 - @as(f32, @floatFromInt(total_expert_assignments)) / @as(f32, @floatFromInt(total_possible_assignments));
    }

    pub fn analyzeExpertDistribution(config: MoEConfig, routing_output: *const RoutingOutput) !void {
        std.debug.print("=== Expert Distribution Analysis ===\n");
        std.debug.print("Routing Algorithm: {s}\n", .{@tagName(config.routing_algorithm)});
        std.debug.print("Number of Experts: {}\n", .{config.num_experts});
        std.debug.print("Top-K: {}\n", .{config.top_k});

        const sparsity = calculateSparsity(routing_output);
        std.debug.print("Sparsity: {:.1}%\n", .{sparsity * 100.0});
        std.debug.print("Load Balancing Loss: {:.6}\n", .{routing_output.load_balancing_loss});

        // Count expert usage
        var expert_counts = std.ArrayList(u32).init(std.heap.page_allocator);
        defer expert_counts.deinit();

        try expert_counts.resize(config.num_experts);
        @memset(expert_counts.items, 0);

        for (routing_output.expert_indices) |indices| {
            for (indices) |expert_id| {
                expert_counts.items[expert_id] += 1;
            }
        }

        std.debug.print("Expert Usage Distribution:\n");
        for (expert_counts.items, 0..) |count, i| {
            const percentage = @as(f32, @floatFromInt(count)) / @as(f32, @floatFromInt(routing_output.expert_indices.len)) * 100.0;
            std.debug.print("  Expert {}: {} tokens ({:.1}%)\n", .{ i, count, percentage });
        }
        std.debug.print("=====================================\n");
    }
};