//! Transformer Components: Feed-Forward Networks
//!
//! This module implements the feed-forward networks (FFN) used in transformer
//! architectures, including both classic and modern variants with gated
//! linear units.
//!
//! ## Educational Objectives
//! - Understand the role of FFNs in transformer processing
//! - Learn why FFNs use much larger hidden dimensions
//! - Implement modern gated activations (SwiGLU, GeGLU)
//! - Connect FFN design choices to model capacity and efficiency
//!
//! ## Transformer Context
//! Feed-forward networks provide the "processing power" in transformers:
//! - **Point-wise Processing**: Apply same transformation to each position
//! - **Capacity**: Typically 4x model dimension for hidden layer
//! - **Non-linearity**: Where most of the model's expressive power comes from
//! - **Parallelization**: Completely parallel across sequence positions

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

// Import our layers
const Tensor = @import("../foundation/tensor.zig").Tensor;
const TensorError = @import("../foundation/tensor.zig").TensorError;
const matrix_ops = @import("../linear_algebra/matrix_ops.zig");
const activations = @import("../neural_primitives/activations.zig");

/// Feed-forward network architectures used in different transformer variants
pub const FFNType = enum {
    Standard,   // Original transformer: Linear -> ReLU -> Linear
    GELU,       // BERT/GPT style: Linear -> GELU -> Linear
    SwiGLU,     // LLaMA style: SwiGLU gated activation
    GeGLU,      // Alternative gated: GeGLU activation
    GLU,        // Classic gated: GLU activation
};

/// Feed-Forward Network Layer
///
/// ## Mathematical Definition
///
/// **Standard FFN (Original Transformer):**
/// ```
/// FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
/// ```
///
/// **Modern Gated FFN (LLaMA SwiGLU):**
/// ```
/// FFN(x) = SwiGLU(xW_gate, xW_up)W_down
/// where SwiGLU(a,b) = a ⊙ SiLU(b)
/// ```
///
/// ## Educational Note: Why Large Hidden Dimensions?
/// FFNs typically use hidden dimensions 4x the model dimension:
///
/// 1. **Expressivity**: More parameters = more representational capacity
/// 2. **Bottleneck**: Model dim → Large hidden → Model dim creates information bottleneck
/// 3. **Empirical**: 4x ratio found to work well across many architectures
/// 4. **Compute**: Modern hardware optimized for these matrix shapes
///
/// ## Memory vs Compute Trade-off
/// - **Standard FFN**: 2 weight matrices, simple activation
/// - **Gated FFN**: 3 weight matrices, more complex but often better performance
/// - **Parameter count**: Gated FFN ~1.5x parameters for similar capacity
pub const FeedForward = struct {
    /// FFN architecture type
    ffn_type: FFNType,

    /// Model dimension (input/output)
    d_model: usize,

    /// Hidden dimension (typically 4 * d_model)
    d_ff: usize,

    /// First linear transformation [d_model × d_ff] (or gate for gated variants)
    w1: Tensor(f32),

    /// Second linear transformation [d_ff × d_model] (or up projection for gated)
    w2: Tensor(f32),

    /// Third linear transformation [d_ff × d_model] (only for gated variants)
    w3: ?Tensor(f32),

    /// Memory allocator
    allocator: Allocator,

    /// Initialize feed-forward network
    ///
    /// ## Educational Note: Parameter Scaling
    /// FFN parameters scale quadratically with dimensions:
    /// - Standard FFN: 2 * d_model * d_ff parameters
    /// - Gated FFN: 3 * d_model * d_ff parameters
    /// - For LLaMA-7B: ~21B parameters just in FFN layers!
    ///
    /// This is why efficient implementations and quantization are crucial.
    pub fn init(allocator: Allocator, d_model: usize, d_ff: usize, ffn_type: FFNType) !FeedForward {
        var w1 = try Tensor(f32).init(allocator, &[_]usize{ d_model, d_ff });
        var w2 = try Tensor(f32).init(allocator, &[_]usize{ d_ff, d_model });
        var w3: ?Tensor(f32) = null;

        // Gated variants need a third weight matrix
        if (ffn_type == .SwiGLU or ffn_type == .GeGLU or ffn_type == .GLU) {
            w3 = try Tensor(f32).init(allocator, &[_]usize{ d_model, d_ff });
        }

        // Initialize weights with appropriate scaling
        const fan_in = @as(f32, @floatFromInt(d_model));
        const fan_out = @as(f32, @floatFromInt(d_ff));
        const xavier_std = @sqrt(2.0 / (fan_in + fan_out));

        initializeWeights(&w1, xavier_std);
        initializeWeights(&w2, @sqrt(2.0 / (fan_out + fan_in))); // Reverse for w2
        if (w3) |*w3_ptr| {
            initializeWeights(w3_ptr, xavier_std);
        }

        return FeedForward{
            .ffn_type = ffn_type,
            .d_model = d_model,
            .d_ff = d_ff,
            .w1 = w1,
            .w2 = w2,
            .w3 = w3,
            .allocator = allocator,
        };
    }

    /// Clean up resources
    pub fn deinit(self: *FeedForward) void {
        self.w1.deinit();
        self.w2.deinit();
        if (self.w3) |*w3| {
            w3.deinit();
        }
    }

    /// Forward pass through feed-forward network
    ///
    /// ## Input Shape
    /// - input: [batch_size, seq_len, d_model]
    ///
    /// ## Output Shape
    /// - output: [batch_size, seq_len, d_model]
    ///
    /// ## Educational Note: Position-wise Processing
    /// FFNs are applied identically to each position in the sequence:
    /// - Same weights shared across all positions
    /// - No interaction between positions (unlike attention)
    /// - Can be parallelized perfectly across sequence dimension
    pub fn forward(self: *const FeedForward, input: Tensor(f32)) !Tensor(f32) {
        return switch (self.ffn_type) {
            .Standard => self.forwardStandard(input),
            .GELU => self.forwardGELU(input),
            .SwiGLU => self.forwardSwiGLU(input),
            .GeGLU => self.forwardGeGLU(input),
            .GLU => self.forwardGLU(input),
        };
    }

    /// Standard FFN: Linear -> ReLU -> Linear
    fn forwardStandard(self: *const FeedForward, input: Tensor(f32)) !Tensor(f32) {
        // First linear transformation: [batch, seq, d_model] @ [d_model, d_ff] -> [batch, seq, d_ff]
        var hidden = try input.matmul(self.w1, self.allocator);
        defer hidden.deinit();

        // Apply ReLU activation
        var activated = try activations.relu(f32, hidden);
        defer activated.deinit();

        // Second linear transformation: [batch, seq, d_ff] @ [d_ff, d_model] -> [batch, seq, d_model]
        return try activated.matmul(self.w2, self.allocator);
    }

    /// GELU FFN: Linear -> GELU -> Linear (BERT/GPT style)
    fn forwardGELU(self: *const FeedForward, input: Tensor(f32)) !Tensor(f32) {
        var hidden = try input.matmul(self.w1, self.allocator);
        defer hidden.deinit();

        var activated = try activations.gelu(f32, hidden, self.allocator);
        defer activated.deinit();

        return try activated.matmul(self.w2, self.allocator);
    }

    /// SwiGLU FFN: Gated with SiLU activation (LLaMA style)
    ///
    /// ## Mathematical Definition
    /// ```
    /// gate = input @ W_gate
    /// up = input @ W_up
    /// hidden = gate ⊙ SiLU(up)
    /// output = hidden @ W_down
    /// ```
    ///
    /// ## Educational Note: Why Gating Works
    /// Gating mechanisms allow the network to selectively process information:
    /// - **Gate values**: Control how much information flows through
    /// - **Up projection**: Provides the actual content to be processed
    /// - **Element-wise product**: Combines gating with content
    /// - **Empirical success**: Consistently outperforms standard FFNs
    fn forwardSwiGLU(self: *const FeedForward, input: Tensor(f32)) !Tensor(f32) {
        if (self.w3 == null) return TensorError.IncompatibleShapes;

        // Gate projection
        var gate = try input.matmul(self.w1, self.allocator);
        defer gate.deinit();

        // Up projection
        var up = try input.matmul(self.w3.?, self.allocator);
        defer up.deinit();

        // Apply SiLU to up projection
        var up_activated = try activations.silu(f32, up, self.allocator);
        defer up_activated.deinit();

        // Element-wise multiplication (gating)
        var gated = try gate.elementWiseMultiply(up_activated, self.allocator);
        defer gated.deinit();

        // Down projection
        return try gated.matmul(self.w2, self.allocator);
    }

    /// GeGLU FFN: Gated with GELU activation
    fn forwardGeGLU(self: *const FeedForward, input: Tensor(f32)) !Tensor(f32) {
        if (self.w3 == null) return TensorError.IncompatibleShapes;

        var gate = try input.matmul(self.w1, self.allocator);
        defer gate.deinit();

        var up = try input.matmul(self.w3.?, self.allocator);
        defer up.deinit();

        var up_activated = try activations.gelu(f32, up, self.allocator);
        defer up_activated.deinit();

        var gated = try gate.elementWiseMultiply(up_activated, self.allocator);
        defer gated.deinit();

        return try gated.matmul(self.w2, self.allocator);
    }

    /// GLU FFN: Gated with sigmoid activation
    fn forwardGLU(self: *const FeedForward, input: Tensor(f32)) !Tensor(f32) {
        if (self.w3 == null) return TensorError.IncompatibleShapes;

        var gate = try input.matmul(self.w1, self.allocator);
        defer gate.deinit();

        var up = try input.matmul(self.w3.?, self.allocator);
        defer up.deinit();

        var up_activated = try activations.sigmoid(f32, up, self.allocator);
        defer up_activated.deinit();

        var gated = try gate.elementWiseMultiply(up_activated, self.allocator);
        defer gated.deinit();

        return try gated.matmul(self.w2, self.allocator);
    }
};

/// Expert Feed-Forward Network for Mixture of Experts (MoE)
///
/// ## Educational Note: Scaling Transformers
/// MoE is a key technique for scaling transformer models:
/// - **Sparse activation**: Only activate subset of parameters per token
/// - **Massive capacity**: Can have many more parameters than dense models
/// - **Efficiency**: Constant compute cost regardless of number of experts
/// - **Routing**: Learn which expert to use for each token
///
/// This is how models like Switch Transformer and PaLM-2 achieve massive scale.
pub const ExpertFFN = struct {
    /// Number of experts
    num_experts: usize,

    /// Individual expert networks
    experts: []FeedForward,

    /// Router network to select experts
    router: Tensor(f32), // [d_model × num_experts]

    /// Top-k experts to activate per token
    top_k: usize,

    allocator: Allocator,

    pub fn init(allocator: Allocator, d_model: usize, d_ff: usize, num_experts: usize, top_k: usize, ffn_type: FFNType) !ExpertFFN {
        // Initialize expert networks
        var experts = try allocator.alloc(FeedForward, num_experts);
        for (0..num_experts) |i| {
            experts[i] = try FeedForward.init(allocator, d_model, d_ff, ffn_type);
        }

        // Initialize router
        var router = try Tensor(f32).init(allocator, &[_]usize{ d_model, num_experts });
        const router_std = @sqrt(1.0 / @as(f32, @floatFromInt(d_model)));
        initializeWeights(&router, router_std);

        return ExpertFFN{
            .num_experts = num_experts,
            .experts = experts,
            .router = router,
            .top_k = top_k,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ExpertFFN) void {
        for (0..self.num_experts) |i| {
            self.experts[i].deinit();
        }
        self.allocator.free(self.experts);
        self.router.deinit();
    }

    /// Forward pass with expert routing
    ///
    /// ## Algorithm
    /// 1. **Route**: Compute routing probabilities for each token
    /// 2. **Select**: Choose top-k experts per token
    /// 3. **Process**: Apply selected experts to tokens
    /// 4. **Combine**: Weighted combination of expert outputs
    pub fn forward(self: *const ExpertFFN, input: Tensor(f32)) !Tensor(f32) {
        // For simplicity, we'll implement a basic version that uses all experts
        // Real MoE would do sophisticated routing and sparse activation

        var output = try Tensor(f32).init(self.allocator, input.shape);
        output.fill(0.0);

        // Simple equal weighting across all experts (not realistic but educational)
        const expert_weight = 1.0 / @as(f32, @floatFromInt(self.num_experts));

        for (0..self.num_experts) |expert_idx| {
            var expert_output = try self.experts[expert_idx].forward(input);
            defer expert_output.deinit();

            // Add weighted expert output
            for (0..output.size) |i| {
                output.data[i] += expert_output.data[i] * expert_weight;
            }
        }

        return output;
    }
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Initialize weights with given standard deviation
fn initializeWeights(tensor: *Tensor(f32), std_dev: f32) void {
    var seed: u32 = 54321;
    for (0..tensor.size) |i| {
        seed = seed *% 1664525 +% 1013904223;
        const rand_f32 = @as(f32, @floatFromInt(seed)) / @as(f32, @floatFromInt(std.math.maxInt(u32)));
        tensor.data[i] = (rand_f32 - 0.5) * 2.0 * std_dev;
    }
}

// Element-wise multiplication is now available in foundation/tensor.zig

// ============================================================================
// COMPREHENSIVE TESTS
// ============================================================================

test "Feed-forward network initialization" {
    const allocator = testing.allocator;

    var ffn = try FeedForward.init(allocator, 64, 256, .Standard);
    defer ffn.deinit();

    try testing.expectEqual(@as(usize, 64), ffn.d_model);
    try testing.expectEqual(@as(usize, 256), ffn.d_ff);
    try testing.expectEqual(FFNType.Standard, ffn.ffn_type);

    // Standard FFN should not have w3
    try testing.expect(ffn.w3 == null);

    // Check weight matrix shapes
    try testing.expectEqualSlices(usize, &[_]usize{ 64, 256 }, ffn.w1.shape);
    try testing.expectEqualSlices(usize, &[_]usize{ 256, 64 }, ffn.w2.shape);
}

test "SwiGLU FFN initialization" {
    const allocator = testing.allocator;

    var ffn = try FeedForward.init(allocator, 32, 128, .SwiGLU);
    defer ffn.deinit();

    try testing.expectEqual(FFNType.SwiGLU, ffn.ffn_type);

    // SwiGLU should have w3 for up projection
    try testing.expect(ffn.w3 != null);
    try testing.expectEqualSlices(usize, &[_]usize{ 32, 128 }, ffn.w3.?.shape);
}

test "Standard FFN forward pass shape preservation" {
    const allocator = testing.allocator;

    var ffn = try FeedForward.init(allocator, 16, 64, .Standard);
    defer ffn.deinit();

    // Input: [batch=2, seq_len=3, d_model=16]
    var input = try Tensor(f32).init(allocator, &[_]usize{ 2, 3, 16 });
    defer input.deinit();
    input.fill(0.1); // Small positive values

    var output = try ffn.forward(input);
    defer output.deinit();

    // Output should preserve batch and sequence dimensions
    try testing.expectEqualSlices(usize, &[_]usize{ 2, 3, 16 }, output.shape);
}

test "FFN parameter counting" {
    const allocator = testing.allocator;

    // Standard FFN parameters: 2 * d_model * d_ff
    const d_model: usize = 64;
    const d_ff: usize = 256;

    var standard_ffn = try FeedForward.init(allocator, d_model, d_ff, .Standard);
    defer standard_ffn.deinit();

    var gated_ffn = try FeedForward.init(allocator, d_model, d_ff, .SwiGLU);
    defer gated_ffn.deinit();

    // Standard: w1 + w2 = d_model*d_ff + d_ff*d_model = 2*d_model*d_ff
    const standard_params = standard_ffn.w1.size + standard_ffn.w2.size;
    const expected_standard = 2 * d_model * d_ff;
    try testing.expectEqual(expected_standard, standard_params);

    // Gated: w1 + w2 + w3 = 3*d_model*d_ff
    const gated_params = gated_ffn.w1.size + gated_ffn.w2.size + gated_ffn.w3.?.size;
    const expected_gated = 3 * d_model * d_ff;
    try testing.expectEqual(expected_gated, gated_params);

    // Gated should have 1.5x more parameters
    try testing.expectEqual(standard_params * 3, gated_params * 2);
}

test "Element-wise multiplication for gating" {
    const allocator = testing.allocator;

    var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer a.deinit();
    var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer b.deinit();

    // Set known values
    a.fill(2.0);
    b.fill(0.5);

    var result = try a.elementWiseMultiply(b, allocator);
    defer result.deinit();

    // Should be 2.0 * 0.5 = 1.0 everywhere
    for (0..result.size) |i| {
        try testing.expectEqual(@as(f32, 1.0), result.data[i]);
    }
}

test "Expert FFN basic functionality" {
    const allocator = testing.allocator;

    var expert_ffn = try ExpertFFN.init(allocator, 8, 32, 4, 2, .Standard);
    defer expert_ffn.deinit();

    try testing.expectEqual(@as(usize, 4), expert_ffn.num_experts);
    try testing.expectEqual(@as(usize, 2), expert_ffn.top_k);

    // Test forward pass
    var input = try Tensor(f32).init(allocator, &[_]usize{ 1, 2, 8 });
    defer input.deinit();
    input.fill(0.1);

    var output = try expert_ffn.forward(input);
    defer output.deinit();

    try testing.expectEqualSlices(usize, input.shape, output.shape);
}

test "FFN numerical stability" {
    const allocator = testing.allocator;

    var ffn = try FeedForward.init(allocator, 4, 16, .GELU);
    defer ffn.deinit();

    // Test with various input magnitudes
    const test_values = [_]f32{ -10.0, -1.0, 0.0, 1.0, 10.0 };

    for (test_values) |val| {
        var input = try Tensor(f32).init(allocator, &[_]usize{ 1, 1, 4 });
        defer input.deinit();
        input.fill(val);

        var output = try ffn.forward(input);
        defer output.deinit();

        // Check that outputs are finite
        for (0..output.size) |i| {
            try testing.expect(std.math.isFinite(output.data[i]));
        }
    }
}