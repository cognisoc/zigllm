//! Neural Primitives: Activation Functions
//!
//! This module implements the activation functions commonly used in transformer
//! models, with educational focus on their mathematical properties and role
//! in neural networks.
//!
//! ## Educational Objectives
//! - Understand how activation functions introduce non-linearity
//! - Learn the mathematical properties of different activation functions
//! - Connect activation choices to transformer architecture decisions
//! - Implement both scalar and vectorized versions for performance
//!
//! ## Transformer Context
//! Modern transformers use specific activation functions:
//! - **GELU**: Gaussian Error Linear Unit (used in BERT, GPT)
//! - **SiLU/Swish**: Sigmoid Linear Unit (used in LLaMA, PaLM)
//! - **ReLU**: Rectified Linear Unit (classic, simple)
//! - **GLU variants**: Gated Linear Units for enhanced expressivity

const std = @import("std");
const math = std.math;
const testing = std.testing;
const Allocator = std.mem.Allocator;
const Tensor = @import("../foundation/tensor.zig").Tensor;
const TensorError = @import("../foundation/tensor.zig").TensorError;

/// Activation function types supported in transformers
pub const ActivationType = enum {
    ReLU,
    GELU,
    SiLU,   // Also known as Swish
    GLU,    // Gated Linear Unit
    GeGLU,  // GELU-based Gated Linear Unit
    SwiGLU, // SiLU-based Gated Linear Unit
    Tanh,
    Sigmoid,
};

/// Mathematical constants for activation function approximations
const GELU_COEFF_A = 0.044715;
const GELU_SQRT_2_OVER_PI = 0.7978845608028654; // sqrt(2/π)

/// Rectified Linear Unit (ReLU)
///
/// ## Mathematical Definition
/// ```
/// ReLU(x) = max(0, x) = { x if x > 0, 0 otherwise }
/// ```
///
/// ## Properties
/// - **Range**: [0, +∞)
/// - **Derivative**: { 1 if x > 0, 0 if x ≤ 0 }
/// - **Computational Cost**: Minimal (single comparison)
/// - **Gradient**: Non-vanishing for positive inputs, dead neurons for negative
///
/// ## Transformer Usage
/// While ReLU was popular in early neural networks, modern transformers
/// prefer smoother activations like GELU or SiLU for better gradient flow.
pub fn relu(comptime T: type, input: Tensor(T)) TensorError!Tensor(T) {
    if (T != f32 and T != f64) return TensorError.InvalidShape; // Type safety

    var result = try Tensor(T).init(input.allocator, input.shape);

    // Vectorized ReLU implementation
    for (0..input.size) |i| {
        result.data[i] = @max(0.0, input.data[i]);
    }

    return result;
}

/// Gaussian Error Linear Unit (GELU)
///
/// ## Mathematical Definition
/// ```
/// GELU(x) = x * Φ(x) = x * (1/2) * (1 + erf(x/√2))
/// ```
/// Where Φ(x) is the cumulative distribution function of standard normal distribution.
///
/// ## Approximation (Tanh-based)
/// For computational efficiency, we use the tanh approximation:
/// ```
/// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
/// ```
///
/// ## Properties
/// - **Range**: (-0.17, +∞) approximately
/// - **Smooth**: Infinitely differentiable
/// - **Probabilistic**: Based on Gaussian CDF
/// - **Gradient**: Non-zero everywhere, preventing dead neurons
///
/// ## Transformer Usage
/// GELU is widely used in:
/// - **BERT**: Feed-forward layers
/// - **GPT-2/3**: MLP activations
/// - **T5**: Dense layer activations
///
/// The smooth activation helps with gradient flow in deep networks.
pub fn gelu(comptime T: type, input: Tensor(T), allocator: Allocator) TensorError!Tensor(T) {
    if (T != f32 and T != f64) return TensorError.InvalidShape;

    var result = try Tensor(T).init(allocator, input.shape);

    // Tanh-based approximation for computational efficiency
    for (0..input.size) |i| {
        const x = input.data[i];
        const cubic_term = GELU_COEFF_A * x * x * x;
        const inner = GELU_SQRT_2_OVER_PI * (x + cubic_term);
        result.data[i] = 0.5 * x * (1.0 + std.math.tanh(inner));
    }

    return result;
}

/// Sigmoid Linear Unit (SiLU) / Swish
///
/// ## Mathematical Definition
/// ```
/// SiLU(x) = x * sigmoid(x) = x * (1 / (1 + e^(-x)))
/// ```
///
/// ## Properties
/// - **Range**: (-0.28, +∞) approximately
/// - **Smooth**: Infinitely differentiable
/// - **Self-Gating**: Uses its own values for gating
/// - **Asymptotic**: Approaches 0 for x → -∞, approaches x for x → +∞
///
/// ## Advantages over ReLU
/// - No dead neurons (gradient always non-zero)
/// - Smooth activation prevents gradient discontinuities
/// - Self-normalizing properties
///
/// ## Transformer Usage
/// SiLU is used in modern architectures:
/// - **LLaMA**: Feed-forward network activations
/// - **PaLM**: MLP layers
/// - **Switch Transformer**: Expert networks
///
/// Often combined with gating mechanisms (SwiGLU).
pub fn silu(comptime T: type, input: Tensor(T), allocator: Allocator) TensorError!Tensor(T) {
    if (T != f32 and T != f64) return TensorError.InvalidShape;

    var result = try Tensor(T).init(allocator, input.shape);

    for (0..input.size) |i| {
        const x = input.data[i];
        const sigmoid_x = 1.0 / (1.0 + @exp(-x));
        result.data[i] = x * sigmoid_x;
    }

    return result;
}

/// Gated Linear Unit (GLU)
///
/// ## Mathematical Definition
/// For input split into two halves [a, b]:
/// ```
/// GLU([a, b]) = a ⊙ sigmoid(b)
/// ```
/// Where ⊙ denotes element-wise multiplication.
///
/// ## Educational Note: Gating Mechanism
/// GLU introduces a gating mechanism where:
/// - **Gate values (b)**: Control information flow via sigmoid
/// - **Content values (a)**: Carry the actual information
/// - **Product**: Combines gated information flow
///
/// ## Transformer Context
/// Gating mechanisms are crucial in transformers:
/// - **FFN Layers**: Control information flow between layers
/// - **Attention**: Gate values control what information to attend to
/// - **Memory**: Selective information retention and forgetting
///
/// ## Implementation Note
/// Input tensor must have even size in the last dimension to be split.
pub fn glu(comptime T: type, input: Tensor(T), allocator: Allocator) TensorError!Tensor(T) {
    if (T != f32 and T != f64) return TensorError.InvalidShape;
    if (input.ndim() < 1) return TensorError.IncompatibleShapes;

    const last_dim = input.shape[input.shape.len - 1];
    if (last_dim % 2 != 0) return TensorError.IncompatibleShapes;

    const output_last_dim = last_dim / 2;
    var output_shape = try allocator.dupe(usize, input.shape);
    defer allocator.free(output_shape);
    output_shape[output_shape.len - 1] = output_last_dim;

    var result = try Tensor(T).init(allocator, output_shape);

    // Split input into two halves and apply GLU
    const half_size = input.size / 2;
    for (0..half_size) |i| {
        const a = input.data[i];                    // First half (content)
        const b = input.data[i + half_size];        // Second half (gate)
        const sigmoid_b = 1.0 / (1.0 + @exp(-b));  // Sigmoid of gate
        result.data[i] = a * sigmoid_b;             // Gated output
    }

    return result;
}

/// GELU-based Gated Linear Unit (GeGLU)
///
/// ## Mathematical Definition
/// For input split into two halves [a, b]:
/// ```
/// GeGLU([a, b]) = a ⊙ GELU(b)
/// ```
///
/// ## Educational Note: Why GeGLU?
/// GeGLU combines the benefits of:
/// - **GELU**: Smooth, probabilistically-motivated activation
/// - **Gating**: Selective information flow control
/// - **Non-linearity**: Enhanced model expressivity
///
/// ## Transformer Usage
/// Used in advanced transformer variants where both smoothness
/// and gating are desired for optimal performance.
pub fn geglu(comptime T: type, input: Tensor(T), allocator: Allocator) TensorError!Tensor(T) {
    if (T != f32 and T != f64) return TensorError.InvalidShape;
    if (input.ndim() < 1) return TensorError.IncompatibleShapes;

    const last_dim = input.shape[input.shape.len - 1];
    if (last_dim % 2 != 0) return TensorError.IncompatibleShapes;

    const output_last_dim = last_dim / 2;
    var output_shape = try allocator.dupe(usize, input.shape);
    defer allocator.free(output_shape);
    output_shape[output_shape.len - 1] = output_last_dim;

    var result = try Tensor(T).init(allocator, output_shape);

    const half_size = input.size / 2;
    for (0..half_size) |i| {
        const a = input.data[i];               // First half (content)
        const b = input.data[i + half_size];   // Second half (gate)

        // Apply GELU to gate values
        const cubic_term = GELU_COEFF_A * b * b * b;
        const inner = GELU_SQRT_2_OVER_PI * (b + cubic_term);
        const gelu_b = 0.5 * b * (1.0 + std.math.tanh(inner));

        result.data[i] = a * gelu_b;
    }

    return result;
}

/// SiLU-based Gated Linear Unit (SwiGLU)
///
/// ## Mathematical Definition
/// For input split into two halves [a, b]:
/// ```
/// SwiGLU([a, b]) = a ⊙ SiLU(b)
/// ```
///
/// ## Educational Note: LLaMA Architecture
/// SwiGLU is a key component of LLaMA's architecture:
/// - **Feed-forward layers**: Use SwiGLU instead of standard activations
/// - **Performance**: Shown to improve model quality over ReLU-based FFNs
/// - **Efficiency**: Self-gating mechanism reduces parameter redundancy
///
/// ## Mathematical Intuition
/// The combination of:
/// - **Content stream (a)**: Raw information to be processed
/// - **Gate stream (b)**: Processed through SiLU to create smooth gates
/// - **Element-wise product**: Selective information flow
///
/// Creates a more expressive and stable activation than simple non-linearities.
pub fn swiglu(comptime T: type, input: Tensor(T), allocator: Allocator) TensorError!Tensor(T) {
    if (T != f32 and T != f64) return TensorError.InvalidShape;
    if (input.ndim() < 1) return TensorError.IncompatibleShapes;

    const last_dim = input.shape[input.shape.len - 1];
    if (last_dim % 2 != 0) return TensorError.IncompatibleShapes;

    const output_last_dim = last_dim / 2;
    var output_shape = try allocator.dupe(usize, input.shape);
    defer allocator.free(output_shape);
    output_shape[output_shape.len - 1] = output_last_dim;

    var result = try Tensor(T).init(allocator, output_shape);

    const half_size = input.size / 2;
    for (0..half_size) |i| {
        const a = input.data[i];               // First half (content)
        const b = input.data[i + half_size];   // Second half (gate)

        // Apply SiLU to gate values
        const sigmoid_b = 1.0 / (1.0 + @exp(-b));
        const silu_b = b * sigmoid_b;

        result.data[i] = a * silu_b;
    }

    return result;
}

/// Hyperbolic Tangent (Tanh)
///
/// ## Mathematical Definition
/// ```
/// tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)) = (e^(2x) - 1) / (e^(2x) + 1)
/// ```
///
/// ## Properties
/// - **Range**: (-1, 1)
/// - **Symmetric**: tanh(-x) = -tanh(x)
/// - **Saturating**: Gradients approach 0 for large |x|
/// - **Zero-centered**: Output centered around 0
///
/// ## Historical Context
/// Tanh was widely used in early neural networks but has largely been
/// replaced by ReLU and its variants due to vanishing gradient problems.
/// However, it still appears in specific contexts like LSTM gates.
pub fn tanh_activation(comptime T: type, input: Tensor(T), allocator: Allocator) TensorError!Tensor(T) {
    if (T != f32 and T != f64) return TensorError.InvalidShape;

    var result = try Tensor(T).init(allocator, input.shape);

    for (0..input.size) |i| {
        result.data[i] = std.math.tanh(input.data[i]);
    }

    return result;
}

/// Sigmoid Activation
///
/// ## Mathematical Definition
/// ```
/// sigmoid(x) = 1 / (1 + e^(-x))
/// ```
///
/// ## Properties
/// - **Range**: (0, 1)
/// - **Monotonic**: Always increasing
/// - **Saturating**: Gradients approach 0 for large |x|
/// - **Probability interpretation**: Can be interpreted as probability
///
/// ## Transformer Usage
/// While not commonly used as a primary activation in modern transformers,
/// sigmoid appears in:
/// - **Attention mechanisms**: Gating functions
/// - **GLU variants**: Gate activation
/// - **Output layers**: Binary classification
pub fn sigmoid(comptime T: type, input: Tensor(T), allocator: Allocator) TensorError!Tensor(T) {
    if (T != f32 and T != f64) return TensorError.InvalidShape;

    var result = try Tensor(T).init(allocator, input.shape);

    for (0..input.size) |i| {
        result.data[i] = 1.0 / (1.0 + @exp(-input.data[i]));
    }

    return result;
}

/// Generic activation function dispatcher
///
/// ## Educational Note: Design Pattern
/// This function demonstrates a common pattern in neural network frameworks:
/// - **Polymorphic activation**: Single interface for multiple activations
/// - **Type safety**: Compile-time type checking
/// - **Performance**: No runtime overhead from function pointers
/// - **Extensibility**: Easy to add new activation functions
pub fn applyActivation(comptime T: type, activation_type: ActivationType, input: Tensor(T), allocator: Allocator) TensorError!Tensor(T) {
    return switch (activation_type) {
        .ReLU => relu(T, input),
        .GELU => gelu(T, input, allocator),
        .SiLU => silu(T, input, allocator),
        .GLU => glu(T, input, allocator),
        .GeGLU => geglu(T, input, allocator),
        .SwiGLU => swiglu(T, input, allocator),
        .Tanh => tanh_activation(T, input, allocator),
        .Sigmoid => sigmoid(T, input, allocator),
    };
}

// ============================================================================
// COMPREHENSIVE TESTS
// ============================================================================

test "ReLU activation function" {
    const allocator = testing.allocator;

    var input = try Tensor(f32).init(allocator, &[_]usize{4});
    defer input.deinit();

    // Test data: mix of positive, negative, and zero
    try input.set(&[_]usize{0}, -2.0);
    try input.set(&[_]usize{1}, -0.5);
    try input.set(&[_]usize{2}, 0.0);
    try input.set(&[_]usize{3}, 1.5);

    var result = try relu(f32, input);
    defer result.deinit();

    // Verify ReLU properties
    try testing.expectEqual(@as(f32, 0.0), try result.get(&[_]usize{0})); // max(0, -2.0)
    try testing.expectEqual(@as(f32, 0.0), try result.get(&[_]usize{1})); // max(0, -0.5)
    try testing.expectEqual(@as(f32, 0.0), try result.get(&[_]usize{2})); // max(0, 0.0)
    try testing.expectEqual(@as(f32, 1.5), try result.get(&[_]usize{3})); // max(0, 1.5)
}

test "GELU activation function" {
    const allocator = testing.allocator;

    var input = try Tensor(f32).init(allocator, &[_]usize{3});
    defer input.deinit();

    // Test key points
    try input.set(&[_]usize{0}, 0.0);  // Should be exactly 0
    try input.set(&[_]usize{1}, 1.0);  // Should be close to 0.841
    try input.set(&[_]usize{2}, -1.0); // Should be close to -0.159

    var result = try gelu(f32, input, allocator);
    defer result.deinit();

    // GELU(0) = 0
    try testing.expectApproxEqAbs(@as(f32, 0.0), try result.get(&[_]usize{0}), 1e-6);

    // GELU(1) ≈ 0.841 (positive values close to input for large positive x)
    const gelu_1 = try result.get(&[_]usize{1});
    try testing.expect(gelu_1 > 0.8 and gelu_1 < 0.9);

    // GELU(-1) ≈ -0.159 (negative but not zero due to smooth transition)
    const gelu_neg1 = try result.get(&[_]usize{2});
    try testing.expect(gelu_neg1 < 0.0 and gelu_neg1 > -0.2);
}

test "SiLU activation function" {
    const allocator = testing.allocator;

    var input = try Tensor(f32).init(allocator, &[_]usize{4});
    defer input.deinit();

    try input.set(&[_]usize{0}, 0.0);
    try input.set(&[_]usize{1}, 1.0);
    try input.set(&[_]usize{2}, -1.0);
    try input.set(&[_]usize{3}, 5.0);

    var result = try silu(f32, input, allocator);
    defer result.deinit();

    // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    try testing.expectApproxEqAbs(@as(f32, 0.0), try result.get(&[_]usize{0}), 1e-6);

    // SiLU(1) = 1 * sigmoid(1) ≈ 1 * 0.731 ≈ 0.731
    const silu_1 = try result.get(&[_]usize{1});
    try testing.expect(silu_1 > 0.7 and silu_1 < 0.8);

    // SiLU approaches x for large positive x
    const silu_5 = try result.get(&[_]usize{3});
    try testing.expect(@abs(silu_5 - 5.0) < 0.1);
}

test "GLU gating mechanism" {
    const allocator = testing.allocator;

    // Input with 4 elements (will be split into 2 halves)
    var input = try Tensor(f32).init(allocator, &[_]usize{4});
    defer input.deinit();

    // First half: content values
    try input.set(&[_]usize{0}, 2.0);
    try input.set(&[_]usize{1}, 3.0);

    // Second half: gate values
    try input.set(&[_]usize{2}, 0.0);  // sigmoid(0) = 0.5
    try input.set(&[_]usize{3}, 5.0);  // sigmoid(5) ≈ 0.993

    var result = try glu(f32, input, allocator);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.size);

    // First output: 2.0 * sigmoid(0) = 2.0 * 0.5 = 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), try result.get(&[_]usize{0}), 1e-3);

    // Second output: 3.0 * sigmoid(5) ≈ 3.0 * 0.993 ≈ 2.98
    const second_output = try result.get(&[_]usize{1});
    try testing.expect(second_output > 2.9 and second_output < 3.0);
}

test "SwiGLU LLaMA activation" {
    const allocator = testing.allocator;

    var input = try Tensor(f32).init(allocator, &[_]usize{6});
    defer input.deinit();

    // Content values (first half)
    try input.set(&[_]usize{0}, 1.0);
    try input.set(&[_]usize{1}, 2.0);
    try input.set(&[_]usize{2}, 3.0);

    // Gate values (second half)
    try input.set(&[_]usize{3}, 1.0);
    try input.set(&[_]usize{4}, 0.0);
    try input.set(&[_]usize{5}, -1.0);

    var result = try swiglu(f32, input, allocator);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 3), result.size);

    // Verify gating behavior
    for (0..3) |i| {
        const content = input.data[i];
        const gate = input.data[i + 3];
        const expected_silu = gate * (1.0 / (1.0 + @exp(-gate)));
        const expected_output = content * expected_silu;

        try testing.expectApproxEqAbs(expected_output, try result.get(&[_]usize{i}), 1e-6);
    }
}

test "activation function dispatcher" {
    const allocator = testing.allocator;

    var input = try Tensor(f32).init(allocator, &[_]usize{2});
    defer input.deinit();

    try input.set(&[_]usize{0}, 1.0);
    try input.set(&[_]usize{1}, -0.5);

    // Test ReLU through dispatcher
    var relu_result = try applyActivation(f32, .ReLU, input, allocator);
    defer relu_result.deinit();

    try testing.expectEqual(@as(f32, 1.0), try relu_result.get(&[_]usize{0}));
    try testing.expectEqual(@as(f32, 0.0), try relu_result.get(&[_]usize{1}));

    // Test GELU through dispatcher
    var gelu_result = try applyActivation(f32, .GELU, input, allocator);
    defer gelu_result.deinit();

    // Should be different from ReLU result
    try testing.expect((try gelu_result.get(&[_]usize{1})) != 0.0);
}