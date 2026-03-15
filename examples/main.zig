// ZigLlama Educational Examples
//
// This file contains progressive examples for learning about transformers
// and neural network concepts, starting from basic tensor operations
// and building up to full transformer components.

const std = @import("std");
const zigllama = @import("zigllama");

const Tensor = zigllama.tensor.Tensor;

/// Example 1: Basic tensor operations
///
/// Learn about:
/// - Creating tensors with different shapes
/// - Basic indexing and data manipulation
/// - Understanding tensor memory layout
fn example1_basic_tensors(allocator: std.mem.Allocator) !void {
    const print = std.debug.print;

    print("\n=== Example 1: Basic Tensor Operations ===\n", .{});
    print("Understanding the foundation of neural networks\n\n", .{});

    // Create different tensor shapes
    print("1. Creating tensors of different shapes:\n", .{});

    // Vector (1D tensor)
    var vector = try Tensor(f32).init(allocator, &[_]usize{4});
    defer vector.deinit();

    // Fill vector with values [1, 2, 3, 4]
    for (0..4) |i| {
        vector.set(&[_]usize{i}, @floatFromInt(i + 1));
    }

    print("Vector (1D tensor): ");
    vector.print();

    // Matrix (2D tensor)
    var matrix = try Tensor(f32).init(allocator, &[_]usize{3, 2});
    defer matrix.deinit();

    // Fill matrix with sequential values
    var value: f32 = 1;
    for (0..3) |i| {
        for (0..2) |j| {
            matrix.set(&[_]usize{i, j}, value);
            value += 1;
        }
    }

    print("\nMatrix (2D tensor): ");
    matrix.print();

    // 3D tensor (like a batch of matrices)
    var tensor3d = try Tensor(f32).init(allocator, &[_]usize{2, 2, 2});
    defer tensor3d.deinit();
    try tensor3d.fill(0.5);

    print("\n3D Tensor (filled with 0.5): ");
    tensor3d.print();

    print("\n💡 Key Learning: Tensors are just multi-dimensional arrays!\n");
    print("   - 1D: vectors (like word embeddings)\n");
    print("   - 2D: matrices (like attention weights)\n");
    print("   - 3D+: batched operations or more complex structures\n");
}

/// Example 2: Matrix multiplication - the heart of neural networks
///
/// Learn about:
/// - Why matrix multiplication is central to neural networks
/// - How transformers use matrix multiplication for projections
/// - Computational patterns in deep learning
fn example2_matrix_operations(allocator: std.mem.Allocator) !void {
    const print = std.debug.print;

    print("\n=== Example 2: Matrix Multiplication ===\n");
    print("The fundamental operation in neural networks\n\n");

    // Simulate a simple linear transformation: y = Wx + b
    // This is what happens in every neural network layer!

    print("Simulating a neural network linear layer: y = Wx + b\n");
    print("Where W is the weight matrix, x is input, b is bias\n\n");

    // Input: batch of 2 samples, each with 3 features
    var input = try Tensor(f32).init(allocator, &[_]usize{2, 3});
    defer input.deinit();

    // Sample 1: [1.0, 0.5, -0.2]
    input.set(&[_]usize{0, 0}, 1.0);
    input.set(&[_]usize{0, 1}, 0.5);
    input.set(&[_]usize{0, 2}, -0.2);

    // Sample 2: [0.8, -1.0, 0.3]
    input.set(&[_]usize{1, 0}, 0.8);
    input.set(&[_]usize{1, 1}, -1.0);
    input.set(&[_]usize{1, 2}, 0.3);

    print("Input (2 samples × 3 features):");
    input.print();

    // Weight matrix: 3 input features → 4 output features
    var weights = try Tensor(f32).init(allocator, &[_]usize{3, 4});
    defer weights.deinit();

    // Initialize with small random-like values
    const weight_values = [_]f32{
        0.1,  0.3, -0.2,  0.4,   // First input feature connections
        0.2, -0.1,  0.5, -0.3,   // Second input feature connections
       -0.4,  0.2,  0.1,  0.6,   // Third input feature connections
    };

    for (weight_values, 0..) |val, i| {
        const row = i / 4;
        const col = i % 4;
        weights.set(&[_]usize{row, col}, val);
    }

    print("\nWeight matrix (3 features → 4 features):");
    weights.print();

    // Compute matrix multiplication: input @ weights
    var output = try input.matmul(weights, allocator);
    defer output.deinit();

    print("\nOutput after linear transformation:");
    output.print();

    print("\n💡 Key Learning: This is exactly what happens in transformer layers!\n");
    print("   - Query projection: X @ W_q\n");
    print("   - Key projection: X @ W_k  \n");
    print("   - Value projection: X @ W_v\n");
    print("   - Feed-forward layers: multiple W matrices in sequence\n");
}

/// Example 3: Attention mechanism simulation
///
/// Learn about:
/// - How attention works at a high level
/// - Query-Key-Value concept
/// - Similarity computation and softmax
fn example3_attention_concepts(allocator: std.mem.Allocator) !void {
    const print = std.debug.print;

    print("\n=== Example 3: Attention Mechanism Concepts ===\n");
    print("Understanding the core innovation of transformers\n\n");

    print("Simulating attention for the sentence: 'The cat sat'\n");
    print("We'll see how each word 'pays attention' to others\n\n");

    // Simplified: 3 words, each represented by a 4-dimensional vector
    var tokens = try Tensor(f32).init(allocator, &[_]usize{3, 4});
    defer tokens.deinit();

    // "The" - determiner (focusing on definiteness)
    tokens.set(&[_]usize{0, 0}, 1.0); // determiner signal
    tokens.set(&[_]usize{0, 1}, 0.0); // noun signal
    tokens.set(&[_]usize{0, 2}, 0.0); // verb signal
    tokens.set(&[_]usize{0, 3}, 0.2); // position signal

    // "cat" - noun (the main subject)
    tokens.set(&[_]usize{1, 0}, 0.1); // determiner signal
    tokens.set(&[_]usize{1, 1}, 1.0); // noun signal
    tokens.set(&[_]usize{1, 2}, 0.0); // verb signal
    tokens.set(&[_]usize{1, 3}, 0.5); // position signal

    // "sat" - verb (the action)
    tokens.set(&[_]usize{2, 0}, 0.0); // determiner signal
    tokens.set(&[_]usize{2, 1}, 0.3); // noun signal (verbs often relate to nouns)
    tokens.set(&[_]usize{2, 2}, 1.0); // verb signal
    tokens.set(&[_]usize{2, 3}, 0.8); // position signal

    print("Token representations (simplified):");
    tokens.print();

    print("\nIn real attention:\n");
    print("1. Each token generates Query (what it's looking for)\n");
    print("2. Each token generates Key (what it represents)\n");
    print("3. Each token generates Value (what information it provides)\n");
    print("4. Similarity = Query · Key (dot product)\n");
    print("5. Attention weights = softmax(similarities)\n");
    print("6. Output = weighted sum of Values\n\n");

    print("💡 Key Insight: Attention lets each position decide which\n");
    print("   other positions are most relevant for its computation!\n");
    print("   - 'The' might pay attention to 'cat' (what it's determining)\n");
    print("   - 'sat' might pay attention to 'cat' (who is sitting)\n");
}

/// Example 4: Feed-forward network simulation
///
/// Learn about:
/// - Position-wise processing in transformers
/// - Non-linear transformations
/// - How feed-forward layers complement attention
fn example4_feedforward_concepts(allocator: std.mem.Allocator) !void {
    const print = std.debug.print;

    print("\n=== Example 4: Feed-Forward Network Concepts ===\n");
    print("Position-wise processing after attention\n\n");

    print("After attention mixes information between positions,\n");
    print("feed-forward networks process each position independently.\n\n");

    // Simulate hidden state after attention (3 positions, 4 dimensions each)
    var hidden = try Tensor(f32).init(allocator, &[_]usize{3, 4});
    defer hidden.deinit();

    // Fill with example post-attention values
    const hidden_values = [_]f32{
        0.8, -0.3,  0.5,  0.1,  // Position 0 after attention
        0.2,  0.7, -0.4,  0.9,  // Position 1 after attention
       -0.1,  0.4,  0.6, -0.2,  // Position 2 after attention
    };

    for (hidden_values, 0..) |val, i| {
        const row = i / 4;
        const col = i % 4;
        hidden.set(&[_]usize{row, col}, val);
    }

    print("Hidden states after attention:");
    hidden.print();

    // Simulate first feed-forward transformation: hidden_dim → ff_dim
    var ff_weights1 = try Tensor(f32).init(allocator, &[_]usize{4, 6});
    defer ff_weights1.deinit();
    try ff_weights1.fill(0.1); // Simplified weights

    var ff_output1 = try hidden.matmul(ff_weights1, allocator);
    defer ff_output1.deinit();

    print("\nAfter first feed-forward transformation (4→6 dims):");
    ff_output1.print();

    // Apply activation function (ReLU for simplicity)
    var activated = try ff_output1.relu(allocator);
    defer activated.deinit();

    print("\nAfter ReLU activation:");
    activated.print();

    print("\n💡 Key Learning: Feed-forward networks serve several purposes:\n");
    print("   - Increase model capacity (temporary expansion to larger dim)\n");
    print("   - Add non-linearity (ReLU, SwiGLU, etc.)\n");
    print("   - Process each position independently\n");
    print("   - Complement attention's position-mixing behavior\n");
}

/// Main examples runner
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const print = std.debug.print;

    print("🎓 ZigLlama Educational Examples\n", .{});
    print("===============================\n", .{});
    print("Learn about transformers by building them step by step!\n", .{});

    // Run all examples
    try example1_basic_tensors(allocator);
    try example2_matrix_operations(allocator);
    try example3_attention_concepts(allocator);
    try example4_feedforward_concepts(allocator);

    print("\n🎉 Congratulations! You've learned the core concepts of transformers:\n");
    print("   ✅ Tensors and their operations\n");
    print("   ✅ Matrix multiplication in neural networks\n");
    print("   ✅ Attention mechanism intuition\n");
    print("   ✅ Feed-forward network purpose\n\n");
    print("Next steps: Run 'zig build docs' to explore the full architecture!\n");
}