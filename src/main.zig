// ZigLlama - Educational LLaMA Implementation in Zig
//
// This is the main entry point for the ZigLlama library, providing educational
// implementations of transformer models following our progressive architecture.
//
// ## Architecture Layers (Bottom to Top)
// 1. Foundation: Tensors, memory management, basic operations
// 2. Linear Algebra: Matrix operations, SIMD, quantization
// 3. Neural Primitives: Activations, normalizations, embeddings
// 4. Transformers: Attention mechanisms, feed-forward networks
// 5. Models: Complete LLaMA architecture and model loading
// 6. Inference: Generation, sampling, and optimizations

const std = @import("std");

// Progressive architecture modules
pub const foundation = struct {
    pub const tensor = @import("foundation/tensor.zig");
};

// Linear algebra layer with SIMD optimizations and quantization
pub const linear_algebra = struct {
    pub const matrix_ops = @import("linear_algebra/matrix_ops.zig");
    pub const quantization = @import("linear_algebra/quantization.zig");
};

pub const neural_primitives = struct {
    pub const activations = @import("neural_primitives/activations.zig");
    pub const normalization = @import("neural_primitives/normalization.zig");
    pub const embeddings = @import("neural_primitives/embeddings.zig");
};

pub const transformers = struct {
    pub const attention = @import("transformers/attention.zig");
    pub const feed_forward = @import("transformers/feed_forward.zig");
    pub const transformer_block = @import("transformers/transformer_block.zig");
};

pub const models = struct {
    pub const llama = @import("models/llama.zig");
    pub const config = @import("models/config.zig");
    pub const tokenizer = @import("models/tokenizer.zig");
    pub const gguf = @import("models/gguf.zig");
};

pub const inference = struct {
    pub const generation = @import("inference/generation.zig");
    pub const kv_cache = @import("inference/kv_cache.zig");
    pub const streaming = @import("inference/streaming.zig");
    pub const batching = @import("inference/batching.zig");
    pub const profiling = @import("inference/profiling.zig");
};

/// ZigLlama version following semantic versioning
pub const version = std.SemanticVersion{
    .major = 0,
    .minor = 1,
    .patch = 0,
};

/// Main configuration for the ZigLlama library
pub const Config = struct {
    /// Memory allocator for all operations
    allocator: std.mem.Allocator,

    /// Maximum sequence length supported
    max_seq_len: u32 = 2048,

    /// Default floating point precision
    float_type: type = f32,

    /// Enable verbose educational logging
    verbose_logging: bool = false,
};

/// Educational demonstration of tensor operations
pub fn demonstrateTensorOperations(allocator: std.mem.Allocator, writer: anytype) !void {
    const Tensor = foundation.tensor.Tensor;

    try writer.print("=== ZigLlama Foundation Layer Demo ===\n", .{});
    try writer.print("Demonstrating basic tensor operations...\n\n", .{});

    // Create example tensors
    var matrix_a = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer matrix_a.deinit();
    var matrix_b = try Tensor(f32).init(allocator, &[_]usize{ 3, 2 });
    defer matrix_b.deinit();

    // Fill with example data
    try matrix_a.set(&[_]usize{ 0, 0 }, 1.0);
    try matrix_a.set(&[_]usize{ 0, 1 }, 2.0);
    try matrix_a.set(&[_]usize{ 0, 2 }, 3.0);
    try matrix_a.set(&[_]usize{ 1, 0 }, 4.0);
    try matrix_a.set(&[_]usize{ 1, 1 }, 5.0);
    try matrix_a.set(&[_]usize{ 1, 2 }, 6.0);

    try matrix_b.set(&[_]usize{ 0, 0 }, 1.0);
    try matrix_b.set(&[_]usize{ 0, 1 }, 0.0);
    try matrix_b.set(&[_]usize{ 1, 0 }, 0.0);
    try matrix_b.set(&[_]usize{ 1, 1 }, 1.0);
    try matrix_b.set(&[_]usize{ 2, 0 }, 0.5);
    try matrix_b.set(&[_]usize{ 2, 1 }, 0.5);

    try writer.print("Matrix A (2×3):\n", .{});
    try matrix_a.print(writer);
    try writer.print("\nMatrix B (3×2):\n", .{});
    try matrix_b.print(writer);

    // Perform matrix multiplication
    var result = try matrix_a.matmul(matrix_b, allocator);
    defer result.deinit();

    try writer.print("\nResult A × B (2×2):\n", .{});
    try result.print(writer);

    try writer.print("\n💡 This matrix multiplication is fundamental to transformers!\n", .{});
    try writer.print("   - Query/Key/Value projections use similar operations\n", .{});
    try writer.print("   - Feed-forward layers are sequences of matrix multiplications\n", .{});
}

/// Main entry point for the executable
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const stderr = std.io.getStdErr();
    const stdout = stderr.writer();

    try stdout.print("ZigLlama v{} - Educational LLaMA Implementation\n", .{version});
    try stdout.print("=============================================\n\n", .{});

    try stdout.print("🎓 Educational transformer implementation\n", .{});
    try stdout.print("⚡ Production-ready performance\n", .{});
    try stdout.print("📚 Progressive learning architecture\n\n", .{});

    // Demonstrate foundation layer
    try demonstrateTensorOperations(allocator, stdout);

    try stdout.print("\n📖 Next Steps:\n", .{});
    try stdout.print("   - Run 'zig build test' for comprehensive tests\n", .{});
    try stdout.print("   - Run 'zig build run-examples' for learning examples\n", .{});
    try stdout.print("   - Check docs/ for educational materials\n", .{});
}

// Basic functionality test
test "library structure and version" {
    const testing = std.testing;
    try testing.expect(version.major == 0);
    try testing.expect(version.minor == 1);
}

// Import all test modules for 'zig build test'
test {
    std.testing.refAllDecls(@This());
    _ = @import("foundation/tensor.zig");
    _ = @import("linear_algebra/matrix_ops.zig");
    _ = @import("linear_algebra/quantization.zig");
    _ = @import("neural_primitives/activations.zig");
    _ = @import("neural_primitives/normalization.zig");
    _ = @import("neural_primitives/embeddings.zig");
    _ = @import("transformers/attention.zig");
    _ = @import("transformers/feed_forward.zig");
    _ = @import("transformers/transformer_block.zig");
    _ = @import("models/llama.zig");
    _ = @import("models/config.zig");
    _ = @import("models/tokenizer.zig");
    _ = @import("models/gguf.zig");
    _ = @import("inference/generation.zig");
    _ = @import("inference/kv_cache.zig");
    _ = @import("inference/streaming.zig");
    _ = @import("inference/batching.zig");
    _ = @import("inference/profiling.zig");
}