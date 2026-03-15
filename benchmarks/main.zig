// ZigLlama Performance Benchmarks
//
// Educational performance analysis to understand computational characteristics
// of transformer operations. These benchmarks serve dual purposes:
// 1. Measure performance to guide optimizations
// 2. Teach about computational complexity and scaling behavior

const std = @import("std");
const zigllama = @import("zigllama");
const Tensor = zigllama.foundation.tensor.Tensor;

const BenchResult = struct {
    operation: []const u8,
    size: usize,
    time_us: i64,
    throughput_gflops: f64,
    memory_gb_per_sec: f64,
};

fn printBenchHeader() void {
    const print = std.debug.print;
    print("ZigLlama Performance Benchmarks\n");
    print("==============================\n\n");
    print("Educational performance analysis of tensor operations\n");
    print("Understanding computational characteristics of transformers\n\n");
}

fn printBenchResult(result: BenchResult) void {
    const print = std.debug.print;
    print("{s:20} | Size: {d:4} | Time: {d:8.2}μs | {d:6.2} GFLOPS | {d:6.2} GB/s\n", .{
        result.operation,
        result.size,
        @as(f64, @floatFromInt(result.time_us)),
        result.throughput_gflops,
        result.memory_gb_per_sec,
    });
}

fn benchmarkMatrixMultiplication(allocator: std.mem.Allocator) !void {
    const print = std.debug.print;
    print("Matrix Multiplication Benchmarks\n");
    print("---------------------------------\n");
    print("Operation            | Size      | Time       | GFLOPS | Memory \n");
    print("---------------------|-----------|------------|--------|--------\n");

    // Test different matrix sizes to understand scaling behavior
    const sizes = [_]usize{ 32, 64, 128, 256, 512 };

    for (sizes) |size| {
        // Create matrices filled with ones for predictable results
        var a = try Tensor(f32).init(allocator, &[_]usize{ size, size });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ size, size });
        defer b.deinit();

        a.fill(1.0);
        b.fill(1.0);

        // Warm up (exclude from timing)
        var warmup = try a.matmul(b, allocator);
        warmup.deinit();

        // Benchmark multiple iterations for stable timing
        const iterations = 3;
        var total_time: i64 = 0;

        for (0..iterations) |_| {
            const start = std.time.microTimestamp();
            var result = try a.matmul(b, allocator);
            const end = std.time.microTimestamp();
            result.deinit();

            total_time += (end - start);
        }

        const avg_time = @divFloor(total_time, iterations);

        // Calculate performance metrics
        const ops = 2 * size * size * size; // Multiply-accumulate operations
        const gflops = @as(f64, @floatFromInt(ops)) / (@as(f64, @floatFromInt(avg_time)) / 1e6) / 1e9;

        // Memory bandwidth (rough estimate)
        const memory_ops = 3 * size * size * @sizeOf(f32); // Read A, B, write C
        const memory_gb_per_sec = @as(f64, @floatFromInt(memory_ops)) / (@as(f64, @floatFromInt(avg_time)) / 1e6) / 1e9;

        const result = BenchResult{
            .operation = "MatMul (NxN)",
            .size = size,
            .time_us = avg_time,
            .throughput_gflops = gflops,
            .memory_gb_per_sec = memory_gb_per_sec,
        };

        printBenchResult(result);
    }

    print("\n💡 Educational Notes:\n");
    print("   - Matrix multiplication is O(n³) in operations\n");
    print("   - Memory usage is O(n²) for storing matrices\n");
    print("   - GFLOPS should scale roughly as n³/time\n");
    print("   - Memory bandwidth often limits performance for large matrices\n\n");
}

fn benchmarkElementWiseOperations(allocator: std.mem.Allocator) !void {
    const print = std.debug.print;
    print("Element-wise Operation Benchmarks\n");
    print("----------------------------------\n");
    print("Operation            | Size      | Time       | GFLOPS | Memory \n");
    print("---------------------|-----------|------------|--------|--------\n");

    const sizes = [_]usize{ 1024, 4096, 16384, 65536, 262144 };

    for (sizes) |size| {
        // Create vectors for element-wise addition
        var a = try Tensor(f32).init(allocator, &[_]usize{size});
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{size});
        defer b.deinit();

        a.fill(1.0);
        b.fill(2.0);

        // Benchmark vector addition
        const iterations = 10;
        var total_time: i64 = 0;

        for (0..iterations) |_| {
            const start = std.time.microTimestamp();
            var result = try a.add(b, allocator);
            const end = std.time.microTimestamp();
            result.deinit();

            total_time += (end - start);
        }

        const avg_time = @divFloor(total_time, iterations);

        // Calculate metrics
        const ops = size; // One addition per element
        const gflops = @as(f64, @floatFromInt(ops)) / (@as(f64, @floatFromInt(avg_time)) / 1e6) / 1e9;

        const memory_ops = 3 * size * @sizeOf(f32); // Read A, B, write result
        const memory_gb_per_sec = @as(f64, @floatFromInt(memory_ops)) / (@as(f64, @floatFromInt(avg_time)) / 1e6) / 1e9;

        const result = BenchResult{
            .operation = "Vector Addition",
            .size = size,
            .time_us = avg_time,
            .throughput_gflops = gflops,
            .memory_gb_per_sec = memory_gb_per_sec,
        };

        printBenchResult(result);
    }

    print("\n💡 Educational Notes:\n");
    print("   - Element-wise operations are O(n) in both time and memory\n");
    print("   - These operations are memory bandwidth limited\n");
    print("   - Perfect parallelization candidates (SIMD, GPU)\n");
    print("   - Used extensively in transformers for residual connections\n\n");
}

fn benchmarkTransformerOperations(allocator: std.mem.Allocator) !void {
    const print = std.debug.print;
    print("Transformer-Specific Benchmarks\n");
    print("--------------------------------\n");
    print("Operation            | Size      | Time       | GFLOPS | Memory \n");
    print("---------------------|-----------|------------|--------|--------\n");

    // Simulate typical transformer dimensions
    const configs = [_]struct { seq_len: usize, d_model: usize, name: []const u8 }{
        .{ .seq_len = 64, .d_model = 512, .name = "Small Model" },
        .{ .seq_len = 128, .d_model = 768, .name = "Base Model" },
        .{ .seq_len = 256, .d_model = 1024, .name = "Large Model" },
    };

    for (configs) |config| {
        // Simulate Q, K, V projection: [seq_len, d_model] @ [d_model, d_model]
        var input = try Tensor(f32).init(allocator, &[_]usize{ config.seq_len, config.d_model });
        defer input.deinit();
        var weights = try Tensor(f32).init(allocator, &[_]usize{ config.d_model, config.d_model });
        defer weights.deinit();

        input.fill(1.0);
        weights.fill(0.1);

        const start = std.time.microTimestamp();
        var query = try input.matmul(weights, allocator);
        const end = std.time.microTimestamp();
        query.deinit();

        const time_us = end - start;
        const ops = 2 * config.seq_len * config.d_model * config.d_model;
        const gflops = @as(f64, @floatFromInt(ops)) / (@as(f64, @floatFromInt(time_us)) / 1e6) / 1e9;

        const memory_ops = (config.seq_len * config.d_model + config.d_model * config.d_model + config.seq_len * config.d_model) * @sizeOf(f32);
        const memory_gb_per_sec = @as(f64, @floatFromInt(memory_ops)) / (@as(f64, @floatFromInt(time_us)) / 1e6) / 1e9;

        const result = BenchResult{
            .operation = config.name,
            .size = config.seq_len,
            .time_us = time_us,
            .throughput_gflops = gflops,
            .memory_gb_per_sec = memory_gb_per_sec,
        };

        printBenchResult(result);
    }

    print("\n💡 Educational Notes:\n");
    print("   - Q/K/V projections are the same operation repeated 3 times\n");
    print("   - Sequence length affects memory but not operation count per token\n");
    print("   - d_model dimension has quadratic impact on compute requirements\n");
    print("   - These projections can be batched for efficiency\n\n");
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    printBenchHeader();

    try benchmarkMatrixMultiplication(allocator);
    try benchmarkElementWiseOperations(allocator);
    try benchmarkTransformerOperations(allocator);

    const print = std.debug.print;
    print("🎓 Key Takeaways:\n");
    print("==================\n");
    print("• Matrix multiplication dominates transformer compute requirements\n");
    print("• Memory bandwidth limits performance for element-wise operations\n");
    print("• Scaling behavior helps predict performance with larger models\n");
    print("• Understanding these patterns guides optimization efforts\n\n");

    print("🔍 Next Steps:\n");
    print("===============\n");
    print("• Implement SIMD optimizations for better throughput\n");
    print("• Add quantization to reduce memory bandwidth requirements\n");
    print("• Profile real transformer workloads for optimization opportunities\n");
    print("• Compare with llama.cpp benchmarks for validation\n\n");
}