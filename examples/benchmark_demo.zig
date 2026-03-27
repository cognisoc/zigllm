const std = @import("std");
const builtin = @import("builtin");

/// Simple row-major matrix backed by a flat f32 slice.
/// Used for benchmarking without depending on the library's Tensor type,
/// so this example can be run standalone with `zig run`.
const Matrix = struct {
    data: []f32,
    rows: usize,
    cols: usize,
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !Matrix {
        const data = try allocator.alloc(f32, rows * cols);
        return .{ .data = data, .rows = rows, .cols = cols, .allocator = allocator };
    }

    fn deinit(self: *Matrix) void {
        self.allocator.free(self.data);
    }

    fn fill(self: *Matrix, comptime gen: fn (usize, usize) f32) void {
        for (0..self.rows) |i| {
            for (0..self.cols) |j| {
                self.data[i * self.cols + j] = gen(i, j);
            }
        }
    }
};

/// Naive O(n^3) matmul — baseline for comparison.
fn matmulNaive(a: Matrix, b: Matrix, out: *Matrix) void {
    for (0..a.rows) |i| {
        for (0..b.cols) |j| {
            var sum: f32 = 0.0;
            for (0..a.cols) |k| {
                sum += a.data[i * a.cols + k] * b.data[k * b.cols + j];
            }
            out.data[i * out.cols + j] = sum;
        }
    }
}

/// SIMD-accelerated matmul using direct data access and @Vector.
fn matmulSIMD(a: Matrix, b: Matrix, out: *Matrix) void {
    const simd_width: comptime_int = switch (builtin.cpu.arch) {
        .aarch64 => 4,
        .x86_64 => 4,
        else => 1,
    };
    const VecF32 = @Vector(simd_width, f32);

    const m = a.rows;
    const k = a.cols;
    const n = b.cols;

    for (0..m) |i| {
        const a_row = i * k;
        const r_row = i * n;

        var j: usize = 0;
        while (j + simd_width <= n) : (j += simd_width) {
            var acc: VecF32 = @splat(0.0);
            for (0..k) |l| {
                const a_val: VecF32 = @splat(a.data[a_row + l]);
                const b_vec: VecF32 = b.data[l * n + j ..][0..simd_width].*;
                acc += a_val * b_vec;
            }
            out.data[r_row + j ..][0..simd_width].* = @as([simd_width]f32, acc);
        }
        // Scalar remainder
        while (j < n) : (j += 1) {
            var sum: f32 = 0.0;
            for (0..k) |l| {
                sum += a.data[a_row + l] * b.data[l * n + j];
            }
            out.data[r_row + j] = sum;
        }
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const print = std.debug.print;
    print("ZigLlama Performance Benchmarks\n", .{});
    print("===============================\n\n", .{});
    print("All measurements are real, timed with std.time.Timer.\n", .{});
    print("Arch: {s}, SIMD width: {d} f32s\n\n", .{
        @tagName(builtin.cpu.arch),
        @as(u32, switch (builtin.cpu.arch) {
            .aarch64 => 4,
            .x86_64 => 4,
            else => 1,
        }),
    });

    const sizes = [_]usize{ 64, 128, 256 };
    const warmup_iters: usize = 2;
    const bench_iters: usize = 5;

    print("Matrix Multiplication: Naive vs SIMD\n", .{});
    print("-------------------------------------\n", .{});

    for (sizes) |size| {
        var a = try Matrix.init(allocator, size, size);
        defer a.deinit();
        var b = try Matrix.init(allocator, size, size);
        defer b.deinit();
        var out = try Matrix.init(allocator, size, size);
        defer out.deinit();

        const gen_a = struct {
            fn f(i: usize, j: usize) f32 {
                return @as(f32, @floatFromInt((i + j) % 17)) * 0.1;
            }
        }.f;
        const gen_b = struct {
            fn f(i: usize, j: usize) f32 {
                return @as(f32, @floatFromInt((i * 3 + j) % 13)) * 0.1;
            }
        }.f;
        a.fill(gen_a);
        b.fill(gen_b);

        // Warmup naive
        for (0..warmup_iters) |_| matmulNaive(a, b, &out);

        // Benchmark naive
        var timer = try std.time.Timer.start();
        for (0..bench_iters) |_| matmulNaive(a, b, &out);
        const naive_ns = timer.read();
        const naive_avg_us = @as(f64, @floatFromInt(naive_ns)) / @as(f64, @floatFromInt(bench_iters)) / 1000.0;

        // Warmup SIMD
        for (0..warmup_iters) |_| matmulSIMD(a, b, &out);

        // Benchmark SIMD
        timer = try std.time.Timer.start();
        for (0..bench_iters) |_| matmulSIMD(a, b, &out);
        const simd_ns = timer.read();
        const simd_avg_us = @as(f64, @floatFromInt(simd_ns)) / @as(f64, @floatFromInt(bench_iters)) / 1000.0;

        const ops: u64 = 2 * size * size * size;
        const naive_gflops = @as(f64, @floatFromInt(ops)) / (naive_avg_us / 1e6) / 1e9;
        const simd_gflops = @as(f64, @floatFromInt(ops)) / (simd_avg_us / 1e6) / 1e9;
        const speedup = naive_avg_us / simd_avg_us;

        print("{d}x{d} matmul (avg of {d} runs):\n", .{ size, size, bench_iters });
        print("  Naive: {d:10.1} us  ({d:5.2} GFLOPS)\n", .{ naive_avg_us, naive_gflops });
        print("  SIMD:  {d:10.1} us  ({d:5.2} GFLOPS)\n", .{ simd_avg_us, simd_gflops });
        print("  Speedup: {d:.2}x\n\n", .{speedup});
    }

    // Quantization memory comparison (analytical)
    print("Quantization Memory Usage (analytical):\n", .{});
    print("---------------------------------------\n", .{});
    const model_params: u64 = 7_000_000_000;
    print("7B parameter model:\n", .{});
    print("  FP32:  {d:.1} GB\n", .{@as(f64, @floatFromInt(model_params * 4)) / 1e9});
    print("  FP16:  {d:.1} GB\n", .{@as(f64, @floatFromInt(model_params * 2)) / 1e9});
    print("  Q8_0:  {d:.1} GB (block size 32: 32 int8 + 1 f16 scale)\n", .{@as(f64, @floatFromInt(model_params)) * 34.0 / 32.0 / 1e9});
    print("  Q4_0:  {d:.1} GB (block size 32: 16 bytes + 1 f16 scale)\n", .{@as(f64, @floatFromInt(model_params)) * 18.0 / 32.0 / 1e9});
    print("\n", .{});

    print("Note: End-to-end inference benchmarks require a loaded GGUF model\n", .{});
    print("      (not yet implemented - see Phase 2-3 roadmap).\n", .{});
}
