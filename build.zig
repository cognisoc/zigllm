const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main test step: run all tests via src/main.zig
    const test_step = b.step("test", "Run all tests");

    const main_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_main_tests = b.addRunArtifact(main_tests);
    test_step.dependOn(&run_main_tests.step);

    // Foundation layer tests
    const test_foundation_step = b.step("test-foundation", "Test foundation layer");

    const foundation_tests = b.addTest(.{
        .root_source_file = b.path("src/foundation/tensor.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_foundation_tests = b.addRunArtifact(foundation_tests);
    test_foundation_step.dependOn(&run_foundation_tests.step);

    // Linear algebra layer tests
    const test_linear_algebra_step = b.step("test-linear-algebra", "Test linear algebra layer");

    const linear_algebra_tests = b.addTest(.{
        .root_source_file = b.path("src/linear_algebra/matrix_ops.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_linear_algebra_tests = b.addRunArtifact(linear_algebra_tests);
    test_linear_algebra_step.dependOn(&run_linear_algebra_tests.step);

    // Benchmark step
    _ = b.step("bench", "Run performance benchmarks (use: zig run examples/benchmark_demo.zig)");
}
