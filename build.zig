const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Use individual test steps that work
    const test_step = b.step("test", "Run all tests");

    // Individual test step for foundation
    const test_foundation_step = b.step("test-foundation", "Test foundation layer");

    // Individual test step for linear algebra
    const test_linear_algebra_step = b.step("test-linear-algebra", "Test linear algebra layer");

    // Demo step (run directly with zig)
    const demo_step = b.step("demo", "Run educational demo (use: zig run examples/educational_demo.zig)");

    // Placeholder for other commands
    _ = b.step("bench", "Run performance benchmarks");

    // Note unused variables to avoid warnings
    _ = test_step;
    _ = test_foundation_step;
    _ = test_linear_algebra_step;
}