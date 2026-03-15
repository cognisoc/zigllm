const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main executable demonstrating library usage
    const exe = b.addExecutable(.{
        .name = "zigllama",
        .target = target,
        .optimize = optimize,
    });
    exe.addModule("zigllama", b.addModule("zigllama", .{}));
    exe.linkLibC();
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    const run_step = b.step("run", "Run the main ZigLlama demo");
    run_step.dependOn(&run_cmd.step);

    // Test step for running all tests
    const test_step = b.step("test", "Run all tests");

    // Individual test commands
    const test_foundation_step = b.step("test-foundation", "Test foundation layer");

    // Benchmark step
    const bench_step = b.step("bench", "Run performance benchmarks");

    // Check step for syntax verification
    const check_step = b.step("check", "Check syntax without running");
    check_step.dependOn(&exe.step);
}