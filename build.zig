const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create a simple test step that runs the main file tests
    const test_step = b.addTest(.{
        .name = "test",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Add include paths for all source directories
    test_step.addIncludePath(.{ .path = "src" });
    test_step.addIncludePath(.{ .path = "src/foundation" });
    test_step.addIncludePath(.{ .path = "src/linear_algebra" });
    test_step.addIncludePath(.{ .path = "src/neural_primitives" });
    test_step.addIncludePath(.{ .path = "src/transformers" });
    test_step.addIncludePath(.{ .path = "src/models" });
    test_step.addIncludePath(.{ .path = "src/inference" });

    // Set default step to run tests
    const default_step = b.step("default", "Build and test the project");
    default_step.dependOn(&test_step.step);
}