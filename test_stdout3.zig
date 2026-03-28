const std = @import("std");

pub fn main() !void {
    const stdout = std.fs.File.stdout();
    const writer = stdout.writer();
    try writer.print("OK\n", .{});
}