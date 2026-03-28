const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    const list = std.ArrayList(u32).init(allocator);
    defer list.deinit();
    std.debug.print("OK\n", .{});
}