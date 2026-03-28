const std = @import("std"); pub fn main() !void { const stdout = std.io.getStdOut(); const writer = stdout.writer(); try writer.print("OK
", .{}); }
