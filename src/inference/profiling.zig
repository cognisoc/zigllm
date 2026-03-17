//! Performance Profiling and Benchmarking Tools
//!
//! This module provides comprehensive performance profiling capabilities for
//! language model inference, including timing, memory usage, throughput
//! measurement, and detailed performance analysis tools.
//!
//! ## Educational Value
//! Performance profiling teaches optimization fundamentals:
//! - How to identify bottlenecks in neural network inference
//! - Memory allocation patterns and optimization opportunities
//! - Throughput vs latency trade-offs in different scenarios
//! - Production monitoring and performance regression detection

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;
const Timer = std.time.Timer;
const math = std.math;

const TextGenerator = @import("generation.zig").TextGenerator;
const GenerationConfig = @import("generation.zig").GenerationConfig;
const GenerationResult = @import("generation.zig").GenerationResult;
const BatchProcessor = @import("batching.zig").BatchProcessor;
const BatchStats = @import("batching.zig").BatchStats;
const ModelKVCache = @import("kv_cache.zig").ModelKVCache;

/// Performance measurement point
pub const MeasurementPoint = struct {
    /// Name of the measurement
    name: []const u8,
    /// Start timestamp (nanoseconds)
    start_time: u64,
    /// End timestamp (nanoseconds)
    end_time: u64,
    /// Memory usage at start (bytes)
    start_memory: usize,
    /// Memory usage at end (bytes)
    end_memory: usize,
    /// Additional context data
    context: ?[]const u8,

    pub fn init(name: []const u8) MeasurementPoint {
        return MeasurementPoint{
            .name = name,
            .start_time = 0,
            .end_time = 0,
            .start_memory = 0,
            .end_memory = 0,
            .context = null,
        };
    }

    pub fn duration(self: MeasurementPoint) f64 {
        if (self.end_time > self.start_time) {
            return @as(f64, @floatFromInt(self.end_time - self.start_time)) / 1_000_000.0; // Convert to milliseconds
        }
        return 0.0;
    }

    pub fn memoryDelta(self: MeasurementPoint) i64 {
        return @as(i64, @intCast(self.end_memory)) - @as(i64, @intCast(self.start_memory));
    }
};

/// Aggregated performance statistics
pub const PerformanceStats = struct {
    /// Total number of measurements
    count: u64 = 0,
    /// Minimum duration (ms)
    min_duration: f64 = math.inf(f64),
    /// Maximum duration (ms)
    max_duration: f64 = 0.0,
    /// Average duration (ms)
    avg_duration: f64 = 0.0,
    /// Median duration (ms)
    median_duration: f64 = 0.0,
    /// 95th percentile duration (ms)
    p95_duration: f64 = 0.0,
    /// 99th percentile duration (ms)
    p99_duration: f64 = 0.0,
    /// Standard deviation
    std_deviation: f64 = 0.0,
    /// Total memory allocated
    total_memory_allocated: u64 = 0,
    /// Peak memory usage
    peak_memory_usage: usize = 0,

    /// Update statistics with new measurement
    pub fn update(self: *PerformanceStats, measurement: MeasurementPoint) void {
        const duration_ms = measurement.duration();

        self.count += 1;
        self.min_duration = @min(self.min_duration, duration_ms);
        self.max_duration = @max(self.max_duration, duration_ms);

        // Update running average
        const n = @as(f64, @floatFromInt(self.count));
        self.avg_duration = (self.avg_duration * (n - 1.0) + duration_ms) / n;

        // Update memory statistics
        if (measurement.memoryDelta() > 0) {
            self.total_memory_allocated += @as(u64, @intCast(measurement.memoryDelta()));
        }
        self.peak_memory_usage = @max(self.peak_memory_usage, measurement.end_memory);
    }

    /// Calculate percentiles from duration list
    pub fn calculatePercentiles(self: *PerformanceStats, durations: []f64) void {
        if (durations.len == 0) return;

        // Sort durations
        std.mem.sort(f64, durations, {}, std.sort.asc(f64));

        // Calculate percentiles
        self.median_duration = durations[durations.len / 2];
        self.p95_duration = durations[@min(durations.len - 1, (durations.len * 95) / 100)];
        self.p99_duration = durations[@min(durations.len - 1, (durations.len * 99) / 100)];

        // Calculate standard deviation
        var variance: f64 = 0.0;
        for (durations) |duration| {
            const diff = duration - self.avg_duration;
            variance += diff * diff;
        }
        variance /= @as(f64, @floatFromInt(durations.len));
        self.std_deviation = @sqrt(variance);
    }

    pub fn print(self: PerformanceStats, writer: anytype) !void {
        try writer.print("Performance Statistics:\n", .{});
        try writer.print("  Measurements: {d}\n", .{self.count});
        try writer.print("  Duration (ms):\n", .{});
        try writer.print("    Min: {d:.2}\n", .{self.min_duration});
        try writer.print("    Max: {d:.2}\n", .{self.max_duration});
        try writer.print("    Avg: {d:.2}\n", .{self.avg_duration});
        try writer.print("    Median: {d:.2}\n", .{self.median_duration});
        try writer.print("    P95: {d:.2}\n", .{self.p95_duration});
        try writer.print("    P99: {d:.2}\n", .{self.p99_duration});
        try writer.print("    Std Dev: {d:.2}\n", .{self.std_deviation});
        try writer.print("  Memory:\n", .{});
        try writer.print("    Total allocated: {d} bytes\n", .{self.total_memory_allocated});
        try writer.print("    Peak usage: {d} bytes\n", .{self.peak_memory_usage});
    }
};

/// Performance profiler for detailed measurement
pub const Profiler = struct {
    /// Measurement points by name
    measurements: HashMap([]const u8, ArrayList(MeasurementPoint), StringContext, std.hash_map.default_max_load_percentage),
    /// Aggregated statistics by name
    statistics: HashMap([]const u8, PerformanceStats, StringContext, std.hash_map.default_max_load_percentage),
    /// Active measurements (not yet completed)
    active_measurements: HashMap([]const u8, MeasurementPoint, StringContext, std.hash_map.default_max_load_percentage),
    /// Memory allocator
    allocator: Allocator,
    /// Timer for high-resolution timing
    timer: Timer,
    /// Whether profiling is enabled
    enabled: bool,

    const StringContext = struct {
        pub fn hash(self: @This(), s: []const u8) u64 {
            _ = self;
            return std.hash_map.hashString(s);
        }

        pub fn eql(self: @This(), a: []const u8, b: []const u8) bool {
            _ = self;
            return std.mem.eql(u8, a, b);
        }
    };

    /// Initialize profiler
    pub fn init(allocator: Allocator) !Profiler {
        return Profiler{
            .measurements = HashMap([]const u8, ArrayList(MeasurementPoint), StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .statistics = HashMap([]const u8, PerformanceStats, StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .active_measurements = HashMap([]const u8, MeasurementPoint, StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .allocator = allocator,
            .timer = try Timer.start(),
            .enabled = true,
        };
    }

    /// Clean up profiler
    pub fn deinit(self: *Profiler) void {
        // Free measurement lists
        var measurement_iter = self.measurements.iterator();
        while (measurement_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit();
        }
        self.measurements.deinit();

        // Free statistics keys
        var stats_iter = self.statistics.iterator();
        while (stats_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.statistics.deinit();

        // Free active measurement keys
        var active_iter = self.active_measurements.iterator();
        while (active_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.active_measurements.deinit();
    }

    /// Enable/disable profiling
    pub fn setEnabled(self: *Profiler, enabled: bool) void {
        self.enabled = enabled;
    }

    /// Start measuring a named operation
    pub fn startMeasurement(self: *Profiler, name: []const u8) !void {
        if (!self.enabled) return;

        const owned_name = try self.allocator.dupe(u8, name);
        var measurement = MeasurementPoint.init(owned_name);
        measurement.start_time = self.timer.read();
        measurement.start_memory = self.getCurrentMemoryUsage();

        try self.active_measurements.put(owned_name, measurement);
    }

    /// End measuring a named operation
    pub fn endMeasurement(self: *Profiler, name: []const u8) !void {
        if (!self.enabled) return;

        if (self.active_measurements.fetchRemove(name)) |entry| {
            var measurement = entry.value;
            measurement.end_time = self.timer.read();
            measurement.end_memory = self.getCurrentMemoryUsage();

            // Store completed measurement
            const result = try self.measurements.getOrPut(entry.key);
            if (!result.found_existing) {
                result.value_ptr.* = ArrayList(MeasurementPoint).init(self.allocator);
                // Key is already owned by active_measurements, don't duplicate
            } else {
                // Free the key since we're not using it
                self.allocator.free(entry.key);
            }

            try result.value_ptr.append(measurement);

            // Update statistics
            const stats_result = try self.statistics.getOrPut(result.key_ptr.*);
            if (!stats_result.found_existing) {
                stats_result.value_ptr.* = PerformanceStats{};
            }

            stats_result.value_ptr.update(measurement);
        }
    }

    /// Measure a block of code with RAII
    pub fn measureBlock(self: *Profiler, name: []const u8) MeasurementBlock {
        return MeasurementBlock.init(self, name);
    }

    /// Get statistics for a named operation
    pub fn getStatistics(self: Profiler, name: []const u8) ?PerformanceStats {
        return self.statistics.get(name);
    }

    /// Get all measurements for a named operation
    pub fn getMeasurements(self: Profiler, name: []const u8) ?[]const MeasurementPoint {
        if (self.measurements.get(name)) |list| {
            return list.items;
        }
        return null;
    }

    /// Calculate and update percentiles for all statistics
    pub fn updatePercentiles(self: *Profiler) !void {
        var stats_iter = self.statistics.iterator();
        while (stats_iter.next()) |entry| {
            if (self.measurements.get(entry.key_ptr.*)) |measurements| {
                var durations = try self.allocator.alloc(f64, measurements.items.len);
                defer self.allocator.free(durations);

                for (measurements.items, 0..) |measurement, i| {
                    durations[i] = measurement.duration();
                }

                entry.value_ptr.calculatePercentiles(durations);
            }
        }
    }

    /// Reset all measurements and statistics
    pub fn reset(self: *Profiler) void {
        // Clear measurements
        var measurement_iter = self.measurements.iterator();
        while (measurement_iter.next()) |entry| {
            entry.value_ptr.clearAndFree();
        }

        // Reset statistics
        var stats_iter = self.statistics.iterator();
        while (stats_iter.next()) |entry| {
            entry.value_ptr.* = PerformanceStats{};
        }
    }

    /// Get current memory usage (simplified)
    fn getCurrentMemoryUsage(self: *Profiler) usize {
        // This is a placeholder - real implementation would use system APIs
        // or integrate with allocator tracking
        _ = self;
        return 0;
    }

    /// Print comprehensive profile report
    pub fn printReport(self: *Profiler, writer: anytype) !void {
        try self.updatePercentiles();

        try writer.print("=== Performance Profile Report ===\n\n", .{});

        var stats_iter = self.statistics.iterator();
        while (stats_iter.next()) |entry| {
            try writer.print("Operation: {s}\n", .{entry.key_ptr.*});
            try entry.value_ptr.print(writer);
            try writer.print("\n", .{});
        }
    }
};

/// RAII measurement block
pub const MeasurementBlock = struct {
    profiler: *Profiler,
    name: []const u8,
    started: bool,

    pub fn init(profiler: *Profiler, name: []const u8) MeasurementBlock {
        const block = MeasurementBlock{
            .profiler = profiler,
            .name = name,
            .started = false,
        };

        // Start measurement (ignore errors in RAII context)
        profiler.startMeasurement(name) catch {};

        return block;
    }

    pub fn deinit(self: MeasurementBlock) void {
        if (self.started) {
            self.profiler.endMeasurement(self.name) catch {};
        }
    }
};

/// Benchmark configuration
pub const BenchmarkConfig = struct {
    /// Number of warmup runs
    warmup_runs: u32 = 5,
    /// Number of measurement runs
    measurement_runs: u32 = 100,
    /// Maximum time per run (ms)
    max_time_per_run: u32 = 10000,
    /// Whether to print progress
    show_progress: bool = true,
    /// Memory limit per run
    memory_limit: ?usize = null,

    pub fn validate(self: BenchmarkConfig) !void {
        if (self.measurement_runs == 0) {
            return error.InvalidMeasurementRuns;
        }
        if (self.max_time_per_run == 0) {
            return error.InvalidMaxTime;
        }
    }
};

/// Comprehensive benchmark result
pub const BenchmarkResult = struct {
    /// Test name
    name: []const u8,
    /// Configuration used
    config: BenchmarkConfig,
    /// Performance statistics
    stats: PerformanceStats,
    /// Throughput metrics
    throughput: ThroughputMetrics,
    /// Memory usage metrics
    memory: MemoryMetrics,
    /// Success rate
    success_rate: f32,
    /// Benchmark duration
    total_duration_ms: f64,

    pub fn print(self: BenchmarkResult, writer: anytype) !void {
        try writer.print("=== Benchmark Results: {s} ===\n", .{self.name});
        try writer.print("Configuration:\n", .{});
        try writer.print("  Warmup runs: {d}\n", .{self.config.warmup_runs});
        try writer.print("  Measurement runs: {d}\n", .{self.config.measurement_runs});
        try writer.print("  Success rate: {d:.1}%\n", .{self.success_rate * 100.0});
        try writer.print("  Total duration: {d:.2} ms\n", .{self.total_duration_ms});
        try writer.print("\n", .{});

        try self.stats.print(writer);
        try writer.print("\n", .{});

        try self.throughput.print(writer);
        try writer.print("\n", .{});

        try self.memory.print(writer);
        try writer.print("\n", .{});
    }
};

/// Throughput measurement metrics
pub const ThroughputMetrics = struct {
    /// Requests per second
    requests_per_second: f32 = 0.0,
    /// Tokens per second
    tokens_per_second: f32 = 0.0,
    /// Characters per second
    characters_per_second: f32 = 0.0,
    /// Batches per second
    batches_per_second: f32 = 0.0,

    pub fn print(self: ThroughputMetrics, writer: anytype) !void {
        try writer.print("Throughput Metrics:\n", .{});
        try writer.print("  Requests/second: {d:.1}\n", .{self.requests_per_second});
        try writer.print("  Tokens/second: {d:.1}\n", .{self.tokens_per_second});
        try writer.print("  Characters/second: {d:.1}\n", .{self.characters_per_second});
        try writer.print("  Batches/second: {d:.1}\n", .{self.batches_per_second});
    }
};

/// Memory usage metrics
pub const MemoryMetrics = struct {
    /// Peak memory usage (bytes)
    peak_usage: usize = 0,
    /// Average memory usage (bytes)
    average_usage: usize = 0,
    /// Memory allocated per request (bytes)
    per_request_allocation: usize = 0,
    /// Memory efficiency (useful bytes / total bytes)
    efficiency: f32 = 0.0,

    pub fn print(self: MemoryMetrics, writer: anytype) !void {
        try writer.print("Memory Metrics:\n", .{});
        try writer.print("  Peak usage: {d} bytes ({d:.1} MB)\n", .{ self.peak_usage, @as(f32, @floatFromInt(self.peak_usage)) / 1024.0 / 1024.0 });
        try writer.print("  Average usage: {d} bytes ({d:.1} MB)\n", .{ self.average_usage, @as(f32, @floatFromInt(self.average_usage)) / 1024.0 / 1024.0 });
        try writer.print("  Per-request allocation: {d} bytes\n", .{self.per_request_allocation});
        try writer.print("  Efficiency: {d:.1}%\n", .{self.efficiency * 100.0});
    }
};

/// Benchmark runner
pub const BenchmarkRunner = struct {
    allocator: Allocator,
    profiler: Profiler,
    config: BenchmarkConfig,

    pub fn init(allocator: Allocator, config: BenchmarkConfig) !BenchmarkRunner {
        try config.validate();

        return BenchmarkRunner{
            .allocator = allocator,
            .profiler = try Profiler.init(allocator),
            .config = config,
        };
    }

    pub fn deinit(self: *BenchmarkRunner) void {
        self.profiler.deinit();
    }

    /// Run generation benchmark
    pub fn benchmarkGeneration(
        self: *BenchmarkRunner,
        generator: *TextGenerator,
        test_prompts: []const []const u8,
        gen_config: GenerationConfig
    ) !BenchmarkResult {
        const start_time = std.time.milliTimestamp();
        var successful_runs: u32 = 0;
        var total_tokens: u32 = 0;
        var total_chars: u32 = 0;

        // Warmup
        if (self.config.show_progress) {
            std.debug.print("Running warmup ({d} runs)...\n", .{self.config.warmup_runs});
        }

        for (0..self.config.warmup_runs) |_| {
            for (test_prompts) |prompt| {
                const result = generator.generate(prompt) catch continue;
                result.deinit(self.allocator);
            }
        }

        // Measurement runs
        if (self.config.show_progress) {
            std.debug.print("Running measurements ({d} runs)...\n", .{self.config.measurement_runs});
        }

        for (0..self.config.measurement_runs) |run| {
            if (self.config.show_progress and run % 10 == 0) {
                std.debug.print("  Progress: {d}/{d}\n", .{ run, self.config.measurement_runs });
            }

            for (test_prompts) |prompt| {
                try self.profiler.startMeasurement("generation");

                const result = generator.generate(prompt) catch |err| {
                    try self.profiler.endMeasurement("generation");
                    _ = err;
                    continue;
                };

                try self.profiler.endMeasurement("generation");

                successful_runs += 1;
                total_tokens += result.num_tokens;
                if (result.text) |text| {
                    total_chars += @as(u32, @intCast(text.len));
                }

                result.deinit(self.allocator);
            }
        }

        const end_time = std.time.milliTimestamp();
        const total_duration = @as(f64, @floatFromInt(end_time - start_time));

        // Calculate metrics
        const total_runs = self.config.measurement_runs * @as(u32, @intCast(test_prompts.len));
        const success_rate = @as(f32, @floatFromInt(successful_runs)) / @as(f32, @floatFromInt(total_runs));

        const throughput = ThroughputMetrics{
            .requests_per_second = @as(f32, @floatFromInt(successful_runs)) * 1000.0 / @as(f32, @floatCast(total_duration)),
            .tokens_per_second = @as(f32, @floatFromInt(total_tokens)) * 1000.0 / @as(f32, @floatCast(total_duration)),
            .characters_per_second = @as(f32, @floatFromInt(total_chars)) * 1000.0 / @as(f32, @floatCast(total_duration)),
            .batches_per_second = 0.0, // Not applicable for single generation
        };

        const stats = self.profiler.getStatistics("generation") orelse PerformanceStats{};

        return BenchmarkResult{
            .name = "Text Generation",
            .config = self.config,
            .stats = stats,
            .throughput = throughput,
            .memory = MemoryMetrics{}, // Would be filled with actual memory tracking
            .success_rate = success_rate,
            .total_duration_ms = total_duration,
        };
    }
};

// Performance profiling tests
test "measurement point operations" {
    const testing = std.testing;

    var point = MeasurementPoint.init("test_operation");
    point.start_time = 1000000; // 1ms in nanoseconds
    point.end_time = 3000000;   // 3ms in nanoseconds
    point.start_memory = 1000;
    point.end_memory = 1500;

    try testing.expectEqual(@as(f64, 2.0), point.duration()); // 2ms
    try testing.expectEqual(@as(i64, 500), point.memoryDelta()); // 500 bytes allocated
}

test "performance statistics updates" {
    const testing = std.testing;

    var stats = PerformanceStats{};
    try testing.expectEqual(@as(u64, 0), stats.count);
    try testing.expect(std.math.isInf(stats.min_duration));

    var measurement = MeasurementPoint.init("test");
    measurement.start_time = 1000000;
    measurement.end_time = 3000000; // 2ms duration
    measurement.start_memory = 1000;
    measurement.end_memory = 1200;

    stats.update(measurement);
    try testing.expectEqual(@as(u64, 1), stats.count);
    try testing.expectEqual(@as(f64, 2.0), stats.avg_duration);
    try testing.expectEqual(@as(f64, 2.0), stats.min_duration);
    try testing.expectEqual(@as(f64, 2.0), stats.max_duration);
}

test "benchmark configuration validation" {
    const testing = std.testing;

    const valid_config = BenchmarkConfig{};
    try valid_config.validate();

    var invalid_config = valid_config;
    invalid_config.measurement_runs = 0;
    try testing.expectError(error.InvalidMeasurementRuns, invalid_config.validate());

    invalid_config = valid_config;
    invalid_config.max_time_per_run = 0;
    try testing.expectError(error.InvalidMaxTime, invalid_config.validate());
}

test "throughput metrics calculation" {
    const testing = std.testing;

    const throughput = ThroughputMetrics{
        .requests_per_second = 100.0,
        .tokens_per_second = 2000.0,
        .characters_per_second = 10000.0,
        .batches_per_second = 25.0,
    };

    // Basic validation that metrics are set
    try testing.expectEqual(@as(f32, 100.0), throughput.requests_per_second);
    try testing.expectEqual(@as(f32, 2000.0), throughput.tokens_per_second);
    try testing.expectEqual(@as(f32, 10000.0), throughput.characters_per_second);
    try testing.expectEqual(@as(f32, 25.0), throughput.batches_per_second);
}