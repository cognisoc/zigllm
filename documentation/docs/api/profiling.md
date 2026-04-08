# inference.profiling

## Module Path

```
zigllama.inference.profiling
```

**Source file:** `src/inference/profiling.zig`

---

## Public Types

### `MeasurementPoint`

```zig
pub const MeasurementPoint = struct {
    name: []const u8,
    start_ns: u64,
    end_ns: ?u64,
    parent: ?*MeasurementPoint,
    children: std.ArrayList(*MeasurementPoint),
};
```

A single timed region. Measurement points form a tree so nested operations
(e.g., attention inside a transformer block) are tracked hierarchically.

### `PerformanceStats`

```zig
pub const PerformanceStats = struct {
    min: f64,
    max: f64,
    avg: f64,
    median: f64,
    p95: f64,
    p99: f64,
    std_dev: f64,
    count: usize,
    total: f64,
};
```

Aggregate statistics computed from a series of measurements. All timing values
are in milliseconds.

### `Profiler`

```zig
pub const Profiler = struct {
    measurements: std.StringHashMap(std.ArrayList(f64)),
    active_points: std.ArrayList(*MeasurementPoint),
    enabled: bool,
    allocator: std.mem.Allocator,
};
```

Collects timing data for named code regions. Thread-safe when each thread uses
its own `Profiler` instance.

### `BenchmarkRunner`

```zig
pub const BenchmarkRunner = struct {
    name: []const u8,
    warmup_iterations: usize,
    profiler: Profiler,
    allocator: std.mem.Allocator,
};
```

Runs a function multiple times with warm-up iterations and produces
`PerformanceStats`.

---

## Public Functions

### `Profiler.init`

```zig
pub fn init(allocator: std.mem.Allocator) Profiler
```

Create a new profiler with no recorded measurements.

### `Profiler.deinit`

```zig
pub fn deinit(self: *Profiler) void
```

Free all stored measurements.

### `Profiler.measureBlock`

```zig
pub fn measureBlock(self: *Profiler, name: []const u8) MeasurementGuard
```

Begin timing a named block. Returns a guard object; timing stops when the guard
goes out of scope (via `defer guard.end()`).

```zig
{
    var guard = profiler.measureBlock("attention");
    defer guard.end();
    // ... attention computation ...
}
```

### `Profiler.getStats`

```zig
pub fn getStats(self: *Profiler, name: []const u8) ?PerformanceStats
```

Compute aggregate statistics for the named block. Returns `null` if no
measurements have been recorded under that name.

### `Profiler.report`

```zig
pub fn report(self: *Profiler, writer: anytype) !void
```

Print a formatted table of all recorded blocks with their statistics.

### `BenchmarkRunner.init`

```zig
pub fn init(
    name: []const u8,
    warmup_iterations: usize,
    allocator: std.mem.Allocator,
) BenchmarkRunner
```

Create a benchmark runner.

### `BenchmarkRunner.run`

```zig
pub fn run(
    self: *BenchmarkRunner,
    func: *const fn () void,
    iterations: usize,
) !PerformanceStats
```

Execute `func` for `warmup_iterations + iterations` times, discarding the
warm-up measurements, and return statistics for the measured iterations.

---

## Error Types

- `error{OutOfMemory}`

---

## Usage Example

```zig
const prof = @import("zigllama").inference.profiling;

var profiler = prof.Profiler.init(allocator);
defer profiler.deinit();

// Profile a forward pass
{
    var guard = profiler.measureBlock("forward_pass");
    defer guard.end();
    _ = try model.forward(tokens, allocator);
}

// Get statistics
if (profiler.getStats("forward_pass")) |stats| {
    std.debug.print("Forward pass: avg={d:.2}ms, p99={d:.2}ms\n", .{
        stats.avg, stats.p99,
    });
}

// Or run a benchmark
var bench = prof.BenchmarkRunner.init("inference", 5, allocator);
const stats = try bench.run(inferenceFunc, 100);
std.debug.print("Throughput: {d:.1} tokens/sec\n", .{1000.0 / stats.avg});
```

---

## Related Modules

- [`inference.generation`](generation.md) -- Profile generation latency.
- [`inference.batching`](batching.md) -- `BatchStats` provides coarser
  throughput metrics.
