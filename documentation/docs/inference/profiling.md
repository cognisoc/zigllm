---
title: "Performance Profiling"
description: "RAII measurement blocks, statistical analysis, the BenchmarkRunner, the roofline model, and bottleneck identification for ZigLlama inference."
---

# Performance Profiling

Optimising inference performance requires *measurement before speculation*.
ZigLlama's profiling module provides high-resolution timing, memory
tracking, statistical aggregation, and benchmark orchestration -- all
designed to integrate seamlessly with the inference pipeline without
distorting the results.

---

## 1. Profiler Struct

The `Profiler` is the central data structure for collecting and analysing
performance measurements:

```zig
pub const Profiler = struct {
    measurements: HashMap([]const u8, ArrayList(MeasurementPoint), ...),
    statistics: HashMap([]const u8, PerformanceStats, ...),
    active_measurements: HashMap([]const u8, MeasurementPoint, ...),
    allocator: Allocator,
    timer: Timer,
    enabled: bool,

    pub fn init(allocator: Allocator) !Profiler { ... }
    pub fn deinit(self: *Profiler) void { ... }
    pub fn setEnabled(self: *Profiler, enabled: bool) void { ... }
    pub fn startMeasurement(self: *Profiler, name: []const u8) !void { ... }
    pub fn endMeasurement(self: *Profiler, name: []const u8) !void { ... }
    pub fn measureBlock(self: *Profiler, name: []const u8) MeasurementBlock { ... }
    pub fn getStatistics(self: Profiler, name: []const u8) ?PerformanceStats { ... }
    pub fn updatePercentiles(self: *Profiler) !void { ... }
    pub fn printReport(self: *Profiler, writer: anytype) !void { ... }
    pub fn reset(self: *Profiler) void { ... }
};
```

### 1.1 Design Principles

- **Named measurements**: Each operation is identified by a string key
  (e.g., `"forward_pass"`, `"sampling"`, `"kv_cache_update"`), enabling
  per-component analysis.
- **Zero-overhead when disabled**: When `enabled = false`, all measurement
  methods return immediately.
- **Thread-compatible**: While the profiler itself is not thread-safe
  (each thread should use its own instance), the `MeasurementPoint` data
  is self-contained and can be merged across threads.

---

## 2. RAII Measurement: measureBlock

The `measureBlock` pattern leverages Zig's `defer` to automatically end
a measurement when the enclosing scope exits, even on error paths:

```zig
pub const MeasurementBlock = struct {
    profiler: *Profiler,
    name: []const u8,
    started: bool,

    pub fn init(profiler: *Profiler, name: []const u8) MeasurementBlock { ... }
    pub fn deinit(self: MeasurementBlock) void {
        if (self.started) {
            self.profiler.endMeasurement(self.name) catch {};
        }
    }
};
```

Usage:

```zig
// Automatic measurement -- timing stops when block exits
{
    const block = profiler.measureBlock("forward_pass");
    defer block.deinit();

    // ... model forward pass ...
    // If this errors out, the measurement still ends cleanly
    try model.forward(tokens);
}
```

!!! tip "Nested Measurements"

    Measurements can be nested to profile sub-operations within a larger
    operation:

    ```zig
    {
        const outer = profiler.measureBlock("generation_step");
        defer outer.deinit();

        {
            const inner1 = profiler.measureBlock("forward_pass");
            defer inner1.deinit();
            try model.forward(tokens);
        }
        {
            const inner2 = profiler.measureBlock("sampling");
            defer inner2.deinit();
            _ = try generator.sampleToken(logits);
        }
    }
    ```

---

## 3. PerformanceStats

Each named operation accumulates statistics in a `PerformanceStats` struct:

```zig
pub const PerformanceStats = struct {
    count: u64,
    min_duration: f64,      // milliseconds
    max_duration: f64,
    avg_duration: f64,
    median_duration: f64,
    p95_duration: f64,      // 95th percentile
    p99_duration: f64,      // 99th percentile
    std_deviation: f64,
    total_memory_allocated: u64,
    peak_memory_usage: usize,
};
```

### 3.1 Running Average Update

Statistics are updated incrementally using a running average:

\[
    \bar{x}_n = \frac{\bar{x}_{n-1} \cdot (n-1) + x_n}{n}
\]

This avoids storing all measurements in memory just to compute the mean.

### 3.2 Percentile Calculation

Percentiles require the full measurement history.  The `updatePercentiles`
method sorts the stored durations and computes the median, P95, and P99:

```zig
pub fn calculatePercentiles(self: *PerformanceStats, durations: []f64) void {
    std.mem.sort(f64, durations, {}, std.sort.asc(f64));

    self.median_duration = durations[durations.len / 2];
    self.p95_duration = durations[(durations.len * 95) / 100];
    self.p99_duration = durations[(durations.len * 99) / 100];

    // Standard deviation
    var variance: f64 = 0.0;
    for (durations) |d| {
        const diff = d - self.avg_duration;
        variance += diff * diff;
    }
    variance /= @as(f64, @floatFromInt(durations.len));
    self.std_deviation = @sqrt(variance);
}
```

!!! info "Why P95 and P99?"

    The **average** latency is misleading for user-facing systems because
    it hides long-tail outliers.  The P95 tells you that 95% of requests
    complete within that time; the P99 catches the worst-case spikes that
    users may actually experience.  For production SLOs, P99 is the standard
    metric.

### 3.3 Memory Metrics

Each `MeasurementPoint` records memory usage at start and end:

```zig
pub const MeasurementPoint = struct {
    name: []const u8,
    start_time: u64,
    end_time: u64,
    start_memory: usize,
    end_memory: usize,
    context: ?[]const u8,

    pub fn duration(self: MeasurementPoint) f64 { ... }
    pub fn memoryDelta(self: MeasurementPoint) i64 { ... }
};
```

A positive `memoryDelta` indicates net allocation; a negative delta
indicates net deallocation.  The peak memory across all measurements
is tracked in `PerformanceStats.peak_memory_usage`.

---

## 4. BenchmarkRunner

The `BenchmarkRunner` orchestrates systematic benchmarks with warmup,
measurement rounds, and statistical analysis:

```zig
pub const BenchmarkRunner = struct {
    allocator: Allocator,
    profiler: Profiler,
    config: BenchmarkConfig,

    pub fn init(allocator: Allocator, config: BenchmarkConfig) !BenchmarkRunner { ... }
    pub fn deinit(self: *BenchmarkRunner) void { ... }
    pub fn benchmarkGeneration(self: *BenchmarkRunner, generator: *TextGenerator,
                               test_prompts: []const []const u8,
                               config: GenerationConfig) !BenchmarkResult { ... }
};
```

### 4.1 BenchmarkConfig

```zig
pub const BenchmarkConfig = struct {
    warmup_runs: u32 = 5,           // Warm up CPU caches and JIT
    measurement_runs: u32 = 100,    // Statistical sample size
    max_time_per_run: u32 = 10000,  // Timeout per run (ms)
    show_progress: bool = true,
    memory_limit: ?usize = null,
};
```

!!! warning "Why Warmup Matters"

    The first few runs of any benchmark are typically slower due to cold
    CPU caches, branch predictor training, and (on some platforms) dynamic
    frequency scaling.  Warmup runs ensure that subsequent measurements
    reflect steady-state performance.  Five warmup runs is generally
    sufficient for CPU inference.

### 4.2 BenchmarkResult

```zig
pub const BenchmarkResult = struct {
    name: []const u8,
    config: BenchmarkConfig,
    stats: PerformanceStats,
    throughput: ThroughputMetrics,
    memory: MemoryMetrics,
    success_rate: f32,
    total_duration_ms: f64,
};
```

Where `ThroughputMetrics` and `MemoryMetrics` provide domain-specific
measurements:

```zig
pub const ThroughputMetrics = struct {
    requests_per_second: f32,
    tokens_per_second: f32,
    characters_per_second: f32,
    batches_per_second: f32,
};

pub const MemoryMetrics = struct {
    peak_usage: usize,
    average_usage: usize,
    per_request_allocation: usize,
    efficiency: f32,  // useful bytes / total bytes
};
```

---

## 5. Roofline Model

!!! definition "Roofline Model"

    The roofline model characterises the performance of a computation by
    two hardware parameters and one application parameter:

    - \( \pi \): peak computational throughput (FLOPS)
    - \( \beta \): peak memory bandwidth (bytes/second)
    - \( I \): arithmetic intensity (FLOPS per byte transferred)

    The achievable performance is:

    \[
        \text{Perf} = \min(\pi, \; \beta \cdot I)
    \]

    If \( \beta \cdot I < \pi \), the operation is **memory-bound** (limited
    by how fast data can be fed to the compute units).  Otherwise it is
    **compute-bound** (limited by raw arithmetic throughput).

### 5.1 Arithmetic Intensity of Key Operations

| Operation | FLOPS | Bytes Transferred | Intensity \( I \) | Typical Bound |
|---|---|---|---|---|
| Matrix multiply \( (m,n) \times (n,p) \) | \( 2mnp \) | \( 4(mn + np + mp) \) | \( \approx \frac{m}{2} \) for large \( n \) | Compute (large \( m \)) |
| Attention \( \mathbf{Q}\mathbf{K}^{\!\top} \) | \( 2 n^2 d_h \) | \( 4(2n d_h + n^2) \) | \( \approx d_h / 2 \) | Memory (small \( d_h \)) |
| Softmax | \( 4n \) | \( 8n \) | 0.5 | Memory |
| RMSNorm | \( 3d \) | \( 8d \) | 0.375 | Memory |
| Token embedding lookup | 0 | \( 4d \) | 0 | Memory |

### 5.2 Applying the Roofline to Inference

For single-token generation on a typical CPU:

- \( \pi \approx 100 \) GFLOPS (AVX2, 8 cores)
- \( \beta \approx 50 \) GB/s (DDR4 dual-channel)
- Model weight loading per token: \( \approx 4d^2 L \) bytes (all weight matrices)
- Compute per token: \( \approx 8d^2 L \) FLOPS

The arithmetic intensity is:

\[
    I = \frac{8 d^2 L}{4 d^2 L} = 2 \; \text{FLOPS/byte}
\]

\[
    \beta \cdot I = 50 \times 2 = 100 \; \text{GFLOPS} \approx \pi
\]

Single-token generation sits right at the boundary -- it is
**memory-bound** for single sequences (batch size 1) and shifts toward
**compute-bound** as batch size increases.

!!! tip "Practical Implication"

    For single-request inference, the dominant bottleneck is **loading model
    weights from RAM**, not computing the matrix products.  This is why
    quantisation (which reduces weight size) has such a large impact on
    single-request throughput: Q4 quantisation roughly halves the bytes
    transferred per token, nearly doubling inference speed.

---

## 6. Bottleneck Identification

The profiling module helps identify whether an operation is compute-bound
or memory-bound by comparing actual throughput against the roofline:

| Symptom | Diagnosis | Optimisation |
|---|---|---|
| Throughput scales linearly with batch size | Memory-bound | Quantise weights, increase bandwidth |
| Throughput plateaus with larger batch | Compute-bound | Use BLAS, vectorise inner loops |
| P99 >> P50 | Tail latency spikes | Check GC pauses, thread contention |
| Memory grows linearly with sequence length | KV cache dominant | Use sliding window, quantise cache |
| First-token latency >> per-token latency | Prefill bottleneck | Chunk prefill, cache prompt tokens |

### 6.1 Profiling Workflow

!!! algorithm "Bottleneck Identification Workflow"

    1. **Instrument** the inference pipeline with named measurements:
       `forward_pass`, `kv_cache_update`, `sampling`, `token_decode`.
    2. **Run** the benchmark with `BenchmarkRunner` (5 warmup + 100
       measurement rounds).
    3. **Analyse** the `BenchmarkResult`:
        - Which operation has the highest P99?
        - What fraction of total time does each operation consume?
        - Does memory grow unexpectedly?
    4. **Compute** arithmetic intensity for the bottleneck operation.
    5. **Compare** against the roofline to determine if it is memory-bound
       or compute-bound.
    6. **Apply** the appropriate optimisation and re-measure.

---

## 7. Integration with Inference Pipeline

The profiler integrates at each stage of the generation loop:

```zig
// Example: profiled generation loop
pub fn generateWithProfiling(
    generator: *TextGenerator,
    profiler: *Profiler,
    prompt: []const u8,
) !GenerationResult {
    {
        const block = profiler.measureBlock("tokenize");
        defer block.deinit();
        // ... tokenize prompt ...
    }

    while (generating) {
        {
            const block = profiler.measureBlock("forward_pass");
            defer block.deinit();
            logits = try model.forward(tokens);
        }
        {
            const block = profiler.measureBlock("kv_cache_update");
            defer block.deinit();
            try cache.appendToSequence(seq_id, layer_id, &keys, &values);
        }
        {
            const block = profiler.measureBlock("sampling");
            defer block.deinit();
            next_token = try generator.sampleToken(logits);
        }
    }

    {
        const block = profiler.measureBlock("decode");
        defer block.deinit();
        // ... decode tokens to text ...
    }
}
```

After generation, print the report:

```zig
try profiler.printReport(std.io.getStdErr().writer());
```

Sample output:

```
=== Performance Profile Report ===

Operation: forward_pass
  Measurements: 128
  Duration (ms):
    Min: 12.50
    Max: 18.30
    Avg: 14.20
    Median: 14.00
    P95: 16.80
    P99: 17.90
    Std Dev: 1.20
  Memory:
    Total allocated: 524288 bytes
    Peak usage: 262144 bytes

Operation: sampling
  Measurements: 128
  Duration (ms):
    Min: 0.05
    Max: 0.30
    Avg: 0.08
    Median: 0.07
    P95: 0.15
    P99: 0.25
    Std Dev: 0.04
```

!!! info "Where Does Time Go?"

    For a typical 7B model on CPU, the time breakdown is approximately:

    | Component | Share | Bound |
    |---|---|---|
    | Forward pass (GEMM) | 85--95% | Memory/Compute |
    | KV cache update | 2--5% | Memory |
    | Sampling | 1--3% | Compute |
    | Tokenize/Decode | < 1% | Compute |

    The forward pass dominates, which is why optimisations at layers 1--2
    (SIMD, quantisation, BLAS) have the highest return on investment.

---

## References

[^1]: Williams, S., Waterman, A. & Patterson, D. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *Communications of the ACM*, 52(4), 2009.
[^2]: Ivanov, A. et al. "Data Movement Is All You Need: A Case Study on Optimizing Transformers." *MLSys*, 2021.
[^3]: Kim, S. et al. "Full Stack Optimization of Transformer Inference: A Survey." *arXiv:2302.14017*, 2023.
[^4]: Pope, R. et al. "Efficiently Scaling Transformer Inference." *MLSys*, 2023.
