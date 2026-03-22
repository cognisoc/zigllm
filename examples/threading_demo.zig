// Threading and NUMA Optimization Demonstration
// Educational showcase of parallel computation infrastructure
//
// This example demonstrates:
// 1. CPU topology detection and analysis
// 2. Thread pool creation and management
// 3. Work-stealing load balancing
// 4. Parallel matrix operations
// 5. NUMA-aware memory allocation
// 6. Performance monitoring and optimization

const std = @import("std");
const threading = @import("../src/foundation/threading.zig");
const Tensor = @import("../src/foundation/tensor.zig").Tensor;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Threading and NUMA Optimization Demo ===\n\n");

    // Section 1: CPU Topology Discovery
    try demonstrateCpuTopology();

    // Section 2: Thread Pool Management
    try demonstrateThreadPool(allocator);

    // Section 3: Parallel Matrix Operations
    try demonstrateParallelOperations(allocator);

    // Section 4: NUMA-Aware Memory Allocation
    try demonstrateNumaAllocation(allocator);

    // Section 5: Performance Analysis
    try demonstratePerformanceAnalysis(allocator);

    // Section 6: Work-Stealing Load Balancing
    try demonstrateWorkStealing(allocator);

    std.debug.print("\n=== Demo Complete ===\n");
}

fn demonstrateCpuTopology() !void {
    std.debug.print("=== CPU Topology Discovery ===\n");

    const topology = threading.CpuTopology.detect();
    topology.print();

    std.debug.print("Analysis:\n");
    std.debug.print("• Parallelism Level: {} threads available\n", .{topology.num_threads});
    std.debug.print("• NUMA Topology: {} NUMA node(s)\n", .{topology.num_numa_nodes});
    std.debug.print("• Memory Hierarchy:\n");
    std.debug.print("  - L1 Cache: {:.1} KB per core (fast access)\n", .{@as(f32, @floatFromInt(topology.l1_cache_size)) / 1024.0});
    std.debug.print("  - L2 Cache: {:.1} KB per core (medium access)\n", .{@as(f32, @floatFromInt(topology.l2_cache_size)) / 1024.0});
    std.debug.print("  - L3 Cache: {:.1} MB shared (slower access)\n", .{@as(f32, @floatFromInt(topology.l3_cache_size)) / 1_048_576.0});
    std.debug.print("• Cache Line Size: {} bytes (alignment unit)\n", .{topology.cache_line_size});

    const recommended_threads = @max(1, topology.num_threads - 1);
    std.debug.print("• Recommended Thread Pool Size: {} threads\n", .{recommended_threads});
    std.debug.print("  (Leave 1 thread for OS and background tasks)\n\n");
}

fn demonstrateThreadPool(allocator: std.mem.Allocator) !void {
    std.debug.print("=== Thread Pool Management ===\n");

    // Auto-detect optimal configuration
    const config = threading.ThreadPoolConfig.detect();
    std.debug.print("Auto-detected configuration:\n");
    std.debug.print("• Worker Threads: {}\n", .{config.num_threads});
    std.debug.print("• NUMA Policy: {}\n", .{config.numa_policy});
    std.debug.print("• Work Stealing: {}\n", .{config.work_stealing});
    std.debug.print("• CPU Affinity: {}\n", .{config.affinity_enabled});

    // Create and start thread pool
    var thread_pool = try threading.ThreadPool.init(allocator, config);
    defer thread_pool.deinit();

    try thread_pool.start();
    defer thread_pool.stop();

    // Demonstrate work submission and execution
    std.debug.print("\nSubmitting computational work...\n");

    var work_items: [20]threading.WorkItem = undefined;
    var counters: [20]u32 = undefined;
    @memset(&counters, 0);

    // Create work items that increment counters
    for (0..20) |i| {
        work_items[i] = threading.WorkItem{
            .func = computationalWork,
            .data = &counters[i],
        };
        _ = thread_pool.submit(&work_items[i]);
    }

    // Wait for all work to complete
    for (&work_items) |*item| {
        item.waitForCompletion();
    }

    // Verify results
    var completed_count: u32 = 0;
    for (counters) |counter| {
        if (counter == 1000) { // Each work item increments counter 1000 times
            completed_count += 1;
        }
    }

    std.debug.print("Results: {}/20 work items completed successfully\n", .{completed_count});

    // Show queue distribution
    const queue_sizes = thread_pool.getQueueSizes();
    defer allocator.free(queue_sizes);

    std.debug.print("Queue distribution after completion:\n");
    for (queue_sizes, 0..) |size, i| {
        std.debug.print("  Worker {}: {} items\n", .{ i, size });
    }
    std.debug.print("");
}

fn computationalWork(work_item: *threading.WorkItem) void {
    const counter = @as(*u32, @ptrCast(@alignCast(work_item.data.?)));

    // Simulate computational work
    for (0..1000) |_| {
        counter.* += 1;
    }

    // Simulate some variance in work completion time
    const thread_id = threading.ThreadingUtils.getCurrentThreadId();
    const delay = (thread_id % 100) * 1000; // Microseconds
    std.time.sleep(delay);
}

fn demonstrateParallelOperations(allocator: std.mem.Allocator) !void {
    std.debug.print("=== Parallel Matrix Operations ===\n");

    // Create thread pool for parallel operations
    const config = threading.ThreadPoolConfig{
        .num_threads = 4,
        .numa_policy = .balanced,
        .work_stealing = true,
        .affinity_enabled = false,
    };

    var thread_pool = try threading.ThreadPool.init(allocator, config);
    defer thread_pool.deinit();

    try thread_pool.start();
    defer thread_pool.stop();

    // Create test matrices for multiplication
    const size = 128; // 128x128 matrices
    var a = try Tensor.random(allocator, &[_]u32{ size, size });
    defer a.deinit();
    var b = try Tensor.random(allocator, &[_]u32{ size, size });
    defer b.deinit();
    var c = try Tensor.zeros(allocator, &[_]u32{ size, size });
    defer c.deinit();

    std.debug.print("Matrix dimensions: {}x{} × {}x{} = {}x{}\n", .{ size, size, size, size, size, size });

    // Measure sequential vs parallel performance
    var timer = try std.time.Timer.start();

    // Sequential matrix multiplication (single-threaded)
    const sequential_start = timer.read();
    try sequentialMatmul(&a, &b, &c);
    const sequential_time = timer.read() - sequential_start;

    // Reset result matrix
    try c.fill(0.0);

    // Parallel matrix multiplication
    var parallel_ops = threading.ParallelOps.init(&thread_pool);
    const parallel_start = timer.read();
    try parallel_ops.matmul(&a, &b, &c);
    const parallel_time = timer.read() - parallel_start;

    // Display performance results
    const speedup = @as(f64, @floatFromInt(sequential_time)) / @as(f64, @floatFromInt(parallel_time));
    std.debug.print("Performance Results:\n");
    std.debug.print("• Sequential Time: {:.2} ms\n", .{@as(f64, @floatFromInt(sequential_time)) / 1_000_000.0});
    std.debug.print("• Parallel Time: {:.2} ms\n", .{@as(f64, @floatFromInt(parallel_time)) / 1_000_000.0});
    std.debug.print("• Speedup: {:.2}x\n", .{speedup});
    std.debug.print("• Efficiency: {:.1}% (speedup / thread count)\n", .{speedup / @as(f64, @floatFromInt(config.num_threads)) * 100.0});

    // Test parallel softmax
    std.debug.print("\nTesting parallel softmax operation...\n");
    var input = try Tensor.random(allocator, &[_]u32{ 64, 1000 }); // 64 sequences, 1000 vocab
    defer input.deinit();
    var output = try Tensor.zeros(allocator, &[_]u32{ 64, 1000 });
    defer output.deinit();

    const softmax_start = timer.read();
    try parallel_ops.softmax(&input, &output, 1);
    const softmax_time = timer.read() - softmax_start;

    std.debug.print("Softmax Results:\n");
    std.debug.print("• Input shape: {}x{}\n", .{ 64, 1000 });
    std.debug.print("• Processing time: {:.2} ms\n", .{@as(f64, @floatFromInt(softmax_time)) / 1_000_000.0});

    // Verify softmax properties on a few rows
    var valid_rows: u32 = 0;
    for (0..@min(10, output.shape[0])) |i| {
        var row_sum: f32 = 0.0;
        var all_positive = true;

        for (0..output.shape[1]) |j| {
            const val = try output.get(&[_]u32{ @intCast(i), @intCast(j) });
            row_sum += val;
            if (val < 0.0) {
                all_positive = false;
            }
        }

        if (all_positive and @abs(row_sum - 1.0) < 0.001) {
            valid_rows += 1;
        }
    }
    std.debug.print("• Validation: {}/10 rows sum to 1.0 (±0.001)\n\n", .{valid_rows});
}

fn sequentialMatmul(a: *const Tensor, b: *const Tensor, c: *Tensor) !void {
    for (0..c.shape[0]) |i| {
        for (0..c.shape[1]) |j| {
            var sum: f32 = 0.0;
            for (0..a.shape[1]) |k| {
                const a_val = try a.get(&[_]u32{ @intCast(i), @intCast(k) });
                const b_val = try b.get(&[_]u32{ @intCast(k), @intCast(j) });
                sum += a_val * b_val;
            }
            try c.set(&[_]u32{ @intCast(i), @intCast(j) }, sum);
        }
    }
}

fn demonstrateNumaAllocation(allocator: std.mem.Allocator) !void {
    std.debug.print("=== NUMA-Aware Memory Allocation ===\n");

    const topology = threading.CpuTopology.detect();
    std.debug.print("NUMA Configuration:\n");
    std.debug.print("• NUMA Nodes: {}\n", .{topology.num_numa_nodes});

    // Create NUMA-aware allocators with different policies
    const policies = [_]threading.NumaPolicy{ .local, .interleave, .balanced };
    const policy_names = [_][]const u8{ "local", "interleave", "balanced" };

    for (policies, policy_names) |policy, name| {
        std.debug.print("\n{s} NUMA Policy:\n", .{std.ascii.upperString(name, name)});

        var numa_allocator = threading.NumaAllocator.init(allocator, policy);

        // Demonstrate node assignment for different thread IDs
        std.debug.print("Thread-to-Node Mapping:\n");
        for (0..8) |thread_id| {
            const node = numa_allocator.getPreferredNode(@intCast(thread_id));
            std.debug.print("  Thread {}: Node {}\n", .{ thread_id, node });
        }

        // Test memory allocation with NUMA policy
        const numa_alloc = numa_allocator.allocator();
        const test_memory = try numa_alloc.alloc(f32, 1024);
        defer numa_alloc.free(test_memory);

        // Fill memory with test pattern
        for (test_memory, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt(i));
        }

        std.debug.print("  Allocated {} KB successfully\n", .{test_memory.len * @sizeOf(f32) / 1024});
    }

    // Estimate memory usage for different model configurations
    std.debug.print("\nMemory Usage Estimates:\n");
    const configs = [_]struct { name: []const u8, sequences: u32, seq_len: u32, layers: u32, heads: u32, head_dim: u32 }{
        .{ .name = "GPT-2 Small", .sequences = 1, .seq_len = 1024, .layers = 12, .heads = 12, .head_dim = 64 },
        .{ .name = "LLaMA 7B", .sequences = 1, .seq_len = 2048, .layers = 32, .heads = 32, .head_dim = 128 },
        .{ .name = "Batch (8 seq)", .sequences = 8, .seq_len = 512, .layers = 24, .heads = 16, .head_dim = 64 },
    };

    for (configs) |config| {
        const memory_usage = threading.KVCacheUtils.estimateMemoryUsage(
            config.sequences,
            config.seq_len,
            config.layers,
            config.heads,
            config.head_dim,
        );

        std.debug.print("• {s}: {:.1} MB\n", .{ config.name, @as(f64, @floatFromInt(memory_usage)) / 1_048_576.0 });
    }
    std.debug.print("");
}

fn demonstratePerformanceAnalysis(allocator: std.mem.Allocator) !void {
    std.debug.print("=== Performance Analysis ===\n");

    // Create thread pool with monitoring
    var stats = threading.ThreadPoolStats.init();

    const config = threading.ThreadPoolConfig{
        .num_threads = 4,
        .numa_policy = .balanced,
        .work_stealing = true,
        .affinity_enabled = false,
    };

    var thread_pool = try threading.ThreadPool.init(allocator, config);
    defer thread_pool.deinit();

    try thread_pool.start();
    defer thread_pool.stop();

    // Simulate workload with different characteristics
    std.debug.print("Simulating mixed workload...\n");

    var work_items: [50]threading.WorkItem = undefined;
    var work_data: [50]WorkData = undefined;

    // Create different types of work
    for (0..50) |i| {
        work_data[i] = WorkData{
            .work_type = @intCast(i % 3), // 0=light, 1=medium, 2=heavy
            .iterations = switch (i % 3) {
                0 => 100,   // Light work
                1 => 1000,  // Medium work
                2 => 5000,  // Heavy work
                else => 1000,
            },
            .result = 0,
        };

        work_items[i] = threading.WorkItem{
            .func = simulatedWork,
            .data = &work_data[i],
        };

        _ = thread_pool.submit(&work_items[i]);
        stats.recordTaskExecution();
    }

    // Monitor completion
    var completed: u32 = 0;
    while (completed < 50) {
        threading.ThreadingUtils.sleep(10); // 10ms

        completed = 0;
        for (&work_items) |*item| {
            if (item.isCompleted()) {
                completed += 1;
            }
        }

        if (completed % 10 == 0 and completed > 0) {
            std.debug.print("Progress: {}/50 tasks completed\n", .{completed});
        }
    }

    // Analyze results
    var light_work_total: u64 = 0;
    var medium_work_total: u64 = 0;
    var heavy_work_total: u64 = 0;

    for (work_data) |data| {
        switch (data.work_type) {
            0 => light_work_total += data.result,
            1 => medium_work_total += data.result,
            2 => heavy_work_total += data.result,
            else => {},
        }
    }

    std.debug.print("\nWorkload Analysis:\n");
    std.debug.print("• Light work results: {}\n", .{light_work_total});
    std.debug.print("• Medium work results: {}\n", .{medium_work_total});
    std.debug.print("• Heavy work results: {}\n", .{heavy_work_total});

    // Display final statistics
    stats.print();

    // Queue analysis
    const queue_sizes = thread_pool.getQueueSizes();
    defer allocator.free(queue_sizes);

    std.debug.print("Final queue states:\n");
    for (queue_sizes, 0..) |size, i| {
        std.debug.print("  Worker {}: {} items remaining\n", .{ i, size });
    }
    std.debug.print("");
}

const WorkData = struct {
    work_type: u32,
    iterations: u32,
    result: u64,
};

fn simulatedWork(work_item: *threading.WorkItem) void {
    const data = @as(*WorkData, @ptrCast(@alignCast(work_item.data.?)));

    var accumulator: u64 = 0;
    for (0..data.iterations) |i| {
        // Simulate different computational intensities
        switch (data.work_type) {
            0 => accumulator += i, // Light: simple addition
            1 => accumulator += i * i, // Medium: multiplication
            2 => accumulator += i * i * i, // Heavy: more operations
            else => accumulator += i,
        }
    }

    data.result = accumulator;

    // Add some variability to completion time
    const delay = (data.work_type + 1) * 1000; // Microseconds
    std.time.sleep(delay);
}

fn demonstrateWorkStealing(allocator: std.mem.Allocator) !void {
    std.debug.print("=== Work-Stealing Load Balancing ===\n");

    // Create thread pool
    const config = threading.ThreadPoolConfig{
        .num_threads = 4,
        .numa_policy = .disabled,
        .work_stealing = true,
        .affinity_enabled = false,
    };

    var thread_pool = try threading.ThreadPool.init(allocator, config);
    defer thread_pool.deinit();

    try thread_pool.start();
    defer thread_pool.stop();

    // Create imbalanced workload (submit all to worker 0)
    std.debug.print("Creating imbalanced workload...\n");

    var work_items: [20]threading.WorkItem = undefined;
    var counters: [20]std.atomic.Atomic(u32) = undefined;

    for (&counters) |*counter| {
        counter.* = std.atomic.Atomic(u32).init(0);
    }

    // Submit all work to worker 0 to create imbalance
    for (0..20) |i| {
        work_items[i] = threading.WorkItem{
            .func = atomicWork,
            .data = &counters[i],
        };

        // Force submission to worker 0
        _ = thread_pool.submitToWorker(0, &work_items[i]);
    }

    // Monitor queue sizes during execution
    std.debug.print("Monitoring work distribution:\n");

    var snapshots: [5][4]u32 = undefined;
    for (0..5) |snapshot| {
        threading.ThreadingUtils.sleep(20); // 20ms intervals

        const queue_sizes = thread_pool.getQueueSizes();
        defer allocator.free(queue_sizes);

        for (queue_sizes, 0..) |size, worker| {
            snapshots[snapshot][worker] = size;
        }

        std.debug.print("Snapshot {}: ", .{snapshot + 1});
        for (queue_sizes, 0..) |size, worker| {
            std.debug.print("W{}={} ", .{ worker, size });
        }
        std.debug.print("\n");
    }

    // Wait for completion
    for (&work_items) |*item| {
        item.waitForCompletion();
    }

    // Analyze work stealing effectiveness
    std.debug.print("\nWork Stealing Analysis:\n");
    std.debug.print("Initial distribution (all work on Worker 0):\n");
    for (0..4) |worker| {
        std.debug.print("  Worker {}: {} -> {} tasks\n", .{ worker, snapshots[0][worker], snapshots[4][worker] });
    }

    // Calculate load balancing improvement
    const initial_max = std.mem.max(u32, &snapshots[0]);
    const initial_min = std.mem.min(u32, &snapshots[0]);
    const final_max = std.mem.max(u32, &snapshots[4]);
    const final_min = std.mem.min(u32, &snapshots[4]);

    const initial_imbalance = if (initial_min == 0) initial_max else @as(f32, @floatFromInt(initial_max)) / @as(f32, @floatFromInt(initial_min));
    const final_imbalance = if (final_min == 0) final_max else @as(f32, @floatFromInt(final_max)) / @as(f32, @floatFromInt(final_min));

    std.debug.print("Load Balance Improvement:\n");
    std.debug.print("• Initial imbalance ratio: {:.1}\n", .{initial_imbalance});
    std.debug.print("• Final imbalance ratio: {:.1}\n", .{final_imbalance});

    if (final_imbalance < initial_imbalance) {
        const improvement = (initial_imbalance - final_imbalance) / initial_imbalance * 100.0;
        std.debug.print("• Work stealing improved balance by {:.1}%\n", .{improvement});
    }

    // Verify all work completed correctly
    var total_work: u32 = 0;
    for (counters) |*counter| {
        total_work += counter.load(.Acquire);
    }
    std.debug.print("• Total work completed: {} (expected: {})\n", .{ total_work, 20 * 10000 });
}

fn atomicWork(work_item: *threading.WorkItem) void {
    const counter = @as(*std.atomic.Atomic(u32), @ptrCast(@alignCast(work_item.data.?)));

    // Simulate work by incrementing counter many times
    for (0..10000) |_| {
        _ = counter.fetchAdd(1, .AcqRel);
    }

    // Add some processing time
    threading.ThreadingUtils.sleep(50); // 50ms of "work"
};