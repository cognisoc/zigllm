// Threading and NUMA Optimization Tests
// Comprehensive tests for parallel computation infrastructure

const std = @import("std");
const testing = std.testing;
const Tensor = @import("../src/foundation/tensor.zig").Tensor;
const threading = @import("../src/foundation/threading.zig");

test "CPU topology detection" {
    const topology = threading.CpuTopology.detect();

    // Basic sanity checks
    try testing.expect(topology.num_cores > 0);
    try testing.expect(topology.num_threads > 0);
    try testing.expect(topology.num_numa_nodes > 0);
    try testing.expect(topology.cache_line_size > 0);
    try testing.expect(topology.l1_cache_size > 0);

    std.debug.print("Detected CPU topology:\n");
    topology.print();
}

test "thread pool configuration" {
    const config = threading.ThreadPoolConfig.detect();

    try testing.expect(config.num_threads > 0);
    try testing.expect(config.stack_size > 0);

    std.debug.print("Thread pool config: {} threads\n", .{config.num_threads});
}

test "work stealing queue operations" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var queue = try threading.WorkStealingQueue.init(allocator, 16);
    defer queue.deinit();

    // Test initial state
    try testing.expect(queue.size() == 0);
    try testing.expect(queue.pop() == null);
    try testing.expect(queue.steal() == null);

    // Create test work items
    var work_item1 = threading.WorkItem{
        .func = testWorkFunction,
        .data = null,
    };

    var work_item2 = threading.WorkItem{
        .func = testWorkFunction,
        .data = null,
    };

    // Test push operations
    try testing.expect(queue.push(&work_item1));
    try testing.expect(queue.size() == 1);

    try testing.expect(queue.push(&work_item2));
    try testing.expect(queue.size() == 2);

    // Test pop operations (LIFO)
    const popped1 = queue.pop();
    try testing.expect(popped1 != null);
    try testing.expect(popped1.? == &work_item2);
    try testing.expect(queue.size() == 1);

    // Test steal operations (FIFO)
    const stolen1 = queue.steal();
    try testing.expect(stolen1 != null);
    try testing.expect(stolen1.? == &work_item1);
    try testing.expect(queue.size() == 0);
}

fn testWorkFunction(work_item: *threading.WorkItem) void {
    _ = work_item;
    // Simple test work - just mark as completed
}

test "work item execution" {
    var work_item = threading.WorkItem{
        .func = incrementCounter,
        .data = &counter,
    };

    counter = 0;
    try testing.expect(!work_item.isCompleted());

    work_item.execute();
    try testing.expect(work_item.isCompleted());
    try testing.expect(counter == 1);

    work_item.waitForCompletion(); // Should return immediately
}

var counter: u32 = 0;

fn incrementCounter(work_item: *threading.WorkItem) void {
    _ = work_item;
    counter += 1;
}

test "thread pool initialization and cleanup" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = threading.ThreadPoolConfig{
        .num_threads = 2,
        .numa_policy = .disabled,
        .work_stealing = true,
        .affinity_enabled = false,
    };

    var thread_pool = try threading.ThreadPool.init(allocator, config);
    defer thread_pool.deinit();

    try testing.expect(thread_pool.getWorkerCount() == 2);

    // Test basic operations without starting threads
    var work_item = threading.WorkItem{
        .func = testWorkFunction,
        .data = null,
    };

    const submitted = thread_pool.submit(&work_item);
    try testing.expect(submitted);

    const queue_sizes = thread_pool.getQueueSizes();
    try testing.expect(queue_sizes.len == 2);
    defer allocator.free(queue_sizes);

    var total_queued: u32 = 0;
    for (queue_sizes) |size| {
        total_queued += size;
    }
    try testing.expect(total_queued == 1);
}

test "parallel matrix multiplication" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create small test matrices
    var a = try Tensor.zeros(allocator, &[_]u32{ 4, 3 });
    defer a.deinit();
    var b = try Tensor.zeros(allocator, &[_]u32{ 3, 2 });
    defer b.deinit();
    var c = try Tensor.zeros(allocator, &[_]u32{ 4, 2 });
    defer c.deinit();

    // Fill matrices with test data
    for (0..4) |i| {
        for (0..3) |j| {
            try a.set(&[_]u32{ @intCast(i), @intCast(j) }, @as(f32, @floatFromInt(i + j)));
        }
    }

    for (0..3) |i| {
        for (0..2) |j| {
            try b.set(&[_]u32{ @intCast(i), @intCast(j) }, @as(f32, @floatFromInt(i * j + 1)));
        }
    }

    // Create thread pool
    const config = threading.ThreadPoolConfig{
        .num_threads = 2,
        .numa_policy = .disabled,
        .work_stealing = true,
        .affinity_enabled = false,
    };

    var thread_pool = try threading.ThreadPool.init(allocator, config);
    defer thread_pool.deinit();

    try thread_pool.start();
    defer thread_pool.stop();

    // Perform parallel matrix multiplication
    var parallel_ops = threading.ParallelOps.init(&thread_pool);
    try parallel_ops.matmul(&a, &b, &c);

    // Verify result (spot check a few elements)
    const c00 = try c.get(&[_]u32{ 0, 0 });
    const c01 = try c.get(&[_]u32{ 0, 1 });

    // Expected: C[0,0] = A[0,:] · B[:,0] = [0,1,2] · [1,1,1] = 4
    // Expected: C[0,1] = A[0,:] · B[:,1] = [0,1,2] · [1,2,3] = 8
    try testing.expectApproxEqAbs(@as(f32, 4.0), c00, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 8.0), c01, 0.001);
}

test "parallel softmax" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create test input
    var input = try Tensor.zeros(allocator, &[_]u32{ 3, 4 });
    defer input.deinit();
    var output = try Tensor.zeros(allocator, &[_]u32{ 3, 4 });
    defer output.deinit();

    // Fill with test data
    for (0..3) |i| {
        for (0..4) |j| {
            try input.set(&[_]u32{ @intCast(i), @intCast(j) }, @as(f32, @floatFromInt(j)));
        }
    }

    // Create thread pool
    const config = threading.ThreadPoolConfig{
        .num_threads = 2,
        .numa_policy = .disabled,
        .work_stealing = true,
        .affinity_enabled = false,
    };

    var thread_pool = try threading.ThreadPool.init(allocator, config);
    defer thread_pool.deinit();

    try thread_pool.start();
    defer thread_pool.stop();

    // Perform parallel softmax
    var parallel_ops = threading.ParallelOps.init(&thread_pool);
    try parallel_ops.softmax(&input, &output, 1);

    // Verify softmax properties
    for (0..3) |i| {
        var row_sum: f32 = 0.0;
        for (0..4) |j| {
            const val = try output.get(&[_]u32{ @intCast(i), @intCast(j) });
            try testing.expect(val >= 0.0);
            try testing.expect(val <= 1.0);
            row_sum += val;
        }
        // Each row should sum to approximately 1.0
        try testing.expectApproxEqAbs(@as(f32, 1.0), row_sum, 0.001);
    }
}

test "NUMA allocator initialization" {
    var base_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = base_allocator.deinit();

    var numa_allocator = threading.NumaAllocator.init(base_allocator.allocator(), .balanced);
    const topology = threading.CpuTopology.detect();

    // Test preferred node calculation
    for (0..8) |i| {
        const node = numa_allocator.getPreferredNode(@intCast(i));
        try testing.expect(node < topology.num_numa_nodes);
    }

    // Test allocator interface
    const allocator = numa_allocator.allocator();
    const memory = try allocator.alloc(u8, 1024);
    defer allocator.free(memory);

    try testing.expect(memory.len == 1024);
}

test "threading utilities" {
    const thread_id = threading.ThreadingUtils.getCurrentThreadId();
    try testing.expect(thread_id > 0);

    const cache_line_size = threading.ThreadingUtils.getCacheLineSize();
    try testing.expect(cache_line_size == 64);

    // Test cache-aligned allocation
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const aligned_memory = try threading.ThreadingUtils.allocAligned(allocator, f32, 100);
    try testing.expect(aligned_memory.len == 100);

    // Verify alignment (address should be cache-line aligned)
    const addr = @intFromPtr(aligned_memory.ptr);
    try testing.expect(addr % cache_line_size == 0);
}

test "thread pool statistics" {
    var stats = threading.ThreadPoolStats.init();

    try testing.expect(stats.total_tasks_executed.load(.Acquire) == 0);
    try testing.expect(stats.total_work_stolen.load(.Acquire) == 0);

    stats.recordTaskExecution();
    stats.recordWorkSteal();

    try testing.expect(stats.total_tasks_executed.load(.Acquire) == 1);
    try testing.expect(stats.total_work_stolen.load(.Acquire) == 1);

    std.debug.print("Thread pool stats:\n");
    stats.print();
}

test "NUMA policy string parsing" {
    try testing.expect(threading.NumaPolicy.fromString("local") == .local);
    try testing.expect(threading.NumaPolicy.fromString("interleave") == .interleave);
    try testing.expect(threading.NumaPolicy.fromString("balanced") == .balanced);
    try testing.expect(threading.NumaPolicy.fromString("unknown") == .disabled);
}

test "work stealing between queues" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var queue1 = try threading.WorkStealingQueue.init(allocator, 16);
    defer queue1.deinit();
    var queue2 = try threading.WorkStealingQueue.init(allocator, 16);
    defer queue2.deinit();

    var work_item = threading.WorkItem{
        .func = testWorkFunction,
        .data = null,
    };

    // Add work to queue1
    try testing.expect(queue1.push(&work_item));
    try testing.expect(queue1.size() == 1);
    try testing.expect(queue2.size() == 0);

    // Steal from queue1 to queue2
    const stolen_item = queue1.steal();
    try testing.expect(stolen_item != null);
    try testing.expect(stolen_item.? == &work_item);
    try testing.expect(queue1.size() == 0);

    // Put stolen work in queue2
    try testing.expect(queue2.push(stolen_item.?));
    try testing.expect(queue2.size() == 1);
}

test "thread pool load balancing" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = threading.ThreadPoolConfig{
        .num_threads = 4,
        .numa_policy = .disabled,
        .work_stealing = true,
        .affinity_enabled = false,
    };

    var thread_pool = try threading.ThreadPool.init(allocator, config);
    defer thread_pool.deinit();

    // Submit multiple work items
    var work_items: [10]threading.WorkItem = undefined;
    for (&work_items) |*item| {
        item.* = threading.WorkItem{
            .func = testWorkFunction,
            .data = null,
        };
        _ = thread_pool.submit(item);
    }

    // Check queue distribution
    const queue_sizes = thread_pool.getQueueSizes();
    defer allocator.free(queue_sizes);

    var total_queued: u32 = 0;
    for (queue_sizes) |size| {
        total_queued += size;
    }
    try testing.expect(total_queued == 10);

    // Work should be distributed across workers (round-robin)
    var non_empty_queues: u32 = 0;
    for (queue_sizes) |size| {
        if (size > 0) {
            non_empty_queues += 1;
        }
    }
    try testing.expect(non_empty_queues >= 2); // At least 2 workers should have work
}

test "memory alignment verification" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test different data types and sizes
    const aligned_f32 = try threading.ThreadingUtils.allocAligned(allocator, f32, 64);
    const aligned_u64 = try threading.ThreadingUtils.allocAligned(allocator, u64, 32);

    const cache_line_size = threading.ThreadingUtils.getCacheLineSize();

    const f32_addr = @intFromPtr(aligned_f32.ptr);
    const u64_addr = @intFromPtr(aligned_u64.ptr);

    try testing.expect(f32_addr % cache_line_size == 0);
    try testing.expect(u64_addr % cache_line_size == 0);

    // Test that we can actually use the memory
    aligned_f32[0] = 3.14159;
    aligned_f32[63] = 2.71828;
    try testing.expectApproxEqAbs(@as(f32, 3.14159), aligned_f32[0], 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 2.71828), aligned_f32[63], 0.0001);

    aligned_u64[0] = 12345;
    aligned_u64[31] = 67890;
    try testing.expect(aligned_u64[0] == 12345);
    try testing.expect(aligned_u64[31] == 67890);
};