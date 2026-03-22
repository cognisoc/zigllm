// CPU Threading and NUMA Optimization
// Advanced parallel computation support for transformer inference
//
// Key features:
// 1. Thread pool management for parallel operations
// 2. NUMA-aware memory allocation and computation
// 3. Work-stealing scheduling for load balancing
// 4. CPU topology detection and optimization
// 5. Parallel matrix operations and attention computation

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

// Thread Pool Configuration
pub const ThreadPoolConfig = struct {
    num_threads: u32,
    numa_policy: NumaPolicy = .balanced,
    work_stealing: bool = true,
    affinity_enabled: bool = true,
    stack_size: usize = 2 * 1024 * 1024, // 2MB per thread

    pub fn detect() ThreadPoolConfig {
        const cpu_count = @as(u32, @intCast(std.Thread.getCpuCount() catch 4));
        return ThreadPoolConfig{
            .num_threads = @max(1, cpu_count - 1), // Leave one core for OS
        };
    }
};

// NUMA Memory Policy
pub const NumaPolicy = enum {
    local,      // Prefer local NUMA node
    interleave, // Interleave across all NUMA nodes
    balanced,   // Automatic balancing
    disabled,   // No NUMA awareness

    pub fn fromString(str: []const u8) NumaPolicy {
        if (std.mem.eql(u8, str, "local")) return .local;
        if (std.mem.eql(u8, str, "interleave")) return .interleave;
        if (std.mem.eql(u8, str, "balanced")) return .balanced;
        return .disabled;
    }
};

// CPU Topology Information
pub const CpuTopology = struct {
    num_cores: u32,
    num_threads: u32,
    num_numa_nodes: u32,
    cache_line_size: u32,
    l1_cache_size: u64,
    l2_cache_size: u64,
    l3_cache_size: u64,

    pub fn detect() CpuTopology {
        // Simplified detection - in production, would query /proc/cpuinfo, CPUID, etc.
        const cpu_count = @as(u32, @intCast(std.Thread.getCpuCount() catch 4));

        return CpuTopology{
            .num_cores = cpu_count,
            .num_threads = cpu_count, // Assuming no hyperthreading for simplicity
            .num_numa_nodes = @max(1, cpu_count / 8), // Rough estimate
            .cache_line_size = 64,
            .l1_cache_size = 32 * 1024,   // 32KB
            .l2_cache_size = 256 * 1024,  // 256KB
            .l3_cache_size = 8 * 1024 * 1024, // 8MB
        };
    }

    pub fn print(self: CpuTopology) void {
        std.debug.print("=== CPU Topology ===\n");
        std.debug.print("Cores: {}\n", .{self.num_cores});
        std.debug.print("Threads: {}\n", .{self.num_threads});
        std.debug.print("NUMA Nodes: {}\n", .{self.num_numa_nodes});
        std.debug.print("Cache Line Size: {} bytes\n", .{self.cache_line_size});
        std.debug.print("L1 Cache: {} KB\n", .{self.l1_cache_size / 1024});
        std.debug.print("L2 Cache: {} KB\n", .{self.l2_cache_size / 1024});
        std.debug.print("L3 Cache: {} KB\n", .{self.l3_cache_size / 1024});
        std.debug.print("====================\n");
    }
};

// Work Item for Thread Pool
pub const WorkItem = struct {
    func: *const fn (*WorkItem) void,
    data: ?*anyopaque = null,
    completed: std.atomic.Atomic(bool) = std.atomic.Atomic(bool).init(false),

    pub fn execute(self: *WorkItem) void {
        self.func(self);
        self.completed.store(true, .Release);
    }

    pub fn isCompleted(self: *const WorkItem) bool {
        return self.completed.load(.Acquire);
    }

    pub fn waitForCompletion(self: *const WorkItem) void {
        while (!self.isCompleted()) {
            std.Thread.yield() catch {};
        }
    }
};

// Work-Stealing Queue
pub const WorkStealingQueue = struct {
    items: []?*WorkItem,
    head: std.atomic.Atomic(u32),
    tail: std.atomic.Atomic(u32),
    capacity: u32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, capacity: u32) !WorkStealingQueue {
        const items = try allocator.alloc(?*WorkItem, capacity);
        @memset(items, null);

        return WorkStealingQueue{
            .items = items,
            .head = std.atomic.Atomic(u32).init(0),
            .tail = std.atomic.Atomic(u32).init(0),
            .capacity = capacity,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *WorkStealingQueue) void {
        self.allocator.free(self.items);
    }

    pub fn push(self: *WorkStealingQueue, item: *WorkItem) bool {
        const tail = self.tail.load(.Acquire);
        const next_tail = (tail + 1) % self.capacity;

        if (next_tail == self.head.load(.Acquire)) {
            return false; // Queue full
        }

        self.items[tail] = item;
        self.tail.store(next_tail, .Release);
        return true;
    }

    pub fn pop(self: *WorkStealingQueue) ?*WorkItem {
        const tail = self.tail.load(.Acquire);
        if (tail == self.head.load(.Acquire)) {
            return null; // Queue empty
        }

        const prev_tail = if (tail == 0) self.capacity - 1 else tail - 1;
        const item = self.items[prev_tail];
        self.items[prev_tail] = null;
        self.tail.store(prev_tail, .Release);
        return item;
    }

    pub fn steal(self: *WorkStealingQueue) ?*WorkItem {
        const head = self.head.load(.Acquire);
        if (head == self.tail.load(.Acquire)) {
            return null; // Queue empty
        }

        const item = self.items[head];
        if (item == null) {
            return null;
        }

        const next_head = (head + 1) % self.capacity;
        if (self.head.compareAndSwap(head, next_head, .AcqRel, .Monotonic) == null) {
            return item;
        }

        return null; // Someone else got it
    }

    pub fn size(self: *const WorkStealingQueue) u32 {
        const tail = self.tail.load(.Acquire);
        const head = self.head.load(.Acquire);
        return if (tail >= head) tail - head else self.capacity - head + tail;
    }
};

// Thread Pool Worker
pub const Worker = struct {
    id: u32,
    thread: std.Thread,
    queue: WorkStealingQueue,
    pool: *ThreadPool,
    should_stop: std.atomic.Atomic(bool),

    pub fn init(allocator: std.mem.Allocator, id: u32, pool: *ThreadPool) !Worker {
        return Worker{
            .id = id,
            .thread = undefined,
            .queue = try WorkStealingQueue.init(allocator, 256),
            .pool = pool,
            .should_stop = std.atomic.Atomic(bool).init(false),
        };
    }

    pub fn deinit(self: *Worker) void {
        self.queue.deinit();
    }

    pub fn start(self: *Worker) !void {
        self.thread = try std.Thread.spawn(.{}, workerLoop, .{self});
    }

    pub fn stop(self: *Worker) void {
        self.should_stop.store(true, .Release);
        self.thread.join();
    }

    fn workerLoop(self: *Worker) void {
        while (!self.should_stop.load(.Acquire)) {
            // Try to get work from own queue first
            var work_item = self.queue.pop();

            // If no work, try to steal from other workers
            if (work_item == null) {
                work_item = self.stealWork();
            }

            if (work_item) |item| {
                item.execute();
            } else {
                // No work available, yield to OS
                std.Thread.yield() catch {};
            }
        }
    }

    fn stealWork(self: *Worker) ?*WorkItem {
        // Try to steal work from other workers in round-robin fashion
        for (0..self.pool.workers.len) |i| {
            const target_id = (self.id + 1 + i) % @as(u32, @intCast(self.pool.workers.len));
            if (target_id == self.id) continue;

            if (self.pool.workers[target_id].queue.steal()) |item| {
                return item;
            }
        }
        return null;
    }
};

// Main Thread Pool
pub const ThreadPool = struct {
    workers: []Worker,
    config: ThreadPoolConfig,
    topology: CpuTopology,
    allocator: std.mem.Allocator,
    next_worker: std.atomic.Atomic(u32),

    pub fn init(allocator: std.mem.Allocator, config: ThreadPoolConfig) !ThreadPool {
        const topology = CpuTopology.detect();

        var workers = try allocator.alloc(Worker, config.num_threads);
        errdefer allocator.free(workers);

        var pool = ThreadPool{
            .workers = workers,
            .config = config,
            .topology = topology,
            .allocator = allocator,
            .next_worker = std.atomic.Atomic(u32).init(0),
        };

        // Initialize workers
        for (workers, 0..) |*worker, i| {
            worker.* = try Worker.init(allocator, @intCast(i), &pool);
        }

        return pool;
    }

    pub fn deinit(self: *ThreadPool) void {
        self.stop();
        for (self.workers) |*worker| {
            worker.deinit();
        }
        self.allocator.free(self.workers);
    }

    pub fn start(self: *ThreadPool) !void {
        for (self.workers) |*worker| {
            try worker.start();
        }

        std.debug.print("Thread pool started with {} workers\n", .{self.workers.len});
        if (self.config.numa_policy != .disabled) {
            std.debug.print("NUMA policy: {}\n", .{self.config.numa_policy});
        }
    }

    pub fn stop(self: *ThreadPool) void {
        for (self.workers) |*worker| {
            worker.stop();
        }
        std.debug.print("Thread pool stopped\n");
    }

    pub fn submit(self: *ThreadPool, work_item: *WorkItem) bool {
        // Round-robin assignment to worker queues
        const worker_id = self.next_worker.fetchAdd(1, .AcqRel) % @as(u32, @intCast(self.workers.len));
        return self.workers[worker_id].queue.push(work_item);
    }

    pub fn submitToWorker(self: *ThreadPool, worker_id: u32, work_item: *WorkItem) bool {
        if (worker_id >= self.workers.len) return false;
        return self.workers[worker_id].queue.push(work_item);
    }

    pub fn getWorkerCount(self: *const ThreadPool) u32 {
        return @intCast(self.workers.len);
    }

    pub fn getQueueSizes(self: *const ThreadPool) []u32 {
        var sizes = self.allocator.alloc(u32, self.workers.len) catch return &[_]u32{};
        for (self.workers, 0..) |*worker, i| {
            sizes[i] = worker.queue.size();
        }
        return sizes;
    }
};

// Parallel Matrix Operations
pub const ParallelOps = struct {
    thread_pool: *ThreadPool,

    pub fn init(thread_pool: *ThreadPool) ParallelOps {
        return ParallelOps{ .thread_pool = thread_pool };
    }

    // Parallel Matrix Multiplication Context
    const MatMulContext = struct {
        a: *const Tensor,
        b: *const Tensor,
        c: *Tensor,
        start_row: u32,
        end_row: u32,
        allocator: std.mem.Allocator,

        fn execute(work_item: *WorkItem) void {
            const ctx = @as(*MatMulContext, @ptrCast(@alignCast(work_item.data.?)));

            // Perform matrix multiplication for assigned rows
            for (ctx.start_row..ctx.end_row) |i| {
                for (0..ctx.c.shape[1]) |j| {
                    var sum: f32 = 0.0;
                    for (0..ctx.a.shape[1]) |k| {
                        const a_val = ctx.a.get(&[_]u32{ @intCast(i), @intCast(k) }) catch 0.0;
                        const b_val = ctx.b.get(&[_]u32{ @intCast(k), @intCast(j) }) catch 0.0;
                        sum += a_val * b_val;
                    }
                    ctx.c.set(&[_]u32{ @intCast(i), @intCast(j) }, sum) catch {};
                }
            }
        }
    };

    pub fn matmul(self: *ParallelOps, a: *const Tensor, b: *const Tensor, c: *Tensor) !void {
        const num_workers = self.thread_pool.getWorkerCount();
        const rows_per_worker = @max(1, c.shape[0] / num_workers);

        var contexts = try self.thread_pool.allocator.alloc(MatMulContext, num_workers);
        defer self.thread_pool.allocator.free(contexts);

        var work_items = try self.thread_pool.allocator.alloc(WorkItem, num_workers);
        defer self.thread_pool.allocator.free(work_items);

        // Create work items
        for (0..num_workers) |i| {
            const start_row = @as(u32, @intCast(i * rows_per_worker));
            const end_row = @as(u32, @intCast(@min((i + 1) * rows_per_worker, c.shape[0])));

            if (start_row >= end_row) break;

            contexts[i] = MatMulContext{
                .a = a,
                .b = b,
                .c = c,
                .start_row = start_row,
                .end_row = end_row,
                .allocator = self.thread_pool.allocator,
            };

            work_items[i] = WorkItem{
                .func = MatMulContext.execute,
                .data = &contexts[i],
            };

            _ = self.thread_pool.submit(&work_items[i]);
        }

        // Wait for all work to complete
        for (work_items) |*item| {
            item.waitForCompletion();
        }
    }

    // Parallel Softmax Context
    const SoftmaxContext = struct {
        input: *const Tensor,
        output: *Tensor,
        start_row: u32,
        end_row: u32,

        fn execute(work_item: *WorkItem) void {
            const ctx = @as(*SoftmaxContext, @ptrCast(@alignCast(work_item.data.?)));

            for (ctx.start_row..ctx.end_row) |i| {
                // Find maximum for numerical stability
                var max_val: f32 = -std.math.inf(f32);
                for (0..ctx.input.shape[1]) |j| {
                    const val = ctx.input.get(&[_]u32{ @intCast(i), @intCast(j) }) catch 0.0;
                    max_val = @max(max_val, val);
                }

                // Compute exponentials and sum
                var sum: f32 = 0.0;
                for (0..ctx.input.shape[1]) |j| {
                    const val = ctx.input.get(&[_]u32{ @intCast(i), @intCast(j) }) catch 0.0;
                    const exp_val = @exp(val - max_val);
                    ctx.output.set(&[_]u32{ @intCast(i), @intCast(j) }, exp_val) catch {};
                    sum += exp_val;
                }

                // Normalize
                for (0..ctx.output.shape[1]) |j| {
                    const val = ctx.output.get(&[_]u32{ @intCast(i), @intCast(j) }) catch 0.0;
                    ctx.output.set(&[_]u32{ @intCast(i), @intCast(j) }, val / sum) catch {};
                }
            }
        }
    };

    pub fn softmax(self: *ParallelOps, input: *const Tensor, output: *Tensor, axis: u32) !void {
        if (axis != 1) {
            return error.UnsupportedAxis; // Only support row-wise softmax for now
        }

        const num_workers = self.thread_pool.getWorkerCount();
        const rows_per_worker = @max(1, input.shape[0] / num_workers);

        var contexts = try self.thread_pool.allocator.alloc(SoftmaxContext, num_workers);
        defer self.thread_pool.allocator.free(contexts);

        var work_items = try self.thread_pool.allocator.alloc(WorkItem, num_workers);
        defer self.thread_pool.allocator.free(work_items);

        for (0..num_workers) |i| {
            const start_row = @as(u32, @intCast(i * rows_per_worker));
            const end_row = @as(u32, @intCast(@min((i + 1) * rows_per_worker, input.shape[0])));

            if (start_row >= end_row) break;

            contexts[i] = SoftmaxContext{
                .input = input,
                .output = output,
                .start_row = start_row,
                .end_row = end_row,
            };

            work_items[i] = WorkItem{
                .func = SoftmaxContext.execute,
                .data = &contexts[i],
            };

            _ = self.thread_pool.submit(&work_items[i]);
        }

        for (work_items) |*item| {
            item.waitForCompletion();
        }
    }
};

// NUMA-Aware Memory Allocator
pub const NumaAllocator = struct {
    base_allocator: std.mem.Allocator,
    numa_policy: NumaPolicy,
    topology: CpuTopology,

    pub fn init(base_allocator: std.mem.Allocator, numa_policy: NumaPolicy) NumaAllocator {
        return NumaAllocator{
            .base_allocator = base_allocator,
            .numa_policy = numa_policy,
            .topology = CpuTopology.detect(),
        };
    }

    pub fn allocator(self: *NumaAllocator) std.mem.Allocator {
        return std.mem.Allocator{
            .ptr = self,
            .vtable = &.{
                .alloc = allocFn,
                .resize = resizeFn,
                .free = freeFn,
            },
        };
    }

    fn allocFn(ctx: *anyopaque, len: usize, log2_ptr_align: u8, ra: usize) ?[*]u8 {
        const self = @as(*NumaAllocator, @ptrCast(@alignCast(ctx)));

        // For now, just delegate to base allocator
        // In production, would use numa_alloc_local(), numa_alloc_interleaved(), etc.
        return self.base_allocator.rawAlloc(len, log2_ptr_align, ra);
    }

    fn resizeFn(ctx: *anyopaque, buf: []u8, log2_buf_align: u8, new_len: usize, ra: usize) bool {
        const self = @as(*NumaAllocator, @ptrCast(@alignCast(ctx)));
        return self.base_allocator.rawResize(buf, log2_buf_align, new_len, ra);
    }

    fn freeFn(ctx: *anyopaque, buf: []u8, log2_buf_align: u8, ra: usize) void {
        const self = @as(*NumaAllocator, @ptrCast(@alignCast(ctx)));
        self.base_allocator.rawFree(buf, log2_buf_align, ra);
    }

    pub fn getPreferredNode(self: *NumaAllocator, thread_id: u32) u32 {
        return switch (self.numa_policy) {
            .local => thread_id % self.topology.num_numa_nodes,
            .interleave => thread_id % self.topology.num_numa_nodes,
            .balanced => thread_id % self.topology.num_numa_nodes,
            .disabled => 0,
        };
    }
};

// Threading Utilities
pub const ThreadingUtils = struct {
    pub fn setThreadAffinity(thread_id: u32, core_id: u32) !void {
        // Platform-specific thread affinity setting
        // On Linux: pthread_setaffinity_np()
        // On Windows: SetThreadAffinityMask()
        // For now, this is a stub
        _ = thread_id;
        _ = core_id;
    }

    pub fn getCurrentThreadId() u32 {
        return @intCast(std.Thread.getCurrentId());
    }

    pub fn yield() void {
        std.Thread.yield() catch {};
    }

    pub fn sleep(ms: u64) void {
        std.time.sleep(ms * std.time.ns_per_ms);
    }

    pub fn getCacheLineSize() u32 {
        // Detect cache line size - typically 64 bytes on modern CPUs
        return 64;
    }

    // Cache-aligned memory allocation
    pub fn allocAligned(allocator: std.mem.Allocator, comptime T: type, count: usize) ![]T {
        const cache_line_size = getCacheLineSize();
        const alignment = @max(@alignOf(T), cache_line_size);

        // Allocate extra space for alignment
        const total_size = count * @sizeOf(T) + alignment;
        const raw_mem = try allocator.alloc(u8, total_size);

        // Calculate aligned pointer
        const addr = @intFromPtr(raw_mem.ptr);
        const aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
        const aligned_ptr = @as([*]T, @ptrFromInt(aligned_addr));

        return aligned_ptr[0..count];
    }
};

// Performance Monitoring
pub const ThreadPoolStats = struct {
    total_tasks_executed: std.atomic.Atomic(u64),
    total_work_stolen: std.atomic.Atomic(u64),
    average_queue_depth: f64,
    cpu_utilization: f64,

    pub fn init() ThreadPoolStats {
        return ThreadPoolStats{
            .total_tasks_executed = std.atomic.Atomic(u64).init(0),
            .total_work_stolen = std.atomic.Atomic(u64).init(0),
            .average_queue_depth = 0.0,
            .cpu_utilization = 0.0,
        };
    }

    pub fn recordTaskExecution(self: *ThreadPoolStats) void {
        _ = self.total_tasks_executed.fetchAdd(1, .AcqRel);
    }

    pub fn recordWorkSteal(self: *ThreadPoolStats) void {
        _ = self.total_work_stolen.fetchAdd(1, .AcqRel);
    }

    pub fn print(self: *const ThreadPoolStats) void {
        std.debug.print("=== Thread Pool Statistics ===\n");
        std.debug.print("Total Tasks: {}\n", .{self.total_tasks_executed.load(.Acquire)});
        std.debug.print("Work Stolen: {}\n", .{self.total_work_stolen.load(.Acquire)});
        std.debug.print("Avg Queue Depth: {:.2}\n", .{self.average_queue_depth});
        std.debug.print("CPU Utilization: {:.1}%\n", .{self.cpu_utilization * 100.0});
        std.debug.print("===============================\n");
    }
};