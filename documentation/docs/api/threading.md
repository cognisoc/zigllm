# foundation.threading

## Module Path

```
zigllama.foundation.threading
```

**Source file:** `src/foundation/threading.zig`

> **Internal module.** This API may change between releases.

---

## Public Types

### `ThreadPoolConfig`

```zig
pub const ThreadPoolConfig = struct {
    num_threads: ?usize = null,     // null = auto-detect CPU count
    stack_size: usize = 8 * 1024 * 1024,
    enable_work_stealing: bool = true,
    numa_aware: bool = false,
    pin_threads: bool = false,
};
```

Configuration for the thread pool. When `num_threads` is `null`, the pool sizes
itself to the number of available hardware threads.

### `ThreadPool`

```zig
pub const ThreadPool = struct {
    workers: []Worker,
    config: ThreadPoolConfig,
    running: std.atomic.Value(bool),
};
```

Fixed-size pool of worker threads with optional work-stealing scheduling.

### `Worker`

Internal per-thread state. Each worker owns a local work queue and can steal
from siblings.

### `WorkStealingQueue`

Lock-free deque used by each `Worker` for local task storage. Other workers can
steal from the tail when their own queues are empty.

### `WorkItem`

```zig
pub const WorkItem = struct {
    func: *const fn (*anyopaque) void,
    context: *anyopaque,
    done: std.atomic.Value(bool),
};
```

A unit of work submitted to the pool.

### `ParallelOps`

```zig
pub const ParallelOps = struct {
    pool: *ThreadPool,
};
```

High-level parallel primitives that partition tensor operations across the pool.

### `NumaAllocator`

Allocator that binds allocations to a specific NUMA node. Falls back to the
system allocator on non-NUMA hardware.

### `NumaPolicy`

```zig
pub const NumaPolicy = enum {
    Interleave,
    Local,
    Preferred,
};
```

Memory placement policy for NUMA-aware allocations.

### `CpuTopology`

```zig
pub const CpuTopology = struct {
    num_cores: usize,
    num_threads: usize,
    num_numa_nodes: usize,
    cache_line_size: usize,
};
```

Hardware topology detected at runtime via `/sys/devices` or `cpuid`.

---

## Public Functions

### `ThreadPool.init`

```zig
pub fn init(config: ThreadPoolConfig) !ThreadPool
```

Create and start the thread pool. Workers begin in an idle spin-wait state.

### `ThreadPool.deinit`

```zig
pub fn deinit(self: *ThreadPool) void
```

Signal all workers to exit and join their threads.

### `ThreadPool.submit`

```zig
pub fn submit(self: *ThreadPool, func: *const fn (*anyopaque) void, context: *anyopaque) !*WorkItem
```

Enqueue a work item. Returns a handle whose `done` field can be polled or
awaited.

### `ParallelOps.matmul`

```zig
pub fn matmul(self: ParallelOps, a: Tensor(f32), b: Tensor(f32), allocator: Allocator) !Tensor(f32)
```

Parallel matrix multiplication that tiles rows of `a` across workers.

### `ParallelOps.softmax`

```zig
pub fn softmax(self: ParallelOps, input: Tensor(f32), allocator: Allocator) !Tensor(f32)
```

Row-parallel softmax. Each worker processes a contiguous slice of rows.

---

## Error Types

- `ThreadPool.init` can return `error{SystemResources, OutOfMemory}`.
- `submit` returns `error{QueueFull}` when the work-stealing queue is at
  capacity.

---

## Usage Example

```zig
const threading = @import("zigllama").foundation.threading;

var pool = try threading.ThreadPool.init(.{
    .num_threads = 4,
    .enable_work_stealing = true,
});
defer pool.deinit();

const ops = threading.ParallelOps{ .pool = &pool };
var result = try ops.matmul(weights, input, allocator);
defer result.deinit();
```

---

## Related Modules

- [`foundation.blas_integration`](blas-integration.md) -- BLAS backends that
  may use their own threading.
- [`linear_algebra.matrix_ops`](matrix-ops.md) -- SIMD matrix ops that can
  leverage `ParallelOps`.
