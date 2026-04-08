# foundation.memory_mapping

## Module Path

```
zigllama.foundation.memory_mapping
```

**Source file:** `src/foundation/memory_mapping.zig`

> **Internal module.** This API may change between releases.

---

## Public Types

### `Protection`

```zig
pub const Protection = enum {
    read,
    write,
    exec,
};
```

Memory protection flags passed to the underlying `mmap` / `MapViewOfFile` call.

### `MemoryMap`

```zig
pub const MemoryMap = struct {
    data: [*]align(std.mem.page_size) u8,
    len: usize,
    protection: Protection,
    flags: u32,
};
```

Handle to a memory-mapped region. Owns the mapping and must be released with
`unmap`.

| Field | Type | Description |
|-------|------|-------------|
| `data` | `[*]align(page_size) u8` | Pointer to the mapped region |
| `len` | `usize` | Size in bytes |
| `protection` | `Protection` | Active protection mode |
| `flags` | `u32` | Platform-specific mapping flags |

### `ModelFileMapper`

Convenience wrapper that opens a model file, maps the entire contents, and
exposes typed accessors.

### `GGUFMapper`

Specialization of `ModelFileMapper` that understands the GGUF header layout and
can jump directly to tensor data offsets.

### `MappingOptimizer`

Applies platform hints (e.g., `madvise(MADV_SEQUENTIAL)`) to improve read-ahead
performance for large sequential scans.

---

## Public Functions

### `MemoryMap.fromFile`

```zig
pub fn fromFile(
    path: []const u8,
    protection: Protection,
    flags: u32,
) !MemoryMap
```

Map the entire contents of the file at `path` into the process address space.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `path` | `[]const u8` | Filesystem path to the file |
| `protection` | `Protection` | Desired access mode |
| `flags` | `u32` | Platform mapping flags (0 for defaults) |

**Returns:** `!MemoryMap` or an OS-level error.

### `MemoryMap.lock`

```zig
pub fn lock(self: *MemoryMap) !void
```

Pin the mapped pages in physical memory (`mlock`). Prevents the OS from paging
the region to disk, which is important for latency-sensitive inference.

### `MemoryMap.unlock`

```zig
pub fn unlock(self: *MemoryMap) !void
```

Release the physical-memory pin (`munlock`).

### `MemoryMap.unmap`

```zig
pub fn unmap(self: *MemoryMap) void
```

Release the memory mapping. The `MemoryMap` must not be used after this call.

### `MemoryMap.createTensor`

```zig
pub fn createTensor(
    self: *MemoryMap,
    comptime T: type,
    offset: usize,
    shape: []const usize,
) !Tensor(T)
```

Construct a `Tensor(T)` whose data points into the mapped region at `offset`
bytes. The tensor does **not** own the memory -- the caller must keep the mapping
alive for the lifetime of the tensor.

---

## Error Types

Functions return OS-level errors (`std.posix.MMapError`, `std.posix.OpenError`)
or `TensorError.InvalidShape`.

---

## Usage Example

```zig
const mm = @import("zigllama").foundation.memory_mapping;

var mapping = try mm.MemoryMap.fromFile("model.gguf", .read, 0);
defer mapping.unmap();

// Create a tensor view into the mapped region
var weights = try mapping.createTensor(f32, header_offset, &[_]usize{ 4096, 4096 });
// weights.data points directly into the file -- zero-copy
```

---

## Related Modules

- [`foundation.tensor`](tensor.md) -- The `Tensor` type returned by
  `createTensor`.
- [`foundation.gguf_format`](gguf-format.md) -- Constants needed to compute
  offsets within GGUF files.
- [`models.gguf`](gguf.md) -- High-level GGUF reader built on top of this
  module.
