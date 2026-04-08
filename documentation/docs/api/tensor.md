# foundation.tensor

## Module Path

```
zigllama.foundation.tensor
```

**Source file:** `src/foundation/tensor.zig`

---

## Public Types

### `Shape`

```zig
pub const Shape = []const usize;
```

Alias for a slice of dimension sizes. A shape `[2, 3, 4]` describes a tensor
with 2 x 3 x 4 = 24 elements.

### `TensorError`

```zig
pub const TensorError = error{
    InvalidShape,
    IncompatibleShapes,
    InvalidIndex,
    OutOfMemory,
};
```

Error set returned by tensor operations.

| Variant | Meaning |
|---------|---------|
| `InvalidShape` | Requested shape has zero or negative dimensions |
| `IncompatibleShapes` | Operand shapes do not match for the operation |
| `InvalidIndex` | Index is out of bounds for the tensor shape |
| `OutOfMemory` | Allocator could not satisfy the request |

### `Tensor(T)`

```zig
pub fn Tensor(comptime T: type) type { ... }
```

Generic multi-dimensional tensor parameterized over element type `T`. Uses
row-major (C-style) memory layout.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `data` | `[]T` | Contiguous backing storage |
| `shape` | `[]usize` | Dimension sizes |
| `strides` | `[]usize` | Element offsets per dimension |
| `size` | `usize` | Total number of elements |
| `allocator` | `std.mem.Allocator` | Allocator used for this tensor |

---

## Public Functions

### `Tensor(T).init`

```zig
pub fn init(allocator: std.mem.Allocator, shape: []const usize) !Tensor(T)
```

Allocate a new zero-initialized tensor with the given shape.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `allocator` | `std.mem.Allocator` | Memory allocator |
| `shape` | `[]const usize` | Desired dimensions |

**Returns:** `!Tensor(T)` -- the new tensor, or `TensorError`.

### `Tensor(T).deinit`

```zig
pub fn deinit(self: *Self) void
```

Free all memory owned by this tensor. The tensor must not be used after calling
`deinit`.

### `Tensor(T).get`

```zig
pub fn get(self: Self, indices: []const usize) !T
```

Retrieve the element at the given multi-dimensional index.

**Returns:** the element value, or `TensorError.InvalidIndex`.

### `Tensor(T).set`

```zig
pub fn set(self: *Self, indices: []const usize, value: T) !void
```

Write a value at the given multi-dimensional index.

### `Tensor(T).matmul`

```zig
pub fn matmul(self: Self, other: Self, allocator: std.mem.Allocator) !Self
```

Matrix multiplication. Both tensors must be 2-D with compatible inner
dimensions (self is M x K, other is K x N). Returns an M x N tensor.

### `Tensor(T).add`

```zig
pub fn add(self: Self, other: Self, allocator: std.mem.Allocator) !Self
```

Element-wise addition. Both tensors must have the same shape.

### `Tensor(T).reshape`

```zig
pub fn reshape(self: *Self, new_shape: []const usize) !Self
```

Return a view of the tensor with a different shape. The total number of
elements must remain the same.

### `Tensor(T).print`

```zig
pub fn print(self: Self, writer: anytype) !void
```

Write a human-readable representation of the tensor to `writer`.

---

## Error Types

All fallible functions in this module return members of `TensorError` (see
above).

---

## Usage Example

```zig
const std = @import("std");
const Tensor = @import("zigllama").foundation.tensor.Tensor;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    // Create a 2x3 matrix
    var a = try Tensor(f32).init(alloc, &[_]usize{ 2, 3 });
    defer a.deinit();

    try a.set(&[_]usize{ 0, 0 }, 1.0);
    try a.set(&[_]usize{ 0, 1 }, 2.0);
    try a.set(&[_]usize{ 0, 2 }, 3.0);
    try a.set(&[_]usize{ 1, 0 }, 4.0);
    try a.set(&[_]usize{ 1, 1 }, 5.0);
    try a.set(&[_]usize{ 1, 2 }, 6.0);

    // Create a 3x2 matrix
    var b = try Tensor(f32).init(alloc, &[_]usize{ 3, 2 });
    defer b.deinit();

    try b.set(&[_]usize{ 0, 0 }, 1.0);
    try b.set(&[_]usize{ 1, 1 }, 1.0);
    try b.set(&[_]usize{ 2, 0 }, 0.5);
    try b.set(&[_]usize{ 2, 1 }, 0.5);

    // Matrix multiply => 2x2 result
    var c = try a.matmul(b, alloc);
    defer c.deinit();
}
```

---

## Related Modules

- [`linear_algebra.matrix_ops`](matrix-ops.md) -- SIMD-optimized matrix
  operations that accept `Tensor(f32)`.
- [`linear_algebra.quantization`](quantization.md) -- Quantize/dequantize
  `Tensor` instances.
- [`foundation.memory_mapping`](memory-mapping.md) -- Create tensors backed by
  memory-mapped files.
