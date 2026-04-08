# linear_algebra.matrix_ops

## Module Path

```
zigllama.linear_algebra.matrix_ops
```

**Source file:** `src/linear_algebra/matrix_ops.zig`

---

## Public Types

### `SimdConfig`

```zig
pub const SimdConfig = struct {
    simd_width: usize,
    has_avx: bool,
    has_avx2: bool,
    has_fma: bool,
    has_neon: bool,
};
```

Describes the SIMD capabilities of the current CPU. Detected once at startup and
used by every matrix kernel to select the optimal code path.

| Field | Description |
|-------|-------------|
| `simd_width` | Number of f32 elements processed per SIMD instruction (4, 8, or 16) |
| `has_avx` | x86 AVX (256-bit float ops) |
| `has_avx2` | x86 AVX2 (256-bit integer ops, FMA3) |
| `has_fma` | Fused multiply-add support |
| `has_neon` | ARM NEON (128-bit SIMD) |

---

## Public Functions

### `SimdConfig.detect`

```zig
pub fn detect() SimdConfig
```

Query the CPU for supported instruction sets and return a populated
`SimdConfig`. Safe to call from any thread; the result is deterministic for the
lifetime of the process.

### `matmulSIMD_f32_simple`

```zig
pub fn matmulSIMD_f32_simple(
    a: Tensor(f32),
    b: Tensor(f32),
    allocator: std.mem.Allocator,
) !Tensor(f32)
```

SIMD-accelerated matrix multiplication using a straightforward row-by-column
algorithm. Good baseline for small matrices (< 256 x 256).

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `a` | `Tensor(f32)` | Left operand, shape M x K |
| `b` | `Tensor(f32)` | Right operand, shape K x N |
| `allocator` | `Allocator` | Memory allocator for the result |

**Returns:** `!Tensor(f32)` with shape M x N.

### `matmulSIMD_f32_blocked`

```zig
pub fn matmulSIMD_f32_blocked(
    a: Tensor(f32),
    b: Tensor(f32),
    allocator: std.mem.Allocator,
) !Tensor(f32)
```

Cache-blocked SIMD matrix multiplication. Tiles the computation into blocks
that fit in L1/L2 cache, providing significantly better throughput on large
matrices. This is the default path for model-weight multiplications.

**Parameters:** same as `matmulSIMD_f32_simple`.

**Returns:** `!Tensor(f32)` with shape M x N.

---

## Error Types

Both multiplication functions return `TensorError.IncompatibleShapes` when the
inner dimensions of `a` and `b` do not match, or `error.OutOfMemory`.

---

## Usage Example

```zig
const matrix_ops = @import("zigllama").linear_algebra.matrix_ops;
const Tensor = @import("zigllama").foundation.tensor.Tensor;

// Detect SIMD at startup
const simd = matrix_ops.SimdConfig.detect();
std.debug.print("SIMD width: {}\n", .{simd.simd_width});

// Multiply two tensors with the blocked kernel
var a = try Tensor(f32).init(allocator, &[_]usize{ 512, 768 });
defer a.deinit();
var b = try Tensor(f32).init(allocator, &[_]usize{ 768, 512 });
defer b.deinit();

var c = try matrix_ops.matmulSIMD_f32_blocked(a, b, allocator);
defer c.deinit();
// c.shape == [512, 512]
```

---

## Related Modules

- [`foundation.tensor`](tensor.md) -- `Tensor` type used as input and output.
- [`foundation.blas_integration`](blas-integration.md) -- Alternative BLAS
  backend for matrix multiplication.
- [`foundation.threading`](threading.md) -- Parallel tiling used by the blocked
  kernel.
