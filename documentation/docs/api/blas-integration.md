# foundation.blas_integration

## Module Path

```
zigllama.foundation.blas_integration
```

**Source file:** `src/foundation/blas_integration.zig`

> **Internal module.** This API may change between releases.

---

## Public Types

### `BlasInterface`

```zig
pub const BlasInterface = struct {
    // Level 1 (vector-vector)
    dotFn:  *const fn (n: usize, x: [*]const f32, y: [*]const f32) f32,
    axpyFn: *const fn (n: usize, alpha: f32, x: [*]const f32, y: [*]f32) void,
    scalFn: *const fn (n: usize, alpha: f32, x: [*]f32) void,
    nrm2Fn: *const fn (n: usize, x: [*]const f32) f32,

    // Level 2 (matrix-vector)
    gemvFn: *const fn (
        m: usize, n: usize, alpha: f32,
        a: [*]const f32, lda: usize,
        x: [*]const f32, beta: f32,
        y: [*]f32,
    ) void,

    // Level 3 (matrix-matrix)
    gemmFn: *const fn (
        m: usize, n: usize, k: usize,
        alpha: f32,
        a: [*]const f32, lda: usize,
        b: [*]const f32, ldb: usize,
        beta: f32,
        c: [*]f32, ldc: usize,
    ) void,
};
```

Virtual function table that abstracts over different BLAS providers. Allows
ZigLlama to dispatch to a native BLAS library at runtime without compile-time
coupling.

### `GenericBlas`

Pure-Zig fallback implementation of `BlasInterface`. Used when no external BLAS
library is available. Provides correct results but without hand-tuned assembly
kernels.

### `OpenBlas`

Implementation of `BlasInterface` that forwards calls to OpenBLAS via the C ABI.
Detected and linked at build time through `build.zig`.

### `BlasManager`

```zig
pub const BlasManager = struct {
    backend: BlasInterface,
    thread_count: usize,
};
```

Singleton-like coordinator that selects the best available BLAS backend and
configures thread counts. Retrieved via `BlasManager.global()`.

---

## Public Functions

### Level 1 -- Vector-Vector

```zig
pub fn dot(blas: BlasInterface, n: usize, x: [*]const f32, y: [*]const f32) f32
pub fn axpy(blas: BlasInterface, n: usize, alpha: f32, x: [*]const f32, y: [*]f32) void
pub fn scal(blas: BlasInterface, n: usize, alpha: f32, x: [*]f32) void
pub fn nrm2(blas: BlasInterface, n: usize, x: [*]const f32) f32
```

Standard BLAS Level 1 operations: dot product, scaled addition, scaling, and
Euclidean norm.

### Level 2 -- Matrix-Vector

```zig
pub fn gemv(
    blas: BlasInterface,
    m: usize, n: usize,
    alpha: f32, a: [*]const f32, lda: usize,
    x: [*]const f32,
    beta: f32, y: [*]f32,
) void
```

General matrix-vector multiply: `y = alpha * A * x + beta * y`.

### Level 3 -- Matrix-Matrix

```zig
pub fn gemm(
    blas: BlasInterface,
    m: usize, n: usize, k: usize,
    alpha: f32, a: [*]const f32, lda: usize,
    b: [*]const f32, ldb: usize,
    beta: f32, c: [*]f32, ldc: usize,
) void
```

General matrix-matrix multiply: `C = alpha * A * B + beta * C`.

---

## Error Types

BLAS functions do not return errors. Invalid inputs (e.g., mismatched
dimensions) cause undefined behavior, matching standard BLAS conventions. Callers
must validate shapes before invoking these routines.

---

## Usage Example

```zig
const blas = @import("zigllama").foundation.blas_integration;

const mgr = blas.BlasManager.global();

// Dot product of two 1024-element vectors
const result = blas.dot(mgr.backend, 1024, x_ptr, y_ptr);
```

---

## Related Modules

- [`linear_algebra.matrix_ops`](matrix-ops.md) -- SIMD matrix ops that may
  delegate to BLAS.
- [`foundation.threading`](threading.md) -- Thread pool used by multi-threaded
  BLAS backends.
