# linear_algebra.k_quantization

## Module Path

```
zigllama.linear_algebra.k_quantization
```

**Source file:** `src/linear_algebra/k_quantization.zig`

> **Internal module.** This API may change between releases.

---

## Public Constants

```zig
pub const QK_K: usize = 256;
```

Super-block size shared by all K-quantization formats. Every 256 elements are
quantized together, allowing finer scale granularity than the older Q4_0/Q8_0
formats (which use blocks of 32).

---

## Public Types

### `KQuantType`

```zig
pub const KQuantType = enum {
    Q4_K,
    Q5_K,
    Q6_K,
};
```

The three K-quantization bit widths.

| Variant | Bits/Weight | Bytes per 256-element block |
|---------|------------|----------------------------|
| `Q4_K` | 4.5 | 144 |
| `Q5_K` | 5.5 | 176 |
| `Q6_K` | 6.5 | 210 |

### `BlockQ4K`

```zig
pub const BlockQ4K = struct {
    d: f16,                   // super-block scale
    dmin: f16,                // super-block minimum
    scales: [12]u8,           // sub-block scales (packed)
    qs: [QK_K / 2]u8,        // quantized nibbles
};
```

On-disk and in-memory layout of a single Q4_K block.

### `BlockQ5K`

```zig
pub const BlockQ5K = struct {
    d: f16,
    dmin: f16,
    scales: [12]u8,
    qh: [QK_K / 8]u8,        // high bits
    qs: [QK_K / 2]u8,        // low nibbles
};
```

### `BlockQ6K`

```zig
pub const BlockQ6K = struct {
    ql: [QK_K / 2]u8,        // low 4 bits
    qh: [QK_K / 4]u8,        // high 2 bits
    scales: [QK_K / 16]i8,   // sub-block scales
    d: f16,                   // super-block scale
};
```

### `KQuantizer`

```zig
pub const KQuantizer = struct {
    allocator: std.mem.Allocator,
};
```

Stateless quantizer/dequantizer for K-quant formats.

---

## Public Functions

### `KQuantizer.quantize`

```zig
pub fn quantize(
    self: KQuantizer,
    data: []const f32,
    ktype: KQuantType,
) ![]u8
```

Quantize a contiguous f32 buffer into packed K-quant bytes. The input length
must be a multiple of `QK_K` (256).

**Returns:** caller-owned byte slice.

### `KQuantizer.dequantize`

```zig
pub fn dequantize(
    self: KQuantizer,
    data: []const u8,
    ktype: KQuantType,
) ![]f32
```

Dequantize packed K-quant bytes back to f32.

---

## Error Types

- `error{InvalidAlignment}` -- input length is not a multiple of `QK_K`.
- `error{OutOfMemory}`

---

## Usage Example

```zig
const kq = @import("zigllama").linear_algebra.k_quantization;

var quantizer = kq.KQuantizer{ .allocator = allocator };

const raw_weights: []const f32 = model_data[offset..][0..4096];
const packed = try quantizer.quantize(raw_weights, .Q4_K);
defer allocator.free(packed);

const restored = try quantizer.dequantize(packed, .Q4_K);
defer allocator.free(restored);
```

---

## Related Modules

- [`linear_algebra.quantization`](quantization.md) -- Higher-level quantization
  API.
- [`linear_algebra.iq_quantization`](iq-quantization.md) -- Importance
  quantization, a complementary family.
- [`foundation.gguf_format`](gguf-format.md) -- `GGMLType.Q4_K` etc. map to
  these block formats.
