# linear_algebra.iq_quantization

## Module Path

```
zigllama.linear_algebra.iq_quantization
```

**Source file:** `src/linear_algebra/iq_quantization.zig`

> **Internal module.** This API may change between releases.

---

## Public Types

### `IQType`

```zig
pub const IQType = enum {
    IQ1_S,
    IQ1_M,
    IQ2_XXS,
    IQ2_XS,
    IQ2_S,
    IQ3_XXS,
    IQ3_S,
    IQ4_XS,
    IQ4_NL,
};
```

Importance quantization formats. These achieve very low bit rates (1--4 bits
per weight) by exploiting non-uniform weight distributions.

| Variant | Approx Bits/Weight | Notes |
|---------|-------------------|-------|
| `IQ1_S` | 1.5 | 1-bit with super-block scales |
| `IQ1_M` | 1.75 | 1-bit with mixed scales |
| `IQ2_XXS` | 2.06 | 2-bit extra-extra-small overhead |
| `IQ2_XS` | 2.31 | 2-bit extra-small overhead |
| `IQ2_S` | 2.5 | 2-bit standard |
| `IQ3_XXS` | 3.06 | 3-bit extra-extra-small overhead |
| `IQ3_S` | 3.44 | 3-bit standard |
| `IQ4_XS` | 4.25 | 4-bit extra-small overhead |
| `IQ4_NL` | 4.5 | 4-bit with non-linear mapping |

### `BlockIQ1S`

```zig
pub const BlockIQ1S = struct {
    qs: [QK_K / 8]u8,
    qh: [QK_K / 16]u8,
    d: f16,
};
```

### `BlockIQ2XS`

```zig
pub const BlockIQ2XS = struct {
    d: f16,
    qs: [QK_K / 4]u16,
    scales: [QK_K / 32]u8,
};
```

### `BlockIQ3S`

```zig
pub const BlockIQ3S = struct {
    d: f16,
    qs: [3 * QK_K / 8]u8,
    qh: [QK_K / 8]u8,
    signs: [QK_K / 8]u8,
    scales: [QK_K / 64]u8,
};
```

### `BlockIQ4XS`

```zig
pub const BlockIQ4XS = struct {
    d: f16,
    scales_h: u16,
    scales_l: [QK_K / 64]u8,
    qs: [QK_K / 2]u8,
};
```

### `IQuantizer`

```zig
pub const IQuantizer = struct {
    allocator: std.mem.Allocator,
    lookup_tables: ?*const LookupTables,
};
```

Quantizer/dequantizer for importance-quantized formats. Uses precomputed lookup
tables for fast dequantization of IQ4_NL and IQ3_S.

---

## Public Functions

### `IQuantizer.quantize`

```zig
pub fn quantize(
    self: IQuantizer,
    data: []const f32,
    iq_type: IQType,
) ![]u8
```

Quantize a contiguous f32 buffer. Input length must be a multiple of `QK_K`
(256).

### `IQuantizer.dequantize`

```zig
pub fn dequantize(
    self: IQuantizer,
    data: []const u8,
    iq_type: IQType,
) ![]f32
```

Dequantize packed bytes back to f32.

---

## Error Types

- `error{InvalidAlignment}` -- input length is not a multiple of `QK_K`.
- `error{UnsupportedIQType}` -- the requested IQ format is not yet implemented.
- `error{OutOfMemory}`

---

## Usage Example

```zig
const iq = @import("zigllama").linear_algebra.iq_quantization;

var iquantizer = iq.IQuantizer{
    .allocator = allocator,
    .lookup_tables = null, // auto-initialize on first use
};

const packed = try iquantizer.quantize(float_data, .IQ4_NL);
defer allocator.free(packed);

const restored = try iquantizer.dequantize(packed, .IQ4_NL);
defer allocator.free(restored);
```

---

## Related Modules

- [`linear_algebra.k_quantization`](k-quantization.md) -- K-quant family
  (Q4_K, Q5_K, Q6_K).
- [`linear_algebra.quantization`](quantization.md) -- Higher-level quantization
  API.
- [`foundation.gguf_format`](gguf-format.md) -- `GGMLType` variants that
  correspond to IQ formats.
