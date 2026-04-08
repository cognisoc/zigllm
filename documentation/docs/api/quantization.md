# linear_algebra.quantization

## Module Path

```
zigllama.linear_algebra.quantization
```

**Source file:** `src/linear_algebra/quantization.zig`

---

## Public Types

### `QuantType`

```zig
pub const QuantType = enum {
    F32,
    F16,
    INT8,
    Q4_0,
    Q4_1,
    Q8_0,
};
```

Supported quantization formats. Lower-bit formats trade precision for memory
savings.

| Variant | Bits/Weight | Description |
|---------|------------|-------------|
| `F32` | 32 | Full precision (no quantization) |
| `F16` | 16 | Half precision |
| `INT8` | 8 | Symmetric 8-bit integer |
| `Q4_0` | 4 | 4-bit with per-block scale |
| `Q4_1` | 4 | 4-bit with per-block scale and minimum |
| `Q8_0` | 8 | 8-bit with per-block scale |

### `QuantParams`

```zig
pub const QuantParams = struct {
    quant_type: QuantType,
    block_size: usize = 32,
    symmetric: bool = true,
    calibration_data: ?Tensor(f32) = null,
};
```

Parameters controlling the quantization process.

### `QuantizedTensor(quant_type)`

```zig
pub fn QuantizedTensor(comptime quant_type: QuantType) type { ... }
```

Compile-time-specialized tensor that stores weights in the given quantized
format. Exposes the same shape metadata as `Tensor(f32)` but uses a packed
internal representation.

---

## Public Functions

### `quantizeTensor`

```zig
pub fn quantizeTensor(
    tensor: Tensor(f32),
    quant_type: QuantType,
) !QuantizedTensor(quant_type)
```

Quantize a full-precision tensor into the specified format. The input tensor is
not modified.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `tensor` | `Tensor(f32)` | Source tensor in f32 |
| `quant_type` | `QuantType` | Target format |

**Returns:** a `QuantizedTensor` containing the packed data and quantization
metadata (scales, zero-points).

### `dequantizeTensor`

```zig
pub fn dequantizeTensor(
    qtensor: anytype, // QuantizedTensor(*)
) !Tensor(f32)
```

Reconstruct an approximate f32 tensor from quantized data. The result is
allocated with the allocator stored in the quantized tensor.

### `quantizedMatmul`

```zig
pub fn quantizedMatmul(
    a: anytype,
    b: anytype,
    allocator: std.mem.Allocator,
) !Tensor(f32)
```

Multiply two quantized tensors without fully dequantizing them first. Uses
fused dequantize-multiply kernels for better cache utilization.

---

## Error Types

- `error{UnsupportedQuantType}` -- requested format is not implemented.
- `error{InvalidBlockSize}` -- tensor element count is not a multiple of the
  block size.
- `TensorError.OutOfMemory`

---

## Usage Example

```zig
const quant = @import("zigllama").linear_algebra.quantization;
const Tensor = @import("zigllama").foundation.tensor.Tensor;

var weights = try Tensor(f32).init(allocator, &[_]usize{ 4096, 4096 });
defer weights.deinit();
// ... fill weights ...

// Quantize to 4-bit
var q_weights = try quant.quantizeTensor(weights, .Q4_0);
defer q_weights.deinit();

// Dequantize back to f32 for verification
var recovered = try quant.dequantizeTensor(q_weights);
defer recovered.deinit();
```

---

## Related Modules

- [`linear_algebra.k_quantization`](k-quantization.md) -- K-quant formats
  (Q4_K, Q5_K, Q6_K).
- [`linear_algebra.iq_quantization`](iq-quantization.md) -- Importance
  quantization formats.
- [`foundation.gguf_format`](gguf-format.md) -- `GGMLType` enum maps to these
  quantization variants.
