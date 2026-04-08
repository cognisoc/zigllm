# foundation.gguf_format

## Module Path

```
zigllama.foundation.gguf_format
```

**Source file:** `src/foundation/gguf_format.zig`

> **Internal module.** This API may change between releases.

---

## Public Constants

```zig
pub const GGUF_MAGIC: u32   = 0x46554747; // "GGUF" in little-endian
pub const GGUF_VERSION: u32 = 3;
```

These constants identify the file format and version expected by the reader.

---

## Public Types

### `GGMLType`

```zig
pub const GGMLType = enum(u32) {
    F32       = 0,
    F16       = 1,
    Q4_0      = 2,
    Q4_1      = 3,
    Q5_0      = 6,
    Q5_1      = 7,
    Q8_0      = 8,
    Q8_1      = 9,
    Q2_K      = 10,
    Q3_K      = 11,
    Q4_K      = 12,
    Q5_K      = 13,
    Q6_K      = 14,
    Q8_K      = 15,
    IQ2_XXS   = 16,
    IQ2_XS    = 17,
    IQ3_XXS   = 18,
    IQ1_S     = 19,
    IQ4_NL    = 20,
    IQ3_S     = 21,
    IQ2_S     = 22,
    IQ4_XS    = 23,
    I8        = 24,
    I16       = 25,
    I32       = 26,
    I64       = 27,
    F64       = 28,
    IQ1_M     = 29,
    // ...
};
```

Enumeration of every quantization / data type recognized by the GGML ecosystem.

### `GGUFValueType`

```zig
pub const GGUFValueType = enum(u32) {
    UINT8    = 0,
    INT8     = 1,
    UINT16   = 2,
    INT16    = 3,
    UINT32   = 4,
    INT32    = 5,
    FLOAT32  = 6,
    BOOL     = 7,
    STRING   = 8,
    ARRAY    = 9,
    UINT64   = 10,
    INT64    = 11,
    FLOAT64  = 12,
};
```

Types that can appear as metadata values inside a GGUF file header.

---

## Public Functions

### `GGMLType.blockSize`

```zig
pub fn blockSize(self: GGMLType) usize
```

Return the quantization block size for this type. For unquantized types (`F32`,
`F16`) the block size is 1.

| Type | Block Size |
|------|-----------|
| `F32` | 1 |
| `F16` | 1 |
| `Q4_0` | 32 |
| `Q4_1` | 32 |
| `Q8_0` | 32 |
| `Q2_K` .. `Q6_K` | 256 |

### `GGMLType.typeSize`

```zig
pub fn typeSize(self: GGMLType) usize
```

Return the byte size of one quantization block. Combined with `blockSize` this
lets you compute the memory footprint for an arbitrary number of elements:

```zig
const bytes = (num_elements / ggml_type.blockSize()) * ggml_type.typeSize();
```

---

## Error Types

This module does not define its own error set. Functions return `error{InvalidType}` when
encountering an unknown `GGMLType` discriminant.

---

## Usage Example

```zig
const fmt = @import("zigllama").foundation.gguf_format;

const q_type = fmt.GGMLType.Q4_0;

const block_sz = q_type.blockSize();   // 32
const type_sz  = q_type.typeSize();    // 18 (2-byte scale + 16 nibbles)

const num_params: usize = 4096 * 4096;
const storage_bytes = (num_params / block_sz) * type_sz;
```

---

## Related Modules

- [`models.gguf`](gguf.md) -- High-level GGUF reader that uses these constants.
- [`foundation.memory_mapping`](memory-mapping.md) -- Maps GGUF files into memory.
- [`linear_algebra.quantization`](quantization.md) -- Quantization routines that
  operate on `GGMLType` blocks.
