# models.gguf

## Module Path

```
zigllama.models.gguf
```

**Source file:** `src/models/gguf.zig`

---

## Public Types

### `GGUFHeader`

```zig
pub const GGUFHeader = struct {
    magic: u32,
    version: u32,
    n_tensors: u64,
    n_kv: u64,
};
```

The first 24 bytes of every GGUF file. `magic` must equal `GGUF_MAGIC`
(0x46554747) and `version` must be 3.

### `GGUFTensorInfo`

```zig
pub const GGUFTensorInfo = struct {
    name: []const u8,
    n_dims: u32,
    dimensions: [4]u64,
    ggml_type: GGMLType,
    offset: u64,
};
```

Metadata for a single tensor stored in the GGUF file. The `offset` is relative
to the start of the tensor data section.

### `GGUFValue`

```zig
pub const GGUFValue = union(GGUFValueType) {
    UINT8: u8,
    INT8: i8,
    UINT32: u32,
    INT32: i32,
    FLOAT32: f32,
    STRING: []const u8,
    ARRAY: []GGUFValue,
    BOOL: bool,
    // ...
};
```

Tagged union representing a single metadata value in the GGUF key-value store.

### `GGUFReader`

```zig
pub const GGUFReader = struct {
    file: std.fs.File,
    header: GGUFHeader,
    metadata: std.StringHashMap(GGUFValue),
    tensor_infos: std.StringHashMap(GGUFTensorInfo),
    data_offset: u64,
    allocator: std.mem.Allocator,
};
```

Stateful reader that opens a GGUF file, parses the header and metadata, and
provides random access to individual tensors.

---

## Public Functions

### `GGUFReader.open`

```zig
pub fn open(
    path: []const u8,
    allocator: std.mem.Allocator,
) !GGUFReader
```

Open a GGUF file and parse the header. Does **not** read tensor data
immediately -- tensors are loaded on demand.

### `GGUFReader.close`

```zig
pub fn close(self: *GGUFReader) void
```

Close the underlying file and free metadata.

### `GGUFReader.readHeader`

```zig
pub fn readHeader(self: *GGUFReader) !GGUFHeader
```

Parse and validate the file header. Called automatically by `open`.

### `GGUFReader.readMetadata`

```zig
pub fn readMetadata(self: *GGUFReader) !void
```

Read all key-value pairs from the metadata section into `self.metadata`.

### `GGUFReader.readTensorInfo`

```zig
pub fn readTensorInfo(self: *GGUFReader) !void
```

Read tensor descriptors into `self.tensor_infos`. After this call, tensor
names and shapes are available without loading the actual data.

### `GGUFReader.loadTensor`

```zig
pub fn loadTensor(
    self: *GGUFReader,
    name: []const u8,
    allocator: std.mem.Allocator,
) !Tensor(f32)
```

Read and dequantize a named tensor from the file. The returned `Tensor(f32)`
is always in full precision regardless of the on-disk quantization format.

### `GGUFReader.findTensor`

```zig
pub fn findTensor(
    self: GGUFReader,
    name: []const u8,
) ?GGUFTensorInfo
```

Look up tensor metadata by name without loading data. Returns `null` if the
tensor is not present in the file.

---

## Error Types

- `error{InvalidMagic}` -- file does not start with the GGUF magic number.
- `error{UnsupportedVersion}` -- file version is not 3.
- `error{TensorNotFound}` -- requested tensor name is not in the file.
- `error{UnsupportedQuantType}` -- tensor uses a quantization format not yet
  implemented.
- `std.fs.File.OpenError`

---

## Usage Example

```zig
const gguf = @import("zigllama").models.gguf;

var reader = try gguf.GGUFReader.open("llama-7b.Q4_0.gguf", allocator);
defer reader.close();

// Inspect metadata
try reader.readMetadata();
try reader.readTensorInfo();

// Load a specific weight tensor
if (reader.findTensor("blk.0.attn_q.weight")) |info| {
    std.debug.print("Q weight: {}x{}, type={}\n", .{
        info.dimensions[0], info.dimensions[1], info.ggml_type,
    });
}

var q_weight = try reader.loadTensor("blk.0.attn_q.weight", allocator);
defer q_weight.deinit();
```

---

## Related Modules

- [`foundation.gguf_format`](gguf-format.md) -- Constants (`GGUF_MAGIC`,
  `GGMLType`) used by the reader.
- [`foundation.memory_mapping`](memory-mapping.md) -- Optional memory-mapped
  access for large files.
- [`models.llama`](llama.md) -- Load GGUF weights into a `LLaMAModel`.
