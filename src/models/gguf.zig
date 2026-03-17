//! GGUF Format Support
//!
//! This module implements the GGUF (GPT-Generated Unified Format) for loading
//! pre-trained language models. GGUF is used by llama.cpp and other inference
//! engines for efficient model storage and loading.
//!
//! ## Educational Value
//! Understanding model serialization formats teaches:
//! - How neural network parameters are stored and loaded
//! - Memory layout optimization for inference
//! - Quantization formats and their trade-offs
//! - Model metadata and configuration management

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;
const Tensor = @import("../foundation/tensor.zig").Tensor;

/// GGUF file magic number for format identification
pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian

/// Current GGUF format version
pub const GGUF_VERSION: u32 = 3;

/// GGUF value types for metadata
pub const GGUFType = enum(u32) {
    // Scalar types
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    // Array types
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,

    /// Get the size in bytes for scalar types
    pub fn scalarSize(self: GGUFType) ?usize {
        return switch (self) {
            .UINT8, .INT8, .BOOL => 1,
            .UINT16, .INT16 => 2,
            .UINT32, .INT32, .FLOAT32 => 4,
            .UINT64, .INT64, .FLOAT64 => 8,
            .STRING, .ARRAY => null, // Variable size
        };
    }
};

/// Tensor data types in GGUF
pub const GGMLType = enum(u32) {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,

    /// Get the size of one element in bytes
    pub fn elementSize(self: GGMLType) usize {
        return switch (self) {
            .F32, .I32 => 4,
            .F16, .I16 => 2,
            .I8 => 1,
            // Quantized types have block-based sizes
            .Q4_0, .Q4_1 => 18, // 16 4-bit values + 2 bytes metadata = 18 bytes per block
            .Q5_0, .Q5_1 => 22, // 16 5-bit values + metadata
            .Q8_0 => 34, // 32 8-bit values + 2 bytes scale
            .Q8_1 => 36, // 32 8-bit values + scale + min
            .Q2_K => 84, // K-quantization blocks (varies)
            .Q3_K => 110,
            .Q4_K => 144,
            .Q5_K => 176,
            .Q6_K => 210,
            .Q8_K => 256,
        };
    }

    /// Check if this type is quantized
    pub fn isQuantized(self: GGMLType) bool {
        return switch (self) {
            .F32, .F16, .I8, .I16, .I32 => false,
            else => true,
        };
    }

    /// Get the block size for quantized types
    pub fn blockSize(self: GGMLType) usize {
        return switch (self) {
            .Q4_0, .Q4_1, .Q5_0, .Q5_1 => 16,
            .Q8_0, .Q8_1 => 32,
            .Q2_K, .Q3_K, .Q4_K, .Q5_K, .Q6_K, .Q8_K => 256,
            .F32, .F16, .I8, .I16, .I32 => 1, // Not blocked
        };
    }
};

/// GGUF metadata value
pub const GGUFValue = union(GGUFType) {
    UINT8: u8,
    INT8: i8,
    UINT16: u16,
    INT16: i16,
    UINT32: u32,
    INT32: i32,
    FLOAT32: f32,
    BOOL: bool,
    STRING: []const u8,
    ARRAY: GGUFArray,
    UINT64: u64,
    INT64: i64,
    FLOAT64: f64,

    /// Free any allocated memory in this value
    pub fn deinit(self: GGUFValue, allocator: Allocator) void {
        switch (self) {
            .STRING => |str| allocator.free(str),
            .ARRAY => |arr| arr.deinit(allocator),
            else => {},
        }
    }
};

/// GGUF array value
pub const GGUFArray = struct {
    element_type: GGUFType,
    length: u64,
    data: []u8,

    pub fn deinit(self: GGUFArray, allocator: Allocator) void {
        allocator.free(self.data);
    }
};

/// Tensor information from GGUF
pub const GGUFTensorInfo = struct {
    name: []const u8,
    dimensions: []u64,
    ggml_type: GGMLType,
    offset: u64,

    pub fn deinit(self: GGUFTensorInfo, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.dimensions);
    }

    /// Calculate total number of elements
    pub fn elementCount(self: GGUFTensorInfo) u64 {
        var count: u64 = 1;
        for (self.dimensions) |dim| {
            count *= dim;
        }
        return count;
    }

    /// Calculate total size in bytes
    pub fn sizeBytes(self: GGUFTensorInfo) u64 {
        const elements = self.elementCount();
        const element_size = self.ggml_type.elementSize();

        if (self.ggml_type.isQuantized()) {
            // For quantized types, calculate based on blocks
            const block_size = self.ggml_type.blockSize();
            const num_blocks = (elements + block_size - 1) / block_size;
            return num_blocks * element_size;
        } else {
            return elements * element_size;
        }
    }
};

/// GGUF file header
pub const GGUFHeader = struct {
    magic: u32,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,

    /// Validate header magic and version
    pub fn validate(self: GGUFHeader) !void {
        if (self.magic != GGUF_MAGIC) {
            return error.InvalidGGUFMagic;
        }
        if (self.version != GGUF_VERSION) {
            return error.UnsupportedGGUFVersion;
        }
    }
};

/// GGUF file reader
pub const GGUFReader = struct {
    /// File handle
    file: std.fs.File,
    /// Allocator for dynamic allocations
    allocator: Allocator,
    /// File header
    header: GGUFHeader,
    /// Metadata key-value pairs
    metadata: HashMap([]const u8, GGUFValue, StringContext, std.hash_map.default_max_load_percentage),
    /// Tensor information
    tensors: ArrayList(GGUFTensorInfo),
    /// Alignment for tensor data
    alignment: u64,
    /// Offset to tensor data
    data_offset: u64,

    const StringContext = struct {
        pub fn hash(self: @This(), s: []const u8) u64 {
            _ = self;
            return std.hash_map.hashString(s);
        }

        pub fn eql(self: @This(), a: []const u8, b: []const u8) bool {
            _ = self;
            return std.mem.eql(u8, a, b);
        }
    };

    /// Open and parse GGUF file
    pub fn open(file_path: []const u8, allocator: Allocator) !GGUFReader {
        const file = try std.fs.cwd().openFile(file_path, .{});
        errdefer file.close();

        var reader = GGUFReader{
            .file = file,
            .allocator = allocator,
            .header = undefined,
            .metadata = HashMap([]const u8, GGUFValue, StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .tensors = ArrayList(GGUFTensorInfo).init(allocator),
            .alignment = 32, // Default alignment
            .data_offset = 0,
        };

        try reader.parseHeader();
        try reader.parseMetadata();
        try reader.parseTensors();

        return reader;
    }

    /// Close and clean up resources
    pub fn close(self: *GGUFReader) void {
        // Free metadata
        var metadata_iter = self.metadata.iterator();
        while (metadata_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(self.allocator);
        }
        self.metadata.deinit();

        // Free tensors
        for (self.tensors.items) |tensor_info| {
            tensor_info.deinit(self.allocator);
        }
        self.tensors.deinit();

        self.file.close();
    }

    /// Parse file header
    fn parseHeader(self: *GGUFReader) !void {
        const reader = self.file.reader();

        self.header.magic = try reader.readIntLittle(u32);
        self.header.version = try reader.readIntLittle(u32);
        self.header.tensor_count = try reader.readIntLittle(u64);
        self.header.metadata_kv_count = try reader.readIntLittle(u64);

        try self.header.validate();
    }

    /// Parse metadata section
    fn parseMetadata(self: *GGUFReader) !void {
        const reader = self.file.reader();

        for (0..self.header.metadata_kv_count) |_| {
            // Read key
            const key = try self.readString(reader);
            errdefer self.allocator.free(key);

            // Read value type
            const value_type = @as(GGUFType, @enumFromInt(try reader.readIntLittle(u32)));

            // Read value
            const value = try self.readValue(reader, value_type);
            errdefer value.deinit(self.allocator);

            try self.metadata.put(key, value);
        }

        // Extract alignment if present
        if (self.metadata.get("general.alignment")) |alignment_value| {
            switch (alignment_value) {
                .UINT32 => |align| self.alignment = align,
                .UINT64 => |align| self.alignment = align,
                else => {},
            }
        }
    }

    /// Parse tensor information
    fn parseTensors(self: *GGUFReader) !void {
        const reader = self.file.reader();

        for (0..self.header.tensor_count) |_| {
            // Read tensor name
            const name = try self.readString(reader);
            errdefer self.allocator.free(name);

            // Read number of dimensions
            const n_dimensions = try reader.readIntLittle(u32);

            // Read dimension sizes
            var dimensions = try self.allocator.alloc(u64, n_dimensions);
            errdefer self.allocator.free(dimensions);

            for (0..n_dimensions) |i| {
                dimensions[i] = try reader.readIntLittle(u64);
            }

            // Read tensor type
            const ggml_type = @as(GGMLType, @enumFromInt(try reader.readIntLittle(u32)));

            // Read offset (will be updated later)
            const offset = try reader.readIntLittle(u64);

            const tensor_info = GGUFTensorInfo{
                .name = name,
                .dimensions = dimensions,
                .ggml_type = ggml_type,
                .offset = offset,
            };

            try self.tensors.append(tensor_info);
        }

        // Calculate data offset (aligned)
        const current_pos = try self.file.getPos();
        self.data_offset = ((current_pos + self.alignment - 1) / self.alignment) * self.alignment;
    }

    /// Read a string from the file
    fn readString(self: *GGUFReader, reader: anytype) ![]u8 {
        const length = try reader.readIntLittle(u64);
        if (length > 1024 * 1024) { // Sanity check: 1MB max string
            return error.StringTooLong;
        }

        var string = try self.allocator.alloc(u8, length);
        errdefer self.allocator.free(string);

        _ = try reader.readAll(string);
        return string;
    }

    /// Read a value of the given type
    fn readValue(self: *GGUFReader, reader: anytype, value_type: GGUFType) !GGUFValue {
        return switch (value_type) {
            .UINT8 => GGUFValue{ .UINT8 = try reader.readIntLittle(u8) },
            .INT8 => GGUFValue{ .INT8 = @as(i8, @bitCast(try reader.readIntLittle(u8))) },
            .UINT16 => GGUFValue{ .UINT16 = try reader.readIntLittle(u16) },
            .INT16 => GGUFValue{ .INT16 = @as(i16, @bitCast(try reader.readIntLittle(u16))) },
            .UINT32 => GGUFValue{ .UINT32 = try reader.readIntLittle(u32) },
            .INT32 => GGUFValue{ .INT32 = @as(i32, @bitCast(try reader.readIntLittle(u32))) },
            .FLOAT32 => GGUFValue{ .FLOAT32 = @as(f32, @bitCast(try reader.readIntLittle(u32))) },
            .BOOL => GGUFValue{ .BOOL = (try reader.readIntLittle(u8)) != 0 },
            .STRING => GGUFValue{ .STRING = try self.readString(reader) },
            .UINT64 => GGUFValue{ .UINT64 = try reader.readIntLittle(u64) },
            .INT64 => GGUFValue{ .INT64 = @as(i64, @bitCast(try reader.readIntLittle(u64))) },
            .FLOAT64 => GGUFValue{ .FLOAT64 = @as(f64, @bitCast(try reader.readIntLittle(u64))) },
            .ARRAY => {
                const element_type = @as(GGUFType, @enumFromInt(try reader.readIntLittle(u32)));
                const length = try reader.readIntLittle(u64);

                // Calculate array data size
                var data_size: u64 = 0;
                if (element_type.scalarSize()) |scalar_size| {
                    data_size = length * scalar_size;
                } else if (element_type == .STRING) {
                    // For string arrays, we need to read each string
                    return error.StringArraysNotSupported; // Simplified for now
                }

                var data = try self.allocator.alloc(u8, data_size);
                _ = try reader.readAll(data);

                return GGUFValue{ .ARRAY = GGUFArray{
                    .element_type = element_type,
                    .length = length,
                    .data = data,
                } };
            },
        };
    }

    /// Get metadata value by key
    pub fn getMetadata(self: GGUFReader, key: []const u8) ?GGUFValue {
        return self.metadata.get(key);
    }

    /// Find tensor by name
    pub fn findTensor(self: GGUFReader, name: []const u8) ?GGUFTensorInfo {
        for (self.tensors.items) |tensor_info| {
            if (std.mem.eql(u8, tensor_info.name, name)) {
                return tensor_info;
            }
        }
        return null;
    }

    /// Load tensor data
    pub fn loadTensor(self: *GGUFReader, tensor_info: GGUFTensorInfo, comptime T: type) !Tensor(T) {
        // Seek to tensor data
        const absolute_offset = self.data_offset + tensor_info.offset;
        try self.file.seekTo(absolute_offset);

        // Convert dimensions to usize array
        var shape = try self.allocator.alloc(usize, tensor_info.dimensions.len);
        defer self.allocator.free(shape);

        for (tensor_info.dimensions, 0..) |dim, i| {
            shape[i] = @as(usize, @intCast(dim));
        }

        // Create tensor
        var tensor = try Tensor(T).init(self.allocator, shape);
        errdefer tensor.deinit();

        // Read tensor data
        const data_size = tensor_info.sizeBytes();
        const tensor_data = try self.allocator.alloc(u8, data_size);
        defer self.allocator.free(tensor_data);

        _ = try self.file.reader().readAll(tensor_data);

        // Convert data based on type
        switch (tensor_info.ggml_type) {
            .F32 => {
                if (T != f32) return error.TypeMismatch;
                @memcpy(@as([*]u8, @ptrCast(tensor.data.ptr))[0..data_size], tensor_data);
            },
            .F16 => {
                // TODO: Implement F16 to F32 conversion
                return error.F16ConversionNotImplemented;
            },
            else => {
                // TODO: Implement quantization dequantization
                return error.QuantizedFormatsNotImplemented;
            },
        }

        return tensor;
    }

    /// Print model information
    pub fn printInfo(self: GGUFReader, writer: anytype) !void {
        try writer.print("GGUF Model Information:\n", .{});
        try writer.print("  Version: {d}\n", .{self.header.version});
        try writer.print("  Tensors: {d}\n", .{self.header.tensor_count});
        try writer.print("  Metadata entries: {d}\n", .{self.header.metadata_kv_count});
        try writer.print("  Data alignment: {d} bytes\n", .{self.alignment});

        // Print key metadata
        if (self.getMetadata("general.name")) |name| {
            switch (name) {
                .STRING => |s| try writer.print("  Model name: {s}\n", .{s}),
                else => {},
            }
        }

        if (self.getMetadata("llama.context_length")) |ctx_len| {
            switch (ctx_len) {
                .UINT32 => |len| try writer.print("  Context length: {d}\n", .{len}),
                else => {},
            }
        }

        // Print tensor summary
        try writer.print("  \nTensors:\n", .{});
        for (self.tensors.items[0..@min(5, self.tensors.items.len)]) |tensor_info| {
            try writer.print("    {s}: {any} ({s})\n", .{
                tensor_info.name,
                tensor_info.dimensions,
                @tagName(tensor_info.ggml_type),
            });
        }

        if (self.tensors.items.len > 5) {
            try writer.print("    ... and {d} more tensors\n", .{self.tensors.items.len - 5});
        }
    }
};

// GGUF format tests
test "GGUF header validation" {
    const testing = std.testing;

    const valid_header = GGUFHeader{
        .magic = GGUF_MAGIC,
        .version = GGUF_VERSION,
        .tensor_count = 100,
        .metadata_kv_count = 50,
    };

    try valid_header.validate();

    const invalid_magic = GGUFHeader{
        .magic = 0x12345678,
        .version = GGUF_VERSION,
        .tensor_count = 100,
        .metadata_kv_count = 50,
    };

    try testing.expectError(error.InvalidGGUFMagic, invalid_magic.validate());

    const invalid_version = GGUFHeader{
        .magic = GGUF_MAGIC,
        .version = 999,
        .tensor_count = 100,
        .metadata_kv_count = 50,
    };

    try testing.expectError(error.UnsupportedGGUFVersion, invalid_version.validate());
}

test "GGML type properties" {
    const testing = std.testing;

    // Test element sizes
    try testing.expectEqual(@as(usize, 4), GGMLType.F32.elementSize());
    try testing.expectEqual(@as(usize, 2), GGMLType.F16.elementSize());
    try testing.expectEqual(@as(usize, 18), GGMLType.Q4_0.elementSize());

    // Test quantization detection
    try testing.expect(!GGMLType.F32.isQuantized());
    try testing.expect(!GGMLType.F16.isQuantized());
    try testing.expect(GGMLType.Q4_0.isQuantized());
    try testing.expect(GGMLType.Q8_0.isQuantized());

    // Test block sizes
    try testing.expectEqual(@as(usize, 1), GGMLType.F32.blockSize());
    try testing.expectEqual(@as(usize, 16), GGMLType.Q4_0.blockSize());
    try testing.expectEqual(@as(usize, 32), GGMLType.Q8_0.blockSize());
}

test "GGUF tensor info calculations" {
    const testing = std.testing;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const name = try allocator.dupe(u8, "test.weight");
    defer allocator.free(name);

    const dimensions = try allocator.dupe(u64, &[_]u64{ 4096, 4096 });
    defer allocator.free(dimensions);

    const tensor_info = GGUFTensorInfo{
        .name = name,
        .dimensions = dimensions,
        .ggml_type = .F32,
        .offset = 0,
    };

    const element_count = tensor_info.elementCount();
    try testing.expectEqual(@as(u64, 4096 * 4096), element_count);

    const size_bytes = tensor_info.sizeBytes();
    try testing.expectEqual(@as(u64, 4096 * 4096 * 4), size_bytes); // f32 = 4 bytes
}

test "GGUF value types" {
    const testing = std.testing;

    // Test scalar sizes
    try testing.expectEqual(@as(usize, 1), GGUFType.UINT8.scalarSize().?);
    try testing.expectEqual(@as(usize, 4), GGUFType.FLOAT32.scalarSize().?);
    try testing.expectEqual(@as(usize, 8), GGUFType.FLOAT64.scalarSize().?);
    try testing.expect(GGUFType.STRING.scalarSize() == null);
    try testing.expect(GGUFType.ARRAY.scalarSize() == null);
}