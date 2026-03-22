// GGUF Format Implementation Tests
// Comprehensive tests for GGUF file format reading and parsing

const std = @import("std");
const testing = std.testing;
const gguf = @import("../src/foundation/gguf_format.zig");

test "GGUF type size calculation" {
    try testing.expect(gguf.GGUFType.uint8.size() == 1);
    try testing.expect(gguf.GGUFType.int8.size() == 1);
    try testing.expect(gguf.GGUFType.uint16.size() == 2);
    try testing.expect(gguf.GGUFType.int16.size() == 2);
    try testing.expect(gguf.GGUFType.uint32.size() == 4);
    try testing.expect(gguf.GGUFType.int32.size() == 4);
    try testing.expect(gguf.GGUFType.float32.size() == 4);
    try testing.expect(gguf.GGUFType.bool.size() == 1);
    try testing.expect(gguf.GGUFType.uint64.size() == 8);
    try testing.expect(gguf.GGUFType.int64.size() == 8);
    try testing.expect(gguf.GGUFType.float64.size() == 8);

    // Variable size types
    try testing.expect(gguf.GGUFType.string.size() == null);
    try testing.expect(gguf.GGUFType.array.size() == null);
}

test "GGML type properties" {
    // Test block sizes
    try testing.expect(gguf.GGMLType.f32.blockSize() == 1);
    try testing.expect(gguf.GGMLType.f16.blockSize() == 1);
    try testing.expect(gguf.GGMLType.q4_0.blockSize() == 32);
    try testing.expect(gguf.GGMLType.q4_1.blockSize() == 32);
    try testing.expect(gguf.GGMLType.q8_0.blockSize() == 32);
    try testing.expect(gguf.GGMLType.q4_k.blockSize() == 256);
    try testing.expect(gguf.GGMLType.iq2_xxs.blockSize() == 256);

    // Test type sizes
    try testing.expect(gguf.GGMLType.f32.typeSize() == 4);
    try testing.expect(gguf.GGMLType.f16.typeSize() == 2);
    try testing.expect(gguf.GGMLType.bf16.typeSize() == 2);
    try testing.expect(gguf.GGMLType.q4_0.typeSize() == 20);
    try testing.expect(gguf.GGMLType.q4_1.typeSize() == 24);
    try testing.expect(gguf.GGMLType.q8_0.typeSize() == 36);

    // Test string representation
    try testing.expectEqualStrings("f32", gguf.GGMLType.f32.toString());
    try testing.expectEqualStrings("q4_0", gguf.GGMLType.q4_0.toString());
    try testing.expectEqualStrings("iq2_xxs", gguf.GGMLType.iq2_xxs.toString());
}

test "GGUF tensor info calculations" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create a test tensor info
    const dims = try allocator.dupe(u64, &[_]u64{ 1024, 768 });
    var tensor_info = gguf.GGUFTensorInfo{
        .name = try allocator.dupe(u8, "test_tensor"),
        .n_dims = 2,
        .dimensions = dims,
        .type = .f32,
        .offset = 1000,
    };

    // Test element count calculation
    const element_count = tensor_info.elementCount();
    try testing.expect(element_count == 1024 * 768);

    // Test size calculation for unquantized type
    const size_f32 = tensor_info.sizeInBytes();
    try testing.expect(size_f32 == element_count * 4); // f32 = 4 bytes per element

    // Test with quantized type
    tensor_info.type = .q4_0;
    const size_q4_0 = tensor_info.sizeInBytes();
    const expected_blocks = (element_count + 31) / 32; // 32 elements per block
    try testing.expect(size_q4_0 == expected_blocks * 20); // 20 bytes per Q4_0 block
}

test "GGUF value formatting" {
    var buffer: [256]u8 = undefined;

    // Test different value types
    const uint8_val = gguf.GGUFValue{ .uint8 = 42 };
    const formatted_uint8 = try std.fmt.bufPrint(&buffer, "{}", .{uint8_val});
    try testing.expectEqualStrings("42", formatted_uint8);

    const float32_val = gguf.GGUFValue{ .float32 = 3.14159 };
    const formatted_float = try std.fmt.bufPrint(&buffer, "{}", .{float32_val});
    try testing.expect(std.mem.indexOf(u8, formatted_float, "3.14159") != null);

    const bool_val = gguf.GGUFValue{ .bool = true };
    const formatted_bool = try std.fmt.bufPrint(&buffer, "{}", .{bool_val});
    try testing.expectEqualStrings("true", formatted_bool);

    const string_val = gguf.GGUFValue{ .string = "hello world" };
    const formatted_string = try std.fmt.bufPrint(&buffer, "{}", .{string_val});
    try testing.expectEqualStrings("\"hello world\"", formatted_string);
}

test "GGUF header validation" {
    // Valid header
    const valid_header = gguf.GGUFHeader{
        .magic = gguf.GGUF_MAGIC,
        .version = gguf.GGUF_VERSION,
        .tensor_count = 100,
        .metadata_kv_count = 20,
    };

    try valid_header.validate();

    // Invalid magic
    const invalid_magic = gguf.GGUFHeader{
        .magic = 0x12345678,
        .version = gguf.GGUF_VERSION,
        .tensor_count = 100,
        .metadata_kv_count = 20,
    };

    try testing.expectError(error.InvalidGGUFMagic, invalid_magic.validate());

    // Invalid version
    const invalid_version = gguf.GGUFHeader{
        .magic = gguf.GGUF_MAGIC,
        .version = 99,
        .tensor_count = 100,
        .metadata_kv_count = 20,
    };

    try testing.expectError(error.UnsupportedGGUFVersion, invalid_version.validate());
}

test "GGUF file metadata access" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gguf_file = gguf.GGUFFile.init(allocator);
    defer gguf_file.deinit();

    // Add test metadata
    try gguf_file.metadata.put(try allocator.dupe(u8, "general.architecture"), gguf.GGUFValue{ .string = try allocator.dupe(u8, "llama") });
    try gguf_file.metadata.put(try allocator.dupe(u8, "llama.context_length"), gguf.GGUFValue{ .uint32 = 2048 });
    try gguf_file.metadata.put(try allocator.dupe(u8, "llama.rope.freq_base"), gguf.GGUFValue{ .float32 = 10000.0 });
    try gguf_file.metadata.put(try allocator.dupe(u8, "training.enabled"), gguf.GGUFValue{ .bool = false });

    // Test string access
    const arch = gguf_file.getMetadataString("general.architecture");
    try testing.expect(arch != null);
    try testing.expectEqualStrings("llama", arch.?);

    // Test integer access
    const ctx_len = gguf_file.getMetadataInt("llama.context_length", u32);
    try testing.expect(ctx_len != null);
    try testing.expect(ctx_len.? == 2048);

    // Test float access
    const freq_base = gguf_file.getMetadataFloat("llama.rope.freq_base", f32);
    try testing.expect(freq_base != null);
    try testing.expectApproxEqAbs(@as(f32, 10000.0), freq_base.?, 0.001);

    // Test non-existent key
    const missing = gguf_file.getMetadataString("missing.key");
    try testing.expect(missing == null);

    // Test wrong type access
    const wrong_type = gguf_file.getMetadataString("llama.context_length"); // It's a uint32, not string
    try testing.expect(wrong_type == null);
}

test "f16 to f32 conversion" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create a dummy loader for testing the conversion function
    const dummy_file = std.io.null_writer.any();
    var reader = gguf.GGUFReader.init(std.fs.File{ .handle = 0 }, allocator);
    var gguf_file = gguf.GGUFFile.init(allocator);
    defer gguf_file.deinit();

    var loader = gguf.GGUFModelLoader{
        .gguf = gguf_file,
        .reader = reader,
        .allocator = allocator,
    };

    // Test common f16 values
    // 0.0
    try testing.expectApproxEqAbs(@as(f32, 0.0), loader.f16ToF32(0x0000), 0.0001);

    // 1.0 in f16: sign=0, exp=15, mantissa=0 -> 0x3C00
    try testing.expectApproxEqAbs(@as(f32, 1.0), loader.f16ToF32(0x3C00), 0.0001);

    // -1.0 in f16: sign=1, exp=15, mantissa=0 -> 0xBC00
    try testing.expectApproxEqAbs(@as(f32, -1.0), loader.f16ToF32(0xBC00), 0.0001);

    // 2.0 in f16: sign=0, exp=16, mantissa=0 -> 0x4000
    try testing.expectApproxEqAbs(@as(f32, 2.0), loader.f16ToF32(0x4000), 0.0001);

    // 0.5 in f16: sign=0, exp=14, mantissa=0 -> 0x3800
    try testing.expectApproxEqAbs(@as(f32, 0.5), loader.f16ToF32(0x3800), 0.0001);
}

test "GGUF array handling" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create a test array of uint32 values
    const array_data = try allocator.alloc(u8, 4 * 5); // 5 uint32 values
    const uint32_slice = std.mem.bytesAsSlice(u32, array_data);
    uint32_slice[0] = 10;
    uint32_slice[1] = 20;
    uint32_slice[2] = 30;
    uint32_slice[3] = 40;
    uint32_slice[4] = 50;

    var test_array = gguf.GGUFArray{
        .type = .uint32,
        .len = 5,
        .data = array_data,
    };

    // Test array properties
    try testing.expect(test_array.type == .uint32);
    try testing.expect(test_array.len == 5);
    try testing.expect(test_array.data.len == 20); // 5 * 4 bytes

    // Verify data can be read back correctly
    const read_data = std.mem.bytesAsSlice(u32, test_array.data);
    try testing.expect(read_data[0] == 10);
    try testing.expect(read_data[4] == 50);

    // Test formatting
    const array_value = gguf.GGUFValue{ .array = test_array };
    var buffer: [64]u8 = undefined;
    const formatted = try std.fmt.bufPrint(&buffer, "{}", .{array_value});
    try testing.expectEqualStrings("[5 items]", formatted);
}

test "quantization block size calculations" {
    // Test that our block size calculations are correct
    const test_cases = [_]struct { type: gguf.GGMLType, expected_blocks: u64, elements: u64 }{
        .{ .type = .f32, .expected_blocks = 1000, .elements = 1000 },
        .{ .type = .q4_0, .expected_blocks = 32, .elements = 1000 }, // ceil(1000/32) = 32
        .{ .type = .q8_0, .expected_blocks = 32, .elements = 1000 }, // ceil(1000/32) = 32
        .{ .type = .q4_k, .expected_blocks = 4, .elements = 1000 }, // ceil(1000/256) = 4
    };

    for (test_cases) |test_case| {
        const block_size = test_case.type.blockSize();
        const actual_blocks = if (block_size == 1)
            test_case.elements
        else
            (test_case.elements + block_size - 1) / block_size;

        try testing.expect(actual_blocks == test_case.expected_blocks);
    }
}

test "GGUF tensor lookup" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var gguf_file = gguf.GGUFFile.init(allocator);
    defer gguf_file.deinit();

    // Create test tensors
    gguf_file.tensors = try allocator.alloc(gguf.GGUFTensorInfo, 3);

    gguf_file.tensors[0] = gguf.GGUFTensorInfo{
        .name = try allocator.dupe(u8, "token_embd.weight"),
        .n_dims = 2,
        .dimensions = try allocator.dupe(u64, &[_]u64{ 32000, 4096 }),
        .type = .q4_0,
        .offset = 0,
    };

    gguf_file.tensors[1] = gguf.GGUFTensorInfo{
        .name = try allocator.dupe(u8, "blk.0.attn_norm.weight"),
        .n_dims = 1,
        .dimensions = try allocator.dupe(u64, &[_]u64{4096}),
        .type = .f32,
        .offset = 1000,
    };

    gguf_file.tensors[2] = gguf.GGUFTensorInfo{
        .name = try allocator.dupe(u8, "output.weight"),
        .n_dims = 2,
        .dimensions = try allocator.dupe(u64, &[_]u64{ 4096, 32000 }),
        .type = .q8_0,
        .offset = 2000,
    };

    // Test successful lookup
    const found_tensor = gguf_file.getTensor("blk.0.attn_norm.weight");
    try testing.expect(found_tensor != null);
    try testing.expectEqualStrings("blk.0.attn_norm.weight", found_tensor.?.name);
    try testing.expect(found_tensor.?.type == .f32);
    try testing.expect(found_tensor.?.offset == 1000);

    // Test failed lookup
    const missing_tensor = gguf_file.getTensor("nonexistent.tensor");
    try testing.expect(missing_tensor == null);

    // Test element count calculation
    const token_embd = gguf_file.getTensor("token_embd.weight").?;
    try testing.expect(token_embd.elementCount() == 32000 * 4096);
}

test "GGUF utilities" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test file validation with non-existent file
    const valid = gguf.GGUFUtils.validateFile("nonexistent.gguf", allocator) catch false;
    try testing.expect(!valid);

    // Test memory usage estimation with non-existent file
    const memory_usage = gguf.GGUFUtils.estimateMemoryUsage("nonexistent.gguf", allocator) catch 0;
    try testing.expect(memory_usage == 0);

    // Note: We can't test with real GGUF files in unit tests, but the error handling works correctly
}

test "endianness handling" {
    // Test that we handle little-endian correctly
    const test_bytes = [_]u8{ 0x78, 0x56, 0x34, 0x12 }; // Little-endian 0x12345678

    const value = std.mem.bytesToValue(u32, &test_bytes);
    try testing.expect(value == 0x12345678);

    // Test with different sizes
    const test_bytes_16 = [_]u8{ 0x34, 0x12 }; // Little-endian 0x1234
    const value_16 = std.mem.bytesToValue(u16, &test_bytes_16);
    try testing.expect(value_16 == 0x1234);
}

test "GGUF magic number and constants" {
    // Test that magic number is correct
    try testing.expect(gguf.GGUF_MAGIC == 0x46554747);

    // Test that version is correct
    try testing.expect(gguf.GGUF_VERSION == 3);

    // Verify magic number bytes
    const magic_bytes = std.mem.toBytes(gguf.GGUF_MAGIC);
    try testing.expect(magic_bytes[0] == 'G');
    try testing.expect(magic_bytes[1] == 'G');
    try testing.expect(magic_bytes[2] == 'U');
    try testing.expect(magic_bytes[3] == 'F');
}

test "tensor size calculations edge cases" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test single element tensor
    var single_element = gguf.GGUFTensorInfo{
        .name = try allocator.dupe(u8, "single"),
        .n_dims = 1,
        .dimensions = try allocator.dupe(u64, &[_]u64{1}),
        .type = .f32,
        .offset = 0,
    };

    try testing.expect(single_element.elementCount() == 1);
    try testing.expect(single_element.sizeInBytes() == 4);

    // Test large tensor with quantization
    var large_tensor = gguf.GGUFTensorInfo{
        .name = try allocator.dupe(u8, "large"),
        .n_dims = 2,
        .dimensions = try allocator.dupe(u64, &[_]u64{ 10000, 10000 }),
        .type = .q4_0,
        .offset = 0,
    };

    const elements = large_tensor.elementCount();
    try testing.expect(elements == 100_000_000);

    const size = large_tensor.sizeInBytes();
    const expected_blocks = (elements + 31) / 32; // Round up to nearest block
    const expected_size = expected_blocks * 20; // 20 bytes per Q4_0 block
    try testing.expect(size == expected_size);
}

test "memory management and cleanup" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test that GGUF values clean up properly
    var string_value = gguf.GGUFValue{ .string = try allocator.dupe(u8, "test string") };
    var array_data = try allocator.alloc(u8, 100);
    var array_value = gguf.GGUFValue{
        .array = gguf.GGUFArray{
            .type = .uint8,
            .len = 100,
            .data = array_data,
        },
    };

    // These should not crash when cleaning up
    string_value.deinit(allocator);
    array_value.deinit(allocator);

    // Primitive types should be no-op for cleanup
    var int_value = gguf.GGUFValue{ .uint32 = 42 };
    int_value.deinit(allocator); // Should be safe
};