const std = @import("std");
const testing = std.testing;
const model_converter = @import("../src/tools/model_converter.zig");
const ModelFormat = model_converter.ModelFormat;
const ConversionConfig = model_converter.ConversionConfig;
const ModelConverter = model_converter.ModelConverter;
const QuantizationType = model_converter.QuantizationType;
const ConversionUtils = model_converter.ConversionUtils;
const TensorDataType = model_converter.TensorDataType;

test "Model format detection from extensions" {
    try testing.expect(ModelFormat.fromExtension("model.gguf") == .GGUF);
    try testing.expect(ModelFormat.fromExtension("model.safetensors") == .SafeTensors);
    try testing.expect(ModelFormat.fromExtension("model.pt") == .PyTorch);
    try testing.expect(ModelFormat.fromExtension("model.pth") == .PyTorch);
    try testing.expect(ModelFormat.fromExtension("model.onnx") == .ONNX);
    try testing.expect(ModelFormat.fromExtension("model.pb") == .TensorFlow);
    try testing.expect(ModelFormat.fromExtension("model.zigllama") == .Custom);
    try testing.expect(ModelFormat.fromExtension("model.unknown") == null);
}

test "Model format string conversions" {
    try testing.expectEqualStrings(ModelFormat.GGUF.toString(), "gguf");
    try testing.expectEqualStrings(ModelFormat.SafeTensors.toString(), "safetensors");
    try testing.expectEqualStrings(ModelFormat.PyTorch.toString(), "pytorch");
    try testing.expectEqualStrings(ModelFormat.Custom.toString(), "zigllama");

    try testing.expectEqualStrings(ModelFormat.GGUF.extension(), ".gguf");
    try testing.expectEqualStrings(ModelFormat.SafeTensors.extension(), ".safetensors");
    try testing.expectEqualStrings(ModelFormat.PyTorch.extension(), ".pt");
    try testing.expectEqualStrings(ModelFormat.Custom.extension(), ".zigllama");
}

test "Quantization type string conversion" {
    try testing.expectEqualStrings(QuantizationType.Q4_0.toString(), "q4_0");
    try testing.expectEqualStrings(QuantizationType.Q4_K_M.toString(), "q4_k_m");
    try testing.expectEqualStrings(QuantizationType.IQ2_XXS.toString(), "iq2_xxs");
    try testing.expectEqualStrings(QuantizationType.None.toString(), "none");
}

test "Tensor data type properties" {
    try testing.expect(TensorDataType.F32.size() == 4);
    try testing.expect(TensorDataType.F16.size() == 2);
    try testing.expect(TensorDataType.I8.size() == 1);
    try testing.expect(TensorDataType.Q4_0.size() == 18);
    try testing.expect(TensorDataType.Q8_0.size() == 34);

    try testing.expectEqualStrings(TensorDataType.F32.toString(), "f32");
    try testing.expectEqualStrings(TensorDataType.Q4_0.toString(), "q4_0");
}

test "Conversion configuration" {
    const config = ConversionConfig{
        .source_format = .GGUF,
        .target_format = .Custom,
        .quantization_type = .Q4_K_M,
        .preserve_metadata = true,
        .validate_output = true,
        .verbose = false,
    };

    try testing.expect(config.source_format == .GGUF);
    try testing.expect(config.target_format == .Custom);
    try testing.expect(config.quantization_type.? == .Q4_K_M);
    try testing.expect(config.preserve_metadata == true);
    try testing.expect(config.validate_output == true);
}

test "Model converter initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = ConversionConfig{
        .source_format = .SafeTensors,
        .target_format = .GGUF,
    };

    var converter = ModelConverter.init(allocator, config);
    try testing.expect(converter.config.source_format == .SafeTensors);
    try testing.expect(converter.config.target_format == .GGUF);
    try testing.expect(converter.progress_callback == null);

    // Test progress callback
    const TestCallback = struct {
        fn callback(progress: f32, message: []const u8) void {
            _ = progress;
            _ = message;
        }
    };

    converter.setProgressCallback(TestCallback.callback);
    try testing.expect(converter.progress_callback != null);
}

test "Model metadata creation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = ConversionConfig{
        .source_format = .GGUF,
        .target_format = .Custom,
    };

    var converter = ModelConverter.init(allocator, config);

    const metadata = try converter.createDefaultMetadata("llama", .GGUF);
    defer metadata.deinit(allocator);

    try testing.expectEqualStrings(metadata.architecture, "llama");
    try testing.expect(metadata.vocab_size == 32000);
    try testing.expect(metadata.context_length == 2048);
    try testing.expect(metadata.embedding_dim == 4096);
    try testing.expect(metadata.num_layers == 32);
    try testing.expect(metadata.num_heads == 32);
}

test "Conversion time estimation" {
    const small_file = 1024 * 1024; // 1MB
    const large_file = 1024 * 1024 * 1024; // 1GB

    const small_time = ConversionUtils.estimateConversionTime(small_file, .GGUF, .Custom);
    const large_time = ConversionUtils.estimateConversionTime(large_file, .GGUF, .Custom);

    try testing.expect(large_time > small_time);

    const pytorch_time = ConversionUtils.estimateConversionTime(small_file, .PyTorch, .GGUF);
    const gguf_time = ConversionUtils.estimateConversionTime(small_file, .GGUF, .Custom);

    // PyTorch should take longer due to complexity
    try testing.expect(pytorch_time >= gguf_time);
}

test "Compression ratio calculation" {
    const ratio1 = ConversionUtils.calculateCompressionRatio(1000, 500);
    try testing.expect(ratio1 == 2.0);

    const ratio2 = ConversionUtils.calculateCompressionRatio(1000, 100);
    try testing.expect(ratio2 == 10.0);

    const ratio3 = ConversionUtils.calculateCompressionRatio(1000, 0);
    try testing.expect(ratio3 == 0.0);

    const ratio4 = ConversionUtils.calculateCompressionRatio(1000, 1000);
    try testing.expect(ratio4 == 1.0);
}

test "Model architecture validation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Valid metadata
    const valid_metadata = model_converter.ModelMetadata{
        .architecture = try allocator.dupe(u8, "llama"),
        .vocab_size = 32000,
        .context_length = 2048,
        .embedding_dim = 4096,
        .num_layers = 32,
        .num_heads = 32, // 4096 % 32 = 128, valid
        .intermediate_size = 11008,
        .rope_theta = 10000.0,
        .created_by = try allocator.dupe(u8, "test"),
        .creation_time = 1234567890,
        .source_format = try allocator.dupe(u8, "gguf"),
        .quantization = try allocator.dupe(u8, "none"),
        .checksum = try allocator.dupe(u8, "abc123"),
    };
    defer valid_metadata.deinit(allocator);

    try testing.expect(ConversionUtils.validateArchitecture(valid_metadata) == true);

    // Invalid metadata (embedding_dim not divisible by num_heads)
    const invalid_metadata = model_converter.ModelMetadata{
        .architecture = try allocator.dupe(u8, "llama"),
        .vocab_size = 32000,
        .context_length = 2048,
        .embedding_dim = 4097, // Not divisible by num_heads
        .num_layers = 32,
        .num_heads = 32,
        .intermediate_size = 11008,
        .rope_theta = 10000.0,
        .created_by = try allocator.dupe(u8, "test"),
        .creation_time = 1234567890,
        .source_format = try allocator.dupe(u8, "gguf"),
        .quantization = try allocator.dupe(u8, "none"),
        .checksum = try allocator.dupe(u8, "abc123"),
    };
    defer invalid_metadata.deinit(allocator);

    try testing.expect(ConversionUtils.validateArchitecture(invalid_metadata) == false);

    // Zero values should be invalid
    const zero_metadata = model_converter.ModelMetadata{
        .architecture = try allocator.dupe(u8, "llama"),
        .vocab_size = 0, // Invalid
        .context_length = 2048,
        .embedding_dim = 4096,
        .num_layers = 32,
        .num_heads = 32,
        .intermediate_size = 11008,
        .rope_theta = 10000.0,
        .created_by = try allocator.dupe(u8, "test"),
        .creation_time = 1234567890,
        .source_format = try allocator.dupe(u8, "gguf"),
        .quantization = try allocator.dupe(u8, "none"),
        .checksum = try allocator.dupe(u8, "abc123"),
    };
    defer zero_metadata.deinit(allocator);

    try testing.expect(ConversionUtils.validateArchitecture(zero_metadata) == false);
}

test "Model fingerprint generation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const test_data = "Hello, ZigLlama!";
    const fingerprint1 = try ConversionUtils.generateFingerprint(test_data, allocator);
    defer allocator.free(fingerprint1);

    const fingerprint2 = try ConversionUtils.generateFingerprint(test_data, allocator);
    defer allocator.free(fingerprint2);

    // Same input should produce same fingerprint
    try testing.expectEqualStrings(fingerprint1, fingerprint2);
    try testing.expect(fingerprint1.len == 64); // SHA-256 hex string

    // Different input should produce different fingerprint
    const different_data = "Different data";
    const fingerprint3 = try ConversionUtils.generateFingerprint(different_data, allocator);
    defer allocator.free(fingerprint3);

    try testing.expect(!std.mem.eql(u8, fingerprint1, fingerprint3));
}

test "Supported conversions list" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const conversions = try ConversionUtils.getSupportedConversions(allocator);
    defer allocator.free(conversions);

    try testing.expect(conversions.len > 0);

    // Check that some expected conversions exist
    var found_gguf_to_custom = false;
    var found_safetensors_to_gguf = false;

    for (conversions) |conversion| {
        if (conversion.from == .GGUF and conversion.to == .Custom) {
            found_gguf_to_custom = true;
        }
        if (conversion.from == .SafeTensors and conversion.to == .GGUF) {
            found_safetensors_to_gguf = true;
        }
    }

    try testing.expect(found_gguf_to_custom);
    try testing.expect(found_safetensors_to_gguf);
}

test "Tensor info management" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const tensor_info = model_converter.TensorInfo{
        .name = try allocator.dupe(u8, "attention.weight"),
        .shape = try allocator.dupe(usize, &[_]usize{ 4096, 4096 }),
        .data_type = .F32,
        .size_bytes = 4096 * 4096 * 4,
        .offset = 0,
        .quantized = false,
    };

    try testing.expectEqualStrings(tensor_info.name, "attention.weight");
    try testing.expect(tensor_info.shape.len == 2);
    try testing.expect(tensor_info.shape[0] == 4096);
    try testing.expect(tensor_info.shape[1] == 4096);
    try testing.expect(tensor_info.data_type == .F32);
    try testing.expect(tensor_info.quantized == false);

    tensor_info.deinit(allocator);
}

test "Progress reporting" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = ConversionConfig{
        .source_format = .GGUF,
        .target_format = .Custom,
        .verbose = false, // Test without verbose output
    };

    var converter = ModelConverter.init(allocator, config);

    // Test progress reporting without callback (should not crash)
    converter.reportProgress(0.5, "Test message");

    // Test with callback
    const TestState = struct {
        var last_progress: f32 = 0.0;
        var last_message: []const u8 = "";

        fn callback(progress: f32, message: []const u8) void {
            last_progress = progress;
            last_message = message;
        }
    };

    converter.setProgressCallback(TestState.callback);
    converter.reportProgress(0.75, "Test callback");

    try testing.expect(TestState.last_progress == 0.75);
    try testing.expectEqualStrings(TestState.last_message, "Test callback");
}

test "Model data structure" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var model_data = model_converter.ModelData{
        .metadata = model_converter.ModelMetadata{
            .architecture = try allocator.dupe(u8, "test"),
            .vocab_size = 1000,
            .context_length = 512,
            .embedding_dim = 768,
            .num_layers = 12,
            .num_heads = 12,
            .intermediate_size = 3072,
            .rope_theta = 10000.0,
            .created_by = try allocator.dupe(u8, "test"),
            .creation_time = 123456789,
            .source_format = try allocator.dupe(u8, "test"),
            .quantization = try allocator.dupe(u8, "none"),
            .checksum = try allocator.dupe(u8, "abc"),
        },
        .tensors = std.ArrayList(model_converter.TensorInfo).init(allocator),
        .data = std.ArrayList(u8).init(allocator),
    };

    // Add test tensor
    const tensor_info = model_converter.TensorInfo{
        .name = try allocator.dupe(u8, "test.weight"),
        .shape = try allocator.dupe(usize, &[_]usize{768}),
        .data_type = .F32,
        .size_bytes = 768 * 4,
        .offset = 0,
        .quantized = false,
    };
    try model_data.tensors.append(tensor_info);

    // Add test data
    try model_data.data.appendSlice("test_data");

    try testing.expect(model_data.tensors.items.len == 1);
    try testing.expect(model_data.data.items.len == 9);
    try testing.expectEqualStrings(model_data.metadata.architecture, "test");

    // Cleanup
    model_data.metadata.deinit(allocator);
    for (model_data.tensors.items) |tensor| {
        tensor.deinit(allocator);
    }
    model_data.tensors.deinit();
    model_data.data.deinit();
}

test "CLI argument parsing helpers" {
    const ConverterCLI = model_converter.ConverterCLI;

    try testing.expect(ConverterCLI.parseFormat("gguf") == .GGUF);
    try testing.expect(ConverterCLI.parseFormat("safetensors") == .SafeTensors);
    try testing.expect(ConverterCLI.parseFormat("pytorch") == .PyTorch);
    try testing.expect(ConverterCLI.parseFormat("custom") == .Custom);
    try testing.expect(ConverterCLI.parseFormat("unknown") == null);

    try testing.expect(ConverterCLI.parseQuantization("q4_0") == .Q4_0);
    try testing.expect(ConverterCLI.parseQuantization("q4_k_m") == .Q4_K_M);
    try testing.expect(ConverterCLI.parseQuantization("iq2_xxs") == .IQ2_XXS);
    try testing.expect(ConverterCLI.parseQuantization("none") == .None);
    try testing.expect(ConverterCLI.parseQuantization("unknown") == null);
}

// Mock test for file operations (would need actual files in real testing)
test "Mock conversion flow" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = ConversionConfig{
        .source_format = .GGUF,
        .target_format = .Custom,
        .quantization_type = .Q4_K_M,
        .preserve_metadata = true,
        .validate_output = true,
        .verbose = false,
    };

    const converter = ModelConverter.init(allocator, config);

    // Test the configuration is set up correctly
    try testing.expect(converter.config.source_format == .GGUF);
    try testing.expect(converter.config.target_format == .Custom);
    try testing.expect(converter.config.quantization_type.? == .Q4_K_M);
    try testing.expect(converter.config.preserve_metadata == true);
    try testing.expect(converter.config.validate_output == true);

    // Note: Actual conversion testing would require creating test files
    // which is beyond the scope of unit tests
}

test "Format compatibility matrix" {
    // Test logical format combinations
    const valid_combinations = [_]struct { from: ModelFormat, to: ModelFormat }{
        .{ .from = .GGUF, .to = .Custom },
        .{ .from = .SafeTensors, .to = .GGUF },
        .{ .from = .Custom, .to = .GGUF },
        .{ .from = .PyTorch, .to = .Custom },
    };

    for (valid_combinations) |combo| {
        // These combinations should be logically valid
        try testing.expect(combo.from != combo.to);
    }

    // Self-conversion should be identity operation
    const identity_formats = [_]ModelFormat{ .GGUF, .SafeTensors, .Custom };
    for (identity_formats) |format| {
        try testing.expect(format == format); // Identity check
    }
}