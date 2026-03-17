//! Comprehensive tests for Models layer
//!
//! This test suite validates the complete model implementations:
//! LLaMA architecture, configuration system, tokenization, and GGUF support.

const std = @import("std");
const testing = std.testing;
const math = std.math;

// Import model components
const ModelConfig = @import("../src/models/config.zig").ModelConfig;
const ModelSize = @import("../src/models/config.zig").ModelSize;
const ActivationType = @import("../src/models/config.zig").ActivationType;
const NormalizationType = @import("../src/models/config.zig").NormalizationType;
const PositionEncodingType = @import("../src/models/config.zig").PositionEncodingType;

const Vocabulary = @import("../src/models/tokenizer.zig").Vocabulary;
const SimpleTokenizer = @import("../src/models/tokenizer.zig").SimpleTokenizer;
const TokenId = @import("../src/models/tokenizer.zig").TokenId;
const SpecialTokens = @import("../src/models/tokenizer.zig").SpecialTokens;
const TokenizerStats = @import("../src/models/tokenizer.zig").TokenizerStats;

const GGUFHeader = @import("../src/models/gguf.zig").GGUFHeader;
const GGMLType = @import("../src/models/gguf.zig").GGMLType;
const GGUFTensorInfo = @import("../src/models/gguf.zig").GGUFTensorInfo;
const GGUF_MAGIC = @import("../src/models/gguf.zig").GGUF_MAGIC;
const GGUF_VERSION = @import("../src/models/gguf.zig").GGUF_VERSION;

test "model configuration validation and scaling" {
    // Test configuration creation
    const llama_7b = ModelConfig.llama(.LLaMA_7B);
    const llama_13b = ModelConfig.llama(.LLaMA_13B);
    const code_llama = ModelConfig.llama(.CodeLlama_7B);

    // Validate configurations
    try llama_7b.validate();
    try llama_13b.validate();
    try code_llama.validate();

    // Test architectural scaling
    try testing.expect(llama_13b.d_model > llama_7b.d_model);
    try testing.expect(llama_13b.num_layers > llama_7b.num_layers);
    try testing.expect(code_llama.max_seq_len > llama_7b.max_seq_len);

    // Test parameter counting
    const params_7b = llama_7b.parameterCount();
    const params_13b = llama_13b.parameterCount();
    try testing.expect(params_13b > params_7b);

    // LLaMA-7B should have approximately 6.7B parameters
    try testing.expect(params_7b >= 6_000_000_000);
    try testing.expect(params_7b <= 8_000_000_000);
}

test "model configuration head dimension consistency" {
    const config = ModelConfig.llama(.LLaMA_7B);

    // Head dimension should divide evenly
    const head_dim = config.headDim();
    try testing.expectEqual(config.d_model, config.num_heads * head_dim);
    try testing.expectEqual(@as(usize, 128), head_dim); // LLaMA-7B uses 128 per head
}

test "model configuration memory estimation" {
    const config = ModelConfig.llama(.LLaMA_7B);
    const memory = config.memoryRequirements(1, 512);

    // Check memory components are reasonable
    try testing.expect(memory.parameters > 0);
    try testing.expect(memory.activations > 0);
    try testing.expect(memory.kv_cache > 0);
    try testing.expect(memory.total >= memory.parameters);

    // Parameter memory should be largest component for small batches
    try testing.expect(memory.parameters > memory.activations);
}

test "activation type parameter scaling" {
    // Test parameter multipliers for different activations
    try testing.expectEqual(@as(f32, 2.0), ActivationType.ReLU.parameterMultiplier());
    try testing.expectEqual(@as(f32, 2.0), ActivationType.GELU.parameterMultiplier());
    try testing.expectEqual(@as(f32, 3.0), ActivationType.SwiGLU.parameterMultiplier());
    try testing.expectEqual(@as(f32, 3.0), ActivationType.GeGLU.parameterMultiplier());
    try testing.expectEqual(@as(f32, 3.0), ActivationType.GLU.parameterMultiplier());
}

test "custom model configuration" {
    const custom = ModelConfig.custom(512, 6, 8, 10000);
    try custom.validate();

    // Check defaults are applied
    try testing.expectEqual(@as(usize, 512 * 4), custom.d_ff); // 4x scaling
    try testing.expectEqual(ActivationType.SwiGLU, custom.activation);
    try testing.expectEqual(NormalizationType.RMSNorm, custom.normalization);
    try testing.expectEqual(PositionEncodingType.RoPE, custom.position_encoding);
}

test "vocabulary basic operations" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var vocab = try Vocabulary.init(allocator, 100);
    defer vocab.deinit();

    // Test special tokens exist
    try testing.expect(vocab.getTokenId("<unk>") != null);
    try testing.expect(vocab.getTokenId("<s>") != null);
    try testing.expect(vocab.getTokenId("</s>") != null);

    // Test adding tokens
    try vocab.addToken("hello", -1.0, 10, false);
    try testing.expectEqual(@as(?TokenId, 10), vocab.getTokenId("hello"));

    const piece = vocab.getTokenPiece(10);
    try testing.expect(piece != null);
    try testing.expectEqualStrings("hello", piece.?.piece);
}

test "simple tokenizer encode/decode" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var tokenizer = try SimpleTokenizer.init(allocator, 100);
    defer tokenizer.deinit();

    // Add some test tokens
    try tokenizer.vocabulary.addToken("hello", -1.0, 4, false);
    try tokenizer.vocabulary.addToken("world", -1.5, 5, false);

    // Test encoding
    const tokens = try tokenizer.encode("hello world");
    defer allocator.free(tokens);

    // Should have BOS, tokens, EOS
    try testing.expect(tokens.len >= 3);
    try testing.expectEqual(SpecialTokens.BOS, tokens[0]);
    try testing.expectEqual(SpecialTokens.EOS, tokens[tokens.len - 1]);

    // Test decoding
    const decoded = try tokenizer.decode(tokens);
    defer allocator.free(decoded);

    // Should contain original words
    try testing.expect(std.mem.indexOf(u8, decoded, "hello") != null);
    try testing.expect(std.mem.indexOf(u8, decoded, "world") != null);
}

test "special token handling" {
    // Test special token identification
    try testing.expect(SpecialTokens.isSpecial(SpecialTokens.UNK));
    try testing.expect(SpecialTokens.isSpecial(SpecialTokens.BOS));
    try testing.expect(SpecialTokens.isSpecial(SpecialTokens.EOS));
    try testing.expect(SpecialTokens.isSpecial(SpecialTokens.PAD));
    try testing.expect(!SpecialTokens.isSpecial(100));

    // Test special token names
    try testing.expectEqualStrings("<unk>", SpecialTokens.name(SpecialTokens.UNK).?);
    try testing.expectEqualStrings("<s>", SpecialTokens.name(SpecialTokens.BOS).?);
    try testing.expectEqualStrings("</s>", SpecialTokens.name(SpecialTokens.EOS).?);
    try testing.expectEqualStrings("<pad>", SpecialTokens.name(SpecialTokens.PAD).?);
    try testing.expect(SpecialTokens.name(100) == null);
}

test "batch tokenization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var tokenizer = try SimpleTokenizer.init(allocator, 100);
    defer tokenizer.deinit();

    const texts = [_][]const u8{ "first text", "second text", "third text" };
    const encoded = try tokenizer.batchEncode(&texts);
    defer {
        for (encoded) |tokens| {
            allocator.free(tokens);
        }
        allocator.free(encoded);
    }

    // All should be encoded
    try testing.expectEqual(@as(usize, 3), encoded.len);
    for (encoded) |tokens| {
        try testing.expect(tokens.len >= 2); // BOS + EOS minimum
        try testing.expectEqual(SpecialTokens.BOS, tokens[0]);
        try testing.expectEqual(SpecialTokens.EOS, tokens[tokens.len - 1]);
    }
}

test "tokenizer statistics analysis" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var vocab = try Vocabulary.createSimpleVocab(allocator);
    var tokenizer = SimpleTokenizer.initWithVocab(vocab, allocator);
    defer tokenizer.deinit();

    const test_texts = [_][]const u8{ "the quick brown", "and the fox" };
    const stats = try TokenizerStats.analyze(tokenizer, &test_texts, allocator);

    // Check statistics make sense
    try testing.expect(stats.vocab_size > 10); // Should have special + test tokens
    try testing.expect(stats.avg_tokens_per_text >= 2.0); // At least BOS + EOS
    try testing.expect(stats.unknown_token_rate >= 0.0 and stats.unknown_token_rate <= 1.0);
    try testing.expect(stats.special_token_count > 0); // Should have BOS/EOS
}

test "GGUF header validation" {
    const valid_header = GGUFHeader{
        .magic = GGUF_MAGIC,
        .version = GGUF_VERSION,
        .tensor_count = 100,
        .metadata_kv_count = 50,
    };
    try valid_header.validate();

    // Test invalid magic
    const invalid_magic = GGUFHeader{
        .magic = 0x12345678,
        .version = GGUF_VERSION,
        .tensor_count = 100,
        .metadata_kv_count = 50,
    };
    try testing.expectError(error.InvalidGGUFMagic, invalid_magic.validate());

    // Test invalid version
    const invalid_version = GGUFHeader{
        .magic = GGUF_MAGIC,
        .version = 999,
        .tensor_count = 100,
        .metadata_kv_count = 50,
    };
    try testing.expectError(error.UnsupportedGGUFVersion, invalid_version.validate());
}

test "GGML type properties and calculations" {
    // Test element sizes
    try testing.expectEqual(@as(usize, 4), GGMLType.F32.elementSize());
    try testing.expectEqual(@as(usize, 2), GGMLType.F16.elementSize());
    try testing.expectEqual(@as(usize, 1), GGMLType.I8.elementSize());

    // Test quantized type detection
    try testing.expect(!GGMLType.F32.isQuantized());
    try testing.expect(!GGMLType.F16.isQuantized());
    try testing.expect(GGMLType.Q4_0.isQuantized());
    try testing.expect(GGMLType.Q8_0.isQuantized());

    // Test block sizes
    try testing.expectEqual(@as(usize, 1), GGMLType.F32.blockSize());
    try testing.expectEqual(@as(usize, 16), GGMLType.Q4_0.blockSize());
    try testing.expectEqual(@as(usize, 32), GGMLType.Q8_0.blockSize());
    try testing.expectEqual(@as(usize, 256), GGMLType.Q2_K.blockSize());
}

test "GGUF tensor info calculations" {
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

    // Test element count calculation
    const element_count = tensor_info.elementCount();
    try testing.expectEqual(@as(u64, 4096 * 4096), element_count);

    // Test size calculation
    const size_bytes = tensor_info.sizeBytes();
    try testing.expectEqual(@as(u64, 4096 * 4096 * 4), size_bytes); // f32 = 4 bytes

    // Test quantized size calculation
    var quantized_info = tensor_info;
    quantized_info.ggml_type = .Q4_0;
    const quantized_size = quantized_info.sizeBytes();
    const expected_blocks = (element_count + 15) / 16; // Q4_0 has 16-element blocks
    try testing.expectEqual(expected_blocks * 18, quantized_size); // Q4_0 block size is 18 bytes
}

test "model size parameter count accuracy" {
    // Test that model sizes have expected parameter counts
    const llama_7b = ModelConfig.llama(.LLaMA_7B);
    const llama_13b = ModelConfig.llama(.LLaMA_13B);

    const params_7b = llama_7b.parameterCount();
    const params_13b = llama_13b.parameterCount();

    // Check parameter scaling is reasonable
    try testing.expect(params_13b > params_7b);
    try testing.expect(params_13b < params_7b * 3); // Shouldn't be more than 3x

    // Check against known ballpark figures
    const expected_7b = ModelSize.LLaMA_7B.parameterCount() * 1e9;
    const expected_13b = ModelSize.LLaMA_13B.parameterCount() * 1e9;

    // Should be within 20% of expected
    const tolerance = 0.2;
    try testing.expect(@abs(@as(f32, @floatFromInt(params_7b)) - expected_7b) < expected_7b * tolerance);
    try testing.expect(@abs(@as(f32, @floatFromInt(params_13b)) - expected_13b) < expected_13b * tolerance);
}

test "memory formatting utilities" {
    var buffer: [64]u8 = undefined;

    // Test various memory sizes
    const kb = try ModelConfig.formatMemorySize(1024, &buffer);
    try testing.expect(std.mem.indexOf(u8, kb, "KB") != null);

    const mb = try ModelConfig.formatMemorySize(1024 * 1024, &buffer);
    try testing.expect(std.mem.indexOf(u8, mb, "MB") != null);

    const gb = try ModelConfig.formatMemorySize(1024 * 1024 * 1024, &buffer);
    try testing.expect(std.mem.indexOf(u8, gb, "GB") != null);
}

test "normalization and position encoding properties" {
    // Test normalization parameter requirements
    try testing.expect(NormalizationType.LayerNorm.hasParameters());
    try testing.expect(NormalizationType.RMSNorm.hasParameters());
    try testing.expect(!NormalizationType.None.hasParameters());

    // Test position encoding parameter requirements
    try testing.expect(!PositionEncodingType.None.requiresParameters());
    try testing.expect(!PositionEncodingType.Sinusoidal.requiresParameters());
    try testing.expect(PositionEncodingType.Learned.requiresParameters());
    try testing.expect(PositionEncodingType.RoPE.requiresParameters());
    try testing.expect(!PositionEncodingType.ALiBi.requiresParameters());
}

test "model configuration edge cases" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    // Test configuration with incompatible dimensions
    var invalid_config = ModelConfig.custom(100, 6, 7, 10000); // 100 not divisible by 7
    try testing.expectError(error.IncompatibleHeadDimension, invalid_config.validate());

    // Test unreasonable values
    invalid_config = ModelConfig.custom(10, 6, 8, 10000); // d_model too small
    try testing.expectError(error.UnreasonableModelDimension, invalid_config.validate());

    invalid_config = ModelConfig.custom(512, 0, 8, 10000); // zero layers
    try testing.expectError(error.UnreasonableLayerCount, invalid_config.validate());

    invalid_config = ModelConfig.custom(512, 6, 0, 10000); // zero heads
    try testing.expectError(error.UnreasonableHeadCount, invalid_config.validate());
}