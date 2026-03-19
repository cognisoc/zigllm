// Comprehensive Model Architecture Tests
// Tests for all implemented model architectures

const std = @import("std");
const testing = std.testing;
const Tensor = @import("../src/foundation/tensor.zig").Tensor;

// Import all model architectures
const llama = @import("../src/models/llama.zig");
const gpt2 = @import("../src/models/gpt2.zig");
const mistral = @import("../src/models/mistral.zig");
const falcon = @import("../src/models/falcon.zig");
const qwen = @import("../src/models/qwen.zig");
const phi = @import("../src/models/phi.zig");
const gptj = @import("../src/models/gptj.zig");
const gpt_neox = @import("../src/models/gpt_neox.zig");
const bloom = @import("../src/models/bloom.zig");

test "LLaMA model architecture" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = llama.LlamaConfig.LLAMA2_7B;
    var model = try llama.LlamaModel.init(allocator, config);
    defer model.deinit();

    // Test forward pass
    const input_ids = [_]u32{ 1, 2, 3, 4, 5 };
    var output = try model.forward(&input_ids, 0);
    defer output.deinit();

    // Verify output shape: [batch_size=1, seq_len=5, vocab_size]
    try testing.expect(output.shape.len == 3);
    try testing.expect(output.shape[0] == 1);
    try testing.expect(output.shape[1] == 5);
    try testing.expect(output.shape[2] == config.vocab_size);

    // Test parameter calculation
    const params = llama.LlamaUtils.calculateParameters(config);
    try testing.expect(params > 6_500_000_000); // Should be ~7B parameters
    try testing.expect(params < 7_500_000_000);
}

test "GPT-2 model architecture" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = gpt2.GPT2Config.GPT2_SMALL;
    var model = try gpt2.GPT2Model.init(allocator, config);
    defer model.deinit();

    // Test forward pass
    const input_ids = [_]u32{ 1, 2, 3, 4 };
    var output = try model.forward(&input_ids, 0);
    defer output.deinit();

    // Verify output shape
    try testing.expect(output.shape.len == 3);
    try testing.expect(output.shape[0] == 1);
    try testing.expect(output.shape[1] == 4);
    try testing.expect(output.shape[2] == config.vocab_size);

    // Test parameter calculation
    const params = gpt2.GPT2Utils.calculateParameters(config);
    try testing.expect(params > 120_000_000); // Should be ~124M parameters
    try testing.expect(params < 130_000_000);
}

test "Mistral model architecture" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = mistral.MistralConfig.MISTRAL_7B_V0_1;
    var model = try mistral.MistralModel.init(allocator, config);
    defer model.deinit();

    // Test forward pass
    const input_ids = [_]u32{ 1, 2, 3 };
    var output = try model.forward(&input_ids, 0);
    defer output.deinit();

    // Verify output shape
    try testing.expect(output.shape.len == 3);
    try testing.expect(output.shape[0] == 1);
    try testing.expect(output.shape[1] == 3);
    try testing.expect(output.shape[2] == config.vocab_size);

    // Test sliding window attention configuration
    try testing.expect(config.sliding_window == 4096);
    try testing.expect(config.n_kv_heads == 8); // Grouped query attention
}

test "Falcon model architecture" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = falcon.FalconConfig.FALCON_7B;
    var model = try falcon.FalconModel.init(allocator, config);
    defer model.deinit();

    // Test forward pass
    const input_ids = [_]u32{ 1, 2 };
    var output = try model.forward(&input_ids, 0);
    defer output.deinit();

    // Verify output shape
    try testing.expect(output.shape.len == 3);
    try testing.expect(output.shape[0] == 1);
    try testing.expect(output.shape[1] == 2);
    try testing.expect(output.shape[2] == config.vocab_size);

    // Test multi-query attention configuration
    try testing.expect(config.n_kv_heads == 1); // Multi-query attention
    try testing.expect(config.parallel_attn); // Parallel attention and MLP
}

test "Qwen model architecture" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = qwen.QwenConfig.QWEN_7B;
    var model = try qwen.QwenModel.init(allocator, config);
    defer model.deinit();

    // Test forward pass
    const input_ids = [_]u32{ 1, 2, 3 };
    var output = try model.forward(&input_ids, 0);
    defer output.deinit();

    // Verify output shape
    try testing.expect(output.shape.len == 3);
    try testing.expect(output.shape[0] == 1);
    try testing.expect(output.shape[1] == 3);
    try testing.expect(output.shape[2] == config.vocab_size);

    // Test Qwen-specific features
    try testing.expect(config.use_dynamic_ntk); // Dynamic NTK scaling
    try testing.expect(config.use_logn_attn); // LogN attention scaling
}

test "Phi model architecture" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = phi.PhiConfig.PHI_2;
    var model = try phi.PhiModel.init(allocator, config);
    defer model.deinit();

    // Test forward pass
    const input_ids = [_]u32{ 1, 2 };
    var output = try model.forward(&input_ids, 0);
    defer output.deinit();

    // Verify output shape
    try testing.expect(output.shape.len == 3);
    try testing.expect(output.shape[0] == 1);
    try testing.expect(output.shape[1] == 2);
    try testing.expect(output.shape[2] == config.vocab_size);

    // Test partial rotary embeddings
    try testing.expect(config.partial_rotary_factor == 0.5); // 50% partial RoPE
}

test "GPT-J model architecture" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = gptj.GPTJConfig.GPT_J_6B;
    var model = try gptj.GPTJModel.init(allocator, config);
    defer model.deinit();

    // Test forward pass
    const input_ids = [_]u32{ 1, 2, 3 };
    var output = try model.forward(&input_ids, 0);
    defer output.deinit();

    // Verify output shape
    try testing.expect(output.shape.len == 3);
    try testing.expect(output.shape[0] == 1);
    try testing.expect(output.shape[1] == 3);
    try testing.expect(output.shape[2] == config.vocab_size);

    // Test parameter calculation
    const params = gptj.GPTJUtils.calculateParameters(config);
    try testing.expect(params > 5_500_000_000); // Should be ~6B parameters
    try testing.expect(params < 6_500_000_000);

    // Test parallel residual connections
    try testing.expect(config.use_parallel_residual);
}

test "GPT-NeoX model architecture" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = gpt_neox.GPTNeoXConfig.GPT_NEOX_1_3B;
    var model = try gpt_neox.GPTNeoXModel.init(allocator, config);
    defer model.deinit();

    // Test forward pass
    const input_ids = [_]u32{ 1, 2 };
    var output = try model.forward(&input_ids, 0);
    defer output.deinit();

    // Verify output shape
    try testing.expect(output.shape.len == 3);
    try testing.expect(output.shape[0] == 1);
    try testing.expect(output.shape[1] == 2);
    try testing.expect(output.shape[2] == config.vocab_size);

    // Test parameter calculation
    const params = gpt_neox.GPTNeoXUtils.calculateParameters(config);
    try testing.expect(params > 1_200_000_000); // Should be ~1.3B parameters
    try testing.expect(params < 1_400_000_000);

    // Test RoPE scaling
    try testing.expect(config.rope_base == 10000.0);
}

test "BLOOM model architecture" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = bloom.BloomConfig.BLOOM_560M;
    var model = try bloom.BloomModel.init(allocator, config);
    defer model.deinit();

    // Test forward pass
    const input_ids = [_]u32{ 1, 2, 3 };
    var output = try model.forward(&input_ids);
    defer output.deinit();

    // Verify output shape
    try testing.expect(output.shape.len == 3);
    try testing.expect(output.shape[0] == 1);
    try testing.expect(output.shape[1] == 3);
    try testing.expect(output.shape[2] == config.vocab_size);

    // Test ALiBi attention
    try testing.expect(config.use_alibi);

    // Test parameter calculation
    const params = bloom.BloomUtils.calculateParameters(config);
    try testing.expect(params > 500_000_000); // Should be ~560M parameters
    try testing.expect(params < 620_000_000);
}

test "ALiBi attention bias computation" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var alibi = try bloom.ALiBi.init(allocator, 8, 16);
    defer alibi.deinit();

    // Test bias tensor shape
    var bias = try alibi.getBias(4);
    try testing.expect(bias.shape.len == 3);
    try testing.expect(bias.shape[0] == 8); // n_heads
    try testing.expect(bias.shape[1] == 4); // seq_len
    try testing.expect(bias.shape[2] == 4); // seq_len

    // Test causal masking (future tokens should be -inf)
    const future_bias = try bias.get(&[_]u32{ 0, 0, 3 }); // position 0 looking at future position 3
    try testing.expect(future_bias == -std.math.inf(f32));

    // Test that slopes are negative powers of 2
    try testing.expect(alibi.slopes.len == 8);
    for (alibi.slopes) |slope| {
        try testing.expect(slope > 0.0);
        try testing.expect(slope <= 1.0);
    }
}

test "RoPE (Rotary Positional Embedding) functionality" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test GPT-J RoPE
    var rope = try gptj.RotaryEmbedding.init(allocator, 64, 2048);
    defer rope.deinit();

    // Create test tensor: [batch=1, seq=4, heads=2, head_dim=32]
    var x = try Tensor.ones(allocator, &[_]u32{ 1, 4, 2, 32 });
    defer x.deinit();

    // Apply RoPE - should not error
    try rope.apply(&x, 0);

    // Test that values changed (RoPE rotation applied)
    const val = try x.get(&[_]u32{ 0, 0, 0, 0 });
    try testing.expect(val != 1.0); // Should be rotated
}

test "Model generation functionality" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test with smaller GPT-2 model for efficiency
    const config = gpt2.GPT2Config{
        .vocab_size = 1000,
        .n_ctx = 16,
        .n_embd = 64,
        .n_head = 4,
        .n_layer = 2,
    };

    var model = try gpt2.GPT2Model.init(allocator, config);
    defer model.deinit();

    // Test generation
    const prompt = [_]u32{ 1, 2, 3 };
    var generated = try model.generate(&prompt, 8, 1.0);
    defer allocator.free(generated);

    // Verify generation properties
    try testing.expect(generated.len == 8);
    try testing.expect(generated[0] == 1); // Should preserve prompt
    try testing.expect(generated[1] == 2);
    try testing.expect(generated[2] == 3);

    // Check that new tokens were generated
    for (generated[3..]) |token| {
        try testing.expect(token < config.vocab_size);
    }
}

test "Cross-architecture consistency" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test that similar configurations produce similar parameter counts
    const llama_config = llama.LlamaConfig{
        .vocab_size = 32000,
        .n_ctx = 2048,
        .n_embd = 4096,
        .n_head = 32,
        .n_layer = 32,
        .intermediate_size = 11008,
    };

    const mistral_config = mistral.MistralConfig{
        .vocab_size = 32000,
        .n_ctx = 2048,
        .n_embd = 4096,
        .n_head = 32,
        .n_layer = 32,
        .intermediate_size = 14336, // Mistral uses larger intermediate size
        .n_kv_heads = 8,
        .sliding_window = 4096,
    };

    const llama_params = llama.LlamaUtils.calculateParameters(llama_config);
    const mistral_params = mistral.MistralUtils.calculateParameters(mistral_config);

    // Both should be in similar parameter ranges for 7B-class models
    try testing.expect(llama_params > 6_000_000_000);
    try testing.expect(llama_params < 8_000_000_000);
    try testing.expect(mistral_params > 6_000_000_000);
    try testing.expect(mistral_params < 8_000_000_000);
}

test "Model configuration variants" {
    // Test that all model families have multiple size variants

    // LLaMA variants
    const llama_7b = llama.LlamaConfig.LLAMA2_7B;
    const llama_13b = llama.LlamaConfig.LLAMA2_13B;
    try testing.expect(llama_13b.n_embd > llama_7b.n_embd);
    try testing.expect(llama_13b.n_layer > llama_7b.n_layer);

    // GPT-2 variants
    const gpt2_small = gpt2.GPT2Config.GPT2_SMALL;
    const gpt2_large = gpt2.GPT2Config.GPT2_LARGE;
    try testing.expect(gpt2_large.n_embd > gpt2_small.n_embd);
    try testing.expect(gpt2_large.n_layer > gpt2_small.n_layer);

    // GPT-NeoX variants
    const neox_410m = gpt_neox.GPTNeoXConfig.GPT_NEOX_410M;
    const neox_20b = gpt_neox.GPTNeoXConfig.GPT_NEOX_20B;
    try testing.expect(neox_20b.n_embd > neox_410m.n_embd);
    try testing.expect(neox_20b.n_layer > neox_410m.n_layer);

    // BLOOM variants
    const bloom_560m = bloom.BloomConfig.BLOOM_560M;
    const bloom_176b = bloom.BloomConfig.BLOOM_176B;
    try testing.expect(bloom_176b.n_embd > bloom_560m.n_embd);
    try testing.expect(bloom_176b.n_layer > bloom_560m.n_layer);
}

test "Architecture-specific features" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test Mistral's grouped query attention
    const mistral_config = mistral.MistralConfig.MISTRAL_7B_V0_1;
    try testing.expect(mistral_config.n_kv_heads < mistral_config.n_head);
    try testing.expect(mistral_config.sliding_window > 0);

    // Test Falcon's multi-query attention
    const falcon_config = falcon.FalconConfig.FALCON_7B;
    try testing.expect(falcon_config.n_kv_heads == 1); // Multi-query attention
    try testing.expect(falcon_config.parallel_attn); // Parallel attention and MLP

    // Test Qwen's dynamic features
    const qwen_config = qwen.QwenConfig.QWEN_7B;
    try testing.expect(qwen_config.use_dynamic_ntk);
    try testing.expect(qwen_config.use_logn_attn);

    // Test Phi's partial RoPE
    const phi_config = phi.PhiConfig.PHI_2;
    try testing.expect(phi_config.partial_rotary_factor > 0.0);
    try testing.expect(phi_config.partial_rotary_factor < 1.0);

    // Test BLOOM's ALiBi
    const bloom_config = bloom.BloomConfig.BLOOM_560M;
    try testing.expect(bloom_config.use_alibi);
    try testing.expect(bloom_config.vocab_size > 200000); // Large multilingual vocab
}