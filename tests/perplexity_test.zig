const std = @import("std");
const testing = std.testing;
const math = std.math;
const perplexity = @import("../src/evaluation/perplexity.zig");
const PerplexityConfig = perplexity.PerplexityConfig;
const PerplexityEvaluator = perplexity.PerplexityEvaluator;
const EvaluationDataset = perplexity.EvaluationDataset;
const BenchmarkSuite = perplexity.BenchmarkSuite;
const PerplexityUtils = perplexity.PerplexityUtils;
const models = @import("../src/models/llama.zig");

test "Perplexity configuration initialization" {
    const config = PerplexityConfig{
        .max_sequence_length = 1024,
        .batch_size = 2,
        .sliding_window = 256,
        .window_overlap = 32,
        .use_log_probs = true,
        .temperature = 1.0,
        .normalize_probs = true,
        .verbose = false,
    };

    try testing.expect(config.max_sequence_length == 1024);
    try testing.expect(config.batch_size == 2);
    try testing.expect(config.sliding_window == 256);
    try testing.expect(config.window_overlap == 32);
    try testing.expect(config.use_log_probs == true);
    try testing.expect(config.temperature == 1.0);
    try testing.expect(config.normalize_probs == true);
    try testing.expect(config.verbose == false);
}

test "Default perplexity configuration" {
    const config = PerplexityConfig{};

    try testing.expect(config.max_sequence_length == 2048);
    try testing.expect(config.batch_size == 1);
    try testing.expect(config.sliding_window == 512);
    try testing.expect(config.window_overlap == 64);
    try testing.expect(config.use_log_probs == true);
    try testing.expect(config.temperature == 1.0);
    try testing.expect(config.normalize_probs == true);
    try testing.expect(config.verbose == false);
}

test "Evaluation dataset management" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create test sequences
    const sequences = try allocator.alloc([]const u8, 3);
    sequences[0] = try allocator.dupe(u8, "Hello world this is a test");
    sequences[1] = try allocator.dupe(u8, "Another test sequence for evaluation");
    sequences[2] = try allocator.dupe(u8, "Third sequence with different content");

    var dataset = EvaluationDataset{
        .sequences = sequences,
        .labels = null,
        .metadata = null,
    };

    try testing.expect(dataset.sequences.len == 3);
    try testing.expectEqualStrings(dataset.sequences[0], "Hello world this is a test");
    try testing.expect(dataset.labels == null);
    try testing.expect(dataset.metadata == null);

    dataset.deinit(allocator);
}

test "Dataset with labels and metadata" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const sequences = try allocator.alloc([]const u8, 2);
    sequences[0] = try allocator.dupe(u8, "Positive example");
    sequences[1] = try allocator.dupe(u8, "Negative example");

    const labels = try allocator.alloc([]const u8, 2);
    labels[0] = try allocator.dupe(u8, "positive");
    labels[1] = try allocator.dupe(u8, "negative");

    const metadata = try allocator.alloc(perplexity.DatasetMetadata, 2);
    metadata[0] = perplexity.DatasetMetadata{
        .source = "test",
        .domain = "classification",
        .language = "en",
        .length = sequences[0].len,
        .difficulty = 0.5,
    };
    metadata[1] = perplexity.DatasetMetadata{
        .source = "test",
        .domain = "classification",
        .language = "en",
        .length = sequences[1].len,
        .difficulty = 0.7,
    };

    var dataset = EvaluationDataset{
        .sequences = sequences,
        .labels = labels,
        .metadata = metadata,
    };

    try testing.expect(dataset.labels != null);
    try testing.expect(dataset.metadata != null);
    try testing.expectEqualStrings(dataset.labels.?[0], "positive");
    try testing.expect(dataset.metadata.?[0].difficulty == 0.5);

    dataset.deinit(allocator);
}

test "Perplexity result structure" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const token_log_probs = try allocator.alloc(f64, 5);
    token_log_probs[0] = -2.3;
    token_log_probs[1] = -1.8;
    token_log_probs[2] = -2.1;
    token_log_probs[3] = -2.5;
    token_log_probs[4] = -1.9;

    const sequence_perplexities = try allocator.alloc(f64, 1);
    sequence_perplexities[0] = 12.5;

    const result = perplexity.PerplexityResult{
        .perplexity = 12.5,
        .log_perplexity = 2.526,
        .bits_per_char = 1.85,
        .bits_per_token = 3.64,
        .total_tokens = 5,
        .total_chars = 25,
        .token_log_probs = token_log_probs,
        .sequence_perplexities = sequence_perplexities,
        .evaluation_time_ms = 150,
        .peak_memory_usage = 1024 * 1024,
    };

    try testing.expect(result.perplexity == 12.5);
    try testing.expect(result.total_tokens == 5);
    try testing.expect(result.token_log_probs.len == 5);
    try testing.expect(result.sequence_perplexities.len == 1);

    result.deinit(allocator);
}

test "Perplexity mathematics" {
    // Test basic perplexity calculations
    const log_probs = [_]f64{ -2.0, -1.5, -2.5, -1.8, -2.2 };
    var total_log_prob: f64 = 0;
    for (log_probs) |log_prob| {
        total_log_prob += log_prob;
    }

    const avg_log_prob = total_log_prob / @as(f64, @floatFromInt(log_probs.len));
    const log_perplexity = -avg_log_prob;
    const perplexity = math.exp(log_perplexity);

    try testing.expect(avg_log_prob == -2.0); // (-10.0 / 5)
    try testing.expect(log_perplexity == 2.0);
    try testing.expect(math.approxEqRel(f64, perplexity, 7.389, 0.001));

    const bits_per_token = log_perplexity / math.ln(2.0);
    try testing.expect(math.approxEqRel(f64, bits_per_token, 2.885, 0.001));
}

test "Tokenization simulation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create mock model and evaluator for tokenization testing
    const config = PerplexityConfig{};

    // Note: This is simplified since we can't create a real model in tests
    // In practice, would need to mock the model properly
    var mock_model: models.LLaMAModel = undefined;
    var mock_tokenizer: u32 = 0; // Placeholder

    var evaluator = PerplexityEvaluator.init(allocator, config, &mock_model, &mock_tokenizer);

    const text = "hello world test";
    const tokens = try evaluator.tokenizeText(text);
    defer allocator.free(tokens);

    try testing.expect(tokens.len > 0);
    try testing.expect(tokens.len <= 4); // Should have tokens for words
}

test "Sliding window evaluation logic" {
    // Test sliding window parameters
    const config = PerplexityConfig{
        .sliding_window = 100,
        .window_overlap = 20,
    };

    const sequence_length = 250;
    var window_start: usize = 0;
    var windows_processed: usize = 0;

    // Simulate sliding window iteration
    while (window_start < sequence_length) {
        const window_end = @min(window_start + config.sliding_window, sequence_length);
        const window_size = window_end - window_start;

        try testing.expect(window_size <= config.sliding_window);
        windows_processed += 1;

        if (window_end >= sequence_length) break;
        window_start = window_end - config.window_overlap;
    }

    try testing.expect(windows_processed >= 2); // Should process multiple windows
}

test "Confidence interval calculation" {
    const sequence_perplexities = [_]f64{ 10.0, 12.0, 8.5, 15.0, 11.0, 9.0, 13.5, 14.0, 7.5, 16.0 };

    const interval = PerplexityUtils.calculateConfidenceInterval(&sequence_perplexities, 0.95);

    try testing.expect(interval.lower < interval.upper);
    try testing.expect(interval.lower >= 7.5); // Should be near minimum
    try testing.expect(interval.upper <= 16.0); // Should be near maximum
}

test "Perplexity comparison" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const token_probs1 = try allocator.alloc(f64, 3);
    token_probs1[0] = -2.0;
    token_probs1[1] = -2.0;
    token_probs1[2] = -2.0;

    const token_probs2 = try allocator.alloc(f64, 3);
    token_probs2[0] = -1.5;
    token_probs2[1] = -1.5;
    token_probs2[2] = -1.5;

    const seq_perp1 = try allocator.alloc(f64, 1);
    seq_perp1[0] = 10.0;
    const seq_perp2 = try allocator.alloc(f64, 1);
    seq_perp2[0] = 15.0;

    const result1 = perplexity.PerplexityResult{
        .perplexity = 10.0,
        .log_perplexity = 2.3,
        .bits_per_char = 1.5,
        .bits_per_token = 3.3,
        .total_tokens = 100,
        .total_chars = 500,
        .token_log_probs = token_probs1,
        .sequence_perplexities = seq_perp1,
        .evaluation_time_ms = 100,
        .peak_memory_usage = 1024,
    };

    const result2 = perplexity.PerplexityResult{
        .perplexity = 15.0,
        .log_perplexity = 2.7,
        .bits_per_char = 1.8,
        .bits_per_token = 3.9,
        .total_tokens = 100,
        .total_chars = 500,
        .token_log_probs = token_probs2,
        .sequence_perplexities = seq_perp2,
        .evaluation_time_ms = 120,
        .peak_memory_usage = 1024,
    };

    const comparison = PerplexityUtils.compareResults(result1, result2);

    try testing.expect(comparison.absolute_difference == -5.0); // 10.0 - 15.0
    try testing.expect(comparison.relative_difference < 0); // First is better (lower perplexity)
    try testing.expect(comparison.better_result == .First);

    result1.deinit(allocator);
    result2.deinit(allocator);
}

test "Standard benchmark identifiers" {
    try testing.expectEqualStrings(perplexity.StandardBenchmarks.WIKITEXT_103, "wikitext-103");
    try testing.expectEqualStrings(perplexity.StandardBenchmarks.PENN_TREEBANK, "penn-treebank");
    try testing.expectEqualStrings(perplexity.StandardBenchmarks.LAMBADA, "lambada");
    try testing.expectEqualStrings(perplexity.StandardBenchmarks.HELLASWAG, "hellaswag");
    try testing.expectEqualStrings(perplexity.StandardBenchmarks.SYNTHETIC_SMALL, "synthetic-small");
    try testing.expectEqualStrings(perplexity.StandardBenchmarks.SYNTHETIC_LARGE, "synthetic-large");
}

test "Temperature scaling effects" {
    const base_log_probs = [_]f64{ -2.0, -1.5, -2.5 };

    // Temperature < 1.0 should make probabilities sharper (more confident)
    const temp_05 = 0.5;
    var scaled_probs_05: [3]f64 = undefined;
    for (base_log_probs, 0..) |log_prob, i| {
        scaled_probs_05[i] = log_prob / temp_05;
    }

    // Temperature > 1.0 should make probabilities softer (less confident)
    const temp_20 = 2.0;
    var scaled_probs_20: [3]f64 = undefined;
    for (base_log_probs, 0..) |log_prob, i| {
        scaled_probs_20[i] = log_prob / temp_20;
    }

    // With temp < 1.0, log probs should be more extreme (further from 0)
    try testing.expect(@abs(scaled_probs_05[0]) > @abs(base_log_probs[0]));

    // With temp > 1.0, log probs should be less extreme (closer to 0)
    try testing.expect(@abs(scaled_probs_20[0]) < @abs(base_log_probs[0]));
}

test "Perplexity bounds validation" {
    // Perplexity should always be >= 1.0
    const perfect_log_prob = 0.0; // log(1.0) = 0.0 → perfect prediction
    const perfect_perplexity = math.exp(-perfect_log_prob);
    try testing.expect(perfect_perplexity == 1.0);

    // Very uncertain predictions should have high perplexity
    const uncertain_log_prob = -10.0; // Very low probability
    const uncertain_perplexity = math.exp(-uncertain_log_prob);
    try testing.expect(uncertain_perplexity > 20000.0);

    // Reasonable predictions should have moderate perplexity
    const reasonable_log_prob = -2.3; // About 10% probability
    const reasonable_perplexity = math.exp(-reasonable_log_prob);
    try testing.expect(reasonable_perplexity > 5.0 and reasonable_perplexity < 15.0);
}

test "Bits per token calculation" {
    const log_perplexity = 2.0; // Perplexity = e^2 ≈ 7.39
    const bits_per_token = log_perplexity / math.ln(2.0);

    // Should be approximately 2.88 bits per token
    try testing.expect(math.approxEqRel(f64, bits_per_token, 2.885, 0.01));

    // Relationship check: if perplexity = 2^bits, then bits = log2(perplexity)
    const perplexity = math.exp(log_perplexity);
    const expected_bits = math.log2(perplexity);
    try testing.expect(math.approxEqRel(f64, bits_per_token, expected_bits, 0.001));
}

test "Dataset memory management" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const sequences = try allocator.alloc([]const u8, 2);
    sequences[0] = try allocator.dupe(u8, "First sequence");
    sequences[1] = try allocator.dupe(u8, "Second sequence");

    var dataset = EvaluationDataset{
        .sequences = sequences,
        .labels = null,
        .metadata = null,
    };

    // Test that deinit properly frees memory
    dataset.deinit(allocator);
    // If this test passes without memory leaks, deinit works correctly
}

test "Error handling for empty sequences" {
    // Test that empty sequences are handled gracefully
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const empty_dataset = EvaluationDataset{
        .sequences = &[_][]const u8{},
        .labels = null,
        .metadata = null,
    };

    try testing.expect(empty_dataset.sequences.len == 0);

    // Test empty string handling
    const empty_sequences = try allocator.alloc([]const u8, 1);
    empty_sequences[0] = try allocator.dupe(u8, "");

    var dataset_with_empty = EvaluationDataset{
        .sequences = empty_sequences,
        .labels = null,
        .metadata = null,
    };

    try testing.expect(dataset_with_empty.sequences.len == 1);
    try testing.expect(dataset_with_empty.sequences[0].len == 0);

    dataset_with_empty.deinit(allocator);
}

test "Window overlap validation" {
    const config1 = PerplexityConfig{
        .sliding_window = 100,
        .window_overlap = 20,
    };

    const config2 = PerplexityConfig{
        .sliding_window = 100,
        .window_overlap = 0,
    };

    try testing.expect(config1.window_overlap < config1.sliding_window);
    try testing.expect(config2.window_overlap == 0);

    // Overlap should be smaller than window size
    try testing.expect(config1.window_overlap < config1.sliding_window);
}

// Note: Full integration tests would require actual model instances
// These tests focus on the mathematical and structural components
// that can be tested independently of the model implementation