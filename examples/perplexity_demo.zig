const std = @import("std");
const perplexity = @import("../src/evaluation/perplexity.zig");
const PerplexityConfig = perplexity.PerplexityConfig;
const PerplexityEvaluator = perplexity.PerplexityEvaluator;
const EvaluationDataset = perplexity.EvaluationDataset;
const BenchmarkSuite = perplexity.BenchmarkSuite;
const PerplexityUtils = perplexity.PerplexityUtils;
const StandardBenchmarks = perplexity.StandardBenchmarks;
const models = @import("../src/models/llama.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("🎯 ZigLlama Perplexity Evaluation & Benchmark Suite Demo\n");
    std.debug.print("=======================================================\n\n");

    // Demo 1: Basic Perplexity Configuration
    std.debug.print("📋 1. Perplexity Configuration Options\n");
    std.debug.print("=======================================\n");

    const configs = [_]struct { name: []const u8, config: PerplexityConfig }{
        .{
            .name = "Default Configuration",
            .config = PerplexityConfig{},
        },
        .{
            .name = "High-Precision Configuration",
            .config = PerplexityConfig{
                .max_sequence_length = 4096,
                .sliding_window = 1024,
                .window_overlap = 128,
                .use_log_probs = true,
                .temperature = 1.0,
                .verbose = true,
            },
        },
        .{
            .name = "Fast Evaluation Configuration",
            .config = PerplexityConfig{
                .max_sequence_length = 1024,
                .sliding_window = 256,
                .window_overlap = 32,
                .batch_size = 4,
                .verbose = false,
            },
        },
        .{
            .name = "Temperature-Scaled Configuration",
            .config = PerplexityConfig{
                .temperature = 0.7, // Makes predictions sharper
                .normalize_probs = true,
            },
        },
    };

    for (configs) |cfg| {
        std.debug.print("🔧 {s}:\n", .{cfg.name});
        std.debug.print("   Max sequence length: {d}\n", .{cfg.config.max_sequence_length});
        std.debug.print("   Sliding window: {d}\n", .{cfg.config.sliding_window});
        std.debug.print("   Window overlap: {d}\n", .{cfg.config.window_overlap});
        std.debug.print("   Batch size: {d}\n", .{cfg.config.batch_size});
        std.debug.print("   Temperature: {d:.1f}\n", .{cfg.config.temperature});
        std.debug.print("   Use log probs: {}\n", .{cfg.config.use_log_probs});
        std.debug.print("   Verbose: {}\n\n", .{cfg.config.verbose});
    }

    // Demo 2: Dataset Creation and Management
    std.debug.print("📊 2. Evaluation Dataset Creation\n");
    std.debug.print("==================================\n");

    const sample_texts = [_][]const u8{
        "The quick brown fox jumps over the lazy dog. This pangram contains all letters of the alphabet.",
        "In the realm of artificial intelligence, large language models have revolutionized natural language processing.",
        "Perplexity is a measure of how well a probability distribution or model predicts a sample of text.",
        "Machine learning algorithms learn patterns from data to make predictions on new, unseen examples.",
        "The transformer architecture uses self-attention mechanisms to process sequential data effectively.",
        "Neural networks consist of interconnected nodes that process information through weighted connections.",
        "Deep learning has achieved remarkable success in computer vision, speech recognition, and language understanding.",
        "Gradient descent is an optimization algorithm used to train neural networks by minimizing loss functions.",
    };

    // Create evaluation dataset
    const sequences = try allocator.alloc([]const u8, sample_texts.len);
    for (sample_texts, 0..) |text, i| {
        sequences[i] = try allocator.dupe(u8, text);
    }

    // Create metadata
    const metadata = try allocator.alloc(perplexity.DatasetMetadata, sample_texts.len);
    for (metadata, 0..) |*meta, i| {
        meta.* = perplexity.DatasetMetadata{
            .source = "demo",
            .domain = if (i < 4) "general" else "technical",
            .language = "en",
            .length = sequences[i].len,
            .difficulty = if (i < 2) 0.3 else if (i < 6) 0.6 else 0.8,
        };
    }

    var demo_dataset = EvaluationDataset{
        .sequences = sequences,
        .labels = null,
        .metadata = metadata,
    };
    defer demo_dataset.deinit(allocator);

    std.debug.print("📝 Created demo dataset:\n");
    std.debug.print("   Sequences: {d}\n", .{demo_dataset.sequences.len});
    std.debug.print("   Average length: {d:.1f} characters\n", .{calculateAverageLength(demo_dataset.sequences)});
    std.debug.print("   Domains: general, technical\n");
    std.debug.print("   Difficulty range: 0.3 - 0.8\n\n");

    // Show sample sequences
    std.debug.print("📄 Sample sequences:\n");
    for (demo_dataset.sequences[0..3], 0..) |seq, i| {
        const truncated = if (seq.len > 80) seq[0..80] else seq;
        std.debug.print("   {d}. {s}{s}\n", .{ i + 1, truncated, if (seq.len > 80) "..." else "" });
    }
    std.debug.print("\n");

    // Demo 3: Mathematical Foundations
    std.debug.print("🔬 3. Perplexity Mathematics Demonstration\n");
    std.debug.print("===========================================\n");

    const example_log_probs = [_]f64{ -1.5, -2.0, -1.8, -2.2, -1.6, -2.5, -1.9, -2.1 };

    std.debug.print("📈 Example log probabilities: ");
    for (example_log_probs, 0..) |log_prob, i| {
        std.debug.print("{d:.1f}", .{log_prob});
        if (i < example_log_probs.len - 1) std.debug.print(", ");
    }
    std.debug.print("\n");

    var total_log_prob: f64 = 0;
    for (example_log_probs) |log_prob| {
        total_log_prob += log_prob;
    }

    const avg_log_prob = total_log_prob / @as(f64, @floatFromInt(example_log_probs.len));
    const log_perplexity = -avg_log_prob;
    const perplexity_value = std.math.exp(log_perplexity);
    const bits_per_token = log_perplexity / std.math.ln(2.0);

    std.debug.print("🧮 Calculations:\n");
    std.debug.print("   Average log probability: {d:.3f}\n", .{avg_log_prob});
    std.debug.print("   Log perplexity: {d:.3f}\n", .{log_perplexity});
    std.debug.print("   Perplexity: {d:.2f}\n", .{perplexity_value});
    std.debug.print("   Bits per token: {d:.2f}\n", .{bits_per_token});
    std.debug.print("   Interpretation: {s}\n\n", .{interpretPerplexity(perplexity_value)});

    // Demo 4: Temperature Effects
    std.debug.print("🌡️  4. Temperature Scaling Effects\n");
    std.debug.print("===================================\n");

    const base_log_prob = -2.0;
    const temperatures = [_]f32{ 0.1, 0.5, 1.0, 1.5, 2.0 };

    std.debug.print("🎚️  Temperature scaling (base log prob: {d:.1f}):\n", .{base_log_prob});
    for (temperatures) |temp| {
        const scaled_log_prob = base_log_prob / temp;
        const scaled_perplexity = std.math.exp(-scaled_log_prob);
        std.debug.print("   T={d:.1f}: log_prob={d:.2f}, perplexity={d:.1f} ({s})\n", .{
            temp,
            scaled_log_prob,
            scaled_perplexity,
            if (temp < 1.0) "sharper" else if (temp > 1.0) "softer" else "baseline",
        });
    }
    std.debug.print("\n");

    // Demo 5: Confidence Intervals
    std.debug.print("📊 5. Statistical Analysis\n");
    std.debug.print("===========================\n");

    const sequence_perplexities = [_]f64{ 8.5, 12.3, 9.7, 15.2, 11.1, 7.8, 13.6, 10.4, 14.8, 9.2, 12.7, 8.9 };

    std.debug.print("📋 Sample sequence perplexities:\n");
    for (sequence_perplexities, 0..) |perp, i| {
        if (i % 6 == 0) std.debug.print("   ");
        std.debug.print("{d:.1f}", .{perp});
        if (i % 6 == 5) {
            std.debug.print("\n");
        } else if (i < sequence_perplexities.len - 1) {
            std.debug.print("  ");
        }
    }
    std.debug.print("\n");

    // Calculate statistics
    var sum: f64 = 0;
    var sum_sq: f64 = 0;
    for (sequence_perplexities) |perp| {
        sum += perp;
        sum_sq += perp * perp;
    }

    const mean = sum / @as(f64, @floatFromInt(sequence_perplexities.len));
    const variance = (sum_sq / @as(f64, @floatFromInt(sequence_perplexities.len))) - (mean * mean);
    const std_dev = std.math.sqrt(variance);

    const interval_95 = PerplexityUtils.calculateConfidenceInterval(&sequence_perplexities, 0.95);
    const interval_90 = PerplexityUtils.calculateConfidenceInterval(&sequence_perplexities, 0.90);

    std.debug.print("📈 Statistical Summary:\n");
    std.debug.print("   Mean: {d:.2f}\n", .{mean});
    std.debug.print("   Standard deviation: {d:.2f}\n", .{std_dev});
    std.debug.print("   95% confidence interval: [{d:.1f}, {d:.1f}]\n", .{ interval_95.lower, interval_95.upper });
    std.debug.print("   90% confidence interval: [{d:.1f}, {d:.1f}]\n\n", .{ interval_90.lower, interval_90.upper });

    // Demo 6: Sliding Window Analysis
    std.debug.print("🪟 6. Sliding Window Evaluation\n");
    std.debug.print("================================\n");

    const long_sequence_length = 2500;
    const window_configs = [_]struct { window: usize, overlap: usize }{
        .{ .window = 512, .overlap = 64 },
        .{ .window = 1024, .overlap = 128 },
        .{ .window = 256, .overlap = 32 },
    };

    for (window_configs) |cfg| {
        const num_windows = calculateNumWindows(long_sequence_length, cfg.window, cfg.overlap);
        const coverage = calculateCoverage(long_sequence_length, cfg.window, cfg.overlap, num_windows);

        std.debug.print("🪟 Window={d}, Overlap={d}:\n", .{ cfg.window, cfg.overlap });
        std.debug.print("   Number of windows: {d}\n", .{num_windows});
        std.debug.print("   Sequence coverage: {d:.1f}%\n", .{coverage * 100});
        std.debug.print("   Effective tokens/window: {d}\n\n", .{cfg.window - cfg.overlap});
    }

    // Demo 7: Standard Benchmarks
    std.debug.print("🏆 7. Standard Benchmark Datasets\n");
    std.debug.print("==================================\n");

    const benchmarks = [_]struct { id: []const u8, description: []const u8, typical_perplexity: []const u8 }{
        .{ .id = StandardBenchmarks.WIKITEXT_103, .description = "Wikipedia articles (long-form)", .typical_perplexity = "15-25" },
        .{ .id = StandardBenchmarks.PENN_TREEBANK, .description = "Financial news articles", .typical_perplexity = "80-120" },
        .{ .id = StandardBenchmarks.LAMBADA, .description = "Narrative completion", .typical_perplexity = "10-20" },
        .{ .id = StandardBenchmarks.HELLASWAG, .description = "Commonsense reasoning", .typical_perplexity = "5-15" },
        .{ .id = StandardBenchmarks.SYNTHETIC_SMALL, .description = "Generated evaluation data (small)", .typical_perplexity = "8-12" },
        .{ .id = StandardBenchmarks.SYNTHETIC_LARGE, .description = "Generated evaluation data (large)", .typical_perplexity = "6-10" },
    };

    std.debug.print("📚 Standard evaluation benchmarks:\n");
    for (benchmarks) |benchmark| {
        std.debug.print("   • {s}\n", .{benchmark.id});
        std.debug.print("     Description: {s}\n", .{benchmark.description});
        std.debug.print("     Typical perplexity range: {s}\n\n", .{benchmark.typical_perplexity});
    }

    // Demo 8: Result Comparison
    std.debug.print("⚖️  8. Model Comparison Analysis\n");
    std.debug.print("=================================\n");

    // Simulate two model results for comparison
    const model_a_perplexity = 12.5;
    const model_b_perplexity = 15.8;

    const mock_result_a = createMockResult(allocator, model_a_perplexity, 1000, 5000) catch return;
    defer mock_result_a.deinit(allocator);

    const mock_result_b = createMockResult(allocator, model_b_perplexity, 1000, 5000) catch return;
    defer mock_result_b.deinit(allocator);

    const comparison = PerplexityUtils.compareResults(mock_result_a, mock_result_b);

    std.debug.print("🔍 Comparison Results:\n");
    std.debug.print("   Model A perplexity: {d:.2f}\n", .{mock_result_a.perplexity});
    std.debug.print("   Model B perplexity: {d:.2f}\n", .{mock_result_b.perplexity});
    std.debug.print("   Absolute difference: {d:.2f}\n", .{comparison.absolute_difference});
    std.debug.print("   Relative difference: {d:.1f}%\n", .{comparison.relative_difference * 100});
    std.debug.print("   Better model: {s}\n", .{if (comparison.better_result == .First) "Model A" else "Model B"});
    std.debug.print("   Statistically significant: {}\n\n", .{comparison.is_statistically_significant});

    // Demo 9: Performance Metrics
    std.debug.print("⏱️  9. Performance and Efficiency Metrics\n");
    std.debug.print("=========================================\n");

    const performance_scenarios = [_]struct { name: []const u8, tokens: usize, time_ms: u64 }{
        .{ .name = "Small model (125M)", .tokens = 1000, .time_ms = 150 },
        .{ .name = "Medium model (350M)", .tokens = 1000, .time_ms = 280 },
        .{ .name = "Large model (1.3B)", .tokens = 1000, .time_ms = 650 },
        .{ .name = "Batch processing", .tokens = 5000, .time_ms = 1200 },
    };

    std.debug.print("🚀 Processing Speed Analysis:\n");
    for (performance_scenarios) |scenario| {
        const tokens_per_second = (@as(f64, @floatFromInt(scenario.tokens)) * 1000.0) / @as(f64, @floatFromInt(scenario.time_ms));
        const ms_per_token = @as(f64, @floatFromInt(scenario.time_ms)) / @as(f64, @floatFromInt(scenario.tokens));

        std.debug.print("   • {s}:\n", .{scenario.name});
        std.debug.print("     Tokens/second: {d:.1f}\n", .{tokens_per_second});
        std.debug.print("     ms/token: {d:.2f}\n", .{ms_per_token});
        std.debug.print("     Total time: {d}ms for {d} tokens\n\n", .{ scenario.time_ms, scenario.tokens });
    }

    // Demo 10: Best Practices and Recommendations
    std.debug.print("💡 10. Best Practices & Recommendations\n");
    std.debug.print("========================================\n");

    const recommendations = [_][]const u8{
        "Use log probabilities for numerical stability with long sequences",
        "Apply sliding windows for sequences longer than model context",
        "Set appropriate temperature for your evaluation task (1.0 for standard)",
        "Include multiple domains in your evaluation dataset",
        "Report confidence intervals for robust comparisons",
        "Use standard benchmarks for reproducible results",
        "Validate results with multiple independent runs",
        "Consider computational resources when choosing window sizes",
        "Monitor memory usage during large dataset evaluation",
        "Document evaluation configuration for reproducibility",
    };

    std.debug.print("📝 Evaluation Best Practices:\n");
    for (recommendations, 0..) |rec, i| {
        std.debug.print("   {d:2d}. {s}\n", .{ i + 1, rec });
    }
    std.debug.print("\n");

    // Final summary
    std.debug.print("🎉 Demo Complete - ZigLlama Perplexity Suite\n");
    std.debug.print("============================================\n");
    std.debug.print("✨ Key Features Demonstrated:\n");
    std.debug.print("   • Comprehensive perplexity evaluation\n");
    std.debug.print("   • Statistical analysis with confidence intervals\n");
    std.debug.print("   • Sliding window processing for long sequences\n");
    std.debug.print("   • Temperature scaling effects\n");
    std.debug.print("   • Model comparison and significance testing\n");
    std.debug.print("   • Standard benchmark dataset integration\n");
    std.debug.print("   • Performance metrics and optimization\n");
    std.debug.print("   • Production-ready evaluation pipeline\n\n");
    std.debug.print("🦙 ZigLlama: Educational AI with Professional Evaluation Tools ✨\n");
}

fn calculateAverageLength(sequences: []const []const u8) f64 {
    var total: usize = 0;
    for (sequences) |seq| {
        total += seq.len;
    }
    return @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(sequences.len));
}

fn interpretPerplexity(perplexity_value: f64) []const u8 {
    if (perplexity_value < 5.0) return "Excellent (very confident predictions)";
    if (perplexity_value < 15.0) return "Good (confident predictions)";
    if (perplexity_value < 30.0) return "Moderate (reasonable predictions)";
    if (perplexity_value < 100.0) return "Poor (uncertain predictions)";
    return "Very poor (highly uncertain predictions)";
}

fn calculateNumWindows(sequence_length: usize, window_size: usize, overlap: usize) usize {
    if (sequence_length <= window_size) return 1;

    const stride = window_size - overlap;
    return (sequence_length - overlap + stride - 1) / stride;
}

fn calculateCoverage(sequence_length: usize, window_size: usize, overlap: usize, num_windows: usize) f64 {
    if (num_windows == 1) return 1.0;

    const stride = window_size - overlap;
    const covered_length = @min(overlap + (num_windows - 1) * stride + (window_size - overlap), sequence_length);
    return @as(f64, @floatFromInt(covered_length)) / @as(f64, @floatFromInt(sequence_length));
}

fn createMockResult(allocator: std.mem.Allocator, perplexity_value: f64, num_tokens: usize, num_chars: usize) !perplexity.PerplexityResult {
    const token_log_probs = try allocator.alloc(f64, num_tokens);
    const sequence_perplexities = try allocator.alloc(f64, 1);

    // Fill with consistent values based on perplexity
    const log_perplexity = std.math.ln(perplexity_value);
    const avg_log_prob = -log_perplexity;

    for (token_log_probs) |*log_prob| {
        log_prob.* = avg_log_prob;
    }
    sequence_perplexities[0] = perplexity_value;

    return perplexity.PerplexityResult{
        .perplexity = perplexity_value,
        .log_perplexity = log_perplexity,
        .bits_per_char = log_perplexity / std.math.ln(2.0) * @as(f64, @floatFromInt(num_tokens)) / @as(f64, @floatFromInt(num_chars)),
        .bits_per_token = log_perplexity / std.math.ln(2.0),
        .total_tokens = num_tokens,
        .total_chars = num_chars,
        .token_log_probs = token_log_probs,
        .sequence_perplexities = sequence_perplexities,
        .evaluation_time_ms = 100,
        .peak_memory_usage = 1024 * 1024,
    };
}