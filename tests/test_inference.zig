//! Comprehensive tests for Inference layer
//!
//! This test suite validates the complete inference system:
//! text generation, sampling strategies, KV caching, streaming,
//! batching, and performance profiling capabilities.

const std = @import("std");
const testing = std.testing;
const math = std.math;

// Import inference components
const GenerationConfig = @import("../src/inference/generation.zig").GenerationConfig;
const SamplingStrategy = @import("../src/inference/generation.zig").SamplingStrategy;
const GenerationResult = @import("../src/inference/generation.zig").GenerationResult;
const StopReason = @import("../src/inference/generation.zig").StopReason;
const GenerationStats = @import("../src/inference/generation.zig").GenerationStats;
const TokenProb = @import("../src/inference/generation.zig").TokenProb;

const KVCacheEntry = @import("../src/inference/kv_cache.zig").KVCacheEntry;
const LayerKVCache = @import("../src/inference/kv_cache.zig").LayerKVCache;
const ModelKVCache = @import("../src/inference/kv_cache.zig").ModelKVCache;
const CacheStrategy = @import("../src/inference/kv_cache.zig").CacheStrategy;
const CacheStats = @import("../src/inference/kv_cache.zig").CacheStats;

const StreamingConfig = @import("../src/inference/streaming.zig").StreamingConfig;
const TokenChunk = @import("../src/inference/streaming.zig").TokenChunk;
const StreamStatus = @import("../src/inference/streaming.zig").StreamStatus;

const BatchConfig = @import("../src/inference/batching.zig").BatchConfig;
const BatchingStrategy = @import("../src/inference/batching.zig").BatchingStrategy;
const RequestPriority = @import("../src/inference/batching.zig").RequestPriority;
const BatchStats = @import("../src/inference/batching.zig").BatchStats;
const BatchResult = @import("../src/inference/batching.zig").BatchResult;

const Profiler = @import("../src/inference/profiling.zig").Profiler;
const PerformanceStats = @import("../src/inference/profiling.zig").PerformanceStats;
const MeasurementPoint = @import("../src/inference/profiling.zig").MeasurementPoint;
const BenchmarkConfig = @import("../src/inference/profiling.zig").BenchmarkConfig;
const ThroughputMetrics = @import("../src/inference/profiling.zig").ThroughputMetrics;

const TokenId = @import("../src/models/tokenizer.zig").TokenId;
const ModelConfig = @import("../src/models/config.zig").ModelConfig;
const Tensor = @import("../src/foundation/tensor.zig").Tensor;

// ===================== Generation Tests =====================

test "generation configuration validation and presets" {
    // Test validation
    const valid_config = GenerationConfig.balanced();
    try valid_config.validate();

    var invalid_config = valid_config;
    invalid_config.temperature = -1.0;
    try testing.expectError(error.InvalidTemperature, invalid_config.validate());

    invalid_config = valid_config;
    invalid_config.top_p = 2.0;
    try testing.expectError(error.InvalidTopP, invalid_config.validate());

    // Test presets have expected properties
    const creative = GenerationConfig.creative();
    const balanced = GenerationConfig.balanced();
    const focused = GenerationConfig.focused();
    const deterministic = GenerationConfig.deterministic();

    try testing.expect(creative.temperature > balanced.temperature);
    try testing.expect(balanced.temperature > focused.temperature);
    try testing.expect(focused.temperature > deterministic.temperature);
    try testing.expectEqual(@as(f32, 0.0), deterministic.temperature);

    try testing.expectEqual(SamplingStrategy.Greedy, deterministic.strategy);
    try testing.expectEqual(SamplingStrategy.Combined, creative.strategy);
}

test "sampling strategy properties" {
    // Test strategy names
    try testing.expectEqualStrings("Greedy", SamplingStrategy.Greedy.name());
    try testing.expectEqualStrings("Top-K", SamplingStrategy.TopK.name());
    try testing.expectEqualStrings("Top-P (Nucleus)", SamplingStrategy.TopP.name());
    try testing.expectEqualStrings("Temperature", SamplingStrategy.Temperature.name());
    try testing.expectEqualStrings("Combined", SamplingStrategy.Combined.name());
}

test "generation statistics calculation" {
    // Test statistics computation
    const stats = GenerationStats.calculate(100, 2000.0); // 100 tokens in 2 seconds

    try testing.expectEqual(@as(f64, 2000.0), stats.generation_time_ms);
    try testing.expectApproxEqAbs(@as(f32, 50.0), stats.tokens_per_second, 0.1);
    try testing.expectApproxEqAbs(@as(f32, 20.0), stats.time_per_token_ms, 0.1);
    try testing.expectEqual(@as(u32, 100), stats.num_forward_passes);

    // Test edge cases
    const zero_time_stats = GenerationStats.calculate(10, 0.0);
    try testing.expectEqual(@as(f32, 0.0), zero_time_stats.tokens_per_second);

    const zero_token_stats = GenerationStats.calculate(0, 1000.0);
    try testing.expectEqual(@as(f32, 0.0), zero_token_stats.time_per_token_ms);
}

test "stop reason descriptions" {
    try testing.expectEqualStrings("Maximum token limit reached", StopReason.MaxTokens.description());
    try testing.expectEqualStrings("Stop token encountered", StopReason.StopToken.description());
    try testing.expectEqualStrings("Stop string encountered", StopReason.StopString.description());
    try testing.expectEqualStrings("End of sequence", StopReason.EndOfSequence.description());
    try testing.expectEqualStrings("Generation error", StopReason.Error.description());
}

test "token probability operations" {
    const token_prob = TokenProb.fromLogit(42, -2.5);

    try testing.expectEqual(@as(TokenId, 42), token_prob.token_id);
    try testing.expectEqual(@as(f32, -2.5), token_prob.log_prob);
    try testing.expectEqual(@as(f32, 0.0), token_prob.probability); // Will be set after softmax
}

// ===================== KV Cache Tests =====================

test "KV cache entry operations" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var cache = try KVCacheEntry.init(allocator, 10, 64);
    defer cache.deinit();

    // Test initial state
    try testing.expectEqual(@as(usize, 0), cache.current_length);
    try testing.expectEqual(@as(usize, 10), cache.max_length);
    try testing.expectEqual(@as(usize, 10), cache.remainingCapacity());
    try testing.expect(!cache.isFull());

    // Create and add test tensors
    var keys = try Tensor(f32).init(allocator, &[_]usize{ 3, 64 });
    defer keys.deinit();
    var values = try Tensor(f32).init(allocator, &[_]usize{ 3, 64 });
    defer values.deinit();

    // Fill with test data
    for (0..keys.size) |i| {
        keys.data[i] = @as(f32, @floatFromInt(i));
        values.data[i] = @as(f32, @floatFromInt(i)) + 1000.0;
    }

    try cache.append(keys, values);
    try testing.expectEqual(@as(usize, 3), cache.current_length);
    try testing.expectEqual(@as(usize, 7), cache.remainingCapacity());

    // Test retrieval
    const cached_keys = try cache.getCachedKeys(allocator);
    defer cached_keys.deinit();
    const cached_values = try cache.getCachedValues(allocator);
    defer cached_values.deinit();

    try testing.expectEqual(@as(usize, 3), cached_keys.shape[0]);
    try testing.expectEqual(@as(usize, 64), cached_keys.shape[1]);
    try testing.expectEqual(@as(f32, 0.0), cached_keys.data[0]);
    try testing.expectEqual(@as(f32, 1000.0), cached_values.data[0]);

    // Test cache overflow
    var large_keys = try Tensor(f32).init(allocator, &[_]usize{ 8, 64 });
    defer large_keys.deinit();
    var large_values = try Tensor(f32).init(allocator, &[_]usize{ 8, 64 });
    defer large_values.deinit();

    try testing.expectError(error.CacheOverflow, cache.append(large_keys, large_values));
}

test "layer KV cache multi-head management" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var layer_cache = try LayerKVCache.init(allocator, 8, 64, 10);
    defer layer_cache.deinit();

    // Test initialization
    try testing.expectEqual(@as(usize, 8), layer_cache.num_heads);
    try testing.expectEqual(@as(usize, 64), layer_cache.head_dim);
    try testing.expectEqual(@as(usize, 10), layer_cache.max_seq_len);

    // Test memory usage calculation
    const expected_memory = 8 * 10 * 64 * @sizeOf(f32) * 2; // keys + values
    try testing.expectEqual(expected_memory, layer_cache.getMemoryUsage());

    try testing.expectEqual(@as(usize, 0), layer_cache.getCurrentLength());
    try testing.expect(!layer_cache.isFull());
}

test "model KV cache integration" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = ModelConfig.custom(512, 6, 8, 10000);
    var model_cache = try ModelKVCache.init(allocator, config);
    defer model_cache.deinit();

    try testing.expectEqual(@as(usize, 6), model_cache.num_layers);
    try testing.expectEqual(@as(usize, 0), model_cache.getCurrentLength());
    try testing.expect(!model_cache.isFull());

    // Test memory statistics
    const memory_stats = model_cache.getMemoryStats();
    try testing.expect(memory_stats.total_bytes > 0);
    try testing.expect(memory_stats.per_layer_bytes > 0);
    try testing.expectEqual(@as(f32, 0.0), memory_stats.utilization_percent);

    // Test invalid layer access
    var dummy_keys = try Tensor(f32).init(allocator, &[_]usize{ 1, 8, 2, 64 });
    defer dummy_keys.deinit();
    var dummy_values = try Tensor(f32).init(allocator, &[_]usize{ 1, 8, 2, 64 });
    defer dummy_values.deinit();

    try testing.expectError(error.InvalidLayerIndex, model_cache.updateLayer(10, dummy_keys, dummy_values));
}

test "cache strategy decision making" {
    // Test different cache strategies
    try testing.expect(CacheStrategy.Always.shouldCache(10, null));
    try testing.expect(CacheStrategy.Always.shouldCache(1000, null));

    try testing.expect(!CacheStrategy.LongSequenceOnly.shouldCache(100, null));
    try testing.expect(CacheStrategy.LongSequenceOnly.shouldCache(1000, null));

    try testing.expect(!CacheStrategy.Disabled.shouldCache(1000, null));

    // Test adaptive strategy
    try testing.expect(CacheStrategy.Adaptive.shouldCache(100, 1000000)); // Enough memory
    try testing.expect(!CacheStrategy.Adaptive.shouldCache(100, 100)); // Not enough memory
}

test "cache statistics tracking" {
    var stats = CacheStats{};
    try testing.expectEqual(@as(f32, 0.0), stats.hitRate());

    stats.cache_hits = 80;
    stats.cache_misses = 20;
    try testing.expectApproxEqAbs(@as(f32, 0.8), stats.hitRate(), 0.01);

    stats.cache_updates = 100;
    stats.cache_compactions = 5;
    try testing.expectEqual(@as(u64, 100), stats.cache_updates);
    try testing.expectEqual(@as(u64, 5), stats.cache_compactions);
}

// ===================== Streaming Tests =====================

test "streaming configuration validation" {
    const valid_config = StreamingConfig{};
    try valid_config.validate();

    var invalid_config = valid_config;
    invalid_config.buffer_size = 0;
    try testing.expectError(error.InvalidBufferSize, invalid_config.validate());

    invalid_config = valid_config;
    invalid_config.min_chunk_size = 10;
    invalid_config.max_chunk_size = 5;
    try testing.expectError(error.InvalidChunkSizes, invalid_config.validate());

    invalid_config = valid_config;
    invalid_config.timeout_ms = 0;
    try testing.expectError(error.InvalidTimeout, invalid_config.validate());
}

test "token chunk creation and properties" {
    const chunk = TokenChunk.init(42, "hello", -2.5, -10.0, 5);

    try testing.expectEqual(@as(TokenId, 42), chunk.token_id);
    try testing.expectEqualStrings("hello", chunk.text.?);
    try testing.expectEqual(@as(f32, -2.5), chunk.log_prob);
    try testing.expectEqual(@as(f32, -10.0), chunk.cumulative_log_prob);
    try testing.expectEqual(@as(u32, 5), chunk.position);
    try testing.expect(chunk.timestamp > 0);
}

test "stream status lifecycle" {
    var status = StreamStatus.init();

    // Test initial state
    try testing.expectEqual(@as(u32, 0), status.tokens_generated);
    try testing.expect(status.is_active);
    try testing.expect(status.stop_reason == null);
    try testing.expectEqual(@as(f32, 0.0), status.current_tps);
    try testing.expectEqual(@as(f32, 0.0), status.average_tps);

    // Simulate token generation
    std.time.sleep(10_000_000); // 10ms
    status.update(5);

    try testing.expectEqual(@as(u32, 5), status.tokens_generated);
    try testing.expect(status.current_tps >= 0.0); // Should be positive
    try testing.expect(status.average_tps >= 0.0);

    // Finish stream
    status.finish(.MaxTokens);
    try testing.expect(!status.is_active);
    try testing.expect(status.stop_reason == .MaxTokens);
}

// ===================== Batching Tests =====================

test "batch configuration validation" {
    const valid_config = BatchConfig{};
    try valid_config.validate();

    var invalid_config = valid_config;
    invalid_config.max_batch_size = 0;
    try testing.expectError(error.InvalidMaxBatchSize, invalid_config.validate());

    invalid_config = valid_config;
    invalid_config.min_batch_size = 0;
    try testing.expectError(error.InvalidMinBatchSize, invalid_config.validate());

    invalid_config = valid_config;
    invalid_config.min_batch_size = 10;
    invalid_config.max_batch_size = 5;
    try testing.expectError(error.InvalidMinBatchSize, invalid_config.validate());

    invalid_config = valid_config;
    invalid_config.max_queue_size = 0;
    try testing.expectError(error.InvalidMaxQueueSize, invalid_config.validate());

    invalid_config = valid_config;
    invalid_config.num_workers = 0;
    try testing.expectError(error.InvalidNumWorkers, invalid_config.validate());
}

test "batching strategy descriptions" {
    try testing.expectEqualStrings("Fixed batch size", BatchingStrategy.FixedSize.description());
    try testing.expectEqualStrings("Dynamic with timeout", BatchingStrategy.DynamicTimeout.description());
    try testing.expectEqualStrings("Adaptive batching", BatchingStrategy.Adaptive.description());
    try testing.expectEqualStrings("Continuous processing", BatchingStrategy.Continuous.description());
}

test "request priority ordering" {
    try testing.expectEqual(std.math.Order.lt, RequestPriority.Low.compare(.Normal));
    try testing.expectEqual(std.math.Order.eq, RequestPriority.High.compare(.High));
    try testing.expectEqual(std.math.Order.gt, RequestPriority.Critical.compare(.Low));
    try testing.expectEqual(std.math.Order.gt, RequestPriority.Critical.compare(.Normal));
}

test "batch statistics accumulation" {
    var stats = BatchStats{};
    try testing.expectEqual(@as(u64, 0), stats.total_requests);
    try testing.expectEqual(@as(f32, 0.0), stats.avg_latency_ms);

    const result1 = BatchResult{
        .request_id = 1,
        .tokens = &[_]TokenId{},
        .text = &[_]u8{},
        .log_probs = &[_]f32{},
        .total_log_prob = 0.0,
        .num_tokens = 10,
        .stop_reason = .MaxTokens,
        .latency_ms = 100,
        .queue_time_ms = 50,
    };

    stats.updateWithResult(result1, 4);
    try testing.expectEqual(@as(u64, 1), stats.total_requests);
    try testing.expectEqual(@as(u64, 1), stats.total_batches);
    try testing.expectEqual(@as(f32, 100.0), stats.avg_latency_ms);
    try testing.expectEqual(@as(f32, 50.0), stats.avg_queue_time_ms);
    try testing.expectEqual(@as(f32, 4.0), stats.avg_batch_size);

    // Add another result
    const result2 = BatchResult{
        .request_id = 2,
        .tokens = &[_]TokenId{},
        .text = &[_]u8{},
        .log_probs = &[_]f32{},
        .total_log_prob = 0.0,
        .num_tokens = 15,
        .stop_reason = .EndOfSequence,
        .latency_ms = 200,
        .queue_time_ms = 30,
    };

    stats.updateWithResult(result2, 6);
    try testing.expectEqual(@as(u64, 2), stats.total_requests);
    try testing.expectEqual(@as(u64, 2), stats.total_batches);
    try testing.expectEqual(@as(f32, 150.0), stats.avg_latency_ms); // (100 + 200) / 2
    try testing.expectEqual(@as(f32, 40.0), stats.avg_queue_time_ms); // (50 + 30) / 2
    try testing.expectEqual(@as(f32, 5.0), stats.avg_batch_size); // (4 + 6) / 2
}

// ===================== Profiling Tests =====================

test "measurement point duration and memory calculations" {
    var point = MeasurementPoint.init("test_operation");
    point.start_time = 1_000_000; // 1ms in nanoseconds
    point.end_time = 3_000_000;   // 3ms in nanoseconds
    point.start_memory = 1000;
    point.end_memory = 1500;

    try testing.expectEqual(@as(f64, 2.0), point.duration()); // 2ms
    try testing.expectEqual(@as(i64, 500), point.memoryDelta()); // 500 bytes allocated

    // Test edge cases
    var zero_point = MeasurementPoint.init("zero");
    zero_point.start_time = 1000;
    zero_point.end_time = 1000;
    try testing.expectEqual(@as(f64, 0.0), zero_point.duration());

    var negative_point = MeasurementPoint.init("negative");
    negative_point.start_memory = 2000;
    negative_point.end_memory = 1500;
    try testing.expectEqual(@as(i64, -500), negative_point.memoryDelta());
}

test "performance statistics updates and aggregation" {
    var stats = PerformanceStats{};
    try testing.expectEqual(@as(u64, 0), stats.count);
    try testing.expect(std.math.isInf(stats.min_duration));

    // Add first measurement
    var measurement1 = MeasurementPoint.init("test");
    measurement1.start_time = 1_000_000;
    measurement1.end_time = 3_000_000; // 2ms duration
    measurement1.start_memory = 1000;
    measurement1.end_memory = 1200;

    stats.update(measurement1);
    try testing.expectEqual(@as(u64, 1), stats.count);
    try testing.expectEqual(@as(f64, 2.0), stats.avg_duration);
    try testing.expectEqual(@as(f64, 2.0), stats.min_duration);
    try testing.expectEqual(@as(f64, 2.0), stats.max_duration);
    try testing.expectEqual(@as(u64, 200), stats.total_memory_allocated);

    // Add second measurement
    var measurement2 = MeasurementPoint.init("test");
    measurement2.start_time = 2_000_000;
    measurement2.end_time = 6_000_000; // 4ms duration
    measurement2.start_memory = 1200;
    measurement2.end_memory = 1500;

    stats.update(measurement2);
    try testing.expectEqual(@as(u64, 2), stats.count);
    try testing.expectEqual(@as(f64, 3.0), stats.avg_duration); // (2 + 4) / 2
    try testing.expectEqual(@as(f64, 2.0), stats.min_duration);
    try testing.expectEqual(@as(f64, 4.0), stats.max_duration);
    try testing.expectEqual(@as(u64, 500), stats.total_memory_allocated); // 200 + 300

    // Test percentile calculation
    var durations = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    stats.calculatePercentiles(&durations);

    try testing.expectEqual(@as(f64, 5.0), stats.median_duration);
    try testing.expectEqual(@as(f64, 9.0), stats.p95_duration);
    try testing.expectEqual(@as(f64, 9.0), stats.p99_duration);
    try testing.expect(stats.std_deviation > 0.0);
}

test "benchmark configuration validation" {
    const valid_config = BenchmarkConfig{};
    try valid_config.validate();

    var invalid_config = valid_config;
    invalid_config.measurement_runs = 0;
    try testing.expectError(error.InvalidMeasurementRuns, invalid_config.validate());

    invalid_config = valid_config;
    invalid_config.max_time_per_run = 0;
    try testing.expectError(error.InvalidMaxTime, invalid_config.validate());
}

test "throughput metrics properties" {
    const throughput = ThroughputMetrics{
        .requests_per_second = 100.0,
        .tokens_per_second = 2000.0,
        .characters_per_second = 10000.0,
        .batches_per_second = 25.0,
    };

    try testing.expectEqual(@as(f32, 100.0), throughput.requests_per_second);
    try testing.expectEqual(@as(f32, 2000.0), throughput.tokens_per_second);
    try testing.expectEqual(@as(f32, 10000.0), throughput.characters_per_second);
    try testing.expectEqual(@as(f32, 25.0), throughput.batches_per_second);
}

// ===================== Integration Tests =====================

test "profiler integration with measurements" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var profiler = try Profiler.init(allocator);
    defer profiler.deinit();

    // Test enabling/disabling
    try testing.expect(profiler.enabled);
    profiler.setEnabled(false);
    try testing.expect(!profiler.enabled);
    profiler.setEnabled(true);

    // Test measurement lifecycle
    try profiler.startMeasurement("test_op");
    std.time.sleep(1_000_000); // 1ms
    try profiler.endMeasurement("test_op");

    const stats = profiler.getStatistics("test_op");
    try testing.expect(stats != null);
    try testing.expectEqual(@as(u64, 1), stats.?.count);
    try testing.expect(stats.?.avg_duration >= 0.8); // Should be around 1ms, allow for variance

    const measurements = profiler.getMeasurements("test_op");
    try testing.expect(measurements != null);
    try testing.expectEqual(@as(usize, 1), measurements.?.len);
}

test "inference layer component interaction" {
    // Test that all major components can be instantiated together
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Generation config
    const gen_config = GenerationConfig.balanced();
    try gen_config.validate();

    // Streaming config
    const stream_config = StreamingConfig{};
    try stream_config.validate();

    // Batch config
    const batch_config = BatchConfig{};
    try batch_config.validate();

    // Benchmark config
    const bench_config = BenchmarkConfig{};
    try bench_config.validate();

    // Model config for cache
    const model_config = ModelConfig.custom(512, 6, 8, 10000);
    var cache = try ModelKVCache.init(allocator, model_config);
    defer cache.deinit();

    // Profiler
    var profiler = try Profiler.init(allocator);
    defer profiler.deinit();

    // All components should be working together
    try testing.expect(gen_config.strategy != .Greedy or gen_config.temperature == 0.0);
    try testing.expect(stream_config.buffer_size > 0);
    try testing.expect(batch_config.max_batch_size > 0);
    try testing.expect(bench_config.measurement_runs > 0);
    try testing.expect(cache.num_layers == 6);
    try testing.expect(profiler.enabled);
}

test "comprehensive error handling across components" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test cache overflow
    var cache = try KVCacheEntry.init(allocator, 2, 64);
    defer cache.deinit();

    var large_tensor = try Tensor(f32).init(allocator, &[_]usize{ 5, 64 });
    defer large_tensor.deinit();

    try testing.expectError(error.CacheOverflow, cache.append(large_tensor, large_tensor));

    // Test invalid configurations
    var bad_gen_config = GenerationConfig.balanced();
    bad_gen_config.temperature = -1.0;
    try testing.expectError(error.InvalidTemperature, bad_gen_config.validate());

    var bad_stream_config = StreamingConfig{};
    bad_stream_config.buffer_size = 0;
    try testing.expectError(error.InvalidBufferSize, bad_stream_config.validate());

    var bad_batch_config = BatchConfig{};
    bad_batch_config.max_batch_size = 0;
    try testing.expectError(error.InvalidMaxBatchSize, bad_batch_config.validate());

    var bad_bench_config = BenchmarkConfig{};
    bad_bench_config.measurement_runs = 0;
    try testing.expectError(error.InvalidMeasurementRuns, bad_bench_config.validate());
}