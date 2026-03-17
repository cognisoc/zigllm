//! Batched Inference Engine
//!
//! This module implements efficient batch processing for language model inference,
//! allowing multiple prompts to be processed simultaneously for better throughput
//! and resource utilization in production environments.
//!
//! ## Educational Value
//! Batched inference teaches production optimization concepts:
//! - How to maximize GPU/CPU utilization through parallelism
//! - Dynamic batching and request queuing strategies
//! - Memory management and optimization in batch contexts
//! - Load balancing and request scheduling algorithms

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;
const Thread = std.Thread;
const Mutex = std.Thread.Mutex;
const Condition = std.Thread.Condition;

const Tensor = @import("../foundation/tensor.zig").Tensor;
const LLaMAModel = @import("../models/llama.zig").LLaMAModel;
const SimpleTokenizer = @import("../models/tokenizer.zig").SimpleTokenizer;
const TokenId = @import("../models/tokenizer.zig").TokenId;
const SpecialTokens = @import("../models/tokenizer.zig").SpecialTokens;
const GenerationConfig = @import("generation.zig").GenerationConfig;
const GenerationResult = @import("generation.zig").GenerationResult;
const StopReason = @import("generation.zig").StopReason;
const ModelKVCache = @import("kv_cache.zig").ModelKVCache;

/// Unique identifier for batch requests
pub const RequestId = u64;

/// Batch processing request
pub const BatchRequest = struct {
    /// Unique request identifier
    id: RequestId,
    /// Input prompt text
    prompt: []const u8,
    /// Generation configuration
    config: GenerationConfig,
    /// Maximum tokens for this request
    max_tokens: u32,
    /// Current generated tokens
    generated_tokens: ArrayList(TokenId),
    /// Current generated text
    generated_text: ArrayList(u8),
    /// Cumulative log probability
    cumulative_log_prob: f32,
    /// Request creation timestamp
    created_at: i64,
    /// Request processing start time
    started_at: ?i64,
    /// Request completion time
    completed_at: ?i64,
    /// Whether request is completed
    completed: bool,
    /// Stop reason (if completed)
    stop_reason: ?StopReason,

    pub fn init(id: RequestId, prompt: []const u8, config: GenerationConfig, allocator: Allocator) !BatchRequest {
        return BatchRequest{
            .id = id,
            .prompt = try allocator.dupe(u8, prompt),
            .config = config,
            .max_tokens = config.max_tokens,
            .generated_tokens = ArrayList(TokenId).init(allocator),
            .generated_text = ArrayList(u8).init(allocator),
            .cumulative_log_prob = 0.0,
            .created_at = std.time.milliTimestamp(),
            .started_at = null,
            .completed_at = null,
            .completed = false,
            .stop_reason = null,
        };
    }

    pub fn deinit(self: *BatchRequest, allocator: Allocator) void {
        allocator.free(self.prompt);
        self.generated_tokens.deinit();
        self.generated_text.deinit();
    }

    pub fn addToken(self: *BatchRequest, token_id: TokenId, text: []const u8, log_prob: f32) !void {
        try self.generated_tokens.append(token_id);
        try self.generated_text.appendSlice(text);
        self.cumulative_log_prob += log_prob;
    }

    pub fn markCompleted(self: *BatchRequest, stop_reason: StopReason) void {
        self.completed = true;
        self.stop_reason = stop_reason;
        self.completed_at = std.time.milliTimestamp();
    }

    pub fn getLatency(self: BatchRequest) ?i64 {
        if (self.started_at != null and self.completed_at != null) {
            return self.completed_at.? - self.started_at.?;
        }
        return null;
    }

    pub fn getQueueTime(self: BatchRequest) ?i64 {
        if (self.started_at) |started| {
            return started - self.created_at;
        }
        return null;
    }
};

/// Batch processing result
pub const BatchResult = struct {
    /// Request ID
    request_id: RequestId,
    /// Generated tokens
    tokens: []TokenId,
    /// Generated text
    text: []u8,
    /// Log probabilities
    log_probs: []f32,
    /// Total log probability
    total_log_prob: f32,
    /// Number of tokens generated
    num_tokens: u32,
    /// Stop reason
    stop_reason: StopReason,
    /// Processing latency in milliseconds
    latency_ms: i64,
    /// Queue waiting time in milliseconds
    queue_time_ms: i64,

    pub fn deinit(self: BatchResult, allocator: Allocator) void {
        allocator.free(self.tokens);
        allocator.free(self.text);
        allocator.free(self.log_probs);
    }
};

/// Batching strategy for request grouping
pub const BatchingStrategy = enum {
    /// Fixed batch size, wait for full batches
    FixedSize,
    /// Dynamic batching with timeout
    DynamicTimeout,
    /// Adaptive batching based on queue length and latency
    Adaptive,
    /// Continuous batching (process as soon as any request arrives)
    Continuous,

    pub fn description(self: BatchingStrategy) []const u8 {
        return switch (self) {
            .FixedSize => "Fixed batch size",
            .DynamicTimeout => "Dynamic with timeout",
            .Adaptive => "Adaptive batching",
            .Continuous => "Continuous processing",
        };
    }
};

/// Batch processing configuration
pub const BatchConfig = struct {
    /// Batching strategy
    strategy: BatchingStrategy = .DynamicTimeout,
    /// Maximum batch size
    max_batch_size: u32 = 8,
    /// Minimum batch size for processing
    min_batch_size: u32 = 1,
    /// Maximum wait time for batching (ms)
    max_wait_time_ms: u32 = 100,
    /// Maximum queue size
    max_queue_size: u32 = 1000,
    /// Number of worker threads
    num_workers: u32 = 1,
    /// Enable request prioritization
    enable_priority: bool = false,
    /// Memory limit per batch (bytes)
    memory_limit_bytes: ?usize = null,

    pub fn validate(self: BatchConfig) !void {
        if (self.max_batch_size == 0) {
            return error.InvalidMaxBatchSize;
        }
        if (self.min_batch_size == 0 or self.min_batch_size > self.max_batch_size) {
            return error.InvalidMinBatchSize;
        }
        if (self.max_queue_size == 0) {
            return error.InvalidMaxQueueSize;
        }
        if (self.num_workers == 0) {
            return error.InvalidNumWorkers;
        }
    }
};

/// Request priority for queue ordering
pub const RequestPriority = enum(u8) {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,

    pub fn compare(a: RequestPriority, b: RequestPriority) std.math.Order {
        return std.math.order(@intFromEnum(a), @intFromEnum(b));
    }
};

/// Batch processing statistics
pub const BatchStats = struct {
    /// Total requests processed
    total_requests: u64 = 0,
    /// Total batches processed
    total_batches: u64 = 0,
    /// Average batch size
    avg_batch_size: f32 = 0.0,
    /// Average latency per request (ms)
    avg_latency_ms: f32 = 0.0,
    /// Average queue time (ms)
    avg_queue_time_ms: f32 = 0.0,
    /// Requests per second (throughput)
    requests_per_second: f32 = 0.0,
    /// Tokens per second (total)
    tokens_per_second: f32 = 0.0,
    /// Current queue size
    current_queue_size: u32 = 0,
    /// Peak queue size seen
    peak_queue_size: u32 = 0,
    /// Number of timeouts
    timeout_count: u32 = 0,
    /// Number of errors
    error_count: u32 = 0,

    pub fn updateWithResult(self: *BatchStats, result: BatchResult, batch_size: u32) void {
        self.total_requests += 1;

        // Update averages using running average
        const n = @as(f32, @floatFromInt(self.total_requests));
        self.avg_latency_ms = (self.avg_latency_ms * (n - 1.0) + @as(f32, @floatFromInt(result.latency_ms))) / n;
        self.avg_queue_time_ms = (self.avg_queue_time_ms * (n - 1.0) + @as(f32, @floatFromInt(result.queue_time_ms))) / n;

        // Update batch statistics
        if (batch_size > 0) {
            const batch_n = @as(f32, @floatFromInt(self.total_batches + 1));
            self.avg_batch_size = (self.avg_batch_size * @as(f32, @floatFromInt(self.total_batches)) + @as(f32, @floatFromInt(batch_size))) / batch_n;
            self.total_batches += 1;
        }

        // Update throughput metrics would require timing information
        // This is simplified - real implementation would track time windows
    }

    pub fn print(self: BatchStats, writer: anytype) !void {
        try writer.print("Batch Processing Statistics:\n", .{});
        try writer.print("  Total requests: {d}\n", .{self.total_requests});
        try writer.print("  Total batches: {d}\n", .{self.total_batches});
        try writer.print("  Average batch size: {d:.1}\n", .{self.avg_batch_size});
        try writer.print("  Average latency: {d:.1} ms\n", .{self.avg_latency_ms});
        try writer.print("  Average queue time: {d:.1} ms\n", .{self.avg_queue_time_ms});
        try writer.print("  Requests/second: {d:.1}\n", .{self.requests_per_second});
        try writer.print("  Tokens/second: {d:.1}\n", .{self.tokens_per_second});
        try writer.print("  Current queue: {d}\n", .{self.current_queue_size});
        try writer.print("  Peak queue: {d}\n", .{self.peak_queue_size});
        try writer.print("  Timeouts: {d}\n", .{self.timeout_count});
        try writer.print("  Errors: {d}\n", .{self.error_count});
    }
};

/// Thread-safe request queue
const RequestQueue = struct {
    requests: ArrayList(BatchRequest),
    mutex: Mutex,
    condition: Condition,
    allocator: Allocator,
    closed: bool,

    pub fn init(allocator: Allocator) RequestQueue {
        return RequestQueue{
            .requests = ArrayList(BatchRequest).init(allocator),
            .mutex = Mutex{},
            .condition = Condition{},
            .allocator = allocator,
            .closed = false,
        };
    }

    pub fn deinit(self: *RequestQueue) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.requests.items) |*request| {
            request.deinit(self.allocator);
        }
        self.requests.deinit();
    }

    pub fn push(self: *RequestQueue, request: BatchRequest) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.closed) {
            return error.QueueClosed;
        }

        try self.requests.append(request);
        self.condition.signal();
    }

    pub fn popBatch(self: *RequestQueue, max_size: u32, timeout_ms: u32) []BatchRequest {
        self.mutex.lock();
        defer self.mutex.unlock();

        const timeout_ns = @as(u64, timeout_ms) * 1_000_000;
        const start_time = std.time.nanoTimestamp();

        // Wait for requests or timeout
        while (self.requests.items.len == 0 and !self.closed) {
            const elapsed = @as(u64, @intCast(std.time.nanoTimestamp() - start_time));
            if (elapsed >= timeout_ns) {
                break; // Timeout
            }

            const remaining = timeout_ns - elapsed;
            _ = self.condition.timedWait(&self.mutex, remaining) catch break;
        }

        // Extract batch
        const available = self.requests.items.len;
        if (available == 0) {
            return &[_]BatchRequest{};
        }

        const batch_size = @min(available, max_size);
        const batch = self.allocator.alloc(BatchRequest, batch_size) catch {
            return &[_]BatchRequest{};
        };

        for (0..batch_size) |i| {
            batch[i] = self.requests.orderedRemove(0);
            batch[i].started_at = std.time.milliTimestamp();
        }

        return batch;
    }

    pub fn size(self: *RequestQueue) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.requests.items.len;
    }

    pub fn close(self: *RequestQueue) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.closed = true;
        self.condition.broadcast();
    }
};

/// Batched inference engine
pub const BatchProcessor = struct {
    /// Model for inference
    model: *LLaMAModel,
    /// Tokenizer for text processing
    tokenizer: *SimpleTokenizer,
    /// Batch configuration
    config: BatchConfig,
    /// Request queue
    queue: RequestQueue,
    /// Processing statistics
    stats: BatchStats,
    /// Memory allocator
    allocator: Allocator,
    /// Worker threads
    workers: []Thread,
    /// Whether processor is running
    is_running: bool,
    /// Next request ID
    next_request_id: RequestId,
    /// Results callback
    result_callback: ?*const fn (result: BatchResult, user_data: ?*anyopaque) void,
    /// User data for callback
    user_data: ?*anyopaque,

    /// Initialize batch processor
    pub fn init(
        model: *LLaMAModel,
        tokenizer: *SimpleTokenizer,
        config: BatchConfig,
        allocator: Allocator
    ) !BatchProcessor {
        try config.validate();

        const workers = try allocator.alloc(Thread, config.num_workers);

        return BatchProcessor{
            .model = model,
            .tokenizer = tokenizer,
            .config = config,
            .queue = RequestQueue.init(allocator),
            .stats = BatchStats{},
            .allocator = allocator,
            .workers = workers,
            .is_running = false,
            .next_request_id = 1,
            .result_callback = null,
            .user_data = null,
        };
    }

    /// Clean up batch processor
    pub fn deinit(self: *BatchProcessor) void {
        self.stop();
        self.queue.deinit();
        self.allocator.free(self.workers);
    }

    /// Set result callback function
    pub fn setResultCallback(
        self: *BatchProcessor,
        callback: *const fn (result: BatchResult, user_data: ?*anyopaque) void,
        user_data: ?*anyopaque
    ) void {
        self.result_callback = callback;
        self.user_data = user_data;
    }

    /// Start batch processing
    pub fn start(self: *BatchProcessor) !void {
        if (self.is_running) {
            return error.AlreadyRunning;
        }

        self.is_running = true;

        // Start worker threads
        for (self.workers, 0..) |*worker, i| {
            const WorkerContext = struct {
                processor: *BatchProcessor,
                worker_id: usize,
            };

            const context = try self.allocator.create(WorkerContext);
            context.* = WorkerContext{
                .processor = self,
                .worker_id = i,
            };

            worker.* = try Thread.spawn(.{}, workerFunction, .{context});
        }
    }

    /// Stop batch processing
    pub fn stop(self: *BatchProcessor) void {
        if (!self.is_running) {
            return;
        }

        self.is_running = false;
        self.queue.close();

        // Wait for workers to finish
        for (self.workers) |worker| {
            worker.join();
        }
    }

    /// Submit request for batch processing
    pub fn submit(self: *BatchProcessor, prompt: []const u8, config: GenerationConfig) !RequestId {
        const request_id = self.next_request_id;
        self.next_request_id += 1;

        const request = try BatchRequest.init(request_id, prompt, config, self.allocator);
        try self.queue.push(request);

        // Update queue statistics
        const queue_size = self.queue.size();
        self.stats.current_queue_size = @as(u32, @intCast(queue_size));
        if (self.stats.current_queue_size > self.stats.peak_queue_size) {
            self.stats.peak_queue_size = self.stats.current_queue_size;
        }

        return request_id;
    }

    /// Get processing statistics
    pub fn getStats(self: BatchProcessor) BatchStats {
        return self.stats;
    }

    /// Worker thread function
    fn workerFunction(context: *anyopaque) void {
        const ctx = @as(*@TypeOf(@as(*BatchProcessor, @ptrCast(@alignCast(context)))), @ptrCast(@alignCast(context)));
        const processor = ctx.processor;
        const worker_id = ctx.worker_id;
        defer processor.allocator.destroy(ctx);

        _ = worker_id; // Suppress unused warning

        while (processor.is_running) {
            // Get batch from queue
            const batch = processor.queue.popBatch(processor.config.max_batch_size, processor.config.max_wait_time_ms);
            if (batch.len == 0) {
                continue; // Timeout or no requests
            }

            defer processor.allocator.free(batch);

            // Process batch
            processor.processBatch(batch) catch |err| {
                // Handle batch processing error
                _ = err;
                processor.stats.error_count += 1;

                // Mark all requests in batch as errored
                for (batch) |*request| {
                    request.markCompleted(.Error);

                    if (processor.result_callback) |callback| {
                        const result = BatchResult{
                            .request_id = request.id,
                            .tokens = &[_]TokenId{},
                            .text = &[_]u8{},
                            .log_probs = &[_]f32{},
                            .total_log_prob = 0.0,
                            .num_tokens = 0,
                            .stop_reason = .Error,
                            .latency_ms = request.getLatency() orelse 0,
                            .queue_time_ms = request.getQueueTime() orelse 0,
                        };

                        callback(result, processor.user_data);
                    }

                    request.deinit(processor.allocator);
                }
            };
        }
    }

    /// Process a batch of requests
    fn processBatch(self: *BatchProcessor, batch: []BatchRequest) !void {
        // For simplicity, process requests sequentially in the batch
        // A full implementation would process them in parallel using batched matrix operations

        for (batch) |*request| {
            try self.processRequest(request);

            // Create result
            if (self.result_callback) |callback| {
                const result = BatchResult{
                    .request_id = request.id,
                    .tokens = try request.generated_tokens.toOwnedSlice(),
                    .text = try request.generated_text.toOwnedSlice(),
                    .log_probs = &[_]f32{}, // Simplified
                    .total_log_prob = request.cumulative_log_prob,
                    .num_tokens = @as(u32, @intCast(request.generated_tokens.items.len)),
                    .stop_reason = request.stop_reason orelse .Error,
                    .latency_ms = request.getLatency() orelse 0,
                    .queue_time_ms = request.getQueueTime() orelse 0,
                };

                self.stats.updateWithResult(result, @as(u32, @intCast(batch.len)));
                callback(result, self.user_data);
            }

            request.deinit(self.allocator);
        }
    }

    /// Process a single request (simplified)
    fn processRequest(self: *BatchProcessor, request: *BatchRequest) !void {
        // This is a simplified implementation that just generates a few tokens
        // A full implementation would use the actual model inference

        const prompt_tokens = try self.tokenizer.encode(request.prompt);
        defer self.allocator.free(prompt_tokens);

        // Simulate generation of a few tokens
        var tokens_generated: u32 = 0;
        const max_tokens = @min(request.max_tokens, 10); // Limit for demo

        while (tokens_generated < max_tokens) {
            // Simulate token generation (would use actual model here)
            const fake_token: TokenId = 100 + tokens_generated; // Fake token ID
            const fake_text = "token"; // Fake decoded text
            const fake_log_prob: f32 = -2.5;

            try request.addToken(fake_token, fake_text, fake_log_prob);
            tokens_generated += 1;

            // Simulate some processing time
            std.time.sleep(10_000_000); // 10ms per token
        }

        request.markCompleted(.MaxTokens);
    }
};

// Batch processing tests
test "batch configuration validation" {
    const testing = std.testing;

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
}

test "request priority comparison" {
    const testing = std.testing;

    try testing.expectEqual(std.math.Order.lt, RequestPriority.Low.compare(.Normal));
    try testing.expectEqual(std.math.Order.eq, RequestPriority.High.compare(.High));
    try testing.expectEqual(std.math.Order.gt, RequestPriority.Critical.compare(.Low));
}

test "batch request lifecycle" {
    const testing = std.testing;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var request = try BatchRequest.init(1, "test prompt", GenerationConfig.balanced(), allocator);
    defer request.deinit(allocator);

    try testing.expectEqual(@as(RequestId, 1), request.id);
    try testing.expectEqualStrings("test prompt", request.prompt);
    try testing.expect(!request.completed);
    try testing.expect(request.stop_reason == null);

    // Add a token
    try request.addToken(42, "hello", -1.5);
    try testing.expectEqual(@as(usize, 1), request.generated_tokens.items.len);
    try testing.expectEqual(@as(TokenId, 42), request.generated_tokens.items[0]);
    try testing.expectEqualStrings("hello", request.generated_text.items);

    // Complete request
    request.markCompleted(.MaxTokens);
    try testing.expect(request.completed);
    try testing.expect(request.stop_reason == .MaxTokens);
    try testing.expect(request.completed_at != null);
}

test "batching strategy descriptions" {
    const testing = std.testing;

    try testing.expectEqualStrings("Fixed batch size", BatchingStrategy.FixedSize.description());
    try testing.expectEqualStrings("Dynamic with timeout", BatchingStrategy.DynamicTimeout.description());
    try testing.expectEqualStrings("Adaptive batching", BatchingStrategy.Adaptive.description());
    try testing.expectEqualStrings("Continuous processing", BatchingStrategy.Continuous.description());
}

test "batch statistics tracking" {
    const testing = std.testing;

    var stats = BatchStats{};
    try testing.expectEqual(@as(u64, 0), stats.total_requests);
    try testing.expectEqual(@as(f32, 0.0), stats.avg_latency_ms);

    const result = BatchResult{
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

    stats.updateWithResult(result, 4);
    try testing.expectEqual(@as(u64, 1), stats.total_requests);
    try testing.expectEqual(@as(u64, 1), stats.total_batches);
    try testing.expectEqual(@as(f32, 100.0), stats.avg_latency_ms);
    try testing.expectEqual(@as(f32, 50.0), stats.avg_queue_time_ms);
    try testing.expectEqual(@as(f32, 4.0), stats.avg_batch_size);
}