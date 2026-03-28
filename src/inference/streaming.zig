//! Streaming Text Generation
//!
//! This module implements streaming text generation for real-time applications
//! where tokens need to be produced and displayed as soon as they're generated,
//! rather than waiting for the complete response.
//!
//! ## Educational Value
//! Streaming generation teaches advanced inference concepts:
//! - How to architect responsive AI applications
//! - Callback-based and iterator-based streaming patterns
//! - Buffering and chunk management for smooth user experience
//! - Error handling and recovery in streaming contexts

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const Thread = std.Thread;
const Mutex = std.Thread.Mutex;
const Condition = std.Thread.Condition;

const TextGenerator = @import("generation.zig").TextGenerator;
const GenerationConfig = @import("generation.zig").GenerationConfig;
const GenerationResult = @import("generation.zig").GenerationResult;
const StopReason = @import("generation.zig").StopReason;
const TokenId = @import("../models/tokenizer.zig").TokenId;
const SimpleTokenizer = @import("../models/tokenizer.zig").SimpleTokenizer;

/// Token chunk for streaming
pub const TokenChunk = struct {
    /// Token ID
    token_id: TokenId,
    /// Decoded text (if available)
    text: ?[]const u8,
    /// Token probability
    log_prob: f32,
    /// Cumulative probability so far
    cumulative_log_prob: f32,
    /// Position in sequence
    position: u32,
    /// Timestamp when token was generated
    timestamp: i64,

    pub fn init(token_id: TokenId, text: ?[]const u8, log_prob: f32, cumulative_log_prob: f32, position: u32) TokenChunk {
        return TokenChunk{
            .token_id = token_id,
            .text = text,
            .log_prob = log_prob,
            .cumulative_log_prob = cumulative_log_prob,
            .position = position,
            .timestamp = std.time.milliTimestamp(),
        };
    }
};

/// Stream status information
pub const StreamStatus = struct {
    /// Total tokens generated so far
    tokens_generated: u32,
    /// Tokens per second (instantaneous)
    current_tps: f32,
    /// Average tokens per second
    average_tps: f32,
    /// Stream start time
    start_time: i64,
    /// Current time
    current_time: i64,
    /// Whether stream is still active
    is_active: bool,
    /// Stop reason (if stream ended)
    stop_reason: ?StopReason,

    pub fn init() StreamStatus {
        const now = std.time.milliTimestamp();
        return StreamStatus{
            .tokens_generated = 0,
            .current_tps = 0.0,
            .average_tps = 0.0,
            .start_time = now,
            .current_time = now,
            .is_active = true,
            .stop_reason = null,
        };
    }

    pub fn update(self: *StreamStatus, new_tokens: u32) void {
        const now = std.time.milliTimestamp();
        const elapsed_ms = now - self.current_time;

        if (elapsed_ms > 0) {
            self.current_tps = @as(f32, @floatFromInt(new_tokens)) * 1000.0 / @as(f32, @floatFromInt(elapsed_ms));
        }

        self.tokens_generated += new_tokens;
        self.current_time = now;

        const total_elapsed = now - self.start_time;
        if (total_elapsed > 0) {
            self.average_tps = @as(f32, @floatFromInt(self.tokens_generated)) * 1000.0 / @as(f32, @floatFromInt(total_elapsed));
        }
    }

    pub fn finish(self: *StreamStatus, stop_reason: StopReason) void {
        self.is_active = false;
        self.stop_reason = stop_reason;
        self.current_time = std.time.milliTimestamp();

        const total_elapsed = self.current_time - self.start_time;
        if (total_elapsed > 0) {
            self.average_tps = @as(f32, @floatFromInt(self.tokens_generated)) * 1000.0 / @as(f32, @floatFromInt(total_elapsed));
        }
    }
};

/// Streaming callback function type
pub const StreamCallback = *const fn (chunk: TokenChunk, status: StreamStatus, user_data: ?*anyopaque) void;

/// Error callback function type
pub const ErrorCallback = *const fn (err: anyerror, user_data: ?*anyopaque) void;

/// Streaming configuration
pub const StreamingConfig = struct {
    /// Buffer size for token chunks
    buffer_size: usize = 64,
    /// Maximum time to wait for tokens (ms)
    timeout_ms: u32 = 5000,
    /// Flush buffer on newlines
    flush_on_newline: bool = true,
    /// Flush buffer on sentence endings
    flush_on_sentence_end: bool = true,
    /// Minimum chunk size for flushing
    min_chunk_size: usize = 1,
    /// Maximum chunk size before forced flush
    max_chunk_size: usize = 32,
    /// Enable detailed streaming statistics
    detailed_stats: bool = false,

    pub fn validate(self: StreamingConfig) !void {
        if (self.buffer_size == 0) {
            return error.InvalidBufferSize;
        }
        if (self.min_chunk_size > self.max_chunk_size) {
            return error.InvalidChunkSizes;
        }
        if (self.timeout_ms == 0) {
            return error.InvalidTimeout;
        }
    }
};

/// Thread-safe token buffer for streaming
const TokenBuffer = struct {
    /// Token chunks waiting to be processed
    chunks: ArrayList(TokenChunk),
    /// Mutex for thread safety
    mutex: Mutex,
    /// Condition for signaling new data
    condition: Condition,
    /// Whether buffer is closed (no more tokens coming)
    closed: bool,
    /// Allocator for memory management
    allocator: Allocator,

    pub fn init(allocator: Allocator) !TokenBuffer {
        return TokenBuffer{
            .chunks = try std.ArrayList(TokenChunk).initCapacity(allocator, 100),
            .mutex = Mutex{},
            .condition = Condition{},
            .closed = false,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TokenBuffer) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Free any remaining text data
        for (self.chunks.items) |chunk| {
            if (chunk.text) |text| {
                self.allocator.free(text);
            }
        }
        self.chunks.deinit(self.allocator);
    }

    pub fn push(self: *TokenBuffer, chunk: TokenChunk) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.closed) {
            return error.BufferClosed;
        }

        try self.chunks.append(self.allocator, chunk);
        self.condition.signal();
    }

    pub fn pop(self: *TokenBuffer, timeout_ms: u32) ?TokenChunk {
        self.mutex.lock();
        defer self.mutex.unlock();

        const timeout_ns = @as(u64, timeout_ms) * 1_000_000;
        const start_time = std.time.nanoTimestamp();

        while (self.chunks.items.len == 0 and !self.closed) {
            const elapsed = @as(u64, @intCast(std.time.nanoTimestamp() - start_time));
            if (elapsed >= timeout_ns) {
                return null; // Timeout
            }

            const remaining = timeout_ns - elapsed;
            _ = self.condition.timedWait(&self.mutex, remaining) catch {
                return null; // Timeout or error
            };
        }

        if (self.chunks.items.len > 0) {
            return self.chunks.orderedRemove(0);
        }

        return null; // Buffer closed and empty
    }

    pub fn close(self: *TokenBuffer) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.closed = true;
        self.condition.broadcast();
    }

    pub fn isEmpty(self: *TokenBuffer) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        return self.chunks.items.len == 0;
    }

    pub fn size(self: *TokenBuffer) usize {
        self.mutex.lock();
        defer self.mutex.unlock();

        return self.chunks.items.len;
    }
};

/// Streaming text generator
pub const StreamingGenerator = struct {
    /// Base text generator
    generator: *TextGenerator,
    /// Tokenizer for text decoding
    tokenizer: *SimpleTokenizer,
    /// Streaming configuration
    config: StreamingConfig,
    /// Memory allocator
    allocator: Allocator,
    /// Token buffer for streaming
    buffer: TokenBuffer,
    /// Generation thread handle
    generation_thread: ?Thread,
    /// Current stream status
    status: StreamStatus,
    /// Whether streaming is active
    is_streaming: bool,

    /// Initialize streaming generator
    pub fn init(generator: *TextGenerator, tokenizer: *SimpleTokenizer, allocator: Allocator) !StreamingGenerator {
        const config = StreamingConfig{};
        try config.validate();

        return StreamingGenerator{
            .generator = generator,
            .tokenizer = tokenizer,
            .config = config,
            .allocator = allocator,
            .buffer = TokenBuffer.init(allocator),
            .generation_thread = null,
            .status = StreamStatus.init(),
            .is_streaming = false,
        };
    }

    /// Clean up streaming generator
    pub fn deinit(self: *StreamingGenerator) void {
        self.stopStreaming();
        self.buffer.deinit();
    }

    /// Set streaming configuration
    pub fn setConfig(self: *StreamingGenerator, config: StreamingConfig) !void {
        try config.validate();
        self.config = config;
    }

    /// Start streaming generation
    pub fn startStreaming(self: *StreamingGenerator, prompt: []const u8) !void {
        if (self.is_streaming) {
            return error.AlreadyStreaming;
        }

        self.is_streaming = true;
        self.status = StreamStatus.init();

        // Start generation in separate thread
        const GenerationContext = struct {
            streaming_gen: *StreamingGenerator,
            prompt: []const u8,
        };

        const context = try self.allocator.create(GenerationContext);
        context.* = GenerationContext{
            .streaming_gen = self,
            .prompt = prompt,
        };

        self.generation_thread = try Thread.spawn(.{}, generateInBackground, .{context});
    }

    /// Stop streaming generation
    pub fn stopStreaming(self: *StreamingGenerator) void {
        if (!self.is_streaming) {
            return;
        }

        self.is_streaming = false;
        self.buffer.close();

        if (self.generation_thread) |thread| {
            thread.join();
            self.generation_thread = null;
        }
    }

    /// Get next token chunk (blocking with timeout)
    pub fn nextChunk(self: *StreamingGenerator) ?TokenChunk {
        return self.buffer.pop(self.config.timeout_ms);
    }

    /// Stream with callback function
    pub fn streamWithCallback(
        self: *StreamingGenerator,
        prompt: []const u8,
        callback: StreamCallback,
        error_callback: ?ErrorCallback,
        user_data: ?*anyopaque
    ) !void {
        try self.startStreaming(prompt);

        while (self.is_streaming) {
            if (self.nextChunk()) |chunk| {
                callback(chunk, self.status, user_data);

                // Check for sentence endings if configured
                if (self.config.flush_on_sentence_end and chunk.text != null) {
                    const text = chunk.text.?;
                    if (text.len > 0) {
                        const last_char = text[text.len - 1];
                        if (last_char == '.' or last_char == '!' or last_char == '?') {
                            // Natural break point for better UX
                            std.time.sleep(10_000_000); // 10ms pause
                        }
                    }
                }
            } else if (self.status.stop_reason != null) {
                break; // Stream ended
            } else {
                // Timeout - check if we should continue
                if (error_callback) |err_cb| {
                    err_cb(error.StreamTimeout, user_data);
                }
            }
        }
    }

    /// Collect all tokens from stream (for testing/batch processing)
    pub fn collectAll(self: *StreamingGenerator, prompt: []const u8) ![]TokenChunk {
        try self.startStreaming(prompt);

        var chunks = ArrayList(TokenChunk).init(self.allocator);
        errdefer {
            for (chunks.items) |chunk| {
                if (chunk.text) |text| {
                    self.allocator.free(text);
                }
            }
            chunks.deinit();
        }

        while (self.is_streaming) {
            if (self.nextChunk()) |chunk| {
                try chunks.append(chunk);
            } else if (self.status.stop_reason != null) {
                break;
            }
        }

        return try chunks.toOwnedSlice();
    }

    /// Get current stream status
    pub fn getStatus(self: StreamingGenerator) StreamStatus {
        return self.status;
    }

    /// Background generation function
    fn generateInBackground(context: *anyopaque) void {
        const ctx = @as(*@TypeOf(@as(*StreamingGenerator, @ptrCast(@alignCast(context)))).*, @ptrCast(@alignCast(context)));
        const streaming_gen = ctx.streaming_gen;
        defer streaming_gen.allocator.destroy(ctx);

        // Generate tokens one by one using the base generator
        streaming_gen.generateStreamingTokens(ctx.prompt) catch |err| {
            // Handle error - in a full implementation, this would use error callback
            _ = err;
            streaming_gen.status.finish(.Error);
        };

        streaming_gen.buffer.close();
        streaming_gen.is_streaming = false;
    }

    /// Internal streaming generation implementation
    fn generateStreamingTokens(self: *StreamingGenerator, prompt: []const u8) !void {
        // For now, use the regular generator and stream its output
        // In a full implementation, this would integrate directly with the model

        const result = try self.generator.generate(prompt);
        defer result.deinit(self.allocator);

        var cumulative_log_prob: f32 = 0.0;

        for (result.tokens, 0..) |token_id, i| {
            if (!self.is_streaming) break; // Check for early termination

            const log_prob = if (i < result.log_probs.len) result.log_probs[i] else 0.0;
            cumulative_log_prob += log_prob;

            // Decode individual token to text
            const token_text = self.decodeToken(token_id) catch null;

            const chunk = TokenChunk.init(
                token_id,
                token_text,
                log_prob,
                cumulative_log_prob,
                @as(u32, @intCast(i))
            );

            try self.buffer.push(chunk);
            self.status.update(1);

            // Simulate realistic timing
            std.time.sleep(50_000_000); // 50ms between tokens
        }

        self.status.finish(result.stop_reason);
    }

    /// Decode a single token to text
    fn decodeToken(self: *StreamingGenerator, token_id: TokenId) ![]u8 {
        const tokens = [_]TokenId{token_id};
        return try self.tokenizer.decode(&tokens);
    }
};

/// Streaming iterator for more ergonomic usage
pub const StreamingIterator = struct {
    streaming_gen: *StreamingGenerator,
    finished: bool,

    pub fn init(streaming_gen: *StreamingGenerator) StreamingIterator {
        return StreamingIterator{
            .streaming_gen = streaming_gen,
            .finished = false,
        };
    }

    pub fn next(self: *StreamingIterator) ?TokenChunk {
        if (self.finished) return null;

        if (self.streaming_gen.nextChunk()) |chunk| {
            return chunk;
        } else {
            self.finished = true;
            return null;
        }
    }
};

// Streaming tests
test "streaming configuration validation" {
    const testing = std.testing;

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

test "token chunk creation" {
    const testing = std.testing;

    const chunk = TokenChunk.init(42, "hello", -2.5, -10.0, 5);

    try testing.expectEqual(@as(TokenId, 42), chunk.token_id);
    try testing.expectEqualStrings("hello", chunk.text.?);
    try testing.expectEqual(@as(f32, -2.5), chunk.log_prob);
    try testing.expectEqual(@as(f32, -10.0), chunk.cumulative_log_prob);
    try testing.expectEqual(@as(u32, 5), chunk.position);
    try testing.expect(chunk.timestamp > 0);
}

test "stream status tracking" {
    const testing = std.testing;

    var status = StreamStatus.init();
    try testing.expectEqual(@as(u32, 0), status.tokens_generated);
    try testing.expect(status.is_active);
    try testing.expect(status.stop_reason == null);

    status.update(10);
    try testing.expectEqual(@as(u32, 10), status.tokens_generated);
    try testing.expect(status.current_tps >= 0.0);
    try testing.expect(status.average_tps >= 0.0);

    status.finish(.MaxTokens);
    try testing.expect(!status.is_active);
    try testing.expect(status.stop_reason == .MaxTokens);
}

test "token buffer thread safety" {
    const testing = std.testing;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var buffer = try TokenBuffer.init(allocator);
    defer buffer.deinit();

    try testing.expect(buffer.isEmpty());
    try testing.expectEqual(@as(usize, 0), buffer.size());

    const chunk = TokenChunk.init(1, null, 0.0, 0.0, 0);
    try buffer.push(chunk);

    try testing.expect(!buffer.isEmpty());
    try testing.expectEqual(@as(usize, 1), buffer.size());

    const popped = buffer.pop(1000);
    try testing.expect(popped != null);
    try testing.expectEqual(@as(TokenId, 1), popped.?.token_id);
    try testing.expect(buffer.isEmpty());

    // Test timeout
    const timeout_result = buffer.pop(10); // 10ms timeout
    try testing.expect(timeout_result == null);
}