//! KV Cache Optimization System
//!
//! This module implements Key-Value caching for transformer models, a critical
//! optimization that dramatically improves inference speed by avoiding
//! recomputation of attention keys and values for previously processed tokens.
//!
//! ## Educational Value
//! KV caching teaches fundamental inference optimization concepts:
//! - How autoregressive attention can be optimized
//! - Memory vs compute trade-offs in neural network inference
//! - Cache management strategies for long sequences
//! - Production-level performance optimization techniques

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const math = std.math;

const Tensor = @import("../foundation/tensor.zig").Tensor;
const ModelConfig = @import("../models/config.zig").ModelConfig;

/// KV cache entry for a single layer and single head
pub const KVCacheEntry = struct {
    /// Cached keys: [seq_len, head_dim]
    keys: Tensor(f32),
    /// Cached values: [seq_len, head_dim]
    values: Tensor(f32),
    /// Current sequence length in cache
    current_length: usize,
    /// Maximum sequence length this cache can hold
    max_length: usize,

    /// Initialize cache entry
    pub fn init(allocator: Allocator, max_length: usize, head_dim: usize) !KVCacheEntry {
        const keys = try Tensor(f32).init(allocator, &[_]usize{ max_length, head_dim });
        const values = try Tensor(f32).init(allocator, &[_]usize{ max_length, head_dim });

        return KVCacheEntry{
            .keys = keys,
            .values = values,
            .current_length = 0,
            .max_length = max_length,
        };
    }

    /// Clean up cache entry
    pub fn deinit(self: *KVCacheEntry) void {
        self.keys.deinit();
        self.values.deinit();
    }

    /// Add new keys and values to the cache
    pub fn append(self: *KVCacheEntry, new_keys: Tensor(f32), new_values: Tensor(f32)) !void {
        const new_seq_len = new_keys.shape[0];

        // Check if we have space
        if (self.current_length + new_seq_len > self.max_length) {
            return error.CacheOverflow;
        }

        // Copy new keys and values to cache
        const key_offset = self.current_length * self.keys.shape[1];
        const value_offset = self.current_length * self.values.shape[1];

        @memcpy(
            self.keys.data[key_offset..key_offset + new_keys.size],
            new_keys.data
        );
        @memcpy(
            self.values.data[value_offset..value_offset + new_values.size],
            new_values.data
        );

        self.current_length += new_seq_len;
    }

    /// Get cached keys up to current length
    pub fn getCachedKeys(self: KVCacheEntry, allocator: Allocator) !Tensor(f32) {
        const shape = [_]usize{ self.current_length, self.keys.shape[1] };
        var result = try Tensor(f32).init(allocator, &shape);

        const size_to_copy = self.current_length * self.keys.shape[1];
        @memcpy(result.data[0..size_to_copy], self.keys.data[0..size_to_copy]);

        return result;
    }

    /// Get cached values up to current length
    pub fn getCachedValues(self: KVCacheEntry, allocator: Allocator) !Tensor(f32) {
        const shape = [_]usize{ self.current_length, self.values.shape[1] };
        var result = try Tensor(f32).init(allocator, &shape);

        const size_to_copy = self.current_length * self.values.shape[1];
        @memcpy(result.data[0..size_to_copy], self.values.data[0..size_to_copy]);

        return result;
    }

    /// Reset cache to empty state
    pub fn reset(self: *KVCacheEntry) void {
        self.current_length = 0;
        // No need to zero memory - we track length
    }

    /// Check if cache is full
    pub fn isFull(self: KVCacheEntry) bool {
        return self.current_length >= self.max_length;
    }

    /// Get remaining capacity
    pub fn remainingCapacity(self: KVCacheEntry) usize {
        return self.max_length - self.current_length;
    }
};

/// Multi-head KV cache for a single layer
pub const LayerKVCache = struct {
    /// Cache entries for each head
    head_caches: []KVCacheEntry,
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head
    head_dim: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Allocator for memory management
    allocator: Allocator,

    /// Initialize layer cache
    pub fn init(allocator: Allocator, num_heads: usize, head_dim: usize, max_seq_len: usize) !LayerKVCache {
        var head_caches = try allocator.alloc(KVCacheEntry, num_heads);
        errdefer allocator.free(head_caches);

        for (head_caches, 0..) |*cache, i| {
            cache.* = KVCacheEntry.init(allocator, max_seq_len, head_dim) catch |err| {
                // Clean up previously initialized caches
                for (head_caches[0..i]) |*prev_cache| {
                    prev_cache.deinit();
                }
                allocator.free(head_caches);
                return err;
            };
        }

        return LayerKVCache{
            .head_caches = head_caches,
            .num_heads = num_heads,
            .head_dim = head_dim,
            .max_seq_len = max_seq_len,
            .allocator = allocator,
        };
    }

    /// Clean up layer cache
    pub fn deinit(self: *LayerKVCache) void {
        for (self.head_caches) |*cache| {
            cache.deinit();
        }
        self.allocator.free(self.head_caches);
    }

    /// Add keys and values for all heads
    pub fn appendMultiHead(self: *LayerKVCache, keys: Tensor(f32), values: Tensor(f32)) !void {
        // Keys and values shape: [batch_size, num_heads, seq_len, head_dim]
        // For simplicity, assume batch_size = 1
        const seq_len = keys.shape[2];

        for (0..self.num_heads) |head_idx| {
            // Extract keys and values for this head
            var head_keys = try self.extractHeadTensor(keys, head_idx, seq_len);
            defer head_keys.deinit();

            var head_values = try self.extractHeadTensor(values, head_idx, seq_len);
            defer head_values.deinit();

            // Append to head cache
            try self.head_caches[head_idx].append(head_keys, head_values);
        }
    }

    /// Extract tensor for a specific head
    fn extractHeadTensor(self: *LayerKVCache, tensor: Tensor(f32), head_idx: usize, seq_len: usize) !Tensor(f32) {
        var result = try Tensor(f32).init(self.allocator, &[_]usize{ seq_len, self.head_dim });

        // Calculate offset into the tensor
        const head_offset = head_idx * seq_len * self.head_dim;
        const size_to_copy = seq_len * self.head_dim;

        @memcpy(result.data, tensor.data[head_offset..head_offset + size_to_copy]);

        return result;
    }

    /// Get cached keys for all heads
    pub fn getAllCachedKeys(self: *LayerKVCache) !Tensor(f32) {
        const current_seq_len = self.head_caches[0].current_length;
        const shape = [_]usize{ 1, self.num_heads, current_seq_len, self.head_dim }; // [batch, heads, seq, head_dim]

        var result = try Tensor(f32).init(self.allocator, &shape);
        errdefer result.deinit();

        for (0..self.num_heads) |head_idx| {
            const head_keys = try self.head_caches[head_idx].getCachedKeys(self.allocator);
            defer head_keys.deinit();

            // Copy head keys to result tensor
            const head_offset = head_idx * current_seq_len * self.head_dim;
            @memcpy(result.data[head_offset..head_offset + head_keys.size], head_keys.data);
        }

        return result;
    }

    /// Get cached values for all heads
    pub fn getAllCachedValues(self: *LayerKVCache) !Tensor(f32) {
        const current_seq_len = self.head_caches[0].current_length;
        const shape = [_]usize{ 1, self.num_heads, current_seq_len, self.head_dim };

        var result = try Tensor(f32).init(self.allocator, &shape);
        errdefer result.deinit();

        for (0..self.num_heads) |head_idx| {
            const head_values = try self.head_caches[head_idx].getCachedValues(self.allocator);
            defer head_values.deinit();

            const head_offset = head_idx * current_seq_len * self.head_dim;
            @memcpy(result.data[head_offset..head_offset + head_values.size], head_values.data);
        }

        return result;
    }

    /// Reset all head caches
    pub fn reset(self: *LayerKVCache) void {
        for (self.head_caches) |*cache| {
            cache.reset();
        }
    }

    /// Get current sequence length (should be same for all heads)
    pub fn getCurrentLength(self: LayerKVCache) usize {
        return self.head_caches[0].current_length;
    }

    /// Check if any head cache is full
    pub fn isFull(self: LayerKVCache) bool {
        return self.head_caches[0].isFull(); // All heads should have same length
    }

    /// Get memory usage in bytes
    pub fn getMemoryUsage(self: LayerKVCache) usize {
        const keys_memory = self.num_heads * self.max_seq_len * self.head_dim * @sizeOf(f32);
        const values_memory = self.num_heads * self.max_seq_len * self.head_dim * @sizeOf(f32);
        return keys_memory + values_memory;
    }
};

/// Complete KV cache for all model layers
pub const ModelKVCache = struct {
    /// Cache for each transformer layer
    layer_caches: []LayerKVCache,
    /// Number of layers in the model
    num_layers: usize,
    /// Model configuration
    config: ModelConfig,
    /// Allocator for memory management
    allocator: Allocator,
    /// Cache statistics
    stats: CacheStats,

    /// Initialize complete model cache
    pub fn init(allocator: Allocator, config: ModelConfig) !ModelKVCache {
        const head_dim = config.headDim();

        var layer_caches = try allocator.alloc(LayerKVCache, config.num_layers);
        errdefer allocator.free(layer_caches);

        for (layer_caches, 0..) |*cache, i| {
            cache.* = LayerKVCache.init(allocator, config.num_heads, head_dim, config.max_seq_len) catch |err| {
                // Clean up previously initialized layer caches
                for (layer_caches[0..i]) |*prev_cache| {
                    prev_cache.deinit();
                }
                allocator.free(layer_caches);
                return err;
            };
        }

        return ModelKVCache{
            .layer_caches = layer_caches,
            .num_layers = config.num_layers,
            .config = config,
            .allocator = allocator,
            .stats = CacheStats{},
        };
    }

    /// Clean up model cache
    pub fn deinit(self: *ModelKVCache) void {
        for (self.layer_caches) |*cache| {
            cache.deinit();
        }
        self.allocator.free(self.layer_caches);
    }

    /// Update cache for a specific layer
    pub fn updateLayer(self: *ModelKVCache, layer_idx: usize, keys: Tensor(f32), values: Tensor(f32)) !void {
        if (layer_idx >= self.num_layers) {
            return error.InvalidLayerIndex;
        }

        try self.layer_caches[layer_idx].appendMultiHead(keys, values);
        self.stats.cache_updates += 1;
    }

    /// Get cached keys and values for a layer
    pub fn getLayerCache(self: *ModelKVCache, layer_idx: usize) !struct { keys: Tensor(f32), values: Tensor(f32) } {
        if (layer_idx >= self.num_layers) {
            return error.InvalidLayerIndex;
        }

        const keys = try self.layer_caches[layer_idx].getAllCachedKeys();
        const values = try self.layer_caches[layer_idx].getAllCachedValues();

        self.stats.cache_hits += 1;
        return .{ .keys = keys, .values = values };
    }

    /// Reset all caches
    pub fn reset(self: *ModelKVCache) void {
        for (self.layer_caches) |*cache| {
            cache.reset();
        }
        self.stats = CacheStats{};
    }

    /// Get current sequence length (should be consistent across layers)
    pub fn getCurrentLength(self: ModelKVCache) usize {
        if (self.layer_caches.len > 0) {
            return self.layer_caches[0].getCurrentLength();
        }
        return 0;
    }

    /// Check if cache is full
    pub fn isFull(self: ModelKVCache) bool {
        return self.layer_caches[0].isFull();
    }

    /// Get total memory usage
    pub fn getTotalMemoryUsage(self: ModelKVCache) usize {
        var total: usize = 0;
        for (self.layer_caches) |cache| {
            total += cache.getMemoryUsage();
        }
        return total;
    }

    /// Get memory usage statistics
    pub fn getMemoryStats(self: ModelKVCache) struct {
        total_bytes: usize,
        per_layer_bytes: usize,
        utilization_percent: f32,
    } {
        const total = self.getTotalMemoryUsage();
        const per_layer = if (self.num_layers > 0) total / self.num_layers else 0;
        const current_len = self.getCurrentLength();
        const utilization = @as(f32, @floatFromInt(current_len)) / @as(f32, @floatFromInt(self.config.max_seq_len)) * 100.0;

        return .{
            .total_bytes = total,
            .per_layer_bytes = per_layer,
            .utilization_percent = utilization,
        };
    }

    /// Compact cache by removing old tokens (sliding window)
    pub fn compact(self: *ModelKVCache, keep_last_n: usize) !void {
        // This is a simplified implementation
        // In practice, you'd implement a sliding window mechanism
        if (keep_last_n >= self.getCurrentLength()) {
            return; // Nothing to compact
        }

        // For now, just reset if we need to compact
        // A proper implementation would shift the cache contents
        self.reset();
        self.stats.cache_compactions += 1;
    }
};

/// KV cache performance statistics
pub const CacheStats = struct {
    /// Number of cache updates
    cache_updates: u64 = 0,
    /// Number of cache hits
    cache_hits: u64 = 0,
    /// Number of cache misses
    cache_misses: u64 = 0,
    /// Number of cache compactions
    cache_compactions: u64 = 0,

    /// Calculate hit rate
    pub fn hitRate(self: CacheStats) f32 {
        const total_accesses = self.cache_hits + self.cache_misses;
        if (total_accesses == 0) return 0.0;
        return @as(f32, @floatFromInt(self.cache_hits)) / @as(f32, @floatFromInt(total_accesses));
    }

    /// Print statistics
    pub fn print(self: CacheStats, writer: anytype) !void {
        try writer.print("KV Cache Statistics:\n", .{});
        try writer.print("  Updates: {d}\n", .{self.cache_updates});
        try writer.print("  Hits: {d}\n", .{self.cache_hits});
        try writer.print("  Misses: {d}\n", .{self.cache_misses});
        try writer.print("  Hit Rate: {d:.1}%\n", .{self.hitRate() * 100.0});
        try writer.print("  Compactions: {d}\n", .{self.cache_compactions});
    }
};

/// KV cache strategy for different use cases
pub const CacheStrategy = enum {
    /// Always cache (best for interactive chat)
    Always,
    /// Cache only for long sequences
    LongSequenceOnly,
    /// Adaptive caching based on memory pressure
    Adaptive,
    /// No caching (for batch processing)
    Disabled,

    /// Should cache be used for given sequence length?
    pub fn shouldCache(self: CacheStrategy, seq_len: usize, available_memory: ?usize) bool {
        return switch (self) {
            .Always => true,
            .LongSequenceOnly => seq_len > 512,
            .Adaptive => {
                if (available_memory) |memory| {
                    // Use cache if we have enough memory
                    const required_memory = seq_len * 1024; // Rough estimate
                    return memory > required_memory * 2; // 2x safety factor
                }
                return seq_len > 256; // Default threshold
            },
            .Disabled => false,
        };
    }
};

// KV cache tests
test "KV cache entry basic operations" {
    const testing = std.testing;
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

    // Create test tensors
    var keys = try Tensor(f32).init(allocator, &[_]usize{ 3, 64 });
    defer keys.deinit();
    var values = try Tensor(f32).init(allocator, &[_]usize{ 3, 64 });
    defer values.deinit();

    // Fill with test data
    for (keys.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i));
    }
    for (values.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) + 1000.0;
    }

    // Test append
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
}

test "layer KV cache multi-head operations" {
    const testing = std.testing;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var layer_cache = try LayerKVCache.init(allocator, 8, 64, 10); // 8 heads, 64 dim, 10 max seq
    defer layer_cache.deinit();

    // Test memory usage calculation
    const expected_memory = 8 * 10 * 64 * @sizeOf(f32) * 2; // keys + values
    try testing.expectEqual(expected_memory, layer_cache.getMemoryUsage());

    // Test initial state
    try testing.expectEqual(@as(usize, 0), layer_cache.getCurrentLength());
    try testing.expect(!layer_cache.isFull());
}

test "model KV cache initialization and cleanup" {
    const testing = std.testing;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = @import("../models/config.zig").ModelConfig.custom(512, 6, 8, 10000);

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
}

test "cache statistics tracking" {
    const testing = std.testing;

    var stats = CacheStats{};
    try testing.expectEqual(@as(f32, 0.0), stats.hitRate());

    stats.cache_hits = 80;
    stats.cache_misses = 20;
    try testing.expectApproxEqAbs(@as(f32, 0.8), stats.hitRate(), 0.01);

    stats.cache_updates = 100;
    try testing.expectEqual(@as(u64, 100), stats.cache_updates);
}

test "cache strategy decisions" {
    const testing = std.testing;

    try testing.expect(CacheStrategy.Always.shouldCache(10, null));
    try testing.expect(CacheStrategy.Always.shouldCache(1000, null));

    try testing.expect(!CacheStrategy.LongSequenceOnly.shouldCache(100, null));
    try testing.expect(CacheStrategy.LongSequenceOnly.shouldCache(1000, null));

    try testing.expect(!CacheStrategy.Disabled.shouldCache(1000, null));

    // Test adaptive strategy
    try testing.expect(CacheStrategy.Adaptive.shouldCache(100, 1000000)); // Enough memory
    try testing.expect(!CacheStrategy.Adaptive.shouldCache(100, 100)); // Not enough memory
}