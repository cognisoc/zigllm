// Advanced KV Cache Implementation
// Supports multi-sequence batching, sliding window attention, and efficient memory management
//
// Key features:
// 1. Multi-sequence batching for efficient inference
// 2. Sliding window attention for long sequences
// 3. Memory-efficient storage and retrieval
// 4. Dynamic allocation and growth
// 5. Sequence-aware cache management

const std = @import("std");
const Tensor = @import("../foundation/tensor.zig").Tensor;

// KV Cache Entry for a single sequence
pub const KVCacheEntry = struct {
    keys: Tensor(f32), // [seq_len, n_heads, head_dim]
    values: Tensor(f32), // [seq_len, n_heads, head_dim]
    sequence_length: usize,
    max_length: usize,
    layer_id: usize,

    pub fn init(allocator: std.mem.Allocator, max_length: usize, n_heads: usize, head_dim: usize, layer_id: usize) !KVCacheEntry {
        var keys = try Tensor(f32).init(allocator, &[_]usize{ max_length, n_heads, head_dim });
        keys.fill(0.0);
        var values = try Tensor(f32).init(allocator, &[_]usize{ max_length, n_heads, head_dim });
        values.fill(0.0);
        return KVCacheEntry{
            .keys = keys,
            .values = values,
            .sequence_length = 0,
            .max_length = max_length,
            .layer_id = layer_id,
        };
    }

    pub fn append(self: *KVCacheEntry, keys: *const Tensor(f32), values: *const Tensor(f32)) !void {
        const new_tokens = keys.shape[0];

        if (self.sequence_length + new_tokens > self.max_length) {
            return error.CacheOverflow;
        }

        // Copy keys and values to cache
        for (0..new_tokens) |i| {
            for (0..keys.shape[1]) |h| {
                for (0..keys.shape[2]) |d| {
                    const key_val = try keys.get(&[_]usize{ i, h, d });
                    const value_val = try values.get(&[_]usize{ i, h, d });

                    try self.keys.set(&[_]usize{ self.sequence_length + i, h, d }, key_val);
                    try self.values.set(&[_]usize{ self.sequence_length + i, h, d }, value_val);
                }
            }
        }

        self.sequence_length += new_tokens;
    }

    pub fn get(self: *const KVCacheEntry, start_pos: usize, length: usize) !struct { keys: Tensor(f32), values: Tensor(f32) } {
        if (start_pos + length > self.sequence_length) {
            return error.InvalidCacheAccess;
        }

        const allocator = self.keys.allocator;
        var cached_keys = try Tensor(f32).init(allocator, &[_]usize{ length, self.keys.shape[1], self.keys.shape[2] });
        cached_keys.fill(0.0);
        var cached_values = try Tensor(f32).init(allocator, &[_]usize{ length, self.values.shape[1], self.values.shape[2] });
        cached_values.fill(0.0);

        for (0..length) |i| {
            for (0..self.keys.shape[1]) |h| {
                for (0..self.keys.shape[2]) |d| {
                    const key_val = try self.keys.get(&[_]usize{ start_pos + i, h, d });
                    const value_val = try self.values.get(&[_]usize{ start_pos + i, h, d });

                    try cached_keys.set(&[_]usize{ i, h, d }, key_val);
                    try cached_values.set(&[_]usize{ i, h, d }, value_val);
                }
            }
        }

        return .{ .keys = cached_keys, .values = cached_values };
    }

    pub fn clear(self: *KVCacheEntry) void {
        self.sequence_length = 0;
        self.keys.fill(0.0);
        self.values.fill(0.0);
    }

    pub fn deinit(self: *KVCacheEntry) void {
        self.keys.deinit();
        self.values.deinit();
    }
};

// Multi-Sequence KV Cache Manager
pub const MultiSequenceKVCache = struct {
    sequences: std.AutoHashMap(u64, []KVCacheEntry),
    n_layers: usize,
    n_heads: usize,
    head_dim: usize,
    max_seq_length: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n_layers: usize, n_heads: usize, head_dim: usize, max_seq_length: usize) MultiSequenceKVCache {
        return MultiSequenceKVCache{
            .sequences = std.AutoHashMap(u64, []KVCacheEntry).init(allocator),
            .n_layers = n_layers,
            .n_heads = n_heads,
            .head_dim = head_dim,
            .max_seq_length = max_seq_length,
            .allocator = allocator,
        };
    }

    pub fn getOrCreateSequence(self: *MultiSequenceKVCache, sequence_id: u64) ![]KVCacheEntry {
        if (self.sequences.get(sequence_id)) |entries| {
            return entries;
        }

        // Create new sequence cache
        const cache_entries = try self.allocator.alloc(KVCacheEntry, self.n_layers);
        for (0..self.n_layers) |i| {
            cache_entries[i] = try KVCacheEntry.init(
                self.allocator,
                self.max_seq_length,
                self.n_heads,
                self.head_dim,
                i,
            );
        }

        try self.sequences.put(sequence_id, cache_entries);
        return cache_entries;
    }

    pub fn appendToSequence(self: *MultiSequenceKVCache, sequence_id: u64, layer_id: usize, keys: *const Tensor(f32), values: *const Tensor(f32)) !void {
        const cache_entries = try self.getOrCreateSequence(sequence_id);
        try cache_entries[layer_id].append(keys, values);
    }

    pub fn getFromSequence(self: *MultiSequenceKVCache, sequence_id: u64, layer_id: usize, start_pos: usize, length: usize) !?struct { keys: Tensor(f32), values: Tensor(f32) } {
        const cache_entries = self.sequences.get(sequence_id) orelse return null;
        return try cache_entries[layer_id].get(start_pos, length);
    }

    pub fn getSequenceLength(self: *MultiSequenceKVCache, sequence_id: u64) usize {
        const cache_entries = self.sequences.get(sequence_id) orelse return 0;
        return cache_entries[0].sequence_length; // All layers should have same length
    }

    pub fn clearSequence(self: *MultiSequenceKVCache, sequence_id: u64) void {
        const cache_entries = self.sequences.get(sequence_id) orelse return;
        for (cache_entries) |*entry| {
            entry.clear();
        }
    }

    pub fn removeSequence(self: *MultiSequenceKVCache, sequence_id: u64) void {
        const cache_entries = self.sequences.get(sequence_id) orelse return;

        // Clean up memory
        for (cache_entries) |*entry| {
            entry.deinit();
        }
        self.allocator.free(cache_entries);

        _ = self.sequences.remove(sequence_id);
    }

    pub fn getMemoryUsage(self: *MultiSequenceKVCache) u64 {
        var total_memory: u64 = 0;

        var iterator = self.sequences.iterator();
        while (iterator.next()) |entry| {
            const cache_entries = entry.value_ptr.*;
            for (cache_entries) |cache_entry| {
                const entry_size = cache_entry.max_length * cache_entry.keys.shape[1] * cache_entry.keys.shape[2] * 2; // keys + values
                total_memory += entry_size * @sizeOf(f32); // assuming f32 tensors
            }
        }

        return total_memory;
    }

    pub fn deinit(self: *MultiSequenceKVCache) void {
        var iterator = self.sequences.iterator();
        while (iterator.next()) |entry| {
            const cache_entries = entry.value_ptr.*;
            for (cache_entries) |*cache_entry| {
                cache_entry.deinit();
            }
            self.allocator.free(cache_entries);
        }
        self.sequences.deinit();
    }
};

// Sliding Window KV Cache
// Automatically manages sliding window attention by evicting old tokens
pub const SlidingWindowKVCache = struct {
    cache: MultiSequenceKVCache,
    window_size: usize,

    pub fn init(allocator: std.mem.Allocator, n_layers: usize, n_heads: usize, head_dim: usize, window_size: usize) SlidingWindowKVCache {
        return SlidingWindowKVCache{
            .cache = MultiSequenceKVCache.init(allocator, n_layers, n_heads, head_dim, window_size * 2), // Extra buffer
            .window_size = window_size,
        };
    }

    pub fn appendWithSliding(self: *SlidingWindowKVCache, sequence_id: u64, layer_id: usize, keys: *const Tensor(f32), values: *const Tensor(f32)) !void {
        const current_length = self.cache.getSequenceLength(sequence_id);
        const new_tokens = keys.shape[0];

        if (current_length + new_tokens > self.window_size) {
            // Need to slide the window
            try self.slideWindow(sequence_id, layer_id, new_tokens);
        }

        try self.cache.appendToSequence(sequence_id, layer_id, keys, values);
    }

    fn slideWindow(self: *SlidingWindowKVCache, sequence_id: u64, layer_id: usize, new_tokens: usize) !void {
        const cache_entries = try self.cache.getOrCreateSequence(sequence_id);
        const current_entry = &cache_entries[layer_id];

        const current_length = current_entry.sequence_length;
        const tokens_to_evict = (current_length + new_tokens) - self.window_size;

        if (tokens_to_evict >= current_length) {
            // Evict everything
            current_entry.clear();
            return;
        }

        // Shift the cache to remove oldest tokens
        const keep_length = current_length - tokens_to_evict;

        // Create temporary storage
        var temp_keys = try Tensor(f32).init(current_entry.keys.allocator, &[_]usize{ keep_length, current_entry.keys.shape[1], current_entry.keys.shape[2] });
        temp_keys.fill(0.0);
        defer temp_keys.deinit();
        var temp_values = try Tensor(f32).init(current_entry.values.allocator, &[_]usize{ keep_length, current_entry.values.shape[1], current_entry.values.shape[2] });
        temp_values.fill(0.0);
        defer temp_values.deinit();

        // Copy the tokens we want to keep
        for (0..keep_length) |i| {
            for (0..current_entry.keys.shape[1]) |h| {
                for (0..current_entry.keys.shape[2]) |d| {
                    const key_val = try current_entry.keys.get(&[_]usize{ tokens_to_evict + i, h, d });
                    const value_val = try current_entry.values.get(&[_]usize{ tokens_to_evict + i, h, d });

                    try temp_keys.set(&[_]usize{ i, h, d }, key_val);
                    try temp_values.set(&[_]usize{ i, h, d }, value_val);
                }
            }
        }

        // Clear the cache and copy back
        current_entry.clear();
        for (0..keep_length) |i| {
            for (0..current_entry.keys.shape[1]) |h| {
                for (0..current_entry.keys.shape[2]) |d| {
                    const key_val = try temp_keys.get(&[_]usize{ i, h, d });
                    const value_val = try temp_values.get(&[_]usize{ i, h, d });

                    try current_entry.keys.set(&[_]usize{ i, h, d }, key_val);
                    try current_entry.values.set(&[_]usize{ i, h, d }, value_val);
                }
            }
        }

        current_entry.sequence_length = keep_length;
    }

    pub fn getWithWindow(self: *SlidingWindowKVCache, sequence_id: u64, layer_id: usize, _: usize) !?struct { keys: Tensor(f32), values: Tensor(f32) } {
        const current_length = self.cache.getSequenceLength(sequence_id);
        if (current_length == 0) {
            return null;
        }

        // For sliding window, we typically want to see the last window_size tokens
        const window_start = if (current_length > self.window_size) current_length - self.window_size else 0;
        const window_length = current_length - window_start;

        return try self.cache.getFromSequence(sequence_id, layer_id, window_start, window_length);
    }

    pub fn deinit(self: *SlidingWindowKVCache) void {
        self.cache.deinit();
    }
};

// Exported type alias for use by other modules
pub const ModelKVCache = MultiSequenceKVCache;

// KV Cache Statistics for monitoring
pub const KVCacheStats = struct {
    total_sequences: u32,
    total_memory_bytes: u64,
    average_sequence_length: f32,
    cache_hit_rate: f32,
    cache_efficiency: f32, // ratio of used vs allocated memory

    pub fn compute(cache: *const MultiSequenceKVCache) KVCacheStats {
        var total_sequences: u32 = 0;
        var total_length: u64 = 0;
        var total_memory: u64 = 0;
        var used_memory: u64 = 0;

        var iterator = cache.sequences.iterator();
        while (iterator.next()) |entry| {
            total_sequences += 1;
            const cache_entries = entry.value_ptr.*;

            if (cache_entries.len > 0) {
                const seq_length = cache_entries[0].sequence_length;
                total_length += seq_length;

                for (cache_entries) |cache_entry| {
                    const max_memory = cache_entry.max_length * cache_entry.keys.shape[1] * cache_entry.keys.shape[2] * 2 * @sizeOf(f32);
                    const used_mem = seq_length * cache_entry.keys.shape[1] * cache_entry.keys.shape[2] * 2 * @sizeOf(f32);
                    total_memory += max_memory;
                    used_memory += used_mem;
                }
            }
        }

        return KVCacheStats{
            .total_sequences = total_sequences,
            .total_memory_bytes = total_memory,
            .average_sequence_length = if (total_sequences > 0) @as(f32, @floatFromInt(total_length)) / @as(f32, @floatFromInt(total_sequences)) else 0.0,
            .cache_hit_rate = 0.0, // Would need additional tracking for hits/misses
            .cache_efficiency = if (total_memory > 0) @as(f32, @floatFromInt(used_memory)) / @as(f32, @floatFromInt(total_memory)) else 0.0,
        };
    }

    pub fn print(self: KVCacheStats) void {
        std.debug.print("=== KV Cache Statistics ===\n", .{});
        std.debug.print("Total Sequences: {}\n", .{self.total_sequences});
        std.debug.print("Total Memory: {d:.2} MB\n", .{@as(f32, @floatFromInt(self.total_memory_bytes)) / 1_048_576.0});
        std.debug.print("Average Sequence Length: {d:.1}\n", .{self.average_sequence_length});
        std.debug.print("Cache Efficiency: {d:.1}%\n", .{self.cache_efficiency * 100.0});
        std.debug.print("===========================\n", .{});
    }
};

// Utility functions for KV cache management
pub const KVCacheUtils = struct {
    pub fn estimateMemoryUsage(n_sequences: u64, max_seq_length: u64, n_layers: u64, n_heads: u64, head_dim: u64) u64 {
        const per_token_memory = n_layers * n_heads * head_dim * 2 * @sizeOf(f32); // keys + values
        return n_sequences * max_seq_length * per_token_memory;
    }

    pub fn recommendedWindowSize(model_context_length: u32, available_memory_mb: u64) u32 {
        // Simple heuristic based on available memory
        const memory_bytes = available_memory_mb * 1_048_576;
        const estimated_per_token = 4096; // rough estimate in bytes

        const max_tokens_from_memory: u32 = @intCast(@min(memory_bytes / estimated_per_token, model_context_length));
        return @min(max_tokens_from_memory, model_context_length);
    }
};

// Tests
test "KV cache entry basic operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var entry = try KVCacheEntry.init(allocator, 16, 2, 4, 0);
    defer entry.deinit();

    try testing.expectEqual(@as(usize, 0), entry.sequence_length);
    try testing.expectEqual(@as(usize, 16), entry.max_length);
}

test "KV cache utility memory estimation" {
    const testing = std.testing;

    const memory = KVCacheUtils.estimateMemoryUsage(1, 512, 32, 32, 128);
    try testing.expect(memory > 0);
}

test "KV cache recommended window size" {
    const testing = std.testing;

    const window = KVCacheUtils.recommendedWindowSize(2048, 1024);
    try testing.expect(window > 0);
    try testing.expect(window <= 2048);
}
