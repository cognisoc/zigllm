//! Tokenization and Vocabulary System
//!
//! This module implements tokenization for transformer models, focusing on
//! the SentencePiece tokenizer used by LLaMA models. It provides both
//! educational understanding and production-ready performance.
//!
//! ## Educational Value
//! Tokenization is the bridge between human language and neural networks:
//! - How text is converted to numerical representations
//! - Subword tokenization strategies and their trade-offs
//! - Vocabulary management and out-of-vocabulary handling
//! - Special tokens and their roles in model behavior

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const HashMap = std.HashMap;

/// Token ID type used throughout the system
pub const TokenId = u32;

/// Special token definitions
pub const SpecialTokens = struct {
    /// Unknown/out-of-vocabulary token
    pub const UNK: TokenId = 0;
    /// Beginning of sequence
    pub const BOS: TokenId = 1;
    /// End of sequence
    pub const EOS: TokenId = 2;
    /// Padding token for batch processing
    pub const PAD: TokenId = 3;

    /// Check if a token ID is a special token
    pub fn isSpecial(token_id: TokenId) bool {
        return token_id <= 3;
    }

    /// Get human-readable name for special tokens
    pub fn name(token_id: TokenId) ?[]const u8 {
        return switch (token_id) {
            UNK => "<unk>",
            BOS => "<s>",
            EOS => "</s>",
            PAD => "<pad>",
            else => null,
        };
    }
};

/// Token piece representing a subword unit
pub const TokenPiece = struct {
    /// The text content of this token
    piece: []const u8,
    /// Score for SentencePiece (higher = more preferred)
    score: f32,
    /// Token ID in the vocabulary
    id: TokenId,
    /// Whether this is a special token
    is_special: bool,

    /// Create a new token piece
    pub fn init(piece: []const u8, score: f32, id: TokenId, is_special: bool) TokenPiece {
        return TokenPiece{
            .piece = piece,
            .score = score,
            .id = id,
            .is_special = is_special,
        };
    }
};

/// Vocabulary management for tokenization
pub const Vocabulary = struct {
    /// Allocator for memory management
    allocator: Allocator,
    /// Map from token piece to token ID
    piece_to_id: HashMap([]const u8, TokenId, StringContext, std.hash_map.default_max_load_percentage),
    /// Map from token ID to token piece
    id_to_piece: ArrayList(TokenPiece),
    /// Vocabulary size (number of tokens)
    vocab_size: usize,

    const StringContext = struct {
        pub fn hash(self: @This(), s: []const u8) u64 {
            _ = self;
            return std.hash_map.hashString(s);
        }

        pub fn eql(self: @This(), a: []const u8, b: []const u8) bool {
            _ = self;
            return std.mem.eql(u8, a, b);
        }
    };

    /// Initialize vocabulary
    pub fn init(allocator: Allocator, vocab_size: usize) !Vocabulary {
        var vocab = Vocabulary{
            .allocator = allocator,
            .piece_to_id = HashMap([]const u8, TokenId, StringContext, std.hash_map.default_max_load_percentage).init(allocator),
            .id_to_piece = try std.ArrayList(TokenPiece).initCapacity(allocator, vocab_size),
            .vocab_size = vocab_size,
        };

        // Reserve space for efficiency
        try vocab.id_to_piece.ensureTotalCapacity(allocator, vocab_size);

        // Add special tokens
        try vocab.addToken("<unk>", 0.0, SpecialTokens.UNK, true);
        try vocab.addToken("<s>", 0.0, SpecialTokens.BOS, true);
        try vocab.addToken("</s>", 0.0, SpecialTokens.EOS, true);
        try vocab.addToken("<pad>", 0.0, SpecialTokens.PAD, true);

        return vocab;
    }

    /// Clean up vocabulary
    pub fn deinit(self: *Vocabulary) void {
        // Free allocated strings
        for (self.id_to_piece.items) |piece| {
            self.allocator.free(piece.piece);
        }
        self.id_to_piece.deinit(self.allocator);
        self.piece_to_id.deinit();
    }

    /// Add a token to the vocabulary
    pub fn addToken(self: *Vocabulary, piece: []const u8, score: f32, id: TokenId, is_special: bool) !void {
        // Allocate owned copy of the piece string
        const owned_piece = try self.allocator.dupe(u8, piece);
        errdefer self.allocator.free(owned_piece);

        // Create token piece
        const token_piece = TokenPiece.init(owned_piece, score, id, is_special);

        // Ensure vector is large enough
        if (id >= self.id_to_piece.items.len) {
            try self.id_to_piece.resize(self.allocator, id + 1);
        }

        // Add to both maps
        try self.piece_to_id.put(owned_piece, id);
        self.id_to_piece.items[id] = token_piece;
    }

    /// Get token ID from piece string
    pub fn getTokenId(self: Vocabulary, piece: []const u8) ?TokenId {
        return self.piece_to_id.get(piece);
    }

    /// Get token piece from ID
    pub fn getTokenPiece(self: Vocabulary, id: TokenId) ?TokenPiece {
        if (id >= self.id_to_piece.items.len) return null;
        return self.id_to_piece.items[id];
    }

    /// Check if token exists in vocabulary
    pub fn contains(self: Vocabulary, piece: []const u8) bool {
        return self.piece_to_id.contains(piece);
    }

    /// Get current vocabulary size
    pub fn size(self: Vocabulary) usize {
        return self.id_to_piece.items.len;
    }

    /// Load vocabulary from SentencePiece model file
    pub fn loadSentencePiece(self: *Vocabulary, file_path: []const u8) !void {
        // TODO: Implement SentencePiece model loading
        // This would parse the protobuf format used by SentencePiece
        _ = self;
        _ = file_path;
        return error.NotImplemented;
    }

    /// Create a simple vocabulary for testing
    pub fn createSimpleVocab(allocator: Allocator) !Vocabulary {
        var vocab = try Vocabulary.init(allocator, 1000);

        // Add common English subwords for demonstration
        const test_tokens = [_]struct { piece: []const u8, score: f32 }{
            .{ .piece = "the", .score = -1.0 },
            .{ .piece = "and", .score = -1.5 },
            .{ .piece = "ing", .score = -2.0 },
            .{ .piece = "ed", .score = -2.5 },
            .{ .piece = "er", .score = -3.0 },
            .{ .piece = "ly", .score = -3.5 },
            .{ .piece = "tion", .score = -4.0 },
            .{ .piece = "ment", .score = -4.5 },
            .{ .piece = "ness", .score = -5.0 },
            .{ .piece = "able", .score = -5.5 },
        };

        var token_id: TokenId = 4; // Start after special tokens
        for (test_tokens) |token| {
            try vocab.addToken(token.piece, token.score, token_id, false);
            token_id += 1;
        }

        return vocab;
    }
};

/// Simple tokenizer implementation
/// For production use, this would be replaced with a full SentencePiece implementation
pub const SimpleTokenizer = struct {
    /// Vocabulary for token management
    vocabulary: Vocabulary,
    /// Allocator for memory management
    allocator: Allocator,

    /// Initialize tokenizer with vocabulary
    pub fn init(allocator: Allocator, vocab_size: usize) !SimpleTokenizer {
        return SimpleTokenizer{
            .vocabulary = try Vocabulary.init(allocator, vocab_size),
            .allocator = allocator,
        };
    }

    /// Initialize with pre-built vocabulary
    pub fn initWithVocab(vocabulary: Vocabulary, allocator: Allocator) SimpleTokenizer {
        return SimpleTokenizer{
            .vocabulary = vocabulary,
            .allocator = allocator,
        };
    }

    /// Clean up tokenizer
    pub fn deinit(self: *SimpleTokenizer) void {
        self.vocabulary.deinit();
    }

    /// Tokenize text into token IDs
    /// This is a simplified implementation - production would use BPE/SentencePiece
    pub fn encode(self: SimpleTokenizer, text: []const u8) ![]TokenId {
        var tokens = try std.ArrayList(TokenId).initCapacity(self.allocator, text.len);
        errdefer tokens.deinit(self.allocator);

        // Add beginning of sequence token
        try tokens.append(self.allocator, SpecialTokens.BOS);

        // Simple word-based tokenization for demonstration
        var word_iter = std.mem.splitSequence(u8, text, " ");
        while (word_iter.next()) |word| {
            if (word.len == 0) continue;

            // Try to find exact word match
            if (self.vocabulary.getTokenId(word)) |token_id| {
                try tokens.append(self.allocator, token_id);
            } else {
                // Fallback to character-level or unknown token
                // In production, this would use subword algorithms
                try tokens.append(self.allocator, SpecialTokens.UNK);
            }
        }

        // Add end of sequence token
        try tokens.append(self.allocator, SpecialTokens.EOS);

        return try tokens.toOwnedSlice(self.allocator);
    }

    /// Decode token IDs back to text
    pub fn decode(self: SimpleTokenizer, token_ids: []const TokenId) ![]u8 {
        var result = try std.ArrayList(u8).initCapacity(self.allocator, token_ids.len * 10);
        errdefer result.deinit(self.allocator);

        for (token_ids, 0..) |token_id, i| {
            // Skip special tokens in output (except spaces)
            if (token_id == SpecialTokens.BOS or token_id == SpecialTokens.EOS or token_id == SpecialTokens.PAD) {
                continue;
            }

            if (self.vocabulary.getTokenPiece(token_id)) |piece| {
                if (token_id == SpecialTokens.UNK) {
                    try result.appendSlice(self.allocator, "<unk>");
                } else {
                    // Add space before tokens (except first)
                    if (i > 0 and result.items.len > 0) {
                        try result.append(self.allocator, ' ');
                    }
                    try result.appendSlice(self.allocator, piece.piece);
                }
            }
        }

        return try result.toOwnedSlice(self.allocator);
    }

    /// Get vocabulary size
    pub fn vocabSize(self: SimpleTokenizer) usize {
        return self.vocabulary.size();
    }

    /// Batch encode multiple texts
    pub fn batchEncode(self: SimpleTokenizer, texts: []const []const u8) ![][]TokenId {
        var results = try self.allocator.alloc([]TokenId, texts.len);
        errdefer {
            for (results[0..texts.len]) |tokens| {
                self.allocator.free(tokens);
            }
            self.allocator.free(results);
        }

        for (texts, 0..) |text, i| {
            results[i] = try self.encode(text);
        }

        return results;
    }

    /// Batch decode multiple token sequences
    pub fn batchDecode(self: SimpleTokenizer, token_sequences: []const []const TokenId) ![][]u8 {
        var results = try self.allocator.alloc([]u8, token_sequences.len);
        errdefer {
            for (results[0..token_sequences.len]) |text| {
                self.allocator.free(text);
            }
            self.allocator.free(results);
        }

        for (token_sequences, 0..) |tokens, i| {
            results[i] = try self.decode(tokens);
        }

        return results;
    }

    /// Pad sequences to the same length for batch processing
    pub fn padSequences(self: SimpleTokenizer, sequences: [][]TokenId, max_length: ?usize) !void {
        // Find maximum length if not provided
        var max_len = max_length orelse 0;
        if (max_len == 0) {
            for (sequences) |seq| {
                max_len = @max(max_len, seq.len);
            }
        }

        // Resize and pad each sequence
        for (sequences) |*seq| {
            if (seq.len < max_len) {
                const old_seq = seq.*;
                seq.* = try self.allocator.realloc(old_seq, max_len);
                // Fill with padding tokens
                for (seq.*[old_seq.len..]) |*token| {
                    token.* = SpecialTokens.PAD;
                }
            } else if (seq.len > max_len) {
                // Truncate sequence
                const old_seq = seq.*;
                seq.* = try self.allocator.realloc(old_seq, max_len);
            }
        }
    }
};

/// BPE (Byte Pair Encoding) Tokenizer
///
/// ## Educational Note: Subword Tokenization
/// BPE is the standard tokenization for LLaMA models. It works by:
/// 1. Starting with individual characters/bytes
/// 2. Iteratively merging the most frequent adjacent pairs
/// 3. Building a vocabulary of common subword units
///
/// This gives a balance between character-level (no OOV) and word-level (semantic) tokens.
pub const BPETokenizer = struct {
    /// Vocabulary (reuses existing Vocabulary struct)
    vocabulary: Vocabulary,
    /// Memory allocator
    allocator: Allocator,
    /// Merge pairs ordered by rank (lower rank = higher priority)
    merges: ArrayList(MergePair),
    /// Fast lookup: "left right" → merge rank
    merge_ranks: HashMap([]const u8, u32, StringContext, std.hash_map.default_max_load_percentage),

    const StringContext = struct {
        pub fn hash(self: @This(), s: []const u8) u64 {
            _ = self;
            return std.hash_map.hashString(s);
        }

        pub fn eql(self: @This(), a: []const u8, b: []const u8) bool {
            _ = self;
            return std.mem.eql(u8, a, b);
        }
    };

    pub const MergePair = struct {
        left: []const u8,
        right: []const u8,
        rank: u32,
    };

    /// Initialize BPE tokenizer
    pub fn init(allocator: Allocator, vocab_size: usize) !BPETokenizer {
        return BPETokenizer{
            .vocabulary = try Vocabulary.init(allocator, vocab_size),
            .allocator = allocator,
            .merges = ArrayList(MergePair).init(allocator),
            .merge_ranks = HashMap([]const u8, u32, StringContext, std.hash_map.default_max_load_percentage).init(allocator),
        };
    }

    /// Clean up resources
    pub fn deinit(self: *BPETokenizer) void {
        // Free merge rank keys (owned strings)
        var rank_iter = self.merge_ranks.iterator();
        while (rank_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.merge_ranks.deinit();

        // Free merge pair strings
        for (self.merges.items) |merge| {
            self.allocator.free(merge.left);
            self.allocator.free(merge.right);
        }
        self.merges.deinit();

        self.vocabulary.deinit();
    }

    /// Add a merge rule. Rank determines priority (lower = merged first).
    pub fn addMerge(self: *BPETokenizer, left: []const u8, right: []const u8, rank: u32) !void {
        const owned_left = try self.allocator.dupe(u8, left);
        errdefer self.allocator.free(owned_left);
        const owned_right = try self.allocator.dupe(u8, right);
        errdefer self.allocator.free(owned_right);

        try self.merges.append(MergePair{
            .left = owned_left,
            .right = owned_right,
            .rank = rank,
        });

        // Build key "left right" for fast lookup
        const key = try std.fmt.allocPrint(self.allocator, "{s} {s}", .{ left, right });
        errdefer self.allocator.free(key);
        try self.merge_ranks.put(key, rank);
    }

    /// Encode text to token IDs using BPE algorithm
    ///
    /// ## Algorithm
    /// 1. Replace spaces with SentencePiece marker (▁ = U+2581)
    /// 2. Split into UTF-8 characters as initial symbols
    /// 3. Repeatedly merge the pair with lowest rank
    /// 4. Look up each resulting symbol in vocabulary
    /// 5. Wrap with BOS and EOS tokens
    pub fn encode(self: *const BPETokenizer, text: []const u8) ![]TokenId {
        var tokens = ArrayList(TokenId).init(self.allocator);
        errdefer tokens.deinit();

        // Add BOS
        try tokens.append(SpecialTokens.BOS);

        if (text.len == 0) {
            try tokens.append(SpecialTokens.EOS);
            return try tokens.toOwnedSlice();
        }

        // Step 1: Prepare text — replace spaces with ▁ (SentencePiece style)
        var prepared = ArrayList(u8).init(self.allocator);
        defer prepared.deinit();

        // Prepend ▁ for SentencePiece convention
        try prepared.appendSlice("\xe2\x96\x81"); // ▁ in UTF-8

        for (text) |c| {
            if (c == ' ') {
                try prepared.appendSlice("\xe2\x96\x81"); // ▁
            } else {
                try prepared.append(c);
            }
        }

        // Step 2: Split into initial UTF-8 characters
        var symbols = ArrayList([]const u8).init(self.allocator);
        defer {
            for (symbols.items) |sym| {
                self.allocator.free(sym);
            }
            symbols.deinit();
        }

        var pos: usize = 0;
        while (pos < prepared.items.len) {
            const byte = prepared.items[pos];
            const char_len: usize = if (byte < 0x80) 1 else if (byte < 0xE0) 2 else if (byte < 0xF0) 3 else 4;
            const end = @min(pos + char_len, prepared.items.len);
            const sym = try self.allocator.dupe(u8, prepared.items[pos..end]);
            try symbols.append(sym);
            pos = end;
        }

        // Step 3: BPE merge loop — repeatedly merge lowest-rank pair
        while (symbols.items.len > 1) {
            // Find the pair with lowest merge rank
            var best_rank: u32 = std.math.maxInt(u32);
            var best_idx: ?usize = null;

            for (0..symbols.items.len - 1) |i| {
                const key = try std.fmt.allocPrint(self.allocator, "{s} {s}", .{ symbols.items[i], symbols.items[i + 1] });
                defer self.allocator.free(key);

                if (self.merge_ranks.get(key)) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }

            // No more merges possible
            if (best_idx == null) break;

            const idx = best_idx.?;

            // Merge: concatenate symbols[idx] and symbols[idx+1]
            const merged = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ symbols.items[idx], symbols.items[idx + 1] });

            // Free old symbols
            self.allocator.free(symbols.items[idx]);
            self.allocator.free(symbols.items[idx + 1]);

            // Replace idx with merged, remove idx+1
            symbols.items[idx] = merged;
            _ = symbols.orderedRemove(idx + 1);
        }

        // Step 4: Look up each symbol in vocabulary
        for (symbols.items) |sym| {
            if (self.vocabulary.getTokenId(sym)) |id| {
                try tokens.append(id);
            } else {
                try tokens.append(SpecialTokens.UNK);
            }
        }

        // Add EOS
        try tokens.append(SpecialTokens.EOS);

        return try tokens.toOwnedSlice();
    }

    /// Decode token IDs back to text
    ///
    /// Converts token IDs → pieces via vocabulary lookup, concatenates,
    /// and replaces ▁ with spaces.
    pub fn decode(self: *const BPETokenizer, token_ids: []const TokenId) ![]u8 {
        var result = ArrayList(u8).init(self.allocator);
        errdefer result.deinit();

        for (token_ids) |id| {
            // Skip BOS and EOS
            if (id == SpecialTokens.BOS or id == SpecialTokens.EOS or id == SpecialTokens.PAD) {
                continue;
            }

            if (id == SpecialTokens.UNK) {
                try result.appendSlice("<unk>");
                continue;
            }

            if (self.vocabulary.getTokenPiece(id)) |piece| {
                try result.appendSlice(piece.piece);
            }
        }

        // Replace ▁ (U+2581, UTF-8: E2 96 81) with space, strip leading space
        var final = ArrayList(u8).init(self.allocator);
        errdefer final.deinit();

        var i: usize = 0;
        while (i < result.items.len) {
            if (i + 2 < result.items.len and
                result.items[i] == 0xE2 and
                result.items[i + 1] == 0x96 and
                result.items[i + 2] == 0x81)
            {
                // Replace ▁ with space (skip leading)
                if (final.items.len > 0) {
                    try final.append(' ');
                }
                i += 3;
            } else {
                try final.append(result.items[i]);
                i += 1;
            }
        }

        result.deinit();
        return try final.toOwnedSlice();
    }

    /// Load vocabulary and merges from GGUF metadata arrays
    pub fn loadFromGGUF(
        self: *BPETokenizer,
        token_pieces: []const []const u8,
        token_scores: ?[]const f32,
        merge_rules: ?[]const []const u8,
    ) !void {
        // Add tokens to vocabulary
        for (token_pieces, 0..) |piece, i| {
            const id: TokenId = @intCast(i);
            const score: f32 = if (token_scores) |scores| (if (i < scores.len) scores[i] else 0.0) else 0.0;
            const is_special = id <= 3;

            // Skip if already added (special tokens)
            if (self.vocabulary.getTokenId(piece) != null) continue;

            try self.vocabulary.addToken(piece, score, id, is_special);
        }

        // Add merge rules
        if (merge_rules) |merges| {
            for (merges, 0..) |rule, rank| {
                // Merge rule format: "left right" (space-separated)
                if (std.mem.indexOf(u8, rule, " ")) |space_pos| {
                    const left = rule[0..space_pos];
                    const right = rule[space_pos + 1 ..];
                    try self.addMerge(left, right, @intCast(rank));
                }
            }
        }
    }
};

/// Tokenizer statistics for analysis and debugging
pub const TokenizerStats = struct {
    vocab_size: usize,
    avg_tokens_per_text: f32,
    unknown_token_rate: f32,
    special_token_count: usize,

    /// Analyze tokenizer performance on a dataset
    pub fn analyze(tokenizer: SimpleTokenizer, texts: []const []const u8, allocator: Allocator) !TokenizerStats {
        var total_tokens: usize = 0;
        var unknown_tokens: usize = 0;
        var special_tokens: usize = 0;

        for (texts) |text| {
            const tokens = try tokenizer.encode(text);
            defer allocator.free(tokens);

            total_tokens += tokens.len;

            for (tokens) |token_id| {
                if (token_id == SpecialTokens.UNK) {
                    unknown_tokens += 1;
                }
                if (SpecialTokens.isSpecial(token_id)) {
                    special_tokens += 1;
                }
            }
        }

        return TokenizerStats{
            .vocab_size = tokenizer.vocabSize(),
            .avg_tokens_per_text = @as(f32, @floatFromInt(total_tokens)) / @as(f32, @floatFromInt(texts.len)),
            .unknown_token_rate = @as(f32, @floatFromInt(unknown_tokens)) / @as(f32, @floatFromInt(total_tokens)),
            .special_token_count = special_tokens,
        };
    }

    /// Print tokenizer statistics
    pub fn print(self: TokenizerStats, writer: anytype) !void {
        try writer.print("Tokenizer Statistics:\n", .{});
        try writer.print("  Vocabulary size: {d}\n", .{self.vocab_size});
        try writer.print("  Average tokens per text: {d:.1}\n", .{self.avg_tokens_per_text});
        try writer.print("  Unknown token rate: {d:.1}%\n", .{self.unknown_token_rate * 100.0});
        try writer.print("  Special tokens used: {d}\n", .{self.special_token_count});
    }
};

// Comprehensive tests for tokenization system
test "vocabulary management" {
    const testing = std.testing;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var vocab = try Vocabulary.init(allocator, 100);
    defer vocab.deinit();

    // Test special tokens are present
    try testing.expect(vocab.getTokenId("<unk>") != null);
    try testing.expect(vocab.getTokenId("<s>") != null);
    try testing.expect(vocab.getTokenId("</s>") != null);
    try testing.expect(vocab.getTokenId("<pad>") != null);

    // Test adding new tokens
    try vocab.addToken("hello", -1.0, 4, false);
    try testing.expect(vocab.getTokenId("hello") != null);
    try testing.expectEqual(@as(TokenId, 4), vocab.getTokenId("hello").?);

    // Test vocabulary contains
    try testing.expect(vocab.contains("hello"));
    try testing.expect(!vocab.contains("world"));
}

test "simple tokenization" {
    const testing = std.testing;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var tokenizer = try SimpleTokenizer.init(allocator, 100);
    defer tokenizer.deinit();

    // Add test tokens
    try tokenizer.vocabulary.addToken("hello", -1.0, 4, false);
    try tokenizer.vocabulary.addToken("world", -1.5, 5, false);

    // Test encoding
    const tokens = try tokenizer.encode("hello world");
    defer allocator.free(tokens);

    try testing.expect(tokens.len >= 3); // BOS + tokens + EOS
    try testing.expectEqual(SpecialTokens.BOS, tokens[0]);
    try testing.expectEqual(SpecialTokens.EOS, tokens[tokens.len - 1]);

    // Test decoding
    const decoded = try tokenizer.decode(tokens);
    defer allocator.free(decoded);

    try testing.expect(std.mem.indexOf(u8, decoded, "hello") != null);
    try testing.expect(std.mem.indexOf(u8, decoded, "world") != null);
}

test "batch processing" {
    const testing = std.testing;
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

    try testing.expectEqual(@as(usize, 3), encoded.len);
    for (encoded) |tokens| {
        try testing.expect(tokens.len >= 2); // At least BOS + EOS
        try testing.expectEqual(SpecialTokens.BOS, tokens[0]);
        try testing.expectEqual(SpecialTokens.EOS, tokens[tokens.len - 1]);
    }
}

test "special token handling" {
    const testing = std.testing;

    try testing.expect(SpecialTokens.isSpecial(SpecialTokens.UNK));
    try testing.expect(SpecialTokens.isSpecial(SpecialTokens.BOS));
    try testing.expect(SpecialTokens.isSpecial(SpecialTokens.EOS));
    try testing.expect(SpecialTokens.isSpecial(SpecialTokens.PAD));
    try testing.expect(!SpecialTokens.isSpecial(100));

    try testing.expectEqualStrings("<unk>", SpecialTokens.name(SpecialTokens.UNK).?);
    try testing.expectEqualStrings("<s>", SpecialTokens.name(SpecialTokens.BOS).?);
    try testing.expectEqualStrings("</s>", SpecialTokens.name(SpecialTokens.EOS).?);
    try testing.expect(SpecialTokens.name(100) == null);
}

test "tokenizer statistics" {
    const testing = std.testing;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const vocab = try Vocabulary.createSimpleVocab(allocator);
    var tokenizer = SimpleTokenizer.initWithVocab(vocab, allocator);
    defer tokenizer.deinit();

    const test_texts = [_][]const u8{ "the quick brown", "and the fox" };
    const stats = try TokenizerStats.analyze(tokenizer, &test_texts, allocator);

    try testing.expect(stats.vocab_size > 0);
    try testing.expect(stats.avg_tokens_per_text > 0);
    try testing.expect(stats.unknown_token_rate >= 0.0);
    try testing.expect(stats.special_token_count > 0);
}

test "BPE tokenizer encode/decode round-trip" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var bpe = try BPETokenizer.init(allocator, 100);
    defer bpe.deinit();

    // Build vocabulary with all intermediate merge results
    // ▁ = U+2581 = 0xE2 0x96 0x81 in UTF-8 (3 bytes: \xe2, \x96, \x81)
    // Initial UTF-8 chars from "hello" → ▁, h, e, l, l, o
    // Note: ▁ is a single UTF-8 char (3 bytes) — split correctly

    // Individual characters (initial symbols)
    try bpe.vocabulary.addToken("\xe2\x96\x81", -10.0, 4, false); // ▁
    try bpe.vocabulary.addToken("h", -10.0, 5, false);
    try bpe.vocabulary.addToken("e", -10.0, 6, false);
    try bpe.vocabulary.addToken("l", -10.0, 7, false);
    try bpe.vocabulary.addToken("o", -10.0, 8, false);

    // Merge results
    try bpe.vocabulary.addToken("\xe2\x96\x81h", -5.0, 9, false);    // ▁ + h
    try bpe.vocabulary.addToken("ll", -5.0, 10, false);              // l + l
    try bpe.vocabulary.addToken("llo", -4.0, 11, false);             // ll + o
    try bpe.vocabulary.addToken("\xe2\x96\x81he", -3.0, 12, false);  // ▁h + e
    try bpe.vocabulary.addToken("\xe2\x96\x81hello", -1.0, 13, false); // ▁he + llo

    // Add merges in priority order (lower rank = merge first)
    try bpe.addMerge("\xe2\x96\x81", "h", 0); // ▁ + h → ▁h
    try bpe.addMerge("l", "l", 1);             // l + l → ll
    try bpe.addMerge("ll", "o", 2);            // ll + o → llo
    try bpe.addMerge("\xe2\x96\x81h", "e", 3); // ▁h + e → ▁he
    try bpe.addMerge("\xe2\x96\x81he", "llo", 4); // ▁he + llo → ▁hello

    const tokens = try bpe.encode("hello");
    defer allocator.free(tokens);

    // Should have BOS + ▁hello + EOS
    try testing.expectEqual(@as(usize, 3), tokens.len);
    try testing.expectEqual(SpecialTokens.BOS, tokens[0]);
    try testing.expectEqual(@as(TokenId, 13), tokens[1]); // ▁hello
    try testing.expectEqual(SpecialTokens.EOS, tokens[2]);

    // Decode back
    const decoded = try bpe.decode(tokens);
    defer allocator.free(decoded);

    try testing.expectEqualStrings("hello", decoded);
}

test "BPE tokenizer BOS/EOS handling" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var bpe = try BPETokenizer.init(allocator, 100);
    defer bpe.deinit();

    // Empty text should still have BOS + EOS
    const tokens = try bpe.encode("");
    defer allocator.free(tokens);

    try testing.expectEqual(@as(usize, 2), tokens.len);
    try testing.expectEqual(SpecialTokens.BOS, tokens[0]);
    try testing.expectEqual(SpecialTokens.EOS, tokens[1]);
}

test "BPE tokenizer unknown token fallback" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var bpe = try BPETokenizer.init(allocator, 100);
    defer bpe.deinit();

    // No vocabulary or merges added — everything should be UNK
    const tokens = try bpe.encode("hi");
    defer allocator.free(tokens);

    // BOS + UNK tokens + EOS
    try testing.expectEqual(SpecialTokens.BOS, tokens[0]);
    try testing.expectEqual(SpecialTokens.EOS, tokens[tokens.len - 1]);

    // All non-BOS/EOS tokens should be UNK (since no vocab entries)
    for (tokens[1 .. tokens.len - 1]) |tok| {
        try testing.expectEqual(SpecialTokens.UNK, tok);
    }
}

test "BPE tokenizer space handling" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var bpe = try BPETokenizer.init(allocator, 100);
    defer bpe.deinit();

    // Individual characters
    try bpe.vocabulary.addToken("\xe2\x96\x81", -10.0, 4, false); // ▁
    try bpe.vocabulary.addToken("a", -10.0, 5, false);
    try bpe.vocabulary.addToken("b", -10.0, 6, false);

    // Merged tokens
    try bpe.vocabulary.addToken("\xe2\x96\x81a", -1.0, 7, false);
    try bpe.vocabulary.addToken("\xe2\x96\x81b", -1.0, 8, false);

    // Merges: ▁ + a → ▁a, ▁ + b → ▁b
    try bpe.addMerge("\xe2\x96\x81", "a", 0);
    try bpe.addMerge("\xe2\x96\x81", "b", 1);

    const tokens = try bpe.encode("a b");
    defer allocator.free(tokens);

    // Should have BOS + ▁a + ▁b + EOS
    try testing.expectEqual(@as(usize, 4), tokens.len);
    try testing.expectEqual(SpecialTokens.BOS, tokens[0]);
    try testing.expectEqual(@as(TokenId, 7), tokens[1]); // ▁a
    try testing.expectEqual(@as(TokenId, 8), tokens[2]); // ▁b
    try testing.expectEqual(SpecialTokens.EOS, tokens[3]);

    // Decode should restore "a b"
    const decoded = try bpe.decode(tokens);
    defer allocator.free(decoded);

    try testing.expectEqualStrings("a b", decoded);
}

test "BPE tokenizer loadFromGGUF" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var bpe = try BPETokenizer.init(allocator, 100);
    defer bpe.deinit();

    const pieces = [_][]const u8{ "<unk>", "<s>", "</s>", "<pad>", "he", "ll", "o" };
    const scores = [_]f32{ 0, 0, 0, 0, -1.0, -1.5, -2.0 };
    const merges = [_][]const u8{"h e"};

    try bpe.loadFromGGUF(&pieces, &scores, &merges);

    // Check vocabulary loaded
    try testing.expect(bpe.vocabulary.getTokenId("he") != null);
    try testing.expect(bpe.vocabulary.getTokenId("ll") != null);
    try testing.expect(bpe.vocabulary.getTokenId("o") != null);

    // Check merge loaded
    try testing.expectEqual(@as(usize, 1), bpe.merges.items.len);
}