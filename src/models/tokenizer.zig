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
            .id_to_piece = ArrayList(TokenPiece).init(allocator),
            .vocab_size = vocab_size,
        };

        // Reserve space for efficiency
        try vocab.id_to_piece.ensureTotalCapacity(vocab_size);

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
        self.id_to_piece.deinit();
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
            try self.id_to_piece.resize(id + 1);
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
        var tokens = ArrayList(TokenId).init(self.allocator);
        errdefer tokens.deinit();

        // Add beginning of sequence token
        try tokens.append(SpecialTokens.BOS);

        // Simple word-based tokenization for demonstration
        var word_iter = std.mem.split(u8, text, " ");
        while (word_iter.next()) |word| {
            if (word.len == 0) continue;

            // Try to find exact word match
            if (self.vocabulary.getTokenId(word)) |token_id| {
                try tokens.append(token_id);
            } else {
                // Fallback to character-level or unknown token
                // In production, this would use subword algorithms
                try tokens.append(SpecialTokens.UNK);
            }
        }

        // Add end of sequence token
        try tokens.append(SpecialTokens.EOS);

        return try tokens.toOwnedSlice();
    }

    /// Decode token IDs back to text
    pub fn decode(self: SimpleTokenizer, token_ids: []const TokenId) ![]u8 {
        var result = ArrayList(u8).init(self.allocator);
        errdefer result.deinit();

        for (token_ids, 0..) |token_id, i| {
            // Skip special tokens in output (except spaces)
            if (token_id == SpecialTokens.BOS or token_id == SpecialTokens.EOS or token_id == SpecialTokens.PAD) {
                continue;
            }

            if (self.vocabulary.getTokenPiece(token_id)) |piece| {
                if (token_id == SpecialTokens.UNK) {
                    try result.appendSlice("<unk>");
                } else {
                    // Add space before tokens (except first)
                    if (i > 0 and result.items.len > 0) {
                        try result.append(' ');
                    }
                    try result.appendSlice(piece.piece);
                }
            }
        }

        return try result.toOwnedSlice();
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

    var vocab = try Vocabulary.createSimpleVocab(allocator);
    var tokenizer = SimpleTokenizer.initWithVocab(vocab, allocator);
    defer tokenizer.deinit();

    const test_texts = [_][]const u8{ "the quick brown", "and the fox" };
    const stats = try TokenizerStats.analyze(tokenizer, &test_texts, allocator);

    try testing.expect(stats.vocab_size > 0);
    try testing.expect(stats.avg_tokens_per_text > 0);
    try testing.expect(stats.unknown_token_rate >= 0.0);
    try testing.expect(stats.special_token_count > 0);
}