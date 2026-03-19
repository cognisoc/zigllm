const std = @import("std");
const Allocator = std.mem.Allocator;
const foundation = @import("../foundation/tensor.zig");
const Tensor = foundation.Tensor;

/// Grammar constraint types supported
pub const GrammarType = enum {
    JSON,           // JSON schema constraints
    RegexPattern,   // Regular expression constraints
    ContextFree,    // Context-free grammar constraints
    XMLSchema,      // XML schema constraints
    EBNF,          // Extended Backus-Naur Form
};

/// JSON schema constraint specification
pub const JSONConstraint = struct {
    schema: []const u8,        // JSON schema specification
    require_valid: bool,       // Require valid JSON structure
    allow_partial: bool,       // Allow partial JSON during generation

    /// Common JSON schema patterns
    pub const OBJECT = "{\"type\":\"object\"}";
    pub const ARRAY = "{\"type\":\"array\"}";
    pub const STRING = "{\"type\":\"string\"}";
    pub const NUMBER = "{\"type\":\"number\"}";
    pub const BOOLEAN = "{\"type\":\"boolean\"}";

    /// Create constraint for structured data
    pub fn createStructured(allocator: Allocator, fields: []const []const u8) !JSONConstraint {
        var schema = std.ArrayList(u8).init(allocator);
        defer schema.deinit();

        try schema.appendSlice("{\"type\":\"object\",\"properties\":{");

        for (fields, 0..) |field, i| {
            if (i > 0) try schema.appendSlice(",");
            try schema.appendSlice("\"");
            try schema.appendSlice(field);
            try schema.appendSlice("\":{\"type\":\"string\"}");
        }

        try schema.appendSlice("}}");

        const schema_owned = try allocator.dupe(u8, schema.items);

        return JSONConstraint{
            .schema = schema_owned,
            .require_valid = true,
            .allow_partial = true,
        };
    }
};

/// Regular expression constraint
pub const RegexConstraint = struct {
    pattern: []const u8,       // Regex pattern
    flags: RegexFlags,         // Regex flags
    max_length: ?usize,        // Maximum match length

    pub const RegexFlags = struct {
        case_insensitive: bool = false,
        multiline: bool = false,
        dot_all: bool = false,
    };

    /// Common patterns
    pub const EMAIL = "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$";
    pub const PHONE = "^\\+?[1-9]\\d{1,14}$";
    pub const UUID = "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$";
    pub const URL = "^https?://[^\\s/$.?#].[^\\s]*$";
    pub const IPV4 = "^(?:[0-9]{1,3}\\.){3}[0-9]{1,3}$";
    pub const DATE_ISO = "^\\d{4}-\\d{2}-\\d{2}$";
    pub const TIME_24H = "^([01]?[0-9]|2[0-3]):[0-5][0-9]$";
};

/// Context-free grammar constraint
pub const CFGConstraint = struct {
    rules: []const GrammarRule, // Grammar production rules
    start_symbol: []const u8,   // Starting non-terminal
    terminals: []const []const u8, // Terminal symbols

    pub const GrammarRule = struct {
        left: []const u8,       // Left-hand side (non-terminal)
        right: []const []const u8, // Right-hand side alternatives
    };

    /// Create simple grammar for structured text
    pub fn createSimple(allocator: Allocator) !CFGConstraint {
        const rules = try allocator.alloc(GrammarRule, 3);

        // S -> "Hello" Name
        // Name -> "Alice" | "Bob" | "Charlie"
        // Simple greeting grammar

        const hello_parts = try allocator.alloc([]const u8, 2);
        hello_parts[0] = "\"Hello\"";
        hello_parts[1] = "Name";

        const name_options = try allocator.alloc([]const u8, 3);
        name_options[0] = "\"Alice\"";
        name_options[1] = "\"Bob\"";
        name_options[2] = "\"Charlie\"";

        rules[0] = GrammarRule{ .left = "S", .right = &[_][]const u8{hello_parts[0], hello_parts[1]} };
        rules[1] = GrammarRule{ .left = "Name", .right = name_options };

        const terminals = try allocator.alloc([]const u8, 4);
        terminals[0] = "\"Hello\"";
        terminals[1] = "\"Alice\"";
        terminals[2] = "\"Bob\"";
        terminals[3] = "\"Charlie\"";

        return CFGConstraint{
            .rules = rules,
            .start_symbol = "S",
            .terminals = terminals,
        };
    }
};

/// Grammar constraint engine state
pub const GrammarState = struct {
    constraint_type: GrammarType,
    current_position: usize,    // Current position in generation
    stack: std.ArrayList([]const u8), // Parser stack for CFG
    context: std.StringHashMap(bool), // Context information
    partial_match: []u8,        // Current partial match buffer
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, constraint_type: GrammarType) Self {
        return Self{
            .constraint_type = constraint_type,
            .current_position = 0,
            .stack = std.ArrayList([]const u8).init(allocator),
            .context = std.StringHashMap(bool).init(allocator),
            .partial_match = &[_]u8{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.stack.deinit();
        self.context.deinit();
        if (self.partial_match.len > 0) {
            self.allocator.free(self.partial_match);
        }
    }

    /// Update state with new token
    pub fn updateWithToken(self: *Self, token: []const u8) !void {
        // Extend partial match
        const new_size = self.partial_match.len + token.len;
        const new_match = try self.allocator.realloc(self.partial_match, new_size);
        @memcpy(new_match[self.partial_match.len..], token);
        self.partial_match = new_match;
        self.current_position += token.len;
    }

    /// Check if current state allows continuation
    pub fn canContinue(self: *Self) bool {
        return switch (self.constraint_type) {
            .JSON => self.isValidJSONState(),
            .RegexPattern => self.isValidRegexState(),
            .ContextFree => self.isValidCFGState(),
            .XMLSchema => self.isValidXMLState(),
            .EBNF => self.isValidEBNFState(),
        };
    }

    fn isValidJSONState(self: *Self) bool {
        // Simplified JSON validation
        var brace_count: i32 = 0;
        var bracket_count: i32 = 0;
        var in_string: bool = false;
        var escaped: bool = false;

        for (self.partial_match) |char| {
            if (escaped) {
                escaped = false;
                continue;
            }

            switch (char) {
                '\\' => if (in_string) escaped = true,
                '"' => in_string = !in_string,
                '{' => if (!in_string) brace_count += 1,
                '}' => if (!in_string) brace_count -= 1,
                '[' => if (!in_string) bracket_count += 1,
                ']' => if (!in_string) bracket_count -= 1,
                else => {},
            }

            // Invalid if we have negative counts
            if (brace_count < 0 or bracket_count < 0) return false;
        }

        return true; // Valid partial JSON
    }

    fn isValidRegexState(self: *Self) bool {
        // This would require a full regex engine implementation
        // Simplified: just check if it could be a valid prefix
        return self.partial_match.len < 1000; // Basic length check
    }

    fn isValidCFGState(self: *Self) bool {
        // Simplified CFG validation using stack
        // In practice, this would implement a proper parser
        return self.stack.items.len < 100; // Prevent infinite recursion
    }

    fn isValidXMLState(self: *Self) bool {
        // Simplified XML validation
        var tag_stack = std.ArrayList([]const u8).init(self.allocator);
        defer tag_stack.deinit();

        // Basic XML tag matching (simplified)
        var i: usize = 0;
        while (i < self.partial_match.len) {
            if (self.partial_match[i] == '<') {
                // Found tag start, basic validation
                var j = i + 1;
                while (j < self.partial_match.len and self.partial_match[j] != '>') {
                    j += 1;
                }
                if (j >= self.partial_match.len) return true; // Incomplete tag is okay

                // Extract tag name (simplified)
                const tag_content = self.partial_match[i + 1..j];
                if (tag_content.len > 0 and tag_content[0] != '/') {
                    // Opening tag (simplified validation)
                    return true;
                }
            }
            i += 1;
        }

        return true;
    }

    fn isValidEBNFState(self: *Self) bool {
        // EBNF validation would be complex
        // Simplified: basic structure checking
        return self.partial_match.len < 2000;
    }
};

/// Grammar-constrained sampler
pub const GrammarConstrainedSampler = struct {
    allocator: Allocator,
    rng: std.rand.DefaultPrng,

    const Self = @This();

    pub fn init(allocator: Allocator, seed: ?u64) Self {
        const actual_seed = seed orelse @as(u64, @intCast(std.time.timestamp()));
        return Self{
            .allocator = allocator,
            .rng = std.rand.DefaultPrng.init(actual_seed),
        };
    }

    /// Sample tokens that satisfy grammar constraints
    pub fn sampleConstrained(self: *Self, logits: Tensor(f32),
                           grammar_state: *GrammarState,
                           tokenizer: anytype) !u32 {
        // Convert logits to probabilities
        const probs = try self.softmax(logits);
        defer probs.deinit(self.allocator);

        // Get valid token candidates
        const valid_tokens = try self.getValidTokens(probs, grammar_state, tokenizer);
        defer self.allocator.free(valid_tokens);

        if (valid_tokens.len == 0) {
            // No valid tokens - return most likely token as fallback
            return self.sampleMostLikely(probs);
        }

        // Sample from valid tokens only
        return self.sampleFromValidTokens(probs, valid_tokens);
    }

    /// JSON-constrained sampling
    pub fn sampleJSONConstrained(self: *Self, logits: Tensor(f32),
                                constraint: JSONConstraint,
                                grammar_state: *GrammarState,
                                tokenizer: anytype) !u32 {
        const probs = try self.softmax(logits);
        defer probs.deinit(self.allocator);

        const valid_tokens = try self.getJSONValidTokens(probs, constraint, grammar_state, tokenizer);
        defer self.allocator.free(valid_tokens);

        return self.sampleFromValidTokens(probs, valid_tokens);
    }

    /// Regex-constrained sampling
    pub fn sampleRegexConstrained(self: *Self, logits: Tensor(f32),
                                 constraint: RegexConstraint,
                                 grammar_state: *GrammarState,
                                 tokenizer: anytype) !u32 {
        const probs = try self.softmax(logits);
        defer probs.deinit(self.allocator);

        const valid_tokens = try self.getRegexValidTokens(probs, constraint, grammar_state, tokenizer);
        defer self.allocator.free(valid_tokens);

        return self.sampleFromValidTokens(probs, valid_tokens);
    }

    /// Context-free grammar constrained sampling
    pub fn sampleCFGConstrained(self: *Self, logits: Tensor(f32),
                               constraint: CFGConstraint,
                               grammar_state: *GrammarState,
                               tokenizer: anytype) !u32 {
        const probs = try self.softmax(logits);
        defer probs.deinit(self.allocator);

        const valid_tokens = try self.getCFGValidTokens(probs, constraint, grammar_state, tokenizer);
        defer self.allocator.free(valid_tokens);

        return self.sampleFromValidTokens(probs, valid_tokens);
    }

    // Helper functions

    fn softmax(self: *Self, logits: Tensor(f32)) !Tensor(f32) {
        const result_data = try self.allocator.alloc(f32, logits.data.len);

        // Find maximum for numerical stability
        var max_val = logits.data[0];
        for (logits.data[1..]) |val| {
            max_val = @max(max_val, val);
        }

        // Compute exp(x - max) and sum
        var sum: f32 = 0.0;
        for (logits.data, 0..) |val, i| {
            result_data[i] = std.math.exp(val - max_val);
            sum += result_data[i];
        }

        // Normalize
        for (result_data) |*val| {
            val.* /= sum;
        }

        return Tensor(f32){ .data = result_data, .shape = logits.shape };
    }

    fn getValidTokens(self: *Self, probs: Tensor(f32), grammar_state: *GrammarState,
                     tokenizer: anytype) ![]u32 {
        var valid_tokens = std.ArrayList(u32).init(self.allocator);

        for (0..probs.data.len) |i| {
            const token_id = @as(u32, @intCast(i));

            // Get token text (would need actual tokenizer interface)
            const token_text = try self.getTokenText(token_id, tokenizer);
            defer self.allocator.free(token_text);

            // Test if token maintains grammar validity
            if (try self.wouldTokenBeValid(token_text, grammar_state)) {
                try valid_tokens.append(token_id);
            }
        }

        return valid_tokens.toOwnedSlice();
    }

    fn getJSONValidTokens(self: *Self, probs: Tensor(f32), constraint: JSONConstraint,
                         grammar_state: *GrammarState, tokenizer: anytype) ![]u32 {
        var valid_tokens = std.ArrayList(u32).init(self.allocator);

        for (0..probs.data.len) |i| {
            const token_id = @as(u32, @intCast(i));
            const token_text = try self.getTokenText(token_id, tokenizer);
            defer self.allocator.free(token_text);

            if (try self.wouldTokenBeValidJSON(token_text, constraint, grammar_state)) {
                try valid_tokens.append(token_id);
            }
        }

        return valid_tokens.toOwnedSlice();
    }

    fn getRegexValidTokens(self: *Self, probs: Tensor(f32), constraint: RegexConstraint,
                          grammar_state: *GrammarState, tokenizer: anytype) ![]u32 {
        var valid_tokens = std.ArrayList(u32).init(self.allocator);

        for (0..probs.data.len) |i| {
            const token_id = @as(u32, @intCast(i));
            const token_text = try self.getTokenText(token_id, tokenizer);
            defer self.allocator.free(token_text);

            if (try self.wouldTokenBeValidRegex(token_text, constraint, grammar_state)) {
                try valid_tokens.append(token_id);
            }
        }

        return valid_tokens.toOwnedSlice();
    }

    fn getCFGValidTokens(self: *Self, probs: Tensor(f32), constraint: CFGConstraint,
                        grammar_state: *GrammarState, tokenizer: anytype) ![]u32 {
        var valid_tokens = std.ArrayList(u32).init(self.allocator);

        for (0..probs.data.len) |i| {
            const token_id = @as(u32, @intCast(i));
            const token_text = try self.getTokenText(token_id, tokenizer);
            defer self.allocator.free(token_text);

            if (try self.wouldTokenBeValidCFG(token_text, constraint, grammar_state)) {
                try valid_tokens.append(token_id);
            }
        }

        return valid_tokens.toOwnedSlice();
    }

    fn getTokenText(self: *Self, token_id: u32, tokenizer: anytype) ![]u8 {
        // Placeholder - would need actual tokenizer interface
        _ = tokenizer;
        const text = try std.fmt.allocPrint(self.allocator, "token_{d}", .{token_id});
        return text;
    }

    fn wouldTokenBeValid(self: *Self, token_text: []const u8, grammar_state: *GrammarState) !bool {
        // Create temporary state to test validity
        var temp_state = GrammarState.init(self.allocator, grammar_state.constraint_type);
        defer temp_state.deinit();

        // Copy current state
        temp_state.current_position = grammar_state.current_position;
        temp_state.partial_match = try self.allocator.dupe(u8, grammar_state.partial_match);

        // Test token addition
        try temp_state.updateWithToken(token_text);
        return temp_state.canContinue();
    }

    fn wouldTokenBeValidJSON(self: *Self, token_text: []const u8, constraint: JSONConstraint,
                           grammar_state: *GrammarState) !bool {
        _ = constraint;
        return try self.wouldTokenBeValid(token_text, grammar_state);
    }

    fn wouldTokenBeValidRegex(self: *Self, token_text: []const u8, constraint: RegexConstraint,
                             grammar_state: *GrammarState) !bool {
        _ = constraint;
        return try self.wouldTokenBeValid(token_text, grammar_state);
    }

    fn wouldTokenBeValidCFG(self: *Self, token_text: []const u8, constraint: CFGConstraint,
                           grammar_state: *GrammarState) !bool {
        _ = constraint;
        return try self.wouldTokenBeValid(token_text, grammar_state);
    }

    fn sampleMostLikely(self: *Self, probs: Tensor(f32)) u32 {
        var max_prob: f32 = 0.0;
        var max_idx: u32 = 0;

        for (probs.data, 0..) |prob, i| {
            if (prob > max_prob) {
                max_prob = prob;
                max_idx = @as(u32, @intCast(i));
            }
        }

        return max_idx;
    }

    fn sampleFromValidTokens(self: *Self, probs: Tensor(f32), valid_tokens: []const u32) u32 {
        if (valid_tokens.len == 0) return 0;

        // Calculate total probability mass of valid tokens
        var total_mass: f32 = 0.0;
        for (valid_tokens) |token_id| {
            total_mass += probs.data[token_id];
        }

        // Generate random value and sample
        const random = self.rng.random();
        const rand_val = random.float(f32) * total_mass;

        var cumulative: f32 = 0.0;
        for (valid_tokens) |token_id| {
            cumulative += probs.data[token_id];
            if (cumulative >= rand_val) {
                return token_id;
            }
        }

        return valid_tokens[valid_tokens.len - 1];
    }
};

/// Structured output generator using grammar constraints
pub const StructuredGenerator = struct {
    sampler: GrammarConstrainedSampler,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, seed: ?u64) Self {
        return Self{
            .sampler = GrammarConstrainedSampler.init(allocator, seed),
            .allocator = allocator,
        };
    }

    /// Generate JSON matching a schema
    pub fn generateJSON(self: *Self, model: anytype, schema: []const u8, max_tokens: u32) ![]u8 {
        const constraint = JSONConstraint{
            .schema = schema,
            .require_valid = true,
            .allow_partial = true,
        };

        var grammar_state = GrammarState.init(self.allocator, .JSON);
        defer grammar_state.deinit();

        var result = std.ArrayList(u8).init(self.allocator);
        defer result.deinit();

        for (0..max_tokens) |_| {
            // Get model logits (would need actual model interface)
            const logits = try model.getLogits();
            defer logits.deinit(self.allocator);

            // Sample constrained token
            const token_id = try self.sampler.sampleJSONConstrained(
                logits, constraint, &grammar_state, model.tokenizer
            );

            // Get token text and append
            const token_text = try self.sampler.getTokenText(token_id, model.tokenizer);
            defer self.allocator.free(token_text);

            try result.appendSlice(token_text);
            try grammar_state.updateWithToken(token_text);

            // Check if we have a complete valid JSON
            if (self.isCompleteJSON(result.items)) {
                break;
            }
        }

        return result.toOwnedSlice();
    }

    /// Generate text matching a regular expression
    pub fn generateRegex(self: *Self, model: anytype, pattern: []const u8, max_tokens: u32) ![]u8 {
        const constraint = RegexConstraint{
            .pattern = pattern,
            .flags = RegexConstraint.RegexFlags{},
            .max_length = max_tokens * 10, // Rough estimate
        };

        var grammar_state = GrammarState.init(self.allocator, .RegexPattern);
        defer grammar_state.deinit();

        var result = std.ArrayList(u8).init(self.allocator);
        defer result.deinit();

        for (0..max_tokens) |_| {
            const logits = try model.getLogits();
            defer logits.deinit(self.allocator);

            const token_id = try self.sampler.sampleRegexConstrained(
                logits, constraint, &grammar_state, model.tokenizer
            );

            const token_text = try self.sampler.getTokenText(token_id, model.tokenizer);
            defer self.allocator.free(token_text);

            try result.appendSlice(token_text);
            try grammar_state.updateWithToken(token_text);

            // Check if pattern is complete (simplified)
            if (self.matchesPattern(result.items, pattern)) {
                break;
            }
        }

        return result.toOwnedSlice();
    }

    fn isCompleteJSON(self: *Self, text: []const u8) bool {
        _ = self;
        // Simplified JSON completeness check
        var brace_count: i32 = 0;
        var in_string: bool = false;
        var escaped: bool = false;

        for (text) |char| {
            if (escaped) {
                escaped = false;
                continue;
            }

            switch (char) {
                '\\' => if (in_string) escaped = true,
                '"' => in_string = !in_string,
                '{' => if (!in_string) brace_count += 1,
                '}' => if (!in_string) brace_count -= 1,
                else => {},
            }
        }

        return brace_count == 0 and !in_string;
    }

    fn matchesPattern(self: *Self, text: []const u8, pattern: []const u8) bool {
        _ = self;
        _ = text;
        _ = pattern;
        // Simplified pattern matching - would need full regex engine
        return false; // Always generate max tokens for now
    }
};