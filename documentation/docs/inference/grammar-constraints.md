---
title: "Grammar-Constrained Generation"
description: "Constraining LLM output to valid JSON, regex patterns, and context-free grammars using token masking in ZigLlama."
---

# Grammar-Constrained Generation

Language models generate free-form text, but many applications require
output that conforms to a specific structure -- valid JSON for API responses,
matching a regex for data extraction, or following a context-free grammar for
code generation.  Grammar-constrained generation forces the model to produce
only tokens that maintain validity with respect to a formal specification,
without retraining or fine-tuning.

---

## 1. Constrained Generation

!!! definition "Constrained Decoding"

    Given a grammar \( G \) and the current partial output
    \( x_{<t} \), define the set of **valid next tokens** as:

    \[
        V_G(x_{<t}) = \{ v \in V : x_{<t} \oplus v \text{ is a valid prefix of } G \}
    \]

    where \( \oplus \) denotes concatenation.  At each generation step,
    tokens outside \( V_G \) are masked from the sampling distribution.

The key insight is that constraints are applied *at the logit level*, before
sampling.  This preserves the model's relative preferences among valid
tokens while absolutely preventing invalid ones.

---

## 2. GrammarType Enum

ZigLlama supports five grammar specification formats:

```zig
pub const GrammarType = enum {
    JSON,           // JSON schema constraints
    RegexPattern,   // Regular expression patterns
    ContextFree,    // Context-free grammar (production rules)
    XMLSchema,      // XML schema constraints
    EBNF,           // Extended Backus-Naur Form
};
```

| Grammar Type | Expressiveness | Parse Complexity | Common Use Case |
|---|---|---|---|
| `JSON` | Structured data | \( O(n) \) | API responses, function calling |
| `RegexPattern` | Regular languages | \( O(n) \) | Data extraction, formatting |
| `ContextFree` | Context-free languages | \( O(n^3) \) worst case | Programming languages, nested structures |
| `XMLSchema` | Structured markup | \( O(n) \) | Document generation |
| `EBNF` | Context-free (declarative) | \( O(n^3) \) worst case | Grammar-driven generation |

---

## 3. JSON Constraints

JSON is the most common constrained output format.  The `JSONConstraint`
struct specifies a JSON schema that the output must match:

```zig
pub const JSONConstraint = struct {
    schema: []const u8,        // JSON Schema specification
    require_valid: bool,       // Require structurally valid JSON
    allow_partial: bool,       // Allow incomplete JSON during generation

    pub const OBJECT = "{\"type\":\"object\"}";
    pub const ARRAY = "{\"type\":\"array\"}";
    pub const STRING = "{\"type\":\"string\"}";
    pub const NUMBER = "{\"type\":\"number\"}";
    pub const BOOLEAN = "{\"type\":\"boolean\"}";
};
```

### 3.1 JSON State Tracking

The `GrammarState` struct tracks the parser state during generation,
maintaining a stack of open brackets and braces:

```zig
fn isValidJSONState(self: *Self) bool {
    var brace_count: i32 = 0;
    var bracket_count: i32 = 0;
    var in_string: bool = false;
    var escaped: bool = false;

    for (self.partial_match) |char| {
        if (escaped) { escaped = false; continue; }
        switch (char) {
            '\\' => if (in_string) escaped = true,
            '"' => in_string = !in_string,
            '{' => if (!in_string) brace_count += 1,
            '}' => if (!in_string) brace_count -= 1,
            '[' => if (!in_string) bracket_count += 1,
            ']' => if (!in_string) bracket_count -= 1,
            else => {},
        }
        if (brace_count < 0 or bracket_count < 0) return false;
    }
    return true;
}
```

### 3.2 Structured Data Extraction

The `createStructured` helper generates a JSON schema for extracting
specific fields:

```zig
const constraint = try JSONConstraint.createStructured(allocator, &[_][]const u8{
    "name", "age", "email",
});
// Generates schema: {"type":"object","properties":{
//   "name":{"type":"string"},
//   "age":{"type":"string"},
//   "email":{"type":"string"}
// }}
```

---

## 4. Regex Constraints

`RegexConstraint` restricts output to strings matching a regular expression:

```zig
pub const RegexConstraint = struct {
    pattern: []const u8,
    flags: RegexFlags,
    max_length: ?usize,

    pub const RegexFlags = struct {
        case_insensitive: bool = false,
        multiline: bool = false,
        dot_all: bool = false,
    };
};
```

Built-in patterns for common formats:

| Pattern Constant | Regex | Description |
|---|---|---|
| `EMAIL` | `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$` | Email address |
| `PHONE` | `^\+?[1-9]\d{1,14}$` | E.164 phone number |
| `UUID` | `^[0-9a-fA-F]{8}-...$` | UUID v4 format |
| `URL` | `^https?://[^\s/$.?#].[^\s]*$` | HTTP(S) URL |
| `IPV4` | `^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$` | IPv4 address |
| `DATE_ISO` | `^\d{4}-\d{2}-\d{2}$` | ISO 8601 date |

---

## 5. Context-Free Grammar Constraints

For languages more expressive than regular expressions, ZigLlama supports
context-free grammars specified as production rules:

```zig
pub const CFGConstraint = struct {
    rules: []const GrammarRule,
    start_symbol: []const u8,
    terminals: []const []const u8,

    pub const GrammarRule = struct {
        left: []const u8,          // Non-terminal (left-hand side)
        right: []const []const u8,  // Production alternatives
    };
};
```

### 5.1 Example: Simple Greeting Grammar

```
S    -> "Hello" Name
Name -> "Alice" | "Bob" | "Charlie"
```

```zig
const cfg = try CFGConstraint.createSimple(allocator);
// rules[0]: S -> "Hello" Name
// rules[1]: Name -> "Alice" | "Bob" | "Charlie"
```

### 5.2 CFG Parsing During Generation

The grammar state maintains a **parser stack** that tracks which
non-terminals still need to be expanded.  At each step, the set of valid
tokens is determined by what the top of the stack expects:

!!! algorithm "CFG-Constrained Token Selection"

    **Input:** parser stack \( S \), grammar rules \( G \), vocabulary \( V \)

    1. Let \( A \) = top of stack.
    2. **if** \( A \) is a terminal: valid tokens = tokens that match \( A \).
    3. **if** \( A \) is a non-terminal: for each rule \( A \to \alpha \),
       check if any token in \( V \) is a valid start of \( \alpha \).
    4. Return the union of all valid tokens.

---

## 6. Token Masking

The core mechanism for constrained generation is **logit masking**: setting
the logits of invalid tokens to \( -\infty \) before the softmax, ensuring
they receive zero probability.

!!! definition "Token Masking"

    Given logits \( \mathbf{z} \in \mathbb{R}^{|V|} \) and valid token set
    \( V_G \), the masked logits are:

    \[
        z'_v = \begin{cases}
            z_v & \text{if } v \in V_G \\
            -\infty & \text{otherwise}
        \end{cases}
    \]

    After softmax, invalid tokens receive probability exactly 0:

    \[
        p'(v) = \frac{\exp(z'_v)}{\sum_j \exp(z'_j)} = 0 \quad \text{for } v \notin V_G
    \]

### 6.1 Implementation

The `GrammarConstrainedSampler` iterates over the vocabulary, tests each
token against the grammar state, and builds a list of valid token IDs:

```zig
fn getValidTokens(self: *Self, probs: Tensor(f32), grammar_state: *GrammarState,
                  tokenizer: anytype) ![]u32 {
    var valid_tokens = std.ArrayList(u32).init(self.allocator);

    for (0..probs.data.len) |i| {
        const token_id = @as(u32, @intCast(i));
        const token_text = try self.getTokenText(token_id, tokenizer);
        defer self.allocator.free(token_text);

        if (try self.wouldTokenBeValid(token_text, grammar_state)) {
            try valid_tokens.append(token_id);
        }
    }
    return valid_tokens.toOwnedSlice();
}
```

!!! complexity "Masking Cost"

    Token masking requires testing **every** token in the vocabulary against
    the grammar state: \( O(|V| \cdot C_{\text{check}}) \) where
    \( C_{\text{check}} \) is the cost of one validity check.  For JSON and
    regex this is \( O(n) \) per check (where \( n \) is the current output
    length), giving total cost \( O(|V| \cdot n) \) per generation step.

!!! warning "Performance Impact"

    For large vocabularies (\( |V| = 32{,}000 \) or more), the masking
    step can become a bottleneck.  Production systems (e.g., llama.cpp)
    optimise this by precomputing valid-token sets for common grammar
    states and caching the results.

---

## 7. Examples

### 7.1 Generating Valid JSON

```zig
const generation = @import("inference/generation.zig");
const grammar = @import("inference/grammar_constraints.zig");

// Create structured generator
var gen = grammar.StructuredGenerator.init(allocator, 42);

// Generate JSON matching a schema
const schema =
    \\{"type":"object","properties":{
    \\  "name":{"type":"string"},
    \\  "age":{"type":"number"},
    \\  "active":{"type":"boolean"}
    \\}}
;

const json_output = try gen.generateJSON(&model, schema, 256);
defer allocator.free(json_output);
// Output: {"name":"Alice Smith","age":30,"active":true}
```

### 7.2 Structured Data Extraction

```zig
// Extract specific fields from free-form text
const constraint = try grammar.JSONConstraint.createStructured(allocator, &[_][]const u8{
    "person_name", "occupation", "location",
});

var state = grammar.GrammarState.init(allocator, .JSON);
defer state.deinit();

var sampler = grammar.GrammarConstrainedSampler.init(allocator, 42);
// Use sampler with model to generate constrained output
```

### 7.3 Regex-Constrained Generation

```zig
// Generate a valid email address
const email_output = try gen.generateRegex(
    &model,
    grammar.RegexConstraint.EMAIL,
    64,
);
defer allocator.free(email_output);
```

---

## 8. Comparison with llama.cpp Grammar Support

ZigLlama's grammar constraint system is modelled after llama.cpp's GBNF
(Grammar-Based Notation Format) support, with several differences:

| Feature | ZigLlama | llama.cpp |
|---|---|---|
| Grammar format | Multiple (`GrammarType` enum) | GBNF (custom format) |
| JSON support | Native `JSONConstraint` | Via GBNF rules |
| Regex support | Native `RegexConstraint` | Limited (via GBNF) |
| CFG support | `CFGConstraint` with production rules | Full GBNF parser |
| Token caching | Per-validity-check | Precomputed valid sets |
| Performance | Educational (per-token check) | Optimised (batch check) |
| Integration | `GrammarConstrainedSampler` | Built into `llama_sampler` |

!!! tip "Production Optimisation"

    The primary optimisation opportunity is **precomputing valid token
    sets** for each grammar state.  Since many grammar states recur (e.g.,
    "expecting a JSON key" appears at every object level), caching the
    valid-token set for each state avoids redundant vocabulary scans.
    llama.cpp implements this via a state machine that maps grammar
    positions to precomputed token bitmasks.

---

## 9. Grammar State Machine

The `GrammarState` struct maintains the parser state across generation
steps:

```zig
pub const GrammarState = struct {
    constraint_type: GrammarType,
    current_position: usize,
    stack: std.ArrayList([]const u8),
    context: std.StringHashMap(bool),
    partial_match: []u8,
    allocator: Allocator,

    pub fn updateWithToken(self: *Self, token: []const u8) !void { ... }
    pub fn canContinue(self: *Self) bool { ... }
};
```

The `canContinue` method dispatches to grammar-specific validators based
on the constraint type, checking whether the partial output so far is a
valid prefix of the grammar.

---

## References

[^1]: Gerganov, G. "GBNF Grammar Support in llama.cpp." GitHub, 2023.
[^2]: Willard, B. & Louf, R. "Efficient Guided Generation for Large Language Models." *arXiv:2307.09702*, 2023.
[^3]: Scholak, T., Schucher, N. & Bahdanau, D. "PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models." *EMNLP*, 2021.
[^4]: Shin, R. et al. "Constrained Language Models Yield Few-Shot Semantic Parsers." *EMNLP*, 2021.
