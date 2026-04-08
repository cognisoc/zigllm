# inference.grammar_constraints

## Module Path

```
zigllama.inference.grammar_constraints
```

**Source file:** `src/inference/grammar_constraints.zig`

---

## Public Types

### `GrammarType`

```zig
pub const GrammarType = enum {
    JSON,
    RegexPattern,
    ContextFree,
    XMLSchema,
    EBNF,
};
```

| Variant | Description |
|---------|-------------|
| `JSON` | Constrain output to valid JSON matching a schema |
| `RegexPattern` | Constrain output to match a regular expression |
| `ContextFree` | Constrain output using context-free grammar rules |
| `XMLSchema` | Constrain output to valid XML matching a schema |
| `EBNF` | Constrain output using Extended Backus-Naur Form |

### `JSONConstraint`

```zig
pub const JSONConstraint = struct {
    schema: []const u8,
    require_valid: bool,
    allow_partial: bool,
};
```

Built-in schema constants for common patterns:

```zig
pub const OBJECT  = "{\"type\":\"object\"}";
pub const ARRAY   = "{\"type\":\"array\"}";
pub const STRING  = "{\"type\":\"string\"}";
pub const NUMBER  = "{\"type\":\"number\"}";
pub const BOOLEAN = "{\"type\":\"boolean\"}";
```

### `RegexConstraint`

```zig
pub const RegexConstraint = struct {
    pattern: []const u8,
    flags: RegexFlags,
    max_length: ?usize,
};

pub const RegexFlags = struct {
    case_insensitive: bool = false,
    multiline: bool = false,
    dot_all: bool = false,
};
```

Built-in patterns: `EMAIL`, `PHONE`, `UUID`, `URL`, `IPV4`, `DATE_ISO`,
`TIME_24H`.

### `CFGConstraint`

```zig
pub const CFGConstraint = struct {
    rules: []const GrammarRule,
    start_symbol: []const u8,
    terminals: []const []const u8,
};
```

### `GrammarRule`

```zig
pub const GrammarRule = struct {
    lhs: []const u8,
    rhs: []const []const u8,
};
```

A single production rule in a context-free grammar: `lhs -> rhs[0] rhs[1] ...`

---

## Public Functions

### `applyConstraint`

```zig
pub fn applyConstraint(
    logits: *Tensor(f32),
    grammar: GrammarType,
    state: *GrammarState,
) !void
```

Modify `logits` in-place to mask out tokens that would violate the grammar
constraint. Tokens that are invalid in the current grammar state have their
logits set to `-inf`.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `logits` | `*Tensor(f32)` | Vocabulary logits (modified in-place) |
| `grammar` | `GrammarType` | Active grammar type |
| `state` | `*GrammarState` | Mutable parser state tracking progress |

### `JSONConstraint.init`

```zig
pub fn init(schema: []const u8) JSONConstraint
```

Create a JSON constraint from a JSON Schema string.

### `JSONConstraint.createStructured`

```zig
pub fn createStructured(
    allocator: std.mem.Allocator,
    fields: []const []const u8,
) !JSONConstraint
```

Build a JSON object schema requiring the given field names, each typed as
string.

### `validateOutput`

```zig
pub fn validateOutput(
    text: []const u8,
    grammar: GrammarType,
    constraint: anytype,
) bool
```

Check whether `text` is valid according to the grammar. Useful as a
post-generation verification step.

---

## Error Types

- `error{InvalidSchema}` -- the provided schema/pattern is malformed.
- `error{InvalidGrammar}` -- a CFG rule references an undefined symbol.
- `error{OutOfMemory}`

---

## Usage Example

```zig
const gc = @import("zigllama").inference.grammar_constraints;

// Constrain generation to valid JSON with specific fields
const constraint = try gc.JSONConstraint.createStructured(
    allocator,
    &[_][]const u8{ "name", "age", "email" },
);

// During generation, apply the constraint before sampling
var state = gc.GrammarState.init(.JSON, allocator);
defer state.deinit();

// In the generation loop:
try gc.applyConstraint(&logits, .JSON, &state);
const token = sampleFromLogits(logits);

// Validate the final output
const valid = gc.validateOutput(output_text, .JSON, constraint);
```

---

## Related Modules

- [`inference.generation`](generation.md) -- Generation loop where constraints
  are applied.
- [`inference.advanced_sampling`](advanced-sampling.md) -- Can be combined with
  grammar constraints for controlled generation.
