# models.tokenizer

## Module Path

```
zigllama.models.tokenizer
```

**Source file:** `src/models/tokenizer.zig`

---

## Public Types

### `TokenId`

```zig
pub const TokenId = u32;
```

Numeric identifier for a single token.

### `SpecialTokens`

```zig
pub const SpecialTokens = struct {
    pub const UNK: TokenId = 0;
    pub const BOS: TokenId = 1;
    pub const EOS: TokenId = 2;
    pub const PAD: TokenId = 3;
};
```

Reserved token IDs shared across all tokenizer implementations.

### `TokenPiece`

```zig
pub const TokenPiece = struct {
    piece: []const u8,
    score: f32,
};
```

A vocabulary entry pairing a text fragment with a merge priority score.

### `Vocabulary`

```zig
pub const Vocabulary = struct {
    piece_to_id: std.StringHashMap(TokenId),
    id_to_piece: []TokenPiece,
    size: usize,
};
```

Bidirectional mapping between text pieces and token IDs.

### `SimpleTokenizer`

```zig
pub const SimpleTokenizer = struct {
    vocab: Vocabulary,
    allocator: std.mem.Allocator,
};
```

Whitespace/character-level tokenizer intended for testing and educational use.

### `BPETokenizer`

```zig
pub const BPETokenizer = struct {
    vocab: Vocabulary,
    merges: []MergePair,
    allocator: std.mem.Allocator,
};
```

Byte-Pair Encoding tokenizer compatible with LLaMA / SentencePiece vocabularies.

---

## Public Functions

### `SimpleTokenizer.encode`

```zig
pub fn encode(self: SimpleTokenizer, text: []const u8) ![]TokenId
```

Tokenize `text` into a sequence of token IDs. Unrecognized characters map to
`SpecialTokens.UNK`.

### `SimpleTokenizer.decode`

```zig
pub fn decode(self: SimpleTokenizer, tokens: []const TokenId) ![]u8
```

Convert a sequence of token IDs back to UTF-8 text.

### `BPETokenizer.encode`

```zig
pub fn encode(self: BPETokenizer, text: []const u8) ![]TokenId
```

BPE encoding: split text into bytes, then iteratively merge the most frequent
pairs according to the merge table.

### `BPETokenizer.decode`

```zig
pub fn decode(self: BPETokenizer, tokens: []const TokenId) ![]u8
```

Concatenate the text pieces for each token ID.

### `batchEncode`

```zig
pub fn batchEncode(
    tokenizer: anytype,
    texts: []const []const u8,
    allocator: std.mem.Allocator,
) ![][]TokenId
```

Encode multiple strings in one call. Returns a slice of token-ID slices.

### `batchDecode`

```zig
pub fn batchDecode(
    tokenizer: anytype,
    token_seqs: []const []const TokenId,
    allocator: std.mem.Allocator,
) ![][]u8
```

Decode multiple token sequences in one call.

### `padSequences`

```zig
pub fn padSequences(
    sequences: [][]TokenId,
    pad_id: TokenId,
    allocator: std.mem.Allocator,
) ![][]TokenId
```

Pad all sequences to the length of the longest one using `pad_id`.

---

## Error Types

- `error{UnknownToken}` -- encountered a token ID not present in the vocabulary.
- `error{OutOfMemory}`

---

## Usage Example

```zig
const tok = @import("zigllama").models.tokenizer;

var tokenizer = try tok.BPETokenizer.init("tokenizer.model", allocator);
defer tokenizer.deinit();

const ids = try tokenizer.encode("Hello, world!");
defer allocator.free(ids);
// ids might be [1, 15043, 29892, 3186, 29991]

const text = try tokenizer.decode(ids);
defer allocator.free(text);
// text == "Hello, world!"
```

---

## Related Modules

- [`models.llama`](llama.md) -- Passes token IDs from the tokenizer to the
  model.
- [`inference.generation`](generation.md) -- Decodes generated token IDs back
  to text.
- [`models.gguf`](gguf.md) -- GGUF files can embed a vocabulary that
  initializes the tokenizer.
