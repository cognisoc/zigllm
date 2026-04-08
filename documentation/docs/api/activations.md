# neural_primitives.activations

## Module Path

```
zigllama.neural_primitives.activations
```

**Source file:** `src/neural_primitives/activations.zig`

---

## Public Functions

All activation functions operate element-wise and are designed for use in
neural network layers. Functions accept and return `f32`.

### `relu`

```zig
pub fn relu(x: f32) f32
```

Rectified Linear Unit: `max(0, x)`. The most widely used activation in deep
learning, though LLaMA models prefer SiLU/SwiGLU.

### `gelu`

```zig
pub fn gelu(x: f32) f32
```

Gaussian Error Linear Unit:
`0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
Used in BERT and GPT-2.

### `silu`

```zig
pub fn silu(x: f32) f32
```

Sigmoid Linear Unit (also called Swish): `x * sigmoid(x)`. The primary
activation in LLaMA feed-forward layers.

### `swiglu`

```zig
pub fn swiglu(a: f32, b: f32) f32
```

SwiGLU gated activation: `silu(a) * b`. The gating mechanism used in LLaMA
feed-forward networks. Takes two inputs -- one passes through SiLU, the other
acts as a gate.

### `geglu`

```zig
pub fn geglu(a: f32, b: f32) f32
```

GeGLU gated activation: `gelu(a) * b`. Variant of SwiGLU using GELU instead
of SiLU.

### `glu`

```zig
pub fn glu(a: f32, b: f32) f32
```

Gated Linear Unit: `sigmoid(a) * b`. The original gating mechanism from
Dauphin et al. (2017).

### `sigmoid`

```zig
pub fn sigmoid(x: f32) f32
```

Logistic sigmoid: `1 / (1 + exp(-x))`. Maps any real number to the range
(0, 1).

### `tanh_activation`

```zig
pub fn tanh_activation(x: f32) f32
```

Hyperbolic tangent: `(exp(x) - exp(-x)) / (exp(x) + exp(-x))`. Maps to the
range (-1, 1). Named `tanh_activation` to avoid shadowing `std.math.tanh`.

---

## Error Types

Activation functions are pure scalar operations and do not return errors.
Overflow/underflow is handled gracefully -- `sigmoid` clamps to avoid `inf`,
and `gelu` uses the tanh approximation for numerical stability.

---

## Usage Example

```zig
const act = @import("zigllama").neural_primitives.activations;

// Single-element activations
const x: f32 = -0.5;
const r = act.relu(x);          // 0.0
const s = act.silu(x);          // -0.1881
const g = act.gelu(x);          // -0.1543

// SwiGLU gating (used in LLaMA FFN)
const gate_input: f32 = 1.2;
const up_input: f32 = 0.8;
const result = act.swiglu(gate_input, up_input);

// Apply to a tensor manually
for (tensor.data) |*elem| {
    elem.* = act.silu(elem.*);
}
```

---

## Related Modules

- [`transformers.feed_forward`](feed-forward.md) -- Feed-forward networks that
  use these activations.
- [`neural_primitives.normalization`](normalization.md) -- Often applied before
  or after activation layers.
