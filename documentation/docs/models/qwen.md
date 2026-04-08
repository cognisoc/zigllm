# Qwen

## Overview

Qwen (Tongyi Qianwen) is Alibaba Cloud's family of large language models, spanning
multiple generations (Qwen, Qwen2, Qwen2.5, Qwen3) and modalities (language, code,
math, vision, MoE)[^1]. The Qwen architecture is notable for its comprehensive
implementation of **advanced RoPE scaling techniques** -- NTK-Aware scaling, YARN,
Dynamic NTK -- that enable context length extension far beyond the training
distribution.

ZigLLM implements the Qwen architecture in `src/models/qwen.zig`, covering both the
original Qwen (0.5B--72B) and Qwen2 (0.5B--72B) series with full support for the
RoPE scaling variants.

!!! info "Educational Value"
    Qwen is the recommended architecture for studying position encoding extensions.
    It implements more RoPE scaling methods than any other model in ZigLLM, making
    it the definitive reference for understanding how to extend context length in
    RoPE-based transformers.

---

## Key Features

| Feature | Qwen 1.x | Qwen 2.x |
|:--------|:---------|:---------|
| Attention | MHA | **GQA** |
| Context | 32K | **128K** |
| RoPE scaling | Dynamic NTK + LogN | **YARN** |
| Sliding window | No | **Yes (4096)** |
| Bias | Yes (0.5B/1.8B) / No (7B+) | No |
| Normalization | RMSNorm | RMSNorm |
| Activation | SwiGLU | SwiGLU |
| Vocab size | 151,936 | 151,936--152,064 |

### Extensive Variant Family

| Series | Sizes | Key Feature |
|:-------|:------|:------------|
| Qwen | 0.5B, 1.8B, 7B, 14B, 72B | Dynamic NTK + LogN attention |
| Qwen2 | 0.5B, 7B, 72B | GQA + YARN + sliding window |
| Qwen2-MoE | 57B (14B active) | Mixture of Experts |
| Qwen2-VL | 2B, 7B, 72B | Vision-language multimodal |
| Qwen3 | 0.6B--235B | Latest generation |
| Qwen3-MoE | 30B (3B active) | Efficient MoE |

---

## RoPE Scaling Methods

The core innovation in the Qwen family is the exploration of different strategies
for extending RoPE beyond its training context length. ZigLLM supports four
scaling types.

### RopeScaling Configuration

```zig
pub const RopeScaling = struct {
    type: RopeScalingType,
    factor: f32,

    pub const RopeScalingType = enum {
        linear,   // Simple frequency division
        dynamic,  // NTK-Aware dynamic scaling
        yarn,     // Yet Another RoPE extensioN
        ntk,      // NTK-Aware interpolation
    };
};
```

### Linear Scaling

The simplest approach: divide all frequencies by a constant factor \( \alpha \).

\[
\theta'_i = \frac{\theta_i}{\alpha}
\]

This uniformly compresses the frequency spectrum, allowing the model to handle
sequences up to \( \alpha \times L_\text{train} \) positions. However, it degrades
resolution at short distances.

### NTK-Aware Scaling

NTK-Aware interpolation[^2] scales frequencies non-uniformly, preserving high-frequency
(local) components while compressing low-frequency (long-range) components:

\[
\theta'_i = \text{base}'^{-2i/d}, \quad \text{base}' = \text{base} \cdot \alpha^{d/(d-2)}
\]

where \( \alpha = L_\text{target} / L_\text{train} \) is the extension factor.

!!! definition "Why NTK-Aware?"
    Neural Tangent Kernel (NTK) theory suggests that the effective learning rate
    of a neural network depends on the frequency spectrum of its positional
    encodings. NTK-Aware scaling adjusts the RoPE base frequency to maintain the
    same effective spectral distribution when extrapolating beyond training length.

### Dynamic NTK Scaling

Used by Qwen 1.x models. The scaling factor is computed dynamically based on the
actual sequence length at inference time:

\[
\alpha = \begin{cases}
1 & \text{if } L \le L_\text{train} \\
\frac{L}{L_\text{train}} & \text{if } L > L_\text{train}
\end{cases}
\]

```zig
// Qwen 1.x config enables dynamic NTK
.use_dynamic_ntk = true,
.use_logn_attn = true,
```

This means no manual configuration is needed -- the model automatically adjusts
RoPE frequencies when the input exceeds the training length.

### YARN (Yet Another RoPE extensioN)

Used by Qwen2 models. YARN[^3] combines NTK-Aware scaling with an attention
temperature adjustment and a partitioned frequency treatment:

\[
\text{YARN}(\theta_i, m) = \begin{cases}
\theta_i \cdot m & \text{if } i < d_\text{low} \text{ (high-freq, unscaled)} \\
\theta'_i \cdot m / t(i) & \text{if } d_\text{low} \le i < d_\text{high} \text{ (interpolated)} \\
\theta'_i \cdot m / \alpha & \text{if } i \ge d_\text{high} \text{ (low-freq, fully scaled)}
\end{cases}
\]

where \( t(i) \) is a smooth interpolation between 1 and \( \alpha \).

```zig
// Qwen2 config uses YARN scaling
.rope_scaling = RopeScaling{ .type = .yarn, .factor = 4.0 },
```

### Comparison

| Method | Quality at Short | Quality at Long | Config Needed | Used By |
|:-------|:---------------:|:---------------:|:-------------|:--------|
| None | Baseline | Degrades | No | LLaMA 1 |
| Linear | Degraded | Good | Factor | -- |
| NTK | Good | Good | Factor | -- |
| Dynamic NTK | Baseline | Good | No | **Qwen 1.x** |
| YARN | Baseline | Very Good | Factor | **Qwen 2.x** |

---

## LogN Attention Scaling

Unique to Qwen 1.x, LogN attention scaling adjusts the attention output magnitude
based on the ratio of inference length to training length:

\[
\text{scale}(l) = \frac{\log_2(L_\text{train})}{\log_2(2 + l)}
\]

where \( l \) is the layer index. This prevents attention entropy from diverging
at longer sequences.

```zig
if (config.use_logn_attn) {
    logn_list = try allocator.alloc(f32, config.num_hidden_layers);
    for (logn_list.?, 0..) |*logn, i| {
        const layer_idx = @as(f32, @floatFromInt(i));
        logn.* = std.math.log(@as(f32, 512.0)) / std.math.log(2.0 + layer_idx);
    }
}
```

---

## Configuration

### QwenConfig Struct

```zig
pub const QwenConfig = struct {
    variant: QwenVariant,
    vocab_size: u32,
    hidden_size: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,      // GQA support
    num_hidden_layers: u32,
    intermediate_size: u32,
    max_position_embeddings: u32,
    rms_norm_eps: f32,
    rope_theta: f32,               // 1,000,000 for all Qwen variants
    rope_scaling: ?RopeScaling,    // null for Qwen1, YARN for Qwen2
    use_bias: bool,
    use_sliding_window: bool,
    sliding_window: u32,
    use_flash_attn: bool,
    attention_dropout: f32,
    use_cache: bool,
    use_dynamic_ntk: bool,         // Qwen1 feature
    use_logn_attn: bool,           // Qwen1 feature
};
```

### Selected Configurations

| Parameter | Qwen-7B | Qwen2-7B | Qwen2-72B |
|:----------|:--------|:---------|:----------|
| `hidden_size` | 4096 | 3584 | 8192 |
| `num_attention_heads` | 32 | 28 | 64 |
| `num_key_value_heads` | 32 (MHA) | 4 (GQA) | 8 (GQA) |
| `num_hidden_layers` | 32 | 28 | 80 |
| `intermediate_size` | 11008 | 18944 | 29568 |
| `max_position_embeddings` | 32768 | 131072 | 131072 |
| `rope_theta` | 1000000 | 1000000 | 1000000 |
| `rope_scaling` | null | YARN (4x) | YARN (4x) |
| `use_sliding_window` | false | true | true |
| `sliding_window` | 0 | 4096 | 4096 |
| `use_dynamic_ntk` | true | false | false |
| `use_logn_attn` | true | false | false |
| `use_bias` | false | false | false |

!!! tip "High rope_theta"
    All Qwen variants use `rope_theta = 1,000,000` (compared to LLaMA's 10,000).
    A higher base frequency stretches the frequency spectrum, which itself provides
    some context length extension even without additional scaling.

---

## Architecture Components

### Qwen Attention

```zig
pub const QwenAttention = struct {
    config: QwenConfig,
    c_attn: LinearLayer,      // Combined Q,K,V projection
    c_proj: LinearLayer,      // Output projection
    attn_dropout: DropoutLayer,
    resid_dropout: DropoutLayer,
    head_dim: u32,
    kv_head_dim: u32,
    logn_list: ?[]f32,        // LogN scaling factors
};
```

The attention forward pass includes Qwen-specific features:

```zig
pub fn forward(self: *Self, hidden_states: Tensor(f32), layer_idx: u32,
               attention_mask: ?Tensor(f32), position_ids: ?Tensor(u32)) !Tensor(f32) {
    // 1. Single QKV projection (GPT-2 style)
    const qkv = try self.c_attn.forward(hidden_states);

    // 2. Split into Q, K, V with GQA support
    var query = try self.extractTensor(qkv, 0, q_size);
    var key = try self.extractTensor(qkv, q_size, kv_size);
    var value = try self.extractTensor(qkv, q_size + kv_size, kv_size);

    // 3. Apply RoPE with Qwen-specific scaling
    if (position_ids) |pos_ids| {
        query = try self.applyQwenRoPE(query, pos_ids,
            self.config.rope_scaling, self.config.use_dynamic_ntk);
        key = try self.applyQwenRoPE(key, pos_ids,
            self.config.rope_scaling, self.config.use_dynamic_ntk);
    }

    // 4. Grouped-query attention
    var attn_output = try self.computeGroupedQueryAttention(
        query, key, value, attention_mask);

    // 5. LogN attention scaling (Qwen 1.x only)
    if (self.config.use_logn_attn and self.logn_list != null) {
        attn_output = try self.scaleAttentionOutput(
            attn_output, self.logn_list.?[layer_idx]);
    }

    // 6. Output projection
    return try self.c_proj.forward(attn_output);
}
```

### Qwen MLP (SwiGLU)

```zig
pub const QwenMLP = struct {
    w1: LinearLayer,       // Gate projection [d, d_ff]
    w2: LinearLayer,       // Up projection [d, d_ff]
    c_proj: LinearLayer,   // Down projection [d_ff, d]

    pub fn forward(self: *Self, hidden_states: Tensor(f32)) !Tensor(f32) {
        const gate = try self.w1.forward(hidden_states);
        const up = try self.w2.forward(hidden_states);
        const gate_swish = try neural_primitives.swish(gate, self.allocator);
        const gated = try self.elementwiseMultiply(gate_swish, up);
        return try self.c_proj.forward(gated);
    }
};
```

### Qwen Block

```zig
pub const QwenBlock = struct {
    ln_1: RMSNorm,           // Pre-attention norm
    attn: QwenAttention,     // Attention with RoPE scaling
    ln_2: RMSNorm,           // Pre-MLP norm
    mlp: QwenMLP,            // SwiGLU FFN

    pub fn forward(self: *Self, hidden_states: Tensor(f32), layer_idx: u32,
                   attention_mask: ?Tensor(f32), position_ids: ?Tensor(u32)) !Tensor(f32) {
        var residual = hidden_states;
        const normed = try self.ln_1.forward(hidden_states);
        const attn_out = try self.attn.forward(normed, layer_idx,
            attention_mask, position_ids);
        hidden_states = try self.addTensors(residual, attn_out);

        residual = hidden_states;
        const normed2 = try self.ln_2.forward(hidden_states);
        const mlp_out = try self.mlp.forward(normed2);
        return try self.addTensors(residual, mlp_out);
    }
};
```

---

## Qwen 1.x vs Qwen 2.x

The evolution from Qwen 1 to Qwen 2 mirrors the broader industry trend toward
more efficient inference.

| Aspect | Qwen 1.x | Qwen 2.x |
|:-------|:---------|:---------|
| KV heads | Full MHA (\( n_\text{kv} = n_\text{heads} \)) | GQA (\( n_\text{kv} \ll n_\text{heads} \)) |
| Context scaling | Dynamic NTK (runtime) | YARN (fixed factor) |
| Attention scaling | LogN layer-dependent | None needed |
| Sliding window | No | Yes (W=4096) |
| Max context | 32K (native) + extrapolation | 128K (native) |

!!! info "Qwen2's Design Philosophy"
    Qwen2 moved from runtime-adaptive scaling (Dynamic NTK) to a fixed YARN
    configuration. This reflects the finding that YARN with a properly chosen
    factor provides better quality than dynamic adjustment, while being simpler
    to implement and optimize.

---

## Full Model

```zig
pub const QwenModel = struct {
    config: QwenConfig,
    wte: EmbeddingLayer,      // Token embeddings
    h: []QwenBlock,           // Transformer blocks
    ln_f: RMSNorm,            // Final normalization
    lm_head: LinearLayer,     // Output projection

    pub fn forward(self: *Self, input_ids: Tensor(u32),
                   attention_mask: ?Tensor(f32),
                   position_ids: ?Tensor(u32)) !Tensor(f32) {
        var hidden = try self.wte.forward(input_ids);
        for (self.h, 0..) |*block, i| {
            hidden = try block.forward(hidden, @intCast(i),
                attention_mask, position_ids);
        }
        hidden = try self.ln_f.forward(hidden);
        return try self.lm_head.forward(hidden);
    }
};
```

---

## References

[^1]: Bai, J. et al. "Qwen Technical Report." arXiv:2309.16609, 2023.
[^2]: bloc97. "NTK-Aware Scaled RoPE." Reddit/LocalLLaMA, 2023.
[^3]: Peng, B. et al. "YaRN: Efficient Context Window Extension of Large Language Models." arXiv:2309.00071, 2023.
[^4]: Yang, A. et al. "Qwen2 Technical Report." arXiv:2407.10671, 2024.
