---
title: "Advanced Sampling Methods"
description: "Mirostat, typical sampling, tail-free sampling, contrastive search, and the AdvancedSampler API in ZigLlama."
---

# Advanced Sampling Methods

The basic strategies (greedy, temperature, top-k, top-p) treat every
generation step independently.  Advanced methods go further: they target a
specific entropy level, select tokens based on information-theoretic
criteria, or balance fluency against diversity using hidden-state similarity.
This page covers four such methods and their ZigLlama implementations.

---

## 1. Mirostat v1: Entropy-Targeting Sampling

!!! definition "Mirostat v1 (Basu et al. 2021)"

    Mirostat maintains a target *cross-entropy* (equivalently, perplexity)
    by dynamically adjusting the top-K cutoff.  The algorithm uses a feedback
    loop: if the observed entropy is too high, reduce \( K \); if too low,
    increase \( K \).

    **Parameters:**

    - \( \tau \): target entropy (bits)
    - \( \eta \): learning rate for entropy adaptation
    - \( \epsilon \): convergence threshold

The key insight is that text quality correlates with a *consistent* level
of surprise.  Too-low entropy produces repetitive text; too-high entropy
produces incoherent text.  Mirostat keeps the entropy near the target
regardless of the underlying distribution shape.

!!! algorithm "Mirostat v1 Algorithm"

    **Input:** logits \( \mathbf{z} \), target entropy \( \tau \), learning
    rate \( \eta \), tolerance \( \epsilon \)

    1. Compute probabilities \( \mathbf{p} = \text{softmax}(\mathbf{z}) \).
    2. Compute observed entropy \( H = -\sum_i p_i \log_2 p_i \).
    3. Set initial \( K \leftarrow |V| \).
    4. **repeat**
        1. Select top-\( K \) tokens; compute their sub-distribution
           entropy \( H_K \).
        2. **if** \( |H_K - \tau| < \epsilon \) **then break**.
        3. **if** \( H_K > \tau \) **then** \( K \leftarrow K - 1 \).
        4. **else** \( K \leftarrow K + 1 \).
        5. Update: \( \tau \leftarrow \tau + \eta \cdot (H_K - \tau) \).
    5. Sample from the top-\( K \) distribution.

```zig
fn mirostatV1(self: *Self, logits: Tensor(f32), config: MirostatConfig) !u32 {
    const probs = try self.softmax(logits);
    defer probs.deinit(self.allocator);

    var k = config.max_iterations;
    var current_tau = config.tau;
    var iteration: u32 = 0;

    while (iteration < config.max_iterations) {
        const selected_indices = try self.selectTopK(probs, k);
        defer self.allocator.free(selected_indices);

        var selected_entropy: f32 = 0.0;
        var selected_mass: f32 = 0.0;
        for (selected_indices) |idx| {
            const p = probs.data[idx];
            selected_mass += p;
            if (p > 1e-10) selected_entropy -= p * std.math.log2(p);
        }
        if (selected_mass > 0) selected_entropy /= selected_mass;

        if (@abs(selected_entropy - current_tau) < config.epsilon) break;

        if (selected_entropy > current_tau) {
            k = @max(1, k - 1);
        } else {
            k = @min(@as(u32, @intCast(probs.data.len)), k + 1);
        }

        current_tau += config.eta * (selected_entropy - current_tau);
        iteration += 1;
    }

    const final_selection = try self.selectTopK(probs, k);
    defer self.allocator.free(final_selection);
    return self.sampleFromIndices(probs, final_selection);
}
```

!!! complexity "Mirostat v1 Complexity"

    Each iteration of the feedback loop re-selects top-K tokens, costing
    \( O(|V| \log |V|) \) for the sort.  With at most `max_iterations`
    rounds, the total is \( O(I \cdot |V| \log |V|) \) where \( I \) is
    typically 5--20.

---

## 2. Mirostat v2: Surprisal-Based Selection

!!! definition "Mirostat v2"

    A simplified variant that avoids the iterative feedback loop.  It sorts
    tokens by **surprisal** \( s(x) = -\log_2 p(x) \) in ascending order
    (least surprising first), then accumulates tokens until the
    probability-weighted average surprisal reaches the target \( \tau \):

    \[
        \frac{\sum_{i=1}^{k} p(x_i) \cdot s(x_i)}{\sum_{i=1}^{k} p(x_i)} \ge \tau
    \]

This single-pass approach converges faster and produces comparable quality
to v1 in practice[^1].

```zig
fn mirostatV2(self: *Self, logits: Tensor(f32), config: MirostatConfig) !u32 {
    const probs = try self.softmax(logits);
    defer probs.deinit(self.allocator);

    // Compute surprisal for each token
    const surprisal = try self.allocator.alloc(f32, probs.data.len);
    defer self.allocator.free(surprisal);
    for (probs.data, 0..) |p, i| {
        surprisal[i] = if (p > 1e-10) -std.math.log2(p) else std.math.inf(f32);
    }

    // Sort by surprisal ascending
    const sorted_indices = try self.argsort(surprisal);
    defer self.allocator.free(sorted_indices);

    // Accumulate until weighted entropy reaches tau
    var selected_count: usize = 1;
    var cumulative_prob: f32 = 0.0;
    var weighted_surprisal: f32 = 0.0;

    for (sorted_indices, 0..) |idx, count| {
        cumulative_prob += probs.data[idx];
        weighted_surprisal += probs.data[idx] * surprisal[idx];
        const current_entropy = weighted_surprisal / cumulative_prob;
        if (current_entropy >= config.tau or count >= sorted_indices.len - 1) {
            selected_count = count + 1;
            break;
        }
    }

    selected_count = @max(selected_count, 1);
    return self.sampleFromIndices(probs, sorted_indices[0..selected_count]);
}
```

---

## 3. Typical Sampling

!!! definition "Typical Sampling (Meister et al. 2023)"

    Select tokens whose **information content** is close to the expected
    information content (entropy) of the distribution.  A token is
    "typical" if its surprisal is near the distribution entropy:

    \[
        H(X) = -\sum_{x \in V} p(x) \log p(x)
    \]

    For each token, compute the absolute deviation from typicality:

    \[
        \delta(x) = \bigl| {-\log p(x)} - H(X) \bigr|
    \]

    Sort tokens by \( \delta(x) \) ascending (most typical first), then
    include tokens until the cumulative probability mass reaches a
    threshold \( m \)[^2].

The intuition is that high-probability tokens may be *too* predictable
(low information), while low-probability tokens carry *too much* surprise.
Typical sampling finds the "Goldilocks zone" of information content.

```zig
pub fn sampleTypical(self: *Self, logits: Tensor(f32), config: TypicalConfig) !u32 {
    const probs = try self.softmax(logits);
    defer probs.deinit(self.allocator);

    // Distribution entropy
    var entropy: f32 = 0.0;
    for (probs.data) |p| {
        if (p > 1e-10) entropy -= p * std.math.log2(p);
    }

    // Deviation from typical information content
    const typical_info = try self.allocator.alloc(f32, probs.data.len);
    defer self.allocator.free(typical_info);
    for (probs.data, 0..) |p, i| {
        const information = if (p > 1e-10) -std.math.log2(p) else std.math.inf(f32);
        typical_info[i] = @abs(information - entropy);
    }

    // Sort by typicality (ascending = most typical first)
    const sorted_indices = try self.argsort(typical_info);
    defer self.allocator.free(sorted_indices);

    // Select until mass threshold
    var cumulative_mass: f32 = 0.0;
    var selected_count: usize = 0;
    for (sorted_indices, 0..) |idx, count| {
        cumulative_mass += probs.data[idx];
        selected_count = count + 1;
        if (cumulative_mass >= config.mass) break;
    }

    selected_count = @max(selected_count, config.min_tokens);
    return self.sampleFromIndices(probs, sorted_indices[0..selected_count]);
}
```

!!! info "Typical vs Top-P"

    Top-P selects the most *probable* tokens.  Typical sampling selects the
    most *informationally typical* tokens -- these sets can differ
    significantly.  A very high-probability token (e.g., "the" after "in")
    may be *atypical* because its information content is far below the
    entropy, while a moderately probable token may be more "typical" of the
    distribution's expected behaviour.

---

## 4. Tail-Free Sampling

!!! definition "Tail-Free Sampling"

    Remove the "tail" of the probability distribution by analysing the
    **second derivative** of the sorted probability sequence.  The tail
    begins where the probability curve transitions from the meaningful
    "body" to the low-probability noise.

    1. Sort probabilities in descending order: \( p_{(1)} \ge p_{(2)} \ge \cdots \)
    2. Compute the second derivative (discrete approximation):
       \[
           p''_i = p_{(i-1)} - 2\, p_{(i)} + p_{(i+1)}
       \]
    3. Accumulate the normalised second derivatives until a threshold
       \( z \) is reached.
    4. All tokens beyond the cutoff are removed.

```zig
pub fn sampleTailFree(self: *Self, logits: Tensor(f32), config: TailFreeConfig) !u32 {
    const probs = try self.softmax(logits);
    defer probs.deinit(self.allocator);

    const sorted_indices = try self.argsortDescending(probs.data);
    defer self.allocator.free(sorted_indices);

    // Second derivative computation
    const second_derivatives = try self.allocator.alloc(f32, sorted_indices.len);
    defer self.allocator.free(second_derivatives);

    second_derivatives[0] = 0.0;
    second_derivatives[sorted_indices.len - 1] = 0.0;
    for (1..sorted_indices.len - 1) |i| {
        const p_prev = probs.data[sorted_indices[i - 1]];
        const p_curr = probs.data[sorted_indices[i]];
        const p_next = probs.data[sorted_indices[i + 1]];
        second_derivatives[i] = p_prev - 2.0 * p_curr + p_next;
    }

    // Find tail cutoff
    var cutoff_idx: usize = sorted_indices.len;
    var sum_second_deriv: f32 = 0.0;
    for (1..sorted_indices.len - 1) |i| {
        sum_second_deriv += second_derivatives[i];
        if (sum_second_deriv / @as(f32, @floatFromInt(i)) > config.z) {
            cutoff_idx = i;
            break;
        }
    }

    cutoff_idx = @max(cutoff_idx, config.min_tokens);
    return self.sampleFromIndices(probs, sorted_indices[0..cutoff_idx]);
}
```

---

## 5. Contrastive Search

!!! definition "Contrastive Search (Su et al. 2022)"

    Balance **fluency** (high model probability) and **diversity** (low
    similarity to previous context) by scoring each candidate token with:

    \[
        x_t = \arg\max_{x \in V_K} \left\{
            (1 - \alpha) \cdot \log p(x \mid x_{<t})
            \;-\; \alpha \cdot \max_{v \in x_{<t}} \cos(\mathbf{h}_x, \mathbf{h}_v)
        \right\}
    \]

    where \( \mathbf{h}_x \) is the hidden-state representation of token
    \( x \), and \( \alpha \in [0, 1] \) controls the trade-off.

    - \( \alpha = 0 \): pure likelihood (greedy among top-K).
    - \( \alpha = 1 \): pure diversity (maximally different from context).
    - Typical range: \( \alpha = 0.1\text{--}0.3 \)[^3].

ZigLlama implements a simplified version that uses probability-space
similarity as a proxy when hidden states are not directly available:

```zig
pub fn sampleContrastive(self: *Self, logits: Tensor(f32), alpha: f32, k: u32) !u32 {
    const probs = try self.softmax(logits);
    defer probs.deinit(self.allocator);

    const top_k_indices = try self.selectTopK(probs, k);
    defer self.allocator.free(top_k_indices);

    const scores = try self.allocator.alloc(f32, top_k_indices.len);
    defer self.allocator.free(scores);

    for (top_k_indices, 0..) |idx, i| {
        const likelihood = std.math.log(probs.data[idx]);
        var diversity_penalty: f32 = 0.0;
        for (top_k_indices) |other_idx| {
            if (other_idx != idx) {
                const similarity = 1.0 - @abs(probs.data[idx] - probs.data[other_idx]);
                diversity_penalty += similarity * probs.data[other_idx];
            }
        }
        scores[i] = likelihood - alpha * diversity_penalty;
    }

    // Select highest-scoring token
    var best_idx: usize = 0;
    var best_score: f32 = scores[0];
    for (scores, 0..) |score, i| {
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }
    return top_k_indices[best_idx];
}
```

!!! warning "Full Contrastive Search"

    The full algorithm requires access to the model's hidden-state
    representations to compute cosine similarity.  ZigLlama's simplified
    version uses probability proximity as a proxy, which captures some of
    the diversity benefit but is less effective than the original formulation.

---

## 6. AdvancedSampler API

All advanced methods are accessed through the `AdvancedSampler` struct:

```zig
pub const AdvancedSampler = struct {
    allocator: Allocator,
    rng: std.rand.DefaultPrng,

    pub fn init(allocator: Allocator, seed: ?u64) Self { ... }

    pub fn sampleMirostat(self: *Self, logits: Tensor(f32), config: MirostatConfig) !u32 { ... }
    pub fn sampleTypical(self: *Self, logits: Tensor(f32), config: TypicalConfig) !u32 { ... }
    pub fn sampleTailFree(self: *Self, logits: Tensor(f32), config: TailFreeConfig) !u32 { ... }
    pub fn sampleLocallyTypical(self: *Self, logits: Tensor(f32), config: LocallyTypicalConfig) !u32 { ... }
    pub fn sampleContrastive(self: *Self, logits: Tensor(f32), alpha: f32, k: u32) !u32 { ... }
};
```

Configuration structs for each method:

```zig
pub const MirostatConfig = struct {
    version: enum { V1, V2 },
    tau: f32,             // Target entropy
    eta: f32,             // Learning rate
    epsilon: f32,         // Convergence threshold
    max_iterations: u32,  // Max adaptation rounds
};

pub const TypicalConfig = struct {
    mass: f32,            // Typical mass threshold (0.0--1.0)
    min_tokens: u32,      // Minimum tokens to keep
};

pub const TailFreeConfig = struct {
    z: f32,               // Tail-free threshold (0.0--1.0)
    min_tokens: u32,      // Minimum tokens to keep
};
```

The `SamplingCoordinator` provides adaptive strategy selection based on
distribution characteristics (entropy, peak probability):

```zig
pub const SamplingCoordinator = struct {
    pub fn adaptiveSample(self: *Self, logits: Tensor(f32)) !u32 {
        // High confidence (max_prob > 0.8) -> contrastive
        // Low entropy (< 2.0)              -> typical
        // High entropy (> 6.0)             -> mirostat v2
        // Moderate entropy                 -> tail-free
    }
};
```

---

## 7. Comparison Table

| Method | Parameters | Stateful | Adaptive | Best Use Case |
|---|---|---|---|---|
| **Mirostat v1** | \( \tau, \eta, \epsilon \) | Yes (K evolves) | Entropy-targeting | Long-form text with consistent quality |
| **Mirostat v2** | \( \tau, \eta \) | Minimal | Surprisal-based | General purpose, simpler than v1 |
| **Typical** | mass, min_tokens | No | Distribution-aware | Avoiding both repetition and incoherence |
| **Tail-Free** | z, min_tokens | No | Curvature-based | Removing low-quality tail tokens |
| **Contrastive** | \( \alpha \), K | Yes (needs context) | Similarity-based | Reducing repetition in dialogue |

!!! tip "When to Use Advanced Sampling"

    - **Mirostat**: When you want consistent perceived quality across varying
      prompt difficulties.  Good default for chat applications.
    - **Typical**: When the base distribution is highly variable and you want
      to avoid both the most and least likely tokens.
    - **Tail-Free**: When top-K/top-P still let through nonsensical tokens
      and you want a principled tail removal.
    - **Contrastive**: When repetition is the primary quality issue, especially
      in long conversations or story generation.

---

## References

[^1]: Basu, S. et al. "Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity." *ICLR*, 2021.
[^2]: Meister, C. et al. "Typical Decoding for Natural Language Generation." *EMNLP*, 2023.
[^3]: Su, Y. et al. "A Contrastive Framework for Neural Text Generation." *NeurIPS*, 2022.
[^4]: Holtzman, A. et al. "The Curious Case of Neural Text Degeneration." *ICLR*, 2020.
