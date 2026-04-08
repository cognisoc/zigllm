---
title: "Glossary"
description: "Alphabetical glossary of 80+ terms used throughout the ZigLlama documentation, with definitions, mathematical notation, and module references."
---

# Glossary

This glossary defines the core terminology used throughout the ZigLlama
documentation.  Each entry includes a concise definition, mathematical notation
where applicable, and a reference to the ZigLlama module or source file that
implements the concept.  Terms are listed alphabetically.

---

## A

**Activation Function**
:   A non-linear function applied element-wise to the output of a linear
    transformation.  Without activation functions, any composition of linear
    layers collapses to a single linear map.  Common examples include ReLU,
    GELU, SiLU, and SwiGLU.
    :material-code-tags: `activation_functions.zig`

**ALiBi (Attention with Linear Biases)**
:   A position encoding method that adds a head-specific linear bias to
    attention scores rather than modifying the input embeddings.  For head
    \( h \) and positions \( i, j \): \( \text{bias}_{h}(i, j) = -m_h \cdot |i - j| \),
    where \( m_h \) is a head-specific slope.
    :material-code-tags: `multi_head_attention.zig` (ALiBi path)

**Attention**
:   The mechanism by which a model computes a weighted combination of value
    vectors, where the weights are derived from the similarity between query
    and key vectors.  See *Scaled Dot-Product Attention* for the standard
    formulation.
    :material-code-tags: `multi_head_attention.zig`

**Autoregressive**
:   A generation mode in which each output token is conditioned on all
    previously generated tokens.  At step \( t \), the model computes
    \( P(x_t \mid x_1, x_2, \dots, x_{t-1}) \).  All decoder-only models
    in ZigLlama operate autoregressively during inference.
    :material-code-tags: `text_generation.zig`

## B

**Batch Processing**
:   The technique of processing multiple independent sequences simultaneously
    to improve hardware utilisation.  ZigLlama's batch processor groups
    sequences by length and processes them in parallel.
    :material-code-tags: `batch_processing.zig`

**BLAS (Basic Linear Algebra Subprograms)**
:   A standardised API for matrix and vector operations (GEMM, GEMV, etc.).
    ZigLlama optionally links against OpenBLAS, MKL, or Apple Accelerate
    through a vtable-based interface, falling back to pure-Zig SIMD kernels
    when no external library is available.
    :material-code-tags: `blas_integration.zig`

**BPE (Byte Pair Encoding)**
:   A subword tokenisation algorithm that iteratively merges the most frequent
    adjacent character or byte pairs in a corpus.  The LLaMA tokeniser uses a
    SentencePiece-based BPE variant.  GPT-2 uses byte-level BPE.
    :material-code-tags: `tokenizer.zig`

## C

**Causal Mask**
:   A binary (or additive \(-\infty\)) mask applied to the attention score matrix
    to prevent a position from attending to future positions.  For positions
    \( i, j \): \( \text{mask}(i, j) = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases} \).
    :material-code-tags: `multi_head_attention.zig`

**Context Window**
:   The maximum number of tokens a model can attend to in a single forward pass.
    Determined by the positional encoding scheme and training configuration.
    Typical values range from 2048 (LLaMA 1) to 32768 (Mistral) or more.
    :material-code-tags: `model_config.zig`

**Cross-Attention**
:   An attention mechanism where queries come from one sequence and keys/values
    come from a different sequence.  Used in encoder-decoder models and
    multi-modal architectures (e.g., attending from language tokens to vision
    features).
    :material-code-tags: `multi_head_attention.zig`

## D

**d_model**
:   The dimensionality of the hidden representations throughout the
    transformer.  All residual connections, layer norms, and embeddings operate
    in \(\mathbb{R}^{d_\text{model}}\).  For LLaMA 7B, \( d_\text{model} = 4096 \).
    :material-code-tags: `model_config.zig`

**Decoder**
:   The autoregressive half of the original transformer architecture.  In
    decoder-only models (LLaMA, GPT-2, Mistral), the entire model is a stack
    of decoder blocks with causal masking.
    :material-code-tags: `transformer_block.zig`

**Dequantization**
:   The process of converting quantised (low-bit) weight representations back
    to floating-point values for computation.  For a block-quantised weight
    \( w_q \) with scale \( s \) and zero-point \( z \):
    \( w \approx s \cdot (w_q - z) \).
    :material-code-tags: `quantization.zig`, `k_quantization.zig`

## E

**Embedding**
:   A learned mapping from discrete token indices to continuous vectors.
    Formally, an embedding table \( \mathbf{E} \in \mathbb{R}^{V \times d_\text{model}} \)
    maps token index \( i \) to row \( \mathbf{E}[i] \).
    :material-code-tags: `embeddings.zig`

**Encoder**
:   The bidirectional half of the transformer architecture.  Encoder blocks
    apply self-attention without causal masking, allowing each position to
    attend to all others.  Used in BERT-family models.
    :material-code-tags: `transformer_block.zig` (BERT path)

## F

**Feed-Forward Network (FFN)**
:   The position-wise fully connected sub-layer within each transformer block.
    In the standard formulation: \( \text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2 \).
    LLaMA uses a SwiGLU variant with three weight matrices and no bias terms.
    :material-code-tags: `feed_forward.zig`

## G

**GGML**
:   A C tensor library for machine learning created by Georgi Gerganov,
    forming the computational foundation of llama.cpp.  ZigLlama's tensor
    implementation is informed by GGML's design but is written in pure Zig.
    :material-code-tags: `tensor.zig`

**GGUF (GGML Unified Format)**
:   A binary file format (version 3) for storing quantised model weights and
    metadata.  The format consists of a magic number, metadata key-value pairs,
    tensor descriptors, and alignment-padded tensor data.
    :material-code-tags: `gguf_format.zig`

**GQA (Grouped-Query Attention)**
:   An attention variant where multiple query heads share a single key-value
    head pair.  If there are \( n_q \) query heads and \( n_{kv} \) KV heads,
    each KV head is shared by \( n_q / n_{kv} \) query heads.  Reduces KV
    cache size by a factor of \( n_q / n_{kv} \).  GQA generalises both
    MHA (\( n_{kv} = n_q \)) and MQA (\( n_{kv} = 1 \)).
    :material-code-tags: `multi_head_attention.zig`

**Gradient**
:   The vector of partial derivatives of a scalar loss with respect to model
    parameters: \( \nabla_\theta \mathcal{L} \).  ZigLlama is an
    inference-only system and does not compute gradients, but understanding
    them explains architectural choices such as residual connections and
    pre-norm placement.

## H

**Head Dimension**
:   The dimensionality of each attention head, typically
    \( d_k = d_\text{model} / n_\text{heads} \).  For LLaMA 7B with
    \( d_\text{model} = 4096 \) and 32 heads, \( d_k = 128 \).
    :material-code-tags: `multi_head_attention.zig`

## I

**Importance Quantization (IQ)**
:   A family of ultra-low-bit quantisation formats (IQ1_S, IQ2_XXS, IQ2_XS,
    IQ3_XXS, IQ4_NL, etc.) that use non-uniform quantisation levels selected
    to minimise reconstruction error for the most important weights.
    :material-code-tags: `iq_quantization.zig`

**Inference**
:   The process of computing model outputs from inputs without updating model
    weights.  ZigLlama is exclusively an inference engine -- it loads
    pre-trained weights and generates text but does not train models.
    :material-code-tags: `text_generation.zig`

## K

**K-Quantization**
:   A family of block quantisation formats (Q2_K through Q6_K) introduced in
    llama.cpp that use per-block scales and mins with varying bit-widths.
    K-quant formats achieve better quality-per-bit than the older Q4_0/Q8_0
    formats by using super-blocks with nested quantisation of scale factors.
    :material-code-tags: `k_quantization.zig`

**KV Cache**
:   A buffer that stores previously computed key and value projections for all
    layers, enabling \( O(1) \) per-token re-use during autoregressive
    generation instead of \( O(T) \) recomputation.  For a model with \( L \)
    layers, \( n_{kv} \) heads, and head dimension \( d_k \), the cache stores
    \( 2 \cdot L \cdot n_{kv} \cdot T \cdot d_k \) elements.
    :material-code-tags: `kv_cache.zig`

## L

**Layer Normalization (LayerNorm)**
:   A normalization technique that computes mean and variance across the
    feature dimension for each individual sample:
    \( \text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \),
    where \( \mu \) and \( \sigma^2 \) are the mean and variance of \( x \).
    Used in GPT-2, BERT, and BLOOM.
    :material-code-tags: `normalization.zig`

**Logits**
:   The raw (unnormalized) output scores produced by the final linear
    projection of the model.  A logit vector
    \( \mathbf{z} \in \mathbb{R}^{V} \) is converted to a probability
    distribution via softmax: \( P(w_i) = \frac{e^{z_i}}{\sum_j e^{z_j}} \).
    :material-code-tags: `text_generation.zig`

**LoRA (Low-Rank Adaptation)**
:   A parameter-efficient fine-tuning method that factorises weight updates
    as \( \Delta W = BA \) where \( B \in \mathbb{R}^{d \times r} \) and
    \( A \in \mathbb{R}^{r \times d} \) with rank \( r \ll d \).  ZigLlama
    does not implement LoRA training but can load models with merged LoRA
    weights.

## M

**Masked Language Model (MLM)**
:   A pre-training objective where random tokens are replaced with a `[MASK]`
    token, and the model predicts the original token.  Used by BERT.  Distinct
    from causal (autoregressive) language modelling.
    :material-code-tags: `transformer_block.zig` (BERT path)

**MHA (Multi-Head Attention)**
:   The standard attention mechanism where \( n \) independent attention heads
    each compute scaled dot-product attention on separate learned projections,
    and the results are concatenated and linearly projected:
    \( \text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_n) W^O \).
    :material-code-tags: `multi_head_attention.zig`

**Mixture of Experts (MoE)**
:   An architecture where each token is routed to a subset of \( k \) "expert"
    feed-forward networks out of \( N \) total experts.  A gating network
    produces routing weights: \( y = \sum_{i \in \text{top-}k} g_i \cdot E_i(x) \).
    :material-code-tags: `mixture_of_experts.zig`

**mmap (Memory-Mapped I/O)**
:   An operating system facility that maps file contents into the virtual
    address space, enabling the process to access file data as if it were
    in-memory.  ZigLlama uses `mmap` to load multi-gigabyte GGUF model files
    with near-zero copy overhead and OS-managed paging.
    :material-code-tags: `memory_mapping.zig`

**MQA (Multi-Query Attention)**
:   The special case of grouped-query attention with a single KV head shared
    across all query heads (\( n_{kv} = 1 \)).  Reduces KV cache size by a
    factor of \( n_\text{heads} \).  Used by Falcon and StarCoder.
    :material-code-tags: `multi_head_attention.zig`

## N

**NEON**
:   ARM's SIMD instruction set for 128-bit vector operations.  ZigLlama's
    `@Vector` types compile to NEON instructions on ARM platforms (e.g.,
    Apple Silicon, Raspberry Pi), enabling portable SIMD without
    architecture-specific intrinsics.
    :material-code-tags: `simd_operations.zig`

**Nucleus Sampling**
:   See *Top-P*.

**NUMA (Non-Uniform Memory Access)**
:   A memory architecture where access latency depends on which CPU socket
    "owns" the memory.  ZigLlama's thread pool supports NUMA-aware work
    distribution to minimise cross-socket memory traffic during parallel
    matrix operations.
    :material-code-tags: `threading.zig`

## P

**Perplexity**
:   A measure of how well a probability model predicts a sample.  Defined as
    the exponentiated average negative log-likelihood:
    \( \text{PPL} = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{<t})\right) \).
    Lower is better.  ZigLlama includes a perplexity evaluation tool.
    :material-code-tags: `perplexity.zig`

**Position Encoding**
:   Any mechanism that injects information about token position into the
    model's representations.  Approaches include sinusoidal embeddings
    (original transformer), learned embeddings (GPT-2), rotary embeddings
    (RoPE), and linear biases (ALiBi).
    :material-code-tags: `rope.zig`, `embeddings.zig`

**Pre-Norm**
:   A transformer block ordering where normalization is applied *before* the
    attention and FFN sub-layers rather than after.  The residual connection
    wraps the normalized sub-layer:
    \( x' = x + \text{Attn}(\text{Norm}(x)) \).  Used by LLaMA, Mistral,
    and most modern architectures.  Improves training stability.
    :material-code-tags: `transformer_block.zig`

## Q

**Quantization**
:   The process of representing model weights at reduced numerical precision
    (e.g., 4-bit, 2-bit) to decrease memory footprint and increase inference
    throughput.  ZigLlama supports 18+ formats spanning basic (Q4_0, Q8_0),
    K-quant (Q2_K through Q6_K), and IQ (IQ1_S through IQ4_NL) families.
    :material-code-tags: `quantization.zig`, `k_quantization.zig`, `iq_quantization.zig`

## R

**ReLU (Rectified Linear Unit)**
:   The activation function \( \text{ReLU}(x) = \max(0, x) \).  Simple and
    efficient but suffers from the "dying ReLU" problem where neurons with
    negative inputs produce zero gradients permanently.  Largely superseded by
    GELU and SiLU in modern transformers.
    :material-code-tags: `activation_functions.zig`

**Repetition Penalty**
:   A decoding modification that reduces the probability of tokens that have
    already appeared in the generated text.  Applied by dividing the logit
    of a repeated token by a penalty factor \( \alpha > 1 \):
    \( z'_i = z_i / \alpha \) if token \( i \) has appeared previously.
    :material-code-tags: `sampling.zig`

**Residual Connection**
:   An additive skip connection that lets the input of a sub-layer pass
    directly to its output: \( y = x + f(x) \).  Residual connections
    mitigate the vanishing gradient problem in deep networks and are present
    around every attention and FFN sub-layer in the transformer.
    :material-code-tags: `transformer_block.zig`

**RMSNorm (Root Mean Square Layer Normalization)**
:   A normalization variant that normalises by the root mean square of the
    input, omitting the mean-centering step:
    \( \text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \epsilon}} \).
    Faster than LayerNorm with comparable quality.  Default for LLaMA, Mistral,
    and most ZigLlama architectures.
    :material-code-tags: `normalization.zig`

**RoPE (Rotary Position Embedding)**
:   A position encoding that applies a rotation in 2D subspaces of the query
    and key vectors.  For position \( m \) and dimension pair \( (2i, 2i+1) \):
    \( \begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix} \),
    where \( \theta_i = 10000^{-2i/d} \).  RoPE encodes relative position
    naturally because \( \langle q'_m, k'_n \rangle \) depends only on \( m - n \).
    :material-code-tags: `rope.zig`

## S

**Sampling**
:   The process of selecting the next token from the probability distribution
    produced by the model.  ZigLlama implements eight strategies: greedy,
    top-k, top-p (nucleus), temperature, Mirostat v1, Mirostat v2, typical,
    and tail-free sampling.
    :material-code-tags: `sampling.zig`

**Scaled Dot-Product Attention**
:   The core attention computation:
    \( \text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^{\!\top}}{\sqrt{d_k}}\right) V \).
    The \( 1/\sqrt{d_k} \) scaling prevents the dot products from growing
    large in magnitude, which would push softmax into saturation.
    :material-code-tags: `multi_head_attention.zig`

**Segment Embedding**
:   An embedding added to distinguish between different segments of the input
    (e.g., sentence A vs. sentence B in BERT's next-sentence prediction task).
    :material-code-tags: `embeddings.zig` (BERT path)

**Selective Scan**
:   The input-dependent state transition mechanism in the Mamba architecture.
    Unlike traditional SSMs with fixed dynamics, selective scan computes
    state transition matrices \( \mathbf{A}, \mathbf{B}, \mathbf{C} \) as
    functions of the input, enabling content-based reasoning.
    :material-code-tags: `mamba.zig`

**Self-Attention**
:   Attention where queries, keys, and values are all derived from the same
    input sequence: \( Q = XW^Q,\ K = XW^K,\ V = XW^V \).  Used in every
    transformer block (with causal masking in decoders).
    :material-code-tags: `multi_head_attention.zig`

**SIMD (Single Instruction, Multiple Data)**
:   A hardware capability for performing the same operation on multiple data
    elements simultaneously.  ZigLlama uses Zig's `@Vector` type to express
    SIMD operations that compile to AVX2, AVX-512, or NEON instructions
    depending on the target platform.
    :material-code-tags: `simd_operations.zig`

**Sliding Window Attention**
:   An attention variant where each token attends only to the \( w \) most
    recent tokens rather than the entire sequence.  Reduces attention
    complexity from \( O(T^2) \) to \( O(T \cdot w) \).  Used in Mistral.
    :material-code-tags: `multi_head_attention.zig`

**Softmax**
:   The function that converts a vector of real-valued scores into a
    probability distribution:
    \( \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}} \).
    Implemented with the log-sum-exp trick for numerical stability:
    subtract \( \max(z) \) before exponentiation.
    :material-code-tags: `activation_functions.zig`

**SSM (State-Space Model)**
:   A sequence model defined by the continuous-time dynamics
    \( h'(t) = Ah(t) + Bx(t),\ y(t) = Ch(t) + Dx(t) \), discretised for
    use with sequential data.  Mamba is an SSM with input-dependent
    (selective) state transitions.
    :material-code-tags: `mamba.zig`

**Streaming**
:   An output mode where tokens are emitted to the caller as soon as they are
    generated, rather than waiting for the entire sequence to complete.
    Enables responsive user interfaces and reduces perceived latency.
    :material-code-tags: `streaming.zig`

**SwiGLU**
:   A gated activation function combining Swish (SiLU) with a gated linear
    unit:
    \( \text{SwiGLU}(x, W_1, W_3) = (\text{SiLU}(xW_1)) \odot (xW_3) \).
    The default FFN activation in LLaMA and most modern architectures.
    Requires three weight matrices instead of two, with the intermediate
    dimension typically set to \( \frac{2}{3} \cdot 4 \cdot d_\text{model} \).
    :material-code-tags: `activation_functions.zig`, `feed_forward.zig`

## T

**Temperature**
:   A scaling parameter applied to logits before softmax to control the
    entropy of the output distribution.  At temperature \( \tau \):
    \( P(w_i) \propto \exp(z_i / \tau) \).  \( \tau < 1 \) sharpens the
    distribution (more deterministic); \( \tau > 1 \) flattens it (more
    random); \( \tau = 1 \) is the unmodified distribution.
    :material-code-tags: `sampling.zig`

**Tensor**
:   A multi-dimensional array of numerical values.  In ZigLlama, the generic
    `Tensor(T)` struct stores data in row-major (C) order with explicit shape
    and stride metadata.  Scalars are rank-0 tensors, vectors rank-1, matrices
    rank-2, and so on.
    :material-code-tags: `tensor.zig`

**Token**
:   The atomic unit of text processed by the model.  Tokens are produced by
    the tokeniser and may represent whole words, subword fragments, individual
    characters, or individual bytes, depending on the vocabulary.
    :material-code-tags: `tokenizer.zig`

**Tokenizer**
:   The component that converts raw text into a sequence of integer token IDs
    and vice versa.  ZigLlama supports SentencePiece (LLaMA), byte-level BPE
    (GPT-2), and WordPiece (BERT) tokenisation schemes.
    :material-code-tags: `tokenizer.zig`

**Top-K**
:   A sampling strategy that restricts the candidate set to the \( k \) tokens
    with the highest probability, then renormalises.  Setting \( k = 1 \) is
    equivalent to greedy decoding.
    :material-code-tags: `sampling.zig`

**Top-P (Nucleus Sampling)**
:   A sampling strategy that includes tokens in decreasing probability order
    until the cumulative probability exceeds a threshold \( p \in (0, 1] \),
    then renormalises.  Dynamically adjusts the candidate set size based on
    the shape of the distribution.
    :material-code-tags: `sampling.zig`

**Transformer**
:   The neural network architecture introduced by Vaswani et al. (2017),
    based on self-attention rather than recurrence or convolution.  A
    transformer consists of stacked transformer blocks, each containing a
    multi-head attention sub-layer and a feed-forward sub-layer, connected
    by residual connections and normalization.
    :material-code-tags: `transformer_block.zig`

**Transformer Block**
:   A single repeating unit of the transformer architecture.  In pre-norm
    configuration (LLaMA):
    \( x' = x + \text{Attn}(\text{RMSNorm}(x)) \),
    \( x'' = x' + \text{FFN}(\text{RMSNorm}(x')) \).
    ZigLlama stacks \( L \) such blocks (e.g., \( L = 32 \) for LLaMA 7B).
    :material-code-tags: `transformer_block.zig`

## V

**ViT (Vision Transformer)**
:   A transformer architecture applied to images by dividing the image into
    fixed-size patches, projecting each patch into an embedding, and
    processing the resulting sequence with standard transformer blocks.  Used
    as the vision encoder in ZigLlama's multi-modal architecture.
    :material-code-tags: `multi_modal.zig`
