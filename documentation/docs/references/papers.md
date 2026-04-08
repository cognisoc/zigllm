---
title: "Academic Papers"
description: "Annotated bibliography of the research papers whose algorithms ZigLlama implements, organised by topic with full citations and arXiv links."
---

# Academic Papers

This page collects the primary research papers behind the algorithms, model
architectures, and inference techniques implemented in ZigLlama.  Each entry
provides a full citation, an arXiv link where available, and a one-sentence
annotation explaining the paper's relevance to the project.

Papers are grouped by topic rather than chronology so that readers studying a
particular subsystem -- positional encodings, quantisation, sampling -- can find
the relevant literature in one place.

---

## 1. Foundational Transformers

These papers introduced the core architecture that every model in ZigLlama
builds upon.

| # | Citation | Link | Relevance to ZigLlama |
|---|----------|------|-----------------------|
| 1 | Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). "Attention Is All You Need." *Advances in Neural Information Processing Systems 30 (NeurIPS).* | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | Defines the original transformer architecture -- scaled dot-product attention, multi-head attention, and the encoder-decoder structure -- which forms the blueprint for every Layer 4 module in ZigLlama. |
| 2 | Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). "Language Models are Unsupervised Multitask Learners." *OpenAI Technical Report (GPT-2).* | [OpenAI blog](https://openai.com/research/better-language-models) | Demonstrates that a decoder-only transformer trained on large-scale web text can perform diverse NLP tasks without task-specific fine-tuning; ZigLlama's GPT-2 architecture support reproduces this model family. |
| 3 | Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *Proceedings of NAACL-HLT.* | [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) | Introduces bidirectional pre-training with masked language modelling; ZigLlama implements BERT's encoder architecture and segment embeddings as one of its 18 supported model families. |

---

## 2. LLaMA Family

The model family from which ZigLlama takes its name and its primary reference
architecture.

| # | Citation | Link | Relevance to ZigLlama |
|---|----------|------|-----------------------|
| 4 | Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., & Lample, G. (2023). "LLaMA: Open and Efficient Foundation Language Models." *arXiv preprint.* | [arXiv:2302.13971](https://arxiv.org/abs/2302.13971) | The primary reference architecture for ZigLlama -- RMSNorm, SwiGLU, RoPE, and pre-norm transformer blocks are implemented exactly as described in this paper. |
| 5 | Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. (2023). "Llama 2: Open Foundation and Fine-Tuned Chat Models." *arXiv preprint.* | [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) | Extends the LLaMA architecture with grouped-query attention and RLHF-tuned chat variants; ZigLlama supports the GQA mechanism and Llama 2 model weights. |

---

## 3. Positional Encodings

Mechanisms for injecting sequence-position information into transformer
representations.

| # | Citation | Link | Relevance to ZigLlama |
|---|----------|------|-----------------------|
| 6 | Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv preprint.* | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) | Introduces RoPE, the rotary position embedding used in LLaMA and most ZigLlama-supported architectures; the `rope.zig` module implements the rotation matrices and frequency schedules described here. |
| 7 | Press, O., Smith, N. A., & Lewis, M. (2021). "Train Short, Test Long: Attention with Linear Biases Enables Input Length Generalization." *ICLR 2022.* | [arXiv:2108.12409](https://arxiv.org/abs/2108.12409) | Proposes ALiBi, a position encoding that adds a linear bias to attention scores instead of modifying embeddings; ZigLlama implements ALiBi for architectures (e.g., BLOOM) that use it. |
| 8 | Peng, B., Quesnelle, J., Fan, H., & Shivam, E. (2023). "YaRN: Efficient Context Window Extension of Large Language Models." *arXiv preprint.* | [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) | Describes a method for extending RoPE-based context windows beyond training length via NTK-aware interpolation; relevant to ZigLlama's context-extension support in the RoPE module. |

---

## 4. Activation Functions

Non-linear functions applied within feed-forward sub-layers.

| # | Citation | Link | Relevance to ZigLlama |
|---|----------|------|-----------------------|
| 9 | Hendrycks, D. & Gimpel, K. (2016). "Gaussian Error Linear Units (GELUs)." *arXiv preprint.* | [arXiv:1606.08415](https://arxiv.org/abs/1606.08415) | Defines GELU, used as the activation function in BERT, GPT-2, and several other ZigLlama-supported models; implemented in `activation_functions.zig`. |
| 10 | Ramachandran, P., Zoph, B., & Le, Q. V. (2017). "Searching for Activation Functions." *arXiv preprint.* | [arXiv:1710.05941](https://arxiv.org/abs/1710.05941) | Discovers the Swish activation \( x \cdot \sigma(x) \) through automated search; Swish is the basis for SiLU, which ZigLlama implements as the gating function in SwiGLU. |
| 11 | Shazeer, N. (2020). "GLU Variants Improve Transformer." *arXiv preprint.* | [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) | Shows that gated linear units (especially SwiGLU) outperform standard FFN layers in transformers; ZigLlama's feed-forward module implements SwiGLU as the default for LLaMA-family models. |

---

## 5. Normalization

Layer-wise normalization techniques that stabilise training and inference.

| # | Citation | Link | Relevance to ZigLlama |
|---|----------|------|-----------------------|
| 12 | Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). "Layer Normalization." *arXiv preprint.* | [arXiv:1607.06450](https://arxiv.org/abs/1607.06450) | Introduces LayerNorm, the normalization method used in the original transformer, GPT-2, and BERT; ZigLlama implements it in `normalization.zig`. |
| 13 | Zhang, B. & Sennrich, R. (2019). "Root Mean Square Layer Normalization." *Advances in Neural Information Processing Systems 32.* | [arXiv:1910.07467](https://arxiv.org/abs/1910.07467) | Proposes RMSNorm, which drops the mean-centering step of LayerNorm for faster computation; RMSNorm is the normalization used in LLaMA and the majority of ZigLlama's model architectures. |

---

## 6. Efficient Attention

Variants that reduce the computational or memory cost of attention.

| # | Citation | Link | Relevance to ZigLlama |
|---|----------|------|-----------------------|
| 14 | Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebron, F., & Sanghai, S. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." *arXiv preprint.* | [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) | Introduces grouped-query attention, which reduces KV cache size by sharing key-value heads across query groups; ZigLlama implements GQA in `multi_head_attention.zig` for Llama 2 and Mistral. |
| 15 | Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need." *arXiv preprint.* | [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) | Proposes multi-query attention (MQA), the limiting case of GQA with a single KV head; ZigLlama supports MQA as a configuration option for models such as Falcon. |
| 16 | Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Re, C. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *Advances in Neural Information Processing Systems 35.* | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) | Demonstrates that tiling attention computation to fit in SRAM yields significant wall-clock speedups; ZigLlama's attention implementation uses cache-aware blocking strategies inspired by this work. |

---

## 7. Quantization

Techniques for reducing model weight precision while preserving accuracy.

| # | Citation | Link | Relevance to ZigLlama |
|---|----------|------|-----------------------|
| 17 | Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *arXiv preprint.* | [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) | Introduces GPTQ, a one-shot weight quantisation method based on approximate second-order information; provides theoretical context for the quantisation formats (Q4_0, Q4_1) that ZigLlama implements in Layer 2. |
| 18 | Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., & Han, S. (2023). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." *arXiv preprint.* | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) | Proposes keeping salient weight channels at higher precision based on activation magnitudes; informs ZigLlama's importance-based quantisation strategies in the IQ format family. |
| 19 | Chee, J., Cai, Y., Kuleshov, V., & De Sa, C. (2023). "QuIP: 2-Bit Quantization of Large Language Models With Guarantees." *arXiv preprint.* | [arXiv:2307.13304](https://arxiv.org/abs/2307.13304) | Achieves 2-bit quantisation with theoretical error guarantees using incoherence processing; provides foundational theory for ZigLlama's ultra-low-bit IQ1_S and IQ2 quantisation formats. |

---

## 8. Sampling and Decoding

Strategies for converting logit distributions into text tokens.

| # | Citation | Link | Relevance to ZigLlama |
|---|----------|------|-----------------------|
| 20 | Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). "The Curious Case of Neural Text Degeneration." *ICLR 2020.* | [arXiv:1904.09751](https://arxiv.org/abs/1904.09751) | Introduces nucleus (top-p) sampling, showing that truncating the probability distribution to a cumulative threshold produces more coherent text than pure top-k; ZigLlama implements nucleus sampling in `sampling.zig`. |
| 21 | Basu, S., Banerjee, S., Ganguly, N., & Naskar, A. (2021). "Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity." *ICLR 2021.* | [arXiv:2007.14966](https://arxiv.org/abs/2007.14966) | Proposes Mirostat, an adaptive sampling algorithm that targets a desired perplexity level; ZigLlama implements both Mirostat v1 and v2 in the sampling module. |
| 22 | Meister, C., Pimentel, T., Wiher, G., & Cotterell, R. (2023). "Typical Decoding for Natural Language Generation." *ICLR 2023.* | [arXiv:2202.00666](https://arxiv.org/abs/2202.00666) | Proposes locally typical sampling, which selects tokens whose information content is close to the expected information; ZigLlama implements typical sampling as one of its eight decoding strategies. |

---

## 9. State-Space Models

Alternatives to the attention mechanism based on structured state-space
representations.

| # | Citation | Link | Relevance to ZigLlama |
|---|----------|------|-----------------------|
| 23 | Gu, A. & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." *arXiv preprint.* | [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) | Introduces the Mamba architecture with selective scan, achieving linear-time sequence processing; ZigLlama implements the Mamba model as one of its 18 supported architectures, including the selective-scan mechanism. |

---

## 10. Mixture of Experts

Sparse architectures that activate a subset of model parameters per token.

| # | Citation | Link | Relevance to ZigLlama |
|---|----------|------|-----------------------|
| 24 | Fedus, W., Zoph, B., & Shazeer, N. (2021). "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." *arXiv preprint.* | [arXiv:2101.03961](https://arxiv.org/abs/2101.03961) | Proposes the Switch Transformer, a simplified MoE design with a single-expert routing strategy; ZigLlama's Mixture-of-Experts module implements top-k expert routing as described in this lineage. |

---

## 11. Multi-Modal Models

Architectures that process both visual and textual inputs.

| # | Citation | Link | Relevance to ZigLlama |
|---|----------|------|-----------------------|
| 25 | Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). "Learning Transferable Visual Models From Natural Language Supervision." *ICML 2021.* | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) | Introduces CLIP, which aligns image and text representations through contrastive learning; provides the vision-encoder foundation used in ZigLlama's multi-modal architecture support. |
| 26 | Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). "Visual Instruction Tuning." *NeurIPS 2023.* | [arXiv:2304.08485](https://arxiv.org/abs/2304.08485) | Proposes LLaVA, a vision-language model that connects a CLIP vision encoder to a LLaMA language model via a projection layer; ZigLlama's multi-modal module implements this vision-language architecture. |

---

## 12. Model-Specific Papers

Publications for specific model architectures supported by ZigLlama beyond
the LLaMA family.

| # | Citation | Link | Relevance to ZigLlama |
|---|----------|------|-----------------------|
| 27 | Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., de las Casas, D., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., Lavaud, L. R., Lachaux, M.-A., Stock, P., Le Scao, T., Lavril, T., Wang, T., Lacroix, T., & El Sayed, W. (2023). "Mistral 7B." *arXiv preprint.* | [arXiv:2310.06825](https://arxiv.org/abs/2310.06825) | Introduces Mistral 7B with sliding window attention and grouped-query attention; ZigLlama implements the Mistral architecture including its windowed attention variant. |
| 28 | Penedo, G., Malartic, Q., Hesslow, D., Cojocaru, R., Cappelli, A., Alobeidli, H., Pannier, B., Almazrouei, E., & Launay, J. (2023). "The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only." *NeurIPS 2023 Datasets and Benchmarks Track.* | [arXiv:2306.01116](https://arxiv.org/abs/2306.01116) | Describes the Falcon model family and its training data; ZigLlama implements the Falcon architecture with multi-query attention as one of its supported model families. |
| 29 | Gunasekar, S., Zhang, Y., Anber, J., Hejazinia, R., Lauter, K., Galashov, A., Langford, J., Luber, N., Goodson, B., Holtermann, H., et al. (2023). "Textbooks Are All You Need." *arXiv preprint.* | [arXiv:2306.11644](https://arxiv.org/abs/2306.11644) | Introduces the Phi model family trained on high-quality "textbook" data, achieving strong performance at small scale; ZigLlama implements the Phi architecture with its partial RoPE and dense attention configuration. |
| 30 | Li, R., Allal, L. B., Zi, Y., Muennighoff, N., Kocetkov, D., Mou, C., Marone, M., Akiki, C., Li, J., Chim, J., et al. (2023). "StarCoder: May the Source Be with You!" *arXiv preprint.* | [arXiv:2305.06161](https://arxiv.org/abs/2305.06161) | Describes StarCoder, a code-generation model trained on permissively licensed source code; ZigLlama supports the StarCoder architecture with its multi-query attention and fill-in-the-middle capabilities. |
| 31 | Scao, T. L., Fan, A., Akiki, C., Pavlick, E., Ilic, S., Hesslow, D., Castagne, R., Luccioni, A. S., Yvon, F., Galle, M., et al. (2022). "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model." *arXiv preprint.* | [arXiv:2211.05100](https://arxiv.org/abs/2211.05100) | Introduces BLOOM, a multilingual model using ALiBi positional encodings and LayerNorm; ZigLlama implements the BLOOM architecture with ALiBi as one of its supported model families. |
| 32 | Gemma Team, Google DeepMind. (2024). "Gemma: Open Models Based on Gemini Research and Technology." *arXiv preprint.* | [arXiv:2403.08295](https://arxiv.org/abs/2403.08295) | Describes the Gemma family of lightweight open models derived from Gemini research; ZigLlama implements the Gemma architecture with its GeGLU activation and RoPE configuration. |

---

## Citation Index

The following table provides a quick-reference sorted by first-author surname
for locating a specific paper above.

| First Author | Year | Short Title | Section |
|---|---|---|---|
| Ainslie | 2023 | GQA | [6. Efficient Attention](#6-efficient-attention) |
| Ba | 2016 | Layer Normalization | [5. Normalization](#5-normalization) |
| Basu | 2021 | Mirostat | [8. Sampling and Decoding](#8-sampling-and-decoding) |
| Chee | 2023 | QuIP | [7. Quantization](#7-quantization) |
| Dao | 2022 | FlashAttention | [6. Efficient Attention](#6-efficient-attention) |
| Devlin | 2019 | BERT | [1. Foundational Transformers](#1-foundational-transformers) |
| Fedus | 2021 | Switch Transformers | [10. Mixture of Experts](#10-mixture-of-experts) |
| Frantar | 2022 | GPTQ | [7. Quantization](#7-quantization) |
| Gemma Team | 2024 | Gemma | [12. Model-Specific Papers](#12-model-specific-papers) |
| Gu | 2023 | Mamba | [9. State-Space Models](#9-state-space-models) |
| Gunasekar | 2023 | Phi | [12. Model-Specific Papers](#12-model-specific-papers) |
| Hendrycks | 2016 | GELU | [4. Activation Functions](#4-activation-functions) |
| Holtzman | 2020 | Nucleus Sampling | [8. Sampling and Decoding](#8-sampling-and-decoding) |
| Jiang | 2023 | Mistral 7B | [12. Model-Specific Papers](#12-model-specific-papers) |
| Li | 2023 | StarCoder | [12. Model-Specific Papers](#12-model-specific-papers) |
| Lin | 2023 | AWQ | [7. Quantization](#7-quantization) |
| Liu | 2023 | LLaVA | [11. Multi-Modal Models](#11-multi-modal-models) |
| Meister | 2023 | Typical Decoding | [8. Sampling and Decoding](#8-sampling-and-decoding) |
| Penedo | 2023 | Falcon | [12. Model-Specific Papers](#12-model-specific-papers) |
| Peng | 2023 | YaRN | [3. Positional Encodings](#3-positional-encodings) |
| Press | 2021 | ALiBi | [3. Positional Encodings](#3-positional-encodings) |
| Radford (2019) | 2019 | GPT-2 | [1. Foundational Transformers](#1-foundational-transformers) |
| Radford (2021) | 2021 | CLIP | [11. Multi-Modal Models](#11-multi-modal-models) |
| Ramachandran | 2017 | Swish | [4. Activation Functions](#4-activation-functions) |
| Scao | 2022 | BLOOM | [12. Model-Specific Papers](#12-model-specific-papers) |
| Shazeer (2019) | 2019 | MQA | [6. Efficient Attention](#6-efficient-attention) |
| Shazeer (2020) | 2020 | GLU Variants | [4. Activation Functions](#4-activation-functions) |
| Su | 2021 | RoPE / RoFormer | [3. Positional Encodings](#3-positional-encodings) |
| Touvron (Feb 2023) | 2023 | LLaMA | [2. LLaMA Family](#2-llama-family) |
| Touvron (Jul 2023) | 2023 | Llama 2 | [2. LLaMA Family](#2-llama-family) |
| Vaswani | 2017 | Attention Is All You Need | [1. Foundational Transformers](#1-foundational-transformers) |
| Zhang | 2019 | RMSNorm | [5. Normalization](#5-normalization) |
