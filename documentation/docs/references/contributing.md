---
title: "Contributing"
description: "Guidelines for contributing code, documentation, tests, and tools to the ZigLlama project."
---

# Contributing

ZigLlama is an educational project, and contributions that improve its clarity,
correctness, and coverage are welcome.  This page describes the areas where help
is most needed, the standards every contribution must meet, and the process for
getting changes merged.

---

## 1. Areas for Contribution

The following areas are where external contributions have the most impact.
They are listed in rough order of accessibility -- documentation improvements
require no Zig experience, while new model architectures require deep
familiarity with both the codebase and the relevant research.

### 1.1 Documentation

- **Prose improvements.** Clarify explanations, fix typos, improve
  mathematical notation, or add missing cross-references.
- **Diagrams.** Add or refine Mermaid diagrams for data-flow, architecture
  overviews, or algorithm walkthroughs.
- **Paper annotations.** Expand the [Academic Papers](papers.md) bibliography
  with additional references or more detailed annotations.
- **Glossary entries.** Add missing terms to the [Glossary](glossary.md) or
  improve existing definitions.
- **Tutorials and examples.** Write new walkthroughs that guide readers
  through specific subsystems or use cases.

### 1.2 Tests

- **Unit tests.** Cover untested edge cases in existing modules, especially
  boundary conditions in quantisation and numerical-stability paths.
- **Integration tests.** Verify that multi-module pipelines produce correct
  end-to-end results (e.g., tokenise-encode-decode round-trips).
- **Reference tests.** Compare ZigLlama outputs against reference
  implementations (llama.cpp, PyTorch) for specific model configurations.
- **Performance regression tests.** Add benchmarks that detect unintentional
  performance degradation.

### 1.3 Educational Optimisations

Optimisations are welcome when they *teach* something and remain readable.
For example:

- Cache-blocking strategies for matrix multiplication.
- SIMD vectorisation patterns that are clearly documented.
- Memory-layout transformations with before/after performance data.

The key criterion is that a reader should be able to understand *why* the
optimisation works by reading the code and its comments.

### 1.4 Visualisation Tools

- Attention-pattern visualisation for debugging and education.
- Token-probability heatmaps for sampling strategy comparison.
- Memory-layout diagrams generated from live model data.
- Quantisation error distribution plots.

### 1.5 Language Bindings

Thin wrappers that expose ZigLlama's inference pipeline to other languages:

- C header generation (leveraging Zig's native C ABI).
- Python bindings via `ctypes` or `cffi`.
- Wasm compilation for browser-based demonstrations.

---

## 2. Code Style

ZigLlama follows standard Zig conventions with an emphasis on educational
clarity.

### 2.1 General Principles

| Principle | Rationale |
|-----------|-----------|
| **Clarity over cleverness.** | Every line should be understandable to a reader who knows Zig but not the specific algorithm.  Prefer explicit loops over opaque one-liners. |
| **Consistent naming.** | Use `snake_case` for functions and variables, `PascalCase` for types and namespaces, `SCREAMING_SNAKE_CASE` for compile-time constants. |
| **Explicit allocators.** | Every heap allocation must go through a passed-in `std.mem.Allocator`.  Never use a global allocator. |
| **`defer` / `errdefer` cleanup.** | Pair every allocation with a `defer` or `errdefer` at the point of allocation.  Readers should be able to verify lifetime correctness locally. |
| **No `anytype` without justification.** | Use concrete types or `comptime`-known generics.  If `anytype` is necessary, document the expected interface in a doc comment. |

### 2.2 Formatting

- Run `zig fmt` on all source files before submitting.  The CI pipeline
  rejects improperly formatted code.
- Maximum line length is 100 characters for code, 80 characters for comments
  and documentation.
- Use four-space indentation (Zig default).

### 2.3 Comments and Documentation

- Every public function and type must have a `///` doc comment explaining
  its purpose, parameters, return value, and error conditions.
- Algorithms must reference the paper or textbook section they implement
  (e.g., "See Vaswani et al., 2017, Section 3.2.2").
- Performance-critical code should include a brief complexity note
  (e.g., "O(n * d_k) per head").

---

## 3. Documentation Standards

Documentation in ZigLlama is as important as the code it describes.  Every
module page should include:

### 3.1 Mathematical Foundations

- Define the mathematical operation the module implements using LaTeX
  notation compatible with MathJax.
- State relevant theorems, lemmas, or properties with citations.
- Show the derivation or simplification steps that explain *why* the code
  is structured as it is.

### 3.2 Performance Context

- State the asymptotic complexity of the core operations.
- Note any SIMD, cache-blocking, or quantisation optimisations and their
  measured impact.
- Compare against naive implementations where instructive.

### 3.3 Paper References

- Every algorithm must link back to the relevant entry in the
  [Academic Papers](papers.md) page.
- Use footnote-style citations (`[^N]`) keyed to the bibliography.

### 3.4 Notation Conventions

Follow the notation defined in the [Foundations index](../foundations/index.md):

| Entity | Convention | Example |
|--------|-----------|---------|
| Scalars | Lowercase italic | \( \alpha, x, d_k \) |
| Vectors | Bold lowercase | \( \mathbf{x}, \mathbf{q} \) |
| Matrices | Bold uppercase | \( \mathbf{W}, \mathbf{K} \) |
| Higher-order tensors | Calligraphic | \( \mathcal{T} \) |

---

## 4. Testing Requirements

All pull requests must pass the existing test suite and, where applicable, add
new tests.

### 4.1 Test Categories

| Category | Purpose | Location |
|----------|---------|----------|
| **Unit tests** | Verify individual functions in isolation. | Inline `test` blocks in each `.zig` file. |
| **Integration tests** | Verify multi-module pipelines. | `tests/integration/` |
| **Reference tests** | Compare outputs against known-good results from llama.cpp or PyTorch. | `tests/reference/` |
| **Performance tests** | Guard against regressions in throughput or latency. | `tests/perf/` |

### 4.2 Test Quality Expectations

- Tests must be **deterministic**.  Use fixed seeds for any random operations.
- Tests must be **self-contained**.  Do not depend on downloaded model files
  or external services.
- Tests must be **fast**.  Individual unit tests should complete in under one
  second.  Performance tests may take longer but must be marked accordingly.
- Tests must include **edge cases**: zero-length inputs, maximum-length
  sequences, degenerate quantisation parameters, single-element tensors.

### 4.3 Running Tests

```bash
# Run the full test suite
zig build test

# Run tests for a specific module
zig build test -- --filter "tensor"

# Run with verbose output
zig build test -- --verbose
```

---

## 5. Pull Request Process

### 5.1 Before You Start

1. **Check existing issues and pull requests** to avoid duplicating work.
2. **Open an issue** describing what you plan to change, especially for
   non-trivial contributions.  This allows early feedback before you invest
   significant effort.
3. **Fork the repository** and create a feature branch from `main`.

### 5.2 During Development

1. Make small, focused commits with clear messages.
2. Run `zig fmt` on every modified file.
3. Run `zig build test` and confirm all tests pass.
4. Add or update tests to cover your changes.
5. Update documentation if your change affects public APIs or user-facing
   behaviour.

### 5.3 Submitting

1. Push your feature branch and open a pull request against `main`.
2. Fill in the PR template:
    - **Summary:** One-paragraph description of the change.
    - **Motivation:** Why is this change needed?
    - **Testing:** What tests were added or modified?
    - **Documentation:** What documentation was added or updated?
3. Ensure CI passes (formatting, tests, documentation build).
4. Address review feedback with additional commits (do not force-push during
   review).

### 5.4 Review Criteria

Reviewers evaluate contributions on:

- **Correctness.** Does the code produce the right results?
- **Clarity.** Can a reader unfamiliar with the specific algorithm follow the
  logic?
- **Test coverage.** Are edge cases and error paths exercised?
- **Documentation.** Is the mathematical background and performance context
  provided?
- **Style.** Does the code follow the conventions described above?

---

## 6. What We Are Not Accepting

To preserve ZigLlama's educational mission, the following types of
contributions will generally be declined:

### 6.1 Complex Optimisations That Sacrifice Clarity

- Heavily unrolled or macro-generated kernels where the algorithm is no
  longer readable.
- Platform-specific assembly that cannot be understood from the Zig source
  alone.
- Micro-optimisations that yield marginal speedups but significantly
  increase code complexity.

The guiding question is: *Can a motivated graduate student understand this code
in one sitting?*  If the answer is no, the optimisation does not belong in
ZigLlama, even if it is faster.

### 6.2 Hardware-Specific Code That Limits Accessibility

- GPU compute shaders (CUDA, Metal, Vulkan) that require specific hardware
  to run or test.
- Proprietary library dependencies that are not freely available on all
  major platforms.
- Platform-specific system calls without portable fallbacks.

ZigLlama targets a `zig build` workflow that works on Linux, macOS, and
Windows with no external dependencies beyond the Zig compiler.  Contributions
must preserve this property.

### 6.3 Training and Fine-Tuning

ZigLlama is an inference-only project.  Backward passes, gradient computation,
optimiser implementations, and training loops are out of scope.

---

## Code of Conduct

All participants in the ZigLlama project are expected to behave
professionally and respectfully.  Harassment, discrimination, and personal
attacks are not tolerated.  Technical disagreements should be resolved through
evidence and reasoned argument.
