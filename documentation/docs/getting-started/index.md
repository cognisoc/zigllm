---
title: "Getting Started"
description: "Install ZigLlama, build from source, run your first inference, and understand the project layout."
---

# Getting Started

This section walks you through everything you need to go from a fresh checkout
to running your first LLM inference with ZigLlama.

---

## Prerequisites at a Glance

| Requirement | Minimum | Recommended |
|---|---|---|
| Zig compiler | 0.13.0 | latest 0.13.x |
| Operating system | Linux, macOS, or Windows | Linux x86-64 |
| RAM | 4 GB | 16 GB+ (for larger models) |
| CPU | Any x86-64 or AArch64 | AVX2-capable x86-64 |

No additional system libraries are required.  BLAS back-ends (OpenBLAS, MKL,
Accelerate) are optional and covered in the installation guide.

---

## Section Contents

<div class="grid cards" markdown>

- :material-download: **[Installation](installation.md)**

    Install the Zig toolchain, clone the repository, verify your setup, and
    optionally configure hardware-accelerated BLAS libraries.

- :material-play-circle: **[Quick Start](quickstart.md)**

    Run the bundled demos, write your first inference program, execute
    layer-by-layer tests, and explore the 12 example files.

- :material-hammer-wrench: **[Building from Source](building.md)**

    Understand `build.zig`, the available build targets and modes,
    cross-compilation, and CI/CD integration.

- :material-file-tree: **[Project Structure](project-structure.md)**

    Navigate the repository: source code layout, the 6-layer mapping, test
    organisation, examples, and documentation.

</div>

---

## Recommended Reading Order

If you are new to ZigLlama, work through the pages in the order listed above:

1. **Installation** -- make sure `zig build test` passes.
2. **Quick Start** -- run the demos and try the minimal code sample.
3. **Building from Source** -- learn how to customise builds.
4. **Project Structure** -- orient yourself in the codebase before diving into
   the layer-by-layer documentation.

After completing this section, continue to the
[Architecture](../architecture/index.md) overview or jump directly to
[Layer 1: Foundations](../foundations/index.md).
