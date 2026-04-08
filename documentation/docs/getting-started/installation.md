---
title: "Installation"
description: "Install the Zig toolchain, clone ZigLlama, verify the build, and optionally configure BLAS acceleration."
---

# Installation

This page covers every step required to build and test ZigLlama on a fresh
machine.  By the end you will have a working checkout with all 285+ tests
passing.

---

## Prerequisites

### Zig 0.13+

ZigLlama targets **Zig 0.13.0** or later.  Zig is distributed as a single
static binary -- no installer, no system packages, no dependency chains.

=== "Linux"

    ```bash
    # Download the latest 0.13.x release
    curl -LO https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz
    tar xf zig-linux-x86_64-0.13.0.tar.xz

    # Move to a permanent location
    sudo mv zig-linux-x86_64-0.13.0 /opt/zig

    # Add to PATH (append to ~/.bashrc or ~/.zshrc)
    export PATH="/opt/zig:$PATH"
    ```

=== "macOS"

    ```bash
    # Via Homebrew (recommended)
    brew install zig

    # Or download manually
    curl -LO https://ziglang.org/download/0.13.0/zig-macos-aarch64-0.13.0.tar.xz
    tar xf zig-macos-aarch64-0.13.0.tar.xz
    sudo mv zig-macos-aarch64-0.13.0 /opt/zig
    export PATH="/opt/zig:$PATH"
    ```

=== "Windows"

    ```powershell
    # Via Scoop
    scoop install zig

    # Or download the zip from https://ziglang.org/download/
    # Extract and add the directory to your system PATH.
    ```

Verify the installation:

```bash
zig version
# Expected output: 0.13.0 (or later)
```

!!! warning "Version compatibility"

    ZigLlama uses language features stabilised in 0.13.  Earlier releases
    (0.11, 0.12) will fail to compile.  If your distribution ships an older
    Zig, use the upstream tarball instead of the package-manager version.

### Hardware Requirements

| Component | Minimum | Recommended | Notes |
|---|---|---|---|
| CPU | Any x86-64 or AArch64 | AVX2-capable x86-64 | SIMD kernels auto-detect; scalar fallback always available |
| RAM | 4 GB | 16 GB | Running tests requires < 1 GB; loading a 7B model in Q4 needs ~4 GB |
| Disk | 100 MB (source) | 10 GB+ | Space for downloaded GGUF model files |

### Optional Tools

| Tool | Purpose | Install |
|---|---|---|
| `git` | Clone the repository | System package manager |
| `python 3.9+` | Build MkDocs documentation locally | `apt install python3` / `brew install python` |
| `perf` (Linux) | CPU profiling of benchmarks | `apt install linux-tools-common` |

---

## Clone the Repository

```bash
git clone https://github.com/dipankar/zigllama.git
cd zigllama
```

The checkout is self-contained.  There are no Git submodules, no vendored C
libraries, and no code-generation steps.

---

## Verify the Installation

Run the full test suite to confirm everything works:

```bash
zig build test
```

!!! tip "Expected output"

    On a clean checkout the command should exit with status 0 and report
    **285+ tests passing**.  The first build compiles the entire project from
    scratch; subsequent runs use the Zig build cache and finish in seconds.

You can also run tests for individual layers to isolate any problems:

```bash
# Foundation layer only (tensors, memory management)
zig build test-foundation

# Linear algebra layer only (SIMD ops, quantisation)
zig build test-linear-algebra
```

If all tests pass, your installation is complete.  Jump to the
[Quick Start](quickstart.md) to run your first inference demo.

---

## Optional: BLAS Libraries

ZigLlama includes a pure-Zig matrix multiplication kernel that works
everywhere.  For maximum throughput on large models, you can optionally link
against an optimised BLAS library.

!!! info "When do you need BLAS?"

    For educational exploration and running the test suite, the built-in
    kernels are more than sufficient.  BLAS integration matters when you are
    benchmarking against llama.cpp or loading full-size models (7B+) and
    need production-grade matrix throughput.

### OpenBLAS

=== "Debian / Ubuntu"

    ```bash
    sudo apt install libopenblas-dev
    ```

=== "Fedora / RHEL"

    ```bash
    sudo dnf install openblas-devel
    ```

=== "macOS (Homebrew)"

    ```bash
    brew install openblas
    export LIBRARY_PATH="/opt/homebrew/opt/openblas/lib:$LIBRARY_PATH"
    ```

### Intel MKL

Intel oneAPI Math Kernel Library provides the fastest BLAS on Intel CPUs.

```bash
# Install via the Intel package repository (Debian/Ubuntu)
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] \
  https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt update && sudo apt install intel-oneapi-mkl-devel
source /opt/intel/oneapi/setvars.sh
```

### Apple Accelerate

On macOS, the Accelerate framework is pre-installed.  No additional
configuration is needed; ZigLlama's BLAS integration layer detects it
automatically at build time.

```bash
# Verify Accelerate is available
xcrun --show-sdk-path
# Should print something like /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
```

---

## Platform Support Matrix

| Platform | Architecture | Status | SIMD | BLAS |
|---|---|---|---|---|
| Linux | x86-64 | Fully supported | AVX, AVX2 | OpenBLAS, MKL |
| Linux | AArch64 | Fully supported | NEON | OpenBLAS |
| macOS | Apple Silicon | Fully supported | NEON | Accelerate |
| macOS | x86-64 (Intel) | Fully supported | AVX, AVX2 | OpenBLAS, MKL, Accelerate |
| Windows | x86-64 | Supported (community) | AVX, AVX2 | OpenBLAS |
| FreeBSD | x86-64 | Expected to work | AVX, AVX2 | OpenBLAS |

!!! tip "Cross-compilation"

    Zig's cross-compilation support means you can build a Linux AArch64 binary
    on your x86-64 workstation:

    ```bash
    zig build -Dtarget=aarch64-linux-gnu
    ```

    See [Building from Source](building.md) for the full cross-compilation
    guide.

---

## Troubleshooting

### `error: expected expression` or parser failures

**Cause**: Zig version too old.

**Fix**: Upgrade to Zig 0.13.0 or later.  Check with `zig version`.

---

### `error: FileNotFound` when running tests

**Cause**: Working directory is not the repository root.

**Fix**: Make sure you `cd zigllama` before running `zig build test`.

---

### BLAS not detected at build time

**Cause**: The BLAS shared library is not on the default library search path.

**Fix**: Set `LIBRARY_PATH` (compile time) and `LD_LIBRARY_PATH` (run time) to
the directory containing `libopenblas.so` or `libmkl_rt.so`.

```bash
export LIBRARY_PATH=/opt/openblas/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/openblas/lib:$LD_LIBRARY_PATH
```

---

### Out-of-memory during large model loading

**Cause**: The model file exceeds available RAM.

**Fix**: Use a quantised model (Q4_K or smaller) or enable memory mapping.
ZigLlama's GGUF loader uses `mmap` by default on Linux and macOS, so the
operating system can page in only the portions of the model file that are
actively needed.

---

### Slow SIMD performance

**Cause**: The CPU does not support AVX2, or the build was compiled in Debug
mode.

**Fix**: Verify CPU support with:

```bash
# Linux
grep -o 'avx2' /proc/cpuinfo | head -1

# macOS
sysctl -a | grep machdep.cpu.features | grep AVX2
```

If AVX2 is available, rebuild in ReleaseFast mode:

```bash
zig build test -Doptimize=ReleaseFast
```

See [Building from Source](building.md) for guidance on build modes.

---

### Tests pass but examples fail to run

**Cause**: Examples are standalone Zig files that import from `src/`.  They
must be run from the repository root.

**Fix**:

```bash
cd /path/to/zigllama
zig run examples/simple_demo.zig
```

---

## Next Steps

- [Quick Start Guide](quickstart.md) -- run your first inference.
- [Building from Source](building.md) -- customise the build for your platform.
- [Project Structure](project-structure.md) -- understand the codebase layout.
