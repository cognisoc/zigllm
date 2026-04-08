---
title: "CLI Interface"
description: "Command-line interface for launching and configuring the ZigLlama HTTP server."
---

# CLI Interface

The ZigLlama CLI (`src/server/cli.zig`) is the primary entry point for running
the inference server.  It parses command-line arguments, prints a startup
banner, and delegates to `http_server.runServer`.

---

## Command-Line Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--help`, `-h` | -- | -- | Print usage information and exit. |
| `--host <host>` | `string` | `127.0.0.1` | Network interface to bind. Use `0.0.0.0` for all interfaces. |
| `--port <port>` | `u16` | `8080` | TCP port number. |
| `--api-key <key>` | `string` | *(none)* | Require this Bearer token on every request. |
| `--model <path>` | `string` | *(none)* | Path to a GGUF model file. Without this flag the server returns demo responses. |
| `--max-requests <n>` | `u32` | `10` | Maximum number of concurrent requests. |
| `--timeout <ms>` | `u32` | `60000` | Per-request timeout in milliseconds. |
| `--no-cors` | `bool` | `false` | Disable CORS headers. |
| `--no-log` | `bool` | `false` | Suppress per-request log lines. |

!!! info "Argument parsing"
    Arguments are parsed with a simple linear scan.  Flags that take a value
    consume the **next** positional argument (`args[i + 1]`).  Unknown flags
    are silently ignored.

---

## Usage Examples

### Minimal launch (demo mode)

```bash
zig build run-server
```

The server starts on `127.0.0.1:8080` with no model loaded.  All inference
endpoints return placeholder responses -- useful for verifying that the HTTP
layer and client integration work correctly before downloading a multi-gigabyte
model.

### Custom host and port

```bash
zig build run-server -- --host 0.0.0.0 --port 3000
```

### Loading a real model with authentication

```bash
zig build run-server -- \
    --model ./models/llama-7b-q4_k_m.gguf \
    --api-key sk-dev-key-12345
```

### Production-style configuration

```bash
zig build run-server -- \
    --host 0.0.0.0 \
    --port 8080 \
    --model ./models/llama-7b-q4_k_m.gguf \
    --api-key sk-prod-$(openssl rand -hex 16) \
    --max-requests 50 \
    --timeout 30000
```

!!! tip "Timeout tuning"
    Set `--timeout` to at least `max_tokens * avg_ms_per_token`.  For a 7B
    model generating 2048 tokens at ~10 ms/token, 30 000 ms provides adequate
    headroom.

---

## Configuration: CLI vs Environment Variables

The CLI currently reads all configuration from command-line flags.  A common
pattern for containerised deployments is to wrap the binary in a shell script
that maps environment variables to flags:

```bash
#!/usr/bin/env bash
exec zigllama-server \
    --host "${ZIGLLAMA_HOST:-0.0.0.0}" \
    --port "${ZIGLLAMA_PORT:-8080}" \
    --model "${ZIGLLAMA_MODEL:?MODEL path required}" \
    --api-key "${ZIGLLAMA_API_KEY}" \
    --max-requests "${ZIGLLAMA_MAX_REQUESTS:-20}" \
    --timeout "${ZIGLLAMA_TIMEOUT:-60000}"
```

!!! warning "No `.env` file support"
    ZigLlama does not read `.env` files natively.  Use `direnv`, `dotenv`, or
    your container runtime's environment injection to bridge the gap.

---

## Interactive Mode

When launched without `--model`, the server enters **demo mode**.  In a future
release this will be extended to a full interactive REPL with the following
planned capabilities:

- Load and unload models at runtime (`/load`, `/unload`).
- Switch chat templates interactively (`/template chatml`).
- Inspect KV cache statistics (`/cache stats`).
- Run single-shot generation without an HTTP client (`/generate "prompt"`).

!!! info "Current status"
    Interactive mode is not yet implemented.  The server currently runs in a
    blocking accept loop.  Contributions are welcome -- see the
    [Contributing](../references/contributing.md) guide.

---

## Startup Output

On launch the CLI prints a banner followed by configuration details and
example `curl` commands tailored to the active host, port, and API key:

```
  ZIGLLAMA

     Educational LLaMA Implementation -- Production-Ready Server

  Starting ZigLlama Server
  Address: http://0.0.0.0:8080
  API Key: Required
  Max concurrent requests: 50
  Request timeout: 30000ms
  CORS enabled: true
  Request logging: true
  Model: ./models/llama-7b-q4_k_m.gguf

  Available Endpoints:
    GET  http://0.0.0.0:8080/health
    GET  http://0.0.0.0:8080/v1/models
    POST http://0.0.0.0:8080/v1/chat/completions
    POST http://0.0.0.0:8080/v1/completions

  Example Requests:

    # Health check
    curl http://0.0.0.0:8080/health

    # Chat completion
    curl -H "Authorization: Bearer sk-..." \
         -X POST http://0.0.0.0:8080/v1/chat/completions \
         -H "Content-Type: application/json" \
         -d '{"model":"llama-7b","messages":[{"role":"user","content":"Hello"}]}'
```

The dynamic `curl` snippets include the `Authorization` header only when an
API key is configured, reducing copy-paste friction during development.

---

## Error Handling

Invalid flag values (e.g., non-numeric `--port`) are caught at parse time and
logged via `std.log.err` before the process returns:

```
error: Invalid port number: abc. Error: error.InvalidCharacter
```

If `--model` points to a file that does not exist or is not a valid GGUF file,
the error surfaces when `ZigLlamaServer.loadModel` is called, after the server
has already printed its startup banner.  Future versions will validate the
model path during argument parsing to fail fast.

---

## Source Reference

| File | Purpose |
|------|---------|
| `src/server/cli.zig` | `main`, `printHelp`, `printBanner`, `printEndpoints` |
| `src/server/http_server.zig` | `runServer`, `ServerConfig` |
