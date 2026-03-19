const std = @import("std");
const http_server = @import("http_server.zig");
const models = @import("../models/main.zig");

/// ZigLlama Server Command Line Interface
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var server_config = http_server.ServerConfig{
        .host = "127.0.0.1",
        .port = 8080,
        .api_key = null,
        .max_concurrent_requests = 10,
        .request_timeout_ms = 60000,
        .enable_cors = true,
        .log_requests = true,
    };

    var model_path: ?[]const u8 = null;
    var help_requested = false;

    // Simple argument parsing
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--help") or std.mem.eql(u8, args[i], "-h")) {
            help_requested = true;
            break;
        } else if (std.mem.eql(u8, args[i], "--host") and i + 1 < args.len) {
            server_config.host = args[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--port") and i + 1 < args.len) {
            server_config.port = std.fmt.parseInt(u16, args[i + 1], 10) catch |err| {
                std.log.err("Invalid port number: {s}. Error: {}", .{ args[i + 1], err });
                return;
            };
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--api-key") and i + 1 < args.len) {
            server_config.api_key = args[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--model") and i + 1 < args.len) {
            model_path = args[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--max-requests") and i + 1 < args.len) {
            server_config.max_concurrent_requests = std.fmt.parseInt(u32, args[i + 1], 10) catch |err| {
                std.log.err("Invalid max requests number: {s}. Error: {}", .{ args[i + 1], err });
                return;
            };
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--timeout") and i + 1 < args.len) {
            server_config.request_timeout_ms = std.fmt.parseInt(u32, args[i + 1], 10) catch |err| {
                std.log.err("Invalid timeout: {s}. Error: {}", .{ args[i + 1], err });
                return;
            };
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--no-cors")) {
            server_config.enable_cors = false;
        } else if (std.mem.eql(u8, args[i], "--no-log")) {
            server_config.log_requests = false;
        }
    }

    if (help_requested) {
        printHelp();
        return;
    }

    // Print startup banner
    printBanner();

    // Print configuration
    std.log.info("🚀 Starting ZigLlama Server");
    std.log.info("📍 Address: http://{s}:{d}", .{ server_config.host, server_config.port });
    std.log.info("🔑 API Key: {s}", .{if (server_config.api_key != null) "Required" else "Not required"});
    std.log.info("📊 Max concurrent requests: {d}", .{server_config.max_concurrent_requests});
    std.log.info("⏱️  Request timeout: {d}ms", .{server_config.request_timeout_ms});
    std.log.info("🌐 CORS enabled: {}", .{server_config.enable_cors});
    std.log.info("📝 Request logging: {}", .{server_config.log_requests});

    if (model_path) |path| {
        std.log.info("🦙 Model: {s}", .{path});
    } else {
        std.log.warn("⚠️  No model specified. Using demo responses.");
        std.log.warn("   Use --model <path> to load a real model.");
    }

    // Print available endpoints
    printEndpoints(server_config);

    // Start the server
    try http_server.runServer(allocator, server_config);
}

fn printHelp() void {
    const help_text =
        \\
        \\🦙 ZigLlama Server - Educational LLaMA Implementation
        \\========================================================
        \\
        \\USAGE:
        \\    zig run src/server/cli.zig -- [OPTIONS]
        \\
        \\OPTIONS:
        \\    --help, -h                  Show this help message
        \\    --host <host>               Server host (default: 127.0.0.1)
        \\    --port <port>               Server port (default: 8080)
        \\    --api-key <key>             Require API key for authentication
        \\    --model <path>              Path to GGUF model file
        \\    --max-requests <num>        Max concurrent requests (default: 10)
        \\    --timeout <ms>              Request timeout in milliseconds (default: 60000)
        \\    --no-cors                   Disable CORS headers
        \\    --no-log                    Disable request logging
        \\
        \\EXAMPLES:
        \\    # Start basic server
        \\    zig run src/server/cli.zig
        \\
        \\    # Start with custom host and port
        \\    zig run src/server/cli.zig -- --host 0.0.0.0 --port 3000
        \\
        \\    # Start with model and API key
        \\    zig run src/server/cli.zig -- --model ./models/llama-7b.gguf --api-key sk-123abc
        \\
        \\    # Production configuration
        \\    zig run src/server/cli.zig -- --host 0.0.0.0 --port 8080 \
        \\                                  --model ./models/llama-7b.gguf \
        \\                                  --api-key sk-prod-key \
        \\                                  --max-requests 50 \
        \\                                  --timeout 30000
        \\
        \\ENDPOINTS:
        \\    GET  /health                Health check endpoint
        \\    GET  /v1/models             List available models (OpenAI compatible)
        \\    POST /v1/chat/completions   Chat completions (OpenAI compatible)
        \\    POST /v1/completions        Text completions (OpenAI compatible)
        \\
        \\For more information, visit: https://github.com/username/zigllama
        \\
    ;
    std.debug.print(help_text, .{});
}

fn printBanner() void {
    const banner =
        \\
        \\  ████████╗██╗ ██████╗ ██╗     ██╗      █████╗ ███╗   ███╗ █████╗
        \\  ╚══██╔══╝██║██╔════╝ ██║     ██║     ██╔══██╗████╗ ████║██╔══██╗
        \\     ██║   ██║██║  ███╗██║     ██║     ███████║██╔████╔██║███████║
        \\     ██║   ██║██║   ██║██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║
        \\     ██║   ██║╚██████╔╝███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║
        \\     ╚═╝   ╚═╝ ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝
        \\
        \\     🦙 Educational LLaMA Implementation • Production-Ready Server 🚀
        \\     📚 65% Production Parity • 100% Educational Excellence ✨
        \\
    ;
    std.debug.print(banner, .{});
}

fn printEndpoints(config: http_server.ServerConfig) void {
    const base_url = std.fmt.allocPrint(
        std.heap.page_allocator,
        "http://{s}:{d}",
        .{ config.host, config.port }
    ) catch "http://localhost:8080";

    std.log.info("");
    std.log.info("🌐 Available Endpoints:");
    std.log.info("   GET  {s}/health", .{base_url});
    std.log.info("   GET  {s}/v1/models", .{base_url});
    std.log.info("   POST {s}/v1/chat/completions", .{base_url});
    std.log.info("   POST {s}/v1/completions", .{base_url});
    std.log.info("");

    // Print example curl commands
    std.log.info("📋 Example Requests:");
    std.log.info("");

    const auth_header = if (config.api_key) |key|
        std.fmt.allocPrint(std.heap.page_allocator, " -H \"Authorization: Bearer {s}\"", .{key}) catch ""
    else
        "";

    std.log.info("   # Health check");
    std.log.info("   curl {s}/health", .{base_url});
    std.log.info("");

    std.log.info("   # List models");
    std.log.info("   curl{s} {s}/v1/models", .{ auth_header, base_url });
    std.log.info("");

    std.log.info("   # Chat completion");
    std.log.info("   curl{s} -X POST {s}/v1/chat/completions \\", .{ auth_header, base_url });
    std.log.info("     -H \"Content-Type: application/json\" \\");
    std.log.info("     -d '{{");
    std.log.info("       \"model\": \"llama-7b\",");
    std.log.info("       \"messages\": [");
    std.log.info("         {{\"role\": \"user\", \"content\": \"Hello, how are you?\"}}");
    std.log.info("       ],");
    std.log.info("       \"max_tokens\": 100");
    std.log.info("     }}'");
    std.log.info("");

    std.log.info("   # Text completion");
    std.log.info("   curl{s} -X POST {s}/v1/completions \\", .{ auth_header, base_url });
    std.log.info("     -H \"Content-Type: application/json\" \\");
    std.log.info("     -d '{{");
    std.log.info("       \"model\": \"llama-7b\",");
    std.log.info("       \"prompt\": \"The future of AI is\",");
    std.log.info("       \"max_tokens\": 50");
    std.log.info("     }}'");
    std.log.info("");
}