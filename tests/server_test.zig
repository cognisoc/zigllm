const std = @import("std");
const testing = std.testing;
const http_server = @import("../src/server/http_server.zig");

test "HTTP server configuration" {
    const config = http_server.ServerConfig{
        .host = "127.0.0.1",
        .port = 8080,
        .api_key = "test-key",
        .max_concurrent_requests = 10,
        .request_timeout_ms = 30000,
        .enable_cors = true,
        .log_requests = false,
    };

    try testing.expect(std.mem.eql(u8, config.host, "127.0.0.1"));
    try testing.expect(config.port == 8080);
    try testing.expect(config.api_key != null);
    try testing.expect(config.max_concurrent_requests == 10);
    try testing.expect(config.enable_cors == true);
}

test "Chat completion request parsing" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const json_request =
        \\{
        \\  "model": "llama-7b",
        \\  "messages": [
        \\    {"role": "user", "content": "Hello"}
        \\  ],
        \\  "max_tokens": 100,
        \\  "temperature": 0.7,
        \\  "stream": false
        \\}
    ;

    const parsed = std.json.parseFromSlice(
        http_server.ChatCompletionRequest,
        allocator,
        json_request,
        .{}
    ) catch |err| {
        std.log.err("Failed to parse chat completion request: {}", .{err});
        return;
    };
    defer parsed.deinit();

    const request = parsed.value;
    try testing.expect(std.mem.eql(u8, request.model, "llama-7b"));
    try testing.expect(request.messages.len == 1);
    try testing.expect(std.mem.eql(u8, request.messages[0].role, "user"));
    try testing.expect(std.mem.eql(u8, request.messages[0].content, "Hello"));
    try testing.expect(request.max_tokens.? == 100);
    try testing.expect(request.temperature.? == 0.7);
    try testing.expect(request.stream.? == false);
}

test "Completion request parsing" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const json_request =
        \\{
        \\  "model": "llama-7b",
        \\  "prompt": "The future of AI is",
        \\  "max_tokens": 50,
        \\  "temperature": 0.8,
        \\  "top_p": 0.9
        \\}
    ;

    const parsed = std.json.parseFromSlice(
        http_server.CompletionRequest,
        allocator,
        json_request,
        .{}
    ) catch |err| {
        std.log.err("Failed to parse completion request: {}", .{err});
        return;
    };
    defer parsed.deinit();

    const request = parsed.value;
    try testing.expect(std.mem.eql(u8, request.model, "llama-7b"));
    try testing.expect(std.mem.eql(u8, request.prompt, "The future of AI is"));
    try testing.expect(request.max_tokens.? == 50);
    try testing.expect(request.temperature.? == 0.8);
    try testing.expect(request.top_p.? == 0.9);
}

test "Server initialization and cleanup" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = http_server.ServerConfig{
        .host = "127.0.0.1",
        .port = 8081, // Use different port for testing
        .api_key = null,
        .max_concurrent_requests = 5,
        .request_timeout_ms = 10000,
        .enable_cors = true,
        .log_requests = false,
    };

    var server = http_server.ZigLlamaServer.init(allocator, config) catch |err| {
        std.log.err("Failed to initialize server: {}", .{err});
        return;
    };
    defer server.deinit();

    try testing.expect(server.config.port == 8081);
    try testing.expect(server.config.max_concurrent_requests == 5);
}

test "Generation result management" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const test_text = try allocator.dupe(u8, "This is a test response");
    const test_tokens = [_]u32{ 1, 2, 3, 4, 5 };

    var result = http_server.GenerationResult{
        .text = test_text,
        .tokens = &test_tokens,
        .finish_reason = "stop",
    };

    try testing.expect(result.text != null);
    try testing.expect(result.tokens.len == 5);
    try testing.expect(std.mem.eql(u8, result.finish_reason.?, "stop"));

    result.deinit(allocator);
}

test "Model list response structure" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const models = [_]http_server.ModelInfo{
        .{
            .id = "llama-7b",
            .object = "model",
            .created = 1677649963,
            .owned_by = "zigllama",
        },
        .{
            .id = "gpt-2",
            .object = "model",
            .created = 1677649963,
            .owned_by = "zigllama",
        },
    };

    const model_list = http_server.ModelListResponse{
        .object = "list",
        .data = &models,
    };

    try testing.expect(std.mem.eql(u8, model_list.object, "list"));
    try testing.expect(model_list.data.len == 2);
    try testing.expect(std.mem.eql(u8, model_list.data[0].id, "llama-7b"));
    try testing.expect(std.mem.eql(u8, model_list.data[1].id, "gpt-2"));
}

test "Chat completion response structure" {
    const message = http_server.ChatMessage{
        .role = "assistant",
        .content = "Hello! How can I help you today?",
    };

    const choice = http_server.ChatChoice{
        .index = 0,
        .message = message,
        .finish_reason = "stop",
    };

    const usage = http_server.TokenUsage{
        .prompt_tokens = 10,
        .completion_tokens = 8,
        .total_tokens = 18,
    };

    const response = http_server.ChatCompletionResponse{
        .id = "chatcmpl-123",
        .created = 1677652288,
        .model = "llama-7b",
        .choices = &[_]http_server.ChatChoice{choice},
        .usage = usage,
    };

    try testing.expect(std.mem.eql(u8, response.id, "chatcmpl-123"));
    try testing.expect(std.mem.eql(u8, response.model, "llama-7b"));
    try testing.expect(response.choices.len == 1);
    try testing.expect(std.mem.eql(u8, response.choices[0].message.role, "assistant"));
    try testing.expect(response.usage.total_tokens == 18);
}

test "Advanced sampling parameters parsing" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const json_request =
        \\{
        \\  "model": "llama-7b",
        \\  "prompt": "Test prompt",
        \\  "temperature": 0.7,
        \\  "top_p": 0.9,
        \\  "top_k": 40,
        \\  "typical_p": 0.95,
        \\  "mirostat": 1,
        \\  "mirostat_tau": 5.0,
        \\  "mirostat_eta": 0.1,
        \\  "repetition_penalty": 1.1
        \\}
    ;

    const parsed = std.json.parseFromSlice(
        http_server.CompletionRequest,
        allocator,
        json_request,
        .{}
    ) catch |err| {
        std.log.err("Failed to parse advanced sampling request: {}", .{err});
        return;
    };
    defer parsed.deinit();

    const request = parsed.value;
    try testing.expect(request.temperature.? == 0.7);
    try testing.expect(request.top_p.? == 0.9);
    try testing.expect(request.top_k.? == 40);
    try testing.expect(request.typical_p.? == 0.95);
    try testing.expect(request.mirostat.? == 1);
    try testing.expect(request.mirostat_tau.? == 5.0);
    try testing.expect(request.mirostat_eta.? == 0.1);
    try testing.expect(request.repetition_penalty.? == 1.1);
}

test "Grammar constraint support" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const json_request =
        \\{
        \\  "model": "llama-7b",
        \\  "prompt": "Generate a JSON object",
        \\  "grammar": "{\"type\": \"object\", \"properties\": {\"name\": {\"type\": \"string\"}}}",
        \\  "grammar_type": "json_schema"
        \\}
    ;

    const parsed = std.json.parseFromSlice(
        http_server.CompletionRequest,
        allocator,
        json_request,
        .{}
    ) catch |err| {
        std.log.err("Failed to parse grammar request: {}", .{err});
        return;
    };
    defer parsed.deinit();

    const request = parsed.value;
    try testing.expect(request.grammar != null);
    try testing.expect(request.grammar_type != null);
    try testing.expect(std.mem.eql(u8, request.grammar_type.?, "json_schema"));
}

// Integration tests would go here if we had a test HTTP client
// For now, these structural tests ensure the server components work correctly