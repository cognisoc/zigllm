const std = @import("std");
const http = std.http;
const json = std.json;
const Allocator = std.mem.Allocator;

// Import our inference and model components
const models = @import("../models/llama.zig");
const inference = @import("../inference/generation.zig");
const advanced_sampling = @import("../inference/advanced_sampling.zig");
const grammar_constraints = @import("../inference/grammar_constraints.zig");
const chat_templates = @import("../models/chat_templates.zig");

/// HTTP server configuration
pub const ServerConfig = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    max_connections: u32 = 100,
    timeout_seconds: u32 = 30,
    cors_enabled: bool = true,
    api_key: ?[]const u8 = null,  // Optional API key authentication
    max_tokens: u32 = 2048,       // Default max tokens per request
    enable_streaming: bool = true, // Enable streaming responses
};

/// OpenAI-compatible request structures
pub const ChatCompletionRequest = struct {
    model: []const u8,
    messages: []ChatMessage,
    max_tokens: ?u32 = null,
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    top_k: ?u32 = null,
    frequency_penalty: ?f32 = null,
    presence_penalty: ?f32 = null,
    stop: ?[][]const u8 = null,
    stream: ?bool = null,
    // ZigLlama extensions
    sampling_strategy: ?[]const u8 = null, // "mirostat", "typical", etc.
    grammar: ?[]const u8 = null,          // JSON schema or regex pattern
    mirostat_tau: ?f32 = null,
    typical_mass: ?f32 = null,
};

pub const ChatMessage = struct {
    role: []const u8, // "system", "user", "assistant"
    content: []const u8,
    name: ?[]const u8 = null,
};

pub const CompletionRequest = struct {
    model: []const u8,
    prompt: []const u8,
    max_tokens: ?u32 = null,
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    top_k: ?u32 = null,
    stop: ?[][]const u8 = null,
    stream: ?bool = null,
    // ZigLlama extensions
    sampling_strategy: ?[]const u8 = null,
    grammar: ?[]const u8 = null,
};

/// OpenAI-compatible response structures
pub const ChatCompletionResponse = struct {
    id: []const u8,
    object: []const u8 = "chat.completion",
    created: u64,
    model: []const u8,
    choices: []ChatChoice,
    usage: TokenUsage,
};

pub const ChatChoice = struct {
    index: u32,
    message: ChatMessage,
    finish_reason: ?[]const u8 = null, // "stop", "length", "content_filter"
};

pub const CompletionResponse = struct {
    id: []const u8,
    object: []const u8 = "text_completion",
    created: u64,
    model: []const u8,
    choices: []CompletionChoice,
    usage: TokenUsage,
};

pub const CompletionChoice = struct {
    text: []const u8,
    index: u32,
    finish_reason: ?[]const u8 = null,
};

pub const TokenUsage = struct {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
};

pub const StreamChunk = struct {
    id: []const u8,
    object: []const u8 = "chat.completion.chunk",
    created: u64,
    model: []const u8,
    choices: []StreamChoice,
};

pub const StreamChoice = struct {
    index: u32,
    delta: ChatMessage,
    finish_reason: ?[]const u8 = null,
};

/// Model information for /v1/models endpoint
pub const ModelInfo = struct {
    id: []const u8,
    object: []const u8 = "model",
    created: u64,
    owned_by: []const u8,
    permission: []ModelPermission = &[_]ModelPermission{},
};

pub const ModelPermission = struct {
    id: []const u8,
    object: []const u8 = "model_permission",
    created: u64,
    allow_create_engine: bool = false,
    allow_sampling: bool = true,
    allow_logprobs: bool = true,
    allow_search_indices: bool = false,
    allow_view: bool = true,
    allow_fine_tuning: bool = false,
    organization: []const u8 = "*",
    group: ?[]const u8 = null,
    is_blocking: bool = false,
};

/// HTTP server with OpenAI-compatible API
pub const ZigLlamaServer = struct {
    allocator: Allocator,
    config: ServerConfig,
    server: http.Server,
    model: ?*models.LLaMAModel,
    tokenizer: ?*anyopaque, // Generic tokenizer interface
    request_counter: std.atomic.Value(u64),
    template_manager: chat_templates.ChatTemplateManager,
    default_template: chat_templates.TemplateType,

    const Self = @This();

    pub fn init(allocator: Allocator, config: ServerConfig) !Self {
        const server = http.Server.init(allocator, .{ .reuse_address = true });

        var template_manager = chat_templates.ChatTemplateManager.init(allocator);

        // Pre-load common templates
        try template_manager.loadTemplate(.Llama2);
        try template_manager.loadTemplate(.Llama3);
        try template_manager.loadTemplate(.ChatML);
        try template_manager.loadTemplate(.Mistral);

        return Self{
            .allocator = allocator,
            .config = config,
            .server = server,
            .model = null,
            .tokenizer = null,
            .request_counter = std.atomic.Value(u64).init(0),
            .template_manager = template_manager,
            .default_template = .ChatML, // Default to ChatML for broad compatibility
        };
    }

    pub fn deinit(self: *Self) void {
        self.template_manager.deinit();
        self.server.deinit();
    }

    /// Load model for inference
    pub fn loadModel(self: *Self, model: *models.LLaMAModel, tokenizer: *anyopaque, model_name: ?[]const u8) !void {
        self.model = model;
        self.tokenizer = tokenizer;

        // Auto-detect appropriate chat template based on model name
        if (model_name) |name| {
            const detected_template = chat_templates.ChatTemplateManager.detectTemplate(name);
            self.default_template = detected_template;

            // Load the detected template if not already loaded
            self.template_manager.loadTemplate(detected_template) catch |err| {
                std.log.warn("Failed to load template {s}: {}", .{ detected_template.toString(), err });
            };

            std.log.info("Auto-detected chat template: {s} for model: {s}", .{ detected_template.toString(), name });
        }
    }

    /// Start the HTTP server
    pub fn start(self: *Self) !void {
        const address = try std.net.Address.resolveIp(self.config.host, self.config.port);
        try self.server.listen(address);

        std.log.info("🚀 ZigLlama HTTP server starting on {}:{}", .{ self.config.host, self.config.port });
        std.log.info("📡 OpenAI-compatible API endpoints:");
        std.log.info("   GET  /v1/models");
        std.log.info("   POST /v1/chat/completions");
        std.log.info("   POST /v1/completions");
        std.log.info("   GET  /health");

        while (true) {
            var response = try self.server.accept(.{ .allocator = self.allocator });
            defer response.deinit();

            // Handle request in a separate function for clarity
            self.handleRequest(&response) catch |err| {
                std.log.err("Error handling request: {}", .{err});
                self.sendErrorResponse(&response, 500, "Internal server error") catch {};
            };
        }
    }

    /// Handle incoming HTTP requests
    fn handleRequest(self: *Self, response: *http.Server.Response) !void {
        try response.wait();

        // CORS headers if enabled
        if (self.config.cors_enabled) {
            try response.headers.append("access-control-allow-origin", "*");
            try response.headers.append("access-control-allow-methods", "GET, POST, OPTIONS");
            try response.headers.append("access-control-allow-headers", "Content-Type, Authorization");
        }

        // Handle OPTIONS preflight requests
        if (std.mem.eql(u8, response.request.method, "OPTIONS")) {
            try response.headers.append("content-length", "0");
            try response.do();
            return;
        }

        // API key authentication if configured
        if (self.config.api_key) |expected_key| {
            const auth_header = response.request.headers.getFirstValue("authorization") orelse "";
            const bearer_prefix = "Bearer ";

            if (!std.mem.startsWith(u8, auth_header, bearer_prefix)) {
                return self.sendErrorResponse(response, 401, "Missing or invalid authorization header");
            }

            const provided_key = auth_header[bearer_prefix.len..];
            if (!std.mem.eql(u8, provided_key, expected_key)) {
                return self.sendErrorResponse(response, 401, "Invalid API key");
            }
        }

        // Route requests
        const path = response.request.target;
        const method = response.request.method;

        std.log.info("{s} {s}", .{ method, path });

        if (std.mem.eql(u8, method, "GET")) {
            if (std.mem.eql(u8, path, "/health")) {
                return try self.handleHealth(response);
            } else if (std.mem.eql(u8, path, "/v1/models")) {
                return try self.handleModels(response);
            }
        } else if (std.mem.eql(u8, method, "POST")) {
            if (std.mem.eql(u8, path, "/v1/chat/completions")) {
                return try self.handleChatCompletions(response);
            } else if (std.mem.eql(u8, path, "/v1/completions")) {
                return try self.handleCompletions(response);
            }
        }

        // 404 for unknown endpoints
        return self.sendErrorResponse(response, 404, "Endpoint not found");
    }

    /// Health check endpoint
    fn handleHealth(self: *Self, response: *http.Server.Response) !void {
        _ = self;

        const health_response =
            \\{"status":"healthy","service":"zigllama","version":"1.0.0"}
        ;

        try response.headers.append("content-type", "application/json");
        try response.headers.append("content-length", try std.fmt.allocPrint(self.allocator, "{d}", .{health_response.len}));
        try response.do();
        try response.writeAll(health_response);
        try response.finish();
    }

    /// List available models endpoint
    fn handleModels(self: *Self, response: *http.Server.Response) !void {
        const models_list =
            \\{"object":"list","data":[
            \\{"id":"llama-7b","object":"model","created":1677610602,"owned_by":"zigllama"},
            \\{"id":"llama-13b","object":"model","created":1677610602,"owned_by":"zigllama"},
            \\{"id":"gpt2-124m","object":"model","created":1677610602,"owned_by":"zigllama"},
            \\{"id":"mistral-7b","object":"model","created":1677610602,"owned_by":"zigllama"}
            \\]}
        ;

        try response.headers.append("content-type", "application/json");
        const content_length = try std.fmt.allocPrint(self.allocator, "{d}", .{models_list.len});
        defer self.allocator.free(content_length);
        try response.headers.append("content-length", content_length);
        try response.do();
        try response.writeAll(models_list);
        try response.finish();
    }

    /// Chat completions endpoint (OpenAI compatible)
    fn handleChatCompletions(self: *Self, response: *http.Server.Response) !void {
        // Read request body
        const body = try response.reader().readAllAlloc(self.allocator, 1024 * 1024); // 1MB max
        defer self.allocator.free(body);

        // Parse JSON request
        const parsed = json.parseFromSlice(ChatCompletionRequest, self.allocator, body) catch |err| {
            std.log.err("Failed to parse chat completion request: {}", .{err});
            return self.sendErrorResponse(response, 400, "Invalid JSON request");
        };
        defer parsed.deinit();

        const request = parsed.value;

        // Validate model is loaded
        if (self.model == null) {
            return self.sendErrorResponse(response, 503, "No model loaded");
        }

        // Convert chat messages to prompt
        const prompt = try self.convertChatToPrompt(request.messages);
        defer self.allocator.free(prompt);

        // Generate response
        const generation_result = try self.generateText(prompt, request);
        defer generation_result.deinit(self.allocator);

        // Handle streaming vs non-streaming
        if (request.stream orelse false) {
            try self.sendStreamingChatResponse(response, request, generation_result);
        } else {
            try self.sendChatResponse(response, request, generation_result);
        }
    }

    /// Text completions endpoint
    fn handleCompletions(self: *Self, response: *http.Server.Response) !void {
        const body = try response.reader().readAllAlloc(self.allocator, 1024 * 1024);
        defer self.allocator.free(body);

        const parsed = json.parseFromSlice(CompletionRequest, self.allocator, body) catch |err| {
            std.log.err("Failed to parse completion request: {}", .{err});
            return self.sendErrorResponse(response, 400, "Invalid JSON request");
        };
        defer parsed.deinit();

        const request = parsed.value;

        if (self.model == null) {
            return self.sendErrorResponse(response, 503, "No model loaded");
        }

        // Generate response
        const generation_result = try self.generateTextFromPrompt(request.prompt, request);
        defer generation_result.deinit(self.allocator);

        // Send response
        if (request.stream orelse false) {
            try self.sendStreamingCompletionResponse(response, request, generation_result);
        } else {
            try self.sendCompletionResponse(response, request, generation_result);
        }
    }

    /// Convert chat messages to a single prompt string
    fn convertChatToPrompt(self: *Self, messages: []const ChatMessage) ![]u8 {
        // Convert OpenAI ChatMessage format to our internal format
        var template_messages = try self.allocator.alloc(chat_templates.ChatMessage, messages.len);
        defer self.allocator.free(template_messages);

        for (messages, 0..) |message, i| {
            template_messages[i] = chat_templates.ChatMessage{
                .role = message.role,
                .content = message.content,
                .name = null,
            };
        }

        // Validate conversation format
        if (!chat_templates.ChatTemplateUtils.validateConversation(template_messages)) {
            std.log.warn("Invalid conversation format detected");
        }

        // Apply template based on default or detected model type
        const template_result = self.template_manager.applyTemplate(
            self.default_template,
            template_messages
        ) catch |err| switch (err) {
            error.TemplateNotFound => {
                std.log.warn("Template {s} not found, falling back to ChatML", .{self.default_template.toString()});

                // Fallback to ChatML template
                try self.template_manager.loadTemplate(.ChatML);
                return try self.template_manager.applyTemplate(.ChatML, template_messages);
            },
            else => return err,
        };

        std.log.debug("Applied template {s} to conversation", .{self.default_template.toString()});
        return template_result;
    }

    /// Generate text using configured sampling strategy
    fn generateText(self: *Self, prompt: []const u8, request: ChatCompletionRequest) !GenerationResult {
        // Create sampling configuration based on request
        var sampler = advanced_sampling.SamplingCoordinator.init(self.allocator, null);

        // Use advanced sampling if specified
        if (request.sampling_strategy) |strategy| {
            if (std.mem.eql(u8, strategy, "mirostat")) {
                const config = advanced_sampling.MirostatConfig{
                    .version = .V2,
                    .tau = request.mirostat_tau orelse 3.0,
                    .eta = 0.1,
                    .epsilon = 0.01,
                    .max_iterations = 10,
                };
                // Would use mirostat sampling here
                return try self.generateWithMirostat(prompt, config);
            } else if (std.mem.eql(u8, strategy, "typical")) {
                const config = advanced_sampling.TypicalConfig{
                    .mass = request.typical_mass orelse 0.9,
                    .min_tokens = 3,
                };
                return try self.generateWithTypical(prompt, config);
            }
        }

        // Use grammar constraints if specified
        if (request.grammar) |grammar_str| {
            return try self.generateWithGrammar(prompt, grammar_str);
        }

        // Default generation
        return try self.generateDefault(prompt, request);
    }

    fn generateTextFromPrompt(self: *Self, prompt: []const u8, request: CompletionRequest) !GenerationResult {
        // Similar to generateText but for completion requests
        return try self.generateDefault(prompt, @as(ChatCompletionRequest, .{
            .model = request.model,
            .messages = &[_]ChatMessage{},
            .max_tokens = request.max_tokens,
            .temperature = request.temperature,
            .top_p = request.top_p,
            .top_k = request.top_k,
        }));
    }

    // Generation method implementations
    fn generateWithMirostat(self: *Self, prompt: []const u8, config: advanced_sampling.MirostatConfig) !GenerationResult {
        _ = prompt;
        _ = config;
        // Placeholder - would implement actual Mirostat generation
        return GenerationResult{
            .text = try self.allocator.dupe(u8, "Generated with Mirostat sampling"),
            .tokens = &[_]u32{42},
            .finish_reason = "stop",
        };
    }

    fn generateWithTypical(self: *Self, prompt: []const u8, config: advanced_sampling.TypicalConfig) !GenerationResult {
        _ = prompt;
        _ = config;
        return GenerationResult{
            .text = try self.allocator.dupe(u8, "Generated with Typical sampling"),
            .tokens = &[_]u32{42},
            .finish_reason = "stop",
        };
    }

    fn generateWithGrammar(self: *Self, prompt: []const u8, grammar: []const u8) !GenerationResult {
        _ = prompt;
        _ = grammar;
        return GenerationResult{
            .text = try self.allocator.dupe(u8, "{\"response\": \"Generated with grammar constraints\"}"),
            .tokens = &[_]u32{42},
            .finish_reason = "stop",
        };
    }

    fn generateDefault(self: *Self, prompt: []const u8, request: ChatCompletionRequest) !GenerationResult {
        _ = prompt;
        _ = request;
        return GenerationResult{
            .text = try self.allocator.dupe(u8, "Hello! I'm ZigLlama, an educational AI assistant."),
            .tokens = &[_]u32{ 15496, 0, 40, 1, 76, 57014, 43, 84329, 11, 459, 16627, 15592, 18328 },
            .finish_reason = "stop",
        };
    }

    /// Send non-streaming chat response
    fn sendChatResponse(self: *Self, response: *http.Server.Response, request: ChatCompletionRequest, result: GenerationResult) !void {
        const request_id = self.generateRequestId();
        defer self.allocator.free(request_id);

        const chat_response = ChatCompletionResponse{
            .id = request_id,
            .created = @as(u64, @intCast(std.time.timestamp())),
            .model = request.model,
            .choices = &[_]ChatChoice{.{
                .index = 0,
                .message = ChatMessage{
                    .role = "assistant",
                    .content = result.text orelse "",
                },
                .finish_reason = result.finish_reason,
            }},
            .usage = TokenUsage{
                .prompt_tokens = 10, // Placeholder
                .completion_tokens = @as(u32, @intCast(result.tokens.len)),
                .total_tokens = 10 + @as(u32, @intCast(result.tokens.len)),
            },
        };

        const json_response = try json.stringifyAlloc(self.allocator, chat_response, .{});
        defer self.allocator.free(json_response);

        try response.headers.append("content-type", "application/json");
        const content_length = try std.fmt.allocPrint(self.allocator, "{d}", .{json_response.len});
        defer self.allocator.free(content_length);
        try response.headers.append("content-length", content_length);

        try response.do();
        try response.writeAll(json_response);
        try response.finish();
    }

    /// Send streaming chat response
    fn sendStreamingChatResponse(self: *Self, response: *http.Server.Response, request: ChatCompletionRequest, result: GenerationResult) !void {
        _ = request;
        _ = result;

        try response.headers.append("content-type", "text/event-stream");
        try response.headers.append("cache-control", "no-cache");
        try response.headers.append("connection", "keep-alive");

        try response.do();

        // Send chunks (simplified streaming implementation)
        const chunk_data = "data: {\"id\":\"chatcmpl-123\",\"object\":\"chat.completion.chunk\",\"created\":1677652288,\"model\":\"llama-7b\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n";
        try response.writeAll(chunk_data);

        const final_chunk = "data: [DONE]\n\n";
        try response.writeAll(final_chunk);

        try response.finish();
    }

    /// Send completion response
    fn sendCompletionResponse(self: *Self, response: *http.Server.Response, request: CompletionRequest, result: GenerationResult) !void {
        const request_id = self.generateRequestId();
        defer self.allocator.free(request_id);

        const completion_response = CompletionResponse{
            .id = request_id,
            .created = @as(u64, @intCast(std.time.timestamp())),
            .model = request.model,
            .choices = &[_]CompletionChoice{.{
                .text = result.text orelse "",
                .index = 0,
                .finish_reason = result.finish_reason,
            }},
            .usage = TokenUsage{
                .prompt_tokens = 10,
                .completion_tokens = @as(u32, @intCast(result.tokens.len)),
                .total_tokens = 10 + @as(u32, @intCast(result.tokens.len)),
            },
        };

        const json_response = try json.stringifyAlloc(self.allocator, completion_response, .{});
        defer self.allocator.free(json_response);

        try response.headers.append("content-type", "application/json");
        const content_length = try std.fmt.allocPrint(self.allocator, "{d}", .{json_response.len});
        defer self.allocator.free(content_length);
        try response.headers.append("content-length", content_length);

        try response.do();
        try response.writeAll(json_response);
        try response.finish();
    }

    fn sendStreamingCompletionResponse(self: *Self, response: *http.Server.Response, request: CompletionRequest, result: GenerationResult) !void {
        _ = request;
        _ = result;

        try response.headers.append("content-type", "text/event-stream");
        try response.headers.append("cache-control", "no-cache");
        try response.headers.append("connection", "keep-alive");

        try response.do();

        const chunk_data = "data: {\"id\":\"cmpl-123\",\"object\":\"text_completion\",\"created\":1677652288,\"model\":\"llama-7b\",\"choices\":[{\"text\":\" world!\",\"index\":0,\"finish_reason\":null}]}\n\n";
        try response.writeAll(chunk_data);

        const final_chunk = "data: [DONE]\n\n";
        try response.writeAll(final_chunk);

        try response.finish();
    }

    /// Send error response
    fn sendErrorResponse(self: *Self, response: *http.Server.Response, status_code: u16, message: []const u8) !void {
        const error_response = try std.fmt.allocPrint(self.allocator,
            "{{\"error\":{{\"message\":\"{s}\",\"type\":\"invalid_request_error\",\"code\":{d}}}}}",
            .{ message, status_code });
        defer self.allocator.free(error_response);

        try response.headers.append("content-type", "application/json");
        const content_length = try std.fmt.allocPrint(self.allocator, "{d}", .{error_response.len});
        defer self.allocator.free(content_length);
        try response.headers.append("content-length", content_length);

        response.status = @enumFromInt(status_code);
        try response.do();
        try response.writeAll(error_response);
        try response.finish();
    }

    /// Apply CORS middleware
    fn applyCors(self: *Self, response: *http.Server.Response) !void {
        _ = self;
        try response.headers.append("access-control-allow-origin", "*");
        try response.headers.append("access-control-allow-methods", "GET, POST, PUT, DELETE, OPTIONS");
        try response.headers.append("access-control-allow-headers", "content-type, authorization");
        try response.headers.append("access-control-max-age", "86400");
    }

    /// Validate API key
    fn validateApiKey(self: *Self, request: *http.Server.Request) bool {
        if (self.config.api_key) |expected_key| {
            if (request.headers.getFirstValue("authorization")) |auth_header| {
                if (std.mem.startsWith(u8, auth_header, "Bearer ")) {
                    const provided_key = auth_header[7..];
                    return std.mem.eql(u8, provided_key, expected_key);
                }
            }
            return false;
        }
        return true; // No API key required
    }

    /// Extract request body as string
    fn extractRequestBody(self: *Self, request: *http.Server.Request) ![]u8 {
        var body = std.ArrayList(u8).init(self.allocator);
        defer body.deinit();

        var buffer: [4096]u8 = undefined;
        while (true) {
            const bytes_read = request.reader().read(&buffer) catch |err| switch (err) {
                error.EndOfStream => break,
                else => return err,
            };
            if (bytes_read == 0) break;
            try body.appendSlice(buffer[0..bytes_read]);
        }

        return try body.toOwnedSlice();
    }

    /// Generate unique request ID
    fn generateRequestId(self: *Self) ![]u8 {
        const counter = self.request_counter.fetchAdd(1, .monotonic);
        const timestamp = std.time.timestamp();
        return try std.fmt.allocPrint(self.allocator, "req_{x}_{d}", .{ timestamp, counter });
    }
};

/// Generation result structure
pub const GenerationResult = struct {
    text: ?[]const u8,
    tokens: []const u32,
    finish_reason: ?[]const u8,

    pub fn deinit(self: GenerationResult, allocator: Allocator) void {
        if (self.text) |text| {
            allocator.free(text);
        }
    }
};

/// Standalone server application
pub fn runServer(allocator: Allocator, config: ServerConfig) !void {
    var server = try ZigLlamaServer.init(allocator, config);
    defer server.deinit();

    // In a real implementation, would load model here
    std.log.info("⚠️  Note: Using demo responses. Load a model for real inference.");

    try server.start();
}