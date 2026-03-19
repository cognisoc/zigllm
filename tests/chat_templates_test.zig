const std = @import("std");
const testing = std.testing;
const chat_templates = @import("../src/models/chat_templates.zig");
const ChatMessage = chat_templates.ChatMessage;
const ChatTemplate = chat_templates.ChatTemplate;
const TemplateType = chat_templates.TemplateType;
const ChatTemplateManager = chat_templates.ChatTemplateManager;
const ChatTemplateUtils = chat_templates.ChatTemplateUtils;

test "Template type from string conversion" {
    try testing.expect(TemplateType.fromString("llama2") == .Llama2);
    try testing.expect(TemplateType.fromString("mistral") == .Mistral);
    try testing.expect(TemplateType.fromString("chatml") == .ChatML);
    try testing.expect(TemplateType.fromString("unknown") == null);

    try testing.expectEqualStrings(TemplateType.Llama3.toString(), "llama3");
    try testing.expectEqualStrings(TemplateType.Mistral.toString(), "mistral");
}

test "Llama2 template creation and application" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const template = try ChatTemplate.create(.Llama2, allocator);
    defer template.deinit(allocator);

    const messages = [_]ChatMessage{
        .{ .role = "system", .content = "You are a helpful assistant." },
        .{ .role = "user", .content = "Hello, how are you?" },
    };

    const result = try template.apply(&messages, allocator);
    defer allocator.free(result);

    try testing.expect(std.mem.indexOf(u8, result, "<s>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "<<SYS>>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "You are a helpful assistant.") != null);
    try testing.expect(std.mem.indexOf(u8, result, "[INST]") != null);
    try testing.expect(std.mem.indexOf(u8, result, "Hello, how are you?") != null);
}

test "Llama3 template creation and application" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const template = try ChatTemplate.create(.Llama3, allocator);
    defer template.deinit(allocator);

    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "What is the capital of France?" },
        .{ .role = "assistant", .content = "The capital of France is Paris." },
    };

    const result = try template.apply(&messages, allocator);
    defer allocator.free(result);

    try testing.expect(std.mem.indexOf(u8, result, "<|begin_of_text|>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "<|start_header_id|>user<|end_header_id|>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "<|start_header_id|>assistant<|end_header_id|>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "<|eot_id|>") != null);
}

test "Mistral template creation and application" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const template = try ChatTemplate.create(.Mistral, allocator);
    defer template.deinit(allocator);

    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Explain quantum computing" },
    };

    const result = try template.apply(&messages, allocator);
    defer allocator.free(result);

    try testing.expect(std.mem.indexOf(u8, result, "<s>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "[INST]") != null);
    try testing.expect(std.mem.indexOf(u8, result, "Explain quantum computing") != null);
    try testing.expect(std.mem.indexOf(u8, result, "[/INST]") != null);
}

test "ChatML template creation and application" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const template = try ChatTemplate.create(.ChatML, allocator);
    defer template.deinit(allocator);

    const messages = [_]ChatMessage{
        .{ .role = "system", .content = "You are ChatGPT, a helpful AI assistant." },
        .{ .role = "user", .content = "Write a haiku about programming" },
    };

    const result = try template.apply(&messages, allocator);
    defer allocator.free(result);

    try testing.expect(std.mem.indexOf(u8, result, "<|im_start|>system") != null);
    try testing.expect(std.mem.indexOf(u8, result, "<|im_start|>user") != null);
    try testing.expect(std.mem.indexOf(u8, result, "<|im_end|>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "Write a haiku about programming") != null);
}

test "Alpaca template creation and application" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const template = try ChatTemplate.create(.Alpaca, allocator);
    defer template.deinit(allocator);

    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Describe machine learning" },
    };

    const result = try template.apply(&messages, allocator);
    defer allocator.free(result);

    try testing.expect(std.mem.indexOf(u8, result, "### Human:") != null);
    try testing.expect(std.mem.indexOf(u8, result, "### Assistant:") != null);
    try testing.expect(std.mem.indexOf(u8, result, "Describe machine learning") != null);
}

test "Claude template creation and application" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const template = try ChatTemplate.create(.Claude, allocator);
    defer template.deinit(allocator);

    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "What is artificial intelligence?" },
    };

    const result = try template.apply(&messages, allocator);
    defer allocator.free(result);

    try testing.expect(std.mem.indexOf(u8, result, "\n\nHuman:") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\n\nAssistant:") != null);
    try testing.expect(std.mem.indexOf(u8, result, "What is artificial intelligence?") != null);
}

test "Template manager initialization and usage" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var manager = ChatTemplateManager.init(allocator);
    defer manager.deinit();

    // Load templates
    try manager.loadTemplate(.Llama2);
    try manager.loadTemplate(.ChatML);

    try testing.expect(manager.getTemplate(.Llama2) != null);
    try testing.expect(manager.getTemplate(.ChatML) != null);
    try testing.expect(manager.getTemplate(.Mistral) == null);

    // Apply template via manager
    const messages = [_]ChatMessage{
        .{ .role = "user", .content = "Test message" },
    };

    const result = try manager.applyTemplate(.ChatML, &messages);
    defer allocator.free(result);

    try testing.expect(std.mem.indexOf(u8, result, "<|im_start|>") != null);
}

test "Template auto-detection from model names" {
    try testing.expect(ChatTemplateManager.detectTemplate("llama-2-7b-chat") == .Llama2);
    try testing.expect(ChatTemplateManager.detectTemplate("Llama-3-8B-Instruct") == .Llama3);
    try testing.expect(ChatTemplateManager.detectTemplate("CodeLlama-7b-Python") == .CodeLlama);
    try testing.expect(ChatTemplateManager.detectTemplate("Mistral-7B-Instruct-v0.2") == .Mistral);
    try testing.expect(ChatTemplateManager.detectTemplate("gpt-4-turbo") == .GPT4);
    try testing.expect(ChatTemplateManager.detectTemplate("alpaca-7b") == .Alpaca);
    try testing.expect(ChatTemplateManager.detectTemplate("vicuna-13b") == .Vicuna);
    try testing.expect(ChatTemplateManager.detectTemplate("orca-mini") == .Orca);
    try testing.expect(ChatTemplateManager.detectTemplate("claude-3-haiku") == .Claude);
    try testing.expect(ChatTemplateManager.detectTemplate("unknown-model") == .ChatML); // Default
}

test "List available templates" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var manager = ChatTemplateManager.init(allocator);
    defer manager.deinit();

    const template_list = try manager.listTemplates(allocator);
    defer allocator.free(template_list);

    try testing.expect(template_list.len == 11);

    // Check if some expected templates are present
    var found_llama2 = false;
    var found_chatML = false;
    var found_mistral = false;

    for (template_list) |template_type| {
        if (template_type == .Llama2) found_llama2 = true;
        if (template_type == .ChatML) found_chatML = true;
        if (template_type == .Mistral) found_mistral = true;
    }

    try testing.expect(found_llama2);
    try testing.expect(found_chatML);
    try testing.expect(found_mistral);
}

test "Message validation" {
    const valid_message = ChatMessage{ .role = "user", .content = "Hello world" };
    const empty_content = ChatMessage{ .role = "user", .content = "" };
    const invalid_role = ChatMessage{ .role = "invalid", .content = "Hello" };

    try testing.expect(ChatTemplateUtils.validateMessage(valid_message) == true);
    try testing.expect(ChatTemplateUtils.validateMessage(empty_content) == false);
    try testing.expect(ChatTemplateUtils.validateMessage(invalid_role) == false);
}

test "Conversation validation" {
    const valid_conversation = [_]ChatMessage{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "Hello" },
        .{ .role = "assistant", .content = "Hi there!" },
        .{ .role = "user", .content = "How are you?" },
    };

    const invalid_conversation = [_]ChatMessage{
        .{ .role = "user", .content = "Hello" },
        .{ .role = "user", .content = "Hello again" }, // Two user messages in a row
    };

    const empty_conversation: [0]ChatMessage = .{};

    try testing.expect(ChatTemplateUtils.validateConversation(&valid_conversation) == true);
    try testing.expect(ChatTemplateUtils.validateConversation(&invalid_conversation) == false);
    try testing.expect(ChatTemplateUtils.validateConversation(&empty_conversation) == false);
}

test "System message extraction" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const messages_with_system = [_]ChatMessage{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "Hello" },
    };

    const messages_without_system = [_]ChatMessage{
        .{ .role = "user", .content = "Hello" },
        .{ .role = "assistant", .content = "Hi!" },
    };

    const system_message = try ChatTemplateUtils.extractSystemMessage(&messages_with_system, allocator);
    defer if (system_message) |msg| allocator.free(msg);

    const no_system_message = try ChatTemplateUtils.extractSystemMessage(&messages_without_system, allocator);

    try testing.expect(system_message != null);
    try testing.expectEqualStrings(system_message.?, "You are helpful.");
    try testing.expect(no_system_message == null);
}

test "Token count estimation" {
    const short_text = "Hello world";
    const long_text = "This is a much longer text that should have more tokens estimated for it because it contains significantly more characters and words.";

    const short_tokens = ChatTemplateUtils.estimateTokenCount(short_text);
    const long_tokens = ChatTemplateUtils.estimateTokenCount(long_text);

    try testing.expect(short_tokens < long_tokens);
    try testing.expect(short_tokens >= 2); // "Hello world" should have at least 2 tokens
    try testing.expect(long_tokens >= 20); // Long text should have many tokens
}

test "Conversation truncation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const template = try ChatTemplate.create(.ChatML, allocator);
    defer template.deinit(allocator);

    const long_conversation = [_]ChatMessage{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "Message 1" },
        .{ .role = "assistant", .content = "Response 1" },
        .{ .role = "user", .content = "Message 2" },
        .{ .role = "assistant", .content = "Response 2" },
        .{ .role = "user", .content = "Message 3" },
        .{ .role = "assistant", .content = "Response 3" },
    };

    const truncated = try ChatTemplateUtils.truncateConversation(
        &long_conversation,
        10, // Very small token limit
        allocator,
        template,
    );
    defer allocator.free(truncated);

    // Should keep system message and only the most recent messages
    try testing.expect(truncated.len < long_conversation.len);
    try testing.expect(truncated.len >= 1); // At least system message

    // First message should be system (if present)
    if (truncated.len > 0) {
        try testing.expectEqualStrings(truncated[0].role, "system");
    }
}

test "Stop sequences retrieval" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const llama2_template = try ChatTemplate.create(.Llama2, allocator);
    defer llama2_template.deinit(allocator);

    const chatML_template = try ChatTemplate.create(.ChatML, allocator);
    defer chatML_template.deinit(allocator);

    const llama2_stops = llama2_template.getStopSequences();
    const chatML_stops = chatML_template.getStopSequences();

    try testing.expect(llama2_stops.len > 0);
    try testing.expect(chatML_stops.len > 0);
    try testing.expect(!std.mem.eql(u8, llama2_stops[0], chatML_stops[0])); // Different stop sequences
}

test "EOS token handling" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const template = try ChatTemplate.create(.Llama2, allocator);
    defer template.deinit(allocator);

    const text_without_eos = "This is some generated text";
    const text_with_eos = "This is some generated text</s>";

    try testing.expect(template.shouldAddEOS(text_without_eos) == true);
    try testing.expect(template.shouldAddEOS(text_with_eos) == false);
}

test "Complex conversation with multiple roles" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const template = try ChatTemplate.create(.Llama3, allocator);
    defer template.deinit(allocator);

    const messages = [_]ChatMessage{
        .{ .role = "system", .content = "You are a helpful coding assistant." },
        .{ .role = "user", .content = "Write a Python function to calculate fibonacci numbers." },
        .{ .role = "assistant", .content = "Here's a Python function:\n\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)" },
        .{ .role = "user", .content = "Can you optimize it?" },
    };

    const result = try template.apply(&messages, allocator);
    defer allocator.free(result);

    // Verify all messages are included
    try testing.expect(std.mem.indexOf(u8, result, "coding assistant") != null);
    try testing.expect(std.mem.indexOf(u8, result, "Python function") != null);
    try testing.expect(std.mem.indexOf(u8, result, "def fibonacci") != null);
    try testing.expect(std.mem.indexOf(u8, result, "optimize") != null);

    // Verify proper Llama3 formatting
    try testing.expect(std.mem.indexOf(u8, result, "<|start_header_id|>system<|end_header_id|>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "<|start_header_id|>user<|end_header_id|>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "<|start_header_id|>assistant<|end_header_id|>") != null);
}