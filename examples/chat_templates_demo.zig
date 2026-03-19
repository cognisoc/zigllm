const std = @import("std");
const chat_templates = @import("../src/models/chat_templates.zig");
const ChatMessage = chat_templates.ChatMessage;
const TemplateType = chat_templates.TemplateType;
const ChatTemplateManager = chat_templates.ChatTemplateManager;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("🦙 ZigLlama Chat Template System Demo\n", .{});
    std.debug.print("=====================================\n\n", .{});

    // Create template manager
    var manager = ChatTemplateManager.init(allocator);
    defer manager.deinit();

    // Sample conversation
    const messages = [_]ChatMessage{
        .{ .role = "system", .content = "You are a helpful AI assistant that explains complex topics clearly." },
        .{ .role = "user", .content = "Can you explain how transformers work in neural networks?" },
        .{ .role = "assistant", .content = "Certainly! Transformers are neural network architectures that use self-attention mechanisms to process sequential data. The key innovation is the attention mechanism, which allows the model to focus on different parts of the input when making predictions." },
        .{ .role = "user", .content = "What makes attention so powerful?" },
    };

    std.debug.print("📝 Sample Conversation:\n");
    for (messages, 0..) |message, i| {
        std.debug.print("  {d}. {s}: {s}\n", .{ i + 1, message.role, message.content });
    }
    std.debug.print("\n");

    // Demonstrate different templates
    const template_types = [_]TemplateType{
        .Llama2, .Llama3, .ChatML, .Mistral, .Alpaca, .Claude,
    };

    for (template_types) |template_type| {
        std.debug.print("🔧 Template: {s}\n", .{template_type.toString()});
        std.debug.print("{s}\n", .{"=" ** 50});

        // Apply template
        const templated_text = manager.applyTemplate(template_type, &messages) catch |err| {
            std.debug.print("❌ Failed to apply template: {}\n\n", .{err});
            continue;
        };
        defer allocator.free(templated_text);

        // Print templated result (truncated for readability)
        const display_text = if (templated_text.len > 300)
            templated_text[0..300]
        else
            templated_text;

        std.debug.print("📋 Result:\n{s}", .{display_text});
        if (templated_text.len > 300) {
            std.debug.print("... (truncated, full length: {d} chars)", .{templated_text.len});
        }
        std.debug.print("\n\n");
    }

    // Template auto-detection demo
    std.debug.print("🔍 Template Auto-Detection Demo\n", .{});
    std.debug.print("================================\n");

    const model_names = [_][]const u8{
        "llama-2-7b-chat",
        "Llama-3-8B-Instruct",
        "CodeLlama-7b-Python",
        "Mistral-7B-Instruct-v0.2",
        "gpt-4-turbo",
        "alpaca-13b",
        "vicuna-7b-v1.5",
        "orca-mini-3b",
        "claude-3-haiku",
        "some-unknown-model",
    };

    for (model_names) |model_name| {
        const detected = ChatTemplateManager.detectTemplate(model_name);
        std.debug.print("📱 Model: {s:25} → Template: {s}\n", .{ model_name, detected.toString() });
    }
    std.debug.print("\n");

    // Template comparison for same input
    std.debug.print("⚖️  Template Comparison\n", .{});
    std.debug.print("========================\n");

    const simple_messages = [_]ChatMessage{
        .{ .role = "user", .content = "Hello, how are you?" },
    };

    std.debug.print("Input: \"Hello, how are you?\"\n\n");

    const comparison_templates = [_]TemplateType{ .Llama2, .Llama3, .ChatML, .Mistral };
    for (comparison_templates) |template_type| {
        const result = manager.applyTemplate(template_type, &simple_messages) catch continue;
        defer allocator.free(result);

        std.debug.print("{s:8} → {s}\n", .{ template_type.toString(), result });
    }
    std.debug.print("\n");

    // Template statistics
    std.debug.print("📊 Template Statistics\n", .{});
    std.debug.print("======================\n");

    const test_conversation = [_]ChatMessage{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "Write a haiku about programming." },
        .{ .role = "assistant", .content = "Code flows like water,\nLogic builds bridges of thought,\nBugs teach us patience." },
    };

    for (comparison_templates) |template_type| {
        const result = manager.applyTemplate(template_type, &test_conversation) catch continue;
        defer allocator.free(result);

        const token_estimate = chat_templates.ChatTemplateUtils.estimateTokenCount(result);
        std.debug.print("{s:8} → {d:4} chars, ~{d:3} tokens\n", .{ template_type.toString(), result.len, token_estimate });
    }
    std.debug.print("\n");

    // List all available templates
    std.debug.print("📋 Available Templates\n", .{});
    std.debug.print("======================\n");

    const template_list = try manager.listTemplates(allocator);
    defer allocator.free(template_list);

    for (template_list, 0..) |template_type, i| {
        std.debug.print("{d:2}. {s}\n", .{ i + 1, template_type.toString() });
    }
    std.debug.print("\n");

    // Template features comparison
    std.debug.print("🛠️  Template Features\n", .{});
    std.debug.print("====================\n");

    const feature_templates = [_]TemplateType{ .Llama2, .Llama3, .ChatML, .Mistral };
    for (feature_templates) |template_type| {
        try manager.loadTemplate(template_type);
        if (manager.getTemplate(template_type)) |template| {
            const stop_sequences = template.getStopSequences();
            std.debug.print("{s:8} → BOS: \"{s}\", EOS: \"{s}\", Stops: {d}\n", .{
                template_type.toString(),
                template.bos_token,
                template.eos_token,
                stop_sequences.len,
            });
        }
    }
    std.debug.print("\n");

    // Message validation demo
    std.debug.print("✅ Message Validation Demo\n", .{});
    std.debug.print("===========================\n");

    const valid_msg = ChatMessage{ .role = "user", .content = "Valid message" };
    const invalid_role = ChatMessage{ .role = "invalid_role", .content = "Content" };
    const empty_content = ChatMessage{ .role = "user", .content = "" };

    const test_messages = [_]struct { msg: ChatMessage, desc: []const u8 }{
        .{ .msg = valid_msg, .desc = "Valid user message" },
        .{ .msg = invalid_role, .desc = "Invalid role" },
        .{ .msg = empty_content, .desc = "Empty content" },
    };

    for (test_messages) |test_case| {
        const is_valid = chat_templates.ChatTemplateUtils.validateMessage(test_case.msg);
        std.debug.print("{s} → {s}\n", .{ test_case.desc, if (is_valid) "✅ Valid" else "❌ Invalid" });
    }
    std.debug.print("\n");

    // Conversation validation demo
    const valid_conversation = [_]ChatMessage{
        .{ .role = "system", .content = "System prompt" },
        .{ .role = "user", .content = "User message" },
        .{ .role = "assistant", .content = "Assistant response" },
    };

    const invalid_conversation = [_]ChatMessage{
        .{ .role = "user", .content = "First user message" },
        .{ .role = "user", .content = "Second user message" }, // Invalid: two user messages in a row
    };

    const conversations = [_]struct { conv: []const ChatMessage, desc: []const u8 }{
        .{ .conv = &valid_conversation, .desc = "Valid conversation" },
        .{ .conv = &invalid_conversation, .desc = "Invalid conversation (consecutive user msgs)" },
        .{ .conv = &[_]ChatMessage{}, .desc = "Empty conversation" },
    };

    for (conversations) |test_case| {
        const is_valid = chat_templates.ChatTemplateUtils.validateConversation(test_case.conv);
        std.debug.print("{s} → {s}\n", .{ test_case.desc, if (is_valid) "✅ Valid" else "❌ Invalid" });
    }
    std.debug.print("\n");

    std.debug.print("🎉 Chat Template System Demo Complete!\n");
    std.debug.print("=======================================\n");
    std.debug.print("✨ The chat template system provides:\n");
    std.debug.print("   • Support for 11 major chat template formats\n");
    std.debug.print("   • Automatic template detection from model names\n");
    std.debug.print("   • Message and conversation validation\n");
    std.debug.print("   • Token estimation and truncation utilities\n");
    std.debug.print("   • Production-ready template management\n");
    std.debug.print("   • Integration with HTTP/REST API server\n\n");
    std.debug.print("🦙 ZigLlama: Educational Excellence with Production Power! ✨\n");
}