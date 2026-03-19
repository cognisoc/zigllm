const std = @import("std");
const Allocator = std.mem.Allocator;

/// Chat message structure
pub const ChatMessage = struct {
    role: []const u8,
    content: []const u8,
    name: ?[]const u8 = null,
};

/// Chat template types supporting various model formats
pub const TemplateType = enum {
    Llama2,
    CodeLlama,
    Llama3,
    Mistral,
    ChatML,
    Alpaca,
    Vicuna,
    Orca,
    GPT4,
    Claude,
    Custom,

    /// Get template type from string
    pub fn fromString(template_str: []const u8) ?TemplateType {
        const templates = std.ComptimeStringMap(TemplateType, .{
            .{ "llama2", .Llama2 },
            .{ "code-llama", .CodeLlama },
            .{ "llama3", .Llama3 },
            .{ "mistral", .Mistral },
            .{ "chatml", .ChatML },
            .{ "alpaca", .Alpaca },
            .{ "vicuna", .Vicuna },
            .{ "orca", .Orca },
            .{ "gpt4", .GPT4 },
            .{ "claude", .Claude },
            .{ "custom", .Custom },
        });
        return templates.get(template_str);
    }

    /// Get string representation
    pub fn toString(self: TemplateType) []const u8 {
        return switch (self) {
            .Llama2 => "llama2",
            .CodeLlama => "code-llama",
            .Llama3 => "llama3",
            .Mistral => "mistral",
            .ChatML => "chatml",
            .Alpaca => "alpaca",
            .Vicuna => "vicuna",
            .Orca => "orca",
            .GPT4 => "gpt4",
            .Claude => "claude",
            .Custom => "custom",
        };
    }
};

/// Chat template configuration
pub const ChatTemplate = struct {
    template_type: TemplateType,
    system_prefix: []const u8,
    system_suffix: []const u8,
    user_prefix: []const u8,
    user_suffix: []const u8,
    assistant_prefix: []const u8,
    assistant_suffix: []const u8,
    bos_token: []const u8,
    eos_token: []const u8,
    separator: []const u8,
    stop_sequences: [][]const u8,
    add_generation_prompt: bool,

    const Self = @This();

    /// Create template for specific model type
    pub fn create(template_type: TemplateType, allocator: Allocator) !Self {
        return switch (template_type) {
            .Llama2 => createLlama2Template(allocator),
            .CodeLlama => createCodeLlamaTemplate(allocator),
            .Llama3 => createLlama3Template(allocator),
            .Mistral => createMistralTemplate(allocator),
            .ChatML => createChatMLTemplate(allocator),
            .Alpaca => createAlpacaTemplate(allocator),
            .Vicuna => createVicunaTemplate(allocator),
            .Orca => createOrcaTemplate(allocator),
            .GPT4 => createGPT4Template(allocator),
            .Claude => createClaudeTemplate(allocator),
            .Custom => createCustomTemplate(allocator),
        };
    }

    /// Apply template to conversation
    pub fn apply(self: Self, messages: []const ChatMessage, allocator: Allocator) ![]u8 {
        var result = std.ArrayList(u8).init(allocator);
        defer result.deinit();

        // Add BOS token
        try result.appendSlice(self.bos_token);

        var system_message: ?ChatMessage = null;
        var conversation_messages = std.ArrayList(ChatMessage).init(allocator);
        defer conversation_messages.deinit();

        // Separate system message from conversation
        for (messages) |message| {
            if (std.mem.eql(u8, message.role, "system")) {
                system_message = message;
            } else {
                try conversation_messages.append(message);
            }
        }

        // Add system message if present
        if (system_message) |sys_msg| {
            try result.appendSlice(self.system_prefix);
            try result.appendSlice(sys_msg.content);
            try result.appendSlice(self.system_suffix);
            if (conversation_messages.items.len > 0) {
                try result.appendSlice(self.separator);
            }
        }

        // Add conversation messages
        for (conversation_messages.items, 0..) |message, i| {
            if (std.mem.eql(u8, message.role, "user")) {
                try result.appendSlice(self.user_prefix);
                try result.appendSlice(message.content);
                try result.appendSlice(self.user_suffix);
            } else if (std.mem.eql(u8, message.role, "assistant")) {
                try result.appendSlice(self.assistant_prefix);
                try result.appendSlice(message.content);
                try result.appendSlice(self.assistant_suffix);
            }

            // Add separator between messages (but not after the last one)
            if (i < conversation_messages.items.len - 1) {
                try result.appendSlice(self.separator);
            }
        }

        // Add generation prompt if needed
        if (self.add_generation_prompt and conversation_messages.items.len > 0) {
            const last_message = conversation_messages.items[conversation_messages.items.len - 1];
            if (std.mem.eql(u8, last_message.role, "user")) {
                try result.appendSlice(self.separator);
                try result.appendSlice(self.assistant_prefix);
            }
        }

        return try result.toOwnedSlice();
    }

    /// Get stop sequences for this template
    pub fn getStopSequences(self: Self) [][]const u8 {
        return self.stop_sequences;
    }

    /// Check if EOS should be added
    pub fn shouldAddEOS(self: Self, text: []const u8) bool {
        if (self.eos_token.len == 0) return false;
        return !std.mem.endsWith(u8, text, self.eos_token);
    }

    pub fn deinit(self: Self, allocator: Allocator) void {
        allocator.free(self.stop_sequences);
    }
};

// Template implementations for different model families

fn createLlama2Template(allocator: Allocator) !ChatTemplate {
    const stop_sequences = try allocator.alloc([]const u8, 2);
    stop_sequences[0] = "</s>";
    stop_sequences[1] = "[/INST]";

    return ChatTemplate{
        .template_type = .Llama2,
        .system_prefix = "[INST] <<SYS>>\n",
        .system_suffix = "\n<</SYS>>\n\n",
        .user_prefix = "[INST] ",
        .user_suffix = " [/INST]",
        .assistant_prefix = " ",
        .assistant_suffix = " </s>",
        .bos_token = "<s>",
        .eos_token = "</s>",
        .separator = "<s>",
        .stop_sequences = stop_sequences,
        .add_generation_prompt = true,
    };
}

fn createCodeLlamaTemplate(allocator: Allocator) !ChatTemplate {
    const stop_sequences = try allocator.alloc([]const u8, 3);
    stop_sequences[0] = "</s>";
    stop_sequences[1] = "[/INST]";
    stop_sequences[2] = "```";

    return ChatTemplate{
        .template_type = .CodeLlama,
        .system_prefix = "[INST] <<SYS>>\n",
        .system_suffix = "\n<</SYS>>\n\n",
        .user_prefix = "[INST] ",
        .user_suffix = " [/INST]",
        .assistant_prefix = " ",
        .assistant_suffix = " </s>",
        .bos_token = "<s>",
        .eos_token = "</s>",
        .separator = "<s>",
        .stop_sequences = stop_sequences,
        .add_generation_prompt = true,
    };
}

fn createLlama3Template(allocator: Allocator) !ChatTemplate {
    const stop_sequences = try allocator.alloc([]const u8, 3);
    stop_sequences[0] = "<|end_of_text|>";
    stop_sequences[1] = "<|eot_id|>";
    stop_sequences[2] = "<|start_header_id|>";

    return ChatTemplate{
        .template_type = .Llama3,
        .system_prefix = "<|start_header_id|>system<|end_header_id|>\n\n",
        .system_suffix = "<|eot_id|>",
        .user_prefix = "<|start_header_id|>user<|end_header_id|>\n\n",
        .user_suffix = "<|eot_id|>",
        .assistant_prefix = "<|start_header_id|>assistant<|end_header_id|>\n\n",
        .assistant_suffix = "<|eot_id|>",
        .bos_token = "<|begin_of_text|>",
        .eos_token = "<|end_of_text|>",
        .separator = "",
        .stop_sequences = stop_sequences,
        .add_generation_prompt = true,
    };
}

fn createMistralTemplate(allocator: Allocator) !ChatTemplate {
    const stop_sequences = try allocator.alloc([]const u8, 2);
    stop_sequences[0] = "</s>";
    stop_sequences[1] = "[/INST]";

    return ChatTemplate{
        .template_type = .Mistral,
        .system_prefix = "",
        .system_suffix = "",
        .user_prefix = "[INST] ",
        .user_suffix = " [/INST]",
        .assistant_prefix = "",
        .assistant_suffix = "</s>",
        .bos_token = "<s>",
        .eos_token = "</s>",
        .separator = " ",
        .stop_sequences = stop_sequences,
        .add_generation_prompt = true,
    };
}

fn createChatMLTemplate(allocator: Allocator) !ChatTemplate {
    const stop_sequences = try allocator.alloc([]const u8, 3);
    stop_sequences[0] = "<|im_end|>";
    stop_sequences[1] = "<|im_start|>";
    stop_sequences[2] = "<|endoftext|>";

    return ChatTemplate{
        .template_type = .ChatML,
        .system_prefix = "<|im_start|>system\n",
        .system_suffix = "<|im_end|>",
        .user_prefix = "<|im_start|>user\n",
        .user_suffix = "<|im_end|>",
        .assistant_prefix = "<|im_start|>assistant\n",
        .assistant_suffix = "<|im_end|>",
        .bos_token = "",
        .eos_token = "<|endoftext|>",
        .separator = "\n",
        .stop_sequences = stop_sequences,
        .add_generation_prompt = true,
    };
}

fn createAlpacaTemplate(allocator: Allocator) !ChatTemplate {
    const stop_sequences = try allocator.alloc([]const u8, 2);
    stop_sequences[0] = "### Human:";
    stop_sequences[1] = "### Assistant:";

    return ChatTemplate{
        .template_type = .Alpaca,
        .system_prefix = "### System:\n",
        .system_suffix = "\n\n",
        .user_prefix = "### Human:\n",
        .user_suffix = "\n\n",
        .assistant_prefix = "### Assistant:\n",
        .assistant_suffix = "\n\n",
        .bos_token = "",
        .eos_token = "",
        .separator = "",
        .stop_sequences = stop_sequences,
        .add_generation_prompt = true,
    };
}

fn createVicunaTemplate(allocator: Allocator) !ChatTemplate {
    const stop_sequences = try allocator.alloc([]const u8, 2);
    stop_sequences[0] = "USER:";
    stop_sequences[1] = "ASSISTANT:";

    return ChatTemplate{
        .template_type = .Vicuna,
        .system_prefix = "",
        .system_suffix = "",
        .user_prefix = "USER: ",
        .user_suffix = "\n",
        .assistant_prefix = "ASSISTANT: ",
        .assistant_suffix = "</s>",
        .bos_token = "",
        .eos_token = "</s>",
        .separator = "",
        .stop_sequences = stop_sequences,
        .add_generation_prompt = true,
    };
}

fn createOrcaTemplate(allocator: Allocator) !ChatTemplate {
    const stop_sequences = try allocator.alloc([]const u8, 3);
    stop_sequences[0] = "<|im_end|>";
    stop_sequences[1] = "User:";
    stop_sequences[2] = "Assistant:";

    return ChatTemplate{
        .template_type = .Orca,
        .system_prefix = "System:\n",
        .system_suffix = "\n",
        .user_prefix = "User:\n",
        .user_suffix = "\n",
        .assistant_prefix = "Assistant:\n",
        .assistant_suffix = "<|im_end|>",
        .bos_token = "",
        .eos_token = "<|im_end|>",
        .separator = "",
        .stop_sequences = stop_sequences,
        .add_generation_prompt = true,
    };
}

fn createGPT4Template(allocator: Allocator) !ChatTemplate {
    const stop_sequences = try allocator.alloc([]const u8, 1);
    stop_sequences[0] = "<|endoftext|>";

    return ChatTemplate{
        .template_type = .GPT4,
        .system_prefix = "",
        .system_suffix = "",
        .user_prefix = "",
        .user_suffix = "",
        .assistant_prefix = "",
        .assistant_suffix = "",
        .bos_token = "",
        .eos_token = "<|endoftext|>",
        .separator = "\n\n",
        .stop_sequences = stop_sequences,
        .add_generation_prompt = false,
    };
}

fn createClaudeTemplate(allocator: Allocator) !ChatTemplate {
    const stop_sequences = try allocator.alloc([]const u8, 2);
    stop_sequences[0] = "\n\nHuman:";
    stop_sequences[1] = "\n\nAssistant:";

    return ChatTemplate{
        .template_type = .Claude,
        .system_prefix = "",
        .system_suffix = "",
        .user_prefix = "\n\nHuman: ",
        .user_suffix = "",
        .assistant_prefix = "\n\nAssistant: ",
        .assistant_suffix = "",
        .bos_token = "",
        .eos_token = "",
        .separator = "",
        .stop_sequences = stop_sequences,
        .add_generation_prompt = true,
    };
}

fn createCustomTemplate(allocator: Allocator) !ChatTemplate {
    const stop_sequences = try allocator.alloc([]const u8, 0);

    return ChatTemplate{
        .template_type = .Custom,
        .system_prefix = "",
        .system_suffix = "",
        .user_prefix = "",
        .user_suffix = "",
        .assistant_prefix = "",
        .assistant_suffix = "",
        .bos_token = "",
        .eos_token = "",
        .separator = "",
        .stop_sequences = stop_sequences,
        .add_generation_prompt = false,
    };
}

/// Chat template manager for handling multiple templates
pub const ChatTemplateManager = struct {
    templates: std.HashMap(TemplateType, ChatTemplate, TemplateTypeContext, std.hash_map.default_max_load_percentage),
    allocator: Allocator,

    const Self = @This();

    const TemplateTypeContext = struct {
        pub fn hash(self: @This(), template_type: TemplateType) u64 {
            _ = self;
            return std.hash_map.hashString(template_type.toString());
        }

        pub fn eql(self: @This(), a: TemplateType, b: TemplateType) bool {
            _ = self;
            return a == b;
        }
    };

    pub fn init(allocator: Allocator) Self {
        return Self{
            .templates = std.HashMap(TemplateType, ChatTemplate, TemplateTypeContext, std.hash_map.default_max_load_percentage).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        var iterator = self.templates.iterator();
        while (iterator.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.templates.deinit();
    }

    /// Load template for specific type
    pub fn loadTemplate(self: *Self, template_type: TemplateType) !void {
        if (self.templates.contains(template_type)) return;

        const template = try ChatTemplate.create(template_type, self.allocator);
        try self.templates.put(template_type, template);
    }

    /// Get template
    pub fn getTemplate(self: *Self, template_type: TemplateType) ?*ChatTemplate {
        return self.templates.getPtr(template_type);
    }

    /// Apply template to messages
    pub fn applyTemplate(self: *Self, template_type: TemplateType, messages: []const ChatMessage) ![]u8 {
        try self.loadTemplate(template_type);
        if (self.getTemplate(template_type)) |template| {
            return try template.apply(messages, self.allocator);
        }
        return error.TemplateNotFound;
    }

    /// Auto-detect template from model name
    pub fn detectTemplate(model_name: []const u8) TemplateType {
        const lower_name = std.ascii.lowerString(std.heap.page_allocator, model_name) catch model_name;
        defer if (lower_name.ptr != model_name.ptr) std.heap.page_allocator.free(lower_name);

        if (std.mem.indexOf(u8, lower_name, "llama-3") != null or
            std.mem.indexOf(u8, lower_name, "llama3") != null) {
            return .Llama3;
        } else if (std.mem.indexOf(u8, lower_name, "code-llama") != null or
                   std.mem.indexOf(u8, lower_name, "codellama") != null) {
            return .CodeLlama;
        } else if (std.mem.indexOf(u8, lower_name, "llama-2") != null or
                   std.mem.indexOf(u8, lower_name, "llama2") != null or
                   std.mem.indexOf(u8, lower_name, "llama") != null) {
            return .Llama2;
        } else if (std.mem.indexOf(u8, lower_name, "mistral") != null or
                   std.mem.indexOf(u8, lower_name, "mixtral") != null) {
            return .Mistral;
        } else if (std.mem.indexOf(u8, lower_name, "gpt-4") != null or
                   std.mem.indexOf(u8, lower_name, "gpt4") != null) {
            return .GPT4;
        } else if (std.mem.indexOf(u8, lower_name, "alpaca") != null) {
            return .Alpaca;
        } else if (std.mem.indexOf(u8, lower_name, "vicuna") != null) {
            return .Vicuna;
        } else if (std.mem.indexOf(u8, lower_name, "orca") != null) {
            return .Orca;
        } else if (std.mem.indexOf(u8, lower_name, "claude") != null) {
            return .Claude;
        } else {
            return .ChatML; // Default to ChatML for unknown models
        }
    }

    /// List available templates
    pub fn listTemplates(self: *Self, allocator: Allocator) ![]TemplateType {
        var template_list = std.ArrayList(TemplateType).init(allocator);

        const all_templates = [_]TemplateType{
            .Llama2, .CodeLlama, .Llama3, .Mistral, .ChatML,
            .Alpaca, .Vicuna, .Orca, .GPT4, .Claude, .Custom,
        };

        for (all_templates) |template_type| {
            try template_list.append(template_type);
        }

        return try template_list.toOwnedSlice();
    }
};

/// Utilities for working with chat templates
pub const ChatTemplateUtils = struct {
    /// Extract system message from conversation
    pub fn extractSystemMessage(messages: []const ChatMessage, allocator: Allocator) !?[]u8 {
        for (messages) |message| {
            if (std.mem.eql(u8, message.role, "system")) {
                return try allocator.dupe(u8, message.content);
            }
        }
        return null;
    }

    /// Count tokens in templated conversation (rough estimate)
    pub fn estimateTokenCount(templated_text: []const u8) usize {
        // Rough estimation: 1 token per 4 characters (English text average)
        return templated_text.len / 4;
    }

    /// Validate message format
    pub fn validateMessage(message: ChatMessage) bool {
        if (message.content.len == 0) return false;

        const valid_roles = [_][]const u8{ "system", "user", "assistant", "function" };
        for (valid_roles) |role| {
            if (std.mem.eql(u8, message.role, role)) return true;
        }
        return false;
    }

    /// Validate conversation format
    pub fn validateConversation(messages: []const ChatMessage) bool {
        if (messages.len == 0) return false;

        for (messages) |message| {
            if (!validateMessage(message)) return false;
        }

        // Check alternating user/assistant pattern (after system message)
        var expect_user = true;
        for (messages) |message| {
            if (std.mem.eql(u8, message.role, "system")) continue;

            if (expect_user and !std.mem.eql(u8, message.role, "user")) return false;
            if (!expect_user and !std.mem.eql(u8, message.role, "assistant")) return false;

            expect_user = !expect_user;
        }

        return true;
    }

    /// Truncate conversation to fit context window
    pub fn truncateConversation(messages: []const ChatMessage, max_tokens: usize,
                               allocator: Allocator, template: ChatTemplate) ![]ChatMessage {
        var result = std.ArrayList(ChatMessage).init(allocator);
        defer result.deinit();

        var system_message: ?ChatMessage = null;
        var conversation_messages = std.ArrayList(ChatMessage).init(allocator);
        defer conversation_messages.deinit();

        // Separate system and conversation messages
        for (messages) |message| {
            if (std.mem.eql(u8, message.role, "system")) {
                system_message = message;
            } else {
                try conversation_messages.append(message);
            }
        }

        // Always include system message if present
        if (system_message) |sys_msg| {
            try result.append(sys_msg);
        }

        // Add conversation messages from the end until we hit token limit
        var estimated_tokens: usize = 0;
        if (system_message) |sys_msg| {
            estimated_tokens += estimateTokenCount(sys_msg.content);
        }

        var i = conversation_messages.items.len;
        while (i > 0) {
            i -= 1;
            const message = conversation_messages.items[i];
            const message_tokens = estimateTokenCount(message.content);

            if (estimated_tokens + message_tokens > max_tokens) break;

            estimated_tokens += message_tokens;
        }

        // Add remaining messages in correct order
        const start_idx = if (i == 0) 0 else i + 1;
        for (conversation_messages.items[start_idx..]) |message| {
            try result.append(message);
        }

        return try result.toOwnedSlice();
    }
};