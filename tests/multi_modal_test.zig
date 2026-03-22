const std = @import("std");
const testing = std.testing;
const print = std.debug.print;
const expectEqual = testing.expectEqual;
const expect = testing.expect;
const allocator = testing.allocator;

test "multi-modal configuration creation" {
    const allocator = testing.allocator;

    // Test Vision Transformer configuration
    const vit_config = MultiModal.ViTConfig{
        .image_size = 224,
        .patch_size = 16,
        .embed_dim = 768,
        .num_layers = 12,
        .num_heads = 12,
        .mlp_dim = 3072,
    };

    try testing.expect(vit_config.image_size == 224);
    try testing.expect(vit_config.patch_size == 16);
    try testing.expect(vit_config.embed_dim == 768);

    // Test projection configuration
    const projection_config = MultiModal.ProjectionConfig{
        .vision_dim = 768,
        .text_dim = 4096,
        .projection_type = .mlp,
        .hidden_layers = 2,
        .hidden_dim = 2048,
    };

    try testing.expect(projection_config.vision_dim == 768);
    try testing.expect(projection_config.text_dim == 4096);
    try testing.expect(projection_config.projection_type == .mlp);

    // Test multi-modal configuration
    const multi_modal_config = MultiModal.MultiModalConfig{
        .model_type = .llava,
        .vision_config = vit_config,
        .projection_config = projection_config,
        .max_image_tokens = 576,
    };

    try testing.expect(multi_modal_config.model_type == .llava);
    try testing.expect(multi_modal_config.max_image_tokens == 576);

    _ = allocator;
}

test "patch embedding initialization and forward pass" {
    const allocator = testing.allocator;

    const config = MultiModal.ViTConfig{
        .image_size = 224,
        .patch_size = 16,
        .in_channels = 3,
        .embed_dim = 768,
    };

    var patch_embed = MultiModal.PatchEmbedding.init(allocator, config) catch |err| {
        print("Error initializing patch embedding: {}\n", .{err});
        return err;
    };
    defer patch_embed.deinit(allocator);

    // Test dimensions
    const expected_patch_dim = config.patch_size * config.patch_size * config.in_channels;
    try testing.expect(patch_embed.projection.rows == expected_patch_dim);
    try testing.expect(patch_embed.projection.cols == config.embed_dim);

    // Test forward pass
    const batch_size = 2;
    const num_patches = (config.image_size / config.patch_size) * (config.image_size / config.patch_size);
    const patch_dim = expected_patch_dim;

    const patches = try Matrix.init(allocator, batch_size, num_patches * patch_dim);
    defer patches.deinit(allocator);

    // Fill with test data
    for (patches.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i % 100)) / 100.0;
    }

    const embeddings = patch_embed.forward(patches, allocator, null) catch |err| {
        print("Error in patch embedding forward pass: {}\n", .{err});
        return err;
    };
    defer embeddings.deinit(allocator);

    try testing.expect(embeddings.rows == batch_size);
    try testing.expect(embeddings.cols == num_patches * config.embed_dim);
}

test "position embedding initialization and forward pass" {
    const allocator = testing.allocator;

    const config = MultiModal.ViTConfig{
        .image_size = 224,
        .patch_size = 16,
        .embed_dim = 768,
    };

    var pos_embed = MultiModal.PositionEmbedding.init(allocator, config) catch |err| {
        print("Error initializing position embedding: {}\n", .{err});
        return err;
    };
    defer pos_embed.deinit(allocator);

    // Test dimensions
    const num_patches = (config.image_size / config.patch_size) * (config.image_size / config.patch_size);
    try testing.expect(pos_embed.embeddings.rows == num_patches + 1); // +1 for CLS token
    try testing.expect(pos_embed.embeddings.cols == config.embed_dim);

    // Test forward pass
    const batch_size = 2;
    const seq_len = num_patches + 1;

    const input = try Matrix.init(allocator, batch_size, seq_len * config.embed_dim);
    defer input.deinit(allocator);

    // Fill with test data
    for (input.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i % 50)) / 50.0;
    }

    const positioned = pos_embed.forward(input, allocator) catch |err| {
        print("Error in position embedding forward pass: {}\n", .{err});
        return err;
    };
    defer positioned.deinit(allocator);

    try testing.expect(positioned.rows == input.rows);
    try testing.expect(positioned.cols == input.cols);

    // Verify that position embeddings were added (values should be different)
    var different = false;
    for (0..@min(10, input.data.len)) |i| {
        if (@abs(positioned.data[i] - input.data[i]) > 1e-6) {
            different = true;
            break;
        }
    }
    try testing.expect(different);
}

test "layer normalization" {
    const allocator = testing.allocator;

    var layer_norm = MultiModal.LayerNorm.init(allocator, 768) catch |err| {
        print("Error initializing layer norm: {}\n", .{err});
        return err;
    };
    defer layer_norm.deinit(allocator);

    // Test forward pass
    const batch_size = 2;
    const embed_dim = 768;

    const input = try Matrix.init(allocator, batch_size, embed_dim);
    defer input.deinit(allocator);

    // Fill with test data (different ranges to test normalization)
    for (0..batch_size) |b| {
        for (0..embed_dim) |d| {
            const idx = b * embed_dim + d;
            input.data[idx] = @as(f32, @floatFromInt(d)) + @as(f32, @floatFromInt(b)) * 10.0;
        }
    }

    const normalized = layer_norm.forward(input, allocator) catch |err| {
        print("Error in layer norm forward pass: {}\n", .{err});
        return err;
    };
    defer normalized.deinit(allocator);

    try testing.expect(normalized.rows == input.rows);
    try testing.expect(normalized.cols == input.cols);

    // Check that each row has approximately zero mean and unit variance
    for (0..batch_size) |b| {
        var sum: f32 = 0.0;
        var sum_sq: f32 = 0.0;

        for (0..embed_dim) |d| {
            const val = normalized.data[b * embed_dim + d];
            sum += val;
            sum_sq += val * val;
        }

        const mean = sum / @as(f32, @floatFromInt(embed_dim));
        const variance = sum_sq / @as(f32, @floatFromInt(embed_dim)) - mean * mean;

        try testing.expect(@abs(mean) < 0.1); // Mean should be close to 0
        try testing.expect(@abs(variance - 1.0) < 0.2); // Variance should be close to 1
    }
}

test "MLP forward pass" {
    const allocator = testing.allocator;

    const input_dim = 768;
    const hidden_dim = 2048;
    const dropout = 0.1;

    var mlp = MultiModal.MLP.init(allocator, input_dim, hidden_dim, dropout) catch |err| {
        print("Error initializing MLP: {}\n", .{err});
        return err;
    };
    defer mlp.deinit(allocator);

    // Test forward pass
    const batch_size = 2;

    const input = try Matrix.init(allocator, batch_size, input_dim);
    defer input.deinit(allocator);

    // Fill with test data
    for (input.data, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i % 100)) / 100.0 - 0.5;
    }

    const output = mlp.forward(input, allocator, null) catch |err| {
        print("Error in MLP forward pass: {}\n", .{err});
        return err;
    };
    defer output.deinit(allocator);

    try testing.expect(output.rows == batch_size);
    try testing.expect(output.cols == input_dim);

    // Verify output is different from input (MLP transformation occurred)
    var different = false;
    for (0..@min(10, input.data.len)) |i| {
        if (@abs(output.data[i] - input.data[i]) > 1e-3) {
            different = true;
            break;
        }
    }
    try testing.expect(different);
}

test "multi-modal projection layers" {
    const allocator = testing.allocator;

    // Test linear projection
    {
        const linear_config = MultiModal.ProjectionConfig{
            .vision_dim = 768,
            .text_dim = 4096,
            .projection_type = .linear,
        };

        var linear_proj = MultiModal.MultiModalProjection.init(allocator, linear_config) catch |err| {
            print("Error initializing linear projection: {}\n", .{err});
            return err;
        };
        defer linear_proj.deinit(allocator);

        const vision_features = try Matrix.init(allocator, 2, 768);
        defer vision_features.deinit(allocator);

        for (vision_features.data, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt(i % 100)) / 100.0;
        }

        const projected = linear_proj.forward(vision_features, allocator, null) catch |err| {
            print("Error in linear projection forward pass: {}\n", .{err});
            return err;
        };
        defer projected.deinit(allocator);

        try testing.expect(projected.rows == 2);
        try testing.expect(projected.cols == 4096);
    }

    // Test MLP projection
    {
        const mlp_config = MultiModal.ProjectionConfig{
            .vision_dim = 768,
            .text_dim = 4096,
            .projection_type = .mlp,
            .hidden_layers = 2,
            .hidden_dim = 2048,
        };

        var mlp_proj = MultiModal.MultiModalProjection.init(allocator, mlp_config) catch |err| {
            print("Error initializing MLP projection: {}\n", .{err});
            return err;
        };
        defer mlp_proj.deinit(allocator);

        const vision_features = try Matrix.init(allocator, 2, 768);
        defer vision_features.deinit(allocator);

        for (vision_features.data, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt(i % 100)) / 100.0;
        }

        const projected = mlp_proj.forward(vision_features, allocator, null) catch |err| {
            print("Error in MLP projection forward pass: {}\n", .{err});
            return err;
        };
        defer projected.deinit(allocator);

        try testing.expect(projected.rows == 2);
        try testing.expect(projected.cols == 4096);
    }

    // Test gated projection
    {
        const gated_config = MultiModal.ProjectionConfig{
            .vision_dim = 768,
            .text_dim = 4096,
            .projection_type = .gated,
        };

        var gated_proj = MultiModal.MultiModalProjection.init(allocator, gated_config) catch |err| {
            print("Error initializing gated projection: {}\n", .{err});
            return err;
        };
        defer gated_proj.deinit(allocator);

        const vision_features = try Matrix.init(allocator, 2, 768);
        defer vision_features.deinit(allocator);

        for (vision_features.data, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt(i % 100)) / 100.0;
        }

        const projected = gated_proj.forward(vision_features, allocator, null) catch |err| {
            print("Error in gated projection forward pass: {}\n", .{err});
            return err;
        };
        defer projected.deinit(allocator);

        try testing.expect(projected.rows == 2);
        try testing.expect(projected.cols == 4096);
    }
}

test "vision transformer full model" {
    const allocator = testing.allocator;

    const config = MultiModal.ViTConfig{
        .image_size = 224,
        .patch_size = 16,
        .in_channels = 3,
        .embed_dim = 384, // Smaller for testing
        .num_layers = 2, // Fewer layers for testing
        .num_heads = 6,
        .mlp_dim = 1536,
    };

    var vit = MultiModal.VisionTransformer.init(allocator, config) catch |err| {
        print("Error initializing Vision Transformer: {}\n", .{err});
        return err;
    };
    defer vit.deinit(allocator);

    // Test forward pass
    const batch_size = 1;
    const image_size = config.image_size * config.image_size * config.in_channels;

    const images = try Matrix.init(allocator, batch_size, image_size);
    defer images.deinit(allocator);

    // Fill with realistic image data
    for (images.data, 0..) |*val, i| {
        val.* = (@as(f32, @floatFromInt(i % 256)) / 255.0 - 0.5) * 2.0; // Normalized to [-1, 1]
    }

    const output = vit.forward(images, allocator, null) catch |err| {
        print("Error in Vision Transformer forward pass: {}\n", .{err});
        return err;
    };
    defer output.deinit(allocator);

    try testing.expect(output.rows == batch_size);
    try testing.expect(output.cols == config.embed_dim);

    // Verify output values are reasonable
    var all_zeros = true;
    var all_same = true;
    const first_val = output.data[0];

    for (output.data) |val| {
        if (@abs(val) > 1e-6) all_zeros = false;
        if (@abs(val - first_val) > 1e-6) all_same = false;
    }

    try testing.expect(!all_zeros); // Output should not be all zeros
    try testing.expect(!all_same); // Output should have variation
}

test "multi-modal model integration" {
    const allocator = testing.allocator;

    const projection_config = MultiModal.ProjectionConfig{
        .vision_dim = 384,
        .text_dim = 768,
        .projection_type = .linear,
    };

    const multi_modal_config = MultiModal.MultiModalConfig{
        .model_type = .llava,
        .vision_config = .{
            .image_size = 224,
            .patch_size = 16,
            .embed_dim = 384,
            .num_layers = 2,
            .num_heads = 6,
        },
        .projection_config = projection_config,
    };

    var model = MultiModal.MultiModalModel.init(allocator, multi_modal_config) catch |err| {
        print("Error initializing multi-modal model: {}\n", .{err});
        return err;
    };
    defer model.deinit(allocator);

    // Test image encoding
    const batch_size = 1;
    const image_size = 224 * 224 * 3;

    const images = try Matrix.init(allocator, batch_size, image_size);
    defer images.deinit(allocator);

    for (images.data, 0..) |*val, i| {
        val.* = (@as(f32, @floatFromInt(i % 256)) / 255.0 - 0.5) * 2.0;
    }

    const encoded_features = model.encodeImages(images, allocator, null) catch |err| {
        print("Error in image encoding: {}\n", .{err});
        return err;
    };
    defer encoded_features.deinit(allocator);

    try testing.expect(encoded_features.rows == batch_size);
    try testing.expect(encoded_features.cols == projection_config.text_dim);

    // Test image token creation
    const image_tokens = model.createImageTokens(encoded_features, allocator) catch |err| {
        print("Error creating image tokens: {}\n", .{err});
        return err;
    };
    defer image_tokens.deinit();

    try testing.expect(image_tokens.items.len > 0);
    try testing.expect(image_tokens.items[0] == multi_modal_config.image_start_token_id.?);
    try testing.expect(image_tokens.items[image_tokens.items.len - 1] == multi_modal_config.image_end_token_id.?);

    // Test multi-modal input processing
    const text_tokens = [_]u32{ 1, 2, 3, 4, 5 };
    const processed = model.processMultiModalInput(images, &text_tokens, allocator, null) catch |err| {
        print("Error processing multi-modal input: {}\n", .{err});
        return err;
    };
    defer processed.tokens.deinit();
    if (processed.image_features) |*features| {
        features.deinit(allocator);
    }

    try testing.expect(processed.tokens.items.len > text_tokens.len); // Should include image tokens
    try testing.expect(processed.image_features != null);

    // Test statistics
    const stats = model.getStats();
    try testing.expect(stats.total_images_processed > 0);
}

test "image preprocessing utilities" {
    const allocator = testing.allocator;

    // Test resize
    {
        const input_size = [2]u32{ 256, 256 };
        const target_size = [2]u32{ 224, 224 };
        const channels = 3;

        const input_image = try allocator.alloc(f32, channels * input_size[0] * input_size[1]);
        defer allocator.free(input_image);

        // Fill with test pattern
        for (input_image, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt(i % 256)) / 255.0;
        }

        const resized = MultiModal.ImagePreprocess.resize(
            input_image,
            input_size,
            target_size,
            channels,
            allocator,
        ) catch |err| {
            print("Error resizing image: {}\n", .{err});
            return err;
        };
        defer allocator.free(resized);

        try testing.expect(resized.len == channels * target_size[0] * target_size[1]);

        // Verify values are in reasonable range
        for (resized) |val| {
            try testing.expect(val >= 0.0 and val <= 1.0);
        }
    }

    // Test normalization
    {
        const size = [2]u32{ 224, 224 };
        const channels = 3;
        const mean = [3]f32{ 0.485, 0.456, 0.406 };
        const std_dev = [3]f32{ 0.229, 0.224, 0.225 };

        var image = try allocator.alloc(f32, channels * size[0] * size[1]);
        defer allocator.free(image);

        // Fill with values in [0, 1] range
        for (image, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt(i % 256)) / 255.0;
        }

        const original_values = try allocator.dupe(f32, image);
        defer allocator.free(original_values);

        MultiModal.ImagePreprocess.normalize(image, mean, std_dev, size);

        // Verify normalization was applied (values should be different)
        var different = false;
        for (image, original_values) |norm_val, orig_val| {
            if (@abs(norm_val - orig_val) > 1e-6) {
                different = true;
                break;
            }
        }
        try testing.expect(different);
    }

    // Test center crop
    {
        const input_size = [2]u32{ 256, 256 };
        const crop_size = [2]u32{ 224, 224 };
        const channels = 3;

        const input_image = try allocator.alloc(f32, channels * input_size[0] * input_size[1]);
        defer allocator.free(input_image);

        // Fill with test pattern
        for (input_image, 0..) |*val, i| {
            val.* = @as(f32, @floatFromInt(i % 256)) / 255.0;
        }

        const cropped = MultiModal.ImagePreprocess.centerCrop(
            input_image,
            input_size,
            crop_size,
            channels,
            allocator,
        ) catch |err| {
            print("Error center cropping image: {}\n", .{err});
            return err;
        };
        defer allocator.free(cropped);

        try testing.expect(cropped.len == channels * crop_size[0] * crop_size[1]);

        // Verify values are in reasonable range
        for (cropped) |val| {
            try testing.expect(val >= 0.0 and val <= 1.0);
        }
    }
}

test "multi-modal model statistics" {
    var stats = MultiModal.MultiModalStats.init();

    try testing.expect(stats.total_images_processed == 0);
    try testing.expect(stats.total_processing_time == 0);
    try testing.expect(stats.average_processing_time == 0.0);

    // Simulate processing
    stats.total_images_processed = 10;
    stats.total_processing_time = 50000; // 50ms
    stats.updateAverageProcessingTime();

    try testing.expect(stats.average_processing_time == 5000.0); // 5ms per image

    stats.vision_encoder_calls = 5;
    stats.projection_calls = 5;

    try testing.expect(stats.vision_encoder_calls == 5);
    try testing.expect(stats.projection_calls == 5);
}

test "multi-modal educational concepts" {
    // Test that educational content is accessible
    const concepts = MultiModal.MultiModalEducational.concepts;
    const architectures = MultiModal.MultiModalEducational.architectures;
    const techniques = MultiModal.MultiModalEducational.techniques;

    try testing.expect(concepts.vision_transformers.len > 0);
    try testing.expect(architectures.llava.len > 0);
    try testing.expect(techniques.patch_size_selection.len > 0);

    // Verify educational content contains key terms
    try testing.expect(std.mem.indexOf(u8, concepts.vision_transformers, "transformer") != null);
    try testing.expect(std.mem.indexOf(u8, concepts.patch_embedding, "patches") != null);
    try testing.expect(std.mem.indexOf(u8, architectures.clip, "contrastive") != null);
}

// Integration test that demonstrates the full multi-modal pipeline
test "full multi-modal pipeline demonstration" {
    const allocator = testing.allocator;

    print("\n=== Running Full Multi-Modal Pipeline Test ===\n");

    // Configuration
    const projection_config = MultiModal.ProjectionConfig{
        .vision_dim = 384,
        .text_dim = 1024,
        .projection_type = .mlp,
        .hidden_layers = 1,
        .hidden_dim = 512,
    };

    const config = MultiModal.MultiModalConfig{
        .model_type = .llava,
        .vision_config = .{
            .image_size = 224,
            .patch_size = 16,
            .embed_dim = 384,
            .num_layers = 2,
            .num_heads = 6,
        },
        .projection_config = projection_config,
    };

    // Initialize model
    var model = MultiModal.MultiModalModel.init(allocator, config) catch |err| {
        print("Failed to initialize model: {}\n", .{err});
        return err;
    };
    defer model.deinit(allocator);

    print("✓ Multi-modal model initialized successfully\n");

    // Create test image
    const batch_size = 2;
    const image_size = 224 * 224 * 3;
    const images = try Matrix.init(allocator, batch_size, image_size);
    defer images.deinit(allocator);

    // Fill with realistic test data
    for (images.data, 0..) |*val, i| {
        val.* = (@as(f32, @floatFromInt(i % 256)) / 255.0 - 0.485) / 0.229; // ImageNet normalization
    }

    print("✓ Test images created ({} images)\n", .{batch_size});

    // Process images
    const image_features = model.encodeImages(images, allocator, null) catch |err| {
        print("Failed to encode images: {}\n", .{err});
        return err;
    };
    defer image_features.deinit(allocator);

    print("✓ Images encoded to {} features\n", .{image_features.cols});

    // Create text input
    const text_tokens = [_]u32{ 1, 15339, 338, 445, 3273, 29973, 2 }; // "What is this image?"

    // Process multi-modal input
    const processed = model.processMultiModalInput(images, &text_tokens, allocator, null) catch |err| {
        print("Failed to process multi-modal input: {}\n", .{err});
        return err;
    };
    defer processed.tokens.deinit();
    if (processed.image_features) |*features| {
        features.deinit(allocator);
    }

    print("✓ Multi-modal input processed ({} total tokens)\n", .{processed.tokens.items.len});

    // Verify statistics
    const stats = model.getStats();
    try testing.expect(stats.total_images_processed == batch_size * 2); // 2 calls to encodeImages

    print("✓ Statistics verified (processed {} images)\n", .{stats.total_images_processed});
    print("✓ Full multi-modal pipeline test passed!\n");
}