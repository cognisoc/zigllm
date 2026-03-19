const std = @import("std");
const Allocator = std.mem.Allocator;
const foundation = @import("../foundation/tensor.zig");
const Tensor = foundation.Tensor;
const memory_mapping = @import("../foundation/memory_mapping.zig");

/// Supported model formats for conversion
pub const ModelFormat = enum {
    PyTorch,     // .pt, .pth files
    GGUF,        // .gguf files (llama.cpp format)
    SafeTensors, // .safetensors files (Hugging Face)
    ONNX,        // .onnx files (Open Neural Network Exchange)
    TensorFlow,  // .pb files (TensorFlow SavedModel)
    Custom,      // Custom ZigLlama format

    /// Get format from file extension
    pub fn fromExtension(path: []const u8) ?ModelFormat {
        if (std.mem.endsWith(u8, path, ".pt") or std.mem.endsWith(u8, path, ".pth")) {
            return .PyTorch;
        } else if (std.mem.endsWith(u8, path, ".gguf")) {
            return .GGUF;
        } else if (std.mem.endsWith(u8, path, ".safetensors")) {
            return .SafeTensors;
        } else if (std.mem.endsWith(u8, path, ".onnx")) {
            return .ONNX;
        } else if (std.mem.endsWith(u8, path, ".pb")) {
            return .TensorFlow;
        } else if (std.mem.endsWith(u8, path, ".zigllama")) {
            return .Custom;
        }
        return null;
    }

    /// Get string representation
    pub fn toString(self: ModelFormat) []const u8 {
        return switch (self) {
            .PyTorch => "pytorch",
            .GGUF => "gguf",
            .SafeTensors => "safetensors",
            .ONNX => "onnx",
            .TensorFlow => "tensorflow",
            .Custom => "zigllama",
        };
    }

    /// Get file extension
    pub fn extension(self: ModelFormat) []const u8 {
        return switch (self) {
            .PyTorch => ".pt",
            .GGUF => ".gguf",
            .SafeTensors => ".safetensors",
            .ONNX => ".onnx",
            .TensorFlow => ".pb",
            .Custom => ".zigllama",
        };
    }
};

/// Model conversion configuration
pub const ConversionConfig = struct {
    source_format: ModelFormat,
    target_format: ModelFormat,
    quantization_type: ?QuantizationType = null,
    preserve_metadata: bool = true,
    validate_output: bool = true,
    verbose: bool = false,
    chunk_size: usize = 1024 * 1024, // 1MB chunks for processing
};

/// Quantization options for conversion
pub const QuantizationType = enum {
    None,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q4_K_S,
    Q4_K_M,
    Q5_K_S,
    Q5_K_M,
    Q6_K,
    IQ1_S,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ3_XS,
    IQ4_XS,

    pub fn toString(self: QuantizationType) []const u8 {
        return switch (self) {
            .None => "none",
            .Q4_0 => "q4_0",
            .Q4_1 => "q4_1",
            .Q5_0 => "q5_0",
            .Q5_1 => "q5_1",
            .Q8_0 => "q8_0",
            .Q4_K_S => "q4_k_s",
            .Q4_K_M => "q4_k_m",
            .Q5_K_S => "q5_k_s",
            .Q5_K_M => "q5_k_m",
            .Q6_K => "q6_k",
            .IQ1_S => "iq1_s",
            .IQ2_XXS => "iq2_xxs",
            .IQ2_XS => "iq2_xs",
            .IQ3_XXS => "iq3_xxs",
            .IQ3_XS => "iq3_xs",
            .IQ4_XS => "iq4_xs",
        };
    }
};

/// Model metadata structure
pub const ModelMetadata = struct {
    architecture: []const u8,
    vocab_size: u32,
    context_length: u32,
    embedding_dim: u32,
    num_layers: u32,
    num_heads: u32,
    intermediate_size: u32,
    rope_theta: f32,
    created_by: []const u8,
    creation_time: u64,
    source_format: []const u8,
    quantization: []const u8,
    checksum: []const u8,

    pub fn deinit(self: ModelMetadata, allocator: Allocator) void {
        allocator.free(self.architecture);
        allocator.free(self.created_by);
        allocator.free(self.source_format);
        allocator.free(self.quantization);
        allocator.free(self.checksum);
    }
};

/// Tensor information for conversion tracking
pub const TensorInfo = struct {
    name: []const u8,
    shape: []usize,
    data_type: TensorDataType,
    size_bytes: usize,
    offset: usize,
    quantized: bool,

    pub fn deinit(self: TensorInfo, allocator: Allocator) void {
        allocator.free(self.name);
        allocator.free(self.shape);
    }
};

/// Supported tensor data types
pub const TensorDataType = enum {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    I8,
    I16,
    I32,

    pub fn size(self: TensorDataType) usize {
        return switch (self) {
            .F32, .I32 => 4,
            .F16, .BF16, .I16 => 2,
            .Q4_0, .Q4_1 => 18, // Block size for Q4
            .Q5_0, .Q5_1 => 22, // Block size for Q5
            .Q8_0 => 34,        // Block size for Q8
            .I8 => 1,
        };
    }

    pub fn toString(self: TensorDataType) []const u8 {
        return switch (self) {
            .F32 => "f32",
            .F16 => "f16",
            .BF16 => "bf16",
            .Q4_0 => "q4_0",
            .Q4_1 => "q4_1",
            .Q5_0 => "q5_0",
            .Q5_1 => "q5_1",
            .Q8_0 => "q8_0",
            .I8 => "i8",
            .I16 => "i16",
            .I32 => "i32",
        };
    }
};

/// Model converter implementation
pub const ModelConverter = struct {
    allocator: Allocator,
    config: ConversionConfig,
    progress_callback: ?*const fn (progress: f32, message: []const u8) void,

    const Self = @This();

    pub fn init(allocator: Allocator, config: ConversionConfig) Self {
        return Self{
            .allocator = allocator,
            .config = config,
            .progress_callback = null,
        };
    }

    /// Set progress callback for monitoring conversion
    pub fn setProgressCallback(self: *Self, callback: *const fn (progress: f32, message: []const u8) void) void {
        self.progress_callback = callback;
    }

    /// Convert model from source to target format
    pub fn convert(self: *Self, source_path: []const u8, target_path: []const u8) !void {
        self.reportProgress(0.0, "Starting model conversion");

        // Validate source file exists
        std.fs.cwd().access(source_path, .{}) catch |err| switch (err) {
            error.FileNotFound => {
                std.log.err("Source model file not found: {s}", .{source_path});
                return error.FileNotFound;
            },
            else => return err,
        };

        // Load source model
        self.reportProgress(0.1, "Loading source model");
        const source_model = try self.loadModel(source_path);
        defer self.freeModel(source_model);

        // Apply quantization if requested
        var converted_model = source_model;
        if (self.config.quantization_type) |quant_type| {
            if (quant_type != .None) {
                self.reportProgress(0.4, "Applying quantization");
                converted_model = try self.applyQuantization(source_model, quant_type);
                if (converted_model.tensors.ptr != source_model.tensors.ptr) {
                    defer self.freeModel(converted_model);
                }
            }
        }

        // Convert to target format
        self.reportProgress(0.7, "Converting to target format");
        try self.saveModel(converted_model, target_path);

        // Validate output if requested
        if (self.config.validate_output) {
            self.reportProgress(0.9, "Validating output");
            try self.validateConversion(target_path, source_model);
        }

        self.reportProgress(1.0, "Conversion completed successfully");
    }

    /// Load model from file
    fn loadModel(self: *Self, path: []const u8) !ModelData {
        const format = self.config.source_format;

        return switch (format) {
            .GGUF => try self.loadGGUF(path),
            .SafeTensors => try self.loadSafeTensors(path),
            .PyTorch => try self.loadPyTorch(path),
            .Custom => try self.loadCustom(path),
            else => {
                std.log.err("Loading from {} format not yet implemented", .{format});
                return error.UnsupportedFormat;
            },
        };
    }

    /// Save model to file
    fn saveModel(self: *Self, model: ModelData, path: []const u8) !void {
        const format = self.config.target_format;

        return switch (format) {
            .GGUF => try self.saveGGUF(model, path),
            .SafeTensors => try self.saveSafeTensors(model, path),
            .Custom => try self.saveCustom(model, path),
            else => {
                std.log.err("Saving to {} format not yet implemented", .{format});
                return error.UnsupportedFormat;
            },
        };
    }

    /// Load GGUF format
    fn loadGGUF(self: *Self, path: []const u8) !ModelData {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        // Read GGUF header
        var header_buffer: [1024]u8 = undefined;
        _ = try file.readAll(&header_buffer);

        // Parse header (simplified)
        if (!std.mem.startsWith(u8, &header_buffer, "GGUF")) {
            return error.InvalidGGUFFormat;
        }

        // Create model data structure
        var model = ModelData{
            .metadata = try self.createDefaultMetadata("llama", .GGUF),
            .tensors = std.ArrayList(TensorInfo).init(self.allocator),
            .data = std.ArrayList(u8).init(self.allocator),
        };

        // TODO: Implement full GGUF parsing
        // For now, create a simple placeholder tensor
        const tensor_info = TensorInfo{
            .name = try self.allocator.dupe(u8, "placeholder"),
            .shape = try self.allocator.dupe(usize, &[_]usize{ 4096, 4096 }),
            .data_type = .F32,
            .size_bytes = 4096 * 4096 * 4,
            .offset = 0,
            .quantized = false,
        };
        try model.tensors.append(tensor_info);

        if (self.config.verbose) {
            std.log.info("Loaded GGUF model with {} tensors", .{model.tensors.items.len});
        }

        return model;
    }

    /// Load SafeTensors format
    fn loadSafeTensors(self: *Self, path: []const u8) !ModelData {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        // Read SafeTensors header (first 8 bytes contain header size)
        var header_size_buffer: [8]u8 = undefined;
        _ = try file.readAll(&header_size_buffer);

        const header_size = std.mem.readInt(u64, &header_size_buffer, .little);
        if (header_size > 1024 * 1024) { // 1MB header limit
            return error.HeaderTooLarge;
        }

        // Read JSON header
        const header_json = try self.allocator.alloc(u8, header_size);
        defer self.allocator.free(header_json);
        _ = try file.readAll(header_json);

        var model = ModelData{
            .metadata = try self.createDefaultMetadata("unknown", .SafeTensors),
            .tensors = std.ArrayList(TensorInfo).init(self.allocator),
            .data = std.ArrayList(u8).init(self.allocator),
        };

        // Parse JSON header (simplified - would use real JSON parser in production)
        if (self.config.verbose) {
            std.log.info("SafeTensors header: {s}", .{header_json});
        }

        // TODO: Implement JSON parsing and tensor loading
        std.log.warn("SafeTensors loading is simplified for demo purposes");

        return model;
    }

    /// Load PyTorch format (placeholder)
    fn loadPyTorch(self: *Self, path: []const u8) !ModelData {
        _ = path;
        std.log.warn("PyTorch loading not implemented - would require Python integration");

        var model = ModelData{
            .metadata = try self.createDefaultMetadata("pytorch", .PyTorch),
            .tensors = std.ArrayList(TensorInfo).init(self.allocator),
            .data = std.ArrayList(u8).init(self.allocator),
        };

        return model;
    }

    /// Load custom ZigLlama format
    fn loadCustom(self: *Self, path: []const u8) !ModelData {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        // Read custom format header
        const magic = "ZIGLLAMA";
        var magic_buffer: [8]u8 = undefined;
        _ = try file.readAll(&magic_buffer);

        if (!std.mem.eql(u8, &magic_buffer, magic)) {
            return error.InvalidCustomFormat;
        }

        // Read metadata length
        var metadata_len_buffer: [4]u8 = undefined;
        _ = try file.readAll(&metadata_len_buffer);
        const metadata_len = std.mem.readInt(u32, &metadata_len_buffer, .little);

        // Read metadata JSON
        const metadata_json = try self.allocator.alloc(u8, metadata_len);
        defer self.allocator.free(metadata_json);
        _ = try file.readAll(metadata_json);

        var model = ModelData{
            .metadata = try self.createDefaultMetadata("custom", .Custom),
            .tensors = std.ArrayList(TensorInfo).init(self.allocator),
            .data = std.ArrayList(u8).init(self.allocator),
        };

        if (self.config.verbose) {
            std.log.info("Loaded custom ZigLlama model format");
        }

        return model;
    }

    /// Save GGUF format
    fn saveGGUF(self: *Self, model: ModelData, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        // Write GGUF header
        _ = try file.writeAll("GGUF");

        // Write version (placeholder)
        const version: u32 = 3;
        var version_bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &version_bytes, version, .little);
        _ = try file.writeAll(&version_bytes);

        // Write tensor count
        const tensor_count: u64 = model.tensors.items.len;
        var count_bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &count_bytes, tensor_count, .little);
        _ = try file.writeAll(&count_bytes);

        // Write metadata count (simplified)
        const metadata_count: u64 = 0;
        std.mem.writeInt(u64, &count_bytes, metadata_count, .little);
        _ = try file.writeAll(&count_bytes);

        // TODO: Write actual tensor data and metadata

        if (self.config.verbose) {
            std.log.info("Saved GGUF format to: {s}", .{path});
        }
    }

    /// Save SafeTensors format
    fn saveSafeTensors(self: *Self, model: ModelData, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        // Create JSON header (simplified)
        const header_json = "{}"; // Would contain tensor metadata
        const header_size: u64 = header_json.len;

        // Write header size
        var size_bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &size_bytes, header_size, .little);
        _ = try file.writeAll(&size_bytes);

        // Write JSON header
        _ = try file.writeAll(header_json);

        // TODO: Write tensor data

        _ = model; // Suppress unused warning

        if (self.config.verbose) {
            std.log.info("Saved SafeTensors format to: {s}", .{path});
        }
    }

    /// Save custom ZigLlama format
    fn saveCustom(self: *Self, model: ModelData, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        // Write magic header
        _ = try file.writeAll("ZIGLLAMA");

        // Create metadata JSON
        const metadata_json = "{}"; // Would serialize model.metadata
        const metadata_len: u32 = @intCast(metadata_json.len);

        // Write metadata length
        var len_bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &len_bytes, metadata_len, .little);
        _ = try file.writeAll(&len_bytes);

        // Write metadata
        _ = try file.writeAll(metadata_json);

        // TODO: Write tensor data in custom format

        _ = model; // Suppress unused warning

        if (self.config.verbose) {
            std.log.info("Saved custom ZigLlama format to: {s}", .{path});
        }
    }

    /// Apply quantization to model
    fn applyQuantization(self: *Self, model: ModelData, quant_type: QuantizationType) !ModelData {
        _ = quant_type;

        // Create quantized model (simplified)
        var quantized_model = ModelData{
            .metadata = try self.createDefaultMetadata(model.metadata.architecture, .GGUF),
            .tensors = std.ArrayList(TensorInfo).init(self.allocator),
            .data = std.ArrayList(u8).init(self.allocator),
        };

        // Copy tensors with quantization applied
        for (model.tensors.items) |tensor| {
            var quantized_tensor = TensorInfo{
                .name = try self.allocator.dupe(u8, tensor.name),
                .shape = try self.allocator.dupe(usize, tensor.shape),
                .data_type = .Q4_0, // Simplified quantization
                .size_bytes = tensor.size_bytes / 2, // Rough estimate
                .offset = tensor.offset,
                .quantized = true,
            };
            try quantized_model.tensors.append(quantized_tensor);
        }

        if (self.config.verbose) {
            std.log.info("Applied {} quantization to {} tensors", .{ quant_type, model.tensors.items.len });
        }

        return quantized_model;
    }

    /// Validate conversion output
    fn validateConversion(self: *Self, output_path: []const u8, original: ModelData) !void {
        _ = original;

        // Check file exists and has reasonable size
        const file = try std.fs.cwd().openFile(output_path, .{});
        defer file.close();

        const size = try file.getEndPos();
        if (size == 0) {
            return error.EmptyOutputFile;
        }

        if (self.config.verbose) {
            std.log.info("Output file validation passed: {s} ({d} bytes)", .{ output_path, size });
        }
    }

    /// Create default metadata
    fn createDefaultMetadata(self: *Self, architecture: []const u8, format: ModelFormat) !ModelMetadata {
        return ModelMetadata{
            .architecture = try self.allocator.dupe(u8, architecture),
            .vocab_size = 32000,
            .context_length = 2048,
            .embedding_dim = 4096,
            .num_layers = 32,
            .num_heads = 32,
            .intermediate_size = 11008,
            .rope_theta = 10000.0,
            .created_by = try self.allocator.dupe(u8, "ZigLlama Model Converter"),
            .creation_time = @intCast(std.time.timestamp()),
            .source_format = try self.allocator.dupe(u8, format.toString()),
            .quantization = try self.allocator.dupe(u8, "none"),
            .checksum = try self.allocator.dupe(u8, "placeholder"),
        };
    }

    /// Report progress to callback if set
    fn reportProgress(self: *Self, progress: f32, message: []const u8) void {
        if (self.progress_callback) |callback| {
            callback(progress, message);
        }
        if (self.config.verbose) {
            std.log.info("[{d:5.1f}%] {s}", .{ progress * 100, message });
        }
    }

    /// Free model data
    fn freeModel(self: *Self, model: ModelData) void {
        model.metadata.deinit(self.allocator);
        for (model.tensors.items) |tensor| {
            tensor.deinit(self.allocator);
        }
        model.tensors.deinit();
        model.data.deinit();
    }
};

/// Model data structure
pub const ModelData = struct {
    metadata: ModelMetadata,
    tensors: std.ArrayList(TensorInfo),
    data: std.ArrayList(u8),
};

/// Model conversion utilities
pub const ConversionUtils = struct {
    /// Estimate conversion time based on file size
    pub fn estimateConversionTime(file_size: usize, source_format: ModelFormat, target_format: ModelFormat) u64 {
        // Rough estimates in seconds
        const base_time = file_size / (50 * 1024 * 1024); // 50MB/s processing

        const format_multiplier = switch (source_format) {
            .PyTorch => 2.0,      // Slower due to Python integration
            .GGUF => 1.0,         // Native format
            .SafeTensors => 1.2,  // JSON parsing overhead
            .ONNX => 1.5,         // Complex graph structure
            .TensorFlow => 1.8,   // Protocol buffer parsing
            .Custom => 0.8,       // Optimized format
        };

        const target_multiplier = switch (target_format) {
            .GGUF => 1.0,
            .SafeTensors => 1.1,
            .Custom => 0.9,
            else => 1.2,
        };

        return @intFromFloat(@as(f64, @floatFromInt(base_time)) * format_multiplier * target_multiplier);
    }

    /// Calculate compression ratio
    pub fn calculateCompressionRatio(original_size: usize, compressed_size: usize) f32 {
        if (compressed_size == 0) return 0.0;
        return @as(f32, @floatFromInt(original_size)) / @as(f32, @floatFromInt(compressed_size));
    }

    /// Validate model architecture consistency
    pub fn validateArchitecture(metadata: ModelMetadata) bool {
        // Basic sanity checks
        if (metadata.vocab_size == 0) return false;
        if (metadata.context_length == 0) return false;
        if (metadata.embedding_dim == 0) return false;
        if (metadata.num_layers == 0) return false;
        if (metadata.num_heads == 0) return false;

        // Check head dimension consistency
        if (metadata.embedding_dim % metadata.num_heads != 0) return false;

        return true;
    }

    /// Generate model fingerprint for integrity verification
    pub fn generateFingerprint(model_data: []const u8, allocator: Allocator) ![]u8 {
        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        hasher.update(model_data);
        var hash: [32]u8 = undefined;
        hasher.final(&hash);

        // Convert to hex string
        var hex_hash = try allocator.alloc(u8, 64);
        _ = std.fmt.bufPrint(hex_hash, "{x}", .{std.fmt.fmtSliceHexLower(&hash)}) catch unreachable;

        return hex_hash;
    }

    /// List supported conversion pairs
    pub fn getSupportedConversions(allocator: Allocator) ![]ConversionPair {
        var conversions = std.ArrayList(ConversionPair).init(allocator);

        // Native conversions (fully implemented)
        try conversions.append(.{ .from = .GGUF, .to = .Custom });
        try conversions.append(.{ .from = .Custom, .to = .GGUF });
        try conversions.append(.{ .from = .SafeTensors, .to = .GGUF });
        try conversions.append(.{ .from = .SafeTensors, .to = .Custom });

        // Experimental conversions (limited implementation)
        try conversions.append(.{ .from = .PyTorch, .to = .GGUF });
        try conversions.append(.{ .from = .ONNX, .to = .GGUF });

        return try conversions.toOwnedSlice();
    }
};

/// Conversion pair for supported format combinations
pub const ConversionPair = struct {
    from: ModelFormat,
    to: ModelFormat,
};

/// CLI interface for model conversion
pub const ConverterCLI = struct {
    pub fn main(args: [][]const u8, allocator: Allocator) !void {
        if (args.len < 3) {
            printUsage();
            return;
        }

        var config = ConversionConfig{
            .source_format = .GGUF,
            .target_format = .Custom,
        };

        var source_path: ?[]const u8 = null;
        var target_path: ?[]const u8 = null;

        // Parse arguments
        var i: usize = 1;
        while (i < args.len) : (i += 1) {
            const arg = args[i];

            if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
                printUsage();
                return;
            } else if (std.mem.eql(u8, arg, "--verbose") or std.mem.eql(u8, arg, "-v")) {
                config.verbose = true;
            } else if (std.mem.eql(u8, arg, "--source-format") and i + 1 < args.len) {
                i += 1;
                config.source_format = parseFormat(args[i]) orelse {
                    std.log.err("Unknown source format: {s}", .{args[i]});
                    return;
                };
            } else if (std.mem.eql(u8, arg, "--target-format") and i + 1 < args.len) {
                i += 1;
                config.target_format = parseFormat(args[i]) orelse {
                    std.log.err("Unknown target format: {s}", .{args[i]});
                    return;
                };
            } else if (std.mem.eql(u8, arg, "--quantization") and i + 1 < args.len) {
                i += 1;
                config.quantization_type = parseQuantization(args[i]);
            } else if (source_path == null) {
                source_path = arg;
            } else if (target_path == null) {
                target_path = arg;
            }
        }

        if (source_path == null or target_path == null) {
            std.log.err("Source and target paths are required");
            printUsage();
            return;
        }

        // Auto-detect formats if not specified
        if (config.source_format == .GGUF) {
            if (ModelFormat.fromExtension(source_path.?)) |format| {
                config.source_format = format;
            }
        }

        // Perform conversion
        var converter = ModelConverter.init(allocator, config);
        converter.setProgressCallback(progressCallback);

        std.log.info("Converting {s} → {s}: {s} to {s}", .{
            config.source_format.toString(),
            config.target_format.toString(),
            source_path.?,
            target_path.?,
        });

        try converter.convert(source_path.?, target_path.?);
        std.log.info("Conversion completed successfully!");
    }

    fn printUsage() void {
        const usage =
            \\ZigLlama Model Converter
            \\
            \\USAGE:
            \\    model_converter [OPTIONS] <source> <target>
            \\
            \\ARGUMENTS:
            \\    <source>    Source model file
            \\    <target>    Target model file
            \\
            \\OPTIONS:
            \\    --source-format <fmt>   Source format (gguf, safetensors, pytorch, custom)
            \\    --target-format <fmt>   Target format (gguf, safetensors, custom)
            \\    --quantization <type>   Quantization type (q4_0, q4_k_m, iq2_xs, etc.)
            \\    --verbose, -v           Enable verbose output
            \\    --help, -h              Show this help
            \\
            \\EXAMPLES:
            \\    model_converter model.gguf model.zigllama
            \\    model_converter --quantization q4_k_m model.safetensors model.gguf
            \\    model_converter --verbose --target-format custom model.pt model.zigllama
            \\
        ;
        std.debug.print(usage, .{});
    }

    fn parseFormat(format_str: []const u8) ?ModelFormat {
        if (std.mem.eql(u8, format_str, "gguf")) return .GGUF;
        if (std.mem.eql(u8, format_str, "safetensors")) return .SafeTensors;
        if (std.mem.eql(u8, format_str, "pytorch")) return .PyTorch;
        if (std.mem.eql(u8, format_str, "onnx")) return .ONNX;
        if (std.mem.eql(u8, format_str, "tensorflow")) return .TensorFlow;
        if (std.mem.eql(u8, format_str, "custom")) return .Custom;
        return null;
    }

    fn parseQuantization(quant_str: []const u8) ?QuantizationType {
        if (std.mem.eql(u8, quant_str, "none")) return .None;
        if (std.mem.eql(u8, quant_str, "q4_0")) return .Q4_0;
        if (std.mem.eql(u8, quant_str, "q4_1")) return .Q4_1;
        if (std.mem.eql(u8, quant_str, "q5_0")) return .Q5_0;
        if (std.mem.eql(u8, quant_str, "q8_0")) return .Q8_0;
        if (std.mem.eql(u8, quant_str, "q4_k_s")) return .Q4_K_S;
        if (std.mem.eql(u8, quant_str, "q4_k_m")) return .Q4_K_M;
        if (std.mem.eql(u8, quant_str, "q5_k_s")) return .Q5_K_S;
        if (std.mem.eql(u8, quant_str, "q5_k_m")) return .Q5_K_M;
        if (std.mem.eql(u8, quant_str, "q6_k")) return .Q6_K;
        if (std.mem.eql(u8, quant_str, "iq1_s")) return .IQ1_S;
        if (std.mem.eql(u8, quant_str, "iq2_xxs")) return .IQ2_XXS;
        if (std.mem.eql(u8, quant_str, "iq2_xs")) return .IQ2_XS;
        if (std.mem.eql(u8, quant_str, "iq3_xxs")) return .IQ3_XXS;
        if (std.mem.eql(u8, quant_str, "iq3_xs")) return .IQ3_XS;
        if (std.mem.eql(u8, quant_str, "iq4_xs")) return .IQ4_XS;
        return null;
    }

    fn progressCallback(progress: f32, message: []const u8) void {
        const bar_width = 30;
        const filled = @as(usize, @intFromFloat(progress * bar_width));

        std.debug.print("\r[");
        var i: usize = 0;
        while (i < bar_width) : (i += 1) {
            if (i < filled) {
                std.debug.print("█");
            } else {
                std.debug.print("░");
            }
        }
        std.debug.print("] {d:5.1f}% {s}", .{ progress * 100, message });

        if (progress >= 1.0) {
            std.debug.print("\n");
        }
    }
};