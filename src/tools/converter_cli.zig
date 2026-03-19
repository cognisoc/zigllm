const std = @import("std");
const model_converter = @import("model_converter.zig");
const ModelFormat = model_converter.ModelFormat;
const ConversionConfig = model_converter.ConversionConfig;
const ModelConverter = model_converter.ModelConverter;
const QuantizationType = model_converter.QuantizationType;
const ConversionUtils = model_converter.ConversionUtils;

/// ZigLlama Model Converter CLI
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        try printUsage();
        return;
    }

    // Parse subcommand
    const subcommand = args[1];

    if (std.mem.eql(u8, subcommand, "convert")) {
        try handleConvertCommand(args[2..], allocator);
    } else if (std.mem.eql(u8, subcommand, "info")) {
        try handleInfoCommand(args[2..], allocator);
    } else if (std.mem.eql(u8, subcommand, "list-formats")) {
        try handleListFormatsCommand(allocator);
    } else if (std.mem.eql(u8, subcommand, "supported")) {
        try handleSupportedCommand(allocator);
    } else if (std.mem.eql(u8, subcommand, "help") or std.mem.eql(u8, subcommand, "--help")) {
        try printUsage();
    } else {
        std.log.err("Unknown subcommand: {s}", .{subcommand});
        try printUsage();
        return;
    }
}

/// Handle convert subcommand
fn handleConvertCommand(args: [][]const u8, allocator: std.mem.Allocator) !void {
    if (args.len < 2) {
        std.log.err("Convert requires source and target paths");
        try printConvertUsage();
        return;
    }

    var config = ConversionConfig{
        .source_format = .GGUF, // Will auto-detect
        .target_format = .Custom,
        .quantization_type = null,
        .preserve_metadata = true,
        .validate_output = true,
        .verbose = false,
    };

    var source_path: ?[]const u8 = null;
    var target_path: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];

        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try printConvertUsage();
            return;
        } else if (std.mem.eql(u8, arg, "--verbose") or std.mem.eql(u8, arg, "-v")) {
            config.verbose = true;
        } else if (std.mem.eql(u8, arg, "--no-validate")) {
            config.validate_output = false;
        } else if (std.mem.eql(u8, arg, "--no-metadata")) {
            config.preserve_metadata = false;
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
            config.quantization_type = parseQuantization(args[i]) orelse {
                std.log.err("Unknown quantization type: {s}", .{args[i]});
                return;
            };
        } else if (std.mem.eql(u8, arg, "--chunk-size") and i + 1 < args.len) {
            i += 1;
            config.chunk_size = std.fmt.parseInt(usize, args[i], 10) catch |err| {
                std.log.err("Invalid chunk size: {s} ({})", .{ args[i], err });
                return;
            };
        } else if (source_path == null) {
            source_path = arg;
        } else if (target_path == null) {
            target_path = arg;
        } else {
            std.log.warn("Ignoring extra argument: {s}", .{arg});
        }
    }

    if (source_path == null or target_path == null) {
        std.log.err("Source and target paths are required");
        try printConvertUsage();
        return;
    }

    // Auto-detect source format if not explicitly set
    if (ModelFormat.fromExtension(source_path.?)) |detected_format| {
        if (config.source_format == .GGUF) { // Default value, not explicitly set
            config.source_format = detected_format;
            if (config.verbose) {
                std.log.info("Auto-detected source format: {s}", .{detected_format.toString()});
            }
        }
    }

    // Auto-detect target format if not explicitly set
    if (ModelFormat.fromExtension(target_path.?)) |detected_format| {
        if (config.target_format == .Custom) { // Check if it's still default
            config.target_format = detected_format;
            if (config.verbose) {
                std.log.info("Auto-detected target format: {s}", .{detected_format.toString()});
            }
        }
    }

    // Show conversion info
    std.log.info("🚀 ZigLlama Model Converter");
    std.log.info("===========================");
    std.log.info("📁 Source: {s} ({})", .{ source_path.?, config.source_format });
    std.log.info("🎯 Target: {s} ({})", .{ target_path.?, config.target_format });

    if (config.quantization_type) |quant| {
        std.log.info("🗜️  Quantization: {s}", .{quant.toString()});
    }

    // Estimate conversion time
    const file_stat = std.fs.cwd().statFile(source_path.?) catch |err| {
        std.log.err("Failed to stat source file: {}", .{err});
        return;
    };

    const estimated_time = ConversionUtils.estimateConversionTime(
        file_stat.size,
        config.source_format,
        config.target_format,
    );

    std.log.info("📊 File size: {d:.1f} MB", .{@as(f64, @floatFromInt(file_stat.size)) / (1024.0 * 1024.0)});
    std.log.info("⏱️  Estimated time: {d}s", .{estimated_time});
    std.log.info("");

    // Perform conversion
    var converter = ModelConverter.init(allocator, config);
    converter.setProgressCallback(progressCallback);

    const start_time = std.time.milliTimestamp();

    converter.convert(source_path.?, target_path.?) catch |err| {
        std.log.err("❌ Conversion failed: {}", .{err});
        return;
    };

    const end_time = std.time.milliTimestamp();
    const actual_time = @as(f64, @floatFromInt(end_time - start_time)) / 1000.0;

    // Show results
    const target_stat = std.fs.cwd().statFile(target_path.?) catch |err| {
        std.log.warn("Failed to stat output file: {}", .{err});
        std.log.info("✅ Conversion completed in {d:.1f}s", .{actual_time});
        return;
    };

    const compression_ratio = ConversionUtils.calculateCompressionRatio(file_stat.size, target_stat.size);

    std.log.info("");
    std.log.info("🎉 Conversion Results");
    std.log.info("====================");
    std.log.info("✅ Status: Success");
    std.log.info("⏱️  Time: {d:.1f}s (estimated: {d}s)", .{ actual_time, estimated_time });
    std.log.info("📁 Input size: {d:.1f} MB", .{@as(f64, @floatFromInt(file_stat.size)) / (1024.0 * 1024.0)});
    std.log.info("📄 Output size: {d:.1f} MB", .{@as(f64, @floatFromInt(target_stat.size)) / (1024.0 * 1024.0)});
    std.log.info("📈 Compression ratio: {d:.1f}x", .{compression_ratio});

    if (compression_ratio > 1.0) {
        const saved_mb = (@as(f64, @floatFromInt(file_stat.size)) - @as(f64, @floatFromInt(target_stat.size))) / (1024.0 * 1024.0);
        std.log.info("💾 Space saved: {d:.1f} MB", .{saved_mb});
    }
}

/// Handle info subcommand
fn handleInfoCommand(args: [][]const u8, allocator: std.mem.Allocator) !void {
    if (args.len == 0) {
        std.log.err("Info requires a model file path");
        return;
    }

    const file_path = args[0];

    // Check file exists
    const file_stat = std.fs.cwd().statFile(file_path) catch |err| {
        std.log.err("Failed to access file {s}: {}", .{ file_path, err });
        return;
    };

    // Detect format
    const detected_format = ModelFormat.fromExtension(file_path);

    std.log.info("🔍 ZigLlama Model Info");
    std.log.info("======================");
    std.log.info("📁 File: {s}", .{file_path});
    std.log.info("📊 Size: {d:.1f} MB ({d} bytes)", .{
        @as(f64, @floatFromInt(file_stat.size)) / (1024.0 * 1024.0),
        file_stat.size,
    });

    if (detected_format) |format| {
        std.log.info("🏷️  Format: {} ({s})", .{ format, format.toString() });
    } else {
        std.log.info("🏷️  Format: Unknown (unsupported extension)");
    }

    const modified_time = file_stat.mtime;
    const datetime = std.time.epoch.EpochSeconds{ .secs = @intCast(@divFloor(modified_time, std.time.ns_per_s)) };
    const day_seconds = datetime.getDaySeconds();
    const year_day = datetime.getYearDay();

    std.log.info("🕒 Modified: {d}-{d:0>2d}-{d:0>2d} {d:0>2d}:{d:0>2d}:{d:0>2d}",
        .{
            year_day.year, year_day.month.numeric(), year_day.day_index + 1,
            day_seconds.getHoursIntoDay(), day_seconds.getMinutesIntoHour(), day_seconds.getSecondsIntoMinute(),
        });

    // Try to analyze the file format
    if (detected_format) |format| {
        switch (format) {
            .GGUF => try analyzeGGUF(file_path, allocator),
            .SafeTensors => try analyzeSafeTensors(file_path, allocator),
            .Custom => try analyzeCustom(file_path, allocator),
            else => std.log.info("ℹ️  Detailed analysis not available for {} format", .{format}),
        }
    }
}

/// Handle list-formats subcommand
fn handleListFormatsCommand(allocator: std.mem.Allocator) !void {
    _ = allocator;

    std.log.info("📋 Supported Model Formats");
    std.log.info("===========================");

    const formats = [_]struct { format: ModelFormat, description: []const u8 }{
        .{ .format = .GGUF, .description = "GGUF (llama.cpp) - Efficient quantized format" },
        .{ .format = .SafeTensors, .description = "SafeTensors (Hugging Face) - Safe tensor format" },
        .{ .format = .PyTorch, .description = "PyTorch (.pt/.pth) - PyTorch native format" },
        .{ .format = .ONNX, .description = "ONNX (.onnx) - Open Neural Network Exchange" },
        .{ .format = .TensorFlow, .description = "TensorFlow (.pb) - TensorFlow SavedModel" },
        .{ .format = .Custom, .description = "ZigLlama (.zigllama) - Educational custom format" },
    };

    for (formats) |fmt| {
        const extension = fmt.format.extension();
        std.log.info("  {s:<12} {s:<15} - {s}", .{ fmt.format.toString(), extension, fmt.description });
    }

    std.log.info("");
    std.log.info("🔧 Quantization Types");
    std.log.info("======================");

    const quantizations = [_]struct { quant: QuantizationType, description: []const u8 }{
        .{ .quant = .Q4_0, .description = "4-bit quantization (basic)" },
        .{ .quant = .Q4_1, .description = "4-bit quantization (improved)" },
        .{ .quant = .Q5_0, .description = "5-bit quantization (basic)" },
        .{ .quant = .Q5_1, .description = "5-bit quantization (improved)" },
        .{ .quant = .Q8_0, .description = "8-bit quantization" },
        .{ .quant = .Q4_K_S, .description = "4-bit K-quantization (small)" },
        .{ .quant = .Q4_K_M, .description = "4-bit K-quantization (medium)" },
        .{ .quant = .Q5_K_S, .description = "5-bit K-quantization (small)" },
        .{ .quant = .Q5_K_M, .description = "5-bit K-quantization (medium)" },
        .{ .quant = .Q6_K, .description = "6-bit K-quantization" },
        .{ .quant = .IQ1_S, .description = "1-bit importance quantization" },
        .{ .quant = .IQ2_XXS, .description = "2-bit importance quantization (extra small)" },
        .{ .quant = .IQ2_XS, .description = "2-bit importance quantization (small)" },
        .{ .quant = .IQ3_XXS, .description = "3-bit importance quantization (extra small)" },
        .{ .quant = .IQ3_XS, .description = "3-bit importance quantization (small)" },
        .{ .quant = .IQ4_XS, .description = "4-bit importance quantization (small)" },
    };

    for (quantizations) |quant| {
        std.log.info("  {s:<10} - {s}", .{ quant.quant.toString(), quant.description });
    }
}

/// Handle supported subcommand
fn handleSupportedCommand(allocator: std.mem.Allocator) !void {
    std.log.info("🔄 Supported Conversion Pairs");
    std.log.info("==============================");

    const conversions = try ConversionUtils.getSupportedConversions(allocator);
    defer allocator.free(conversions);

    var native_count: usize = 0;
    var experimental_count: usize = 0;

    std.log.info("✅ Native Conversions (Full Support):");
    for (conversions) |conversion| {
        // Consider GGUF, SafeTensors, Custom as native
        const is_native = (conversion.from == .GGUF or conversion.from == .SafeTensors or conversion.from == .Custom) and
            (conversion.to == .GGUF or conversion.to == .SafeTensors or conversion.to == .Custom);

        if (is_native) {
            std.log.info("  {s} → {s}", .{ conversion.from.toString(), conversion.to.toString() });
            native_count += 1;
        }
    }

    std.log.info("");
    std.log.info("🧪 Experimental Conversions (Limited Support):");
    for (conversions) |conversion| {
        const is_experimental = !((conversion.from == .GGUF or conversion.from == .SafeTensors or conversion.from == .Custom) and
            (conversion.to == .GGUF or conversion.to == .SafeTensors or conversion.to == .Custom));

        if (is_experimental) {
            std.log.info("  {s} → {s}", .{ conversion.from.toString(), conversion.to.toString() });
            experimental_count += 1;
        }
    }

    std.log.info("");
    std.log.info("📊 Summary: {d} native, {d} experimental ({d} total)", .{ native_count, experimental_count, conversions.len });
}

/// Parse format string
fn parseFormat(format_str: []const u8) ?ModelFormat {
    if (std.mem.eql(u8, format_str, "gguf")) return .GGUF;
    if (std.mem.eql(u8, format_str, "safetensors")) return .SafeTensors;
    if (std.mem.eql(u8, format_str, "pytorch")) return .PyTorch;
    if (std.mem.eql(u8, format_str, "onnx")) return .ONNX;
    if (std.mem.eql(u8, format_str, "tensorflow")) return .TensorFlow;
    if (std.mem.eql(u8, format_str, "custom") or std.mem.eql(u8, format_str, "zigllama")) return .Custom;
    return null;
}

/// Parse quantization string
fn parseQuantization(quant_str: []const u8) ?QuantizationType {
    if (std.mem.eql(u8, quant_str, "none")) return .None;
    if (std.mem.eql(u8, quant_str, "q4_0")) return .Q4_0;
    if (std.mem.eql(u8, quant_str, "q4_1")) return .Q4_1;
    if (std.mem.eql(u8, quant_str, "q5_0")) return .Q5_0;
    if (std.mem.eql(u8, quant_str, "q5_1")) return .Q5_1;
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

/// Progress callback for conversion
fn progressCallback(progress: f32, message: []const u8) void {
    const bar_width = 40;
    const filled = @as(usize, @intFromFloat(progress * @as(f32, @floatFromInt(bar_width))));

    std.debug.print("\r🔄 [");
    var i: usize = 0;
    while (i < bar_width) : (i += 1) {
        if (i < filled) {
            std.debug.print("█");
        } else if (i == filled and progress < 1.0) {
            std.debug.print("▶");
        } else {
            std.debug.print("░");
        }
    }
    std.debug.print("] {d:5.1f}% {s}", .{ progress * 100, message });

    if (progress >= 1.0) {
        std.debug.print("\n");
    }
}

/// Analyze GGUF file format
fn analyzeGGUF(file_path: []const u8, allocator: std.mem.Allocator) !void {
    _ = allocator;

    std.log.info("");
    std.log.info("🔍 GGUF Analysis");
    std.log.info("================");

    const file = std.fs.cwd().openFile(file_path, .{}) catch |err| {
        std.log.warn("Failed to analyze GGUF file: {}", .{err});
        return;
    };
    defer file.close();

    var header_buffer: [16]u8 = undefined;
    _ = file.readAll(&header_buffer) catch |err| {
        std.log.warn("Failed to read GGUF header: {}", .{err});
        return;
    };

    if (std.mem.startsWith(u8, &header_buffer, "GGUF")) {
        std.log.info("✅ Valid GGUF magic header found");

        // Read version (simplified parsing)
        if (header_buffer.len >= 8) {
            const version = std.mem.readInt(u32, header_buffer[4..8], .little);
            std.log.info("📋 GGUF Version: {d}", .{version});
        }
    } else {
        std.log.warn("❌ Invalid GGUF magic header");
    }

    std.log.info("ℹ️  Full GGUF analysis requires complete parser implementation");
}

/// Analyze SafeTensors file format
fn analyzeSafeTensors(file_path: []const u8, allocator: std.mem.Allocator) !void {
    _ = allocator;

    std.log.info("");
    std.log.info("🔍 SafeTensors Analysis");
    std.log.info("=======================");

    const file = std.fs.cwd().openFile(file_path, .{}) catch |err| {
        std.log.warn("Failed to analyze SafeTensors file: {}", .{err});
        return;
    };
    defer file.close();

    var header_size_buffer: [8]u8 = undefined;
    _ = file.readAll(&header_size_buffer) catch |err| {
        std.log.warn("Failed to read SafeTensors header: {}", .{err});
        return;
    };

    const header_size = std.mem.readInt(u64, &header_size_buffer, .little);
    std.log.info("📋 Header size: {d} bytes", .{header_size});

    if (header_size > 0 and header_size < 1024 * 1024) {
        std.log.info("✅ Header size looks reasonable");
    } else {
        std.log.warn("❌ Header size seems invalid: {d}", .{header_size});
    }

    std.log.info("ℹ️  Full SafeTensors analysis requires JSON parser implementation");
}

/// Analyze custom ZigLlama format
fn analyzeCustom(file_path: []const u8, allocator: std.mem.Allocator) !void {
    _ = allocator;

    std.log.info("");
    std.log.info("🔍 ZigLlama Custom Format Analysis");
    std.log.info("==================================");

    const file = std.fs.cwd().openFile(file_path, .{}) catch |err| {
        std.log.warn("Failed to analyze custom file: {}", .{err});
        return;
    };
    defer file.close();

    var magic_buffer: [8]u8 = undefined;
    _ = file.readAll(&magic_buffer) catch |err| {
        std.log.warn("Failed to read custom header: {}", .{err});
        return;
    };

    if (std.mem.eql(u8, &magic_buffer, "ZIGLLAMA")) {
        std.log.info("✅ Valid ZigLlama magic header found");

        var metadata_len_buffer: [4]u8 = undefined;
        _ = file.readAll(&metadata_len_buffer) catch {
            std.log.warn("Failed to read metadata length");
            return;
        };

        const metadata_len = std.mem.readInt(u32, &metadata_len_buffer, .little);
        std.log.info("📋 Metadata length: {d} bytes", .{metadata_len});
    } else {
        std.log.warn("❌ Invalid ZigLlama magic header");
    }

    std.log.info("ℹ️  This is the ZigLlama educational format");
}

/// Print main usage
fn printUsage() !void {
    const usage =
        \\🦙 ZigLlama Model Converter
        \\===========================
        \\
        \\Educational tool for converting between different model formats with support
        \\for quantization and comprehensive format analysis.
        \\
        \\USAGE:
        \\    zigllama-converter <subcommand> [options]
        \\
        \\SUBCOMMANDS:
        \\    convert         Convert between model formats
        \\    info           Show detailed information about a model file
        \\    list-formats   List all supported formats and quantizations
        \\    supported      Show supported conversion pairs
        \\    help           Show this help message
        \\
        \\EXAMPLES:
        \\    # Convert GGUF to custom format
        \\    zigllama-converter convert model.gguf model.zigllama
        \\
        \\    # Convert with quantization
        \\    zigllama-converter convert --quantization q4_k_m model.safetensors model.gguf
        \\
        \\    # Show model information
        \\    zigllama-converter info model.gguf
        \\
        \\    # List supported formats
        \\    zigllama-converter list-formats
        \\
        \\For detailed help on a subcommand, use: zigllama-converter <subcommand> --help
        \\
        \\🎓 Educational Features:
        \\   • Support for 6 major model formats
        \\   • 16 quantization options including K-quant and IQ series
        \\   • Detailed format analysis and validation
        \\   • Progress tracking with visual feedback
        \\   • Comprehensive error handling and reporting
        \\
        \\🦙 ZigLlama: Where Education Meets Production-Ready AI ✨
        \\
    ;
    std.debug.print(usage, .{});
}

/// Print convert subcommand usage
fn printConvertUsage() !void {
    const usage =
        \\zigllama-converter convert - Convert between model formats
        \\
        \\USAGE:
        \\    zigllama-converter convert [OPTIONS] <source> <target>
        \\
        \\ARGUMENTS:
        \\    <source>    Source model file path
        \\    <target>    Target model file path
        \\
        \\OPTIONS:
        \\    --source-format <fmt>   Source format (auto-detected by default)
        \\    --target-format <fmt>   Target format (auto-detected by default)
        \\    --quantization <type>   Apply quantization during conversion
        \\    --chunk-size <bytes>    Processing chunk size (default: 1MB)
        \\    --no-validate          Skip output validation
        \\    --no-metadata          Don't preserve metadata
        \\    --verbose, -v          Enable verbose output
        \\    --help, -h             Show this help
        \\
        \\FORMATS:
        \\    gguf, safetensors, pytorch, onnx, tensorflow, custom
        \\
        \\QUANTIZATIONS:
        \\    q4_0, q4_1, q5_0, q5_1, q8_0           - Basic quantization
        \\    q4_k_s, q4_k_m, q5_k_s, q5_k_m, q6_k  - K-quantization
        \\    iq1_s, iq2_xxs, iq2_xs, iq3_xxs, iq3_xs, iq4_xs  - Importance quantization
        \\
        \\EXAMPLES:
        \\    # Basic conversion (auto-detect formats)
        \\    zigllama-converter convert model.gguf model.zigllama
        \\
        \\    # Convert with quantization
        \\    zigllama-converter convert --quantization q4_k_m input.safetensors output.gguf
        \\
        \\    # Explicit formats with verbose output
        \\    zigllama-converter convert --source-format pytorch --target-format gguf --verbose model.pt model.gguf
        \\
        \\    # Fast conversion with large chunks
        \\    zigllama-converter convert --chunk-size 10485760 large-model.safetensors large-model.gguf
        \\
    ;
    std.debug.print(usage, .{});
}