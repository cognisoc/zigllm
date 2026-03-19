const std = @import("std");
const os = std.os;
const Allocator = std.mem.Allocator;
const foundation = @import("tensor.zig");
const Tensor = foundation.Tensor;

/// Memory mapping support for large model files
/// Provides efficient loading of model weights without full memory allocation
pub const MemoryMap = struct {
    ptr: [*]align(std.mem.page_size) u8,
    len: usize,
    fd: os.fd_t,
    locked: bool,

    const Self = @This();

    /// Memory mapping protection flags
    pub const Protection = struct {
        read: bool = true,
        write: bool = false,
        exec: bool = false,
    };

    /// Memory mapping flags
    pub const Flags = struct {
        shared: bool = false,      // Share mapping with other processes
        private: bool = true,      // Create private copy-on-write mapping
        anonymous: bool = false,   // Create anonymous mapping (no file)
        populate: bool = false,    // Populate page tables (pre-fault pages)
        huge_pages: bool = false,  // Use huge pages if available
    };

    /// Create memory mapping from file
    pub fn fromFile(path: []const u8, protection: Protection, flags: Flags) !Self {
        // Open file
        const file = std.fs.cwd().openFile(path, .{}) catch |err| switch (err) {
            error.FileNotFound => {
                std.log.err("Memory mapping file not found: {s}", .{path});
                return err;
            },
            else => return err,
        };
        defer file.close();

        // Get file size
        const file_size = try file.getEndPos();
        if (file_size == 0) {
            return error.EmptyFile;
        }

        // Map file to memory
        const fd = file.handle;
        const prot = createProtectionFlags(protection);
        const map_flags = createMappingFlags(flags);

        const ptr = os.mmap(
            null,
            file_size,
            prot,
            map_flags,
            fd,
            0,
        ) catch |err| switch (err) {
            error.MemoryMappingNotSupported => {
                std.log.err("Memory mapping not supported on this platform");
                return error.UnsupportedOperation;
            },
            error.OutOfMemory => {
                std.log.err("Failed to map file: out of virtual memory");
                return error.OutOfMemory;
            },
            else => return err,
        };

        return Self{
            .ptr = @as([*]align(std.mem.page_size) u8, @ptrCast(ptr)),
            .len = file_size,
            .fd = fd,
            .locked = false,
        };
    }

    /// Create anonymous memory mapping (no backing file)
    pub fn anonymous(size: usize, protection: Protection, flags: Flags) !Self {
        const prot = createProtectionFlags(protection);
        var map_flags = createMappingFlags(flags);
        map_flags |= os.MAP.ANONYMOUS;

        const ptr = os.mmap(
            null,
            size,
            prot,
            map_flags,
            -1,
            0,
        ) catch |err| switch (err) {
            error.OutOfMemory => {
                std.log.err("Failed to create anonymous mapping: out of virtual memory");
                return error.OutOfMemory;
            },
            else => return err,
        };

        return Self{
            .ptr = @as([*]align(std.mem.page_size) u8, @ptrCast(ptr)),
            .len = size,
            .fd = -1,
            .locked = false,
        };
    }

    /// Unmap memory
    pub fn deinit(self: *Self) void {
        if (self.locked) {
            self.unlock() catch {};
        }

        os.munmap(@as([*]align(std.mem.page_size) u8, self.ptr)[0..self.len]) catch |err| {
            std.log.warn("Failed to unmap memory: {}", .{err});
        };
    }

    /// Lock pages in physical memory (prevent swapping)
    pub fn lock(self: *Self) !void {
        if (self.locked) return;

        os.mlock(self.ptr[0..self.len]) catch |err| switch (err) {
            error.MemoryLockingNotSupported => {
                std.log.warn("Memory locking not supported on this platform");
                return error.UnsupportedOperation;
            },
            error.OutOfMemory => {
                std.log.warn("Failed to lock memory: insufficient physical memory or locked memory limit reached");
                return error.OutOfMemory;
            },
            error.PermissionDenied => {
                std.log.warn("Failed to lock memory: insufficient permissions");
                return error.PermissionDenied;
            },
            else => return err,
        };

        self.locked = true;
        std.log.info("Locked {d} MB of memory in RAM", .{self.len / (1024 * 1024)});
    }

    /// Unlock pages (allow swapping)
    pub fn unlock(self: *Self) !void {
        if (!self.locked) return;

        os.munlock(self.ptr[0..self.len]) catch |err| switch (err) {
            else => {
                std.log.warn("Failed to unlock memory: {}", .{err});
                return err;
            },
        };

        self.locked = false;
    }

    /// Prefault pages (load into memory)
    pub fn prefault(self: *Self) !void {
        // Touch every page to ensure it's loaded
        const page_size = std.mem.page_size;
        var offset: usize = 0;

        while (offset < self.len) {
            // Read first byte of each page
            _ = self.ptr[offset];
            offset += page_size;
        }

        std.log.info("Prefaulted {d} MB of memory mapping", .{self.len / (1024 * 1024)});
    }

    /// Advise kernel about memory usage patterns
    pub fn advise(self: *Self, advice: MemoryAdvice) !void {
        const madv_advice = switch (advice) {
            .Normal => os.MADV.NORMAL,
            .Random => os.MADV.RANDOM,
            .Sequential => os.MADV.SEQUENTIAL,
            .WillNeed => os.MADV.WILLNEED,
            .DontNeed => os.MADV.DONTNEED,
        };

        os.madvise(self.ptr[0..self.len], madv_advice) catch |err| switch (err) {
            error.InvalidArgument => {
                std.log.warn("Invalid memory advice argument");
            },
            else => {
                std.log.warn("Failed to set memory advice: {}", .{err});
            },
        };
    }

    /// Get slice of mapped memory
    pub fn getSlice(self: *Self, comptime T: type, offset: usize, count: usize) ![]T {
        const byte_size = count * @sizeOf(T);
        if (offset + byte_size > self.len) {
            return error.OutOfBounds;
        }

        const byte_ptr = self.ptr + offset;
        const typed_ptr = @as([*]T, @ptrCast(@alignCast(byte_ptr)));
        return typed_ptr[0..count];
    }

    /// Create tensor from mapped memory region
    pub fn createTensor(self: *Self, comptime T: type, offset: usize,
                       shape: []const usize) !Tensor(T) {
        const total_elements = blk: {
            var total: usize = 1;
            for (shape) |dim| total *= dim;
            break :blk total;
        };

        const data = try self.getSlice(T, offset, total_elements);

        return Tensor(T){
            .data = data,
            .shape = shape,
        };
    }

    /// Synchronize mapped memory with backing file
    pub fn sync(self: *Self, async_sync: bool) !void {
        if (self.fd == -1) return; // Anonymous mapping, nothing to sync

        const flags = if (async_sync) os.MS.ASYNC else os.MS.SYNC;

        os.msync(self.ptr[0..self.len], flags) catch |err| switch (err) {
            error.InvalidArgument => {
                std.log.warn("Invalid msync arguments");
            },
            else => {
                std.log.warn("Failed to sync memory mapping: {}", .{err});
                return err;
            },
        };
    }

    /// Get memory usage statistics
    pub fn getStats(self: *Self) MemoryStats {
        return MemoryStats{
            .total_size = self.len,
            .page_size = std.mem.page_size,
            .num_pages = (self.len + std.mem.page_size - 1) / std.mem.page_size,
            .is_locked = self.locked,
            .estimated_resident = if (self.locked) self.len else self.len / 2, // Rough estimate
        };
    }

    // Helper functions
    fn createProtectionFlags(protection: Protection) u32 {
        var prot: u32 = 0;
        if (protection.read) prot |= os.PROT.READ;
        if (protection.write) prot |= os.PROT.WRITE;
        if (protection.exec) prot |= os.PROT.EXEC;
        return prot;
    }

    fn createMappingFlags(flags: Flags) u32 {
        var map_flags: u32 = 0;
        if (flags.shared) map_flags |= os.MAP.SHARED;
        if (flags.private) map_flags |= os.MAP.PRIVATE;
        if (flags.populate) {
            // Note: MAP_POPULATE is Linux-specific
            const MAP_POPULATE = 0x8000;
            map_flags |= MAP_POPULATE;
        }
        if (flags.huge_pages) {
            // Note: MAP_HUGETLB is Linux-specific
            const MAP_HUGETLB = 0x40000;
            map_flags |= MAP_HUGETLB;
        }
        return map_flags;
    }
};

/// Memory usage advice for kernel optimization
pub const MemoryAdvice = enum {
    Normal,     // No special treatment
    Random,     // Expect random page references
    Sequential, // Expect sequential page references
    WillNeed,   // Expect access in near future
    DontNeed,   // Don't expect access in near future
};

/// Memory mapping statistics
pub const MemoryStats = struct {
    total_size: usize,        // Total mapped size
    page_size: usize,         // System page size
    num_pages: usize,         // Number of pages
    is_locked: bool,          // Whether memory is locked
    estimated_resident: usize, // Estimated resident memory
};

/// High-level model file mapping manager
pub const ModelFileMapper = struct {
    mappings: std.ArrayList(MemoryMap),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .mappings = std.ArrayList(MemoryMap).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.mappings.items) |*mapping| {
            mapping.deinit();
        }
        self.mappings.deinit();
    }

    /// Load model file with memory mapping
    pub fn loadModelFile(self: *Self, path: []const u8, lock_memory: bool,
                        prefault: bool) !*MemoryMap {
        const protection = MemoryMap.Protection{
            .read = true,
            .write = false,
            .exec = false,
        };

        const flags = MemoryMap.Flags{
            .private = true,
            .shared = false,
            .populate = prefault,
        };

        var mapping = try MemoryMap.fromFile(path, protection, flags);

        // Lock in memory if requested
        if (lock_memory) {
            mapping.lock() catch |err| switch (err) {
                error.UnsupportedOperation, error.PermissionDenied => {
                    std.log.warn("Could not lock model in memory: {}", .{err});
                },
                else => return err,
            };
        }

        // Prefault pages if requested and not done via MAP_POPULATE
        if (prefault and !flags.populate) {
            try mapping.prefault();
        }

        // Advise kernel about access patterns
        mapping.advise(.Sequential) catch {};

        try self.mappings.append(mapping);
        return &self.mappings.items[self.mappings.items.len - 1];
    }

    /// Create tensor from model file section
    pub fn createModelTensor(self: *Self, mapping: *MemoryMap, comptime T: type,
                            offset: usize, shape: []const usize) !Tensor(T) {
        _ = self; // No state needed for this operation
        return try mapping.createTensor(T, offset, shape);
    }

    /// Get total memory usage across all mappings
    pub fn getTotalMemoryUsage(self: *Self) MemoryStats {
        var total_stats = MemoryStats{
            .total_size = 0,
            .page_size = std.mem.page_size,
            .num_pages = 0,
            .is_locked = false,
            .estimated_resident = 0,
        };

        var any_locked = false;
        for (self.mappings.items) |*mapping| {
            const stats = mapping.getStats();
            total_stats.total_size += stats.total_size;
            total_stats.num_pages += stats.num_pages;
            total_stats.estimated_resident += stats.estimated_resident;
            if (stats.is_locked) any_locked = true;
        }

        total_stats.is_locked = any_locked;
        return total_stats;
    }

    /// Print memory usage summary
    pub fn printMemoryUsage(self: *Self) void {
        const stats = self.getTotalMemoryUsage();
        const total_mb = @as(f64, @floatFromInt(stats.total_size)) / (1024.0 * 1024.0);
        const resident_mb = @as(f64, @floatFromInt(stats.estimated_resident)) / (1024.0 * 1024.0);

        std.log.info("Memory mapping usage:");
        std.log.info("  Total mapped: {d:.1f} MB ({d} pages)", .{ total_mb, stats.num_pages });
        std.log.info("  Estimated resident: {d:.1f} MB", .{resident_mb});
        std.log.info("  Memory locked: {}", .{stats.is_locked});
        std.log.info("  Page size: {d} KB", .{stats.page_size / 1024});
    }
};

/// GGUF file memory mapping utilities
pub const GGUFMapper = struct {
    file_mapper: ModelFileMapper,

    const Self = @This();

    pub fn init(allocator: Allocator) Self {
        return Self{
            .file_mapper = ModelFileMapper.init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.file_mapper.deinit();
    }

    /// Load GGUF file with optimal memory mapping settings
    pub fn loadGGUFFile(self: *Self, path: []const u8, lock_weights: bool) !*MemoryMap {
        std.log.info("Loading GGUF file with memory mapping: {s}", .{path});

        // Load with memory mapping optimized for model inference
        const mapping = try self.file_mapper.loadModelFile(
            path,
            lock_weights,  // Lock if requested
            true          // Prefault pages for better performance
        );

        // Advise kernel that we'll access data sequentially initially,
        // then randomly during inference
        mapping.advise(.Sequential) catch {};

        std.log.info("GGUF file loaded successfully");
        self.file_mapper.printMemoryUsage();

        return mapping;
    }

    /// Create weight tensor from GGUF mapping
    pub fn createWeightTensor(self: *Self, mapping: *MemoryMap, comptime T: type,
                             offset: usize, shape: []const usize) !Tensor(T) {
        const tensor = try mapping.createTensor(T, offset, shape);

        // For weight tensors, advise kernel we'll access randomly
        const tensor_bytes = tensor.data.len * @sizeOf(T);
        const tensor_ptr = @as([*]u8, @ptrCast(tensor.data.ptr));

        os.madvise(tensor_ptr[0..tensor_bytes], os.MADV.RANDOM) catch |err| {
            std.log.debug("Could not set random access advice for tensor: {}", .{err});
        };

        return tensor;
    }
};

/// Utilities for optimal memory mapping based on system characteristics
pub const MappingOptimizer = struct {
    /// Detect optimal mapping strategy based on system resources
    pub fn detectOptimalStrategy(file_size: usize) MappingStrategy {
        // Get available system memory
        const available_memory = getAvailableMemory() catch file_size * 2;

        if (file_size > available_memory) {
            // Large model, prefer memory mapping with selective locking
            return MappingStrategy{
                .use_mmap = true,
                .lock_memory = false,
                .prefault = false,
                .advice = .Random,
            };
        } else if (file_size > available_memory / 2) {
            // Medium model, use mapping with prefaulting
            return MappingStrategy{
                .use_mmap = true,
                .lock_memory = false,
                .prefault = true,
                .advice = .WillNeed,
            };
        } else {
            // Small model, can afford to lock in memory
            return MappingStrategy{
                .use_mmap = true,
                .lock_memory = true,
                .prefault = true,
                .advice = .WillNeed,
            };
        }
    }

    /// Get available system memory (rough estimate)
    fn getAvailableMemory() !usize {
        // This is platform-specific and simplified
        // In practice, would use proper system calls
        return 8 * 1024 * 1024 * 1024; // Assume 8GB available
    }
};

pub const MappingStrategy = struct {
    use_mmap: bool,
    lock_memory: bool,
    prefault: bool,
    advice: MemoryAdvice,
};