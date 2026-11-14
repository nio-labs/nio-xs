const std = @import("std");
const posix = std.posix;
const GGUFFile = @import("gguf.zig").GGUFFile;

pub const MmapStreamer = struct {
    file: std.fs.File,
    mapped_data: []align(std.mem.page_size) const u8,
    gguf: GGUFFile,
    resident_window_layers: u32,
    current_layer: u32 = 0,

    pub fn init(allocator: std.mem.Allocator, file_path: []const u8, window_layers: u32) !MmapStreamer {
        const file = try std.fs.cwd().openFile(file_path, .{ .mode = .read_only });
        errdefer file.close();

        const file_size = try file.getEndPos();
        const mapped = try posix.mmap(
            null,
            file_size,
            posix.PROT.READ,
            .{ .TYPE = .SHARED },
            file.handle,
            0,
        );
        errdefer posix.munmap(mapped);

        // Tell kernel we plan sequential access on the header
        _ = posix.madvise(mapped.ptr, @min(mapped.len, 1024 * 1024 * 16), posix.MADV.WILLNEED) catch {};

        const gguf = try GGUFFile.parse(allocator, mapped);

        return MmapStreamer{
            .file = file,
            .mapped_data = mapped,
            .gguf = gguf,
            .resident_window_layers = window_layers,
            .current_layer = 0,
        };
    }

    /// Prepare layer `layer_idx` for execution (prefetch if necessary)
    pub fn prefetchLayer(self: *MmapStreamer, layer_idx: u32) void {
        var prefix_buf: [64]u8 = undefined;
        const prefix = std.fmt.bufPrint(&prefix_buf, "blk.{d}.", .{layer_idx}) catch return;

        var it = self.gguf.tensors.iterator();
        while (it.next()) |entry| {
            if (std.mem.startsWith(u8, entry.key_ptr.*, prefix)) {
                const info = entry.value_ptr;
                const start = self.gguf.data_offset + info.offset;
                const end = start + info.size_bytes;
                if (end <= self.mapped_data.len) {
                    const page_mask: usize = std.mem.page_size - 1;
                    const inv_mask: usize = ~page_mask;
                    const aligned_start = start & inv_mask;
                    const aligned_end = (end + page_mask) & inv_mask;
                    const aligned_len = aligned_end - aligned_start;
                    const ptr: [*]align(std.mem.page_size) u8 = @constCast(@alignCast(self.mapped_data.ptr + aligned_start));
                    _ = posix.madvise(ptr, aligned_len, posix.MADV.WILLNEED) catch {};
                }
            }
        }
    }

    /// Release layer `layer_idx` from physical RAM via MADV_DONTNEED
    pub fn evictLayer(self: *MmapStreamer, layer_idx: u32) void {
        var prefix_buf: [64]u8 = undefined;
        const prefix = std.fmt.bufPrint(&prefix_buf, "blk.{d}.", .{layer_idx}) catch return;

        var it = self.gguf.tensors.iterator();
        while (it.next()) |entry| {
            if (std.mem.startsWith(u8, entry.key_ptr.*, prefix)) {
                const info = entry.value_ptr;
                const start = self.gguf.data_offset + info.offset;
                const end = start + info.size_bytes;
                if (end <= self.mapped_data.len) {
                    const page_mask: usize = std.mem.page_size - 1;
                    const inv_mask: usize = ~page_mask;
                    const aligned_start = start & inv_mask;
                    const aligned_end = (end + page_mask) & inv_mask;
                    const aligned_len = aligned_end - aligned_start;
                    const ptr: [*]align(std.mem.page_size) u8 = @constCast(@alignCast(self.mapped_data.ptr + aligned_start));
                    _ = posix.madvise(ptr, aligned_len, posix.MADV.DONTNEED) catch {};
                }
            }
        }
    }

    /// Actively touch all pages of layer `layer_idx` to fault model weights into resident RAM (VmRSS)
    pub fn touchLayer(self: *MmapStreamer, layer_idx: u32) usize {
        var prefix_buf: [64]u8 = undefined;
        const prefix = std.fmt.bufPrint(&prefix_buf, "blk.{d}.", .{layer_idx}) catch return 0;

        var touched_bytes: usize = 0;
        var it = self.gguf.tensors.iterator();
        while (it.next()) |entry| {
            if (std.mem.startsWith(u8, entry.key_ptr.*, prefix)) {
                const info = entry.value_ptr;
                const start = self.gguf.data_offset + info.offset;
                const end = start + info.size_bytes;
                if (end <= self.mapped_data.len) {
                    var offset = start;
                    while (offset < end) : (offset += std.mem.page_size) {
                        const ptr: *volatile const u8 = @ptrCast(&self.mapped_data[offset]);
                        const b = ptr.*;
                        _ = b;
                        touched_bytes += std.mem.page_size;
                    }
                }
            }
        }
        return touched_bytes;
    }

    /// Actively touch embedding and output norm tensors into physical RAM
    pub fn touchEmbeddings(self: *MmapStreamer) usize {
        var touched_bytes: usize = 0;
        var it = self.gguf.tensors.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (std.mem.startsWith(u8, name, "token_embd.") or
                std.mem.startsWith(u8, name, "output_norm.") or
                std.mem.startsWith(u8, name, "output."))
            {
                const info = entry.value_ptr;
                const start = self.gguf.data_offset + info.offset;
                const end = start + info.size_bytes;
                if (end <= self.mapped_data.len) {
                    var offset = start;
                    while (offset < end) : (offset += std.mem.page_size) {
                        const ptr: *volatile const u8 = @ptrCast(&self.mapped_data[offset]);
                        const b = ptr.*;
                        _ = b;
                        touched_bytes += std.mem.page_size;
                    }
                }
            }
        }
        return touched_bytes;
    }

    /// Advance rolling sliding-window: prefetches layer + window, faults active layer, evicts trailing layer
    pub fn stepLayer(self: *MmapStreamer, layer_idx: u32) void {
        self.current_layer = layer_idx;
        const total_blocks = self.gguf.metadata.block_count;

        // Prefetch next layer in window
        if (layer_idx + self.resident_window_layers < total_blocks) {
            self.prefetchLayer(layer_idx + self.resident_window_layers);
        }

        // Fault current layer into physical resident RAM
        _ = self.touchLayer(layer_idx);

        // Evict layer that passed outside the trailing window
        if (layer_idx >= self.resident_window_layers) {
            self.evictLayer(layer_idx - self.resident_window_layers);
        }
    }

    /// Evict all faulted pages of this model back to the kernel via MADV_DONTNEED
    pub fn evictAll(self: *MmapStreamer) void {
        const page_mask: usize = std.mem.page_size - 1;
        const inv_mask: usize = ~page_mask;
        const aligned_len = (self.mapped_data.len + page_mask) & inv_mask;
        const ptr: [*]align(std.mem.page_size) u8 = @constCast(@alignCast(self.mapped_data.ptr));
        _ = posix.madvise(ptr, aligned_len, posix.MADV.DONTNEED) catch {};
    }

    pub fn deinit(self: *MmapStreamer) void {
        self.gguf.deinit();
        posix.munmap(self.mapped_data);
        self.file.close();
    }
};
