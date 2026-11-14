const std = @import("std");
const MmapStreamer = @import("mmap_streamer.zig").MmapStreamer;

pub export fn nio_engine_open(path_c: [*:0]const u8, window_layers: u32) ?*anyopaque {
    const path = std.mem.span(path_c);
    const allocator = std.heap.page_allocator;

    const streamer = allocator.create(MmapStreamer) catch return null;
    streamer.* = MmapStreamer.init(allocator, path, window_layers) catch {
        allocator.destroy(streamer);
        return null;
    };

    return @ptrCast(streamer);
}

pub export fn nio_engine_get_block_count(ctx: ?*anyopaque) u32 {
    if (ctx == null) return 0;
    const streamer: *MmapStreamer = @ptrCast(@alignCast(ctx.?));
    return streamer.gguf.metadata.block_count;
}

pub export fn nio_engine_get_embedding_length(ctx: ?*anyopaque) u64 {
    if (ctx == null) return 0;
    const streamer: *MmapStreamer = @ptrCast(@alignCast(ctx.?));
    return streamer.gguf.metadata.embedding_length;
}

pub export fn nio_engine_step_layer(ctx: ?*anyopaque, layer_idx: u32) void {
    if (ctx == null) return;
    const streamer: *MmapStreamer = @ptrCast(@alignCast(ctx.?));
    streamer.stepLayer(layer_idx);
}

pub export fn nio_engine_deinit(ctx: ?*anyopaque) void {
    if (ctx == null) return;
    const streamer: *MmapStreamer = @ptrCast(@alignCast(ctx.?));
    const allocator = streamer.gguf.allocator;
    streamer.deinit();
    allocator.destroy(streamer);
}
