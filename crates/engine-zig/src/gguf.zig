const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;

pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian

pub const GGMLType = enum(u32) {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    _,
};

pub const GGUFValueType = enum(u32) {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
    _,
};

pub const TensorInfo = struct {
    name: []const u8,
    n_dims: u32,
    dims: [4]u64,
    type: GGMLType,
    offset: u64,
    size_bytes: usize,
};

pub const ModelMetadata = struct {
    architecture: []const u8 = "qwen2",
    context_length: u64 = 4096,
    embedding_length: u64 = 1536,
    block_count: u32 = 28,
    feed_forward_length: u64 = 8960,
    head_count: u32 = 12,
    head_count_kv: u32 = 2,
    rope_freq_base: f32 = 1000000.0,
    alignment: u64 = 32,
};

pub const GGUFFile = struct {
    allocator: Allocator,
    data: []const u8,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
    metadata: ModelMetadata,
    tensors: std.StringHashMap(TensorInfo),
    data_offset: usize,

    pub fn parse(allocator: Allocator, bytes: []const u8) !GGUFFile {
        if (bytes.len < 16) return error.FileTooSmall;

        var stream = std.io.fixedBufferStream(bytes);
        var reader = stream.reader();

        const magic = try reader.readInt(u32, .little);
        if (magic != GGUF_MAGIC) return error.InvalidMagic;

        const version = try reader.readInt(u32, .little);
        if (version != 2 and version != 3) return error.UnsupportedVersion;

        const tensor_count = try reader.readInt(u64, .little);
        const metadata_kv_count = try reader.readInt(u64, .little);

        var meta = ModelMetadata{};
        var i: usize = 0;
        while (i < metadata_kv_count) : (i += 1) {
            const key_len = try reader.readInt(u64, .little);
            const key_pos = stream.pos;
            if (key_pos + key_len > bytes.len) return error.UnexpectedEOF;
            const key = bytes[key_pos .. key_pos + key_len];
            stream.pos += key_len;

            const val_type_raw = try reader.readInt(u32, .little);
            const val_type = @as(GGUFValueType, @enumFromInt(val_type_raw));

            try skipOrReadValue(&stream, bytes, &meta, key, val_type);
        }

        // Tensors
        var tensors = std.StringHashMap(TensorInfo).init(allocator);
        i = 0;
        while (i < tensor_count) : (i += 1) {
            const name_len = try reader.readInt(u64, .little);
            const name_pos = stream.pos;
            if (name_pos + name_len > bytes.len) return error.UnexpectedEOF;
            const name = bytes[name_pos .. name_pos + name_len];
            stream.pos += name_len;

            const n_dims = try reader.readInt(u32, .little);
            if (n_dims > 4) return error.TooManyDimensions;

            var dims = [_]u64{ 1, 1, 1, 1 };
            var d: usize = 0;
            while (d < n_dims) : (d += 1) {
                dims[d] = try reader.readInt(u64, .little);
            }

            const qtype_raw = try reader.readInt(u32, .little);
            const qtype = @as(GGMLType, @enumFromInt(qtype_raw));
            const offset = try reader.readInt(u64, .little);

            // Calculate element count & approximate size
            var n_elements: u64 = 1;
            for (dims[0..n_dims]) |dim| {
                n_elements *= dim;
            }

            const size_bytes = calculateTensorSize(qtype, n_elements);

            try tensors.put(name, TensorInfo{
                .name = name,
                .n_dims = n_dims,
                .dims = dims,
                .type = qtype,
                .offset = offset,
                .size_bytes = size_bytes,
            });
        }

        // Align data offset to alignment
        const alignment = meta.alignment;
        const current_pos = stream.pos;
        const remainder = current_pos % alignment;
        const data_offset = if (remainder == 0) current_pos else current_pos + (alignment - remainder);

        return GGUFFile{
            .allocator = allocator,
            .data = bytes,
            .version = version,
            .tensor_count = tensor_count,
            .metadata_kv_count = metadata_kv_count,
            .metadata = meta,
            .tensors = tensors,
            .data_offset = data_offset,
        };
    }

    pub fn getTensorSlice(self: *const GGUFFile, name: []const u8) ?[]const u8 {
        const info = self.tensors.get(name) orelse return null;
        const start = self.data_offset + info.offset;
        const end = start + info.size_bytes;
        if (end > self.data.len) return null;
        return self.data[start..end];
    }

    pub fn deinit(self: *GGUFFile) void {
        self.tensors.deinit();
    }
};

fn skipOrReadValue(stream: *std.io.FixedBufferStream([]const u8), bytes: []const u8, meta: *ModelMetadata, key: []const u8, val_type: GGUFValueType) !void {
    var reader = stream.reader();
    switch (val_type) {
        .UINT8, .INT8, .BOOL => stream.pos += 1,
        .UINT16, .INT16 => stream.pos += 2,
        .UINT32, .INT32, .FLOAT32 => {
            const val = try reader.readInt(u32, .little);
            if (mem.endsWith(u8, key, ".block_count")) {
                meta.block_count = val;
            } else if (mem.endsWith(u8, key, ".head_count")) {
                meta.head_count = val;
            } else if (mem.endsWith(u8, key, ".head_count_kv")) {
                meta.head_count_kv = val;
            }
        },
        .UINT64, .INT64, .FLOAT64 => {
            const val = try reader.readInt(u64, .little);
            if (mem.endsWith(u8, key, ".context_length")) {
                meta.context_length = val;
            } else if (mem.endsWith(u8, key, ".embedding_length")) {
                meta.embedding_length = val;
            } else if (mem.endsWith(u8, key, ".feed_forward_length")) {
                meta.feed_forward_length = val;
            } else if (mem.eql(u8, key, "general.alignment")) {
                meta.alignment = val;
            }
        },
        .STRING => {
            const str_len = try reader.readInt(u64, .little);
            const str_pos = stream.pos;
            if (str_pos + str_len > bytes.len) return error.UnexpectedEOF;
            const s = bytes[str_pos .. str_pos + str_len];
            stream.pos += str_len;
            if (mem.eql(u8, key, "general.architecture")) {
                meta.architecture = s;
            }
        },
        .ARRAY => {
            const elem_type_raw = try reader.readInt(u32, .little);
            const elem_type = @as(GGUFValueType, @enumFromInt(elem_type_raw));
            const count = try reader.readInt(u64, .little);
            var i: usize = 0;
            while (i < count) : (i += 1) {
                try skipOrReadValue(stream, bytes, meta, "", elem_type);
            }
        },
        _ => {},
    }
}

fn calculateTensorSize(qtype: GGMLType, n_elements: u64) usize {
    return switch (qtype) {
        .F32 => @as(usize, @intCast(n_elements * 4)),
        .F16, .BF16 => @as(usize, @intCast(n_elements * 2)),
        .Q4_0 => @as(usize, @intCast((n_elements + 31) / 32 * 18)),
        .Q8_0 => @as(usize, @intCast((n_elements + 31) / 32 * 34)),
        .Q4_K => @as(usize, @intCast((n_elements + 255) / 256 * 144)),
        .Q5_K => @as(usize, @intCast((n_elements + 255) / 256 * 176)),
        .Q6_K => @as(usize, @intCast((n_elements + 255) / 256 * 210)),
        else => @as(usize, @intCast(n_elements * 2)), // fallback
    };
}
