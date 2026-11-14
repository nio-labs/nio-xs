const std = @import("std");
const math = std.math;

pub const QK_K = 256;
pub const K_SCALE_SIZE = 12;

/// Standard GGML block_q4_K (256 values in 144 bytes)
pub const BlockQ4_K = extern struct {
    d: f16, // delta
    dmin: f16, // min delta
    scales: [K_SCALE_SIZE]u8, // 12 bytes of 6-bit scales
    qs: [QK_K / 2]u8, // 128 bytes of nibbles
};

/// Standard GGML block_q8_0 (32 values in 34 bytes)
pub const BlockQ8_0 = extern struct {
    d: f16,
    qs: [32]i8,
};

/// Fast dequantization of a single Q4_K block (256 elements) into an f32 buffer
pub fn dequantizeQ4_K(block: *const BlockQ4_K, out: *[QK_K]f32) void {
    const d: f32 = @floatCast(block.d);
    const min: f32 = @floatCast(block.dmin);

    // Unpack scales
    var sc: [8]u8 = undefined;
    var msc: [8]u8 = undefined;

    const s = &block.scales;
    sc[0] = s[0] & 63;
    sc[1] = s[1] & 63;
    sc[2] = s[2] & 63;
    sc[3] = s[3] & 63;
    sc[4] = s[4] & 63;
    sc[5] = s[5] & 63;
    sc[6] = s[6] & 63;
    sc[7] = s[7] & 63;

    msc[0] = (s[0] >> 6) | ((s[8] & 0x03) << 2);
    msc[1] = (s[1] >> 6) | ((s[8] & 0x0c) >> 0);
    msc[2] = (s[2] >> 6) | ((s[8] & 0x30) >> 2);
    msc[3] = (s[3] >> 6) | ((s[8] & 0xc0) >> 4);
    msc[4] = (s[4] >> 6) | ((s[9] & 0x03) << 2);
    msc[5] = (s[5] >> 6) | ((s[9] & 0x0c) >> 0);
    msc[6] = (s[6] >> 6) | ((s[9] & 0x30) >> 2);
    msc[7] = (s[7] >> 6) | ((s[9] & 0xc0) >> 4);

    var idx: usize = 0;
    var q_idx: usize = 0;

    // 8 sub-blocks of 32 elements each
    inline for (0..8) |sb| {
        const d_sub = d * @as(f32, @floatFromInt(sc[sb]));
        const m_sub = min * @as(f32, @floatFromInt(msc[sb]));

        inline for (0..16) |_| {
            const byte = block.qs[q_idx];
            q_idx += 1;
            const q0: f32 = @floatFromInt(byte & 0x0F);
            const q1: f32 = @floatFromInt(byte >> 4);

            out[idx] = q0 * d_sub - m_sub;
            out[idx + 1] = q1 * d_sub - m_sub;
            idx += 2;
        }
    }
}

/// SIMD Vectorized dot-product of dequantized block against an f32 activation vector
pub fn vecDotF32(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    const Vec8 = @Vector(8, f32);
    var sum_vec: Vec8 = @splat(0.0);

    const n = a.len;
    const n_vec = n - (n % 8);
    var i: usize = 0;

    while (i < n_vec) : (i += 8) {
        const va: Vec8 = a[i..][0..8].*;
        const vb: Vec8 = b[i..][0..8].*;
        sum_vec += va * vb;
    }

    var sum: f32 = @reduce(.Add, sum_vec);
    while (i < n) : (i += 1) {
        sum += a[i] * b[i];
    }
    return sum;
}
