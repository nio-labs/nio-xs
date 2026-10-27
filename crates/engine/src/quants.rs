#![allow(unsafe_op_in_unsafe_fn)]
// Quantization kernels for AST Micro Engine: IQ3_XXS, IQ4_NL, Q8_0, Q4_0, and F32.

pub const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

pub const KSIGNS_IQ2XS: [u8; 128] = [
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
];

pub const IQ3XXS_GRID: [u32; 256] = [
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
];

pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Dequantizes IQ4_NL (dtype 20): 32 weights per 18-byte block (2 bytes f16 scale, 16 bytes nibbles).
pub fn dequantize_row_iq4_nl(raw_bytes: &[u8], out: &mut [f32]) {
    assert_eq!(out.len() % 32, 0);
    let nb = out.len() / 32;
    assert_eq!(raw_bytes.len(), nb * 18);

    for i in 0..nb {
        let block = &raw_bytes[i * 18..(i + 1) * 18];
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..18];
        let y = &mut out[i * 32..(i + 1) * 32];
        for j in 0..16 {
            let q = qs[j];
            y[j] = d * (KVALUES_IQ4NL[(q & 0x0f) as usize] as f32);
            y[j + 16] = d * (KVALUES_IQ4NL[(q >> 4) as usize] as f32);
        }
    }
}

/// Dequantizes IQ3_XXS (dtype 18): 256 weights per 98-byte block (2 bytes f16 scale, 64 bytes qs, 32 bytes scales+signs).
pub fn dequantize_row_iq3_xxs(raw_bytes: &[u8], out: &mut [f32]) {
    assert_eq!(out.len() % 256, 0);
    let nb = out.len() / 256;
    assert_eq!(raw_bytes.len(), nb * 98);

    for i in 0..nb {
        let block = &raw_bytes[i * 98..(i + 1) * 98];
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..66];
        let scales_and_signs = &block[66..98];

        let mut qs_idx = 0usize;
        let mut y_idx = i * 256;

        for ib32 in 0..8 {
            let aux32 = u32::from_le_bytes([
                scales_and_signs[4 * ib32],
                scales_and_signs[4 * ib32 + 1],
                scales_and_signs[4 * ib32 + 2],
                scales_and_signs[4 * ib32 + 3],
            ]);
            let db = d * (0.5 + ((aux32 >> 28) as f32)) * 0.5;

            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                let g1_bytes = IQ3XXS_GRID[qs[qs_idx + 2 * l] as usize].to_le_bytes();
                let g2_bytes = IQ3XXS_GRID[qs[qs_idx + 2 * l + 1] as usize].to_le_bytes();

                for j in 0..4 {
                    let s1 = if (signs & KMASK_IQ2XS[j]) != 0 { -1.0 } else { 1.0 };
                    let s2 = if (signs & KMASK_IQ2XS[j + 4]) != 0 { -1.0 } else { 1.0 };
                    out[y_idx + j] = db * (g1_bytes[j] as f32) * s1;
                    out[y_idx + j + 4] = db * (g2_bytes[j] as f32) * s2;
                }
                y_idx += 8;
            }
            qs_idx += 8;
        }
    }
}

/// Dequantizes Q8_0 (dtype 8): 32 weights per 34-byte block (2 bytes f16 scale, 32 bytes i8 values).
pub fn dequantize_row_q8_0(raw_bytes: &[u8], out: &mut [f32]) {
    assert_eq!(out.len() % 32, 0);
    let nb = out.len() / 32;
    assert_eq!(raw_bytes.len(), nb * 34);

    for i in 0..nb {
        let block = &raw_bytes[i * 34..(i + 1) * 34];
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..34];
        let y = &mut out[i * 32..(i + 1) * 32];
        for j in 0..32 {
            y[j] = d * (qs[j] as i8 as f32);
        }
    }
}

/// Dequantizes Q4_0 (dtype 2): 32 weights per 18-byte block (2 bytes f16 scale, 16 bytes nibbles - 8).
pub fn dequantize_row_q4_0(raw_bytes: &[u8], out: &mut [f32]) {
    assert_eq!(out.len() % 32, 0);
    let nb = out.len() / 32;
    assert_eq!(raw_bytes.len(), nb * 18);

    for i in 0..nb {
        let block = &raw_bytes[i * 18..(i + 1) * 18];
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[2..18];
        let y = &mut out[i * 32..(i + 1) * 32];
        for j in 0..16 {
            let v0 = (qs[j] & 0x0f) as i8 - 8;
            let v1 = (qs[j] >> 4) as i8 - 8;
            y[j] = d * (v0 as f32);
            y[j + 16] = d * (v1 as f32);
        }
    }
}

/// Copies raw F32 bytes into out.
pub fn dequantize_row_f32(raw_bytes: &[u8], out: &mut [f32]) {
    assert_eq!(raw_bytes.len(), out.len() * 4);
    for (i, chunk) in raw_bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
}

pub const SIGNS64_IQ2XS: [u64; 128] = [
    0x0101010101010101, 0xff010101010101ff, 0xff0101010101ff01, 0x010101010101ffff,
    0xff01010101ff0101, 0x0101010101ff01ff, 0x0101010101ffff01, 0xff01010101ffffff,
    0xff010101ff010101, 0x01010101ff0101ff, 0x01010101ff01ff01, 0xff010101ff01ffff,
    0x01010101ffff0101, 0xff010101ffff01ff, 0xff010101ffffff01, 0x01010101ffffffff,
    0xff0101ff01010101, 0x010101ff010101ff, 0x010101ff0101ff01, 0xff0101ff0101ffff,
    0x010101ff01ff0101, 0xff0101ff01ff01ff, 0xff0101ff01ffff01, 0x010101ff01ffffff,
    0x010101ffff010101, 0xff0101ffff0101ff, 0xff0101ffff01ff01, 0x010101ffff01ffff,
    0xff0101ffffff0101, 0x010101ffffff01ff, 0x010101ffffffff01, 0xff0101ffffffffff,
    0xff01ff0101010101, 0x0101ff01010101ff, 0x0101ff010101ff01, 0xff01ff010101ffff,
    0x0101ff0101ff0101, 0xff01ff0101ff01ff, 0xff01ff0101ffff01, 0x0101ff0101ffffff,
    0x0101ff01ff010101, 0xff01ff01ff0101ff, 0xff01ff01ff01ff01, 0x0101ff01ff01ffff,
    0xff01ff01ffff0101, 0x0101ff01ffff01ff, 0x0101ff01ffffff01, 0xff01ff01ffffffff,
    0x0101ffff01010101, 0xff01ffff010101ff, 0xff01ffff0101ff01, 0x0101ffff0101ffff,
    0xff01ffff01ff0101, 0x0101ffff01ff01ff, 0x0101ffff01ffff01, 0xff01ffff01ffffff,
    0xff01ffffff010101, 0x0101ffffff0101ff, 0x0101ffffff01ff01, 0xff01ffffff01ffff,
    0x0101ffffffff0101, 0xff01ffffffff01ff, 0xff01ffffffffff01, 0x0101ffffffffffff,
    0xffff010101010101, 0x01ff0101010101ff, 0x01ff01010101ff01, 0xffff01010101ffff,
    0x01ff010101ff0101, 0xffff010101ff01ff, 0xffff010101ffff01, 0x01ff010101ffffff,
    0x01ff0101ff010101, 0xffff0101ff0101ff, 0xffff0101ff01ff01, 0x01ff0101ff01ffff,
    0xffff0101ffff0101, 0x01ff0101ffff01ff, 0x01ff0101ffffff01, 0xffff0101ffffffff,
    0x01ff01ff01010101, 0xffff01ff010101ff, 0xffff01ff0101ff01, 0x01ff01ff0101ffff,
    0xffff01ff01ff0101, 0x01ff01ff01ff01ff, 0x01ff01ff01ffff01, 0xffff01ff01ffffff,
    0xffff01ffff010101, 0x01ff01ffff0101ff, 0x01ff01ffff01ff01, 0xffff01ffff01ffff,
    0x01ff01ffffff0101, 0xffff01ffffff01ff, 0xffff01ffffffff01, 0x01ff01ffffffffff,
    0x01ffff0101010101, 0xffffff01010101ff, 0xffffff010101ff01, 0x01ffff010101ffff,
    0xffffff0101ff0101, 0x01ffff0101ff01ff, 0x01ffff0101ffff01, 0xffffff0101ffffff,
    0xffffff01ff010101, 0x01ffff01ff0101ff, 0x01ffff01ff01ff01, 0xffffff01ff01ffff,
    0x01ffff01ffff0101, 0xffffff01ffff01ff, 0xffffff01ffffff01, 0x01ffff01ffffffff,
    0xffffffff01010101, 0x01ffffff010101ff, 0x01ffffff0101ff01, 0xffffffff0101ffff,
    0x01ffffff01ff0101, 0xffffffff01ff01ff, 0xffffffff01ffff01, 0x01ffffff01ffffff,
    0x01ffffffff010101, 0xffffffffff0101ff, 0xffffffffff01ff01, 0x01ffffffff01ffff,
    0xffffffffffff0101, 0x01ffffffffff01ff, 0xffffffffffff01, 0xffffffffffffffff,
];

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_0 {
    pub d: half::f16,
    pub qs: [i8; 32],
}

pub fn quantize_row_q8_0(x: &[f32], out: &mut [BlockQ8_0]) {
    assert_eq!(x.len() % 32, 0);
    assert_eq!(out.len(), x.len() / 32);
    for (i, block) in out.iter_mut().enumerate() {
        let chunk = &x[i * 32..(i + 1) * 32];
        let mut amax = 0.0f32;
        for &v in chunk {
            let av = v.abs();
            if av > amax {
                amax = av;
            }
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        block.d = half::f16::from_f32(d);
        for j in 0..32 {
            let x0 = chunk[j] * id;
            block.qs[j] = x0.round().clamp(-128.0, 127.0) as i8;
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_K {
    pub d: f32,
    pub qs: [i8; 256],
    pub bsums: [i16; 16],
}

pub fn quantize_row_q8_k(x: &[f32], out: &mut [BlockQ8_K]) {
    assert_eq!(x.len() % 256, 0);
    assert_eq!(out.len(), x.len() / 256);
    for (i, block) in out.iter_mut().enumerate() {
        let chunk = &x[i * 256..(i + 1) * 256];
        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &v in chunk {
            let av = v.abs();
            if av > amax {
                amax = av;
                max = v;
            }
        }
        if amax == 0.0 {
            block.d = 0.0;
            block.qs.fill(0);
            block.bsums.fill(0);
            continue;
        }
        let iscale = -127.0 / max;
        for j in 0..256 {
            let v = (iscale * chunk[j]).round() as i32;
            block.qs[j] = v.clamp(-128, 127) as i8;
        }
        for j in 0..16 {
            let mut sum = 0i32;
            for ii in 0..16 {
                sum += block.qs[j * 16 + ii] as i32;
            }
            block.bsums[j] = sum.clamp(-32768, 32767) as i16;
        }
        block.d = 1.0 / iscale;
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum_float_8(x: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;
    let mut res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn mul_add_epi8(x: core::arch::x86_64::__m256i, y: core::arch::x86_64::__m256i) -> core::arch::x86_64::__m256i {
    use core::arch::x86_64::*;
    let ax = _mm256_sign_epi8(x, x);
    let sy = _mm256_sign_epi8(y, x);
    _mm256_maddubs_epi16(ax, sy)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_i16_pairs_float(x: core::arch::x86_64::__m256i) -> core::arch::x86_64::__m256 {
    use core::arch::x86_64::*;
    let ones = _mm256_set1_epi16(1);
    let summed_pairs = _mm256_madd_epi16(ones, x);
    _mm256_cvtepi32_ps(summed_pairs)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn mul_sum_i8_pairs_float(x: core::arch::x86_64::__m256i, y: core::arch::x86_64::__m256i) -> core::arch::x86_64::__m256 {
    use core::arch::x86_64::*;
    let ax = _mm256_sign_epi8(x, x);
    let sy = _mm256_sign_epi8(y, x);
    let dot = _mm256_maddubs_epi16(ax, sy);
    sum_i16_pairs_float(dot)
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn vec_dot_q8_0_q8_0_avx2(weight_bytes: &[u8], x_blocks: &[BlockQ8_0]) -> f32 {
    use core::arch::x86_64::*;
    let nb = x_blocks.len();
    assert_eq!(weight_bytes.len(), nb * 34);

    let mut acc = _mm256_setzero_ps();
    for ib in 0..nb {
        let block_ptr = weight_bytes.as_ptr().add(ib * 34);
        let d_f16 = half::f16::from_le_bytes([*block_ptr, *block_ptr.add(1)]);
        let d = _mm256_set1_ps(d_f16.to_f32() * x_blocks[ib].d.to_f32());
        let qx = _mm256_loadu_si256(block_ptr.add(2) as *const __m256i);
        let qy = _mm256_loadu_si256(x_blocks[ib].qs.as_ptr() as *const __m256i);
        let q = mul_sum_i8_pairs_float(qx, qy);
        acc = _mm256_fmadd_ps(d, q, acc);
    }
    hsum_float_8(acc)
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn vec_dot_iq4_nl_q8_0_avx2(weight_bytes: &[u8], x_blocks: &[BlockQ8_0]) -> f32 {
    use core::arch::x86_64::*;
    let nb = x_blocks.len();
    assert_eq!(weight_bytes.len(), nb * 18);

    let values128 = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let m4b = _mm_set1_epi8(0x0f);
    let mone = _mm256_set1_epi16(1);

    let mut accum1 = _mm256_setzero_ps();
    let mut accum2 = _mm256_setzero_ps();
    let mut ib = 0;
    while ib + 1 < nb {
        let b0_ptr = weight_bytes.as_ptr().add(ib * 18);
        let b1_ptr = weight_bytes.as_ptr().add((ib + 1) * 18);

        let q4bits_1 = _mm_loadu_si128(b0_ptr.add(2) as *const __m128i);
        let q4bits_2 = _mm_loadu_si128(b1_ptr.add(2) as *const __m128i);

        let q8b_1 = _mm256_loadu_si256(x_blocks[ib].qs.as_ptr() as *const __m256i);
        let q8b_2 = _mm256_loadu_si256(x_blocks[ib + 1].qs.as_ptr() as *const __m256i);

        let q4b_1_low = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b));
        let q4b_1_high = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b));
        let q4b_1 = _mm256_insertf128_si256(_mm256_castsi128_si256(q4b_1_low), q4b_1_high, 1);

        let q4b_2_low = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b));
        let q4b_2_high = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b));
        let q4b_2 = _mm256_insertf128_si256(_mm256_castsi128_si256(q4b_2_low), q4b_2_high, 1);

        let p16_1 = mul_add_epi8(q4b_1, q8b_1);
        let p16_2 = mul_add_epi8(q4b_2, q8b_2);

        let p_1 = _mm256_madd_epi16(p16_1, mone);
        let p_2 = _mm256_madd_epi16(p16_2, mone);

        let d0 = half::f16::from_le_bytes([*b0_ptr, *b0_ptr.add(1)]).to_f32() * x_blocks[ib].d.to_f32();
        let d1 = half::f16::from_le_bytes([*b1_ptr, *b1_ptr.add(1)]).to_f32() * x_blocks[ib + 1].d.to_f32();

        accum1 = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(p_1), accum1);
        accum2 = _mm256_fmadd_ps(_mm256_set1_ps(d1), _mm256_cvtepi32_ps(p_2), accum2);

        ib += 2;
    }

    let mut sumf = hsum_float_8(_mm256_add_ps(accum1, accum2));
    while ib < nb {
        let b_ptr = weight_bytes.as_ptr().add(ib * 18);
        let d = half::f16::from_le_bytes([*b_ptr, *b_ptr.add(1)]).to_f32() * x_blocks[ib].d.to_f32();
        let qs = std::slice::from_raw_parts(b_ptr.add(2), 16);
        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;
        for j in 0..16 {
            sumi1 += (x_blocks[ib].qs[j] as i32) * (KVALUES_IQ4NL[(qs[j] & 0xf) as usize] as i32);
            sumi2 += (x_blocks[ib].qs[j + 16] as i32) * (KVALUES_IQ4NL[(qs[j] >> 4) as usize] as i32);
        }
        sumf += d * (sumi1 + sumi2) as f32;
        ib += 1;
    }
    sumf
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn vec_dot_iq3_xxs_q8_k_avx2(weight_bytes: &[u8], x_blocks: &[BlockQ8_K]) -> f32 {
    use core::arch::x86_64::*;
    let nb = x_blocks.len();
    assert_eq!(weight_bytes.len(), nb * 98);

    let mut accumf = _mm256_setzero_ps();

    for i in 0..nb {
        let b_ptr = weight_bytes.as_ptr().add(i * 98);
        let d_f16 = half::f16::from_le_bytes([*b_ptr, *b_ptr.add(1)]).to_f32();
        let d = d_f16 * x_blocks[i].d;

        let mut q3_ptr = b_ptr.add(2); // 64 bytes grid indices
        let mut gas_ptr = b_ptr.add(66); // 32 bytes scales/signs
        let mut q8_ptr = x_blocks[i].qs.as_ptr();

        let mut sumi1 = _mm256_setzero_si256();
        let mut sumi2 = _mm256_setzero_si256();

        for _ in 0..4 {
            let q8_1 = _mm256_loadu_si256(q8_ptr as *const __m256i);
            q8_ptr = q8_ptr.add(32);
            let q8_2 = _mm256_loadu_si256(q8_ptr as *const __m256i);
            q8_ptr = q8_ptr.add(32);

            let q3 = std::slice::from_raw_parts(q3_ptr, 16);
            q3_ptr = q3_ptr.add(16);

            let q2_1 = _mm256_set_epi32(
                IQ3XXS_GRID[q3[7] as usize] as i32,
                IQ3XXS_GRID[q3[6] as usize] as i32,
                IQ3XXS_GRID[q3[5] as usize] as i32,
                IQ3XXS_GRID[q3[4] as usize] as i32,
                IQ3XXS_GRID[q3[3] as usize] as i32,
                IQ3XXS_GRID[q3[2] as usize] as i32,
                IQ3XXS_GRID[q3[1] as usize] as i32,
                IQ3XXS_GRID[q3[0] as usize] as i32,
            );

            let q2_2 = _mm256_set_epi32(
                IQ3XXS_GRID[q3[15] as usize] as i32,
                IQ3XXS_GRID[q3[14] as usize] as i32,
                IQ3XXS_GRID[q3[13] as usize] as i32,
                IQ3XXS_GRID[q3[12] as usize] as i32,
                IQ3XXS_GRID[q3[11] as usize] as i32,
                IQ3XXS_GRID[q3[10] as usize] as i32,
                IQ3XXS_GRID[q3[9] as usize] as i32,
                IQ3XXS_GRID[q3[8] as usize] as i32,
            );

            let aux32_0 = u32::from_le_bytes([*gas_ptr, *gas_ptr.add(1), *gas_ptr.add(2), *gas_ptr.add(3)]);
            let aux32_1 = u32::from_le_bytes([*gas_ptr.add(4), *gas_ptr.add(5), *gas_ptr.add(6), *gas_ptr.add(7)]);
            gas_ptr = gas_ptr.add(8);

            let s2_1 = _mm256_set_epi64x(
                SIGNS64_IQ2XS[((aux32_0 >> 21) & 127) as usize] as i64,
                SIGNS64_IQ2XS[((aux32_0 >> 14) & 127) as usize] as i64,
                SIGNS64_IQ2XS[((aux32_0 >> 7) & 127) as usize] as i64,
                SIGNS64_IQ2XS[((aux32_0 >> 0) & 127) as usize] as i64,
            );

            let s2_2 = _mm256_set_epi64x(
                SIGNS64_IQ2XS[((aux32_1 >> 21) & 127) as usize] as i64,
                SIGNS64_IQ2XS[((aux32_1 >> 14) & 127) as usize] as i64,
                SIGNS64_IQ2XS[((aux32_1 >> 7) & 127) as usize] as i64,
                SIGNS64_IQ2XS[((aux32_1 >> 0) & 127) as usize] as i64,
            );

            let q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            let q8s_2 = _mm256_sign_epi8(q8_2, s2_2);

            let dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
            let dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);

            let ls1 = (aux32_0 >> 28) as i16;
            let ls2 = (aux32_1 >> 28) as i16;

            let p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2 * ls1 + 1));
            let p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2 * ls2 + 1));

            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)),
            accumf,
        );
    }

    0.25f32 * hsum_float_8(accumf)
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_iq4_nl_prequant(out: &mut [f32], weight_bytes: &[u8], x_blocks: &[BlockQ8_0], n_rows: usize, n_cols: usize) {
    use core::arch::x86_64::*;
    let nb = x_blocks.len();
    let row_bytes = (n_cols / 32) * 18;
    assert_eq!(weight_bytes.len(), n_rows * row_bytes);

    let values128 = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let m4b = _mm_set1_epi8(0x0f);
    let mone = _mm256_set1_epi16(1);

    let mut r = 0;
    while r + 3 < n_rows {
        let r0_ptr = weight_bytes.as_ptr().add(r * row_bytes);
        let r1_ptr = weight_bytes.as_ptr().add((r + 1) * row_bytes);
        let r2_ptr = weight_bytes.as_ptr().add((r + 2) * row_bytes);
        let r3_ptr = weight_bytes.as_ptr().add((r + 3) * row_bytes);

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for ib in 0..nb {
            let q8b = _mm256_loadu_si256(x_blocks[ib].qs.as_ptr() as *const __m256i);
            let x_d = x_blocks[ib].d.to_f32();

            // Row 0
            let b0 = r0_ptr.add(ib * 18);
            let q4_0 = _mm_loadu_si128(b0.add(2) as *const __m128i);
            let q4b_0 = _mm256_insertf128_si256(
                _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(q4_0, m4b))),
                _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4_0, 4), m4b)),
                1,
            );
            let p16_0 = mul_add_epi8(q4b_0, q8b);
            let p_0 = _mm256_madd_epi16(p16_0, mone);
            let d0 = half::f16::from_le_bytes([*b0, *b0.add(1)]).to_f32() * x_d;
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(p_0), acc0);

            // Row 1
            let b1 = r1_ptr.add(ib * 18);
            let q4_1 = _mm_loadu_si128(b1.add(2) as *const __m128i);
            let q4b_1 = _mm256_insertf128_si256(
                _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(q4_1, m4b))),
                _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4_1, 4), m4b)),
                1,
            );
            let p16_1 = mul_add_epi8(q4b_1, q8b);
            let p_1 = _mm256_madd_epi16(p16_1, mone);
            let d1 = half::f16::from_le_bytes([*b1, *b1.add(1)]).to_f32() * x_d;
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), _mm256_cvtepi32_ps(p_1), acc1);

            // Row 2
            let b2 = r2_ptr.add(ib * 18);
            let q4_2 = _mm_loadu_si128(b2.add(2) as *const __m128i);
            let q4b_2 = _mm256_insertf128_si256(
                _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(q4_2, m4b))),
                _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4_2, 4), m4b)),
                1,
            );
            let p16_2 = mul_add_epi8(q4b_2, q8b);
            let p_2 = _mm256_madd_epi16(p16_2, mone);
            let d2 = half::f16::from_le_bytes([*b2, *b2.add(1)]).to_f32() * x_d;
            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(d2), _mm256_cvtepi32_ps(p_2), acc2);

            // Row 3
            let b3 = r3_ptr.add(ib * 18);
            let q4_3 = _mm_loadu_si128(b3.add(2) as *const __m128i);
            let q4b_3 = _mm256_insertf128_si256(
                _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(q4_3, m4b))),
                _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4_3, 4), m4b)),
                1,
            );
            let p16_3 = mul_add_epi8(q4b_3, q8b);
            let p_3 = _mm256_madd_epi16(p16_3, mone);
            let d3 = half::f16::from_le_bytes([*b3, *b3.add(1)]).to_f32() * x_d;
            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(d3), _mm256_cvtepi32_ps(p_3), acc3);
        }

        out[r] = hsum_float_8(acc0);
        out[r + 1] = hsum_float_8(acc1);
        out[r + 2] = hsum_float_8(acc2);
        out[r + 3] = hsum_float_8(acc3);
        r += 4;
    }

    while r < n_rows {
        let row_slice = &weight_bytes[r * row_bytes..(r + 1) * row_bytes];
        out[r] = vec_dot_iq4_nl_q8_0_avx2(row_slice, x_blocks);
        r += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_iq3_xxs_prequant(out: &mut [f32], weight_bytes: &[u8], x_q8_k: &[BlockQ8_K], n_rows: usize, n_cols: usize) {
    let row_bytes = (n_cols / 256) * 98;
    assert_eq!(weight_bytes.len(), n_rows * row_bytes);

    let mut r = 0;
    while r + 3 < n_rows {
        let row0 = std::slice::from_raw_parts(weight_bytes.as_ptr().add(r * row_bytes), row_bytes);
        let row1 = std::slice::from_raw_parts(weight_bytes.as_ptr().add((r + 1) * row_bytes), row_bytes);
        let row2 = std::slice::from_raw_parts(weight_bytes.as_ptr().add((r + 2) * row_bytes), row_bytes);
        let row3 = std::slice::from_raw_parts(weight_bytes.as_ptr().add((r + 3) * row_bytes), row_bytes);

        out[r] = vec_dot_iq3_xxs_q8_k_avx2(row0, x_q8_k);
        out[r + 1] = vec_dot_iq3_xxs_q8_k_avx2(row1, x_q8_k);
        out[r + 2] = vec_dot_iq3_xxs_q8_k_avx2(row2, x_q8_k);
        out[r + 3] = vec_dot_iq3_xxs_q8_k_avx2(row3, x_q8_k);
        r += 4;
    }
    while r < n_rows {
        let row_slice = &weight_bytes[r * row_bytes..(r + 1) * row_bytes];
        out[r] = vec_dot_iq3_xxs_q8_k_avx2(row_slice, x_q8_k);
        r += 1;
    }
}

pub fn gemv_iq4_nl(out: &mut [f32], weight_bytes: &[u8], x: &[f32], n_rows: usize, n_cols: usize) {
    assert_eq!(n_cols % 32, 0);
    assert_eq!(out.len(), n_rows);
    let nb = n_cols / 32;
    let mut x_q8_0 = vec![BlockQ8_0 { d: half::f16::ZERO, qs: [0; 32] }; nb];
    quantize_row_q8_0(x, &mut x_q8_0);
    unsafe { gemv_iq4_nl_prequant(out, weight_bytes, &x_q8_0, n_rows, n_cols) };
}

pub fn gemv_iq3_xxs(out: &mut [f32], weight_bytes: &[u8], x: &[f32], n_rows: usize, n_cols: usize) {
    assert_eq!(n_cols % 256, 0);
    assert_eq!(out.len(), n_rows);
    let nb = n_cols / 256;
    let mut x_q8_k = vec![BlockQ8_K { d: 0.0, qs: [0; 256], bsums: [0; 16] }; nb];
    quantize_row_q8_k(x, &mut x_q8_k);
    unsafe { gemv_iq3_xxs_prequant(out, weight_bytes, &x_q8_k, n_rows, n_cols) };
}

pub fn gemv_q8_0(out: &mut [f32], weight_bytes: &[u8], x: &[f32], n_rows: usize, n_cols: usize) {
    assert_eq!(n_cols % 32, 0);
    assert_eq!(out.len(), n_rows);
    let row_bytes = (n_cols / 32) * 34;
    assert_eq!(weight_bytes.len(), n_rows * row_bytes);

    let nb = n_cols / 32;
    let mut x_q8_0 = vec![BlockQ8_0 { d: half::f16::ZERO, qs: [0; 32] }; nb];
    quantize_row_q8_0(x, &mut x_q8_0);

    for r in 0..n_rows {
        let row_slice = &weight_bytes[r * row_bytes..(r + 1) * row_bytes];
        out[r] = unsafe { vec_dot_q8_0_q8_0_avx2(row_slice, &x_q8_0) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_dot_q8_0() {
        let n = 64;
        let mut x = vec![0f32; n];
        for i in 0..n {
            x[i] = ((i as f32) * 0.1).sin();
        }
        let nb = n / 32;
        let mut x_q8_0 = vec![BlockQ8_0 { d: half::f16::ZERO, qs: [0; 32] }; nb];
        quantize_row_q8_0(&x, &mut x_q8_0);

        // Synthetic Q8_0 weights
        let mut w_bytes = vec![0u8; nb * 34];
        for b in 0..nb {
            let d_bytes = half::f16::from_f32(0.05).to_le_bytes();
            w_bytes[b * 34] = d_bytes[0];
            w_bytes[b * 34 + 1] = d_bytes[1];
            for j in 0..32 {
                w_bytes[b * 34 + 2 + j] = ((j as i8) - 16) as u8;
            }
        }

        // Scalar dequantize
        let mut w_dequant = vec![0f32; n];
        dequantize_row_q8_0(&w_bytes, &mut w_dequant);

        let scalar_dot: f32 = w_dequant.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let simd_dot = unsafe { vec_dot_q8_0_q8_0_avx2(&w_bytes, &x_q8_0) };

        println!("Q8_0 scalar_dot: {scalar_dot}, simd_dot: {simd_dot}");
        let diff = (scalar_dot - simd_dot).abs();
        assert!(diff < 0.05 * scalar_dot.abs().max(1.0), "Q8_0 mismatch: {scalar_dot} vs {simd_dot}");
    }

    #[test]
    fn test_vec_dot_iq4_nl() {
        let n = 64;
        let mut x = vec![0f32; n];
        for i in 0..n {
            x[i] = ((i as f32) * 0.2).cos();
        }
        let nb = n / 32;
        let mut x_q8_0 = vec![BlockQ8_0 { d: half::f16::ZERO, qs: [0; 32] }; nb];
        quantize_row_q8_0(&x, &mut x_q8_0);

        // Synthetic IQ4_NL weights
        let mut w_bytes = vec![0u8; nb * 18];
        for b in 0..nb {
            let d_bytes = half::f16::from_f32(0.02).to_le_bytes();
            w_bytes[b * 18] = d_bytes[0];
            w_bytes[b * 18 + 1] = d_bytes[1];
            for j in 0..16 {
                w_bytes[b * 18 + 2 + j] = ((j as u8) | ((15 - j as u8) << 4));
            }
        }

        // Scalar dequantize
        let mut w_dequant = vec![0f32; n];
        dequantize_row_iq4_nl(&w_bytes, &mut w_dequant);

        let scalar_dot: f32 = w_dequant.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let simd_dot = unsafe { vec_dot_iq4_nl_q8_0_avx2(&w_bytes, &x_q8_0) };

        println!("IQ4_NL scalar_dot: {scalar_dot}, simd_dot: {simd_dot}");
        let diff = (scalar_dot - simd_dot).abs();
        assert!(diff < 0.05 * scalar_dot.abs().max(1.0), "IQ4_NL mismatch: {scalar_dot} vs {simd_dot}");
    }

    #[test]
    fn test_vec_dot_iq3_xxs() {
        let n = 256;
        let mut x = vec![0f32; n];
        for i in 0..n {
            x[i] = ((i as f32) * 0.05).sin();
        }
        let nb = n / 256;
        let mut x_q8_k = vec![BlockQ8_K { d: 0.0, qs: [0; 256], bsums: [0; 16] }; nb];
        quantize_row_q8_k(&x, &mut x_q8_k);

        // Synthetic IQ3_XXS weights
        let mut w_bytes = vec![0u8; nb * 98];
        for b in 0..nb {
            let d_bytes = half::f16::from_f32(0.01).to_le_bytes();
            w_bytes[b * 98] = d_bytes[0];
            w_bytes[b * 98 + 1] = d_bytes[1];
            for j in 0..64 {
                w_bytes[b * 98 + 2 + j] = (j % 256) as u8;
            }
            for j in 0..32 {
                w_bytes[b * 98 + 66 + j] = ((j * 7) % 256) as u8;
            }
        }

        // Scalar dequantize
        let mut w_dequant = vec![0f32; n];
        dequantize_row_iq3_xxs(&w_bytes, &mut w_dequant);

        let scalar_dot: f32 = w_dequant.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let simd_dot = unsafe { vec_dot_iq3_xxs_q8_k_avx2(&w_bytes, &x_q8_k) };

        println!("IQ3_XXS scalar_dot: {scalar_dot}, simd_dot: {simd_dot}");
        let diff = (scalar_dot - simd_dot).abs();
        assert!(diff < 0.05 * scalar_dot.abs().max(1.0), "IQ3_XXS mismatch: {scalar_dot} vs {simd_dot}");
    }
}
