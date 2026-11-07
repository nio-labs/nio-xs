#![allow(unsafe_op_in_unsafe_fn)]
// Quantization kernels for AST Micro Engine: IQ3_XXS, IQ4_NL, Q8_0, Q4_0, and F32.

pub const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

pub const KSIGNS_IQ2XS: [u8; 128] = [
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149,
    150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170,
    43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212,
    85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232,
    105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123,
    252, 125, 126, 255,
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

#[inline]
fn best_index_iq4nl(val_to_match: f32) -> u8 {
    if val_to_match <= KVALUES_IQ4NL[0] as f32 {
        return 0;
    }
    if val_to_match >= KVALUES_IQ4NL[15] as f32 {
        return 15;
    }
    let mut l = 0usize;
    let mut r = 15usize;
    while r - l > 1 {
        let m = (l + r) / 2;
        if (KVALUES_IQ4NL[m] as f32) > val_to_match {
            r = m;
        } else {
            l = m;
        }
    }
    let diff_l = val_to_match - (KVALUES_IQ4NL[l] as f32);
    let diff_r = (KVALUES_IQ4NL[r] as f32) - val_to_match;
    if diff_l < diff_r { l as u8 } else { r as u8 }
}

/// Quantizes 32 f32 weights into an 18-byte IQ4_NL block (dtype 20).
pub fn quantize_block_iq4_nl(x: &[f32; 32], out_bytes: &mut [u8; 18]) {
    let mut amax = 0.0f32;
    let mut max_val = 0.0f32;
    for &v in x {
        let av = v.abs();
        if av > amax {
            amax = av;
            max_val = v;
        }
    }
    if amax < 1e-15 {
        out_bytes.fill(0);
        return;
    }

    let mut d = -max_val / (KVALUES_IQ4NL[0] as f32);
    let id_val = 1.0 / d;
    let mut sumqx = 0.0f32;
    let mut sumq2 = 0.0f32;
    for &v in x {
        let idx = best_index_iq4nl(id_val * v) as usize;
        let q = KVALUES_IQ4NL[idx] as f32;
        sumqx += q * v;
        sumq2 += q * q;
    }
    if sumq2 > 0.0 {
        d = sumqx / sumq2;
    }
    let mut best = d * sumqx;

    for itry in -7..=7 {
        let id_try = (itry as f32 + KVALUES_IQ4NL[0] as f32) / max_val;
        let mut sqx = 0.0f32;
        let mut sq2 = 0.0f32;
        for &v in x {
            let idx = best_index_iq4nl(id_try * v) as usize;
            let q = KVALUES_IQ4NL[idx] as f32;
            sqx += q * v;
            sq2 += q * q;
        }
        if sq2 > 0.0 && sqx * sqx > best * sq2 {
            d = sqx / sq2;
            best = d * sqx;
        }
    }

    let d_f16 = half::f16::from_f32(d);
    let d_bytes = d_f16.to_le_bytes();
    out_bytes[0] = d_bytes[0];
    out_bytes[1] = d_bytes[1];

    let d_actual = d_f16.to_f32();
    let id_actual = if d_actual != 0.0 { 1.0 / d_actual } else { 0.0 };
    let mut l_indices = [0u8; 32];
    for (j, &v) in x.iter().enumerate() {
        l_indices[j] = best_index_iq4nl(id_actual * v);
    }
    for j in 0..16 {
        out_bytes[2 + j] = l_indices[j] | (l_indices[j + 16] << 4);
    }
}

/// Quantizes an array of f32 weights into raw IQ4_NL bytes (dtype 20).
pub fn quantize_row_iq4_nl(src: &[f32], dst: &mut [u8]) {
    assert_eq!(src.len() % 32, 0);
    let nb = src.len() / 32;
    assert_eq!(dst.len(), nb * 18);
    for i in 0..nb {
        let x: &[f32; 32] = src[i * 32..(i + 1) * 32].try_into().unwrap();
        let out_block: &mut [u8; 18] = (&mut dst[i * 18..(i + 1) * 18]).try_into().unwrap();
        quantize_block_iq4_nl(x, out_block);
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
                    let s1 = if (signs & KMASK_IQ2XS[j]) != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let s2 = if (signs & KMASK_IQ2XS[j + 4]) != 0 {
                        -1.0
                    } else {
                        1.0
                    };
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

// Expand the canonical sign masks at compile time. Every byte consumed by
// _mm256_sign_epi8 must be +1 or -1; a zero byte silently drops that lane.
pub const SIGNS64_IQ2XS: [u64; 128] = {
    let mut signs = [0u64; 128];
    let mut i = 0;
    while i < signs.len() {
        let mut lane = 0;
        while lane < 8 {
            let byte = if KSIGNS_IQ2XS[i] & (1 << lane) == 0 {
                1u64
            } else {
                255u64
            };
            signs[i] |= byte << (lane * 8);
            lane += 1;
        }
        i += 1;
    }
    signs
};

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn f16_to_f32_hw(bits: u16) -> f32 {
    use core::arch::x86_64::*;
    let v = _mm_cvtsi32_si128(bits as i32);
    let f = _mm_cvtph_ps(v);
    _mm_cvtss_f32(f)
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_0 {
    pub d: f32,
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
        let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
        // Q8_0 uses an FP16 scale, even though our internal block stores it in
        // f32 for SIMD access. Match the GGML CPU quantizer's rounding too.
        block.d = half::f16::from_f32(d).to_f32();
        for j in 0..32 {
            let x0 = chunk[j] * id;
            block.qs[j] = x0.round_ties_even().clamp(-128.0, 127.0) as i8;
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn quantize_row_q8_k_avx2(x: &[f32], out: &mut [BlockQ8_K]) {
    use core::arch::x86_64::*;
    let nb = out.len();
    let x_ptr = x.as_ptr();

    unsafe {
        for i in 0..nb {
            let chunk_ptr = x_ptr.add(i * 256);
            let block = &mut out[i];

            let mut max_vec = _mm256_setzero_ps();
            let mut min_vec = _mm256_setzero_ps();

            for k in 0..32 {
                let v = _mm256_loadu_ps(chunk_ptr.add(k * 8));
                max_vec = _mm256_max_ps(max_vec, v);
                min_vec = _mm256_min_ps(min_vec, v);
            }

            let mut max_arr = [0.0f32; 8];
            let mut min_arr = [0.0f32; 8];
            _mm256_storeu_ps(max_arr.as_mut_ptr(), max_vec);
            _mm256_storeu_ps(min_arr.as_mut_ptr(), min_vec);

            let mut pos_max = max_arr[0];
            let mut neg_min = min_arr[0];
            for k in 1..8 {
                if max_arr[k] > pos_max {
                    pos_max = max_arr[k];
                }
                if min_arr[k] < neg_min {
                    neg_min = min_arr[k];
                }
            }

            let max = if -neg_min > pos_max { neg_min } else { pos_max };
            if max == 0.0 {
                block.d = 0.0;
                block.qs.fill(0);
                continue;
            }

            let iscale = -127.0 / max;
            block.d = 1.0 / iscale;
            let iscale_vec = _mm256_set1_ps(iscale);

            for k in 0..8 {
                let offset = k * 32;
                let f0 = _mm256_loadu_ps(chunk_ptr.add(offset));
                let f1 = _mm256_loadu_ps(chunk_ptr.add(offset + 8));
                let f2 = _mm256_loadu_ps(chunk_ptr.add(offset + 16));
                let f3 = _mm256_loadu_ps(chunk_ptr.add(offset + 24));

                let i0 = _mm256_cvtps_epi32(_mm256_mul_ps(f0, iscale_vec));
                let i1 = _mm256_cvtps_epi32(_mm256_mul_ps(f1, iscale_vec));
                let i2 = _mm256_cvtps_epi32(_mm256_mul_ps(f2, iscale_vec));
                let i3 = _mm256_cvtps_epi32(_mm256_mul_ps(f3, iscale_vec));

                let p01 = _mm256_packs_epi32(i0, i1);
                let p23 = _mm256_packs_epi32(i2, i3);

                let p01_perm = _mm256_permute4x64_epi64(p01, 0xd8);
                let p23_perm = _mm256_permute4x64_epi64(p23, 0xd8);

                let packed8 = _mm256_packs_epi16(p01_perm, p23_perm);
                let packed8_perm = _mm256_permute4x64_epi64(packed8, 0xd8);

                _mm256_storeu_si256(
                    block.qs.as_mut_ptr().add(offset) as *mut __m256i,
                    packed8_perm,
                );
            }
        }
    }
}

pub fn quantize_row_q8_k(x: &[f32], out: &mut [BlockQ8_K]) {
    assert_eq!(x.len() % 256, 0);
    assert_eq!(out.len(), x.len() / 256);
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe {
            quantize_row_q8_k_avx2(x, out);
            return;
        }
    }
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

#[inline(always)]
pub fn fast_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
#[cfg(target_arch = "x86_64")]
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
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mul_add_epi8(
    x: core::arch::x86_64::__m256i,
    y: core::arch::x86_64::__m256i,
) -> core::arch::x86_64::__m256i {
    use core::arch::x86_64::*;
    let ax = _mm256_sign_epi8(x, x);
    let sy = _mm256_sign_epi8(y, x);
    _mm256_maddubs_epi16(ax, sy)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_i16_pairs_float(x: core::arch::x86_64::__m256i) -> core::arch::x86_64::__m256 {
    use core::arch::x86_64::*;
    let ones = _mm256_set1_epi16(1);
    let summed_pairs = _mm256_madd_epi16(ones, x);
    _mm256_cvtepi32_ps(summed_pairs)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mul_sum_i8_pairs_float(
    x: core::arch::x86_64::__m256i,
    y: core::arch::x86_64::__m256i,
) -> core::arch::x86_64::__m256 {
    use core::arch::x86_64::*;
    let ax = _mm256_sign_epi8(x, x);
    let sy = _mm256_sign_epi8(y, x);
    let dot = _mm256_maddubs_epi16(ax, sy);
    sum_i16_pairs_float(dot)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_q8_0_q8_0_avx2(weight_bytes: &[u8], x_blocks: &[BlockQ8_0]) -> f32 {
    use core::arch::x86_64::*;
    let nb = x_blocks.len();
    assert_eq!(weight_bytes.len(), nb * 34);

    let mut acc = _mm256_setzero_ps();
    for ib in 0..nb {
        if (ib & 3) == 0 && ib + 8 < nb {
            _mm_prefetch(
                weight_bytes.as_ptr().add((ib + 8) * 34) as *const i8,
                _MM_HINT_T0,
            );
        }
        let block_ptr = weight_bytes.as_ptr().add(ib * 34);
        let d_f32 = f16_to_f32_hw((block_ptr as *const u16).read_unaligned());
        let d = _mm256_set1_ps(d_f32 * x_blocks[ib].d);
        let qx = _mm256_loadu_si256(block_ptr.add(2) as *const __m256i);
        let qy = _mm256_loadu_si256(x_blocks[ib].qs.as_ptr() as *const __m256i);
        let q = mul_sum_i8_pairs_float(qx, qy);
        acc = _mm256_fmadd_ps(d, q, acc);
    }
    hsum_float_8(acc)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_q8_0_18blocks_avx2(
    weight_ptr: *const u8,
    x_qs: &[core::arch::x86_64::__m256i; 18],
    x_d: &[f32; 18],
) -> f32 {
    use core::arch::x86_64::*;

    // Prefetch row ahead
    _mm_prefetch(weight_ptr.add(612) as *const i8, _MM_HINT_T0);

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    let mut ib = 0;
    while ib < 18 {
        let b0_ptr = weight_ptr.add(ib * 34);
        let b1_ptr = weight_ptr.add((ib + 1) * 34);

        let d0_f32 = f16_to_f32_hw((b0_ptr as *const u16).read_unaligned());
        let d1_f32 = f16_to_f32_hw((b1_ptr as *const u16).read_unaligned());

        let d0 = _mm256_set1_ps(d0_f32 * x_d[ib]);
        let d1 = _mm256_set1_ps(d1_f32 * x_d[ib + 1]);

        let qx0 = _mm256_loadu_si256(b0_ptr.add(2) as *const __m256i);
        let qx1 = _mm256_loadu_si256(b1_ptr.add(2) as *const __m256i);

        let q0 = mul_sum_i8_pairs_float(qx0, x_qs[ib]);
        let q1 = mul_sum_i8_pairs_float(qx1, x_qs[ib + 1]);

        acc0 = _mm256_fmadd_ps(d0, q0, acc0);
        acc1 = _mm256_fmadd_ps(d1, q1, acc1);

        ib += 2;
    }

    hsum_float_8(_mm256_add_ps(acc0, acc1))
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_q8_0_28blocks_2rows_avx2(
    r0_ptr: *const u8,
    r1_ptr: *const u8,
    x_qs: &[core::arch::x86_64::__m256i; 28],
    x_d: &[f32; 28],
) -> (f32, f32) {
    use core::arch::x86_64::*;

    // Prefetch 4 rows ahead (4 * 952 = 3808 bytes)
    _mm_prefetch(r0_ptr.add(3808) as *const i8, _MM_HINT_T0);
    _mm_prefetch(r0_ptr.add(3808 + 512) as *const i8, _MM_HINT_T0);
    _mm_prefetch(r1_ptr.add(3808) as *const i8, _MM_HINT_T0);
    _mm_prefetch(r1_ptr.add(3808 + 512) as *const i8, _MM_HINT_T0);

    let mut r0_acc0 = _mm256_setzero_ps();
    let mut r0_acc1 = _mm256_setzero_ps();
    let mut r1_acc0 = _mm256_setzero_ps();
    let mut r1_acc1 = _mm256_setzero_ps();

    let mut ib = 0;
    while ib < 28 {
        if ib + 4 < 28 {
            _mm_prefetch(r0_ptr.add((ib + 4) * 34) as *const i8, _MM_HINT_T0);
            _mm_prefetch(r1_ptr.add((ib + 4) * 34) as *const i8, _MM_HINT_T0);
        }
        let act_q0 = x_qs[ib];
        let act_q1 = x_qs[ib + 1];
        let act_d0 = x_d[ib];
        let act_d1 = x_d[ib + 1];

        // Row 0
        let r0_b0 = r0_ptr.add(ib * 34);
        let r0_b1 = r0_ptr.add((ib + 1) * 34);
        let r0_d0 = _mm256_set1_ps(f16_to_f32_hw((r0_b0 as *const u16).read_unaligned()) * act_d0);
        let r0_d1 = _mm256_set1_ps(f16_to_f32_hw((r0_b1 as *const u16).read_unaligned()) * act_d1);
        let r0_qx0 = _mm256_loadu_si256(r0_b0.add(2) as *const __m256i);
        let r0_qx1 = _mm256_loadu_si256(r0_b1.add(2) as *const __m256i);
        let r0_q0 = mul_sum_i8_pairs_float(r0_qx0, act_q0);
        let r0_q1 = mul_sum_i8_pairs_float(r0_qx1, act_q1);
        r0_acc0 = _mm256_fmadd_ps(r0_d0, r0_q0, r0_acc0);
        r0_acc1 = _mm256_fmadd_ps(r0_d1, r0_q1, r0_acc1);

        // Row 1
        let r1_b0 = r1_ptr.add(ib * 34);
        let r1_b1 = r1_ptr.add((ib + 1) * 34);
        let r1_d0 = _mm256_set1_ps(f16_to_f32_hw((r1_b0 as *const u16).read_unaligned()) * act_d0);
        let r1_d1 = _mm256_set1_ps(f16_to_f32_hw((r1_b1 as *const u16).read_unaligned()) * act_d1);
        let r1_qx0 = _mm256_loadu_si256(r1_b0.add(2) as *const __m256i);
        let r1_qx1 = _mm256_loadu_si256(r1_b1.add(2) as *const __m256i);
        let r1_q0 = mul_sum_i8_pairs_float(r1_qx0, act_q0);
        let r1_q1 = mul_sum_i8_pairs_float(r1_qx1, act_q1);
        r1_acc0 = _mm256_fmadd_ps(r1_d0, r1_q0, r1_acc0);
        r1_acc1 = _mm256_fmadd_ps(r1_d1, r1_q1, r1_acc1);

        ib += 2;
    }

    let val0 = hsum_float_8(_mm256_add_ps(r0_acc0, r0_acc1));
    let val1 = hsum_float_8(_mm256_add_ps(r1_acc0, r1_acc1));
    (val0, val1)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_q8_0_28blocks_avx2(
    weight_ptr: *const u8,
    x_qs: &[core::arch::x86_64::__m256i; 28],
    x_d: &[f32; 28],
) -> f32 {
    use core::arch::x86_64::*;

    // Prefetch row ahead (28 blocks * 34 bytes = 952 bytes)
    _mm_prefetch(weight_ptr.add(952) as *const i8, _MM_HINT_T0);
    _mm_prefetch(weight_ptr.add(952 + 512) as *const i8, _MM_HINT_T0);

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    let mut ib = 0;
    while ib < 28 {
        let b0_ptr = weight_ptr.add(ib * 34);
        let b1_ptr = weight_ptr.add((ib + 1) * 34);

        let d0_f32 = f16_to_f32_hw((b0_ptr as *const u16).read_unaligned());
        let d1_f32 = f16_to_f32_hw((b1_ptr as *const u16).read_unaligned());

        let d0 = _mm256_set1_ps(d0_f32 * x_d[ib]);
        let d1 = _mm256_set1_ps(d1_f32 * x_d[ib + 1]);

        let qx0 = _mm256_loadu_si256(b0_ptr.add(2) as *const __m256i);
        let qx1 = _mm256_loadu_si256(b1_ptr.add(2) as *const __m256i);

        let q0 = mul_sum_i8_pairs_float(qx0, x_qs[ib]);
        let q1 = mul_sum_i8_pairs_float(qx1, x_qs[ib + 1]);

        acc0 = _mm256_fmadd_ps(d0, q0, acc0);
        acc1 = _mm256_fmadd_ps(d1, q1, acc1);

        ib += 2;
    }

    hsum_float_8(_mm256_add_ps(acc0, acc1))
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_q8_0_48blocks_2rows_avx2(
    r0_ptr: *const u8,
    r1_ptr: *const u8,
    x_qs: &[core::arch::x86_64::__m256i; 48],
    x_d: &[f32; 48],
) -> (f32, f32) {
    use core::arch::x86_64::*;

    // Prefetch 4 rows ahead (4 * 1632 = 6528 bytes)
    _mm_prefetch(r0_ptr.add(6528) as *const i8, _MM_HINT_T0);
    _mm_prefetch(r0_ptr.add(6528 + 512) as *const i8, _MM_HINT_T0);
    _mm_prefetch(r0_ptr.add(6528 + 1024) as *const i8, _MM_HINT_T0);
    _mm_prefetch(r1_ptr.add(6528) as *const i8, _MM_HINT_T0);
    _mm_prefetch(r1_ptr.add(6528 + 512) as *const i8, _MM_HINT_T0);
    _mm_prefetch(r1_ptr.add(6528 + 1024) as *const i8, _MM_HINT_T0);

    let mut r0_acc0 = _mm256_setzero_ps();
    let mut r0_acc1 = _mm256_setzero_ps();
    let mut r1_acc0 = _mm256_setzero_ps();
    let mut r1_acc1 = _mm256_setzero_ps();

    let mut ib = 0;
    while ib < 48 {
        if ib + 4 < 48 {
            _mm_prefetch(r0_ptr.add((ib + 4) * 34) as *const i8, _MM_HINT_T0);
            _mm_prefetch(r1_ptr.add((ib + 4) * 34) as *const i8, _MM_HINT_T0);
        }
        let act_q0 = x_qs[ib];
        let act_q1 = x_qs[ib + 1];
        let act_d0 = x_d[ib];
        let act_d1 = x_d[ib + 1];

        // Row 0
        let r0_b0 = r0_ptr.add(ib * 34);
        let r0_b1 = r0_ptr.add((ib + 1) * 34);
        let r0_d0 = _mm256_set1_ps(f16_to_f32_hw((r0_b0 as *const u16).read_unaligned()) * act_d0);
        let r0_d1 = _mm256_set1_ps(f16_to_f32_hw((r0_b1 as *const u16).read_unaligned()) * act_d1);
        let r0_qx0 = _mm256_loadu_si256(r0_b0.add(2) as *const __m256i);
        let r0_qx1 = _mm256_loadu_si256(r0_b1.add(2) as *const __m256i);
        let r0_q0 = mul_sum_i8_pairs_float(r0_qx0, act_q0);
        let r0_q1 = mul_sum_i8_pairs_float(r0_qx1, act_q1);
        r0_acc0 = _mm256_fmadd_ps(r0_d0, r0_q0, r0_acc0);
        r0_acc1 = _mm256_fmadd_ps(r0_d1, r0_q1, r0_acc1);

        // Row 1
        let r1_b0 = r1_ptr.add(ib * 34);
        let r1_b1 = r1_ptr.add((ib + 1) * 34);
        let r1_d0 = _mm256_set1_ps(f16_to_f32_hw((r1_b0 as *const u16).read_unaligned()) * act_d0);
        let r1_d1 = _mm256_set1_ps(f16_to_f32_hw((r1_b1 as *const u16).read_unaligned()) * act_d1);
        let r1_qx0 = _mm256_loadu_si256(r1_b0.add(2) as *const __m256i);
        let r1_qx1 = _mm256_loadu_si256(r1_b1.add(2) as *const __m256i);
        let r1_q0 = mul_sum_i8_pairs_float(r1_qx0, act_q0);
        let r1_q1 = mul_sum_i8_pairs_float(r1_qx1, act_q1);
        r1_acc0 = _mm256_fmadd_ps(r1_d0, r1_q0, r1_acc0);
        r1_acc1 = _mm256_fmadd_ps(r1_d1, r1_q1, r1_acc1);

        ib += 2;
    }

    let val0 = hsum_float_8(_mm256_add_ps(r0_acc0, r0_acc1));
    let val1 = hsum_float_8(_mm256_add_ps(r1_acc0, r1_acc1));
    (val0, val1)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_q8_0_48blocks_avx2(
    weight_ptr: *const u8,
    x_qs: &[core::arch::x86_64::__m256i; 48],
    x_d: &[f32; 48],
) -> f32 {
    use core::arch::x86_64::*;

    // Prefetch row ahead (48 blocks * 34 bytes = 1632 bytes)
    _mm_prefetch(weight_ptr.add(1632) as *const i8, _MM_HINT_T0);
    _mm_prefetch(weight_ptr.add(1632 + 512) as *const i8, _MM_HINT_T0);

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    let mut ib = 0;
    while ib < 48 {
        let b0_ptr = weight_ptr.add(ib * 34);
        let b1_ptr = weight_ptr.add((ib + 1) * 34);

        let d0_f32 = f16_to_f32_hw((b0_ptr as *const u16).read_unaligned());
        let d1_f32 = f16_to_f32_hw((b1_ptr as *const u16).read_unaligned());

        let d0 = _mm256_set1_ps(d0_f32 * x_d[ib]);
        let d1 = _mm256_set1_ps(d1_f32 * x_d[ib + 1]);

        let qx0 = _mm256_loadu_si256(b0_ptr.add(2) as *const __m256i);
        let qx1 = _mm256_loadu_si256(b1_ptr.add(2) as *const __m256i);

        let q0 = mul_sum_i8_pairs_float(qx0, x_qs[ib]);
        let q1 = mul_sum_i8_pairs_float(qx1, x_qs[ib + 1]);

        acc0 = _mm256_fmadd_ps(d0, q0, acc0);
        acc1 = _mm256_fmadd_ps(d1, q1, acc1);

        ib += 2;
    }

    hsum_float_8(_mm256_add_ps(acc0, acc1))
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_q8_0_96blocks_2rows_avx2(
    r0_ptr: *const u8,
    r1_ptr: *const u8,
    x_qs: &[core::arch::x86_64::__m256i; 96],
    x_d: &[f32; 96],
) -> (f32, f32) {
    use core::arch::x86_64::*;

    // Prefetch 4 rows ahead (4 * 3264 = 13056 bytes)
    _mm_prefetch(r0_ptr.add(13056) as *const i8, _MM_HINT_T0);
    _mm_prefetch(r0_ptr.add(13056 + 1024) as *const i8, _MM_HINT_T0);
    _mm_prefetch(r1_ptr.add(13056) as *const i8, _MM_HINT_T0);
    _mm_prefetch(r1_ptr.add(13056 + 1024) as *const i8, _MM_HINT_T0);

    let mut r0_acc0 = _mm256_setzero_ps();
    let mut r0_acc1 = _mm256_setzero_ps();
    let mut r1_acc0 = _mm256_setzero_ps();
    let mut r1_acc1 = _mm256_setzero_ps();

    let mut ib = 0;
    while ib < 96 {
        if ib + 4 < 96 {
            _mm_prefetch(r0_ptr.add((ib + 4) * 34) as *const i8, _MM_HINT_T0);
            _mm_prefetch(r1_ptr.add((ib + 4) * 34) as *const i8, _MM_HINT_T0);
        }
        let act_q0 = x_qs[ib];
        let act_q1 = x_qs[ib + 1];
        let act_d0 = x_d[ib];
        let act_d1 = x_d[ib + 1];

        // Row 0
        let r0_b0 = r0_ptr.add(ib * 34);
        let r0_b1 = r0_ptr.add((ib + 1) * 34);
        let r0_d0 = _mm256_set1_ps(f16_to_f32_hw((r0_b0 as *const u16).read_unaligned()) * act_d0);
        let r0_d1 = _mm256_set1_ps(f16_to_f32_hw((r0_b1 as *const u16).read_unaligned()) * act_d1);
        let r0_qx0 = _mm256_loadu_si256(r0_b0.add(2) as *const __m256i);
        let r0_qx1 = _mm256_loadu_si256(r0_b1.add(2) as *const __m256i);
        let r0_q0 = mul_sum_i8_pairs_float(r0_qx0, act_q0);
        let r0_q1 = mul_sum_i8_pairs_float(r0_qx1, act_q1);
        r0_acc0 = _mm256_fmadd_ps(r0_d0, r0_q0, r0_acc0);
        r0_acc1 = _mm256_fmadd_ps(r0_d1, r0_q1, r0_acc1);

        // Row 1
        let r1_b0 = r1_ptr.add(ib * 34);
        let r1_b1 = r1_ptr.add((ib + 1) * 34);
        let r1_d0 = _mm256_set1_ps(f16_to_f32_hw((r1_b0 as *const u16).read_unaligned()) * act_d0);
        let r1_d1 = _mm256_set1_ps(f16_to_f32_hw((r1_b1 as *const u16).read_unaligned()) * act_d1);
        let r1_qx0 = _mm256_loadu_si256(r1_b0.add(2) as *const __m256i);
        let r1_qx1 = _mm256_loadu_si256(r1_b1.add(2) as *const __m256i);
        let r1_q0 = mul_sum_i8_pairs_float(r1_qx0, act_q0);
        let r1_q1 = mul_sum_i8_pairs_float(r1_qx1, act_q1);
        r1_acc0 = _mm256_fmadd_ps(r1_d0, r1_q0, r1_acc0);
        r1_acc1 = _mm256_fmadd_ps(r1_d1, r1_q1, r1_acc1);

        ib += 2;
    }

    let val0 = hsum_float_8(_mm256_add_ps(r0_acc0, r0_acc1));
    let val1 = hsum_float_8(_mm256_add_ps(r1_acc0, r1_acc1));
    (val0, val1)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_q8_0_96blocks_avx2(
    weight_ptr: *const u8,
    x_qs: &[core::arch::x86_64::__m256i; 96],
    x_d: &[f32; 96],
) -> f32 {
    use core::arch::x86_64::*;

    // Prefetch row ahead (96 blocks * 34 bytes = 3264 bytes)
    _mm_prefetch(weight_ptr.add(3264) as *const i8, _MM_HINT_T0);
    _mm_prefetch(weight_ptr.add(3264 + 1024) as *const i8, _MM_HINT_T0);

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    let mut ib = 0;
    while ib < 96 {
        let b0_ptr = weight_ptr.add(ib * 34);
        let b1_ptr = weight_ptr.add((ib + 1) * 34);

        let d0_f32 = f16_to_f32_hw((b0_ptr as *const u16).read_unaligned());
        let d1_f32 = f16_to_f32_hw((b1_ptr as *const u16).read_unaligned());

        let d0 = _mm256_set1_ps(d0_f32 * x_d[ib]);
        let d1 = _mm256_set1_ps(d1_f32 * x_d[ib + 1]);

        let qx0 = _mm256_loadu_si256(b0_ptr.add(2) as *const __m256i);
        let qx1 = _mm256_loadu_si256(b1_ptr.add(2) as *const __m256i);

        let q0 = mul_sum_i8_pairs_float(qx0, x_qs[ib]);
        let q1 = mul_sum_i8_pairs_float(qx1, x_qs[ib + 1]);

        acc0 = _mm256_fmadd_ps(d0, q0, acc0);
        acc1 = _mm256_fmadd_ps(d1, q1, acc1);

        ib += 2;
    }

    hsum_float_8(_mm256_add_ps(acc0, acc1))
}

#[cfg(target_arch = "x86_64")]
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
        if (ib & 3) == 0 && ib + 8 < nb {
            _mm_prefetch(
                weight_bytes.as_ptr().add((ib + 8) * 18) as *const i8,
                _MM_HINT_T0,
            );
        }
        let b0_ptr = weight_bytes.as_ptr().add(ib * 18);
        let b1_ptr = weight_bytes.as_ptr().add((ib + 1) * 18);

        let q4bits_1 = _mm_loadu_si128(b0_ptr.add(2) as *const __m128i);
        let q4bits_2 = _mm_loadu_si128(b1_ptr.add(2) as *const __m128i);

        let q8b_1 = _mm256_loadu_si256(x_blocks[ib].qs.as_ptr() as *const __m256i);
        let q8b_2 = _mm256_loadu_si256(x_blocks[ib + 1].qs.as_ptr() as *const __m256i);

        let q4b_1_low = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b));
        let q4b_1_high =
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b));
        let q4b_1 = _mm256_insertf128_si256(_mm256_castsi128_si256(q4b_1_low), q4b_1_high, 1);

        let q4b_2_low = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b));
        let q4b_2_high =
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b));
        let q4b_2 = _mm256_insertf128_si256(_mm256_castsi128_si256(q4b_2_low), q4b_2_high, 1);

        let p16_1 = mul_add_epi8(q4b_1, q8b_1);
        let p16_2 = mul_add_epi8(q4b_2, q8b_2);

        let p_1 = _mm256_madd_epi16(p16_1, mone);
        let p_2 = _mm256_madd_epi16(p16_2, mone);

        let d0 = f16_to_f32_hw((b0_ptr as *const u16).read_unaligned()) * x_blocks[ib].d;
        let d1 = f16_to_f32_hw((b1_ptr as *const u16).read_unaligned()) * x_blocks[ib + 1].d;

        accum1 = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(p_1), accum1);
        accum2 = _mm256_fmadd_ps(_mm256_set1_ps(d1), _mm256_cvtepi32_ps(p_2), accum2);

        ib += 2;
    }

    let mut sumf = hsum_float_8(_mm256_add_ps(accum1, accum2));
    while ib < nb {
        let b_ptr = weight_bytes.as_ptr().add(ib * 18);
        let d = f16_to_f32_hw((b_ptr as *const u16).read_unaligned()) * x_blocks[ib].d;
        let qs = std::slice::from_raw_parts(b_ptr.add(2), 16);
        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;
        for j in 0..16 {
            sumi1 += (x_blocks[ib].qs[j] as i32) * (KVALUES_IQ4NL[(qs[j] & 0xf) as usize] as i32);
            sumi2 +=
                (x_blocks[ib].qs[j + 16] as i32) * (KVALUES_IQ4NL[(qs[j] >> 4) as usize] as i32);
        }
        sumf += d * (sumi1 + sumi2) as f32;
        ib += 1;
    }
    sumf
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_iq4_nl_48blocks_2rows_avx2(
    r0_ptr: *const u8,
    r1_ptr: *const u8,
    x_qs: &[core::arch::x86_64::__m256i; 48],
    x_d: &[f32; 48],
) -> (f32, f32) {
    use core::arch::x86_64::*;
    _mm_prefetch(r0_ptr.add(864) as *const i8, _MM_HINT_T0);
    _mm_prefetch(r1_ptr.add(864) as *const i8, _MM_HINT_T0);

    let values128 = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let m4b = _mm_set1_epi8(0x0f);
    let mone = _mm256_set1_epi16(1);

    let mut r0_acc0 = _mm256_setzero_ps();
    let mut r0_acc1 = _mm256_setzero_ps();
    let mut r1_acc0 = _mm256_setzero_ps();
    let mut r1_acc1 = _mm256_setzero_ps();

    let mut ib = 0;
    while ib < 48 {
        let act_q0 = x_qs[ib];
        let act_q1 = x_qs[ib + 1];
        let act_d0 = x_d[ib];
        let act_d1 = x_d[ib + 1];

        // Row 0
        let r0_b0 = r0_ptr.add(ib * 18);
        let r0_b1 = r0_ptr.add((ib + 1) * 18);
        let r0_d0 = _mm256_set1_ps(f16_to_f32_hw((r0_b0 as *const u16).read_unaligned()) * act_d0);
        let r0_d1 = _mm256_set1_ps(f16_to_f32_hw((r0_b1 as *const u16).read_unaligned()) * act_d1);
        let r0_q4_0 = _mm_loadu_si128(r0_b0.add(2) as *const __m128i);
        let r0_q4_1 = _mm_loadu_si128(r0_b1.add(2) as *const __m128i);
        let r0_q4b_0 = _mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(r0_q4_0, m4b))),
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(r0_q4_0, 4), m4b)),
            1,
        );
        let r0_q4b_1 = _mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(r0_q4_1, m4b))),
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(r0_q4_1, 4), m4b)),
            1,
        );
        let r0_p_0 = _mm256_madd_epi16(mul_add_epi8(r0_q4b_0, act_q0), mone);
        let r0_p_1 = _mm256_madd_epi16(mul_add_epi8(r0_q4b_1, act_q1), mone);
        r0_acc0 = _mm256_fmadd_ps(r0_d0, _mm256_cvtepi32_ps(r0_p_0), r0_acc0);
        r0_acc1 = _mm256_fmadd_ps(r0_d1, _mm256_cvtepi32_ps(r0_p_1), r0_acc1);

        // Row 1
        let r1_b0 = r1_ptr.add(ib * 18);
        let r1_b1 = r1_ptr.add((ib + 1) * 18);
        let r1_d0 = _mm256_set1_ps(f16_to_f32_hw((r1_b0 as *const u16).read_unaligned()) * act_d0);
        let r1_d1 = _mm256_set1_ps(f16_to_f32_hw((r1_b1 as *const u16).read_unaligned()) * act_d1);
        let r1_q4_0 = _mm_loadu_si128(r1_b0.add(2) as *const __m128i);
        let r1_q4_1 = _mm_loadu_si128(r1_b1.add(2) as *const __m128i);
        let r1_q4b_0 = _mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(r1_q4_0, m4b))),
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(r1_q4_0, 4), m4b)),
            1,
        );
        let r1_q4b_1 = _mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(r1_q4_1, m4b))),
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(r1_q4_1, 4), m4b)),
            1,
        );
        let r1_p_0 = _mm256_madd_epi16(mul_add_epi8(r1_q4b_0, act_q0), mone);
        let r1_p_1 = _mm256_madd_epi16(mul_add_epi8(r1_q4b_1, act_q1), mone);
        r1_acc0 = _mm256_fmadd_ps(r1_d0, _mm256_cvtepi32_ps(r1_p_0), r1_acc0);
        r1_acc1 = _mm256_fmadd_ps(r1_d1, _mm256_cvtepi32_ps(r1_p_1), r1_acc1);

        ib += 2;
    }

    let val0 = hsum_float_8(_mm256_add_ps(r0_acc0, r0_acc1));
    let val1 = hsum_float_8(_mm256_add_ps(r1_acc0, r1_acc1));
    (val0, val1)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_iq4_nl_48blocks_avx2(
    weight_ptr: *const u8,
    x_qs: &[core::arch::x86_64::__m256i; 48],
    x_d: &[f32; 48],
) -> f32 {
    use core::arch::x86_64::*;
    _mm_prefetch(weight_ptr.add(864) as *const i8, _MM_HINT_T0);

    let values128 = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let m4b = _mm_set1_epi8(0x0f);
    let mone = _mm256_set1_epi16(1);

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    let mut ib = 0;
    while ib < 48 {
        let b0_ptr = weight_ptr.add(ib * 18);
        let b1_ptr = weight_ptr.add((ib + 1) * 18);

        let d0 = _mm256_set1_ps(f16_to_f32_hw((b0_ptr as *const u16).read_unaligned()) * x_d[ib]);
        let d1 =
            _mm256_set1_ps(f16_to_f32_hw((b1_ptr as *const u16).read_unaligned()) * x_d[ib + 1]);

        let q4_0 = _mm_loadu_si128(b0_ptr.add(2) as *const __m128i);
        let q4_1 = _mm_loadu_si128(b1_ptr.add(2) as *const __m128i);

        let q4b_0 = _mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(q4_0, m4b))),
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4_0, 4), m4b)),
            1,
        );
        let q4b_1 = _mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(q4_1, m4b))),
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4_1, 4), m4b)),
            1,
        );

        let p_0 = _mm256_madd_epi16(mul_add_epi8(q4b_0, x_qs[ib]), mone);
        let p_1 = _mm256_madd_epi16(mul_add_epi8(q4b_1, x_qs[ib + 1]), mone);

        acc0 = _mm256_fmadd_ps(d0, _mm256_cvtepi32_ps(p_0), acc0);
        acc1 = _mm256_fmadd_ps(d1, _mm256_cvtepi32_ps(p_1), acc1);

        ib += 2;
    }

    hsum_float_8(_mm256_add_ps(acc0, acc1))
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_iq4_nl_96blocks_2rows_avx2(
    r0_ptr: *const u8,
    r1_ptr: *const u8,
    x_qs: &[core::arch::x86_64::__m256i; 96],
    x_d: &[f32; 96],
) -> (f32, f32) {
    use core::arch::x86_64::*;
    _mm_prefetch(r0_ptr.add(1728) as *const i8, _MM_HINT_T0);
    _mm_prefetch(r1_ptr.add(1728) as *const i8, _MM_HINT_T0);

    let values128 = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let m4b = _mm_set1_epi8(0x0f);
    let mone = _mm256_set1_epi16(1);

    let mut r0_acc0 = _mm256_setzero_ps();
    let mut r0_acc1 = _mm256_setzero_ps();
    let mut r1_acc0 = _mm256_setzero_ps();
    let mut r1_acc1 = _mm256_setzero_ps();

    let mut ib = 0;
    while ib < 96 {
        let act_q0 = x_qs[ib];
        let act_q1 = x_qs[ib + 1];
        let act_d0 = x_d[ib];
        let act_d1 = x_d[ib + 1];

        // Row 0
        let r0_b0 = r0_ptr.add(ib * 18);
        let r0_b1 = r0_ptr.add((ib + 1) * 18);
        let r0_d0 = _mm256_set1_ps(f16_to_f32_hw((r0_b0 as *const u16).read_unaligned()) * act_d0);
        let r0_d1 = _mm256_set1_ps(f16_to_f32_hw((r0_b1 as *const u16).read_unaligned()) * act_d1);
        let r0_q4_0 = _mm_loadu_si128(r0_b0.add(2) as *const __m128i);
        let r0_q4_1 = _mm_loadu_si128(r0_b1.add(2) as *const __m128i);
        let r0_q4b_0 = _mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(r0_q4_0, m4b))),
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(r0_q4_0, 4), m4b)),
            1,
        );
        let r0_q4b_1 = _mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(r0_q4_1, m4b))),
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(r0_q4_1, 4), m4b)),
            1,
        );
        let r0_p_0 = _mm256_madd_epi16(mul_add_epi8(r0_q4b_0, act_q0), mone);
        let r0_p_1 = _mm256_madd_epi16(mul_add_epi8(r0_q4b_1, act_q1), mone);
        r0_acc0 = _mm256_fmadd_ps(r0_d0, _mm256_cvtepi32_ps(r0_p_0), r0_acc0);
        r0_acc1 = _mm256_fmadd_ps(r0_d1, _mm256_cvtepi32_ps(r0_p_1), r0_acc1);

        // Row 1
        let r1_b0 = r1_ptr.add(ib * 18);
        let r1_b1 = r1_ptr.add((ib + 1) * 18);
        let r1_d0 = _mm256_set1_ps(f16_to_f32_hw((r1_b0 as *const u16).read_unaligned()) * act_d0);
        let r1_d1 = _mm256_set1_ps(f16_to_f32_hw((r1_b1 as *const u16).read_unaligned()) * act_d1);
        let r1_q4_0 = _mm_loadu_si128(r1_b0.add(2) as *const __m128i);
        let r1_q4_1 = _mm_loadu_si128(r1_b1.add(2) as *const __m128i);
        let r1_q4b_0 = _mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(r1_q4_0, m4b))),
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(r1_q4_0, 4), m4b)),
            1,
        );
        let r1_q4b_1 = _mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(r1_q4_1, m4b))),
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(r1_q4_1, 4), m4b)),
            1,
        );
        let r1_p_0 = _mm256_madd_epi16(mul_add_epi8(r1_q4b_0, act_q0), mone);
        let r1_p_1 = _mm256_madd_epi16(mul_add_epi8(r1_q4b_1, act_q1), mone);
        r1_acc0 = _mm256_fmadd_ps(r1_d0, _mm256_cvtepi32_ps(r1_p_0), r1_acc0);
        r1_acc1 = _mm256_fmadd_ps(r1_d1, _mm256_cvtepi32_ps(r1_p_1), r1_acc1);

        ib += 2;
    }

    let val0 = hsum_float_8(_mm256_add_ps(r0_acc0, r0_acc1));
    let val1 = hsum_float_8(_mm256_add_ps(r1_acc0, r1_acc1));
    (val0, val1)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_iq4_nl_96blocks_avx2(
    weight_ptr: *const u8,
    x_qs: &[core::arch::x86_64::__m256i; 96],
    x_d: &[f32; 96],
) -> f32 {
    use core::arch::x86_64::*;
    _mm_prefetch(weight_ptr.add(1728) as *const i8, _MM_HINT_T0);

    let values128 = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let m4b = _mm_set1_epi8(0x0f);
    let mone = _mm256_set1_epi16(1);

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    let mut ib = 0;
    while ib < 96 {
        let b0_ptr = weight_ptr.add(ib * 18);
        let b1_ptr = weight_ptr.add((ib + 1) * 18);

        let d0 = _mm256_set1_ps(f16_to_f32_hw((b0_ptr as *const u16).read_unaligned()) * x_d[ib]);
        let d1 =
            _mm256_set1_ps(f16_to_f32_hw((b1_ptr as *const u16).read_unaligned()) * x_d[ib + 1]);

        let q4_0 = _mm_loadu_si128(b0_ptr.add(2) as *const __m128i);
        let q4_1 = _mm_loadu_si128(b1_ptr.add(2) as *const __m128i);

        let q4b_0 = _mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(q4_0, m4b))),
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4_0, 4), m4b)),
            1,
        );
        let q4b_1 = _mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(q4_1, m4b))),
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4_1, 4), m4b)),
            1,
        );

        let p_0 = _mm256_madd_epi16(mul_add_epi8(q4b_0, x_qs[ib]), mone);
        let p_1 = _mm256_madd_epi16(mul_add_epi8(q4b_1, x_qs[ib + 1]), mone);

        acc0 = _mm256_fmadd_ps(d0, _mm256_cvtepi32_ps(p_0), acc0);
        acc1 = _mm256_fmadd_ps(d1, _mm256_cvtepi32_ps(p_1), acc1);

        ib += 2;
    }

    hsum_float_8(_mm256_add_ps(acc0, acc1))
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_iq4_nl_28blocks_2rows_avx2(
    r0_ptr: *const u8,
    r1_ptr: *const u8,
    x_qs: &[core::arch::x86_64::__m256i; 28],
    x_d: &[f32; 28],
) -> (f32, f32) {
    use core::arch::x86_64::*;
    _mm_prefetch(r0_ptr.add(504) as *const i8, _MM_HINT_T0);
    _mm_prefetch(r1_ptr.add(504) as *const i8, _MM_HINT_T0);

    let values128 = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let m4b = _mm_set1_epi8(0x0f);
    let mone = _mm256_set1_epi16(1);

    let mut r0_acc0 = _mm256_setzero_ps();
    let mut r0_acc1 = _mm256_setzero_ps();
    let mut r1_acc0 = _mm256_setzero_ps();
    let mut r1_acc1 = _mm256_setzero_ps();

    let mut ib = 0;
    while ib < 28 {
        let act_q0 = x_qs[ib];
        let act_q1 = x_qs[ib + 1];
        let act_d0 = x_d[ib];
        let act_d1 = x_d[ib + 1];

        // Row 0
        let r0_b0 = r0_ptr.add(ib * 18);
        let r0_b1 = r0_ptr.add((ib + 1) * 18);
        let r0_d0 = _mm256_set1_ps(f16_to_f32_hw((r0_b0 as *const u16).read_unaligned()) * act_d0);
        let r0_d1 = _mm256_set1_ps(f16_to_f32_hw((r0_b1 as *const u16).read_unaligned()) * act_d1);
        let r0_q4_0 = _mm_loadu_si128(r0_b0.add(2) as *const __m128i);
        let r0_q4_1 = _mm_loadu_si128(r0_b1.add(2) as *const __m128i);
        let r0_q4b_0 = _mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(r0_q4_0, m4b))),
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(r0_q4_0, 4), m4b)),
            1,
        );
        let r0_q4b_1 = _mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(r0_q4_1, m4b))),
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(r0_q4_1, 4), m4b)),
            1,
        );
        let r0_p_0 = _mm256_madd_epi16(mul_add_epi8(r0_q4b_0, act_q0), mone);
        let r0_p_1 = _mm256_madd_epi16(mul_add_epi8(r0_q4b_1, act_q1), mone);
        r0_acc0 = _mm256_fmadd_ps(r0_d0, _mm256_cvtepi32_ps(r0_p_0), r0_acc0);
        r0_acc1 = _mm256_fmadd_ps(r0_d1, _mm256_cvtepi32_ps(r0_p_1), r0_acc1);

        // Row 1
        let r1_b0 = r1_ptr.add(ib * 18);
        let r1_b1 = r1_ptr.add((ib + 1) * 18);
        let r1_d0 = _mm256_set1_ps(f16_to_f32_hw((r1_b0 as *const u16).read_unaligned()) * act_d0);
        let r1_d1 = _mm256_set1_ps(f16_to_f32_hw((r1_b1 as *const u16).read_unaligned()) * act_d1);
        let r1_q4_0 = _mm_loadu_si128(r1_b0.add(2) as *const __m128i);
        let r1_q4_1 = _mm_loadu_si128(r1_b1.add(2) as *const __m128i);
        let r1_q4b_0 = _mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(r1_q4_0, m4b))),
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(r1_q4_0, 4), m4b)),
            1,
        );
        let r1_q4b_1 = _mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(r1_q4_1, m4b))),
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(r1_q4_1, 4), m4b)),
            1,
        );
        let r1_p_0 = _mm256_madd_epi16(mul_add_epi8(r1_q4b_0, act_q0), mone);
        let r1_p_1 = _mm256_madd_epi16(mul_add_epi8(r1_q4b_1, act_q1), mone);
        r1_acc0 = _mm256_fmadd_ps(r1_d0, _mm256_cvtepi32_ps(r1_p_0), r1_acc0);
        r1_acc1 = _mm256_fmadd_ps(r1_d1, _mm256_cvtepi32_ps(r1_p_1), r1_acc1);

        ib += 2;
    }

    let val0 = hsum_float_8(_mm256_add_ps(r0_acc0, r0_acc1));
    let val1 = hsum_float_8(_mm256_add_ps(r1_acc0, r1_acc1));
    (val0, val1)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_iq4_nl_28blocks_avx2(
    weight_ptr: *const u8,
    x_qs: &[core::arch::x86_64::__m256i; 28],
    x_d: &[f32; 28],
) -> f32 {
    use core::arch::x86_64::*;
    _mm_prefetch(weight_ptr.add(504) as *const i8, _MM_HINT_T0);

    let values128 = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let m4b = _mm_set1_epi8(0x0f);
    let mone = _mm256_set1_epi16(1);

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    let mut ib = 0;
    while ib < 28 {
        let b0_ptr = weight_ptr.add(ib * 18);
        let b1_ptr = weight_ptr.add((ib + 1) * 18);

        let d0 = _mm256_set1_ps(f16_to_f32_hw((b0_ptr as *const u16).read_unaligned()) * x_d[ib]);
        let d1 =
            _mm256_set1_ps(f16_to_f32_hw((b1_ptr as *const u16).read_unaligned()) * x_d[ib + 1]);

        let q4_0 = _mm_loadu_si128(b0_ptr.add(2) as *const __m128i);
        let q4_1 = _mm_loadu_si128(b1_ptr.add(2) as *const __m128i);

        let q4b_0 = _mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(q4_0, m4b))),
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4_0, 4), m4b)),
            1,
        );
        let q4b_1 = _mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_shuffle_epi8(values128, _mm_and_si128(q4_1, m4b))),
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4_1, 4), m4b)),
            1,
        );

        let p_0 = _mm256_madd_epi16(mul_add_epi8(q4b_0, x_qs[ib]), mone);
        let p_1 = _mm256_madd_epi16(mul_add_epi8(q4b_1, x_qs[ib + 1]), mone);

        acc0 = _mm256_fmadd_ps(d0, _mm256_cvtepi32_ps(p_0), acc0);
        acc1 = _mm256_fmadd_ps(d1, _mm256_cvtepi32_ps(p_1), acc1);

        ib += 2;
    }

    hsum_float_8(_mm256_add_ps(acc0, acc1))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn gate_up_iq4_nl_silu_avx2(
    w1_ptr: *const u8,
    w3_ptr: *const u8,
    x_blocks: &[BlockQ8_0],
    nb: usize,
) -> f32 {
    use core::arch::x86_64::*;

    let values128 = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let m4b = _mm_set1_epi8(0x0f);
    let mone = _mm256_set1_epi16(1);

    let mut acc_w1_0 = _mm256_setzero_ps();
    let mut acc_w1_1 = _mm256_setzero_ps();
    let mut acc_w3_0 = _mm256_setzero_ps();
    let mut acc_w3_1 = _mm256_setzero_ps();

    let mut ib = 0;
    while ib + 1 < nb {
        if (ib & 3) == 0 && ib + 8 < nb {
            _mm_prefetch(w1_ptr.add((ib + 8) * 18) as *const i8, _MM_HINT_T0);
            _mm_prefetch(w3_ptr.add((ib + 8) * 18) as *const i8, _MM_HINT_T0);
        }
        let q8b_1 = _mm256_loadu_si256(x_blocks[ib].qs.as_ptr() as *const __m256i);
        let q8b_2 = _mm256_loadu_si256(x_blocks[ib + 1].qs.as_ptr() as *const __m256i);
        let x_d0 = x_blocks[ib].d;
        let x_d1 = x_blocks[ib + 1].d;

        // W1 blocks
        let w1_b0 = w1_ptr.add(ib * 18);
        let w1_b1 = w1_ptr.add((ib + 1) * 18);
        let w1_q4_1 = _mm_loadu_si128(w1_b0.add(2) as *const __m128i);
        let w1_q4_2 = _mm_loadu_si128(w1_b1.add(2) as *const __m128i);

        let w1_q4b_1_low = _mm_shuffle_epi8(values128, _mm_and_si128(w1_q4_1, m4b));
        let w1_q4b_1_high =
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(w1_q4_1, 4), m4b));
        let w1_q4b_1 =
            _mm256_insertf128_si256(_mm256_castsi128_si256(w1_q4b_1_low), w1_q4b_1_high, 1);

        let w1_q4b_2_low = _mm_shuffle_epi8(values128, _mm_and_si128(w1_q4_2, m4b));
        let w1_q4b_2_high =
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(w1_q4_2, 4), m4b));
        let w1_q4b_2 =
            _mm256_insertf128_si256(_mm256_castsi128_si256(w1_q4b_2_low), w1_q4b_2_high, 1);

        let w1_p16_1 = mul_add_epi8(w1_q4b_1, q8b_1);
        let w1_p16_2 = mul_add_epi8(w1_q4b_2, q8b_2);
        let w1_p_1 = _mm256_madd_epi16(w1_p16_1, mone);
        let w1_p_2 = _mm256_madd_epi16(w1_p16_2, mone);

        let w1_d0 = f16_to_f32_hw((w1_b0 as *const u16).read_unaligned()) * x_d0;
        let w1_d1 = f16_to_f32_hw((w1_b1 as *const u16).read_unaligned()) * x_d1;
        acc_w1_0 = _mm256_fmadd_ps(_mm256_set1_ps(w1_d0), _mm256_cvtepi32_ps(w1_p_1), acc_w1_0);
        acc_w1_1 = _mm256_fmadd_ps(_mm256_set1_ps(w1_d1), _mm256_cvtepi32_ps(w1_p_2), acc_w1_1);

        // W3 blocks
        let w3_b0 = w3_ptr.add(ib * 18);
        let w3_b1 = w3_ptr.add((ib + 1) * 18);
        let w3_q4_1 = _mm_loadu_si128(w3_b0.add(2) as *const __m128i);
        let w3_q4_2 = _mm_loadu_si128(w3_b1.add(2) as *const __m128i);

        let w3_q4b_1_low = _mm_shuffle_epi8(values128, _mm_and_si128(w3_q4_1, m4b));
        let w3_q4b_1_high =
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(w3_q4_1, 4), m4b));
        let w3_q4b_1 =
            _mm256_insertf128_si256(_mm256_castsi128_si256(w3_q4b_1_low), w3_q4b_1_high, 1);

        let w3_q4b_2_low = _mm_shuffle_epi8(values128, _mm_and_si128(w3_q4_2, m4b));
        let w3_q4b_2_high =
            _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(w3_q4_2, 4), m4b));
        let w3_q4b_2 =
            _mm256_insertf128_si256(_mm256_castsi128_si256(w3_q4b_2_low), w3_q4b_2_high, 1);

        let w3_p16_1 = mul_add_epi8(w3_q4b_1, q8b_1);
        let w3_p16_2 = mul_add_epi8(w3_q4b_2, q8b_2);
        let w3_p_1 = _mm256_madd_epi16(w3_p16_1, mone);
        let w3_p_2 = _mm256_madd_epi16(w3_p16_2, mone);

        let w3_d0 = f16_to_f32_hw((w3_b0 as *const u16).read_unaligned()) * x_d0;
        let w3_d1 = f16_to_f32_hw((w3_b1 as *const u16).read_unaligned()) * x_d1;
        acc_w3_0 = _mm256_fmadd_ps(_mm256_set1_ps(w3_d0), _mm256_cvtepi32_ps(w3_p_1), acc_w3_0);
        acc_w3_1 = _mm256_fmadd_ps(_mm256_set1_ps(w3_d1), _mm256_cvtepi32_ps(w3_p_2), acc_w3_1);

        ib += 2;
    }

    let w1_val = hsum_float_8(_mm256_add_ps(acc_w1_0, acc_w1_1));
    let w3_val = hsum_float_8(_mm256_add_ps(acc_w3_0, acc_w3_1));
    fast_silu(w1_val) * w3_val
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_iq3_xxs_q8_k_avx2(weight_bytes: &[u8], x_blocks: &[BlockQ8_K]) -> f32 {
    use core::arch::x86_64::*;
    let nb = x_blocks.len();
    assert_eq!(weight_bytes.len(), nb * 98);

    let mut accumf = _mm256_setzero_ps();

    for i in 0..nb {
        let b_ptr = weight_bytes.as_ptr().add(i * 98);

        let d = f16_to_f32_hw((b_ptr as *const u16).read_unaligned()) * x_blocks[i].d;

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

            let q2_1 = _mm256_set_epi32(
                IQ3XXS_GRID[*q3_ptr.add(7) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(6) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(5) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(4) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(3) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(2) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(1) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(0) as usize] as i32,
            );

            let q2_2 = _mm256_set_epi32(
                IQ3XXS_GRID[*q3_ptr.add(15) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(14) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(13) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(12) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(11) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(10) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(9) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(8) as usize] as i32,
            );
            q3_ptr = q3_ptr.add(16);

            let aux32_0 = (gas_ptr as *const u32).read_unaligned();
            let aux32_1 = (gas_ptr.add(4) as *const u32).read_unaligned();
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_iq3_xxs_q8_k_2rows_avx2_ptr(
    b0_base: *const u8,
    b1_base: *const u8,
    x_blocks: *const BlockQ8_K,
    nb: usize,
) -> (f32, f32) {
    use core::arch::x86_64::*;
    let mut accumf0 = _mm256_setzero_ps();
    let mut accumf1 = _mm256_setzero_ps();

    for i in 0..nb {
        let b0_ptr = b0_base.add(i * 98);
        let b1_ptr = b1_base.add(i * 98);
        if i + 2 < nb {
            _mm_prefetch(b0_ptr.add(196) as *const i8, _MM_HINT_T0);
            _mm_prefetch(b1_ptr.add(196) as *const i8, _MM_HINT_T0);
        }

        let block = &*x_blocks.add(i);
        let d0 = f16_to_f32_hw((b0_ptr as *const u16).read_unaligned()) * block.d;
        let d1 = f16_to_f32_hw((b1_ptr as *const u16).read_unaligned()) * block.d;

        let mut q3_0_ptr = b0_ptr.add(2);
        let mut gas_0_ptr = b0_ptr.add(66);
        let mut q3_1_ptr = b1_ptr.add(2);
        let mut gas_1_ptr = b1_ptr.add(66);
        let mut q8_ptr = block.qs.as_ptr();

        let mut sumi1_0 = _mm256_setzero_si256();
        let mut sumi2_0 = _mm256_setzero_si256();
        let mut sumi1_1 = _mm256_setzero_si256();
        let mut sumi2_1 = _mm256_setzero_si256();

        for _ in 0..4 {
            let q8_1 = _mm256_loadu_si256(q8_ptr as *const __m256i);
            q8_ptr = q8_ptr.add(32);
            let q8_2 = _mm256_loadu_si256(q8_ptr as *const __m256i);
            q8_ptr = q8_ptr.add(32);

            let q2_1_0 = _mm256_set_epi32(
                IQ3XXS_GRID[*q3_0_ptr.add(7) as usize] as i32,
                IQ3XXS_GRID[*q3_0_ptr.add(6) as usize] as i32,
                IQ3XXS_GRID[*q3_0_ptr.add(5) as usize] as i32,
                IQ3XXS_GRID[*q3_0_ptr.add(4) as usize] as i32,
                IQ3XXS_GRID[*q3_0_ptr.add(3) as usize] as i32,
                IQ3XXS_GRID[*q3_0_ptr.add(2) as usize] as i32,
                IQ3XXS_GRID[*q3_0_ptr.add(1) as usize] as i32,
                IQ3XXS_GRID[*q3_0_ptr.add(0) as usize] as i32,
            );
            let q2_2_0 = _mm256_set_epi32(
                IQ3XXS_GRID[*q3_0_ptr.add(15) as usize] as i32,
                IQ3XXS_GRID[*q3_0_ptr.add(14) as usize] as i32,
                IQ3XXS_GRID[*q3_0_ptr.add(13) as usize] as i32,
                IQ3XXS_GRID[*q3_0_ptr.add(12) as usize] as i32,
                IQ3XXS_GRID[*q3_0_ptr.add(11) as usize] as i32,
                IQ3XXS_GRID[*q3_0_ptr.add(10) as usize] as i32,
                IQ3XXS_GRID[*q3_0_ptr.add(9) as usize] as i32,
                IQ3XXS_GRID[*q3_0_ptr.add(8) as usize] as i32,
            );
            q3_0_ptr = q3_0_ptr.add(16);

            let aux32_0_0 = (gas_0_ptr as *const u32).read_unaligned();
            let aux32_1_0 = (gas_0_ptr.add(4) as *const u32).read_unaligned();
            gas_0_ptr = gas_0_ptr.add(8);

            let s2_1_0 = _mm256_set_epi64x(
                SIGNS64_IQ2XS[((aux32_0_0 >> 21) & 127) as usize] as i64,
                SIGNS64_IQ2XS[((aux32_0_0 >> 14) & 127) as usize] as i64,
                SIGNS64_IQ2XS[((aux32_0_0 >> 7) & 127) as usize] as i64,
                SIGNS64_IQ2XS[((aux32_0_0 >> 0) & 127) as usize] as i64,
            );
            let s2_2_0 = _mm256_set_epi64x(
                SIGNS64_IQ2XS[((aux32_1_0 >> 21) & 127) as usize] as i64,
                SIGNS64_IQ2XS[((aux32_1_0 >> 14) & 127) as usize] as i64,
                SIGNS64_IQ2XS[((aux32_1_0 >> 7) & 127) as usize] as i64,
                SIGNS64_IQ2XS[((aux32_1_0 >> 0) & 127) as usize] as i64,
            );

            let q8s_1_0 = _mm256_sign_epi8(q8_1, s2_1_0);
            let q8s_2_0 = _mm256_sign_epi8(q8_2, s2_2_0);
            let dot1_0 = _mm256_maddubs_epi16(q2_1_0, q8s_1_0);
            let dot2_0 = _mm256_maddubs_epi16(q2_2_0, q8s_2_0);
            let ls1_0 = (aux32_0_0 >> 28) as i16;
            let ls2_0 = (aux32_1_0 >> 28) as i16;
            let p1_0 = _mm256_madd_epi16(dot1_0, _mm256_set1_epi16(2 * ls1_0 + 1));
            let p2_0 = _mm256_madd_epi16(dot2_0, _mm256_set1_epi16(2 * ls2_0 + 1));
            sumi1_0 = _mm256_add_epi32(sumi1_0, p1_0);
            sumi2_0 = _mm256_add_epi32(sumi2_0, p2_0);

            let q2_1_1 = _mm256_set_epi32(
                IQ3XXS_GRID[*q3_1_ptr.add(7) as usize] as i32,
                IQ3XXS_GRID[*q3_1_ptr.add(6) as usize] as i32,
                IQ3XXS_GRID[*q3_1_ptr.add(5) as usize] as i32,
                IQ3XXS_GRID[*q3_1_ptr.add(4) as usize] as i32,
                IQ3XXS_GRID[*q3_1_ptr.add(3) as usize] as i32,
                IQ3XXS_GRID[*q3_1_ptr.add(2) as usize] as i32,
                IQ3XXS_GRID[*q3_1_ptr.add(1) as usize] as i32,
                IQ3XXS_GRID[*q3_1_ptr.add(0) as usize] as i32,
            );
            let q2_2_1 = _mm256_set_epi32(
                IQ3XXS_GRID[*q3_1_ptr.add(15) as usize] as i32,
                IQ3XXS_GRID[*q3_1_ptr.add(14) as usize] as i32,
                IQ3XXS_GRID[*q3_1_ptr.add(13) as usize] as i32,
                IQ3XXS_GRID[*q3_1_ptr.add(12) as usize] as i32,
                IQ3XXS_GRID[*q3_1_ptr.add(11) as usize] as i32,
                IQ3XXS_GRID[*q3_1_ptr.add(10) as usize] as i32,
                IQ3XXS_GRID[*q3_1_ptr.add(9) as usize] as i32,
                IQ3XXS_GRID[*q3_1_ptr.add(8) as usize] as i32,
            );
            q3_1_ptr = q3_1_ptr.add(16);

            let aux32_0_1 = (gas_1_ptr as *const u32).read_unaligned();
            let aux32_1_1 = (gas_1_ptr.add(4) as *const u32).read_unaligned();
            gas_1_ptr = gas_1_ptr.add(8);

            let s2_1_1 = _mm256_set_epi64x(
                SIGNS64_IQ2XS[((aux32_0_1 >> 21) & 127) as usize] as i64,
                SIGNS64_IQ2XS[((aux32_0_1 >> 14) & 127) as usize] as i64,
                SIGNS64_IQ2XS[((aux32_0_1 >> 7) & 127) as usize] as i64,
                SIGNS64_IQ2XS[((aux32_0_1 >> 0) & 127) as usize] as i64,
            );
            let s2_2_1 = _mm256_set_epi64x(
                SIGNS64_IQ2XS[((aux32_1_1 >> 21) & 127) as usize] as i64,
                SIGNS64_IQ2XS[((aux32_1_1 >> 14) & 127) as usize] as i64,
                SIGNS64_IQ2XS[((aux32_1_1 >> 7) & 127) as usize] as i64,
                SIGNS64_IQ2XS[((aux32_1_1 >> 0) & 127) as usize] as i64,
            );

            let q8s_1_1 = _mm256_sign_epi8(q8_1, s2_1_1);
            let q8s_2_1 = _mm256_sign_epi8(q8_2, s2_2_1);
            let dot1_1 = _mm256_maddubs_epi16(q2_1_1, q8s_1_1);
            let dot2_1 = _mm256_maddubs_epi16(q2_2_1, q8s_2_1);
            let ls1_1 = (aux32_0_1 >> 28) as i16;
            let ls2_1 = (aux32_1_1 >> 28) as i16;
            let p1_1 = _mm256_madd_epi16(dot1_1, _mm256_set1_epi16(2 * ls1_1 + 1));
            let p2_1 = _mm256_madd_epi16(dot2_1, _mm256_set1_epi16(2 * ls2_1 + 1));
            sumi1_1 = _mm256_add_epi32(sumi1_1, p1_1);
            sumi2_1 = _mm256_add_epi32(sumi2_1, p2_1);
        }

        accumf0 = _mm256_fmadd_ps(
            _mm256_set1_ps(d0),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1_0, sumi2_0)),
            accumf0,
        );
        accumf1 = _mm256_fmadd_ps(
            _mm256_set1_ps(d1),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1_1, sumi2_1)),
            accumf1,
        );
    }

    (
        0.25f32 * hsum_float_8(accumf0),
        0.25f32 * hsum_float_8(accumf1),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_iq3_xxs_q8_k_2tokens_avx2_ptr(
    w_base: *const u8,
    x0_blocks: *const BlockQ8_K,
    x1_blocks: *const BlockQ8_K,
    nb: usize,
) -> (f32, f32) {
    use core::arch::x86_64::*;
    let mut accumf0 = _mm256_setzero_ps();
    let mut accumf1 = _mm256_setzero_ps();

    for i in 0..nb {
        let b_ptr = w_base.add(i * 98);
        if i + 2 < nb {
            _mm_prefetch(b_ptr.add(196) as *const i8, _MM_HINT_T0);
        }
        let blk0 = &*x0_blocks.add(i);
        let blk1 = &*x1_blocks.add(i);

        let w_scale = f16_to_f32_hw((b_ptr as *const u16).read_unaligned());
        let d0 = w_scale * blk0.d;
        let d1 = w_scale * blk1.d;

        let mut q3_ptr = b_ptr.add(2);
        let mut gas_ptr = b_ptr.add(66);
        let mut q8_0_ptr = blk0.qs.as_ptr();
        let mut q8_1_ptr = blk1.qs.as_ptr();

        let mut sumi1_0 = _mm256_setzero_si256();
        let mut sumi2_0 = _mm256_setzero_si256();
        let mut sumi1_1 = _mm256_setzero_si256();
        let mut sumi2_1 = _mm256_setzero_si256();

        for _ in 0..4 {
            let q8_1_0 = _mm256_loadu_si256(q8_0_ptr as *const __m256i);
            q8_0_ptr = q8_0_ptr.add(32);
            let q8_2_0 = _mm256_loadu_si256(q8_0_ptr as *const __m256i);
            q8_0_ptr = q8_0_ptr.add(32);

            let q8_1_1 = _mm256_loadu_si256(q8_1_ptr as *const __m256i);
            q8_1_ptr = q8_1_ptr.add(32);
            let q8_2_1 = _mm256_loadu_si256(q8_1_ptr as *const __m256i);
            q8_1_ptr = q8_1_ptr.add(32);

            let q2_1 = _mm256_set_epi32(
                IQ3XXS_GRID[*q3_ptr.add(7) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(6) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(5) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(4) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(3) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(2) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(1) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(0) as usize] as i32,
            );
            let q2_2 = _mm256_set_epi32(
                IQ3XXS_GRID[*q3_ptr.add(15) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(14) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(13) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(12) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(11) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(10) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(9) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(8) as usize] as i32,
            );
            q3_ptr = q3_ptr.add(16);

            let aux32_0 = (gas_ptr as *const u32).read_unaligned();
            let aux32_1 = (gas_ptr.add(4) as *const u32).read_unaligned();
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

            let ls1 = (aux32_0 >> 28) as i16;
            let ls2 = (aux32_1 >> 28) as i16;
            let scale1 = _mm256_set1_epi16(2 * ls1 + 1);
            let scale2 = _mm256_set1_epi16(2 * ls2 + 1);

            let q8s_1_0 = _mm256_sign_epi8(q8_1_0, s2_1);
            let q8s_2_0 = _mm256_sign_epi8(q8_2_0, s2_2);
            let dot1_0 = _mm256_maddubs_epi16(q2_1, q8s_1_0);
            let dot2_0 = _mm256_maddubs_epi16(q2_2, q8s_2_0);
            sumi1_0 = _mm256_add_epi32(sumi1_0, _mm256_madd_epi16(dot1_0, scale1));
            sumi2_0 = _mm256_add_epi32(sumi2_0, _mm256_madd_epi16(dot2_0, scale2));

            let q8s_1_1 = _mm256_sign_epi8(q8_1_1, s2_1);
            let q8s_2_1 = _mm256_sign_epi8(q8_2_1, s2_2);
            let dot1_1 = _mm256_maddubs_epi16(q2_1, q8s_1_1);
            let dot2_1 = _mm256_maddubs_epi16(q2_2, q8s_2_1);
            sumi1_1 = _mm256_add_epi32(sumi1_1, _mm256_madd_epi16(dot1_1, scale1));
            sumi2_1 = _mm256_add_epi32(sumi2_1, _mm256_madd_epi16(dot2_1, scale2));
        }

        accumf0 = _mm256_fmadd_ps(
            _mm256_set1_ps(d0),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1_0, sumi2_0)),
            accumf0,
        );
        accumf1 = _mm256_fmadd_ps(
            _mm256_set1_ps(d1),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1_1, sumi2_1)),
            accumf1,
        );
    }

    (
        0.25f32 * hsum_float_8(accumf0),
        0.25f32 * hsum_float_8(accumf1),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_iq3_xxs_q8_k_3tokens_avx2_ptr(
    w_base: *const u8,
    x0_blocks: *const BlockQ8_K,
    x1_blocks: *const BlockQ8_K,
    x2_blocks: *const BlockQ8_K,
    nb: usize,
) -> (f32, f32, f32) {
    use core::arch::x86_64::*;
    let mut accumf0 = _mm256_setzero_ps();
    let mut accumf1 = _mm256_setzero_ps();
    let mut accumf2 = _mm256_setzero_ps();

    for i in 0..nb {
        let b_ptr = w_base.add(i * 98);
        if i + 2 < nb {
            _mm_prefetch(b_ptr.add(196) as *const i8, _MM_HINT_T0);
        }
        let blk0 = &*x0_blocks.add(i);
        let blk1 = &*x1_blocks.add(i);
        let blk2 = &*x2_blocks.add(i);

        let w_scale = f16_to_f32_hw((b_ptr as *const u16).read_unaligned());
        let d0 = w_scale * blk0.d;
        let d1 = w_scale * blk1.d;
        let d2 = w_scale * blk2.d;

        let mut q3_ptr = b_ptr.add(2);
        let mut gas_ptr = b_ptr.add(66);
        let mut q8_0_ptr = blk0.qs.as_ptr();
        let mut q8_1_ptr = blk1.qs.as_ptr();
        let mut q8_2_ptr = blk2.qs.as_ptr();

        let mut sumi1_0 = _mm256_setzero_si256();
        let mut sumi2_0 = _mm256_setzero_si256();
        let mut sumi1_1 = _mm256_setzero_si256();
        let mut sumi2_1 = _mm256_setzero_si256();
        let mut sumi1_2 = _mm256_setzero_si256();
        let mut sumi2_2 = _mm256_setzero_si256();

        for _ in 0..4 {
            let q8_1_0 = _mm256_loadu_si256(q8_0_ptr as *const __m256i);
            q8_0_ptr = q8_0_ptr.add(32);
            let q8_2_0 = _mm256_loadu_si256(q8_0_ptr as *const __m256i);
            q8_0_ptr = q8_0_ptr.add(32);

            let q8_1_1 = _mm256_loadu_si256(q8_1_ptr as *const __m256i);
            q8_1_ptr = q8_1_ptr.add(32);
            let q8_2_1 = _mm256_loadu_si256(q8_1_ptr as *const __m256i);
            q8_1_ptr = q8_1_ptr.add(32);

            let q8_1_2 = _mm256_loadu_si256(q8_2_ptr as *const __m256i);
            q8_2_ptr = q8_2_ptr.add(32);
            let q8_2_2 = _mm256_loadu_si256(q8_2_ptr as *const __m256i);
            q8_2_ptr = q8_2_ptr.add(32);

            let q2_1 = _mm256_set_epi32(
                IQ3XXS_GRID[*q3_ptr.add(7) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(6) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(5) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(4) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(3) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(2) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(1) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(0) as usize] as i32,
            );
            let q2_2 = _mm256_set_epi32(
                IQ3XXS_GRID[*q3_ptr.add(15) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(14) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(13) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(12) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(11) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(10) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(9) as usize] as i32,
                IQ3XXS_GRID[*q3_ptr.add(8) as usize] as i32,
            );
            q3_ptr = q3_ptr.add(16);

            let aux32_0 = (gas_ptr as *const u32).read_unaligned();
            let aux32_1 = (gas_ptr.add(4) as *const u32).read_unaligned();
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

            let ls1 = (aux32_0 >> 28) as i16;
            let ls2 = (aux32_1 >> 28) as i16;
            let scale1 = _mm256_set1_epi16(2 * ls1 + 1);
            let scale2 = _mm256_set1_epi16(2 * ls2 + 1);

            let q8s_1_0 = _mm256_sign_epi8(q8_1_0, s2_1);
            let q8s_2_0 = _mm256_sign_epi8(q8_2_0, s2_2);
            let dot1_0 = _mm256_maddubs_epi16(q2_1, q8s_1_0);
            let dot2_0 = _mm256_maddubs_epi16(q2_2, q8s_2_0);
            sumi1_0 = _mm256_add_epi32(sumi1_0, _mm256_madd_epi16(dot1_0, scale1));
            sumi2_0 = _mm256_add_epi32(sumi2_0, _mm256_madd_epi16(dot2_0, scale2));

            let q8s_1_1 = _mm256_sign_epi8(q8_1_1, s2_1);
            let q8s_2_1 = _mm256_sign_epi8(q8_2_1, s2_2);
            let dot1_1 = _mm256_maddubs_epi16(q2_1, q8s_1_1);
            let dot2_1 = _mm256_maddubs_epi16(q2_2, q8s_2_1);
            sumi1_1 = _mm256_add_epi32(sumi1_1, _mm256_madd_epi16(dot1_1, scale1));
            sumi2_1 = _mm256_add_epi32(sumi2_1, _mm256_madd_epi16(dot2_1, scale2));

            let q8s_1_2 = _mm256_sign_epi8(q8_1_2, s2_1);
            let q8s_2_2 = _mm256_sign_epi8(q8_2_2, s2_2);
            let dot1_2 = _mm256_maddubs_epi16(q2_1, q8s_1_2);
            let dot2_2 = _mm256_maddubs_epi16(q2_2, q8s_2_2);
            sumi1_2 = _mm256_add_epi32(sumi1_2, _mm256_madd_epi16(dot1_2, scale1));
            sumi2_2 = _mm256_add_epi32(sumi2_2, _mm256_madd_epi16(dot2_2, scale2));
        }

        accumf0 = _mm256_fmadd_ps(
            _mm256_set1_ps(d0),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1_0, sumi2_0)),
            accumf0,
        );
        accumf1 = _mm256_fmadd_ps(
            _mm256_set1_ps(d1),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1_1, sumi2_1)),
            accumf1,
        );
        accumf2 = _mm256_fmadd_ps(
            _mm256_set1_ps(d2),
            _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1_2, sumi2_2)),
            accumf2,
        );
    }

    (
        0.25f32 * hsum_float_8(accumf0),
        0.25f32 * hsum_float_8(accumf1),
        0.25f32 * hsum_float_8(accumf2),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_iq3_xxs_q8_k_2rows_avx2(
    w0_bytes: &[u8],
    w1_bytes: &[u8],
    x_blocks: &[BlockQ8_K],
) -> (f32, f32) {
    let nb = x_blocks.len();
    assert_eq!(w0_bytes.len(), nb * 98);
    assert_eq!(w1_bytes.len(), nb * 98);
    vec_dot_iq3_xxs_q8_k_2rows_avx2_ptr(w0_bytes.as_ptr(), w1_bytes.as_ptr(), x_blocks.as_ptr(), nb)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn gemv_iq4_nl_prequant(
    out: &mut [f32],
    weight_bytes: &[u8],
    x_blocks: &[BlockQ8_0],
    n_rows: usize,
    n_cols: usize,
) {
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

        if r + 7 < n_rows {
            _mm_prefetch(
                weight_bytes.as_ptr().add((r + 4) * row_bytes) as *const i8,
                _MM_HINT_T0,
            );
            _mm_prefetch(
                weight_bytes.as_ptr().add((r + 5) * row_bytes) as *const i8,
                _MM_HINT_T0,
            );
            _mm_prefetch(
                weight_bytes.as_ptr().add((r + 6) * row_bytes) as *const i8,
                _MM_HINT_T0,
            );
            _mm_prefetch(
                weight_bytes.as_ptr().add((r + 7) * row_bytes) as *const i8,
                _MM_HINT_T0,
            );
        }

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for ib in 0..nb {
            if (ib & 3) == 0 && ib + 8 < nb {
                _mm_prefetch(r0_ptr.add((ib + 8) * 18) as *const i8, _MM_HINT_T0);
                _mm_prefetch(r1_ptr.add((ib + 8) * 18) as *const i8, _MM_HINT_T0);
                _mm_prefetch(r2_ptr.add((ib + 8) * 18) as *const i8, _MM_HINT_T0);
                _mm_prefetch(r3_ptr.add((ib + 8) * 18) as *const i8, _MM_HINT_T0);
            }
            let q8b = _mm256_loadu_si256(x_blocks[ib].qs.as_ptr() as *const __m256i);
            let x_d = x_blocks[ib].d;

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
            let d0 = f16_to_f32_hw((b0 as *const u16).read_unaligned()) * x_d;
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
            let d1 = f16_to_f32_hw((b1 as *const u16).read_unaligned()) * x_d;
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
            let d2 = f16_to_f32_hw((b2 as *const u16).read_unaligned()) * x_d;
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
            let d3 = f16_to_f32_hw((b3 as *const u16).read_unaligned()) * x_d;
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn iq3_xxs_subblock_step_avx2(
    q3_ptr: *const u8,
    gas_ptr: *const u8,
    q8_1: core::arch::x86_64::__m256i,
    q8_2: core::arch::x86_64::__m256i,
    sumi: core::arch::x86_64::__m256i,
) -> core::arch::x86_64::__m256i {
    use core::arch::x86_64::*;
    let q2_1 = _mm256_set_epi32(
        IQ3XXS_GRID[*q3_ptr.add(7) as usize] as i32,
        IQ3XXS_GRID[*q3_ptr.add(6) as usize] as i32,
        IQ3XXS_GRID[*q3_ptr.add(5) as usize] as i32,
        IQ3XXS_GRID[*q3_ptr.add(4) as usize] as i32,
        IQ3XXS_GRID[*q3_ptr.add(3) as usize] as i32,
        IQ3XXS_GRID[*q3_ptr.add(2) as usize] as i32,
        IQ3XXS_GRID[*q3_ptr.add(1) as usize] as i32,
        IQ3XXS_GRID[*q3_ptr.add(0) as usize] as i32,
    );
    let q2_2 = _mm256_set_epi32(
        IQ3XXS_GRID[*q3_ptr.add(15) as usize] as i32,
        IQ3XXS_GRID[*q3_ptr.add(14) as usize] as i32,
        IQ3XXS_GRID[*q3_ptr.add(13) as usize] as i32,
        IQ3XXS_GRID[*q3_ptr.add(12) as usize] as i32,
        IQ3XXS_GRID[*q3_ptr.add(11) as usize] as i32,
        IQ3XXS_GRID[*q3_ptr.add(10) as usize] as i32,
        IQ3XXS_GRID[*q3_ptr.add(9) as usize] as i32,
        IQ3XXS_GRID[*q3_ptr.add(8) as usize] as i32,
    );

    let aux32_0 = (gas_ptr as *const u32).read_unaligned();
    let aux32_1 = (gas_ptr.add(4) as *const u32).read_unaligned();

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

    let dot1 = _mm256_maddubs_epi16(q2_1, _mm256_sign_epi8(q8_1, s2_1));
    let dot2 = _mm256_maddubs_epi16(q2_2, _mm256_sign_epi8(q8_2, s2_2));
    let ls1 = (aux32_0 >> 28) as i16;
    let ls2 = (aux32_1 >> 28) as i16;
    let p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2 * ls1 + 1));
    let p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2 * ls2 + 1));
    _mm256_add_epi32(sumi, _mm256_add_epi32(p1, p2))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_iq3_xxs_q8_k_4rows_avx2_ptr(
    b0_base: *const u8,
    b1_base: *const u8,
    b2_base: *const u8,
    b3_base: *const u8,
    x_blocks: *const BlockQ8_K,
    nb: usize,
) -> (f32, f32, f32, f32) {
    use core::arch::x86_64::*;
    let mut accumf0 = _mm256_setzero_ps();
    let mut accumf1 = _mm256_setzero_ps();
    let mut accumf2 = _mm256_setzero_ps();
    let mut accumf3 = _mm256_setzero_ps();

    for i in 0..nb {
        let b0_ptr = b0_base.add(i * 98);
        let b1_ptr = b1_base.add(i * 98);
        let b2_ptr = b2_base.add(i * 98);
        let b3_ptr = b3_base.add(i * 98);
        if i + 2 < nb {
            _mm_prefetch(b0_ptr.add(196) as *const i8, _MM_HINT_T0);
            _mm_prefetch(b1_ptr.add(196) as *const i8, _MM_HINT_T0);
            _mm_prefetch(b2_ptr.add(196) as *const i8, _MM_HINT_T0);
            _mm_prefetch(b3_ptr.add(196) as *const i8, _MM_HINT_T0);
        }

        let block = &*x_blocks.add(i);
        let d0 = f16_to_f32_hw((b0_ptr as *const u16).read_unaligned()) * block.d;
        let d1 = f16_to_f32_hw((b1_ptr as *const u16).read_unaligned()) * block.d;
        let d2 = f16_to_f32_hw((b2_ptr as *const u16).read_unaligned()) * block.d;
        let d3 = f16_to_f32_hw((b3_ptr as *const u16).read_unaligned()) * block.d;

        let mut q8_ptr = block.qs.as_ptr();
        let mut sumi_0 = _mm256_setzero_si256();
        let mut sumi_1 = _mm256_setzero_si256();
        let mut sumi_2 = _mm256_setzero_si256();
        let mut sumi_3 = _mm256_setzero_si256();

        for sub in 0..4 {
            let q8_1 = _mm256_loadu_si256(q8_ptr as *const __m256i);
            q8_ptr = q8_ptr.add(32);
            let q8_2 = _mm256_loadu_si256(q8_ptr as *const __m256i);
            q8_ptr = q8_ptr.add(32);

            let q3_off = 2 + sub * 16;
            let gas_off = 66 + sub * 8;
            sumi_0 = iq3_xxs_subblock_step_avx2(
                b0_ptr.add(q3_off),
                b0_ptr.add(gas_off),
                q8_1,
                q8_2,
                sumi_0,
            );
            sumi_1 = iq3_xxs_subblock_step_avx2(
                b1_ptr.add(q3_off),
                b1_ptr.add(gas_off),
                q8_1,
                q8_2,
                sumi_1,
            );
            sumi_2 = iq3_xxs_subblock_step_avx2(
                b2_ptr.add(q3_off),
                b2_ptr.add(gas_off),
                q8_1,
                q8_2,
                sumi_2,
            );
            sumi_3 = iq3_xxs_subblock_step_avx2(
                b3_ptr.add(q3_off),
                b3_ptr.add(gas_off),
                q8_1,
                q8_2,
                sumi_3,
            );
        }

        accumf0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(sumi_0), accumf0);
        accumf1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), _mm256_cvtepi32_ps(sumi_1), accumf1);
        accumf2 = _mm256_fmadd_ps(_mm256_set1_ps(d2), _mm256_cvtepi32_ps(sumi_2), accumf2);
        accumf3 = _mm256_fmadd_ps(_mm256_set1_ps(d3), _mm256_cvtepi32_ps(sumi_3), accumf3);
    }

    (
        0.25f32 * hsum_float_8(accumf0),
        0.25f32 * hsum_float_8(accumf1),
        0.25f32 * hsum_float_8(accumf2),
        0.25f32 * hsum_float_8(accumf3),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_iq3_xxs_q8_k_2rows_2tokens_avx2_ptr(
    w0_base: *const u8,
    w1_base: *const u8,
    x0_blocks: *const BlockQ8_K,
    x1_blocks: *const BlockQ8_K,
    nb: usize,
) -> (f32, f32, f32, f32) {
    use core::arch::x86_64::*;
    let mut accumf_r0_t0 = _mm256_setzero_ps();
    let mut accumf_r0_t1 = _mm256_setzero_ps();
    let mut accumf_r1_t0 = _mm256_setzero_ps();
    let mut accumf_r1_t1 = _mm256_setzero_ps();

    for i in 0..nb {
        let b0_ptr = w0_base.add(i * 98);
        let b1_ptr = w1_base.add(i * 98);
        if i + 2 < nb {
            _mm_prefetch(b0_ptr.add(196) as *const i8, _MM_HINT_T0);
            _mm_prefetch(b1_ptr.add(196) as *const i8, _MM_HINT_T0);
        }
        let blk0 = &*x0_blocks.add(i);
        let blk1 = &*x1_blocks.add(i);

        let w0_scale = f16_to_f32_hw((b0_ptr as *const u16).read_unaligned());
        let w1_scale = f16_to_f32_hw((b1_ptr as *const u16).read_unaligned());
        let d_r0_t0 = w0_scale * blk0.d;
        let d_r0_t1 = w0_scale * blk1.d;
        let d_r1_t0 = w1_scale * blk0.d;
        let d_r1_t1 = w1_scale * blk1.d;

        let mut q8_0_ptr = blk0.qs.as_ptr();
        let mut q8_1_ptr = blk1.qs.as_ptr();

        let mut sumi_r0_t0 = _mm256_setzero_si256();
        let mut sumi_r0_t1 = _mm256_setzero_si256();
        let mut sumi_r1_t0 = _mm256_setzero_si256();
        let mut sumi_r1_t1 = _mm256_setzero_si256();

        for sub in 0..4 {
            let q8_1_t0 = _mm256_loadu_si256(q8_0_ptr as *const __m256i);
            q8_0_ptr = q8_0_ptr.add(32);
            let q8_2_t0 = _mm256_loadu_si256(q8_0_ptr as *const __m256i);
            q8_0_ptr = q8_0_ptr.add(32);

            let q8_1_t1 = _mm256_loadu_si256(q8_1_ptr as *const __m256i);
            q8_1_ptr = q8_1_ptr.add(32);
            let q8_2_t1 = _mm256_loadu_si256(q8_1_ptr as *const __m256i);
            q8_1_ptr = q8_1_ptr.add(32);

            let q3_off = 2 + sub * 16;
            let gas_off = 66 + sub * 8;

            // Row 0
            {
                let q3_ptr = b0_ptr.add(q3_off);
                let gas_ptr = b0_ptr.add(gas_off);
                let q2_1 = _mm256_set_epi32(
                    IQ3XXS_GRID[*q3_ptr.add(7) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(6) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(5) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(4) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(3) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(2) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(1) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(0) as usize] as i32,
                );
                let q2_2 = _mm256_set_epi32(
                    IQ3XXS_GRID[*q3_ptr.add(15) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(14) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(13) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(12) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(11) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(10) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(9) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(8) as usize] as i32,
                );
                let aux32_0 = (gas_ptr as *const u32).read_unaligned();
                let aux32_1 = (gas_ptr.add(4) as *const u32).read_unaligned();
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
                let scale1 = _mm256_set1_epi16(2 * ((aux32_0 >> 28) as i16) + 1);
                let scale2 = _mm256_set1_epi16(2 * ((aux32_1 >> 28) as i16) + 1);

                let dot1_t0 = _mm256_maddubs_epi16(q2_1, _mm256_sign_epi8(q8_1_t0, s2_1));
                let dot2_t0 = _mm256_maddubs_epi16(q2_2, _mm256_sign_epi8(q8_2_t0, s2_2));
                sumi_r0_t0 = _mm256_add_epi32(
                    sumi_r0_t0,
                    _mm256_add_epi32(
                        _mm256_madd_epi16(dot1_t0, scale1),
                        _mm256_madd_epi16(dot2_t0, scale2),
                    ),
                );

                let dot1_t1 = _mm256_maddubs_epi16(q2_1, _mm256_sign_epi8(q8_1_t1, s2_1));
                let dot2_t1 = _mm256_maddubs_epi16(q2_2, _mm256_sign_epi8(q8_2_t1, s2_2));
                sumi_r0_t1 = _mm256_add_epi32(
                    sumi_r0_t1,
                    _mm256_add_epi32(
                        _mm256_madd_epi16(dot1_t1, scale1),
                        _mm256_madd_epi16(dot2_t1, scale2),
                    ),
                );
            }

            // Row 1
            {
                let q3_ptr = b1_ptr.add(q3_off);
                let gas_ptr = b1_ptr.add(gas_off);
                let q2_1 = _mm256_set_epi32(
                    IQ3XXS_GRID[*q3_ptr.add(7) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(6) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(5) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(4) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(3) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(2) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(1) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(0) as usize] as i32,
                );
                let q2_2 = _mm256_set_epi32(
                    IQ3XXS_GRID[*q3_ptr.add(15) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(14) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(13) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(12) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(11) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(10) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(9) as usize] as i32,
                    IQ3XXS_GRID[*q3_ptr.add(8) as usize] as i32,
                );
                let aux32_0 = (gas_ptr as *const u32).read_unaligned();
                let aux32_1 = (gas_ptr.add(4) as *const u32).read_unaligned();
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
                let scale1 = _mm256_set1_epi16(2 * ((aux32_0 >> 28) as i16) + 1);
                let scale2 = _mm256_set1_epi16(2 * ((aux32_1 >> 28) as i16) + 1);

                let dot1_t0 = _mm256_maddubs_epi16(q2_1, _mm256_sign_epi8(q8_1_t0, s2_1));
                let dot2_t0 = _mm256_maddubs_epi16(q2_2, _mm256_sign_epi8(q8_2_t0, s2_2));
                sumi_r1_t0 = _mm256_add_epi32(
                    sumi_r1_t0,
                    _mm256_add_epi32(
                        _mm256_madd_epi16(dot1_t0, scale1),
                        _mm256_madd_epi16(dot2_t0, scale2),
                    ),
                );

                let dot1_t1 = _mm256_maddubs_epi16(q2_1, _mm256_sign_epi8(q8_1_t1, s2_1));
                let dot2_t1 = _mm256_maddubs_epi16(q2_2, _mm256_sign_epi8(q8_2_t1, s2_2));
                sumi_r1_t1 = _mm256_add_epi32(
                    sumi_r1_t1,
                    _mm256_add_epi32(
                        _mm256_madd_epi16(dot1_t1, scale1),
                        _mm256_madd_epi16(dot2_t1, scale2),
                    ),
                );
            }
        }

        accumf_r0_t0 = _mm256_fmadd_ps(
            _mm256_set1_ps(d_r0_t0),
            _mm256_cvtepi32_ps(sumi_r0_t0),
            accumf_r0_t0,
        );
        accumf_r0_t1 = _mm256_fmadd_ps(
            _mm256_set1_ps(d_r0_t1),
            _mm256_cvtepi32_ps(sumi_r0_t1),
            accumf_r0_t1,
        );
        accumf_r1_t0 = _mm256_fmadd_ps(
            _mm256_set1_ps(d_r1_t0),
            _mm256_cvtepi32_ps(sumi_r1_t0),
            accumf_r1_t0,
        );
        accumf_r1_t1 = _mm256_fmadd_ps(
            _mm256_set1_ps(d_r1_t1),
            _mm256_cvtepi32_ps(sumi_r1_t1),
            accumf_r1_t1,
        );
    }

    (
        0.25f32 * hsum_float_8(accumf_r0_t0),
        0.25f32 * hsum_float_8(accumf_r0_t1),
        0.25f32 * hsum_float_8(accumf_r1_t0),
        0.25f32 * hsum_float_8(accumf_r1_t1),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn vec_dot_iq3_xxs_q8_k_2rows_3tokens_avx2_ptr(
    w0_base: *const u8,
    w1_base: *const u8,
    x0_blocks: *const BlockQ8_K,
    x1_blocks: *const BlockQ8_K,
    x2_blocks: *const BlockQ8_K,
    nb: usize,
) -> (f32, f32, f32, f32, f32, f32) {
    use core::arch::x86_64::*;
    let mut accumf_r0_t0 = _mm256_setzero_ps();
    let mut accumf_r0_t1 = _mm256_setzero_ps();
    let mut accumf_r0_t2 = _mm256_setzero_ps();
    let mut accumf_r1_t0 = _mm256_setzero_ps();
    let mut accumf_r1_t1 = _mm256_setzero_ps();
    let mut accumf_r1_t2 = _mm256_setzero_ps();

    for i in 0..nb {
        let b0_ptr = w0_base.add(i * 98);
        let b1_ptr = w1_base.add(i * 98);
        if i + 2 < nb {
            _mm_prefetch(b0_ptr.add(196) as *const i8, _MM_HINT_T0);
            _mm_prefetch(b1_ptr.add(196) as *const i8, _MM_HINT_T0);
        }
        let blk0 = &*x0_blocks.add(i);
        let blk1 = &*x1_blocks.add(i);
        let blk2 = &*x2_blocks.add(i);

        let w0_scale = f16_to_f32_hw((b0_ptr as *const u16).read_unaligned());
        let w1_scale = f16_to_f32_hw((b1_ptr as *const u16).read_unaligned());

        let mut q8_0_ptr = blk0.qs.as_ptr();
        let mut q8_1_ptr = blk1.qs.as_ptr();
        let mut q8_2_ptr = blk2.qs.as_ptr();

        let mut sumi_r0_t0 = _mm256_setzero_si256();
        let mut sumi_r0_t1 = _mm256_setzero_si256();
        let mut sumi_r0_t2 = _mm256_setzero_si256();
        let mut sumi_r1_t0 = _mm256_setzero_si256();
        let mut sumi_r1_t1 = _mm256_setzero_si256();
        let mut sumi_r1_t2 = _mm256_setzero_si256();

        for sub in 0..4 {
            let q8_1_t0 = _mm256_loadu_si256(q8_0_ptr as *const __m256i);
            q8_0_ptr = q8_0_ptr.add(32);
            let q8_2_t0 = _mm256_loadu_si256(q8_0_ptr as *const __m256i);
            q8_0_ptr = q8_0_ptr.add(32);

            let q8_1_t1 = _mm256_loadu_si256(q8_1_ptr as *const __m256i);
            q8_1_ptr = q8_1_ptr.add(32);
            let q8_2_t1 = _mm256_loadu_si256(q8_1_ptr as *const __m256i);
            q8_1_ptr = q8_1_ptr.add(32);

            let q8_1_t2 = _mm256_loadu_si256(q8_2_ptr as *const __m256i);
            q8_2_ptr = q8_2_ptr.add(32);
            let q8_2_t2 = _mm256_loadu_si256(q8_2_ptr as *const __m256i);
            q8_2_ptr = q8_2_ptr.add(32);

            let q3_off = 2 + sub * 16;
            let gas_off = 66 + sub * 8;

            sumi_r0_t0 = iq3_xxs_subblock_step_avx2(
                b0_ptr.add(q3_off),
                b0_ptr.add(gas_off),
                q8_1_t0,
                q8_2_t0,
                sumi_r0_t0,
            );
            sumi_r0_t1 = iq3_xxs_subblock_step_avx2(
                b0_ptr.add(q3_off),
                b0_ptr.add(gas_off),
                q8_1_t1,
                q8_2_t1,
                sumi_r0_t1,
            );
            sumi_r0_t2 = iq3_xxs_subblock_step_avx2(
                b0_ptr.add(q3_off),
                b0_ptr.add(gas_off),
                q8_1_t2,
                q8_2_t2,
                sumi_r0_t2,
            );

            sumi_r1_t0 = iq3_xxs_subblock_step_avx2(
                b1_ptr.add(q3_off),
                b1_ptr.add(gas_off),
                q8_1_t0,
                q8_2_t0,
                sumi_r1_t0,
            );
            sumi_r1_t1 = iq3_xxs_subblock_step_avx2(
                b1_ptr.add(q3_off),
                b1_ptr.add(gas_off),
                q8_1_t1,
                q8_2_t1,
                sumi_r1_t1,
            );
            sumi_r1_t2 = iq3_xxs_subblock_step_avx2(
                b1_ptr.add(q3_off),
                b1_ptr.add(gas_off),
                q8_1_t2,
                q8_2_t2,
                sumi_r1_t2,
            );
        }

        accumf_r0_t0 = _mm256_fmadd_ps(
            _mm256_set1_ps(w0_scale * blk0.d),
            _mm256_cvtepi32_ps(sumi_r0_t0),
            accumf_r0_t0,
        );
        accumf_r0_t1 = _mm256_fmadd_ps(
            _mm256_set1_ps(w0_scale * blk1.d),
            _mm256_cvtepi32_ps(sumi_r0_t1),
            accumf_r0_t1,
        );
        accumf_r0_t2 = _mm256_fmadd_ps(
            _mm256_set1_ps(w0_scale * blk2.d),
            _mm256_cvtepi32_ps(sumi_r0_t2),
            accumf_r0_t2,
        );
        accumf_r1_t0 = _mm256_fmadd_ps(
            _mm256_set1_ps(w1_scale * blk0.d),
            _mm256_cvtepi32_ps(sumi_r1_t0),
            accumf_r1_t0,
        );
        accumf_r1_t1 = _mm256_fmadd_ps(
            _mm256_set1_ps(w1_scale * blk1.d),
            _mm256_cvtepi32_ps(sumi_r1_t1),
            accumf_r1_t1,
        );
        accumf_r1_t2 = _mm256_fmadd_ps(
            _mm256_set1_ps(w1_scale * blk2.d),
            _mm256_cvtepi32_ps(sumi_r1_t2),
            accumf_r1_t2,
        );
    }

    (
        0.25f32 * hsum_float_8(accumf_r0_t0),
        0.25f32 * hsum_float_8(accumf_r0_t1),
        0.25f32 * hsum_float_8(accumf_r0_t2),
        0.25f32 * hsum_float_8(accumf_r1_t0),
        0.25f32 * hsum_float_8(accumf_r1_t1),
        0.25f32 * hsum_float_8(accumf_r1_t2),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_iq3_xxs_prequant(
    out: &mut [f32],
    weight_bytes: &[u8],
    x_q8_k: &[BlockQ8_K],
    n_rows: usize,
    n_cols: usize,
) {
    let row_bytes = (n_cols / 256) * 98;
    assert_eq!(weight_bytes.len(), n_rows * row_bytes);

    let nb = x_q8_k.len();
    let mut r = 0;
    while r + 1 < n_rows {
        let r0_ptr = weight_bytes.as_ptr().add(r * row_bytes);
        let r1_ptr = weight_bytes.as_ptr().add((r + 1) * row_bytes);

        let (v0, v1) = vec_dot_iq3_xxs_q8_k_2rows_avx2_ptr(r0_ptr, r1_ptr, x_q8_k.as_ptr(), nb);
        out[r] = v0;
        out[r + 1] = v1;
        r += 2;
    }
    while r < n_rows {
        let row_slice = &weight_bytes[r * row_bytes..(r + 1) * row_bytes];
        out[r] = vec_dot_iq3_xxs_q8_k_avx2(row_slice, x_q8_k);
        r += 1;
    }
}

#[allow(dead_code)]
pub fn gemv_iq4_nl(out: &mut [f32], weight_bytes: &[u8], x: &[f32], n_rows: usize, n_cols: usize) {
    assert_eq!(n_cols % 32, 0);
    assert_eq!(out.len(), n_rows);
    let nb = n_cols / 32;
    let mut x_q8_0 = vec![
        BlockQ8_0 {
            d: 0.0,
            qs: [0; 32]
        };
        nb
    ];
    quantize_row_q8_0(x, &mut x_q8_0);
    unsafe { gemv_iq4_nl_prequant(out, weight_bytes, &x_q8_0, n_rows, n_cols) };
}

#[allow(dead_code)]
pub fn gemv_iq3_xxs(out: &mut [f32], weight_bytes: &[u8], x: &[f32], n_rows: usize, n_cols: usize) {
    assert_eq!(n_cols % 256, 0);
    assert_eq!(out.len(), n_rows);
    let nb = n_cols / 256;
    let mut x_q8_k = vec![
        BlockQ8_K {
            d: 0.0,
            qs: [0; 256],
            bsums: [0; 16]
        };
        nb
    ];
    quantize_row_q8_k(x, &mut x_q8_k);
    unsafe { gemv_iq3_xxs_prequant(out, weight_bytes, &x_q8_k, n_rows, n_cols) };
}

#[allow(dead_code)]
pub fn gemv_q8_0(out: &mut [f32], weight_bytes: &[u8], x: &[f32], n_rows: usize, n_cols: usize) {
    assert_eq!(n_cols % 32, 0);
    assert_eq!(out.len(), n_rows);
    let row_bytes = (n_cols / 32) * 34;
    assert_eq!(weight_bytes.len(), n_rows * row_bytes);

    let nb = n_cols / 32;
    let mut x_q8_0 = vec![
        BlockQ8_0 {
            d: 0.0,
            qs: [0; 32]
        };
        nb
    ];
    quantize_row_q8_0(x, &mut x_q8_0);

    for r in 0..n_rows {
        let row_slice = &weight_bytes[r * row_bytes..(r + 1) * row_bytes];
        out[r] = unsafe { vec_dot_q8_0_q8_0_avx2(row_slice, &x_q8_0) };
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn q8_activation_scale_and_halfway_values_match_ggml_cpu() {
        let mut input = [0.0; 64];
        input[..8].copy_from_slice(&[127.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 0.0]);
        input[32] = 1.2345;
        let mut blocks = [BlockQ8_0 {
            d: 0.0,
            qs: [0; 32],
        }; 2];
        quantize_row_q8_0(&input, &mut blocks);
        assert_eq!(&blocks[0].qs[..8], &[127, 0, 0, 2, -2, 2, -2, 0]);
        let scale = input[32] / 127.0;
        assert_ne!(scale, half::f16::from_f32(scale).to_f32());
        assert_eq!(blocks[1].d, half::f16::from_f32(scale).to_f32());
    }

    #[test]
    fn iq3_all_sign_patterns_match_scalar_in_every_simd_kernel() {
        if !is_x86_feature_detected!("avx2")
            || !is_x86_feature_detected!("fma")
            || !is_x86_feature_detected!("f16c")
        {
            return;
        }
        let mut activation = BlockQ8_K {
            d: 1.0,
            qs: [0; 256],
            bsums: [0; 16],
        };
        for (i, q) in activation.qs.iter_mut().enumerate() {
            *q = (i % 8 + 1) as i8;
        }
        for index in 0..128u32 {
            let mut weight = [0u8; 98];
            weight[..2].copy_from_slice(&half::f16::ONE.to_le_bytes());
            let packed = index | (index << 7) | (index << 14) | (index << 21);
            for group in weight[66..].chunks_exact_mut(4) {
                group.copy_from_slice(&packed.to_le_bytes());
            }
            // IQ3 grid index 0 contains four 4s; scale index 0 and d=1 give
            // weights of +/-1. The eighth sign enforces even parity.
            let mask = index | ((index.count_ones() & 1) << 7);
            let expected: f32 = (0..256)
                .map(|i| {
                    let x = activation.qs[i] as f32;
                    if mask & (1 << (i % 8)) == 0 { x } else { -x }
                })
                .sum();
            if index == 126 {
                assert_eq!(expected, -576.0);
            }
            let w = weight.as_ptr();
            let x = &activation as *const _;
            let mut actual = Vec::new();
            unsafe {
                actual.push(vec_dot_iq3_xxs_q8_k_avx2(&weight, &[activation]));
                let (a, b) = vec_dot_iq3_xxs_q8_k_2rows_avx2_ptr(w, w, x, 1);
                actual.extend([a, b]);
                let (a, b, c, d) = vec_dot_iq3_xxs_q8_k_4rows_avx2_ptr(w, w, w, w, x, 1);
                actual.extend([a, b, c, d]);
                let (a, b) = vec_dot_iq3_xxs_q8_k_2tokens_avx2_ptr(w, x, x, 1);
                actual.extend([a, b]);
                let (a, b, c) = vec_dot_iq3_xxs_q8_k_3tokens_avx2_ptr(w, x, x, x, 1);
                actual.extend([a, b, c]);
                let (a, b, c, d) = vec_dot_iq3_xxs_q8_k_2rows_2tokens_avx2_ptr(w, w, x, x, 1);
                actual.extend([a, b, c, d]);
                let (a, b, c, d, e, f) =
                    vec_dot_iq3_xxs_q8_k_2rows_3tokens_avx2_ptr(w, w, x, x, x, 1);
                actual.extend([a, b, c, d, e, f]);
            }
            assert!(
                actual.iter().all(|v| *v == expected),
                "sign pattern {index}: {actual:?}, expected {expected}"
            );
        }
    }

    #[test]
    fn silu_preserves_the_sigmoid_at_former_cutoffs() {
        for x in [-16.0f32, -8.0, -7.99, 0.0, 3.99, 4.0, 8.0, 16.0] {
            let expected = (x as f64 / (1.0 + (-(x as f64)).exp())) as f32;
            assert!((fast_silu(x) - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn q8_matrix_vector_matches_dequantized_reference() {
        let cols = 64;
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.17).sin()).collect();
        let mut weights = vec![0u8; 2 * (cols / 32) * 34];
        for (block, bytes) in weights.chunks_exact_mut(34).enumerate() {
            bytes[..2]
                .copy_from_slice(&half::f16::from_f32(0.03 * (block + 1) as f32).to_le_bytes());
            for (i, byte) in bytes[2..].iter_mut().enumerate() {
                *byte = (i as i8 - 16) as u8;
            }
        }
        let mut actual = vec![0.0; 2];
        gemv_q8_0(&mut actual, &weights, &x, 2, cols);
        for (row, bytes) in weights.chunks_exact(68).enumerate() {
            let mut dequantized = vec![0.0; cols];
            dequantize_row_q8_0(bytes, &mut dequantized);
            let expected: f32 = dequantized.iter().zip(&x).map(|(w, v)| w * v).sum();
            assert!((actual[row] - expected).abs() < 0.05 * expected.abs().max(1.0));
        }
    }

    #[test]
    fn test_vec_dot_q8_0() {
        let n = 64;
        let mut x = vec![0f32; n];
        for i in 0..n {
            x[i] = ((i as f32) * 0.1).sin();
        }
        let nb = n / 32;
        let mut x_q8_0 = vec![
            BlockQ8_0 {
                d: 0.0,
                qs: [0; 32]
            };
            nb
        ];
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
        assert!(
            diff < 0.05 * scalar_dot.abs().max(1.0),
            "Q8_0 mismatch: {scalar_dot} vs {simd_dot}"
        );
    }

    #[test]
    fn test_vec_dot_iq4_nl() {
        let n = 64;
        let mut x = vec![0f32; n];
        for i in 0..n {
            x[i] = ((i as f32) * 0.2).cos();
        }
        let nb = n / 32;
        let mut x_q8_0 = vec![
            BlockQ8_0 {
                d: 0.0,
                qs: [0; 32]
            };
            nb
        ];
        quantize_row_q8_0(&x, &mut x_q8_0);

        // Synthetic IQ4_NL weights
        let mut w_bytes = vec![0u8; nb * 18];
        for b in 0..nb {
            let d_bytes = half::f16::from_f32(0.02).to_le_bytes();
            w_bytes[b * 18] = d_bytes[0];
            w_bytes[b * 18 + 1] = d_bytes[1];
            for j in 0..16 {
                w_bytes[b * 18 + 2 + j] = (j as u8) | ((15 - j as u8) << 4);
            }
        }

        // Scalar dequantize
        let mut w_dequant = vec![0f32; n];
        dequantize_row_iq4_nl(&w_bytes, &mut w_dequant);

        let scalar_dot: f32 = w_dequant.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let simd_dot = unsafe { vec_dot_iq4_nl_q8_0_avx2(&w_bytes, &x_q8_0) };

        println!("IQ4_NL scalar_dot: {scalar_dot}, simd_dot: {simd_dot}");
        let diff = (scalar_dot - simd_dot).abs();
        assert!(
            diff < 0.05 * scalar_dot.abs().max(1.0),
            "IQ4_NL mismatch: {scalar_dot} vs {simd_dot}"
        );
    }

    #[test]
    fn test_vec_dot_iq3_xxs() {
        let n = 256;
        let mut x = vec![0f32; n];
        for i in 0..n {
            x[i] = ((i as f32) * 0.05).sin();
        }
        let nb = n / 256;
        let mut x_q8_k = vec![
            BlockQ8_K {
                d: 0.0,
                qs: [0; 256],
                bsums: [0; 16]
            };
            nb
        ];
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
        assert!(
            diff < 0.05 * scalar_dot.abs().max(1.0),
            "IQ3_XXS mismatch: {scalar_dot} vs {simd_dot}"
        );

        let (dot0, dot1) = unsafe { vec_dot_iq3_xxs_q8_k_2rows_avx2(&w_bytes, &w_bytes, &x_q8_k) };
        assert!(
            (dot0 - simd_dot).abs() < 1e-4,
            "2rows dot0 mismatch: {dot0} vs {simd_dot}"
        );
        assert!(
            (dot1 - simd_dot).abs() < 1e-4,
            "2rows dot1 mismatch: {dot1} vs {simd_dot}"
        );

        let (t_dot0, t_dot1) = unsafe {
            vec_dot_iq3_xxs_q8_k_2tokens_avx2_ptr(
                w_bytes.as_ptr(),
                x_q8_k.as_ptr(),
                x_q8_k.as_ptr(),
                nb,
            )
        };
        assert!(
            (t_dot0 - simd_dot).abs() < 1e-4,
            "2tokens dot0 mismatch: {t_dot0} vs {simd_dot}"
        );
        assert!(
            (t_dot1 - simd_dot).abs() < 1e-4,
            "2tokens dot1 mismatch: {t_dot1} vs {simd_dot}"
        );

        let mut x2 = vec![0f32; n];
        for i in 0..n {
            x2[i] = ((i as f32) * 0.12).cos();
        }
        let mut x2_q8_k = vec![
            BlockQ8_K {
                d: 0.0,
                qs: [0; 256],
                bsums: [0; 16]
            };
            nb
        ];
        quantize_row_q8_k(&x2, &mut x2_q8_k);
        let simd_dot2 = unsafe { vec_dot_iq3_xxs_q8_k_avx2(&w_bytes, &x2_q8_k) };

        let (t_dot0_diff, t_dot1_diff) = unsafe {
            vec_dot_iq3_xxs_q8_k_2tokens_avx2_ptr(
                w_bytes.as_ptr(),
                x_q8_k.as_ptr(),
                x2_q8_k.as_ptr(),
                nb,
            )
        };
        assert!(
            (t_dot0_diff - simd_dot).abs() < 1e-4,
            "2tokens different x0 mismatch: {t_dot0_diff} vs {simd_dot}"
        );
        assert!(
            (t_dot1_diff - simd_dot2).abs() < 1e-4,
            "2tokens different x1 mismatch: {t_dot1_diff} vs {simd_dot2}"
        );
    }

    #[test]
    fn test_quantize_and_dot_iq4_nl_48blocks() {
        let n = 48 * 32;
        let mut x = vec![0f32; n];
        let mut w0 = vec![0f32; n];
        let mut w1 = vec![0f32; n];
        for i in 0..n {
            x[i] = ((i as f32) * 0.07).sin() * 0.1;
            w0[i] = ((i as f32) * 0.13).cos() * 0.2;
            w1[i] = ((i as f32) * 0.17).sin() * 0.15;
        }

        let mut x_q8_0 = vec![
            BlockQ8_0 {
                d: 0.0,
                qs: [0; 32]
            };
            48
        ];
        quantize_row_q8_0(&x, &mut x_q8_0);

        let mut x_qs = unsafe { [core::arch::x86_64::_mm256_setzero_si256(); 48] };
        let mut x_d = [0.0f32; 48];
        for i in 0..48 {
            x_qs[i] = unsafe {
                core::arch::x86_64::_mm256_loadu_si256(x_q8_0[i].qs.as_ptr() as *const _)
            };
            x_d[i] = x_q8_0[i].d;
        }

        let mut raw_w0 = vec![0u8; 48 * 18];
        let mut raw_w1 = vec![0u8; 48 * 18];
        quantize_row_iq4_nl(&w0, &mut raw_w0);
        quantize_row_iq4_nl(&w1, &mut raw_w1);

        let mut deq0 = vec![0f32; n];
        let mut deq1 = vec![0f32; n];
        dequantize_row_iq4_nl(&raw_w0, &mut deq0);
        dequantize_row_iq4_nl(&raw_w1, &mut deq1);

        let ref0: f32 = deq0.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let ref1: f32 = deq1.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

        let (simd0, simd1) = unsafe {
            vec_dot_iq4_nl_48blocks_2rows_avx2(raw_w0.as_ptr(), raw_w1.as_ptr(), &x_qs, &x_d)
        };
        let single0 = unsafe { vec_dot_iq4_nl_48blocks_avx2(raw_w0.as_ptr(), &x_qs, &x_d) };

        assert!(
            (ref0 - simd0).abs() < 0.05 * ref0.abs().max(1.0),
            "Beluga row 0 mismatch: {ref0} vs {simd0}"
        );
        assert!(
            (ref1 - simd1).abs() < 0.05 * ref1.abs().max(1.0),
            "Beluga row 1 mismatch: {ref1} vs {simd1}"
        );
        assert!(
            (simd0 - single0).abs() < 1e-4,
            "Beluga dual vs single mismatch: {simd0} vs {single0}"
        );
    }

    #[test]
    fn test_quantize_and_dot_iq4_nl_28blocks() {
        let n = 28 * 32;
        let mut x = vec![0f32; n];
        let mut w0 = vec![0f32; n];
        let mut w1 = vec![0f32; n];
        for i in 0..n {
            x[i] = ((i as f32) * 0.09).cos() * 0.1;
            w0[i] = ((i as f32) * 0.11).sin() * 0.2;
            w1[i] = ((i as f32) * 0.15).cos() * 0.15;
        }

        let mut x_q8_0 = vec![
            BlockQ8_0 {
                d: 0.0,
                qs: [0; 32]
            };
            28
        ];
        quantize_row_q8_0(&x, &mut x_q8_0);

        let mut x_qs = unsafe { [core::arch::x86_64::_mm256_setzero_si256(); 28] };
        let mut x_d = [0.0f32; 28];
        for i in 0..28 {
            x_qs[i] = unsafe {
                core::arch::x86_64::_mm256_loadu_si256(x_q8_0[i].qs.as_ptr() as *const _)
            };
            x_d[i] = x_q8_0[i].d;
        }

        let mut raw_w0 = vec![0u8; 28 * 18];
        let mut raw_w1 = vec![0u8; 28 * 18];
        quantize_row_iq4_nl(&w0, &mut raw_w0);
        quantize_row_iq4_nl(&w1, &mut raw_w1);

        let mut deq0 = vec![0f32; n];
        let mut deq1 = vec![0f32; n];
        dequantize_row_iq4_nl(&raw_w0, &mut deq0);
        dequantize_row_iq4_nl(&raw_w1, &mut deq1);

        let ref0: f32 = deq0.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let ref1: f32 = deq1.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

        let (simd0, simd1) = unsafe {
            vec_dot_iq4_nl_28blocks_2rows_avx2(raw_w0.as_ptr(), raw_w1.as_ptr(), &x_qs, &x_d)
        };
        let single0 = unsafe { vec_dot_iq4_nl_28blocks_avx2(raw_w0.as_ptr(), &x_qs, &x_d) };

        assert!(
            (ref0 - simd0).abs() < 0.05 * ref0.abs().max(1.0),
            "Manta row 0 mismatch: {ref0} vs {simd0}"
        );
        assert!(
            (ref1 - simd1).abs() < 0.05 * ref1.abs().max(1.0),
            "Manta row 1 mismatch: {ref1} vs {simd1}"
        );
        assert!(
            (simd0 - single0).abs() < 1e-4,
            "Manta dual vs single mismatch: {simd0} vs {single0}"
        );
    }

    #[test]
    fn test_quantize_and_dot_iq4_nl_96blocks() {
        let n = 96 * 32;
        let mut x = vec![0f32; n];
        let mut w0 = vec![0f32; n];
        let mut w1 = vec![0f32; n];
        for i in 0..n {
            x[i] = ((i as f32) * 0.07).cos() * 0.1;
            w0[i] = ((i as f32) * 0.13).sin() * 0.2;
            w1[i] = ((i as f32) * 0.17).cos() * 0.15;
        }

        let mut x_q8_0 = vec![
            BlockQ8_0 {
                d: 0.0,
                qs: [0; 32]
            };
            96
        ];
        quantize_row_q8_0(&x, &mut x_q8_0);

        let mut x_qs = unsafe { [core::arch::x86_64::_mm256_setzero_si256(); 96] };
        let mut x_d = [0.0f32; 96];
        for i in 0..96 {
            x_qs[i] = unsafe {
                core::arch::x86_64::_mm256_loadu_si256(x_q8_0[i].qs.as_ptr() as *const _)
            };
            x_d[i] = x_q8_0[i].d;
        }

        let mut raw_w0 = vec![0u8; 96 * 18];
        let mut raw_w1 = vec![0u8; 96 * 18];
        quantize_row_iq4_nl(&w0, &mut raw_w0);
        quantize_row_iq4_nl(&w1, &mut raw_w1);

        let mut deq0 = vec![0f32; n];
        let mut deq1 = vec![0f32; n];
        dequantize_row_iq4_nl(&raw_w0, &mut deq0);
        dequantize_row_iq4_nl(&raw_w1, &mut deq1);

        let ref0: f32 = deq0.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let ref1: f32 = deq1.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

        let (simd0, simd1) = unsafe {
            vec_dot_iq4_nl_96blocks_2rows_avx2(raw_w0.as_ptr(), raw_w1.as_ptr(), &x_qs, &x_d)
        };
        let single0 = unsafe { vec_dot_iq4_nl_96blocks_avx2(raw_w0.as_ptr(), &x_qs, &x_d) };

        assert!(
            (ref0 - simd0).abs() < 0.05 * ref0.abs().max(1.0),
            "Blue row 0 mismatch: {ref0} vs {simd0}"
        );
        assert!(
            (ref1 - simd1).abs() < 0.05 * ref1.abs().max(1.0),
            "Blue row 1 mismatch: {ref1} vs {simd1}"
        );
        assert!(
            (simd0 - single0).abs() < 1e-4,
            "Blue dual vs single mismatch: {simd0} vs {single0}"
        );
    }

    #[test]
    fn test_quantize_and_dot_q8_0_96blocks() {
        let n = 96 * 32;
        let mut x = vec![0f32; n];
        for i in 0..n {
            x[i] = ((i as f32) * 0.05).sin();
        }
        let mut x_q8_0 = vec![
            BlockQ8_0 {
                d: 0.0,
                qs: [0; 32]
            };
            96
        ];
        quantize_row_q8_0(&x, &mut x_q8_0);

        let mut x_qs = unsafe { [core::arch::x86_64::_mm256_setzero_si256(); 96] };
        let mut x_d = [0.0f32; 96];
        for i in 0..96 {
            x_qs[i] = unsafe {
                core::arch::x86_64::_mm256_loadu_si256(x_q8_0[i].qs.as_ptr() as *const _)
            };
            x_d[i] = x_q8_0[i].d;
        }

        // Two rows of Q8_0 weights
        let mut w0_bytes = vec![0u8; 96 * 34];
        let mut w1_bytes = vec![0u8; 96 * 34];
        for b in 0..96 {
            let d0_bytes = half::f16::from_f32(0.04).to_le_bytes();
            w0_bytes[b * 34] = d0_bytes[0];
            w0_bytes[b * 34 + 1] = d0_bytes[1];
            let d1_bytes = half::f16::from_f32(0.06).to_le_bytes();
            w1_bytes[b * 34] = d1_bytes[0];
            w1_bytes[b * 34 + 1] = d1_bytes[1];
            for j in 0..32 {
                w0_bytes[b * 34 + 2 + j] = ((((j as i32 + b as i32) % 32) - 16) as i8) as u8;
                w1_bytes[b * 34 + 2 + j] = ((((j as i32 * 2 + b as i32) % 32) - 16) as i8) as u8;
            }
        }

        let mut w0_dequant = vec![0f32; n];
        let mut w1_dequant = vec![0f32; n];
        dequantize_row_q8_0(&w0_bytes, &mut w0_dequant);
        dequantize_row_q8_0(&w1_bytes, &mut w1_dequant);

        let ref0: f32 = w0_dequant.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        let ref1: f32 = w1_dequant.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

        let (simd0, simd1) = unsafe {
            vec_dot_q8_0_96blocks_2rows_avx2(w0_bytes.as_ptr(), w1_bytes.as_ptr(), &x_qs, &x_d)
        };
        let single0 = unsafe { vec_dot_q8_0_96blocks_avx2(w0_bytes.as_ptr(), &x_qs, &x_d) };

        assert!(
            (ref0 - simd0).abs() < 0.05 * ref0.abs().max(1.0),
            "Blue Q8_0 row 0 mismatch: {ref0} vs {simd0}"
        );
        assert!(
            (ref1 - simd1).abs() < 0.05 * ref1.abs().max(1.0),
            "Blue Q8_0 row 1 mismatch: {ref1} vs {simd1}"
        );
        assert!(
            (simd0 - single0).abs() < 1e-4,
            "Blue Q8_0 dual vs single mismatch: {simd0} vs {single0}"
        );
    }
}

#[cfg(any(test, not(target_arch = "x86_64")))]
#[path = "quants_scalar.rs"]
pub(crate) mod scalar;
#[cfg(not(target_arch = "x86_64"))]
pub use scalar::*;
