use super::{
    BlockQ8_0, BlockQ8_K, dequantize_row_iq3_xxs, dequantize_row_iq4_nl, dequantize_row_q8_0,
};

pub unsafe fn vec_dot_q8_0_q8_0_avx2(weights: &[u8], x: &[BlockQ8_0]) -> f32 {
    let mut w = vec![0.0; x.len() * 32];
    dequantize_row_q8_0(weights, &mut w);
    w.chunks_exact(32)
        .zip(x)
        .map(|(w, b)| {
            w.iter()
                .zip(b.qs)
                .map(|(v, q)| v * q as f32 * b.d)
                .sum::<f32>()
        })
        .sum()
}
pub unsafe fn vec_dot_iq4_nl_q8_0_avx2(weights: &[u8], x: &[BlockQ8_0]) -> f32 {
    let mut w = vec![0.0; x.len() * 32];
    dequantize_row_iq4_nl(weights, &mut w);
    w.chunks_exact(32)
        .zip(x)
        .map(|(w, b)| {
            w.iter()
                .zip(b.qs)
                .map(|(v, q)| v * q as f32 * b.d)
                .sum::<f32>()
        })
        .sum()
}
pub unsafe fn vec_dot_iq3_xxs_q8_k_avx2(weights: &[u8], x: &[BlockQ8_K]) -> f32 {
    let mut w = vec![0.0; x.len() * 256];
    dequantize_row_iq3_xxs(weights, &mut w);
    w.chunks_exact(256)
        .zip(x)
        .map(|(w, b)| {
            w.iter()
                .zip(b.qs)
                .map(|(v, q)| v * q as f32 * b.d)
                .sum::<f32>()
        })
        .sum()
}
pub unsafe fn gemv_iq4_nl_prequant(
    out: &mut [f32],
    weights: &[u8],
    x: &[BlockQ8_0],
    rows: usize,
    cols: usize,
) {
    let len = cols / 32 * 18;
    for (o, row) in out.iter_mut().take(rows).zip(weights.chunks_exact(len)) {
        *o = unsafe { vec_dot_iq4_nl_q8_0_avx2(row, x) };
    }
}
pub unsafe fn gemv_iq3_xxs_prequant(
    out: &mut [f32],
    weights: &[u8],
    x: &[BlockQ8_K],
    rows: usize,
    cols: usize,
) {
    let len = cols / 256 * 98;
    for (o, row) in out.iter_mut().take(rows).zip(weights.chunks_exact(len)) {
        *o = unsafe { vec_dot_iq3_xxs_q8_k_avx2(row, x) };
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    #[test]
    fn scalar_dot_products_match_simd() {
        if !std::is_x86_feature_detected!("avx2")
            || !std::is_x86_feature_detected!("fma")
            || !std::is_x86_feature_detected!("f16c")
        {
            return;
        }
        let mut iq3 = vec![0u8; 98];
        iq3[..2].copy_from_slice(&half::f16::from_f32(0.125).to_le_bytes());
        for (i, b) in iq3[2..].iter_mut().enumerate() {
            *b = (i.wrapping_mul(31)) as u8;
        }
        let x3 = [BlockQ8_K {
            d: 0.01,
            qs: std::array::from_fn(|i| (i as i8).wrapping_sub(64)),
            bsums: [0; 16],
        }];
        let mut iq4 = vec![0u8; 18];
        iq4[..2].copy_from_slice(&half::f16::from_f32(0.125).to_le_bytes());
        for (i, b) in iq4[2..].iter_mut().enumerate() {
            *b = (i.wrapping_mul(13)) as u8;
        }
        let mut q8 = vec![0u8; 34];
        q8[..2].copy_from_slice(&half::f16::from_f32(0.125).to_le_bytes());
        for (i, b) in q8[2..].iter_mut().enumerate() {
            *b = (i as i8 - 16) as u8;
        }
        let x8 = [BlockQ8_0 {
            d: 0.02,
            qs: std::array::from_fn(|i| i as i8 - 16),
        }];
        let pairs = unsafe {
            [
                (
                    vec_dot_iq3_xxs_q8_k_avx2(&iq3, &x3),
                    super::super::vec_dot_iq3_xxs_q8_k_avx2(&iq3, &x3),
                ),
                (
                    vec_dot_iq4_nl_q8_0_avx2(&iq4, &x8),
                    super::super::vec_dot_iq4_nl_q8_0_avx2(&iq4, &x8),
                ),
                (
                    vec_dot_q8_0_q8_0_avx2(&q8, &x8),
                    super::super::vec_dot_q8_0_q8_0_avx2(&q8, &x8),
                ),
            ]
        };
        for (scalar, simd) in pairs {
            assert!(
                (scalar - simd).abs() < 0.001 * simd.abs().max(1.0),
                "scalar={scalar}, simd={simd}"
            );
        }
    }
}

#[cfg(test)]
mod portable_tests {
    use super::*;
    #[test]
    fn scalar_q8_dot_uses_quantization_scales() {
        let mut weights = vec![0u8; 34];
        weights[..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        weights[2..].fill(2);
        let x = [BlockQ8_0 {
            d: 0.25,
            qs: [4; 32],
        }];
        assert!((unsafe { vec_dot_q8_0_q8_0_avx2(&weights, &x) } - 32.0).abs() < 1e-6);
    }
}
