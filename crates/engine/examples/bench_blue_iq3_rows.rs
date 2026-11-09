//! Compare existing IQ3 row kernels at Blue's matrix dimensions, without weights changes.
use nioxs_engine::quants::{self, BlockQ8_K};
use std::{hint::black_box, time::Instant};

fn calculate(weights: &[u8], x: &[BlockQ8_K], out: &mut [f32], rows_per_step: usize) {
    let nb = x.len();
    let stride = nb * 98;
    for row in (0..out.len()).step_by(rows_per_step) {
        unsafe {
            let p = weights.as_ptr().add(row * stride);
            if rows_per_step == 4 {
                let (a, b, c, d) = quants::vec_dot_iq3_xxs_q8_k_4rows_avx2_ptr(
                    p,
                    p.add(stride),
                    p.add(stride * 2),
                    p.add(stride * 3),
                    x.as_ptr(),
                    nb,
                );
                out[row..row + 4].copy_from_slice(&[a, b, c, d]);
            } else {
                let (a, b) =
                    quants::vec_dot_iq3_xxs_q8_k_2rows_avx2_ptr(p, p.add(stride), x.as_ptr(), nb);
                out[row..row + 2].copy_from_slice(&[a, b]);
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    anyhow::ensure!(
        is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c"),
        "AVX2, FMA and F16C required"
    );
    for (rows, cols) in [(8192, 3072), (3072, 8192)] {
        let mut weights = vec![0u8; rows * (cols / 256) * 98];
        let mut seed = 12345u32;
        for block in weights.chunks_exact_mut(98) {
            block[..2].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
            for byte in &mut block[2..] {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                *byte = (seed >> 24) as u8;
            }
        }
        let input: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.07).sin()).collect();
        let mut x = vec![
            BlockQ8_K {
                d: 0.0,
                qs: [0; 256],
                bsums: [0; 16]
            };
            cols / 256
        ];
        quants::quantize_row_q8_k(&input, &mut x);
        let mut two = vec![0.0; rows];
        let mut four = two.clone();
        calculate(&weights, &x, &mut two, 2);
        calculate(&weights, &x, &mut four, 4);
        let different_bits = two
            .iter()
            .zip(&four)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let max_abs_difference = two
            .iter()
            .zip(&four)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mut timings = Vec::new();
        for width in [2, 4, 4, 2, 2, 4] {
            let start = Instant::now();
            for _ in 0..4 {
                calculate(
                    black_box(&weights),
                    black_box(&x),
                    black_box(&mut four),
                    width,
                );
                black_box(&four);
            }
            timings.push(serde_json::json!({"rows_per_step":width,"seconds_per_matrix":start.elapsed().as_secs_f64()/4.0}));
        }
        println!(
            "{}",
            serde_json::json!({"rows":rows,"cols":cols,"different_output_bits":different_bits,"max_abs_difference":max_abs_difference,"timings":timings})
        );
    }
    Ok(())
}
