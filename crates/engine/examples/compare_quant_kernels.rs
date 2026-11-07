//! Emit real model rows and deterministic inputs for an independent GGML oracle.
use anyhow::Result;
use nioxs_engine::{gguf::GgufModel, quants};
use serde_json::json;
use std::fs::File;

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("expected MODEL"))?;
    let file = File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gguf = GgufModel::read(&mut &file)?;
    let mut names = vec!["token_embd.weight".to_owned()];
    for layer in [0, 12, 23] {
        for projection in [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ] {
            names.push(format!("blk.{layer}.{projection}.weight"));
        }
    }
    let mut fixtures = Vec::new();
    for name in names {
        let Some(meta) = gguf.tensors.get(&name) else {
            continue;
        };
        let cols = meta.shape[1];
        let (block_size, block_bytes) = match meta.dtype {
            8 => (32, 34),
            18 => (256, 98),
            20 => (32, 18),
            _ => continue,
        };
        let row_bytes = cols / block_size * block_bytes;
        for variant in 0..3 {
            let x: Vec<f32> = (0..cols)
                .map(|i| match variant {
                    0 => (i as f32 * 0.17).sin() + (i as f32 * 0.023).cos(),
                    1 => match i % 8 {
                        0 => 127.0,
                        1 => 0.5,
                        2 => -0.5,
                        3 => 1.5,
                        4 => -1.5,
                        5 => 2.5,
                        6 => -2.5,
                        _ => 0.0,
                    },
                    _ => (i as f32 * 0.31).sin() * 0.0003,
                })
                .collect();
            let rows: Vec<_> = [0usize, 1, meta.shape[0] / 2, meta.shape[0] - 1]
                .into_iter()
                .collect();
            let (scales, quantized) = if meta.dtype == 18 {
                let mut blocks = vec![
                    quants::BlockQ8_K {
                        d: 0.0,
                        qs: [0; 256],
                        bsums: [0; 16]
                    };
                    cols / 256
                ];
                quants::quantize_row_q8_k(&x, &mut blocks);
                (
                    blocks.iter().map(|b| b.d).collect::<Vec<_>>(),
                    blocks.iter().flat_map(|b| b.qs).collect::<Vec<_>>(),
                )
            } else {
                let mut blocks = vec![
                    quants::BlockQ8_0 {
                        d: 0.0,
                        qs: [0; 32]
                    };
                    cols / 32
                ];
                quants::quantize_row_q8_0(&x, &mut blocks);
                (
                    blocks.iter().map(|b| b.d).collect::<Vec<_>>(),
                    blocks.iter().flat_map(|b| b.qs).collect::<Vec<_>>(),
                )
            };
            let mut output = Vec::new();
            let mut dedicated: Vec<f32> = Vec::new();
            for &row in &rows {
                let offset = gguf.tensor_data_offset + meta.offset as usize + row * row_bytes;
                let bytes = &mmap[offset..offset + row_bytes];
                let mut result = [0.0f32];
                match meta.dtype {
                    8 => quants::gemv_q8_0(&mut result, bytes, &x, 1, cols),
                    18 => quants::gemv_iq3_xxs(&mut result, bytes, &x, 1, cols),
                    20 => quants::gemv_iq4_nl(&mut result, bytes, &x, 1, cols),
                    _ => unreachable!(),
                }
                output.push(result[0]);
                #[cfg(target_arch = "x86_64")]
                if cols == 896
                    && matches!(meta.dtype, 8 | 20)
                    && is_x86_feature_detected!("avx2")
                    && is_x86_feature_detected!("fma")
                    && is_x86_feature_detected!("f16c")
                {
                    let mut blocks = vec![
                        quants::BlockQ8_0 {
                            d: 0.0,
                            qs: [0; 32]
                        };
                        cols / 32
                    ];
                    quants::quantize_row_q8_0(&x, &mut blocks);
                    let mut qs: [core::arch::x86_64::__m256i; 28] = unsafe { std::mem::zeroed() };
                    let mut scales = [0.0f32; 28];
                    for (i, b) in blocks.iter().enumerate() {
                        scales[i] = b.d;
                        qs[i] =
                            unsafe { core::arch::x86_64::_mm256_loadu_si256(b.qs.as_ptr().cast()) };
                    }
                    dedicated.push(unsafe {
                        if meta.dtype == 20 {
                            quants::vec_dot_iq4_nl_28blocks_avx2(bytes.as_ptr(), &qs, &scales)
                        } else {
                            quants::vec_dot_q8_0_28blocks_avx2(bytes.as_ptr(), &qs, &scales)
                        }
                    });
                }
            }
            fixtures.push(json!({"tensor":name, "dtype":meta.dtype, "columns":cols,
                "data_offset":gguf.tensor_data_offset + meta.offset as usize, "row_bytes":row_bytes,
                "variant":variant, "rows":rows, "input":x, "scales":scales,
                "quantized":quantized, "native":output, "dedicated":dedicated}));
        }
    }
    println!("{}", serde_json::to_string(&fixtures)?);
    Ok(())
}
