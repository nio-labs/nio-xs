#![allow(unsafe_op_in_unsafe_fn)]
use super::{GemvTask, worker_bounds};
use crate::quants::{self, BlockQ8_0, BlockQ8_K};
use std::slice;

unsafe fn matrix(
    out: usize,
    weight: usize,
    x: usize,
    rows: usize,
    cols: usize,
    iq3: bool,
    worker: usize,
    workers: usize,
) {
    let (start, end) = worker_bounds(rows, worker, workers);
    if start == end {
        return;
    }
    let row_bytes = if iq3 { cols / 256 * 98 } else { cols / 32 * 18 };
    let o = slice::from_raw_parts_mut((out as *mut f32).add(start), end - start);
    let w = slice::from_raw_parts(
        (weight as *const u8).add(start * row_bytes),
        (end - start) * row_bytes,
    );
    if iq3 {
        let b = slice::from_raw_parts(x as *const BlockQ8_K, cols / 256);
        quants::scalar::gemv_iq3_xxs_prequant(o, w, b, end - start, cols);
    } else {
        let b = slice::from_raw_parts(x as *const BlockQ8_0, cols / 32);
        quants::scalar::gemv_iq4_nl_prequant(o, w, b, end - start, cols);
    }
}
unsafe fn gate(
    out: usize,
    w1: usize,
    w3: usize,
    x: usize,
    rows: usize,
    cols: usize,
    iq3: bool,
    id: usize,
    workers: usize,
) {
    let (start, end) = worker_bounds(rows, id, workers);
    let bytes = if iq3 { cols / 256 * 98 } else { cols / 32 * 18 };
    for r in start..end {
        let a = slice::from_raw_parts((w1 as *const u8).add(r * bytes), bytes);
        let b = slice::from_raw_parts((w3 as *const u8).add(r * bytes), bytes);
        let (v1, v3) = if iq3 {
            let input = slice::from_raw_parts(x as *const BlockQ8_K, cols / 256);
            (
                quants::scalar::vec_dot_iq3_xxs_q8_k_avx2(a, input),
                quants::scalar::vec_dot_iq3_xxs_q8_k_avx2(b, input),
            )
        } else {
            let input = slice::from_raw_parts(x as *const BlockQ8_0, cols / 32);
            (
                quants::scalar::vec_dot_iq4_nl_q8_0_avx2(a, input),
                quants::scalar::vec_dot_iq4_nl_q8_0_avx2(b, input),
            )
        };
        *((out as *mut f32).add(r)) = quants::fast_silu(v1) * v3;
    }
}
unsafe fn vocab(
    weight: usize,
    x: usize,
    start: usize,
    count: usize,
    bytes: usize,
    suppress: bool,
    history: usize,
    history_len: usize,
    suppressed: usize,
    suppressed_len: usize,
    pen: f32,
    out_tok: usize,
    out_score: usize,
    iq4: bool,
    id: usize,
    workers: usize,
) {
    let (lo, hi) = worker_bounds(count, id, workers);
    let hist = if history_len == 0 {
        &[]
    } else {
        slice::from_raw_parts(history as *const u32, history_len)
    };
    let block = if suppressed_len == 0 {
        &[]
    } else {
        slice::from_raw_parts(suppressed as *const u32, suppressed_len)
    };
    let pen = pen.max(1.0);
    let n = if iq4 { bytes / 18 } else { bytes / 34 };
    let xs = slice::from_raw_parts(x as *const BlockQ8_0, n);
    let mut best = f32::NEG_INFINITY;
    let mut tok = 0;
    for i in lo..hi {
        let t = (start + i) as u32;
        if (suppress && matches!(t, 0 | 1 | 2 | 151643 | 151645))
            || (!iq4 && n == 18 && matches!(t, 3725 | 47182))
            || block.contains(&t)
        {
            continue;
        }
        let w = slice::from_raw_parts((weight as *const u8).add((start + i) * bytes), bytes);
        let mut score = if iq4 {
            quants::scalar::vec_dot_iq4_nl_q8_0_avx2(w, xs)
        } else {
            quants::scalar::vec_dot_q8_0_q8_0_avx2(w, xs)
        };
        score = crate::decoding::repetition_score(score, t, hist, pen);
        if score > best {
            best = score;
            tok = t;
        }
    }
    *((out_tok as *mut u32).add(id)) = tok;
    *((out_score as *mut f32).add(id)) = best;
}
impl GemvTask {
    pub unsafe fn execute_scalar(&self, id: usize, workers: usize) {
        use GemvTask::*;
        match *self {
            Iq4Nl {
                out_ptr,
                weight_ptr,
                x_blocks_ptr,
                n_rows,
                n_cols,
            } => matrix(
                out_ptr,
                weight_ptr,
                x_blocks_ptr,
                n_rows,
                n_cols,
                false,
                id,
                workers,
            ),
            Iq3Xxs {
                out_ptr,
                weight_ptr,
                x_q8_k_ptr,
                n_rows,
                n_cols,
            } => matrix(
                out_ptr, weight_ptr, x_q8_k_ptr, n_rows, n_cols, true, id, workers,
            ),
            Iq3Xxs2Tokens {
                out0_ptr,
                out1_ptr,
                weight_ptr,
                x0_q8_k_ptr,
                x1_q8_k_ptr,
                n_rows,
                n_cols,
            } => {
                for (o, x) in [(out0_ptr, x0_q8_k_ptr), (out1_ptr, x1_q8_k_ptr)] {
                    matrix(o, weight_ptr, x, n_rows, n_cols, true, id, workers)
                }
            }
            Iq3Xxs3Tokens {
                out0_ptr,
                out1_ptr,
                out2_ptr,
                weight_ptr,
                x0_q8_k_ptr,
                x1_q8_k_ptr,
                x2_q8_k_ptr,
                n_rows,
                n_cols,
            } => {
                for (o, x) in [
                    (out0_ptr, x0_q8_k_ptr),
                    (out1_ptr, x1_q8_k_ptr),
                    (out2_ptr, x2_q8_k_ptr),
                ] {
                    matrix(o, weight_ptr, x, n_rows, n_cols, true, id, workers)
                }
            }
            QkvIq4Nl {
                q_ptr,
                k_ptr,
                v_ptr,
                wq_ptr,
                wk_ptr,
                wv_ptr,
                x_blocks_ptr,
                q_rows,
                k_rows,
                v_rows,
                n_cols,
            } => {
                for (o, w, n) in [
                    (q_ptr, wq_ptr, q_rows),
                    (k_ptr, wk_ptr, k_rows),
                    (v_ptr, wv_ptr, v_rows),
                ] {
                    matrix(o, w, x_blocks_ptr, n, n_cols, false, id, workers)
                }
            }
            QkvIq3Xxs {
                q_ptr,
                k_ptr,
                v_ptr,
                wq_ptr,
                wk_ptr,
                wv_ptr,
                x_q8_k_ptr,
                q_rows,
                k_rows,
                v_rows,
                n_cols,
            } => {
                for (o, w, n) in [
                    (q_ptr, wq_ptr, q_rows),
                    (k_ptr, wk_ptr, k_rows),
                    (v_ptr, wv_ptr, v_rows),
                ] {
                    matrix(o, w, x_q8_k_ptr, n, n_cols, true, id, workers)
                }
            }
            QkvIq3Xxs2Tokens {
                q0_ptr,
                q1_ptr,
                k0_ptr,
                k1_ptr,
                v0_ptr,
                v1_ptr,
                wq_ptr,
                wk_ptr,
                wv_ptr,
                x0_q8_k_ptr,
                x1_q8_k_ptr,
                q_rows,
                k_rows,
                v_rows,
                n_cols,
            } => {
                for (q, k, v, x) in [
                    (q0_ptr, k0_ptr, v0_ptr, x0_q8_k_ptr),
                    (q1_ptr, k1_ptr, v1_ptr, x1_q8_k_ptr),
                ] {
                    for (o, w, n) in [
                        (q, wq_ptr, q_rows),
                        (k, wk_ptr, k_rows),
                        (v, wv_ptr, v_rows),
                    ] {
                        matrix(o, w, x, n, n_cols, true, id, workers)
                    }
                }
            }
            QkvIq3Xxs3Tokens {
                q0_ptr,
                q1_ptr,
                q2_ptr,
                k0_ptr,
                k1_ptr,
                k2_ptr,
                v0_ptr,
                v1_ptr,
                v2_ptr,
                wq_ptr,
                wk_ptr,
                wv_ptr,
                x0_q8_k_ptr,
                x1_q8_k_ptr,
                x2_q8_k_ptr,
                q_rows,
                k_rows,
                v_rows,
                n_cols,
            } => {
                for (q, k, v, x) in [
                    (q0_ptr, k0_ptr, v0_ptr, x0_q8_k_ptr),
                    (q1_ptr, k1_ptr, v1_ptr, x1_q8_k_ptr),
                    (q2_ptr, k2_ptr, v2_ptr, x2_q8_k_ptr),
                ] {
                    for (o, w, n) in [
                        (q, wq_ptr, q_rows),
                        (k, wk_ptr, k_rows),
                        (v, wv_ptr, v_rows),
                    ] {
                        matrix(o, w, x, n, n_cols, true, id, workers)
                    }
                }
            }
            GateUpIq4Nl {
                inter_out_ptr,
                w1_ptr,
                w3_ptr,
                x_blocks_ptr,
                n_rows,
                n_cols,
                ..
            } => gate(
                inter_out_ptr,
                w1_ptr,
                w3_ptr,
                x_blocks_ptr,
                n_rows,
                n_cols,
                false,
                id,
                workers,
            ),
            GateUpIq3Xxs {
                inter_out_ptr,
                w1_ptr,
                w3_ptr,
                x_q8_k_ptr,
                n_rows,
                n_cols,
            } => gate(
                inter_out_ptr,
                w1_ptr,
                w3_ptr,
                x_q8_k_ptr,
                n_rows,
                n_cols,
                true,
                id,
                workers,
            ),
            GateUpIq3Xxs2Tokens {
                inter0_out_ptr,
                inter1_out_ptr,
                w1_ptr,
                w3_ptr,
                x0_q8_k_ptr,
                x1_q8_k_ptr,
                n_rows,
                n_cols,
            } => {
                for (o, x) in [(inter0_out_ptr, x0_q8_k_ptr), (inter1_out_ptr, x1_q8_k_ptr)] {
                    gate(o, w1_ptr, w3_ptr, x, n_rows, n_cols, true, id, workers)
                }
            }
            GateUpIq3Xxs3Tokens {
                inter0_out_ptr,
                inter1_out_ptr,
                inter2_out_ptr,
                w1_ptr,
                w3_ptr,
                x0_q8_k_ptr,
                x1_q8_k_ptr,
                x2_q8_k_ptr,
                n_rows,
                n_cols,
            } => {
                for (o, x) in [
                    (inter0_out_ptr, x0_q8_k_ptr),
                    (inter1_out_ptr, x1_q8_k_ptr),
                    (inter2_out_ptr, x2_q8_k_ptr),
                ] {
                    gate(o, w1_ptr, w3_ptr, x, n_rows, n_cols, true, id, workers)
                }
            }
            Q8_0 {
                out_ptr,
                weight_ptr,
                x_norm_ptr,
                n_rows,
                n_cols,
            } => {
                let (lo, hi) = worker_bounds(n_rows, id, workers);
                if lo < hi {
                    let bytes = n_cols / 32 * 34;
                    let o = slice::from_raw_parts_mut((out_ptr as *mut f32).add(lo), hi - lo);
                    let w = slice::from_raw_parts(
                        (weight_ptr as *const u8).add(lo * bytes),
                        (hi - lo) * bytes,
                    );
                    let x = slice::from_raw_parts(x_norm_ptr as *const f32, n_cols);
                    quants::gemv_q8_0(o, w, x, hi - lo, n_cols)
                }
            }
            VocabQ8_0 {
                weight_ptr,
                x_blocks_ptr,
                start_tok,
                vocab_count,
                bytes_per_token,
                suppress_eos,
                history_ptr,
                history_len,
                suppressed_ptr,
                suppressed_len,
                repeat_penalty,
                top_tok_ptr,
                top_score_ptr,
            } => vocab(
                weight_ptr,
                x_blocks_ptr,
                start_tok,
                vocab_count,
                bytes_per_token,
                suppress_eos,
                history_ptr,
                history_len,
                suppressed_ptr,
                suppressed_len,
                repeat_penalty,
                top_tok_ptr,
                top_score_ptr,
                false,
                id,
                workers,
            ),
            VocabIq4Nl {
                weight_ptr,
                x_blocks_ptr,
                start_tok,
                vocab_count,
                bytes_per_token,
                suppress_eos,
                history_ptr,
                history_len,
                suppressed_ptr,
                suppressed_len,
                repeat_penalty,
                top_tok_ptr,
                top_score_ptr,
            } => vocab(
                weight_ptr,
                x_blocks_ptr,
                start_tok,
                vocab_count,
                bytes_per_token,
                suppress_eos,
                history_ptr,
                history_len,
                suppressed_ptr,
                suppressed_len,
                repeat_penalty,
                top_tok_ptr,
                top_score_ptr,
                true,
                id,
                workers,
            ),
            Attention {
                q_ptr,
                k_cache_ptr,
                v_cache_ptr,
                out_ptr,
                seq_len,
                n_head,
                n_kv_head,
                head_dim,
            } => {
                let (lo, hi) = worker_bounds(n_head, id, workers);
                let q = slice::from_raw_parts(q_ptr as *const f32, n_head * head_dim);
                let k = slice::from_raw_parts(
                    k_cache_ptr as *const f32,
                    seq_len * n_kv_head * head_dim,
                );
                let v = slice::from_raw_parts(
                    v_cache_ptr as *const f32,
                    seq_len * n_kv_head * head_dim,
                );
                let out = slice::from_raw_parts_mut(out_ptr as *mut f32, n_head * head_dim);
                let stride = n_kv_head * head_dim;
                let mut scores = vec![0.0f32; seq_len];
                for h in lo..hi {
                    let kh = h / (n_head / n_kv_head);
                    let mut max = f32::NEG_INFINITY;
                    for t in 0..seq_len {
                        let off = t * stride + kh * head_dim;
                        let s = q[h * head_dim..(h + 1) * head_dim]
                            .iter()
                            .zip(&k[off..off + head_dim])
                            .map(|(a, b)| a * b)
                            .sum::<f32>()
                            / (head_dim as f32).sqrt();
                        scores[t] = s;
                        max = max.max(s)
                    }
                    let mut sum = 0.0;
                    for s in &mut scores {
                        *s = (*s - max).exp();
                        sum += *s
                    }
                    for d in 0..head_dim {
                        out[h * head_dim + d] = (0..seq_len)
                            .map(|t| scores[t] * v[t * stride + kh * head_dim + d])
                            .sum::<f32>()
                            / sum;
                    }
                }
            }
            GemvTask::GenericLinear {
                out_ptr,
                weight_ptr,
                x_f32_ptr,
                x_q8_k_ptr,
                n_rows,
                n_cols,
                dtype,
            } => {
                let (lo, hi) = worker_bounds(n_rows, id, workers);
                if lo >= hi {
                    return;
                }
                let (block_size, type_size) = match dtype {
                    0 => (1, 4),
                    2 => (32, 18),
                    8 => (32, 34),
                    12 => (256, 144),
                    13 => (256, 176),
                    14 => (256, 210),
                    18 => (256, 98),
                    20 => (32, 18),
                    21 => (256, 110),
                    22 => (256, 82),
                    other => panic!("unsupported dtype {other}"),
                };
                let row_bytes = (n_cols / block_size) * type_size;
                let x_f32 = slice::from_raw_parts(x_f32_ptr as *const f32, n_cols);
                let out = slice::from_raw_parts_mut(out_ptr as *mut f32, n_rows);
                match dtype {
                    12 => {
                        use candle_core::quantized::k_quants::{BlockQ4K, BlockQ8K, GgmlType};
                        let ys = slice::from_raw_parts(x_q8_k_ptr as *const BlockQ8K, n_cols / 256);
                        for r in lo..hi {
                            let r_ptr = (weight_ptr as *const u8).add(r * row_bytes);
                            let xs = slice::from_raw_parts(r_ptr as *const BlockQ4K, n_cols / 256);
                            out[r] = BlockQ4K::vec_dot(n_cols, xs, ys).unwrap_or(0.0);
                        }
                    }
                    13 => {
                        use candle_core::quantized::k_quants::{BlockQ5K, BlockQ8K, GgmlType};
                        let ys = slice::from_raw_parts(x_q8_k_ptr as *const BlockQ8K, n_cols / 256);
                        for r in lo..hi {
                            let r_ptr = (weight_ptr as *const u8).add(r * row_bytes);
                            let xs = slice::from_raw_parts(r_ptr as *const BlockQ5K, n_cols / 256);
                            out[r] = BlockQ5K::vec_dot(n_cols, xs, ys).unwrap_or(0.0);
                        }
                    }
                    14 => {
                        use candle_core::quantized::k_quants::{BlockQ6K, BlockQ8K, GgmlType};
                        let ys = slice::from_raw_parts(x_q8_k_ptr as *const BlockQ8K, n_cols / 256);
                        for r in lo..hi {
                            let r_ptr = (weight_ptr as *const u8).add(r * row_bytes);
                            let xs = slice::from_raw_parts(r_ptr as *const BlockQ6K, n_cols / 256);
                            out[r] = BlockQ6K::vec_dot(n_cols, xs, ys).unwrap_or(0.0);
                        }
                    }
                    18 => {
                        let x_q8_k = slice::from_raw_parts(
                            x_q8_k_ptr as *const quants::BlockQ8_K,
                            n_cols / 256,
                        );
                        for r in lo..hi {
                            let r_ptr = (weight_ptr as *const u8).add(r * row_bytes);
                            let row_slice = slice::from_raw_parts(r_ptr, row_bytes);
                            out[r] = quants::scalar::vec_dot_iq3_xxs_q8_k_avx2(row_slice, x_q8_k);
                        }
                    }
                    21 => {
                        let mut dequant_buf = vec![0.0f32; n_cols];
                        for r in lo..hi {
                            let r_ptr = (weight_ptr as *const u8).add(r * row_bytes);
                            let row_slice = slice::from_raw_parts(r_ptr, row_bytes);
                            quants::dequantize_row_iq3_s(row_slice, &mut dequant_buf);
                            out[r] = quants::dot_f32(&dequant_buf, x_f32);
                        }
                    }
                    22 => {
                        let mut dequant_buf = vec![0.0f32; n_cols];
                        for r in lo..hi {
                            let r_ptr = (weight_ptr as *const u8).add(r * row_bytes);
                            let row_slice = slice::from_raw_parts(r_ptr, row_bytes);
                            quants::dequantize_row_iq2_s(row_slice, &mut dequant_buf);
                            out[r] = quants::dot_f32(&dequant_buf, x_f32);
                        }
                    }
                    2 => {
                        let mut dequant_buf = vec![0.0f32; n_cols];
                        for r in lo..hi {
                            let r_ptr = (weight_ptr as *const u8).add(r * row_bytes);
                            let row_slice = slice::from_raw_parts(r_ptr, row_bytes);
                            quants::dequantize_row_q4_0(row_slice, &mut dequant_buf);
                            out[r] = quants::dot_f32(&dequant_buf, x_f32);
                        }
                    }
                    _ => panic!("unhandled generic linear dtype {dtype}"),
                }
            }
            GemvTask::GenericLinear2Tokens {
                out0_ptr,
                out1_ptr,
                weight_ptr,
                x0_f32_ptr,
                x1_f32_ptr,
                x0_q8_k_ptr,
                x1_q8_k_ptr,
                n_rows,
                n_cols,
                dtype,
            } => {
                Self::GenericLinear {
                    out_ptr: out0_ptr,
                    weight_ptr,
                    x_f32_ptr: x0_f32_ptr,
                    x_q8_k_ptr: x0_q8_k_ptr,
                    n_rows,
                    n_cols,
                    dtype,
                }
                .execute_scalar(id, workers);
                Self::GenericLinear {
                    out_ptr: out1_ptr,
                    weight_ptr,
                    x_f32_ptr: x1_f32_ptr,
                    x_q8_k_ptr: x1_q8_k_ptr,
                    n_rows,
                    n_cols,
                    dtype,
                }
                .execute_scalar(id, workers);
            }
            GemvTask::GenericLinear3Tokens {
                out0_ptr,
                out1_ptr,
                out2_ptr,
                weight_ptr,
                x0_f32_ptr,
                x1_f32_ptr,
                x2_f32_ptr,
                x0_q8_k_ptr,
                x1_q8_k_ptr,
                x2_q8_k_ptr,
                n_rows,
                n_cols,
                dtype,
            } => {
                Self::GenericLinear {
                    out_ptr: out0_ptr,
                    weight_ptr,
                    x_f32_ptr: x0_f32_ptr,
                    x_q8_k_ptr: x0_q8_k_ptr,
                    n_rows,
                    n_cols,
                    dtype,
                }
                .execute_scalar(id, workers);
                Self::GenericLinear {
                    out_ptr: out1_ptr,
                    weight_ptr,
                    x_f32_ptr: x1_f32_ptr,
                    x_q8_k_ptr: x1_q8_k_ptr,
                    n_rows,
                    n_cols,
                    dtype,
                }
                .execute_scalar(id, workers);
                Self::GenericLinear {
                    out_ptr: out2_ptr,
                    weight_ptr,
                    x_f32_ptr: x2_f32_ptr,
                    x_q8_k_ptr: x2_q8_k_ptr,
                    n_rows,
                    n_cols,
                    dtype,
                }
                .execute_scalar(id, workers);
            }
            GemvTask::VocabQ5K {
                weight_ptr,
                x_q8_k_ptr,
                n_cols,
                start_tok,
                vocab_count,
                bytes_per_token,
                suppress_eos,
                history_ptr,
                history_len,
                suppressed_ptr,
                suppressed_len,
                repeat_penalty,
                top_tok_ptr,
                top_score_ptr,
            } => {
                let (w0, w1) = worker_bounds(vocab_count, id, workers);
                let (lo, hi) = (start_tok + w0, start_tok + w1);
                if lo >= hi {
                    return;
                }
                use candle_core::quantized::k_quants::{BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8K, GgmlType};
                let nb = n_cols / 256;
                let ys = slice::from_raw_parts(x_q8_k_ptr as *const BlockQ8K, nb);
                let hist = slice::from_raw_parts(history_ptr as *const u32, history_len);
                let sup = slice::from_raw_parts(suppressed_ptr as *const u32, suppressed_len);
                let mut best_s = f32::NEG_INFINITY;
                let mut best_t = lo as u32;
                for tid in (lo as u32)..(hi as u32) {
                    if (suppress_eos
                        && (tid == 0 || tid == 1 || tid == 2 || tid == 151645 || tid == 151643))
                        || (!sup.is_empty() && sup.contains(&tid))
                    {
                        continue;
                    }
                    let r_ptr = (weight_ptr as *const u8).add(tid as usize * bytes_per_token);
                    let val = match bytes_per_token / nb {
                        144 => {
                            let xs = slice::from_raw_parts(r_ptr as *const BlockQ4K, nb);
                            BlockQ4K::vec_dot(n_cols, xs, ys).unwrap_or(f32::NEG_INFINITY)
                        }
                        210 => {
                            let xs = slice::from_raw_parts(r_ptr as *const BlockQ6K, nb);
                            BlockQ6K::vec_dot(n_cols, xs, ys).unwrap_or(f32::NEG_INFINITY)
                        }
                        _ => {
                            let xs = slice::from_raw_parts(r_ptr as *const BlockQ5K, nb);
                            BlockQ5K::vec_dot(n_cols, xs, ys).unwrap_or(f32::NEG_INFINITY)
                        }
                    };
                    if val > best_s {
                        let s = crate::decoding::repetition_score(val, tid, hist, repeat_penalty);
                        if s > best_s {
                            best_s = s;
                            best_t = tid;
                        }
                    }
                }
                slice::from_raw_parts_mut(top_tok_ptr as *mut u32, workers)[id] = best_t;
                slice::from_raw_parts_mut(top_score_ptr as *mut f32, workers)[id] = best_s;
            }
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
impl GemvTask {
    pub unsafe fn execute(&self, id: usize, workers: usize) {
        unsafe { self.execute_scalar(id, workers) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn attention_handles_long_context() {
        let n = 8192;
        let q = vec![0.0f32; 8];
        let k = vec![0.0f32; n * 8];
        let v = vec![2.0f32; n * 8];
        let mut out = vec![0.0f32; 8];
        let task = GemvTask::Attention {
            q_ptr: q.as_ptr() as usize,
            k_cache_ptr: k.as_ptr() as usize,
            v_cache_ptr: v.as_ptr() as usize,
            out_ptr: out.as_mut_ptr() as usize,
            seq_len: n,
            n_head: 1,
            n_kv_head: 1,
            head_dim: 8,
        };
        unsafe { task.execute_scalar(0, 1) };
        assert!(out.iter().all(|x| (x - 2.0).abs() < 0.001));
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod parity_tests {
    use super::*;
    #[test]
    fn scalar_matrix_and_vocab_match_x86() {
        if !std::is_x86_feature_detected!("avx2")
            || !std::is_x86_feature_detected!("fma")
            || !std::is_x86_feature_detected!("f16c")
        {
            return;
        }
        let mut weights = vec![0u8; 3 * 98];
        for row in weights.chunks_exact_mut(98) {
            row[..2].copy_from_slice(&half::f16::from_f32(0.125).to_le_bytes());
            for (i, b) in row[2..].iter_mut().enumerate() {
                *b = (i.wrapping_mul(31)) as u8;
            }
        }
        let input = [BlockQ8_K {
            d: 0.01,
            qs: std::array::from_fn(|i| (i as i8).wrapping_sub(64)),
            bsums: [0; 16],
        }];
        let mut a = vec![0.0f32; 3];
        let mut b = a.clone();
        let make = |out: usize| GemvTask::Iq3Xxs {
            out_ptr: out,
            weight_ptr: weights.as_ptr() as usize,
            x_q8_k_ptr: input.as_ptr() as usize,
            n_rows: 3,
            n_cols: 256,
        };
        unsafe {
            make(a.as_mut_ptr() as usize).execute_scalar(0, 1);
            make(b.as_mut_ptr() as usize).execute(0, 1)
        }
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 0.001 * y.abs().max(1.0));
        }
        let mut vocab_weights = vec![0u8; 12 * 34];
        for (i, row) in vocab_weights.chunks_exact_mut(34).enumerate() {
            row[..2].copy_from_slice(&half::f16::from_f32(0.1).to_le_bytes());
            row[2..].fill(i as u8);
        }
        let x8 = [BlockQ8_0 {
            d: 1.0,
            qs: [1; 32],
        }];
        let mut at = [0];
        let mut bt = [0];
        let mut av = [0.0f32];
        let mut bv = [0.0f32];
        let empty: Vec<u32> = Vec::new();
        let make_vocab = |tok: usize, score: usize| GemvTask::VocabQ8_0 {
            weight_ptr: vocab_weights.as_ptr() as usize,
            x_blocks_ptr: x8.as_ptr() as usize,
            start_tok: 3,
            vocab_count: 8,
            bytes_per_token: 34,
            suppress_eos: false,
            history_ptr: empty.as_ptr() as usize,
            history_len: 0,
            suppressed_ptr: empty.as_ptr() as usize,
            suppressed_len: 0,
            repeat_penalty: 1.1,
            top_tok_ptr: tok,
            top_score_ptr: score,
        };
        unsafe {
            make_vocab(at.as_mut_ptr() as usize, av.as_mut_ptr() as usize).execute_scalar(0, 1);
            make_vocab(bt.as_mut_ptr() as usize, bv.as_mut_ptr() as usize).execute(0, 1)
        }
        assert_eq!(at, bt);
        assert!((av[0] - bv[0]).abs() < 0.001);
    }
}
