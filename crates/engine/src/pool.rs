#![allow(unsafe_op_in_unsafe_fn)]

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use crate::quants::{self, BlockQ8_0, BlockQ8_K};

pub const NUM_WORKERS: usize = 4;

#[derive(Clone, Copy)]
pub enum GemvTask {
    Iq4Nl {
        out_ptr: usize,
        weight_ptr: usize,
        x_blocks_ptr: usize,
        n_rows: usize,
        n_cols: usize,
    },
    QkvIq4Nl {
        q_ptr: usize,
        k_ptr: usize,
        v_ptr: usize,
        wq_ptr: usize,
        wk_ptr: usize,
        wv_ptr: usize,
        x_blocks_ptr: usize,
        q_rows: usize,
        k_rows: usize,
        v_rows: usize,
        n_cols: usize,
    },
    QkvIq3Xxs {
        q_ptr: usize,
        k_ptr: usize,
        v_ptr: usize,
        wq_ptr: usize,
        wk_ptr: usize,
        wv_ptr: usize,
        x_q8_k_ptr: usize,
        q_rows: usize,
        k_rows: usize,
        v_rows: usize,
        n_cols: usize,
    },
    GateUpIq4Nl {
        w1_out_ptr: usize,
        w3_out_ptr: usize,
        inter_out_ptr: usize,
        w1_ptr: usize,
        w3_ptr: usize,
        x_blocks_ptr: usize,
        n_rows: usize,
        n_cols: usize,
    },
    GateUpIq3Xxs {
        inter_out_ptr: usize,
        w1_ptr: usize,
        w3_ptr: usize,
        x_q8_k_ptr: usize,
        n_rows: usize,
        n_cols: usize,
    },
    GateUpIq3Xxs2Tokens {
        inter0_out_ptr: usize,
        inter1_out_ptr: usize,
        w1_ptr: usize,
        w3_ptr: usize,
        x0_q8_k_ptr: usize,
        x1_q8_k_ptr: usize,
        n_rows: usize,
        n_cols: usize,
    },
    Iq3Xxs {
        out_ptr: usize,
        weight_ptr: usize,
        x_q8_k_ptr: usize,
        n_rows: usize,
        n_cols: usize,
    },
    Iq3Xxs2Tokens {
        out0_ptr: usize,
        out1_ptr: usize,
        weight_ptr: usize,
        x0_q8_k_ptr: usize,
        x1_q8_k_ptr: usize,
        n_rows: usize,
        n_cols: usize,
    },
    QkvIq3Xxs2Tokens {
        q0_ptr: usize,
        q1_ptr: usize,
        k0_ptr: usize,
        k1_ptr: usize,
        v0_ptr: usize,
        v1_ptr: usize,
        wq_ptr: usize,
        wk_ptr: usize,
        wv_ptr: usize,
        x0_q8_k_ptr: usize,
        x1_q8_k_ptr: usize,
        q_rows: usize,
        k_rows: usize,
        v_rows: usize,
        n_cols: usize,
    },
    Q8_0 {
        out_ptr: usize,
        weight_ptr: usize,
        x_norm_ptr: usize,
        n_rows: usize,
        n_cols: usize,
    },
    VocabQ8_0 {
        weight_ptr: usize,
        x_blocks_ptr: usize,
        start_tok: usize,
        vocab_count: usize,
        bytes_per_token: usize,
        suppress_eos: bool,
        history_ptr: usize,
        history_len: usize,
        suppressed_ptr: usize,
        suppressed_len: usize,
        repeat_penalty: f32,
        top_tok_ptr: usize,
        top_score_ptr: usize,
    },
    VocabIq4Nl {
        weight_ptr: usize,
        x_blocks_ptr: usize,
        start_tok: usize,
        vocab_count: usize,
        bytes_per_token: usize,
        suppress_eos: bool,
        history_ptr: usize,
        history_len: usize,
        suppressed_ptr: usize,
        suppressed_len: usize,
        repeat_penalty: f32,
        top_tok_ptr: usize,
        top_score_ptr: usize,
    },
    Attention {
        q_ptr: usize,
        k_cache_ptr: usize,
        v_cache_ptr: usize,
        out_ptr: usize,
        seq_len: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
    },
}

impl GemvTask {
    #[inline(always)]
    pub unsafe fn execute(&self, worker_id: usize, num_workers: usize) {
        match *self {
            GemvTask::Iq4Nl { out_ptr, weight_ptr, x_blocks_ptr, n_rows, n_cols } => {
                let chunk = (n_rows + num_workers - 1) / num_workers;
                let start_r = (worker_id * chunk).min(n_rows);
                let end_r = ((worker_id + 1) * chunk).min(n_rows);
                let rows = end_r - start_r;
                if rows == 0 {
                    return;
                }
                let row_bytes = (n_cols / 32) * 18;
                let out_slice = std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start_r), rows);
                let weight_slice = std::slice::from_raw_parts((weight_ptr as *const u8).add(start_r * row_bytes), rows * row_bytes);
                let x_blocks = std::slice::from_raw_parts(x_blocks_ptr as *const BlockQ8_0, n_cols / 32);
                quants::gemv_iq4_nl_prequant(out_slice, weight_slice, x_blocks, rows, n_cols);
            }
            GemvTask::QkvIq4Nl {
                q_ptr, k_ptr, v_ptr,
                wq_ptr, wk_ptr, wv_ptr,
                x_blocks_ptr,
                q_rows, k_rows, v_rows,
                n_cols,
            } => {
                let row_bytes = (n_cols / 32) * 18;
                let x_blocks = std::slice::from_raw_parts(x_blocks_ptr as *const BlockQ8_0, n_cols / 32);

                // Q partition
                let q_chunk = (q_rows + num_workers - 1) / num_workers;
                let q_start = (worker_id * q_chunk).min(q_rows);
                let q_end = ((worker_id + 1) * q_chunk).min(q_rows);
                if q_end > q_start {
                    let out = std::slice::from_raw_parts_mut((q_ptr as *mut f32).add(q_start), q_end - q_start);
                    let weights = std::slice::from_raw_parts((wq_ptr as *const u8).add(q_start * row_bytes), (q_end - q_start) * row_bytes);
                    quants::gemv_iq4_nl_prequant(out, weights, x_blocks, q_end - q_start, n_cols);
                }

                // K partition
                let k_chunk = (k_rows + num_workers - 1) / num_workers;
                let k_start = (worker_id * k_chunk).min(k_rows);
                let k_end = ((worker_id + 1) * k_chunk).min(k_rows);
                if k_end > k_start {
                    let out = std::slice::from_raw_parts_mut((k_ptr as *mut f32).add(k_start), k_end - k_start);
                    let weights = std::slice::from_raw_parts((wk_ptr as *const u8).add(k_start * row_bytes), (k_end - k_start) * row_bytes);
                    quants::gemv_iq4_nl_prequant(out, weights, x_blocks, k_end - k_start, n_cols);
                }

                // V partition
                let v_chunk = (v_rows + num_workers - 1) / num_workers;
                let v_start = (worker_id * v_chunk).min(v_rows);
                let v_end = ((worker_id + 1) * v_chunk).min(v_rows);
                if v_end > v_start {
                    let out = std::slice::from_raw_parts_mut((v_ptr as *mut f32).add(v_start), v_end - v_start);
                    let weights = std::slice::from_raw_parts((wv_ptr as *const u8).add(v_start * row_bytes), (v_end - v_start) * row_bytes);
                    quants::gemv_iq4_nl_prequant(out, weights, x_blocks, v_end - v_start, n_cols);
                }
            }
            GemvTask::QkvIq3Xxs {
                q_ptr, k_ptr, v_ptr,
                wq_ptr, wk_ptr, wv_ptr,
                x_q8_k_ptr,
                q_rows, k_rows, v_rows,
                n_cols,
            } => {
                let row_bytes = (n_cols / 256) * 98;
                let x_q8_k = std::slice::from_raw_parts(x_q8_k_ptr as *const BlockQ8_K, (n_cols + 255) / 256);

                // Q partition
                let q_chunk = (q_rows + num_workers - 1) / num_workers;
                let q_start = (worker_id * q_chunk).min(q_rows);
                let q_end = ((worker_id + 1) * q_chunk).min(q_rows);
                if q_end > q_start {
                    let out = std::slice::from_raw_parts_mut((q_ptr as *mut f32).add(q_start), q_end - q_start);
                    let weights = std::slice::from_raw_parts((wq_ptr as *const u8).add(q_start * row_bytes), (q_end - q_start) * row_bytes);
                    quants::gemv_iq3_xxs_prequant(out, weights, x_q8_k, q_end - q_start, n_cols);
                }

                // K partition
                let k_chunk = (k_rows + num_workers - 1) / num_workers;
                let k_start = (worker_id * k_chunk).min(k_rows);
                let k_end = ((worker_id + 1) * k_chunk).min(k_rows);
                if k_end > k_start {
                    let out = std::slice::from_raw_parts_mut((k_ptr as *mut f32).add(k_start), k_end - k_start);
                    let weights = std::slice::from_raw_parts((wk_ptr as *const u8).add(k_start * row_bytes), (k_end - k_start) * row_bytes);
                    quants::gemv_iq3_xxs_prequant(out, weights, x_q8_k, k_end - k_start, n_cols);
                }

                // V partition
                let v_chunk = (v_rows + num_workers - 1) / num_workers;
                let v_start = (worker_id * v_chunk).min(v_rows);
                let v_end = ((worker_id + 1) * v_chunk).min(v_rows);
                if v_end > v_start {
                    let out = std::slice::from_raw_parts_mut((v_ptr as *mut f32).add(v_start), v_end - v_start);
                    let weights = std::slice::from_raw_parts((wv_ptr as *const u8).add(v_start * row_bytes), (v_end - v_start) * row_bytes);
                    quants::gemv_iq3_xxs_prequant(out, weights, x_q8_k, v_end - v_start, n_cols);
                }
            }
            GemvTask::QkvIq3Xxs2Tokens {
                q0_ptr, q1_ptr,
                k0_ptr, k1_ptr,
                v0_ptr, v1_ptr,
                wq_ptr, wk_ptr, wv_ptr,
                x0_q8_k_ptr, x1_q8_k_ptr,
                q_rows, k_rows, v_rows,
                n_cols,
            } => {
                let nb = (n_cols + 255) / 256;
                let row_bytes = nb * 98;
                let x0_q8_k = x0_q8_k_ptr as *const BlockQ8_K;
                let x1_q8_k = x1_q8_k_ptr as *const BlockQ8_K;

                // Q partition
                let q_chunk = (q_rows + num_workers - 1) / num_workers;
                let q_start = (worker_id * q_chunk).min(q_rows);
                let q_end = ((worker_id + 1) * q_chunk).min(q_rows);
                if q_end > q_start {
                    let q0_base = (q0_ptr as *mut f32).add(q_start);
                    let q1_base = (q1_ptr as *mut f32).add(q_start);
                    for j in 0..(q_end - q_start) {
                        let r = q_start + j;
                        let w_ptr = (wq_ptr as *const u8).add(r * row_bytes);
                        let (v0, v1) = quants::vec_dot_iq3_xxs_q8_k_2tokens_avx2_ptr(w_ptr, x0_q8_k, x1_q8_k, nb);
                        *q0_base.add(j) = v0;
                        *q1_base.add(j) = v1;
                    }
                }

                // K partition
                let k_chunk = (k_rows + num_workers - 1) / num_workers;
                let k_start = (worker_id * k_chunk).min(k_rows);
                let k_end = ((worker_id + 1) * k_chunk).min(k_rows);
                if k_end > k_start {
                    let k0_base = (k0_ptr as *mut f32).add(k_start);
                    let k1_base = (k1_ptr as *mut f32).add(k_start);
                    for j in 0..(k_end - k_start) {
                        let r = k_start + j;
                        let w_ptr = (wk_ptr as *const u8).add(r * row_bytes);
                        let (v0, v1) = quants::vec_dot_iq3_xxs_q8_k_2tokens_avx2_ptr(w_ptr, x0_q8_k, x1_q8_k, nb);
                        *k0_base.add(j) = v0;
                        *k1_base.add(j) = v1;
                    }
                }

                // V partition
                let v_chunk = (v_rows + num_workers - 1) / num_workers;
                let v_start = (worker_id * v_chunk).min(v_rows);
                let v_end = ((worker_id + 1) * v_chunk).min(v_rows);
                if v_end > v_start {
                    let v0_base = (v0_ptr as *mut f32).add(v_start);
                    let v1_base = (v1_ptr as *mut f32).add(v_start);
                    for j in 0..(v_end - v_start) {
                        let r = v_start + j;
                        let w_ptr = (wv_ptr as *const u8).add(r * row_bytes);
                        let (v0, v1) = quants::vec_dot_iq3_xxs_q8_k_2tokens_avx2_ptr(w_ptr, x0_q8_k, x1_q8_k, nb);
                        *v0_base.add(j) = v0;
                        *v1_base.add(j) = v1;
                    }
                }
            }
            GemvTask::GateUpIq4Nl {
                inter_out_ptr,
                w1_ptr, w3_ptr,
                x_blocks_ptr,
                n_rows, n_cols,
                ..
            } => {
                let row_bytes = (n_cols / 32) * 18;
                let x_blocks = std::slice::from_raw_parts(x_blocks_ptr as *const BlockQ8_0, n_cols / 32);
                let chunk = (n_rows + num_workers - 1) / num_workers;
                let start_r = (worker_id * chunk).min(n_rows);
                let end_r = ((worker_id + 1) * chunk).min(n_rows);
                let rows = end_r - start_r;
                if rows > 0 {
                    let weights1_ptr = (w1_ptr as *const u8).add(start_r * row_bytes);
                    let weights3_ptr = (w3_ptr as *const u8).add(start_r * row_bytes);
                    let inter = std::slice::from_raw_parts_mut((inter_out_ptr as *mut f32).add(start_r), rows);

                    let nb = n_cols / 32;
                    if nb == 48 {
                        let mut x_qs = [core::arch::x86_64::_mm256_setzero_si256(); 48];
                        let mut x_d = [0.0f32; 48];
                        for i in 0..48 {
                            x_qs[i] = core::arch::x86_64::_mm256_loadu_si256(x_blocks[i].qs.as_ptr() as *const core::arch::x86_64::__m256i);
                            x_d[i] = x_blocks[i].d;
                        }
                        let mut j = 0;
                        while j + 1 < rows {
                            let w1_r0 = weights1_ptr.add(j * 864);
                            let w1_r1 = weights1_ptr.add((j + 1) * 864);
                            let w3_r0 = weights3_ptr.add(j * 864);
                            let w3_r1 = weights3_ptr.add((j + 1) * 864);
                            let (w1_0, w1_1) = quants::vec_dot_iq4_nl_48blocks_2rows_avx2(w1_r0, w1_r1, &x_qs, &x_d);
                            let (w3_0, w3_1) = quants::vec_dot_iq4_nl_48blocks_2rows_avx2(w3_r0, w3_r1, &x_qs, &x_d);
                            inter[j] = quants::fast_silu(w1_0) * w3_0;
                            inter[j + 1] = quants::fast_silu(w1_1) * w3_1;
                            j += 2;
                        }
                        if j < rows {
                            let w1_r0 = weights1_ptr.add(j * 864);
                            let w3_r0 = weights3_ptr.add(j * 864);
                            let w1_0 = quants::vec_dot_iq4_nl_48blocks_avx2(w1_r0, &x_qs, &x_d);
                            let w3_0 = quants::vec_dot_iq4_nl_48blocks_avx2(w3_r0, &x_qs, &x_d);
                            inter[j] = quants::fast_silu(w1_0) * w3_0;
                        }
                    } else if nb == 28 {
                        let mut x_qs = [core::arch::x86_64::_mm256_setzero_si256(); 28];
                        let mut x_d = [0.0f32; 28];
                        for i in 0..28 {
                            x_qs[i] = core::arch::x86_64::_mm256_loadu_si256(x_blocks[i].qs.as_ptr() as *const core::arch::x86_64::__m256i);
                            x_d[i] = x_blocks[i].d;
                        }
                        let mut j = 0;
                        while j + 1 < rows {
                            let w1_r0 = weights1_ptr.add(j * 504);
                            let w1_r1 = weights1_ptr.add((j + 1) * 504);
                            let w3_r0 = weights3_ptr.add(j * 504);
                            let w3_r1 = weights3_ptr.add((j + 1) * 504);
                            let (w1_0, w1_1) = quants::vec_dot_iq4_nl_28blocks_2rows_avx2(w1_r0, w1_r1, &x_qs, &x_d);
                            let (w3_0, w3_1) = quants::vec_dot_iq4_nl_28blocks_2rows_avx2(w3_r0, w3_r1, &x_qs, &x_d);
                            inter[j] = quants::fast_silu(w1_0) * w3_0;
                            inter[j + 1] = quants::fast_silu(w1_1) * w3_1;
                            j += 2;
                        }
                        if j < rows {
                            let w1_r0 = weights1_ptr.add(j * 504);
                            let w3_r0 = weights3_ptr.add(j * 504);
                            let w1_0 = quants::vec_dot_iq4_nl_28blocks_avx2(w1_r0, &x_qs, &x_d);
                            let w3_0 = quants::vec_dot_iq4_nl_28blocks_avx2(w3_r0, &x_qs, &x_d);
                            inter[j] = quants::fast_silu(w1_0) * w3_0;
                        }
                    } else {
                        for j in 0..rows {
                            let w1_row_ptr = weights1_ptr.add(j * row_bytes);
                            let w3_row_ptr = weights3_ptr.add(j * row_bytes);
                            if j + 1 < rows {
                                let next_w1 = weights1_ptr.add((j + 1) * row_bytes);
                                let next_w3 = weights3_ptr.add((j + 1) * row_bytes);
                                core::arch::x86_64::_mm_prefetch(next_w1 as *const i8, core::arch::x86_64::_MM_HINT_T0);
                                core::arch::x86_64::_mm_prefetch(next_w1.add(64) as *const i8, core::arch::x86_64::_MM_HINT_T0);
                                core::arch::x86_64::_mm_prefetch(next_w3 as *const i8, core::arch::x86_64::_MM_HINT_T0);
                                core::arch::x86_64::_mm_prefetch(next_w3.add(64) as *const i8, core::arch::x86_64::_MM_HINT_T0);
                            }
                            inter[j] = quants::gate_up_iq4_nl_silu_avx2(w1_row_ptr, w3_row_ptr, x_blocks, nb);
                        }
                    }
                }
            }
            GemvTask::VocabQ8_0 {
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
            } => {
                let chunk = (vocab_count + num_workers - 1) / num_workers;
                let worker_start = (start_tok + worker_id * chunk).min(start_tok + vocab_count);
                let worker_end = (start_tok + (worker_id + 1) * chunk).min(start_tok + vocab_count);
                if worker_start >= worker_end {
                    return;
                }
                let num_blocks = bytes_per_token / 34;
                let weight_bytes = std::slice::from_raw_parts(weight_ptr as *const u8, (start_tok + vocab_count) * bytes_per_token);
                let x_blocks = std::slice::from_raw_parts(x_blocks_ptr as *const BlockQ8_0, num_blocks);
                let history = std::slice::from_raw_parts(history_ptr as *const u32, history_len);
                let suppressed = if suppressed_len > 0 {
                    std::slice::from_raw_parts(suppressed_ptr as *const u32, suppressed_len)
                } else {
                    &[]
                };

                let is_skate = num_blocks == 18;
                let is_manta = num_blocks == 28;
                let is_beluga = num_blocks == 48;

                // Pin activation vectors in L1 registers/stack for Skate (18 blocks), Manta (28 blocks), Beluga (48 blocks)
                let mut x_qs_skate = [core::arch::x86_64::_mm256_setzero_si256(); 18];
                let mut x_d_skate = [0.0f32; 18];
                let mut x_qs_manta = [core::arch::x86_64::_mm256_setzero_si256(); 28];
                let mut x_d_manta = [0.0f32; 28];
                let mut x_qs_beluga = [core::arch::x86_64::_mm256_setzero_si256(); 48];
                let mut x_d_beluga = [0.0f32; 48];
                if is_skate {
                    for i in 0..18 {
                        x_qs_skate[i] = core::arch::x86_64::_mm256_loadu_si256(x_blocks[i].qs.as_ptr() as *const core::arch::x86_64::__m256i);
                        x_d_skate[i] = x_blocks[i].d;
                    }
                } else if is_manta {
                    for i in 0..28 {
                        x_qs_manta[i] = core::arch::x86_64::_mm256_loadu_si256(x_blocks[i].qs.as_ptr() as *const core::arch::x86_64::__m256i);
                        x_d_manta[i] = x_blocks[i].d;
                    }
                } else if is_beluga {
                    for i in 0..48 {
                        x_qs_beluga[i] = core::arch::x86_64::_mm256_loadu_si256(x_blocks[i].qs.as_ptr() as *const core::arch::x86_64::__m256i);
                        x_d_beluga[i] = x_blocks[i].d;
                    }
                }

                let recent_slice = if history_len > 96 {
                    &history[history_len - 96..]
                } else {
                    history
                };

                let mut best_tok = worker_start as u32;
                let mut best_score = f32::NEG_INFINITY;
                let pen = if repeat_penalty > 1.0 { repeat_penalty } else { 1.25 };

                if is_manta {
                    let mut tok_id = worker_start as u32;
                    let end_id = worker_end as u32;
                    while tok_id + 1 < end_id {
                        let tok0 = tok_id;
                        let tok1 = tok_id + 1;
                        let skip0 = (suppress_eos && (tok0 == 151645 || tok0 == 151643)) || (!suppressed.is_empty() && suppressed.contains(&tok0));
                        let skip1 = (suppress_eos && (tok1 == 151645 || tok1 == 151643)) || (!suppressed.is_empty() && suppressed.contains(&tok1));
                        let r0_ptr = weight_bytes.as_ptr().add((tok0 as usize) * 952);
                        let r1_ptr = weight_bytes.as_ptr().add((tok1 as usize) * 952);
                        let (val0, val1) = quants::vec_dot_q8_0_28blocks_2rows_avx2(r0_ptr, r1_ptr, &x_qs_manta, &x_d_manta);

                        if !skip0 && val0 > best_score {
                            let mut score = val0;
                            let repeat_count = recent_slice.iter().filter(|&&id| id == tok0).count();
                            if repeat_count > 0 {
                                score -= (repeat_count as f32) * 2.5;
                                if score > 0.0 { score /= pen.powi(repeat_count as i32); }
                                else { score *= pen.powi(repeat_count as i32); }
                            }
                            if score > best_score {
                                best_score = score;
                                best_tok = tok0;
                            }
                        }
                        if !skip1 && val1 > best_score {
                            let mut score = val1;
                            let repeat_count = recent_slice.iter().filter(|&&id| id == tok1).count();
                            if repeat_count > 0 {
                                score -= (repeat_count as f32) * 2.5;
                                if score > 0.0 { score /= pen.powi(repeat_count as i32); }
                                else { score *= pen.powi(repeat_count as i32); }
                            }
                            if score > best_score {
                                best_score = score;
                                best_tok = tok1;
                            }
                        }
                        tok_id += 2;
                    }
                    if tok_id < end_id {
                        let tok0 = tok_id;
                        if !((suppress_eos && (tok0 == 151645 || tok0 == 151643)) || (!suppressed.is_empty() && suppressed.contains(&tok0))) {
                            let r0_ptr = weight_bytes.as_ptr().add((tok0 as usize) * 952);
                            let val0 = quants::vec_dot_q8_0_28blocks_avx2(r0_ptr, &x_qs_manta, &x_d_manta);
                            if val0 > best_score {
                                let mut score = val0;
                                let repeat_count = recent_slice.iter().filter(|&&id| id == tok0).count();
                                if repeat_count > 0 {
                                    score -= (repeat_count as f32) * 2.5;
                                    if score > 0.0 { score /= pen.powi(repeat_count as i32); }
                                    else { score *= pen.powi(repeat_count as i32); }
                                }
                                if score > best_score {
                                    best_score = score;
                                    best_tok = tok0;
                                }
                            }
                        }
                    }
                } else if is_beluga {
                    let mut tok_id = worker_start as u32;
                    let end_id = worker_end as u32;
                    while tok_id + 1 < end_id {
                        let tok0 = tok_id;
                        let tok1 = tok_id + 1;
                        let skip0 = (suppress_eos && (tok0 == 151645 || tok0 == 151643)) || (!suppressed.is_empty() && suppressed.contains(&tok0));
                        let skip1 = (suppress_eos && (tok1 == 151645 || tok1 == 151643)) || (!suppressed.is_empty() && suppressed.contains(&tok1));
                        let r0_ptr = weight_bytes.as_ptr().add((tok0 as usize) * 1632);
                        let r1_ptr = weight_bytes.as_ptr().add((tok1 as usize) * 1632);
                        let (val0, val1) = quants::vec_dot_q8_0_48blocks_2rows_avx2(r0_ptr, r1_ptr, &x_qs_beluga, &x_d_beluga);

                        if !skip0 && val0 > best_score {
                            let mut score = val0;
                            let repeat_count = recent_slice.iter().filter(|&&id| id == tok0).count();
                            if repeat_count > 0 {
                                score -= (repeat_count as f32) * 2.5;
                                if score > 0.0 { score /= pen.powi(repeat_count as i32); }
                                else { score *= pen.powi(repeat_count as i32); }
                            }
                            if score > best_score {
                                best_score = score;
                                best_tok = tok0;
                            }
                        }
                        if !skip1 && val1 > best_score {
                            let mut score = val1;
                            let repeat_count = recent_slice.iter().filter(|&&id| id == tok1).count();
                            if repeat_count > 0 {
                                score -= (repeat_count as f32) * 2.5;
                                if score > 0.0 { score /= pen.powi(repeat_count as i32); }
                                else { score *= pen.powi(repeat_count as i32); }
                            }
                            if score > best_score {
                                best_score = score;
                                best_tok = tok1;
                            }
                        }
                        tok_id += 2;
                    }
                    if tok_id < end_id {
                        let tok0 = tok_id;
                        if !((suppress_eos && (tok0 == 151645 || tok0 == 151643)) || (!suppressed.is_empty() && suppressed.contains(&tok0))) {
                            let r0_ptr = weight_bytes.as_ptr().add((tok0 as usize) * 1632);
                            let val0 = quants::vec_dot_q8_0_48blocks_avx2(r0_ptr, &x_qs_beluga, &x_d_beluga);
                            if val0 > best_score {
                                let mut score = val0;
                                let repeat_count = recent_slice.iter().filter(|&&id| id == tok0).count();
                                if repeat_count > 0 {
                                    score -= (repeat_count as f32) * 2.5;
                                    if score > 0.0 { score /= pen.powi(repeat_count as i32); }
                                    else { score *= pen.powi(repeat_count as i32); }
                                }
                                if score > best_score {
                                    best_score = score;
                                    best_tok = tok0;
                                }
                            }
                        }
                    }
                } else if is_skate {
                    for tok_id in (worker_start as u32)..(worker_end as u32) {
                        if (suppress_eos && (tok_id == 0 || tok_id == 1 || tok_id == 2)) || (!suppressed.is_empty() && suppressed.contains(&tok_id)) || tok_id == 3725 || tok_id == 47182 {
                            continue;
                        }
                        let row_offset = (tok_id as usize) * 612;
                        let row_ptr = weight_bytes.as_ptr().add(row_offset);
                        let val = quants::vec_dot_q8_0_18blocks_avx2(row_ptr, &x_qs_skate, &x_d_skate);
                        if val > best_score {
                            let mut score = val;
                            let repeat_count = recent_slice.iter().filter(|&&id| id == tok_id).count();
                            if repeat_count > 0 {
                                score -= (repeat_count as f32) * 2.5;
                                if score > 0.0 {
                                    score /= pen.powi(repeat_count as i32);
                                } else {
                                    score *= pen.powi(repeat_count as i32);
                                }
                            }
                            if score > best_score {
                                best_score = score;
                                best_tok = tok_id;
                            }
                        }
                    }
                } else {
                    for tok_id in (worker_start as u32)..(worker_end as u32) {
                        if (suppress_eos && (tok_id == 0 || tok_id == 1 || tok_id == 2 || tok_id == 151645 || tok_id == 151643)) || (!suppressed.is_empty() && suppressed.contains(&tok_id)) {
                            continue;
                        }
                        let row_offset = (tok_id as usize) * bytes_per_token;
                        let row_ptr = weight_bytes.as_ptr().add(row_offset);
                        let row_slice = std::slice::from_raw_parts(row_ptr, bytes_per_token);
                        let val = quants::vec_dot_q8_0_q8_0_avx2(row_slice, x_blocks);
                        if val > best_score {
                            let mut score = val;
                            let repeat_count = recent_slice.iter().filter(|&&id| id == tok_id).count();
                            if repeat_count > 0 {
                                score -= (repeat_count as f32) * 2.5;
                                if score > 0.0 {
                                    score /= pen.powi(repeat_count as i32);
                                } else {
                                    score *= pen.powi(repeat_count as i32);
                                }
                            }
                            if score > best_score {
                                best_score = score;
                                best_tok = tok_id;
                            }
                        }
                    }
                }

                let top_toks = std::slice::from_raw_parts_mut(top_tok_ptr as *mut u32, num_workers);
                let top_scores = std::slice::from_raw_parts_mut(top_score_ptr as *mut f32, num_workers);
                top_toks[worker_id] = best_tok;
                top_scores[worker_id] = best_score;
            }
            GemvTask::VocabIq4Nl {
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
            } => {
                let chunk = (vocab_count + num_workers - 1) / num_workers;
                let worker_start = (start_tok + worker_id * chunk).min(start_tok + vocab_count);
                let worker_end = (start_tok + (worker_id + 1) * chunk).min(start_tok + vocab_count);
                if worker_start >= worker_end {
                    return;
                }
                let num_blocks = bytes_per_token / 18;
                let weight_bytes = std::slice::from_raw_parts(weight_ptr as *const u8, (start_tok + vocab_count) * bytes_per_token);
                let x_blocks = std::slice::from_raw_parts(x_blocks_ptr as *const BlockQ8_0, num_blocks);
                let history = std::slice::from_raw_parts(history_ptr as *const u32, history_len);
                let suppressed = if suppressed_len > 0 {
                    std::slice::from_raw_parts(suppressed_ptr as *const u32, suppressed_len)
                } else {
                    &[]
                };

                let is_manta = num_blocks == 28;
                let is_beluga = num_blocks == 48;

                let mut x_qs_manta = [core::arch::x86_64::_mm256_setzero_si256(); 28];
                let mut x_d_manta = [0.0f32; 28];
                let mut x_qs_beluga = [core::arch::x86_64::_mm256_setzero_si256(); 48];
                let mut x_d_beluga = [0.0f32; 48];

                if is_manta {
                    for i in 0..28 {
                        x_qs_manta[i] = core::arch::x86_64::_mm256_loadu_si256(x_blocks[i].qs.as_ptr() as *const core::arch::x86_64::__m256i);
                        x_d_manta[i] = x_blocks[i].d;
                    }
                } else if is_beluga {
                    for i in 0..48 {
                        x_qs_beluga[i] = core::arch::x86_64::_mm256_loadu_si256(x_blocks[i].qs.as_ptr() as *const core::arch::x86_64::__m256i);
                        x_d_beluga[i] = x_blocks[i].d;
                    }
                }

                let recent_slice = if history_len > 96 {
                    &history[history_len - 96..]
                } else {
                    history
                };

                let mut best_tok = worker_start as u32;
                let mut best_score = f32::NEG_INFINITY;
                let pen = if repeat_penalty > 1.0 { repeat_penalty } else { 1.25 };

                if is_manta {
                    let mut tok_id = worker_start as u32;
                    let end_id = worker_end as u32;
                    while tok_id + 1 < end_id {
                        let tok0 = tok_id;
                        let tok1 = tok_id + 1;
                        let skip0 = (suppress_eos && (tok0 == 151645 || tok0 == 151643)) || (!suppressed.is_empty() && suppressed.contains(&tok0));
                        let skip1 = (suppress_eos && (tok1 == 151645 || tok1 == 151643)) || (!suppressed.is_empty() && suppressed.contains(&tok1));
                        let r0_ptr = weight_bytes.as_ptr().add((tok0 as usize) * 504);
                        let r1_ptr = weight_bytes.as_ptr().add((tok1 as usize) * 504);
                        let (val0, val1) = quants::vec_dot_iq4_nl_28blocks_2rows_avx2(r0_ptr, r1_ptr, &x_qs_manta, &x_d_manta);

                        if !skip0 && val0 > best_score {
                            let mut score = val0;
                            let repeat_count = recent_slice.iter().filter(|&&id| id == tok0).count();
                            if repeat_count > 0 {
                                score -= (repeat_count as f32) * 2.5;
                                if score > 0.0 { score /= pen.powi(repeat_count as i32); }
                                else { score *= pen.powi(repeat_count as i32); }
                            }
                            if score > best_score {
                                best_score = score;
                                best_tok = tok0;
                            }
                        }
                        if !skip1 && val1 > best_score {
                            let mut score = val1;
                            let repeat_count = recent_slice.iter().filter(|&&id| id == tok1).count();
                            if repeat_count > 0 {
                                score -= (repeat_count as f32) * 2.5;
                                if score > 0.0 { score /= pen.powi(repeat_count as i32); }
                                else { score *= pen.powi(repeat_count as i32); }
                            }
                            if score > best_score {
                                best_score = score;
                                best_tok = tok1;
                            }
                        }
                        tok_id += 2;
                    }
                    if tok_id < end_id {
                        let tok0 = tok_id;
                        if !((suppress_eos && (tok0 == 151645 || tok0 == 151643)) || (!suppressed.is_empty() && suppressed.contains(&tok0))) {
                            let r0_ptr = weight_bytes.as_ptr().add((tok0 as usize) * 504);
                            let val0 = quants::vec_dot_iq4_nl_28blocks_avx2(r0_ptr, &x_qs_manta, &x_d_manta);
                            if val0 > best_score {
                                let mut score = val0;
                                let repeat_count = recent_slice.iter().filter(|&&id| id == tok0).count();
                                if repeat_count > 0 {
                                    score -= (repeat_count as f32) * 2.5;
                                    if score > 0.0 { score /= pen.powi(repeat_count as i32); }
                                    else { score *= pen.powi(repeat_count as i32); }
                                }
                                if score > best_score {
                                    best_score = score;
                                    best_tok = tok0;
                                }
                            }
                        }
                    }
                } else if is_beluga {
                    let mut tok_id = worker_start as u32;
                    let end_id = worker_end as u32;
                    while tok_id + 1 < end_id {
                        let tok0 = tok_id;
                        let tok1 = tok_id + 1;
                        let skip0 = (suppress_eos && (tok0 == 151645 || tok0 == 151643)) || (!suppressed.is_empty() && suppressed.contains(&tok0));
                        let skip1 = (suppress_eos && (tok1 == 151645 || tok1 == 151643)) || (!suppressed.is_empty() && suppressed.contains(&tok1));
                        let r0_ptr = weight_bytes.as_ptr().add((tok0 as usize) * 864);
                        let r1_ptr = weight_bytes.as_ptr().add((tok1 as usize) * 864);
                        let (val0, val1) = quants::vec_dot_iq4_nl_48blocks_2rows_avx2(r0_ptr, r1_ptr, &x_qs_beluga, &x_d_beluga);

                        if !skip0 && val0 > best_score {
                            let mut score = val0;
                            let repeat_count = recent_slice.iter().filter(|&&id| id == tok0).count();
                            if repeat_count > 0 {
                                score -= (repeat_count as f32) * 2.5;
                                if score > 0.0 { score /= pen.powi(repeat_count as i32); }
                                else { score *= pen.powi(repeat_count as i32); }
                            }
                            if score > best_score {
                                best_score = score;
                                best_tok = tok0;
                            }
                        }
                        if !skip1 && val1 > best_score {
                            let mut score = val1;
                            let repeat_count = recent_slice.iter().filter(|&&id| id == tok1).count();
                            if repeat_count > 0 {
                                score -= (repeat_count as f32) * 2.5;
                                if score > 0.0 { score /= pen.powi(repeat_count as i32); }
                                else { score *= pen.powi(repeat_count as i32); }
                            }
                            if score > best_score {
                                best_score = score;
                                best_tok = tok1;
                            }
                        }
                        tok_id += 2;
                    }
                    if tok_id < end_id {
                        let tok0 = tok_id;
                        if !((suppress_eos && (tok0 == 151645 || tok0 == 151643)) || (!suppressed.is_empty() && suppressed.contains(&tok0))) {
                            let r0_ptr = weight_bytes.as_ptr().add((tok0 as usize) * 864);
                            let val0 = quants::vec_dot_iq4_nl_48blocks_avx2(r0_ptr, &x_qs_beluga, &x_d_beluga);
                            if val0 > best_score {
                                let mut score = val0;
                                let repeat_count = recent_slice.iter().filter(|&&id| id == tok0).count();
                                if repeat_count > 0 {
                                    score -= (repeat_count as f32) * 2.5;
                                    if score > 0.0 { score /= pen.powi(repeat_count as i32); }
                                    else { score *= pen.powi(repeat_count as i32); }
                                }
                                if score > best_score {
                                    best_score = score;
                                    best_tok = tok0;
                                }
                            }
                        }
                    }
                } else {
                    for tok_id in (worker_start as u32)..(worker_end as u32) {
                        if (suppress_eos && (tok_id == 0 || tok_id == 1 || tok_id == 2 || tok_id == 151645 || tok_id == 151643)) || (!suppressed.is_empty() && suppressed.contains(&tok_id)) {
                            continue;
                        }
                        let row_offset = (tok_id as usize) * bytes_per_token;
                        let row_ptr = weight_bytes.as_ptr().add(row_offset);
                        let row_slice = std::slice::from_raw_parts(row_ptr, bytes_per_token);
                        let val = quants::vec_dot_iq4_nl_q8_0_avx2(row_slice, x_blocks);
                        if val > best_score {
                            let mut score = val;
                            let repeat_count = recent_slice.iter().filter(|&&id| id == tok_id).count();
                            if repeat_count > 0 {
                                score -= (repeat_count as f32) * 2.5;
                                if score > 0.0 {
                                    score /= pen.powi(repeat_count as i32);
                                } else {
                                    score *= pen.powi(repeat_count as i32);
                                }
                            }
                            if score > best_score {
                                best_score = score;
                                best_tok = tok_id;
                            }
                        }
                    }
                }

                let top_toks = std::slice::from_raw_parts_mut(top_tok_ptr as *mut u32, num_workers);
                let top_scores = std::slice::from_raw_parts_mut(top_score_ptr as *mut f32, num_workers);
                top_toks[worker_id] = best_tok;
                top_scores[worker_id] = best_score;
            }
            GemvTask::GateUpIq3Xxs {
                inter_out_ptr,
                w1_ptr,
                w3_ptr,
                x_q8_k_ptr,
                n_rows,
                n_cols,
            } => {
                let chunk = (n_rows + num_workers - 1) / num_workers;
                let start_r = (worker_id * chunk).min(n_rows);
                let end_r = ((worker_id + 1) * chunk).min(n_rows);
                let rows = end_r - start_r;
                if rows == 0 {
                    return;
                }
                let nb = (n_cols + 255) / 256;
                let row_bytes = nb * 98;
                let x_q8_k = x_q8_k_ptr as *const BlockQ8_K;
                let inter_base = (inter_out_ptr as *mut f32).add(start_r);

                for j in 0..rows {
                    let r = start_r + j;
                    let w1_row_ptr = (w1_ptr as *const u8).add(r * row_bytes);
                    let w3_row_ptr = (w3_ptr as *const u8).add(r * row_bytes);
                    let (w1_val, w3_val) = quants::vec_dot_iq3_xxs_q8_k_2rows_avx2_ptr(w1_row_ptr, w3_row_ptr, x_q8_k, nb);
                    *inter_base.add(j) = quants::fast_silu(w1_val) * w3_val;
                }
            }
            GemvTask::GateUpIq3Xxs2Tokens {
                inter0_out_ptr,
                inter1_out_ptr,
                w1_ptr,
                w3_ptr,
                x0_q8_k_ptr,
                x1_q8_k_ptr,
                n_rows,
                n_cols,
            } => {
                let chunk = (n_rows + num_workers - 1) / num_workers;
                let start_r = (worker_id * chunk).min(n_rows);
                let end_r = ((worker_id + 1) * chunk).min(n_rows);
                let rows = end_r - start_r;
                if rows == 0 {
                    return;
                }
                let nb = (n_cols + 255) / 256;
                let row_bytes = nb * 98;
                let x0_q8_k = x0_q8_k_ptr as *const BlockQ8_K;
                let x1_q8_k = x1_q8_k_ptr as *const BlockQ8_K;
                let inter0_base = (inter0_out_ptr as *mut f32).add(start_r);
                let inter1_base = (inter1_out_ptr as *mut f32).add(start_r);

                for j in 0..rows {
                    let r = start_r + j;
                    let w1_row_ptr = (w1_ptr as *const u8).add(r * row_bytes);
                    let w3_row_ptr = (w3_ptr as *const u8).add(r * row_bytes);
                    let (w1_0, w1_1) = quants::vec_dot_iq3_xxs_q8_k_2tokens_avx2_ptr(w1_row_ptr, x0_q8_k, x1_q8_k, nb);
                    let (w3_0, w3_1) = quants::vec_dot_iq3_xxs_q8_k_2tokens_avx2_ptr(w3_row_ptr, x0_q8_k, x1_q8_k, nb);
                    *inter0_base.add(j) = quants::fast_silu(w1_0) * w3_0;
                    *inter1_base.add(j) = quants::fast_silu(w1_1) * w3_1;
                }
            }
            GemvTask::Iq3Xxs { out_ptr, weight_ptr, x_q8_k_ptr, n_rows, n_cols } => {
                let chunk = (n_rows + num_workers - 1) / num_workers;
                let start_r = (worker_id * chunk).min(n_rows);
                let end_r = ((worker_id + 1) * chunk).min(n_rows);
                let rows = end_r - start_r;
                if rows == 0 {
                    return;
                }
                let row_bytes = (n_cols / 256) * 98;
                let out_slice = std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start_r), rows);
                let weight_slice = std::slice::from_raw_parts((weight_ptr as *const u8).add(start_r * row_bytes), rows * row_bytes);
                let x_q8_k = std::slice::from_raw_parts(x_q8_k_ptr as *const BlockQ8_K, (n_cols + 255) / 256);
                quants::gemv_iq3_xxs_prequant(out_slice, weight_slice, x_q8_k, rows, n_cols);
            }
            GemvTask::Iq3Xxs2Tokens {
                out0_ptr,
                out1_ptr,
                weight_ptr,
                x0_q8_k_ptr,
                x1_q8_k_ptr,
                n_rows,
                n_cols,
            } => {
                let chunk = (n_rows + num_workers - 1) / num_workers;
                let start_r = (worker_id * chunk).min(n_rows);
                let end_r = ((worker_id + 1) * chunk).min(n_rows);
                let rows = end_r - start_r;
                if rows == 0 {
                    return;
                }
                let nb = (n_cols + 255) / 256;
                let row_bytes = nb * 98;
                let x0_q8_k = x0_q8_k_ptr as *const BlockQ8_K;
                let x1_q8_k = x1_q8_k_ptr as *const BlockQ8_K;
                let out0_base = (out0_ptr as *mut f32).add(start_r);
                let out1_base = (out1_ptr as *mut f32).add(start_r);

                for j in 0..rows {
                    let r = start_r + j;
                    let w_row_ptr = (weight_ptr as *const u8).add(r * row_bytes);
                    let (v0, v1) = quants::vec_dot_iq3_xxs_q8_k_2tokens_avx2_ptr(w_row_ptr, x0_q8_k, x1_q8_k, nb);
                    *out0_base.add(j) = v0;
                    *out1_base.add(j) = v1;
                }
            }
            GemvTask::Q8_0 { out_ptr, weight_ptr, x_norm_ptr, n_rows, n_cols } => {
                let chunk = (n_rows + num_workers - 1) / num_workers;
                let start_r = (worker_id * chunk).min(n_rows);
                let end_r = ((worker_id + 1) * chunk).min(n_rows);
                let rows = end_r - start_r;
                if rows == 0 {
                    return;
                }
                let row_bytes = (n_cols / 32) * 34;
                let out_slice = std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(start_r), rows);
                let weight_slice = std::slice::from_raw_parts((weight_ptr as *const u8).add(start_r * row_bytes), rows * row_bytes);
                let x_norm = std::slice::from_raw_parts(x_norm_ptr as *const f32, n_cols);
                quants::gemv_q8_0(out_slice, weight_slice, x_norm, rows, n_cols);
            }
            GemvTask::Attention {
                q_ptr,
                k_cache_ptr,
                v_cache_ptr,
                out_ptr,
                seq_len,
                n_head,
                n_kv_head,
                head_dim,
            } => {
                let chunk = (n_head + num_workers - 1) / num_workers;
                let start_h = (worker_id * chunk).min(n_head);
                let end_h = ((worker_id + 1) * chunk).min(n_head);
                if start_h >= end_h {
                    return;
                }
                let q = std::slice::from_raw_parts(q_ptr as *const f32, n_head * head_dim);
                let k_cache = std::slice::from_raw_parts(k_cache_ptr as *const f32, seq_len * n_kv_head * head_dim);
                let v_cache = std::slice::from_raw_parts(v_cache_ptr as *const f32, seq_len * n_kv_head * head_dim);
                let out = std::slice::from_raw_parts_mut(out_ptr as *mut f32, n_head * head_dim);

                let mut scores = [0.0f32; 2048];
                let scores_slice = &mut scores[..seq_len];

                let kv_repeats = n_head / n_kv_head;
                let scale = 1.0 / (head_dim as f32).sqrt();
                let kv_stride = n_kv_head * head_dim;
                let head_dim_vec = head_dim / 8;

                for h in start_h..end_h {
                    let kv_h = h / kv_repeats;
                    let q_head_ptr = q.as_ptr().add(h * head_dim);

                    // Preload query vectors in AVX2 registers/stack (head_dim <= 128 -> head_dim_vec <= 16)
                    let num_q_vec = head_dim_vec.min(16);
                    let mut q_vec = [core::arch::x86_64::_mm256_setzero_ps(); 16];
                    for vd in 0..num_q_vec {
                        q_vec[vd] = core::arch::x86_64::_mm256_loadu_ps(q_head_ptr.add(vd * 8));
                    }

                    let mut max_s = f32::NEG_INFINITY;
                    for t in 0..seq_len {
                        let k_start = t * kv_stride + kv_h * head_dim;
                        let k_ptr = k_cache.as_ptr().add(k_start);

                        // Lookahead prefetch for k_cache
                        if t + 4 < seq_len {
                            let k_next = (t + 4) * kv_stride + kv_h * head_dim;
                            core::arch::x86_64::_mm_prefetch(k_cache.as_ptr().add(k_next) as *const i8, core::arch::x86_64::_MM_HINT_T0);
                        }

                        let mut dot_vec = core::arch::x86_64::_mm256_setzero_ps();
                        for vd in 0..num_q_vec {
                            let k_v = core::arch::x86_64::_mm256_loadu_ps(k_ptr.add(vd * 8));
                            dot_vec = core::arch::x86_64::_mm256_fmadd_ps(q_vec[vd], k_v, dot_vec);
                        }

                        let mut res = core::arch::x86_64::_mm256_extractf128_ps(dot_vec, 1);
                        res = core::arch::x86_64::_mm_add_ps(res, core::arch::x86_64::_mm256_castps256_ps128(dot_vec));
                        res = core::arch::x86_64::_mm_add_ps(res, core::arch::x86_64::_mm_movehl_ps(res, res));
                        res = core::arch::x86_64::_mm_add_ss(res, core::arch::x86_64::_mm_movehdup_ps(res));
                        let mut dot = core::arch::x86_64::_mm_cvtss_f32(res);

                        for d in (head_dim_vec * 8)..head_dim {
                            dot += *q_head_ptr.add(d) * *k_ptr.add(d);
                        }

                        let s = dot * scale;
                        scores_slice[t] = s;
                        if s > max_s {
                            max_s = s;
                        }
                    }

                    let mut sum_exp = 0.0f32;
                    for t in 0..seq_len {
                        let w = (scores_slice[t] - max_s).exp();
                        scores_slice[t] = w;
                        sum_exp += w;
                    }
                    let inv_sum = 1.0 / sum_exp;
                    for t in 0..seq_len {
                        scores_slice[t] *= inv_sum;
                    }

                    // Accumulate V in registers / stack without re-loading/storing out_ptr on every token step
                    let mut acc_v = [core::arch::x86_64::_mm256_setzero_ps(); 16];
                    let mut tail_acc = [0.0f32; 8];
                    let tail_start = head_dim_vec * 8;
                    let tail_len = head_dim - tail_start;

                    for t in 0..seq_len {
                        // Lookahead prefetch for v_cache
                        if t + 4 < seq_len {
                            let v_next = (t + 4) * kv_stride + kv_h * head_dim;
                            core::arch::x86_64::_mm_prefetch(v_cache.as_ptr().add(v_next) as *const i8, core::arch::x86_64::_MM_HINT_T0);
                        }

                        let s = scores_slice[t];
                        let s_vec = core::arch::x86_64::_mm256_set1_ps(s);
                        let v_ptr = v_cache.as_ptr().add(t * kv_stride + kv_h * head_dim);

                        for vd in 0..num_q_vec {
                            let v_v = core::arch::x86_64::_mm256_loadu_ps(v_ptr.add(vd * 8));
                            acc_v[vd] = core::arch::x86_64::_mm256_fmadd_ps(s_vec, v_v, acc_v[vd]);
                        }
                        if tail_len > 0 {
                            for d in 0..tail_len {
                                tail_acc[d] += s * *v_ptr.add(tail_start + d);
                            }
                        }
                    }

                    let out_ptr = out.as_mut_ptr().add(h * head_dim);
                    for vd in 0..num_q_vec {
                        core::arch::x86_64::_mm256_storeu_ps(out_ptr.add(vd * 8), acc_v[vd]);
                    }
                    if tail_len > 0 {
                        for d in 0..tail_len {
                            *out_ptr.add(tail_start + d) = tail_acc[d];
                        }
                    }
                }
            }
        }
    }
}

struct SharedState {
    task: std::cell::UnsafeCell<Option<GemvTask>>,
    generation: AtomicU32,
    done_count: AtomicU32,
    running: AtomicBool,
    spinning: AtomicBool,
    mutex: Mutex<()>,
    condvar: Condvar,
}

unsafe impl Sync for SharedState {}

unsafe fn pin_thread_to_cpu(cpu_id: usize) {
    let mut set: libc::cpu_set_t = std::mem::zeroed();
    libc::CPU_SET(cpu_id, &mut set);
    libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
}

pub struct SpinThreadPool {
    shared: Arc<SharedState>,
    _threads: Vec<JoinHandle<()>>,
}

impl SpinThreadPool {
    pub fn new() -> Self {
        // Pin calling thread (Worker 0) to physical CPU 0
        unsafe {
            pin_thread_to_cpu(0);
        }

        let shared = Arc::new(SharedState {
            task: std::cell::UnsafeCell::new(None),
            generation: AtomicU32::new(0),
            done_count: AtomicU32::new(0),
            running: AtomicBool::new(true),
            spinning: AtomicBool::new(true),
            mutex: Mutex::new(()),
            condvar: Condvar::new(),
        });

        let mut threads = Vec::with_capacity(NUM_WORKERS - 1);

        for worker_id in 1..NUM_WORKERS {
            let shared_c = Arc::clone(&shared);
            let handle = thread::Builder::new()
                .name(format!("ast-worker-{}", worker_id))
                .stack_size(512 * 1024)
                .spawn(move || {
                    // Pin worker thread to physical CPU (worker 1 -> CPU 2, worker 2 -> CPU 4, worker 3 -> CPU 6)
                    unsafe {
                        pin_thread_to_cpu(worker_id * 2);
                    }
                    let mut local_gen = 0u32;
                    while shared_c.running.load(Ordering::Relaxed) {
                        // Check if in spin mode
                        if shared_c.spinning.load(Ordering::Acquire) {
                            let cur_gen = shared_c.generation.load(Ordering::Acquire);
                            if cur_gen != local_gen {
                                local_gen = cur_gen;
                                let task = unsafe { *shared_c.task.get() };
                                if let Some(t) = task {
                                    unsafe {
                                        t.execute(worker_id, NUM_WORKERS);
                                    }
                                }
                                shared_c.done_count.fetch_add(1, Ordering::Release);
                            } else {
                                core::hint::spin_loop();
                            }
                        } else {
                            // Parked on Condvar to save CPU
                            let guard = shared_c.mutex.lock().unwrap();
                            let _guard = shared_c
                                .condvar
                                .wait_while(guard, |_| !shared_c.spinning.load(Ordering::Acquire) && shared_c.running.load(Ordering::Relaxed))
                                .unwrap();
                        }
                    }
                })
                .expect("failed to spawn worker thread");
            threads.push(handle);
        }

        Self {
            shared,
            _threads: threads,
        }
    }

    #[inline(always)]
    pub fn run_task(&self, task: GemvTask) {
        unsafe {
            *self.shared.task.get() = Some(task);
        }
        self.shared.done_count.store(0, Ordering::Release);
        self.shared.generation.fetch_add(1, Ordering::Release);

        // Worker 0 executes on calling thread
        unsafe {
            task.execute(0, NUM_WORKERS);
        }

        // Wait for workers 1..NUM_WORKERS to complete
        while self.shared.done_count.load(Ordering::Acquire) < (NUM_WORKERS - 1) as u32 {
            core::hint::spin_loop();
        }
    }

    pub fn park(&self) {
        let _guard = self.shared.mutex.lock().unwrap();
        self.shared.spinning.store(false, Ordering::Release);
        self.shared.condvar.notify_all();
    }

    pub fn unpark(&self) {
        let _guard = self.shared.mutex.lock().unwrap();
        self.shared.spinning.store(true, Ordering::Release);
        self.shared.condvar.notify_all();
    }
}

impl Drop for SpinThreadPool {
    fn drop(&mut self) {
        self.shared.running.store(false, Ordering::Release);
        self.shared.spinning.store(false, Ordering::Release);
        let _guard = self.shared.mutex.lock().unwrap();
        self.shared.condvar.notify_all();
    }
}
