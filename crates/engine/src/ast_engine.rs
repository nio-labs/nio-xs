use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use memmap2::Mmap;
use std::collections::HashSet;
use std::fs::File;
use std::path::Path;

use crate::gguf;
use crate::quants;
use crate::pool::{self, GemvTask, SpinThreadPool};

pub fn memory_mb() -> (f64, f64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let get_val = |key: &str| -> f64 {
        status
            .lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0)
            / 1024.0
    };
    (get_val("VmRSS:"), get_val("VmHWM:"))
}

#[allow(dead_code)]
fn random_f32() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static SEED: Cell<u32> = Cell::new(0xdeadbeef);
    }
    SEED.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        s.set(x);
        (x as f32) / (u32::MAX as f32)
    })
}

#[allow(dead_code)]
#[inline(always)]
fn insert_top_k(top: &mut [(u32, f32); 16], tok: u32, score: f32) {
    if score <= top[15].1 {
        return;
    }
    let mut pos = 15;
    while pos > 0 && top[pos - 1].1 < score {
        top[pos] = top[pos - 1];
        pos -= 1;
    }
    top[pos] = (tok, score);
}

#[allow(dead_code)]
fn merge_top_k(a: &[(u32, f32); 16], b: &[(u32, f32); 16]) -> [(u32, f32); 16] {
    let mut res = [(0u32, f32::NEG_INFINITY); 16];
    let mut i = 0;
    let mut j = 0;
    for k in 0..16 {
        if a[i].1 >= b[j].1 {
            res[k] = a[i];
            i += 1;
        } else {
            res[k] = b[j];
            j += 1;
        }
    }
    res
}

#[allow(dead_code)]
fn sample_top_p(top: &[(u32, f32); 16], temperature: f32, top_p: f32) -> u32 {
    if temperature < 0.05 {
        return top[0].0;
    }
    let max_l = top[0].1;
    let mut exp_sum = 0.0f32;
    let mut probs = [(0u32, 0.0f32); 16];
    for i in 0..16 {
        let p = ((top[i].1 - max_l) / temperature).exp();
        exp_sum += p;
        probs[i] = (top[i].0, p);
    }
    let inv_sum = 1.0 / exp_sum;
    for i in 0..16 {
        probs[i].1 *= inv_sum;
    }

    let mut cum = 0.0f32;
    let mut cutoff = 16;
    for i in 0..16 {
        cum += probs[i].1;
        if cum >= top_p {
            cutoff = i + 1;
            break;
        }
    }

    let p_slice = &probs[..cutoff];
    let total_p: f32 = p_slice.iter().map(|p| p.1).sum();
    let r = random_f32() * total_p;
    let mut acc = 0.0f32;
    for &(tok, p) in p_slice {
        acc += p;
        if acc >= r {
            return tok;
        }
    }
    top[0].0
}

#[inline(always)]
pub fn rms_norm_slice(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let len = x.len();
    let mut sum_sq = 0.0f32;
    for i in 0..len {
        sum_sq += x[i] * x[i];
    }
    let scale = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    for i in 0..len {
        out[i] = x[i] * scale * weight[i];
    }
}

#[inline(always)]
pub fn apply_rope(vec: &mut [f32], n_heads: usize, head_dim: usize, cos: &[f32], sin: &[f32], neox: bool) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        let offset = h * head_dim;
        for i in 0..half {
            let (i0, i1) = if neox {
                (offset + i, offset + i + half)
            } else {
                (offset + 2 * i, offset + 2 * i + 1)
            };
            let x0 = vec[i0];
            let x1 = vec[i1];
            let c = cos[i];
            let s = sin[i];
            vec[i0] = x0 * c - x1 * s;
            vec[i1] = x0 * s + x1 * c;
        }
    }
}

#[inline(always)]
pub fn compute_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    seq_len: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    out: &mut [f32],
    scores: &mut Vec<f32>,
) {
    let kv_repeats = n_head / n_kv_head;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_stride = n_kv_head * head_dim;

    scores.resize(seq_len, 0.0);

    for h in 0..n_head {
        let kv_h = h / kv_repeats;
        let q_head = &q[h * head_dim..(h + 1) * head_dim];

        let mut max_s = f32::NEG_INFINITY;
        for t in 0..seq_len {
            let k_start = t * kv_stride + kv_h * head_dim;
            let k_token = &k_cache[k_start..k_start + head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_head[d] * k_token[d];
            }
            let s = dot * scale;
            scores[t] = s;
            if s > max_s {
                max_s = s;
            }
        }

        let mut sum_exp = 0.0f32;
        for t in 0..seq_len {
            let w = (scores[t] - max_s).exp();
            scores[t] = w;
            sum_exp += w;
        }
        let inv_sum = 1.0 / sum_exp;
        for t in 0..seq_len {
            scores[t] *= inv_sum;
        }

        for d in 0..head_dim {
            let mut val = 0.0f32;
            for t in 0..seq_len {
                let v_start = t * kv_stride + kv_h * head_dim;
                val += scores[t] * v_cache[v_start + d];
            }
            out[h * head_dim + d] = val;
        }
    }
}

pub struct ScratchBuffers {
    pub x: Vec<f32>,
    pub x_norm: Vec<f32>,
    pub x_q8_0: Vec<quants::BlockQ8_0>,
    pub x_q8_k: Vec<quants::BlockQ8_K>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub attn_q8_0: Vec<quants::BlockQ8_0>,
    pub attn_q8_k: Vec<quants::BlockQ8_K>,
    pub wo_out: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub ffn_q8_0: Vec<quants::BlockQ8_0>,
    pub ffn_q8_k: Vec<quants::BlockQ8_K>,
    pub w1: Vec<f32>,
    pub w3: Vec<f32>,
    pub inter: Vec<f32>,
    pub inter_q8_k: Vec<quants::BlockQ8_K>,
    pub mlp_out: Vec<f32>,
    pub attn_scores: Vec<f32>,
}

impl ScratchBuffers {
    pub fn new(embedding_length: usize, wq_rows: usize, wk_rows: usize, wv_rows: usize, w1_rows: usize) -> Self {
        Self {
            x: vec![0f32; embedding_length],
            x_norm: vec![0f32; embedding_length],
            x_q8_0: vec![quants::BlockQ8_0 { d: 0.0, qs: [0; 32] }; embedding_length / 32],
            x_q8_k: vec![quants::BlockQ8_K { d: 0.0, qs: [0; 256], bsums: [0; 16] }; (embedding_length + 255) / 256],
            q: vec![0f32; wq_rows],
            k: vec![0f32; wk_rows],
            v: vec![0f32; wv_rows],
            attn_out: vec![0f32; wq_rows],
            attn_q8_0: vec![quants::BlockQ8_0 { d: 0.0, qs: [0; 32] }; wq_rows / 32],
            attn_q8_k: vec![quants::BlockQ8_K { d: 0.0, qs: [0; 256], bsums: [0; 16] }; (wq_rows + 255) / 256],
            wo_out: vec![0f32; embedding_length],
            ffn_norm: vec![0f32; embedding_length],
            ffn_q8_0: vec![quants::BlockQ8_0 { d: 0.0, qs: [0; 32] }; embedding_length / 32],
            ffn_q8_k: vec![quants::BlockQ8_K { d: 0.0, qs: [0; 256], bsums: [0; 16] }; (embedding_length + 255) / 256],
            w1: vec![0f32; w1_rows],
            w3: vec![0f32; w1_rows],
            inter: vec![0f32; w1_rows],
            inter_q8_k: vec![quants::BlockQ8_K { d: 0.0, qs: [0; 256], bsums: [0; 16] }; (w1_rows + 255) / 256],
            mlp_out: vec![0f32; embedding_length],
            attn_scores: Vec::with_capacity(512),
        }
    }
}

#[derive(Clone, Default)]
pub struct LayerKvCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub cur_len: usize,
}

impl LayerKvCache {
    pub fn reset(&mut self) {
        self.k.clear();
        self.v.clear();
        self.cur_len = 0;
    }

    pub fn truncate(&mut self, target_len: usize, kv_stride: usize) {
        if target_len < self.cur_len {
            self.cur_len = target_len;
            let target_elems = target_len * kv_stride;
            self.k.truncate(target_elems);
            self.v.truncate(target_elems);
        }
    }
}

#[derive(Clone, Debug)]
pub struct LinearSlice {
    pub offset: usize,
    pub byte_len: usize,
    pub n_rows: usize,
    pub n_cols: usize,
    pub dtype: u32,
}

impl LinearSlice {
    pub fn new(meta: &gguf::TensorMeta, data_offset: usize) -> Self {
        let offset = data_offset + meta.offset as usize;
        let elem_count: usize = meta.shape.iter().product();
        let (block_size, type_size) = match meta.dtype {
            0 => (1, 4),
            2 => (32, 18),
            8 => (32, 34),
            18 => (256, 98),
            20 => (32, 18),
            other => panic!("unsupported dtype {other}"),
        };
        let byte_len = (elem_count / block_size) * type_size;
        let n_rows = meta.shape[0];
        let n_cols = meta.shape[1];
        Self {
            offset,
            byte_len,
            n_rows,
            n_cols,
            dtype: meta.dtype,
        }
    }
}

pub struct LayerDesc {
    pub idx: usize,
    pub wq: LinearSlice,
    pub wk: LinearSlice,
    pub wv: LinearSlice,
    pub wo: LinearSlice,
    pub w1: LinearSlice,
    pub w2: LinearSlice,
    pub w3: LinearSlice,
    pub q_bias: Vec<f32>,
    pub k_bias: Vec<f32>,
    pub v_bias: Vec<f32>,
    pub q_norm_weight: Vec<f32>,
    pub k_norm_weight: Vec<f32>,
    pub attn_norm_weight: Vec<f32>,
    pub ffn_norm_weight: Vec<f32>,
    pub min_offset: usize,
    pub max_offset: usize,
}

impl LayerDesc {
    #[inline(always)]
    pub fn forward_token(
        &self,
        mmap: &[u8],
        scratch: &mut ScratchBuffers,
        pool: &SpinThreadPool,
        _pos: usize,
        cos: &[f32],
        sin: &[f32],
        neox: bool,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        eps: f32,
        kv_cache: &mut LayerKvCache,
    ) {
        // 1. Attention Norm
        rms_norm_slice(&scratch.x, &self.attn_norm_weight, eps, &mut scratch.x_norm);
        if self.wq.dtype == 18 {
            quants::quantize_row_q8_k(&scratch.x_norm, &mut scratch.x_q8_k);
        } else {
            quants::quantize_row_q8_0(&scratch.x_norm, &mut scratch.x_q8_0);
        }

        // 2. Q, K, V Projections (Pure AVX2 parallelized across spin threadpool)
        let wq_slice = &mmap[self.wq.offset..self.wq.offset + self.wq.byte_len];
        let wk_slice = &mmap[self.wk.offset..self.wk.offset + self.wk.byte_len];
        let wv_slice = &mmap[self.wv.offset..self.wv.offset + self.wv.byte_len];
        if self.wq.dtype == 18 {
            pool.run_task(GemvTask::QkvIq3Xxs {
                q_ptr: scratch.q.as_mut_ptr() as usize,
                k_ptr: scratch.k.as_mut_ptr() as usize,
                v_ptr: scratch.v.as_mut_ptr() as usize,
                wq_ptr: wq_slice.as_ptr() as usize,
                wk_ptr: wk_slice.as_ptr() as usize,
                wv_ptr: wv_slice.as_ptr() as usize,
                x_q8_k_ptr: scratch.x_q8_k.as_ptr() as usize,
                q_rows: self.wq.n_rows,
                k_rows: self.wk.n_rows,
                v_rows: self.wv.n_rows,
                n_cols: self.wq.n_cols,
            });
        } else if self.wq.dtype == 8 {
            pool.run_task(GemvTask::Q8_0 {
                out_ptr: scratch.q.as_mut_ptr() as usize,
                weight_ptr: wq_slice.as_ptr() as usize,
                x_norm_ptr: scratch.x_norm.as_ptr() as usize,
                n_rows: self.wq.n_rows,
                n_cols: self.wq.n_cols,
            });
            pool.run_task(GemvTask::Q8_0 {
                out_ptr: scratch.k.as_mut_ptr() as usize,
                weight_ptr: wk_slice.as_ptr() as usize,
                x_norm_ptr: scratch.x_norm.as_ptr() as usize,
                n_rows: self.wk.n_rows,
                n_cols: self.wk.n_cols,
            });
            pool.run_task(GemvTask::Q8_0 {
                out_ptr: scratch.v.as_mut_ptr() as usize,
                weight_ptr: wv_slice.as_ptr() as usize,
                x_norm_ptr: scratch.x_norm.as_ptr() as usize,
                n_rows: self.wv.n_rows,
                n_cols: self.wv.n_cols,
            });
        } else {
            pool.run_task(GemvTask::QkvIq4Nl {
                q_ptr: scratch.q.as_mut_ptr() as usize,
                k_ptr: scratch.k.as_mut_ptr() as usize,
                v_ptr: scratch.v.as_mut_ptr() as usize,
                wq_ptr: wq_slice.as_ptr() as usize,
                wk_ptr: wk_slice.as_ptr() as usize,
                wv_ptr: wv_slice.as_ptr() as usize,
                x_blocks_ptr: scratch.x_q8_0.as_ptr() as usize,
                q_rows: self.wq.n_rows,
                k_rows: self.wk.n_rows,
                v_rows: self.wv.n_rows,
                n_cols: self.wq.n_cols,
            });
        }

        if !self.q_bias.is_empty() {
            for i in 0..scratch.q.len() {
                scratch.q[i] += self.q_bias[i];
            }
        }
        if !self.k_bias.is_empty() {
            for i in 0..scratch.k.len() {
                scratch.k[i] += self.k_bias[i];
            }
        }
        if !self.v_bias.is_empty() {
            for i in 0..scratch.v.len() {
                scratch.v[i] += self.v_bias[i];
            }
        }

        // Qwen3 / Manta normalizes each query and key head before applying RoPE.
        for (values, weight) in [(&mut scratch.q, &self.q_norm_weight), (&mut scratch.k, &self.k_norm_weight)] {
            if !weight.is_empty() {
                for head in values.chunks_exact_mut(head_dim) {
                    let scale = (head.iter().map(|x| x * x).sum::<f32>() / head_dim as f32 + eps).sqrt().recip();
                    for (x, w) in head.iter_mut().zip(weight.iter()) { *x *= scale * w; }
                }
            }
        }

        // 3. RoPE
        apply_rope(&mut scratch.q, n_head, head_dim, cos, sin, neox);
        apply_rope(&mut scratch.k, n_kv_head, head_dim, cos, sin, neox);

        // 4. KV Cache append
        kv_cache.k.extend_from_slice(&scratch.k);
        kv_cache.v.extend_from_slice(&scratch.v);
        kv_cache.cur_len += 1;

        // 5. Multi-head Attention (parallelized across spin threadpool)
        pool.run_task(GemvTask::Attention {
            q_ptr: scratch.q.as_ptr() as usize,
            k_cache_ptr: kv_cache.k.as_ptr() as usize,
            v_cache_ptr: kv_cache.v.as_ptr() as usize,
            out_ptr: scratch.attn_out.as_mut_ptr() as usize,
            seq_len: kv_cache.cur_len,
            n_head,
            n_kv_head,
            head_dim,
        });

        // 6. Output Projection (wo)
        let wo_slice = &mmap[self.wo.offset..self.wo.offset + self.wo.byte_len];
        if self.wo.dtype == 18 {
            quants::quantize_row_q8_k(&scratch.attn_out, &mut scratch.attn_q8_k);
            pool.run_task(GemvTask::Iq3Xxs {
                out_ptr: scratch.wo_out.as_mut_ptr() as usize,
                weight_ptr: wo_slice.as_ptr() as usize,
                x_q8_k_ptr: scratch.attn_q8_k.as_ptr() as usize,
                n_rows: self.wo.n_rows,
                n_cols: self.wo.n_cols,
            });
        } else if self.wo.dtype == 8 {
            pool.run_task(GemvTask::Q8_0 {
                out_ptr: scratch.wo_out.as_mut_ptr() as usize,
                weight_ptr: wo_slice.as_ptr() as usize,
                x_norm_ptr: scratch.attn_out.as_ptr() as usize,
                n_rows: self.wo.n_rows,
                n_cols: self.wo.n_cols,
            });
        } else {
            quants::quantize_row_q8_0(&scratch.attn_out, &mut scratch.attn_q8_0);
            pool.run_task(GemvTask::Iq4Nl {
                out_ptr: scratch.wo_out.as_mut_ptr() as usize,
                weight_ptr: wo_slice.as_ptr() as usize,
                x_blocks_ptr: scratch.attn_q8_0.as_ptr() as usize,
                n_rows: self.wo.n_rows,
                n_cols: self.wo.n_cols,
            });
        }

        // 7. Residual connection (x = x + wo_out)
        for i in 0..scratch.x.len() {
            scratch.x[i] += scratch.wo_out[i];
        }

        // 8. FFN Norm
        rms_norm_slice(&scratch.x, &self.ffn_norm_weight, eps, &mut scratch.ffn_norm);
        if self.w1.dtype == 18 {
            quants::quantize_row_q8_k(&scratch.ffn_norm, &mut scratch.ffn_q8_k);
        } else {
            quants::quantize_row_q8_0(&scratch.ffn_norm, &mut scratch.ffn_q8_0);
        }

        // 9. FFN Gate (w1) and Up (w3)
        let w1_slice = &mmap[self.w1.offset..self.w1.offset + self.w1.byte_len];
        let w3_slice = &mmap[self.w3.offset..self.w3.offset + self.w3.byte_len];
        if self.w1.dtype == 18 {
            pool.run_task(GemvTask::Iq3Xxs {
                out_ptr: scratch.w1.as_mut_ptr() as usize,
                weight_ptr: w1_slice.as_ptr() as usize,
                x_q8_k_ptr: scratch.ffn_q8_k.as_ptr() as usize,
                n_rows: self.w1.n_rows,
                n_cols: self.w1.n_cols,
            });
            pool.run_task(GemvTask::Iq3Xxs {
                out_ptr: scratch.w3.as_mut_ptr() as usize,
                weight_ptr: w3_slice.as_ptr() as usize,
                x_q8_k_ptr: scratch.ffn_q8_k.as_ptr() as usize,
                n_rows: self.w3.n_rows,
                n_cols: self.w3.n_cols,
            });
            for j in 0..self.w1.n_rows {
                let w1_val = scratch.w1[j];
                let silu = w1_val / (1.0 + (-w1_val).exp());
                scratch.inter[j] = silu * scratch.w3[j];
            }
        } else if self.w1.dtype == 8 {
            pool.run_task(GemvTask::Q8_0 {
                out_ptr: scratch.w1.as_mut_ptr() as usize,
                weight_ptr: w1_slice.as_ptr() as usize,
                x_norm_ptr: scratch.ffn_norm.as_ptr() as usize,
                n_rows: self.w1.n_rows,
                n_cols: self.w1.n_cols,
            });
            pool.run_task(GemvTask::Q8_0 {
                out_ptr: scratch.w3.as_mut_ptr() as usize,
                weight_ptr: w3_slice.as_ptr() as usize,
                x_norm_ptr: scratch.ffn_norm.as_ptr() as usize,
                n_rows: self.w3.n_rows,
                n_cols: self.w3.n_cols,
            });
            for j in 0..self.w1.n_rows {
                let w1_val = scratch.w1[j];
                let silu = w1_val / (1.0 + (-w1_val).exp());
                scratch.inter[j] = silu * scratch.w3[j];
            }
        } else {
            pool.run_task(GemvTask::GateUpIq4Nl {
                w1_out_ptr: scratch.w1.as_mut_ptr() as usize,
                w3_out_ptr: scratch.w3.as_mut_ptr() as usize,
                inter_out_ptr: scratch.inter.as_mut_ptr() as usize,
                w1_ptr: w1_slice.as_ptr() as usize,
                w3_ptr: w3_slice.as_ptr() as usize,
                x_blocks_ptr: scratch.ffn_q8_0.as_ptr() as usize,
                n_rows: self.w1.n_rows,
                n_cols: self.w1.n_cols,
            });
        }

        // 11. Quantize inter to Q8_K and Down Projection (w2)
        quants::quantize_row_q8_k(&scratch.inter, &mut scratch.inter_q8_k);
        let w2_slice = &mmap[self.w2.offset..self.w2.offset + self.w2.byte_len];
        if self.w2.dtype == 8 {
            pool.run_task(GemvTask::Q8_0 {
                out_ptr: scratch.mlp_out.as_mut_ptr() as usize,
                weight_ptr: w2_slice.as_ptr() as usize,
                x_norm_ptr: scratch.inter.as_ptr() as usize,
                n_rows: self.w2.n_rows,
                n_cols: self.w2.n_cols,
            });
        } else {
            pool.run_task(GemvTask::Iq3Xxs {
                out_ptr: scratch.mlp_out.as_mut_ptr() as usize,
                weight_ptr: w2_slice.as_ptr() as usize,
                x_q8_k_ptr: scratch.inter_q8_k.as_ptr() as usize,
                n_rows: self.w2.n_rows,
                n_cols: self.w2.n_cols,
            });
        }

        // 12. Residual connection (x = x + mlp_out)
        for i in 0..scratch.x.len() {
            scratch.x[i] += scratch.mlp_out[i];
        }
    }

    #[inline(always)]
    pub fn madvise_layer(&self, mmap: &[u8]) {
        let page_size = 4096;
        let aligned_start = (self.min_offset / page_size) * page_size;
        let aligned_len = ((self.max_offset - aligned_start + page_size - 1) / page_size) * page_size;
        unsafe {
            libc::madvise(
                mmap.as_ptr().add(aligned_start) as *mut libc::c_void,
                aligned_len,
                libc::MADV_DONTNEED,
            );
        }
    }

    pub fn forward(
        &self,
        mmap: &[u8],
        scratch: &mut ScratchBuffers,
        pool: &SpinThreadPool,
        pos: usize,
        cos: &[f32],
        sin: &[f32],
        neox: bool,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        eps: f32,
        kv_cache: &mut LayerKvCache,
    ) {
        self.forward_token(
            mmap,
            scratch,
            pool,
            pos,
            cos,
            sin,
            neox,
            n_head,
            n_kv_head,
            head_dim,
            eps,
            kv_cache,
        );
        self.madvise_layer(mmap);
    }
}

fn precompute_freqs_cis_flat(
    head_dim: usize,
    freq_base: f32,
    context_length: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos_table = Vec::with_capacity(context_length * half);
    let mut sin_table = Vec::with_capacity(context_length * half);
    let theta: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();

    for pos in 0..context_length {
        for i in 0..half {
            let angle = pos as f32 * theta[i];
            cos_table.push(angle.cos());
            sin_table.push(angle.sin());
        }
    }
    (cos_table, sin_table)
}

pub struct AstMicroEngine {
    mmap: Mmap,
    gguf: gguf::GgufModel,
    pub block_count: usize,
    pub vocab_size: usize,
    head_count: usize,
    head_count_kv: usize,
    head_dim: usize,
    _embedding_length: usize,
    rms_norm_eps: f64,
    cos_table: Vec<f32>,
    sin_table: Vec<f32>,
    rope_neox: bool,
    output_norm_weight: Vec<f32>,
    token_embd_offset: usize,
    token_embd_dtype: u32,
    bytes_per_token: usize,
    output_weight: Option<LinearSlice>,
    kv_caches: Vec<LayerKvCache>,
    layers: Vec<LayerDesc>,
    scratch: ScratchBuffers,
    sampling_temp: f32,
    sampling_top_p: f32,
    repeat_penalty: f32,
    repeat_last_n: usize,
    pub suppressed_tokens: HashSet<u32>,
    pub turbo: bool,
    pub pool: SpinThreadPool,
    pub resident_layers: usize,
}

impl AstMicroEngine {
    pub fn eos_token_id(&self) -> Option<u32> {
        self.gguf.metadata.get("tokenizer.ggml.eos_token_id").and_then(|v| v.to_u32().ok())
    }

    pub fn tokenizer_tokens(&self) -> Result<Vec<String>> {
        match self.gguf.metadata.get("tokenizer.ggml.tokens") {
            Some(gguf::GgufValue::Array(tokens)) => tokens.iter().map(|t| t.to_string()).collect(),
            _ => Err(anyhow!("model is missing tokenizer.ggml.tokens")),
        }
    }

    pub fn new(model_path: &Path) -> Result<Self> {
        let file = File::open(model_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        #[cfg(target_os = "linux")]
        unsafe {
            // Advise Transparent HugePages (THP) for collapsing 4KB page walks into 2MB entries
            libc::madvise(
                mmap.as_ptr() as *mut libc::c_void,
                mmap.len(),
                libc::MADV_HUGEPAGE,
            );
        }

        let mut cursor = std::io::Cursor::new(&mmap[..]);
        let gguf = gguf::GgufModel::read(&mut cursor)?;

        let arch: String = gguf
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .unwrap_or_else(|| "llama".to_string());
        let rope_neox = matches!(arch.as_str(), "qwen2" | "qwen3");

        let md_get = |s: &str| -> Result<&gguf::GgufValue> {
            gguf.metadata
                .get(&format!("{arch}.{s}"))
                .or_else(|| gguf.metadata.get(&format!("llama.{s}")))
                .or_else(|| gguf.metadata.get(&format!("qwen2.{s}")))
                .ok_or_else(|| anyhow!("missing metadata {arch}.{s}"))
        };

        let head_count = md_get("attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("attention.head_count_kv")?.to_u32()? as usize;
        let embedding_length = md_get("embedding_length")?.to_u32()? as usize;
        let context_length = (md_get("context_length")?.to_u32()? as usize).min(4096);
        let block_count = md_get("block_count")?.to_u32()? as usize;
        let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("rope.freq_base")
            .ok()
            .and_then(|m| m.to_f32().ok())
            .unwrap_or(10000.0);

        let head_dim = embedding_length / head_count;
        let (cos_table, sin_table) = precompute_freqs_cis_flat(head_dim, rope_freq_base, context_length);

        let output_norm_weight = Self::read_weight_vec(&mmap, &gguf, "output_norm.weight")?;

        let embd_info = gguf
            .tensors
            .get("token_embd.weight")
            .ok_or_else(|| anyhow!("missing token_embd.weight"))?;
        let token_embd_offset = gguf.tensor_data_offset + embd_info.offset as usize;
        let token_embd_dtype = embd_info.dtype;
        if token_embd_dtype != 8 {
            anyhow::bail!("AST-Micro requires Q8_0 token embeddings, found dtype {token_embd_dtype}");
        }

        let (block_size, type_size) = match token_embd_dtype {
            0 => (1, 4),
            2 => (32, 18),
            8 => (32, 34),
            18 => (256, 98),
            20 => (32, 18),
            other => anyhow::bail!("unsupported token_embd dtype {other}"),
        };
        let bytes_per_token = (embedding_length / block_size) * type_size;

        // Llama chat checkpoints may keep a distinct output head. Preserve and
        // use it when present; older AST artifacts omit it and use tied embeddings.
        let output_weight = if let Some(meta) = gguf.tensors.get("output.weight") {
            if meta.shape.len() != 2
                || meta.shape[0] != embd_info.shape[0]
                || meta.shape[1] != embedding_length
            {
                anyhow::bail!("output.weight shape {:?} does not match vocab/embedding dimensions", meta.shape);
            }
            if meta.dtype != 8 {
                anyhow::bail!("AST-Micro separate output.weight requires Q8_0, found dtype {}", meta.dtype);
            }
            Some(LinearSlice::new(meta, gguf.tensor_data_offset))
        } else {
            None
        };

        let kv_caches = vec![LayerKvCache::default(); block_count];

        let mut layers = Vec::with_capacity(block_count);
        let mut max_wq_rows = 0;
        let mut max_wk_rows = 0;
        let mut max_wv_rows = 0;
        let mut max_w1_rows = 0;

        for idx in 0..block_count {
            let prefix = format!("blk.{idx}");
            let meta_wq = gguf.tensors.get(&format!("{prefix}.attn_q.weight")).ok_or_else(|| anyhow!("missing {prefix}.attn_q.weight"))?;
            let meta_wk = gguf.tensors.get(&format!("{prefix}.attn_k.weight")).ok_or_else(|| anyhow!("missing {prefix}.attn_k.weight"))?;
            let meta_wv = gguf.tensors.get(&format!("{prefix}.attn_v.weight")).ok_or_else(|| anyhow!("missing {prefix}.attn_v.weight"))?;
            let meta_wo = gguf.tensors.get(&format!("{prefix}.attn_output.weight")).ok_or_else(|| anyhow!("missing {prefix}.attn_output.weight"))?;
            let meta_w1 = gguf.tensors.get(&format!("{prefix}.ffn_gate.weight")).ok_or_else(|| anyhow!("missing {prefix}.ffn_gate.weight"))?;
            let meta_w2 = gguf.tensors.get(&format!("{prefix}.ffn_down.weight")).ok_or_else(|| anyhow!("missing {prefix}.ffn_down.weight"))?;
            let meta_w3 = gguf.tensors.get(&format!("{prefix}.ffn_up.weight")).ok_or_else(|| anyhow!("missing {prefix}.ffn_up.weight"))?;

            if [meta_wq, meta_wk, meta_wv, meta_wo, meta_w1, meta_w2, meta_w3].iter().any(|meta| meta.dtype == 8)
                && std::env::var_os("NIOXS_AST_EXPERIMENTAL_Q8").as_deref() != Some(std::ffi::OsStr::new("1"))
            {
                anyhow::bail!("Q8_0 projection support is experimental; set NIOXS_AST_EXPERIMENTAL_Q8=1 to opt in for evaluation");
            }

            for meta in [meta_wq, meta_wk, meta_wv, meta_wo, meta_w1, meta_w3] {
                if ![8, 20].contains(&meta.dtype) || meta.shape.len() != 2 || meta.shape[1] % 32 != 0 {
                    anyhow::bail!("unsupported AST-Micro tensor {}: dtype {} shape {:?}; expected Q8_0 or IQ4_NL matrix", meta.name, meta.dtype, meta.shape);
                }
            }
            if meta_w2.shape.len() != 2 || !matches!(meta_w2.dtype, 8 | 18) || (meta_w2.dtype == 18 && meta_w2.shape[1] % 256 != 0) || (meta_w2.dtype == 8 && meta_w2.shape[1] % 32 != 0) {
                anyhow::bail!("unsupported AST-Micro tensor {}: dtype {} shape {:?}; expected Q8_0 or IQ3_XXS matrix", meta_w2.name, meta_w2.dtype, meta_w2.shape);
            }

            let wq = LinearSlice::new(meta_wq, gguf.tensor_data_offset);
            let wk = LinearSlice::new(meta_wk, gguf.tensor_data_offset);
            let wv = LinearSlice::new(meta_wv, gguf.tensor_data_offset);
            let wo = LinearSlice::new(meta_wo, gguf.tensor_data_offset);
            let w1 = LinearSlice::new(meta_w1, gguf.tensor_data_offset);
            let w2 = LinearSlice::new(meta_w2, gguf.tensor_data_offset);
            let w3 = LinearSlice::new(meta_w3, gguf.tensor_data_offset);

            max_wq_rows = max_wq_rows.max(wq.n_rows);
            max_wk_rows = max_wk_rows.max(wk.n_rows);
            max_wv_rows = max_wv_rows.max(wv.n_rows);
            max_w1_rows = max_w1_rows.max(w1.n_rows);

            let min_offset = [wq.offset, wk.offset, wv.offset, wo.offset, w1.offset, w2.offset, w3.offset].into_iter().min().unwrap();
            let max_offset = [
                wq.offset + wq.byte_len,
                wk.offset + wk.byte_len,
                wv.offset + wv.byte_len,
                wo.offset + wo.byte_len,
                w1.offset + w1.byte_len,
                w2.offset + w2.byte_len,
                w3.offset + w3.byte_len,
            ].into_iter().max().unwrap();

            let q_bias = Self::read_optional_weight_vec(&mmap, &gguf, &format!("{prefix}.attn_q.bias"))?;
            let k_bias = Self::read_optional_weight_vec(&mmap, &gguf, &format!("{prefix}.attn_k.bias"))?;
            let v_bias = Self::read_optional_weight_vec(&mmap, &gguf, &format!("{prefix}.attn_v.bias"))?;
            let q_norm_weight = Self::read_optional_weight_vec(&mmap, &gguf, &format!("{prefix}.attn_q_norm.weight"))?;
            let k_norm_weight = Self::read_optional_weight_vec(&mmap, &gguf, &format!("{prefix}.attn_k_norm.weight"))?;
            if arch == "qwen3" && (q_norm_weight.len() != head_dim || k_norm_weight.len() != head_dim) {
                anyhow::bail!("Qwen3 layer {idx} requires query/key norm weights of length {head_dim}");
            }
            if (!q_norm_weight.is_empty() && q_norm_weight.len() != head_dim) || (!k_norm_weight.is_empty() && k_norm_weight.len() != head_dim) {
                anyhow::bail!("invalid attention normalization weights in layer {idx}");
            }
            let attn_norm_weight = Self::read_weight_vec(&mmap, &gguf, &format!("{prefix}.attn_norm.weight"))?;
            let ffn_norm_weight = Self::read_weight_vec(&mmap, &gguf, &format!("{prefix}.ffn_norm.weight"))?;

            layers.push(LayerDesc {
                idx,
                wq,
                wk,
                wv,
                wo,
                w1,
                w2,
                w3,
                q_bias,
                k_bias,
                v_bias,
                q_norm_weight,
                k_norm_weight,
                attn_norm_weight,
                ffn_norm_weight,
                min_offset,
                max_offset,
            });
        }

        let scratch = ScratchBuffers::new(embedding_length, max_wq_rows, max_wk_rows, max_wv_rows, max_w1_rows);

        let vocab_size = if let Some(output) = &output_weight {
            output.n_rows
        } else {
            embd_info.shape[0]
        };

        let mut engine = Self {
            mmap,
            gguf,
            block_count,
            vocab_size,
            head_count,
            head_count_kv,
            head_dim,
            _embedding_length: embedding_length,
            rms_norm_eps,
            cos_table,
            sin_table,
            rope_neox,
            output_norm_weight,
            token_embd_offset,
            token_embd_dtype,
            bytes_per_token,
            output_weight,
            kv_caches,
            layers,
            scratch,
            sampling_temp: 0.7,
            sampling_top_p: 0.9,
            repeat_penalty: 1.15,
            repeat_last_n: 64,
            suppressed_tokens: HashSet::new(),
            turbo: false,
            pool: {
                let pool = SpinThreadPool::new();
                pool.park();
                pool
            },
            resident_layers: 0,
        };
        engine.evict_all();
        unsafe {
            libc::malloc_trim(0);
        }
        Ok(engine)
    }

    #[inline(always)]
    pub fn set_resident_layers(&mut self, n: usize) {
        self.resident_layers = n;
    }

    #[inline(always)]
    pub fn set_turbo(&mut self, enabled: bool) {
        self.turbo = enabled;
    }

    #[inline(always)]
    pub fn unpark(&self) {
        self.pool.unpark();
    }

    #[inline(always)]
    pub fn park(&self) {
        self.pool.park();
    }

    #[inline(always)]
    pub fn madvise_dontneed_range(mmap: &[u8], offset: usize, len: usize) {
        let page_size = 4096;
        let aligned_start = (offset / page_size) * page_size;
        let aligned_len = ((offset + len - aligned_start + page_size - 1) / page_size) * page_size;
        unsafe {
            libc::madvise(
                mmap.as_ptr().add(aligned_start) as *mut libc::c_void,
                aligned_len,
                libc::MADV_DONTNEED,
            );
        }
    }

    pub fn set_suppressed_tokens(&mut self, tokens: impl IntoIterator<Item = u32>) {
        self.suppressed_tokens = tokens.into_iter().collect();
    }

    pub fn add_suppressed_token(&mut self, token: u32) {
        self.suppressed_tokens.insert(token);
    }

    pub fn set_sampling(&mut self, temperature: f32, top_p: f32, repeat_penalty: f32, repeat_last_n: usize) {
        if temperature >= 0.0 {
            self.sampling_temp = temperature.clamp(0.0, 2.0);
        }
        if top_p > 0.0 && top_p <= 1.0 {
            self.sampling_top_p = top_p;
        }
        if repeat_penalty >= 1.0 {
            self.repeat_penalty = repeat_penalty.clamp(1.0, 2.0);
        }
        if repeat_last_n > 0 {
            self.repeat_last_n = repeat_last_n;
        }
    }

    pub fn reset_cache(&mut self) {
        for c in &mut self.kv_caches {
            c.reset();
        }
        self.evict_all();
        self.pool.park();
        unsafe {
            libc::malloc_trim(0);
        }
    }

    pub fn truncate_cache(&mut self, target_len: usize) {
        let kv_stride = self.head_count_kv * self.head_dim;
        for c in &mut self.kv_caches {
            c.truncate(target_len, kv_stride);
        }
    }

    pub fn evict_all(&mut self) {
        unsafe {
            libc::madvise(
                self.mmap.as_ptr() as *mut libc::c_void,
                self.mmap.len(),
                libc::MADV_DONTNEED,
            );
            #[cfg(target_os = "linux")]
            libc::madvise(
                self.mmap.as_ptr() as *mut libc::c_void,
                self.mmap.len(),
                libc::MADV_HUGEPAGE,
            );
        }
    }

    fn read_weight_vec(mmap: &Mmap, gguf: &gguf::GgufModel, name: &str) -> Result<Vec<f32>> {
        let meta = gguf
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))?;
        let start = gguf.tensor_data_offset + meta.offset as usize;
        let elem_count: usize = meta.shape.iter().product();
        let (block_size, type_size) = match meta.dtype {
            0 => (1, 4),
            2 => (32, 18),
            8 => (32, 34),
            18 => (256, 98),
            20 => (32, 18),
            other => anyhow::bail!("unsupported tensor dtype {other} for {name}"),
        };
        let byte_len = (elem_count / block_size) * type_size;
        let slice = &mmap[start..start + byte_len];

        let mut data = vec![0f32; elem_count];
        match meta.dtype {
            0 => quants::dequantize_row_f32(slice, &mut data),
            2 => quants::dequantize_row_q4_0(slice, &mut data),
            8 => quants::dequantize_row_q8_0(slice, &mut data),
            18 => quants::dequantize_row_iq3_xxs(slice, &mut data),
            20 => quants::dequantize_row_iq4_nl(slice, &mut data),
            other => anyhow::bail!("unimplemented dequantize for dtype {other}"),
        }

        Ok(data)
    }

    fn read_optional_weight_vec(mmap: &Mmap, gguf: &gguf::GgufModel, name: &str) -> Result<Vec<f32>> {
        if gguf.tensors.contains_key(name) {
            Self::read_weight_vec(mmap, gguf, name)
        } else {
            Ok(Vec::new())
        }
    }

    #[inline(always)]
    fn embed_token_slice(mmap: &[u8], offset: usize, bytes_per_token: usize, dtype: u32, tok: u32, out: &mut [f32]) -> Result<()> {
        let start = offset + (tok as usize) * bytes_per_token;
        let end = start + bytes_per_token;
        let slice = &mmap[start..end];
        match dtype {
            8 => quants::dequantize_row_q8_0(slice, out),
            2 => quants::dequantize_row_q4_0(slice, out),
            0 => quants::dequantize_row_f32(slice, out),
            other => anyhow::bail!("unsupported token_embd dtype {other}"),
        }
        Ok(())
    }

    /// Forward multiple tokens in a layer-outer fashion, loading and evicting each layer's weights
    /// into memory only ONCE for the entire batch of tokens.
    /// Returns the normalized hidden state (after rms_norm) for each token in `tokens`.
    pub fn forward_batch_layer_outer(
        &mut self,
        tokens: &[u32],
        index_pos: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let n_tokens = tokens.len();
        if n_tokens == 0 {
            return Ok(Vec::new());
        }

        self.pool.unpark();

        let half = self.head_dim / 2;
        let limit = self.cos_table.len() / half;
        if index_pos.checked_add(n_tokens).is_none_or(|end| end > limit) {
            return Err(anyhow!("context exceeds model runtime limit of {limit} tokens"));
        }
        let vocab = self.gguf.tensors.get("token_embd.weight")
            .and_then(|t| t.shape.first()).copied().ok_or_else(|| anyhow!("invalid token embedding shape"))?;
        if let Some(token) = tokens.iter().find(|&&token| token as usize >= vocab) {
            return Err(anyhow!("token id {token} exceeds model vocabulary size {vocab}"));
        }

        let token_embd_offset = self.token_embd_offset;
        let bytes_per_token = self.bytes_per_token;
        let token_embd_dtype = self.token_embd_dtype;
        let embd_len = self._embedding_length;

        // 1. Embed all tokens into activation vectors
        let mut activations: Vec<Vec<f32>> = Vec::with_capacity(n_tokens);
        for &tok in tokens {
            let mut x = vec![0.0f32; embd_len];
            Self::embed_token_slice(
                &self.mmap,
                token_embd_offset,
                bytes_per_token,
                token_embd_dtype,
                tok,
                &mut x,
            )?;
            activations.push(x);
        }

        // 2. Pass all tokens through layers, layer-by-layer (layer-outer)
        for layer_idx in 0..self.block_count {
            let layer = &self.layers[layer_idx];
            for step in 0..n_tokens {
                let pos = index_pos + step;
                let cos = &self.cos_table[pos * half..(pos + 1) * half];
                let sin = &self.sin_table[pos * half..(pos + 1) * half];

                self.scratch.x.copy_from_slice(&activations[step]);
                layer.forward_token(
                    &self.mmap,
                    &mut self.scratch,
                    &self.pool,
                    pos,
                    cos,
                    sin,
                    self.rope_neox,
                    self.head_count,
                    self.head_count_kv,
                    self.head_dim,
                    self.rms_norm_eps as f32,
                    &mut self.kv_caches[layer_idx],
                );
                activations[step].copy_from_slice(&self.scratch.x);
            }

            // Layer weights were used for all tokens; evict once per layer if above resident threshold
            if layer_idx >= self.resident_layers {
                layer.madvise_layer(&self.mmap);
            }
        }

        // 3. Output RMS norm for each token
        let mut out_norms = Vec::with_capacity(n_tokens);
        for x in &activations {
            let mut norm = vec![0.0f32; embd_len];
            rms_norm_slice(
                x,
                &self.output_norm_weight,
                self.rms_norm_eps as f32,
                &mut norm,
            );
            out_norms.push(norm);
        }

        // Keep self.scratch.x and self.scratch.x_norm aligned with the last token
        if let Some(last_norm) = out_norms.last() {
            self.scratch.x_norm.copy_from_slice(last_norm);
        }
        if let Some(last_x) = activations.last() {
            self.scratch.x.copy_from_slice(last_x);
        }

        Ok(out_norms)
    }

    pub fn forward_stream(&mut self, tokens: &[u32], index_pos: usize) -> Result<Tensor> {
        self.forward_batch_layer_outer(tokens, index_pos)?;
        // Return dummy 1-element tensor for caller interface compatibility
        Tensor::new(&[0.0f32], &Device::Cpu).map_err(|e| anyhow!("{e}"))
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Tensor> {
        self.forward_stream(tokens, 0)
    }

    /// Return unsampled logits after prefill/forward for runtime diagnostics.
    pub fn diagnostic_logits(&mut self) -> Result<Vec<f32>> {
        quants::quantize_row_q8_0(&self.scratch.x_norm, &mut self.scratch.x_q8_0);
        let (offset, vocab_size, row_bytes) = if let Some(output) = &self.output_weight {
            (output.offset, output.n_rows, (output.n_cols / 32) * 34)
        } else {
            let embedding = self.gguf.tensors.get("token_embd.weight")
                .ok_or_else(|| anyhow!("missing token embeddings"))?;
            (self.token_embd_offset, *embedding.shape.first().ok_or_else(|| anyhow!("invalid embedding shape"))?, self.bytes_per_token)
        };
        let end = offset.checked_add(vocab_size.checked_mul(row_bytes)
            .ok_or_else(|| anyhow!("embedding size overflow"))?).ok_or_else(|| anyhow!("embedding offset overflow"))?;
        let bytes = self.mmap.get(offset..end).ok_or_else(|| anyhow!("truncated output matrix"))?;
        use rayon::prelude::*;
        Ok(bytes.par_chunks_exact(row_bytes)
            .map(|row| unsafe { quants::vec_dot_q8_0_q8_0_avx2(row, &self.scratch.x_q8_0) }).collect())
    }

    pub fn next_token(&mut self, _hidden: &Tensor, candidates: Option<&[u32]>) -> Result<u32> {
        self.next_token_with_history(candidates, &[], false)
    }

    pub fn next_token_with_history(
        &mut self,
        candidates: Option<&[u32]>,
        history: &[u32],
        suppress_eos: bool,
    ) -> Result<u32> {
        let x_norm = self.scratch.x_norm.clone();
        self.sample_token_from_x_norm(&x_norm, candidates, history, suppress_eos)
    }

    pub fn sample_token_from_x_norm(
        &mut self,
        x_norm: &[f32],
        candidates: Option<&[u32]>,
        history: &[u32],
        suppress_eos: bool,
    ) -> Result<u32> {
        self.pool.unpark();
        quants::quantize_row_q8_0(x_norm, &mut self.scratch.x_q8_0);
        let repeat_penalty = self.repeat_penalty;
        let recent: HashSet<u32> = history
            .iter()
            .rev()
            .take(self.repeat_last_n.max(1))
            .copied()
            .collect();

        if let Some(cands) = candidates {
            if cands.is_empty() {
                return Ok(2); // EOS
            }
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;
            let (matrix_offset, row_bytes) = if let Some(output) = &self.output_weight {
                (output.offset, (output.n_cols / 32) * 34)
            } else {
                (self.token_embd_offset, self.bytes_per_token)
            };
            for (i, &tok_id) in cands.iter().enumerate() {
                if suppress_eos && (tok_id == 0 || tok_id == 1 || tok_id == 2) {
                    continue;
                }
                if self.suppressed_tokens.contains(&tok_id) {
                    continue;
                }
                let start = matrix_offset + (tok_id as usize) * row_bytes;
                let slice = &self.mmap[start..start + row_bytes];
                let val = unsafe { quants::vec_dot_q8_0_q8_0_avx2(slice, &self.scratch.x_q8_0) };
                let mut score = val;
                if recent.contains(&tok_id) {
                    score = if score > 0.0 {
                        score / repeat_penalty
                    } else {
                        score * repeat_penalty
                    };
                }
                if score > max_val {
                    max_val = score;
                    max_idx = i;
                }
            }
            Ok(cands[max_idx])
        } else {
            let (matrix_offset, vocab_size, bytes_per_token) = if let Some(output) = &self.output_weight {
                (output.offset, output.n_rows, (output.n_cols / 32) * 34)
            } else {
                let embd_info = self.gguf.tensors.get("token_embd.weight").unwrap();
                (self.token_embd_offset, embd_info.shape[0], self.bytes_per_token)
            };

            let suppressed_vec: Vec<u32> = self.suppressed_tokens.iter().copied().collect();
            let mut top_toks = [0u32; pool::NUM_WORKERS];
            let mut top_scores = [f32::NEG_INFINITY; pool::NUM_WORKERS];

            self.pool.run_task(GemvTask::VocabQ8_0 {
                weight_ptr: (self.mmap.as_ptr() as usize) + matrix_offset,
                x_blocks_ptr: self.scratch.x_q8_0.as_ptr() as usize,
                start_tok: 0,
                vocab_count: vocab_size,
                bytes_per_token,
                suppress_eos,
                history_ptr: history.as_ptr() as usize,
                history_len: history.len(),
                suppressed_ptr: suppressed_vec.as_ptr() as usize,
                suppressed_len: suppressed_vec.len(),
                repeat_penalty: self.repeat_penalty,
                top_tok_ptr: top_toks.as_mut_ptr() as usize,
                top_score_ptr: top_scores.as_mut_ptr() as usize,
            });

            let mut best_token = 0u32;
            let mut max_score = f32::NEG_INFINITY;
            for i in 0..pool::NUM_WORKERS {
                if top_scores[i] > max_score {
                    max_score = top_scores[i];
                    best_token = top_toks[i];
                }
            }

            Ok(best_token)
        }
    }

    /// Perform one speculative decoding step with Prompt Lookup Decoding (PLD).
    /// Uses layer-outer multi-token evaluation to amortize memory bandwidth across draft tokens,
    /// and strictly enforces 100.000% mathematical parity with autoregressive decoding.
    pub fn speculative_pld_step(
        &mut self,
        all_tokens: &[u32],
        current_confirmed: u32,
        current_pos: usize,
        gen_history: &[u32],
        max_draft: usize,
        eos_tokens: &[u32],
    ) -> Result<SpeculativeStepResult> {
        let is_eos = |t: u32| eos_tokens.contains(&t);
        let drafts = if max_draft > 0 {
            find_candidate_draft_tokens(all_tokens, max_draft, eos_tokens)
        } else {
            Vec::new()
        };

        let mut batch = Vec::with_capacity(1 + drafts.len());
        batch.push(current_confirmed);
        batch.extend_from_slice(&drafts);

        let out_norms = self.forward_batch_layer_outer(&batch, current_pos)?;

        let mut matched_drafts = 0;
        let mut final_token = 0;
        let mut final_is_eos = false;
        let mut accepted_tokens = Vec::with_capacity(batch.len() + 1);
        let mut local_history: Vec<u32> = gen_history.to_vec();

        for i in 0..=drafts.len() {
            let true_tok = self.sample_token_from_x_norm(&out_norms[i], None, &local_history, false)?;
            if is_eos(true_tok) {
                final_token = true_tok;
                final_is_eos = true;
                break;
            }

            if i < drafts.len() && true_tok == drafts[i] {
                matched_drafts += 1;
                accepted_tokens.push(drafts[i]);
                local_history.push(drafts[i]);
            } else {
                final_token = true_tok;
                break;
            }
        }

        let next_pos = current_pos + 1 + matched_drafts;
        self.truncate_cache(next_pos);

        if final_token != 0 {
            accepted_tokens.push(final_token);
        }

        let next_confirmed = final_token;

        Ok(SpeculativeStepResult {
            accepted_tokens,
            next_confirmed,
            next_pos,
            is_eos: final_is_eos,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SpeculativeStepResult {
    pub accepted_tokens: Vec<u32>,
    pub next_confirmed: u32,
    pub next_pos: usize,
    pub is_eos: bool,
}

/// Find up to `max_draft` candidate tokens from historical context using prompt lookup.
/// Searches backwards for matching n-grams of lengths 5, 4, down to 3.
/// Minimum n-gram length is 3 to guarantee high-confidence draft continuations and avoid
/// low-entropy thrashing on common bigrams/whitespace.
pub fn find_candidate_draft_tokens(
    context: &[u32],
    max_draft: usize,
    eos_tokens: &[u32],
) -> Vec<u32> {
    if context.len() < 4 || max_draft == 0 {
        return Vec::new();
    }

    let is_eos = |t: u32| eos_tokens.contains(&t);

    for ngram_len in (3..=5.min(context.len().saturating_sub(1))).rev() {
        let query = &context[context.len() - ngram_len..];
        let search_limit = context.len().saturating_sub(ngram_len);
        if search_limit == 0 {
            continue;
        }

        for match_idx in (0..search_limit).rev() {
            if &context[match_idx..match_idx + ngram_len] == query {
                let candidate_start = match_idx + ngram_len;
                let max_take = (context.len() - ngram_len).saturating_sub(candidate_start).min(max_draft);
                if max_take > 0 {
                    let mut candidates = Vec::with_capacity(max_take);
                    for &t in &context[candidate_start..candidate_start + max_take] {
                        if is_eos(t) {
                            break;
                        }
                        candidates.push(t);
                    }
                    if !candidates.is_empty() {
                        return candidates;
                    }
                }
            }
        }
    }

    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_candidate_draft_tokens_basic() {
        let context = vec![10, 20, 30, 40, 50, 99, 10, 20, 30];
        let drafts = find_candidate_draft_tokens(&context, 2, &[2]);
        assert_eq!(drafts, vec![40, 50]);
    }

    #[test]
    fn test_find_candidate_draft_tokens_recency() {
        let context = vec![10, 20, 30, 1, 2, 3, 10, 20, 30, 88, 99, 4, 5, 10, 20, 30];
        let drafts = find_candidate_draft_tokens(&context, 2, &[2]);
        assert_eq!(drafts, vec![88, 99]);
    }

    #[test]
    fn test_find_candidate_draft_tokens_eos_cutoff() {
        let eos = 2;
        let context = vec![10, 20, 30, 40, eos, 50, 10, 20, 30];
        let drafts = find_candidate_draft_tokens(&context, 3, &[eos]);
        assert_eq!(drafts, vec![40]);
    }

    #[test]
    fn test_layer_kv_cache_truncate() {
        let mut cache = LayerKvCache::default();
        let kv_stride = 128;
        cache.k = vec![1.0; 10 * kv_stride];
        cache.v = vec![2.0; 10 * kv_stride];
        cache.cur_len = 10;

        cache.truncate(6, kv_stride);
        assert_eq!(cache.cur_len, 6);
        assert_eq!(cache.k.len(), 6 * kv_stride);
        assert_eq!(cache.v.len(), 6 * kv_stride);

        cache.truncate(8, kv_stride);
        assert_eq!(cache.cur_len, 6);
        assert_eq!(cache.k.len(), 6 * kv_stride);
    }
}
