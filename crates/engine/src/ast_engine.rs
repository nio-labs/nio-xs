use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use memmap2::Mmap;
use std::collections::HashSet;
use std::fs::File;
use std::path::Path;

use crate::gguf;
use crate::quants;

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
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub attn_q8_0: Vec<quants::BlockQ8_0>,
    pub wo_out: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub ffn_q8_0: Vec<quants::BlockQ8_0>,
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
            x_q8_0: vec![quants::BlockQ8_0 { d: half::f16::ZERO, qs: [0; 32] }; embedding_length / 32],
            q: vec![0f32; wq_rows],
            k: vec![0f32; wk_rows],
            v: vec![0f32; wv_rows],
            attn_out: vec![0f32; wq_rows],
            attn_q8_0: vec![quants::BlockQ8_0 { d: half::f16::ZERO, qs: [0; 32] }; wq_rows / 32],
            wo_out: vec![0f32; embedding_length],
            ffn_norm: vec![0f32; embedding_length],
            ffn_q8_0: vec![quants::BlockQ8_0 { d: half::f16::ZERO, qs: [0; 32] }; embedding_length / 32],
            w1: vec![0f32; w1_rows],
            w3: vec![0f32; w1_rows],
            inter: vec![0f32; w1_rows],
            inter_q8_k: vec![quants::BlockQ8_K { d: 0.0, qs: [0; 256], bsums: [0; 16] }; w1_rows / 256],
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
    pub fn forward(
        &self,
        mmap: &[u8],
        scratch: &mut ScratchBuffers,
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
        quants::quantize_row_q8_0(&scratch.x_norm, &mut scratch.x_q8_0);

        // 2. Q, K, V Projections (Pure AVX2 in L1 cache)
        let wq_slice = &mmap[self.wq.offset..self.wq.offset + self.wq.byte_len];
        let wk_slice = &mmap[self.wk.offset..self.wk.offset + self.wk.byte_len];
        let wv_slice = &mmap[self.wv.offset..self.wv.offset + self.wv.byte_len];
        for (out, weight, desc) in [(&mut scratch.q, wq_slice, &self.wq), (&mut scratch.k, wk_slice, &self.wk), (&mut scratch.v, wv_slice, &self.wv)] {
            if desc.dtype == 8 { quants::gemv_q8_0(out, weight, &scratch.x_norm, desc.n_rows, desc.n_cols); }
            else { unsafe { quants::gemv_iq4_nl_prequant(out, weight, &scratch.x_q8_0, desc.n_rows, desc.n_cols); } }
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

        // Qwen3 normalizes each query and key head before applying RoPE.
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

        // 5. Multi-head Attention
        compute_attention(
            &scratch.q,
            &kv_cache.k,
            &kv_cache.v,
            kv_cache.cur_len,
            n_head,
            n_kv_head,
            head_dim,
            &mut scratch.attn_out,
            &mut scratch.attn_scores,
        );

        // 6. Output Projection (wo)
        quants::quantize_row_q8_0(&scratch.attn_out, &mut scratch.attn_q8_0);
        let wo_slice = &mmap[self.wo.offset..self.wo.offset + self.wo.byte_len];
        if self.wo.dtype == 8 { quants::gemv_q8_0(&mut scratch.wo_out, wo_slice, &scratch.attn_out, self.wo.n_rows, self.wo.n_cols); }
        else { unsafe { quants::gemv_iq4_nl_prequant(&mut scratch.wo_out, wo_slice, &scratch.attn_q8_0, self.wo.n_rows, self.wo.n_cols); } }

        // 7. Residual connection (x = x + wo_out)
        for i in 0..scratch.x.len() {
            scratch.x[i] += scratch.wo_out[i];
        }

        // 8. FFN Norm
        rms_norm_slice(&scratch.x, &self.ffn_norm_weight, eps, &mut scratch.ffn_norm);
        quants::quantize_row_q8_0(&scratch.ffn_norm, &mut scratch.ffn_q8_0);

        // 9. FFN Gate (w1) and Up (w3)
        let w1_slice = &mmap[self.w1.offset..self.w1.offset + self.w1.byte_len];
        let w3_slice = &mmap[self.w3.offset..self.w3.offset + self.w3.byte_len];
        for (out, weight, desc) in [(&mut scratch.w1, w1_slice, &self.w1), (&mut scratch.w3, w3_slice, &self.w3)] {
            if desc.dtype == 8 { quants::gemv_q8_0(out, weight, &scratch.ffn_norm, desc.n_rows, desc.n_cols); }
            else { unsafe { quants::gemv_iq4_nl_prequant(out, weight, &scratch.ffn_q8_0, desc.n_rows, desc.n_cols); } }
        }

        // 10. SwiGLU activation: inter = SiLU(w1) * w3
        for j in 0..self.w1.n_rows {
            let w1_val = scratch.w1[j];
            let silu = w1_val / (1.0 + (-w1_val).exp());
            scratch.inter[j] = silu * scratch.w3[j];
        }

        // 11. Quantize inter to Q8_K and Down Projection (w2)
        let w2_slice = &mmap[self.w2.offset..self.w2.offset + self.w2.byte_len];
        if self.w2.dtype == 8 { quants::gemv_q8_0(&mut scratch.mlp_out, w2_slice, &scratch.inter, self.w2.n_rows, self.w2.n_cols); }
        else {
            quants::quantize_row_q8_k(&scratch.inter, &mut scratch.inter_q8_k);
            unsafe { quants::gemv_iq3_xxs_prequant(&mut scratch.mlp_out, w2_slice, &scratch.inter_q8_k, self.w2.n_rows, self.w2.n_cols); }
        }

        // 12. Residual connection (x = x + mlp_out)
        for i in 0..scratch.x.len() {
            scratch.x[i] += scratch.mlp_out[i];
        }

        // 13. madvise each layer to keep Peak RAM safely locked around ~68-71MB
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
    block_count: usize,
    head_count: usize,
    head_count_kv: usize,
    head_dim: usize,
    embedding_length: usize,
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

        Ok(Self {
            mmap,
            gguf,
            block_count,
            head_count,
            head_count_kv,
            head_dim,
            embedding_length,
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
        })
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
        unsafe {
            libc::malloc_trim(0);
        }
    }

    pub fn evict_all(&mut self) {
        unsafe {
            libc::madvise(
                self.mmap.as_ptr() as *mut libc::c_void,
                self.mmap.len(),
                libc::MADV_DONTNEED,
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

    pub fn forward_stream(&mut self, tokens: &[u32], index_pos: usize) -> Result<Tensor> {
        let half = self.head_dim / 2;
        let limit = self.cos_table.len() / half;
        if index_pos.checked_add(tokens.len()).is_none_or(|end| end > limit) {
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

        for (step, &tok) in tokens.iter().enumerate() {
            let pos = index_pos + step;
            let cos = &self.cos_table[pos * half..(pos + 1) * half];
            let sin = &self.sin_table[pos * half..(pos + 1) * half];

            Self::embed_token_slice(
                &self.mmap,
                token_embd_offset,
                bytes_per_token,
                token_embd_dtype,
                tok,
                &mut self.scratch.x,
            )?;

            for layer_idx in 0..self.block_count {
                let layer = &self.layers[layer_idx];
                layer.forward(
                    &self.mmap,
                    &mut self.scratch,
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
            }

            rms_norm_slice(
                &self.scratch.x,
                &self.output_norm_weight,
                self.rms_norm_eps as f32,
                &mut self.scratch.x_norm,
            );
        }

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
        quants::quantize_row_q8_0(&self.scratch.x_norm, &mut self.scratch.x_q8_0);
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

            let embd_bytes = &self.mmap[matrix_offset..matrix_offset + vocab_size * bytes_per_token];

            use rayon::prelude::*;
            let chunk_size = 4096;
            let x_q8_0 = &self.scratch.x_q8_0;
            let top_16 = (0..vocab_size as u32)
                .into_par_iter()
                .chunks(chunk_size)
                .map(|chunk| {
                    let mut local_top = [(0u32, f32::NEG_INFINITY); 16];
                    for tok_id in chunk {
                        if suppress_eos && (tok_id == 0 || tok_id == 1 || tok_id == 2) {
                            continue;
                        }
                        let row_offset = (tok_id as usize) * bytes_per_token;
                        let row_slice = &embd_bytes[row_offset..row_offset + bytes_per_token];
                        let val = unsafe { quants::vec_dot_q8_0_q8_0_avx2(row_slice, x_q8_0) };
                        let mut score = val;
                        if recent.contains(&tok_id) {
                            score = if score > 0.0 {
                                score / repeat_penalty
                            } else {
                                score * repeat_penalty
                            };
                        }
                        insert_top_k(&mut local_top, tok_id, score);
                    }
                    local_top
                })
                .reduce(
                    || [(0u32, f32::NEG_INFINITY); 16],
                    |a, b| merge_top_k(&a, &b),
                );

            let best_token = sample_top_p(&top_16, self.sampling_temp, self.sampling_top_p);
            Ok(best_token)
        }
    }
}
