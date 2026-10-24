use anyhow::{Result, anyhow};
use candle_core::quantized::{ggml_file::qtensor_from_ggml, gguf_file, GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_transformers::quantized_nn::RmsNorm;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

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

#[derive(Debug, Clone)]
pub struct Mlp {
    pub feed_forward_w1: QMatMul,
    pub feed_forward_w2: QMatMul,
    pub feed_forward_w3: QMatMul,
}

impl Mlp {
    pub fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

pub struct LayerWeights {
    pub attention_wq: QMatMul,
    pub attention_wk: QMatMul,
    pub attention_wv: QMatMul,
    pub attention_bq: Tensor,
    pub attention_bk: Tensor,
    pub attention_bv: Tensor,
    pub attention_wo: QMatMul,
    pub attention_norm: RmsNorm,
    pub mlp: Mlp,
    pub ffn_norm: RmsNorm,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub neg_inf: Tensor,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> candle_core::Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

fn repeat_kv(xs: Tensor, n_rep: usize) -> candle_core::Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        xs.unsqueeze(2)?
            .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
            .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

impl LayerWeights {
    pub fn forward_attn(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> candle_core::Result<Tensor> {
        let (b_sz, seq_len, n_embd) = x.dims3()?;

        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q.broadcast_add(&self.attention_bq)?;
        let k = k.broadcast_add(&self.attention_bk)?;
        let v = v.broadcast_add(&self.attention_bv)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let cos_n = cos.narrow(0, index_pos, seq_len)?;
        let sin_n = sin.narrow(0, index_pos, seq_len)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos_n, &sin_n)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos_n, &sin_n)?;

        let (k, v) = match kv_cache.as_ref() {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        *kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match mask {
            None => att,
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, &self.neg_inf)?
            }
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        self.attention_wo.forward(&y)
    }

    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> candle_core::Result<Tensor> {
        let residual = x;
        let x_norm = self.attention_norm.forward(x)?;
        let attn = self.forward_attn(&x_norm, mask, index_pos, cos, sin, kv_cache)?;
        let x = (attn + residual)?;

        let x_norm = self.ffn_norm.forward(&x)?;
        let mlp_out = self.mlp.forward(&x_norm)?;
        x + mlp_out
    }
}

fn precompute_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    context_length: usize,
    device: &Device,
) -> candle_core::Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

#[derive(Clone)]
pub struct PrefixCache {
    pub tokens: Vec<u32>,
    pub kv_caches: Vec<Option<(Tensor, Tensor)>>,
}

#[derive(Clone)]
pub struct TensorRef {
    pub offset: usize,
    pub size: usize,
    pub shape: Vec<usize>,
    pub dtype: GgmlDType,
}

fn resolve_tensor(content: &gguf_file::Content, name: &str) -> Result<TensorRef> {
    let info = content
        .tensor_infos
        .get(name)
        .ok_or_else(|| anyhow!("missing tensor {name}"))?;
    let offset = (content.tensor_data_offset + info.offset) as usize;
    let size = (info.shape.elem_count() / info.ggml_dtype.block_size()) * info.ggml_dtype.type_size();
    Ok(TensorRef {
        offset,
        size,
        shape: info.shape.dims().to_vec(),
        dtype: info.ggml_dtype,
    })
}

#[derive(Clone)]
pub struct LayerStaticData {
    pub attention_bq: Tensor,
    pub attention_bk: Tensor,
    pub attention_bv: Tensor,
    pub attention_norm: RmsNorm,
    pub ffn_norm: RmsNorm,
    pub wq: TensorRef,
    pub wk: TensorRef,
    pub wv: TensorRef,
    pub wo: TensorRef,
    pub w1: TensorRef,
    pub w2: TensorRef,
    pub w3: TensorRef,
}

pub struct AstMicroEngine {
    pub mmap: Mmap,
    pub content: gguf_file::Content,
    pub device: Device,
    pub block_count: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub head_dim: usize,
    pub embedding_length: usize,
    pub rms_norm_eps: f64,
    pub cos: Tensor,
    pub sin: Tensor,
    pub neg_inf: Tensor,
    pub output_norm: RmsNorm,
    pub output_weight: Option<QMatMul>,
    pub token_embd_offset: usize,
    pub token_embd_dtype: GgmlDType,
    pub bytes_per_token: usize,
    pub vocab_size: usize,
    pub output_offset: usize,
    pub output_dtype: GgmlDType,
    pub layer_ranges: Vec<Vec<(usize, usize)>>,
    pub kv_caches: Vec<Option<(Tensor, Tensor)>>,
    pub masks: HashMap<usize, Tensor>,
    pub cached_layers: Option<Vec<LayerWeights>>,
    pub prefix_cache: Option<PrefixCache>,
    pub layer_static: Vec<LayerStaticData>,
}

impl AstMicroEngine {
    pub fn new(model_path: &Path) -> Result<Self> {
        let file = File::open(model_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut cursor = std::io::Cursor::new(&mmap[..]);
        let mut content = gguf_file::Content::read(&mut cursor)?;
        let device = Device::Cpu;

        let md_get = |s: &str| -> Result<&gguf_file::Value> {
            content
                .metadata
                .get(s)
                .ok_or_else(|| anyhow!("missing metadata {s}"))
        };

        let head_count = md_get("qwen2.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("qwen2.attention.head_count_kv")?.to_u32()? as usize;
        let embedding_length = md_get("qwen2.embedding_length")?.to_u32()? as usize;
        let context_length = (md_get("qwen2.context_length")?.to_u32()? as usize).min(4096);
        let block_count = md_get("qwen2.block_count")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen2.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen2.rope.freq_base")
            .ok()
            .and_then(|m| m.to_f32().ok())
            .unwrap_or(1000000.0);

        // Purge heavy embedded GGUF tokenizer metadata tables that are unused by the runtime
        content.metadata.remove("tokenizer.ggml.tokens");
        content.metadata.remove("tokenizer.ggml.scores");
        content.metadata.remove("tokenizer.ggml.token_type");
        content.metadata.remove("tokenizer.ggml.merges");

        // Evict GGUF header pages from kernel physical RAM
        let header_bytes = content.tensor_data_offset as usize;
        let page_size = 4096;
        let header_page_end = ((header_bytes + page_size - 1) / page_size) * page_size;
        unsafe {
            libc::madvise(
                mmap.as_ptr() as *mut libc::c_void,
                header_page_end,
                libc::MADV_DONTNEED,
            );
            libc::malloc_trim(0);
        }

        let head_dim = embedding_length / head_count;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, &device)?;
        let (cos, sin) = precompute_freqs_cis(head_dim, rope_freq_base, context_length, &device)?;

        let norm_tensor = content.tensor(&mut cursor, "output_norm.weight", &device)?;
        let output_norm = RmsNorm::from_qtensor(norm_tensor, rms_norm_eps)?;

        // AST-Micro streaming mode (<86MB peak RAM) is the default and only engine execution profile.
        let output_weight = None;

        let embd_info = content
            .tensor_infos
            .get("token_embd.weight")
            .ok_or_else(|| anyhow!("missing token_embd.weight"))?;
        let token_embd_offset = (content.tensor_data_offset + embd_info.offset) as usize;
        let token_embd_dtype = embd_info.ggml_dtype;
        let bytes_per_token = (embedding_length / token_embd_dtype.block_size()) * token_embd_dtype.type_size();

        let (output_offset, output_dtype, vocab_size) = if let Some(info) = content.tensor_infos.get("output.weight") {
            let offset = (content.tensor_data_offset + info.offset) as usize;
            let v_size = info.shape.dims().iter().copied().max().unwrap_or(0);
            (offset, info.ggml_dtype, v_size)
        } else {
            let v_size = embd_info.shape.dims().iter().copied().max().unwrap_or(0);
            (token_embd_offset, token_embd_dtype, v_size)
        };

        let mut layer_ranges = Vec::with_capacity(block_count);
        for idx in 0..block_count {
            let prefix = format!("blk.{idx}");
            let tensor_names = [
                format!("{prefix}.attn_q.weight"),
                format!("{prefix}.attn_k.weight"),
                format!("{prefix}.attn_v.weight"),
                format!("{prefix}.attn_output.weight"),
                format!("{prefix}.attn_q.bias"),
                format!("{prefix}.attn_k.bias"),
                format!("{prefix}.attn_v.bias"),
                format!("{prefix}.attn_norm.weight"),
                format!("{prefix}.ffn_gate.weight"),
                format!("{prefix}.ffn_down.weight"),
                format!("{prefix}.ffn_up.weight"),
                format!("{prefix}.ffn_norm.weight"),
            ];
            let mut ranges = Vec::new();
            for name in &tensor_names {
                if let Some(info) = content.tensor_infos.get(name) {
                    let start = (content.tensor_data_offset + info.offset) as usize;
                    let size = (info.shape.elem_count() / info.ggml_dtype.block_size()) * info.ggml_dtype.type_size();
                    ranges.push((start, size));
                }
            }
            layer_ranges.push(ranges);
        }

        let kv_caches = vec![None; block_count];

        let mut engine = Self {
            mmap,
            content,
            device: device.clone(),
            block_count,
            head_count,
            head_count_kv,
            head_dim,
            embedding_length,
            rms_norm_eps,
            cos,
            sin,
            neg_inf,
            output_norm,
            output_weight,
            token_embd_offset,
            token_embd_dtype,
            bytes_per_token,
            vocab_size,
            output_offset,
            output_dtype,
            layer_ranges,
            kv_caches,
            masks: HashMap::new(),
            cached_layers: None,
            prefix_cache: None,
            layer_static: Vec::new(),
        };

        let mut layer_static = Vec::with_capacity(block_count);
        for idx in 0..block_count {
            let prefix = format!("blk.{idx}");
            let bq = engine.read_tensor(&format!("{prefix}.attn_q.bias"))?.dequantize(&device)?;
            let bk = engine.read_tensor(&format!("{prefix}.attn_k.bias"))?.dequantize(&device)?;
            let bv = engine.read_tensor(&format!("{prefix}.attn_v.bias"))?.dequantize(&device)?;
            let attention_norm = RmsNorm::from_qtensor(
                engine.read_tensor(&format!("{prefix}.attn_norm.weight"))?,
                rms_norm_eps,
            )?;
            let ffn_norm = RmsNorm::from_qtensor(
                engine.read_tensor(&format!("{prefix}.ffn_norm.weight"))?,
                rms_norm_eps,
            )?;

            layer_static.push(LayerStaticData {
                attention_bq: bq,
                attention_bk: bk,
                attention_bv: bv,
                attention_norm,
                ffn_norm,
                wq: resolve_tensor(&engine.content, &format!("{prefix}.attn_q.weight"))?,
                wk: resolve_tensor(&engine.content, &format!("{prefix}.attn_k.weight"))?,
                wv: resolve_tensor(&engine.content, &format!("{prefix}.attn_v.weight"))?,
                wo: resolve_tensor(&engine.content, &format!("{prefix}.attn_output.weight"))?,
                w1: resolve_tensor(&engine.content, &format!("{prefix}.ffn_gate.weight"))?,
                w2: resolve_tensor(&engine.content, &format!("{prefix}.ffn_down.weight"))?,
                w3: resolve_tensor(&engine.content, &format!("{prefix}.ffn_up.weight"))?,
            });
        }
        engine.layer_static = layer_static;

        engine.cached_layers = None;
        Ok(engine)
    }

    pub fn reset_cache(&mut self) {
        self.kv_caches = vec![None; self.block_count];
        if self.cached_layers.is_none() {
            unsafe {
                libc::madvise(
                    self.mmap.as_ptr() as *mut libc::c_void,
                    self.mmap.len(),
                    libc::MADV_DONTNEED,
                );
                libc::malloc_trim(0);
            }
        }
    }

    pub fn embed(&self, tokens: &[u32]) -> Result<Tensor> {
        let seq_len = tokens.len();
        let mut raw_bytes = Vec::with_capacity(seq_len * self.bytes_per_token);
        for &tok in tokens {
            let start = self.token_embd_offset + (tok as usize) * self.bytes_per_token;
            let end = start + self.bytes_per_token;
            raw_bytes.extend_from_slice(&self.mmap[start..end]);
        }
        let qtensor = qtensor_from_ggml(
            self.token_embd_dtype,
            &raw_bytes,
            vec![1, seq_len, self.embedding_length],
            &self.device,
        )?;
        Ok(qtensor.dequantize(&self.device)?)
    }

    fn read_tensor(&self, name: &str) -> Result<QTensor> {
        let info = self
            .content
            .tensor_infos
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor {name}"))?;
        let start = (self.content.tensor_data_offset + info.offset) as usize;
        let size = (info.shape.elem_count() / info.ggml_dtype.block_size()) * info.ggml_dtype.type_size();
        let slice = &self.mmap[start..start + size];
        qtensor_from_ggml(info.ggml_dtype, slice, info.shape.dims().to_vec(), &self.device)
            .map_err(|e| anyhow!("{e}"))
    }

    pub fn read_tensor_ref(&self, t: &TensorRef) -> Result<QTensor> {
        let slice = &self.mmap[t.offset..t.offset + t.size];
        qtensor_from_ggml(t.dtype, slice, t.shape.clone(), &self.device)
            .map_err(|e| anyhow!("{e}"))
    }

    pub fn load_layer(&self, idx: usize) -> Result<LayerWeights> {
        let stat = &self.layer_static[idx];
        let wq = QMatMul::from_qtensor(self.read_tensor_ref(&stat.wq)?)?;
        let wk = QMatMul::from_qtensor(self.read_tensor_ref(&stat.wk)?)?;
        let wv = QMatMul::from_qtensor(self.read_tensor_ref(&stat.wv)?)?;
        let wo = QMatMul::from_qtensor(self.read_tensor_ref(&stat.wo)?)?;

        let mlp = Mlp {
            feed_forward_w1: QMatMul::from_qtensor(self.read_tensor_ref(&stat.w1)?)?,
            feed_forward_w2: QMatMul::from_qtensor(self.read_tensor_ref(&stat.w2)?)?,
            feed_forward_w3: QMatMul::from_qtensor(self.read_tensor_ref(&stat.w3)?)?,
        };

        Ok(LayerWeights {
            attention_wq: wq,
            attention_wk: wk,
            attention_wv: wv,
            attention_bq: stat.attention_bq.clone(),
            attention_bk: stat.attention_bk.clone(),
            attention_bv: stat.attention_bv.clone(),
            attention_wo: wo,
            attention_norm: stat.attention_norm.clone(),
            mlp,
            ffn_norm: stat.ffn_norm.clone(),
            n_head: self.head_count,
            n_kv_head: self.head_count_kv,
            head_dim: self.head_dim,
            neg_inf: self.neg_inf.clone(),
        })
    }

    fn mask(&mut self, t: usize) -> candle_core::Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }

    fn causal_mask(&mut self, seq_len: usize, total_len: usize) -> candle_core::Result<Tensor> {
        let index_pos = total_len - seq_len;
        if index_pos == 0 {
            return self.mask(seq_len);
        }
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| (0..total_len).map(move |j| u8::from(j > index_pos + i)))
            .collect();
        Tensor::from_slice(&mask, (seq_len, total_len), &self.device)
    }

    pub fn save_prefix(&mut self, tokens: &[u32], prefix_len: usize) {
        if prefix_len == 0 || prefix_len > tokens.len() {
            return;
        }
        let mut prefix_kv = Vec::with_capacity(self.block_count);
        for kv in &self.kv_caches {
            if let Some((k, v)) = kv {
                let k_sub = k.narrow(2, 0, prefix_len).ok();
                let v_sub = v.narrow(2, 0, prefix_len).ok();
                if let (Some(k_sub), Some(v_sub)) = (k_sub, v_sub) {
                    prefix_kv.push(Some((k_sub, v_sub)));
                } else {
                    prefix_kv.push(None);
                }
            } else {
                prefix_kv.push(None);
            }
        }
        self.prefix_cache = Some(PrefixCache {
            tokens: tokens[..prefix_len].to_vec(),
            kv_caches: prefix_kv,
        });
    }

    pub fn restore_prefix(&mut self, prefix_len: usize) -> bool {
        if let Some(ref cache) = self.prefix_cache {
            if cache.tokens.len() >= prefix_len {
                let mut restored = Vec::with_capacity(self.block_count);
                for kv in &cache.kv_caches {
                    if let Some((k, v)) = kv {
                        let k_sub = k.narrow(2, 0, prefix_len).ok();
                        let v_sub = v.narrow(2, 0, prefix_len).ok();
                        if let (Some(k_sub), Some(v_sub)) = (k_sub, v_sub) {
                            restored.push(Some((k_sub, v_sub)));
                        } else {
                            return false;
                        }
                    } else {
                        restored.push(None);
                    }
                }
                self.kv_caches = restored;
                return true;
            }
        }
        false
    }

    pub fn prefill(&mut self, prompt_tokens: &[u32]) -> Result<Tensor> {
        let mut matched_len = 0usize;
        if let Some(ref cache) = self.prefix_cache {
            let common = prompt_tokens
                .iter()
                .zip(cache.tokens.iter())
                .take_while(|(a, b)| a == b)
                .count();
            if common >= 16 {
                matched_len = common;
            }
        }

        let hidden = if matched_len > 0 && self.restore_prefix(matched_len) {
            let remaining = &prompt_tokens[matched_len..];
            if remaining.is_empty() {
                self.forward_stream(&prompt_tokens[matched_len - 1..], matched_len - 1)?
            } else {
                self.forward_stream(remaining, matched_len)?
            }
        } else {
            self.reset_cache();
            self.forward_stream(prompt_tokens, 0)?
        };

        // Automatically cache the largest valid boundary (<|im_end|>\n = 151645 for Qwen, 2 for NioViper) before the final prompt:
        let prefix_end = prompt_tokens[..prompt_tokens.len().saturating_sub(1)]
            .iter()
            .rposition(|&tok| tok == 151645 || tok == 2)
            .map(|pos| (pos + 2).min(prompt_tokens.len()))
            .unwrap_or_else(|| {
                prompt_tokens
                    .iter()
                    .position(|&tok| tok == 151645 || tok == 2)
                    .map(|pos| (pos + 2).min(prompt_tokens.len()))
                    .unwrap_or_else(|| prompt_tokens.len().min(128))
            });

        if prefix_end >= 16 && prefix_end <= prompt_tokens.len() {
            self.save_prefix(prompt_tokens, prefix_end);
        }

        Ok(hidden)
    }

    pub fn prefetch_layer(&self, idx: usize) {
        if let Some(ranges) = self.layer_ranges.get(idx) {
            let page_size = 4096;
            for &(start, size) in ranges {
                let page_start = (start / page_size) * page_size;
                let page_end = ((start + size + page_size - 1) / page_size) * page_size;
                let len = page_end - page_start;
                unsafe {
                    libc::madvise(
                        self.mmap.as_ptr().add(page_start) as *mut libc::c_void,
                        len,
                        libc::MADV_WILLNEED,
                    );
                }
            }
        }
    }

    pub fn evict_layer(&self, idx: usize) {
        if let Some(ranges) = self.layer_ranges.get(idx) {
            let page_size = 4096;
            for &(start, size) in ranges {
                let page_start = (start / page_size) * page_size;
                let page_end = ((start + size + page_size - 1) / page_size) * page_size;
                let len = page_end - page_start;
                unsafe {
                    libc::madvise(
                        self.mmap.as_ptr().add(page_start) as *mut libc::c_void,
                        len,
                        libc::MADV_DONTNEED,
                    );
                }
            }
        }
    }

    pub fn forward_stream(&mut self, tokens: &[u32], index_pos: usize) -> Result<Tensor> {
        let seq_len = tokens.len();
        let total_len = index_pos + seq_len;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.causal_mask(seq_len, total_len)?)
        };

        let mut layer_in = self.embed(tokens)?;

        if let Some(ref layers) = self.cached_layers {
            for (layer_idx, layer) in layers.iter().enumerate() {
                layer_in = layer.forward(
                    &layer_in,
                    mask.as_ref(),
                    index_pos,
                    &self.cos,
                    &self.sin,
                    &mut self.kv_caches[layer_idx],
                )?;
            }
        } else {
            // Prefetch the first layer
            self.prefetch_layer(0);
            for layer_idx in 0..self.block_count {
                // Prefetch NEXT layer while computing current one (pipeline overlap)
                if layer_idx + 1 < self.block_count {
                    self.prefetch_layer(layer_idx + 1);
                }

                let layer = self.load_layer(layer_idx)?;
                layer_in = layer.forward(
                    &layer_in,
                    mask.as_ref(),
                    index_pos,
                    &self.cos,
                    &self.sin,
                    &mut self.kv_caches[layer_idx],
                )?;
                drop(layer);

                // Layers are kept in pagecache during active token generation for fast throughput,
                // and evicted en masse via reset_cache() when generation completes.
            }
        }

        let out = self.output_norm.forward(&layer_in)?;
        let last_hidden = out.i((.., seq_len - 1, ..))?;
        Ok(last_hidden)
    }

    pub fn truncate_kv(&mut self, target_len: usize) {
        for kv in &mut self.kv_caches {
            if let Some((k, v)) = kv {
                if let (Ok(k_sub), Ok(v_sub)) = (k.narrow(2, 0, target_len), v.narrow(2, 0, target_len)) {
                    *k = k_sub;
                    *v = v_sub;
                }
            }
        }
    }

    pub fn forward_all(&mut self, tokens: &[u32], index_pos: usize) -> Result<Tensor> {
        let seq_len = tokens.len();
        let total_len = index_pos + seq_len;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.causal_mask(seq_len, total_len)?)
        };

        let mut layer_in = self.embed(tokens)?;

        if let Some(ref layers) = self.cached_layers {
            for (layer_idx, layer) in layers.iter().enumerate() {
                layer_in = layer.forward(
                    &layer_in,
                    mask.as_ref(),
                    index_pos,
                    &self.cos,
                    &self.sin,
                    &mut self.kv_caches[layer_idx],
                )?;
            }
        } else {
            self.prefetch_layer(0);
            for layer_idx in 0..self.block_count {
                if layer_idx + 1 < self.block_count {
                    self.prefetch_layer(layer_idx + 1);
                }

                let layer = self.load_layer(layer_idx)?;
                layer_in = layer.forward(
                    &layer_in,
                    mask.as_ref(),
                    index_pos,
                    &self.cos,
                    &self.sin,
                    &mut self.kv_caches[layer_idx],
                )?;
                drop(layer);

            }
        }

        let out = self.output_norm.forward(&layer_in)?;
        Ok(out)
    }

    pub fn next_token(&mut self, hidden: &Tensor, candidates: Option<&[u32]>) -> Result<u32> {
        if let Some(cands) = candidates {
            if cands.is_empty() {
                return Ok(151643); // EOS
            }
            if let Some(ref output_weight) = self.output_weight {
                let logits = output_weight.forward(hidden)?;
                let logits = logits.squeeze(0)?;
                let logits_v: Vec<f32> = logits.to_vec1()?;
                let mut max_idx = cands[0];
                let mut max_val = f32::NEG_INFINITY;
                for &tok in cands {
                    if let Some(&val) = logits_v.get(tok as usize) {
                        if val > max_val {
                            max_val = val;
                            max_idx = tok;
                        }
                    }
                }
                return Ok(max_idx);
            }
            let mut raw_bytes = Vec::with_capacity(cands.len() * self.bytes_per_token);
            for &tok in cands {
                let start = self.output_offset + (tok as usize) * self.bytes_per_token;
                let end = start + self.bytes_per_token;
                raw_bytes.extend_from_slice(&self.mmap[start..end]);
            }
            let q_cand = qtensor_from_ggml(
                self.output_dtype,
                &raw_bytes,
                vec![cands.len(), self.embedding_length],
                &self.device,
            )?;
            let cand_weights = q_cand.dequantize(&self.device)?;
            let logits = hidden.matmul(&cand_weights.t()?)?.squeeze(0)?;
            let logits_v: Vec<f32> = logits.to_vec1()?;
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;
            for (i, &val) in logits_v.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }
            Ok(cands[max_idx])
        } else if let Some(ref output_weight) = self.output_weight {
            let logits = output_weight.forward(hidden)?;
            let logits = logits.squeeze(0)?;
            let logits_v: Vec<f32> = logits.to_vec1()?;
            let mut max_idx = 0u32;
            let mut max_val = f32::NEG_INFINITY;
            for (i, &val) in logits_v.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = i as u32;
                }
            }
            Ok(max_idx)
        } else {
            let chunk_size = 38016;
            let mut max_idx = 0u32;
            let mut max_val = f32::NEG_INFINITY;

            for chunk_start in (0..self.vocab_size).step_by(chunk_size) {
                let chunk_end = (chunk_start + chunk_size).min(self.vocab_size);
                let count = chunk_end - chunk_start;
                let byte_start = self.output_offset + chunk_start * self.bytes_per_token;
                let byte_len = count * self.bytes_per_token;
                let slice = &self.mmap[byte_start..byte_start + byte_len];

                let q_chunk = qtensor_from_ggml(
                    self.output_dtype,
                    slice,
                    vec![count, self.embedding_length],
                    &self.device,
                )?;
                let logits_v = if let Ok(qmat) = QMatMul::from_qtensor(q_chunk) {
                    let logits = qmat.forward(hidden)?.squeeze(0)?;
                    logits.to_vec1::<f32>()?
                } else {
                    let q_chunk = qtensor_from_ggml(
                        self.output_dtype,
                        slice,
                        vec![count, self.embedding_length],
                        &self.device,
                    )?;
                    let cand_weights = q_chunk.dequantize(&self.device)?;
                    let logits = hidden.matmul(&cand_weights.t()?)?.squeeze(0)?;
                    logits.to_vec1::<f32>()?
                };

                for (i, &val) in logits_v.iter().enumerate() {
                    if val > max_val {
                        max_val = val;
                        max_idx = (chunk_start + i) as u32;
                    }
                }
            }

            Ok(max_idx)
        }
    }

    pub fn evict_all(&self) {
        unsafe {
            libc::madvise(
                self.mmap.as_ptr() as *mut libc::c_void,
                self.mmap.len(),
                libc::MADV_DONTNEED,
            );
            libc::malloc_trim(0);
        }
    }
}
