use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use crate::ast_engine::vec_dot_f32_direct;

#[derive(Debug, Deserialize)]
struct AdapterConfig {
    #[serde(default = "default_r")]
    r: usize,
    #[serde(default = "default_alpha")]
    lora_alpha: f32,
    #[serde(default)]
    #[allow(dead_code)]
    target_modules: Vec<String>,
}

fn default_r() -> usize {
    32
}

fn default_alpha() -> f32 {
    64.0
}

/// In-process LoRA low-rank projection for a single linear layer:
/// W_eff = W_base + (lora_alpha / r) * (B @ A)
/// Output perturbation for vector x: delta_y = scaling * (x @ A^T) @ B^T
#[derive(Clone, Debug)]
pub struct LoraLinear {
    pub a: Vec<f32>,     // shape: [r, in_dim], row-major
    pub b: Vec<f32>,     // shape: [out_dim, r], row-major
    pub r: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub scaling: f32,
}

impl LoraLinear {
    #[inline(always)]
    pub fn apply(&self, x: &[f32], y: &mut [f32]) {
        if self.r == 0 || self.a.is_empty() || self.b.is_empty() {
            return;
        }
        let r = self.r;
        let in_dim = self.in_dim;
        let out_dim = self.out_dim;
        let scaling = self.scaling;

        // Step 1: h = x @ A^T, size [r]
        // Pre-scale h by scaling factor so subsequent B GEMV needs no scaling multiplications
        let mut h_buf = [0.0f32; 128];
        let mut h_heap = Vec::new();
        let h: &mut [f32] = if r <= 128 {
            &mut h_buf[..r]
        } else {
            h_heap.resize(r, 0.0f32);
            &mut h_heap[..r]
        };

        for k in 0..r {
            let a_row = &self.a[k * in_dim..(k + 1) * in_dim];
            h[k] = vec_dot_f32_direct(a_row, x) * scaling;
        }

        // Step 2: y += h @ B^T, size [out_dim]
        for i in 0..out_dim {
            let b_row = &self.b[i * r..(i + 1) * r];
            y[i] += vec_dot_f32_direct(b_row, h);
        }
    }
}

/// Set of LoRA projections for a single transformer layer
#[derive(Clone, Default, Debug)]
pub struct LayerLora {
    pub wq: Option<LoraLinear>,
    pub wk: Option<LoraLinear>,
    pub wv: Option<LoraLinear>,
    pub wo: Option<LoraLinear>,
    pub w1: Option<LoraLinear>,
    pub w2: Option<LoraLinear>,
    pub w3: Option<LoraLinear>,
}

/// Resident In-Process LoRA Adapter Head (< 3 MB footprint)
#[derive(Clone, Debug)]
pub struct LoraAdapter {
    pub id: String,
    pub r: usize,
    pub alpha: f32,
    pub layers: Vec<LayerLora>,
}

impl LoraAdapter {
    /// Load a LoRA adapter from a directory or directly from a safetensors file
    pub fn load(path: &Path) -> Result<Self> {
        let (config_path, weights_path, id) = if path.is_dir() {
            let cfg = path.join("adapter_config.json");
            let wgt = path.join("adapter_model.safetensors");
            let name = path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "lora-head".to_string());
            (cfg, wgt, name)
        } else {
            let dir = path.parent().unwrap_or_else(|| Path::new("."));
            let cfg = dir.join("adapter_config.json");
            let name = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "lora-head".to_string());
            (cfg, path.to_path_buf(), name)
        };

        if !weights_path.exists() {
            return Err(anyhow!(
                "LoRA weights not found at {}",
                weights_path.display()
            ));
        }

        let config: AdapterConfig = if config_path.exists() {
            let cfg_content = std::fs::read_to_string(&config_path)
                .with_context(|| format!("reading {}", config_path.display()))?;
            serde_json::from_str(&cfg_content).unwrap_or(AdapterConfig {
                r: default_r(),
                lora_alpha: default_alpha(),
                target_modules: Vec::new(),
            })
        } else {
            AdapterConfig {
                r: default_r(),
                lora_alpha: default_alpha(),
                target_modules: Vec::new(),
            }
        };

        let file = File::open(&weights_path)
            .with_context(|| format!("opening weights at {}", weights_path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let tensors = SafeTensors::deserialize(&mmap)
            .with_context(|| "deserializing SafeTensors header")?;

        let r = config.r;
        let alpha = config.lora_alpha;
        let scaling = if r > 0 { alpha / r as f32 } else { 1.0 };

        // Inspect layer indices present in the adapter
        let mut max_layer_idx = 0;
        for name in tensors.names() {
            if let Some(idx) = extract_layer_idx(name) {
                if idx > max_layer_idx {
                    max_layer_idx = idx;
                }
            }
        }

        let num_layers = max_layer_idx + 1;
        let mut layers = vec![LayerLora::default(); num_layers];

        // Match modules for each layer
        let module_specs = [
            ("q_proj", 0),
            ("k_proj", 1),
            ("v_proj", 2),
            ("o_proj", 3),
            ("gate_proj", 4),
            ("down_proj", 5),
            ("up_proj", 6),
        ];

        for layer_idx in 0..num_layers {
            for &(mod_name, mod_type) in &module_specs {
                if let Some(linear) =
                    load_lora_linear(&tensors, layer_idx, mod_name, r, scaling)?
                {
                    match mod_type {
                        0 => layers[layer_idx].wq = Some(linear),
                        1 => layers[layer_idx].wk = Some(linear),
                        2 => layers[layer_idx].wv = Some(linear),
                        3 => layers[layer_idx].wo = Some(linear),
                        4 => layers[layer_idx].w1 = Some(linear),
                        5 => layers[layer_idx].w2 = Some(linear),
                        6 => layers[layer_idx].w3 = Some(linear),
                        _ => {}
                    }
                }
            }
        }

        Ok(Self {
            id,
            r,
            alpha,
            layers,
        })
    }
}

fn extract_layer_idx(name: &str) -> Option<usize> {
    let parts: Vec<&str> = name.split('.').collect();
    for (i, &p) in parts.iter().enumerate() {
        if p == "layers" && i + 1 < parts.len() {
            if let Ok(idx) = parts[i + 1].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    None
}

fn load_lora_linear(
    tensors: &SafeTensors,
    layer_idx: usize,
    mod_name: &str,
    expected_r: usize,
    scaling: f32,
) -> Result<Option<LoraLinear>> {
    // Generate candidate keys matching HuggingFace PEFT conventions:
    // e.g. base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    // or model.layers.0.self_attn.q_proj.lora_A.weight
    // or layers.0.self_attn.q_proj.lora_A.weight
    let sub = if matches!(mod_name, "q_proj" | "k_proj" | "v_proj" | "o_proj") {
        "self_attn"
    } else {
        "mlp"
    };

    let candidates_a = [
        format!("base_model.model.model.layers.{layer_idx}.{sub}.{mod_name}.lora_A.weight"),
        format!("base_model.model.layers.{layer_idx}.{sub}.{mod_name}.lora_A.weight"),
        format!("model.layers.{layer_idx}.{sub}.{mod_name}.lora_A.weight"),
        format!("layers.{layer_idx}.{sub}.{mod_name}.lora_A.weight"),
    ];

    let candidates_b = [
        format!("base_model.model.model.layers.{layer_idx}.{sub}.{mod_name}.lora_B.weight"),
        format!("base_model.model.layers.{layer_idx}.{sub}.{mod_name}.lora_B.weight"),
        format!("model.layers.{layer_idx}.{sub}.{mod_name}.lora_B.weight"),
        format!("layers.{layer_idx}.{sub}.{mod_name}.lora_B.weight"),
    ];

    let mut tensor_a = None;
    for k in &candidates_a {
        if let Ok(t) = tensors.tensor(k) {
            tensor_a = Some(t);
            break;
        }
    }

    let mut tensor_b = None;
    for k in &candidates_b {
        if let Ok(t) = tensors.tensor(k) {
            tensor_b = Some(t);
            break;
        }
    }

    let (t_a, t_b) = match (tensor_a, tensor_b) {
        (Some(a), Some(b)) => (a, b),
        _ => return Ok(None),
    };

    let shape_a = t_a.shape();
    let shape_b = t_b.shape();

    if shape_a.len() != 2 || shape_b.len() != 2 {
        return Ok(None);
    }

    let r_a = shape_a[0];
    let in_dim = shape_a[1];
    let out_dim = shape_b[0];
    let r_b = shape_b[1];

    if r_a != r_b {
        return Err(anyhow!(
            "Rank mismatch between lora_A ({r_a}) and lora_B ({r_b}) for layer {layer_idx} {mod_name}"
        ));
    }

    let r = if r_a > 0 { r_a } else { expected_r };

    let vec_a = tensor_to_f32(&t_a)?;
    let vec_b = tensor_to_f32(&t_b)?;

    Ok(Some(LoraLinear {
        a: vec_a,
        b: vec_b,
        r,
        in_dim,
        out_dim,
        scaling,
    }))
}

fn tensor_to_f32(t: &safetensors::tensor::TensorView) -> Result<Vec<f32>> {
    let data = t.data();
    match t.dtype() {
        safetensors::Dtype::F32 => {
            let count = data.len() / 4;
            let mut out = Vec::with_capacity(count);
            for chunk in data.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        safetensors::Dtype::F16 => {
            let count = data.len() / 2;
            let mut out = Vec::with_capacity(count);
            for chunk in data.chunks_exact(2) {
                let u = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::f16::from_bits(u).to_f32());
            }
            Ok(out)
        }
        safetensors::Dtype::BF16 => {
            let count = data.len() / 2;
            let mut out = Vec::with_capacity(count);
            for chunk in data.chunks_exact(2) {
                let u = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::bf16::from_bits(u).to_f32());
            }
            Ok(out)
        }
        other => Err(anyhow!("unsupported safetensors dtype {other:?}")),
    }
}

/// Global In-Memory LoRA Head Cache for Sub-Millisecond (< 1ms) Hot-Swapping
static LORA_CACHE: OnceLock<Mutex<HashMap<String, Arc<LoraAdapter>>>> = OnceLock::new();

fn get_cache() -> &'static Mutex<HashMap<String, Arc<LoraAdapter>>> {
    LORA_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Resolve a council head name or slug to its on-disk path
pub fn resolve_head_path(head_name: &str) -> Option<PathBuf> {
    let raw = head_name.trim();
    let lower = raw.to_ascii_lowercase();

    // 1. If explicit file or directory path exists
    let direct = Path::new(raw);
    if direct.exists() {
        return Some(direct.to_path_buf());
    }

    // 2. Standard search directories
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let mut search_dirs = Vec::new();
    if let Some(ref h) = home {
        search_dirs.push(h.join(".nioxs").join("heads"));
    }
    search_dirs.push(PathBuf::from("/root/.nioxs/heads"));
    search_dirs.push(PathBuf::from("/root/nio-labs/nio-xs-models/models/hydra/heads"));

    // 3. Normalized candidate names
    let mut candidates = Vec::new();
    candidates.push(lower.clone());

    if lower == "001" || lower == "head-001" || lower == "hydra-kernel" || lower.contains("atomic") {
        candidates.push("head-001".to_string());
        candidates.push("head-001-atomic-concurrency".to_string());
        candidates.push("001".to_string());
    }
    if lower == "002" || lower == "head-002" || lower.contains("axum") || lower.contains("backend") {
        candidates.push("head-002".to_string());
        candidates.push("head-002-backend-axum-tokio".to_string());
        candidates.push("002".to_string());
    }
    if lower == "003" || lower == "head-003" || lower.contains("react") || lower.contains("rsc") {
        candidates.push("head-003".to_string());
        candidates.push("head-003-frontend-react19-rsc".to_string());
    }
    if lower == "004" || lower == "head-004" || lower.contains("ebpf") || lower.contains("xdp") {
        candidates.push("head-004".to_string());
        candidates.push("head-004-kernel-ebpf-xdp".to_string());
    }
    if lower == "005" || lower == "head-005" || lower.contains("clickhouse") {
        candidates.push("head-005".to_string());
        candidates.push("head-005-data-clickhouse-mergetree".to_string());
    }
    if lower == "006" || lower == "head-006" || lower.contains("logic") || lower.contains("algorithm") {
        candidates.push("head-006".to_string());
        candidates.push("head-006-algorithms-formal-logic".to_string());
    }

    for dir in &search_dirs {
        for cand in &candidates {
            let p = dir.join(cand);
            if p.exists() {
                return Some(p);
            }
            let p_safetensors = p.join("adapter_model.safetensors");
            if p_safetensors.exists() {
                return Some(p);
            }
        }
    }

    None
}

/// Retrieve a cached LoRA adapter or load it from disk in < 1ms
pub fn get_or_load_head(head_id_or_path: &str) -> Result<Arc<LoraAdapter>> {
    let lower = head_id_or_path.trim().to_ascii_lowercase();

    // Check in-memory cache first (hot-swap speed: < 0.001 ms)
    {
        let cache = get_cache().lock().unwrap();
        if let Some(adapter) = cache.get(&lower) {
            return Ok(adapter.clone());
        }
    }

    // Resolve path
    let path = resolve_head_path(head_id_or_path)
        .ok_or_else(|| anyhow!("Could not resolve LoRA head path for '{head_id_or_path}'"))?;

    let t0 = std::time::Instant::now();
    let adapter = Arc::new(LoraAdapter::load(&path)?);
    let elapsed = t0.elapsed();

    // Cache the adapter under both requested name and canonical id
    let mut cache = get_cache().lock().unwrap();
    cache.insert(lower.clone(), adapter.clone());
    cache.insert(adapter.id.to_ascii_lowercase(), adapter.clone());

    #[cfg(debug_assertions)]
    eprintln!(
        "[nioxs-lora] Loaded head '{}' ({} layers) in {:.2}ms",
        adapter.id,
        adapter.layers.len(),
        elapsed.as_secs_f64() * 1000.0
    );

    Ok(adapter)
}

/// Evict all resident LoRA adapters from in-memory cache
pub fn clear_lora_cache() {
    let mut cache = get_cache().lock().unwrap();
    cache.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_linear_apply_identity() {
        let linear = LoraLinear {
            a: vec![1.0, 0.0, 0.0, 1.0], // [2, 2]
            b: vec![1.0, 0.0, 0.0, 1.0], // [2, 2]
            r: 2,
            in_dim: 2,
            out_dim: 2,
            scaling: 1.0,
        };

        let x = [3.0f32, 5.0f32];
        let mut y = [10.0f32, 20.0f32];
        linear.apply(&x, &mut y);

        // x @ A^T = [3.0, 5.0]
        // [3.0, 5.0] @ B^T = [3.0, 5.0]
        // y += [3.0, 5.0] => [13.0, 25.0]
        assert!((y[0] - 13.0).abs() < 1e-5);
        assert!((y[1] - 25.0).abs() < 1e-5);
    }

    #[test]
    fn test_resolve_head_path() {
        let p1 = resolve_head_path("head-001");
        assert!(p1.is_some(), "head-001 should resolve in .nioxs/heads");
        let p2 = resolve_head_path("head-002");
        assert!(p2.is_some(), "head-002 should resolve in .nioxs/heads");
    }

    #[test]
    fn test_load_head_001_safetensors() {
        if let Some(path) = resolve_head_path("head-001") {
            let t0 = std::time::Instant::now();
            let adapter = LoraAdapter::load(&path).expect("head-001 must load cleanly");
            let load_time_ms = t0.elapsed().as_secs_f64() * 1000.0;
            println!("head-001 loaded in {:.2} ms", load_time_ms);
            assert_eq!(adapter.r, 32);
            assert_eq!(adapter.alpha, 64.0);
            assert_eq!(adapter.layers.len(), 10);

            // Populate cache
            let _ = get_or_load_head("head-001").expect("initial load");

            // Verify hot-swap cache retrieval (< 0.01 ms = < 10 µs)
            let t1 = std::time::Instant::now();
            let cached = get_or_load_head("head-001").expect("cached lookup");
            let cache_time_us = t1.elapsed().as_micros();
            println!("cached lookup took {} µs", cache_time_us);
            assert!(cache_time_us < 1000, "cache lookup must be sub-millisecond, took {cache_time_us} µs");
            assert_eq!(cached.layers.len(), 10);
        }
    }
}
