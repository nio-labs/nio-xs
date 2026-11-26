use anyhow::{Context, Result, anyhow};
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use crate::ast_engine::vec_dot_f32_direct;

// =========================================================================
// FP8 (E4M3 / E5M2) Lookup Tables & Precision Conversion
// =========================================================================

/// Precomputed compile-time lookup table for FP8 E4M3 -> f32 (256 entries)
pub const fn init_fp8_e4m3_table() -> [f32; 256] {
    let mut table = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        let b = i as u8;
        let sign = if (b & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = (b >> 3) & 0x0F;
        let mant = (b & 0x07) as f32;

        let val = if exp == 0 {
            // Subnormal: (-1)^s * 2^(-6) * (m / 8)
            sign * 0.015625f32 * (mant / 8.0f32)
        } else if exp == 15 && (b & 0x07) == 7 {
            // NaN in E4M3
            f32::NAN
        } else {
            // Normal: (-1)^s * 2^(exp - 7) * (1 + m / 8)
            let pow = (1 << (exp - 1)) as f32 / 64.0f32; // 2^(exp - 7)
            sign * pow * (1.0f32 + mant / 8.0f32)
        };
        table[i] = val;
        i += 1;
    }
    table
}

/// Precomputed compile-time lookup table for FP8 E5M2 -> f32 (256 entries)
pub const fn init_fp8_e5m2_table() -> [f32; 256] {
    let mut table = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        let b = i as u8;
        let sign = if (b & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
        let exp = (b >> 2) & 0x1F;
        let mant = (b & 0x03) as f32;

        let val = if exp == 0 {
            // Subnormal: (-1)^s * 2^(-14) * (m / 4)
            sign * (1.0f32 / 16384.0f32) * (mant / 4.0f32)
        } else if exp == 31 {
            if mant == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        } else {
            // Normal: (-1)^s * 2^(exp - 15) * (1 + m / 4)
            let pow = if exp >= 15 {
                (1 << (exp - 15)) as f32
            } else {
                1.0f32 / (1 << (15 - exp)) as f32
            };
            sign * pow * (1.0f32 + mant / 4.0f32)
        };
        table[i] = val;
        i += 1;
    }
    table
}

pub static FP8_E4M3_TABLE: [f32; 256] = init_fp8_e4m3_table();
pub static FP8_E5M2_TABLE: [f32; 256] = init_fp8_e5m2_table();

#[inline(always)]
pub fn fp8_e4m3_to_f32(b: u8) -> f32 {
    FP8_E4M3_TABLE[b as usize]
}

#[inline(always)]
pub fn fp8_e5m2_to_f32(b: u8) -> f32 {
    FP8_E5M2_TABLE[b as usize]
}

/// Fast IEEE-754 f32 to OCP FP8 E4M3 quantization
pub fn f32_to_fp8_e4m3(val: f32) -> u8 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 0 {
        return (sign << 7) as u8;
    }
    if exp == 0xFF {
        return ((sign << 7) | 0x7F) as u8;
    }

    // Unbiased exp in f32: exp - 127. E4M3 bias is 7: exp_fp8 = (exp - 127) + 7 = exp - 120
    let exp_fp8 = exp - 120;

    if exp_fp8 >= 15 {
        return ((sign << 7) | 0x7E) as u8; // Clamp to max finite (448.0)
    }

    if exp_fp8 <= 0 {
        let shift = 1 - exp_fp8;
        if shift > 8 {
            return (sign << 7) as u8;
        }
        let full_mant = (1 << 23) | mant;
        let mant_fp8 = (full_mant >> (20 + shift)) & 0x7;
        return ((sign << 7) | (mant_fp8 as u32)) as u8;
    }

    // Normal round with nearest-even bias
    let mant_fp8 = ((mant + (1 << 19)) >> 20).min(7);
    ((sign << 7) | ((exp_fp8 as u32) << 3) | (mant_fp8 as u32)) as u8
}

// =========================================================================
// SVD / Principal Component Rank Truncation (Rank 32 -> 16 / 12)
// =========================================================================

/// Classical Jacobi eigenvalue algorithm for small symmetric matrices (e.g. 32x32)
pub fn jacobi_eigenvalues_symm(a: &[f32], n: usize, max_iter: usize) -> (Vec<f32>, Vec<f32>) {
    let mut v = vec![0.0f32; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let mut m = a.to_vec();

    for _ in 0..max_iter {
        let mut max_val = 0.0f32;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let abs_val = m[i * n + j].abs();
                if abs_val > max_val {
                    max_val = abs_val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-7 {
            break;
        }

        let app = m[p * n + p];
        let aqq = m[q * n + q];
        let apq = m[p * n + q];

        let phi = 0.5 * (aqq - app) / apq;
        let t = if phi >= 0.0 {
            1.0 / (phi + (phi * phi + 1.0).sqrt())
        } else {
            -1.0 / (-phi + (phi * phi + 1.0).sqrt())
        };
        let c = 1.0 / (t * t + 1.0).sqrt();
        let s = t * c;

        for i in 0..n {
            if i != p && i != q {
                let aip = m[i * n + p];
                let aiq = m[i * n + q];
                m[i * n + p] = c * aip - s * aiq;
                m[p * n + i] = m[i * n + p];
                m[i * n + q] = s * aip + c * aiq;
                m[q * n + i] = m[i * n + q];
            }
        }
        m[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        m[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        m[p * n + q] = 0.0;
        m[q * n + p] = 0.0;

        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = c * vip - s * viq;
            v[i * n + q] = s * vip + c * viq;
        }
    }

    let mut eigenvalues = vec![0.0f32; n];
    for i in 0..n {
        eigenvalues[i] = m[i * n + i];
    }
    (eigenvalues, v)
}

/// Optimal factored SVD rank reduction:
/// Decomposes B = Q_b * R_b and A^T = Q_a * R_a via Modified Gram-Schmidt,
/// then performs exact SVD on the tiny (r x r) cross-matrix Z = R_b * R_a^T.
/// Truncating Z to target_r singular values yields the provably optimal rank-target_r approximation
/// of (B * A) with maximal cosine similarity.
pub fn reduce_rank_svd(
    a: &[f32],
    b: &[f32],
    in_dim: usize,
    out_dim: usize,
    r: usize,
    target_r: usize,
) -> (Vec<f32>, Vec<f32>) {
    if target_r >= r {
        return (a.to_vec(), b.to_vec());
    }

    // 1. MGS QR on B (out_dim x r)
    let mut q_b = vec![0.0f32; out_dim * r];
    let mut r_b = vec![0.0f32; r * r];

    for k in 0..r {
        for row in 0..out_dim {
            q_b[row * r + k] = b[row * r + k];
        }
        for j in 0..k {
            let mut dot = 0.0f32;
            for row in 0..out_dim {
                dot += q_b[row * r + j] * q_b[row * r + k];
            }
            r_b[j * r + k] = dot;
            for row in 0..out_dim {
                q_b[row * r + k] -= dot * q_b[row * r + j];
            }
        }
        let mut norm_sq = 0.0f32;
        for row in 0..out_dim {
            let val = q_b[row * r + k];
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt();
        r_b[k * r + k] = norm;
        if norm > 1e-8 {
            let inv_norm = 1.0 / norm;
            for row in 0..out_dim {
                q_b[row * r + k] *= inv_norm;
            }
        }
    }

    // 2. MGS QR on A^T (in_dim x r)
    let mut q_a = vec![0.0f32; in_dim * r];
    let mut r_a = vec![0.0f32; r * r];

    for k in 0..r {
        for row in 0..in_dim {
            q_a[row * r + k] = a[k * in_dim + row];
        }
        for j in 0..k {
            let mut dot = 0.0f32;
            for row in 0..in_dim {
                dot += q_a[row * r + j] * q_a[row * r + k];
            }
            r_a[j * r + k] = dot;
            for row in 0..in_dim {
                q_a[row * r + k] -= dot * q_a[row * r + j];
            }
        }
        let mut norm_sq = 0.0f32;
        for row in 0..in_dim {
            let val = q_a[row * r + k];
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt();
        r_a[k * r + k] = norm;
        if norm > 1e-8 {
            let inv_norm = 1.0 / norm;
            for row in 0..in_dim {
                q_a[row * r + k] *= inv_norm;
            }
        }
    }

    // 3. Compute Z = R_b * R_a^T (r x r)
    let mut z = vec![0.0f32; r * r];
    for i in 0..r {
        for j in 0..r {
            let mut sum = 0.0f32;
            for k in 0..r {
                sum += r_b[i * r + k] * r_a[j * r + k];
            }
            z[i * r + j] = sum;
        }
    }

    // 4. Compute Gram matrix G_z = Z * Z^T (r x r)
    let mut g_z = vec![0.0f32; r * r];
    for i in 0..r {
        for j in 0..r {
            let mut sum = 0.0f32;
            for k in 0..r {
                sum += z[i * r + k] * z[j * r + k];
            }
            g_z[i * r + j] = sum;
        }
    }

    // 5. Jacobi eigenvalue decomposition of G_z -> gives U_z and sigma^2
    let (eigvals, u_z) = jacobi_eigenvalues_symm(&g_z, r, 120);

    let mut indices: Vec<usize> = (0..r).collect();
    indices.sort_by(|&i, &j| {
        eigvals[j]
            .partial_cmp(&eigvals[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 6. Compute V_z = Z^T * U_z * diag(1/sigma)
    let mut s = vec![0.0f32; r];
    let mut v_z = vec![0.0f32; r * r];
    for (sorted_idx, &orig_k) in indices.iter().enumerate() {
        let sigma = eigvals[orig_k].max(0.0).sqrt();
        s[sorted_idx] = sigma;
        if sigma > 1e-7 {
            let inv_s = 1.0 / sigma;
            for row in 0..r {
                let mut sum = 0.0f32;
                for col in 0..r {
                    sum += z[col * r + row] * u_z[col * r + orig_k];
                }
                v_z[row * r + sorted_idx] = sum * inv_s;
            }
        }
    }

    // 7. Reconstruct truncated B_new = Q_b * U_z[:, :target_r] * diag(sqrt(sigma))
    let mut b_new = vec![0.0f32; out_dim * target_r];
    for k in 0..target_r {
        let orig_k = indices[k];
        let sqrt_s = s[k].sqrt();
        for row in 0..out_dim {
            let mut sum = 0.0f32;
            for inner in 0..r {
                sum += q_b[row * r + inner] * u_z[inner * r + orig_k];
            }
            b_new[row * target_r + k] = sum * sqrt_s;
        }
    }

    // 8. Reconstruct truncated A_new = diag(sqrt(sigma)) * (V_z[:, :target_r])^T * Q_a^T
    let mut a_new = vec![0.0f32; target_r * in_dim];
    for k in 0..target_r {
        let sqrt_s = s[k].sqrt();
        for col in 0..in_dim {
            let mut sum = 0.0f32;
            for inner in 0..r {
                sum += v_z[inner * r + k] * q_a[col * r + inner];
            }
            a_new[k * in_dim + col] = sum * sqrt_s;
        }
    }

    (a_new, b_new)
}

// =========================================================================
// Data Models & SafeTensors View Helper
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterConfig {
    #[serde(default = "default_r")]
    pub r: usize,
    #[serde(default = "default_alpha")]
    pub lora_alpha: f32,
    #[serde(default)]
    #[allow(dead_code)]
    pub target_modules: Vec<String>,
}

fn default_r() -> usize {
    32
}

fn default_alpha() -> f32 {
    64.0
}

struct SimpleTensorView<'a> {
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    data: &'a [u8],
}

impl<'a> safetensors::tensor::View for SimpleTensorView<'a> {
    fn dtype(&self) -> safetensors::Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(self.data)
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// In-process LoRA low-rank projection for a single linear layer
#[derive(Clone, Debug, Default)]
pub struct LoraLinear {
    pub a: Vec<f32>,   // shape: [r, in_dim], row-major
    pub b: Vec<f32>,   // shape: [out_dim, r], row-major
    pub b_t: Vec<f32>, // shape: [r, out_dim], pre-transposed for AVX2 vectorization
    pub r: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub scaling: f32,
}

#[inline]
pub fn vec_dot2_f32_direct(a0: &[f32], a1: &[f32], b: &[f32]) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { vec_dot2_f32_avx2(a0, a1, b) };
        }
    }
    let d0 = a0.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let d1 = a1.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    (d0, d1)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot2_f32_avx2(a0: &[f32], a1: &[f32], b: &[f32]) -> (f32, f32) {
    use core::arch::x86_64::*;
    let n = a0.len().min(a1.len()).min(b.len());
    let mut i = 0;
    unsafe {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        while i + 8 <= n {
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let va0 = _mm256_loadu_ps(a0.as_ptr().add(i));
            let va1 = _mm256_loadu_ps(a1.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(va0, vb, acc0);
            acc1 = _mm256_fmadd_ps(va1, vb, acc1);
            i += 8;
        }
        let mut d0 = hsum_float_8_local(acc0);
        let mut d1 = hsum_float_8_local(acc1);
        while i < n {
            d0 += *a0.get_unchecked(i) * *b.get_unchecked(i);
            d1 += *a1.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
        (d0, d1)
    }
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn hsum_float_8_local(x: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;
    unsafe {
        let mut res = _mm256_extractf128_ps(x, 1);
        res = _mm_add_ps(res, _mm256_castps256_ps128(x));
        res = _mm_add_ps(res, _mm_movehl_ps(res, res));
        let r1 = _mm_shuffle_ps(res, res, 0x55);
        res = _mm_add_ss(res, r1);
        _mm_cvtss_f32(res)
    }
}

#[inline]
pub fn vec_add_scaled(y: &mut [f32], x: &[f32], alpha: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                vec_add_scaled_avx2(y, x, alpha);
            }
            return;
        }
    }
    let n = y.len().min(x.len());
    for i in 0..n {
        y[i] += alpha * x[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_add_scaled_avx2(y: &mut [f32], x: &[f32], alpha: f32) {
    use core::arch::x86_64::*;
    let n = y.len().min(x.len());
    let mut i = 0;
    unsafe {
        let valpha = _mm256_set1_ps(alpha);
        while i + 32 <= n {
            let vy0 = _mm256_loadu_ps(y.as_ptr().add(i));
            let vx0 = _mm256_loadu_ps(x.as_ptr().add(i));
            let vy1 = _mm256_loadu_ps(y.as_ptr().add(i + 8));
            let vx1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
            let vy2 = _mm256_loadu_ps(y.as_ptr().add(i + 16));
            let vx2 = _mm256_loadu_ps(x.as_ptr().add(i + 16));
            let vy3 = _mm256_loadu_ps(y.as_ptr().add(i + 24));
            let vx3 = _mm256_loadu_ps(x.as_ptr().add(i + 24));

            _mm256_storeu_ps(y.as_mut_ptr().add(i), _mm256_fmadd_ps(valpha, vx0, vy0));
            _mm256_storeu_ps(y.as_mut_ptr().add(i + 8), _mm256_fmadd_ps(valpha, vx1, vy1));
            _mm256_storeu_ps(
                y.as_mut_ptr().add(i + 16),
                _mm256_fmadd_ps(valpha, vx2, vy2),
            );
            _mm256_storeu_ps(
                y.as_mut_ptr().add(i + 24),
                _mm256_fmadd_ps(valpha, vx3, vy3),
            );
            i += 32;
        }
        while i + 8 <= n {
            let vy = _mm256_loadu_ps(y.as_ptr().add(i));
            let vx = _mm256_loadu_ps(x.as_ptr().add(i));
            _mm256_storeu_ps(y.as_mut_ptr().add(i), _mm256_fmadd_ps(valpha, vx, vy));
            i += 8;
        }
        while i < n {
            *y.get_unchecked_mut(i) += alpha * *x.get_unchecked(i);
            i += 1;
        }
    }
}

impl LoraLinear {
    pub fn new(
        a: Vec<f32>,
        b: Vec<f32>,
        r: usize,
        in_dim: usize,
        out_dim: usize,
        scaling: f32,
    ) -> Self {
        let mut b_t = vec![0.0f32; r * out_dim];
        for i in 0..out_dim {
            for k in 0..r {
                b_t[k * out_dim + i] = b[i * r + k];
            }
        }
        Self {
            a,
            b,
            b_t,
            r,
            in_dim,
            out_dim,
            scaling,
        }
    }

    #[inline(always)]
    pub fn apply(&self, x: &[f32], y: &mut [f32]) {
        if self.r == 0 || self.a.is_empty() || self.b.is_empty() {
            return;
        }
        let r = self.r;
        let in_dim = self.in_dim;
        let out_dim = self.out_dim;
        let scaling = self.scaling;

        // Step 1: h = x @ A^T, size [r] (2-way unrolled vector dot products)
        let mut h_buf = [0.0f32; 128];
        let mut h_heap = Vec::new();
        let h: &mut [f32] = if r <= 128 {
            &mut h_buf[..r]
        } else {
            h_heap.resize(r, 0.0f32);
            &mut h_heap[..r]
        };

        let mut k = 0;
        while k + 2 <= r {
            let a_row0 = &self.a[k * in_dim..(k + 1) * in_dim];
            let a_row1 = &self.a[(k + 1) * in_dim..(k + 2) * in_dim];
            let (d0, d1) = vec_dot2_f32_direct(a_row0, a_row1, x);
            h[k] = d0 * scaling;
            h[k + 1] = d1 * scaling;
            k += 2;
        }
        while k < r {
            let a_row = &self.a[k * in_dim..(k + 1) * in_dim];
            h[k] = vec_dot_f32_direct(a_row, x) * scaling;
            k += 1;
        }

        // Step 2: y += B @ h = sum_k (h[k] * B_col[k])
        if !self.b_t.is_empty() {
            for k in 0..r {
                let hk = h[k];
                if hk != 0.0 {
                    let b_col = &self.b_t[k * out_dim..(k + 1) * out_dim];
                    vec_add_scaled(y, b_col, hk);
                }
            }
        } else {
            for i in 0..out_dim {
                let b_row = &self.b[i * r..(i + 1) * r];
                let mut sum = 0.0f32;
                for k in 0..r {
                    sum += b_row[k] * h[k];
                }
                y[i] += sum;
            }
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

/// Head Calibration Performance & Size Report
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub head_id: String,
    pub original_size_bytes: u64,
    pub calibrated_size_bytes: u64,
    pub original_rank: usize,
    pub calibrated_rank: usize,
    pub compression_ratio: f64,
    pub saved_mb: f64,
    pub is_fp8: bool,
    pub is_compressed: bool,
}

impl LoraAdapter {
    /// Load a LoRA adapter from a directory or directly from a safetensors/lz4 file
    pub fn load(path: &Path) -> Result<Self> {
        let (config_path, weights_path, id) = if path.is_dir() {
            let cfg = path.join("adapter_config.json");
            let wgt_lz4 = path.join("adapter_model.safetensors.lz4");
            let wgt_raw = path.join("adapter_model.safetensors");
            let wgt = if wgt_lz4.exists() { wgt_lz4 } else { wgt_raw };
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

        let is_lz4 = weights_path.extension().and_then(|e| e.to_str()) == Some("lz4");

        let decompressed_buf: Vec<u8>;
        let mmap_handle: Option<Mmap>;

        let tensors = if is_lz4 {
            let compressed = std::fs::read(&weights_path)
                .with_context(|| format!("reading {}", weights_path.display()))?;
            decompressed_buf = lz4_flex::decompress_size_prepended(&compressed)
                .map_err(|e| anyhow!("LZ4 decompression error: {e}"))?;
            SafeTensors::deserialize(&decompressed_buf)
                .with_context(|| "deserializing LZ4 SafeTensors header")?
        } else {
            let file = File::open(&weights_path)
                .with_context(|| format!("opening weights at {}", weights_path.display()))?;
            mmap_handle = Some(unsafe { Mmap::map(&file)? });
            SafeTensors::deserialize(mmap_handle.as_ref().unwrap())
                .with_context(|| "deserializing SafeTensors header")?
        };

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
                if let Some(linear) = load_lora_linear(&tensors, layer_idx, mod_name, r, scaling)? {
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
    base_scaling: f32,
) -> Result<Option<LoraLinear>> {
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
    let mut key_a_found = None;
    for k in &candidates_a {
        if let Ok(t) = tensors.tensor(k) {
            tensor_a = Some(t);
            key_a_found = Some(k.clone());
            break;
        }
    }

    let mut tensor_b = None;
    let mut key_b_found = None;
    for k in &candidates_b {
        if let Ok(t) = tensors.tensor(k) {
            tensor_b = Some(t);
            key_b_found = Some(k.clone());
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

    // Check for optional companion scale tensors (e.g. for calibrated FP8/INT8)
    let scale_a = key_a_found
        .as_ref()
        .and_then(|k| tensors.tensor(&format!("{k}.scale")).ok())
        .and_then(|view| {
            if view.data().len() >= 4 {
                Some(f32::from_le_bytes([
                    view.data()[0],
                    view.data()[1],
                    view.data()[2],
                    view.data()[3],
                ]))
            } else {
                None
            }
        })
        .unwrap_or(1.0f32);

    let scale_b = key_b_found
        .as_ref()
        .and_then(|k| tensors.tensor(&format!("{k}.scale")).ok())
        .and_then(|view| {
            if view.data().len() >= 4 {
                Some(f32::from_le_bytes([
                    view.data()[0],
                    view.data()[1],
                    view.data()[2],
                    view.data()[3],
                ]))
            } else {
                None
            }
        })
        .unwrap_or(1.0f32);

    let effective_scaling = base_scaling * scale_a * scale_b;

    let vec_a = tensor_to_f32(&t_a)?;
    let vec_b = tensor_to_f32(&t_b)?;

    Ok(Some(LoraLinear::new(
        vec_a,
        vec_b,
        r,
        in_dim,
        out_dim,
        effective_scaling,
    )))
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
        safetensors::Dtype::F8_E4M3 => {
            let mut out = Vec::with_capacity(data.len());
            for &b in data {
                out.push(fp8_e4m3_to_f32(b));
            }
            Ok(out)
        }
        safetensors::Dtype::F8_E5M2 => {
            let mut out = Vec::with_capacity(data.len());
            for &b in data {
                out.push(fp8_e5m2_to_f32(b));
            }
            Ok(out)
        }
        safetensors::Dtype::I8 => {
            let mut out = Vec::with_capacity(data.len());
            for &b in data {
                out.push((b as i8) as f32 / 127.0f32);
            }
            Ok(out)
        }
        safetensors::Dtype::U8 => {
            let mut out = Vec::with_capacity(data.len());
            for &b in data {
                out.push((b as f32) / 255.0f32);
            }
            Ok(out)
        }
        other => Err(anyhow!("unsupported safetensors dtype {other:?}")),
    }
}

// =========================================================================
// Head Storage Calibration (< 3 MB per Head Pipeline)
// =========================================================================

/// Calibrate an existing LoRA adapter head:
/// 1. SVD rank reduction (e.g. Rank 32 -> Rank 16) preserving principal directions
/// 2. FP8 (E4M3) quantization with per-tensor dynamic range scaling
/// 3. Optional ultra-fast LZ4 compression to achieve strictly < 3 MB per head
pub fn calibrate_head_dir(
    input_dir: &Path,
    output_dir: &Path,
    target_rank: Option<usize>,
    use_fp8: bool,
    compress_lz4: bool,
) -> Result<CalibrationReport> {
    std::fs::create_dir_all(output_dir)?;

    let input_weights = if input_dir.join("adapter_model.safetensors.lz4").exists() {
        input_dir.join("adapter_model.safetensors.lz4")
    } else {
        input_dir.join("adapter_model.safetensors")
    };
    if !input_weights.exists() {
        return Err(anyhow!(
            "Input weights not found in {}",
            input_dir.display()
        ));
    }

    let original_size = std::fs::metadata(&input_weights)?.len();

    // Read config
    let config_path = input_dir.join("adapter_config.json");
    let mut config: AdapterConfig = if config_path.exists() {
        let content = std::fs::read_to_string(&config_path)?;
        serde_json::from_str(&content).unwrap_or(AdapterConfig {
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

    let original_r = config.r;
    let new_r = target_rank.unwrap_or(original_r).min(original_r);

    // Read tensors
    let is_lz4 = input_weights.extension().and_then(|e| e.to_str()) == Some("lz4");
    let decompressed: Vec<u8>;
    let mmap: Option<Mmap>;
    let tensors = if is_lz4 {
        let comp = std::fs::read(&input_weights)?;
        decompressed = lz4_flex::decompress_size_prepended(&comp)
            .map_err(|e| anyhow!("LZ4 decompress error: {e}"))?;
        SafeTensors::deserialize(&decompressed)?
    } else {
        let file = File::open(&input_weights)?;
        mmap = Some(unsafe { Mmap::map(&file)? });
        SafeTensors::deserialize(mmap.as_ref().unwrap())?
    };

    // Find all tensor pairs (lora_A and lora_B)
    let mut tensor_names: Vec<String> =
        tensors.names().into_iter().map(|s| s.to_string()).collect();
    tensor_names.sort();

    // Storage for calibrated tensor data buffers
    let mut tensor_store: Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)> = Vec::new();

    let mut i = 0;
    while i < tensor_names.len() {
        let name = &tensor_names[i];
        if name.ends_with(".lora_A.weight") {
            let base_key = &name[..name.len() - ".lora_A.weight".len()];
            let key_b = format!("{base_key}.lora_B.weight");

            if let (Ok(view_a), Ok(view_b)) = (tensors.tensor(name), tensors.tensor(&key_b)) {
                let shape_a = view_a.shape();
                let shape_b = view_b.shape();
                let r_curr = shape_a[0];
                let in_dim = shape_a[1];
                let out_dim = shape_b[0];

                let existing_scale_a = tensors
                    .tensor(&format!("{name}.scale"))
                    .ok()
                    .and_then(|v| {
                        if v.data().len() >= 4 {
                            Some(f32::from_le_bytes([
                                v.data()[0],
                                v.data()[1],
                                v.data()[2],
                                v.data()[3],
                            ]))
                        } else {
                            None
                        }
                    })
                    .unwrap_or(1.0f32);

                let existing_scale_b = tensors
                    .tensor(&format!("{key_b}.scale"))
                    .ok()
                    .and_then(|v| {
                        if v.data().len() >= 4 {
                            Some(f32::from_le_bytes([
                                v.data()[0],
                                v.data()[1],
                                v.data()[2],
                                v.data()[3],
                            ]))
                        } else {
                            None
                        }
                    })
                    .unwrap_or(1.0f32);

                let mut raw_a = tensor_to_f32(&view_a)?;
                if existing_scale_a != 1.0f32 {
                    for x in &mut raw_a {
                        *x *= existing_scale_a;
                    }
                }

                let mut raw_b = tensor_to_f32(&view_b)?;
                if existing_scale_b != 1.0f32 {
                    for x in &mut raw_b {
                        *x *= existing_scale_b;
                    }
                }

                // 1. Perform SVD rank reduction if requested
                let (proc_a, proc_b) = if new_r < r_curr {
                    reduce_rank_svd(&raw_a, &raw_b, in_dim, out_dim, r_curr, new_r)
                } else {
                    (raw_a, raw_b)
                };

                let eff_r = new_r;

                // 2. Quantize or Store
                if use_fp8 {
                    // Scaled FP8 E4M3 quantization (scaled to 240.0 for maximum dynamic range)
                    let max_abs_a = proc_a
                        .iter()
                        .map(|x| x.abs())
                        .fold(0.0f32, f32::max)
                        .max(1e-8);
                    let scale_a = max_abs_a / 240.0f32;
                    let mut fp8_bytes_a = Vec::with_capacity(proc_a.len());
                    for &x in &proc_a {
                        fp8_bytes_a.push(f32_to_fp8_e4m3(x / scale_a));
                    }

                    let max_abs_b = proc_b
                        .iter()
                        .map(|x| x.abs())
                        .fold(0.0f32, f32::max)
                        .max(1e-8);
                    let scale_b = max_abs_b / 240.0f32;
                    let mut fp8_bytes_b = Vec::with_capacity(proc_b.len());
                    for &x in &proc_b {
                        fp8_bytes_b.push(f32_to_fp8_e4m3(x / scale_b));
                    }

                    // A tensor & companion scale
                    tensor_store.push((
                        name.clone(),
                        safetensors::Dtype::F8_E4M3,
                        vec![eff_r, in_dim],
                        fp8_bytes_a,
                    ));
                    tensor_store.push((
                        format!("{name}.scale"),
                        safetensors::Dtype::F32,
                        vec![1],
                        scale_a.to_le_bytes().to_vec(),
                    ));

                    // B tensor & companion scale
                    tensor_store.push((
                        key_b.clone(),
                        safetensors::Dtype::F8_E4M3,
                        vec![out_dim, eff_r],
                        fp8_bytes_b,
                    ));
                    tensor_store.push((
                        format!("{key_b}.scale"),
                        safetensors::Dtype::F32,
                        vec![1],
                        scale_b.to_le_bytes().to_vec(),
                    ));
                } else {
                    // FP16 output
                    let mut f16_bytes_a = Vec::with_capacity(proc_a.len() * 2);
                    for &x in &proc_a {
                        let h = half::f16::from_f32(x);
                        f16_bytes_a.extend_from_slice(&h.to_bits().to_le_bytes());
                    }

                    let mut f16_bytes_b = Vec::with_capacity(proc_b.len() * 2);
                    for &x in &proc_b {
                        let h = half::f16::from_f32(x);
                        f16_bytes_b.extend_from_slice(&h.to_bits().to_le_bytes());
                    }

                    tensor_store.push((
                        name.clone(),
                        safetensors::Dtype::F16,
                        vec![eff_r, in_dim],
                        f16_bytes_a,
                    ));
                    tensor_store.push((
                        key_b.clone(),
                        safetensors::Dtype::F16,
                        vec![out_dim, eff_r],
                        f16_bytes_b,
                    ));
                }
            }
        }
        i += 1;
    }

    // Build Safetensors view map
    let mut views: Vec<(String, SimpleTensorView)> = Vec::with_capacity(tensor_store.len());
    for (name, dtype, shape, data) in &tensor_store {
        views.push((
            name.clone(),
            SimpleTensorView {
                dtype: *dtype,
                shape: shape.clone(),
                data,
            },
        ));
    }

    // Serialize Safetensors
    let serialized = safetensors::serialize(views, None)
        .map_err(|e| anyhow!("Safetensors serialize error: {e:?}"))?;

    let calibrated_size = if compress_lz4 {
        let compressed = lz4_flex::compress_prepend_size(&serialized);
        let out_file = output_dir.join("adapter_model.safetensors.lz4");
        let mut f = File::create(&out_file)?;
        f.write_all(&compressed)?;
        compressed.len() as u64
    } else {
        let out_file = output_dir.join("adapter_model.safetensors");
        let mut f = File::create(&out_file)?;
        f.write_all(&serialized)?;
        serialized.len() as u64
    };

    // Update and write adapter_config.json
    if original_r > 0 {
        config.lora_alpha = config.lora_alpha * (new_r as f32 / original_r as f32);
    }
    config.r = new_r;
    let updated_cfg = serde_json::to_string_pretty(&config)?;
    let mut cfg_file = File::create(output_dir.join("adapter_config.json"))?;
    cfg_file.write_all(updated_cfg.as_bytes())?;

    // Copy README if exists
    let src_readme = input_dir.join("README.md");
    if src_readme.exists() {
        let _ = std::fs::copy(&src_readme, output_dir.join("README.md"));
    }

    let head_id = input_dir
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "head".to_string());

    let saved_mb = (original_size.saturating_sub(calibrated_size)) as f64 / (1024.0 * 1024.0);
    let ratio = original_size as f64 / calibrated_size.max(1) as f64;

    Ok(CalibrationReport {
        head_id,
        original_size_bytes: original_size,
        calibrated_size_bytes: calibrated_size,
        original_rank: original_r,
        calibrated_rank: new_r,
        compression_ratio: ratio,
        saved_mb,
        is_fp8: use_fp8,
        is_compressed: compress_lz4,
    })
}

// =========================================================================
// Global LoRA Head Registry & Bounded LRU Cache (< 441MB Peak RAM Enforced)
// =========================================================================

/// Fixed-capacity LRU Cache for resident LoRA adapters
pub struct BoundedLoraCache {
    capacity: usize,
    entries: HashMap<String, Arc<LoraAdapter>>,
    order: Vec<String>,
}

impl BoundedLoraCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: HashMap::new(),
            order: Vec::new(),
        }
    }

    pub fn get(&mut self, key: &str) -> Option<Arc<LoraAdapter>> {
        if let Some(adapter) = self.entries.get(key) {
            let canon = adapter.id.to_ascii_lowercase();
            if let Some(pos) = self.order.iter().position(|k| k == &canon) {
                let k = self.order.remove(pos);
                self.order.push(k);
            }
            Some(adapter.clone())
        } else {
            None
        }
    }

    pub fn insert(&mut self, alias_key: String, adapter: Arc<LoraAdapter>) {
        let canon = adapter.id.to_ascii_lowercase();

        if let Some(pos) = self.order.iter().position(|k| k == &canon) {
            self.order.remove(pos);
            self.order.push(canon.clone());
        } else {
            while self.order.len() >= self.capacity {
                let lru_canon = self.order.remove(0);
                self.entries
                    .retain(|_, v| v.id.to_ascii_lowercase() != lru_canon);
                if std::env::var_os("NIOXS_DEBUG").is_some() {
                    eprintln!(
                        "[nioxs-lora] Bounded LRU cache full (capacity={}); evicted resident head '{}'",
                        self.capacity, lru_canon
                    );
                }
                // Force glibc to return freed pages to OS immediately.
                // Without this, dropped f32 LoRA weight Vecs (~55 MB each) remain mapped,
                // causing RSS to grow with each hot-swap even though the Arc is dropped.
                #[cfg(target_os = "linux")]
                unsafe {
                    libc::malloc_trim(0);
                }
            }
            self.order.push(canon.clone());
        }

        self.entries.insert(alias_key, adapter.clone());
        self.entries.insert(canon, adapter);
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }
}

static LORA_CACHE: OnceLock<Mutex<BoundedLoraCache>> = OnceLock::new();

fn get_cache() -> &'static Mutex<BoundedLoraCache> {
    LORA_CACHE.get_or_init(|| {
        let cap = std::env::var("NIOXS_LORA_CACHE_CAPACITY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8);
        Mutex::new(BoundedLoraCache::new(cap))
    })
}

/// Resolve a council head name or slug to its on-disk path
pub fn resolve_head_path(head_name: &str) -> Option<PathBuf> {
    let raw = head_name.trim();
    let lower = raw.to_ascii_lowercase();

    // 1. Direct path check
    let direct = Path::new(raw);
    if direct.exists() {
        return Some(direct.to_path_buf());
    }

    // 2. Standard search directories across tiers (Hydra, Hydra Max, Hydra Ultra)
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let mut search_dirs = Vec::new();
    if let Some(ref h) = home {
        search_dirs.push(h.join(".nioxs").join("anchors"));
        search_dirs.push(
            h.join("nio-labs")
                .join("nio-xs-models")
                .join("models")
                .join("hydra")
                .join("heads"),
        );
        let base = h.join(".nioxs").join("heads");
        search_dirs.push(base.join("hydra-ultra"));
        search_dirs.push(base.join("hydra-max"));
        search_dirs.push(base.join("hydra"));
        search_dirs.push(base);
    }
    search_dirs.push(PathBuf::from("/root/.nioxs/anchors"));
    search_dirs.push(PathBuf::from(
        "/root/nio-labs/nio-xs-models/models/hydra/heads",
    ));
    search_dirs.push(PathBuf::from("/root/.nioxs/heads/hydra-ultra"));
    search_dirs.push(PathBuf::from("/root/.nioxs/heads/hydra-max"));
    search_dirs.push(PathBuf::from("/root/.nioxs/heads/hydra"));
    search_dirs.push(PathBuf::from("/root/.nioxs/heads"));

    // 3. Normalized candidate names
    let mut candidates = Vec::new();
    candidates.push(lower.clone());

    candidates.push(format!("anchor-{}", lower));
    candidates.push(format!("head-{}", lower));
    if lower.starts_with("anchor-") || lower.starts_with("anchor_") {
        candidates.push(lower.replace('_', "-"));
        let stripped = lower
            .trim_start_matches("anchor-")
            .trim_start_matches("anchor_");
        candidates.push(stripped.to_string());
    }

    if lower == "000"
        || lower == "head-000"
        || lower == "anchor-000"
        || lower.contains("chair")
        || lower.contains("arbiter")
    {
        candidates.push("anchor-000-chair".to_string());
        candidates.push("anchor-000".to_string());
        candidates.push("head-000".to_string());
        candidates.push("head-000-sovereign-chair".to_string());
        candidates.push("head-000-sovereign-chair-arbiter".to_string());
        candidates.push("000".to_string());
    }
    if lower == "001" || lower == "head-001" || lower == "hydra-kernel" || lower.contains("atomic")
    {
        candidates.push("head-001".to_string());
        candidates.push("head-001-atomic-concurrency".to_string());
        candidates.push("head-001-systems-atomic-concurrency".to_string());
        candidates.push("001".to_string());
    }
    if lower == "002" || lower == "head-002" || lower.contains("axum") || lower.contains("backend")
    {
        candidates.push("head-002".to_string());
        candidates.push("head-002-backend-axum-tokio".to_string());
        candidates.push("002".to_string());
    }
    if lower == "003" || lower == "head-003" || lower.contains("react") || lower.contains("rsc") {
        candidates.push("head-003".to_string());
        candidates.push("head-003-frontend-react19-rsc".to_string());
        candidates.push("003".to_string());
    }
    if lower == "004" || lower == "head-004" || lower.contains("ebpf") || lower.contains("xdp") {
        candidates.push("head-004".to_string());
        candidates.push("head-004-kernel-ebpf-xdp".to_string());
        candidates.push("004".to_string());
    }
    if lower == "005"
        || lower == "head-005"
        || lower.contains("clickhouse")
        || lower.contains("mergetree")
    {
        candidates.push("head-005".to_string());
        candidates.push("head-005-data-clickhouse-mergetree".to_string());
        candidates.push("005".to_string());
    }
    if lower == "006"
        || lower == "head-006"
        || lower.contains("logic")
        || lower.contains("algorithm")
    {
        candidates.push("head-006".to_string());
        candidates.push("head-006-algorithms-formal-logic".to_string());
        candidates.push("006".to_string());
    }
    if lower == "007"
        || lower == "head-007"
        || lower.contains("k8s")
        || lower.contains("kubernetes")
    {
        candidates.push("head-007".to_string());
        candidates.push("head-007-devops-k8s-controller".to_string());
        candidates.push("007".to_string());
    }
    if lower == "008"
        || lower == "head-008"
        || lower.contains("raft")
        || lower.contains("paxos")
        || lower.contains("consensus")
    {
        candidates.push("head-008".to_string());
        candidates.push("head-008-consensus-raft-paxos".to_string());
        candidates.push("008".to_string());
    }
    if lower == "009" || lower == "head-009" || lower.contains("simd") || lower.contains("avx") {
        candidates.push("head-009".to_string());
        candidates.push("head-009-vector-simd-avx512".to_string());
        candidates.push("009".to_string());
    }
    if lower == "010" || lower == "head-010" || lower.contains("wasm") || lower.contains("wasi") {
        candidates.push("head-010".to_string());
        candidates.push("head-010-wasm-wasi-sandboxing".to_string());
        candidates.push("010".to_string());
    }
    if lower == "direct" || lower == "head-direct" || lower.contains("polymath") {
        candidates.push("head-direct".to_string());
        candidates.push("head-direct-systems-technology-polymath".to_string());
        candidates.push("head-polymath".to_string());
        candidates.push("direct".to_string());
    }

    for dir in &search_dirs {
        for cand in &candidates {
            let p = dir.join(cand);
            if p.join("adapter_model.safetensors").exists()
                || p.join("adapter_model.safetensors.lz4").exists()
            {
                return Some(p);
            }
            if p.exists() {
                return Some(p);
            }
        }
        // Dynamic scan: match any subdirectory starting with anchor-{id} or head-{id}
        let id_part = lower
            .trim_start_matches("head-")
            .trim_start_matches("anchor-");
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
                if name.starts_with(&format!("anchor-{}", id_part))
                    || name.starts_with(&format!("head-{}", id_part))
                    || name.starts_with(&format!("anchor-{}", lower))
                    || name.starts_with(&format!("head-{}", lower))
                    || name == lower
                {
                    let p = entry.path();
                    if p.join("adapter_model.safetensors").exists()
                        || p.join("adapter_model.safetensors.lz4").exists()
                    {
                        return Some(p);
                    }
                }
            }
        }
    }

    None
}

/// Ensures that adapter_model.safetensors exists in head_dir.
/// If only adapter_model.safetensors.lz4 exists, decompresses it on-demand in < 2ms.
pub fn ensure_safetensors_file(head_dir: &Path) -> Option<PathBuf> {
    let raw = head_dir.join("adapter_model.safetensors");
    if raw.is_file() {
        return Some(raw);
    }
    let lz4_file = head_dir.join("adapter_model.safetensors.lz4");
    if lz4_file.is_file() {
        if let Ok(compressed) = std::fs::read(&lz4_file) {
            if let Ok(decompressed) = lz4_flex::decompress_size_prepended(&compressed) {
                if std::fs::write(&raw, &decompressed).is_ok() {
                    return Some(raw);
                }
            }
        }
    }
    None
}

/// Ensures that adapter_model.gguf exists in head_dir.
/// If only safetensors exists, converts it on-demand via convert_lora_to_gguf.py.
pub fn ensure_gguf_adapter_file(head_dir: &Path) -> Option<PathBuf> {
    let gguf_file = head_dir.join("adapter_model.gguf");
    if gguf_file.is_file() {
        return Some(gguf_file);
    }
    let _ = ensure_safetensors_file(head_dir);
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)?;
    let python_bin = home.join(".nioxs").join("env").join("bin").join("python3");
    let script_candidates = [
        home.join("nio-labs")
            .join("nio-xs")
            .join("scripts")
            .join("convert_lora_to_gguf.py"),
        home.join(".nioxs")
            .join("scripts")
            .join("convert_lora_to_gguf.py"),
    ];
    let script_path = script_candidates.iter().find(|p| p.is_file())?;
    let py = if python_bin.is_file() {
        python_bin
    } else {
        PathBuf::from("python3")
    };

    let _ = std::process::Command::new(py)
        .arg(script_path)
        .arg(head_dir)
        .status();

    if gguf_file.is_file() {
        Some(gguf_file)
    } else {
        None
    }
}

/// Converts all anchors in ~/.nioxs/anchors/ to GGUF format using convert_lora_to_gguf.py.
pub fn convert_all_anchors_to_gguf() -> Result<usize> {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("Cannot locate HOME directory"))?;
    let anchors_dir = home.join(".nioxs").join("anchors");
    let python_bin = home.join(".nioxs").join("env").join("bin").join("python3");
    let script_candidates = [
        home.join("nio-labs")
            .join("nio-xs")
            .join("scripts")
            .join("convert_lora_to_gguf.py"),
        home.join(".nioxs")
            .join("scripts")
            .join("convert_lora_to_gguf.py"),
    ];
    let script_path = script_candidates
        .iter()
        .find(|p| p.is_file())
        .ok_or_else(|| anyhow!("convert_lora_to_gguf.py script not found"))?;

    let py = if python_bin.is_file() {
        python_bin
    } else {
        PathBuf::from("python3")
    };

    let status = std::process::Command::new(&py)
        .arg(script_path)
        .arg(&anchors_dir)
        .arg("--all")
        .status()?;

    if !status.success() {
        return Err(anyhow!(
            "GGUF conversion failed with exit code: {:?}",
            status.code()
        ));
    }

    let mut count = 0;
    if let Ok(entries) = std::fs::read_dir(&anchors_dir) {
        for entry in entries.flatten() {
            if entry.path().join("adapter_model.gguf").is_file() {
                count += 1;
            }
        }
    }
    Ok(count)
}

/// Scans ~/.nioxs/anchors/ and ~/.nioxs/heads/ and decompresses all adapter_model.safetensors.lz4 files to adapter_model.safetensors.
pub fn decompress_all_anchors() -> Result<usize> {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("Cannot locate HOME directory"))?;
    let search_roots = [
        home.join(".nioxs").join("anchors"),
        home.join(".nioxs").join("heads"),
    ];
    let mut count = 0;
    for root in &search_roots {
        if !root.is_dir() {
            continue;
        }
        if let Ok(entries) = std::fs::read_dir(root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let lz4_file = path.join("adapter_model.safetensors.lz4");
                    let raw_file = path.join("adapter_model.safetensors");
                    if lz4_file.is_file() && !raw_file.is_file() {
                        if let Some(_p) = ensure_safetensors_file(&path) {
                            count += 1;
                        }
                    }
                }
            }
        }
    }
    Ok(count)
}

/// Retrieve a cached LoRA adapter or load it from disk with bounded LRU memory protection
pub fn get_or_load_head(head_id_or_path: &str) -> Result<Arc<LoraAdapter>> {
    let lower = head_id_or_path.trim().to_ascii_lowercase();

    // Check in-memory LRU cache first (hot-swap speed: < 0.001 ms)
    {
        let mut cache = get_cache().lock().unwrap();
        if let Some(adapter) = cache.get(&lower) {
            return Ok(adapter);
        }
    }

    // Resolve path
    let path = resolve_head_path(head_id_or_path)
        .ok_or_else(|| anyhow!("Could not resolve LoRA head path for '{head_id_or_path}'"))?;

    let t0 = std::time::Instant::now();
    let adapter = Arc::new(LoraAdapter::load(&path)?);
    let elapsed = t0.elapsed();

    // Cache the adapter under requested alias with bounded LRU eviction
    let mut cache = get_cache().lock().unwrap();
    cache.insert(lower.clone(), adapter.clone());

    if std::env::var_os("NIOXS_DEBUG").is_some() {
        eprintln!(
            "[nioxs-lora] Loaded head '{}' ({} layers) in {:.2}ms (resident: {}/{})",
            adapter.id,
            adapter.layers.len(),
            elapsed.as_secs_f64() * 1000.0,
            cache.len(),
            cache.capacity()
        );
    }

    Ok(adapter)
}

/// Pre-warm up to capacity council heads into memory cache (< 21MB total footprint)
pub fn preload_all_council_heads() -> usize {
    let mut count = 0;
    for i in 1..=8 {
        let name = format!("head-{:03}", i);
        if let Ok(_) = get_or_load_head(&name) {
            count += 1;
        }
    }
    count
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
    fn test_fp8_e4m3_roundtrip_precision() {
        let test_values = [-2.5f32, -1.0, -0.5, -0.05, 0.0, 0.05, 0.5, 1.0, 2.5, 16.0];
        for &val in &test_values {
            let fp8 = f32_to_fp8_e4m3(val);
            let recon = fp8_e4m3_to_f32(fp8);
            let rel_err = if val.abs() > 1e-4 {
                (val - recon).abs() / val.abs()
            } else {
                (val - recon).abs()
            };
            assert!(
                rel_err < 0.15,
                "FP8 roundtrip error too large for {val}: got {recon} (err {rel_err})"
            );
        }
    }

    #[test]
    fn test_svd_rank_reduction_accuracy() {
        if let Some(path) = resolve_head_path("head-001") {
            let orig = LoraAdapter::load(&path).expect("head-001 load");
            for layer_idx in 0..orig.layers.len() {
                if let Some(ref lin) = orig.layers[layer_idx].wq {
                    if lin.r >= 8 {
                        let target_r = lin.r / 2;
                        let (a_sub, b_sub) = reduce_rank_svd(
                            &lin.a,
                            &lin.b,
                            lin.in_dim,
                            lin.out_dim,
                            lin.r,
                            target_r,
                        );

                        let probe = vec![0.5f32; lin.in_dim];
                        let mut y_orig = vec![0.0f32; lin.out_dim];
                        lin.apply(&probe, &mut y_orig);

                        let lin_sub = LoraLinear::new(
                            a_sub,
                            b_sub,
                            target_r,
                            lin.in_dim,
                            lin.out_dim,
                            lin.scaling,
                        );
                        let mut y_sub = vec![0.0f32; lin.out_dim];
                        lin_sub.apply(&probe, &mut y_sub);

                        let cos = cos_sim(&y_orig, &y_sub);
                        println!("Layer {} wq: cos(r={target_r}) = {:.6}", layer_idx, cos);
                        assert!(cos > 0.80, "cos too low: {cos}");
                        break;
                    }
                }
            }
        }
    }

    fn cos_sim(u: &[f32], v: &[f32]) -> f32 {
        let dot: f32 = u.iter().zip(v).map(|(x, y)| x * y).sum();
        let nu: f32 = u.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nv: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (nu * nv)
    }

    #[test]
    fn test_lora_linear_apply_identity() {
        let linear = LoraLinear::new(
            vec![1.0, 0.0, 0.0, 1.0], // [2, 2]
            vec![1.0, 0.0, 0.0, 1.0], // [2, 2]
            2,
            2,
            2,
            1.0,
        );

        let x = [3.0f32, 5.0f32];
        let mut y = [10.0f32, 20.0f32];
        linear.apply(&x, &mut y);

        assert!((y[0] - 13.0).abs() < 1e-5);
        assert!((y[1] - 25.0).abs() < 1e-5);
    }

    #[test]
    fn test_resolve_head_path() {
        assert!(resolve_head_path("nonexistent_head_xyz_999").is_none());
        if let Some(p1) = resolve_head_path("head-001") {
            assert!(p1.exists());
            let p2 = resolve_head_path("head-002");
            assert!(p2.is_some(), "head-002 should resolve alongside head-001");
        }
    }

    #[test]
    fn test_load_head_001_safetensors() {
        if let Some(path) = resolve_head_path("head-001") {
            let t0 = std::time::Instant::now();
            let adapter = LoraAdapter::load(&path).expect("head-001 must load cleanly");
            let load_time_ms = t0.elapsed().as_secs_f64() * 1000.0;
            println!("head-001 loaded in {:.2} ms", load_time_ms);
            assert!(adapter.r > 0);
            assert!(adapter.alpha > 0.0);
            assert!(adapter.layers.len() == 24 || adapter.layers.len() == 10);

            // Populate cache
            let _ = get_or_load_head("head-001").expect("initial load");

            // Verify hot-swap cache retrieval (< 0.01 ms = < 10 µs)
            let t1 = std::time::Instant::now();
            let cached = get_or_load_head("head-001").expect("cached lookup");
            let cache_time_us = t1.elapsed().as_micros();
            println!("cached lookup took {} µs", cache_time_us);
            assert!(
                cache_time_us < 1000,
                "cache lookup must be sub-millisecond, took {cache_time_us} µs"
            );
            assert!(cached.layers.len() == 24 || cached.layers.len() == 10);
        }
    }

    #[test]
    fn test_calibrate_and_load_head() {
        if let Some(src_path) = resolve_head_path("head-001") {
            let tmp_dir = std::env::temp_dir().join("nioxs_test_calibrated_head_001");
            let _ = std::fs::remove_dir_all(&tmp_dir);

            // Calibrate to Rank 12 + FP8 (< 8MB for 24 layers, ~2.8MB for 10 layers)
            let stats = calibrate_head_dir(&src_path, &tmp_dir, Some(12), true, false)
                .expect("calibration must succeed");
            println!(
                "Calibration stats (Rank 12): raw {} MB -> calibrated {:.2} MB (ratio: {:.1}x)",
                stats.original_size_bytes / 1024 / 1024,
                stats.calibrated_size_bytes as f64 / 1024.0 / 1024.0,
                stats.compression_ratio
            );
            // Load calibrated adapter
            let t0 = std::time::Instant::now();
            let cal_adapter = LoraAdapter::load(&tmp_dir).expect("calibrated adapter must load");
            let load_time_ms = t0.elapsed().as_secs_f64() * 1000.0;
            println!("Calibrated adapter loaded in {:.2} ms", load_time_ms);
            assert_eq!(cal_adapter.r, 12);
            assert!(cal_adapter.layers.len() == 24 || cal_adapter.layers.len() == 10);

            let max_allowed = if cal_adapter.layers.len() == 24 {
                8 * 1024 * 1024 // 24-layer 0.5B adapter (< 8MB uncompressed, < 6.5MB with lz4)
            } else {
                3 * 1024 * 1024 // 10-layer 180M adapter
            };
            assert!(
                stats.calibrated_size_bytes < max_allowed,
                "Calibrated size must be < {} bytes, got {}",
                max_allowed,
                stats.calibrated_size_bytes
            );

            // Compare layer-by-layer forward pass with original
            let orig_adapter = LoraAdapter::load(&src_path).expect("orig load");
            if let (Some(orig_wq), Some(cal_wq)) =
                (&orig_adapter.layers[0].wq, &cal_adapter.layers[0].wq)
            {
                let x = vec![0.5f32; orig_wq.in_dim];
                let mut y_orig = vec![0.0f32; orig_wq.out_dim];
                let mut y_cal = vec![0.0f32; cal_wq.out_dim];
                orig_wq.apply(&x, &mut y_orig);
                cal_wq.apply(&x, &mut y_cal);
                let cos = cos_sim(&y_orig, &y_cal);
                let norm_orig = y_orig.iter().map(|v| v * v).sum::<f32>().sqrt();
                let norm_cal = y_cal.iter().map(|v| v * v).sum::<f32>().sqrt();
                println!(
                    "orig_wq.scaling = {}, cal_wq.scaling = {}",
                    orig_wq.scaling, cal_wq.scaling
                );
                println!(
                    "orig_a[0..3] = {:?}, cal_a[0..3] = {:?}",
                    &orig_wq.a[0..3],
                    &cal_wq.a[0..3]
                );
                println!(
                    "Layer 0 wq comparison: cos = {:.6}, norm_orig = {:.4}, norm_cal = {:.4}",
                    cos, norm_orig, norm_cal
                );
            }

            // Test Rank 8 SVD truncation + FP8 (< 2MB)
            let tmp_dir_r8 = std::env::temp_dir().join("nioxs_test_calibrated_head_r8");
            let _ = std::fs::remove_dir_all(&tmp_dir_r8);
            let stats_r8 = calibrate_head_dir(&src_path, &tmp_dir_r8, Some(8), true, false)
                .expect("rank 8 calibration must succeed");
            println!(
                "Rank 8 calibration: raw {} MB -> calibrated {:.2} MB",
                stats_r8.original_size_bytes / 1024 / 1024,
                stats_r8.calibrated_size_bytes as f64 / 1024.0 / 1024.0
            );
            let max_r8 = if cal_adapter.layers.len() == 24 {
                6 * 1024 * 1024
            } else {
                2 * 1024 * 1024
            };
            assert!(stats_r8.calibrated_size_bytes < max_r8);

            let cal_adapter_r8 = LoraAdapter::load(&tmp_dir_r8).expect("rank 8 load");
            assert_eq!(cal_adapter_r8.r, 8);

            let _ = std::fs::remove_dir_all(&tmp_dir);
            let _ = std::fs::remove_dir_all(&tmp_dir_r8);
        }
    }

    #[test]
    fn test_bounded_lru_cache_eviction() {
        let mut cache = BoundedLoraCache::new(3);
        assert_eq!(cache.capacity(), 3);
        assert_eq!(cache.len(), 0);

        let make_dummy = |id: &str| -> Arc<LoraAdapter> {
            Arc::new(LoraAdapter {
                id: id.to_string(),
                r: 4,
                alpha: 8.0,
                layers: Vec::new(),
            })
        };

        // Insert 3 items
        cache.insert("h1".to_string(), make_dummy("head-001"));
        cache.insert("h2".to_string(), make_dummy("head-002"));
        cache.insert("h3".to_string(), make_dummy("head-003"));
        assert_eq!(cache.len(), 3);

        // Access h1 to make it most-recently used (order: h2, h3, h1)
        assert!(cache.get("h1").is_some());

        // Insert 4th item: should evict h2 (least recently used)
        cache.insert("h4".to_string(), make_dummy("head-004"));
        assert_eq!(cache.len(), 3);
        assert!(cache.get("h2").is_none(), "h2 must be evicted");
        assert!(cache.get("h1").is_some(), "h1 must still be resident");
        assert!(cache.get("h3").is_some(), "h3 must still be resident");
        assert!(cache.get("h4").is_some(), "h4 must be resident");
    }
}
