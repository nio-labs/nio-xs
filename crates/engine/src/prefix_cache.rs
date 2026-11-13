use anyhow::Result;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::ast_engine::AstMicroEngine;

pub fn resolve_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("NIOXS_CACHE_DIR") {
        let p = PathBuf::from(dir);
        if !p.as_os_str().is_empty() {
            return p;
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".nioxs").join("cache")
    } else {
        PathBuf::from(".nioxs").join("cache")
    }
}

const MAGIC: &[u8; 8] = b"NIOXSPFX";
const VERSION: u32 = 1;

pub fn save_prefix_cache(path: &Path, tokens: &[u32], engine: &AstMicroEngine) -> Result<()> {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, file);

    writer.write_all(MAGIC)?;
    writer.write_all(&VERSION.to_le_bytes())?;

    // Tokens
    writer.write_all(&(tokens.len() as u32).to_le_bytes())?;
    for &tok in tokens {
        writer.write_all(&tok.to_le_bytes())?;
    }

    // x_norm
    let x_norm = &engine.scratch.x_norm;
    writer.write_all(&(x_norm.len() as u32).to_le_bytes())?;
    let x_bytes = unsafe {
        std::slice::from_raw_parts(
            x_norm.as_ptr() as *const u8,
            x_norm.len() * std::mem::size_of::<f32>(),
        )
    };
    writer.write_all(x_bytes)?;

    // kv_caches
    let num_layers = engine.kv_caches.len() as u32;
    writer.write_all(&num_layers.to_le_bytes())?;
    for cache in &engine.kv_caches {
        writer.write_all(&(cache.cur_len as u32).to_le_bytes())?;
        // k
        writer.write_all(&(cache.k.len() as u32).to_le_bytes())?;
        let k_bytes = unsafe {
            std::slice::from_raw_parts(
                cache.k.as_ptr() as *const u8,
                cache.k.len() * std::mem::size_of::<f32>(),
            )
        };
        writer.write_all(k_bytes)?;

        // v
        writer.write_all(&(cache.v.len() as u32).to_le_bytes())?;
        let v_bytes = unsafe {
            std::slice::from_raw_parts(
                cache.v.as_ptr() as *const u8,
                cache.v.len() * std::mem::size_of::<f32>(),
            )
        };
        writer.write_all(v_bytes)?;
    }

    writer.flush()?;
    Ok(())
}

pub fn try_load_prefix_cache(
    path: &Path,
    prompt_tokens: &[u32],
    engine: &mut AstMicroEngine,
) -> Result<Option<Vec<u32>>> {
    if !path.exists() {
        return Ok(None);
    }
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };
    let mut reader = BufReader::with_capacity(4 * 1024 * 1024, file);

    let mut magic = [0u8; 8];
    if reader.read_exact(&mut magic).is_err() || &magic != MAGIC {
        return Ok(None);
    }
    let mut ver_buf = [0u8; 4];
    if reader.read_exact(&mut ver_buf).is_err() || u32::from_le_bytes(ver_buf) != VERSION {
        return Ok(None);
    }

    // Read cached tokens
    let mut u32_buf = [0u8; 4];
    if reader.read_exact(&mut u32_buf).is_err() {
        return Ok(None);
    }
    let num_tokens = u32::from_le_bytes(u32_buf) as usize;
    if num_tokens == 0 || num_tokens > prompt_tokens.len() {
        return Ok(None);
    }

    let mut cached_tokens = vec![0u32; num_tokens];
    let tok_bytes = unsafe {
        std::slice::from_raw_parts_mut(
            cached_tokens.as_mut_ptr() as *mut u8,
            num_tokens * std::mem::size_of::<u32>(),
        )
    };
    if reader.read_exact(tok_bytes).is_err() {
        return Ok(None);
    }

    // Must match prompt prefix exactly
    if !prompt_tokens.starts_with(&cached_tokens) {
        return Ok(None);
    }

    // Read x_norm
    if reader.read_exact(&mut u32_buf).is_err() {
        return Ok(None);
    }
    let x_norm_len = u32::from_le_bytes(u32_buf) as usize;
    if x_norm_len != engine.scratch.x_norm.len() {
        return Ok(None);
    }
    let x_bytes = unsafe {
        std::slice::from_raw_parts_mut(
            engine.scratch.x_norm.as_mut_ptr() as *mut u8,
            x_norm_len * std::mem::size_of::<f32>(),
        )
    };
    if reader.read_exact(x_bytes).is_err() {
        return Ok(None);
    }

    // Read kv_caches
    if reader.read_exact(&mut u32_buf).is_err() {
        return Ok(None);
    }
    let num_layers = u32::from_le_bytes(u32_buf) as usize;
    if num_layers != engine.kv_caches.len() {
        return Ok(None);
    }

    for cache in &mut engine.kv_caches {
        // cur_len
        if reader.read_exact(&mut u32_buf).is_err() {
            return Ok(None);
        }
        cache.cur_len = u32::from_le_bytes(u32_buf) as usize;

        // k
        if reader.read_exact(&mut u32_buf).is_err() {
            return Ok(None);
        }
        let k_len = u32::from_le_bytes(u32_buf) as usize;
        cache.k.resize(k_len, 0.0);
        let k_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                cache.k.as_mut_ptr() as *mut u8,
                k_len * std::mem::size_of::<f32>(),
            )
        };
        if reader.read_exact(k_bytes).is_err() {
            return Ok(None);
        }

        // v
        if reader.read_exact(&mut u32_buf).is_err() {
            return Ok(None);
        }
        let v_len = u32::from_le_bytes(u32_buf) as usize;
        cache.v.resize(v_len, 0.0);
        let v_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                cache.v.as_mut_ptr() as *mut u8,
                v_len * std::mem::size_of::<f32>(),
            )
        };
        if reader.read_exact(v_bytes).is_err() {
            return Ok(None);
        }
    }

    Ok(Some(cached_tokens))
}
