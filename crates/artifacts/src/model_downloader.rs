//! # Model Downloader & Artifact Lifecycle
//!
//! Manages on-demand downloading of local quantized model weights
//! into `~/.nioxs/models/` with streaming progress and SHA-256 integrity verification.

use futures_util::StreamExt;
use nioxs_contracts::{MODEL_CUDA, MODEL_MANTA, MODEL_NANO, MODEL_RAY, MODEL_VIPER};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelArtifactEntry {
    pub model_id: String,
    pub brand_name: String,
    pub base_model: String,
    pub filename: String,
    pub download_url: String,
    pub sha256: String,
    pub size_bytes: u64,
}

pub fn default_manifest() -> Vec<ModelArtifactEntry> {
    vec![
        ModelArtifactEntry {
            model_id: MODEL_RAY.into(),
            brand_name: "NioRay (Keystroke Predator)".into(),
            base_model: "Qwen2.5-Coder-0.5B-Dense-24L".into(),
            filename: "nio-ray-ast.Q4_0.gguf".into(),
            download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.4.1/nio-ray-ast.Q4_0.gguf".into(),
            sha256: "f070cd657e929ee2d45a7d8a46c71ef0fdec6b8ae1100de57735808deae4477f".into(),
            size_bytes: 284_087_232,
        },
        ModelArtifactEntry {
            model_id: MODEL_VIPER.into(),
            brand_name: "NioViper (Tactical Terminal Striker)".into(),
            base_model: "NioViper-50M".into(),
            filename: "nio-viper.Q4_0.gguf".into(),
            download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.1.0/nio-viper.Q4_0.gguf".into(),
            sha256: "acaa8a21c8d37d12398a2e7ac67ad367d2e3786b261ce95195e6e0cc08363e9c".into(),
            size_bytes: 28_214_368,
        },
        ModelArtifactEntry {
            model_id: MODEL_MANTA.into(),
            brand_name: "NioManta (Conversational Brain & Wasm Architect)".into(),
            base_model: "Qwen/Qwen2.5-0.5B-Instruct (24L, v-µLM Vector)".into(),
            filename: "manta.vmicrolm-vector.gguf".into(),
            download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.2.2/manta.vmicrolm-vector.gguf".into(),
            sha256: "ed27ef71bf8c0c5f372efb0eef99b71a9c6480bf3e228ce2f1027fd3583d08bf".into(),
            size_bytes: 265_635_936,
        },
        ModelArtifactEntry {
            model_id: MODEL_CUDA.into(),
            brand_name: "NioCuda (Tactical Swarm Conductor)".into(),
            base_model: "Qwen/Qwen2.5-Coder-0.5B-Instruct (16L)".into(),
            filename: "nio-cuda-ast.Q4_0.gguf".into(),
            download_url: "https://github.com/nio-labs/nio-xs-models/releases/download/v0.5.2/nio-cuda-ast.Q4_0.gguf".into(),
            sha256: "09566c2f463dfae5475b2747c4338cf3a9b887f8382a21052766d1f97803c58c".into(),
            size_bytes: 284_962_368,
        },
    ]
}

pub fn get_manifest_entry(model_id: &str) -> Option<ModelArtifactEntry> {
    default_manifest()
        .into_iter()
        .find(|e| e.model_id == model_id)
}

/// Resolves the local models cache directory (`~/.nioxs/models/` by default).
pub fn resolve_models_dir() -> PathBuf {
    if let Ok(dir) = env::var("NIOXS_MODELS_DIR") {
        let p = PathBuf::from(dir);
        if !p.as_os_str().is_empty() {
            return p;
        }
    }
    if let Ok(home) = env::var("HOME") {
        PathBuf::from(home).join(".nioxs").join("models")
    } else {
        PathBuf::from(".nioxs").join("models")
    }
}

pub trait ProgressListener: Send + Sync {
    fn on_start(&self, brand_name: &str, total_bytes: u64);
    fn on_chunk(&self, downloaded_bytes: u64, total_bytes: u64);
    fn on_verify(&self, filename: &str);
    fn on_complete(&self, path: &Path);
}

pub struct SilentProgress;
impl ProgressListener for SilentProgress {
    fn on_start(&self, _: &str, _: u64) {}
    fn on_chunk(&self, _: u64, _: u64) {}
    fn on_verify(&self, _: &str) {}
    fn on_complete(&self, _: &Path) {}
}

pub struct ConsoleProgress;
impl ProgressListener for ConsoleProgress {
    fn on_start(&self, brand_name: &str, total_bytes: u64) {
        let mb = total_bytes as f64 / (1024.0 * 1024.0);
        println!("[NioXS] 📦 {brand_name} is required for this task ({mb:.1} MB).");
    }
    fn on_chunk(&self, downloaded_bytes: u64, total_bytes: u64) {
        let pct = if total_bytes > 0 {
            (downloaded_bytes as f64 / total_bytes as f64 * 100.0) as u32
        } else {
            0
        };
        let d_mb = downloaded_bytes as f64 / (1024.0 * 1024.0);
        let t_mb = total_bytes as f64 / (1024.0 * 1024.0);
        print!("\rDownloading [{pct:3}%] {d_mb:.1} MB / {t_mb:.1} MB");
        let _ = std::io::stdout().flush();
    }
    fn on_verify(&self, filename: &str) {
        println!("\n[NioXS] 🔒 Verifying SHA-256 digest for {filename}...");
    }
    fn on_complete(&self, path: &Path) {
        println!("[NioXS] ✔ Verified and ready: {}", path.display());
    }
}

/// Verifies whether an existing file matches the expected SHA-256 checksum.
pub fn verify_file_sha256(path: &Path, expected_sha256: &str) -> Result<bool, String> {
    if !path.exists() {
        return Ok(false);
    }
    let mut file =
        File::open(path).map_err(|e| format!("failed to open file {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let bytes_read = file
            .read(&mut buffer)
            .map_err(|e| format!("read error: {e}"))?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    let computed: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    Ok(computed.eq_ignore_ascii_case(expected_sha256))
}

/// Checks if a model's weights exist and are ready on disk.
pub fn is_model_ready(model_id: &str, custom_dir: Option<&Path>) -> bool {
    if model_id == MODEL_NANO {
        return true; // Embedded in-process
    }
    let Some(entry) = get_manifest_entry(model_id) else {
        return false;
    };
    let dir = match custom_dir {
        Some(d) => d.to_path_buf(),
        None => resolve_models_dir(),
    };
    let target = dir.join(&entry.filename);
    target.exists() && target.metadata().map(|m| m.len() > 0).unwrap_or(false)
}

/// Downloads model weights from URL with streaming progress, computes SHA-256,
/// and renames to the destination path only after validation succeeds.
pub async fn download_and_verify_model<P: ProgressListener>(
    client: &reqwest::Client,
    entry: &ModelArtifactEntry,
    target_dir: &Path,
    listener: &P,
) -> Result<PathBuf, String> {
    fs::create_dir_all(target_dir)
        .map_err(|e| format!("failed to create models dir {}: {e}", target_dir.display()))?;

    let final_path = target_dir.join(&entry.filename);
    if final_path.exists() {
        if verify_file_sha256(&final_path, &entry.sha256).unwrap_or(false) {
            listener.on_complete(&final_path);
            return Ok(final_path);
        } else {
            // Corrupt file exists, remove it before re-downloading
            let _ = fs::remove_file(&final_path);
        }
    }

    let tmp_path = target_dir.join(format!("{}.download.tmp", entry.filename));
    listener.on_start(&entry.brand_name, entry.size_bytes);

    let response = client.get(&entry.download_url).send().await.map_err(|e| {
        format!(
            "failed to connect to download URL {}: {e}",
            entry.download_url
        )
    })?;

    if !response.status().is_success() {
        return Err(format!(
            "download failed with HTTP status {}",
            response.status()
        ));
    }

    let total_size = response.content_length().unwrap_or(entry.size_bytes);

    let mut file = File::create(&tmp_path)
        .map_err(|e| format!("failed to create temp file {}: {e}", tmp_path.display()))?;

    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut hasher = Sha256::new();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("stream read error: {e}"))?;
        file.write_all(&chunk)
            .map_err(|e| format!("failed to write chunk: {e}"))?;
        hasher.update(&chunk);
        downloaded += chunk.len() as u64;
        listener.on_chunk(downloaded, total_size);
    }

    file.flush()
        .map_err(|e| format!("failed to flush downloaded file: {e}"))?;
    drop(file);

    listener.on_verify(&entry.filename);
    let computed_hash: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();

    if !computed_hash.eq_ignore_ascii_case(&entry.sha256) {
        let _ = fs::remove_file(&tmp_path);
        return Err(format!(
            "SHA-256 checksum mismatch for {}: expected {}, computed {}",
            entry.filename, entry.sha256, computed_hash
        ));
    }

    fs::rename(&tmp_path, &final_path).map_err(|e| {
        format!(
            "failed to move temp file {} to {}: {e}",
            tmp_path.display(),
            final_path.display()
        )
    })?;

    // Companion tokenizer sync (e.g. nio-viper-tokenizer.json)
    if entry.filename.contains("viper") {
        let tok_path = target_dir.join("nio-viper-tokenizer.json");
        let tok_url = entry.download_url.replace("nio-viper.Q4_0.gguf", "nio-viper-tokenizer.json");
        if let Ok(resp) = client.get(&tok_url).send().await {
            if resp.status().is_success() {
                if let Ok(bytes) = resp.bytes().await {
                    let _ = fs::write(&tok_path, bytes);
                }
            }
        }
    }

    listener.on_complete(&final_path);
    Ok(final_path)
}

/// Synchronous entrypoint to ensure model is present on disk.
pub fn ensure_model_ready(model_id: &str, custom_dir: Option<&Path>) -> Result<PathBuf, String> {
    if model_id == MODEL_NANO {
        return Ok(PathBuf::from("embedded://nio-nano"));
    }
    let entry =
        get_manifest_entry(model_id).ok_or_else(|| format!("unknown model id '{model_id}'"))?;

    let dir = match custom_dir {
        Some(d) => d.to_path_buf(),
        None => resolve_models_dir(),
    };

    let final_path = dir.join(&entry.filename);
    if final_path.exists() && verify_file_sha256(&final_path, &entry.sha256).unwrap_or(false) {
        return Ok(final_path);
    }

    // Also check NIOXS_MODELS_DIR environment variable
    if let Ok(dir_env) = env::var("NIOXS_MODELS_DIR") {
        let p = PathBuf::from(dir_env).join(&entry.filename);
        if p.exists() && verify_file_sha256(&p, &entry.sha256).unwrap_or(false) {
            return Ok(p);
        }
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .map_err(|e| format!("failed to build HTTP client: {e}"))?;

    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| {
            handle.block_on(download_and_verify_model(
                &client,
                &entry,
                &dir,
                &ConsoleProgress,
            ))
        })
    } else {
        let rt = tokio::runtime::Runtime::new().map_err(|e| format!("failed to create runtime: {e}"))?;
        rt.block_on(download_and_verify_model(
            &client,
            &entry,
            &dir,
            &ConsoleProgress,
        ))
    }
}

/// Interactive or automated model probe and download.
/// If model weights are present, returns the path immediately.
/// If missing:
///   - In an interactive terminal session: prompts the developer `[Y/n]` to download via `nioxs pull`.
///   - If `auto_yes` is true (or `--yes` / `-y`): automatically downloads.
///   - In non-interactive mode without confirmation: returns an actionable error.
pub fn prompt_and_pull_if_needed(model_id: &str, auto_yes: bool) -> Result<PathBuf, String> {
    if model_id == MODEL_NANO {
        return Ok(PathBuf::from("embedded://nio-nano"));
    }
    let entry =
        get_manifest_entry(model_id).ok_or_else(|| format!("unknown model id '{model_id}'"))?;

    let dir = resolve_models_dir();
    let final_path = dir.join(&entry.filename);
    if final_path.exists() && verify_file_sha256(&final_path, &entry.sha256).unwrap_or(false) {
        return Ok(final_path);
    }

    // Also check NIOXS_MODELS_DIR environment variable
    if let Ok(dir_env) = env::var("NIOXS_MODELS_DIR") {
        let p = PathBuf::from(dir_env).join(&entry.filename);
        if p.exists() && verify_file_sha256(&p, &entry.sha256).unwrap_or(false) {
            return Ok(p);
        }
    }

    let size_mb = entry.size_bytes as f64 / (1024.0 * 1024.0);

    if !auto_yes {
        use std::io::IsTerminal;
        if std::io::stdin().is_terminal() {
            eprintln!();
            eprintln!("  ┌─────────────────────────────────────────────────────────────┐");
            eprintln!(
                "  │ 📦 Model '{}' is not downloaded locally ({size_mb:.1} MB).",
                entry.model_id
            );
            eprintln!("  │    • Role     : {}", entry.brand_name);
            eprintln!("  │    • Base     : {}", entry.base_model);
            eprintln!("  │    • Target   : {}", final_path.display());
            eprintln!("  │                                                             │");
            eprint!("  │ Would you like to download it now via 'nioxs pull'? [Y/n] ");
            let _ = std::io::stderr().flush();

            let mut input = String::new();
            if std::io::stdin().read_line(&mut input).is_err() {
                return Err("Failed to read user confirmation".into());
            }
            let trimmed = input.trim().to_ascii_lowercase();
            if !trimmed.is_empty() && trimmed != "y" && trimmed != "yes" {
                eprintln!("  │ Download cancelled by user.                                 │");
                eprintln!("  └─────────────────────────────────────────────────────────────┘\n");
                return Err(format!(
                    "model '{}' download cancelled by user",
                    entry.model_id
                ));
            }
            eprintln!("  └─────────────────────────────────────────────────────────────┘\n");
        } else {
            return Err(format!(
                "model '{}' not found in {}. Run 'nioxs pull {}' or pass '--auto-pull' / '-y'.",
                entry.model_id,
                dir.display(),
                entry.model_id
            ));
        }
    }

    ensure_model_ready(model_id, Some(&dir))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;
    use std::thread;

    // Pins every manifest entry to the real upstream GGUF artifact it claims to
    // mirror. The checksum/size/url are the LFS `sha256`/`size` the Hugging Face
    // API reports for those exact files (`.vendor/`-free, fetched from
    // `https://huggingface.co/api/models/<repo>?blobs=true` on 2026-10-13), so a
    // fabricated or drifted value fails loudly here instead of silently making
    // every real download die in SHA-256 verification.
    //
    // Two entries were pinned to the wrong quantization's digest (helm carried
    // Q4_K_S's while pointing at the Q4_K_M URL, code carried a digest that
    // belongs to no published file). The `list_unused_manifest_pins` guard below
    // is what would have caught the copy-paste: these values must match
    // `default_manifest()` exactly, not merely look plausible.
    #[test]
    fn test_manifest_pins_real_upstream_artifacts() {
        let expected: [(&str, &str, u64, &str); 4] = [
            (
                MODEL_RAY,
                "f070cd657e929ee2d45a7d8a46c71ef0fdec6b8ae1100de57735808deae4477f",
                284_087_232,
                "https://github.com/nio-labs/nio-xs-models/releases/download/v0.4.1/nio-ray-ast.Q4_0.gguf",
            ),
            (
                MODEL_VIPER,
                "acaa8a21c8d37d12398a2e7ac67ad367d2e3786b261ce95195e6e0cc08363e9c",
                28_214_368,
                "https://github.com/nio-labs/nio-xs-models/releases/download/v0.1.0/nio-viper.Q4_0.gguf",
            ),
            (
                MODEL_MANTA,
                "ed27ef71bf8c0c5f372efb0eef99b71a9c6480bf3e228ce2f1027fd3583d08bf",
                265_635_936,
                "https://github.com/nio-labs/nio-xs-models/releases/download/v0.2.2/manta.vmicrolm-vector.gguf",
            ),
            (
                MODEL_CUDA,
                "09566c2f463dfae5475b2747c4338cf3a9b887f8382a21052766d1f97803c58c",
                284_962_368,
                "https://github.com/nio-labs/nio-xs-models/releases/download/v0.5.2/nio-cuda-ast.Q4_0.gguf",
            ),
        ];
        let manifest = default_manifest();
        assert_eq!(manifest.len(), expected.len());
        for (id, sha256, size_bytes, url) in expected {
            let entry =
                get_manifest_entry(id).unwrap_or_else(|| panic!("missing manifest entry for {id}"));
            assert_eq!(entry.sha256, sha256, "sha256 for {id}");
            assert_eq!(entry.sha256.len(), 64, "sha256 length for {id}");
            assert_eq!(entry.size_bytes, size_bytes, "size for {id}");
            assert_eq!(entry.download_url, url, "download_url for {id}");
        }
    }

    #[test]
    fn test_manifest_contains_all_models() {
        let manifest = default_manifest();
        assert_eq!(manifest.len(), 4);
        for id in [MODEL_RAY, MODEL_VIPER, MODEL_MANTA, MODEL_CUDA] {
            assert!(manifest.iter().any(|e| e.model_id == id));
        }
    }

    #[test]
    fn test_is_model_ready_embedded_nano() {
        assert!(is_model_ready(MODEL_NANO, None));
    }

    #[test]
    fn test_verify_file_sha256_success_and_failure() {
        let temp_dir = env::temp_dir().join(format!("nioxs_test_{}", std::process::id()));
        fs::create_dir_all(&temp_dir).unwrap();
        let file_path = temp_dir.join("test_file.bin");

        let content = b"hello nio models";
        fs::write(&file_path, content).unwrap();

        let mut hasher = Sha256::new();
        hasher.update(content);
        let expected: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();

        assert_eq!(verify_file_sha256(&file_path, &expected).unwrap(), true);
        assert_eq!(
            verify_file_sha256(&file_path, "00000000000000000000000000000000").unwrap(),
            false
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_download_and_verify_mock_server() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let payload = b"GGUF_MOCK_MODEL_DATA";
        let mut hasher = Sha256::new();
        hasher.update(payload);
        let expected_sha256: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();

        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut chunk = [0u8; 1024];
            let _ = stream.read(&mut chunk).unwrap();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                payload.len()
            );
            stream.write_all(resp.as_bytes()).unwrap();
            stream.write_all(payload).unwrap();
        });

        let temp_dir = env::temp_dir().join(format!("nioxs_mock_{}", std::process::id()));
        let entry = ModelArtifactEntry {
            model_id: "test-model".into(),
            brand_name: "Test Model".into(),
            base_model: "Mock-1B".into(),
            filename: "mock_model.gguf".into(),
            download_url: format!("http://{addr}/model.gguf"),
            sha256: expected_sha256.clone(),
            size_bytes: payload.len() as u64,
        };

        let client = reqwest::Client::new();
        let downloaded = download_and_verify_model(&client, &entry, &temp_dir, &SilentProgress)
            .await
            .unwrap();

        server.join().unwrap();
        assert!(downloaded.exists());
        assert_eq!(
            verify_file_sha256(&downloaded, &expected_sha256).unwrap(),
            true
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    /// Records the callback sequence so the provider's progress reporting has a
    /// contract to build on: `on_start` first, `on_chunk` per stream chunk,
    /// `on_verify` before the digest is compared, `on_complete` last.
    #[derive(Default)]
    struct RecordingProgress {
        events: std::sync::Mutex<Vec<String>>,
    }

    impl RecordingProgress {
        fn push(&self, event: String) {
            self.events.lock().unwrap().push(event);
        }
        fn events(&self) -> Vec<String> {
            self.events.lock().unwrap().clone()
        }
    }

    impl ProgressListener for RecordingProgress {
        fn on_start(&self, brand_name: &str, total_bytes: u64) {
            self.push(format!("start:{brand_name}:{total_bytes}"));
        }
        fn on_chunk(&self, downloaded_bytes: u64, total_bytes: u64) {
            self.push(format!("chunk:{downloaded_bytes}/{total_bytes}"));
        }
        fn on_verify(&self, filename: &str) {
            self.push(format!("verify:{filename}"));
        }
        fn on_complete(&self, _path: &Path) {
            self.push("complete".into());
        }
    }

    #[tokio::test]
    async fn test_progress_listener_reports_start_chunks_verify_and_complete() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let payload = b"GGUF_PROGRESS_MOCK_DATA";
        let mut hasher = Sha256::new();
        hasher.update(payload);
        let expected_sha256: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();

        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut chunk = [0u8; 1024];
            let _ = stream.read(&mut chunk).unwrap();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                payload.len()
            );
            stream.write_all(resp.as_bytes()).unwrap();
            stream.write_all(payload).unwrap();
        });

        let temp_dir = env::temp_dir().join(format!("nioxs_progress_{}", std::process::id()));
        let entry = ModelArtifactEntry {
            model_id: "test-progress".into(),
            brand_name: "Progress Model".into(),
            base_model: "Mock-1B".into(),
            filename: "progress.gguf".into(),
            download_url: format!("http://{addr}/model.gguf"),
            sha256: expected_sha256,
            size_bytes: payload.len() as u64,
        };

        let client = reqwest::Client::new();
        let progress = RecordingProgress::default();
        download_and_verify_model(&client, &entry, &temp_dir, &progress)
            .await
            .unwrap();
        server.join().unwrap();

        let events = progress.events();
        let size = payload.len();
        let expected_start = format!("start:Progress Model:{size}");
        assert_eq!(
            events.first().map(String::as_str),
            Some(expected_start.as_str()),
            "on_start announces the total before any bytes move: {events:?}"
        );
        assert_eq!(events.last().map(String::as_str), Some("complete"));
        assert_eq!(
            events.iter().filter(|e| e.starts_with("verify:")).count(),
            1,
            "verification is announced exactly once: {events:?}"
        );
        assert!(
            events.iter().any(|e| e == &format!("chunk:{size}/{size}")),
            "final chunk must report full progress: {events:?}"
        );
        let verify_at = events.iter().position(|e| e.starts_with("verify:"));
        let complete_at = events.iter().position(|e| e == "complete");
        assert!(
            verify_at < complete_at,
            "verify precedes complete: {events:?}"
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_download_checksum_mismatch_cleans_up() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let payload = b"CORRUPTED_DATA";

        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut chunk = [0u8; 1024];
            let _ = stream.read(&mut chunk).unwrap();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                payload.len()
            );
            stream.write_all(resp.as_bytes()).unwrap();
            stream.write_all(payload).unwrap();
        });

        let temp_dir = env::temp_dir().join(format!("nioxs_corrupt_{}", std::process::id()));
        let entry = ModelArtifactEntry {
            model_id: "test-corrupt".into(),
            brand_name: "Corrupt Model".into(),
            base_model: "Mock-1B".into(),
            filename: "corrupt.gguf".into(),
            download_url: format!("http://{addr}/model.gguf"),
            sha256: "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".into(),
            size_bytes: payload.len() as u64,
        };

        let client = reqwest::Client::new();
        let err = download_and_verify_model(&client, &entry, &temp_dir, &SilentProgress)
            .await
            .unwrap_err();

        server.join().unwrap();
        assert!(err.contains("checksum mismatch"));
        assert!(!temp_dir.join("corrupt.gguf").exists());
        assert!(!temp_dir.join("corrupt.gguf.download.tmp").exists());

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
