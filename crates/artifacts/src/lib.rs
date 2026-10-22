use serde::{Deserialize, Serialize};

pub mod model_downloader;
pub use model_downloader::{
    ConsoleProgress, ModelArtifactEntry, ProgressListener, SilentProgress, default_manifest,
    download_and_verify_model, ensure_model_ready, get_manifest_entry, is_model_ready,
    prompt_and_pull_if_needed, resolve_models_dir, verify_file_sha256,
};

/// Contract only. Verification and installation belong to Phase 5.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactDescriptor {
    pub name: String,
    pub version: String,
    pub platform: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub url: String,
}
