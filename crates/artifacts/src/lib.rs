use serde::{Deserialize, Serialize};

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
