//! # Baby Orca (`15 × 135M` Sequential Micro-Pod Fleet)
//!
//! Ultra-light edge fleet (`~1.5 GB` disk total, `~200-300 MB` active RAM — 1 pod at a time):
//! - `1 × 135M` Lead (`baby-orca-lead-135m-q4_k_m.gguf`)
//! - `1 × 135M` Web Scout (`baby-orca-web-scout-135m-q4_k_m.gguf`)
//! - `1 × 135M` Anti-Hallucination Guard (`baby-orca-guard-135m-q4_k_m.gguf`)
//! - `12 × 135M` Specialist Pods (`coding-expertise`, `reasoning-think`, `python`, `rust`,
//!   `sql`, `bash-cli`, `testing`, `security`, `systems`, `web-api`, `data-json`, `docs-summary`)

use std::path::{Path, PathBuf};

pub const BABY_ORCA_PODS: &[&str] = &[
    "coding-expertise",
    "reasoning-think",
    "python",
    "rust",
    "typescript",
    "go",
    "cpp",
    "sql",
    "bash-cli",
    "regex-cli",
    "testing",
    "security",
    "systems",
    "git-devops",
    "web-api",
    "data-json",
    "architecture-diagrams",
    "debugging",
    "math-logic",
    "docs-summary",
];

/// Keeps only the first 2 concise error lines (stripping rustc ASCII art) for 135M healing.
pub fn compact_verify_issue(err: &str) -> String {
    let lines: Vec<&str> = err
        .lines()
        .map(|l| l.trim())
        .filter(|l| {
            !l.is_empty() && !l.starts_with('|') && !l.starts_with("-->") && !l.starts_with('=')
        })
        .take(2)
        .collect();
    if lines.is_empty() {
        "syntax or type error".to_string()
    } else {
        lines.join(" | ")
    }
}

pub fn resolve_baby_orca_dir(models_dir: &Path) -> PathBuf {
    let nested = models_dir.join("baby_orca").join("gguf");
    if nested.join("baby-orca-lead-135m-q4_k_m.gguf").is_file() {
        return nested;
    }
    models_dir.to_path_buf()
}

pub fn pod_gguf_path(dir: &Path, pod: &str) -> PathBuf {
    dir.join(format!("baby-orca-{pod}-135m-q4_k_m.gguf"))
}

pub fn parse_lead_route(raw: &str) -> &'static str {
    let lower = raw.trim().to_ascii_lowercase();
    for pod in BABY_ORCA_PODS {
        if lower.contains(pod) {
            return pod;
        }
    }
    "coding-expertise"
}

pub fn parse_web_scout(raw: &str, fallback_query: &str) -> Option<String> {
    let trimmed = raw.trim();
    let upper = trimmed.to_ascii_uppercase();
    if !upper.starts_with("YES") {
        return None;
    }
    if let Some(idx) = upper.find("QUERY:") {
        let q = trimmed[idx + 6..].trim();
        if !q.is_empty() && !q.eq_ignore_ascii_case("NONE") {
            return Some(q.to_string());
        }
    }
    Some(fallback_query.to_string())
}

pub fn parse_guard_verdict(raw: &str) -> (bool, Option<String>) {
    let trimmed = raw.trim();
    let upper = trimmed.to_ascii_uppercase();
    if upper.starts_with("HALLUCINATION") {
        let reason = trimmed
            .split_once(':')
            .map(|(_, r)| r.trim().to_string())
            .filter(|r| !r.is_empty())
            .unwrap_or_else(|| "Unverified claim or inconsistency detected.".to_string());
        (false, Some(reason))
    } else {
        (true, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_lead_route() {
        assert_eq!(parse_lead_route("rust"), "rust");
        assert_eq!(parse_lead_route("Route: bash-cli\n"), "bash-cli");
        assert_eq!(parse_lead_route("unknown-pod"), "coding-expertise");
    }

    #[test]
    fn test_parse_web_scout() {
        assert_eq!(parse_web_scout("NO", "hello"), None);
        assert_eq!(
            parse_web_scout("YES\nQUERY: latest rust release", "what is latest rust?"),
            Some("latest rust release".to_string())
        );
        assert_eq!(
            parse_web_scout("YES", "latest node version"),
            Some("latest node version".to_string())
        );
    }

    #[test]
    fn test_parse_guard_verdict() {
        assert_eq!(parse_guard_verdict("GROUNDED"), (true, None));
        assert_eq!(
            parse_guard_verdict("HALLUCINATION: invented API method"),
            (false, Some("invented API method".to_string()))
        );
    }
}
