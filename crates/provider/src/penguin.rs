//! # Nio Penguin (`vMicroLM-Growth`)
//!
//! Self-healing, self-growing hybrid conversational + coding assistant:
//! - `0.5B` Scout (`penguin-scout-0.5b-q4_k_m.gguf`) for fast chat, memory recall & first pass
//! - `1.5B` Builder (`penguin-builder-1.5b-q4_k_m.gguf`) for dual-mode direct + bounded `<think>` healing
//! - SimSimi-inspired instant `/teach` & healed memory (`~/.nioxs/penguin_memory.jsonl`)
//! - Verified replay buffer (`~/.nioxs/penguin_replay.jsonl`) for sleep-cycle LoRA growth

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PenguinMemory {
    pub id: u64,
    pub scope: String,
    pub trigger: String,
    pub content: String,
    pub source: String,
    pub verified: bool,
    pub created_at: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PenguinReplayEntry {
    pub prompt: String,
    pub failed_draft: Option<String>,
    pub verifier_error: Option<String>,
    pub verified_solution: String,
    pub used_web: bool,
    pub created_at: u64,
}

pub fn nioxs_home_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".nioxs")
}

pub fn memory_path() -> PathBuf {
    nioxs_home_dir().join("penguin_memory.jsonl")
}

pub fn replay_path() -> PathBuf {
    nioxs_home_dir().join("penguin_replay.jsonl")
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn tokenize(text: &str) -> HashSet<String> {
    text.to_ascii_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| s.len() >= 2)
        .map(|s| s.to_string())
        .collect()
}

pub fn load_memories(path: &Path) -> Vec<PenguinMemory> {
    let Ok(raw) = fs::read_to_string(path) else {
        return Vec::new();
    };
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<PenguinMemory>(l).ok())
        .collect()
}

pub fn append_memory(
    path: &Path,
    scope: &str,
    trigger: &str,
    content: &str,
    source: &str,
    verified: bool,
) -> Result<PenguinMemory, String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create memory dir: {e}"))?;
    }
    let existing = load_memories(path);
    let next_id = existing.iter().map(|m| m.id).max().unwrap_or(0) + 1;
    let entry = PenguinMemory {
        id: next_id,
        scope: scope.to_string(),
        trigger: trigger.trim().to_string(),
        content: content.trim().to_string(),
        source: source.to_string(),
        verified,
        created_at: unix_now(),
    };
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("Failed to open memory file: {e}"))?;
    let line = serde_json::to_string(&entry).map_err(|e| format!("Serialize error: {e}"))?;
    writeln!(f, "{line}").map_err(|e| format!("Write error: {e}"))?;
    Ok(entry)
}

pub fn append_replay(path: &Path, entry: &PenguinReplayEntry) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create replay dir: {e}"))?;
    }
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("Failed to open replay file: {e}"))?;
    let line = serde_json::to_string(entry).map_err(|e| format!("Serialize error: {e}"))?;
    writeln!(f, "{line}").map_err(|e| format!("Write error: {e}"))?;
    Ok(())
}

pub fn retrieve_memories(path: &Path, query: &str, limit: usize) -> Vec<PenguinMemory> {
    let q_tokens = tokenize(query);
    if q_tokens.is_empty() {
        return Vec::new();
    }
    let mut scored: Vec<(usize, PenguinMemory)> = load_memories(path)
        .into_iter()
        .filter_map(|m| {
            let mut m_tokens = tokenize(&m.trigger);
            m_tokens.extend(tokenize(&m.content));
            let overlap = q_tokens.intersection(&m_tokens).count();
            if overlap > 0 {
                let bonus = if m.verified { 1 } else { 0 };
                Some((overlap * 2 + bonus, m))
            } else {
                None
            }
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.id.cmp(&a.1.id)));
    scored.into_iter().take(limit).map(|(_, m)| m).collect()
}

pub fn forget_memories(path: &Path, target: &str) -> Result<usize, String> {
    if target.eq_ignore_ascii_case("all") {
        let count = load_memories(path).len();
        if path.exists() {
            fs::remove_file(path).map_err(|e| format!("Failed to remove memory file: {e}"))?;
        }
        return Ok(count);
    }
    let id: u64 = target
        .parse()
        .map_err(|_| format!("Invalid memory id '{target}' (use a number or 'all')"))?;
    let all = load_memories(path);
    let before = all.len();
    let kept: Vec<PenguinMemory> = all.into_iter().filter(|m| m.id != id).collect();
    let removed = before.saturating_sub(kept.len());
    let mut out = String::new();
    for m in kept {
        if let Ok(line) = serde_json::to_string(&m) {
            out.push_str(&line);
            out.push('\n');
        }
    }
    fs::write(path, out).map_err(|e| format!("Failed to update memory file: {e}"))?;
    Ok(removed)
}

pub fn looks_like_code_task(prompt: &str, response: &str) -> bool {
    if response.contains("```") || response.contains("def ") || response.contains("fn ") {
        return true;
    }
    let lower = prompt.to_ascii_lowercase();
    [
        "write a function",
        "python",
        "rust",
        "javascript",
        "typescript",
        "implement",
        "assert ",
        "compile",
        "refactor",
        "debug",
        "algorithm",
        "code",
    ]
    .iter()
    .any(|kw| lower.contains(kw))
}

pub async fn fetch_web_grounding(query: &str) -> Option<String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(6))
        .user_agent("NioXS-Penguin/0.6")
        .build()
        .ok()?;
    let resp = client
        .get("https://api.duckduckgo.com/")
        .query(&[
            ("q", query),
            ("format", "json"),
            ("no_html", "1"),
            ("skip_disambig", "1"),
        ])
        .send()
        .await
        .ok()?;
    let val: serde_json::Value = resp.json().await.ok()?;
    let mut snippets = Vec::new();
    if let Some(abs) = val.get("AbstractText").and_then(|v| v.as_str()) {
        if !abs.trim().is_empty() {
            snippets.push(abs.trim().to_string());
        }
    }
    if let Some(topics) = val.get("RelatedTopics").and_then(|v| v.as_array()) {
        for t in topics.iter().take(3) {
            if let Some(txt) = t.get("Text").and_then(|v| v.as_str()) {
                if !txt.trim().is_empty() {
                    snippets.push(txt.trim().to_string());
                }
            }
        }
    }
    if snippets.is_empty() {
        None
    } else {
        Some(snippets.join("\n- "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_penguin_teach_retrieve_and_forget() {
        let dir = std::env::temp_dir().join(format!("nioxs_penguin_test_{}", unix_now()));
        let mem_file = dir.join("penguin_memory.jsonl");
        let _ = fs::remove_dir_all(&dir);

        let m1 = append_memory(
            &mem_file,
            "global",
            "axum 0.8 route syntax",
            "In axum 0.8, use /{id} path params instead of /:id.",
            "teach",
            true,
        )
        .unwrap();
        assert_eq!(m1.id, 1);

        let hits = retrieve_memories(&mem_file, "How do I write an axum 0.8 route?", 3);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].content.contains("/{id}"));

        let removed = forget_memories(&mem_file, "1").unwrap();
        assert_eq!(removed, 1);
        assert!(load_memories(&mem_file).is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_looks_like_code_task() {
        assert!(!looks_like_code_task(
            "Hello, how are you today?",
            "I am doing well!"
        ));
        assert!(looks_like_code_task(
            "Write a Python function fib(n)",
            "def fib(n): ..."
        ));
    }
}
