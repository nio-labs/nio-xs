//! # Nio Meerkat (`vMicroLM-Growth`)
//!
//! Self-healing, self-growing hybrid conversational + coding assistant:
//! - `0.5B` Scout (`meerkat-scout-0.5b-q4_k_m.gguf`) for fast chat, memory recall & first pass
//! - `1.5B` Builder (`meerkat-builder-1.5b-q4_k_m.gguf`) for dual-mode direct + bounded `<think>` healing & 4-tool loop
//! - SimSimi-inspired instant `/teach` & healed memory (`~/.nioxs/meerkat_memory.jsonl`)
//! - Verified replay buffer (`~/.nioxs/meerkat_replay.jsonl`) for sleep-cycle LoRA growth

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeerkatMemory {
    pub id: u64,
    pub scope: String,
    pub trigger: String,
    pub content: String,
    pub source: String,
    pub verified: bool,
    pub created_at: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeerkatReplayEntry {
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
    nioxs_home_dir().join("meerkat_memory.jsonl")
}

pub fn replay_path() -> PathBuf {
    nioxs_home_dir().join("meerkat_replay.jsonl")
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

pub fn load_memories(path: &Path) -> Vec<MeerkatMemory> {
    let Ok(raw) = fs::read_to_string(path) else {
        return Vec::new();
    };
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<MeerkatMemory>(l).ok())
        .collect()
}

pub fn append_memory(
    path: &Path,
    scope: &str,
    trigger: &str,
    content: &str,
    source: &str,
    verified: bool,
) -> Result<MeerkatMemory, String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create memory dir: {e}"))?;
    }
    let existing = load_memories(path);
    let next_id = existing.iter().map(|m| m.id).max().unwrap_or(0) + 1;
    let entry = MeerkatMemory {
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

pub fn append_replay(path: &Path, entry: &MeerkatReplayEntry) -> Result<(), String> {
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

const WEB_MEMORY_TTL_SECS: u64 = 24 * 60 * 60;

fn is_memory_fresh(m: &MeerkatMemory, now: u64) -> bool {
    if m.source == "web_healed" {
        now.saturating_sub(m.created_at) <= WEB_MEMORY_TTL_SECS
    } else {
        true
    }
}

pub fn retrieve_memories(path: &Path, query: &str, limit: usize) -> Vec<MeerkatMemory> {
    let q_tokens = tokenize(query);
    if q_tokens.is_empty() {
        return Vec::new();
    }
    let now = unix_now();
    let mut scored: Vec<(usize, MeerkatMemory)> = load_memories(path)
        .into_iter()
        .filter(|m| is_memory_fresh(m, now))
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
    let kept: Vec<MeerkatMemory> = all.into_iter().filter(|m| m.id != id).collect();
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

pub fn extract_natural_teaching(prompt: &str) -> Option<(String, String)> {
    let trimmed = prompt.trim();
    let lower = trimmed.to_ascii_lowercase();
    let prefixes = [
        "remember that ",
        "remember: ",
        "remember ",
        "from now on, ",
        "from now on ",
        "note that ",
        "note: ",
        "keep in mind that ",
        "keep in mind ",
        "learn that ",
        "learn: ",
        "correction: ",
        "actually, ",
        "no, ",
    ];
    for p in prefixes {
        if lower.starts_with(p) {
            let fact = trimmed[p.len()..].trim();
            if fact.len() >= 6 {
                let trigger = fact
                    .split(" is ")
                    .next()
                    .or_else(|| fact.split(" are ").next())
                    .unwrap_or(fact)
                    .trim();
                return Some((trigger.to_string(), fact.to_string()));
            }
        }
    }
    None
}

pub fn looks_like_code_task(_prompt: &str, response: &str) -> bool {
    response.contains("```") || response.contains("def ") || response.contains("fn ")
}

pub fn looks_like_knowledge_gap(resp: &str) -> bool {
    let lower = resp.to_ascii_lowercase();
    [
        "[search]",
        "[need_web]",
        "as an ai",
        "as of my last update",
        "knowledge cutoff",
        "don't have access",
        "cannot access",
        "do not have access",
        "i don't have a specific",
        "real-time",
        "i'm not sure",
        "i am not sure",
        "cannot verify",
        "check the official",
    ]
    .iter()
    .any(|kw| lower.contains(kw))
}

fn strip_html_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub async fn fetch_web_grounding(query: &str) -> Option<String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(8))
        .user_agent("Mozilla/5.0 (X11; Linux x86_64) NioXS-Meerkat/0.6")
        .build()
        .ok()?;
    let mut snippets = Vec::new();
    if let Ok(resp) = client
        .get("https://api.duckduckgo.com/")
        .query(&[
            ("q", query),
            ("format", "json"),
            ("no_html", "1"),
            ("skip_disambig", "1"),
        ])
        .send()
        .await
    {
        if let Ok(val) = resp.json::<serde_json::Value>().await {
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
        }
    }

    if snippets.is_empty() {
        if let Ok(resp) = client
            .get("https://html.duckduckgo.com/html/")
            .query(&[("q", query)])
            .send()
            .await
        {
            if let Ok(html) = resp.text().await {
                for part in html.split("class=\"result__snippet\"").skip(1).take(4) {
                    if let Some((_, after_gt)) = part.split_once('>') {
                        if let Some((raw_snip, _)) = after_gt.split_once("</a>") {
                            let clean = strip_html_tags(raw_snip);
                            if !clean.is_empty() {
                                snippets.push(clean);
                            }
                        }
                    }
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

fn truncate_chars(s: &str, max_chars: usize) -> String {
    let mut out: String = s.chars().take(max_chars).collect();
    if s.chars().count() > max_chars {
        out.push_str("\n...[truncated]");
    }
    out
}

/// Parses the first flat `<tool name="...">arg</tool>` tag emitted by the 1.5B Builder.
pub fn extract_tool_call(text: &str) -> Option<(String, String)> {
    let start = text.find("<tool ")?;
    let after_tag = &text[start + 6..];
    let gt = after_tag.find('>')?;
    let attr = &after_tag[..gt];
    let name = if let Some(rest) = attr.split("name=\"").nth(1) {
        rest.split('"').next()?.trim()
    } else if let Some(rest) = attr.split("name='").nth(1) {
        rest.split('\'').next()?.trim()
    } else {
        return None;
    };
    if name.is_empty() {
        return None;
    }
    let body_start = start + 6 + gt + 1;
    let body_rest = &text[body_start..];
    let end = body_rest.find("</tool>")?;
    let arg = body_rest[..end].trim().to_string();
    Some((name.to_string(), arg))
}

async fn fetch_web_page(url: &str) -> String {
    let trimmed = url.trim();
    if !trimmed.starts_with("http://") && !trimmed.starts_with("https://") {
        return "Error: web_fetch requires an http:// or https:// URL.".to_string();
    }
    let Ok(client) = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(8))
        .user_agent("Mozilla/5.0 (X11; Linux x86_64) NioXS-Meerkat/0.6")
        .build()
    else {
        return "Error: failed to build HTTP client.".to_string();
    };
    match client.get(trimmed).send().await {
        Ok(resp) => match resp.text().await {
            Ok(html) => {
                let clean = strip_html_tags(&html);
                if clean.is_empty() {
                    "Page returned empty text.".to_string()
                } else {
                    truncate_chars(&clean, 2500)
                }
            }
            Err(e) => format!("Error reading response body: {e}"),
        },
        Err(e) => format!("Error fetching URL: {e}"),
    }
}

fn read_local_file(path_str: &str) -> String {
    let trimmed = path_str.trim();
    if trimmed.is_empty() {
        return "Error: empty file path.".to_string();
    }
    let path = Path::new(trimmed);
    match fs::read_to_string(path) {
        Ok(content) => truncate_chars(&content, 3000),
        Err(e) => format!("Error reading '{trimmed}': {e}"),
    }
}

async fn run_python_snippet(code: &str) -> String {
    let trimmed = code.trim();
    if trimmed.is_empty() {
        return "Error: empty python snippet.".to_string();
    }
    let script_path = std::env::temp_dir().join(format!(
        "nioxs_meerkat_tool_{}_{}.py",
        std::process::id(),
        unix_now()
    ));
    if let Err(e) = fs::write(&script_path, trimmed) {
        return format!("Error writing temp python script: {e}");
    }
    let mut cmd = tokio::process::Command::new("python3");
    cmd.arg(&script_path);
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), cmd.output()).await;
    let _ = fs::remove_file(&script_path);
    match result {
        Ok(Ok(out)) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let combined = match (stdout.trim().is_empty(), stderr.trim().is_empty()) {
                (false, true) => stdout.trim().to_string(),
                (true, false) => format!("stderr:\n{}", stderr.trim()),
                (false, false) => format!("{}\nstderr:\n{}", stdout.trim(), stderr.trim()),
                (true, true) => "(python3 exited 0 with no output)".to_string(),
            };
            truncate_chars(&combined, 1500)
        }
        Ok(Err(e)) => format!("Error running python3: {e}"),
        Err(_) => "Error: python3 execution timed out after 5s.".to_string(),
    }
}

/// Executes one of the 4 bounded Meerkat tools (`web_search`, `web_fetch`, `read_file`, `run_python`).
pub async fn execute_tool(name: &str, arg: &str) -> String {
    match name.trim() {
        "web_search" => fetch_web_grounding(arg.trim())
            .await
            .unwrap_or_else(|| "No web search results found.".to_string()),
        "web_fetch" => fetch_web_page(arg).await,
        "read_file" => read_local_file(arg),
        "run_python" => run_python_snippet(arg).await,
        other => format!(
            "Error: unsupported tool '{other}'. Allowed tools: web_search, web_fetch, read_file, run_python."
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meerkat_teach_retrieve_and_forget() {
        let dir = std::env::temp_dir().join(format!("nioxs_meerkat_test_{}", unix_now()));
        let mem_file = dir.join("meerkat_memory.jsonl");
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

    #[test]
    fn test_extract_natural_teaching() {
        let hit = extract_natural_teaching("Remember that my favorite language is Rust");
        assert_eq!(
            hit,
            Some((
                "my favorite language".to_string(),
                "my favorite language is Rust".to_string()
            ))
        );
        assert!(extract_natural_teaching("What is my favorite language?").is_none());
    }

    #[test]
    fn test_extract_tool_call() {
        let sample = "Let me check that.\n<tool name=\"read_file\">Cargo.toml</tool>";
        assert_eq!(
            extract_tool_call(sample),
            Some(("read_file".to_string(), "Cargo.toml".to_string()))
        );
        let py = "<tool name='run_python'>print(2 + 2)</tool>";
        assert_eq!(
            extract_tool_call(py),
            Some(("run_python".to_string(), "print(2 + 2)".to_string()))
        );
        assert!(extract_tool_call("No tool tag here.").is_none());
    }

    #[test]
    fn test_web_healed_memory_ttl() {
        let now = unix_now();
        let fresh_web = MeerkatMemory {
            id: 1,
            scope: "global".into(),
            trigger: "latest rust".into(),
            content: "1.100.0".into(),
            source: "web_healed".into(),
            verified: true,
            created_at: now.saturating_sub(60),
        };
        let stale_web = MeerkatMemory {
            created_at: now.saturating_sub(WEB_MEMORY_TTL_SECS + 10),
            ..fresh_web.clone()
        };
        let perm_teach = MeerkatMemory {
            source: "teach".into(),
            created_at: now.saturating_sub(WEB_MEMORY_TTL_SECS * 90),
            ..fresh_web.clone()
        };
        assert!(is_memory_fresh(&fresh_web, now));
        assert!(!is_memory_fresh(&stale_web, now));
        assert!(is_memory_fresh(&perm_teach, now));
    }
}
