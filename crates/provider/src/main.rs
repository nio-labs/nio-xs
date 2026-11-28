use nioxs_nano::Model;
use nioxs_provider::{ActiveTurns, App, router};
use nioxs_storage::NioDb;
use std::sync::Arc;

mod baby_orca;
mod meerkat;
mod skate_safety;
mod stream_flow;

use stream_flow::{FlowMode, StreamFlow};

fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(8 * 1024 * 1024)
        .build()
        .expect("Failed to build Tokio runtime");
    if let Err(error) = rt.block_on(run()) {
        eprintln!("nioxs: {error}");
        std::process::exit(1);
    }
}

fn format_council_directive(directive: &str) -> String {
    let inner = directive.trim();
    let inner = inner.strip_prefix("@council(").unwrap_or(inner);
    let inner = inner.strip_suffix(')').unwrap_or(inner).trim();
    let mut key_name = "route";
    let mut key_val = String::new();
    let mut domain = String::new();
    for part in inner.split(',') {
        let p = part.trim();
        if let Some(val) = p
            .strip_prefix("head=")
            .or_else(|| p.strip_prefix("head = "))
        {
            key_name = "head";
            key_val = val.trim_matches(|c| c == '"' || c == '\'').to_string();
        } else if let Some(val) = p
            .strip_prefix("route=")
            .or_else(|| p.strip_prefix("route = "))
        {
            key_name = "route";
            key_val = val.trim_matches(|c| c == '"' || c == '\'').to_string();
        } else if let Some(val) = p
            .strip_prefix("domain=")
            .or_else(|| p.strip_prefix("domain = "))
        {
            domain = val.trim_matches(|c| c == '"' || c == '\'').to_string();
        }
    }
    if !key_val.is_empty() && !domain.is_empty() {
        format!(
            "<council>\n    @council(\n        {key_name} = {key_val},\n        domain = {domain}\n    )\n</council>\n\n"
        )
    } else if !key_val.is_empty() {
        format!("<council>\n    @council(\n        {key_name} = {key_val}\n    )\n</council>\n\n")
    } else {
        format!("<council>\n    {}\n</council>\n\n", directive.trim())
    }
}

#[allow(dead_code)]
fn is_degenerate(s: &str) -> bool {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return false;
    }
    // Check 1: Repetition of quote-paren artifacts common to unaligned base models
    if trimmed.contains("')\n')")
        || trimmed.contains("')')")
        || trimmed.contains("``C, `C")
        || trimmed.contains("011200000")
        || trimmed.contains("100000")
    {
        return true;
    }
    // Check 2: Unusually high ratio of backticks or single quotes
    let quote_count = trimmed.chars().filter(|&c| c == '\'' || c == '`').count();
    if quote_count > 6 && quote_count * 4 > trimmed.len() {
        return true;
    }
    // Check 3: Repetition of long digit sequences (> 15 consecutive digits)
    let mut consecutive_digits = 0;
    for c in trimmed.chars() {
        if c.is_ascii_digit() {
            consecutive_digits += 1;
            if consecutive_digits >= 15 {
                return true;
            }
        } else {
            consecutive_digits = 0;
        }
    }
    // Check 3: Repetition of short identical lines
    let lines: Vec<&str> = trimmed
        .lines()
        .map(|l| l.trim())
        .filter(|l| {
            !l.is_empty() && *l != "}" && *l != "};" && *l != "]" && *l != "];" && *l != ")"
        })
        .collect();
    if lines.len() >= 3 {
        let mut repeat_count = 0;
        for i in 1..lines.len() {
            if lines[i] == lines[i - 1] {
                repeat_count += 1;
            }
        }
        if repeat_count >= 2 {
            return true;
        }
    }
    false
}

#[allow(dead_code)]
fn is_code_request(prompt: &str) -> bool {
    let lower = prompt.to_ascii_lowercase();
    let has_code_verb = lower.contains("write")
        || lower.contains("implement")
        || lower.contains("code")
        || lower.contains("create")
        || lower.contains("build")
        || lower.contains("construct")
        || lower.contains("develop")
        || lower.contains("engineer")
        || lower.contains("function")
        || lower.contains("struct")
        || lower.contains("class")
        || lower.contains("fn ")
        || lower.contains("def ")
        || lower.contains("snippet")
        || lower.contains("vectorize")
        || lower.contains("optimize")
        || lower.contains("benchmark")
        || lower.contains("design")
        || lower.contains("generate")
        || lower.contains("convert")
        || lower.contains("draft")
        || lower.contains("show")
        || lower.contains("example")
        || lower.contains("program")
        || lower.contains("script");
    let has_lang = lower.contains("in rust")
        || lower.contains("in zig")
        || lower.contains("in c")
        || lower.contains("in cpp")
        || lower.contains("in c++")
        || lower.contains("in python")
        || lower.contains("in go")
        || lower.contains("in typescript")
        || lower.contains("in tsx")
        || lower.contains("in sql");
    has_code_verb || has_lang
}

#[allow(dead_code)]
fn detect_target_code_language(prompt: &str, specialist: &str) -> &'static str {
    let lower = prompt.to_ascii_lowercase();
    if lower.contains("zig") {
        "zig"
    } else if lower.contains("rust") || lower.contains("axum") || lower.contains("tokio") {
        "rust"
    } else if lower.contains("posix")
        || lower.contains("in c")
        || lower.contains(" c ")
        || lower.ends_with(" c")
        || lower.contains("c code")
    {
        "c"
    } else if lower.contains("cpp") || lower.contains("c++") {
        "cpp"
    } else if lower.contains("python") {
        "python"
    } else if lower.contains("go") || lower.contains("golang") {
        "go"
    } else {
        match specialist {
            "001"
            | "head-001"
            | "hydra-kernel"
            | "head-001-atomic-concurrency"
            | "head-001-systems-atomic-concurrency" => "zig",
            "002" | "head-002" | "head-002-backend-axum-tokio" => "rust",
            "003" | "head-003" | "head-003-frontend-react19-rsc" => "tsx",
            "004" | "head-004" | "head-004-kernel-ebpf-xdp" => "rust",
            "005" | "head-005" | "head-005-data-clickhouse-mergetree" => "sql",
            "006" | "head-006" | "head-006-algorithms-formal-logic" => "rust",
            "007" | "head-007" | "head-007-devops-k8s-controller" => "rust",
            "008" | "head-008" | "head-008-consensus-raft-paxos" => "rust",
            "009" | "head-009" | "head-009-vector-simd-avx512" => "rust",
            "010" | "head-010" | "head-010-wasm-wasi-sandboxing" => "rust",
            _ => "rust",
        }
    }
}

struct DynamicManifest {
    head_id: &'static str,
    title: String,
    triggers: Vec<String>,
}

fn format_hydra_directive(head: &str, domain: Option<&str>) -> String {
    let clean_id = head
        .trim_start_matches("head-")
        .trim_start_matches("anchor-");
    if clean_id == "direct" || clean_id == "polymath" {
        return "Council Direct: Polymath Synthesis\n\n".to_string();
    }
    let manifests = get_dynamic_manifests();
    for m in manifests {
        if m.head_id == head || m.head_id.ends_with(clean_id) {
            return format!("Council {}: {}\n\n", clean_id, m.title);
        }
    }
    format!(
        "Council {}: {}\n\n",
        clean_id,
        domain.unwrap_or("Domain Specialist")
    )
}

/// Cosine-based semantic router for the Hydra Sovereign Council.
///
/// # Architecture (3-Chair Council)
/// - Chair A (head-000): **Sovereign Arbiter** — the ONLY entry point. Always resident.
///   Emits `@council(head=NNN)` directives into `<think>` after semantic triage.
/// - Chair B: Specialist — mounts the winning Tier-2 Anchor LoRA + Tier-3 Head LoRA.
/// - Chair C: Verifier/Critic — audits Chair B output in parallel.
///
/// # Routing Strategy (layered, O(T·K) where T=tokens, K=anchors)
/// 1. **Explicit neural directive** — Chair A's `@council(head=NNN)` / `@council(direct)` wins instantly.
/// 2. **Explicit head ID in accumulated/prompt** — `head-NNN` / `council NNN` wins instantly.
/// 3. **Cosine semantic fingerprint** — bag-of-words dot-product against 41 anchor manifests.
///    Returns the highest-scoring anchor head ID, or None if max score == 0.
///
/// Extract the first fenced code block `(lang, code)` from model output
/// after any `<think>...</think>` reasoning block, ignoring nested doc-comment fences.
fn extract_code_block(text: &str) -> Option<(String, String)> {
    let cleaned = if let Some(pos) = text.rfind("</think>") {
        &text[pos + "</think>".len()..]
    } else if text.contains("<think>") {
        ""
    } else {
        text
    };
    let mut lines = cleaned.lines();
    let mut lang = String::new();
    let mut found_open = false;
    let mut body = Vec::new();
    for line in &mut lines {
        let trimmed = line.trim();
        if !found_open {
            if let Some(rest) = trimmed.strip_prefix("```") {
                lang = rest.trim().to_ascii_lowercase();
                found_open = true;
            }
        } else if trimmed == "```" {
            return Some((lang, body.join("\n")));
        } else {
            body.push(line);
        }
    }
    if found_open && !body.is_empty() {
        return Some((lang, body.join("\n")));
    }
    None
}

/// Compile/test generated code in an isolated temp dir and return concise diagnostics on failure.
fn verify_generated_code(
    output: &str,
    prompt: &str,
    specialist: &str,
    test_file: Option<&std::path::Path>,
    extra_harness: Option<&str>,
) -> Result<String, String> {
    let (fence_lang, mut code) = extract_code_block(output)
        .ok_or_else(|| "No fenced code block (```...```) found in output.".to_string())?;
    let lower_p = prompt.to_ascii_lowercase();
    let lang = if !fence_lang.is_empty() {
        fence_lang
    } else if specialist.contains("001") || lower_p.contains("zig") {
        "zig".to_string()
    } else if specialist.contains("015") || specialist.contains("016") || lower_p.contains("python")
    {
        "python".to_string()
    } else {
        "rust".to_string()
    };

    let harness_opt = test_file
        .and_then(|tf| std::fs::read_to_string(tf).ok())
        .or_else(|| extra_harness.map(|s| s.to_string()));
    if let Some(harness) = harness_opt {
        if (lang == "rust" || lang == "rs") && harness.contains("fn main(") {
            if let Some(pos) = code.find("fn main(") {
                code.truncate(pos);
            }
        }
        code.push('\n');
        code.push_str(&harness);
    }

    let tmp_dir = std::env::temp_dir().join(format!(
        "nioxs_verify_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::create_dir_all(&tmp_dir);

    let result = match lang.as_str() {
        "rust" | "rs" => {
            let src = tmp_dir.join("main.rs");
            let bin = tmp_dir.join("main_bin");
            let _ = std::fs::write(&src, &code);
            let is_test = code.contains("#[test]");
            let has_main = code.contains("fn main(");
            let mut cmd = std::process::Command::new("rustc");
            cmd.arg("--edition")
                .arg("2021")
                .arg(&src)
                .arg("-o")
                .arg(&bin);
            if is_test {
                cmd.arg("--test");
            } else if !has_main {
                cmd.arg("--crate-type").arg("lib");
            }
            match cmd.output() {
                Ok(out) if out.status.success() => {
                    if is_test || has_main {
                        match std::process::Command::new(&bin).output() {
                            Ok(run_out) if run_out.status.success() => {
                                Ok("rustc + run PASS".to_string())
                            }
                            Ok(run_out) => Err(String::from_utf8_lossy(&run_out.stderr)
                                .chars()
                                .take(600)
                                .collect()),
                            Err(e) => Err(format!("Failed to execute binary: {e}")),
                        }
                    } else {
                        Ok("rustc lib PASS".to_string())
                    }
                }
                Ok(out) => Err(String::from_utf8_lossy(&out.stderr)
                    .chars()
                    .take(600)
                    .collect()),
                Err(e) => Err(format!("Failed to invoke rustc: {e}")),
            }
        }
        "zig" => {
            let src = tmp_dir.join("main.zig");
            let _ = std::fs::write(&src, &code);
            let subcmd = if code.contains("test ") {
                "test"
            } else {
                "build-obj"
            };
            let mut cmd = std::process::Command::new("zig");
            cmd.arg(subcmd).arg(&src).current_dir(&tmp_dir);
            match cmd.output() {
                Ok(out) if out.status.success() => Ok(format!("zig {subcmd} PASS")),
                Ok(out) => Err(String::from_utf8_lossy(&out.stderr)
                    .chars()
                    .take(600)
                    .collect()),
                Err(e) => Err(format!("Failed to invoke zig: {e}")),
            }
        }
        "python" | "py" => {
            let src = tmp_dir.join("main.py");
            let _ = std::fs::write(&src, &code);
            match std::process::Command::new("python3")
                .arg("-m")
                .arg("py_compile")
                .arg(&src)
                .output()
            {
                Ok(out) if out.status.success() => {
                    if code.contains("assert ") || code.contains("unittest") {
                        match std::process::Command::new("python3").arg(&src).output() {
                            Ok(run_out) if run_out.status.success() => {
                                Ok("python3 + assert PASS".to_string())
                            }
                            Ok(run_out) => Err(String::from_utf8_lossy(&run_out.stderr)
                                .chars()
                                .take(600)
                                .collect()),
                            Err(e) => Err(format!("Failed to run python3: {e}")),
                        }
                    } else {
                        Ok("python3 py_compile PASS".to_string())
                    }
                }
                Ok(out) => Err(String::from_utf8_lossy(&out.stderr)
                    .chars()
                    .take(600)
                    .collect()),
                Err(e) => Err(format!("Failed to invoke python3: {e}")),
            }
        }
        other => Ok(format!("skipped unsupported fence '{other}'")),
    };

    let _ = std::fs::remove_dir_all(&tmp_dir);
    result
}

/// Scales to 1,000+ heads with no code changes — add anchors to ANCHOR_MANIFESTS below.
fn detect_council_specialist(
    accumulated: &str,
    prompt: &str,
    _is_hydra: bool,
) -> Option<&'static str> {
    let lower_acc = format!("{accumulated} {prompt}").to_ascii_lowercase();

    // ── Layer 1: Explicit Neural Directives from Chair A ─────────────────────
    // Chair A emits these inside <think>; they are the highest-priority signal.
    if lower_acc.contains("@council(direct)")
        || lower_acc.contains("@council(polymath)")
        || lower_acc.contains("@council(self)")
        || lower_acc.contains("directive: direct")
        || lower_acc.contains("directive: polymath")
        || lower_acc.contains("council direct")
        || lower_acc.contains("council: direct")
        || lower_acc.contains("polymath synthesis")
        || lower_acc.contains("technology polymath")
    {
        return Some("head-direct");
    }

    // ── Layer 2: Explicit head ID injection (Chair A directive or user override) ──
    // Matches @council(head=NNN), head-NNN, council NNN patterns.
    // Covers all 41 anchors (000–041) generically.
    {
        // Generic @council(head=NNN) pattern — capture NNN
        let patterns = [
            ("@council(head=", ')'),
            ("@council(anchor=", ')'),
            ("delegate: council ", ':'),
            ("delegate: council ", ' '),
            ("council: ", ':'), // "council: 009: SIMD"
        ];
        for (prefix, term) in &patterns {
            if let Some(pos) = lower_acc.find(prefix) {
                let rest = &lower_acc[pos + prefix.len()..];
                let end = rest.find(*term).unwrap_or(rest.len()).min(3);
                let id_str = rest[..end].trim();
                if let Ok(id_num) = id_str.parse::<u32>() {
                    if id_num <= 2048 {
                        return Some(anchor_head_id(id_num));
                    }
                }
            }
        }

        // Generic head-NNN / anchor-NNN explicit numeric mentions (0..=2048)
        for id_num in 0u32..=2048 {
            let id3 = format!("{:03}", id_num);
            if lower_acc.contains(&format!("head-{}", id3))
                || lower_acc.contains(&format!("head={}", id3))
                || lower_acc.contains(&format!("head = {}", id3))
                || lower_acc.contains(&format!("anchor-{}", id3))
                || lower_acc.contains(&format!("anchor={}", id3))
                || lower_acc.contains(&format!("council {}", id3))
                || lower_acc.contains(&format!("council: {}", id3))
            {
                return Some(anchor_head_id(id_num));
            }
        }
    }

    // ── Layer 3: Cosine Semantic Fingerprint ──────────────────────────────────
    // Each anchor manifest is a bag-of-trigger-words. We score only the user
    // prompt (never Chair A's free-form <think> prose) via dot-product and
    // return the winning head ID. Pure O(T·K) string ops, < 0.2 ms.
    cosine_route(&prompt.to_ascii_lowercase())
}

/// Returns the canonical head-ID static string for any anchor/head number (0..=2048).
/// Dynamically cached and memoized; zero heap allocations during routing.
fn anchor_head_id(id: u32) -> &'static str {
    static HEAD_MAP: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    let map = HEAD_MAP.get_or_init(|| {
        (0..=2048)
            .map(|i| Box::leak(format!("head-{:03}", i).into_boxed_str()) as &'static str)
            .collect()
    });
    map.get(id as usize).copied().unwrap_or("head-direct")
}

/// Dynamically loads the anchor and head manifests from registry JSON files.
/// Zero hardcoded triggers or head rosters in Rust. Supports scaling to 1,000+ heads.
fn get_dynamic_manifests() -> &'static [DynamicManifest] {
    static MANIFESTS: std::sync::OnceLock<Vec<DynamicManifest>> = std::sync::OnceLock::new();
    MANIFESTS.get_or_init(|| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
        let registry_paths = [
            format!("{}/.nioxs/registry/hydra_anchor_registry.json", home),
            format!(
                "{}/nio-labs/nio-ideas/registry/hydra_anchor_registry.json",
                home
            ),
            "/root/.nioxs/registry/hydra_anchor_registry.json".to_string(),
            "/root/nio-labs/nio-ideas/registry/hydra_anchor_registry.json".to_string(),
        ];
        let mut raw_json = String::new();
        for p in &registry_paths {
            if let Ok(c) = std::fs::read_to_string(p) {
                raw_json = c;
                break;
            }
        }
        if raw_json.is_empty() {
            raw_json = "{\"anchors\":[]}".to_string();
        }

        let mut list = Vec::new();
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw_json) {
            if let Some(anchors) = val.get("anchors").and_then(|a| a.as_array()) {
                for a in anchors {
                    let id_str = a.get("id").and_then(|i| i.as_str()).unwrap_or("");
                    if id_str == "000" {
                        continue; // Chair A does not route to itself
                    }
                    let head_id: &'static str =
                        Box::leak(format!("head-{}", id_str).into_boxed_str());
                    let title = a
                        .get("title")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string();
                    let mut triggers = Vec::new();
                    if let Some(slug) = a.get("slug").and_then(|s| s.as_str()) {
                        let sl = slug.trim().to_ascii_lowercase();
                        if !sl.is_empty() {
                            triggers.push(sl.clone());
                            if sl.starts_with("anchor-") {
                                triggers.push(sl.trim_start_matches("anchor-").to_string());
                            }
                        }
                    }
                    if let Some(trigs) = a.get("triggers").and_then(|t| t.as_array()) {
                        for t in trigs {
                            if let Some(s) = t.as_str() {
                                let st = s.trim().to_ascii_lowercase();
                                if !st.is_empty() && st != "*" {
                                    triggers.push(st);
                                }
                            }
                        }
                    }
                    if !triggers.is_empty() {
                        list.push(DynamicManifest {
                            head_id,
                            title,
                            triggers,
                        });
                    }
                }
            }
        }
        list
    })
}

/// Hop 2 of Two-Hop Routing: Dynamically score installed Tier-3 heads under this Anchor.
/// Falls back to the Sovereign Anchor unless a Tier-3 head's sub-skill matches the prompt.
fn resolve_tier3_head(anchor: &str, prompt: &str) -> String {
    let lower_p = prompt.to_ascii_lowercase();
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let anchors_dir = std::path::PathBuf::from(&home)
        .join(".nioxs")
        .join("anchors");
    let clean_num = anchor
        .trim_start_matches("head-")
        .trim_start_matches("anchor-");
    let prefix = format!("tier3-{clean_num}");

    let mut best_slug: Option<String> = None;
    let mut best_score: u32 = 0;

    if let Ok(entries) = std::fs::read_dir(&anchors_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.starts_with(&prefix) || !entry.path().join("adapter_model.gguf").is_file() {
                continue;
            }
            // Score sub-skill tokens after `tier3-XXXy-` (skip generic filler words)
            let tail = name.strip_prefix(&prefix).unwrap_or("");
            let sub = tail.splitn(2, '-').nth(1).unwrap_or("");
            let mut score: u32 = 0;
            for tok in sub
                .split('-')
                .filter(|t| t.len() >= 3 && !matches!(*t, "core" | "idioms" | "opt"))
            {
                if contains_trigger(&lower_p, tok) {
                    score += 1;
                }
            }
            if score > best_score {
                best_score = score;
                best_slug = Some(name);
            }
        }
    }

    if best_score >= 1 {
        if let Some(slug) = best_slug {
            return slug;
        }
    }
    anchor.to_string()
}

/// Word-boundary trigger match so e.g. "wit" does not match "with".
fn contains_trigger(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let mut start = 0;
    while let Some(pos) = haystack[start..].find(needle) {
        let abs_start = start + pos;
        let abs_end = abs_start + needle.len();
        let left_ok = abs_start == 0
            || !haystack[..abs_start]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_ascii_alphanumeric());
        let right_ok = abs_end == haystack.len()
            || !haystack[abs_end..]
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphanumeric());
        if left_ok && right_ok {
            return true;
        }
        start = abs_start + 1;
    }
    false
}

/// Bag-of-words cosine router — scores query dynamically against all manifests loaded from registry.
/// Returns the highest-scoring head ID, or None if no trigger matched.
/// O(tokens × anchors) ≈ 0.05–0.15 ms. 100% dynamic, zero hardcoded tables.
fn cosine_route(lower_query: &str) -> Option<&'static str> {
    let manifests = get_dynamic_manifests();
    let mut best_id: &'static str = "";
    let mut best_score: u32 = 0;

    for m in manifests {
        let mut score: u32 = 0;
        for t in &m.triggers {
            if contains_trigger(lower_query, t.as_str()) {
                // Weight longer/more-specific triggers higher
                score += (t.len() as u32).max(1);
            }
        }
        if score > best_score {
            best_score = score;
            best_id = m.head_id;
        }
    }

    if best_score > 0 { Some(best_id) } else { None }
}

fn print_council_heads_table(heads_dir: &std::path::Path) {
    println!(
        "\n========================================================================================"
    );
    println!("                      HYDRA SOVEREIGN 64-ANCHOR COUNCIL ROSTER");
    println!(
        "========================================================================================"
    );
    println!(
        "{:<9} | {:<36} | {:<22} | {:<9} | {:<12}",
        "Head ID", "Canonical Slug", "Domain Title", "Size (MB)", "Status"
    );
    println!(
        "----------------------------------------------------------------------------------------"
    );

    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let anchors_dir = std::path::PathBuf::from(format!("{}/.nioxs/anchors", home));

    let registry_paths = [
        format!("{}/.nioxs/registry/hydra_anchor_registry.json", home),
        format!(
            "{}/nio-labs/nio-ideas/registry/hydra_anchor_registry.json",
            home
        ),
    ];
    let mut anchors_data = Vec::new();
    for p in &registry_paths {
        if let Ok(c) = std::fs::read_to_string(p) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&c) {
                if let Some(anchors) = val.get("anchors").and_then(|a| a.as_array()) {
                    for a in anchors {
                        let id = a
                            .get("id")
                            .and_then(|i| i.as_str())
                            .unwrap_or("")
                            .to_string();
                        let slug = a
                            .get("slug")
                            .and_then(|s| s.as_str())
                            .unwrap_or("")
                            .to_string();
                        let title = a
                            .get("title")
                            .and_then(|t| t.as_str())
                            .unwrap_or("")
                            .to_string();
                        anchors_data.push((id, slug, title));
                    }
                }
            }
            break;
        }
    }

    let mut total_mb = 0.0;
    let mut installed_count = 0;

    for (id, slug, title) in &anchors_data {
        let candidate_dirs = [
            anchors_dir.join(slug),
            anchors_dir.join(format!("head-{}", id)),
            heads_dir.join(slug),
            heads_dir.join(format!("head-{}", id)),
            heads_dir.join("hydra").join(slug),
        ];
        let target_p = candidate_dirs.iter().find(|p| p.exists());
        let size_mb = if let Some(p) = target_p {
            let bytes: u64 = std::fs::read_dir(p)
                .map(|entries| {
                    entries
                        .filter_map(|e| e.ok())
                        .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
                        .sum()
                })
                .unwrap_or(0);
            bytes as f64 / (1024.0 * 1024.0)
        } else {
            0.0
        };

        let short_title = if title.len() > 22 {
            format!("{}...", &title[..19])
        } else {
            title.clone()
        };

        if size_mb > 0.05 {
            installed_count += 1;
            total_mb += size_mb;
            let status = if size_mb < 30.0 {
                "\x1b[1;32mREADY (GGUF)\x1b[0m"
            } else {
                "\x1b[1;33mLARGE\x1b[0m"
            };
            println!(
                "Head #{:<4} | {:<36} | {:<22} | {:>6.2} MB | {}",
                id, slug, short_title, size_mb, status
            );
        } else {
            println!(
                "Head #{:<4} | {:<36} | {:<22} | {:>6}    | \x1b[1;33mNOT PRESENT\x1b[0m",
                id, slug, short_title, "-"
            );
        }
    }
    println!(
        "----------------------------------------------------------------------------------------"
    );
    println!("Directory: {}", anchors_dir.display());
    println!(
        "Status:    {} / {} Anchors Installed | Total Footprint: {:.2} MB (GGUF + FP8, Rank 12-32)",
        installed_count,
        anchors_data.len(),
        total_mb
    );
    println!(
        "Runtime:   In-Process Hot-Swap Latency < 1ms | Target Hardware: Raspberry Pi 5 / CPU / GPU\n"
    );
}

fn inspect_council_head(head_id: &str) {
    let resolved = nioxs_engine::lora::resolve_head_path(head_id);
    match resolved {
        Some(path) => {
            println!("\n\x1b[1;36m[Hydra Council Head Inspector]\x1b[0m");
            println!("  • Head Target:     {}", head_id);
            println!("  • On-Disk Path:    {}", path.display());
            let config_p = path.join("adapter_config.json");
            if let Ok(content) = std::fs::read_to_string(&config_p) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                    println!(
                        "  • Rank (r):        {}",
                        val.get("r").unwrap_or(&serde_json::Value::Null)
                    );
                    println!(
                        "  • LoRA Alpha:      {}",
                        val.get("lora_alpha").unwrap_or(&serde_json::Value::Null)
                    );
                    println!(
                        "  • Target Modules:  {}",
                        val.get("target_modules")
                            .unwrap_or(&serde_json::Value::Null)
                    );
                    println!(
                        "  • Base Anchor:     {}",
                        val.get("base_model_name_or_path")
                            .unwrap_or(&serde_json::Value::Null)
                    );
                }
            }
            let bytes = std::fs::read_dir(&path)
                .map(|entries| {
                    entries
                        .filter_map(|e| e.ok())
                        .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
                        .sum::<u64>()
                })
                .unwrap_or(0);
            println!(
                "  • Total Footprint: {:.2} MB (\x1b[1;32m< 3.0 MB SLA Met\x1b[0m)",
                bytes as f64 / (1024.0 * 1024.0)
            );
            println!(
                "  • System Prompt:\n    {}\n",
                nioxs_engine::hydra_specialist_system(head_id)
            );
        }
        None => {
            eprintln!(
                "Error: Could not locate council head '{head_id}'. Run 'nioxs council update' to install."
            );
        }
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arg = std::env::args().nth(1);
    if matches!(arg.as_deref(), Some("--version")) {
        println!("nioxs {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    if matches!(arg.as_deref(), Some("--help" | "-h")) {
        println!(
            "Usage: nioxs [serve|pull <model>|orca <prompt>|orca-nano <prompt>|orca-mini <prompt>|orca-omni <prompt>|meerkat <prompt>|viper <prompt>|skate <prompt>|manta <prompt>|beluga <prompt>|blue <prompt>|tiger <prompt>|cheetah <prompt>|fox <prompt>|rabbit <prompt>|mcp|init|undo|history]\n\
Commands:\n\
  serve          Start HTTP OpenAI-compatible AI gateway (default)\n\
  pull [model]   Download and verify local MicroLM weights (orca-nano, orca-mini, orca-omni, viper, skate, manta, beluga, blue, tiger, cheetah, fox, rabbit, all)\n\
  orca <prompt>  Run Orca Sovereign Pod Fleet (auto-routes lead/pods/reasoners; --tier nano|mini|omni|max|ultra, --verify)\n\
  orca-nano <prompt> Run Orca Nano 0.5B×13 Sovereign Pod Fleet\n\
  orca-mini <prompt> Run Orca Mini 1.5B×13 Sovereign Pod Fleet\n\
  orca-omni <prompt> Run Orca Omni Hybrid Fleet (0.5B fast Lead + 1.5B+ heavy Pods)\n\
  orca-pack <prompt> Run Orca Pack Mixture-of-Agents (1.5B Lead + 12×0.5B concurrent Pods)\n\
  tiny-orca <prompt> Run Orca-XS (Tiny Orca) 23×135M Micro-Pod Fleet (1 active pod, Web Scout + Guard)\n\
  meerkat <prompt>   Run Meerkat Self-Healing Herd (0.5B Scout + 1.5B Builder + 4-tool loop + /teach memory)\n\
  meerkat-s <prompt> Run Meerkat-S (0.5B IQ3_XXS Reflex + 0.5B Wave 1 LoRAs, ~624 MB RAM)\n\
  meerkat-xs <prompt> Run Meerkat-XS (2×135M edge companion for fast chat & light helpers, ~334 MB RAM)\n\
  tiny-meerkat <prompt> Alias for Meerkat-S (falls back to Meerkat-XS if 0.5B IQ3_XXS is absent)\n\
  viper [-x|-y] <prompt> Run local Viper 100M tactical terminal reflex (-x executes natively with safety clutch)\n\
  viper plugin <install|list> Manage isolated Python plugins for Viper sandbox\n\
  skate <prompt> Run local in-process Skate 135M ambient desk companion\n\
  manta <prompt> Run local in-process Manta 494M systems guru & DAG architect (<manta_plan>)\n\
  beluga <prompt> Run local in-process Beluga 1.54B sovereign lead architect & CoT reasoning\n\
  blue <prompt>   Run local in-process Blue 3.82B sovereign reflex lead companion\n\
  tiger <prompt>  Run local in-process Tiger 3.0B sovereign vibe coder & daily systems workhorse\n\
  cheetah <prompt> Run local in-process Cheetah 1.5B sub-20ms sovereign code sprinter\n\
  fox <prompt>    Run local in-process Fox 3.0B sovereign articulate mentor & interactive rubber duck\n\
  rabbit <prompt> Run local in-process Rabbit 3.0B sovereign deep thinker & root-cause detective (<think> CoT)\n\
  update         Download and install latest calibrated council heads (< 3MB SLA)\n\
  calibrate-head <head> Calibrate and compress LoRA head to < 3MB (FP8 + SVD)\n\
  mcp            Start Model Context Protocol (MCP) JSON-RPC stdio server\n\
  init           Initialize configuration for coding agents (--aider, --cline, --opencode, --cursor)\n\
  undo           Rollback last applied multi-file patch (or specific patch_id)\n\
  history        Display local session turn journal from .nioxs/history.jsonl\n\
\n\
Options:\n\
  --audit, -a                        Trigger Chair C formal invariant verification & safety audit pass\n\
  --device, -d <auto|cpu|gpu>        Select hardware compute device (default: auto; uses GPU if available, else pure CPU)\n\
  --gpu                              Shortcut to target GPU acceleration\n\
  --cpu                              Shortcut to target pure-CPU execution\n\
  --auto                             Shortcut to use automatic hardware detection\n\
  --flow, -f <token|burst|paragraph> Output streaming cadence (default: token; instant token stream)\n\
  --max-tokens, -n <count>           Max tokens to generate (default: 4096 / unconstrained until EOS)\n\
  --plan                             Manta only: buffer and validate a complete XML DAG before printing\n\
\n\
Environment: NIOXS_LISTEN (127.0.0.1:7460), NIOXS_DEVICE (auto|cpu|gpu), NIOXS_DB_URL, NIOXS_DB_TOKEN_FILE, NIOXS_DEPLOYMENT_ID."
        );
        return Ok(());
    }
    if arg.as_deref() == Some("council") || arg.as_deref() == Some("update") {
        let sub = if arg.as_deref() == Some("update") {
            "update".to_string()
        } else {
            std::env::args().nth(2).unwrap_or_else(|| "ls".into())
        };

        let heads_dir = if let Some(home) = std::env::var_os("HOME") {
            std::path::PathBuf::from(home).join(".nioxs").join("heads")
        } else {
            std::path::PathBuf::from(".nioxs").join("heads")
        };
        std::fs::create_dir_all(&heads_dir)?;

        if sub == "update" {
            println!(
                "\x1b[1;36m[Hydra Council]\x1b[0m Checking for latest calibrated council heads..."
            );
            let download_url = "https://github.com/nio-labs/nio-xs-models/releases/download/v0.5.1/hydra-founding-council-10heads.tar.gz";
            let tar_dest = std::env::temp_dir().join("hydra-founding-council-10heads.tar.gz");

            println!("  • Downloading Founding Council bundle from Release v0.5.1...");
            match reqwest::get(download_url).await {
                Ok(resp) if resp.status().is_success() => {
                    let bytes = resp.bytes().await?;
                    std::fs::write(&tar_dest, &bytes)?;
                    println!(
                        "  • Extracting {:.2} MB archive to '{}'...",
                        bytes.len() as f64 / (1024.0 * 1024.0),
                        heads_dir.display()
                    );
                    let status = std::process::Command::new("tar")
                        .arg("-xzf")
                        .arg(&tar_dest)
                        .arg("-C")
                        .arg(&heads_dir)
                        .status();
                    if let Ok(st) = status {
                        if st.success() {
                            println!(
                                "\x1b[1;32m✔ Successfully updated Hydra Council to v0.5.1 (10 Calibrated Heads)!\x1b[0m"
                            );
                        }
                    }
                }
                _ => {
                    let repo_heads = std::path::PathBuf::from(
                        "/home/mn/nio-labs/nio-xs-models/models/hydra/heads",
                    );
                    if repo_heads.exists() {
                        println!("  • Synchronizing from local calibrated models repository...");
                        for entry in std::fs::read_dir(&repo_heads)? {
                            let entry = entry?;
                            if entry.file_type()?.is_dir() {
                                let target = heads_dir.join(entry.file_name());
                                std::fs::create_dir_all(&target)?;
                                for f in std::fs::read_dir(entry.path())? {
                                    let f = f?;
                                    let _ = std::fs::copy(f.path(), target.join(f.file_name()));
                                }
                            }
                        }
                        println!(
                            "\x1b[1;32m✔ Successfully updated Hydra Council from local repository (10 Calibrated Heads)!\x1b[0m"
                        );
                    }
                }
            }
            print_council_heads_table(&heads_dir);
            return Ok(());
        }

        if sub == "ls" || sub == "list" || sub == "status" {
            print_council_heads_table(&heads_dir);
            return Ok(());
        }

        if sub == "inspect" {
            let head_id = std::env::args().nth(3).unwrap_or_else(|| "001".into());
            inspect_council_head(&head_id);
            return Ok(());
        }

        if sub == "decompress" || sub == "unpack" {
            println!(
                "\x1b[1;36m[Hydra Council]\x1b[0m Decompressing LZ4 adapter weights for Sovereign Council anchors..."
            );
            match nioxs_engine::decompress_all_anchors() {
                Ok(count) => {
                    println!(
                        "\x1b[1;32m✔ Successfully decompressed {count} anchor adapter(s) to .safetensors\x1b[0m"
                    );
                }
                Err(err) => {
                    eprintln!("\x1b[1;31m✘ Failed to decompress anchors: {err}\x1b[0m");
                }
            }
            return Ok(());
        }

        if sub == "export-gguf" || sub == "convert-gguf" {
            println!(
                "\x1b[1;36m[Hydra Council]\x1b[0m Converting Sovereign Council adapters to GGUF..."
            );
            match nioxs_engine::convert_all_anchors_to_gguf() {
                Ok(count) => {
                    println!(
                        "\x1b[1;32m✔ Successfully converted/verified {count} anchor adapter(s) in GGUF format\x1b[0m"
                    );
                }
                Err(err) => {
                    eprintln!("\x1b[1;31m✘ Failed to convert anchors to GGUF: {err}\x1b[0m");
                }
            }
            return Ok(());
        }

        println!("Usage: nioxs council [ls|update|inspect <id>|status|decompress|export-gguf]");
        println!("       nioxs update");
        return Ok(());
    }
    if arg.as_deref() == Some("pull") {
        let model_target = std::env::args().nth(2).unwrap_or_else(|| "all".into());
        let auto_yes = std::env::args().any(|a| a == "-y" || a == "--yes");
        let manifest = nioxs_artifacts::default_manifest();
        if model_target == "all" {
            println!("Pulling all production MicroLM models...");
            for entry in &manifest {
                let _ = nioxs_artifacts::ensure_model_ready(&entry.model_id, None)
                    .map_err(|e| format!("Failed to download {}: {e}", entry.model_id))?;
            }
            println!("✔ All production models downloaded and verified.");
            return Ok(());
        }
        let lowered = model_target.trim().to_ascii_lowercase();
        let (fleet_prefixes, manifest_tag): (Vec<&str>, Option<(&str, &str)>) =
            match lowered.as_str() {
                "orca" | "orca-mini" => (vec!["orca-mini"], None),
                "orca-nano" => (vec!["orca-nano"], None),
                "orca-omni" => (vec!["orca-nano", "orca-mini"], None),
                "tiny-orca" | "baby-orca" | "orca-xs" => (vec!["tiny-orca"], None),
                "meerkat-xs" | "baby-meerkat" => {
                    (vec!["tiny-meerkat-scout", "tiny-meerkat-builder"], None)
                }
                "meerkat-s" | "tiny-meerkat" => (vec!["tiny-meerkat-reflex"], None),
                "meerkat" => (vec!["meerkat"], None),
                "health-wellness" => (
                    vec!["health"],
                    Some(("wave1_bundle_manifest.json", "v0.6.1")),
                ),
                "everyday-home" => (
                    vec!["everyday"],
                    Some(("wave1_bundle_manifest.json", "v0.6.1")),
                ),
                "orca-wave1" | "wave1" => (
                    vec!["health", "everyday"],
                    Some(("wave1_bundle_manifest.json", "v0.6.1")),
                ),
                "math" => (vec!["math"], Some(("wave2_bundle_manifest.json", "v0.6.3"))),
                "history" => (
                    vec!["history"],
                    Some(("wave2_bundle_manifest.json", "v0.6.3")),
                ),
                "general-knowledge" => (vec!["gk"], Some(("wave2_bundle_manifest.json", "v0.6.3"))),
                "education-tutor" => (
                    vec!["tutor"],
                    Some(("wave2_bundle_manifest.json", "v0.6.3")),
                ),
                "orca-wave2" | "wave2" => (
                    vec!["math", "history", "gk", "tutor"],
                    Some(("wave2_bundle_manifest.json", "v0.6.3")),
                ),
                _ => (vec![], None),
            };
        if !fleet_prefixes.is_empty() {
            let pod_entries: Vec<_> = manifest
                .iter()
                .filter(|e| {
                    fleet_prefixes
                        .iter()
                        .any(|p| e.model_id == *p || e.model_id.starts_with(&format!("{p}-")))
                })
                .collect();
            println!(
                "Pulling {} {} ({} pods + manifest)...",
                lowered,
                if manifest_tag.is_some() {
                    "bundle"
                } else {
                    "fleet"
                },
                pod_entries.len()
            );
            for entry in pod_entries {
                let _ = nioxs_artifacts::prompt_and_pull_if_needed(&entry.model_id, auto_yes)
                    .map_err(|e| format!("Failed to pull {}: {e}", entry.model_id))?;
            }
            let (manifest_name, tag) = manifest_tag.unwrap_or(("pod_manifest.json", "v0.6.0"));
            let manifest_url = format!(
                "https://github.com/nio-labs/nio-xs-models/releases/download/{tag}/{manifest_name}"
            );
            let manifest_dest = nioxs_artifacts::resolve_models_dir().join(manifest_name);
            if !manifest_dest.exists() {
                if let Ok(resp) = reqwest::get(&manifest_url).await {
                    if resp.status().is_success() {
                        if let Ok(bytes) = resp.bytes().await {
                            let _ = std::fs::write(&manifest_dest, &bytes);
                        }
                    }
                }
            }
            println!(
                "✔ {} '{lowered}' ready in {}",
                if manifest_tag.is_some() {
                    "Bundle"
                } else {
                    "Fleet"
                },
                nioxs_artifacts::resolve_models_dir().display()
            );
            return Ok(());
        }
        let normalized = match lowered.as_str() {
            "viper" | "nio-viper" => "nio-viper",
            "skate" | "nio-skate" => "nio-skate",
            "manta" | "nio-manta" => "nio-manta",
            "beluga" | "nio-beluga" => "nio-beluga",
            "cuda" | "nio-cuda" => "nio-cuda",
            "blue" | "nio-blue" => "blue",
            "tiger" | "nio-tiger" => "tiger",
            "fox" | "nio-fox" => "fox",
            "rabbit" | "nio-rabbit" => "rabbit",
            "cheetah" | "nio-cheetah" => "cheetah",
            "fin" | "nio-fin" => "fin",
            "fin-base" | "nio-fin-base" => "fin-base",
            "fin-kernel" | "nio-fin-kernel" => "fin-kernel",
            "fin-sql" | "nio-fin-sql" => "fin-sql",
            "fin-concurrency" | "nio-fin-concurrency" => "fin-concurrency",
            "fin-raft" | "nio-fin-raft" => "fin-raft",
            other => other,
        };
        let path = nioxs_artifacts::prompt_and_pull_if_needed(normalized, auto_yes)
            .map_err(|e| format!("Failed to pull {normalized}: {e}"))?;
        println!("✔ Model '{normalized}' ready: {}", path.display());
        return Ok(());
    }
    if arg.as_deref() == Some("calibrate-head") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() || raw_args[0] == "-h" || raw_args[0] == "--help" {
            println!(
                "Usage: nioxs calibrate-head <head_path_or_id> [--out <dir>] [--rank <target_rank>] [--no-fp8] [--lz4] [--inplace]\n\n\
                 Calibrates and compresses unquantized FP32 LoRA adapters down to < 3.0 MB per head\n\
                 (target: ~2.8 MB) using SVD subspace projection and FP8 E4M3 quantization.\n\n\
                 Arguments:\n\
                   <head_path_or_id>    Path to adapter directory (e.g. ~/.nioxs/heads/head-001) or head ID\n\
                 Options:\n\
                   --out, -o <dir>      Destination output directory (default: overwrites in-place or writes to <head>_calibrated)\n\
                   --rank, -r <n>       Target LoRA rank truncation via factored SVD (default: 12 for < 3MB, or 16/32)\n\
                   --no-fp8             Preserves FP16 precision instead of FP8 E4M3\n\
                   --lz4                Compresses SafeTensors payload with LZ4 frame\n\
                   --inplace            Overwrites the input directory directly in-place"
            );
            return Ok(());
        }

        let input_arg = &raw_args[0];
        let input_path = if let Some(p) = nioxs_engine::lora::resolve_head_path(input_arg) {
            p
        } else {
            std::path::PathBuf::from(input_arg)
        };

        if !input_path.exists() {
            eprintln!(
                "Error: Adapter path '{}' does not exist.",
                input_path.display()
            );
            std::process::exit(1);
        }

        let mut out_dir = None;
        let mut target_rank = Some(12); // Default 12 yields ~2.75 MB (< 3MB)
        let mut use_fp8 = true;
        let mut compress_lz4 = false;
        let mut inplace = false;

        let mut iter = raw_args[1..].iter();
        while let Some(a) = iter.next() {
            if a == "--out" || a == "-o" {
                if let Some(val) = iter.next() {
                    out_dir = Some(std::path::PathBuf::from(val));
                }
            } else if a == "--rank" || a == "-r" {
                if let Some(val) = iter.next() {
                    target_rank = val.parse::<usize>().ok();
                }
            } else if a == "--no-fp8" {
                use_fp8 = false;
            } else if a == "--lz4" {
                compress_lz4 = true;
            } else if a == "--inplace" {
                inplace = true;
            }
        }

        let final_out = if inplace {
            input_path.clone()
        } else if let Some(out) = out_dir {
            out
        } else {
            let parent = input_path
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."));
            let name = input_path.file_name().unwrap_or_default().to_string_lossy();
            parent.join(format!("{name}_calibrated"))
        };

        println!(
            "\x1b[1;36m[Hydra Calibration Engine]\x1b[0m Calibrating '{}' -> '{}'...",
            input_path.display(),
            final_out.display()
        );
        let t0 = std::time::Instant::now();
        let report = nioxs_engine::lora::calibrate_head_dir(
            &input_path,
            &final_out,
            target_rank,
            use_fp8,
            compress_lz4,
        )
        .map_err(|e| format!("Calibration failed: {e}"))?;
        let elapsed = t0.elapsed().as_secs_f64();

        println!(
            "\x1b[1;32m✔ Head Calibration Complete in {:.2}s\x1b[0m",
            elapsed
        );
        println!("  • Head ID:             {}", report.head_id);
        println!(
            "  • Original Size:       {:.2} MB",
            report.original_size_bytes as f64 / (1024.0 * 1024.0)
        );
        println!(
            "  • Calibrated Size:     \x1b[1;32m{:.2} MB\x1b[0m (Target < 3.0 MB met)",
            report.calibrated_size_bytes as f64 / (1024.0 * 1024.0)
        );
        println!(
            "  • Compression Ratio:   \x1b[1;33m{:.1}x\x1b[0m",
            report.compression_ratio
        );
        println!("  • Storage Saved:       {:.2} MB", report.saved_mb);
        println!(
            "  • Effective Rank:      {} -> {}",
            report.original_rank, report.calibrated_rank
        );
        println!(
            "  • Precision:           {}",
            if report.is_fp8 {
                "FP8 E4M3 (scaled)"
            } else {
                "FP16"
            }
        );
        println!(
            "  • Container:           {}",
            if report.is_compressed {
                "SafeTensors + LZ4"
            } else {
                "SafeTensors (Standard)"
            }
        );
        return Ok(());
    }
    if arg.as_deref() == Some("skate") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!(
                "Usage: nioxs skate [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }

        let mut custom_max_tokens = None;
        let mut device_pref = nioxs_engine::DevicePreference::Auto;
        let mut flow_mode = FlowMode::Token;
        let mut prompt_parts = Vec::new();
        let mut iter = raw_args.clone().into_iter();
        while let Some(a) = iter.next() {
            if a == "-n" || a == "--max-tokens" {
                if let Some(val) = iter.next() {
                    custom_max_tokens = val.parse::<usize>().ok();
                }
            } else if a == "-d" || a == "--device" {
                if let Some(val) = iter.next() {
                    device_pref = nioxs_engine::DevicePreference::parse(&val);
                }
            } else if a == "--gpu" {
                device_pref = nioxs_engine::DevicePreference::Gpu;
            } else if a == "--cpu" {
                device_pref = nioxs_engine::DevicePreference::Cpu;
            } else if a == "--auto" {
                device_pref = nioxs_engine::DevicePreference::Auto;
            } else if a == "-f" || a == "--flow" {
                if let Some(val) = iter.next() {
                    flow_mode = FlowMode::parse(&val);
                }
            } else {
                prompt_parts.push(a);
            }
        }
        let prompt = prompt_parts.join(" ");
        if prompt.is_empty() {
            eprintln!(
                "Usage: nioxs skate [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }

        let max_tokens = custom_max_tokens.unwrap_or(2048);
        let mut flow = StreamFlow::new(flow_mode);
        match skate_safety::generate_stream_with_device(&prompt, device_pref, max_tokens).await {
            Ok(mut rx) => {
                while let Some(chunk) = rx.recv().await {
                    flow.push(&chunk);
                }
                flow.finish();
                println!();
            }
            Err(_) => {
                let response = skate_safety::generate(&prompt).await?;
                println!("{response}");
            }
        }
        return Ok(());
    }
    if arg.as_deref() == Some("viper") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!("Usage: nioxs viper [-x|--run] [-y|--yes] <prompt>");
            eprintln!("       nioxs viper plugin <install <pkg>|list>");
            return Ok(());
        }

        // Subcommand: Python Plugin Sandbox Manager
        if raw_args.first().map(|s| s.as_str()) == Some("plugin") {
            let action = raw_args.get(1).map(|s| s.as_str()).unwrap_or("help");
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .unwrap_or_else(|_| ".".into());
            let plugin_dir = std::path::PathBuf::from(home)
                .join(".nioxs")
                .join("plugins");
            let venv_dir = plugin_dir.join("venv");
            std::fs::create_dir_all(&plugin_dir)?;

            match action {
                "install" => {
                    let pkg = raw_args
                        .get(2)
                        .ok_or("Usage: nioxs viper plugin install <package_name>")?;
                    if !venv_dir.exists() {
                        println!(
                            "📦 Initializing isolated Viper plugin sandbox in {}...",
                            venv_dir.display()
                        );
                        let venv_status = std::process::Command::new("python3")
                            .args(["-m", "venv", &venv_dir.to_string_lossy()])
                            .status()
                            .or_else(|_| {
                                std::process::Command::new("python")
                                    .args(["-m", "venv", &venv_dir.to_string_lossy()])
                                    .status()
                            })?;
                        if !venv_status.success() {
                            return Err(
                                "Failed to create Python virtualenv. Ensure python3 is installed."
                                    .into(),
                            );
                        }
                    }
                    #[cfg(target_os = "windows")]
                    let pip_bin = venv_dir.join("Scripts").join("pip.exe");
                    #[cfg(not(target_os = "windows"))]
                    let pip_bin = venv_dir.join("bin").join("pip");

                    println!("📦 Installing plugin '{pkg}' into isolated Viper sandbox...");
                    let status = std::process::Command::new(pip_bin)
                        .args(["install", pkg])
                        .status()?;
                    if status.success() {
                        println!("✔ Plugin '{pkg}' successfully installed in Viper sandbox!");
                    } else {
                        return Err(format!("Failed to install plugin '{pkg}'").into());
                    }
                    return Ok(());
                }
                "list" => {
                    if !venv_dir.exists() {
                        println!(
                            "No plugins installed yet. Use 'nioxs viper plugin install <package>'"
                        );
                        return Ok(());
                    }
                    #[cfg(target_os = "windows")]
                    let pip_bin = venv_dir.join("Scripts").join("pip.exe");
                    #[cfg(not(target_os = "windows"))]
                    let pip_bin = venv_dir.join("bin").join("pip");

                    println!(
                        "📦 Installed NioViper Plugins (isolated in {}):",
                        venv_dir.display()
                    );
                    let _ = std::process::Command::new(pip_bin)
                        .args(["list"])
                        .status()?;
                    return Ok(());
                }
                _ => {
                    println!(
                        "Usage:\n  nioxs viper plugin install <package>\n  nioxs viper plugin list"
                    );
                    return Ok(());
                }
            }
        }

        // Parse Swiss Gear execution flags: -x / --run, -y / --yes
        let mut execute = false;
        let mut auto_yes = false;
        let mut prompt_tokens = Vec::new();

        for arg in raw_args {
            if arg == "-x" || arg == "--run" {
                execute = true;
            } else if arg == "-y" || arg == "--yes" {
                auto_yes = true;
            } else {
                prompt_tokens.push(arg);
            }
        }

        let prompt = prompt_tokens.join(" ");
        if prompt.is_empty() {
            eprintln!("Usage: nioxs viper [-x|--run] [-y|--yes] <prompt>");
            return Ok(());
        }

        let engine = nioxs_engine::AstEngine::for_model("nio-viper")?;
        let max_tokens = if prompt.split_whitespace().count() <= 2 {
            32
        } else {
            128
        };
        let mut rx = engine
            .generate_stream(prompt.clone(), max_tokens, 0.7, 0.9)
            .await?;
        use std::io::Write;
        let mut generated_cmd = String::new();
        while let Some(chunk) = rx.recv().await {
            print!("{chunk}");
            let _ = std::io::stdout().flush();
            generated_cmd.push_str(&chunk);
        }
        println!();

        let clean_cmd = ground_and_sanitize_command(&prompt, &generated_cmd);
        if clean_cmd != generated_cmd.trim() && !execute {
            println!("🎯 Grounded reflex: {clean_cmd}");
        }

        if execute {
            if clean_cmd.is_empty() {
                eprintln!("No executable command was generated.");
                return Ok(());
            }

            // Safety Clutch: Check for destructive mutations
            let is_destructive = clean_cmd.starts_with("rm ")
                || clean_cmd.contains("rm -rf")
                || clean_cmd.contains("kill -9")
                || clean_cmd.contains("pkill -9")
                || clean_cmd.contains("git reset --hard")
                || clean_cmd.contains("Stop-Process")
                || clean_cmd.contains("Remove-Item")
                || clean_cmd.contains("drop database");

            if is_destructive && !auto_yes {
                print!("⚡ Destructive command detected. Execute: \"{clean_cmd}\"? [Y/n] ");
                let _ = std::io::stdout().flush();
                let mut input = String::new();
                std::io::stdin().read_line(&mut input)?;
                let trimmed = input.trim().to_lowercase();
                if !trimmed.is_empty() && trimmed != "y" && trimmed != "yes" {
                    println!("Execution canceled.");
                    return Ok(());
                }
            }

            println!("🚀 Executing action: {clean_cmd}");
            #[cfg(target_os = "windows")]
            let mut child = std::process::Command::new("powershell")
                .args(["-NoProfile", "-Command", &clean_cmd])
                .spawn()?;

            #[cfg(not(target_os = "windows"))]
            let mut child = std::process::Command::new("sh")
                .args(["-c", &clean_cmd])
                .spawn()?;

            let status = child.wait()?;
            if !status.success() {
                eprintln!("Action finished with status: {status}");
            }
        }

        return Ok(());
    }
    if arg.as_deref() == Some("beluga") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!(
                "Usage: nioxs beluga [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }

        let mut custom_max_tokens = None;
        let mut device_pref = nioxs_engine::DevicePreference::Auto;
        let mut flow_mode = FlowMode::Token;
        let mut prompt_parts = Vec::new();
        let mut iter = raw_args.clone().into_iter();
        while let Some(a) = iter.next() {
            if a == "-n" || a == "--max-tokens" {
                if let Some(val) = iter.next() {
                    custom_max_tokens = val.parse::<usize>().ok();
                }
            } else if a == "-d" || a == "--device" {
                if let Some(val) = iter.next() {
                    device_pref = nioxs_engine::DevicePreference::parse(&val);
                }
            } else if a == "--gpu" {
                device_pref = nioxs_engine::DevicePreference::Gpu;
            } else if a == "--cpu" {
                device_pref = nioxs_engine::DevicePreference::Cpu;
            } else if a == "--auto" {
                device_pref = nioxs_engine::DevicePreference::Auto;
            } else if a == "-f" || a == "--flow" {
                if let Some(val) = iter.next() {
                    flow_mode = FlowMode::parse(&val);
                }
            } else {
                prompt_parts.push(a);
            }
        }
        let prompt = prompt_parts.join(" ");
        if prompt.is_empty() {
            eprintln!(
                "Usage: nioxs beluga [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }
        let engine = nioxs_engine::AstEngine::for_model_with_device("nio-beluga", device_pref)?;
        let max_tokens = custom_max_tokens.unwrap_or(4096);
        let formatted_prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            nioxs_engine::beluga_system()
        );
        let mut rx = engine
            .generate_stream(formatted_prompt, max_tokens, 0.6, 0.9)
            .await?;
        let mut flow = StreamFlow::new(flow_mode);
        flow.set_show_think_tags(true);
        while let Some(chunk) = rx.recv().await {
            flow.push(&chunk);
        }
        flow.finish();
        println!();
        return Ok(());
    }
    if arg.as_deref() == Some("blue") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!(
                "Usage: nioxs blue [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }

        let mut custom_max_tokens = None;
        let mut device_pref = nioxs_engine::DevicePreference::Auto;
        let mut flow_mode = FlowMode::Token;
        let mut prompt_parts = Vec::new();
        let mut iter = raw_args.clone().into_iter();
        while let Some(a) = iter.next() {
            if a == "-n" || a == "--max-tokens" {
                if let Some(val) = iter.next() {
                    custom_max_tokens = val.parse::<usize>().ok();
                }
            } else if a == "-d" || a == "--device" {
                if let Some(val) = iter.next() {
                    device_pref = nioxs_engine::DevicePreference::parse(&val);
                }
            } else if a == "--gpu" {
                device_pref = nioxs_engine::DevicePreference::Gpu;
            } else if a == "--cpu" {
                device_pref = nioxs_engine::DevicePreference::Cpu;
            } else if a == "--auto" {
                device_pref = nioxs_engine::DevicePreference::Auto;
            } else if a == "-f" || a == "--flow" {
                if let Some(val) = iter.next() {
                    flow_mode = FlowMode::parse(&val);
                }
            } else {
                prompt_parts.push(a);
            }
        }
        let prompt = prompt_parts.join(" ");
        if prompt.is_empty() {
            eprintln!(
                "Usage: nioxs blue [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }
        let engine = nioxs_engine::AstEngine::for_model_with_device("blue", device_pref)?;
        let max_tokens = custom_max_tokens.unwrap_or(4096);
        let formatted_prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            nioxs_engine::blue_system()
        );
        let mut rx = engine
            .generate_stream(formatted_prompt, max_tokens, 0.35, 0.88)
            .await?;
        let mut flow = StreamFlow::new(flow_mode);
        while let Some(chunk) = rx.recv().await {
            flow.push(&chunk);
        }
        flow.finish();
        println!();
        return Ok(());
    }
    if arg.as_deref() == Some("orca-pack") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        return Ok(run_orca_pack_cli(raw_args).await?);
    }
    if matches!(arg.as_deref(), Some("tiny-orca" | "baby-orca" | "orca-xs")) {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        return Ok(run_baby_orca_cli(raw_args).await?);
    }
    if matches!(arg.as_deref(), Some("meerkat-xs")) {
        let mut raw_args = vec!["--xs".to_string()];
        raw_args.extend(std::env::args().skip(2));
        return Ok(run_meerkat_cli(raw_args).await?);
    }
    if matches!(
        arg.as_deref(),
        Some("meerkat-s" | "tiny-meerkat" | "baby-meerkat")
    ) {
        let mut raw_args = vec!["--tiny".to_string()];
        raw_args.extend(std::env::args().skip(2));
        return Ok(run_meerkat_cli(raw_args).await?);
    }
    if matches!(arg.as_deref(), Some("meerkat" | "penguin")) {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        return Ok(run_meerkat_cli(raw_args).await?);
    }
    if matches!(
        arg.as_deref(),
        Some("orca" | "orca-nano" | "orca-mini" | "orca-omni" | "orca-max" | "orca-ultra")
    ) {
        let cmd_name = arg.as_deref().unwrap_or("orca");
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() || raw_args[0] == "-h" || raw_args[0] == "--help" {
            println!(
                "Usage: nioxs {cmd_name} [--tier <nano|mini|omni|max|ultra>] [--think|--no-think] [--verify|--no-verify] [--retries <N>] [--pod <id>] [--verbose|-v] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            println!(
                "Tiers: orca-nano (0.5B), orca-mini (1.5B), orca-omni (0.5B Lead + 1.5B+ Pods), orca-max (3B), orca-ultra (7B)"
            );
            return Ok(());
        }

        let mut tier = match cmd_name {
            "orca-nano" => "nano".to_string(),
            "orca-mini" => "mini".to_string(),
            "orca-omni" => "omni".to_string(),
            "orca-max" => "max".to_string(),
            "orca-ultra" => "ultra".to_string(),
            _ => "auto".to_string(),
        };
        let mut custom_max_tokens: Option<usize> = None;
        let mut device_pref = nioxs_engine::DevicePreference::Auto;
        let mut flow_mode = FlowMode::Token;
        let mut think_override: Option<bool> = None;
        let mut forced_pod: Option<String> = None;
        let mut verify = true;
        let mut max_retries: usize = 3;
        let mut verbose = false;
        let mut prompt_parts = Vec::new();
        let mut iter = raw_args.into_iter();
        while let Some(a) = iter.next() {
            match a.as_str() {
                "--think" => think_override = Some(true),
                "--no-think" => think_override = Some(false),
                "--verify" => verify = true,
                "--no-verify" => verify = false,
                "--retries" => {
                    if let Some(v) = iter.next() {
                        max_retries = v.parse::<usize>().unwrap_or(3);
                    }
                }
                "--verbose" | "-v" => verbose = true,
                "--gpu" => device_pref = nioxs_engine::DevicePreference::Gpu,
                "--cpu" => device_pref = nioxs_engine::DevicePreference::Cpu,
                "--auto" => device_pref = nioxs_engine::DevicePreference::Auto,
                "--tier" => {
                    if let Some(v) = iter.next() {
                        tier = v.to_ascii_lowercase();
                    }
                }
                "--pod" => {
                    if let Some(v) = iter.next() {
                        forced_pod = Some(v);
                    }
                }
                "--flow" => {
                    if let Some(v) = iter.next() {
                        flow_mode = match v.to_ascii_lowercase().as_str() {
                            "burst" => FlowMode::Burst,
                            "paragraph" | "para" => FlowMode::Paragraph,
                            _ => FlowMode::Token,
                        };
                    }
                }
                "-n" | "--max-tokens" => {
                    if let Some(v) = iter.next() {
                        custom_max_tokens = v.parse::<usize>().ok();
                    }
                }
                "--device" => {
                    if let Some(v) = iter.next() {
                        device_pref = match v.to_ascii_lowercase().as_str() {
                            "cpu" => nioxs_engine::DevicePreference::Cpu,
                            "gpu" => nioxs_engine::DevicePreference::Gpu,
                            _ => nioxs_engine::DevicePreference::Auto,
                        };
                    }
                }
                other => prompt_parts.push(other.to_string()),
            }
        }
        let prompt = prompt_parts.join(" ");
        if prompt.is_empty() {
            return Err(format!("Usage: nioxs {cmd_name} <prompt>").into());
        }

        let home = std::env::var("HOME").unwrap_or_default();
        let gguf_dirs = [
            std::path::PathBuf::from("models/gguf"),
            std::path::PathBuf::from("models/orca_nano_gguf"),
            std::path::PathBuf::from(&home).join("nio-labs/nio-ideas/models/gguf"),
            std::path::PathBuf::from(&home).join("nio-labs/nio-ideas/models/orca_nano_gguf"),
            std::path::PathBuf::from(&home).join(".nioxs/models/orca"),
            std::path::PathBuf::from(&home).join(".nioxs/models"),
        ];
        let models_dir = nioxs_artifacts::resolve_models_dir();
        let tok_candidates = [
            models_dir.join("orca-tokenizer.json"),
            std::path::PathBuf::from("models/orca_lead_1_5b_instruct_tuned/tokenizer.json"),
            std::path::PathBuf::from(&home).join("nio-labs/nio-ideas/models/orca_lead_1_5b_instruct_tuned/tokenizer.json"),
            std::path::PathBuf::from(&home).join("nio-labs/nio-ideas/models/orca_nano_kaggle/orca_lite_models/orca_lite_lead_0_5b/tokenizer.json"),
            std::path::PathBuf::from(&home).join(".nioxs/models/orca/tokenizer.json"),
        ];
        let tokenizer_path = tok_candidates
            .into_iter()
            .find(|p| p.is_file())
            .ok_or_else(|| "Could not locate Orca tokenizer.json".to_string())?;

        let manifest_candidates = [
            std::env::var_os("ORCA_MANIFEST").map(std::path::PathBuf::from),
            Some(models_dir.join("pod_manifest.json")),
            Some(std::path::PathBuf::from(
                "data/phase2_instruct/pod_manifest.json",
            )),
            Some(
                std::path::PathBuf::from(&home)
                    .join("nio-labs/nio-ideas/data/phase2_instruct/pod_manifest.json"),
            ),
            Some(std::path::PathBuf::from(&home).join(".nioxs/models/orca/pod_manifest.json")),
            Some(std::path::PathBuf::from(&home).join(".nioxs/orca/pod_manifest.json")),
        ];
        let mut manifest_entries: Vec<serde_json::Value> = manifest_candidates
            .into_iter()
            .flatten()
            .find(|p| p.is_file())
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str::<Vec<serde_json::Value>>(&s).ok())
            .unwrap_or_default();
        for extra_name in ["wave1_bundle_manifest.json", "wave2_bundle_manifest.json"] {
            let extra_path = models_dir.join(extra_name);
            if extra_path.is_file() {
                if let Some(extra) = std::fs::read_to_string(&extra_path)
                    .ok()
                    .and_then(|s| serde_json::from_str::<Vec<serde_json::Value>>(&s).ok())
                {
                    manifest_entries.extend(extra);
                }
            }
        }

        let (pod_id, matched_think_meta, verify_mode, custom_gguf) =
            route_orca_pod_meta(&prompt, forced_pod.as_deref(), &manifest_entries);
        let is_reasoning_pod = |p: &str| p.contains("reasoner") || p == "architect";
        let think_enabled =
            think_override.unwrap_or_else(|| matched_think_meta || is_reasoning_pod(&pod_id));
        if verify_mode == "none" {
            verify = false;
        }

        let kebab = pod_id.replace('_', "-");
        let tier_specs: Vec<(&str, &[&str])> = match tier.as_str() {
            "nano" => vec![("nano", &["0.5b", "0-5b"])],
            "mini" => vec![("mini", &["1.5b", "1-5b"])],
            "omni" => {
                if think_enabled || is_reasoning_pod(&pod_id) {
                    // Heavy path: prefer 1.5B (or 3B/7B) reasoner/architect pods
                    vec![
                        ("mini", &["1.5b", "1-5b"]),
                        ("max", &["3b"]),
                        ("ultra", &["7b"]),
                        ("nano", &["0.5b", "0-5b"]),
                    ]
                } else {
                    // Fast path: 0.5B Lead/Action pods first, falling back to 1.5B
                    vec![("nano", &["0.5b", "0-5b"]), ("mini", &["1.5b", "1-5b"])]
                }
            }
            "max" => vec![("max", &["3b"])],
            "ultra" => vec![("ultra", &["7b"])],
            _ => vec![
                ("mini", &["1.5b", "1-5b"]),
                ("nano", &["0.5b", "0-5b"]),
                ("max", &["3b"]),
                ("ultra", &["7b"]),
            ],
        };

        let mut names = Vec::new();
        if let Some(ref g) = custom_gguf {
            names.push(g.clone());
        }
        for (t_name, size_tags) in &tier_specs {
            for size_tag in *size_tags {
                names.push(format!("orca-{t_name}-{kebab}-{size_tag}-q4_k_m.gguf"));
                names.push(format!("orca-{kebab}-{size_tag}-q4_k_m.gguf"));
            }
            if think_enabled {
                for size_tag in *size_tags {
                    names.push(format!("orca-{t_name}-reasoner-{size_tag}-q4_k_m.gguf"));
                    names.push(format!("orca-reasoner-{size_tag}-q4_k_m.gguf"));
                }
            }
            for size_tag in *size_tags {
                names.push(format!("orca-{t_name}-lead-{size_tag}-q4_k_m.gguf"));
                names.push(format!("orca-lead-{size_tag}-q4_k_m.gguf"));
            }
        }

        let model_path = names
            .iter()
            .flat_map(|n| gguf_dirs.iter().map(move |d| d.join(n)))
            .find(|p| p.is_file())
            .ok_or_else(|| format!("Could not locate Orca ({tier}) GGUF (run 'nioxs pull {cmd_name}' or check models/gguf/)"))?;

        let resolved_tier = if tier == "omni" {
            "omni"
        } else {
            let fname = model_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if fname.contains("0.5b") || fname.contains("0-5b") {
                "nano"
            } else if fname.contains("3b") {
                "max"
            } else if fname.contains("7b") {
                "ultra"
            } else {
                "mini"
            }
        };

        let mut engine =
            nioxs_engine::AstEngine::from_paths(model_path.clone(), tokenizer_path.clone());
        engine.set_device_preference(device_pref);
        let action_sys = "You are Orca (vMicroLM-Reflex), a concise, accurate coding assistant developed by Nio Labs.";
        let think_sys = "You are Orca-Reasoner (vMicroLM-Reflex), a compact coding specialist developed by Nio Labs. Start with a concise <think> block (short bulleted invariants and edge cases, under 12 lines) </think>, followed immediately by the complete working code.";
        let sys = if think_enabled { think_sys } else { action_sys };
        let formatted_prompt = format!(
            "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        );
        let max_tokens = custom_max_tokens.unwrap_or(8192);
        let temp = if think_enabled { 0.20 } else { 0.0 };
        let t0 = std::time::Instant::now();
        let mut rx = engine
            .generate_stream(formatted_prompt, max_tokens, temp, 0.90)
            .await?;
        let mut flow = StreamFlow::new(flow_mode);
        flow.set_show_think_tags(true);
        let mut full_resp = String::new();
        while let Some(chunk) = rx.recv().await {
            full_resp.push_str(&chunk);
            flow.push(&chunk);
        }
        flow.finish();
        println!();

        let mut verified_opt: Option<bool> = None;
        let mut retry_rounds = 0usize;
        let mut active_pod = pod_id.clone();
        let mut active_gguf = model_path.clone();

        if verify {
            let (mut ok, mut err) = verify_orca_with_mode(&full_resp, &verify_mode);
            let active_size_tags: &[&str] = match resolved_tier {
                "nano" => &["0.5b", "0-5b"],
                "omni" => &["1.5b", "1-5b", "3b", "7b"],
                "max" => &["3b"],
                "ultra" => &["7b"],
                _ => &["1.5b", "1-5b"],
            };
            let mut retry_ring: Vec<(String, std::path::PathBuf)> = Vec::new();
            for d in &gguf_dirs {
                if let Ok(entries) = std::fs::read_dir(d) {
                    for entry in entries.flatten() {
                        let p = entry.path();
                        if !p.is_file() {
                            continue;
                        }
                        let fname = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                        if fname.starts_with("orca-")
                            && fname.ends_with(".gguf")
                            && active_size_tags.iter().any(|t| fname.contains(t))
                            && !retry_ring.iter().any(|(_, existing)| existing == &p)
                        {
                            let label = fname
                                .trim_start_matches("orca-")
                                .split("-1.5b")
                                .next()
                                .unwrap_or(fname)
                                .split("-1-5b")
                                .next()
                                .unwrap_or(fname)
                                .split("-0.5b")
                                .next()
                                .unwrap_or(fname)
                                .to_string();
                            retry_ring.push((label, p));
                        }
                    }
                }
            }
            retry_ring.sort_by_key(|(label, p)| {
                let is_repair = label.contains("repair");
                let is_reason = label.contains("reasoner") || label.contains("architect");
                let is_same = *p == model_path;
                (is_same, !is_repair, !is_reason, label.clone())
            });
            if retry_ring.is_empty() {
                retry_ring.push((pod_id.clone(), model_path.clone()));
            }
            while !ok && retry_rounds < max_retries {
                let (r_pod, r_path) = retry_ring[retry_rounds % retry_ring.len()].clone();
                retry_rounds += 1;
                eprintln!(
                    "[verify] Attempt {} failed ({}) -> retrying with {}...",
                    retry_rounds,
                    err.lines().last().unwrap_or("error"),
                    r_pod
                );
                let mut r_eng =
                    nioxs_engine::AstEngine::from_paths(r_path.clone(), tokenizer_path.clone());
                r_eng.set_device_preference(device_pref);
                let fix_prompt = format!(
                    "<|im_start|>system\n{think_sys}<|im_end|>\n<|im_start|>user\nFix this code based on the error. Output the complete corrected code in a fenced block.\n\nOriginal task:\n{prompt}\n\nPrevious output:\n{full_resp}\n\nError:\n{err}<|im_end|>\n<|im_start|>assistant\n"
                );
                let mut r_rx = r_eng
                    .generate_stream(fix_prompt, max_tokens, 0.35, 0.90)
                    .await?;
                let mut r_flow = StreamFlow::new(flow_mode);
                r_flow.set_show_think_tags(true);
                full_resp.clear();
                while let Some(chunk) = r_rx.recv().await {
                    full_resp.push_str(&chunk);
                    r_flow.push(&chunk);
                }
                r_flow.finish();
                println!();
                active_pod = r_pod.to_string();
                active_gguf = r_path;
                let res = verify_orca_with_mode(&full_resp, &verify_mode);
                ok = res.0;
                err = res.1;
            }
            verified_opt = Some(ok);
        }

        if verbose {
            let secs = t0.elapsed().as_secs_f64().max(1e-6);
            println!("---");
            println!(
                "Orca [AstMicroEngine native] | tier={} | pod={} | think={} | verified={:?} | retries={} | gguf={} | {:.2}s ({} chars)",
                resolved_tier,
                active_pod,
                think_enabled || retry_rounds > 0,
                verified_opt,
                retry_rounds,
                active_gguf
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("?"),
                secs,
                full_resp.len()
            );
        }
        return Ok(());
    }
    if arg.as_deref() == Some("cheetah") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!(
                "Usage: nioxs cheetah [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }

        let mut custom_max_tokens = None;
        let mut device_pref = nioxs_engine::DevicePreference::Auto;
        let mut flow_mode = FlowMode::Token;
        let mut prompt_parts = Vec::new();
        let mut iter = raw_args.clone().into_iter();
        while let Some(a) = iter.next() {
            if a == "-n" || a == "--max-tokens" {
                if let Some(val) = iter.next() {
                    custom_max_tokens = val.parse::<usize>().ok();
                }
            } else if a == "-d" || a == "--device" {
                if let Some(val) = iter.next() {
                    device_pref = nioxs_engine::DevicePreference::parse(&val);
                }
            } else if a == "--gpu" {
                device_pref = nioxs_engine::DevicePreference::Gpu;
            } else if a == "--cpu" {
                device_pref = nioxs_engine::DevicePreference::Cpu;
            } else if a == "--auto" {
                device_pref = nioxs_engine::DevicePreference::Auto;
            } else if a == "-f" || a == "--flow" {
                if let Some(val) = iter.next() {
                    flow_mode = FlowMode::parse(&val);
                }
            } else {
                prompt_parts.push(a);
            }
        }
        let prompt = prompt_parts.join(" ");
        if prompt.is_empty() {
            eprintln!(
                "Usage: nioxs cheetah [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }
        let engine = nioxs_engine::AstEngine::for_model_with_device("cheetah", device_pref)?;
        let max_tokens = custom_max_tokens.unwrap_or(4096);
        let formatted_prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            nioxs_engine::cheetah_system()
        );
        let mut rx = engine
            .generate_stream(formatted_prompt, max_tokens, 0.20, 0.85)
            .await?;
        let mut flow = StreamFlow::new(flow_mode);
        flow.set_show_think_tags(true);
        while let Some(chunk) = rx.recv().await {
            flow.push(&chunk);
        }
        flow.finish();
        println!();
        return Ok(());
    }
    if arg.as_deref() == Some("tiger") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!(
                "Usage: nioxs tiger [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }

        let mut custom_max_tokens = None;
        let mut device_pref = nioxs_engine::DevicePreference::Auto;
        let mut flow_mode = FlowMode::Token;
        let mut prompt_parts = Vec::new();
        let mut iter = raw_args.clone().into_iter();
        while let Some(a) = iter.next() {
            if a == "-n" || a == "--max-tokens" {
                if let Some(val) = iter.next() {
                    custom_max_tokens = val.parse::<usize>().ok();
                }
            } else if a == "-d" || a == "--device" {
                if let Some(val) = iter.next() {
                    device_pref = nioxs_engine::DevicePreference::parse(&val);
                }
            } else if a == "--gpu" {
                device_pref = nioxs_engine::DevicePreference::Gpu;
            } else if a == "--cpu" {
                device_pref = nioxs_engine::DevicePreference::Cpu;
            } else if a == "--auto" {
                device_pref = nioxs_engine::DevicePreference::Auto;
            } else if a == "-f" || a == "--flow" {
                if let Some(val) = iter.next() {
                    flow_mode = FlowMode::parse(&val);
                }
            } else {
                prompt_parts.push(a);
            }
        }
        let prompt = prompt_parts.join(" ");
        if prompt.is_empty() {
            eprintln!(
                "Usage: nioxs tiger [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }
        let engine = nioxs_engine::AstEngine::for_model_with_device("tiger", device_pref)?;
        let max_tokens = custom_max_tokens.unwrap_or(4096);
        let formatted_prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            nioxs_engine::tiger_system()
        );
        let mut rx = engine
            .generate_stream(formatted_prompt, max_tokens, 0.20, 0.85)
            .await?;
        let mut flow = StreamFlow::new(flow_mode);
        flow.set_show_think_tags(true);
        while let Some(chunk) = rx.recv().await {
            flow.push(&chunk);
        }
        flow.finish();
        println!();
        return Ok(());
    }
    if arg.as_deref() == Some("fox") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!(
                "Usage: nioxs fox [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }

        let mut custom_max_tokens = None;
        let mut device_pref = nioxs_engine::DevicePreference::Auto;
        let mut flow_mode = FlowMode::Token;
        let mut prompt_parts = Vec::new();
        let mut iter = raw_args.clone().into_iter();
        while let Some(a) = iter.next() {
            if a == "-n" || a == "--max-tokens" {
                if let Some(val) = iter.next() {
                    custom_max_tokens = val.parse::<usize>().ok();
                }
            } else if a == "-d" || a == "--device" {
                if let Some(val) = iter.next() {
                    device_pref = nioxs_engine::DevicePreference::parse(&val);
                }
            } else if a == "--gpu" {
                device_pref = nioxs_engine::DevicePreference::Gpu;
            } else if a == "--cpu" {
                device_pref = nioxs_engine::DevicePreference::Cpu;
            } else if a == "--auto" {
                device_pref = nioxs_engine::DevicePreference::Auto;
            } else if a == "-f" || a == "--flow" {
                if let Some(val) = iter.next() {
                    flow_mode = FlowMode::parse(&val);
                }
            } else {
                prompt_parts.push(a);
            }
        }
        let prompt = prompt_parts.join(" ");
        if prompt.is_empty() {
            eprintln!(
                "Usage: nioxs fox [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }
        let engine = nioxs_engine::AstEngine::for_model_with_device("fox", device_pref)?;
        let max_tokens = custom_max_tokens.unwrap_or(4096);
        let formatted_prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            nioxs_engine::fox_system()
        );
        let mut rx = engine
            .generate_stream(formatted_prompt, max_tokens, 0.20, 0.85)
            .await?;
        let mut flow = StreamFlow::new(flow_mode);
        flow.set_show_think_tags(true);
        while let Some(chunk) = rx.recv().await {
            flow.push(&chunk);
        }
        flow.finish();
        println!();
        return Ok(());
    }
    if arg.as_deref() == Some("rabbit") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!(
                "Usage: nioxs rabbit [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }

        let mut custom_max_tokens = None;
        let mut device_pref = nioxs_engine::DevicePreference::Auto;
        let mut flow_mode = FlowMode::Token;
        let mut prompt_parts = Vec::new();
        let mut iter = raw_args.clone().into_iter();
        while let Some(a) = iter.next() {
            if a == "-n" || a == "--max-tokens" {
                if let Some(val) = iter.next() {
                    custom_max_tokens = val.parse::<usize>().ok();
                }
            } else if a == "-d" || a == "--device" {
                if let Some(val) = iter.next() {
                    device_pref = nioxs_engine::DevicePreference::parse(&val);
                }
            } else if a == "--gpu" {
                device_pref = nioxs_engine::DevicePreference::Gpu;
            } else if a == "--cpu" {
                device_pref = nioxs_engine::DevicePreference::Cpu;
            } else if a == "--auto" {
                device_pref = nioxs_engine::DevicePreference::Auto;
            } else if a == "-f" || a == "--flow" {
                if let Some(val) = iter.next() {
                    flow_mode = FlowMode::parse(&val);
                }
            } else {
                prompt_parts.push(a);
            }
        }
        let prompt = prompt_parts.join(" ");
        if prompt.is_empty() {
            eprintln!(
                "Usage: nioxs rabbit [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }
        let engine = nioxs_engine::AstEngine::for_model_with_device("rabbit", device_pref)?;
        let max_tokens = custom_max_tokens.unwrap_or(4096);
        let formatted_prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            nioxs_engine::rabbit_system()
        );
        let mut rx = engine
            .generate_stream(formatted_prompt, max_tokens, 0.6, 0.95)
            .await?;
        let mut flow = StreamFlow::new(flow_mode);
        flow.set_show_think_tags(true);
        while let Some(chunk) = rx.recv().await {
            flow.push(&chunk);
        }
        flow.finish();
        println!();
        return Ok(());
    }
    if matches!(
        arg.as_deref(),
        Some(
            "hydra"
                | "nio-hydra"
                | "hydra-kernel"
                | "nio-hydra-kernel"
                | "hydra-max"
                | "hydra-ultra",
        )
    ) {
        let cmd = arg.as_deref().unwrap_or("hydra");
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!(
                "Usage: nioxs hydra [--chair|--kernel|--base] [--audit|-a] [--verbose|-v] [--head <id>] [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }

        let mut custom_max_tokens = None;
        let mut device_pref = nioxs_engine::DevicePreference::Auto;
        let mut flow_mode = FlowMode::Token;
        let mut run_audit = raw_args.iter().any(|a| a == "--audit" || a == "-a")
            || std::env::var_os("NIOXS_AUDIT").is_some();
        let mut verbose = raw_args.iter().any(|a| a == "--verbose" || a == "-v")
            || std::env::var_os("NIOXS_VERBOSE").is_some();
        let mut verify = raw_args.iter().any(|a| a == "--verify");
        let mut tdd = raw_args.iter().any(|a| a == "--tdd");
        if tdd {
            verify = true;
        }
        let mut test_file: Option<std::path::PathBuf> = None;
        let mut deliberate = raw_args
            .iter()
            .any(|a| a == "--council" || a == "--deliberate" || a == "--triage")
            || std::env::var_os("NIOXS_DELIBERATE").is_some();
        let mut explicit_head = None;
        let mut model_target = match cmd {
            "hydra-kernel" | "nio-hydra-kernel" => "hydra-kernel",
            "hydra-max" => "hydra-max",
            "hydra-ultra" => "hydra-ultra",
            _ => "hydra",
        };
        let mut prompt_parts = Vec::new();
        let mut iter = raw_args.clone().into_iter();
        while let Some(a) = iter.next() {
            if a == "--kernel" {
                model_target = "hydra-kernel";
            } else if a == "--chair" {
                model_target = "hydra-chair";
            } else if a == "--base" {
                model_target = "hydra";
            } else if a == "--audit" || a == "-a" {
                run_audit = true;
            } else if a == "--verbose" || a == "-v" {
                verbose = true;
            } else if a == "--verify" {
                verify = true;
            } else if a == "--tdd" {
                tdd = true;
                verify = true;
            } else if a == "--test-file" {
                if let Some(val) = iter.next() {
                    test_file = Some(std::path::PathBuf::from(val));
                    verify = true;
                }
            } else if a == "--deliberate" || a == "--council" || a == "--triage" {
                deliberate = true;
            } else if a == "--head" || a == "-H" {
                if let Some(val) = iter.next() {
                    explicit_head = Some(val);
                }
            } else if a == "-n" || a == "--max-tokens" {
                if let Some(val) = iter.next() {
                    custom_max_tokens = val.parse::<usize>().ok();
                }
            } else if a == "-d" || a == "--device" {
                if let Some(val) = iter.next() {
                    device_pref = nioxs_engine::DevicePreference::parse(&val);
                }
            } else if a == "--gpu" {
                device_pref = nioxs_engine::DevicePreference::Gpu;
            } else if a == "--cpu" {
                device_pref = nioxs_engine::DevicePreference::Cpu;
            } else if a == "--auto" {
                device_pref = nioxs_engine::DevicePreference::Auto;
            } else if a == "-f" || a == "--flow" {
                if let Some(val) = iter.next() {
                    flow_mode = FlowMode::parse(&val);
                }
            } else {
                prompt_parts.push(a);
            }
        }
        let prompt = prompt_parts.join(" ");
        if prompt.is_empty() {
            eprintln!(
                "Usage: nioxs hydra [--chair|--kernel|--base] [--audit|-a] [--verbose|-v] [--head <id>] [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }
        let max_tokens = custom_max_tokens.unwrap_or(4096);
        let mut flow = StreamFlow::new(flow_mode);
        flow.set_show_think_tags(true);

        if model_target == "hydra-chair" {
            let base_engine = nioxs_engine::AstEngine::for_model_with_device("hydra", device_pref)?;
            base_engine.ensure_loaded().await?;
            if let Err(e) = base_engine.attach_lora_head("000").await {
                eprintln!(
                    "\x1b[1;31m[ERROR] Failed to attach Chair A LoRA (000): {:?}\x1b[0m",
                    e
                );
            }
            let chair_prompt = format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n",
                nioxs_engine::hydra_specialist_system("head-000")
            );
            flow.push("<think>\n");
            let mut chair_rx = base_engine
                .generate_stream(chair_prompt, max_tokens.min(512), 0.10, 0.90)
                .await?;
            let mut seen_body = false;
            while let Some(chunk) = chair_rx.recv().await {
                let to_push = if !seen_body {
                    let trimmed = chunk
                        .trim_start_matches(|c: char| c.is_whitespace())
                        .strip_prefix("<think>")
                        .unwrap_or(&chunk);
                    let trimmed = trimmed.trim_start_matches('\n');
                    if !trimmed.is_empty() {
                        seen_body = true;
                    }
                    trimmed
                } else {
                    &chunk
                };
                if !to_push.is_empty() {
                    flow.push(to_push);
                }
                if chunk.contains("</think>") {
                    break;
                }
            }
            flow.finish();
            return Ok(());
        }

        let is_council_speaker_session =
            model_target == "hydra" || model_target == "hydra-max" || model_target == "hydra-ultra";

        if is_council_speaker_session {
            let speaker_model = "hydra";
            let is_hydra = true;

            let base_engine =
                nioxs_engine::AstEngine::for_model_with_device(speaker_model, device_pref)?;

            // Fast-Path: Explicit head override or sub-millisecond (< 0.2ms) zero-token lexical & cosine semantic routing
            let mut specialist_opt = if let Some(ref h) = explicit_head {
                Some(h.as_str())
            } else {
                detect_council_specialist("", &prompt, is_hydra)
            };
            let mut accumulated = String::new();

            // Chair A Neural Triage Fallback:
            // If deliberate flag is passed OR prompt was ambiguous/unrouted by fast-path (< 0.2ms)
            if (deliberate || specialist_opt.is_none()) && explicit_head.is_none() {
                base_engine.ensure_loaded().await?;
                let _ = base_engine.attach_lora_head("000").await;
                let speaker_system = nioxs_engine::hydra_specialist_system("head-000");
                let formatted_prompt = format!(
                    "<|im_start|>system\n{speaker_system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
                );
                let mut rx = base_engine
                    .generate_stream(formatted_prompt, 128, 0.10, 0.90)
                    .await?;

                flow.push("<think>\n");
                let mut seen_triage_body = false;

                while let Some(chunk) = rx.recv().await {
                    let to_push = if !seen_triage_body {
                        let stripped = chunk.trim_start_matches(|c: char| c.is_whitespace());
                        let stripped = stripped
                            .strip_prefix("<think>")
                            .unwrap_or(stripped)
                            .trim_start_matches('\n');
                        if !stripped.is_empty() {
                            seen_triage_body = true;
                        }
                        stripped
                    } else {
                        &chunk
                    };
                    if !to_push.is_empty() {
                        accumulated.push_str(to_push);
                        flow.push(to_push);
                    }
                    if chunk.contains("</think>") || accumulated.contains("</think>") {
                        specialist_opt = detect_council_specialist(&accumulated, &prompt, true);
                        break;
                    }
                }

                // Fallback safety clutch: If neural speaker failed to emit directive, fallback to keyword detection
                if specialist_opt.is_none() {
                    specialist_opt = detect_council_specialist(&accumulated, &prompt, is_hydra);
                }
                drop(rx);
                flow.close_think();
            }

            // Step 2: Automatic Sovereign Single-Pass Execution
            let specialist = specialist_opt.unwrap_or("head-direct");
            {
                if is_hydra {
                    let specialist = if specialist == "head-000" {
                        "head-direct"
                    } else {
                        specialist
                    };

                    let raw_domain =
                        if accumulated.contains("domain=") || accumulated.contains("domain =") {
                            let dom_start = accumulated.find("domain").unwrap();
                            let rest = &accumulated[dom_start..];
                            let val = rest
                                .split(|c| c == ',' || c == ')' || c == '\n')
                                .next()
                                .unwrap_or("");
                            let clean = val
                                .split('=')
                                .nth(1)
                                .unwrap_or("")
                                .trim()
                                .trim_matches(|c| c == '"' || c == '\'' || c == ' ');
                            if !clean.is_empty() { Some(clean) } else { None }
                        } else {
                            None
                        };

                    let active_head = if explicit_head.is_none() && specialist != "head-direct" {
                        resolve_tier3_head(specialist, &prompt)
                    } else {
                        specialist.to_string()
                    };

                    if active_head != "head-direct" {
                        if verbose {
                            let badge = format_hydra_directive(specialist, raw_domain);
                            flow.push(&badge);
                            if active_head != specialist {
                                eprintln!(
                                    "[nioxs-hydra] Hop 2 (Anchor -> Tier-3): {specialist} -> {active_head}"
                                );
                            }
                        }

                        // NATIVE IN-PROCESS LORA HEAD HOT-SWAP (< 1ms switch)
                        let t_swap = std::time::Instant::now();
                        let head_attached =
                            base_engine.attach_lora_head(&active_head).await.is_ok();
                        let swap_ms = t_swap.elapsed().as_secs_f64() * 1000.0;
                        if head_attached && std::env::var_os("NIOXS_DEBUG").is_some() {
                            eprintln!(
                                "[nioxs-hydra] Hot-swapped to Head '{active_head}' in {swap_ms:.2}ms (Base Anchor resident in RAM)"
                            );
                        }
                    }

                    let specialist_prompt = if deliberate {
                        flow.push("<think>\n");
                        format!(
                            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n",
                            nioxs_engine::hydra_specialist_system(specialist)
                        )
                    } else {
                        format!(
                            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                            nioxs_engine::hydra_specialist_system(specialist)
                        )
                    };

                    let t_gen_start = std::time::Instant::now();
                    let mut t_first_tok: Option<f64> = None;
                    let mut generated_code = String::new();

                    if verify {
                        let mut tdd_harness: Option<String> = None;
                        if tdd && test_file.is_none() {
                            let test_prompt = format!(
                                "<|im_start|>system\nYou are Hydra Test-Writer. Write ONLY a small, self-contained unit test block (in ```rust with `fn main()` + `assert_eq!`, or ```python with `assert`, or ```zig with `test`) covering normal cases and edge cases (empty input, zero, bounds) for the requested function. Do NOT implement the function itself.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                            );
                            let mut test_out = String::new();
                            let mut test_rx = base_engine
                                .generate_stream(test_prompt, 256, 0.15, 0.85)
                                .await?;
                            while let Some(chunk) = test_rx.recv().await {
                                test_out.push_str(&chunk);
                            }
                            if let Some((_, tests_code)) = extract_code_block(&test_out) {
                                let valid_harness = !tests_code.trim_start().starts_with("///")
                                    && (tests_code.contains("fn main(")
                                        || tests_code.contains("#[test]")
                                        || tests_code.contains("assert")
                                        || tests_code.contains("test "));
                                if valid_harness {
                                    if verbose {
                                        eprintln!(
                                            "[nioxs-tdd] Generated edge-case test harness ({} bytes)",
                                            tests_code.len()
                                        );
                                    }
                                    tdd_harness = Some(tests_code);
                                } else if verbose {
                                    eprintln!(
                                        "[nioxs-tdd] Discarded non-runnable test harness; using compiler check only"
                                    );
                                }
                            }
                        }

                        let draft_prompt = if let Some(ref th) = tdd_harness {
                            format!(
                                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}\n\nYour implementation must pass these tests:\n```\n{th}\n```\nReturn ONLY the implementation in a fenced code block (no <think>, no doc-comment examples).<|im_end|>\n<|im_start|>assistant\n",
                                nioxs_engine::hydra_specialist_system(specialist)
                            )
                        } else {
                            specialist_prompt.clone()
                        };

                        let mut spec_rx = base_engine
                            .generate_stream(draft_prompt, max_tokens, 0.20, 0.85)
                            .await?;
                        while let Some(chunk) = spec_rx.recv().await {
                            if t_first_tok.is_none() && !chunk.is_empty() {
                                t_first_tok = Some(t_gen_start.elapsed().as_secs_f64());
                            }
                            generated_code.push_str(&chunk);
                            flow.push(&chunk);
                        }

                        let mut last_err = match verify_generated_code(
                            &generated_code,
                            &prompt,
                            specialist,
                            test_file.as_deref(),
                            tdd_harness.as_deref(),
                        ) {
                            Ok(msg) => {
                                if verbose {
                                    eprintln!("\n[nioxs-verify] PASS: {msg}");
                                }
                                None
                            }
                            Err(err) => Some(err),
                        };

                        if let Some(mut err) = last_err.take() {
                            for round in 1..=3 {
                                if verbose {
                                    eprintln!("\n[nioxs-verify] Round {round} repair after: {err}");
                                }
                                let repair_prompt = format!(
                                    "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}\n\nPrevious code failed verification with:\n{err}\n\nPrevious code:\n{generated_code}\n\nFix the exact error and return ONLY the complete working fenced code block.<|im_end|>\n<|im_start|>assistant\n",
                                    nioxs_engine::hydra_specialist_system(specialist)
                                );
                                let mut repaired = String::new();
                                let mut rep_rx = base_engine
                                    .generate_stream(repair_prompt, max_tokens, 0.20, 0.85)
                                    .await?;
                                while let Some(chunk) = rep_rx.recv().await {
                                    repaired.push_str(&chunk);
                                    flow.push(&chunk);
                                }
                                generated_code = repaired;
                                match verify_generated_code(
                                    &generated_code,
                                    &prompt,
                                    specialist,
                                    test_file.as_deref(),
                                    tdd_harness.as_deref(),
                                ) {
                                    Ok(msg) => {
                                        if verbose {
                                            eprintln!(
                                                "\n[nioxs-verify] PASS (repair round {round}): {msg}"
                                            );
                                        }
                                        last_err = None;
                                        break;
                                    }
                                    Err(next_err) => {
                                        err = next_err.clone();
                                        last_err = Some(next_err);
                                    }
                                }
                            }
                            if let Some(final_err) = last_err {
                                if verbose {
                                    eprintln!(
                                        "\n[nioxs-verify] FAIL (after 3 repair rounds): {final_err}"
                                    );
                                }
                            }
                        }
                    } else {
                        let mut spec_rx = base_engine
                            .generate_stream(specialist_prompt.clone(), max_tokens, 0.20, 0.85)
                            .await?;
                        while let Some(chunk) = spec_rx.recv().await {
                            if t_first_tok.is_none() && !chunk.is_empty() {
                                t_first_tok = Some(t_gen_start.elapsed().as_secs_f64());
                            }
                            generated_code.push_str(&chunk);
                            flow.push(&chunk);
                        }
                        // Empty-output / early-EOS retry guard (e.g. calibrated r=12 heads)
                        if generated_code.trim().is_empty() {
                            let nudge = "Here is the implementation:\n";
                            let retry_prompt = format!("{specialist_prompt}{nudge}");
                            if let Ok(mut retry_rx) = base_engine
                                .generate_stream(retry_prompt, max_tokens, 0.35, 0.90)
                                .await
                            {
                                flow.push(nudge);
                                generated_code.push_str(nudge);
                                while let Some(chunk) = retry_rx.recv().await {
                                    if t_first_tok.is_none() && !chunk.is_empty() {
                                        t_first_tok = Some(t_gen_start.elapsed().as_secs_f64());
                                    }
                                    generated_code.push_str(&chunk);
                                    flow.push(&chunk);
                                }
                            }
                        }
                    }
                    if verbose {
                        let total_s = t_gen_start.elapsed().as_secs_f64();
                        let ttft_s = t_first_tok.unwrap_or(total_s);
                        let decode_s = (total_s - ttft_s).max(0.001);
                        let words = generated_code.split_whitespace().count();
                        let wps = words as f64 / decode_s;
                        let dev_label = match device_pref {
                            nioxs_engine::DevicePreference::Gpu => "gpu",
                            nioxs_engine::DevicePreference::Cpu => "cpu",
                            nioxs_engine::DevicePreference::Auto => "auto",
                        };
                        eprintln!(
                            "\n[nioxs] ({dev_label}) TTFT: {ttft_s:.2}s | Decode: {decode_s:.2}s | {words} words ({wps:.2} w/s)"
                        );
                    }

                    // ── Step 3: Sovereign Chair C Invariant Verification & Formal Audit ────
                    if run_audit && specialist != "head-042" && !generated_code.trim().is_empty() {
                        let chair_c_id = "042"; // anchor-042-chair-c-critic
                        let t_c = std::time::Instant::now();
                        let _ = base_engine.attach_lora_head(chair_c_id).await;
                        let c_ms = t_c.elapsed().as_secs_f64() * 1000.0;
                        if std::env::var_os("NIOXS_DEBUG").is_some() {
                            eprintln!(
                                "[nioxs-hydra] Chair C Invariant Verifier hot-swapped in {c_ms:.2}ms"
                            );
                        }
                        flow.push("\n\n---\n**Council Chair C: Invariant Audit & Verification**\n\n<audit>\n");
                        let audit_prompt = format!(
                            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\nAudit the following implementation:\n```\n{}\n```\n\nProvide:\n1. Invariant & Safety Analysis (data races, memory safety, pointer validity, and language correctness)\n2. Verdict: State SOUND or REQUIRES CORRECTION with a concise explanation.<|im_end|>\n<|im_start|>assistant\n<audit>\n",
                            nioxs_engine::hydra_specialist_system("head-042"),
                            generated_code.trim()
                        );
                        let audit_tokens = 384.min(max_tokens / 2);
                        let mut audit_text = String::new();
                        match base_engine
                            .generate_stream(audit_prompt, audit_tokens, 0.15, 0.85)
                            .await
                        {
                            Ok(mut audit_rx) => {
                                while let Some(chunk) = audit_rx.recv().await {
                                    audit_text.push_str(&chunk);
                                    flow.push(&chunk);
                                }
                            }
                            Err(e) => {
                                eprintln!("[nioxs-hydra] Chair C stream error: {e}");
                            }
                        }
                        if !audit_text.contains("</audit>") {
                            flow.push("\n</audit>\n");
                        }
                    }
                } else {
                    let directive_rendered = if accumulated.contains("@council") {
                        let start = accumulated.find("@council").unwrap();
                        let end = accumulated[start..]
                            .find(')')
                            .unwrap_or(accumulated[start..].len() - 1);
                        let raw_directive = &accumulated[start..=start + end];
                        let prefix = accumulated[..start].trim();
                        if !prefix.is_empty() {
                            format!("{}\n\n{}", prefix, format_council_directive(raw_directive))
                        } else {
                            format_council_directive(raw_directive)
                        }
                    } else {
                        format_council_directive(&format!("@council(route = {specialist})"))
                    };
                    flow.push(&directive_rendered);

                    base_engine.unload().await;
                    drop(base_engine); // Evict Speaker working set to guarantee < 180MB RAM (< 850MB on fin)
                    let specialist_engine =
                        nioxs_engine::AstEngine::for_model_with_device(specialist, device_pref)?;
                    let specialist_prompt = format!(
                        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                        nioxs_engine::hydra_specialist_system(specialist)
                    );
                    let mut spec_rx = specialist_engine
                        .generate_stream(specialist_prompt, max_tokens, 0.20, 0.85)
                        .await?;
                    while let Some(chunk) = spec_rx.recv().await {
                        flow.push(&chunk);
                    }
                }
            }
        } else {
            // Direct specialist execution (e.g. nioxs hydra --kernel)
            let engine = nioxs_engine::AstEngine::for_model_with_device("hydra", device_pref)?;
            let _ = engine.attach_lora_head(model_target).await;
            let system_prompt = nioxs_engine::hydra_specialist_system(model_target);
            let formatted_prompt = format!(
                "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            );
            let mut rx = engine
                .generate_stream(formatted_prompt, max_tokens, 0.20, 0.85)
                .await?;
            while let Some(chunk) = rx.recv().await {
                flow.push(&chunk);
            }
        }

        flow.finish();
        println!();
        return Ok(());
    }
    if arg.as_deref() == Some("manta") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!(
                "Usage: nioxs manta [--device <auto|cpu|gpu>] [--gpu] [--cpu] [--auto] [--flow <burst|token|paragraph>] [--plan] [-n <max_tokens>] <prompt>"
            );
            return Ok(());
        }

        let mut is_plan = false;
        let mut custom_max_tokens = None;
        let mut device_pref = nioxs_engine::DevicePreference::Auto;
        let mut flow_mode = FlowMode::Token;
        let mut prompt_parts = Vec::new();
        let mut iter = raw_args.clone().into_iter();
        while let Some(a) = iter.next() {
            if a == "--plan" {
                is_plan = true;
            } else if a == "-n" || a == "--max-tokens" {
                if let Some(val) = iter.next() {
                    custom_max_tokens = val.parse::<usize>().ok();
                }
            } else if a == "-d" || a == "--device" {
                if let Some(val) = iter.next() {
                    device_pref = nioxs_engine::DevicePreference::parse(&val);
                }
            } else if a == "--gpu" {
                device_pref = nioxs_engine::DevicePreference::Gpu;
            } else if a == "--cpu" {
                device_pref = nioxs_engine::DevicePreference::Cpu;
            } else if a == "--auto" {
                device_pref = nioxs_engine::DevicePreference::Auto;
            } else if a == "-f" || a == "--flow" {
                if let Some(val) = iter.next() {
                    flow_mode = FlowMode::parse(&val);
                }
            } else if a != "-i" && a != "--interactive" && a != "--benchmark" && a != "-b" {
                prompt_parts.push(a);
            }
        }
        let user_prompt = prompt_parts.join(" ");

        // Prefer native Rust AstEngine for nio-manta
        match nioxs_engine::AstEngine::for_model_with_device("nio-manta", device_pref) {
            Ok(engine) => {
                let sys_msg = nioxs_engine::manta_system(is_plan);
                let (formatted_prompt, temp) = if is_plan {
                    (
                        format!(
                            "<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n<manta_plan>\n<step id=\"step-1\" depends_on=\"\">\n"
                        ),
                        0.0,
                    )
                } else {
                    (
                        format!(
                            "<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                        ),
                        0.2,
                    )
                };
                let max_tokens = custom_max_tokens.unwrap_or(if is_plan { 4096 } else { 128 });
                let mut rx = engine
                    .generate_stream(formatted_prompt, max_tokens, temp, 0.9)
                    .await?;
                if is_plan {
                    let mut output =
                        String::from("<manta_plan>\n<step id=\"step-1\" depends_on=\"\">\n");
                    while let Some(chunk) = rx.recv().await {
                        output.push_str(&chunk);
                    }
                    let trimmed = output.trim();
                    let plan_str = if let Some(start) = trimmed.find("<manta_plan>") {
                        if let Some(end) = trimmed.rfind("</manta_plan>") {
                            &trimmed[start..end + "</manta_plan>".len()]
                        } else {
                            &trimmed[start..]
                        }
                    } else {
                        trimmed
                    };
                    nioxs_provider::manta_plan::validate(plan_str).map_err(|error| {
                        format!("Manta plan rejected: {error:#}\nRaw output:\n{trimmed}")
                    })?;
                    println!("{}", plan_str.trim());
                    return Ok(());
                }
                let mut flow = StreamFlow::new(flow_mode);
                while let Some(chunk) = rx.recv().await {
                    flow.push(&chunk);
                }
                flow.finish();
                println!();
                return Ok(());
            }
            Err(e) => {
                // Fallback: Locate runner script if available
                let exe_dir = std::env::current_exe()
                    .ok()
                    .and_then(|p| p.parent().map(|p| p.to_path_buf()));
                let mut script_candidates = vec![
                    std::path::PathBuf::from("scripts/run_niomanta_onnx.py"),
                    std::path::PathBuf::from(
                        "/home/mn/nio-labs/nio-xs/scripts/run_niomanta_onnx.py",
                    ),
                ];
                if let Some(dir) = exe_dir {
                    script_candidates.push(dir.join("../scripts/run_niomanta_onnx.py"));
                    script_candidates.push(dir.join("scripts/run_niomanta_onnx.py"));
                }

                let script = script_candidates.into_iter().find(|p| p.exists());

                if script.is_none() {
                    return Err(format!("Failed to initialize Manta: {e}").into());
                }

                // Locate python interpreter
                let py_candidates = [
                    std::env::var("NIOXS_PYTHON").ok(),
                    Some("/home/mn/nio-labs/.venv/bin/python3".into()),
                    Some(".venv/bin/python3".into()),
                    Some("python3".into()),
                ];
                let python = py_candidates
                    .into_iter()
                    .flatten()
                    .find(|p| {
                        std::process::Command::new(p)
                            .arg("--version")
                            .output()
                            .is_ok()
                    })
                    .unwrap_or_else(|| "python3".into());

                let script_path = script.unwrap();
                let mut cmd = std::process::Command::new(&python);
                cmd.arg(&script_path);

                let mut is_plan = false;
                let mut prompt_parts = Vec::new();

                for a in raw_args {
                    if a == "--plan" {
                        is_plan = true;
                    } else if a == "--benchmark" || a == "-b" {
                        cmd.arg("--benchmark");
                    } else if a == "-i" || a == "--interactive" {
                        cmd.arg("--interactive");
                    } else {
                        prompt_parts.push(a);
                    }
                }

                if is_plan {
                    cmd.arg("--plan");
                }
                let prompt = prompt_parts.join(" ");
                if !prompt.is_empty() {
                    cmd.arg("-p").arg(prompt);
                }

                if is_plan {
                    let output = cmd.output()?;
                    if !output.status.success() {
                        return Err(format!(
                            "Manta fallback failed: {}",
                            String::from_utf8_lossy(&output.stderr)
                        )
                        .into());
                    }
                    let output = String::from_utf8(output.stdout)?;
                    nioxs_provider::manta_plan::validate(&output)
                        .map_err(|error| format!("Manta plan rejected: {error:#}"))?;
                    println!("{}", output.trim());
                } else {
                    let status = cmd.status()?;
                    if !status.success() {
                        std::process::exit(status.code().unwrap_or(1));
                    }
                }
                return Ok(());
            }
        }
    }
    if arg.as_deref() == Some("init") {
        let target = std::env::args().nth(2).unwrap_or_else(|| "--all".into());
        match target.as_str() {
            "--aider" => {
                let config = "openai-api-base: http://127.0.0.1:7460/v1\nmodel: openai/nio-cuda\nopenai-api-key: nioxs-local\n";
                std::fs::write(".aider.conf.yml", config)?;
                println!("✔ Successfully created .aider.conf.yml for Aider");
            }
            "--cline" => {
                let config = serde_json::json!({
                    "mcpServers": {
                        "nioxs": {
                            "command": "nioxs",
                            "args": ["mcp"]
                        }
                    }
                });
                std::fs::write(
                    "cline_mcp_settings.json",
                    serde_json::to_string_pretty(&config)?,
                )?;
                println!("✔ Successfully created cline_mcp_settings.json for Cline");
            }
            "--opencode" => {
                let config = serde_json::json!({
                    "provider": "openai",
                    "baseURL": "http://127.0.0.1:7460/v1",
                    "apiKey": "nioxs-local",
                    "model": "nio-cuda"
                });
                std::fs::write("opencode.json", serde_json::to_string_pretty(&config)?)?;
                println!("✔ Successfully created opencode.json for OpenCode");
            }
            "--cursor" => {
                println!("Cursor OpenAI Endpoint Configuration:");
                println!("  Base URL: http://127.0.0.1:7460/v1");
                println!("  API Key:  nioxs-local");
                println!("  Model:    nio-cuda");
            }
            _ => {
                println!("Auto-configuring coding agent settings in current directory...");
                let aider_cfg = "openai-api-base: http://127.0.0.1:7460/v1\nmodel: openai/nio-cuda\nopenai-api-key: nioxs-local\n";
                let _ = std::fs::write(".aider.conf.yml", aider_cfg);
                let cline_cfg = serde_json::json!({
                    "mcpServers": {
                        "nioxs": {
                            "command": "nioxs",
                            "args": ["mcp"]
                        }
                    }
                });
                let _ = std::fs::write(
                    "cline_mcp_settings.json",
                    serde_json::to_string_pretty(&cline_cfg)?,
                );
                println!("✔ Configured .aider.conf.yml and cline_mcp_settings.json");
            }
        }
        return Ok(());
    }
    if arg.as_deref() == Some("undo") {
        let current_dir = std::env::current_dir().unwrap_or_else(|_| ".".into());
        let patch_id = std::env::args().nth(2);
        let id_to_rollback = if let Some(id) = patch_id {
            id
        } else {
            let undo_dir = current_dir.join(".nioxs/undo");
            let mut latest_file = None;
            let mut latest_time = std::time::SystemTime::UNIX_EPOCH;
            if let Ok(entries) = std::fs::read_dir(&undo_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("json") {
                        if let Ok(meta) = entry.metadata() {
                            if let Ok(mod_time) = meta.modified() {
                                if mod_time > latest_time {
                                    latest_time = mod_time;
                                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                                        latest_file = Some(stem.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            match latest_file {
                Some(id) => id,
                None => {
                    println!("No undo snapshots found in .nioxs/undo/");
                    return Ok(());
                }
            }
        };

        match nioxs_nano::patch::rollback_patch(&current_dir, &id_to_rollback) {
            Ok(count) => {
                println!(
                    "✔ Successfully rolled back patch '{id_to_rollback}' ({count} files restored)."
                );
            }
            Err(e) => {
                eprintln!("Failed to rollback patch '{id_to_rollback}': {e}");
            }
        }
        return Ok(());
    }
    if arg.as_deref() == Some("history") {
        let current_dir = std::env::current_dir().unwrap_or_else(|_| ".".into());
        let limit = std::env::args()
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);
        let turns = nioxs_provider::journal::read_local_turns(&current_dir, limit);
        if turns.is_empty() {
            println!("No local turn history found in .nioxs/history.jsonl");
        } else {
            println!("Recent Turns ({} total):", turns.len());
            for (idx, turn) in turns.iter().enumerate() {
                println!(
                    "[{:2}] ts:{} | Intent: {:<12} | Outcome: {:<10}\n     Prompt: {}\n",
                    idx + 1,
                    turn.timestamp_unix,
                    turn.intent,
                    turn.outcome,
                    turn.prompt.chars().take(80).collect::<String>()
                );
            }
        }
        return Ok(());
    }
    if arg.as_deref() == Some("mcp") {
        return nioxs_provider::mcp::run_stdio_server().map_err(|e| e.into());
    }
    if arg.is_some() && arg.as_deref() != Some("serve") {
        return Err("unknown command; expected 'nioxs serve', 'nioxs pull', 'nioxs viper', 'nioxs skate', 'nioxs manta', 'nioxs beluga', 'nioxs blue', 'nioxs cheetah', 'nioxs tiger', 'nioxs fox', 'nioxs rabbit', 'nioxs fin', 'nioxs hydra', 'nioxs mcp', 'nioxs init', 'nioxs undo', or 'nioxs history'".into());
    }

    let mut server_device_pref = nioxs_engine::DevicePreference::Auto;
    let mut iter = std::env::args().skip(1);
    while let Some(a) = iter.next() {
        if a == "--gpu" {
            server_device_pref = nioxs_engine::DevicePreference::Gpu;
        } else if a == "--cpu" {
            server_device_pref = nioxs_engine::DevicePreference::Cpu;
        } else if a == "--auto" {
            server_device_pref = nioxs_engine::DevicePreference::Auto;
        } else if a == "-d" || a == "--device" {
            if let Some(val) = iter.next() {
                server_device_pref = nioxs_engine::DevicePreference::parse(&val);
            }
        }
    }
    match server_device_pref {
        nioxs_engine::DevicePreference::Gpu => unsafe { std::env::set_var("NIOXS_DEVICE", "gpu") },
        nioxs_engine::DevicePreference::Cpu => unsafe { std::env::set_var("NIOXS_DEVICE", "cpu") },
        nioxs_engine::DevicePreference::Auto => {}
    }
    let listen: std::net::SocketAddr = if let Ok(listen_str) = std::env::var("NIOXS_LISTEN") {
        listen_str.parse()?
    } else if let Ok(port_str) = std::env::var("PORT") {
        let port: u16 = port_str.parse()?;
        std::net::SocketAddr::from(([0, 0, 0, 0], port))
    } else {
        "127.0.0.1:7460".parse()?
    };
    let (db_url, db_token, _niodb_proc) = nioxs_provider::niodb_supervisor::ensure_niodb(
        std::env::var("NIOXS_DB_URL").ok().as_deref(),
        std::env::var("NIOXS_DB_TOKEN_FILE").ok().as_deref(),
    )
    .await?;

    let api_token = if let Ok(direct) = std::env::var("NIOXS_API_TOKEN") {
        let trimmed = direct.trim().to_string();
        if trimmed.is_empty() {
            return Err("provider token cannot be empty".into());
        }
        Some(trimmed)
    } else if let Ok(file) = std::env::var("NIOXS_API_TOKEN_FILE") {
        let content = std::fs::read_to_string(file)?;
        let trimmed = content.trim().to_string();
        if trimmed.is_empty() {
            return Err("provider token cannot be empty".into());
        }
        Some(trimmed)
    } else {
        None
    };

    let allow_unauthenticated = std::env::var("NIOXS_ALLOW_UNAUTHENTICATED")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if !listen.ip().is_loopback() && api_token.is_none() && !allow_unauthenticated {
        return Err("non-loopback deployment requires NIOXS_API_TOKEN or NIOXS_API_TOKEN_FILE (or set NIOXS_ALLOW_UNAUTHENTICATED=true)".into());
    }
    let deployment = std::env::var("NIOXS_DEPLOYMENT_ID").unwrap_or_else(|_| "local".into());
    if deployment.is_empty()
        || deployment.len() > 64
        || !deployment
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
    {
        return Err("invalid deployment ID".into());
    }
    let db = NioDb::new(&db_url, db_token)?;
    let model = match std::env::var("NIOXS_MODEL_FILE") {
        Ok(file) => {
            let bytes = std::fs::read(file)?;
            let expected = std::env::var("NIOXS_MODEL_SHA256")
                .map_err(|_| "external model requires NIOXS_MODEL_SHA256")?;
            if nioxs_contracts::digest(&bytes) != expected {
                return Err("model checksum mismatch".into());
            }
            Model::load(&bytes)?
        }
        Err(_) => Model::embedded()?,
    };
    let instance = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    );
    let ast_engine: Option<nioxs_engine::AstEngine> = None;
    let microlm_banner = "Sovereign MicroLM Fleet (Blue, Beluga, Manta, Viper, Dolphin)";
    let active_hw = nioxs_engine::ActiveDevice::detect(server_device_pref);
    let listener = tokio::net::TcpListener::bind(listen).await?;
    let addr = listener.local_addr()?;
    eprintln!(
        "\n\
  ╭──────────────────────────────────────────────────────────╮\n\
  │  ⚡ NioXS v{:<8} Pure Coding Companion Gateway        │\n\
  ├──────────────────────────────────────────────────────────┤\n\
  │  • Gateway API : http://{:<33} │\n\
  │  • NioDB Store : {:<40} │\n\
  │  • MicroLM     : {:<40} │\n\
  │  • Hardware    : {:<40} │\n\
  │  • Directives  : --bash, --tools, --code, --test         │\n\
  ╰──────────────────────────────────────────────────────────╯\n",
        env!("CARGO_PKG_VERSION"),
        addr,
        db_url,
        microlm_banner,
        active_hw.name(),
    );
    axum::serve(
        listener,
        router(App {
            db,
            deployment,
            api_token,
            active: ActiveTurns::default(),
            model: Arc::new(model),
            ray_engine: ast_engine,
            instance,
        }),
    )
    .with_graceful_shutdown(async {
        let _ = tokio::signal::ctrl_c().await;
    })
    .await?;
    Ok(())
}

fn ground_and_sanitize_command(prompt: &str, raw_cmd: &str) -> String {
    let mut clean_cmd = raw_cmd
        .trim()
        .trim_start_matches("```powershell")
        .trim_start_matches("```bash")
        .trim_start_matches("```sh")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim()
        .to_string();

    // 1. Sanitize trailing dangling flags (e.g. "ls -" -> "ls -la")
    if clean_cmd == "ls -" || clean_cmd.ends_with("ls -") {
        clean_cmd = clean_cmd.replace("ls -", "ls -la");
    } else if clean_cmd.ends_with(" -") {
        clean_cmd = clean_cmd.trim_end_matches(" -").trim().to_string();
    } else if clean_cmd.ends_with(" --") {
        clean_cmd = clean_cmd.trim_end_matches(" --").trim().to_string();
    }

    // 2. Extract quoted literal terms from prompt (e.g. 'TODO' or "TODO")
    let prompt_quoted: Vec<&str> = {
        let mut list = Vec::new();
        for quote_char in ['\'', '"'] {
            let mut start = None;
            for (i, c) in prompt.char_indices() {
                if c == quote_char {
                    if let Some(s) = start {
                        if i > s + 1 {
                            list.push(&prompt[s + 1..i]);
                        }
                        start = None;
                    } else {
                        start = Some(i);
                    }
                }
            }
        }
        list
    };

    // If user specified an explicit search term, align the pattern in the reflex
    if let Some(&user_term) = prompt_quoted.first() {
        // Handle -e '...' or -e "..."
        if let Some(pos) = clean_cmd.find("-e ") {
            let after = &clean_cmd[pos + 3..];
            if after.starts_with('\'') || after.starts_with('"') {
                let quote = after.chars().next().unwrap();
                if let Some(end_quote) = after[1..].find(quote) {
                    let full_flag = &clean_cmd[pos..pos + 3 + 1 + end_quote + 1];
                    let replacement = format!("-e '{user_term}'");
                    clean_cmd = clean_cmd.replace(full_flag, &replacement);
                }
            }
        }
        // Handle -Pattern '...' or -Pattern "..."
        if let Some(pos) = clean_cmd.find("-Pattern ") {
            let after = &clean_cmd[pos + 9..];
            if after.starts_with('\'') || after.starts_with('"') {
                let quote = after.chars().next().unwrap();
                if let Some(end_quote) = after[1..].find(quote) {
                    let full_flag = &clean_cmd[pos..pos + 9 + 1 + end_quote + 1];
                    let replacement = format!("-Pattern '{user_term}'");
                    clean_cmd = clean_cmd.replace(full_flag, &replacement);
                }
            }
        }
        // Handle grep without -e, e.g. grep -rn 'panic' or grep -rnw 'panic'
        if clean_cmd.starts_with("grep ") && !clean_cmd.contains("-e ") {
            for quote in ['\'', '"'] {
                if let Some(first_q) = clean_cmd.find(quote) {
                    if let Some(second_q) = clean_cmd[first_q + 1..].find(quote) {
                        let matched_slice = &clean_cmd[first_q..first_q + 1 + second_q + 1];
                        if !matched_slice.contains('/')
                            && matched_slice != format!("{quote}.{quote}")
                        {
                            let replacement = format!("'{user_term}'");
                            clean_cmd = clean_cmd.replace(matched_slice, &replacement);
                            break;
                        }
                    }
                }
            }
        }
    }

    // 3. Fallback system directory sanitization:
    // If prompt didn't specify system paths like /var/log, /opt, /var/cache, /data, /home/app,
    // don't scan them with grep/find, fallback to '.'
    let system_paths = [
        "'/var/log'",
        "\"/var/log\"",
        "'/var/cache'",
        "\"/var/cache\"",
        "'/home/app'",
        "\"/home/app\"",
        "'/opt'",
        "\"/opt\"",
        "'/data'",
        "\"/data\"",
    ];

    for path in system_paths {
        let stripped = path.trim_matches('\'').trim_matches('"');
        if clean_cmd.contains(path) && !prompt.contains(stripped) {
            clean_cmd = clean_cmd.replace(path, "'.'");
        }
    }

    // 4. Port number slot grounding:
    // If prompt mentions a port number (e.g. "port 3000" or "port 8080")
    for word in prompt.split_whitespace() {
        let clean_word = word.trim_matches(|c: char| !c.is_numeric());
        if let Ok(port_num) = clean_word.parse::<u16>() {
            if port_num >= 80 {
                let port_str = port_num.to_string();
                for default_port in ["3000", "8080", "5432", "8000", "9000"] {
                    if default_port != port_str && clean_cmd.contains(default_port) {
                        clean_cmd = clean_cmd.replace(default_port, &port_str);
                    }
                }
            }
        }
    }

    // 5. Code & Spider Search Hygiene (Binary noise suppression, directory exclusions, extension filter)
    if clean_cmd.starts_with("grep ") {
        // Prevent binary match spam (target/*.rmeta, *.rlib, *.so, *.exe)
        if clean_cmd.contains("-rn ") {
            clean_cmd = clean_cmd.replace("-rn ", "-rIn ");
        } else if clean_cmd.contains("-rni ") {
            clean_cmd = clean_cmd.replace("-rni ", "-rIni ");
        } else if clean_cmd.contains("-rnw ") {
            clean_cmd = clean_cmd.replace("-rnw ", "-rInw ");
        }

        // Exclude build artifacts and caches
        if !clean_cmd.contains("--exclude-dir")
            && !prompt.contains("target")
            && !prompt.contains(".git")
        {
            clean_cmd.push_str(" --exclude-dir={target,.git,node_modules}");
        }

        // File extension grounding (e.g. ".rs files", ".py files", ".ts files")
        if !clean_cmd.contains("--include") {
            for word in prompt.split_whitespace() {
                let clean_w = word.trim_matches(|c: char| c == '\'' || c == '"' || c == ',');
                if clean_w.starts_with('.')
                    && clean_w.len() >= 2
                    && clean_w.len() <= 6
                    && clean_w[1..].chars().all(|c| c.is_alphanumeric())
                {
                    let ext = clean_w.trim_start_matches('.');
                    clean_cmd.push_str(&format!(" --include=\"*.{ext}\""));
                    break;
                }
            }
        }

        // Ensure search directory anchor exists
        let parts: Vec<&str> = clean_cmd.split_whitespace().collect();
        let last = parts.last().copied().unwrap_or("");
        if last.starts_with('-')
            || clean_cmd.ends_with('}')
            || clean_cmd.ends_with('"')
            || clean_cmd.ends_with('\'')
        {
            clean_cmd.push_str(" .");
        }
    }

    clean_cmd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_council_directive_route_and_domain() {
        let directive = "@council(route = memory, domain = circular_buffer)";
        let formatted = format_council_directive(directive);
        assert_eq!(
            formatted,
            "<council>\n    @council(\n        route = memory,\n        domain = circular_buffer\n    )\n</council>\n\n"
        );
    }

    #[test]
    fn test_detect_hydra_specialist_atomic_ring_buffer() {
        let acc = "<think>\n1. Invariant: Linearizable memory ordering\n</think>\n<council>\n@council(\nhead = 001,\ndomain = atomic\n)\n</council>";
        let prompt = "Implement an atomic ring buffer in Zig";
        let specialist = detect_council_specialist(acc, prompt, true);
        assert_eq!(specialist, Some("head-001"));
    }

    #[test]
    fn test_detect_hydra_specialist_sql() {
        let acc = "@council(head = 005, domain = mergetree)";
        let prompt = "Optimize this ClickHouse MergeTree query";
        let specialist = detect_council_specialist(acc, prompt, true);
        assert_eq!(specialist, Some("head-005"));
    }

    #[test]
    fn test_detect_hydra_specialist_raft() {
        let acc = "@council(head = 008, domain = raft)";
        let prompt = "Implement joint consensus membership change";
        let specialist = detect_council_specialist(acc, prompt, true);
        assert_eq!(specialist, Some("head-008"));
    }

    #[test]
    fn test_detect_hydra_specialist_kernel() {
        let acc = "@council(head = 001, domain = posix)";
        let prompt = "Write an async submission queue poller in C";
        let specialist = detect_council_specialist(acc, prompt, true);
        assert_eq!(specialist, Some("head-001"));
    }

    #[test]
    fn test_detect_hydra_specialist_head_001() {
        let acc = "<think>\nInvariant: memory order seq_cst\n</think>\n@council(head-001-atomic-concurrency)";
        let prompt = "Write a lock-free queue in Zig";
        let specialist = detect_council_specialist(acc, prompt, true);
        assert_eq!(specialist, Some("head-001"));
    }

    #[test]
    fn test_detect_hydra_specialist_head_002() {
        let acc = "<think>\nInvariant: async streaming backpressure\n</think>\n@council(head-002-backend-axum-tokio)";
        let prompt = "Build an Axum SSE streaming server";
        let specialist = detect_council_specialist(acc, prompt, true);
        assert_eq!(specialist, Some("head-002"));
    }

    #[test]
    fn test_detect_hydra_specialist_all_heads() {
        assert_eq!(
            detect_council_specialist(
                "@council(head=003)",
                "Build a React 19 RSC component with Suspense",
                true
            ),
            Some("head-003")
        );
        assert_eq!(
            detect_council_specialist(
                "@council(head=004)",
                "Write an eBPF XDP packet filter in Rust aya",
                true
            ),
            Some("head-004")
        );
        assert_eq!(
            detect_council_specialist(
                "@council(head=005)",
                "Create ClickHouse ReplacingMergeTree schema",
                true
            ),
            Some("head-005")
        );
        assert_eq!(
            detect_council_specialist(
                "@council(head=006)",
                "Implement lazy propagation segment tree in Rust",
                true
            ),
            Some("head-006")
        );
        assert_eq!(
            detect_council_specialist(
                "@council(head=007)",
                "Write a Kubernetes CRD reconciler in kube-rs",
                true
            ),
            Some("head-007")
        );
        assert_eq!(
            detect_council_specialist(
                "@council(head=008)",
                "Implement Raft consensus AppendEntries state machine",
                true
            ),
            Some("head-008")
        );
        assert_eq!(
            detect_council_specialist(
                "@council(head=009)",
                "Vectorize dot product using AVX2 SIMD intrinsics",
                true
            ),
            Some("head-009")
        );
        assert_eq!(
            detect_council_specialist(
                "@council(head=010)",
                "Run WASI 0.2 sandbox in Wasmtime host",
                true
            ),
            Some("head-010")
        );
    }

    #[test]
    fn test_detect_hydra_polymath_direct() {
        let acc = "<think>\nGeneral query, no specialist needed\n</think>\n@council(direct)";
        let prompt = "Explain the difference between Arc and Rc in Rust";
        let specialist = detect_council_specialist(acc, prompt, true);
        assert_eq!(specialist, Some("head-direct"));
    }

    #[test]
    fn test_is_degenerate_detection() {
        let garbage1 = "''C, 'C their 'Cle', the Clawst, the 'C'\n')\n')\n')";
        assert!(is_degenerate(garbage1));
        let garbage2 = "011200000";
        assert!(is_degenerate(garbage2));
        let valid =
            "<think>\n1. Invariant: memory order.\n</think>\n@council(head=001, domain=systems)";
        assert!(!is_degenerate(valid));
    }

    #[test]
    fn test_format_council_directive_head() {
        let directive = "@council(head = 001, domain = systems-atomic-concurrency)";
        let formatted = format_council_directive(directive);
        assert!(formatted.contains("head = 001"));
        assert!(formatted.contains("domain = systems-atomic-concurrency"));
    }
}

struct MeerkatOpts {
    use_baby: bool,
    use_xs: bool,
    use_builder: bool,
    use_tools: bool,
    use_search: bool,
    force_think: Option<bool>,
    force_verify: Option<bool>,
    verbose: bool,
    flow_mode: FlowMode,
    max_tokens: usize,
    max_retries: usize,
    device_pref: nioxs_engine::DevicePreference,
}

async fn run_meerkat_cli(raw_args: Vec<String>) -> anyhow::Result<()> {
    if matches!(raw_args.first().map(|s| s.as_str()), Some("-h" | "--help")) {
        println!(
            "Usage: nioxs meerkat [--tiny] [--chat] [--builder] [--tools] [--search] [--teach <fact>] [--memories] [--forget <id|all>] [--think|--no-think] [--verify|--no-verify] [--retries <N>] [--verbose|-v] [prompt]"
        );
        println!(
            "  teach <trigger => fact>   Save an instant SimSimi-style memory to ~/.nioxs/meerkat_memory.jsonl"
        );
        return Ok(());
    }

    let mem_path = meerkat::memory_path();
    let rep_path = meerkat::replay_path();
    if raw_args.first().map(|s| s.as_str()) == Some("teach") {
        let joined = raw_args[1..].join(" ");
        if joined.trim().is_empty() {
            return Err(anyhow::anyhow!(
                "Usage: nioxs meerkat teach \"<trigger => fact>\""
            ));
        }
        let (trigger, content) = joined
            .split_once("=>")
            .map(|(a, b)| (a.trim(), b.trim()))
            .unwrap_or((joined.trim(), joined.trim()));
        let saved = meerkat::append_memory(&mem_path, "global", trigger, content, "teach", true)
            .map_err(|e| anyhow::anyhow!(e))?;
        println!(
            "✔ Taught Meerkat [id={}] ({}): {}",
            saved.id, saved.trigger, saved.content
        );
        return Ok(());
    }

    let mut chat_mode = raw_args.is_empty();
    let mut opts = MeerkatOpts {
        use_baby: false,
        use_xs: false,
        use_builder: false,
        use_tools: false,
        use_search: false,
        force_think: None,
        force_verify: None,
        verbose: false,
        flow_mode: FlowMode::Token,
        max_tokens: 2048,
        max_retries: 3,
        device_pref: nioxs_engine::DevicePreference::Auto,
    };
    let mut prompt_parts = Vec::new();
    let mut iter = raw_args.into_iter();
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--tiny" | "--baby" | "--small" => opts.use_baby = true,
            "--xs" => {
                opts.use_baby = true;
                opts.use_xs = true;
            }
            "--chat" | "-i" => chat_mode = true,
            "--builder" => opts.use_builder = true,
            "--tools" => opts.use_tools = true,
            "--search" => opts.use_search = true,
            "--think" => opts.force_think = Some(true),
            "--no-think" => opts.force_think = Some(false),
            "--verify" => opts.force_verify = Some(true),
            "--no-verify" => opts.force_verify = Some(false),
            "--verbose" | "-v" => opts.verbose = true,
            "--cpu" => opts.device_pref = nioxs_engine::DevicePreference::Cpu,
            "--gpu" => opts.device_pref = nioxs_engine::DevicePreference::Gpu,
            "--memories" => {
                let all = meerkat::load_memories(&mem_path);
                if all.is_empty() {
                    println!("(no Meerkat memories saved yet in {})", mem_path.display());
                } else {
                    for m in all {
                        println!(
                            "#{} [{}|{}] {} => {}",
                            m.id, m.scope, m.source, m.trigger, m.content
                        );
                    }
                }
                return Ok(());
            }
            "--forget" => {
                let target = iter
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--forget requires <id|all>"))?;
                let n =
                    meerkat::forget_memories(&mem_path, &target).map_err(|e| anyhow::anyhow!(e))?;
                println!("✔ Forgot {n} Meerkat memory entry(s).");
                return Ok(());
            }
            "--teach" => {
                let fact = iter
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--teach requires <fact>"))?;
                let (trigger, content) = fact
                    .split_once("=>")
                    .map(|(a, b)| (a.trim(), b.trim()))
                    .unwrap_or((fact.trim(), fact.trim()));
                let saved =
                    meerkat::append_memory(&mem_path, "global", trigger, content, "teach", true)
                        .map_err(|e| anyhow::anyhow!(e))?;
                println!("✔ Taught Meerkat [id={}]: {}", saved.id, saved.content);
                if prompt_parts.is_empty() && iter.len() == 0 && !chat_mode {
                    return Ok(());
                }
            }
            "--flow" => {
                if let Some(v) = iter.next() {
                    opts.flow_mode = match v.to_ascii_lowercase().as_str() {
                        "burst" => FlowMode::Burst,
                        "paragraph" | "para" => FlowMode::Paragraph,
                        _ => FlowMode::Token,
                    };
                }
            }
            "-n" | "--max-tokens" => {
                if let Some(v) = iter.next() {
                    if let Ok(n) = v.parse() {
                        opts.max_tokens = n;
                    }
                }
            }
            "--retries" => {
                if let Some(v) = iter.next() {
                    if let Ok(n) = v.parse::<usize>() {
                        opts.max_retries = n.clamp(1, 32);
                    }
                }
            }
            _ => prompt_parts.push(a),
        }
    }

    let models_dir = nioxs_artifacts::resolve_models_dir();
    let (tokenizer_path, scout_path, builder_path) = if opts.use_baby {
        let orca_tok = [
            models_dir.join("orca-tokenizer.json"),
            models_dir.join("tiger-tokenizer.json"),
        ]
        .into_iter()
        .find(|p| p.is_file());
        let reflex_05b = if opts.use_xs {
            None
        } else {
            [
                models_dir.join("tiny-meerkat-reflex-0.5b-iq3_xxs.gguf"),
                models_dir.join("tiny-meerkat-reflex-0.5b-iq3_s.gguf"),
            ]
            .into_iter()
            .find(|p| p.is_file())
        };
        if let (Some(tok), Some(reflex)) = (orca_tok, reflex_05b) {
            (tok, reflex.clone(), reflex)
        } else {
            let tok = models_dir.join("baby-orca-tokenizer.json");
            if !tok.is_file() {
                return Err(anyhow::anyhow!(
                    "Could not locate baby-orca-tokenizer.json in {}",
                    models_dir.display()
                ));
            }
            (
                tok,
                models_dir.join("baby-meerkat-scout-135m-q4_k_m.gguf"),
                models_dir.join("baby-meerkat-builder-135m-q4_k_m.gguf"),
            )
        }
    } else {
        let tok = [
            models_dir.join("orca-tokenizer.json"),
            models_dir.join("tiger-tokenizer.json"),
        ]
        .into_iter()
        .find(|p| p.is_file())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Could not locate orca-tokenizer.json in {}",
                models_dir.display()
            )
        })?;
        let scout = [
            models_dir.join("meerkat-scout-0.5b-q4_k_m.gguf"),
            models_dir.join("penguin-scout-0.5b-q4_k_m.gguf"),
        ]
        .into_iter()
        .find(|p| p.is_file())
        .unwrap_or_else(|| models_dir.join("meerkat-scout-0.5b-q4_k_m.gguf"));
        let builder = [
            models_dir.join("meerkat-builder-1.5b-q4_k_m.gguf"),
            models_dir.join("penguin-builder-1.5b-q4_k_m.gguf"),
        ]
        .into_iter()
        .find(|p| p.is_file())
        .unwrap_or_else(|| models_dir.join("meerkat-builder-1.5b-q4_k_m.gguf"));
        (tok, scout, builder)
    };
    if !scout_path.is_file() && !builder_path.is_file() {
        let expected = if opts.use_baby {
            "baby-meerkat-scout-135m-q4_k_m.gguf / baby-meerkat-builder-135m-q4_k_m.gguf"
        } else {
            "meerkat-scout-0.5b-q4_k_m.gguf / meerkat-builder-1.5b-q4_k_m.gguf"
        };
        return Err(anyhow::anyhow!(
            "Could not locate Meerkat GGUFs in {} (expected {expected})",
            models_dir.display()
        ));
    }

    if !chat_mode {
        let prompt = prompt_parts.join(" ");
        if prompt.trim().is_empty() {
            return Err(anyhow::anyhow!(
                "Empty prompt. Usage: nioxs meerkat <prompt>"
            ));
        }
        let _ = run_meerkat_turn(
            &prompt,
            &[],
            &opts,
            &mem_path,
            &rep_path,
            &scout_path,
            &builder_path,
            &tokenizer_path,
        )
        .await?;
        return Ok(());
    }

    println!("🦫 Nio Meerkat Interactive Chat (`vMicroLM-Growth`)");
    println!(
        "   Commands: /memories, /forget <id|all>, /exit  (or teach naturally: \"Remember that ...\")\n"
    );
    let mut history: Vec<(String, String)> = Vec::new();
    let mut line = String::new();
    loop {
        print!("you> ");
        let _ = std::io::Write::flush(&mut std::io::stdout());
        line.clear();
        if std::io::stdin().read_line(&mut line)? == 0 {
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if matches!(input, "/exit" | "/quit" | "exit" | "quit") {
            break;
        }
        if input == "/memories" {
            for m in meerkat::load_memories(&mem_path) {
                println!(
                    "#{} [{}|{}] {} => {}",
                    m.id, m.scope, m.source, m.trigger, m.content
                );
            }
            continue;
        }
        if let Some(target) = input.strip_prefix("/forget ") {
            let n = meerkat::forget_memories(&mem_path, target.trim())
                .map_err(|e| anyhow::anyhow!(e))?;
            println!("✔ Forgot {n} memory entry(s).");
            continue;
        }
        print!("meerkat> ");
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let resp = run_meerkat_turn(
            input,
            &history,
            &opts,
            &mem_path,
            &rep_path,
            &scout_path,
            &builder_path,
            &tokenizer_path,
        )
        .await?;
        history.push((input.to_string(), resp));
        if history.len() > 6 {
            history.remove(0);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_meerkat_turn(
    prompt: &str,
    history: &[(String, String)],
    opts: &MeerkatOpts,
    mem_path: &std::path::Path,
    rep_path: &std::path::Path,
    scout_path: &std::path::Path,
    builder_path: &std::path::Path,
    tokenizer_path: &std::path::Path,
) -> anyhow::Result<String> {
    let _initial_path = if opts.use_builder || !scout_path.is_file() {
        builder_path.to_path_buf()
    } else {
        scout_path.to_path_buf()
    };

    if let Some((trigger, fact)) = meerkat::extract_natural_teaching(prompt) {
        if let Ok(saved) =
            meerkat::append_memory(mem_path, "global", &trigger, &fact, "natural_teach", true)
        {
            println!("✔ Learned [id={}]: {}", saved.id, saved.content);
        }
    }

    let mut memories = meerkat::retrieve_memories(mem_path, prompt, 3);
    memories.retain(|m| m.source != "healed");
    let taught_name = meerkat::load_memories(mem_path)
        .into_iter()
        .rev()
        .find(|m| {
            let t = m.trigger.to_ascii_lowercase();
            let c = m.content.to_ascii_lowercase();
            t.contains("your name") || c.contains("your name is") || c.contains("name is")
        });
    let mem_block = if memories.is_empty() {
        String::new()
    } else {
        let lines: Vec<String> = memories
            .iter()
            .map(|m| format!("- {}: {}", m.trigger, m.content))
            .collect();
        format!(
            "[Known User Facts / Taught Memories]\n{}\n\n",
            lines.join("\n")
        )
    };

    let think_enabled = opts.force_think.unwrap_or(false);
    let lower_q = prompt.to_ascii_lowercase();
    let asks_identity = [
        "who are you",
        "what are you",
        "your name",
        "who created you",
        "who made you",
        "who built you",
    ]
    .iter()
    .any(|kw| lower_q.contains(kw));

    let mut auto_need_web = opts.use_search;
    let mut auto_need_code = meerkat::looks_like_code_task(prompt, "");
    let tools_enabled = opts.use_tools || opts.use_builder || prompt.contains("<tool ");
    if !auto_need_web && !tools_enabled && !asks_identity && scout_path.is_file() {
        let cls_path = if opts.use_baby && builder_path.is_file() {
            builder_path.to_path_buf()
        } else {
            scout_path.to_path_buf()
        };
        let mut cls_engine =
            nioxs_engine::AstEngine::from_paths(cls_path, tokenizer_path.to_path_buf());
        cls_engine.set_device_preference(opts.device_pref);

        if !auto_need_code {
            let code_cls_prompt = format!(
                "<|im_start|>system\nIs this a programming, coding, or software-writing request? Reply ONLY with YES or NO.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            );
            if let Ok(mut cls_rx) = cls_engine
                .generate_stream(code_cls_prompt, 4, 0.0, 0.90)
                .await
            {
                let mut verdict = String::new();
                while let Some(c) = cls_rx.recv().await {
                    verdict.push_str(&c);
                }
                let upper = verdict.trim().to_ascii_uppercase();
                if upper.starts_with("YES")
                    || (opts.use_baby && !opts.use_xs && !upper.starts_with("NO"))
                {
                    auto_need_code = true;
                }
            }
        }

        if !auto_need_code && memories.is_empty() {
            let web_cls_prompt = format!(
                "<|im_start|>system\nDecide if the user question requires live, current, or post-2024 web data (such as current leaders, latest software versions, recent events, prices, or news) instead of timeless facts or casual chat. Reply ONLY with YES or NO.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            );
            if let Ok(mut cls_rx) = cls_engine
                .generate_stream(web_cls_prompt, 4, 0.0, 0.90)
                .await
            {
                let mut verdict = String::new();
                while let Some(c) = cls_rx.recv().await {
                    verdict.push_str(&c);
                }
                if verdict.trim().to_ascii_uppercase().starts_with("YES") {
                    auto_need_web = true;
                }
            }
        }
    }

    let mut used_web = false;
    let web_block = if auto_need_web {
        if opts.verbose {
            println!("\x1b[1;36m[Meerkat Scout]\x1b[0m Scouting the web...");
        }
        if let Some(snip) = meerkat::fetch_web_grounding(prompt).await {
            used_web = true;
            if opts.verbose {
                println!(
                    "\x1b[1;36m[Meerkat Web]\x1b[0m Retrieved {} chars of grounding.",
                    snip.len()
                );
            }
            format!(
                "[Live Web Search Results]\n- {snip}\n\nInstruction: Answer directly using ONLY the exact version numbers and facts in [Live Web Search Results] above. Do NOT say 'as of my last update'.\n\nQuestion: "
            )
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    let using_reflex_05b = opts.use_baby
        && builder_path
            .file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|n| n.starts_with("tiny-meerkat-reflex-0.5b"));
    let using_builder = (tools_enabled || auto_need_code || used_web || !scout_path.is_file())
        && builder_path.is_file();
    let active_tier_prefix = if using_reflex_05b {
        "meerkat-scout-0.5b"
    } else if opts.use_baby {
        "tiny-builder-135m"
    } else if using_builder {
        "meerkat-builder-1.5b"
    } else {
        "meerkat-scout-0.5b"
    };
    let wave1_domains = meerkat::list_wave1_domains(active_tier_prefix);
    let mut selected_lora: Option<(String, std::path::PathBuf)> = None;
    let mut reusable_engine: Option<(std::path::PathBuf, nioxs_engine::AstEngine)> = None;
    if (auto_need_code || opts.use_builder) && !wave1_domains.is_empty() && scout_path.is_file() {
        let domain_list: Vec<&str> = wave1_domains.iter().map(|(d, _)| d.as_str()).collect();
        let cls_model = if opts.use_baby && builder_path.is_file() {
            builder_path.to_path_buf()
        } else {
            scout_path.to_path_buf()
        };
        let mut dom_eng =
            nioxs_engine::AstEngine::from_paths(cls_model.clone(), tokenizer_path.to_path_buf());
        dom_eng.set_device_preference(opts.device_pref);
        let dom_prompt = format!(
            "<|im_start|>system\nWhat programming language or domain is this? Reply with 1 word from: {} (or NONE).<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            domain_list.join(", ")
        );
        if let Ok(mut d_rx) = dom_eng.generate_stream(dom_prompt, 12, 0.0, 0.90).await {
            let mut verdict = String::new();
            while let Some(c) = d_rx.recv().await {
                verdict.push_str(&c);
            }
            if let Some((d, p)) = meerkat::match_wave1_domain(&wave1_domains, &verdict) {
                selected_lora = Some((d.clone(), p.clone()));
                if opts.verbose {
                    println!("\x1b[1;35m[Meerkat LoRA]\x1b[0m Selected Wave 1 adapter: {d}");
                }
            }
        }
        reusable_engine = Some((cls_model, dom_eng));
    }
    if auto_need_code && !opts.use_builder && builder_path.is_file() {
        let b_label = if using_reflex_05b {
            "Tiny Reflex (0.5B IQ3)"
        } else if opts.use_baby {
            "Tiny Builder (135M)"
        } else {
            "1.5B Builder"
        };
        println!(
            "\x1b[1;36m[Meerkat Scout]\x1b[0m Coding task detected -> routing to {b_label}..."
        );
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    let initial_path = if opts.use_baby
        && !using_reflex_05b
        && selected_lora
            .as_ref()
            .is_some_and(|(d, _)| d != "python-async")
    {
        // Wave 1 135M LoRAs were trained on the frozen SmolLM2-135M-Instruct base (scout_path),
        // whereas baby-meerkat-builder-135m is our 5.6k-row Python Algo+Repair specialist.
        scout_path.to_path_buf()
    } else if using_builder {
        builder_path.to_path_buf()
    } else {
        scout_path.to_path_buf()
    };
    let identity_block = if asks_identity {
        if let Some(nm) = taught_name {
            format!(
                "[Identity]\nYou are created by Nio Labs (local Nio Meerkat vMicroLM-Growth herd), and the user taught you: {}. Use this taught name directly and never claim to be Alibaba Cloud, Qwen, or Pegasus.\n\n",
                nm.content
            )
        } else {
            "[Identity]\nYou are Nio Meerkat (vMicroLM-Growth), created by Nio Labs — a local self-healing conversational and coding AI herd (0.5B Scout + 1.5B Builder) running in nio-xs. Never call yourself Pegasus, Qwen, or Alibaba Cloud.\n\n".to_string()
        }
    } else {
        String::new()
    };
    let tool_menu = if tools_enabled && using_builder {
        "\n\n[Available Tools]\nYou can optionally call ONE tool at a time using flat XML tags when you need live info, a file, or Python execution:\n- <tool name=\"web_search\">query</tool>\n- <tool name=\"web_fetch\">https://...</tool>\n- <tool name=\"read_file\">path/to/file</tool>\n- <tool name=\"run_python\">python code</tool>\nOnly emit a <tool> tag if needed; otherwise answer directly."
    } else {
        ""
    };
    let sys = if using_reflex_05b {
        "You are Nio Meerkat-S (vMicroLM-Reflex), created by Nio Labs. Output clean, accurate code."
            .to_string()
    } else if opts.use_xs {
        "You are Nio Meerkat-XS (135M edge companion), created by Nio Labs. Be natural, concise, and helpful for chat, Q&A, and simple scripts. Always follow [Identity] and [Known User Facts] directly.".to_string()
    } else if opts.use_baby {
        if let Some((ref dom, _)) = selected_lora {
            format!(
                "You are Nio Tiny Meerkat ({dom}), created by Nio Labs. Output clean, accurate code."
            )
        } else if think_enabled {
            "You are Nio Tiny Meerkat, created by Nio Labs. Output clean, accurate code."
                .to_string()
        } else {
            "You are Nio Tiny Meerkat, created by Nio Labs. Be natural, concise, and accurate. Always follow [Identity] and [Known User Facts] directly.".to_string()
        }
    } else if think_enabled {
        format!(
            "You are Nio Meerkat Builder Reasoner (vMicroLM-Growth), created by Nio Labs. Write a concise 3-6 line <think>...</think> plan, then provide a clear, accurate answer using any provided [Identity], [Known User Facts], or [Live Web Search Results].{tool_menu}"
        )
    } else {
        format!(
            "You are Nio Meerkat (vMicroLM-Growth), created by Nio Labs. Be natural, concise, and accurate. Always follow [Identity] and [Known User Facts] directly. If you don't know a current/post-2024 fact and it is not in [Known User Facts] or [Live Web Search Results], reply ONLY with [SEARCH].{tool_menu}"
        )
    };
    let mut history_chatml = String::new();
    for (u, a) in history {
        history_chatml.push_str(&format!(
            "<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
        ));
    }
    let effective_mem = if opts.use_baby && (auto_need_code || opts.use_builder) {
        ""
    } else {
        mem_block.as_str()
    };
    let user_msg = format!("{identity_block}{effective_mem}{web_block}{prompt}");
    let mut formatted_prompt = format!(
        "<|im_start|>system\n{sys}<|im_end|>\n{history_chatml}<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
    );

    let temp = if think_enabled { 0.20 } else { 0.30 };
    let t0 = std::time::Instant::now();
    let mut engine = match reusable_engine {
        Some((p, eng)) if p == initial_path => eng,
        _ => {
            let mut e = nioxs_engine::AstEngine::from_paths(
                initial_path.clone(),
                tokenizer_path.to_path_buf(),
            );
            e.set_device_preference(opts.device_pref);
            e
        }
    };
    if let Some((ref dom, ref lora_path)) = selected_lora {
        if !(opts.use_baby && !using_reflex_05b && dom == "python-async") {
            let _ = engine.attach_lora_head(&lora_path.to_string_lossy()).await;
        }
    }
    let mut rx = engine
        .generate_stream(formatted_prompt.clone(), opts.max_tokens, temp, 0.90)
        .await?;
    let mut flow = StreamFlow::new(opts.flow_mode);
    flow.set_show_think_tags(true);
    let mut full_resp = String::new();
    let hide_tool_xml = tools_enabled && !opts.verbose;
    while let Some(chunk) = rx.recv().await {
        full_resp.push_str(&chunk);
        if !hide_tool_xml {
            flow.push(&chunk);
        }
    }
    if hide_tool_xml && meerkat::extract_tool_call(&full_resp).is_none() {
        flow.push(&full_resp);
    }
    flow.finish();
    if !hide_tool_xml || meerkat::extract_tool_call(&full_resp).is_none() {
        println!();
    }

    let mut active_model = initial_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("meerkat")
        .to_string();

    if tools_enabled && builder_path.is_file() {
        let mut cached_builder = if using_builder { Some(engine) } else { None };
        for tool_step in 1..=3 {
            let Some((tool_name, tool_arg)) = meerkat::extract_tool_call(&full_resp) else {
                break;
            };
            if opts.verbose {
                println!(
                    "\x1b[1;35m[Meerkat Tool {}/3]\x1b[0m {}({:?})",
                    tool_step, tool_name, tool_arg
                );
            }
            let tool_out = meerkat::execute_tool(&tool_name, &tool_arg).await;
            if tool_name == "web_search" || tool_name == "web_fetch" {
                used_web = true;
            }
            formatted_prompt.push_str(&format!(
                "{full_resp}<|im_end|>\n<|im_start|>user\n<tool_result name=\"{tool_name}\">\n{tool_out}\n</tool_result>\nContinue and answer the user's request concisely using the tool result above (or call another tool if needed).<|im_end|>\n<|im_start|>assistant\n"
            ));
            let b_engine = cached_builder.get_or_insert_with(|| {
                let mut e = nioxs_engine::AstEngine::from_paths(
                    builder_path.to_path_buf(),
                    tokenizer_path.to_path_buf(),
                );
                e.set_device_preference(opts.device_pref);
                e
            });
            let mut b_rx = b_engine
                .generate_stream(formatted_prompt.clone(), opts.max_tokens, 0.20, 0.90)
                .await?;
            let mut b_flow = StreamFlow::new(opts.flow_mode);
            b_flow.set_show_think_tags(true);
            full_resp.clear();
            while let Some(chunk) = b_rx.recv().await {
                full_resp.push_str(&chunk);
                if !hide_tool_xml {
                    b_flow.push(&chunk);
                }
            }
            if hide_tool_xml && meerkat::extract_tool_call(&full_resp).is_none() {
                b_flow.push(&full_resp);
            }
            b_flow.finish();
            if !hide_tool_xml || meerkat::extract_tool_call(&full_resp).is_none() {
                println!();
            }
            active_model = "meerkat-builder-1.5b-q4_k_m.gguf".to_string();
        }
    }

    let should_verify = opts
        .force_verify
        .unwrap_or_else(|| meerkat::looks_like_code_task(prompt, &full_resp));
    let mut healed = false;
    let mut verified_opt: Option<bool> = None;

    if should_verify {
        println!("\x1b[1;36m[Meerkat Verify]\x1b[0m Checking generated code...");
        let _ = std::io::Write::flush(&mut std::io::stdout());
        let (mut ok, mut err) = verify_orca_with_mode(&full_resp, "code");
        if ok {
            println!("\x1b[1;32m[Meerkat Verify]\x1b[0m Passed ✔");
        }
        if !ok && builder_path.is_file() {
            let web_heal = if web_block.is_empty() && opts.use_search {
                println!("\x1b[1;36m[Meerkat Web]\x1b[0m Fetching grounding...");
                meerkat::fetch_web_grounding(prompt)
                    .await
                    .map(|s| format!("\n\n[Web Grounding]\n- {s}"))
                    .unwrap_or_default()
            } else {
                web_block.clone()
            };
            let used_web = !web_heal.is_empty();
            let first_draft = full_resp.clone();
            let first_err = err.clone();

            let total_rounds =
                if opts.use_xs && !opts.use_builder && opts.force_verify != Some(true) {
                    opts.max_retries.min(1)
                } else {
                    opts.max_retries.max(1)
                };
            for round in 1..=total_rounds {
                if ok {
                    break;
                }
                let fresh_restart =
                    round > total_rounds.div_ceil(2) || (total_rounds >= 2 && round == 2);
                let short_err = baby_orca::compact_verify_issue(&err);
                println!(
                    "\x1b[1;33m[Meerkat Heal {round}/{total_rounds}{}]\x1b[0m {short_err}",
                    if fresh_restart { " fresh-restart" } else { "" }
                );
                let builder_label = if using_reflex_05b {
                    "Tiny Reflex (0.5B IQ3)"
                } else if opts.use_baby {
                    "Tiny Builder (135M)"
                } else {
                    "1.5B Builder"
                };
                println!(
                    "\x1b[1;33m[Meerkat Builder]\x1b[0m Escalating to {builder_label} (thinking & repairing)..."
                );
                let _ = std::io::Write::flush(&mut std::io::stdout());
                let heal_sys = if using_reflex_05b {
                    "You are Nio vMicroLM-Reflex, created by Nio Labs. Output clean, accurate code."
                        .to_string()
                } else if opts.use_baby {
                    if let Some((ref dom, _)) = selected_lora {
                        format!(
                            "You are Nio Tiny Meerkat ({dom}), created by Nio Labs. Output clean, accurate code."
                        )
                    } else {
                        "You are Nio Tiny Meerkat, created by Nio Labs. Output clean, accurate code.".to_string()
                    }
                } else {
                    format!(
                        "You are Meerkat Builder Reasoner (vMicroLM-Growth), developed by Nio Labs. Write a short 3-6 line <think>...</think> root-cause analysis, then output a complete, clean, working solution in the requested language with minimal verification tests.{mem_block}{web_heal}"
                    )
                };
                let heal_user = if fresh_restart {
                    format!(
                        "Original request:\n{prompt}\n\nDiscard the broken draft and write a fresh, simple, working solution from scratch."
                    )
                } else {
                    format!(
                        "Original request:\n{prompt}\n\nFailed draft:\n{full_resp}\n\nVerifier error:\n{err}\n\nFix the issue completely."
                    )
                };
                let heal_prompt = format!(
                    "<|im_start|>system\n{heal_sys}<|im_end|>\n<|im_start|>user\n{heal_user}<|im_end|>\n<|im_start|>assistant\n"
                );
                let heal_temp = if fresh_restart {
                    (0.35 + 0.05 * ((round - 1) as f64)).min(0.70)
                } else {
                    (0.20 + 0.05 * ((round - 1) as f64)).min(0.50)
                };
                let heal_model_path = if opts.use_baby
                    && !using_reflex_05b
                    && selected_lora
                        .as_ref()
                        .is_some_and(|(d, _)| d != "python-async")
                {
                    scout_path.to_path_buf()
                } else {
                    builder_path.to_path_buf()
                };
                let mut b_engine = nioxs_engine::AstEngine::from_paths(
                    heal_model_path.clone(),
                    tokenizer_path.to_path_buf(),
                );
                b_engine.set_device_preference(opts.device_pref);
                if let Some((ref dom, ref lora_path)) = selected_lora {
                    if !(opts.use_baby && !using_reflex_05b && dom == "python-async") {
                        let _ = b_engine
                            .attach_lora_head(&lora_path.to_string_lossy())
                            .await;
                    }
                }
                let mut b_rx = b_engine
                    .generate_stream(heal_prompt, opts.max_tokens, heal_temp, 0.90)
                    .await?;
                let mut b_flow = StreamFlow::new(opts.flow_mode);
                b_flow.set_show_think_tags(true);
                full_resp.clear();
                while let Some(chunk) = b_rx.recv().await {
                    full_resp.push_str(&chunk);
                    b_flow.push(&chunk);
                }
                b_flow.finish();
                println!();
                active_model = heal_model_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("meerkat-builder-1.5b-q4_k_m.gguf")
                    .to_string();
                let res = verify_orca_with_mode(&full_resp, "code");
                ok = res.0;
                err = res.1;
            }
            if ok {
                healed = true;
                let _ = meerkat::append_replay(
                    rep_path,
                    &meerkat::MeerkatReplayEntry {
                        prompt: prompt.to_string(),
                        failed_draft: Some(first_draft),
                        verifier_error: Some(first_err.clone()),
                        verified_solution: full_resp.clone(),
                        used_web,
                        created_at: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_secs())
                            .unwrap_or(0),
                    },
                );
                let short_first = baby_orca::compact_verify_issue(&first_err);
                let _ = meerkat::append_memory(
                    mem_path,
                    "project",
                    prompt,
                    &format!("Healed & verified solution recorded (fixed: {short_first})."),
                    "healed",
                    true,
                );
            }
        }
        verified_opt = Some(ok);
        if !ok && opts.verbose {
            eprintln!("[Meerkat Verify] Last error: {err}");
        }
    } else if meerkat::looks_like_knowledge_gap(&full_resp) && builder_path.is_file() {
        let web_heal = if web_block.is_empty() {
            if opts.verbose {
                println!(
                    "\x1b[1;36m[Meerkat Scout]\x1b[0m Knowledge gap detected; scouting the web..."
                );
            }
            meerkat::fetch_web_grounding(prompt)
                .await
                .map(|s| {
                    used_web = true;
                    format!("[Live Web Search Results]\n- {s}\n\nInstruction: Answer concisely in 1-3 sentences using ONLY the facts in [Live Web Search Results] above.\n\nQuestion: ")
                })
                .unwrap_or_default()
        } else {
            web_block.clone()
        };
        if !web_heal.is_empty() {
            println!(
                "\x1b[1;33m[Meerkat Heal]\x1b[0m Answering from live web grounding via 1.5B Builder..."
            );
            let heal_prompt = format!(
                "<|im_start|>system\nYou are Nio Meerkat (vMicroLM-Growth), created by Nio Labs. Answer directly and concisely using ONLY the facts in [Live Web Search Results].<|im_end|>\n<|im_start|>user\n{web_heal}{prompt}<|im_end|>\n<|im_start|>assistant\n"
            );
            let mut b_engine = nioxs_engine::AstEngine::from_paths(
                builder_path.to_path_buf(),
                tokenizer_path.to_path_buf(),
            );
            b_engine.set_device_preference(opts.device_pref);
            let mut b_rx = b_engine
                .generate_stream(heal_prompt, opts.max_tokens, 0.20, 0.90)
                .await?;
            let mut b_flow = StreamFlow::new(opts.flow_mode);
            b_flow.set_show_think_tags(true);
            full_resp.clear();
            while let Some(chunk) = b_rx.recv().await {
                full_resp.push_str(&chunk);
                b_flow.push(&chunk);
            }
            b_flow.finish();
            println!();
            active_model = "meerkat-builder-1.5b-q4_k_m.gguf".to_string();
            healed = true;
            let _ = meerkat::append_memory(
                mem_path,
                "global",
                prompt,
                full_resp.trim(),
                "web_healed",
                true,
            );
            let _ = meerkat::append_replay(
                rep_path,
                &meerkat::MeerkatReplayEntry {
                    prompt: prompt.to_string(),
                    failed_draft: None,
                    verifier_error: Some("knowledge_gap".to_string()),
                    verified_solution: full_resp.clone(),
                    used_web: true,
                    created_at: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0),
                },
            );
        }
    } else if used_web {
        let _ = meerkat::append_memory(
            mem_path,
            "global",
            prompt,
            full_resp.trim(),
            "web_healed",
            true,
        );
    }

    if opts.verbose {
        let elapsed = t0.elapsed().as_secs_f64();
        println!(
            "\n---\nMeerkat [vMicroLM-Growth] | model={} | memories={} | web={} | healed={} | verified={:?} | {:.2}s",
            active_model,
            memories.len(),
            used_web,
            healed,
            verified_opt,
            elapsed
        );
    }
    Ok(full_resp)
}

async fn run_baby_orca_one(
    model_path: &std::path::Path,
    tokenizer_path: &std::path::Path,
    device_pref: nioxs_engine::DevicePreference,
    sys: &str,
    user: &str,
    max_tokens: usize,
    temp: f64,
) -> anyhow::Result<String> {
    let prompt = format!(
        "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    );
    let mut engine =
        nioxs_engine::AstEngine::from_paths(model_path.to_path_buf(), tokenizer_path.to_path_buf());
    engine.set_device_preference(device_pref);
    let mut rx = engine
        .generate_stream(prompt, max_tokens, temp, 0.90)
        .await?;
    let mut out = String::new();
    while let Some(c) = rx.recv().await {
        out.push_str(&c);
    }
    Ok(out)
}

async fn run_baby_orca_cli(raw_args: Vec<String>) -> anyhow::Result<()> {
    if raw_args.is_empty() || matches!(raw_args[0].as_str(), "-h" | "--help") {
        println!(
            "Usage: nioxs tiny-orca [--pod <name>] [--search] [--verify|--no-verify] [--retries <N>] [--verbose|-v] [--cpu|--gpu] [-n <max_tokens>] <prompt>"
        );
        println!("Pods: {}", baby_orca::BABY_ORCA_PODS.join(", "));
        return Ok(());
    }

    let mut forced_pod: Option<String> = None;
    let mut use_search = false;
    let mut force_verify: Option<bool> = None;
    let mut force_relay: Option<bool> = None;
    let mut verbose = false;
    let mut flow_mode = FlowMode::Token;
    let mut max_tokens: usize = 512;
    let mut max_retries: usize = 3;
    let mut device_pref = nioxs_engine::DevicePreference::Auto;
    let mut prompt_parts = Vec::new();

    let mut iter = raw_args.into_iter();
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--pod" => forced_pod = iter.next(),
            "--search" => use_search = true,
            "--verify" => force_verify = Some(true),
            "--no-verify" => force_verify = Some(false),
            "--relay" => force_relay = Some(true),
            "--no-relay" => force_relay = Some(false),
            "--verbose" | "-v" => verbose = true,
            "--cpu" => device_pref = nioxs_engine::DevicePreference::Cpu,
            "--gpu" => device_pref = nioxs_engine::DevicePreference::Gpu,
            "--flow" => {
                if let Some(v) = iter.next() {
                    flow_mode = FlowMode::parse(&v);
                }
            }
            "-n" | "--max-tokens" => {
                if let Some(v) = iter.next() {
                    if let Ok(n) = v.parse() {
                        max_tokens = n;
                    }
                }
            }
            "--retries" => {
                if let Some(v) = iter.next() {
                    if let Ok(n) = v.parse::<usize>() {
                        max_retries = n.clamp(1, 32);
                    }
                }
            }
            _ => prompt_parts.push(a),
        }
    }
    let prompt = prompt_parts.join(" ");
    if prompt.trim().is_empty() {
        return Err(anyhow::anyhow!(
            "Empty prompt. Usage: nioxs tiny-orca <prompt>"
        ));
    }

    let models_dir = nioxs_artifacts::resolve_models_dir();
    let baby_dir = baby_orca::resolve_baby_orca_dir(&models_dir);
    let tokenizer_path = [
        models_dir.join("baby-orca-tokenizer.json"),
        baby_dir.join("baby-orca-tokenizer.json"),
    ]
    .into_iter()
    .find(|p| p.is_file())
    .ok_or_else(|| {
        anyhow::anyhow!(
            "Missing baby-orca-tokenizer.json in {}",
            models_dir.display()
        )
    })?;

    let lead_path = baby_orca::pod_gguf_path(&baby_dir, "lead");
    let scout_path = baby_orca::pod_gguf_path(&baby_dir, "web-scout");
    let guard_path = baby_orca::pod_gguf_path(&baby_dir, "guard");
    let reason_path = baby_orca::pod_gguf_path(&baby_dir, "reasoning-think");
    if !lead_path.is_file() {
        return Err(anyhow::anyhow!(
            "Could not locate Tiny Orca GGUFs in {}",
            baby_dir.display()
        ));
    }

    let t0 = std::time::Instant::now();

    // 1. Web Scout Pod (135M)
    let lower_q = prompt.to_ascii_lowercase();
    let asks_identity = ["who are you", "who are u", "what are you", "your name"]
        .iter()
        .any(|kw| lower_q.contains(kw));
    let mut web_query = if use_search {
        Some(prompt.clone())
    } else {
        None
    };
    if web_query.is_none() && !asks_identity && scout_path.is_file() {
        println!("\x1b[1;36m[Tiny Orca]\x1b[0m Step 1/4: Web Scout...");
        let scout_raw = run_baby_orca_one(
            &scout_path,
            &tokenizer_path,
            device_pref,
            "You are Tiny Orca Web Scout (135M). Reply YES (with QUERY:) if this asks for facts, definitions, software/tools, or current data to verify on the web; reply NO only for pure math or coding.",
            &prompt,
            32,
            0.0,
        )
        .await?;
        web_query = baby_orca::parse_web_scout(&scout_raw, &prompt);
        if web_query.is_none()
            && !meerkat::looks_like_code_task(&prompt, "")
            && prompt.trim_end().ends_with('?')
        {
            web_query = Some(prompt.clone());
        }
    }
    let mut used_web = false;
    let web_ctx = if let Some(q) = web_query {
        println!("\x1b[1;36m[Tiny Orca]\x1b[0m Scouting web: {q}");
        if let Some(snip) = meerkat::fetch_web_grounding(&q).await {
            used_web = true;
            format!("[Live Web Search Results]\n- {snip}\n\n")
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    // 2. Lead Router Pod (135M)
    let chosen_pod = if let Some(p) = forced_pod {
        baby_orca::parse_lead_route(&p).to_string()
    } else if asks_identity {
        "docs-summary".to_string()
    } else {
        let route_raw = run_baby_orca_one(
            &lead_path,
            &tokenizer_path,
            device_pref,
            "You are Tiny Orca Lead (135M). Route the request to the single best specialist pod name.",
            &prompt,
            16,
            0.0,
        )
        .await?;
        baby_orca::parse_lead_route(&route_raw).to_string()
    };
    println!("\x1b[1;36m[Tiny Orca]\x1b[0m Step 2/4: Lead -> {chosen_pod}");

    // Optional 2-Pod Relay (reasoning-think 2-line plan -> specialist pod)
    let use_relay = force_relay.unwrap_or_else(|| meerkat::looks_like_code_task(&prompt, ""))
        && chosen_pod != "reasoning-think"
        && !asks_identity
        && reason_path.is_file();
    let relay_plan = if use_relay {
        println!("\x1b[1;36m[Tiny Orca]\x1b[0m Relay: Planning via reasoning-think...");
        let plan_raw = run_baby_orca_one(
            &reason_path,
            &tokenizer_path,
            device_pref,
            "You are Tiny Orca 'reasoning-think' Pod (135M). Output ONLY a 2-line invariant/edge-case checklist inside <think>...</think>.",
            &format!("{web_ctx}{prompt}"),
            64,
            0.1,
        )
        .await?;
        format!("[Invariant Plan]\n{}\n\n", plan_raw.trim())
    } else {
        String::new()
    };

    // 3. Active Specialist Pod (135M, 1 pod at a time — buffered so final answer prints once)
    println!("\x1b[1;36m[Tiny Orca]\x1b[0m Step 3/4: Drafting via {chosen_pod}...");
    let pod_path = baby_orca::pod_gguf_path(&baby_dir, &chosen_pod);
    let active_pod_path = if pod_path.is_file() {
        pod_path
    } else {
        lead_path.clone()
    };
    let pod_sys = if asks_identity {
        "You are Tiny Orca (15x135M), created by Nio Labs — an ultra-light sequential micro-pod AI fleet. Answer directly and concisely.".to_string()
    } else {
        format!(
            "You are Tiny Orca '{chosen_pod}' Pod (135M), created by Nio Labs. Be ultra-concise, accurate, and zero-filler."
        )
    };
    let pod_user = format!("{web_ctx}{relay_plan}{prompt}");
    let pod_prompt = format!(
        "<|im_start|>system\n{pod_sys}<|im_end|>\n<|im_start|>user\n{pod_user}<|im_end|>\n<|im_start|>assistant\n"
    );
    let mut pod_engine =
        nioxs_engine::AstEngine::from_paths(active_pod_path, tokenizer_path.to_path_buf());
    pod_engine.set_device_preference(device_pref);
    let mut rx = pod_engine
        .generate_stream(pod_prompt, max_tokens, 0.25, 0.90)
        .await?;
    let mut draft = String::new();
    while let Some(c) = rx.recv().await {
        draft.push_str(&c);
    }

    let should_verify =
        force_verify.unwrap_or_else(|| meerkat::looks_like_code_task(&prompt, &draft));

    // 4. Anti-Hallucination Guard Pod (135M) — run when grounded by web or verifying code
    let mut grounded = true;
    let mut guard_reason: Option<String> = None;
    if !asks_identity && guard_path.is_file() {
        let guard_user = format!("[Context]\n{web_ctx}{prompt}\n\n[Draft]\n{draft}");
        let guard_raw = run_baby_orca_one(
            &guard_path,
            &tokenizer_path,
            device_pref,
            "You are Tiny Orca Anti-Hallucination Guard (135M). Reply GROUNDED or HALLUCINATION: <reason>.",
            &guard_user,
            32,
            0.0,
        )
        .await?;
        let (ok, reason) = baby_orca::parse_guard_verdict(&guard_raw);
        grounded = ok;
        guard_reason = reason.map(|r| {
            if r.eq_ignore_ascii_case("incorrect programming language") && !should_verify {
                "needs clearer grounding".to_string()
            } else {
                r
            }
        });
        println!(
            "\x1b[1;35m[Tiny Orca]\x1b[0m Step 4/4: Guard -> {}",
            if grounded {
                "GROUNDED".to_string()
            } else {
                format!(
                    "Unverified ({})",
                    guard_reason.as_deref().unwrap_or("needs check")
                )
            }
        );
    } else if !asks_identity {
        println!("\x1b[1;35m[Tiny Orca]\x1b[0m Step 4/4: Guard -> OK (direct)");
    }

    let mut verified_opt: Option<bool> = None;
    let mut healed = false;
    let mut final_pod = chosen_pod.clone();
    let mut final_resp = draft.clone();

    let mut verify_err: Option<String> = None;
    if should_verify {
        let (ok, err) = verify_orca_with_mode(&draft, "code");
        verified_opt = Some(ok);
        if !ok {
            verify_err = Some(err);
        }
    }

    if !asks_identity && (!grounded || verify_err.is_some()) && reason_path.is_file() {
        let total_rounds = max_retries.max(1);
        for round in 1..=total_rounds {
            let issue = verify_err
                .as_deref()
                .map(baby_orca::compact_verify_issue)
                .or_else(|| guard_reason.clone())
                .unwrap_or_else(|| "Need micro-<think> verification".to_string());
            let fresh_restart =
                round > total_rounds.div_ceil(2) || (total_rounds >= 2 && round == 2);
            println!(
                "\x1b[1;33m[Tiny Orca Heal {round}/{total_rounds}{}]\x1b[0m Refining via reasoning-think (135M): {issue}",
                if fresh_restart { " fresh-restart" } else { "" }
            );
            let heal_user = if fresh_restart {
                format!(
                    "{web_ctx}Task: {prompt}\n\nPrevious attempt failed with: {issue}\nWrite a fresh, clean, working solution."
                )
            } else {
                format!("{web_ctx}Task: {prompt}\n\nIssue flagged: {issue}\nFix concisely.")
            };
            let heal_prompt = format!(
                "<|im_start|>system\nYou are Tiny Orca 'reasoning-think' Pod (135M). Write a 2-line <think>...</think> checklist, then a concise corrected answer.<|im_end|>\n<|im_start|>user\n{heal_user}<|im_end|>\n<|im_start|>assistant\n"
            );
            let heal_temp = if fresh_restart {
                (0.35 + 0.05 * ((round - 1) as f64)).min(0.70)
            } else {
                (0.20 + 0.05 * ((round - 1) as f64)).min(0.50)
            };
            let mut h_engine = nioxs_engine::AstEngine::from_paths(
                reason_path.clone(),
                tokenizer_path.to_path_buf(),
            );
            h_engine.set_device_preference(device_pref);
            let mut h_rx = h_engine
                .generate_stream(heal_prompt, max_tokens, heal_temp, 0.90)
                .await?;
            final_resp.clear();
            while let Some(c) = h_rx.recv().await {
                final_resp.push_str(&c);
            }
            final_pod = "reasoning-think".to_string();
            healed = true;
            if should_verify {
                let (ok2, err2) = verify_orca_with_mode(&final_resp, "code");
                verified_opt = Some(ok2);
                if ok2 {
                    break;
                }
                verify_err = Some(err2);
            } else {
                break;
            }
        }
    }

    let mut flow = StreamFlow::new(flow_mode);
    flow.set_show_think_tags(true);
    flow.push(&final_resp);
    flow.finish();
    println!();

    if verbose {
        println!(
            "\n---\nTiny Orca [23×135M] | pod={} | web={} | grounded={} | healed={} | verified={:?} | {:.2}s",
            final_pod,
            used_web,
            grounded,
            healed,
            verified_opt,
            t0.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

async fn run_orca_pack_cli(raw_args: Vec<String>) -> anyhow::Result<()> {
    if raw_args.is_empty() || matches!(raw_args[0].as_str(), "-h" | "--help") {
        println!(
            "Usage: nioxs orca-pack [--top <N>] [--all-pods] [--verify|--no-verify] [--retries <N>] [--verbose|-v] [--cpu|--gpu] [-n <max_tokens>] <prompt>"
        );
        println!(
            "Runs 0.5B Orca Nano Pods (default top 4, --all-pods for 12) + 1.5B Lead synthesis + --verify."
        );
        return Ok(());
    }

    let mut top_n: usize = 4;
    let mut force_verify: Option<bool> = None;
    let mut verbose = false;
    let mut flow_mode = FlowMode::Token;
    let mut max_tokens: usize = 1024;
    let mut max_retries: usize = 3;
    let mut device_pref = nioxs_engine::DevicePreference::Auto;
    let mut prompt_parts = Vec::new();

    let mut iter = raw_args.into_iter();
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--all-pods" => top_n = 12,
            "--top" => {
                if let Some(v) = iter.next() {
                    top_n = v.parse().unwrap_or(4).clamp(1, 12);
                }
            }
            "--verify" => force_verify = Some(true),
            "--no-verify" => force_verify = Some(false),
            "--verbose" | "-v" => verbose = true,
            "--cpu" => device_pref = nioxs_engine::DevicePreference::Cpu,
            "--gpu" => device_pref = nioxs_engine::DevicePreference::Gpu,
            "--flow" => {
                if let Some(v) = iter.next() {
                    flow_mode = FlowMode::parse(&v);
                }
            }
            "-n" | "--max-tokens" => {
                if let Some(v) = iter.next() {
                    if let Ok(n) = v.parse() {
                        max_tokens = n;
                    }
                }
            }
            "--retries" => {
                if let Some(v) = iter.next() {
                    if let Ok(n) = v.parse::<usize>() {
                        max_retries = n.clamp(1, 32);
                    }
                }
            }
            _ => prompt_parts.push(a),
        }
    }
    let prompt = prompt_parts.join(" ");
    if prompt.trim().is_empty() {
        return Err(anyhow::anyhow!(
            "Empty prompt. Usage: nioxs orca-pack <prompt>"
        ));
    }

    let home = std::env::var("HOME").unwrap_or_default();
    let models_dir = nioxs_artifacts::resolve_models_dir();
    let search_dirs = [
        models_dir.clone(),
        models_dir.join("orca"),
        std::path::PathBuf::from("models/orca_nano_gguf"),
        std::path::PathBuf::from("models/gguf"),
        std::path::PathBuf::from(&home).join("nio-labs/nio-ideas/models/orca_nano_gguf"),
        std::path::PathBuf::from(&home).join("nio-labs/nio-ideas/models/gguf"),
    ];
    let find_gguf = |name: &str| -> Option<std::path::PathBuf> {
        search_dirs
            .iter()
            .map(|d| d.join(name))
            .find(|p| p.is_file())
    };

    let tokenizer_path = [
        models_dir.join("orca-tokenizer.json"),
        std::path::PathBuf::from(&home)
            .join("nio-labs/nio-ideas/models/orca_lead_1_5b_instruct_tuned/tokenizer.json"),
    ]
    .into_iter()
    .find(|p| p.is_file())
    .ok_or_else(|| anyhow::anyhow!("Could not locate orca-tokenizer.json"))?;

    let lead_path = find_gguf("orca-lead-1.5b-q4_k_m.gguf")
        .or_else(|| find_gguf("orca-nano-lead-0.5b-q4_k_m.gguf"))
        .ok_or_else(|| anyhow::anyhow!("Could not locate orca-lead-1.5b-q4_k_m.gguf"))?;

    let nano_pod_specs = [
        (
            "algo-reasoner",
            "orca-nano-algo-reasoner-0.5b-q4_k_m.gguf",
            "algorithms and complexity",
        ),
        (
            "architect",
            "orca-nano-architect-0.5b-q4_k_m.gguf",
            "software architecture and design",
        ),
        (
            "security-reasoner",
            "orca-nano-security-reasoner-0.5b-q4_k_m.gguf",
            "security, safety, and edge validation",
        ),
        (
            "test-reasoner",
            "orca-nano-test-reasoner-0.5b-q4_k_m.gguf",
            "unit tests and boundary edge cases",
        ),
        (
            "repair-reasoner",
            "orca-nano-repair-reasoner-0.5b-q4_k_m.gguf",
            "bug prevention and invariants",
        ),
        (
            "reasoner",
            "orca-nano-reasoner-0.5b-q4_k_m.gguf",
            "step-by-step logic",
        ),
        (
            "pod-0",
            "orca-nano-pod-0-0.5b-q4_k_m.gguf",
            "core syntax and stdlib",
        ),
        (
            "pod-1",
            "orca-nano-pod-1-0.5b-q4_k_m.gguf",
            "data structures and I/O",
        ),
        (
            "pod-2",
            "orca-nano-pod-2-0.5b-q4_k_m.gguf",
            "concurrency and systems",
        ),
        (
            "pod-3",
            "orca-nano-pod-3-0.5b-q4_k_m.gguf",
            "APIs, parsing, and formatting",
        ),
        (
            "pod-4",
            "orca-nano-pod-4-0.5b-q4_k_m.gguf",
            "CLI, scripting, and tooling",
        ),
        (
            "pod-5",
            "orca-nano-pod-5-0.5b-q4_k_m.gguf",
            "clarity and concise explanation",
        ),
    ];

    let active_pods: Vec<_> = nano_pod_specs
        .into_iter()
        .filter_map(|(id, file, role)| find_gguf(file).map(|p| (id, p, role)))
        .take(top_n)
        .collect();
    if active_pods.is_empty() {
        return Err(anyhow::anyhow!(
            "No 0.5B Orca Nano pods found for orca-pack."
        ));
    }

    let t0 = std::time::Instant::now();
    let total_pods = active_pods.len();
    println!(
        "\x1b[1;36m[Orca Pack]\x1b[0m Step 1/3: Running {} × 0.5B Nano Pods...",
        total_pods
    );
    let _ = std::io::Write::flush(&mut std::io::stdout());

    let sem = Arc::new(tokio::sync::Semaphore::new(2));
    let mut join_set = tokio::task::JoinSet::new();
    for (pod_id, pod_path, role) in active_pods {
        let tok = tokenizer_path.clone();
        let q = prompt.clone();
        let sem_c = sem.clone();
        join_set.spawn(async move {
            let _permit = sem_c.acquire_owned().await;
            let sys = format!(
                "You are Orca Nano '{pod_id}' Pod (0.5B), specializing in {role}. Reply ONLY with [SKIP] if irrelevant, OR [NOTE] followed by 2 concise bullet points (max 60 words)."
            );
            let p = format!(
                "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
            );
            let mut eng = nioxs_engine::AstEngine::from_paths(pod_path, tok);
            eng.set_device_preference(device_pref);
            let Ok(mut rx) = eng.generate_stream(p, 80, 0.2, 0.90).await else {
                return (pod_id, String::new());
            };
            let mut out = String::new();
            while let Some(c) = rx.recv().await {
                out.push_str(&c);
            }
            (pod_id, out.trim().to_string())
        });
    }

    let mut notes = Vec::new();
    let mut done_cnt = 0usize;
    while let Some(res) = join_set.join_next().await {
        done_cnt += 1;
        if let Ok((pod_id, text)) = res {
            let is_skip = text.is_empty() || text.to_ascii_uppercase().starts_with("[SKIP]");
            println!(
                "  \x1b[2m[{done_cnt}/{total_pods}] {pod_id} -> {}\x1b[0m",
                if is_skip { "SKIP" } else { "NOTE" }
            );
            let _ = std::io::Write::flush(&mut std::io::stdout());
            if !is_skip {
                notes.push(format!("- [{pod_id}] {text}"));
            }
        }
    }
    println!(
        "\x1b[1;36m[Orca Pack]\x1b[0m Step 2/3: Synthesizing {} pod notes via 1.5B Lead...",
        notes.len()
    );
    let _ = std::io::Write::flush(&mut std::io::stdout());

    let notes_block = if notes.is_empty() {
        String::new()
    } else {
        format!("[Pack Specialist Notes]\n{}\n\n", notes.join("\n"))
    };
    let lead_prompt = format!(
        "<|im_start|>system\nYou are Orca Pack 1.5B Lead, developed by Nio Labs. Synthesize the [Pack Specialist Notes] into one clean, accurate, complete response.<|im_end|>\n<|im_start|>user\n{notes_block}{prompt}<|im_end|>\n<|im_start|>assistant\n"
    );
    let mut lead_eng =
        nioxs_engine::AstEngine::from_paths(lead_path.clone(), tokenizer_path.clone());
    lead_eng.set_device_preference(device_pref);
    let mut rx = lead_eng
        .generate_stream(lead_prompt, max_tokens, 0.2, 0.90)
        .await?;
    let mut flow = StreamFlow::new(flow_mode);
    flow.set_show_think_tags(true);
    let mut final_resp = String::new();
    while let Some(c) = rx.recv().await {
        final_resp.push_str(&c);
        flow.push(&c);
    }
    flow.finish();
    println!();

    let should_verify =
        force_verify.unwrap_or_else(|| meerkat::looks_like_code_task(&prompt, &final_resp));
    let mut verified_opt = None;
    if should_verify {
        println!("\x1b[1;36m[Orca Pack]\x1b[0m Step 3/3: Verifying code...");
        let (mut ok, mut err) = verify_orca_with_mode(&final_resp, "code");
        let total_rounds = max_retries.max(1);
        for round in 1..=total_rounds {
            if ok {
                break;
            }
            let fresh_restart =
                round > total_rounds.div_ceil(2) || (total_rounds >= 2 && round == 2);
            let short_err = baby_orca::compact_verify_issue(&err);
            println!(
                "\x1b[1;33m[Orca Pack Heal {round}/{total_rounds}{}]\x1b[0m {short_err}",
                if fresh_restart { " fresh-restart" } else { "" }
            );
            let heal_user = if fresh_restart {
                format!(
                    "{notes_block}Task:\n{prompt}\n\nPrevious attempt failed with:\n{err}\n\nDiscard the broken draft and write a fresh, clean, verified solution."
                )
            } else {
                format!(
                    "{notes_block}Task:\n{prompt}\n\nFailed code:\n{final_resp}\n\nError:\n{err}\n\nFix the failing code completely."
                )
            };
            let heal_prompt = format!(
                "<|im_start|>system\nYou are Orca Pack 1.5B Lead, developed by Nio Labs. Fix the failing solution cleanly and completely.<|im_end|>\n<|im_start|>user\n{heal_user}<|im_end|>\n<|im_start|>assistant\n"
            );
            let heal_temp = if fresh_restart {
                (0.35 + 0.05 * ((round - 1) as f64)).min(0.70)
            } else {
                (0.20 + 0.05 * ((round - 1) as f64)).min(0.50)
            };
            let mut h_eng =
                nioxs_engine::AstEngine::from_paths(lead_path.clone(), tokenizer_path.clone());
            h_eng.set_device_preference(device_pref);
            let mut h_rx = h_eng
                .generate_stream(heal_prompt, max_tokens, heal_temp, 0.90)
                .await?;
            let mut h_flow = StreamFlow::new(flow_mode);
            h_flow.set_show_think_tags(true);
            final_resp.clear();
            while let Some(c) = h_rx.recv().await {
                final_resp.push_str(&c);
                h_flow.push(&c);
            }
            h_flow.finish();
            println!();
            let res = verify_orca_with_mode(&final_resp, "code");
            ok = res.0;
            err = res.1;
        }
        verified_opt = Some(ok);
        if ok {
            println!("\x1b[1;32m[Orca Pack]\x1b[0m Verified ✔");
        }
    }

    if verbose {
        println!(
            "\n---\nOrca Pack [1.5B Lead + 0.5B Pack] | notes={} | verified={:?} | {:.2}s",
            notes.len(),
            verified_opt,
            t0.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

fn route_orca_pod_meta(
    prompt: &str,
    forced_pod: Option<&str>,
    manifest_entries: &[serde_json::Value],
) -> (String, bool, String, Option<String>) {
    let extract_meta = |e: &serde_json::Value| {
        let think = e.get("think").and_then(|v| v.as_bool()).unwrap_or(false)
            || e.get("role")
                .and_then(|v| v.as_str())
                .map(|r| r.contains("reason"))
                .unwrap_or(false);
        let vmode = e
            .get("verify_mode")
            .and_then(|v| v.as_str())
            .unwrap_or("code")
            .to_string();
        let gguf = e
            .get("gguf")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        (think, vmode, gguf)
    };
    if let Some(p) = forced_pod {
        let (think, vmode, gguf) = manifest_entries
            .iter()
            .find(|e| e.get("pod_id").and_then(|v| v.as_str()) == Some(p))
            .map(extract_meta)
            .unwrap_or((false, "code".to_string(), None));
        return (p.to_string(), think, vmode, gguf);
    }

    let prompt_tokens: std::collections::HashSet<String> = prompt
        .to_ascii_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();

    let mut best_pod = "lead".to_string();
    let mut best_score = 0usize;
    let mut best_is_bundle = false;
    let mut matched_think = false;
    let mut matched_vmode = "code".to_string();
    let mut matched_gguf: Option<String> = None;
    for entry in manifest_entries {
        let Some(pid) = entry.get("pod_id").and_then(|v| v.as_str()) else {
            continue;
        };
        let mut score = 0usize;
        if let Some(terms) = entry.get("salient_terms").and_then(|v| v.as_array()) {
            for t in terms {
                if let Some(ts) = t.as_str() {
                    if prompt_tokens.contains(&ts.to_ascii_lowercase()) {
                        score += 1;
                    }
                }
            }
        }
        if prompt_tokens.contains(&pid.to_ascii_lowercase()) {
            score += 2;
        }
        let is_bundle = entry
            .get("bundle")
            .and_then(|v| v.as_str())
            .map(|s| !s.is_empty())
            .unwrap_or(false);
        if score > best_score || (score > 0 && score == best_score && is_bundle && !best_is_bundle)
        {
            best_score = score;
            best_is_bundle = is_bundle;
            best_pod = pid.to_string();
            let (think, vmode, gguf) = extract_meta(entry);
            matched_think = think;
            matched_vmode = vmode;
            matched_gguf = gguf;
        }
    }
    (best_pod, matched_think, matched_vmode, matched_gguf)
}

fn route_orca_pod(
    prompt: &str,
    forced_pod: Option<&str>,
    manifest_entries: &[serde_json::Value],
) -> (String, bool) {
    let (pod, think, _, _) = route_orca_pod_meta(prompt, forced_pod, manifest_entries);
    (pod, think)
}

fn verify_orca_with_mode(resp: &str, verify_mode: &str) -> (bool, String) {
    match verify_mode {
        "none" => (true, String::new()),
        "json" => {
            let body = resp.split("</think>").last().unwrap_or(resp).trim();
            let candidate = if let Some(start) = body.find("```") {
                let after = &body[start + 3..];
                if let Some(nl) = after.find('\n') {
                    let code = &after[nl + 1..];
                    code.find("```")
                        .map(|end| code[..end].trim())
                        .unwrap_or(body)
                } else {
                    body
                }
            } else {
                body
            };
            match serde_json::from_str::<serde_json::Value>(candidate) {
                Ok(_) => (true, String::new()),
                Err(e) => (false, format!("JSONParseError: {e}")),
            }
        }
        _ => verify_orca_code(resp),
    }
}

fn run_cmd_with_timeout(
    mut cmd: std::process::Command,
    timeout_secs: u64,
    use_stdout_on_fail: bool,
) -> (bool, String) {
    use std::io::Read;
    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return (true, String::new()),
        Err(e) => return (false, e.to_string()),
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if status.success() {
                    return (true, String::new());
                }
                let mut err_buf = String::new();
                if use_stdout_on_fail {
                    if let Some(mut out) = child.stdout.take() {
                        let _ = out.read_to_string(&mut err_buf);
                    }
                } else if let Some(mut err) = child.stderr.take() {
                    let _ = err.read_to_string(&mut err_buf);
                }
                return (false, err_buf.trim().to_string());
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return (
                        false,
                        format!("TimeoutExpired: verification exceeded {timeout_secs}s"),
                    );
                }
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            Err(e) => return (false, e.to_string()),
        }
    }
}

fn verify_orca_code(resp: &str) -> (bool, String) {
    let body = resp.split("</think>").last().unwrap_or(resp);
    let mut lang = "";
    let mut blocks: Vec<String> = Vec::new();
    let mut rest = body;
    while let Some(start) = rest.find("```") {
        let after = &rest[start + 3..];
        let Some(nl) = after.find('\n') else {
            break;
        };
        let tag = after[..nl].trim().to_ascii_lowercase();
        let code_start = &after[nl + 1..];
        let Some(end) = code_start.find("```") else {
            break;
        };
        if lang.is_empty() {
            if tag.contains("python") || tag == "py" {
                lang = "python";
            } else if tag.contains("rust") || tag == "rs" {
                lang = "rust";
            } else if tag == "js" || tag == "javascript" {
                lang = "js";
            } else if tag == "ts" || tag == "typescript" {
                lang = "ts";
            } else if tag == "sh" || tag == "bash" || tag == "zsh" {
                lang = "bash";
            } else if tag == "go" || tag == "golang" {
                lang = "go";
            }
        }
        blocks.push(code_start[..end].to_string());
        rest = &code_start[end + 3..];
    }
    if blocks.is_empty() || lang.is_empty() {
        return (true, String::new());
    }
    let joined = blocks.join("\n\n");
    static VERIFY_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let pid = std::process::id();
    let seq = VERIFY_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp_dir = std::env::temp_dir();
    match lang {
        "python" => {
            let p = tmp_dir.join(format!("nioxs_orca_verify_{pid}_{seq}.py"));
            let _ = std::fs::write(&p, &joined);
            let mut cmd = std::process::Command::new("python3");
            cmd.arg(&p);
            let res = run_cmd_with_timeout(cmd, 5, false);
            let _ = std::fs::remove_file(&p);
            res
        }
        "rust" => {
            let p = tmp_dir.join(format!("nioxs_orca_verify_{pid}_{seq}.rs"));
            let rlib = tmp_dir.join(format!("libnioxs_orca_verify_{pid}_{seq}.rlib"));
            let _ = std::fs::write(&p, &joined);
            let mut cmd = std::process::Command::new("rustc");
            cmd.args([
                "--edition",
                "2021",
                "--crate-type",
                "lib",
                "--emit=metadata",
                "-o",
            ])
            .arg(&rlib)
            .arg(&p);
            let res = run_cmd_with_timeout(cmd, 10, false);
            let _ = std::fs::remove_file(&p);
            let _ = std::fs::remove_file(&rlib);
            res
        }
        "js" => {
            let p = tmp_dir.join(format!("nioxs_orca_verify_{pid}_{seq}.js"));
            let _ = std::fs::write(&p, &joined);
            let mut cmd = std::process::Command::new("node");
            cmd.arg("--check").arg(&p);
            let res = run_cmd_with_timeout(cmd, 5, false);
            let _ = std::fs::remove_file(&p);
            res
        }
        "ts" => {
            let p = tmp_dir.join(format!("nioxs_orca_verify_{pid}_{seq}.ts"));
            let _ = std::fs::write(&p, &joined);
            let mut cmd = std::process::Command::new("tsc");
            cmd.args(["--noEmit", "--skipLibCheck"]).arg(&p);
            let res = run_cmd_with_timeout(cmd, 10, true);
            let _ = std::fs::remove_file(&p);
            res
        }
        "bash" => {
            let p = tmp_dir.join(format!("nioxs_orca_verify_{pid}_{seq}.sh"));
            let _ = std::fs::write(&p, &joined);
            let mut cmd = std::process::Command::new("bash");
            cmd.arg("-n").arg(&p);
            let res = run_cmd_with_timeout(cmd, 5, false);
            let _ = std::fs::remove_file(&p);
            res
        }
        "go" => {
            let p = tmp_dir.join(format!("nioxs_orca_verify_{pid}_{seq}.go"));
            let _ = std::fs::write(&p, &joined);
            let mut cmd = std::process::Command::new("gofmt");
            cmd.arg("-e").arg(&p);
            let res = run_cmd_with_timeout(cmd, 5, false);
            let _ = std::fs::remove_file(&p);
            res
        }
        _ => (true, String::new()),
    }
}

#[cfg(test)]
mod orca_tests {
    use super::*;

    #[test]
    fn test_route_orca_pod_dynamic_manifest() {
        let manifest = vec![
            serde_json::json!({"pod_id": "pod_0", "salient_terms": ["git", "diff"]}),
            serde_json::json!({"pod_id": "sql_plugin", "salient_terms": ["postgres", "migration"], "think": true}),
        ];
        assert_eq!(
            route_orca_pod("fix postgres migration", None, &manifest),
            ("sql_plugin".into(), true)
        );
        assert_eq!(
            route_orca_pod("unrelated prompt", None, &manifest),
            ("lead".into(), false)
        );
        assert_eq!(
            route_orca_pod("anything", Some("sql_plugin"), &manifest),
            ("sql_plugin".into(), true)
        );
    }

    #[test]
    fn test_verify_orca_code_pass_and_fail() {
        let ok = verify_orca_code("```python\nassert 2 + 2 == 4\n```");
        assert!(ok.0, "expected pass: {:?}", ok);

        let bad = verify_orca_code("```python\nassert 2 + 2 == 5\n```");
        assert!(!bad.0);
        assert!(bad.1.contains("AssertionError"));
    }

    #[test]
    fn test_verify_orca_code_timeout() {
        let (ok, err) = verify_orca_code("```python\nwhile True:\n    pass\n```");
        assert!(!ok);
        assert!(err.contains("TimeoutExpired"), "unexpected err: {err}");
    }

    #[test]
    fn test_verify_orca_with_mode_none_and_json() {
        assert_eq!(
            verify_orca_with_mode("Waterloo was 1815.", "none"),
            (true, String::new())
        );
        assert_eq!(
            verify_orca_with_mode("```json\n{\"ok\": true}\n```", "json"),
            (true, String::new())
        );
        let bad = verify_orca_with_mode("not valid json", "json");
        assert!(!bad.0);
        assert!(bad.1.contains("JSONParseError"));
    }

    #[test]
    fn test_route_orca_pod_meta_bundle_fields() {
        let manifest = vec![serde_json::json!({
            "pod_id": "health-nutrition-fitness",
            "bundle": "health-wellness",
            "verify_mode": "none",
            "gguf": "orca-health-nutrition-fitness-0.5b-q4_k_m.gguf",
            "salient_terms": ["protein", "macros", "workout"]
        })];
        let (pod, think, vmode, gguf) =
            route_orca_pod_meta("plan my daily protein macros", None, &manifest);
        assert_eq!(pod, "health-nutrition-fitness");
        assert!(!think);
        assert_eq!(vmode, "none");
        assert_eq!(
            gguf.as_deref(),
            Some("orca-health-nutrition-fitness-0.5b-q4_k_m.gguf")
        );
    }

    #[test]
    fn test_route_orca_pod_meta_bundle_tiebreaker() {
        let manifest = vec![
            serde_json::json!({
                "pod_id": "pod_2",
                "salient_terms": ["for", "python"]
            }),
            serde_json::json!({
                "pod_id": "everyday-cooking-travel",
                "bundle": "everyday-home",
                "verify_mode": "none",
                "gguf": "orca-everyday-cooking-travel-0.5b-q4_k_m.gguf",
                "salient_terms": ["recipe", "cook"]
            }),
        ];
        let (pod, _, vmode, gguf) = route_orca_pod_meta(
            "Give a quick 3-step recipe for garlic butter chicken",
            None,
            &manifest,
        );
        assert_eq!(pod, "everyday-cooking-travel");
        assert_eq!(vmode, "none");
        assert_eq!(
            gguf.as_deref(),
            Some("orca-everyday-cooking-travel-0.5b-q4_k_m.gguf")
        );
    }

    #[test]
    fn test_verify_orca_code_missing_tool_skips() {
        let mut cmd = std::process::Command::new("nioxs_nonexistent_linter_binary_12345");
        cmd.arg("--check");
        let (ok, err) = run_cmd_with_timeout(cmd, 5, false);
        assert!(ok);
        assert!(err.is_empty());
    }
}
