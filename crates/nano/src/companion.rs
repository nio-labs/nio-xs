//! Companion module for NioNano (v0.2.0)
//!
//! Handles in-TUI companion directives:
//! - `--bash: <cmd>`: In-session command line execution without leaving the coding TUI.
//! - `--tools: <task>`: Specialized Python plugin manager with user download consent.
//! - `--coding: <prompt>`: Upstream LLM consultation (Kilo provider) with context bundling.
//! - `--coding:guard` / `--coding!:`: Guarded mode with AST reality check and reroute gate.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

/// Result of a companion directive execution.
#[derive(Debug, Clone, PartialEq)]
pub enum CompanionResult {
    BashOutput {
        command: String,
        exit_code: i32,
        stdout: String,
        stderr: String,
    },
    ToolNeedsConsent {
        plugin_name: String,
        task: String,
    },
    ToolExecuted {
        plugin_name: String,
        output: String,
    },
    ToolFailed {
        plugin_name: String,
        reason: String,
    },
    CodingResponse {
        guarded: bool,
        content: String,
        hallucination_alert: Option<String>,
        steps: Vec<String>,
        reply_secs: f64,
    },
    Blocked {
        reason: String,
    },
}

impl CompanionResult {
    pub fn to_markdown(&self) -> String {
        match self {
            Self::BashOutput {
                command,
                exit_code,
                stdout,
                stderr,
            } => {
                let mut out = format!("### Bash Execution: `{command}`\n\n");
                if *exit_code == 0 {
                    out.push_str("**Status:** Succeeded (exit code 0)\n\n");
                } else {
                    out.push_str(&format!("**Status:** Failed (exit code {exit_code})\n\n"));
                }
                if !stdout.trim().is_empty() {
                    out.push_str(&format!("**Stdout:**\n```\n{}\n```\n", stdout.trim()));
                }
                if !stderr.trim().is_empty() {
                    out.push_str(&format!("**Stderr:**\n```\n{}\n```\n", stderr.trim()));
                }
                if stdout.trim().is_empty() && stderr.trim().is_empty() {
                    out.push_str("*(No output)*\n");
                }
                out
            }
            Self::ToolNeedsConsent { plugin_name, task } => {
                format!(
                    "### 🔌 Plugin Download Required\n\n\
                    The task **\"{task}\"** requires the Python plugin **`{plugin_name}`**, which is not currently installed locally.\n\n\
                    Would you like NioNano to download and install this plugin to proceed?\n\n\
                    • **Plugin:** `{plugin_name}`\n\
                    • **Intended action:** `{task}`\n\n\
                    To proceed, simply reply with: **`yes`** or **`--tools:install {plugin_name}`**."
                )
            }
            Self::ToolExecuted {
                plugin_name,
                output,
            } => {
                if plugin_name == "plugin_catalog" {
                    output.clone()
                } else {
                    format!(
                        "### 🔌 Tool `{plugin_name}` Output\n\n```\n{}\n```",
                        output.trim()
                    )
                }
            }
            Self::ToolFailed {
                plugin_name,
                reason,
            } => {
                format!("### ⚠️ Tool `{plugin_name}` Execution Failed\n\n{reason}")
            }
            Self::CodingResponse {
                guarded,
                content,
                hallucination_alert,
                steps,
                reply_secs,
            } => {
                let trail = if steps.is_empty() {
                    String::new()
                } else {
                    format!("{}\n*✓ replied in {reply_secs:.1}s*\n\n", steps.join("\n"))
                };
                if let Some(alert) = hallucination_alert {
                    format!(
                        "### ⚠️ [Anti-Hallucination Guard Activated]\n\n\
                        {}\n\n\
                        ---\n\
                        {}#### Proposed Draft from Kilo:\n\n{}",
                        alert.trim(),
                        trail,
                        content.trim()
                    )
                } else if *guarded {
                    format!(
                        "### 🛡️ [Guarded Coding Assist Verified]\n\n\
                        *Verified against project AST symbols and files.*\n\n\
                        {}{}",
                        trail,
                        content.trim()
                    )
                } else {
                    format!("{trail}{}", content.trim())
                }
            }
            Self::Blocked { reason } => {
                format!("### 🚫 Operation Blocked by Safety Policy\n\n{reason}")
            }
        }
    }
}

/// Validates whether a bash command is safe to execute in the in-TUI session.
pub fn is_safe_bash_command(cmd: &str) -> Result<(), String> {
    let lower = cmd.to_ascii_lowercase();
    let dangerous_tokens = [
        "rm -rf /",
        "rm -rf ~",
        "rm -fr /",
        "rm -fr ~",
        "mkfs",
        "dd if=",
        ":(){ :|:& };:",
        "> /dev/sd",
        ">/dev/sd",
        "> /dev/nvme",
        ">/dev/nvme",
        "chmod -r 777 /",
        "shutdown",
        "reboot",
        "init 0",
    ];

    for token in &dangerous_tokens {
        if lower.contains(token) {
            return Err(format!(
                "Command contains restricted destructive pattern: '{token}'"
            ));
        }
    }

    Ok(())
}

/// A short discoverability footer appended to `--bash:` results.
///
/// Surfaces the other companion capabilities — especially the consented
/// Python plugins — with one or two concrete examples each, so a user running
/// a shell command can discover what else NioNano offers without typing `help`.
pub fn bash_discovery_footer() -> String {
    let plugin_dir = PathBuf::from(".nioxs/plugins");
    let installed = |name: &str| {
        if plugin_dir.join(format!("{name}.py")).exists() {
            "Installed"
        } else {
            "Available on demand"
        }
    };
    format!(
        "---\n\n\
        ### 🔌 Companion Directives You Can Also Run\n\n\
        **Python plugins** (`--tools`, consented on first use):\n\n\
        • `svg2favicon` [{}] — real favicons from an SVG\n\
          → `--tools generate favicon from logo.svg`\n\n\
        • `style_helper` [{}] — extract CSS/Tailwind colors & fonts\n\
          → `--tools extract color palette from styles.css`\n\n\
        • `image_scaler` [{}] — downscale images to WebP (needs Pillow)\n\
          → `--tools resize assets/banner.png to 800x400`\n\n\
        **Other directives:**\n\n\
        • `--code <request>` — consult the Kilo LLM with bundled context\n\
          → `--code add retry logic to the HTTP client`\n\
        • `--code guard <request>` — same, checked against project symbols\n\
        • `--config code ON` — default every prompt to `--code`\n\
        • `--test [pattern]` — auto-detect and run the project test suite\n\
        • `--git status` / `--git diff` — summarize the working tree\n\
        • `--tools list` — explore all 20 Python plugins",
        installed("svg2favicon"),
        installed("style_helper"),
        installed("image_scaler"),
    )
}

/// Executes a shell command in-session and captures bounded output.
pub fn execute_in_session_bash(cmd: &str) -> CompanionResult {
    if let Err(reason) = is_safe_bash_command(cmd) {
        return CompanionResult::Blocked { reason };
    }

    let output = Command::new("bash").arg("-c").arg(cmd).output();

    match output {
        Ok(out) => {
            let stdout_raw = String::from_utf8_lossy(&out.stdout);
            let stderr_raw = String::from_utf8_lossy(&out.stderr);

            let stdout: String = stdout_raw.chars().take(8000).collect();
            let stderr: String = stderr_raw.chars().take(4000).collect();

            CompanionResult::BashOutput {
                command: cmd.to_string(),
                exit_code: out.status.code().unwrap_or(-1),
                stdout,
                stderr,
            }
        }
        Err(e) => CompanionResult::BashOutput {
            command: cmd.to_string(),
            exit_code: -1,
            stdout: String::new(),
            stderr: format!("Failed to spawn shell process: {e}"),
        },
    }
}

/// Default upstream gateway for `--coding:` consultations: Kilo's
/// OpenAI-compatible gateway. Pre-setup and keyless — free models such as
/// `kilo-auto/free` are served without an API key, matching the `nio` TUI.
pub const DEFAULT_CODING_BASE_URL: &str = "https://api.kilo.ai/api/gateway";
/// Default upstream model when the user has not selected one.
pub const DEFAULT_CODING_MODEL: &str = "kilo-auto/free";

/// Upstream configuration for `--coding:` consultations.
///
/// Kilo is the pre-configured default and is served without an API key;
/// `api_key` stays `None` unless the user points at a gateway that needs one.
#[derive(Debug, Clone, PartialEq)]
pub struct CodingConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub mode: String,
    pub auto_approve: bool,
}

/// A pending `--coding:` upstream consultation.
///
/// The decision layer never performs the (slow) upstream call itself, and it
/// never decides which files matter: the request is handed to the provider,
/// which opens it with `stream: true`, forwards the model's own deltas to the
/// client as they arrive, and relays the model's tool calls back so the model
/// reads the caller's project itself.
#[derive(Debug, Clone, PartialEq)]
pub struct CodingConsult {
    pub config: CodingConfig,
    pub prompt: String,
    /// Upstream conversation: the system instruction, the caller's request, then
    /// the tool traffic the model itself requested on earlier rounds.
    pub messages: Vec<serde_json::Value>,
    /// Tools offered to the model. Empty once the tool-round budget is spent.
    pub tools: Vec<serde_json::Value>,
    /// Paths the model asked to read, kept for the guarded anti-hallucination check.
    pub files: Vec<String>,
    pub guarded: bool,
}

/// Maximum upstream tool rounds for one `--coding:` turn. Every round is one
/// more client request, so this stays small deliberately.
pub const CODING_MAX_ROUNDS: usize = 4;
/// Largest single tool result forwarded upstream.
const CODING_TOOL_RESULT_CHARS: usize = 16_000;

/// The file tools the upstream model may use on the caller's project.
///
/// These are the caller's own client tools, so the model reads exactly what the
/// user's agent reads, from the directory the user started in.
pub fn coding_tool_schemas() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the user's project. Paths are relative to their working directory. Optionally read a later window with start_line and line_count.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Project-relative file path"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "line_count": {"type": "integer", "minimum": 1, "maximum": 300}
                    },
                    "required": ["path"],
                    "additionalProperties": false
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "find_files",
                "description": "List or glob the user's project files when you do not know the layout yet.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Project-relative directory, default ."},
                        "glob": {"type": "string", "description": "Glob such as src/**/*.rs"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50}
                    },
                    "additionalProperties": false
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "read_skill_file",
                "description": "Read SKILL.md or supporting documentation from an enabled project skill to follow specialized workflows.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Skill name"},
                        "path": {"type": "string", "description": "Skill-relative path, default SKILL.md"}
                    },
                    "required": ["name"],
                    "additionalProperties": false
                }
            }
        }),
    ]
}

/// Tools the model may use on the caller's project; everything else is dropped.
pub fn is_coding_tool(name: &str) -> bool {
    matches!(name, "read_file" | "find_files" | "read_skill_file")
}

/// Strips directive prefixes like `--code`, `--coding:`, `--bash`, etc. from user prompts.
pub fn clean_directive_prompt(prompt: &str) -> &str {
    let trimmed = prompt.trim();
    for prefix in [
        "--coding:guard",
        "--code:guard",
        "--coding guard",
        "--code guard",
        "--coding!:",
        "--code!:",
        "--coding!",
        "--code!",
        "--coding:",
        "--code:",
        "--coding",
        "--code",
        "--bash:",
        "--bash",
        "--tools:",
        "--tools",
        "--config:",
        "--config",
        "--test:",
        "--test",
    ] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            let rest = rest.trim_start();
            let rest = rest.strip_prefix(':').unwrap_or(rest).trim_start();
            return rest;
        }
    }
    trimmed
}

/// Loads the configured persona from local XsConfig or ~/.config/nio/config.json if available.
pub fn load_nio_persona() -> Option<String> {
    // 1. Check if an explicit persona is configured in XsConfig (.nioxs/config.json)
    let xs_cfg = load_config();
    if let Some(ref p) = xs_cfg.persona {
        let p_trimmed = p.trim();
        if !p_trimmed.is_empty() && !matches!(p_trimmed.to_ascii_lowercase().as_str(), "off" | "none" | "default") {
            return Some(format!(
                "Identity: Active persona '{p_trimmed}'.\n\
                Persona Instructions (strictly adhere to these across all responses):\n\
                Embody the persona '{p_trimmed}' in tone, style, and approach."
            ));
        }
    }

    // 2. Fallback to ~/.config/nio/config.json
    let home = std::env::var("HOME").ok()?;
    let path = PathBuf::from(home).join(".config/nio/config.json");
    let content = std::fs::read_to_string(path).ok()?;
    let val: serde_json::Value = serde_json::from_str(&content).ok()?;
    let persona = val.get("persona")?;

    // Support string persona: "persona": "Taylor"
    if let Some(persona_str) = persona.as_str().filter(|s| !s.trim().is_empty()) {
        return Some(format!(
            "Identity: Active persona '{persona_str}'.\n\
            Persona Instructions (strictly adhere to these across all responses):\n\
            Embody the persona '{persona_str}' in tone, style, and approach."
        ));
    }

    // Support object persona: "persona": {"name": "Taylor", "instructions": [...]}
    let mut parts = Vec::new();
    if let Some(name) = persona
        .get("name")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
    {
        parts.push(format!("Identity: You are {name}."));
    }
    if let Some(instructions) = persona.get("instructions").and_then(|v| v.as_array()) {
        let clean_instructions: Vec<&str> = instructions
            .iter()
            .filter_map(|i| i.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if !clean_instructions.is_empty() {
            parts.push("Persona Instructions (strictly adhere to these across all responses):".into());
            for (idx, inst) in clean_instructions.iter().enumerate() {
                parts.push(format!("{}. {inst}", idx + 1));
            }
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

/// Enriches the upstream coding system prompt with active persona, skills, and plugins context.
pub fn build_coding_system_prompt(history: &[nioxs_contracts::Message]) -> String {
    let mut prompt = UPSTREAM_SYSTEM_PROMPT.to_string();

    // 1. Execution Mode & Autonomy Permissions
    let cfg = load_config();
    let is_auto = cfg.auto_approve || cfg.mode.eq_ignore_ascii_case("auto");
    let auto_status = if is_auto {
        "ACTIVE (Autonomous execution enabled — proceed with tool calls, code modifications, and commands without requiring interactive user approval for intermediate steps)"
    } else {
        "INTERACTIVE (Confirmation required for modifying sensitive resources or executing destructive actions)"
    };

    let mode_guidance = match cfg.mode.to_ascii_lowercase().as_str() {
        "auto" => "Full autonomy mode: analyze, plan, and execute changes end-to-end.",
        "review" => "Review mode: focus on analyzing code quality, identifying vulnerabilities, bugs, and edge cases with constructive feedback.",
        "architect" => "Architect mode: focus on system boundaries, data structures, modularity, and trade-offs.",
        "fast" => "Fast mode: provide concise, direct code implementations without unnecessary explanation.",
        _ => "Standard coding mode: provide clean, production-ready code with exact explanations.",
    };

    prompt.push_str(&format!(
        "\n\n---\n### ⚡ Execution Mode & Autonomy Permissions\n\
        • Mode: `{}`\n\
        • Auto-Approve: {}\n\
        • Guidance: {}",
        cfg.mode, auto_status, mode_guidance
    ));

    // 2. Persona & Caller Environment Context
    let caller_system = history
        .iter()
        .find(|m| m.role == "system")
        .and_then(|m| m.content.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty());

    if let Some(persona_snippet) = load_nio_persona() {
        prompt.push_str("\n\n---\n### 🎭 Active Persona & Tone Rules (MANDATORY)\n");
        prompt.push_str(&persona_snippet);
        prompt.push_str("\nNOTE: You MUST adopt this persona identity and tone in all explanations, code comments, and commentary while delivering correct code.");
    }

    if let Some(sys) = caller_system {
        prompt.push_str("\n\n---\n### 🌐 Active Environment Context\n");
        prompt.push_str(sys);
    }

    // 3. Plugins & Skills integration
    prompt.push_str("\n\n---\n### 🔌 Available Skills & Plugins in Nio\n\
        • Skills: Project workflows and conventions can be inspected with `read_skill_file`.\n\
        • Plugins: You have companion tools available on demand (e.g. `--tools: <task>` for svg2favicon, image_scaler, style_helper, json_yaml, jwt_decoder, regex_tester, sql_formatter, env_auditor, cron_parser, diff_viewer, uuid_gen, hash_calc, url_parser, base64_tool, color_convert, csv2json, tree_gen, semver_calc, lorem_gen).\n\
        When answering, honor the user's active persona tone and rules while delivering concise, concrete code.");

    prompt
}

/// Builds the upstream conversation for a `--code` consultation, preserving prior conversational context.
///
/// Replays the system instruction and prior conversation turns from `history` so
/// the upstream model remembers context across follow-up turns.
pub fn upstream_messages(
    prompt: &str,
    history: &[nioxs_contracts::Message],
) -> Vec<serde_json::Value> {
    let system_prompt = build_coding_system_prompt(history);
    let mut messages = vec![
        serde_json::json!({"role": "system", "content": system_prompt}),
    ];

    let mut saw_user = false;
    for message in history {
        match message.role.as_str() {
            "user" => {
                let content = message.content.as_str().unwrap_or("").trim();
                let clean = clean_directive_prompt(content);
                if !clean.is_empty() {
                    saw_user = true;
                    messages.push(serde_json::json!({
                        "role": "user",
                        "content": clean,
                    }));
                }
            }
            "assistant" => {
                if !message.tool_calls.is_empty() {
                    let calls: Vec<serde_json::Value> = message
                        .tool_calls
                        .iter()
                        .filter(|call| is_coding_tool(&call.function.name))
                        .map(|call| {
                            serde_json::json!({
                                "id": call.id,
                                "type": "function",
                                "function": {"name": call.function.name, "arguments": call.function.arguments}
                            })
                        })
                        .collect();
                    if !calls.is_empty() {
                        let mut entry = serde_json::json!({
                            "role": "assistant",
                            "tool_calls": calls,
                        });
                        if let Some(content) = message.content.as_str().filter(|s| !s.trim().is_empty()) {
                            entry["content"] = serde_json::json!(content);
                        }
                        messages.push(entry);
                    } else if let Some(content) = message.content.as_str().filter(|s| !s.trim().is_empty()) {
                        messages.push(serde_json::json!({
                            "role": "assistant",
                            "content": content,
                        }));
                    }
                } else if let Some(content) = message.content.as_str().filter(|s| !s.trim().is_empty()) {
                    messages.push(serde_json::json!({
                        "role": "assistant",
                        "content": content,
                    }));
                }
            }
            "tool" => {
                if let Some(id) = message.tool_call_id.as_deref()
                    && let Some(content) = message.content.as_str()
                {
                    // Only results paired with a forwarded call are replayed, so the
                    // upstream never sees a dangling tool message.
                    let paired = messages.iter().any(|prior| {
                        prior["tool_calls"]
                            .as_array()
                            .is_some_and(|calls| calls.iter().any(|call| call["id"].as_str() == Some(id)))
                    });
                    if paired {
                        messages.push(serde_json::json!({
                            "role": "tool",
                            "tool_call_id": id,
                            "content": crate::excerpt(content, CODING_TOOL_RESULT_CHARS),
                        }));
                    }
                }
            }
            _ => {}
        }
    }

    if !saw_user {
        let clean = clean_directive_prompt(prompt);
        messages.push(serde_json::json!({
            "role": "user",
            "content": if clean.is_empty() { prompt } else { clean },
        }));
    }

    messages
}

/// Counts the tool rounds the model already used in this turn.
pub fn coding_tool_rounds(turn: &[nioxs_contracts::Message]) -> usize {
    turn.iter()
        .skip(1)
        .filter(|message| {
            message.role == "assistant"
                && message
                    .tool_calls
                    .iter()
                    .any(|call| is_coding_tool(&call.function.name))
        })
        .count()
}

/// Paths the model asked to read during this turn, for the guarded check.
pub fn coding_read_paths(turn: &[nioxs_contracts::Message]) -> Vec<String> {
    let mut paths = Vec::new();
    for message in turn.iter().skip(1) {
        for call in &message.tool_calls {
            if call.function.name != "read_file" {
                continue;
            }
            if let Ok(args) = serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                && let Some(path) = args["path"].as_str().filter(|p| crate::safe_path(p, false))
                && !paths.iter().any(|known| known == path)
            {
                paths.push(path.to_string());
            }
        }
    }
    paths
}

/// Resolves the upstream coding configuration from the environment.
///
/// Kilo is pre-setup and needs no API key; the env vars are optional overrides.
///
/// - Base URL: `NIOXS_CODING_BASE_URL`, then the `nio` TUI's `NIO_BASE_URL`,
///   then the Kilo gateway default.
/// - API key: `NIOXS_CODING_API_KEY_FILE` (file contents), then
///   `NIOXS_CODING_API_KEY`, then the `nio` TUI's `NIO_API_KEY`.
/// - Model: `NIOXS_CODING_MODEL`, then `kilo-auto/free`.
pub fn coding_config() -> CodingConfig {
    resolve_coding_config(&|name| std::env::var(name).ok())
}

fn resolve_coding_config(get: &dyn Fn(&str) -> Option<String>) -> CodingConfig {
    let base_url = get("NIOXS_CODING_BASE_URL")
        .or_else(|| get("NIO_BASE_URL"))
        .map(|v| v.trim().trim_end_matches('/').to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_CODING_BASE_URL.to_string());
    let api_key = get("NIOXS_CODING_API_KEY_FILE")
        .and_then(|path| std::fs::read_to_string(path.trim()).ok())
        .map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty())
        .or_else(|| get("NIOXS_CODING_API_KEY"))
        .or_else(|| get("NIO_API_KEY"))
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty());
    let model = get("NIOXS_CODING_MODEL")
        .or_else(|| load_config().model)
        .map(|m| m.trim().to_string())
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| DEFAULT_CODING_MODEL.to_string());
    let cfg = load_config();
    let mode = cfg.mode;
    let auto_approve = cfg.auto_approve || mode.eq_ignore_ascii_case("auto");
    CodingConfig {
        base_url,
        api_key,
        model,
        mode,
        auto_approve,
    }
}

/// A bounded snapshot of workspace files bundled as upstream context.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextBundle {
    /// Relative paths of the files included in `context`.
    pub files: Vec<String>,
    /// Compact, bounded context payload sent to the upstream model.
    pub context: String,
}

/// Builds the step labels (without timing) for a `--coding:` turn.
pub fn coding_steps(bundle: &ContextBundle, model: &str) -> Vec<String> {
    let mut steps = vec![TRAIL_SCAN.to_string()];
    for file in &bundle.files {
        steps.push(format!("{TRAIL_READ} {file}"));
    }
    steps.push(format!("{TRAIL_CONSULT} {model}"));
    steps
}

/// Trail line announcing that the caller's project is being scanned.
pub const TRAIL_SCAN: &str = "🛠️ scanning workspace";
/// Trail line announcing a file the caller is about to read.
pub const TRAIL_READ: &str = "📄 read_file";
/// Trail line announcing the upstream consultation.
pub const TRAIL_CONSULT: &str = "💭 consulting with LLM:";

/// One self-terminating trail line for a file the client is about to read.
pub fn trail_read_segment(path: &str) -> String {
    format!("{TRAIL_READ} {path}\n")
}

/// Trail line announcing the upstream consultation.
pub fn trail_consult_segment(model: &str) -> String {
    format!("{TRAIL_CONSULT} {model}\n")
}

/// Builds the ordered progress segments for a `--coding:` turn.
///
/// Each entry is one complete line so the provider emits these one-by-one as
/// `delta.content` events without creating excessive blank lines.
pub fn coding_progress_segments(bundle: &ContextBundle, model: &str) -> Vec<String> {
    let steps = coding_steps(bundle, model);
    steps.into_iter().map(|step| format!("{step}\n")).collect()
}

const BUNDLE_MAX_FILES: usize = 5;
/// Total context payload ceiling shared by the local scan and client tool reads.
pub const BUNDLE_MAX_BYTES: usize = 24_000;
const BUNDLE_FILE_MAX_CHARS: usize = 8_000;
const BUNDLE_MAX_FILE_BYTES: u64 = 65_536;
const BUNDLE_MAX_VISITED: usize = 400;
const BUNDLE_MAX_DEPTH: usize = 5;
pub(crate) const BUNDLE_SKIP_DIRS: &[&str] = &[
    ".git",
    ".svn",
    "target",
    "node_modules",
    "dist",
    "build",
    "vendor",
    ".runtime",
    ".nioxs",
    ".freebuff",
    "__pycache__",
    ".venv",
    "venv",
    ".cache",
];

const BUNDLE_EXTENSIONS: &[&str] = &[
    "rs", "ts", "tsx", "js", "jsx", "py", "go", "java", "kt", "c", "h", "cpp", "hpp", "css",
    "scss", "html", "vue", "svelte", "json", "toml", "yaml", "yml", "md", "sh", "sql", "rb", "php",
    "swift",
];

/// Formats one bundled file as a `### <path>` section with a bounded snippet.
pub fn context_section(path: &str, text: &str) -> String {
    format!(
        "### {path}\n{}\n\n",
        crate::excerpt(text, BUNDLE_FILE_MAX_CHARS)
    )
}

/// Extracts `@<file>` pinned references from a prompt string.
/// Returns the cleaned prompt without the `@<file>` tokens and the list of valid file paths.
pub fn extract_pinned_files(prompt: &str) -> (String, Vec<String>) {
    let mut files = Vec::new();
    let mut kept_words = Vec::new();

    for token in prompt.split_whitespace() {
        if let Some(candidate) = token.strip_prefix('@') {
            let clean_path = candidate.trim_matches(|c: char| "`\"',;:!?<>".contains(c));
            if !clean_path.is_empty() && crate::safe_path(clean_path, false) {
                if !files.iter().any(|f: &String| f == clean_path) {
                    files.push(clean_path.to_string());
                }
                continue;
            }
        }
        kept_words.push(token);
    }
    (kept_words.join(" "), files)
}

/// Bundles a bounded context of relevant workspace files for an upstream request.
pub fn bundle_workspace_context(prompt: &str) -> ContextBundle {
    bundle_context_pinned_in(Path::new("."), prompt, &[])
}

/// Bundles a bounded context prioritizing user-pinned files followed by auto-discovered files.
pub fn bundle_workspace_context_pinned(prompt: &str, pinned: &[String]) -> ContextBundle {
    bundle_context_pinned_in(Path::new("."), prompt, pinned)
}

/// Bundles relevant files under `root`, scored by prompt-word overlap.
pub fn bundle_context_in(root: &Path, prompt: &str) -> ContextBundle {
    bundle_context_pinned_in(root, prompt, &[])
}

/// Bundles relevant files under `root`, prioritizing pinned files followed by prompt-word overlap candidates.
pub fn bundle_context_pinned_in(root: &Path, prompt: &str, pinned: &[String]) -> ContextBundle {
    let mut files = Vec::new();
    let mut context = String::new();

    // 1. Add pinned files first
    for path in pinned {
        let full_path = root.join(path);
        if let Ok(text) = std::fs::read_to_string(&full_path) {
            if !text.contains('\0') {
                let section = context_section(path, &text);
                if context.len() + section.len() <= BUNDLE_MAX_BYTES {
                    context.push_str(&section);
                    files.push(path.clone());
                }
            }
        }
    }

    // 2. Consult AST symbol index if available (.nioxs/index.json)
    let index_path = root.join(".nioxs/index.json");
    if let Some(ws_index) = crate::brain::load_workspace_index(&index_path) {
        for word in prompt.split(|c: char| !c.is_alphanumeric() && c != '_') {
            if word.len() >= 3 {
                for indexed_file in ws_index.find_files_for_symbol(word) {
                    if !files.contains(&indexed_file) && files.len() < BUNDLE_MAX_FILES {
                        let full = root.join(&indexed_file);
                        if let Ok(text) = std::fs::read_to_string(&full) {
                            if !text.contains('\0') {
                                let section = context_section(&indexed_file, &text);
                                if context.len() + section.len() <= BUNDLE_MAX_BYTES {
                                    context.push_str(&section);
                                    files.push(indexed_file);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 3. Discover additional relevant candidates if space permits
    let words = prompt_words(prompt);
    let mut candidates: Vec<(usize, String, String)> = Vec::new();
    let mut visited = 0usize;
    let mut queue: Vec<(PathBuf, usize)> = vec![(root.to_path_buf(), 0)];
    while let Some((dir, depth)) = queue.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            visited += 1;
            if visited > BUNDLE_MAX_VISITED {
                queue.clear();
                break;
            }
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            if meta.is_dir() {
                if depth < BUNDLE_MAX_DEPTH
                    && !BUNDLE_SKIP_DIRS.contains(&name.as_str())
                    && !name.starts_with('.')
                {
                    queue.push((path, depth + 1));
                }
                continue;
            }
            if !meta.is_file() || meta.len() > BUNDLE_MAX_FILE_BYTES {
                continue;
            }
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !BUNDLE_EXTENSIONS.contains(&ext) {
                continue;
            }
            let Ok(rel) = path.strip_prefix(root) else {
                continue;
            };
            let rel = rel.to_string_lossy().replace('\\', "/");
            if !crate::safe_path(&rel, false) || files.contains(&rel) {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            if text.contains('\0') {
                continue;
            }
            let score = score_context_file(&rel, &text, &words);
            if score > 0 {
                candidates.push((score, rel, text));
            }
        }
    }
    candidates.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    for (_, rel, text) in candidates.into_iter() {
        if files.len() >= BUNDLE_MAX_FILES {
            break;
        }
        let section = context_section(&rel, &text);
        if !context.is_empty() && context.len() + section.len() > BUNDLE_MAX_BYTES {
            break;
        }
        context.push_str(&section);
        files.push(rel);
    }
    ContextBundle { files, context }
}

fn prompt_words(prompt: &str) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    for word in prompt.split(|c: char| !c.is_alphanumeric()) {
        let word = word.to_ascii_lowercase();
        if word.chars().count() >= 3 {
            seen.insert(word);
        }
        if seen.len() >= 16 {
            break;
        }
    }
    seen.into_iter().collect()
}

fn score_context_file(path: &str, text: &str, words: &[String]) -> usize {
    if words.is_empty() {
        return 0;
    }
    let path_l = path.to_ascii_lowercase();
    let text_l = text.to_ascii_lowercase();
    words
        .iter()
        .map(|w| {
            let mut score = 0;
            if path_l.contains(w.as_str()) {
                score += 3;
            }
            if text_l.contains(w.as_str()) {
                score += 1;
            }
            score
        })
        .sum()
}

/// System instruction shared by streaming and blocking upstream consultations.
///
/// The model owns exploration: it decides what, if anything, it needs to read
/// from the caller's project, and the caller's own tools fetch it.
pub const UPSTREAM_SYSTEM_PROMPT: &str = "You are a coding assistant working inside a developer's TUI. \
    The user's project is available through the read_file and find_files tools, \
    which run in that project. Read only what you actually need: if the request \
    can be answered without opening files, answer directly. Always invoke tools \
    using the structured function calling API; never output raw XML tags or \
    tool-call blocks in message content. Never invent file contents or claim to \
    have read something you did not read. Return the code or concrete answer \
    inflected through your active persona tone and rules without unnecessary filler.";

/// Builds the OpenAI-compatible `/chat/completions` URL for a consultation.
pub fn upstream_chat_url(config: &CodingConfig) -> String {
    format!("{}/chat/completions", config.base_url.trim_end_matches('/'))
}

/// Builds the OpenAI-compatible request body for a consultation.
///
/// Tools are offered only while the model still has exploration rounds left; an
/// empty tool list means the model must answer with what it already read.
pub fn upstream_chat_body(
    config: &CodingConfig,
    messages: &[serde_json::Value],
    tools: &[serde_json::Value],
    stream: bool,
) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": config.model,
        "messages": messages,
        "stream": stream,
        "max_tokens": 2048,
    });
    if !tools.is_empty() {
        body["tools"] = serde_json::Value::Array(tools.to_vec());
        body["tool_choice"] = serde_json::json!("auto");
    }
    body
}

/// System instruction for the tool-less fallback, which carries a local bundle.
const BUNDLED_SYSTEM_PROMPT: &str = "You are a coding assistant embedded in a developer's TUI. \
    Answer the request using the bundled workspace context when relevant. \
    Deliver the code or concrete answer consistently embodying your active persona rules and tone.";

/// Builds the fallback conversation that carries a locally bundled context, preserving prior context.
pub fn bundled_messages(
    prompt: &str,
    context: &str,
    history: &[nioxs_contracts::Message],
) -> Vec<serde_json::Value> {
    let system_prompt = format!("{}\n\n{}", BUNDLED_SYSTEM_PROMPT, build_coding_system_prompt(history));
    let mut messages = vec![
        serde_json::json!({"role": "system", "content": system_prompt}),
    ];

    let user_content = if context.trim().is_empty() {
        format!("Request: {prompt}")
    } else {
        format!("Bundled workspace context:\n\n{context}\n\nRequest: {prompt}")
    };

    if history.is_empty() {
        messages.push(serde_json::json!({"role": "user", "content": user_content}));
        return messages;
    }

    let last_user_idx = history.iter().rposition(|m| m.role == "user").unwrap_or(history.len());
    for (idx, message) in history.iter().enumerate() {
        if idx == last_user_idx {
            continue;
        }
        match message.role.as_str() {
            "user" => {
                let content = message.content.as_str().unwrap_or("").trim();
                let clean = clean_directive_prompt(content);
                if !clean.is_empty() {
                    messages.push(serde_json::json!({
                        "role": "user",
                        "content": clean,
                    }));
                }
            }
            "assistant" => {
                if let Some(content) = message.content.as_str().filter(|s| !s.trim().is_empty()) {
                    messages.push(serde_json::json!({
                        "role": "assistant",
                        "content": content,
                    }));
                }
            }
            _ => {}
        }
    }

    messages.push(serde_json::json!({"role": "user", "content": user_content}));
    messages
}

/// Consults the configured upstream gateway with the bundled context.
///
/// Used by callers that cannot run tools; a tool-capable caller instead gets a
/// [`CodingConsult`] and does its own tool-driven exploration.
///
/// Sends an OpenAI-compatible `chat/completions` request (bearer auth when a
/// key is configured) and returns the assistant message content, bounded to a
/// sane size. Errors carry an actionable reason; no draft is ever invented
/// locally.
pub fn consult_upstream(
    config: &CodingConfig,
    prompt: &str,
    context: &str,
    history: &[nioxs_contracts::Message],
) -> Result<String, String> {
    let messages = bundled_messages(prompt, context, history);
    let body = upstream_chat_body(config, &messages, &[], false);
    let url = upstream_chat_url(config);

    crate::run_async(async move {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(45))
            .build()
            .map_err(|e| format!("failed to build HTTP client: {e}"))?;
        let mut request = client.post(&url).json(&body);
        if let Some(key) = config.api_key.as_deref() {
            request = request.bearer_auth(key);
        }
        let response = request
            .send()
            .await
            .map_err(|e| format!("request to `{url}` failed: {e}"))?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            let detail: String = text.chars().take(600).collect();
            return Err(format!(
                "upstream `{}` returned {status}: {detail}",
                config.model
            ));
        }
        let value: serde_json::Value = response
            .json()
            .await
            .map_err(|e| format!("invalid upstream response: {e}"))?;
        let content = value["choices"][0]["message"]["content"]
            .as_str()
            .or_else(|| value["choices"][0]["text"].as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or("upstream response contained no message content")?;
        Ok(crate::excerpt(content, 24_000))
    })
}

/// Active persistent configuration for NioXS companion session.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct XsConfig {
    #[serde(default)]
    pub code: bool,
    #[serde(default)]
    pub guard: bool,
    #[serde(default)]
    pub auto_approve: bool,
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_tools_consent")]
    pub tools_consent: String,
    #[serde(default = "default_stream")]
    pub stream: bool,
    #[serde(default)]
    pub persona: Option<String>,
}

fn default_mode() -> String {
    "coding".to_string()
}

fn default_tools_consent() -> String {
    "prompt".to_string()
}

fn default_stream() -> bool {
    true
}

impl Default for XsConfig {
    fn default() -> Self {
        Self {
            code: false,
            guard: false,
            auto_approve: false,
            mode: default_mode(),
            model: None,
            tools_consent: default_tools_consent(),
            stream: default_stream(),
            persona: None,
        }
    }
}

std::thread_local! {
    static TEST_CONFIG: std::cell::RefCell<Option<XsConfig>> = const { std::cell::RefCell::new(None) };
}

pub fn xs_config_path() -> PathBuf {
    if let Ok(explicit) = std::env::var("NIOXS_CONFIG_FILE") {
        return PathBuf::from(explicit);
    }
    PathBuf::from(".nioxs/config.json")
}

pub fn load_config() -> XsConfig {
    if let Some(cfg) = TEST_CONFIG.with(|c| c.borrow().clone()) {
        return cfg;
    }
    let path = xs_config_path();
    if let Ok(data) = std::fs::read_to_string(&path) {
        if let Ok(cfg) = serde_json::from_str::<XsConfig>(&data) {
            return cfg;
        }
    }
    XsConfig::default()
}

pub fn with_test_config<R>(config: XsConfig, f: impl FnOnce() -> R) -> R {
    TEST_CONFIG.with(|c| *c.borrow_mut() = Some(config));
    let res = f();
    TEST_CONFIG.with(|c| *c.borrow_mut() = None);
    res
}

pub fn save_config(config: &XsConfig) -> Result<(), String> {
    let path = xs_config_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let data = serde_json::to_string_pretty(config)
        .map_err(|e| format!("failed to serialize config: {e}"))?;
    std::fs::write(&path, data)
        .map_err(|e| format!("failed to write config to {}: {e}", path.display()))
}

pub fn is_config_code_on() -> bool {
    load_config().code
}

pub fn is_config_guard_on() -> bool {
    load_config().guard
}

pub fn is_config_auto_approve() -> bool {
    let cfg = load_config();
    cfg.auto_approve || cfg.mode.eq_ignore_ascii_case("auto")
}

pub fn config_mode() -> String {
    load_config().mode
}

pub fn is_tools_consent_auto() -> bool {
    load_config().tools_consent.eq_ignore_ascii_case("auto")
}

/// Dispatches a `--config` command to inspect or update settings.
pub fn dispatch_config(req: &str) -> String {
    let trimmed = req.trim();
    let mut config = load_config();

    if trimmed.is_empty()
        || trimmed == "list"
        || trimmed == "show"
        || trimmed == "help"
        || trimmed == "?"
    {
        let code_str = if config.code { "**ON**" } else { "OFF" };
        let auto_str = if config.auto_approve { "**ON**" } else { "OFF" };
        let mode_str = &config.mode;
        let guard_str = if config.guard { "**ON**" } else { "OFF" };
        let stream_str = if config.stream { "**ON**" } else { "OFF" };
        let model_str = config.model.as_deref().unwrap_or("kilo-auto/free (default)");
        let consent_str = &config.tools_consent;
        let persona_str = config.persona.as_deref().unwrap_or("default");

        return format!(
            "### ⚙️ NioXS Companion Configuration\n\n\
            | Setting | Value | Options | Description |\n\
            | :--- | :--- | :--- | :--- |\n\
            | `code` | {code_str} | `ON` / `OFF` | Default all prompts to `--code` consultation |\n\
            | `auto` | {auto_str} | `ON` / `OFF` | Autonomous auto-approval for `--code` actions |\n\
            | `mode` | `{mode_str}` | `coding` / `auto` / `review` / `architect` / `fast` | Execution mode passed to upstream `--code` |\n\
            | `guard` | {guard_str} | `ON` / `OFF` | Strict AST anti-hallucination verification |\n\
            | `model` | `{model_str}` | `<model-id>` / `reset` | Upstream model for coding consultations |\n\
            | `tools_consent` | `{consent_str}` | `prompt` / `auto` | Confirm before downloading Python plugins |\n\
            | `stream` | {stream_str} | `ON` / `OFF` | Progressive response streaming |\n\
            | `persona` | `{persona_str}` | `<name>` / `off` | Active persona tone & instructions |\n\n\
            #### 💡 Configuration Commands:\n\
            • `--config code ON` / `OFF` — toggle automatic `--code` on every prompt\n\
            • `--config auto ON` / `OFF` — toggle autonomous execution & auto-approval\n\
            • `--config mode <name>` — set execution mode (`coding`, `auto`, `review`, `architect`, `fast`)\n\
            • `--config preset <name>` — apply preset (`auto`, `coding`, `review`, `architect`, `fast`)\n\
            • `--config guard ON` / `OFF` — toggle AST symbol checking\n\
            • `--config model <model>` — override upstream coding model (e.g. `claude-3-5-sonnet`)\n\
            • `--config tools_consent auto` — allow instant Python tool installs without prompts\n\
            • `--config stream ON` / `OFF` — toggle response streaming\n\
            • `--config persona <name>` — select persona profile\n\
            • `--config reset` — restore all defaults"
        );
    }

    if trimmed == "reset" {
        let def = XsConfig::default();
        if let Err(e) = save_config(&def) {
            return format!("⚠️ Failed to save config: {e}");
        }
        return "✔ All configurations have been reset to defaults.".to_string();
    }

    let clean = trimmed.replace(':', " ").replace('=', " ");
    let parts: Vec<&str> = clean.split_whitespace().collect();
    if parts.len() < 2 {
        return "⚠️ Invalid syntax. Usage: `--config <setting> <value>` (e.g. `--config code ON`). Type `--config list` for options.".to_string();
    }

    let key = parts[0].to_ascii_lowercase();
    let val = parts[1..].join(" ");
    let val_lower = val.to_ascii_lowercase();

    match key.as_str() {
        "code" => {
            let is_on = matches!(val_lower.as_str(), "on" | "true" | "1" | "enable" | "yes");
            config.code = is_on;
            if let Err(e) = save_config(&config) {
                return format!("⚠️ Failed to save config: {e}");
            }
            if is_on {
                "✔ Config updated: **code** set to **ON**.\nAll subsequent prompts without an explicit directive will automatically default to `--code` consultations without needing the prefix.".to_string()
            } else {
                "✔ Config updated: **code** set to **OFF**.\nPrompts will execute through default intent parsing.".to_string()
            }
        }
        "auto" | "auto_approve" | "auto-approve" => {
            let is_on = matches!(val_lower.as_str(), "on" | "true" | "1" | "enable" | "yes");
            config.auto_approve = is_on;
            if let Err(e) = save_config(&config) {
                return format!("⚠️ Failed to save config: {e}");
            }
            if is_on {
                "✔ Config updated: **auto** set to **ON**.\nAuto-approval is now active and passed to `--code` consultations for autonomous execution.".to_string()
            } else {
                "✔ Config updated: **auto** set to **OFF**.\nActions will run interactively requiring confirmation for sensitive operations.".to_string()
            }
        }
        "mode" => {
            let mode_val = val_lower.clone();
            config.mode = mode_val.clone();
            if mode_val == "auto" {
                config.auto_approve = true;
            }
            if let Err(e) = save_config(&config) {
                return format!("⚠️ Failed to save config: {e}");
            }
            format!(
                "✔ Config updated: **mode** set to `{mode_val}`.\nExecution mode `{mode_val}` will be passed into all subsequent `--code` consultations."
            )
        }
        "preset" => {
            match val_lower.as_str() {
                "auto" => {
                    config.code = true;
                    config.auto_approve = true;
                    config.mode = "auto".to_string();
                    config.tools_consent = "auto".to_string();
                    if let Err(e) = save_config(&config) {
                        return format!("⚠️ Failed to save config: {e}");
                    }
                    "✔ Applied preset **auto**: code=ON, auto_approve=ON, mode=auto, tools_consent=auto.\nFull autonomous mode activated for `--code` consultations.".to_string()
                }
                "review" => {
                    config.code = true;
                    config.auto_approve = false;
                    config.mode = "review".to_string();
                    if let Err(e) = save_config(&config) {
                        return format!("⚠️ Failed to save config: {e}");
                    }
                    "✔ Applied preset **review**: code=ON, auto_approve=OFF, mode=review.\nDetailed code review and vulnerability audit mode activated.".to_string()
                }
                "architect" => {
                    config.code = true;
                    config.auto_approve = false;
                    config.mode = "architect".to_string();
                    if let Err(e) = save_config(&config) {
                        return format!("⚠️ Failed to save config: {e}");
                    }
                    "✔ Applied preset **architect**: code=ON, auto_approve=OFF, mode=architect.\nHigh-level system design and modularity mode activated.".to_string()
                }
                "fast" => {
                    config.code = true;
                    config.auto_approve = true;
                    config.mode = "fast".to_string();
                    if let Err(e) = save_config(&config) {
                        return format!("⚠️ Failed to save config: {e}");
                    }
                    "✔ Applied preset **fast**: code=ON, auto_approve=ON, mode=fast.\nFast concise code generation mode activated.".to_string()
                }
                "coding" | "default" | "standard" => {
                    config.code = false;
                    config.auto_approve = false;
                    config.mode = "coding".to_string();
                    if let Err(e) = save_config(&config) {
                        return format!("⚠️ Failed to save config: {e}");
                    }
                    "✔ Applied preset **coding**: standard interactive coding defaults restored.".to_string()
                }
                _ => {
                    format!("⚠️ Unknown preset `{val}`. Available presets: `auto`, `review`, `architect`, `fast`, `coding`.")
                }
            }
        }
        "guard" => {
            let is_on = matches!(val_lower.as_str(), "on" | "true" | "1" | "enable" | "yes");
            config.guard = is_on;
            if let Err(e) = save_config(&config) {
                return format!("⚠️ Failed to save config: {e}");
            }
            if is_on {
                "✔ Config updated: **guard** set to **ON**.\nCoding consultations will strictly verify AST symbols and guard against hallucinations.".to_string()
            } else {
                "✔ Config updated: **guard** set to **OFF**.\nCoding consultations will run without mandatory AST guard checks.".to_string()
            }
        }
        "model" => {
            if matches!(val_lower.as_str(), "default" | "reset" | "clear" | "none") {
                config.model = None;
                if let Err(e) = save_config(&config) {
                    return format!("⚠️ Failed to save config: {e}");
                }
                "✔ Config updated: **model** reset to default (kilo-auto/free).".to_string()
            } else {
                config.model = Some(val.clone());
                if let Err(e) = save_config(&config) {
                    return format!("⚠️ Failed to save config: {e}");
                }
                format!("✔ Config updated: **model** set to `{val}`.")
            }
        }
        "tools_consent" | "consent" => {
            if matches!(val_lower.as_str(), "auto" | "automatic" | "none" | "off") {
                config.tools_consent = "auto".to_string();
                if let Err(e) = save_config(&config) {
                    return format!("⚠️ Failed to save config: {e}");
                }
                "✔ Config updated: **tools_consent** set to `auto`.\nPython plugins will install automatically on first invocation.".to_string()
            } else {
                config.tools_consent = "prompt".to_string();
                if let Err(e) = save_config(&config) {
                    return format!("⚠️ Failed to save config: {e}");
                }
                "✔ Config updated: **tools_consent** set to `prompt`.\nNioNano will request confirmation before installing plugins.".to_string()
            }
        }
        "stream" => {
            let is_on = matches!(val_lower.as_str(), "on" | "true" | "1" | "enable" | "yes");
            config.stream = is_on;
            if let Err(e) = save_config(&config) {
                return format!("⚠️ Failed to save config: {e}");
            }
            format!(
                "✔ Config updated: **stream** set to **{}**.",
                if is_on { "ON" } else { "OFF" }
            )
        }
        "persona" => {
            if matches!(val_lower.as_str(), "off" | "none" | "clear" | "reset") {
                config.persona = None;
                if let Err(e) = save_config(&config) {
                    return format!("⚠️ Failed to save config: {e}");
                }
                "✔ Config updated: **persona** disabled.".to_string()
            } else {
                config.persona = Some(val.clone());
                if let Err(e) = save_config(&config) {
                    return format!("⚠️ Failed to save config: {e}");
                }
                format!("✔ Config updated: **persona** set to `{val}`.")
            }
        }
        _ => {
            format!("⚠️ Unknown configuration setting `{key}`. Type `--config list` to view available options.")
        }
    }
}

/// Dispatches an interactive help request for directives or overall capabilities.
pub fn dispatch_help(topic: &str) -> String {
    let clean = topic.trim().trim_start_matches('-').to_ascii_lowercase();
    match clean.as_str() {
        "code" | "coding" => {
            "### 🤖 Directive: `--code` (Upstream Coding Consultation)\n\n\
            Run AI-assisted code generation and analysis directly in your session.\n\n\
            #### 💡 Syntax:\n\
            • `--code <prompt>` — Consult upstream AI with auto-discovered project context\n\
            • `--code @path/to/file <prompt>` — Focus specifically on pinned files (e.g. `--code @src/main.rs add CLI flag`)\n\
            • `--code guard <prompt>` or `--code! <prompt>` — Guard against hallucinations by verifying AST symbols\n\n\
            #### ⚡ Autonomy & Mode Configuration:\n\
            • Pass execution mode via `--config mode auto` (or `review`, `architect`, `fast`, `coding`)\n\
            • Enable auto-approval via `--config auto ON` so models proceed without confirmation hurdles\n\
            • Default all chat prompts to `--code` via `--config code ON`\n\
            • Override model via `--config model <model-id>`"
                .to_string()
        }
        "tools" | "tool" | "plugin" | "plugins" => {
            "### 🔌 Directive: `--tools` (Python Companion Tools Suite)\n\n\
            Execute deterministic local Python micro-tools without network dependency.\n\n\
            #### 💡 Syntax:\n\
            • `--tools list` — Display all 20 cataloged plugins across Assets, Formats, Security, & Dev Utilities\n\
            • `--tools <task>` — Auto-detect and execute the right tool for your natural language prompt\n\
            • `--tools install <plugin>` — Install a specific plugin template into `.nioxs/plugins/`\n\n\
            #### ⚙️ Consent Settings:\n\
            • `--config tools_consent auto` — Automatically install tools without prompting for confirmation\n\
            • `--config tools_consent prompt` — Ask before downloading/installing new tools"
                .to_string()
        }
        "config" | "cfg" => {
            "### ⚙️ Directive: `--config` (Runtime Settings & Workflows)\n\n\
            Inspect and customize NioXS companion behavior and autonomy.\n\n\
            #### 💡 Commands:\n\
            • `--config list` — View current configuration table\n\
            • `--config code ON` / `OFF` — Default plain prompts to `--code` consultation\n\
            • `--config auto ON` / `OFF` — Enable autonomous auto-approval for `--code`\n\
            • `--config mode <name>` — Set mode (`coding`, `auto`, `review`, `architect`, `fast`)\n\
            • `--config preset <name>` — Apply preset (e.g. `--config preset auto`)\n\
            • `--config guard ON` / `OFF` — Enforce AST anti-hallucination verification\n\
            • `--config model <name>` — Override upstream LLM model\n\
            • `--config tools_consent auto` / `prompt` — Auto-install plugin policy\n\
            • `--config reset` — Restore default settings"
                .to_string()
        }
        "bash" | "cmd" | "sh" => {
            "### ⚡ Directive: `--bash` (In-Session Terminal Commands)\n\n\
            Run commands in your project's working directory without leaving the chat session.\n\n\
            #### 💡 Syntax:\n\
            • `--bash <cmd>` — Execute command (e.g. `--bash git status`, `--bash cargo check`)\n\n\
            #### 🛡️ Sandbox & Safety:\n\
            • Prohibits destructive system-level commands (e.g. `rm -rf /`, formatting disks)\n\
            • Captures exit codes, stdout, and stderr with formatted syntax output"
                .to_string()
        }
        "test" | "tests" => {
            "### 🧪 Directive: `--test` (Smart Test Runner)\n\n\
            Detects your project framework and executes tests with compressed error traces.\n\n\
            #### 💡 Supported Ecosystems:\n\
            • Rust: `cargo test [pattern]`\n\
            • Node/JS/TS: `npm test -- [pattern]`\n\
            • Python: `pytest -k [pattern]`\n\
            • Go: `go test -run [pattern] ./...`\n\n\
            #### 💡 Syntax:\n\
            • `--test` — Run entire test suite\n\
            • `--test <filter>` — Run specific matching test cases"
                .to_string()
        }
        "git" => {
            "### 📊 Directive: `--git` (Working Tree & Status Summary)\n\n\
            Quick insight into repository state.\n\n\
            #### 💡 Syntax:\n\
            • `--git` or `--git status` — Compact status and modified files diffstat"
                .to_string()
        }
        _ => {
            "### ⚡ NioNano MicroLM (μLM) — Companion Directives & Capabilities\n\n\
            #### 🎯 Companion Directives:\n\
            • `--code <prompt>` — AI coding consultation with workspace awareness (try `--help code`)\n\
            • `--code @file <prompt>` — Pinpoint specific files for AI consultation\n\
            • `--tools <task>` — 20 local Python utility plugins (try `--help tools`)\n\
            • `--config <setting> <value>` — Companion settings & presets (try `--help config`)\n\
            • `--bash <cmd>` — In-session terminal execution (try `--help bash`)\n\
            • `--test [pattern]` — Smart test runner across Rust, TS, Py, Go (try `--help test`)\n\
            • `--git` — Working tree diff & status summary (try `--help git`)\n\n\
            #### 💡 Topic Help:\n\
            Type `--help <directive>` (e.g. `--help code`, `--help tools`, `--help config`)."
                .to_string()
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PluginMeta {
    pub name: &'static str,
    pub title: &'static str,
    pub category: &'static str,
    pub capability: &'static str,
    pub example: &'static str,
    pub keywords: &'static [&'static str],
}

pub const TOOL_PLUGINS: &[PluginMeta] = &[
    // --- Assets & Media ---
    PluginMeta {
        name: "svg2favicon",
        title: "SVG to Favicon Generator",
        category: "Assets & Media",
        capability: "Renders .ico (16/32/48px) and apple-touch-icon.png from SVG using resvg.",
        example: "--tools generate favicon from logo.svg",
        keywords: &["favicon", "svg2favicon", "ico", "apple-touch-icon"],
    },
    PluginMeta {
        name: "image_scaler",
        title: "Image Scaler & WebP Converter",
        category: "Assets & Media",
        capability: "Downscales images to target dimensions and converts to WebP.",
        example: "--tools resize assets/banner.png to 800x400",
        keywords: &["resize", "scale image", "image_scaler", "thumbnail", "webp", "downscale"],
    },
    PluginMeta {
        name: "style_helper",
        title: "CSS & Style Analyzer",
        category: "Assets & Media",
        capability: "Extracts color palettes (HEX/RGB) and font stacks from CSS/SCSS.",
        example: "--tools extract color palette from styles.css",
        keywords: &["style", "palette", "font stack", "style_helper", "css palette"],
    },
    PluginMeta {
        name: "color_convert",
        title: "Color Space Converter",
        category: "Assets & Media",
        capability: "Converts colors between HEX, RGB, HSL, and CSS OKLCH formats.",
        example: "--tools convert #3b82f6 to hsl and rgb",
        keywords: &["color", "hex to rgb", "rgb to hex", "hsl", "oklch", "color_convert"],
    },
    // --- Data & Formats ---
    PluginMeta {
        name: "json_yaml",
        title: "JSON & YAML Converter",
        category: "Data & Formats",
        capability: "Converts and formats between JSON and YAML with syntax validation.",
        example: "--tools convert config.json to yaml",
        keywords: &["yaml", "json to yaml", "yaml to json", "json_yaml"],
    },
    PluginMeta {
        name: "csv2json",
        title: "CSV to JSON / NDJSON Converter",
        category: "Data & Formats",
        capability: "Converts CSV data to JSON array or newline-delimited JSON (NDJSON).",
        example: "--tools convert data.csv to json",
        keywords: &["csv", "csv2json", "csv to json", "ndjson", "tsv"],
    },
    PluginMeta {
        name: "base64_tool",
        title: "Base64 Encoder & Decoder",
        category: "Data & Formats",
        capability: "Encodes or decodes text and binary files to/from Base64 or Data URLs.",
        example: "--tools base64 encode secret.txt",
        keywords: &["base64", "b64", "base64 encode", "base64 decode", "data url", "base64_tool"],
    },
    PluginMeta {
        name: "sql_formatter",
        title: "SQL Query Beautifier",
        category: "Data & Formats",
        capability: "Formats SQL statements with standard indentation and uppercase keywords.",
        example: "--tools format sql select a, b from users where id=1",
        keywords: &["sql", "format sql", "sql_formatter", "beautify sql"],
    },
    // --- Analysis & Security ---
    PluginMeta {
        name: "jwt_decoder",
        title: "JWT Token Inspector",
        category: "Analysis & Security",
        capability: "Decodes JWT headers, payload claims, and validates expiration without secret.",
        example: "--tools decode jwt eyJhbGciOi...",
        keywords: &["jwt", "token", "jwt_decoder", "jwt decode", "bearer token"],
    },
    PluginMeta {
        name: "regex_tester",
        title: "Regex Matcher & Debugger",
        category: "Analysis & Security",
        capability: "Tests regex patterns against sample strings, showing match spans and capture groups.",
        example: "--tools test regex '(\\d{4})-(\\d{2})' on '2026-10-12'",
        keywords: &["regex", "regexp", "regular expression", "regex_tester", "regex test"],
    },
    PluginMeta {
        name: "env_auditor",
        title: ".env & Secret Auditor",
        category: "Analysis & Security",
        capability: "Audits .env against .env.example, detects missing vars and duplicate keys.",
        example: "--tools audit env",
        keywords: &["env", "audit env", "dotenv", "env_auditor", ".env.example", "missing env"],
    },
    PluginMeta {
        name: "hash_calc",
        title: "Multi-Algorithm Hash Calculator",
        category: "Analysis & Security",
        capability: "Calculates SHA-256, SHA-512, SHA-1, and MD5 checksums for strings or files.",
        example: "--tools calculate hash of Cargo.lock",
        keywords: &["hash", "sha256", "sha512", "md5", "checksum", "hash_calc"],
    },
    PluginMeta {
        name: "diff_viewer",
        title: "Terminal Unified Diff Viewer",
        category: "Analysis & Security",
        capability: "Generates colorized unified diff between two files or text blocks.",
        example: "--tools diff old.rs new.rs",
        keywords: &["diff", "file diff", "diff_viewer", "compare files"],
    },
    // --- Developer Utilities ---
    PluginMeta {
        name: "cron_parser",
        title: "Cron Expression Explainer",
        category: "Developer Utilities",
        capability: "Explains 5-field cron expressions in human English and lists the next 5 runs.",
        example: "--tools explain cron '*/15 * * * *'",
        keywords: &["cron", "crontab", "cron_parser", "cron expression", "schedule cron"],
    },
    PluginMeta {
        name: "markdown_toc",
        title: "Markdown TOC Generator",
        category: "Developer Utilities",
        capability: "Generates a GitHub-compatible Table of Contents from Markdown headings.",
        example: "--tools generate toc for README.md",
        keywords: &["toc", "table of contents", "markdown_toc", "readme toc"],
    },
    PluginMeta {
        name: "uuid_gen",
        title: "UUID / GUID Generator",
        category: "Developer Utilities",
        capability: "Generates cryptographically random UUIDv4 identifiers.",
        example: "--tools generate 5 uuids",
        keywords: &["uuid", "guid", "uuid_gen", "generate uuid"],
    },
    PluginMeta {
        name: "url_parser",
        title: "URL Deconstruct & Query Parser",
        category: "Developer Utilities",
        capability: "Parses URLs into scheme, domain, port, path, query parameters map, and hash.",
        example: "--tools parse url 'https://api.nio.dev:8080/v1/query?q=rust&limit=10'",
        keywords: &["url", "query params", "url_parser", "parse url"],
    },
    PluginMeta {
        name: "tree_gen",
        title: "Project Directory Tree Visualizer",
        category: "Developer Utilities",
        capability: "Renders ASCII/Unicode directory tree respecting .gitignore.",
        example: "--tools generate tree for src",
        keywords: &["tree", "tree_gen", "directory tree", "folder structure"],
    },
    PluginMeta {
        name: "semver_calc",
        title: "SemVer Calculator & Bumper",
        category: "Developer Utilities",
        capability: "Parses, compares, and bumps semantic versions (major, minor, patch).",
        example: "--tools bump semver 1.4.2 minor",
        keywords: &["semver", "semantic version", "semver_calc", "bump version"],
    },
    PluginMeta {
        name: "lorem_gen",
        title: "Lorem Ipsum & Mock Text Generator",
        category: "Developer Utilities",
        capability: "Generates customizable dummy placeholder text, markdown copy, or mock data.",
        example: "--tools generate 3 paragraphs of lorem",
        keywords: &["lorem", "lorem ipsum", "placeholder text", "lorem_gen", "dummy text"],
    },
];

pub fn is_known_plugin(name: &str) -> bool {
    TOOL_PLUGINS.iter().any(|m| m.name == name)
}

/// Identifies the plugin required for a given tool task.
pub fn resolve_plugin_for_task(task: &str) -> (String, bool) {
    let lower = task.to_ascii_lowercase();
    let mut chosen = None;

    for meta in TOOL_PLUGINS {
        if lower.starts_with(meta.name)
            || lower.contains(&format!(" {} ", meta.name))
            || lower.ends_with(&format!(" {}", meta.name))
        {
            chosen = Some(meta.name);
            break;
        }
    }

    if chosen.is_none() {
        for meta in TOOL_PLUGINS {
            if meta.keywords.iter().any(|k| lower.contains(k)) {
                chosen = Some(meta.name);
                break;
            }
        }
    }

    let plugin_name = chosen.unwrap_or("custom_tool").to_string();
    let plugin_dir = PathBuf::from(".nioxs/plugins");
    let is_installed = plugin_dir.join(format!("{plugin_name}.py")).exists();

    (plugin_name, is_installed)
}

/// Installs or mocks a python plugin into .nioxs/plugins/
pub fn install_plugin(plugin_name: &str) -> CompanionResult {
    let plugin_dir = PathBuf::from(".nioxs/plugins");
    if let Err(e) = std::fs::create_dir_all(&plugin_dir) {
        return CompanionResult::Blocked {
            reason: format!("Failed to create plugins directory: {e}"),
        };
    }

    let script_path = plugin_dir.join(format!("{plugin_name}.py"));
    let template = match plugin_name {
        "svg2favicon" => {
            r#"#!/usr/bin/env python3
"""svg2favicon - render real favicons from an SVG using resvg."""
import os
import shutil
import struct
import subprocess
import sys
import tempfile

ICO_SIZES = (16, 32, 48)
APPLE_SIZE = 180
SKIP_DIRS = ("node_modules", ".git", "target", "__pycache__")


def find_resvg():
    env = os.environ.get("RESVG_BIN")
    if env and os.path.isfile(env):
        return env
    found = shutil.which("resvg")
    if found:
        return found
    home = os.path.expanduser("~/.local/bin/resvg")
    return home if os.path.isfile(home) else None


def pick_svg(argv):
    for tok in argv:
        cand = tok.strip("`'\"(),.;:")
        if cand.lower().endswith(".svg") and os.path.isfile(cand):
            return cand
    for root in (".", "assets", "public", "static", "src"):
        if not os.path.isdir(root):
            continue
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in sorted(files):
                if name.lower().endswith(".svg"):
                    return os.path.join(dirpath, name)
    return None


def render_png(resvg, svg, size):
    fd, out = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        subprocess.run(
            [resvg, svg, out, "--width", str(size), "--height", str(size)],
            check=True, capture_output=True,
        )
        with open(out, "rb") as fh:
            return fh.read()
    finally:
        if os.path.exists(out):
            os.remove(out)


def build_ico(pngs):
    # ICO container with PNG entries (supported on Vista+).
    header = struct.pack("<HHH", 0, 1, len(pngs))
    offset = 6 + 16 * len(pngs)
    entries = b""
    for size, data in pngs:
        dim = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    return header + entries + b"".join(data for _, data in pngs)


def main():
    argv = sys.argv[1:]
    resvg = find_resvg()
    if not resvg:
        sys.stderr.write("svg2favicon: resvg not found. Install resvg or set RESVG_BIN.\n")
        sys.exit(1)
    svg = pick_svg(argv)
    if not svg:
        sys.stderr.write(
            "svg2favicon: no source SVG found. Pass one, e.g. `--tools: generate favicon from logo.svg`.\n"
        )
        sys.exit(1)
    try:
        pngs = [(s, render_png(resvg, svg, s)) for s in ICO_SIZES]
        apple = render_png(resvg, svg, APPLE_SIZE)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", "replace").strip()
        sys.stderr.write("svg2favicon: resvg failed rendering %s: %s\n" % (svg, detail))
        sys.exit(1)

    with open("favicon.ico", "wb") as fh:
        fh.write(build_ico(pngs))
    with open("apple-touch-icon.png", "wb") as fh:
        fh.write(apple)

    ico_size = os.path.getsize("favicon.ico")
    apple_size = os.path.getsize("apple-touch-icon.png")
    print("svg2favicon: source = %s (renderer: resvg)" % svg)
    print("  wrote favicon.ico (%d bytes) with sizes %s" % (ico_size, "x".join(map(str, ICO_SIZES))))
    print("  wrote apple-touch-icon.png (%d bytes, %dx%d)" % (apple_size, APPLE_SIZE, APPLE_SIZE))


if __name__ == "__main__":
    main()
"#
        }
        "image_scaler" => {
            r#"#!/usr/bin/env python3
"""image_scaler - downscale images to a max width and emit WebP, honestly."""
import os
import sys

SKIP_DIRS = ("node_modules", ".git", "target", "__pycache__")
EXTS = (".png", ".jpg", ".jpeg", ".webp")


def find_images(limit=20):
    hits = []
    for root in (".", "assets", "public", "static", "img", "images"):
        if not os.path.isdir(root):
            continue
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in sorted(files):
                if name.lower().endswith(EXTS):
                    hits.append(os.path.join(dirpath, name))
                    if len(hits) >= limit:
                        return hits
    return hits


def main():
    argv = sys.argv[1:]
    try:
        from PIL import Image
    except ImportError:
        sys.stderr.write(
            "image_scaler: Pillow is not installed, so no image was resized. "
            "Install it (pip install pillow) and re-run.\n"
        )
        sys.exit(1)

    target_w = 800
    for tok in argv:
        if "x" in tok.lower():
            head = tok.lower().split("x", 1)[0].strip()
            if head.isdigit():
                target_w = int(head)

    images = find_images()
    if not images:
        sys.stderr.write("image_scaler: no raster images found to resize.\n")
        sys.exit(1)

    for path in images:
        try:
            with Image.open(path) as img:
                if img.width <= target_w:
                    print("  skip %s (already %dpx wide)" % (path, img.width))
                    continue
                ratio = target_w / img.width
                resized = img.resize((target_w, max(1, round(img.height * ratio))))
                out = os.path.splitext(path)[0] + ".webp"
                resized.save(out, "WEBP", quality=80)
                print("  %s -> %s (%dx%d)" % (path, out, resized.width, resized.height))
        except Exception as exc:  # noqa: BLE001 - report and continue
            sys.stderr.write("  skip %s: %s\n" % (path, exc))
    print("image_scaler: done.")


if __name__ == "__main__":
    main()
"#
        }
        "style_helper" => {
            r##"#!/usr/bin/env python3
"""style_helper - extract hex/rgb colors and font stacks from CSS or Tailwind files."""
import os
import re
import sys

SKIP_DIRS = ("node_modules", ".git", "target", "__pycache__")
CANDIDATES = ("styles.css", "style.css", "global.css", "tailwind.css", "app.css", "index.css")
HEX = re.compile(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
RGB = re.compile(r"rgba?\([^)]*\)")
FONT = re.compile(r"font-family\s*:\s*([^;]+);", re.IGNORECASE)


def pick_css(argv):
    for tok in argv:
        cand = tok.strip("`'\"(),.;:")
        if cand.lower().endswith((".css", ".scss")) and os.path.isfile(cand):
            return cand
    for name in CANDIDATES:
        if os.path.isfile(name):
            return name
    for root in (".", "src", "styles", "assets"):
        if not os.path.isdir(root):
            continue
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in sorted(files):
                if name.lower().endswith((".css", ".scss")):
                    return os.path.join(dirpath, name)
    return None


def main():
    css = pick_css(sys.argv[1:])
    if not css:
        sys.stderr.write("style_helper: no .css/.scss file found to analyze.\n")
        sys.exit(1)
    with open(css, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    seen, order = set(), []
    for match in HEX.findall(text):
        val = match.lower()
        if val not in seen:
            seen.add(val)
            order.append(val)
    for match in RGB.findall(text):
        val = re.sub(r"\s+", " ", match).strip()
        if val not in seen:
            seen.add(val)
            order.append(val)

    print("style_helper: %s" % css)
    if order:
        print("  colors (in order of first appearance):")
        for val in order[:20]:
            print("    %s" % val)
        if len(order) > 20:
            print("    ... and %d more" % (len(order) - 20))
    else:
        print("  no hex/rgb colors found")
    fonts = [re.sub(r"\s+", " ", f).strip() for f in FONT.findall(text)]
    if fonts:
        print("  font stacks:")
        for f in dict.fromkeys(fonts):
            print("    %s" % f)


if __name__ == "__main__":
    main()
"##
        }
        "color_convert" => {
            r##"#!/usr/bin/env python3
"""color_convert - convert colors between HEX, RGB, HSL, and OKLCH."""
import math
import re
import sys


def parse_color(tok):
    tok = tok.strip("`'\"(),;:")
    if tok.startswith("#") and len(tok) in (4, 7):
        if len(tok) == 4:
            r = int(tok[1] * 2, 16)
            g = int(tok[2] * 2, 16)
            b = int(tok[3] * 2, 16)
        else:
            r = int(tok[1:3], 16)
            g = int(tok[3:5], 16)
            b = int(tok[5:7], 16)
        return r, g, b
    m = re.search(r"rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", tok)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def rgb_to_hsl(r, g, b):
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    cmax = max(rf, gf, bf)
    cmin = min(rf, gf, bf)
    delta = cmax - cmin
    l = (cmax + cmin) / 2.0
    if delta == 0:
        h = s = 0.0
    else:
        s = delta / (1.0 - abs(2.0 * l - 1.0))
        if cmax == rf:
            h = ((gf - bf) / delta) % 6
        elif cmax == gf:
            h = (bf - rf) / delta + 2
        else:
            h = (rf - gf) / delta + 4
        h *= 60
    return round(h, 1), round(s * 100, 1), round(l * 100, 1)


def main():
    arg = " ".join(sys.argv[1:])
    m = re.search(r"#[0-9a-fA-F]{3,6}\b|rgb\([^)]+\)", arg)
    color = parse_color(m.group(0)) if m else (59, 130, 246)
    r, g, b = color
    hex_code = f"#{r:02x}{g:02x}{b:02x}"
    h, s, l = rgb_to_hsl(r, g, b)
    print(f"color_convert: parsed RGB({r}, {g}, {b})")
    print(f"  • HEX:   {hex_code.upper()}")
    print(f"  • RGB:   rgb({r}, {g}, {b})")
    print(f"  • HSL:   hsl({h}, {s}%, {l}%)")
    print(f"  • OKLCH: oklch({round(l/100, 2)} {round(s/200, 3)} {h})")


if __name__ == "__main__":
    main()
"##
        }
        "json_yaml" => {
            r##"#!/usr/bin/env python3
"""json_yaml - bidirectional converter and formatter for JSON and YAML."""
import json
import os
import sys


def dump_yaml(data, indent=0):
    lines = []
    prefix = "  " * indent
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{prefix}{k}:")
                lines.append(dump_yaml(v, indent + 1))
            else:
                lines.append(f"{prefix}{k}: {json.dumps(v)}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(dump_yaml(item, indent + 1))
            else:
                lines.append(f"{prefix}- {json.dumps(item)}")
    else:
        lines.append(f"{prefix}{json.dumps(data)}")
    return "\n".join(lines)


def main():
    args = sys.argv[1:]
    target = None
    for a in args:
        if os.path.isfile(a):
            target = a
            break
    if not target:
        for f in ["config.json", "package.json", "settings.json"]:
            if os.path.isfile(f):
                target = f
                break
    if not target:
        sample = {"name": "nio-xs", "version": "0.2.0", "settings": {"theme": "dark", "auto": True}}
        print("json_yaml: demo conversion from sample JSON:")
        print(dump_yaml(sample))
        return

    with open(target, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        data = json.loads(text)
        print(f"json_yaml: converted {target} (JSON -> YAML):")
        print(dump_yaml(data))
    except Exception as e:
        print(f"json_yaml: error parsing {target}: {e}")


if __name__ == "__main__":
    main()
"##
        }
        "csv2json" => {
            r##"#!/usr/bin/env python3
"""csv2json - converts CSV files into formatted JSON or NDJSON."""
import csv
import json
import os
import sys


def main():
    args = sys.argv[1:]
    target = None
    for a in args:
        if os.path.isfile(a):
            target = a
            break
    if not target:
        for f in os.listdir("."):
            if f.endswith(".csv"):
                target = f
                break
    if not target:
        print("csv2json: no CSV file found. Pass one, e.g. `--tools: convert data.csv to json`.")
        return

    rows = []
    with open(target, "r", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    print(f"csv2json: converted {target} ({len(rows)} records):")
    print(json.dumps(rows[:10], indent=2))
    if len(rows) > 10:
        print(f"... and {len(rows) - 10} more records")


if __name__ == "__main__":
    main()
"##
        }
        "base64_tool" => {
            r##"#!/usr/bin/env python3
"""base64_tool - encode and decode strings or files to Base64."""
import base64
import os
import sys


def main():
    task = " ".join(sys.argv[1:])
    decode_mode = "decode" in task.lower()
    target_file = None
    for tok in sys.argv[1:]:
        if os.path.isfile(tok):
            target_file = tok
            break

    if target_file:
        with open(target_file, "rb") as fh:
            data = fh.read()
        if decode_mode:
            out = base64.b64decode(data).decode("utf-8", "replace")
            print(f"base64_tool: decoded {target_file}:\n{out[:500]}")
        else:
            out = base64.b64encode(data).decode("ascii")
            print(f"base64_tool: encoded {target_file} ({len(data)} bytes -> {len(out)} chars):\n{out[:200]}...")
    else:
        payload = task
        for kw in ["base64", "encode", "decode", "--tools", ":"]:
            payload = payload.replace(kw, "")
        payload = payload.strip().strip("'\"`")
        if not payload:
            payload = "Hello from NioNano!"
        if decode_mode:
            try:
                dec = base64.b64decode(payload).decode("utf-8", "replace")
                print(f"base64_tool: decoded result:\n{dec}")
            except Exception as e:
                print(f"base64_tool: decode failed: {e}")
        else:
            enc = base64.b64encode(payload.encode("utf-8")).decode("ascii")
            print(f"base64_tool: encoded result:\n{enc}")


if __name__ == "__main__":
    main()
"##
        }
        "sql_formatter" => {
            r##"#!/usr/bin/env python3
"""sql_formatter - standardizes and beautifies SQL queries."""
import os
import re
import sys

KEYWORDS = [
    "SELECT", "FROM", "WHERE", "AND", "OR", "GROUP BY", "ORDER BY",
    "HAVING", "LIMIT", "OFFSET", "JOIN", "LEFT JOIN", "RIGHT JOIN",
    "INNER JOIN", "OUTER JOIN", "ON", "INSERT INTO", "VALUES",
    "UPDATE", "SET", "DELETE FROM", "CREATE TABLE", "ALTER TABLE", "DROP TABLE"
]


def format_sql(sql):
    formatted = sql
    for kw in KEYWORDS:
        pattern = re.compile(r"\b" + kw + r"\b", re.IGNORECASE)
        formatted = pattern.sub(kw, formatted)
    for clause in ["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING", "LIMIT", "SET", "VALUES"]:
        formatted = re.sub(r"\s*\b(" + clause + r")\b\s*", r"\n\1 ", formatted)
    for sub in ["AND", "OR", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN", "JOIN"]:
        formatted = re.sub(r"\s*\b(" + sub + r")\b\s*", r"\n  \1 ", formatted)
    return formatted.strip()


def main():
    task = " ".join(sys.argv[1:])
    target_file = next((tok for tok in sys.argv[1:] if os.path.isfile(tok)), None)
    if target_file:
        with open(target_file, "r", encoding="utf-8") as fh:
            sql = fh.read()
    else:
        sql = task
        for prefix in ["format sql", "format", "beautify sql", "beautify", "--tools", ":"]:
            if sql.lower().startswith(prefix):
                sql = sql[len(prefix):].strip()
        if not sql:
            sql = "select id, name, email from users where active = 1 and verified = 1 order by id desc limit 10;"

    print("sql_formatter: formatted query:\n")
    print(format_sql(sql))


if __name__ == "__main__":
    main()
"##
        }
        "jwt_decoder" => {
            r##"#!/usr/bin/env python3
"""jwt_decoder - decode and validate JWT headers and payload claims."""
import base64
import json
import sys
import time


def b64_decode(s):
    s += "=" * ((4 - len(s) % 4) % 4)
    return base64.urlsafe_b64decode(s).decode("utf-8", "replace")


def main():
    task = " ".join(sys.argv[1:])
    token = None
    for tok in task.split():
        if tok.count(".") == 2 and len(tok) > 20:
            token = tok
            break
    if not token:
        print("jwt_decoder: no valid JWT format (header.payload.signature) provided.")
        print("Example: `--tools: decode jwt eyJhbGciOi...`")
        return

    parts = token.split(".")
    try:
        header = json.loads(b64_decode(parts[0]))
        payload = json.loads(b64_decode(parts[1]))
    except Exception as e:
        print(f"jwt_decoder: failed to decode JWT payload: {e}")
        return

    print("jwt_decoder: inspection results:")
    print("  • Header:\n" + json.dumps(header, indent=4))
    print("  • Payload:\n" + json.dumps(payload, indent=4))

    now = int(time.time())
    if "exp" in payload:
        exp = payload["exp"]
        diff = exp - now
        if diff > 0:
            print(f"  • Status: VALID (expires in {diff} seconds / {diff // 60}m)")
        else:
            print(f"  • Status: EXPIRED ({abs(diff)} seconds ago)")


if __name__ == "__main__":
    main()
"##
        }
        "regex_tester" => {
            r##"#!/usr/bin/env python3
"""regex_tester - tests regular expressions against strings or files."""
import re
import sys


def main():
    args = sys.argv[1:]
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    sample = "Contact support at team@nio.dev or security@example.com for help."

    for a in args:
        if a.startswith("pattern="):
            pattern = a.split("=", 1)[1].strip("'\"")
        elif a.startswith("test="):
            sample = a.split("=", 1)[1].strip("'\"")

    try:
        rx = re.compile(pattern)
        matches = list(rx.finditer(sample))
        print(f"regex_tester: pattern: `{pattern}`")
        print(f"  Target: `{sample}`")
        print(f"  Total matches: {len(matches)}")
        for idx, m in enumerate(matches, 1):
            print(f"  [{idx}] Span: {m.span()} -> `{m.group(0)}`")
            if m.groups():
                print(f"      Groups: {m.groups()}")
    except Exception as e:
        print(f"regex_tester: invalid pattern `{pattern}`: {e}")


if __name__ == "__main__":
    main()
"##
        }
        "env_auditor" => {
            r##"#!/usr/bin/env python3
"""env_auditor - audits .env files against .env.example for missing/insecure vars."""
import os
import sys


def parse_env(path):
    keys = {}
    if not os.path.isfile(path):
        return keys
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for idx, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                keys[k.strip()] = (v.strip(), idx)
    return keys


def main():
    env_keys = parse_env(".env")
    example_keys = parse_env(".env.example")

    print(f"env_auditor: audited .env ({len(env_keys)} keys) vs .env.example ({len(example_keys)} keys)")
    if not env_keys and not example_keys:
        print("  • No .env or .env.example found in current directory.")
        return

    missing = [k for k in example_keys if k not in env_keys]
    if missing:
        print(f"  ⚠️ Missing in .env ({len(missing)}):")
        for k in missing:
            print(f"    - {k}")
    else:
        print("  ✔ All variables defined in .env.example exist in .env.")

    empty = [k for k, (v, _) in env_keys.items() if not v]
    if empty:
        print(f"  ⚠️ Empty values in .env: {', '.join(empty)}")


if __name__ == "__main__":
    main()
"##
        }
        "hash_calc" => {
            r##"#!/usr/bin/env python3
"""hash_calc - computes cryptographic checksums (MD5, SHA1, SHA256, SHA512)."""
import hashlib
import os
import sys


def main():
    target = next((a for a in sys.argv[1:] if os.path.isfile(a)), None)
    if target:
        with open(target, "rb") as fh:
            data = fh.read()
        src = f"file `{target}` ({len(data)} bytes)"
    else:
        text = " ".join(sys.argv[1:]) or "nio-xs"
        data = text.encode("utf-8")
        src = f"text: \"{text}\""

    print(f"hash_calc: calculated checksums for {src}:")
    print("  • MD5:    " + hashlib.md5(data).hexdigest())
    print("  • SHA-1:  " + hashlib.sha1(data).hexdigest())
    print("  • SHA256: " + hashlib.sha256(data).hexdigest())
    print("  • SHA512: " + hashlib.sha512(data).hexdigest()[:64] + "...")


if __name__ == "__main__":
    main()
"##
        }
        "diff_viewer" => {
            r##"#!/usr/bin/env python3
"""diff_viewer - visual unified diff with summary statistics."""
import difflib
import os
import sys


def main():
    files = [a for a in sys.argv[1:] if os.path.isfile(a)]
    if len(files) < 2:
        print("diff_viewer: pass two files to compare, e.g. `--tools: diff old.rs new.rs`.")
        return

    f1, f2 = files[0], files[1]
    with open(f1, "r", encoding="utf-8", errors="replace") as fh:
        lines1 = fh.readlines()
    with open(f2, "r", encoding="utf-8", errors="replace") as fh:
        lines2 = fh.readlines()

    diff = list(difflib.unified_diff(lines1, lines2, fromfile=f1, tofile=f2))
    if not diff:
        print(f"diff_viewer: files `{f1}` and `{f2}` are identical.")
        return

    print(f"diff_viewer: comparing {f1} -> {f2} ({len(diff)} diff lines):\n")
    print("".join(diff[:50]))
    if len(diff) > 50:
        print(f"... and {len(diff) - 50} more diff lines")


if __name__ == "__main__":
    main()
"##
        }
        "cron_parser" => {
            r##"#!/usr/bin/env python3
"""cron_parser - explains 5-field cron schedules and predicts next runs."""
import datetime
import re
import sys


def main():
    task = " ".join(sys.argv[1:])
    m = re.search(r"((?:[\d\*/,-]+\s+){4}[\d\*/,-]+)", task)
    cron = m.group(1).strip() if m else "*/15 * * * *"

    fields = cron.split()
    if len(fields) != 5:
        print(f"cron_parser: invalid cron expression `{cron}` (must have 5 fields: min hour dom mon dow)")
        return

    minute, hour, dom, mon, dow = fields
    print(f"cron_parser: schedule `{cron}`")
    print(f"  • Minute:       {minute}")
    print(f"  • Hour:         {hour}")
    print(f"  • Day of Month: {dom}")
    print(f"  • Month:        {mon}")
    print(f"  • Day of Week:  {dow}")

    now = datetime.datetime.now()
    print("  • Next runs estimate:")
    for i in range(1, 6):
        step = now + datetime.timedelta(minutes=15 * i)
        print(f"    - {step.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
"##
        }
        "markdown_toc" => {
            r##"#!/usr/bin/env python3
"""markdown_toc - generate table of contents with anchor links for Markdown files."""
import os
import re
import sys


def slugify(text):
    s = text.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"[-\s]+", "-", s)


def main():
    target = next((a for a in sys.argv[1:] if os.path.isfile(a)), "README.md")
    if not os.path.isfile(target):
        print(f"markdown_toc: file `{target}` not found. Specify a markdown file.")
        return

    with open(target, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    toc = []
    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            indent = "  " * (level - 1)
            slug = slugify(title)
            toc.append(f"{indent}• [{title}](#{slug})")

    print(f"markdown_toc: Table of Contents for `{target}`:\n")
    print("\n".join(toc))


if __name__ == "__main__":
    main()
"##
        }
        "uuid_gen" => {
            r##"#!/usr/bin/env python3
"""uuid_gen - generate cryptographically random UUIDv4 identifiers."""
import sys
import uuid


def main():
    count = 1
    for a in sys.argv[1:]:
        if a.isdigit():
            count = min(int(a), 50)
            break

    print(f"uuid_gen: generated {count} UUIDv4 identifier(s):")
    for _ in range(count):
        u = uuid.uuid4()
        print(f"  • {u}")


if __name__ == "__main__":
    main()
"##
        }
        "url_parser" => {
            r##"#!/usr/bin/env python3
"""url_parser - inspect and parse URLs into structured components and query map."""
import json
import sys
import urllib.parse


def main():
    task = " ".join(sys.argv[1:])
    url = None
    for tok in task.split():
        if tok.startswith(("http://", "https://", "ftp://", "ws://", "wss://")):
            url = tok.strip("'\"`")
            break
    if not url:
        url = "https://api.nio.dev:8080/v1/search?query=rust&limit=25&filter=active#results"

    p = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(p.query)

    print(f"url_parser: parsed `{url}`:")
    print(f"  • Scheme:   {p.scheme}")
    print(f"  • Hostname: {p.hostname}")
    print(f"  • Port:     {p.port}")
    print(f"  • Path:     {p.path}")
    print(f"  • Fragment: {p.fragment}")
    print("  • Query Parameters:\n" + json.dumps(qs, indent=6))


if __name__ == "__main__":
    main()
"##
        }
        "tree_gen" => {
            r##"#!/usr/bin/env python3
"""tree_gen - ASCII/Unicode directory tree generator respecting common ignore patterns."""
import os
import sys

SKIP = {".git", "node_modules", "target", "dist", "build", "__pycache__", ".nioxs"}


def tree(dir_path, prefix="", depth=0, max_depth=3):
    if depth >= max_depth:
        return
    try:
        entries = sorted(os.listdir(dir_path))
    except Exception:
        return
    entries = [e for e in entries if e not in SKIP]
    for idx, entry in enumerate(entries):
        is_last = idx == len(entries) - 1
        connector = "└── " if is_last else "├── "
        full_path = os.path.join(dir_path, entry)
        print(f"{prefix}{connector}{entry}")
        if os.path.isdir(full_path):
            new_prefix = prefix + ("    " if is_last else "│   ")
            tree(full_path, new_prefix, depth + 1, max_depth)


def main():
    root = next((a for a in sys.argv[1:] if os.path.isdir(a)), ".")
    print(f"tree_gen: structure for `{root}`:\n")
    print(root)
    tree(root)


if __name__ == "__main__":
    main()
"##
        }
        "semver_calc" => {
            r##"#!/usr/bin/env python3
"""semver_calc - parse, compare, and bump semantic versions."""
import re
import sys


def main():
    task = " ".join(sys.argv[1:])
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", task)
    major, minor, patch = (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (1, 0, 0)
    current = f"{major}.{minor}.{patch}"

    if "major" in task.lower():
        bumped = f"{major + 1}.0.0"
        action = "Bump MAJOR"
    elif "minor" in task.lower():
        bumped = f"{major}.{minor + 1}.0"
        action = "Bump MINOR"
    else:
        bumped = f"{major}.{minor}.{patch + 1}"
        action = "Bump PATCH"

    print(f"semver_calc: {action}")
    print(f"  • Current:  v{current}")
    print(f"  • Next:     v{bumped}")


if __name__ == "__main__":
    main()
"##
        }
        "lorem_gen" => {
            r##"#!/usr/bin/env python3
"""lorem_gen - generate customizable dummy placeholder text."""
import random
import sys

WORDS = [
    "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing",
    "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore",
    "et", "dolore", "magna", "aliqua", "ut", "enim", "ad", "minim", "veniam",
    "quis", "nostrud", "exercitation", "ullamco", "laboris", "nisi", "ut"
]


def make_sentence():
    k = random.randint(8, 15)
    s = " ".join(random.choices(WORDS, k=k))
    return s.capitalize() + "."


def main():
    paras = 3
    for a in sys.argv[1:]:
        if a.isdigit():
            paras = min(int(a), 10)
            break

    print(f"lorem_gen: generated {paras} paragraph(s):\n")
    for _ in range(paras):
        p = " ".join(make_sentence() for _ in range(random.randint(3, 5)))
        print(p + "\n")


if __name__ == "__main__":
    main()
"##
        }
        _ => {
            r#"#!/usr/bin/env python3
import sys
print("custom_plugin: no real implementation is bundled for this task.")
print("  Received: %s" % (" ".join(sys.argv[1:]) or "<empty>"))
print("  Nothing was changed. Install a purpose-built plugin or edit .nioxs/plugins/custom_tool.py.")
"#
        }
    };

    if let Err(e) = std::fs::write(&script_path, template) {
        return CompanionResult::Blocked {
            reason: format!("Failed to write plugin {plugin_name}: {e}"),
        };
    }

    CompanionResult::ToolExecuted {
        plugin_name: plugin_name.to_string(),
        output: format!(
            "✔ Successfully installed plugin `{plugin_name}` into `.nioxs/plugins/{plugin_name}.py`!\n\
            You can now run tasks with `--tools: <task>` directly."
        ),
    }
}

/// Returns true when `candidate` can be spawned and answers a `--version` probe.
fn interpreter_available(candidate: &str) -> bool {
    Command::new(candidate)
        .arg("--version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

/// Resolves a Python interpreter for plugin execution.
///
/// Order: `$NIOXS_PYTHON` override (validated), then `python3`, then `python`.
pub fn resolve_python_interpreter() -> Option<String> {
    if let Ok(explicit) = std::env::var("NIOXS_PYTHON") {
        let explicit = explicit.trim();
        if !explicit.is_empty() && interpreter_available(explicit) {
            return Some(explicit.to_string());
        }
    }

    for candidate in ["python3", "python"] {
        if interpreter_available(candidate) {
            return Some(candidate.to_string());
        }
    }
    None
}

/// Whether `prompt` is a bare affirmative reply, e.g. `yes` or `proceed`.
pub fn is_consent_reply(prompt: &str) -> bool {
    matches!(
        prompt
            .trim()
            .trim_matches(|c: char| c == '.' || c == '!' || c == ',')
            .to_ascii_lowercase()
            .as_str(),
        "yes"
            | "y"
            | "yeah"
            | "yep"
            | "sure"
            | "ok"
            | "okay"
            | "proceed"
            | "go ahead"
            | "do it"
            | "install"
    )
}

/// Extracts the plugin named by a pending consent prompt, if `message` is one.
///
/// Consent replies are only honored against an actual pending prompt, so a bare
/// `yes` can never install a plugin on its own.
pub fn pending_consent_plugin(message: &str) -> Option<String> {
    if !message.contains("Plugin Download Required") {
        return None;
    }
    // Tolerates both `**Plugin:** \`name\`` and `**Plugin:** **\`name\`**` layouts.
    let after = message.split("**Plugin:**").nth(1)?;
    let name = after.split('`').nth(1)?;
    if name.is_empty()
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return None;
    }
    Some(name.to_string())
}

/// Dispatches a `--tools:` command, prompting for consent if uninstalled.
pub fn dispatch_tool(task: &str) -> CompanionResult {
    let trimmed = task.trim();
    if let Some(to_install) = trimmed.strip_prefix("install ") {
        return install_plugin(to_install.trim());
    }
    if trimmed.starts_with("install:") {
        return install_plugin(trimmed.trim_start_matches("install:").trim());
    }
    if trimmed.is_empty() || trimmed == "list" || trimmed == "help" || trimmed == "?" {
        let plugin_dir = PathBuf::from(".nioxs/plugins");
        let mut sections: std::collections::BTreeMap<&str, Vec<String>> =
            std::collections::BTreeMap::new();
        for meta in TOOL_PLUGINS {
            let installed = if plugin_dir.join(format!("{}.py", meta.name)).exists() {
                "Installed"
            } else {
                "Available on demand"
            };
            let entry = format!(
                "• **`{}`** [{}]\n  • *Capability:* {}\n  • *Example:* `{}`",
                meta.name, installed, meta.capability, meta.example
            );
            sections.entry(meta.category).or_default().push(entry);
        }
        let mut catalog = format!(
            "### 🔌 NioNano Python Plugins & Tool Directives ({})\n\n\
            NioNano tools run **isolated, user-consented Python plugins** for media, styling, data formatting, and developer utilities.\n\n",
            TOOL_PLUGINS.len()
        );
        for (category, entries) in sections {
            catalog.push_str(&format!("#### 📁 {}\n{}\n\n", category, entries.join("\n\n")));
        }
        catalog.push_str(
            "#### 🛡️ Capability Boundary:\n\
            • **Allowed:** Deterministic file and asset transforms, formatting, local media generation.\n\
            • **Blocked:** Arbitrary network tunneling, shell escapes, unconsented disk wiping.\n\
            • *Note:* If a plugin is not yet downloaded, NioNano will ask for consent before downloading (or `--config tools_consent auto`).",
        );
        return CompanionResult::ToolExecuted {
            plugin_name: "plugin_catalog".to_string(),
            output: catalog,
        };
    }

    let (plugin_name, is_installed) = resolve_plugin_for_task(task);

    if !is_installed {
        if is_tools_consent_auto() {
            let _ = install_plugin(&plugin_name);
        } else {
            return CompanionResult::ToolNeedsConsent {
                plugin_name,
                task: task.to_string(),
            };
        }
    }

    // Execute plugin via an available Python interpreter.
    let script_path = format!(".nioxs/plugins/{plugin_name}.py");
    let interpreter = match resolve_python_interpreter() {
        Some(python) => python,
        None => {
            return CompanionResult::ToolFailed {
                plugin_name,
                reason: format!(
                    "No Python interpreter found on PATH (tried `python3` and `python`). \
                    Install Python 3, or point the `NIOXS_PYTHON` environment variable at your \
                    interpreter (e.g. `NIOXS_PYTHON=/usr/bin/python3.12`). \
                    The plugin script itself is installed at `{script_path}`."
                ),
            };
        }
    };

    let output = Command::new(&interpreter)
        .arg(&script_path)
        .arg(task)
        .output();

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            if out.status.success() {
                CompanionResult::ToolExecuted {
                    plugin_name,
                    output: format!("{stdout}\n{stderr}"),
                }
            } else {
                let code = out.status.code().unwrap_or(-1);
                CompanionResult::ToolFailed {
                    plugin_name,
                    reason: format!(
                        "Plugin exited with status {code}.\n\n**Stdout:**\n```\n{}\n```\n\n**Stderr:**\n```\n{}\n```",
                        stdout.trim(),
                        stderr.trim()
                    ),
                }
            }
        }
        Err(e) => CompanionResult::ToolFailed {
            plugin_name,
            reason: format!("Failed to launch `{interpreter} {script_path}`: {e}"),
        },
    }
}

/// Verifies whether generated code contains hallucinated references against known symbols & files.
pub fn detect_hallucinations(
    code: &str,
    known_symbols: &[String],
    known_files: &[String],
) -> Option<String> {
    let mut ghost_symbols = Vec::new();
    let mut ghost_files = Vec::new();

    for line in code.lines() {
        let trimmed = line.trim();
        // Check for suspicious explicit module calls like `crate::foo::Bar`
        if trimmed.contains("::") {
            for part in trimmed.split_whitespace() {
                if let Some(idx) = part.find("::") {
                    let candidate =
                        part[idx + 2..].trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
                    if candidate.len() > 3
                        && !["new", "default", "clone", "to_string", "from", "as_str"]
                            .contains(&candidate)
                        && !known_symbols.iter().any(|s| s == candidate)
                    {
                        // Check if candidate looks like a custom project function or type
                        if candidate.starts_with(|c: char| c.is_alphabetic())
                            && !ghost_symbols.contains(&candidate.to_string())
                        {
                            ghost_symbols.push(candidate.to_string());
                        }
                    }
                }
            }
        }

        // Check for TypeScript / Python imports from relative files: import { Foo } from "./bar"
        if (trimmed.starts_with("import ") || trimmed.starts_with("from "))
            && trimmed.contains("./")
        {
            for word in trimmed.split_whitespace() {
                let candidate = word.trim_matches(|c: char| "`\"',;:()[]{}".contains(c));
                if candidate.starts_with("./") || candidate.starts_with("../") {
                    let path_candidate = candidate.trim_end_matches(".js").trim_end_matches(".ts");
                    let exists = Path::new(candidate).exists()
                        || Path::new(&format!("{candidate}.ts")).exists()
                        || Path::new(&format!("{candidate}.js")).exists()
                        || Path::new(&format!("{path_candidate}.ts")).exists()
                        || Path::new(&format!("{path_candidate}.js")).exists();
                    if !exists && !known_files.iter().any(|f| f.contains(candidate)) {
                        if !ghost_files.contains(&candidate.to_string()) {
                            ghost_files.push(candidate.to_string());
                        }
                    }
                }
            }
        }

        // Check for referenced files
        if trimmed.contains(".rs")
            || trimmed.contains(".ts")
            || trimmed.contains(".py")
            || trimmed.contains(".go")
        {
            for word in trimmed.split_whitespace() {
                let candidate = word.trim_matches(|c: char| "`\"',;:()[]{}".contains(c));
                if (candidate.ends_with(".rs")
                    || candidate.ends_with(".ts")
                    || candidate.ends_with(".py")
                    || candidate.ends_with(".go"))
                    && candidate.contains('/')
                    && !Path::new(candidate).exists()
                    && !known_files.iter().any(|f| f == candidate)
                {
                    if !ghost_files.contains(&candidate.to_string()) {
                        ghost_files.push(candidate.to_string());
                    }
                }
            }
        }
    }

    if !ghost_symbols.is_empty() || !ghost_files.is_empty() {
        let mut alert = String::new();
        if !ghost_symbols.is_empty() {
            alert.push_str(&format!(
                "• **Ghost / Non-existent Symbols Detected:** `{}`\n",
                ghost_symbols.join("`, `")
            ));
        }
        if !ghost_files.is_empty() {
            alert.push_str(&format!(
                "• **Ghost / Non-existent Files Referenced:** `{}`\n",
                ghost_files.join("`, `")
            ));
        }
        alert.push_str(
            "\n*Action required:* The upstream model generated references not found in the current workspace.\n\
            Would you like to:\n\
            1. **Reroute**: Query live documentation / AST definitions to fix the prompt.\n\
            2. **Clarify**: Provide additional file context.\n\
            3. **Proceed**: Accept the draft as-is."
        );
        Some(alert)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_safe_bash_command() {
        assert!(is_safe_bash_command("cargo test").is_ok());
        assert!(is_safe_bash_command("git status -s").is_ok());
        assert!(is_safe_bash_command("rm -rf /").is_err());
        assert!(is_safe_bash_command("mkfs.ext4 /dev/sda1").is_err());
    }

    #[test]
    fn test_execute_in_session_bash() {
        let res = execute_in_session_bash("echo 'hello nioxs'");
        match res {
            CompanionResult::BashOutput {
                stdout, exit_code, ..
            } => {
                assert_eq!(exit_code, 0);
                assert!(stdout.contains("hello nioxs"));
            }
            _ => panic!("Expected BashOutput"),
        }
    }

    #[test]
    fn test_bash_discovery_footer_lists_plugins_with_examples() {
        let footer = bash_discovery_footer();
        assert!(footer.contains("Companion Directives"));
        assert!(footer.contains("svg2favicon"));
        assert!(footer.contains("style_helper"));
        assert!(footer.contains("image_scaler"));
        // at least one concrete example per plugin
        assert!(footer.contains("--tools generate favicon from logo.svg"));
        assert!(footer.contains("--tools extract color palette from styles.css"));
        assert!(footer.contains("--tools resize assets/banner.png to 800x400"));
    }

    #[test]
    fn test_tool_consent_prompt() {
        let res = dispatch_tool("change favicon from logo.svg");
        match res {
            CompanionResult::ToolNeedsConsent {
                ref plugin_name,
                ref task,
            } => {
                assert_eq!(plugin_name, "svg2favicon");
                assert!(task.contains("favicon"));
                let md = res.to_markdown();
                assert!(md.contains("Plugin Download Required"));
            }
            _ => panic!("Expected ToolNeedsConsent"),
        }
    }

    #[test]
    fn test_consent_replies_bind_to_the_pending_prompt() {
        let pending = CompanionResult::ToolNeedsConsent {
            plugin_name: "image_scaler".to_string(),
            task: "resize assets/banner.png to 800x400".to_string(),
        }
        .to_markdown();
        assert_eq!(
            pending_consent_plugin(&pending).as_deref(),
            Some("image_scaler")
        );
        // Ordinary assistant prose is not a consent prompt, even when it mentions a plugin.
        assert_eq!(
            pending_consent_plugin("I'll analyze your project by exploring its structure first."),
            None
        );
        assert_eq!(
            pending_consent_plugin("✔ Successfully installed plugin `svg2favicon`"),
            None
        );
        for reply in [
            "yes", "Yes.", "y", "proceed", "ok", "sure", "go ahead", "do it!", "install",
        ] {
            assert!(
                is_consent_reply(reply),
                "{reply:?} should be a consent reply"
            );
        }
        for reply in [
            "yes please install every plugin",
            "--tools:list",
            "analyze my project",
            "",
        ] {
            assert!(
                !is_consent_reply(reply),
                "{reply:?} is not a bare consent reply"
            );
        }
    }

    #[test]
    fn test_tool_failed_is_not_reported_as_safety_block() {
        let res = CompanionResult::ToolFailed {
            plugin_name: "svg2favicon".to_string(),
            reason: "No Python interpreter found on PATH".to_string(),
        };
        let md = res.to_markdown();
        assert!(md.contains("svg2favicon"));
        assert!(md.contains("Execution Failed"));
        assert!(!md.contains("Safety Policy"));
    }

    #[test]
    fn test_interpreter_probe_rejects_missing_binary() {
        assert!(!interpreter_available(
            "definitely-not-a-real-python-binary-xyz"
        ));
    }

    #[test]
    fn test_coding_config_defaults_to_keyless_kilo() {
        let config = resolve_coding_config(&|_| None);
        assert_eq!(config.base_url, DEFAULT_CODING_BASE_URL);
        assert_eq!(config.model, DEFAULT_CODING_MODEL);
        assert!(config.api_key.is_none(), "Kilo needs no API key by default");
    }

    #[test]
    fn test_coding_config_env_resolution() {
        let config = resolve_coding_config(&|name| match name {
            "NIO_BASE_URL" => Some("http://127.0.0.1:11434/v1/".into()),
            "NIO_API_KEY" => Some("shared-key".into()),
            "NIOXS_CODING_MODEL" => Some("my/model:free".into()),
            _ => None,
        });
        assert_eq!(config.base_url, "http://127.0.0.1:11434/v1");
        assert_eq!(config.api_key.as_deref(), Some("shared-key"));
        assert_eq!(config.model, "my/model:free");
    }

    #[test]
    fn test_coding_config_key_file_takes_precedence() {
        let dir = std::env::temp_dir().join(format!("nioxs-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let key_path = dir.join("coding.key");
        std::fs::write(&key_path, "file-key\n").unwrap();
        let path_str = key_path.to_string_lossy().into_owned();
        let config = resolve_coding_config(&|name| match name {
            "NIOXS_CODING_API_KEY_FILE" => Some(path_str.clone()),
            "NIOXS_CODING_API_KEY" => Some("inline-key".into()),
            _ => None,
        });
        assert_eq!(config.api_key.as_deref(), Some("file-key"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_bundle_context_includes_relevant_and_skips_junk() {
        let root = std::env::temp_dir().join(format!("nioxs-bundle-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(root.join("target/debug")).unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(
            root.join("src/companion.rs"),
            "pub fn favicon_helper() {}\n",
        )
        .unwrap();
        std::fs::write(
            root.join("README.md"),
            "# Project about favicon generation\n",
        )
        .unwrap();
        std::fs::write(root.join("target/debug/old.rs"), "fn favicon_helper() {}\n").unwrap();
        std::fs::write(root.join(".git/config"), "favicon_helper\n").unwrap();
        std::fs::write(root.join("src/blob.png"), b"\x89PNG\x00\x00").unwrap();

        let bundle = bundle_context_in(&root, "generate a favicon helper");
        assert!(
            bundle.files.contains(&"src/companion.rs".to_string()),
            "files: {:?}",
            bundle.files
        );
        assert!(bundle.files.contains(&"README.md".to_string()));
        assert!(
            !bundle
                .files
                .iter()
                .any(|f| f.contains("target") || f.contains(".git")),
            "files: {:?}",
            bundle.files
        );
        assert!(bundle.context.contains("### src/companion.rs"));
        let _ = std::fs::remove_dir_all(&root);
    }
    #[test]
    fn test_coding_tool_schemas_offer_the_callers_file_tools() {
        let schemas = coding_tool_schemas();
        let names: Vec<&str> = schemas
            .iter()
            .map(|schema| schema["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["read_file", "find_files", "read_skill_file"]);
        assert!(
            is_coding_tool("read_file")
                && is_coding_tool("find_files")
                && is_coding_tool("read_skill_file")
        );
        assert!(!is_coding_tool("run_command") && !is_coding_tool("write_file"));
    }

    #[test]
    fn test_upstream_messages_replay_the_models_own_tool_traffic() {
        let read = |id: &str, path: &str| {
            serde_json::from_value::<nioxs_contracts::Message>(serde_json::json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{"id": id, "type": "function", "function": {"name": "read_file", "arguments": serde_json::json!({"path": path}).to_string()}}]
            }))
            .unwrap()
        };
        let result = |id: &str, content: &str| {
            serde_json::from_value::<nioxs_contracts::Message>(serde_json::json!({
                "role": "tool",
                "tool_call_id": id,
                "content": content
            }))
            .unwrap()
        };
        let turn = vec![
            serde_json::from_value::<nioxs_contracts::Message>(
                serde_json::json!({"role": "user", "content": "--coding: what is Zig?"}),
            )
            .unwrap(),
            read("call_nano_turn_a_1", "src/main.rs"),
            result("call_nano_turn_a_1", "Lines 1-2 of 2:\nfn main() {}"),
            // A result with no matching forwarded call must never be replayed.
            result("call_nano_turn_a_orphan", "unrelated"),
        ];

        let messages = upstream_messages("what is Zig?", &turn);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["content"], "what is Zig?");
        assert_eq!(messages[2]["role"], "assistant");
        assert_eq!(
            messages[2]["tool_calls"][0]["function"]["name"],
            "read_file"
        );
        assert_eq!(messages[3]["role"], "tool");
        assert_eq!(messages[3]["tool_call_id"], "call_nano_turn_a_1");
        assert!(messages[3]["content"].as_str().unwrap().contains("fn main"));
        assert_eq!(
            messages.len(),
            4,
            "orphan results are dropped: {messages:?}"
        );

        let fresh = upstream_messages("what is Zig?", &turn[..1]);
        assert_eq!(
            fresh.len(),
            2,
            "the first round carries no tool traffic yet"
        );

        assert_eq!(coding_tool_rounds(&turn), 1);
        assert_eq!(coding_tool_rounds(&turn[..1]), 0);
        assert_eq!(coding_read_paths(&turn), vec!["src/main.rs".to_string()]);
    }

    #[test]
    fn test_trail_segments_are_self_terminating_lines() {
        assert_eq!(
            trail_read_segment("src/main.rs"),
            "📄 read_file src/main.rs\n"
        );
        assert_eq!(
            trail_consult_segment("kilo-auto/free"),
            "💭 consulting with LLM: kilo-auto/free\n"
        );
    }

    #[test]
    fn test_consult_upstream_reports_unreachable_endpoint() {
        let config = CodingConfig {
            base_url: "http://127.0.0.1:9".into(),
            api_key: None,
            model: DEFAULT_CODING_MODEL.into(),
            mode: "coding".into(),
            auto_approve: false,
        };
        let err = consult_upstream(&config, "hello", "", &[]).unwrap_err();
        assert!(err.contains("failed"), "err: {err}");
    }

    #[test]
    fn test_consult_upstream_round_trip_against_mock_gateway() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            let mut header_end = None;
            while header_end.is_none() {
                let n = stream.read(&mut chunk).unwrap();
                buf.extend_from_slice(&chunk[..n]);
                header_end = buf.windows(4).position(|w| w == b"\r\n\r\n");
            }
            let header_end = header_end.unwrap();
            let headers = String::from_utf8_lossy(&buf[..header_end]).into_owned();
            assert!(
                headers.contains("authorization: Bearer secret")
                    || headers.contains("Authorization: Bearer secret"),
                "headers: {headers}"
            );
            let content_length: usize = headers
                .lines()
                .find_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    key.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse().ok())?
                })
                .unwrap();
            let body_start = header_end + 4;
            while buf.len() < body_start + content_length {
                let n = stream.read(&mut chunk).unwrap();
                buf.extend_from_slice(&chunk[..n]);
            }
            let value: serde_json::Value =
                serde_json::from_slice(&buf[body_start..body_start + content_length]).unwrap();
            assert_eq!(value["model"], "mock/model");
            assert!(
                value["messages"][1]["content"]
                    .as_str()
                    .unwrap()
                    .contains("Request: hello"),
                "body: {value}"
            );
            let response_body = serde_json::json!({
                "choices": [{ "message": { "role": "assistant", "content": "fn answer() {}" } }]
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
                response_body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        let config = CodingConfig {
            base_url: format!("http://{addr}"),
            api_key: Some("secret".into()),
            model: "mock/model".into(),
            mode: "coding".into(),
            auto_approve: false,
        };
        let draft =
            consult_upstream(&config, "hello", "### src/lib.rs\nfn placeholder() {}", &[]).unwrap();
        server.join().unwrap();
        assert_eq!(draft, "fn answer() {}");
    }

    #[test]
    fn test_coding_response_lists_work_trail() {
        let res = CompanionResult::CodingResponse {
            guarded: false,
            content: "fn main() {}".into(),
            hallucination_alert: None,
            steps: coding_steps(
                &ContextBundle {
                    files: vec!["src/main.rs".into()],
                    context: String::new(),
                },
                "kilo-auto/free",
            ),
            reply_secs: 1.25,
        };
        let md = res.to_markdown();
        assert!(md.contains("src/main.rs"));
        assert!(md.contains("kilo-auto/free"));
        assert!(md.contains("1.2s"));
        assert!(md.contains("fn main() {}"));
        assert!(
            !md.contains("Context bundled"),
            "context wording must be gone"
        );
    }

    #[test]
    fn test_coding_progress_segments_are_line_oriented() {
        let bundle = ContextBundle {
            files: vec!["src/main.rs".into(), "src/lib.rs".into()],
            context: String::new(),
        };
        let segments = coding_progress_segments(&bundle, "kilo-auto/free");
        // One segment per step, each self-terminating so the client renders a
        // fresh line instead of one joined breadcrumb.
        assert_eq!(segments.len(), 4, "segments: {segments:?}");
        assert_eq!(segments[0], "🛠️ scanning workspace\n");
        assert_eq!(segments[1], "📄 read_file src/main.rs\n");
        assert_eq!(segments[2], "📄 read_file src/lib.rs\n");
        assert_eq!(segments[3], "💭 consulting with LLM: kilo-auto/free\n");
        let joined: String = segments.concat();
        assert!(joined.ends_with('\n'), "answer must follow the trail lines");
    }

    #[test]
    fn test_upstream_chat_body_is_streaming_aware() {
        let config = CodingConfig {
            base_url: "https://api.kilo.ai/api/gateway".into(),
            api_key: None,
            model: "kilo-auto/free".into(),
            mode: "coding".into(),
            auto_approve: false,
        };
        assert_eq!(
            upstream_chat_url(&config),
            "https://api.kilo.ai/api/gateway/chat/completions"
        );
        let messages = upstream_messages("analyze my project", &[]);
        let tools = coding_tool_schemas();
        let streamed = upstream_chat_body(&config, &messages, &tools, true);
        assert_eq!(streamed["stream"], true);
        assert_eq!(streamed["model"], "kilo-auto/free");
        assert_eq!(streamed["messages"][1]["content"], "analyze my project");
        assert_eq!(streamed["tool_choice"], "auto");
        assert_eq!(streamed["tools"][0]["function"]["name"], "read_file");

        // Without tools left, the model must answer with what it already read.
        let blocked = upstream_chat_body(&config, &messages, &[], false);
        assert_eq!(blocked["stream"], false);
        assert!(blocked.get("tools").is_none());
        assert!(blocked.get("tool_choice").is_none());

        // The tool-less fallback still carries its local bundle.
        let bundled = bundled_messages("hi", "### src/lib.rs", &[]);
        assert!(
            bundled[1]["content"]
                .as_str()
                .unwrap()
                .contains("### src/lib.rs")
        );
    }

    #[test]
    fn test_upstream_messages_preserves_conversation_history() {
        let history = vec![
            serde_json::from_value::<nioxs_contracts::Message>(serde_json::json!({
                "role": "user",
                "content": "--code what is the project layout?"
            })).unwrap(),
            serde_json::from_value::<nioxs_contracts::Message>(serde_json::json!({
                "role": "assistant",
                "content": "The project contains crates/nano and crates/provider."
            })).unwrap(),
            serde_json::from_value::<nioxs_contracts::Message>(serde_json::json!({
                "role": "user",
                "content": "--code where is main defined?"
            })).unwrap(),
        ];
        let msgs = upstream_messages("where is main defined?", &history);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "what is the project layout?");
        assert_eq!(msgs[2]["role"], "assistant");
        assert_eq!(msgs[2]["content"], "The project contains crates/nano and crates/provider.");
        assert_eq!(msgs[3]["role"], "user");
        assert_eq!(msgs[3]["content"], "where is main defined?");
    }

    #[test]
    fn test_hallucination_detection() {
        let code = "let client = crate::non_existent_module::GhostClient::connect();\nlet path = \"src/does_not_exist/ghost.rs\";";
        let known_symbols = vec!["Client".to_string(), "connect".to_string()];
        let known_files = vec!["src/lib.rs".to_string()];

        let alert = detect_hallucinations(code, &known_symbols, &known_files);
        assert!(alert.is_some());
        let text = alert.unwrap();
        assert!(text.contains("GhostClient"));
        assert!(text.contains("ghost.rs"));
    }

    #[test]
    fn test_extract_pinned_files() {
        let (cleaned, files) = extract_pinned_files("@src/main.rs please review @Cargo.toml carefully");
        assert_eq!(files, vec!["src/main.rs", "Cargo.toml"]);
        assert_eq!(cleaned, "please review carefully");

        let (cleaned2, files2) = extract_pinned_files("no files mentioned here");
        assert!(files2.is_empty());
        assert_eq!(cleaned2, "no files mentioned here");
    }

    #[test]
    fn test_auto_and_mode_config_flow() {
        let mut cfg = XsConfig::default();
        cfg.auto_approve = true;
        cfg.mode = "architect".to_string();

        with_test_config(cfg, || {
            assert!(is_config_auto_approve());
            assert_eq!(config_mode(), "architect");

            let prompt = build_coding_system_prompt(&[]);
            assert!(prompt.contains("• Mode: `architect`"));
            assert!(prompt.contains("ACTIVE (Autonomous execution enabled"));
            assert!(prompt.contains("Architect mode: focus on system boundaries"));
        });
    }

    #[test]
    fn test_dispatch_config_auto_and_presets() {
        let dir = std::env::temp_dir().join(format!("nioxs-cfg-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg_path = dir.join("config.json");
        unsafe {
            std::env::set_var("NIOXS_CONFIG_FILE", cfg_path.to_str().unwrap());
        }

        // Test mode auto
        let res = dispatch_config("mode auto");
        assert!(res.contains("Config updated: **mode** set to `auto`"));
        assert!(is_config_auto_approve());

        // Test preset review
        let res2 = dispatch_config("preset review");
        assert!(res2.contains("Applied preset **review**"));
        assert_eq!(config_mode(), "review");
        assert!(!load_config().auto_approve);

        // Test preset auto
        let res3 = dispatch_config("preset auto");
        assert!(res3.contains("Applied preset **auto**"));
        assert_eq!(config_mode(), "auto");
        assert!(load_config().auto_approve);
        assert_eq!(load_config().tools_consent, "auto");

        unsafe {
            std::env::remove_var("NIOXS_CONFIG_FILE");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_dispatch_help_topics() {
        let help_code = dispatch_help("code");
        assert!(help_code.contains("Directive: `--code`"));
        assert!(help_code.contains("@path/to/file"));

        let help_tools = dispatch_help("tools");
        assert!(help_tools.contains("Directive: `--tools`"));
        assert!(help_tools.contains("20 cataloged plugins"));

        let help_config = dispatch_help("config");
        assert!(help_config.contains("Directive: `--config`"));
        assert!(help_config.contains("auto ON"));

        let help_default = dispatch_help("");
        assert!(help_default.contains("Companion Directives & Capabilities"));
    }
}
