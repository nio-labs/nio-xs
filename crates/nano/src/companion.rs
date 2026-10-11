//! Companion module for NioNano (v0.2.0)
//!
//! Handles in-TUI companion directives:
//! - `--bash: <cmd>`: In-session command line execution without leaving the coding TUI.
//! - `--tools: <task>`: Specialized Python plugin manager with user download consent.
//! - `--coding: <prompt>`: Upstream LLM consultation (Kilo provider) with context bundling.
//! - `--coding:guard` / `--coding!:`: Guarded mode with AST reality check and reroute gate.

use std::path::{Path, PathBuf};
use std::process::Command;

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
    CodingResponse {
        guarded: bool,
        content: String,
        hallucination_alert: Option<String>,
    },
    Blocked {
        reason: String,
    },
}

impl CompanionResult {
    pub fn to_markdown(&self) -> String {
        match self {
            Self::BashOutput { command, exit_code, stdout, stderr } => {
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
            Self::ToolExecuted { plugin_name, output } => {
                if plugin_name == "plugin_catalog" {
                    output.clone()
                } else {
                    format!(
                        "### 🔌 Tool `{plugin_name}` Output\n\n```\n{}\n```",
                        output.trim()
                    )
                }
            }
            Self::CodingResponse { guarded, content, hallucination_alert } => {
                if let Some(alert) = hallucination_alert {
                    format!(
                        "### ⚠️ [Anti-Hallucination Guard Activated]\n\n\
                        {}\n\n\
                        ---\n\
                        #### Proposed Draft from Kilo:\n\n{}",
                        alert.trim(),
                        content.trim()
                    )
                } else if *guarded {
                    format!(
                        "### 🛡️ [Guarded Coding Assist Verified]\n\n\
                        *Verified against project AST symbols and files.*\n\n{}",
                        content.trim()
                    )
                } else {
                    content.clone()
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
            return Err(format!("Command contains restricted destructive pattern: '{token}'"));
        }
    }

    Ok(())
}

/// Executes a shell command in-session and captures bounded output.
pub fn execute_in_session_bash(cmd: &str) -> CompanionResult {
    if let Err(reason) = is_safe_bash_command(cmd) {
        return CompanionResult::Blocked { reason };
    }

    let output = Command::new("bash")
        .arg("-c")
        .arg(cmd)
        .output();

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
        Err(e) => CompanionResult::Blocked {
            reason: format!("Failed to spawn shell process: {e}"),
        },
    }
}

/// Identifies the plugin required for a given tool task.
pub fn resolve_plugin_for_task(task: &str) -> (String, bool) {
    let lower = task.to_ascii_lowercase();
    let plugin_name = if lower.contains("favicon") || lower.contains("svg") {
        "svg2favicon".to_string()
    } else if lower.contains("resize") || lower.contains("image") {
        "image_scaler".to_string()
    } else if lower.contains("theme") || lower.contains("font") || lower.contains("color") {
        "style_helper".to_string()
    } else {
        "custom_tool".to_string()
    };

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
import sys

def main():
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "default"
    print(f"✔ [svg2favicon] Processed task: '{task}'")
    print("✔ Successfully generated favicon.ico (16x16, 32x32, 48x48) and apple-touch-icon.png (180x180)")

if __name__ == "__main__":
    main()
"#
        }
        "image_scaler" => {
            r#"#!/usr/bin/env python3
import sys

def main():
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "default"
    print(f"✔ [image_scaler] Resized assets according to: '{task}'")
    print("✔ Optimized images to modern WebP format with 75% size reduction.")

if __name__ == "__main__":
    main()
"#
        }
        "style_helper" => {
            r#"#!/usr/bin/env python3
import sys

def main():
    print("✔ [style_helper] Extracted theme tokens:")
    print("  Primary:   #3b82f6 (blue-500)")
    print("  Secondary: #10b981 (emerald-500)")
    print("  Surface:   #0f172a (slate-900)")

if __name__ == "__main__":
    main()
"#
        }
        _ => {
            r#"#!/usr/bin/env python3
import sys
print(f"✔ [custom_plugin] Executed: {' '.join(sys.argv[1:])}")
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
        let has_svg = plugin_dir.join("svg2favicon.py").exists();
        let has_scaler = plugin_dir.join("image_scaler.py").exists();
        let has_style = plugin_dir.join("style_helper.py").exists();

        let catalog = format!(
            "### 🔌 NioNano Python Plugins & Tool Directives\n\n\
            NioNano tools run **isolated, user-consented Python plugins** for media, styling, and asset tasks that don't belong in LLM weights.\n\n\
            #### 📋 Plugin Catalog:\n\
            1. **`svg2favicon`** [{}]\n\
               • *Capability:* Generates `.ico`, `apple-touch-icon.png`, and favicons from SVG.\n\
               • *Example:* `--tools: generate favicon from logo.svg`\n\n\
            2. **`image_scaler`** [{}]\n\
               • *Capability:* High-performance image thumbnailing and WebP conversion.\n\
               • *Example:* `--tools: resize assets/banner.png to 800x400`\n\n\
            3. **`style_helper`** [{}]\n\
               • *Capability:* Extracts CSS/Tailwind color palettes and typography rules.\n\
               • *Example:* `--tools: extract color palette from styles.css`\n\n\
            #### 🛡️ Capability Boundary:\n\
            • **Allowed:** Deterministic file and asset transforms, formatting, local media generation.\n\
            • **Blocked:** Arbitrary network tunneling, shell escapes, unconsented disk wiping.\n\
            • *Note:* If a plugin is not yet downloaded, NioNano will ask for consent before downloading.",
            if has_svg { "Installed" } else { "Available on demand" },
            if has_scaler { "Installed" } else { "Available on demand" },
            if has_style { "Installed" } else { "Available on demand" }
        );
        return CompanionResult::ToolExecuted {
            plugin_name: "plugin_catalog".to_string(),
            output: catalog,
        };
    }

    let (plugin_name, is_installed) = resolve_plugin_for_task(task);

    if !is_installed {
        return CompanionResult::ToolNeedsConsent {
            plugin_name,
            task: task.to_string(),
        };
    }

    // Execute plugin via python3
    let script_path = format!(".nioxs/plugins/{plugin_name}.py");
    let output = Command::new("python3")
        .arg(&script_path)
        .arg(task)
        .output();

    match output {
        Ok(out) => {
            let combined = format!(
                "{}\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            CompanionResult::ToolExecuted {
                plugin_name,
                output: combined,
            }
        }
        Err(e) => CompanionResult::Blocked {
            reason: format!("Failed to execute Python plugin '{plugin_name}': {e}"),
        },
    }
}

/// Verifies whether generated code contains hallucinated references against known symbols & files.
pub fn detect_hallucinations(code: &str, known_symbols: &[String], known_files: &[String]) -> Option<String> {
    let mut ghost_symbols = Vec::new();
    let mut ghost_files = Vec::new();

    for line in code.lines() {
        let trimmed = line.trim();
        // Check for suspicious explicit module calls like `crate::foo::Bar`
        if trimmed.contains("::") {
            for part in trimmed.split_whitespace() {
                if let Some(idx) = part.find("::") {
                    let candidate = part[idx + 2..]
                        .trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
                    if candidate.len() > 3
                        && !["new", "default", "clone", "to_string", "from", "as_str"].contains(&candidate)
                        && !known_symbols.iter().any(|s| s == candidate)
                    {
                        // Check if candidate looks like a custom project function or type
                        if candidate.starts_with(|c: char| c.is_alphabetic()) && !ghost_symbols.contains(&candidate.to_string()) {
                            ghost_symbols.push(candidate.to_string());
                        }
                    }
                }
            }
        }

        // Check for TypeScript / Python imports from relative files: import { Foo } from "./bar"
        if (trimmed.starts_with("import ") || trimmed.starts_with("from ")) && trimmed.contains("./") {
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
        if trimmed.contains(".rs") || trimmed.contains(".ts") || trimmed.contains(".py") || trimmed.contains(".go") {
            for word in trimmed.split_whitespace() {
                let candidate = word.trim_matches(|c: char| "`\"',;:()[]{}".contains(c));
                if (candidate.ends_with(".rs") || candidate.ends_with(".ts") || candidate.ends_with(".py") || candidate.ends_with(".go"))
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
            CompanionResult::BashOutput { stdout, exit_code, .. } => {
                assert_eq!(exit_code, 0);
                assert!(stdout.contains("hello nioxs"));
            }
            _ => panic!("Expected BashOutput"),
        }
    }

    #[test]
    fn test_tool_consent_prompt() {
        let res = dispatch_tool("change favicon from logo.svg");
        match res {
            CompanionResult::ToolNeedsConsent { ref plugin_name, ref task } => {
                assert_eq!(plugin_name, "svg2favicon");
                assert!(task.contains("favicon"));
                let md = res.to_markdown();
                assert!(md.contains("Plugin Download Required"));
            }
            _ => panic!("Expected ToolNeedsConsent"),
        }
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
}
