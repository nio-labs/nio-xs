//! # Rust Protector & Healer Engine (The Immunity Subsystem)
//!
//! Provides deterministic protection against hallucinations, path escapes,
//! and unsafe shell commands, and automated compiler-feedback auto-repair
//! with atomic transactional rollbacks.

use crate::compiler::{CompilerDiagnostic, run_project_check};
use crate::patch::{MultiFilePatch, apply_multi_patch, rollback_patch};
use serde::{Deserialize, Serialize};
use std::path::Path;

// ============================================================================
// 1. THE PROTECTOR (Security, Sandbox, & Anti-Hallucination Guard)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProtectorVerdict {
    pub passed: bool,
    pub safe_paths: bool,
    pub safe_bash: bool,
    pub hallucination_alert: Option<String>,
    pub blocked_reason: Option<String>,
}

/// Validates whether a file path is strictly within the trusted project boundary.
pub fn protect_file_write(path: &str) -> Result<(), String> {
    if !crate::safe_path(path, false) {
        return Err(format!(
            "🛡️ [Protector Blocked]: Path '{path}' is outside the trusted project scope or targets sensitive files."
        ));
    }
    Ok(())
}

/// Validates whether a bash command is permitted without dangerous operations.
pub fn protect_bash_command(command: &str) -> Result<(), String> {
    crate::companion::is_safe_bash_command(command).map_err(|err| {
        format!("🛡️ [Protector Blocked]: Command '{command}' was blocked: {err}")
    })
}

/// Inspects generated code for non-existent symbols, packages, or files
/// by verifying against the real project AST symbol call graph.
pub fn protect_code_against_hallucinations(
    code: &str,
    known_symbols: &[String],
    known_files: &[String],
) -> Result<(), String> {
    if let Some(alert) = crate::companion::detect_hallucinations(code, known_symbols, known_files) {
        return Err(format!("🛡️ [Protector Anti-Hallucination Alert]: {alert}"));
    }
    Ok(())
}

/// Comprehensive inspection of a model's proposed actions before execution.
pub fn inspect_proposal(
    file_paths: &[String],
    code_content: &str,
    bash_command: Option<&str>,
    known_symbols: &[String],
    known_files: &[String],
) -> ProtectorVerdict {
    // 1. Verify paths
    let mut safe_paths = true;
    let mut blocked_reason = None;
    for path in file_paths {
        if let Err(err) = protect_file_write(path) {
            safe_paths = false;
            blocked_reason = Some(err);
            break;
        }
    }

    // 2. Verify bash command if present
    let mut safe_bash = true;
    if let Some(cmd) = bash_command {
        if let Err(err) = protect_bash_command(cmd) {
            safe_bash = false;
            if blocked_reason.is_none() {
                blocked_reason = Some(err);
            }
        }
    }

    // 3. Verify AST hallucination
    let hallucination_alert =
        crate::companion::detect_hallucinations(code_content, known_symbols, known_files);

    let passed = safe_paths && safe_bash && hallucination_alert.is_none();

    ProtectorVerdict {
        passed,
        safe_paths,
        safe_bash,
        hallucination_alert,
        blocked_reason,
    }
}

// ============================================================================
// 2. THE HEALER (Compiler Diagnostic Triage & Auto-Fix TDD Repair Loop)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealingResult {
    pub success: bool,
    pub attempts: usize,
    pub applied_patch_ids: Vec<String>,
    pub diagnostics: Vec<String>,
    pub rolled_back: bool,
    pub message: String,
}

/// Applies an initial patch to a workspace and iteratively repairs any compiler
/// errors or test failures by feeding diagnostics to `fixer` in a bounded loop.
/// If repair succeeds, changes remain applied.
/// If repair fails, all modifications are atomically rolled back to leave the
/// workspace in its clean original state.
pub fn execute_healer_loop<F>(
    workspace: &Path,
    initial_patch: &MultiFilePatch,
    test_or_check_cmd: Option<&str>,
    max_retries: usize,
    mut fixer: F,
) -> Result<HealingResult, String>
where
    F: FnMut(&str, &[CompilerDiagnostic]) -> Option<MultiFilePatch>,
{
    // Apply initial patch
    apply_multi_patch(workspace, initial_patch)?;
    let mut applied_patches = vec![initial_patch.id.clone()];
    let mut collected_diags = Vec::new();
    let mut current_attempt = 1;

    loop {
        // Run check command (custom or auto-detected cargo check / tsc / python)
        let check_result = match test_or_check_cmd {
            Some(cmd) => {
                let output = std::process::Command::new("sh")
                    .arg("-c")
                    .arg(cmd)
                    .current_dir(workspace)
                    .output()
                    .map_err(|e| format!("failed to run check command '{cmd}': {e}"))?;

                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let full = format!("{}\n{}", stdout, stderr);

                if output.status.success() {
                    Ok(full)
                } else {
                    Err(full)
                }
            }
            None => {
                let root_str = workspace.to_string_lossy();
                match run_project_check(&root_str) {
                    Ok(()) => Ok("Project checks passed with 0 diagnostics".into()),
                    Err(diags) => {
                        let summary = diags
                            .iter()
                            .map(|d| format!("{}:{}: [{}] {}", d.file_path, d.line_number, d.error_code, d.message))
                            .collect::<Vec<_>>()
                            .join("\n");
                        Err(summary)
                    }
                }
            }
        };

        match check_result {
            Ok(_) => {
                // Success! Keep the changes.
                return Ok(HealingResult {
                    success: true,
                    attempts: current_attempt,
                    applied_patch_ids: applied_patches,
                    diagnostics: collected_diags,
                    rolled_back: false,
                    message: format!("✔ Healer confirmed: changes compile and pass checks in {current_attempt} attempt(s)."),
                });
            }
            Err(failure_output) => {
                let diags: Vec<CompilerDiagnostic> = failure_output
                    .lines()
                    .take(10)
                    .map(|l| CompilerDiagnostic {
                        file_path: "".into(),
                        line_number: 0,
                        error_code: "COMPILE_ERROR".into(),
                        message: l.to_string(),
                        label: "".into(),
                        suggested_replacement: None,
                    })
                    .collect();

                collected_diags.push(format!("Attempt {current_attempt}: {}", failure_output.lines().take(3).collect::<Vec<_>>().join(" | ")));

                if current_attempt >= max_retries {
                    break;
                }

                // Call the fixer closure to generate the next corrective patch
                if let Some(next_patch) = fixer(&failure_output, &diags) {
                    current_attempt += 1;
                    apply_multi_patch(workspace, &next_patch)?;
                    applied_patches.push(next_patch.id.clone());
                } else {
                    break;
                }
            }
        }
    }

    // Retries exhausted or no fix available: atomically rollback all applied patches
    for pid in applied_patches.iter().rev() {
        let _ = rollback_patch(workspace, pid);
    }

    Ok(HealingResult {
        success: false,
        attempts: current_attempt,
        applied_patch_ids: applied_patches,
        diagnostics: collected_diags,
        rolled_back: true,
        message: format!("🩹 [Healer Rollback]: Changes failed compiler verification after {current_attempt} attempt(s). Workspace reverted atomically to pristine state."),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::PatchOperation;
    use std::fs;

    #[test]
    fn test_protector_blocks_sensitive_paths() {
        assert!(protect_file_write("src/main.rs").is_ok());
        assert!(protect_file_write("../outside.rs").is_err());
        assert!(protect_file_write(".env").is_err());
        assert!(protect_file_write(".ssh/id_rsa").is_err());
    }

    #[test]
    fn test_protector_blocks_unsafe_bash() {
        assert!(protect_bash_command("cargo check").is_ok());
        assert!(protect_bash_command("git status").is_ok());
        assert!(protect_bash_command("rm -rf /").is_err());
        assert!(protect_bash_command("sudo reboot").is_err());
    }

    #[test]
    fn test_protector_detects_ast_hallucination() {
        let code = "use non_existent_pkg::magic_fn;\nmagic_fn();";
        let symbols = vec!["known_fn".to_string()];
        let files = vec!["src/lib.rs".to_string()];
        let alert = protect_code_against_hallucinations(code, &symbols, &files);
        assert!(alert.is_err());
        assert!(alert.unwrap_err().contains("magic_fn"));
    }

    #[test]
    fn test_healer_loop_rolls_back_on_unresolvable_error() {
        let temp_dir = std::env::temp_dir().join(format!("nioxs_healer_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let file_path = "sample.txt";
        fs::write(temp_dir.join(file_path), "original clean state\n").unwrap();

        let broken_patch = MultiFilePatch {
            id: "broken_patch_1".into(),
            description: "introduces error".into(),
            operations: vec![PatchOperation {
                file_path: file_path.into(),
                target_text: "clean".into(),
                replacement_text: "broken".into(),
            }],
        };

        // Command "grep -q 'clean' sample.txt" fails when replaced by 'broken'
        let check_cmd = "grep -q 'clean' sample.txt";

        let result = execute_healer_loop(
            &temp_dir,
            &broken_patch,
            Some(check_cmd),
            2,
            |_output, _diags| None, // No fix provided
        )
        .unwrap();

        assert_eq!(result.success, false);
        assert_eq!(result.rolled_back, true);

        // Verify workspace is pristine
        let content = fs::read_to_string(temp_dir.join(file_path)).unwrap();
        assert_eq!(content, "original clean state\n");

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_healer_loop_succeeds_with_corrective_fix() {
        let temp_dir = std::env::temp_dir().join(format!("nioxs_healer_success_{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let file_path = "sample.txt";
        fs::write(temp_dir.join(file_path), "initial\n").unwrap();

        let initial_patch = MultiFilePatch {
            id: "initial_patch".into(),
            description: "step 1".into(),
            operations: vec![PatchOperation {
                file_path: file_path.into(),
                target_text: "initial".into(),
                replacement_text: "flawed".into(),
            }],
        };

        // Needs 'perfect' to pass
        let check_cmd = "grep -q 'perfect' sample.txt";

        let mut fix_attempt = 0;
        let result = execute_healer_loop(
            &temp_dir,
            &initial_patch,
            Some(check_cmd),
            3,
            |_output, _diags| {
                fix_attempt += 1;
                Some(MultiFilePatch {
                    id: format!("fix_{fix_attempt}"),
                    description: "healing patch".into(),
                    operations: vec![PatchOperation {
                        file_path: file_path.into(),
                        target_text: "flawed".into(),
                        replacement_text: "perfect".into(),
                    }],
                })
            },
        )
        .unwrap();

        assert_eq!(result.success, true);
        assert_eq!(result.rolled_back, false);
        assert_eq!(result.attempts, 2);

        let content = fs::read_to_string(temp_dir.join(file_path)).unwrap();
        assert_eq!(content, "perfect\n");

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
