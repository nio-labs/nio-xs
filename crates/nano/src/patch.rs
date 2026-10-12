use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PatchOperation {
    pub file_path: String,
    pub target_text: String,
    pub replacement_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PatchResult {
    Success {
        file_path: String,
        replaced_count: usize,
    },
    NotFound {
        file_path: String,
    },
    Ambiguous {
        file_path: String,
        occurrences: usize,
    },
    Error(String),
}

/// Applies an exact, deterministic patch replacing `target_text` with `replacement_text` in `original_content`.
/// Requires exact matching and unique occurrence by default to avoid unintended edits.
pub fn apply_patch_to_string(
    content: &str,
    target: &str,
    replacement: &str,
    allow_multiple: bool,
) -> Result<(String, usize), String> {
    if target.is_empty() {
        return Err("target text to replace cannot be empty".into());
    }

    let occurrences = content.matches(target).count();
    if occurrences == 0 {
        return Err("target text not found in source".into());
    }

    if occurrences > 1 && !allow_multiple {
        return Err(format!(
            "target text appears {occurrences} times; exact patch requires unique match or allow_multiple"
        ));
    }

    let patched = if allow_multiple {
        content.replace(target, replacement)
    } else {
        content.replacen(target, replacement, 1)
    };

    Ok((patched, occurrences))
}

/// A multi-file atomic patch operation containing targets and replacements per file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MultiFilePatch {
    pub id: String,
    pub description: String,
    pub operations: Vec<PatchOperation>,
}

/// A snapshot backup of original files taken before applying a patch, allowing instant rollback.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UndoSnapshot {
    pub patch_id: String,
    pub timestamp_unix: u64,
    pub backups: Vec<(String, String)>, // (file_path, original_content)
}

/// Validates that all patch operations match uniquely in their respective files without modifying disk.
pub fn dry_run_multi_patch(
    root: &std::path::Path,
    patch: &MultiFilePatch,
) -> Result<Vec<(String, String)>, String> {
    let mut planned = Vec::new();
    for op in &patch.operations {
        let full = root.join(&op.file_path);
        let content = std::fs::read_to_string(&full)
            .map_err(|e| format!("cannot read {}: {e}", op.file_path))?;
        let (patched, _) = apply_patch_to_string(&content, &op.target_text, &op.replacement_text, false)?;
        planned.push((op.file_path.clone(), patched));
    }
    Ok(planned)
}

/// Applies a multi-file patch atomically, backing up original contents to `.nioxs/undo/<patch_id>.json`.
pub fn apply_multi_patch(
    root: &std::path::Path,
    patch: &MultiFilePatch,
) -> Result<UndoSnapshot, String> {
    // 1. Dry-run first to ensure no partial applications
    let planned = dry_run_multi_patch(root, patch)?;

    let mut backups = Vec::new();
    for op in &patch.operations {
        let full = root.join(&op.file_path);
        let orig = std::fs::read_to_string(&full).unwrap_or_default();
        backups.push((op.file_path.clone(), orig));
    }

    let snapshot = UndoSnapshot {
        patch_id: patch.id.clone(),
        timestamp_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        backups,
    };

    // Save snapshot to .nioxs/undo/
    let undo_dir = root.join(".nioxs/undo");
    let _ = std::fs::create_dir_all(&undo_dir);
    let snap_data = serde_json::to_string_pretty(&snapshot).unwrap_or_default();
    let _ = std::fs::write(undo_dir.join(format!("{}.json", patch.id)), snap_data);

    // Write all planned changes
    for (file, content) in planned {
        let full = root.join(&file);
        std::fs::write(&full, content).map_err(|e| format!("failed writing {file}: {e}"))?;
    }

    Ok(snapshot)
}

/// Rolls back files modified by a previous patch using its undo snapshot.
pub fn rollback_patch(root: &std::path::Path, patch_id: &str) -> Result<usize, String> {
    let snap_path = root.join(format!(".nioxs/undo/{patch_id}.json"));
    let data = std::fs::read_to_string(&snap_path)
        .map_err(|_| format!("undo snapshot for '{patch_id}' not found"))?;
    let snapshot: UndoSnapshot = serde_json::from_str(&data)
        .map_err(|e| format!("corrupt snapshot: {e}"))?;

    let count = snapshot.backups.len();
    for (file, original) in snapshot.backups {
        let full = root.join(&file);
        std::fs::write(&full, original).map_err(|e| format!("failed restoring {file}: {e}"))?;
    }
    let _ = std::fs::remove_file(snap_path);
    Ok(count)
}

/// Scaffolds standard deterministic project files
pub fn scaffold_template(template_name: &str) -> Option<(&'static str, &'static str)> {
    match template_name {
        "gitignore" | ".gitignore" => Some((
            ".gitignore",
            "# Generated by NioNano MicroLM (μLM)\ntarget/\nnode_modules/\n.env\n*.log\n.DS_Store\n",
        )),
        "ci" | "github-ci" => Some((
            ".github/workflows/ci.yml",
            "name: CI\non: [push, pull_request]\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - run: cargo test --workspace\n",
        )),
        "license-mit" | "mit" => Some((
            "LICENSE",
            "MIT License\n\nCopyright (c) 2026 NioLabs\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the \"Software\"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so.\n",
        )),
        _ => None,
    }
}/// Represents the result of an automated TDD auto-fix loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoFixReport {
    pub success: bool,
    pub attempts: usize,
    pub original_patch_id: String,
    pub applied_patch_ids: Vec<String>,
    pub diagnostics: Vec<String>,
    pub output: String,
}

/// Executes an autonomous self-correction TDD loop:
/// 1. Applies candidate patch.
/// 2. Runs the verification command.
/// 3. If failing, extracts diagnostics and invokes `fixer` to generate a candidate correction up to `max_attempts`.
/// 4. If all attempts fail, rolls back all applied patches and returns the diagnostic report.
pub fn execute_autofix_tdd_loop<F>(
    workspace: &Path,
    initial_patch: &MultiFilePatch,
    test_command: &str,
    max_attempts: usize,
    mut fixer: F,
) -> Result<AutoFixReport, String>
where
    F: FnMut(&str, &str) -> Option<MultiFilePatch>,
{
    let mut applied_patches = Vec::new();

    // 1. Apply initial patch
    apply_multi_patch(workspace, initial_patch)?;
    applied_patches.push(initial_patch.id.clone());

    let max_retries = max_attempts.clamp(1, 10);
    let mut current_attempt = 1;
    let mut last_output;
    let mut collected_diags = Vec::new();

    loop {
        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(test_command)
            .current_dir(workspace)
            .output()
            .map_err(|e| format!("failed to run test command '{test_command}': {e}"))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let full_output = format!("{}\n{}", stdout, stderr);
        last_output = full_output.clone();

        if output.status.success() {
            return Ok(AutoFixReport {
                success: true,
                attempts: current_attempt,
                original_patch_id: initial_patch.id.clone(),
                applied_patch_ids: applied_patches,
                diagnostics: collected_diags,
                output: last_output,
            });
        }

        let diags = crate::brain::parse_compiler_diagnostics(&full_output);
        let diag_summary = if !diags.is_empty() {
            diags.iter().map(|d| format!("{:?}: {}", d.code, d.message)).collect::<Vec<_>>().join("; ")
        } else {
            full_output.lines().take(5).collect::<Vec<_>>().join("\n")
        };
        collected_diags.push(format!("Attempt {current_attempt} failure: {diag_summary}"));

        if current_attempt >= max_retries {
            break;
        }

        if let Some(next_patch) = fixer(&full_output, &diag_summary) {
            current_attempt += 1;
            apply_multi_patch(workspace, &next_patch)?;
            applied_patches.push(next_patch.id.clone());
        } else {
            break;
        }
    }

    // Rollback all applied patches in reverse order upon failure
    for pid in applied_patches.iter().rev() {
        let _ = rollback_patch(workspace, pid);
    }

    Ok(AutoFixReport {
        success: false,
        attempts: current_attempt,
        original_patch_id: initial_patch.id.clone(),
        applied_patch_ids: applied_patches,
        diagnostics: collected_diags,
        output: last_output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_patch_and_rollback() {
        let temp_dir = std::env::temp_dir().join(format!("nioxs-patch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let file_a = "file_a.txt";
        let file_b = "file_b.txt";
        std::fs::write(temp_dir.join(file_a), "hello apple\n").unwrap();
        std::fs::write(temp_dir.join(file_b), "hello orange\n").unwrap();

        let patch = MultiFilePatch {
            id: "patch_test_1".into(),
            description: "fruit upgrade".into(),
            operations: vec![
                PatchOperation {
                    file_path: file_a.into(),
                    target_text: "apple".into(),
                    replacement_text: "crisp apple".into(),
                },
                PatchOperation {
                    file_path: file_b.into(),
                    target_text: "orange".into(),
                    replacement_text: "sweet orange".into(),
                },
            ],
        };

        let snapshot = apply_multi_patch(&temp_dir, &patch).expect("apply patch");
        assert_eq!(snapshot.backups.len(), 2);
        assert_eq!(std::fs::read_to_string(temp_dir.join(file_a)).unwrap(), "hello crisp apple\n");
        assert_eq!(std::fs::read_to_string(temp_dir.join(file_b)).unwrap(), "hello sweet orange\n");

        let rolled_back = rollback_patch(&temp_dir, "patch_test_1").expect("rollback");
        assert_eq!(rolled_back, 2);
        assert_eq!(std::fs::read_to_string(temp_dir.join(file_a)).unwrap(), "hello apple\n");
        assert_eq!(std::fs::read_to_string(temp_dir.join(file_b)).unwrap(), "hello orange\n");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_unique_patch_succeeds() {
        let original = "fn main() {\n    println!(\"hello\");\n}\n";
        let target = "println!(\"hello\");";
        let replacement = "println!(\"hello world!\");";

        let (patched, count) = apply_patch_to_string(original, target, replacement, false).unwrap();
        assert_eq!(count, 1);
        assert_eq!(patched, "fn main() {\n    println!(\"hello world!\");\n}\n");
    }

    #[test]
    fn test_missing_patch_target_fails() {
        let original = "fn main() {}";
        let target = "non_existent()";
        let err = apply_patch_to_string(original, target, "something", false).unwrap_err();
        assert!(err.contains("not found"));
    }

    #[test]
    fn test_ambiguous_patch_requires_flag() {
        let original = "foo\nfoo\n";
        let err = apply_patch_to_string(original, "foo", "bar", false).unwrap_err();
        assert!(err.contains("appears 2 times"));

        let (patched, count) = apply_patch_to_string(original, "foo", "bar", true).unwrap();
        assert_eq!(count, 2);
        assert_eq!(patched, "bar\nbar\n");
    }

    #[test]
    fn test_template_scaffolding() {
        let (name, content) = scaffold_template("gitignore").unwrap();
        assert_eq!(name, ".gitignore");
        assert!(content.contains("target/"));
    }

    #[test]
    fn test_execute_autofix_tdd_loop() {
        let temp_dir = std::env::temp_dir().join(format!("nioxs-autofix-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let code_file = "check.txt";
        std::fs::write(temp_dir.join(code_file), "initial broken state").unwrap();

        let initial_patch = MultiFilePatch {
            id: "initial_1".into(),
            description: "first attempt".into(),
            operations: vec![PatchOperation {
                file_path: code_file.into(),
                target_text: "initial broken state".into(),
                replacement_text: "intermediate failing state".into(),
            }],
        };

        // Shell command checks if file contains "fixed state"
        let test_cmd = "grep -q 'fixed state' check.txt";

        let mut fix_called = 0;
        let report = execute_autofix_tdd_loop(&temp_dir, &initial_patch, test_cmd, 3, |_out, _diags| {
            fix_called += 1;
            Some(MultiFilePatch {
                id: format!("remedy_{fix_called}"),
                description: "fix attempt".into(),
                operations: vec![PatchOperation {
                    file_path: code_file.into(),
                    target_text: "intermediate failing state".into(),
                    replacement_text: "fixed state".into(),
                }],
            })
        })
        .expect("autofix run");

        assert!(report.success);
        assert_eq!(report.attempts, 2);
        assert_eq!(report.applied_patch_ids.len(), 2);
        assert_eq!(std::fs::read_to_string(temp_dir.join(code_file)).unwrap(), "fixed state");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}

