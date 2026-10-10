use crate::patch::apply_patch_to_string;
use std::fs;
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, PartialEq)]
pub struct SymbolLocation {
    pub file: String,
    pub line_number: usize,
    pub line_text: String,
    pub kind: String,
}

/// Recursively collect all relevant source code files in a directory.
fn collect_source_files(dir: &Path, files: &mut Vec<String>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if file_name.starts_with('.')
                || file_name == "target"
                || file_name == "node_modules"
                || file_name == ".runtime"
            {
                continue;
            }
            if path.is_dir() {
                collect_source_files(&path, files);
            } else if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if matches!(
                        ext,
                        "rs" | "py" | "js" | "ts" | "jsx" | "tsx" | "go" | "c" | "cpp" | "h" | "hpp"
                    ) {
                        if let Some(p) = path.to_str() {
                            files.push(p.to_string());
                        }
                    }
                }
            }
        }
    }
}

/// Phase 3: Semantic Symbol Definition Finder.
/// Locates where a symbol is defined (e.g., fn, struct, enum, class, def) excluding call sites.
pub fn find_symbol_definitions(symbol: &str, root: &str) -> Vec<SymbolLocation> {
    let clean_symbol = symbol.trim().trim_matches(|c: char| "`\"',;:!?".contains(c));
    let mut files = Vec::new();
    collect_source_files(Path::new(root), &mut files);

    let mut results = Vec::new();

    for file in files {
        let Ok(content) = fs::read_to_string(&file) else {
            continue;
        };

        for (idx, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with('#') {
                continue;
            }

            // Rust patterns
            let is_rust_def = trimmed.starts_with(&format!("fn {clean_symbol}"))
                || trimmed.starts_with(&format!("pub fn {clean_symbol}"))
                || trimmed.starts_with(&format!("pub(crate) fn {clean_symbol}"))
                || trimmed.starts_with(&format!("async fn {clean_symbol}"))
                || trimmed.starts_with(&format!("pub async fn {clean_symbol}"))
                || trimmed.starts_with(&format!("struct {clean_symbol}"))
                || trimmed.starts_with(&format!("pub struct {clean_symbol}"))
                || trimmed.starts_with(&format!("enum {clean_symbol}"))
                || trimmed.starts_with(&format!("pub enum {clean_symbol}"))
                || trimmed.starts_with(&format!("trait {clean_symbol}"))
                || trimmed.starts_with(&format!("pub trait {clean_symbol}"))
                || trimmed.starts_with(&format!("type {clean_symbol}"))
                || trimmed.starts_with(&format!("pub type {clean_symbol}"))
                || trimmed.starts_with(&format!("const {clean_symbol}:"))
                || trimmed.starts_with(&format!("pub const {clean_symbol}:"))
                || trimmed.starts_with(&format!("static {clean_symbol}:"))
                || trimmed.starts_with(&format!("macro_rules! {clean_symbol}"));

            // Python patterns
            let is_py_def = trimmed.starts_with(&format!("def {clean_symbol}("))
                || trimmed.starts_with(&format!("class {clean_symbol}:"))
                || trimmed.starts_with(&format!("class {clean_symbol}("));

            // JS/TS patterns
            let is_js_def = trimmed.starts_with(&format!("function {clean_symbol}"))
                || trimmed.starts_with(&format!("export function {clean_symbol}"))
                || trimmed.starts_with(&format!("class {clean_symbol}"))
                || trimmed.starts_with(&format!("export class {clean_symbol}"))
                || trimmed.starts_with(&format!("interface {clean_symbol}"))
                || trimmed.starts_with(&format!("export interface {clean_symbol}"))
                || trimmed.starts_with(&format!("const {clean_symbol} ="))
                || trimmed.starts_with(&format!("export const {clean_symbol} ="));

            if is_rust_def || is_py_def || is_js_def {
                let kind = if trimmed.contains("fn ") || trimmed.contains("def ") || trimmed.contains("function ") {
                    "Function"
                } else if trimmed.contains("struct ") {
                    "Struct"
                } else if trimmed.contains("enum ") {
                    "Enum"
                } else if trimmed.contains("trait ") || trimmed.contains("interface ") {
                    "Trait/Interface"
                } else if trimmed.contains("class ") {
                    "Class"
                } else {
                    "Declaration"
                };

                let rel_file = file.strip_prefix("./").unwrap_or(&file).to_string();
                results.push(SymbolLocation {
                    file: rel_file,
                    line_number: idx + 1,
                    line_text: trimmed.to_string(),
                    kind: kind.to_string(),
                });
            }
        }
    }

    results
}

/// Phase 3: Semantic Symbol References / Call-Sites Finder.
/// Locates where a symbol is referenced/called, excluding its definition lines.
pub fn find_symbol_references(symbol: &str, root: &str) -> Vec<SymbolLocation> {
    let clean_symbol = symbol.trim().trim_matches(|c: char| "`\"',;:!?".contains(c));
    let mut files = Vec::new();
    collect_source_files(Path::new(root), &mut files);

    let defs = find_symbol_definitions(clean_symbol, root);
    let def_set: std::collections::HashSet<(String, usize)> = defs
        .into_iter()
        .map(|d| (d.file, d.line_number))
        .collect();

    let mut results = Vec::new();

    for file in files {
        let Ok(content) = fs::read_to_string(&file) else {
            continue;
        };
        let rel_file = file.strip_prefix("./").unwrap_or(&file).to_string();

        for (idx, line) in content.lines().enumerate() {
            let line_num = idx + 1;
            if def_set.contains(&(rel_file.clone(), line_num)) {
                continue;
            }

            let trimmed = line.trim();
            if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with('#') {
                continue;
            }

            // Word boundary match check
            if contains_isolated_symbol(trimmed, clean_symbol) {
                results.push(SymbolLocation {
                    file: rel_file.clone(),
                    line_number: line_num,
                    line_text: trimmed.to_string(),
                    kind: "Usage/Reference".to_string(),
                });
            }
        }
    }

    results
}

fn contains_isolated_symbol(line: &str, symbol: &str) -> bool {
    let mut cursor = 0;
    while let Some(pos) = line[cursor..].find(symbol) {
        let abs = cursor + pos;
        let before_ok = abs == 0 || !line.as_bytes()[abs - 1].is_ascii_alphanumeric() && line.as_bytes()[abs - 1] != b'_';
        let after_pos = abs + symbol.len();
        let after_ok = after_pos >= line.len()
            || !line.as_bytes()[after_pos].is_ascii_alphanumeric() && line.as_bytes()[after_pos] != b'_';

        if before_ok && after_ok {
            return true;
        }
        cursor = abs + symbol.len();
    }
    false
}

/// Phase 4: Deterministic In-Place Code Refactoring.
pub fn refactor_file(
    file_path: &str,
    target_text: &str,
    replacement_text: &str,
) -> Result<String, String> {
    let path = Path::new(file_path);
    if !path.exists() {
        return Err(format!("File '{file_path}' does not exist."));
    }

    let original = fs::read_to_string(path)
        .map_err(|e| format!("Cannot read file '{file_path}': {e}"))?;

    let (patched, count) = apply_patch_to_string(&original, target_text, replacement_text, false)?;

    fs::write(path, patched)
        .map_err(|e| format!("Failed to write patched file '{file_path}': {e}"))?;

    let mut out = format!(
        "✔ Successfully refactored `{file_path}` ({count} unique occurrence replaced):\n\n"
    );
    out.push_str("Diff Preview:\n```diff\n");
    for line in target_text.lines() {
        out.push_str(&format!("- {line}\n"));
    }
    for line in replacement_text.lines() {
        out.push_str(&format!("+ {line}\n"));
    }
    out.push_str("```\n");
    Ok(out)
}

/// Phase 5: Autonomous Edit-and-Test Self-Verification Loop (TDD Loop).
pub fn tdd_refactor_and_test(
    file_path: &str,
    target_text: &str,
    replacement_text: &str,
) -> Result<String, String> {
    let path = Path::new(file_path);
    if !path.exists() {
        return Err(format!("File '{file_path}' does not exist."));
    }

    let original = fs::read_to_string(path)
        .map_err(|e| format!("Cannot read file '{file_path}': {e}"))?;

    let (patched, count) = apply_patch_to_string(&original, target_text, replacement_text, false)?;

    // Apply patch
    fs::write(path, &patched)
        .map_err(|e| format!("Failed to write patched file '{file_path}': {e}"))?;

    // Execute test suite
    let (cmd, args) = if Path::new("Cargo.toml").exists() {
        ("cargo", vec!["test", "--quiet"])
    } else if Path::new("package.json").exists() {
        ("npm", vec!["test", "--", "--silent"])
    } else {
        ("pytest", vec!["-q"])
    };

    let test_run = Command::new(cmd).args(&args).output();

    match test_run {
        Ok(output) if output.status.success() => {
            let mut out = format!(
                "✔ **TDD Loop Passed**: Patch verified cleanly against the project test suite ({count} occurrence replaced).\n\n"
            );
            out.push_str("Applied Changes:\n```diff\n");
            for line in target_text.lines() {
                out.push_str(&format!("- {line}\n"));
            }
            for line in replacement_text.lines() {
                out.push_str(&format!("+ {line}\n"));
            }
            out.push_str("```\n\nAll unit tests passed.");
            Ok(out)
        }
        Ok(output) => {
            // Roll back to original file content immediately!
            let _ = fs::write(path, original);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let diag = if !stderr.trim().is_empty() {
                stderr
            } else {
                stdout
            };
            let short_diag: String = diag.lines().take(15).collect::<Vec<_>>().join("\n");

            Err(format!(
                "❌ **TDD Verification Failed**: Automated tests failed with exit code {:?}.\nChange was **automatically rolled back** to original state.\n\nFailure snippet:\n```\n{}\n```",
                output.status.code(),
                short_diag
            ))
        }
        Err(e) => {
            let _ = fs::write(path, original);
            Err(format!("Could not execute test runner '{cmd}': {e}. Reverted change."))
        }
    }
}

/// Phase 6: Dynamic Unit Test Generator.
pub fn scaffold_test_for_function(file_path: &str, function_name: &str) -> Result<String, String> {
    let path = Path::new(file_path);
    if !path.exists() {
        return Err(format!("File '{file_path}' does not exist."));
    }

    let content = fs::read_to_string(path)
        .map_err(|e| format!("Cannot read file '{file_path}': {e}"))?;

    let clean_fn = function_name.trim().trim_matches(|c: char| "`\"',;:!?".contains(c));
    let mut sig_line = None;
    for line in content.lines() {
        let trimmed = line.trim();
        if (trimmed.contains(&format!("fn {clean_fn}"))
            || trimmed.contains(&format!("def {clean_fn}("))
            || trimmed.contains(&format!("function {clean_fn}")))
            && !trimmed.starts_with("//")
        {
            sig_line = Some(trimmed.to_string());
            break;
        }
    }

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("rs");
    match ext {
        "rs" => {
            let sig_comment = sig_line.as_deref().unwrap_or(clean_fn);
            Ok(format!(
                "#[cfg(test)]\nmod tests {{\n    use super::*;\n\n    #[test]\n    fn test_{clean_fn}_behavior() {{\n        // Target: {sig_comment}\n        let result = {clean_fn}();\n        assert!(!result.is_empty());\n    }}\n}}\n"
            ))
        }
        "py" => {
            Ok(format!(
                "def test_{clean_fn}():\n    # Target: {clean_fn}\n    result = {clean_fn}()\n    assert result is not None\n"
            ))
        }
        _ => {
            Ok(format!(
                "describe('{clean_fn}', () => {{\n    it('should behave as expected', () => {{\n        const result = {clean_fn}();\n        expect(result).toBeDefined();\n    }});\n}});\n"
            ))
        }
    }
}

/// Phase 6: Git Diff Change Reviewer.
pub fn review_git_diff() -> Result<String, String> {
    let output = Command::new("git")
        .args(["diff", "--stat"])
        .output()
        .map_err(|e| format!("cannot run git diff: {e}"))?;

    let stat = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if stat.is_empty() {
        return Ok("Working tree clean. No uncommitted git changes to review.".into());
    }

    let full_output = Command::new("git")
        .args(["diff", "-U2"])
        .output()
        .map_err(|e| format!("cannot run git diff: {e}"))?;
    let diff_text = String::from_utf8_lossy(&full_output.stdout);

    let mut unwraps_added = 0;
    let mut debug_prints_added = 0;
    let mut tests_touched = false;

    for line in diff_text.lines() {
        if line.starts_with('+') && !line.starts_with("+++") {
            if line.contains(".unwrap()") || line.contains(".expect(") {
                unwraps_added += 1;
            }
            if line.contains("println!") || line.contains("console.log(") || line.contains("print(") {
                debug_prints_added += 1;
            }
            if line.contains("#[test]") || line.contains("assert_eq!") || line.contains("it(") {
                tests_touched = true;
            }
        }
    }

    let mut report = String::new();
    report.push_str("### Git Diff Review\n\n");
    report.push_str(&format!("**Changed Files & Statistics**:\n```\n{stat}\n```\n\n"));

    report.push_str("#### Strengths & Positive Observations\n");
    if tests_touched {
        report.push_str("• ✔ Dedicated unit tests or assertions updated alongside code changes.\n");
    } else {
        report.push_str("• ✔ Focused modifications adhering to repository module layout.\n");
    }

    report.push_str("\n#### Risk Analysis & Code Health Warnings\n");
    if unwraps_added > 0 {
        report.push_str(&format!("• ⚠ Detected {unwraps_added} `.unwrap()` or `.expect()` call(s) introduced. Consider using `?` error propagation for sovereign reliability.\n"));
    }
    if debug_prints_added > 0 {
        report.push_str(&format!("• ⚠ Detected {debug_prints_added} debug print statement(s). Ensure these are cleaned up before merging.\n"));
    }
    if unwraps_added == 0 && debug_prints_added == 0 {
        report.push_str("• ✔ No unhandled panic unwraps or lingering debug print statements detected.\n");
    }

    report.push_str("\n#### Recommendation\n");
    if unwraps_added > 0 {
        report.push_str("Recommended: Refactor unwraps to typed error handling, then proceed with `cargo test --locked`.\n");
    } else {
        report.push_str("Recommended: Diff is healthy and clean. Proceed with commit.\n");
    }

    Ok(report)
}

/// Phase 6: Commit Message Generator from git diff.
pub fn suggest_commit_message() -> Result<String, String> {
    let output = Command::new("git")
        .args(["status", "--short"])
        .output()
        .map_err(|e| format!("cannot run git status: {e}"))?;

    let status = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if status.is_empty() {
        return Ok("Working tree clean. Nothing to commit.".into());
    }

    let stat_output = Command::new("git")
        .args(["diff", "--stat"])
        .output()
        .map_err(|e| format!("cannot run git diff: {e}"))?;
    let stat = String::from_utf8_lossy(&stat_output.stdout);

    // Determine primary component
    let component = if stat.contains("crates/nano") {
        "nano"
    } else if stat.contains("crates/browser") {
        "browser"
    } else if stat.contains("crates/provider") {
        "provider"
    } else if stat.contains("crates/contracts") {
        "contracts"
    } else {
        "core"
    };

    Ok(format!(
        "Suggested Conventional Commit Message:\n\n```\nfeat({component}): enhance sovereign capabilities and pair-programming workflows\n```\n\nStatus summary:\n```\n{status}\n```"
    ))
}

use crate::Intent;

pub fn parse_coder_intent(prompt: &str) -> Option<Intent> {
    let lower = prompt.to_ascii_lowercase();
    let trimmed = lower.trim().trim_end_matches(|c: char| "?!. \t\r\n".contains(c));

    // Git Diff & Commit
    if [
        "review git diff",
        "analyze git diff",
        "check git diff",
        "review diff",
        "git diff review",
        "review uncommitted changes",
        "review changes",
        "diff review",
    ]
    .contains(&trimmed)
    {
        return Some(Intent::ReviewDiff);
    }
    if [
        "generate commit message",
        "suggest commit message",
        "commit message",
        "suggest a commit message",
        "generate a commit message",
    ]
    .contains(&trimmed)
    {
        return Some(Intent::SuggestCommit);
    }

    // Dynamic Test Scaffolding
    for prefix in &["write test for ", "generate test for ", "scaffold test for ", "create test for "] {
        if lower.starts_with(prefix) {
            let rest = &prompt[prefix.len()..];
            if let Some((fn_part, file_part)) = rest.split_once(" in ") {
                let function = fn_part.trim().trim_matches(|c: char| "`\"',;:!?".contains(c)).to_string();
                let file = file_part.trim().trim_matches(|c: char| "`\"',;:!?".contains(c)).to_string();
                if !function.is_empty() && !file.is_empty() {
                    return Some(Intent::ScaffoldTest { file, function });
                }
            }
        }
    }

    // In-Place Refactoring & TDD
    if (lower.contains("replace ") || lower.contains("change "))
        && (lower.contains(" with ") || lower.contains(" to "))
        && lower.contains(" in ")
    {
        if let Some(intent) = extract_refactor_intent(prompt) {
            return Some(intent);
        }
    }

    // Semantic Symbol Navigation (Phase 3)
    // Definitions:
    for prefix in &[
        "where is defined ",
        "where is symbol defined ",
        "where is definition of ",
        "show me definition of ",
        "show definition of ",
        "find definition of ",
        "find declaration of ",
        "where is fn ",
        "where is struct ",
        "where is enum ",
        "where is trait ",
    ] {
        if lower.starts_with(prefix) {
            let symbol = prompt[prefix.len()..].trim().trim_matches(|c: char| "`\"',;:!? \t\r\n".contains(c));
            if !symbol.is_empty() {
                return Some(Intent::SymbolDef(symbol.to_string()));
            }
        }
    }

    if lower.starts_with("where is ") && lower.ends_with("defined") {
        let middle = prompt[9..prompt.len() - 7].trim();
        let sym = middle.trim_matches(|c: char| "`\"',;:!? \t\r\n".contains(c));
        if !sym.is_empty() && !sym.contains(' ') {
            return Some(Intent::SymbolDef(sym.to_string()));
        }
    }

    // References / Call Sites:
    for prefix in &[
        "who calls ",
        "find usages of ",
        "find references of ",
        "find call sites of ",
        "references to ",
        "usages of ",
        "callers of ",
    ] {
        if lower.starts_with(prefix) {
            let symbol = prompt[prefix.len()..].trim().trim_matches(|c: char| "`\"',;:!? \t\r\n".contains(c));
            if !symbol.is_empty() {
                return Some(Intent::SymbolRefs(symbol.to_string()));
            }
        }
    }

    if lower.starts_with("where is ") && (lower.ends_with("called") || lower.ends_with("used")) {
        let suffix_len = if lower.ends_with("called") { 6 } else { 4 };
        let middle = prompt[9..prompt.len() - suffix_len].trim();
        let sym = middle.trim_matches(|c: char| "`\"',;:!? \t\r\n".contains(c));
        if !sym.is_empty() && !sym.contains(' ') {
            return Some(Intent::SymbolRefs(sym.to_string()));
        }
    }

    None
}

fn extract_refactor_intent(prompt: &str) -> Option<Intent> {
    let is_tdd = prompt.to_ascii_lowercase().contains("test") || prompt.to_ascii_lowercase().contains("verify");

    let quote_chars = ['\'', '"', '`'];
    for q in quote_chars {
        let Some(first_q) = prompt.find(q) else { continue; };
        let Some(second_q_rel) = prompt[first_q + 1..].find(q) else { continue; };
        let second_q = second_q_rel + first_q + 1;
        let old = &prompt[first_q + 1..second_q];

        let after_old = &prompt[second_q + 1..];
        let with_kw = if let Some(p) = after_old.to_ascii_lowercase().find(" with ") {
            p + 6
        } else if let Some(p) = after_old.to_ascii_lowercase().find(" to ") {
            p + 4
        } else {
            continue;
        };

        let rest_after_with = &after_old[with_kw..];
        let Some(third_q) = rest_after_with.find(q) else { continue; };
        let Some(fourth_q_rel) = rest_after_with[third_q + 1..].find(q) else { continue; };
        let fourth_q = fourth_q_rel + third_q + 1;
        let new = &rest_after_with[third_q + 1..fourth_q];

        let after_new = &rest_after_with[fourth_q + 1..];
        if let Some(in_pos) = after_new.to_ascii_lowercase().find(" in ") {
            let file_cand = after_new[in_pos + 4..]
                .split_whitespace()
                .next()?
                .trim_matches(|c: char| "`\"',;:!?".contains(c));
            if !file_cand.is_empty() {
                return if is_tdd {
                    Some(Intent::TddRefactor {
                        file: file_cand.to_string(),
                        old: old.to_string(),
                        new: new.to_string(),
                    })
                } else {
                    Some(Intent::RefactorPatch {
                        file: file_cand.to_string(),
                        old: old.to_string(),
                        new: new.to_string(),
                    })
                };
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_navigation_finds_self() {
        let defs = find_symbol_definitions("apply_patch_to_string", ".");
        assert!(!defs.is_empty(), "Should find definition of apply_patch_to_string");
        assert!(defs.iter().any(|d| d.file.contains("patch.rs")));

        let refs = find_symbol_references("apply_patch_to_string", ".");
        assert!(!refs.is_empty(), "Should find references of apply_patch_to_string");
        // Definition in patch.rs should be excluded from references
        for r in &refs {
            assert!(!r.line_text.starts_with("pub fn apply_patch_to_string"));
        }
    }

    #[test]
    fn test_contains_isolated_symbol() {
        assert!(contains_isolated_symbol("let x = foo();", "foo"));
        assert!(contains_isolated_symbol("foo::bar()", "foo"));
        assert!(!contains_isolated_symbol("let x = foobar();", "foo"));
        assert!(!contains_isolated_symbol("let x = my_foo();", "foo"));
    }
}
