//! # Code Assist Brain for NioNano MicroLM (v0.2.0)
//!
//! Provides structured problem solving, AST/symbol call-graph analysis,
//! compiler diagnostic triage, and contextual assistance for developers.

use serde::{Deserialize, Serialize};

/// A compiler or linter diagnostic parsed from build output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Diagnostic {
    pub code: Option<String>,
    pub message: String,
    pub level: DiagnosticLevel,
    pub file: Option<String>,
    pub line: Option<usize>,
    pub column: Option<usize>,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticLevel {
    Error,
    Warning,
    Note,
    Help,
}

/// A node in the codebase symbol graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SymbolNode {
    pub name: String,
    pub kind: String, // fn, struct, enum, trait, type, class, interface
    pub file: String,
    pub line: usize,
    pub callers: Vec<String>,
    pub callees: Vec<String>,
}

/// Extracts symbols across Rust, TypeScript, JavaScript, Python, and Go.
pub fn extract_symbols_from_source(file: &str, content: &str) -> Vec<SymbolNode> {
    let mut symbols = Vec::new();
    let ext = file.rsplit('.').next().unwrap_or("").to_ascii_lowercase();

    for (idx, line) in content.lines().enumerate() {
        let line_num = idx + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
            continue;
        }

        match ext.as_str() {
            "rs" => {
                if let Some(rest) = trimmed
                    .strip_prefix("pub async fn ")
                    .or_else(|| trimmed.strip_prefix("async fn "))
                    .or_else(|| trimmed.strip_prefix("pub fn "))
                    .or_else(|| trimmed.strip_prefix("fn "))
                {
                    if let Some(name) = rest.split(['(', '<', ' ']).next() {
                        if !name.is_empty() {
                            symbols.push(SymbolNode {
                                name: name.to_string(),
                                kind: "fn".into(),
                                file: file.to_string(),
                                line: line_num,
                                callers: vec![],
                                callees: vec![],
                            });
                        }
                    }
                } else if let Some(rest) = trimmed
                    .strip_prefix("pub struct ")
                    .or_else(|| trimmed.strip_prefix("struct "))
                {
                    if let Some(name) = rest.split(['{', ';', '<', ' ']).next() {
                        if !name.is_empty() {
                            symbols.push(SymbolNode {
                                name: name.to_string(),
                                kind: "struct".into(),
                                file: file.to_string(),
                                line: line_num,
                                callers: vec![],
                                callees: vec![],
                            });
                        }
                    }
                } else if let Some(rest) = trimmed
                    .strip_prefix("pub enum ")
                    .or_else(|| trimmed.strip_prefix("enum "))
                {
                    if let Some(name) = rest.split(['{', '<', ' ']).next() {
                        if !name.is_empty() {
                            symbols.push(SymbolNode {
                                name: name.to_string(),
                                kind: "enum".into(),
                                file: file.to_string(),
                                line: line_num,
                                callers: vec![],
                                callees: vec![],
                            });
                        }
                    }
                } else if let Some(rest) = trimmed
                    .strip_prefix("pub trait ")
                    .or_else(|| trimmed.strip_prefix("trait "))
                {
                    if let Some(name) = rest.split(['{', '<', ' ']).next() {
                        if !name.is_empty() {
                            symbols.push(SymbolNode {
                                name: name.to_string(),
                                kind: "trait".into(),
                                file: file.to_string(),
                                line: line_num,
                                callers: vec![],
                                callees: vec![],
                            });
                        }
                    }
                }
            }
            "ts" | "tsx" | "js" | "jsx" => {
                if let Some(rest) = trimmed
                    .strip_prefix("export function ")
                    .or_else(|| trimmed.strip_prefix("function "))
                    .or_else(|| trimmed.strip_prefix("export async function "))
                    .or_else(|| trimmed.strip_prefix("async function "))
                {
                    if let Some(name) = rest.split(['(', '<', ' ']).next() {
                        if !name.is_empty() {
                            symbols.push(SymbolNode {
                                name: name.to_string(),
                                kind: "function".into(),
                                file: file.to_string(),
                                line: line_num,
                                callers: vec![],
                                callees: vec![],
                            });
                        }
                    }
                } else if let Some(rest) = trimmed
                    .strip_prefix("export class ")
                    .or_else(|| trimmed.strip_prefix("class "))
                {
                    if let Some(name) = rest.split(['{', '<', ' ']).next() {
                        if !name.is_empty() {
                            symbols.push(SymbolNode {
                                name: name.to_string(),
                                kind: "class".into(),
                                file: file.to_string(),
                                line: line_num,
                                callers: vec![],
                                callees: vec![],
                            });
                        }
                    }
                } else if let Some(rest) = trimmed
                    .strip_prefix("export interface ")
                    .or_else(|| trimmed.strip_prefix("interface "))
                {
                    if let Some(name) = rest.split(['{', '<', ' ']).next() {
                        if !name.is_empty() {
                            symbols.push(SymbolNode {
                                name: name.to_string(),
                                kind: "interface".into(),
                                file: file.to_string(),
                                line: line_num,
                                callers: vec![],
                                callees: vec![],
                            });
                        }
                    }
                } else if let Some(rest) = trimmed
                    .strip_prefix("export type ")
                    .or_else(|| trimmed.strip_prefix("type "))
                {
                    if let Some(name) = rest.split(['=', '<', ' ']).next() {
                        if !name.is_empty() {
                            symbols.push(SymbolNode {
                                name: name.to_string(),
                                kind: "type".into(),
                                file: file.to_string(),
                                line: line_num,
                                callers: vec![],
                                callees: vec![],
                            });
                        }
                    }
                }
            }
            "py" => {
                if let Some(rest) = trimmed.strip_prefix("def ") {
                    if let Some(name) = rest.split(['(', ':']).next() {
                        if !name.is_empty() {
                            symbols.push(SymbolNode {
                                name: name.trim().to_string(),
                                kind: "def".into(),
                                file: file.to_string(),
                                line: line_num,
                                callers: vec![],
                                callees: vec![],
                            });
                        }
                    }
                } else if let Some(rest) = trimmed.strip_prefix("class ") {
                    if let Some(name) = rest.split(['(', ':', ' ']).next() {
                        if !name.is_empty() {
                            symbols.push(SymbolNode {
                                name: name.trim().to_string(),
                                kind: "class".into(),
                                file: file.to_string(),
                                line: line_num,
                                callers: vec![],
                                callees: vec![],
                            });
                        }
                    }
                }
            }
            "go" => {
                if let Some(rest) = trimmed.strip_prefix("func ") {
                    // func (r *Receiver) Method() or func FunctionName()
                    let rest = if rest.starts_with('(') {
                        if let Some(idx) = rest.find(')') {
                            rest[idx + 1..].trim_start()
                        } else {
                            rest
                        }
                    } else {
                        rest
                    };
                    if let Some(name) = rest.split(['(', ' ']).next() {
                        if !name.is_empty() {
                            symbols.push(SymbolNode {
                                name: name.to_string(),
                                kind: "func".into(),
                                file: file.to_string(),
                                line: line_num,
                                callers: vec![],
                                callees: vec![],
                            });
                        }
                    }
                } else if let Some(rest) = trimmed.strip_prefix("type ") {
                    let parts: Vec<&str> = rest.split_whitespace().collect();
                    if parts.len() >= 2 {
                        symbols.push(SymbolNode {
                            name: parts[0].to_string(),
                            kind: parts[1].to_string(),
                            file: file.to_string(),
                            line: line_num,
                            callers: vec![],
                            callees: vec![],
                        });
                    }
                }
            }
            _ => {}
        }
    }

    symbols
}

/// The solving and reasoning state machine for developer assistance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistPlan {
    pub title: String,
    pub root_cause: String,
    pub steps: Vec<AssistStep>,
    pub verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistStep {
    pub description: String,
    pub file: Option<String>,
    pub target_symbol: Option<String>,
    pub rationale: String,
}

/// Analyzes compiler output (such as `cargo check --message-format=json`) into structured diagnostics.
pub fn parse_compiler_diagnostics(raw_output: &str) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();

    for line in raw_output.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Try parsing JSON format first
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if val.get("reason").and_then(|r| r.as_str()) == Some("compiler-message") {
                if let Some(msg) = val.get("message") {
                    let level_str = msg.get("level").and_then(|l| l.as_str()).unwrap_or("error");
                    let level = match level_str {
                        "warning" => DiagnosticLevel::Warning,
                        "note" => DiagnosticLevel::Note,
                        "help" => DiagnosticLevel::Help,
                        _ => DiagnosticLevel::Error,
                    };
                    let message = msg
                        .get("message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("")
                        .to_string();
                    let code = msg
                        .get("code")
                        .and_then(|c| c.get("code"))
                        .and_then(|c| c.as_str())
                        .map(str::to_string);

                    let mut file = None;
                    let mut line_num = None;
                    let mut col_num = None;

                    if let Some(spans) = msg.get("spans").and_then(|s| s.as_array()) {
                        if let Some(primary) = spans.iter().find(|s| {
                            s.get("is_primary")
                                .and_then(|p| p.as_bool())
                                .unwrap_or(false)
                        }) {
                            file = primary
                                .get("file_name")
                                .and_then(|f| f.as_str())
                                .map(str::to_string);
                            line_num = primary
                                .get("line_start")
                                .and_then(|l| l.as_u64())
                                .map(|n| n as usize);
                            col_num = primary
                                .get("column_start")
                                .and_then(|c| c.as_u64())
                                .map(|n| n as usize);
                        }
                    }

                    diagnostics.push(Diagnostic {
                        code,
                        message,
                        level,
                        file,
                        line: line_num,
                        column: col_num,
                        suggestion: None,
                    });
                }
            }
        } else if trimmed.starts_with("error[E") {
            // Textual rustc error fallback: error[E0382]: use of moved value...
            let code_end = trimmed.find(']').unwrap_or(trimmed.len());
            let code = if code_end > 6 {
                Some(trimmed[6..code_end].to_string())
            } else {
                None
            };
            let message = trimmed[code_end + 1..]
                .trim_start_matches(':')
                .trim()
                .to_string();

            diagnostics.push(Diagnostic {
                code,
                message,
                level: DiagnosticLevel::Error,
                file: None,
                line: None,
                column: None,
                suggestion: None,
            });
        }
    }

    diagnostics
}

/// Solves and generates an actionable Code Assist plan for a given diagnostic.
pub fn formulate_assist_plan(query: &str, diag: &Diagnostic) -> AssistPlan {
    let mut steps = Vec::new();

    if let Some(code) = &diag.code {
        match code.as_str() {
            "E0382" => {
                steps.push(AssistStep {
                    description: "Clone value prior to closure or move site".into(),
                    file: diag.file.clone(),
                    target_symbol: None,
                    rationale:
                        "Resolves use of moved value by creating an independent owned instance."
                            .into(),
                });
                steps.push(AssistStep {
                    description:
                        "Wrap shared data in Arc<T> or Arc<Mutex<T>> if across async boundaries"
                            .into(),
                    file: diag.file.clone(),
                    target_symbol: None,
                    rationale: "Enables multiple concurrent owners across tasks.".into(),
                });
                return AssistPlan {
                    title: format!("Remediating Borrow Checker Error [{code}]"),
                    root_cause: "Value is moved into a scope or closure and accessed after move."
                        .into(),
                    steps,
                    verified: true,
                };
            }
            "E0499" | "E0502" => {
                steps.push(AssistStep {
                    description: "Shorten scope of mutable borrow".into(),
                    file: diag.file.clone(),
                    target_symbol: None,
                    rationale:
                        "Ensures the mutable reference is dropped before subsequent accesses."
                            .into(),
                });
                return AssistPlan {
                    title: format!("Remediating Multiple Borrow Conflict [{code}]"),
                    root_cause: "Cannot borrow as mutable more than once at a time.".into(),
                    steps,
                    verified: true,
                };
            }
            _ => {}
        }
    }

    steps.push(AssistStep {
        description: format!(
            "Review target site and compiler recommendations: {}",
            diag.message
        ),
        file: diag.file.clone(),
        target_symbol: None,
        rationale: "Address compiler diagnostic with idiomatic patterns.".into(),
    });

    AssistPlan {
        title: format!("Code Assist: {}", query.trim()),
        root_cause: diag.message.clone(),
        steps,
        verified: false,
    }
}

/// A persisted workspace AST symbol index mapping symbols to files, lines, and kinds.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct WorkspaceIndex {
    pub version: u32,
    pub updated_at_unix: u64,
    pub files: Vec<String>,
    pub symbols: Vec<SymbolNode>,
}

impl WorkspaceIndex {
    pub fn new() -> Self {
        Self {
            version: 1,
            updated_at_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            files: Vec::new(),
            symbols: Vec::new(),
        }
    }

    /// Finds files containing the given symbol or related callers/callees.
    pub fn find_files_for_symbol(&self, symbol: &str) -> Vec<String> {
        let sym_l = symbol.to_ascii_lowercase();
        let mut matched_files = Vec::new();
        for node in &self.symbols {
            if node.name.to_ascii_lowercase() == sym_l
                || node.name.to_ascii_lowercase().contains(&sym_l)
            {
                if !matched_files.contains(&node.file) {
                    matched_files.push(node.file.clone());
                }
            }
        }
        matched_files
    }
}

/// Scans the workspace directory and builds a complete WorkspaceIndex.
pub fn scan_workspace_index(root: &std::path::Path) -> WorkspaceIndex {
    let mut index = WorkspaceIndex::new();
    let mut queue = vec![root.to_path_buf()];
    let max_visited = 500;
    let mut visited = 0;

    while let Some(dir) = queue.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            visited += 1;
            if visited > max_visited {
                break;
            }
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            if meta.is_dir() {
                if !crate::companion::BUNDLE_SKIP_DIRS.contains(&name.as_str())
                    && !name.starts_with('.')
                {
                    queue.push(path);
                }
                continue;
            }
            if !meta.is_file() || meta.len() > 100_000 {
                continue;
            }
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !["rs", "ts", "tsx", "js", "jsx", "py", "go"].contains(&ext) {
                continue;
            }
            let Ok(rel) = path.strip_prefix(root) else {
                continue;
            };
            let rel_str = rel.to_string_lossy().replace('\\', "/");
            if !crate::safe_path(&rel_str, false) {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(&path) else {
                continue;
            };
            let syms = extract_symbols_from_source(&rel_str, &content);
            if !syms.is_empty() {
                index.files.push(rel_str);
                index.symbols.extend(syms);
            }
        }
    }
    index
}

/// Persists the workspace index to `.nioxs/index.json`.
pub fn save_workspace_index(index: &WorkspaceIndex, dest: &std::path::Path) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let data = serde_json::to_string_pretty(index)
        .map_err(|e| format!("failed to serialize index: {e}"))?;
    std::fs::write(dest, data)
        .map_err(|e| format!("failed to write index to {}: {e}", dest.display()))
}

/// Loads the workspace index from `.nioxs/index.json` if available.
pub fn load_workspace_index(path: &std::path::Path) -> Option<WorkspaceIndex> {
    let data = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&data).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_diagnostic_parsing() {
        let sample = "error[E0382]: use of moved value: `conn`\n  --> src/main.rs:42:5";
        let diags = parse_compiler_diagnostics(sample);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code.as_deref(), Some("E0382"));
        assert!(diags[0].message.contains("use of moved value"));

        let plan = formulate_assist_plan("fix borrow error", &diags[0]);
        assert_eq!(plan.steps.len(), 2);
        assert!(plan.steps[0].description.contains("Clone"));
    }

    #[test]
    fn test_multi_language_symbol_extraction() {
        let ts_code = "export class GatewayService {}\nexport function startGateway() {}";
        let ts_syms = extract_symbols_from_source("gateway.ts", ts_code);
        assert_eq!(ts_syms.len(), 2);
        assert_eq!(ts_syms[0].name, "GatewayService");
        assert_eq!(ts_syms[1].name, "startGateway");

        let py_code = "class DatabaseConnection:\n    pass\ndef execute_query(sql):\n    pass";
        let py_syms = extract_symbols_from_source("db.py", py_code);
        assert_eq!(py_syms.len(), 2);
        assert_eq!(py_syms[0].name, "DatabaseConnection");
        assert_eq!(py_syms[1].name, "execute_query");

        let go_code = "type Server struct {}\nfunc (s *Server) Listen() {}";
        let go_syms = extract_symbols_from_source("main.go", go_code);
        assert_eq!(go_syms.len(), 2);
        assert_eq!(go_syms[0].name, "Server");
        assert_eq!(go_syms[1].name, "Listen");
    }

    #[test]
    fn test_workspace_index_search_and_persistence() {
        let mut index = WorkspaceIndex::new();
        index.files.push("src/auth.rs".into());
        index.symbols.push(SymbolNode {
            name: "validate_token".into(),
            kind: "fn".into(),
            file: "src/auth.rs".into(),
            line: 12,
            callers: vec![],
            callees: vec![],
        });

        let found = index.find_files_for_symbol("validate_token");
        assert_eq!(found, vec!["src/auth.rs"]);

        let temp_dir = std::env::temp_dir().join(format!("nioxs-idx-{}", std::process::id()));
        let file_path = temp_dir.join("index.json");
        assert!(save_workspace_index(&index, &file_path).is_ok());

        let loaded = load_workspace_index(&file_path).expect("loaded index");
        assert_eq!(loaded.symbols.len(), 1);
        assert_eq!(loaded.symbols[0].name, "validate_token");
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
