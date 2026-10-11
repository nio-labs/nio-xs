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
                if let Some(rest) = trimmed.strip_prefix("pub async fn ").or_else(|| trimmed.strip_prefix("async fn ")).or_else(|| trimmed.strip_prefix("pub fn ")).or_else(|| trimmed.strip_prefix("fn ")) {
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
                } else if let Some(rest) = trimmed.strip_prefix("pub struct ").or_else(|| trimmed.strip_prefix("struct ")) {
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
                } else if let Some(rest) = trimmed.strip_prefix("pub enum ").or_else(|| trimmed.strip_prefix("enum ")) {
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
                } else if let Some(rest) = trimmed.strip_prefix("pub trait ").or_else(|| trimmed.strip_prefix("trait ")) {
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
                if let Some(rest) = trimmed.strip_prefix("export function ").or_else(|| trimmed.strip_prefix("function ")).or_else(|| trimmed.strip_prefix("export async function ")).or_else(|| trimmed.strip_prefix("async function ")) {
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
                } else if let Some(rest) = trimmed.strip_prefix("export class ").or_else(|| trimmed.strip_prefix("class ")) {
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
                } else if let Some(rest) = trimmed.strip_prefix("export interface ").or_else(|| trimmed.strip_prefix("interface ")) {
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
                } else if let Some(rest) = trimmed.strip_prefix("export type ").or_else(|| trimmed.strip_prefix("type ")) {
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
                    let message = msg.get("message").and_then(|m| m.as_str()).unwrap_or("").to_string();
                    let code = msg.get("code").and_then(|c| c.get("code")).and_then(|c| c.as_str()).map(str::to_string);
                    
                    let mut file = None;
                    let mut line_num = None;
                    let mut col_num = None;

                    if let Some(spans) = msg.get("spans").and_then(|s| s.as_array()) {
                        if let Some(primary) = spans.iter().find(|s| s.get("is_primary").and_then(|p| p.as_bool()).unwrap_or(false)) {
                            file = primary.get("file_name").and_then(|f| f.as_str()).map(str::to_string);
                            line_num = primary.get("line_start").and_then(|l| l.as_u64()).map(|n| n as usize);
                            col_num = primary.get("column_start").and_then(|c| c.as_u64()).map(|n| n as usize);
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
            let code = if code_end > 6 { Some(trimmed[6..code_end].to_string()) } else { None };
            let message = trimmed[code_end + 1..].trim_start_matches(':').trim().to_string();

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
                    rationale: "Resolves use of moved value by creating an independent owned instance.".into(),
                });
                steps.push(AssistStep {
                    description: "Wrap shared data in Arc<T> or Arc<Mutex<T>> if across async boundaries".into(),
                    file: diag.file.clone(),
                    target_symbol: None,
                    rationale: "Enables multiple concurrent owners across tasks.".into(),
                });
                return AssistPlan {
                    title: format!("Remediating Borrow Checker Error [{code}]"),
                    root_cause: "Value is moved into a scope or closure and accessed after move.".into(),
                    steps,
                    verified: true,
                };
            }
            "E0499" | "E0502" => {
                steps.push(AssistStep {
                    description: "Shorten scope of mutable borrow".into(),
                    file: diag.file.clone(),
                    target_symbol: None,
                    rationale: "Ensures the mutable reference is dropped before subsequent accesses.".into(),
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
        description: format!("Review target site and compiler recommendations: {}", diag.message),
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
}
