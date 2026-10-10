use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerDiagnostic {
    pub file_path: String,
    pub line_number: usize,
    pub error_code: String,
    pub message: String,
    pub label: String,
    pub suggested_replacement: Option<String>,
}

#[derive(Deserialize, Debug)]
struct CargoMessage {
    reason: String,
    message: Option<DiagnosticMessage>,
}

#[derive(Deserialize, Debug)]
struct DiagnosticMessage {
    level: String,
    message: String,
    code: Option<DiagnosticCode>,
    spans: Vec<DiagnosticSpan>,
    children: Vec<DiagnosticChild>,
}

#[derive(Deserialize, Debug)]
struct DiagnosticCode {
    code: String,
}

#[derive(Deserialize, Debug)]
struct DiagnosticChild {
    spans: Vec<DiagnosticSpan>,
}

#[derive(Deserialize, Debug)]
struct DiagnosticSpan {
    file_name: String,
    line_start: usize,
    is_primary: bool,
    label: Option<String>,
    suggested_replacement: Option<String>,
}

pub fn run_cargo_check(root: &str) -> Result<(), Vec<CompilerDiagnostic>> {
    let output = Command::new("cargo")
        .arg("check")
        .arg("--message-format=json")
        .current_dir(root)
        .output()
        .map_err(|e| vec![CompilerDiagnostic {
            file_path: "".to_string(),
            line_number: 0,
            error_code: "IO_ERROR".to_string(),
            message: format!("Failed to run cargo check: {}", e),
            label: "".to_string(),
            suggested_replacement: None,
        }])?;

    let mut diagnostics = Vec::new();
    let stdout = String::from_utf8_lossy(&output.stdout);

    for line in stdout.lines() {
        if let Ok(cargo_msg) = serde_json::from_str::<CargoMessage>(line) {
            if cargo_msg.reason == "compiler-message" {
                if let Some(msg) = cargo_msg.message {
                    if msg.level == "error" {
                        let code = msg.code.map(|c| c.code).unwrap_or_else(|| "UNKNOWN".to_string());
                        
                        let mut suggested = None;
                        for child in &msg.children {
                            if let Some(span) = child.spans.iter().find(|s| s.is_primary && s.suggested_replacement.is_some()) {
                                suggested = span.suggested_replacement.clone();
                            }
                        }
                        
                        if let Some(span) = msg.spans.iter().find(|s| s.is_primary) {
                            diagnostics.push(CompilerDiagnostic {
                                file_path: span.file_name.clone(),
                                line_number: span.line_start,
                                error_code: code,
                                message: msg.message.clone(),
                                label: span.label.clone().unwrap_or_default(),
                                suggested_replacement: suggested,
                            });
                        }
                    }
                }
            }
        }
    }

    if diagnostics.is_empty() && !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        diagnostics.push(CompilerDiagnostic {
            file_path: "".to_string(),
            line_number: 0,
            error_code: "BUILD_FAILED".to_string(),
            message: format!("Build failed without JSON diagnostics: {}", stderr),
            label: "".to_string(),
            suggested_replacement: None,
        });
    }

    if diagnostics.is_empty() {
        Ok(())
    } else {
        Err(diagnostics)
    }
}
