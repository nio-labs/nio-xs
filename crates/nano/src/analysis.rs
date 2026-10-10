//! # Codebase & File Analysis Engine for NioNano MicroLM (μLM)
//!
//! Provides structural parsing, heuristic code health assessment,
//! strengths (Good), anti-pattern detection (Bad), and actionable suggestions.

#[derive(Debug, Clone, PartialEq)]
pub struct FileAnalysis {
    pub path: String,
    pub language: String,
    pub lines: usize,
    pub functions: usize,
    pub types: usize,
    pub tests: usize,
    pub doc_lines: usize,
    pub goods: Vec<String>,
    pub bads: Vec<String>,
    pub suggestions: Vec<String>,
}

pub fn analyze_file_content(path: &str, content: &str) -> FileAnalysis {
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();

    let ext = path.rsplit('.').next().unwrap_or("");
    let language = match ext {
        "rs" => "Rust",
        "ts" | "tsx" => "TypeScript",
        "js" | "jsx" => "JavaScript",
        "py" => "Python",
        "go" => "Go",
        "c" | "h" => "C",
        "cpp" | "hpp" | "cc" => "C++",
        "toml" => "TOML Configuration",
        "json" => "JSON Configuration",
        "yaml" | "yml" => "YAML Configuration",
        "md" => "Markdown Documentation",
        _ => "Source Code",
    };

    let mut functions = 0;
    let mut types = 0;
    let mut tests = 0;
    let mut doc_lines = 0;
    let mut unwrap_count = 0;
    let mut todo_count = 0;
    let mut unsafe_count = 0;
    let mut print_count = 0;

    for line in &lines {
        let trimmed = line.trim();
        if trimmed.starts_with("///")
            || trimmed.starts_with("//!")
            || trimmed.starts_with("/**")
            || trimmed.starts_with('*')
            || trimmed.starts_with('#') && language == "Markdown Documentation"
        {
            doc_lines += 1;
        }
        if trimmed.starts_with("fn ")
            || trimmed.starts_with("pub fn ")
            || trimmed.starts_with("pub async fn ")
            || trimmed.starts_with("async fn ")
            || trimmed.starts_with("def ")
            || trimmed.starts_with("function ")
        {
            functions += 1;
        }
        if trimmed.starts_with("struct ")
            || trimmed.starts_with("pub struct ")
            || trimmed.starts_with("enum ")
            || trimmed.starts_with("pub enum ")
            || trimmed.starts_with("class ")
            || trimmed.starts_with("interface ")
            || trimmed.starts_with("type ")
        {
            types += 1;
        }
        if trimmed.contains("#[test]")
            || trimmed.starts_with("def test_")
            || trimmed.contains("it(")
            || trimmed.contains("test(")
        {
            tests += 1;
        }
        if trimmed.contains(".unwrap()") {
            unwrap_count += 1;
        }
        if trimmed.contains("TODO") || trimmed.contains("FIXME") {
            todo_count += 1;
        }
        if trimmed.contains("unsafe ") || trimmed.contains("unsafe{") {
            unsafe_count += 1;
        }
        if trimmed.contains("println!") || trimmed.contains("console.log") {
            print_count += 1;
        }
    }

    let mut goods = Vec::new();
    let mut bads = Vec::new();
    let mut suggestions = Vec::new();

    // Strengths (Good)
    if tests > 0 {
        goods.push(format!(
            "Automated test coverage present ({tests} test{} declared)",
            if tests == 1 { "" } else { "s" }
        ));
    }
    if doc_lines > 3 {
        goods.push(format!(
            "Documentation present ({doc_lines} lines of docs/comments)"
        ));
    }
    if types > 0 {
        goods.push(format!(
            "Strong type definitions ({types} struct/enum/type model{})",
            if types == 1 { "" } else { "s" }
        ));
    }
    if total_lines < 300 && total_lines > 10 {
        goods.push("Modular file sizing (<300 LOC promotes maintainability)".into());
    }
    if unsafe_count == 0 && language == "Rust" {
        goods.push("Memory safe: zero unsafe blocks detected".into());
    }

    // Areas for Improvement (Bad / Risks)
    if unwrap_count > 0 {
        bads.push(format!(
            "{unwrap_count} instance{} of `.unwrap()` (risk of unhandled runtime panic)",
            if unwrap_count == 1 { "" } else { "s" }
        ));
        suggestions.push("Replace `.unwrap()` with `?` error propagation or `.expect(\"context\")`.".into());
    }
    if todo_count > 0 {
        bads.push(format!(
            "{todo_count} unresolved TODO/FIXME marker{}",
            if todo_count == 1 { "" } else { "s" }
        ));
        suggestions.push("Audit and resolve lingering TODO/FIXME comments before shipping.".into());
    }
    if unsafe_count > 0 {
        bads.push(format!(
            "{unsafe_count} `unsafe` block{} present",
            if unsafe_count == 1 { "" } else { "s" }
        ));
        suggestions.push("Ensure all `unsafe` blocks include explicit `// SAFETY:` invariants.".into());
    }
    if total_lines > 500 {
        bads.push(format!(
            "Large file size ({total_lines} LOC may indicate mixed responsibilities)"
        ));
        suggestions.push("Consider modularizing this file into dedicated submodules or traits.".into());
    }
    if doc_lines == 0 && total_lines > 20 {
        bads.push("Missing doc comments: public functions/types lack API documentation".into());
        suggestions.push("Add `///` doc comments for exported functions and types.".into());
    }
    if tests == 0 && (functions > 3 || types > 2) && !path.contains("test") {
        bads.push("No co-located unit tests detected for declared functions/types".into());
        suggestions.push("Add a `#[cfg(test)] mod tests` module to test core functionality.".into());
    }
    if print_count > 3 && !path.contains("main") && !path.contains("bin") {
        bads.push(format!("{print_count} raw stdout print statements in library code"));
        suggestions.push("Use structured logging (`tracing` or `log`) instead of direct stdout prints.".into());
    }

    if goods.is_empty() {
        goods.push("Clean file structure with standard syntax".into());
    }
    if bads.is_empty() {
        goods.push("Clean implementation: zero unwrap panics, zero unsafe blocks, zero unresolved TODOs".into());
    }
    if suggestions.is_empty() {
        suggestions.push("File is in good health; maintain test coverage when adding new features.".into());
    }

    FileAnalysis {
        path: path.to_string(),
        language: language.to_string(),
        lines: total_lines,
        functions,
        types,
        tests,
        doc_lines,
        goods,
        bads,
        suggestions,
    }
}

pub fn format_file_analysis(analysis: &FileAnalysis) -> String {
    let mut out = format!(
        "### Code Analysis: `{}`\n\n**Overview**:\n• **Language**: {}\n• **Lines of Code**: {} LOC\n• **Functions**: {}\n• **Types / Models**: {}\n• **Test Cases**: {}\n\n",
        analysis.path, analysis.language, analysis.lines, analysis.functions, analysis.types, analysis.tests
    );

    out.push_str("#### Strengths (Good):\n");
    for g in &analysis.goods {
        out.push_str(&format!("• ✔ {g}\n"));
    }
    out.push('\n');

    if !analysis.bads.is_empty() {
        out.push_str("#### Areas for Improvement (Bad / Risks):\n");
        for b in &analysis.bads {
            out.push_str(&format!("• ⚠ {b}\n"));
        }
        out.push('\n');
    }

    out.push_str("#### Actionable Suggestions:\n");
    for (i, s) in analysis.suggestions.iter().enumerate() {
        out.push_str(&format!("{}. {s}\n", i + 1));
    }

    out
}

pub fn format_directory_analysis(dir: &str, paths: &[String]) -> String {
    let total_files = paths.len();
    let mut rust_files = 0;
    let mut ts_files = 0;
    let mut py_files = 0;
    let mut config_files = 0;
    let mut doc_files = 0;
    let mut test_files = 0;

    for p in paths {
        let lower = p.to_ascii_lowercase();
        if lower.ends_with(".rs") {
            rust_files += 1;
        } else if lower.ends_with(".ts") || lower.ends_with(".js") {
            ts_files += 1;
        } else if lower.ends_with(".py") {
            py_files += 1;
        } else if lower.ends_with(".toml")
            || lower.ends_with(".json")
            || lower.ends_with(".yaml")
            || lower.ends_with(".yml")
        {
            config_files += 1;
        } else if lower.ends_with(".md") {
            doc_files += 1;
        }
        if lower.contains("test") || lower.contains("spec") {
            test_files += 1;
        }
    }

    let primary_lang = if rust_files >= ts_files && rust_files >= py_files && rust_files > 0 {
        "Rust"
    } else if ts_files > 0 {
        "TypeScript / JavaScript"
    } else if py_files > 0 {
        "Python"
    } else {
        "Mixed Polyglot"
    };

    let mut goods: Vec<String> = Vec::new();
    let mut bads: Vec<String> = Vec::new();
    let mut suggestions: Vec<String> = Vec::new();

    goods.push(format!(
        "Identified {total_files} tracked files primarily in {primary_lang}"
    ));
    if paths
        .iter()
        .any(|p| p.ends_with("Cargo.toml") || p.ends_with("package.json"))
    {
        goods.push("Standard package manifest detected for declarative dependency management".into());
    }
    if test_files > 0 {
        goods.push(format!(
            "Dedicated testing suite present ({test_files} test files/modules)"
        ));
    }
    if paths
        .iter()
        .any(|p| p.contains(".github/workflows") || p.contains(".gitlab-ci.yml"))
    {
        goods.push("Automated CI/CD workflows configured".into());
    }
    if doc_files > 0 {
        goods.push(format!(
            "Project documentation available ({doc_files} Markdown docs)"
        ));
    }

    if !paths.iter().any(|p| p == ".gitignore") {
        bads.push("Missing `.gitignore`: risk of committing build artifacts or credentials".into());
        suggestions.push("Run `scaffold gitignore` to generate a standard ignore file.".into());
    }
    if !paths.iter().any(|p| p == "LICENSE" || p == "LICENSE.md") {
        bads.push("No explicit open-source LICENSE file detected at repository root".into());
        suggestions.push("Run `scaffold license-mit` to declare license terms.".into());
    }
    if test_files == 0 {
        bads.push("No dedicated test files discovered in this directory".into());
        suggestions.push("Add automated unit tests and run `run test check` to verify behavior.".into());
    }
    if !paths.iter().any(|p| p.to_ascii_lowercase().starts_with("readme")) {
        bads.push("Missing README.md: repository lacks onboarding documentation".into());
        suggestions.push("Create a README.md documenting project goals, setup, and usage.".into());
    }

    if suggestions.is_empty() {
        suggestions.push("Codebase structure is healthy; proceed with running `run test check` and `run build check`.".into());
    }

    let mut out = format!(
        "### Codebase Analysis: `{dir}`\n\n**Overview**:\n• **Primary Ecosystem**: {}\n• **Tracked Files**: {}\n• **Code Files**: {} Rust, {} TS/JS, {} Python\n• **Config & Docs**: {} Configs, {} Docs\n\n",
        primary_lang, total_files, rust_files, ts_files, py_files, config_files, doc_files
    );

    out.push_str("#### Strengths (Good):\n");
    for g in &goods {
        out.push_str(&format!("• ✔ {g}\n"));
    }
    out.push('\n');

    if !bads.is_empty() {
        out.push_str("#### Areas for Improvement (Bad / Risks):\n");
        for b in &bads {
            out.push_str(&format!("• ⚠ {b}\n"));
        }
        out.push('\n');
    }

    out.push_str("#### Actionable Suggestions:\n");
    for (i, s) in suggestions.iter().enumerate() {
        out.push_str(&format!("{}. {s}\n", i + 1));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_analysis_rust() {
        let code = r#"
/// Documentation comment
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[test]
fn test_add() {
    assert_eq!(add(1, 2), 3);
}
"#;
        let analysis = analyze_file_content("src/math.rs", code);
        assert_eq!(analysis.language, "Rust");
        assert_eq!(analysis.functions, 2);
        assert_eq!(analysis.tests, 1);
        assert!(analysis.goods.iter().any(|g| g.contains("Automated test")));
    }

    #[test]
    fn test_file_analysis_unwrap_detection() {
        let code = r#"
fn risky() {
    let x: Option<i32> = None;
    let _ = x.unwrap();
    // TODO: fix this later
}
"#;
        let analysis = analyze_file_content("src/risky.rs", code);
        assert!(analysis.bads.iter().any(|b| b.contains(".unwrap()")));
        assert!(analysis.bads.iter().any(|b| b.contains("TODO")));
        assert!(analysis.suggestions.iter().any(|s| s.contains("Replace `.unwrap()`")));
    }

    #[test]
    fn test_directory_analysis_formatting() {
        let files = vec![
            "Cargo.toml".into(),
            "src/lib.rs".into(),
            "src/main.rs".into(),
            "tests/test_all.rs".into(),
            "README.md".into(),
        ];
        let report = format_directory_analysis(".", &files);
        assert!(report.contains("Codebase Analysis"));
        assert!(report.contains("Rust"));
        assert!(report.contains("Strengths (Good)"));
    }
}
