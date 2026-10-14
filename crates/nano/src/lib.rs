//! # NioNano — MicroLM (μLM)
//!
//! NioNano combines **Symbolic AI** (deterministic code search, unified patch engine,
//! AST exploration, path sanitization, sandboxing) with a lightweight **Language Model (LM)**
//! (sub-megabyte embedded weights, multinomial n-gram distribution, <1ms CPU inference)
//! to provide sovereign pair-programming workflows without external API dependencies or GPU requirements.

pub mod analysis;
pub mod brain;
pub mod coder;
pub mod companion;
pub mod compiler;
pub mod dag;
pub mod decision;
pub mod model;
pub mod navigation;
pub mod patch;
pub mod profiles;
pub mod synthesize;
pub mod tools;
pub mod upstream;
pub mod watcher;
pub mod workflows;

pub use model::Model;
use nioxs_contracts::{
    ChatRequest, Execution, FunctionCall, Handoff, MODEL_NANO, Outcome, PROVIDER, SCHEMA_VERSION,
    ToolCall, digest,
};
pub use patch::{apply_patch_to_string, scaffold_template};
use serde_json::{Value, json};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

static GREETING_ROTATION: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Debug, PartialEq)]
pub enum Intent {
    Connected(Vec<workflows::Step>),
    Capabilities,
    GitInspect(tools::GitView),
    Diagnostics(tools::DiagnosticsKind),
    ProjectTasks,
    Read(String),
    Inspect,
    Files(String),
    Search(String),
    Checks(String),
    WebFetch(String),
    Research(String),
    Wiki(String),
    Github(String),
    Package(String),
    Scaffold(String),
    Greeting,
    Help,
    Clarify(String),
    Plan(Vec<Intent>),
    AnalyzeFile(String),
    AnalyzeDir(String),
    SymbolDef(String),
    SymbolRefs(String),
    SymbolCallers(String),
    SymbolImplementations(String),
    RefactorPatch {
        file: String,
        old: String,
        new: String,
    },
    TddRefactor {
        file: String,
        old: String,
        new: String,
    },
    ScaffoldTest {
        file: String,
        function: String,
    },
    ReviewDiff,
    SuggestCommit,
    ProjectSuggestions,
    TaskPlan(String),
    Bash(String),
    Tool(String),
    TestRunner(String),
    GitSummary(String),
    Coding {
        prompt: String,
        guarded: bool,
    },
    Config(String),
    HelpTopic(String),
    Unsupported,
}
impl Intent {
    pub fn label(&self) -> &str {
        match self {
            Self::Connected(_) => "connected_workflow",
            Self::Capabilities => "capabilities",
            Self::GitInspect(_) => "git_inspect",
            Self::Diagnostics(_) => "diagnostics",
            Self::ProjectTasks => "project_tasks",
            Self::Bash(_) => "bash_command",
            Self::Tool(_) => "run_tool",
            Self::Config(_) => "config",
            Self::TestRunner(_) => "test_runner",
            Self::GitSummary(_) => "git_summary",
            Self::Coding { guarded: true, .. } => "coding_guarded",
            Self::Coding { guarded: false, .. } => "coding",
            Self::Read(_) => "read_file",
            Self::Inspect => "inspect_project",
            Self::Files(_) => "find_files",
            Self::Search(_) => "search_code",
            Self::Checks(_) => "run_checks",
            Self::WebFetch(_) => "web_fetch",
            Self::Research(_) => "research_sources",
            Self::Wiki(_) => "wiki_search",
            Self::Github(_) => "github_repo",
            Self::Package(_) => "package_info",
            Self::Scaffold(_) => "scaffold_template",
            Self::Greeting => "greeting",
            Self::Help | Self::HelpTopic(_) => "help",
            Self::Clarify(_) => "clarify",
            Self::Plan(_) => "plan",
            Self::AnalyzeFile(_) => "analyze_file",
            Self::AnalyzeDir(_) => "analyze_dir",
            Self::SymbolDef(_) => "symbol_definitions",
            Self::SymbolRefs(_) => "symbol_references",
            Self::SymbolCallers(_) => "symbol_callers",
            Self::SymbolImplementations(_) => "symbol_implementations",
            Self::RefactorPatch { .. } => "apply_exact_patch",
            Self::TddRefactor { .. } => "connected_workflow",
            Self::ScaffoldTest { .. } => "scaffold_template",
            Self::ReviewDiff => "inspect_project",
            Self::SuggestCommit => "inspect_project",
            Self::ProjectSuggestions => "project_suggestions",
            Self::TaskPlan(_) => "task_plan",
            Self::Unsupported => "unsupported",
        }
    }
}

/// Labels of the shipped, actionable workflows advertised by `/v1/capabilities`.
/// The informational and abstain labels (`greeting`, `help`, `clarify`,
/// `unsupported`) are intentionally excluded.
pub const SHIPPED_WORKFLOWS: &[&str] = &[
    "connected_workflow",
    "apply_exact_patch",
    "format_file",
    "rename_symbol",
    "search_docs",
    "fetch_reference",
    "capabilities",
    "git_inspect",
    "diagnostics",
    "project_tasks",
    "symbol_definitions",
    "symbol_references",
    "symbol_callers",
    "symbol_implementations",
    "bash_command",
    "run_tool",
    "config",
    "test_runner",
    "git_summary",
    "coding",
    "coding_guarded",
    "read_file",
    "inspect_project",
    "find_files",
    "search_code",
    "run_checks",
    "web_fetch",
    "research_sources",
    "wiki_search",
    "github_repo",
    "package_info",
    "scaffold_template",
    "plan",
    "analyze_file",
    "analyze_dir",
    "project_suggestions",
    "task_plan",
];
#[derive(Clone)]
pub struct ObservedAction {
    pub call: ToolCall,
    pub result_digest: String,
    pub status: String,
}
pub struct Decision {
    pub content: Option<String>,
    pub call: Option<ToolCall>,
    pub execution: Execution,
    pub request_digest: String,
    pub result_digest: Option<String>,
    pub intent: String,
    pub step: usize,
    pub observed: Vec<ObservedAction>,
    /// Ordered progress segments for progressive SSE streaming.
    /// Empty for every intent except a `--code:` turn; the provider emits
    /// each as a separate `delta.content` event so the TUI renders work line by
    /// line (the step trail). `content` remains the fully-assembled markdown for
    /// non-streaming clients.
    pub stream_segments: Vec<String>,
    /// A pending upstream consultation. When set, the provider opens the
    /// upstream request with `stream: true` and forwards the model's own deltas
    /// to the client instead of returning `content` in one block.
    pub coding: Option<companion::CodingConsult>,
}
fn embedded() -> &'static Model {
    static MODEL: OnceLock<Model> = OnceLock::new();
    MODEL.get_or_init(|| Model::embedded().expect("validated embedded model"))
}
pub fn decide(request: &ChatRequest, deployment: &str) -> Result<Decision, String> {
    decide_with_turn(request, deployment, None)
}
pub fn decide_with_turn(
    request: &ChatRequest,
    deployment: &str,
    turn: Option<&str>,
) -> Result<Decision, String> {
    decide_with_model(request, deployment, turn, embedded())
}
pub fn safe_path(path: &str, glob: bool) -> bool {
    !path.is_empty()
        && path.len() <= 240
        && !path.starts_with('/')
        && !path.starts_with('~')
        && path.bytes().all(|b| {
            b.is_ascii_alphanumeric() || b"_./-".contains(&b) || glob && b"*?".contains(&b)
        })
        && path
            .split('/')
            .all(|p| !matches!(p, ".." | ".git" | ".ssh") && !p.starts_with(".env"))
        && !path.ends_with(".key")
        && !path.ends_with(".token")
}
fn paths(prompt: &str) -> Vec<String> {
    prompt
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| "`\"',:?!".contains(c))
                .trim_end_matches('.')
        })
        .filter(|w| {
            w.contains(['.', '/', '*', '?'])
                || ["readme", "makefile", "license"].contains(&w.to_ascii_lowercase().as_str())
        })
        .map(|w| match w.to_ascii_lowercase().as_str() {
            "readme" | "readme.md" => "README.md".into(),
            "cargo.toml" => "Cargo.toml".into(),
            "makefile" => "Makefile".into(),
            "license" => "LICENSE".into(),
            _ => w.into(),
        })
        .collect()
}
pub fn is_web_url(s: &str) -> bool {
    (s.starts_with("https://") || s.starts_with("http://"))
        && s.len() <= 1024
        && !s.contains(char::is_whitespace)
}
fn run_async<F: std::future::Future>(f: F) -> F::Output {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        tokio::task::block_in_place(|| handle.block_on(f))
    } else {
        tokio::runtime::Runtime::new()
            .expect("tokio runtime")
            .block_on(f)
    }
}

fn split_compound_prompt(prompt: &str) -> Option<Vec<&str>> {
    let mut s = prompt.trim();
    let mut lower = s.to_ascii_lowercase();
    let mut changed = true;
    while changed {
        changed = false;
        for filler in &[
            "hey ",
            "hi ",
            "hello ",
            "ok ",
            "well ",
            "so ",
            "now ",
            "please ",
            "can you ",
            "could you ",
        ] {
            if lower.starts_with(filler) {
                lower = lower[filler.len()..].trim_start().to_string();
                s = s[filler.len()..].trim_start();
                changed = true;
                break;
            }
        }
    }
    for delim in &[" and then ", " then ", "; ", " after that "] {
        if lower.contains(delim) {
            let parts: Vec<&str> = prompt
                .split(delim)
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() >= 2 {
                return Some(parts);
            }
        }
    }
    None
}

pub fn parse(model: &Model, prompt: &str) -> Intent {
    if let Some(result) = workflows::parse(prompt.trim()) {
        return match result {
            Ok(steps) => Intent::Connected(steps),
            Err(reason) => Intent::Clarify(reason),
        };
    }
    if prompt.len() > 1024 || !prompt.is_ascii() {
        return Intent::Unsupported;
    }
    if let Some(parts) = split_compound_prompt(prompt) {
        let mut steps = Vec::new();
        for part in parts {
            if model::features(part)
                .iter()
                .any(|f| f == "<unsupported-risk>")
            {
                return Intent::Unsupported;
            }
            let sub = parse_atomic(model, part);
            if matches!(sub, Intent::Unsupported | Intent::Clarify(_)) {
                return Intent::Unsupported;
            }
            steps.push(sub);
        }
        if steps.len() >= 2 {
            return Intent::Plan(steps);
        }
    }
    parse_atomic(model, prompt)
}

pub fn normalize_prompt(prompt: &str) -> String {
    let mut s = prompt.trim().to_ascii_lowercase();
    let conversational_fillers = [
        "could you please ",
        "could you ",
        "can you please ",
        "can you ",
        "would you please ",
        "would you ",
        "please ",
        "i want you to ",
        "i'd like you to ",
        "i would like you to ",
        "i want to ",
        "i'd like to ",
        "give me an ",
        "give me a ",
        "give me ",
        "show me an ",
        "show me a ",
        "show me ",
        "tell me about ",
        "let's take a look at ",
        "take a look at ",
        "let's check ",
        "let's read ",
        "let's view ",
        "let's open ",
        "let's ",
        "how about ",
        "check out ",
        "help me read ",
        "help me view ",
        "help me open ",
        "can we ",
        "could we ",
        "hey ",
        "hi ",
        "hello ",
        "ok ",
        "okay ",
        "well ",
        "so ",
        "now ",
    ];
    let mut changed = true;
    while changed {
        changed = false;
        for filler in &conversational_fillers {
            if s.starts_with(filler) {
                s = s[filler.len()..].trim().to_string();
                changed = true;
                break;
            }
        }
    }

    let words: Vec<&str> = s.split_whitespace().collect();
    let mut normalized_words = Vec::new();
    for word in words {
        let clean = word.trim_matches(|c: char| "`\"',:!?".contains(c));
        let norm = match clean {
            "analyse" | "analysing" | "analysed" | "analyzing" | "analyzed" | "analysis" => {
                "analyze"
            }
            "analyst" => "analyze",
            "eval" | "evaluation" | "evaluating" => "evaluate",
            "auditing" | "audited" => "audit",
            "reviewing" | "reviewed" => "review",
            "inspecting" | "inspected" => "inspect",
            "testing" | "tests" => "test",
            _ => clean,
        };
        normalized_words.push(norm);
    }

    let mut joined = normalized_words.join(" ");
    if joined.starts_with("analyze of ") {
        joined = format!("analyze {}", &joined[11..]);
    } else if joined.starts_with("review of ") {
        joined = format!("review {}", &joined[10..]);
    } else if joined.starts_with("audit of ") {
        joined = format!("audit {}", &joined[9..]);
    }

    if joined == "check code health"
        || joined == "how is the code quality"
        || joined == "how is code quality"
        || joined == "check code quality"
        || joined == "code quality"
        || joined == "code health"
    {
        return "analyze codebase".to_string();
    }
    joined
}

pub fn extract_natural_read_intent(prompt: &str, norm: &str) -> Option<Intent> {
    let lower_prompt = prompt.trim().to_ascii_lowercase();
    let lower_norm = norm.trim().to_ascii_lowercase();
    let stripped_prompt = lower_prompt.trim_end_matches(|c: char| "?!. \t\r\n".contains(c));
    let stripped_norm = lower_norm.trim_end_matches(|c: char| "?!. \t\r\n".contains(c));

    // Exclude other intents explicitly
    if stripped_prompt.starts_with("analyze ")
        || stripped_norm.starts_with("analyze ")
        || stripped_prompt.starts_with("review ")
        || stripped_norm.starts_with("review ")
        || stripped_prompt.starts_with("audit ")
        || stripped_norm.starts_with("audit ")
        || stripped_prompt.starts_with("write ")
        || stripped_norm.starts_with("write ")
        || stripped_prompt.starts_with("scaffold ")
        || stripped_norm.starts_with("scaffold ")
        || stripped_prompt.starts_with("search ")
        || stripped_norm.starts_with("search ")
        || stripped_prompt.starts_with("find ")
        || stripped_norm.starts_with("find ")
    {
        return None;
    }

    let read_prefixes = [
        "what is in ",
        "what's in ",
        "what is inside ",
        "what's inside ",
        "what is in the file ",
        "what's in the file ",
        "what is inside the file ",
        "what's inside the file ",
        "contents of ",
        "content of ",
        "contents of file ",
        "content of file ",
        "the contents of ",
        "the content of ",
        "read file ",
        "read the file ",
        "read ",
        "view file ",
        "view the file ",
        "view ",
        "open file ",
        "open the file ",
        "open ",
        "cat file ",
        "cat the file ",
        "cat ",
        "show file ",
        "show the file ",
        "show contents of ",
        "show content of ",
        "show me the contents of ",
        "show me contents of ",
        "show me what's inside ",
        "show me what is inside ",
        "show me what's in ",
        "show me what is in ",
        "show me file ",
        "show me the file ",
        "show me ",
        "show ",
        "display file ",
        "display the file ",
        "display contents of ",
        "display ",
        "print file ",
        "print the file ",
        "print ",
        "examine file ",
        "examine the file ",
        "examine ",
        "look at file ",
        "look at the file ",
        "look at ",
        "look into file ",
        "look into ",
        "check file ",
        "check the file ",
        "check what's inside ",
        "check what is inside ",
        "check what's in ",
        "check what is in ",
        "inspect file ",
        "inspect the file ",
    ];

    for prefix in &read_prefixes {
        let matched = if stripped_prompt.starts_with(prefix) {
            Some(&prompt[prefix.len()..])
        } else if stripped_norm.starts_with(prefix) {
            Some(&norm[prefix.len()..])
        } else {
            None
        };

        if let Some(rest) = matched {
            let cleaned = rest
                .trim()
                .trim_matches(|c: char| "`\"',;:!? \t\r\n".contains(c));
            if cleaned == "it" || cleaned == "that" || cleaned == "file" || cleaned == "this" {
                return Some(Intent::Clarify(
                    "Which project-relative file should I read?".into(),
                ));
            }

            let candidates = paths(rest);
            if candidates.len() == 1
                && safe_path(&candidates[0], false)
                && !candidates[0].contains(['*', '?'])
            {
                return Some(Intent::Read(candidates[0].clone()));
            }

            if safe_path(cleaned, false)
                && !cleaned.contains(['*', '?', ' '])
                && !cleaned.is_empty()
            {
                let canonical = match cleaned.to_ascii_lowercase().as_str() {
                    "cargo.toml" => "Cargo.toml".to_string(),
                    "readme" | "readme.md" => "README.md".to_string(),
                    "makefile" => "Makefile".to_string(),
                    "license" => "LICENSE".to_string(),
                    _ => cleaned.to_string(),
                };
                return Some(Intent::Read(canonical));
            }
        }
    }

    // Natural inquiry: "what does Cargo.toml contain?" / "how does src/lib.rs look?"
    if (stripped_prompt.starts_with("what does ") || stripped_prompt.starts_with("how does "))
        && (stripped_prompt.contains(" contain") || stripped_prompt.contains(" look"))
    {
        let candidates = paths(prompt);
        if candidates.len() == 1
            && safe_path(&candidates[0], false)
            && !candidates[0].contains(['*', '?'])
        {
            return Some(Intent::Read(candidates[0].clone()));
        }
    }

    // Bare file query: user typed just the filename or path
    let trimmed = prompt
        .trim()
        .trim_matches(|c: char| "`\"',; \t\r\n".contains(c));
    if !trimmed.contains(' ') && safe_path(trimmed, false) && !trimmed.contains(['*', '?']) {
        let lower = trimmed.to_ascii_lowercase();
        let has_ext = trimmed.contains('.');
        let is_known = matches!(
            lower.as_str(),
            "readme" | "makefile" | "license" | "dockerfile"
        );
        if has_ext || is_known {
            let canonical = match lower.as_str() {
                "cargo.toml" => "Cargo.toml".to_string(),
                "readme" | "readme.md" => "README.md".to_string(),
                "makefile" => "Makefile".to_string(),
                "license" => "LICENSE".to_string(),
                _ => trimmed.to_string(),
            };
            return Some(Intent::Read(canonical));
        }
    }

    None
}

fn strip_directive<'a>(prompt: &'a str, name: &str) -> Option<&'a str> {
    let rest = prompt.strip_prefix(name)?;
    if rest.is_empty() {
        return Some("");
    }
    if let Some(after) = rest.strip_prefix(':') {
        return Some(after.trim());
    }
    if rest.starts_with(' ') || rest.starts_with('\t') {
        return Some(rest.trim());
    }
    None
}

pub fn parse_atomic(model: &Model, prompt: &str) -> Intent {
    if prompt.len() > 1024 || !prompt.is_ascii() {
        return Intent::Unsupported;
    }
    let trimmed_prompt = prompt.trim();
    if let Some(result) = workflows::parse(trimmed_prompt) {
        return match result {
            Ok(steps) => Intent::Connected(steps),
            Err(reason) => Intent::Clarify(reason),
        };
    }
    if let Some(result) = navigation::parse_request(trimmed_prompt) {
        return match result {
            Ok((kind, symbol)) => match kind {
                navigation::NavigationKind::Definitions => Intent::SymbolDef(symbol),
                navigation::NavigationKind::References => Intent::SymbolRefs(symbol),
                navigation::NavigationKind::Callers => Intent::SymbolCallers(symbol),
                navigation::NavigationKind::Implementations => {
                    Intent::SymbolImplementations(symbol)
                }
            },
            Err(reason) => Intent::Clarify(reason),
        };
    }
    let normalized = trimmed_prompt.to_ascii_lowercase();
    match normalized.as_str() {
        "--capabilities"
        | "capabilities"
        | "what tools are available?"
        | "list available tools" => return Intent::Capabilities,
        "--git full-diff" | "--git:full-diff" | "show full git diff" | "what changed?" => {
            return Intent::GitInspect(tools::GitView::Diff);
        }
        "--git staged" | "--git:staged" | "show staged diff" => {
            return Intent::GitInspect(tools::GitView::Staged);
        }
        "--git history" | "--git:history" | "show git history" | "show recent commits" => {
            return Intent::GitInspect(tools::GitView::History);
        }
        "--diagnostics"
        | "--diagnostics compiler"
        | "--diagnostics:compiler"
        | "show compiler diagnostics"
        | "check compiler errors"
        | "what failed?" => return Intent::Diagnostics(tools::DiagnosticsKind::Compiler),
        "--diagnostics lint" | "--diagnostics:lint" | "show lint diagnostics" => {
            return Intent::Diagnostics(tools::DiagnosticsKind::Lint);
        }
        "--tasks" | "list project tasks" | "show project scripts" | "what tasks can i run?" => {
            return Intent::ProjectTasks;
        }
        _ => {}
    }
    if strip_directive(trimmed_prompt, "--diagnostics").is_some() {
        return Intent::Clarify(
            "Use --diagnostics compiler or --diagnostics lint (Rust projects).".into(),
        );
    }
    if strip_directive(trimmed_prompt, "--tasks").is_some() {
        return Intent::Clarify(
            "Use --tasks to discover project tasks without executing them.".into(),
        );
    }
    if let Some(cmd) = strip_directive(trimmed_prompt, "--bash") {
        return Intent::Bash(cmd.to_string());
    }
    if let Some(tool_req) = strip_directive(trimmed_prompt, "--tools") {
        return Intent::Tool(tool_req.to_string());
    }
    if let Some(config_req) = strip_directive(trimmed_prompt, "--config") {
        return Intent::Config(config_req.to_string());
    }
    let lower_trimmed = trimmed_prompt.to_ascii_lowercase();
    if let Some(rest) = lower_trimmed.strip_prefix("install ") {
        if companion::is_known_plugin(rest.trim()) {
            return Intent::Tool(format!("install {}", rest.trim()));
        }
    }
    if let Some(test_req) = strip_directive(trimmed_prompt, "--test") {
        return Intent::TestRunner(test_req.to_string());
    }
    if let Some(git_req) = strip_directive(trimmed_prompt, "--git") {
        // Only known subcommands are honored; the bare directive means "status
        // plus diff". An unknown subcommand is clarified instead of being
        // silently collapsed into the default summary.
        return match git_req.trim().to_ascii_lowercase().as_str() {
            "" => Intent::GitSummary(String::new()),
            "status" => Intent::GitSummary("status".into()),
            "diff" => Intent::GitSummary("diff".into()),
            other => Intent::Clarify(format!(
                "Unknown `--git` subcommand `{other}`. Use `--git`, `--git:status`, `--git:diff`, `--git:full-diff`, `--git:staged`, or `--git:history`."
            )),
        };
    }
    if let Some(glob) = strip_directive(trimmed_prompt, "--find-files") {
        let glob = if glob.trim().is_empty() {
            "*"
        } else {
            glob.trim()
        };
        return if safe_path(glob, true) {
            Intent::Files(glob.to_string())
        } else {
            Intent::Clarify("Use a project-relative filename or glob for --find-files.".into())
        };
    }
    if let Some(help_topic) = strip_directive(trimmed_prompt, "--help") {
        return if help_topic.is_empty() {
            Intent::Help
        } else {
            Intent::HelpTopic(help_topic.to_string())
        };
    }
    // The coding consultation directive is spelled `--code`; the older
    // `--coding` spelling is retired and intentionally not recognized.
    if let Some(coding_req) = strip_directive(trimmed_prompt, "--code guard")
        .or_else(|| strip_directive(trimmed_prompt, "--code:guard"))
        .or_else(|| strip_directive(trimmed_prompt, "--code!"))
    {
        return Intent::Coding {
            prompt: coding_req.to_string(),
            guarded: true,
        };
    }
    if let Some(coding_req) = strip_directive(trimmed_prompt, "--code") {
        return Intent::Coding {
            prompt: coding_req.to_string(),
            guarded: companion::is_config_guard_on(),
        };
    }
    if companion::is_config_code_on() {
        return Intent::Coding {
            prompt: trimmed_prompt.to_string(),
            guarded: companion::is_config_guard_on(),
        };
    }
    for word in prompt.split_whitespace() {
        let clean = word.trim_matches(|c: char| "`\"',;:!?<>".contains(c));
        if is_web_url(clean) {
            return Intent::WebFetch(clean.to_string());
        }
    }
    let norm = normalize_prompt(prompt);
    if let Some(coder_intent) =
        coder::parse_coder_intent(prompt).or_else(|| coder::parse_coder_intent(&norm))
    {
        return coder_intent;
    }
    if let Some(read_intent) = extract_natural_read_intent(prompt, &norm) {
        return read_intent;
    }

    let mut s = prompt.trim();
    let mut lower = s.to_ascii_lowercase();
    let mut changed = true;
    while changed {
        changed = false;
        for filler in &[
            "hey ",
            "hi ",
            "hello ",
            "ok ",
            "well ",
            "so ",
            "now ",
            "please ",
            "can you ",
            "could you ",
        ] {
            if lower.starts_with(filler) {
                lower = lower[filler.len()..].trim_start().to_string();
                s = s[filler.len()..].trim_start();
                changed = true;
                break;
            }
        }
    }
    let query_prefix = [
        "search the web for ",
        "search web for ",
        "web search for ",
        "web search ",
        "research ",
        "what is ",
        "what's ",
        "who is ",
        "who's ",
        "where is ",
        "where's ",
        "how is ",
        "how does ",
        "tell me about ",
        "explain ",
        "why is ",
        "why ",
    ]
    .iter()
    .find_map(|p| {
        if lower.starts_with(p) && !lower.starts_with("where is symbol ") {
            Some(s[p.len()..].trim().trim_end_matches('?').trim())
        } else {
            None
        }
    });
    if let Some(query) = query_prefix
        && !query.is_empty()
    {
        let internal_targets = [
            "workspace",
            "the router",
            "the protocol",
            "staging server",
            "source tree",
            "production",
            "remote repository",
            "an account",
            "build artifact",
            "private logs",
        ];
        let lower_q = query.to_ascii_lowercase();
        if lower_q.contains("my project")
            || lower_q.contains("this project")
            || lower_q.contains("our project")
            || lower_q.contains("the project")
            || lower_q.contains("my codebase")
            || lower_q.contains("this codebase")
            || lower_q.contains("our codebase")
            || lower_q.contains("my repo")
            || lower_q.contains("this repo")
            || lower_q.contains("our repo")
            || lower_q.contains("project build on")
            || lower_q.contains("project built on")
        {
            return Intent::Inspect;
        }
        if lower_q.contains("code quality")
            || lower_q.contains("code health")
            || lower_q == "the code"
            || lower_q == "code"
            || lower_q == "codebase"
        {
            return Intent::AnalyzeDir(".".into());
        }
        if [" then ", " && ", " and "]
            .iter()
            .any(|s| lower_q.contains(s))
            || model::features(query)
                .iter()
                .any(|f| f == "<unsupported-risk>")
            || internal_targets
                .iter()
                .any(|target| lower_q.contains(target))
        {
            return Intent::Unsupported;
        }
        return Intent::Research(query.to_string());
    }
    if model::features(prompt)
        .iter()
        .any(|f| f == "<unsupported-risk>")
    {
        return Intent::Unsupported;
    }
    if [" then ", " && ", " and "]
        .iter()
        .any(|s| lower.contains(s))
    {
        return Intent::Unsupported;
    }
    if lower.starts_with("wiki ") || lower.starts_with("wikipedia ") {
        let query = if lower.starts_with("wikipedia ") {
            s[10..].trim()
        } else {
            s[5..].trim()
        };
        if !query.is_empty() {
            return Intent::Wiki(query.to_string());
        }
    }
    if lower.starts_with("github ") || lower.starts_with("git ") {
        let target = if lower.starts_with("github ") {
            s[7..].trim()
        } else {
            s[4..].trim()
        };
        if !target.is_empty() && target.contains('/') {
            return Intent::Github(target.to_string());
        }
    }
    if lower.starts_with("package ") || lower.starts_with("crate ") || lower.starts_with("npm ") {
        let pkg = if lower.starts_with("package ") {
            s[8..].trim()
        } else if lower.starts_with("crate ") {
            s[6..].trim()
        } else {
            s[4..].trim()
        };
        if !pkg.is_empty() && !pkg.contains(' ') {
            return Intent::Package(pkg.to_string());
        }
    }
    if lower.starts_with("scaffold ") || lower.starts_with("template ") {
        let tpl = s[9..].trim();
        if !tpl.is_empty() {
            return Intent::Scaffold(tpl.to_string());
        }
    }
    let norm = normalize_prompt(prompt);
    let trimmed = lower
        .trim()
        .trim_end_matches(|c: char| "!?. \t\r\n".contains(c));
    if [
        "hey",
        "hello",
        "hi",
        "good morning",
        "good evening",
        "howdy",
        "sup",
        "yo",
    ]
    .contains(&trimmed)
        || [
            "hey",
            "hello",
            "hi",
            "good morning",
            "good evening",
            "howdy",
            "sup",
            "yo",
        ]
        .contains(&norm.as_str())
    {
        return Intent::Greeting;
    }
    if [
        "help",
        "what can you do",
        "commands",
        "usage",
        "capabilities",
    ]
    .contains(&trimmed)
        || [
            "help",
            "what can you do",
            "commands",
            "usage",
            "capabilities",
        ]
        .contains(&norm.as_str())
    {
        return Intent::Help;
    }
    if [
        "search my codebase",
        "search the codebase",
        "search codebase",
        "search code",
    ]
    .contains(&trimmed)
        || [
            "search my codebase",
            "search the codebase",
            "search codebase",
            "search code",
        ]
        .contains(&norm.as_str())
    {
        return Intent::Clarify(
            "Which literal text or symbol should I search for in the codebase?".into(),
        );
    }
    let analyze_prefixes = [
        "analyze file ",
        "analyze folder ",
        "analyze dir ",
        "analyze directory ",
        "analyze ",
        "review file ",
        "review folder ",
        "review dir ",
        "review directory ",
        "review ",
        "audit file ",
        "audit folder ",
        "audit dir ",
        "audit ",
    ];
    for prefix in &analyze_prefixes {
        let matched = if norm.starts_with(prefix) {
            Some(&norm[prefix.len()..])
        } else if lower.starts_with(prefix) {
            Some(&prompt[prefix.len()..])
        } else {
            None
        };
        if let Some(raw_target) = matched {
            let target = raw_target
                .trim()
                .trim_matches(|c: char| "`\"',:!?".contains(c));
            let lower_t = target.to_ascii_lowercase();
            if [
                "codebase",
                "the codebase",
                "my codebase",
                "project",
                "the project",
                "my project",
                "code",
                "the code",
                ".",
            ]
            .contains(&lower_t.as_str())
            {
                return Intent::AnalyzeDir(".".into());
            }
            let canonical_target = match lower_t.as_str() {
                "cargo.toml" => "Cargo.toml",
                "readme" | "readme.md" => "README.md",
                "makefile" => "Makefile",
                "license" => "LICENSE",
                _ => target,
            };
            if safe_path(canonical_target, true) && !canonical_target.is_empty() {
                if canonical_target.contains('.') && !canonical_target.ends_with('/') {
                    return Intent::AnalyzeFile(canonical_target.to_string());
                } else {
                    return Intent::AnalyzeDir(canonical_target.to_string());
                }
            }
        }
    }
    if [
        "analyze codebase",
        "review codebase",
        "audit codebase",
        "analyze project",
        "review project",
        "audit project",
        "analyze code",
        "code review",
        "code audit",
        "analyze current codebase",
        "analyze current code base",
    ]
    .contains(&trimmed)
        || [
            "analyze codebase",
            "review codebase",
            "audit codebase",
            "analyze project",
            "review project",
            "audit project",
            "analyze code",
            "code review",
            "code audit",
            "analyze current codebase",
            "analyze current code base",
        ]
        .contains(&norm.as_str())
    {
        return Intent::AnalyzeDir(".".into());
    }
    let candidates = paths(prompt);
    let intent = match model.predict(prompt).label.as_str() {
        "read_file" => {
            if candidates.len() == 1
                && safe_path(&candidates[0], false)
                && !candidates[0].contains(['*', '?'])
            {
                Intent::Read(candidates[0].clone())
            } else {
                Intent::Clarify("Which project-relative file should I read?".into())
            }
        }
        "inspect_project" => Intent::Inspect,
        "find_files" => {
            if candidates.is_empty() {
                Intent::Files("*".into())
            } else if candidates.len() == 1 && safe_path(&candidates[0], true) {
                Intent::Files(candidates[0].clone())
            } else {
                Intent::Clarify("Which project-relative filename or glob should I find?".into())
            }
        }
        "search_code" => {
            let markers = [
                "which source files contain ",
                "look for references to ",
                "search project text for ",
                "search the repository for ",
                "find the code containing ",
                "find source references to ",
                "locate code containing ",
                "locate references to ",
                "search code containing ",
                "find references to ",
                "show occurrences of ",
                "show me matches for ",
                "find occurrences of ",
                "search code for ",
                "search source for ",
                "where is symbol ",
                "find code for ",
                "locate symbol ",
                "find symbol ",
                "search for ",
                "find where ",
                "find ",
            ];
            let query = markers.iter().find_map(|m| {
                lower.find(m).map(|i| {
                    prompt[i + m.len()..]
                        .trim()
                        .trim_matches(|c: char| "`\"'?.".contains(c))
                        .trim_end_matches(" occurs")
                        .trim()
                        .to_string()
                })
            });
            match query {
                Some(q)
                    if !q.is_empty()
                        && q.len() <= 240
                        && ![
                            "it",
                            "that",
                            "this",
                            "the thing",
                            "my codebase",
                            "the codebase",
                            "codebase",
                        ]
                        .contains(&q.as_str()) =>
                {
                    Intent::Search(q)
                }
                _ => Intent::Clarify(
                    "Which literal text or symbol should I search for in the codebase?".into(),
                ),
            }
        }
        "run_checks" => {
            let words = model::words(prompt);
            let kinds: Vec<&str> = [
                ("test", ["test", "tests"].as_slice()),
                ("build", ["build", "compile"].as_slice()),
                ("lint", ["lint", "linter"].as_slice()),
            ]
            .into_iter()
            .filter(|(_, names)| words.iter().any(|w| names.contains(&w.as_str())))
            .map(|(kind, _)| kind)
            .collect();
            if kinds.len() == 1 {
                Intent::Checks(kinds[0].into())
            } else {
                Intent::Clarify("Which configured check should I run: test, build, or lint?".into())
            }
        }
        "clarify" => Intent::Clarify(
            if lower.contains("read") || lower.contains("open") || lower.contains("show") {
                "Which project-relative file should I read?"
            } else if lower.contains("run") || lower.contains("check") {
                "Which configured check should I run: test, build, or lint?"
            } else {
                "Which literal text or symbol should I search for?"
            }
            .into(),
        ),
        _ => Intent::Unsupported,
    };
    // A code-modification verb with no bounded local workflow is handed to the
    // explicit `--code` consultation instead of dead-ending as `unsupported`.
    if matches!(intent, Intent::Unsupported) && unbound_code_modification(prompt) {
        return Intent::Clarify(format!(
            "That looks like a code change with no bounded local workflow. Re-run it as \
            `--code {prompt}` to consult the coding model."
        ));
    }
    intent
}

/// Code-modification verbs that have no bounded local workflow, so an unbound
/// request using one is handed to `--code` rather than reported as unsupported.
fn unbound_code_modification(prompt: &str) -> bool {
    const VERBS: [&str; 6] = [
        "refactor",
        "debug",
        "rename",
        "implement",
        "migrate",
        "rewrite",
    ];
    model::words(prompt)
        .iter()
        .any(|w| VERBS.contains(&w.as_str()))
}
struct Workflow<'a> {
    request: &'a ChatRequest,
    start: usize,
    key: String,
    execution: Execution,
    request_digest: String,
    intent: String,
    observed: Vec<ObservedAction>,
    stream_segments: Vec<String>,
    coding: Option<companion::CodingConsult>,
}
impl Workflow<'_> {
    fn decision(&self, content: Option<String>, call: Option<ToolCall>) -> Decision {
        let step = if self.coding.is_some() {
            companion::coding_tool_rounds(&self.request.messages[self.start..])
        } else {
            self.observed.len()
        };
        Decision {
            content,
            call,
            execution: self.execution.clone(),
            request_digest: self.request_digest.clone(),
            result_digest: self.observed.last().map(|o| o.result_digest.clone()),
            intent: self.intent.clone(),
            step,
            observed: self.observed.clone(),
            stream_segments: self.stream_segments.clone(),
            coding: self.coding.clone(),
        }
    }
    fn end(&mut self, content: String, outcome: Outcome) -> Decision {
        self.execution.outcome = outcome;
        self.decision(Some(content), None)
    }
    fn limitation(&mut self, reason: &str, continue_llm: bool) -> Decision {
        self.execution.handoff = Some(Handoff {
            reason: reason.into(),
            continuation_required: continue_llm,
            completed_action_ids: self.observed.iter().map(|o| o.call.id.clone()).collect(),
        });
        self.end(reason.to_string(), Outcome::CapabilityLimitation)
    }
    fn tool(&mut self, name: &str, args: Value) -> Result<Result<String, Decision>, String> {
        if self.observed.len() >= 8 {
            return Ok(Err(
                self.limitation("the eight-action budget was reached", false)
            ));
        }
        let call = ToolCall {
            id: format!(
                "call_nano_{}_{}",
                self.execution.turn_id,
                &digest(format!("{}:{}:{name}:{args}", self.key, self.observed.len()).as_bytes())
                    [..32]
            ),
            kind: "function".into(),
            function: FunctionCall {
                name: name.into(),
                arguments: args.to_string(),
            },
        };
        let mut assistant_position = None;
        for (index, message) in self.request.messages[self.start + 1..].iter().enumerate() {
            if message.role == "assistant" && message.tool_calls.iter().any(|c| c.id == call.id) {
                if !message.tool_calls.iter().any(|c| {
                    c.id == call.id
                        && c.kind == "function"
                        && c.function.name == name
                        && serde_json::from_str::<Value>(&c.function.arguments).ok()
                            == Some(args.clone())
                }) {
                    return Err("Nano action arguments do not match".into());
                }
                assistant_position = Some(index);
            }
            if message.role == "tool" && message.tool_call_id.as_deref() == Some(call.id.as_str()) {
                if assistant_position.is_none_or(|p| p >= index) {
                    return Err("tool result has no preceding matching Nano action".into());
                }
                let text = message
                    .content
                    .as_str()
                    .ok_or("text tool result required")?
                    .to_string();
                let failed = text.starts_with("Error:")
                    || text.starts_with("error:")
                    || text.contains("user denied")
                    || name == "run_command" && !text.starts_with("exit code: 0\n");
                self.observed.push(ObservedAction {
                    call: call.clone(),
                    result_digest: digest(text.as_bytes()),
                    status: if failed { "failed" } else { "completed" }.into(),
                });
                if failed {
                    return Ok(Err(self.end(
                        format!(
                            "{name} did not complete successfully. No retry was attempted.\n\n{}",
                            excerpt(&text, 6000)
                        ),
                        Outcome::ToolFailed,
                    )));
                }
                return Ok(Ok(text));
            }
        }
        if !self.request.permits_tool(name) {
            return Ok(Err(self.limitation(
                &format!("{name} is unavailable under the caller's current tool policy"),
                false,
            )));
        }
        self.execution.outcome = Outcome::ToolRequired;
        Ok(Err(self.decision(None, Some(call))))
    }
}
macro_rules! tool {
    ($w:expr,$name:expr,$args:expr) => {
        match $w.tool($name, $args)? {
            Ok(text) => text,
            Err(decision) => return Ok(decision),
        }
    };
}
/// One search plus at most six file windows leaves space within Nano's
/// eight-action turn budget. No persisted index can outlive caller evidence.
fn rust_navigation(
    w: &mut Workflow<'_>,
    symbol: &str,
    kind: navigation::NavigationKind,
) -> Result<Decision, String> {
    if !navigation::valid_symbol(symbol) {
        return Ok(w.end(
            "Specify one Rust identifier; qualified paths and expressions are not supported."
                .into(),
            Outcome::Clarification,
        ));
    }
    for required in ["search_code", "read_file"] {
        if !w.request.permits_tool(required) {
            return Ok(w.limitation(&format!("Rust navigation requires an offered {required} tool under the caller's current policy."), false));
        }
    }
    let search = tool!(
        w,
        "search_code",
        json!({"query":symbol.trim_start_matches("r#"),"path":".","glob":"**/*.rs","regex":false,"context_lines":0,"limit":50})
    );
    let search: Value =
        serde_json::from_str(&search).map_err(|_| "invalid navigation search result")?;
    let mut paths = std::collections::BTreeSet::new();
    for hit in search["matches"]
        .as_array()
        .ok_or("navigation search result has no matches")?
    {
        let path = hit["path"]
            .as_str()
            .filter(|p| safe_path(p, false) && p.ends_with(".rs"))
            .ok_or("invalid Rust navigation search path")?;
        paths.insert(path.to_string());
    }
    let mut incomplete = search["scan_limited"] == true || !search["next_offset"].is_null();
    let mut skipped = Vec::new();
    let mut results = Vec::new();
    let mut parsed_files = Vec::new();
    let mut reads = 0;
    for path in paths {
        let mut source = String::new();
        let mut start = 1;
        let mut expected_total = None;
        loop {
            if reads >= 6 {
                incomplete = true;
                skipped.push(json!({"path":path,"reason":"six-window navigation budget reached"}));
                break;
            }
            let text = tool!(
                w,
                "read_file",
                json!({"path":path,"start_line":start,"line_count":300})
            );
            reads += 1;
            let (window, end, total) = match navigation::read_window(&text, start) {
                Ok(window) => window,
                Err(reason) => {
                    incomplete = true;
                    skipped.push(json!({"path":path,"reason":reason}));
                    break;
                }
            };
            if expected_total.is_some_and(|previous| previous != total) {
                incomplete = true;
                skipped
                    .push(json!({"path":path,"reason":"file line count changed between windows"}));
                break;
            }
            expected_total = Some(total);
            source.push_str(window);
            if !source.ends_with('\n') {
                source.push('\n');
            }
            if end == total {
                match navigation::analyze(&path, &source, symbol, &kind) {
                    Ok(matches) => {
                        results.extend(matches);
                        parsed_files.push(path.clone());
                    }
                    Err(reason) => {
                        incomplete = true;
                        skipped.push(json!({"path":path,"reason":reason}));
                    }
                }
                break;
            }
            start = end + 1;
        }
    }
    let results_truncated = results.len() > 50;
    results.truncate(50);
    let mut report = json!({"workflow":kind.label(),"symbol":symbol,"language":"rust",
        "resolution":"syntax_name_match","matches":results,"parsed_files":parsed_files,
        "coverage_incomplete":incomplete,"results_truncated":results_truncated,"skipped_files":skipped,
        "limitations":["Same-name symbols may refer to different definitions; types and imports are not resolved.","Macros are not expanded and conditional compilation is not evaluated.","Only searched files fully read and parsed are analyzed; zero matches does not prove absence outside this coverage."]});
    // Keep the report valid JSON and preserve coverage warnings under output limits.
    while serde_json::to_string_pretty(&report).unwrap().len() > 8000 {
        if !report["matches"].as_array().unwrap().is_empty() {
            report["matches"].as_array_mut().unwrap().pop();
            report["results_truncated"] = json!(true);
        } else if !report["skipped_files"].as_array().unwrap().is_empty() {
            report["skipped_files"].as_array_mut().unwrap().pop();
            report["coverage_details_truncated"] = json!(true);
        } else if !report["parsed_files"].as_array().unwrap().is_empty() {
            report["parsed_files"].as_array_mut().unwrap().pop();
            report["coverage_details_truncated"] = json!(true);
        } else {
            break;
        }
    }
    Ok(w.end(
        serde_json::to_string_pretty(&report).unwrap(),
        Outcome::Completed,
    ))
}

pub fn excerpt(text: &str, limit: usize) -> String {
    let out: String = text.chars().take(limit).collect();
    if text.chars().count() > limit {
        format!("{out}\n[Output excerpt truncated.]")
    } else {
        out
    }
}
fn payload(text: &str) -> &str {
    if text.starts_with("Lines ") {
        text.split_once('\n').map_or(text, |(_, rest)| rest)
    } else {
        text
    }
}
fn files(text: &str) -> Result<(Vec<String>, bool), String> {
    let value: Value = serde_json::from_str(text).map_err(|_| "invalid find_files result")?;
    let paths = value["files"]
        .as_array()
        .ok_or("missing files")?
        .iter()
        .map(|v| {
            v.as_str()
                .filter(|p| safe_path(p, false))
                .map(str::to_string)
                .ok_or("invalid tool path")
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((
        paths,
        value["scan_limited"] == true || !value["next_offset"].is_null(),
    ))
}
pub fn decide_with_model(
    request: &ChatRequest,
    deployment: &str,
    turn_override: Option<&str>,
    model: &Model,
) -> Result<Decision, String> {
    let start = request.validate()?;
    let route = profiles::route(
        &request.model,
        request.messages[start].content.as_str().unwrap_or(""),
    );
    for (index, message) in request.messages[start + 1..].iter().enumerate() {
        if message.role == "tool"
            && !request.messages[start + 1..start + 1 + index]
                .iter()
                .any(|prior| {
                    prior.role == "assistant"
                        && prior
                            .tool_calls
                            .iter()
                            .any(|call| Some(call.id.as_str()) == message.tool_call_id.as_deref())
                })
        {
            return Err("tool result has no preceding assistant action".into());
        }
    }
    let mut prompt = request.messages[start]
        .content
        .as_str()
        .unwrap_or("")
        .trim()
        .to_string();
    if start > 0
        && let Some(previous) = request.messages[..start]
            .iter()
            .rev()
            .find(|m| m.role == "assistant")
            .and_then(|m| m.content.as_str())
    {
        if let Some(plugin) = companion::pending_consent_plugin(previous)
            && companion::is_consent_reply(&prompt)
        {
            // The previous turn asked to download a specific plugin, so this reply approves
            // that exact plugin rather than a hard-coded default.
            prompt = format!("--tools:install {plugin}");
        } else if previous == "Which project-relative file should I read?"
            && paths(&prompt).len() == 1
        {
            prompt = format!("read {prompt}");
        } else if previous == "Which literal text or symbol should I search for?" {
            prompt = format!("search code for {prompt}");
        } else if previous == "Which configured check should I run: test, build, or lint?" {
            prompt = format!("run {prompt}");
        }
    }
    let flow_options = profiles::flow_options(&prompt);
    let is_flow = route.selected == nioxs_contracts::MODEL_FLOW;
    let thinker_options = profiles::thinker_options(&prompt);
    let thinker_with_helm = route.selected == nioxs_contracts::MODEL_THINKER && thinker_options.1;
    let intent = if is_flow && flow_options.0.is_empty() {
        Intent::Clarify(
            "Usage: --no-helm <task> or --helm <task>. Flow uses its specialists without the final Helm gate by default."
                .into(),
        )
    } else if route.selected == nioxs_contracts::MODEL_THINKER
        && thinker_with_helm
        && thinker_options.0.is_empty()
    {
        Intent::Clarify("Usage: --with-helm <task>. NioThinker runs its three reflection phases first, then NioHelm reviews the result.".into())
    } else if profiles::direct_answer(route.selected) {
        Intent::Coding {
            prompt: if thinker_with_helm {
                thinker_options.0
            } else {
                prompt.clone()
            },
            guarded: false,
        }
    } else if companion::is_consent_reply(&prompt) {
        // Without a pending prompt there is nothing to approve; never install on a bare "yes".
        Intent::Clarify(
            "There's no pending confirmation to approve right now. Tell me what you'd like to do — \
            for example `--tools:list`, `read Cargo.toml`, or `--code: <request>`."
                .into(),
        )
    } else {
        let parsed = parse(model, &prompt);
        if route.selected != MODEL_NANO
            && (!prompt.starts_with("--") || is_flow)
            && !matches!(parsed, Intent::Help | Intent::Config(_))
        {
            Intent::Coding {
                prompt: if is_flow {
                    flow_options.0
                } else {
                    prompt.clone()
                },
                guarded: profiles::guarded(route.selected),
            }
        } else {
            parsed
        }
    };
    let request_digest = digest(
        &serde_json::to_vec(&request.messages[..=start]).map_err(|_| "cannot encode history")?,
    );
    let turn = match turn_override {
        Some(id)
            if id.starts_with("turn_")
                && (6..=64).contains(&id.len())
                && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') =>
        {
            id.to_string()
        }
        Some(_) => return Err("invalid X-NioXS-Turn-ID".into()),
        None => request.messages[start + 1..]
            .iter()
            .filter(|message| message.role == "assistant")
            .flat_map(|message| &message.tool_calls)
            .find_map(|call| {
                call.id
                    .strip_prefix("call_nano_")?
                    .rsplit_once('_')
                    .map(|(turn, _)| turn.to_string())
            })
            .filter(|turn| {
                turn.starts_with("turn_")
                    && (6..=64).contains(&turn.len())
                    && turn.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
            })
            .unwrap_or_else(|| {
                format!(
                    "turn_{}",
                    &digest(
                        format!(
                            "{deployment}:{request_digest}:{}:{}",
                            std::process::id(),
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_nanos()
                        )
                        .as_bytes()
                    )[..32]
                )
            }),
    };
    let mut w = Workflow {
        request,
        start,
        key: format!(
            "{deployment}:{turn}:{request_digest}:{}",
            model.model_version
        ),
        execution: Execution {
            schema_version: SCHEMA_VERSION,
            turn_id: turn,
            provider: PROVIDER.into(),
            model: route.selected.into(),
            outcome: Outcome::CapabilityLimitation,
            handoff: None,
        },
        request_digest,
        intent: intent.label().into(),
        observed: vec![],
        stream_segments: vec![],
        coding: None,
    };
    match intent {

        Intent::Connected(steps) => workflows::execute(&mut w, &steps),
        Intent::Capabilities => Ok(w.end(
            serde_json::to_string_pretty(&tools::session_capabilities(request)).unwrap(),
            Outcome::Completed,
        )),
        Intent::GitInspect(view) => {
            let command = view.command();
            let result = tool!(w, "run_command", json!({"command":command}));
            Ok(w.end(format!("Git inspection (`{command}`):\n\n{}", excerpt(&result, 6500)), Outcome::Completed))
        }
        Intent::Diagnostics(kind) => {
            if !request.permits_tool("run_command") {
                return Ok(w.limitation("Rust diagnostics require an offered run_command tool and caller command approval.", false));
            }
            let inventory = tool!(w, "find_files", json!({"path":".","glob":"Cargo.toml","limit":50}));
            let (paths, limited) = files(&inventory)?;
            if !paths.iter().any(|p| p == "Cargo.toml") {
                return Ok(w.limitation(if limited { "Cargo.toml was not found in the incomplete inventory; Rust diagnostics are unavailable." } else { "Rust diagnostics require a root Cargo.toml; other languages are not supported yet." }, false));
            }
            let command = kind.command();
            let result = tool!(w, "run_command", json!({"command":command}));
            Ok(w.end(format!("Rust diagnostics (`{command}`):\n\n{}", excerpt(&result, 6500)), Outcome::Completed))
        }
        Intent::ProjectTasks => {
            let inventory = tool!(w, "find_files", json!({"path":".","glob":"{Cargo.toml,package.json}","limit":50}));
            let (paths, limited) = files(&inventory)?;
            let package = if paths.iter().any(|p| p == "package.json") {
                Some(tool!(w, "read_file", json!({"path":"package.json","line_count":200})))
            } else { None };
            let tasks = tools::project_tasks(&paths, package.as_deref().map(payload), limited)?;
            Ok(w.end(excerpt(&serde_json::to_string_pretty(&tasks).unwrap(), 8000), Outcome::Completed))
        }

        Intent::Unsupported => Ok(w.limitation(
            "⚡ **NioNano Companion Ready.** I couldn't identify a direct matching programming action for that phrasing.\n\n\
            Here are quick actions you can run right now:\n\
            • ⚡ **Run Commands**: `--bash <cmd>` (e.g. `--bash git status`)\n\
            • 🧪 **Run Tests**: `--test [pattern]`\n\
            • 🔌 **Plugins & Tools**: `--tools list`\n\
            • 🛡️ **Guarded Coding**: `--code guard <request>`\n\
            • 🔍 **Inspect**: `inspect project`, `analyze codebase`, or `where is <symbol> defined?`\n\
            • ❓ **Help**: Type `help` for full capability catalog.",
            false,
        )),
        Intent::Bash(cmd) => {
            let res = companion::execute_in_session_bash(&cmd);
            let markdown = format!(
                "{}\n{}",
                res.to_markdown(),
                companion::bash_discovery_footer()
            );
            Ok(w.end(markdown, Outcome::Completed))
        }
        Intent::Tool(task) => {
            let res = companion::dispatch_tool(&task);
            let outcome = match &res {
                companion::CompanionResult::ToolNeedsConsent { .. } => Outcome::Clarification,
                companion::CompanionResult::ToolFailed { .. } => Outcome::ToolFailed,
                _ => Outcome::Completed,
            };
            Ok(w.end(res.to_markdown(), outcome))
        }
        Intent::Config(req) => {
            let res = companion::dispatch_config(&req);
            Ok(w.end(res, Outcome::Completed))
        }
        Intent::TestRunner(pattern) => {
            // Determine project test runner
            let test_cmd = if std::path::Path::new("Cargo.toml").exists() {
                if pattern.is_empty() {
                    "cargo test".to_string()
                } else {
                    format!("cargo test {pattern}")
                }
            } else if std::path::Path::new("package.json").exists() {
                if pattern.is_empty() {
                    "npm test".to_string()
                } else {
                    format!("npm test -- {pattern}")
                }
            } else if std::path::Path::new("pytest.ini").exists() || std::path::Path::new("pyproject.toml").exists() {
                if pattern.is_empty() {
                    "pytest".to_string()
                } else {
                    format!("pytest -k '{pattern}'")
                }
            } else if std::path::Path::new("go.mod").exists() {
                if pattern.is_empty() {
                    "go test ./...".to_string()
                } else {
                    format!("go test -run '{pattern}' ./...")
                }
            } else {
                format!("echo 'No known test configuration detected for pattern: {pattern}'")
            };

            let res = companion::execute_in_session_bash(&test_cmd);
            Ok(w.end(res.to_markdown(), Outcome::Completed))
        }
        Intent::GitSummary(sub) => {
            let cmd = match sub.as_str() {
                "status" => "git status -s",
                "diff" => "git diff --stat",
                _ => "git status -s && git diff --stat",
            };
            let res = companion::execute_in_session_bash(cmd);
            Ok(w.end(res.to_markdown(), Outcome::Completed))
        }
        Intent::Coding { prompt, guarded } => {
            let guarded = guarded || profiles::guarded(route.selected);
            let (clean_prompt, pinned_files) = companion::extract_pinned_files(&prompt);
            let target_prompt = if clean_prompt.is_empty() { prompt.as_str() } else { clean_prompt.as_str() };
            let prompt_text = target_prompt.trim();
            if prompt_text.is_empty() && pinned_files.is_empty() {
                return Ok(w.end(
                    "Usage: `--code <request>` or `--code @<file> <request>`.\n\n\
                    Example: `--code @src/net.rs add retry logic to the HTTP client`"
                        .to_string(),
                    Outcome::Clarification,
                ));
            }
            let config = companion::coding_config();
            // A tool-capable caller owns the working directory, so the model
            // explores it itself: the provider offers `read_file`/`find_files`
            // upstream, relays the model's own tool calls to the client, and
            // replays the results on the next round. NioNano never picks files.
            // Callers without those tools fall back to a local bounded bundle.
            if profiles::direct_answer(route.selected) || !profiles::project_evidence_relevant(prompt_text) || (w.request.permits_read() && w.request.permits_tool("find_files")) {
                let turn = &w.request.messages[w.start..];
                let rounds = companion::coding_tool_rounds(turn);
                let tools = if rounds < profiles::tool_rounds(route.selected) {
                    let schemas = if profiles::direct_answer(route.selected) {
                        w.request.tools.iter().filter(|tool| profiles::answer_tool(route.selected, tool["function"]["name"].as_str().unwrap_or(""))).cloned().collect()
                    } else { companion::coding_tool_schemas() };
                    schemas.into_iter()
                        .filter(|tool| tool["function"]["name"] == "web_fetch" || profiles::project_evidence_relevant(prompt_text))
                        .filter(|tool| w.request.permits_tool(tool["function"]["name"].as_str().unwrap_or("")))
                        .collect()
                } else {
                    Vec::new()
                };
                let mut initial_files = pinned_files.clone();
                initial_files.extend(companion::coding_read_paths(turn));
                let mut messages = profiles::messages(route.selected, companion::upstream_messages(prompt_text, &w.request.messages));
                if tools.is_empty() {
                    messages.push(json!({
                        "role": "system",
                        "content": "No further inspection tools are available for this request (not offered or budget exhausted). Do not request or emit any more tool calls, including XML tool-call text. Answer now using the user request and the tool results already present in the conversation. State clearly if the available evidence is insufficient."
                    }));
                }
                w.coding = Some(companion::CodingConsult {
                    config,
                    prompt: prompt_text.to_string(),
                    messages,
                    tools,
                    files: initial_files,
                    guarded,
                });
                w.execution.outcome = Outcome::Completed;
                return Ok(w.decision(None, None));
            }
            let bundle = companion::bundle_workspace_context_pinned(prompt_text, &pinned_files);
            // Both streaming and JSON clients use the same provider-owned
            // consultation, including catalog selection and upstream metadata.
            w.coding = Some(companion::CodingConsult {
                config,
                prompt: prompt_text.to_string(),
                messages: profiles::messages(route.selected, companion::bundled_messages(prompt_text, &bundle.context, &w.request.messages)),
                files: bundle.files,
                tools: Vec::new(),
                guarded,
            });
            w.execution.outcome = Outcome::Completed;
            Ok(w.decision(None, None))
        }
        Intent::Clarify(question) => Ok(w.end(question, Outcome::Clarification)),
        Intent::Read(path) => {
            let text = tool!(w, "read_file", json!({"path": path, "line_count": 80}));
            let response = synthesize::synthesize_read_response(&path, &text);
            Ok(w.end(response, Outcome::Completed))
        }
        Intent::Files(glob) => {
            let text = tool!(w, "find_files", json!({"path": ".", "glob": glob, "limit": 50}));
            let (paths, limited) = files(&text)?;
            let count = paths.len();
            let body = paths.iter().map(|p| format!("• `{p}`")).collect::<Vec<_>>().join("\n");
            let notice = if limited { "\n\n*(Showing first 50 files)*" } else { "" };
            Ok(w.end(
                format!("Found {count} file{} matching `{glob}`:\n\n{body}{notice}", if count == 1 { "" } else { "s" }),
                Outcome::Completed,
            ))
        }
        Intent::Search(query) => {
            let text = tool!(w, "search_code", json!({"query": query, "path": ".", "regex": false, "context_lines": 1, "limit": 30}));
            let value: Value = serde_json::from_str(&text).map_err(|_| "invalid search result")?;
            let mut ranked: Vec<(usize, String)> = value["matches"].as_array().ok_or("missing matches")?.iter().map(|entry| {
                let path = entry["path"].as_str().filter(|p| safe_path(p, false)).ok_or("invalid search path")?;
                let line = entry["line"].as_u64().filter(|n| *n > 0).ok_or("invalid search line")?;
                let snippet = entry["excerpt"].as_str().ok_or("missing excerpt")?;
                Ok((snippet.to_ascii_lowercase().matches(&query.to_ascii_lowercase()).count(), format!("• `{path}:{line}`\n```\n{snippet}\n```")))
            }).collect::<Result<Vec<_>, String>>()?;
            ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            let count = ranked.len();
            if count == 0 {
                Ok(w.end(
                    format!("No code occurrences of symbol `{query}` were found in this project."),
                    Outcome::Completed,
                ))
            } else {
                let limited = value["scan_limited"] == true || !value["next_offset"].is_null();
                let matches_str = ranked.iter().map(|(_, row)| row.as_str()).collect::<Vec<_>>().join("\n\n");
                let notice = if limited { "\n\n*(Search scan coverage limited)*" } else { "" };
                let s = if count == 1 { "" } else { "s" };
                Ok(w.end(
                    format!("Found {count} occurrence{s} of `{query}`:\n\n{matches_str}{notice}"),
                    Outcome::Completed,
                ))
            }
        }
        Intent::Inspect => {
            let text = tool!(w, "find_files", json!({"path": ".", "limit": 50}));
            let (paths, limited) = files(&text)?;
            let manifests: Vec<&String> = paths.iter().filter(|p| ["Cargo.toml", "package.json", "pyproject.toml", "go.mod", "Makefile"].contains(&p.as_str())).take(4).collect();

            let mut stack = Vec::new();
            if paths.iter().any(|p| p.ends_with(".rs") || p == "Cargo.toml") {
                stack.push("Rust (Cargo workspace / crates)");
            }
            if paths.iter().any(|p| p.ends_with(".ts") || p.ends_with(".js") || p == "package.json") {
                stack.push("TypeScript / JavaScript (Node.js)");
            }
            if paths.iter().any(|p| p.ends_with(".py") || p == "pyproject.toml") {
                stack.push("Python");
            }
            if paths.iter().any(|p| p.ends_with(".go") || p == "go.mod") {
                stack.push("Go");
            }

            let mut report = String::new();
            if !stack.is_empty() {
                report.push_str(&format!("### Project Overview\n\n• **Technology Stack**: {}\n• **Workspace Size**: {} files discovered\n", stack.join(", "), paths.len()));
            } else {
                report.push_str(&format!("Project Workspace ({} files discovered):\n", paths.len()));
            }

            for path in &manifests {
                let content = tool!(w, "read_file", json!({"path": path, "line_count": 200}));
                report.push_str(&format!("\nConfiguration from `{path}`:\n```\n{}\n```\n", excerpt(&content, 2200)));
                if path.as_str() == "package.json" && let Ok(package) = serde_json::from_str::<Value>(payload(&content)) {
                    report.push_str(&format!("Package: {}; version: {}.\n", package["name"].as_str().unwrap_or("not declared"), package["version"].as_str().unwrap_or("not declared")));
                    if let Some(deps) = package["dependencies"].as_object() {
                        report.push_str(&format!("Declared dependencies: {}.\n", deps.keys().take(20).cloned().collect::<Vec<_>>().join(", ")));
                    }
                }
            }
            if manifests.is_empty() {
                report.push_str("\nFiles:\n");
                report.push_str(&paths.iter().map(|p| format!("• `{p}`")).collect::<Vec<_>>().join("\n"));
                report.push('\n');
            }
            if limited {
                report.push_str("*(Inventory truncated to first 50 files)*\n");
            }
            Ok(w.end(excerpt(&report, 8000), Outcome::Completed))
        }
        Intent::Checks(kind) => {
            if !request.permits_tool("run_command") {
                return Ok(w.limitation("commands require an offered run_command tool and Nio's Build-mode approval", false));
            }
            let text = tool!(w, "find_files", json!({"path": ".", "limit": 50}));
            let (paths, limited) = files(&text)?;
            let (command, source): (String, String) = if paths.iter().any(|p| p == ".nioxs.json") {
                let text = tool!(w, "read_file", json!({"path": ".nioxs.json", "line_count": 200}));
                let config: Value = serde_json::from_str(payload(&text)).map_err(|_| "invalid or truncated .nioxs.json")?;
                if config["schema_version"] != 1 {
                    return Err("unsupported .nioxs.json schema".into());
                }
                let Some(command) = config["commands"][&kind].as_str() else {
                    return Ok(w.end(format!("No {kind} command configured in .nioxs.json."), Outcome::Clarification));
                };
                let allowed = ["cargo test --locked", "cargo build --locked", "cargo clippy --locked -- -D warnings", "npm test", "npm run test", "npm run build", "npm run lint", "python3 -m unittest", "python3 -m pytest", "go test ./...", "go build ./...", "go vet ./..."];
                if !allowed.contains(&command) {
                    return Ok(w.limitation("configured command outside Nano's supported named-command set", false));
                }
                (command.into(), ".nioxs.json".into())
            } else if paths.iter().any(|p| p == "Cargo.toml") {
                (match kind.as_str() { "test" => "cargo test --locked", "build" => "cargo build --locked", _ => "cargo clippy --locked -- -D warnings" }.into(), "Cargo.toml (fixed Rust mapping)".into())
            } else if paths.iter().any(|p| p == "package.json") {
                let text = tool!(w, "read_file", json!({"path": "package.json", "line_count": 200}));
                let package: Value = serde_json::from_str(payload(&text)).map_err(|_| "invalid or truncated package.json")?;
                if package["scripts"][&kind].as_str().is_none() {
                    return Ok(w.end(format!("package.json has no {kind} script."), Outcome::Clarification));
                }
                (format!("npm run {kind}"), "package.json scripts".into())
            } else {
                return Ok(w.end(format!("No supported {kind} command found{}; configure .nioxs.json.", if limited { " in incomplete inventory" } else { "" }), Outcome::Clarification));
            };
            let result = tool!(w, "run_command", json!({"command": command}));
            Ok(w.end(format!("{kind} checks completed successfully (`{command}` from {source}):\n\n```\n{}\n```", excerpt(&result, 6500)), Outcome::Completed))
        }
        Intent::WebFetch(url) => {
            let text = if request.permits_tool("web_fetch") {
                tool!(w, "web_fetch", json!({"url": url}))
            } else {
                run_async(nioxs_browser::fetch_url(&url))
                    .map_err(|e| format!("web fetch error: {e}"))?
            };
            let clean = nioxs_browser::extract_text(&text);
            Ok(w.end(
                format!("Web content from {url}:\n\n{}", excerpt(&clean, 8000)),
                Outcome::Completed,
            ))
        }
        Intent::Research(query) => {
            let report = run_async(nioxs_browser::research(&query, 4))
                .map_err(|e| format!("research error: {e}"))?;
            if report.passages.is_empty() {
                Ok(w.end(
                    format!("No informative public web sources found for '{query}'."),
                    Outcome::Completed,
                ))
            } else {
                let synthesized = synthesize::synthesize_research_answer(&query, &report);
                Ok(w.end(excerpt(&synthesized, 8000), Outcome::Completed))
            }
        }
        Intent::Wiki(query) => {
            let results = run_async(nioxs_browser::search_wikipedia(&query, 3))
                .map_err(|e| format!("wikipedia error: {e}"))?;
            if results.is_empty() {
                Ok(w.end(
                    format!("No Wikipedia articles found for '{query}'."),
                    Outcome::Completed,
                ))
            } else {
                let top = &results[0];
                let article_text = match run_async(nioxs_browser::fetch_url(&top.url)) {
                    Ok(html) => nioxs_browser::extract_text(&html),
                    Err(_) => top.snippet.clone(),
                };
                let report = nioxs_browser::ResearchReport {
                    query: query.clone(),
                    passages: vec![nioxs_browser::ResearchPassage {
                        url: top.url.clone(),
                        title: top.title.clone(),
                        excerpt: article_text,
                        snippet: top.snippet.clone(),
                        score: 100,
                    }],
                    sources_visited: vec![top.url.clone()],
                };
                let synthesized = synthesize::synthesize_research_answer(&query, &report);
                Ok(w.end(excerpt(&synthesized, 8000), Outcome::Completed))
            }
        }
        Intent::Github(repo) => {
            let content = run_async(nioxs_browser::fetch_github_repo(&repo))
                .map_err(|e| format!("github fetch error: {e}"))?;
            Ok(w.end(excerpt(&content, 8000), Outcome::Completed))
        }
        Intent::Package(pkg) => {
            // Check crates.io first; if not found, check npm registry
            let metadata = match run_async(nioxs_browser::fetch_crate_info(&pkg)) {
                Ok(meta) => meta,
                Err(_) => match run_async(nioxs_browser::fetch_npm_info(&pkg)) {
                    Ok(meta) => meta,
                    Err(e) => return Ok(w.end(format!("Package '{pkg}' not found: {e}"), Outcome::Completed)),
                },
            };
            let mut report = format!(
                "Package {} ({})\nVersion: {}\nDescription: {}\n",
                metadata.name, metadata.ecosystem, metadata.latest_version, metadata.description
            );
            if let Some(homepage) = metadata.homepage {
                report.push_str(&format!("Homepage: {homepage}\n"));
            }
            if let Some(repo) = metadata.repository {
                report.push_str(&format!("Repository: {repo}\n"));
            }
            if let Some(doc) = metadata.documentation {
                report.push_str(&format!("Documentation: {doc}\n"));
            }
            Ok(w.end(report, Outcome::Completed))
        }
        Intent::Scaffold(template_name) => {
            match scaffold_template(&template_name) {
                Some((filename, content)) => {
                    Ok(w.end(
                        format!("Deterministic scaffold template for '{filename}':\n\n```\n{content}```"),
                        Outcome::Completed,
                    ))
                }
                None => {
                    Ok(w.end(
                        format!("Unknown scaffold template '{template_name}'. Available: gitignore, github-ci, license-mit"),
                        Outcome::Clarification,
                    ))
                }
            }
        }
        Intent::Greeting => {
            let turn = GREETING_ROTATION.fetch_add(1, Ordering::Relaxed);
            let greetings = [
                "⚡ **Hello! I'm NioNano MicroLM (μLM)** — your sovereign in-TUI coding companion.\n\n\
                I run as an ultra-fast (<0.5ms) decision layer alongside your coding agent.\n\n\
                #### 🎯 In-Session Directives\n\
                • `--bash <cmd>` — Run in-session terminal commands without exiting your TUI\n\
                • `--test [pattern]` — Auto-detect runner (`cargo`, `npm`, `pytest`, `go`) and trim error traces\n\
                • `--git status` / `--git diff` — Summarize working tree diffs and modified symbols\n\
                • `--tools <task>` — Run specialized Python plugins (e.g. svg2favicon, asset resizers)\n\
                • `--code guard <prompt>` — Anti-hallucination reality checks against project AST symbols\n\n\
                #### 🔬 Quick Inspections\n\
                • `what is my project build on?` / `inspect project`\n\
                • `analyze codebase` or `analyze <file>`\n\
                • `where is <symbol> defined?` / `who calls <symbol>?`\n\n\
                What would you like to inspect, run, or diagnose today?",
                "🧠 **Welcome! NioNano Coding Companion Ready.**\n\n\
                Trained for sovereign code exploration, AST diagnostic triage, and live assistance:\n\
                • ⚡ **Terminal**: `--bash cargo test` or `--test`\n\
                • 🛡️ **Guarded Coding**: `--code guard <req>` (verified against project AST symbols)\n\
                • 🔌 **Plugins**: `--tools list` or `--tools <task>`\n\
                • 📊 **Git**: `--git status` or `review git diff`\n\n\
                How can I assist your coding session?",
            ];
            let msg = greetings[turn % greetings.len()];
            Ok(w.end(msg.to_string(), Outcome::Completed))
        }
        Intent::Help => {
            Ok(w.end(
                companion::dispatch_help(""),
                Outcome::Completed,
            ))
        }
        Intent::HelpTopic(topic) => {
            Ok(w.end(
                companion::dispatch_help(&topic),
                Outcome::Completed,
            ))
        }
        Intent::AnalyzeFile(path) => {
            let text = tool!(w, "read_file", json!({"path": path, "line_count": 500}));
            let payload_text = payload(&text);
            let analysis = analysis::analyze_file_content(&path, payload_text);
            let report = analysis::format_file_analysis(&analysis);
            Ok(w.end(report, Outcome::Completed))
        }
        Intent::AnalyzeDir(dir) => {
            let text = tool!(w, "find_files", json!({"path": ".", "limit": 100}));
            let (paths, _) = files(&text)?;
            let filtered: Vec<String> = if dir == "." {
                paths
            } else {
                paths.into_iter().filter(|p| p.starts_with(&dir)).collect()
            };
            let report = analysis::format_directory_analysis(&dir, &filtered);
            Ok(w.end(report, Outcome::Completed))
        }
        Intent::SymbolDef(symbol) => rust_navigation(&mut w, &symbol, navigation::NavigationKind::Definitions),
        Intent::SymbolRefs(symbol) => rust_navigation(&mut w, &symbol, navigation::NavigationKind::References),
        Intent::SymbolCallers(symbol) => rust_navigation(&mut w, &symbol, navigation::NavigationKind::Callers),
        Intent::SymbolImplementations(symbol) => rust_navigation(&mut w, &symbol, navigation::NavigationKind::Implementations),
        Intent::RefactorPatch { file, old, new } => workflows::execute(&mut w, &[workflows::Step::Patch { path:file, old, new }]),
        Intent::TddRefactor { file, old, new } => workflows::execute(&mut w, &[workflows::Step::Patch { path:file, old, new }, workflows::Step::Check {kind:"test".into()}]),
        Intent::ScaffoldTest { file, function } => {
            match coder::scaffold_test_for_function(&file, &function) {
                Ok(code) => Ok(w.end(
                    format!("### Generated Test Scaffold for `{function}` (`{file}`):\n\n```\n{code}```"),
                    Outcome::Completed,
                )),
                Err(err) => Ok(w.end(format!("Test scaffolding error: {err}"), Outcome::Completed)),
            }
        }
        Intent::ReviewDiff => {
            match coder::review_git_diff() {
                Ok(report) => Ok(w.end(report, Outcome::Completed)),
                Err(err) => Ok(w.end(format!("Diff review failed: {err}"), Outcome::Completed)),
            }
        }
        Intent::SuggestCommit => {
            match coder::suggest_commit_message() {
                Ok(msg) => Ok(w.end(msg, Outcome::Completed)),
                Err(err) => Ok(w.end(format!("Commit generation failed: {err}"), Outcome::Completed)),
            }
        }
        Intent::ProjectSuggestions => {
            let suggestions = coder::generate_project_suggestions(".");
            Ok(w.end(suggestions, Outcome::Completed))
        }
        Intent::TaskPlan(task) => {
            let plan = coder::generate_task_plan(&task);
            Ok(w.end(plan, Outcome::Completed))
        }
        Intent::Plan(steps) => {
            if let Some(connected) = workflows::from_intents(&steps) {
                return workflows::execute(&mut w, &connected);
            }
            let mut summary = format!("Decomposed and executing {} task steps:\n\n", steps.len());
            for (idx, step) in steps.iter().enumerate() {
                let step_num = idx + 1;
                match step {
                    Intent::Read(path) => {
                        let text = tool!(w, "read_file", json!({"path": path, "line_count": 80}));
                        summary.push_str(&format!("### Step {step_num}: Read {path}\n{}\n\n", synthesize::synthesize_read_response(path, &text)));
                    }
                    Intent::Files(glob) => {
                        let text = tool!(w, "find_files", json!({"path": ".", "glob": glob, "limit": 50}));
                        let (paths, limited) = files(&text)?;
                        summary.push_str(&format!(
                            "### Step {step_num}: Files matching {glob}\n{}\n{}\n\n",
                            paths.iter().map(|p| format!("- {p}")).collect::<Vec<_>>().join("\n"),
                            if limited { "(Inventory incomplete)" } else { "" }
                        ));
                    }
                    Intent::Search(query) => {
                        let text = tool!(w, "search_code", json!({"query": query, "path": ".", "regex": false, "context_lines": 1, "limit": 20}));
                        summary.push_str(&format!("### Step {step_num}: Search code for '{query}'\n{}\n\n", excerpt(&text, 2000)));
                    }
                    Intent::Inspect => {
                        let text = tool!(w, "find_files", json!({"path": ".", "limit": 50}));
                        let (paths, _) = files(&text)?;
                        summary.push_str(&format!("### Step {step_num}: Project Inspection\nFound {} project files.\n\n", paths.len()));
                    }
                    Intent::AnalyzeFile(path) => {
                        let text = tool!(w, "read_file", json!({"path": path, "line_count": 500}));
                        let analysis = analysis::analyze_file_content(path, payload(&text));
                        summary.push_str(&format!("### Step {step_num}: Analysis of {path}\n{}\n\n", analysis::format_file_analysis(&analysis)));
                    }
                    Intent::AnalyzeDir(dir) => {
                        let text = tool!(w, "find_files", json!({"path": ".", "limit": 100}));
                        let (paths, _) = files(&text)?;
                        let filtered: Vec<String> = if dir == "." {
                            paths
                        } else {
                            paths.into_iter().filter(|p| p.starts_with(dir)).collect()
                        };
                        summary.push_str(&format!("### Step {step_num}: Codebase Analysis ({dir})\n{}\n\n", analysis::format_directory_analysis(dir, &filtered)));
                    }
                    Intent::Checks(kind) => {
                        if !request.permits_tool("run_command") {
                            summary.push_str(&format!("### Step {step_num}: {kind} check\n(Requires run_command permission)\n\n"));
                        } else {
                            let text = tool!(w, "find_files", json!({"path": ".", "limit": 50}));
                            let (paths, _) = files(&text)?;
                            let cmd = if paths.iter().any(|p| p == "Cargo.toml") {
                                match kind.as_str() {
                                    "test" => "cargo test --locked",
                                    "build" => "cargo build --locked",
                                    _ => "cargo clippy --locked -- -D warnings",
                                }
                            } else {
                                "npm test"
                            };
                            let result = tool!(w, "run_command", json!({"command": cmd}));
                            summary.push_str(&format!("### Step {step_num}: Run {cmd}\n{}\n\n", excerpt(&result, 2000)));
                        }
                    }
                    Intent::Wiki(q) => {
                        if let Ok(results) = run_async(nioxs_browser::search_wikipedia(q, 1)) {
                            if let Some(first) = results.first() {
                                summary.push_str(&format!("### Step {step_num}: Wikipedia: {}\n{}\n\n", first.title, first.snippet));
                            }
                        }
                    }
                    Intent::Package(p) => {
                        if let Ok(meta) = run_async(nioxs_browser::fetch_crate_info(p)) {
                            summary.push_str(&format!("### Step {step_num}: Crate {} v{}\n{}\n\n", meta.name, meta.latest_version, meta.description));
                        }
                    }
                    Intent::Scaffold(tpl) => {
                        if let Some((fname, content)) = scaffold_template(tpl) {
                            summary.push_str(&format!("### Step {step_num}: Scaffolding {fname}\n```\n{}\n```\n\n", content));
                        }
                    }
                    Intent::Research(q) => {
                        if let Ok(rep) = run_async(nioxs_browser::research(q, 2)) {
                            summary.push_str(&format!("### Step {step_num}: Web Research for '{q}'\nVisited {} sources, cited {} passages.\n\n", rep.sources_visited.len(), rep.passages.len()));
                        }
                    }
                    _ => {}
                }
            }
            Ok(w.end(summary.trim().to_string(), Outcome::Completed))
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use nioxs_contracts::MODEL_NANO;

    fn inspection_request(prompt: &str) -> ChatRequest {
        serde_json::from_value(json!({"model":MODEL_NANO,"messages":[{"role":"user","content":prompt}],
            "tools":tools::registry().iter().map(|entry| entry["schema"].clone()).collect::<Vec<_>>()
        })).unwrap()
    }

    fn return_result(request: &mut ChatRequest, decision: &Decision, result: &str) {
        let call = decision.call.as_ref().expect("caller tool call");
        request.messages.push(assistant_action(call));
        request.messages.push(tool_response(call, result));
    }

    #[test]
    fn navigation_uses_caller_evidence_for_all_four_views() {
        let source = "trait Service { fn ping(&self); }\nstruct Worker;\nimpl Service for Worker { fn ping(&self) {} }\nfn caller() { ping(); }";
        for (prompt, count, expected_label) in [
            ("--definitions ping", 2, "symbol_definitions"),
            ("--references ping", 1, "symbol_references"),
            ("who calls ping?", 1, "symbol_callers"),
            ("--implementations Service", 1, "symbol_implementations"),
        ] {
            let mut request = inspection_request(prompt);
            let search = decide(&request, "test").unwrap();
            assert_eq!(search.call.as_ref().unwrap().function.name, "search_code");
            return_result(
                &mut request,
                &search,
                r#"{"matches":[{"path":"caller_project/lib.rs"},{"path":"caller_project/lib.rs"}]}"#,
            );
            let read = decide(&request, "test").unwrap();
            assert_eq!(read.call.as_ref().unwrap().function.name, "read_file");
            let args: Value =
                serde_json::from_str(&read.call.as_ref().unwrap().function.arguments).unwrap();
            assert_eq!(args["path"], "caller_project/lib.rs");
            return_result(&mut request, &read, &format!("Lines 1-4 of 4:\n{source}"));
            let completed = decide(&request, "test").unwrap();
            assert_eq!(completed.execution.outcome, Outcome::Completed);
            assert_eq!(completed.observed.len(), 2);
            let report: Value = serde_json::from_str(&completed.content.unwrap()).unwrap();
            assert_eq!(report["workflow"], expected_label);
            assert_eq!(report["matches"].as_array().unwrap().len(), count);
            assert_eq!(report["coverage_incomplete"], false);
        }
    }

    #[test]
    fn navigation_reads_later_windows_and_keeps_line_numbers() {
        let mut request = inspection_request("--definitions late");
        let search = decide(&request, "test").unwrap();
        return_result(
            &mut request,
            &search,
            r#"{"matches":[{"path":"src/lib.rs"}],"scan_limited":true}"#,
        );
        let first = decide(&request, "test").unwrap();
        let window = "// padding\n".repeat(300);
        return_result(
            &mut request,
            &first,
            &format!("Lines 1-300 of 301:\n{window}"),
        );
        let next = decide(&request, "test").unwrap();
        let args: Value =
            serde_json::from_str(&next.call.as_ref().unwrap().function.arguments).unwrap();
        assert_eq!(args["start_line"], 301);
        return_result(&mut request, &next, "Lines 301-301 of 301:\nfn late() {}");
        let done = decide(&request, "test").unwrap();
        let report: Value = serde_json::from_str(&done.content.unwrap()).unwrap();
        assert_eq!(report["matches"][0]["line"], 301);
        assert_eq!(report["coverage_incomplete"], true);
    }

    #[test]
    fn navigation_budget_and_incomplete_reads_are_explicit() {
        let mut request = inspection_request("--callers target");
        let search = decide(&request, "test").unwrap();
        let hits: Vec<Value> = (0..8)
            .map(|i| json!({"path":format!("src/{i}.rs")}))
            .collect();
        return_result(&mut request, &search, &json!({"matches":hits}).to_string());
        for _ in 0..6 {
            let read = decide(&request, "test").unwrap();
            return_result(
                &mut request,
                &read,
                "Lines 1-1 of 1:\nfn caller() { target(); }",
            );
        }
        let done = decide(&request, "test").unwrap();
        assert!(done.call.is_none());
        assert_eq!(done.observed.len(), 7);
        let report: Value = serde_json::from_str(&done.content.unwrap()).unwrap();
        assert_eq!(report["matches"].as_array().unwrap().len(), 6);
        assert_eq!(report["skipped_files"].as_array().unwrap().len(), 2);
        assert_eq!(report["coverage_incomplete"], true);
        let mut request = inspection_request("--definitions target");
        let search = decide(&request, "test").unwrap();
        return_result(&mut request, &search, r#"{"matches":[{"path":"lib.rs"}]}"#);
        let read = decide(&request, "test").unwrap();
        return_result(&mut request, &read, "fn target() {}");
        let report: Value =
            serde_json::from_str(&decide(&request, "test").unwrap().content.unwrap()).unwrap();
        assert_eq!(report["coverage_incomplete"], true);
        assert!(report["matches"].as_array().unwrap().is_empty());
    }

    #[test]
    fn navigation_permissions_paths_and_tool_failures_fail_closed() {
        for prompt in [
            "--definitions target",
            "--references target",
            "--callers target",
            "--implementations Target",
        ] {
            for absent in ["read_file", "search_code"] {
                let mut request = inspection_request(prompt);
                request.tools.retain(|t| t["function"]["name"] != absent);
                let blocked = decide(&request, "test").unwrap();
                assert_eq!(blocked.execution.outcome, Outcome::CapabilityLimitation);
                assert!(blocked.call.is_none());
            }
        }
        let mut request = inspection_request("--references target");
        request.tool_choice = Some(json!("none"));
        assert!(decide(&request, "test").unwrap().call.is_none());
        request.tool_choice = None;
        let search = decide(&request, "test").unwrap();
        return_result(
            &mut request,
            &search,
            r#"{"matches":[{"path":"../outside.rs"}]}"#,
        );
        assert!(decide(&request, "test").is_err());
        request.messages.last_mut().unwrap().content = json!("Error: user denied");
        let failed = decide(&request, "test").unwrap();
        assert_eq!(failed.execution.outcome, Outcome::ToolFailed);
        assert!(failed.call.is_none());
    }

    #[test]
    fn navigation_output_limits_preserve_json_and_coverage_flags() {
        let mut request = inspection_request("--callers target");
        let search = decide(&request, "test").unwrap();
        return_result(&mut request, &search, r#"{"matches":[{"path":"lib.rs"}]}"#);
        let read = decide(&request, "test").unwrap();
        let calls = "target(); ".repeat(40);
        return_result(
            &mut request,
            &read,
            &format!("Lines 1-1 of 1:\nfn caller() {{ {calls} }}"),
        );
        let content = decide(&request, "test").unwrap().content.unwrap();
        assert!(content.len() <= 8000);
        let report: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(report["results_truncated"], true);
        assert_eq!(report["coverage_incomplete"], false);
        assert!(!report["limitations"].as_array().unwrap().is_empty());
    }

    #[test]
    fn inspection_capabilities_follow_caller_policy() {
        let mut request = inspection_request("--capabilities");
        request
            .tools
            .push(json!({"type":"function","function":{"name":"custom_plugin"}}));
        let decision = decide(&request, "test").unwrap();
        assert!(decision.call.is_none());
        let caps: Value = serde_json::from_str(decision.content.as_ref().unwrap()).unwrap();
        assert!(
            caps["tools"]
                .as_array()
                .unwrap()
                .iter()
                .all(|t| t["available"] == true)
        );
        assert!(
            caps["caller_offered_tools"]
                .as_array()
                .unwrap()
                .contains(&json!("custom_plugin"))
        );
        request.tool_choice = Some(json!("none"));
        let caps = tools::session_capabilities(&request);
        assert!(
            caps["tools"]
                .as_array()
                .unwrap()
                .iter()
                .all(|t| t["available"] == false)
        );
        request.messages[0].content = json!("--git full-diff");
        let blocked = decide(&request, "test").unwrap();
        assert_eq!(blocked.execution.outcome, Outcome::CapabilityLimitation);
        assert!(blocked.call.is_none());
    }

    #[test]
    fn git_inspection_returns_caller_evidence_and_never_retries_failure() {
        for (prompt, command) in [
            ("what changed?", tools::GitView::Diff.command()),
            ("--git staged", tools::GitView::Staged.command()),
            ("show recent commits", tools::GitView::History.command()),
        ] {
            let mut request = inspection_request(prompt);
            let first = decide(&request, "test").unwrap();
            assert_eq!(first.call.as_ref().unwrap().function.name, "run_command");
            let args: Value =
                serde_json::from_str(&first.call.as_ref().unwrap().function.arguments).unwrap();
            assert_eq!(args["command"], command);
            return_result(&mut request, &first, "exit code: 0\nACTUAL GIT EVIDENCE");
            let completed = decide(&request, "test").unwrap();
            assert_eq!(completed.execution.outcome, Outcome::Completed);
            assert!(completed.content.unwrap().contains("ACTUAL GIT EVIDENCE"));
            request.messages.last_mut().unwrap().content =
                json!("exit code: 128\nfatal: not a git repository");
            let failed = decide(&request, "test").unwrap();
            assert_eq!(failed.execution.outcome, Outcome::ToolFailed);
            assert!(failed.call.is_none());
        }
        assert!(matches!(
            parse_atomic(&Model::embedded().unwrap(), "--git history; rm file"),
            Intent::Clarify(_)
        ));
    }

    #[test]
    fn rust_diagnostics_preserve_nonzero_compiler_output() {
        for (prompt, kind) in [
            ("--diagnostics", tools::DiagnosticsKind::Compiler),
            ("show lint diagnostics", tools::DiagnosticsKind::Lint),
        ] {
            let mut request = inspection_request(prompt);
            let inventory = decide(&request, "test").unwrap();
            return_result(&mut request, &inventory, r#"{"files":["Cargo.toml"]}"#);
            let command = decide(&request, "test").unwrap();
            let args: Value =
                serde_json::from_str(&command.call.as_ref().unwrap().function.arguments).unwrap();
            assert_eq!(args["command"], kind.command());
            return_result(
                &mut request,
                &command,
                "exit code: 101\n{\"reason\":\"compiler-message\",\"message\":\"missing type\"}",
            );
            let failed = decide(&request, "test").unwrap();
            assert_eq!(failed.execution.outcome, Outcome::ToolFailed);
            assert!(failed.content.unwrap().contains("missing type"));
            assert!(failed.call.is_none());
        }
        let mut request = inspection_request("--diagnostics");
        let inventory = decide(&request, "test").unwrap();
        return_result(&mut request, &inventory, r#"{"files":["package.json"]}"#);
        let unsupported = decide(&request, "test").unwrap();
        assert_eq!(unsupported.execution.outcome, Outcome::CapabilityLimitation);
        assert!(unsupported.call.is_none());
    }

    #[test]
    fn task_discovery_reads_scripts_without_executing_them() {
        let mut request = inspection_request("list project tasks");
        request
            .tools
            .retain(|tool| tool["function"]["name"] != "run_command");
        let inventory = decide(&request, "test").unwrap();
        return_result(
            &mut request,
            &inventory,
            r#"{"files":["Cargo.toml","package.json"],"scan_limited":true}"#,
        );
        let read = decide(&request, "test").unwrap();
        assert_eq!(read.call.as_ref().unwrap().function.name, "read_file");
        return_result(
            &mut request,
            &read,
            "Lines 1-1 of 1:\n{\"scripts\":{\"custom\":\"echo untrusted; rm file\"}}",
        );
        let completed = decide(&request, "test").unwrap();
        assert!(completed.call.is_none());
        let tasks: Value = serde_json::from_str(completed.content.as_ref().unwrap()).unwrap();
        assert_eq!(tasks["executed"], false);
        assert_eq!(tasks["inventory_truncated"], true);
        assert_eq!(tasks["tasks"].as_array().unwrap().len(), 5);
        assert!(tools::project_tasks(&[], Some("{invalid"), false).is_err());
        assert!(tools::project_tasks(&[], Some(r#"{"scripts":{"bad":42}}"#), false).is_err());
    }
    #[test]
    fn answer_profiles_consult_with_persona_and_never_dispatch_native_mutations() {
        for name in [
            nioxs_contracts::MODEL_ECHO,
            nioxs_contracts::MODEL_SAGE,
            nioxs_contracts::MODEL_QUILL,
            nioxs_contracts::MODEL_SCOUT,
            nioxs_contracts::MODEL_THINKER,
        ] {
            for prompt in ["Hello", "--bash echo should-not-execute", "--test", "yes"] {
                let request: ChatRequest = serde_json::from_value(json!({"model":name,
                    "messages":[{"role":"system","content":"Persona & Guidelines (always active):\nIdentity: Your name is Morgan."},{"role":"user","content":prompt}],
                    "tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","required":["path"]}}},
                        {"type":"function","function":{"name":"run_command"}},
                        {"type":"function","function":{"name":"patch_file"}},
                        {"type":"function","function":{"name":"web_fetch"}}]})).unwrap();
                let decision = decide(&request, "test").unwrap();
                assert_eq!(decision.execution.model, name);
                assert!(decision.call.is_none());
                let consult = decision.coding.unwrap();
                assert!(
                    consult
                        .messages
                        .iter()
                        .any(|m| m["content"].as_str().is_some_and(|s| s.contains("Morgan")))
                );
                assert!(
                    consult.tools.iter().all(|t| profiles::answer_tool(
                        name,
                        t["function"]["name"].as_str().unwrap()
                    ))
                );
                let safe_tools: Vec<_> = consult
                    .tools
                    .iter()
                    .map(|tool| tool["function"]["name"].as_str().unwrap())
                    .collect();
                if matches!(
                    name,
                    nioxs_contracts::MODEL_SCOUT | nioxs_contracts::MODEL_THINKER
                ) {
                    assert_eq!(safe_tools, ["web_fetch"]);
                } else {
                    assert!(safe_tools.is_empty());
                }
            }
        }
    }

    #[test]
    fn answer_profiles_support_toolless_questions_and_scout_web_only() {
        for name in [
            nioxs_contracts::MODEL_ECHO,
            nioxs_contracts::MODEL_SAGE,
            nioxs_contracts::MODEL_QUILL,
            nioxs_contracts::MODEL_SCOUT,
            nioxs_contracts::MODEL_THINKER,
        ] {
            let request: ChatRequest = serde_json::from_value(
                json!({"model":name,"messages":[{"role":"user","content":"Explain ownership"}]}),
            )
            .unwrap();
            let consult = decide(&request, "test").unwrap().coding.unwrap();
            assert!(consult.files.is_empty());
            assert!(consult.tools.is_empty());
        }
        let request: ChatRequest = serde_json::from_value(json!({"model":"nio-scout","messages":[{"role":"user","content":"Research this URL"}],"tools":[{"type":"function","function":{"name":"web_fetch"}}]})).unwrap();
        assert_eq!(
            decide(&request, "test").unwrap().coding.unwrap().tools[0]["function"]["name"],
            "web_fetch"
        );
    }

    #[test]
    fn flow_no_helm_directive_enters_coding_and_bare_flag_explains_usage() {
        let request: ChatRequest = serde_json::from_value(json!({"model":"nio-flow","messages":[{"role":"user","content":"--no-helm suggest a site"}]})).unwrap();
        let decision = decide(&request, "test").unwrap();
        assert_eq!(decision.coding.unwrap().prompt, "suggest a site");
        let request: ChatRequest = serde_json::from_value(
            json!({"model":"nio-flow","messages":[{"role":"user","content":"--no-helm"}]}),
        )
        .unwrap();
        let decision = decide(&request, "test").unwrap();
        assert_eq!(decision.execution.outcome, Outcome::Clarification);
        assert!(decision.content.unwrap().contains("Usage: --no-helm"));
    }

    #[test]
    fn profiles_default_to_consultation_and_keep_their_identity() {
        for name in [
            nioxs_contracts::MODEL_HELM,
            nioxs_contracts::MODEL_NOVA,
            nioxs_contracts::MODEL_ATOM,
            nioxs_contracts::MODEL_NEON,
        ] {
            let r: ChatRequest = serde_json::from_value(json!({"model":name,"messages":[{"role":"user","content":"consider the authentication implementation"}],"tools":[{"type":"function","function":{"name":"read_file"}},{"type":"function","function":{"name":"find_files"}}]})).unwrap();
            let decision = decide(&r, "test").unwrap();
            assert_eq!(decision.execution.model, name);
            let consult = decision.coding.unwrap();
            assert!(consult.messages.iter().any(|m| {
                m["content"]
                    .as_str()
                    .is_some_and(|s| s.contains(profiles::instructions(name).unwrap()))
            }));
            assert_eq!(
                consult.tools.len(),
                2,
                "only caller-provided tools are offered"
            );
            assert_eq!(consult.guarded, profiles::guarded(name));
        }
        let mut r = request("brainstorm a dashboard", true);
        r.model = nioxs_contracts::MODEL_AUTO.into();
        r.stream = true;
        assert_eq!(
            decide(&r, "test").unwrap().execution.model,
            nioxs_contracts::MODEL_FLOW
        );
    }
    #[test]
    fn nonstreamed_toolless_coding_is_deferred_to_the_provider() {
        let r = request("--code explain ownership", false);
        let decision = decide(&r, "test").unwrap();
        assert!(decision.content.is_none());
        let consult = decision.coding.unwrap();
        assert!(consult.tools.is_empty());
        assert!(consult.messages.iter().any(|m| m["role"] == "user"));
    }

    #[test]
    fn auto_requests_enter_nioflow_orchestration() {
        for prompt in [
            "compare approaches",
            "brainstorm ideas",
            "implement auth",
            "review auth",
        ] {
            let r: ChatRequest = serde_json::from_value(json!({"model":nioxs_contracts::MODEL_AUTO,"messages":[{"role":"user","content":prompt}]})).unwrap();
            assert_eq!(
                decide(&r, "test").unwrap().execution.model,
                nioxs_contracts::MODEL_FLOW
            );
        }
    }
    fn request(prompt: &str, tools: bool) -> ChatRequest {
        serde_json::from_value(json!({"model":MODEL_NANO,"messages":[{"role":"user","content":prompt}],"tools":if tools{vec![json!({"type":"function","function":{"name":"read_file"}})]}else{vec![]}})).unwrap()
    }
    #[test]
    fn streaming_coding_turn_defers_the_upstream_call() {
        let r: ChatRequest = serde_json::from_value(json!({
            "model": MODEL_NANO,
            "messages": [{"role": "user", "content": "--code: analyze my project"}],
            "stream": true,
        }))
        .unwrap();
        let decision = decide(&r, "test").unwrap();
        assert_eq!(decision.intent, "coding");
        let consult = decision
            .coding
            .clone()
            .expect("a streaming coding turn must defer the upstream call to the provider");
        assert_eq!(consult.prompt, "analyze my project");
        assert!(!consult.guarded);
        assert!(
            decision.content.is_none(),
            "the answer streams from the model, so nothing is precomputed"
        );
        assert!(
            decision.stream_segments.is_empty(),
            "synthetic trail segments are omitted in favor of clean end-of-turn footer"
        );
        assert_eq!(decision.execution.outcome, Outcome::Completed);
    }

    /// A tool-capable caller gets the model, not a local file picker: the turn
    /// is handed upstream with the caller's own file tools, and the model's tool
    /// calls are replayed on the next round with their real results.
    #[test]
    fn coding_turn_hands_the_callers_project_to_the_model() {
        let mut r: ChatRequest = serde_json::from_value(json!({
            "model": MODEL_NANO,
            "messages": [{"role": "user", "content": "--code: analyze my project"}],
            "stream": true,
            "tools": [
                {"type":"function","function":{"name":"read_file"}},
                {"type":"function","function":{"name":"find_files"}}
            ],
        }))
        .unwrap();

        let first = decide(&r, "test").unwrap();
        assert!(
            first.call.is_none(),
            "NioNano must not pick files itself; the model decides what to read"
        );
        let consult = first.coding.clone().expect("the model is consulted");
        assert_eq!(consult.prompt, "analyze my project");
        assert_eq!(consult.messages.len(), 2, "system + the request");
        assert_eq!(consult.messages[1]["content"], "analyze my project");
        assert_eq!(
            consult.tools.len(),
            2,
            "only the caller's offered file tools are forwarded"
        );
        assert!(consult.files.is_empty(), "nothing was read yet");
        assert!(
            first.stream_segments.is_empty(),
            "Option C: no synthetic consult header prefix in stream_segments"
        );
        assert_eq!(first.execution.outcome, Outcome::Completed);

        // The model asks the caller to read a file; the next round must carry
        // that traffic back upstream so the model keeps its own context.
        let call = ToolCall {
            id: format!("call_nano_{}_abc123", first.execution.turn_id),
            kind: "function".into(),
            function: FunctionCall {
                name: "read_file".into(),
                arguments: json!({"path": "src/http.rs"}).to_string(),
            },
        };
        r.messages.push(assistant_action(&call));
        r.messages.push(tool_response(
            &call,
            "Lines 1-1 of 1:\nCALLER CONTENT OF src/http.rs",
        ));

        let second = decide(&r, "test").unwrap();
        assert!(
            second.stream_segments.is_empty(),
            "consulting progress line must not repeat on subsequent tool rounds: {:?}",
            second.stream_segments
        );
        let consult = second
            .coding
            .clone()
            .expect("the model still owns the turn");
        assert_eq!(
            second.execution.turn_id, first.execution.turn_id,
            "the continuation stays in the same turn"
        );
        assert_eq!(consult.files, vec!["src/http.rs".to_string()]);
        assert_eq!(consult.messages[2]["role"], "assistant");
        assert_eq!(
            consult.messages[3]["content"],
            "Lines 1-1 of 1:\nCALLER CONTENT OF src/http.rs"
        );
        assert_eq!(
            consult.tools.len(),
            2,
            "rounds remain while the budget lasts"
        );
        assert!(
            !consult.messages.iter().any(|message| {
                message["content"]
                    .as_str()
                    .is_some_and(|text| text.contains("Nio Axis"))
            }),
            "the gateway's own directory must never reach the model: {:?}",
            consult.messages
        );

        // Once the round budget is spent the tools are withdrawn so the model
        // answers with what it actually read.
        for round in 0..companion::CODING_MAX_ROUNDS {
            let call = ToolCall {
                id: format!("call_nano_{}_r{round}", first.execution.turn_id),
                kind: "function".into(),
                function: FunctionCall {
                    name: "read_file".into(),
                    arguments: json!({"path": "src/http.rs"}).to_string(),
                },
            };
            r.messages.push(assistant_action(&call));
            r.messages.push(tool_response(&call, "file contents"));
        }
        let capped = decide(&r, "test").unwrap();
        let consult = capped.coding.clone().expect("the model still answers");
        assert!(
            consult.tools.is_empty(),
            "the tool budget must be exhausted, not unbounded"
        );

        // In a follow-up conversation turn, consult progress trail must NOT be repeated.
        r.messages.push(
            serde_json::from_value(json!({
                "role": "assistant",
                "content": "Analysis complete."
            }))
            .unwrap(),
        );
        r.messages.push(
            serde_json::from_value(json!({
                "role": "user",
                "content": "--code: what next?"
            }))
            .unwrap(),
        );
        let third_turn = decide(&r, "test").unwrap();
        assert!(
            third_turn.stream_segments.is_empty(),
            "consulting progress line must not repeat on subsequent conversation turns: {:?}",
            third_turn.stream_segments
        );
    }

    fn assistant_action(call: &ToolCall) -> nioxs_contracts::Message {
        serde_json::from_value(json!({"role":"assistant","content":null,"tool_calls":[call]}))
            .unwrap()
    }

    fn tool_response(call: &ToolCall, content: &str) -> nioxs_contracts::Message {
        serde_json::from_value(json!({"role":"tool","tool_call_id":call.id,"content":content}))
            .unwrap()
    }

    #[test]
    fn fresh_turns_are_distinct_and_continuations_keep_the_returned_turn() {
        let mut r = request("now read readme.md", true);
        let first = decide(&r, "test").unwrap();
        let second = decide(&r, "test").unwrap();
        assert_ne!(first.execution.turn_id, second.execution.turn_id);
        let call = first.call.unwrap();
        r.messages.push(
            serde_json::from_value(json!({"role":"assistant","content":null,"tool_calls":[call]}))
                .unwrap(),
        );
        r.messages.push(
            serde_json::from_value(
                json!({"role":"tool","tool_call_id":call.id,"content":"read result"}),
            )
            .unwrap(),
        );
        let continuation = decide(&r, "test").unwrap();
        assert_eq!(continuation.execution.turn_id, first.execution.turn_id);
        assert_eq!(continuation.execution.outcome, Outcome::Completed);
    }
    #[test]
    fn bare_consent_reply_without_a_pending_prompt_installs_nothing() {
        // Regression: a plain "yes" used to install svg2favicon no matter what was asked before.
        let mut r = request("--code: analyze my project", true);
        r.messages.push(
            serde_json::from_value(json!({
                "role": "assistant",
                "content": "I'll analyze your project by first exploring its structure."
            }))
            .unwrap(),
        );
        r.messages
            .push(serde_json::from_value(json!({"role": "user", "content": "yes"})).unwrap());
        let decision = decide(&r, "test").unwrap();
        assert!(
            decision.call.is_none(),
            "a bare yes must not dispatch a tool"
        );
        assert_eq!(decision.execution.outcome, Outcome::Clarification);
        assert!(
            decision
                .content
                .unwrap_or_default()
                .contains("no pending confirmation")
        );
    }

    #[test]
    fn consent_reply_approves_the_plugin_the_previous_turn_requested() {
        let pending = companion::CompanionResult::ToolNeedsConsent {
            plugin_name: "image_scaler".to_string(),
            task: "resize assets/banner.png to 800x400".to_string(),
        }
        .to_markdown();
        let r: ChatRequest = serde_json::from_value(json!({
            "model": MODEL_NANO,
            "messages": [
                {"role": "user", "content": "--tools: resize assets/banner.png to 800x400"},
                {"role": "assistant", "content": pending},
                {"role": "user", "content": "yes"}
            ],
            "tools": []
        }))
        .unwrap();
        let decision = decide(&r, "test").unwrap();
        let content = decision.content.unwrap_or_default();
        assert!(
            content.contains("image_scaler"),
            "the approved plugin must be the pending one, got: {content}"
        );
        assert!(
            !content.contains("svg2favicon"),
            "a consent reply must never fall back to a default plugin"
        );
        let _ = std::fs::remove_file(".nioxs/plugins/image_scaler.py");
        let _ = std::fs::remove_dir(".nioxs/plugins");
        let _ = std::fs::remove_dir(".nioxs");
    }

    #[test]
    fn natural_read_phrasing_and_correlated_results() {
        for prompt in [
            "now read readme.md",
            "please open README.md",
            "show me README.md",
        ] {
            let mut r = request(prompt, true);
            let call = decide(&r, "test").unwrap().call.unwrap();
            assert_eq!(
                serde_json::from_str::<Value>(&call.function.arguments).unwrap()["path"],
                "README.md"
            );
            r.messages.push(
                serde_json::from_value(
                    json!({"role":"assistant","content":null,"tool_calls":[call]}),
                )
                .unwrap(),
            );
            r.messages.push(
                serde_json::from_value(
                    json!({"role":"tool","tool_call_id":call.id,"content":"# Actual file"}),
                )
                .unwrap(),
            );
            let done = decide(&r, "test").unwrap();
            assert_eq!(done.execution.outcome, Outcome::Completed);
            assert!(done.content.unwrap().contains("Actual file"));
            r.messages.remove(1);
            assert!(decide(&r, "test").is_err());
        }
    }
    #[test]
    fn unsupported_disabled_and_missing_inputs_never_dispatch() {
        for r in [
            request("write tests for README.md", true),
            request("read /etc/shadow", true),
            request("read README.md", false),
            request("run tests", true),
            request("read it", true),
        ] {
            assert!(decide(&r, "test").unwrap().call.is_none());
        }
        assert_eq!(
            decide(&request("read it", true), "test")
                .unwrap()
                .execution
                .outcome,
            Outcome::Clarification
        );
        assert!(!safe_path("../outside.txt", false));
        assert!(!safe_path(".env", false));
    }
    #[test]
    fn web_fetch_and_research_parsing() {
        let model = embedded();
        assert_eq!(
            parse(model, "read https://example.com"),
            Intent::WebFetch("https://example.com".into())
        );
        assert_eq!(
            parse(model, "research Rust programming language"),
            Intent::Research("Rust programming language".into())
        );
        assert_eq!(
            parse(model, "search web for tokio broadcast"),
            Intent::Research("tokio broadcast".into())
        );
        assert_eq!(
            parse(model, "wiki Rust language"),
            Intent::Wiki("Rust language".into())
        );
        assert_eq!(
            parse(model, "github tokio-rs/tokio"),
            Intent::Github("tokio-rs/tokio".into())
        );
        assert_eq!(parse(model, "crate tokio"), Intent::Package("tokio".into()));
        assert_eq!(
            parse(model, "npm express"),
            Intent::Package("express".into())
        );
        assert_eq!(
            parse(model, "scaffold gitignore"),
            Intent::Scaffold("gitignore".into())
        );
        assert_eq!(parse(model, "what is Jev?"), Intent::Research("Jev".into()));
        assert_eq!(
            parse(model, "hey what is zig?"),
            Intent::Research("zig".into())
        );
        assert_eq!(parse(model, "why zig?"), Intent::Research("zig".into()));
        assert_eq!(parse(model, "hey"), Intent::Greeting);
        assert_eq!(parse(model, "help"), Intent::Help);
        assert_eq!(
            parse(model, "search my codebase"),
            Intent::Clarify(
                "Which literal text or symbol should I search for in the codebase?".into()
            )
        );
    }
    #[test]
    fn test_task_decomposition_planning() {
        let model = Model::embedded().unwrap();
        let plan = parse(&model, "inspect project and then run tests");
        assert_eq!(
            plan,
            Intent::Plan(vec![Intent::Inspect, Intent::Checks("test".into())])
        );

        let plan2 = parse(&model, "scaffold gitignore then crate tokio");
        assert_eq!(
            plan2,
            Intent::Plan(vec![
                Intent::Scaffold("gitignore".into()),
                Intent::Package("tokio".into())
            ])
        );

        let unsupported = parse(&model, "run tests then deploy our workspace");
        assert_eq!(unsupported, Intent::Unsupported);

        let unsupported2 = parse(&model, "read and upload our workspace");
        assert_eq!(unsupported2, Intent::Unsupported);
    }

    #[test]
    fn test_unbound_code_modification_hands_off_to_code() {
        let model = Model::embedded().unwrap();
        for prompt in [
            "refactor the parser",
            "debug the crash",
            "rename foo to bar",
        ] {
            match parse(&model, prompt) {
                Intent::Clarify(msg) => assert!(
                    msg.contains("--code"),
                    "handoff for {prompt:?} must name --code, got {msg:?}"
                ),
                other => panic!("expected a --code handoff for {prompt:?}, got {other:?}"),
            }
        }
        // Destructive prompts still dead-end as unsupported rather than handing off.
        assert_eq!(
            parse(&model, "read and upload our workspace"),
            Intent::Unsupported
        );
    }

    #[test]
    fn test_human_language_analysis_normalization() {
        let model = Model::embedded().unwrap();
        assert_eq!(
            parse(&model, "analyst code"),
            Intent::AnalyzeDir(".".into())
        );
        assert_eq!(
            parse(&model, "analyse codebase"),
            Intent::AnalyzeDir(".".into())
        );
        assert_eq!(
            parse(&model, "analyze codebase"),
            Intent::AnalyzeDir(".".into())
        );
        assert_eq!(
            parse(&model, "can you give me an analysis of the codebase?"),
            Intent::AnalyzeDir(".".into())
        );
        assert_eq!(
            parse(&model, "please review Cargo.toml"),
            Intent::AnalyzeFile("Cargo.toml".into())
        );
        assert_eq!(
            parse(&model, "how is the code quality?"),
            Intent::AnalyzeDir(".".into())
        );
    }
    #[test]
    fn test_phase3_to_phase6_coding_engine() {
        let model = Model::embedded().unwrap();

        // Phase 3: Semantic Symbol Navigation
        assert_eq!(
            parse(&model, "where is parse_atomic defined?"),
            Intent::SymbolDef("parse_atomic".into())
        );
        assert_eq!(
            parse(&model, "who calls apply_patch_to_string?"),
            Intent::SymbolCallers("apply_patch_to_string".into())
        );

        // Phase 4: Deterministic Code Refactoring
        assert_eq!(
            parse(&model, "replace 'old_fn' with 'new_fn' in src/lib.rs"),
            Intent::RefactorPatch {
                file: "src/lib.rs".into(),
                old: "old_fn".into(),
                new: "new_fn".into(),
            }
        );

        // Phase 5: Autonomous Edit-and-Test Loop (TDD)
        assert_eq!(
            parse(
                &model,
                "replace 'old_fn' with 'new_fn' in src/lib.rs and run tests"
            ),
            Intent::TddRefactor {
                file: "src/lib.rs".into(),
                old: "old_fn".into(),
                new: "new_fn".into(),
            }
        );

        // Phase 6: Dynamic Test Generator & Git Diff Reviewer
        assert_eq!(
            parse(&model, "write test for calculate_tax in src/math.rs"),
            Intent::ScaffoldTest {
                file: "src/math.rs".into(),
                function: "calculate_tax".into(),
            }
        );
        assert_eq!(parse(&model, "review git diff"), Intent::ReviewDiff);
        assert_eq!(
            parse(&model, "suggest commit message"),
            Intent::SuggestCommit
        );
    }

    #[test]
    fn test_suggestions_and_task_planning() {
        let model = Model::embedded().unwrap();

        assert_eq!(parse(&model, "any suggestion?"), Intent::ProjectSuggestions);
        assert_eq!(
            parse(&model, "what should I work on?"),
            Intent::ProjectSuggestions
        );
        assert_eq!(
            parse(&model, "suggest next steps"),
            Intent::ProjectSuggestions
        );

        assert_eq!(
            parse(
                &model,
                "plan this task: implement token bucket rate limiter"
            ),
            Intent::TaskPlan("implement token bucket rate limiter".into())
        );
        assert_eq!(
            parse(&model, "plan refactor database queries"),
            Intent::TaskPlan("refactor database queries".into())
        );
    }

    #[test]
    fn test_natural_read_phrasings_and_nlp_synthesis() {
        let model = Model::embedded().unwrap();

        // 1. Natural read prompt variants
        assert_eq!(
            parse(&model, "what is in Cargo.toml?"),
            Intent::Read("Cargo.toml".into())
        );
        assert_eq!(
            parse(&model, "what's inside src/lib.rs"),
            Intent::Read("src/lib.rs".into())
        );
        assert_eq!(
            parse(&model, "can you read src/lib.rs"),
            Intent::Read("src/lib.rs".into())
        );
        assert_eq!(
            parse(&model, "view src/lib.rs"),
            Intent::Read("src/lib.rs".into())
        );
        assert_eq!(
            parse(&model, "cat Cargo.toml"),
            Intent::Read("Cargo.toml".into())
        );
        assert_eq!(
            parse(&model, "open Cargo.toml"),
            Intent::Read("Cargo.toml".into())
        );
        assert_eq!(
            parse(&model, "let's look at Cargo.toml"),
            Intent::Read("Cargo.toml".into())
        );
        assert_eq!(
            parse(&model, "show me what's inside Cargo.toml"),
            Intent::Read("Cargo.toml".into())
        );
        assert_eq!(
            parse(&model, "show contents of README.md"),
            Intent::Read("README.md".into())
        );
        assert_eq!(
            parse(&model, "display src/lib.rs"),
            Intent::Read("src/lib.rs".into())
        );
        assert_eq!(
            parse(&model, "what does Cargo.toml contain?"),
            Intent::Read("Cargo.toml".into())
        );
        assert_eq!(
            parse(&model, "Cargo.toml"),
            Intent::Read("Cargo.toml".into())
        );

        // 2. Synthesize read NLP response
        let toml_sample = "[package]\nname = \"test-crate\"\nversion = \"0.1.0\"\n[dependencies]\ntokio = \"1.0\"\n";
        let toml_res = synthesize::synthesize_read_response("Cargo.toml", toml_sample);
        assert!(toml_res.contains("### 📄 `Cargo.toml`"));
        assert!(toml_res.contains("test-crate"));
        assert!(toml_res.contains("```toml"));
        assert!(toml_res.contains("💡 **Suggested Actions**"));

        let rs_sample = "pub fn add(a: i32, b: i32) -> i32 { a + b }\npub struct Calc;\n";
        let rs_res = synthesize::synthesize_read_response("src/calc.rs", rs_sample);
        assert!(rs_res.contains("### 📄 `src/calc.rs`"));
        assert!(rs_res.contains("`add()`"));
        assert!(rs_res.contains("`Calc`"));
        assert!(rs_res.contains("```rust"));
    }

    #[test]
    fn test_directive_parsing_with_and_without_colons() {
        let model = Model::embedded().unwrap();

        // --code directives
        assert_eq!(
            parse(&model, "--code write a function"),
            Intent::Coding {
                prompt: "write a function".into(),
                guarded: false,
            }
        );
        assert_eq!(
            parse(&model, "--code: write a function"),
            Intent::Coding {
                prompt: "write a function".into(),
                guarded: false,
            }
        );
        assert_eq!(
            parse(&model, "--code guard write a function"),
            Intent::Coding {
                prompt: "write a function".into(),
                guarded: true,
            }
        );
        assert_eq!(
            parse(&model, "--code:guard write a function"),
            Intent::Coding {
                prompt: "write a function".into(),
                guarded: true,
            }
        );
        assert_eq!(
            parse(&model, "--code! write a function"),
            Intent::Coding {
                prompt: "write a function".into(),
                guarded: true,
            }
        );
        // The retired `--coding:` spelling must no longer route as a coding directive.
        assert_ne!(
            parse(&model, "--coding: write a function"),
            Intent::Coding {
                prompt: "write a function".into(),
                guarded: false,
            }
        );

        // --bash directives
        assert_eq!(
            parse(&model, "--bash cargo check"),
            Intent::Bash("cargo check".into())
        );
        assert_eq!(
            parse(&model, "--bash: cargo check"),
            Intent::Bash("cargo check".into())
        );

        // --tools directives
        assert_eq!(parse(&model, "--tools list"), Intent::Tool("list".into()));
        assert_eq!(parse(&model, "--tools: list"), Intent::Tool("list".into()));

        // --test directives
        assert_eq!(
            parse(&model, "--test unit"),
            Intent::TestRunner("unit".into())
        );
        assert_eq!(
            parse(&model, "--test: unit"),
            Intent::TestRunner("unit".into())
        );

        // --git directives carry their subcommand; the bare form means status+diff.
        assert_eq!(parse(&model, "--git"), Intent::GitSummary(String::new()));
        assert_eq!(
            parse(&model, "--git status"),
            Intent::GitSummary("status".into())
        );
        assert_eq!(
            parse(&model, "--git:status"),
            Intent::GitSummary("status".into())
        );
        assert_eq!(
            parse(&model, "--git diff"),
            Intent::GitSummary("diff".into())
        );
        assert_eq!(
            parse(&model, "--git:diff"),
            Intent::GitSummary("diff".into())
        );
        assert!(matches!(parse(&model, "--git commit"), Intent::Clarify(_)));

        // --config directives
        assert_eq!(
            parse(&model, "--config code ON"),
            Intent::Config("code ON".into())
        );
        assert_eq!(
            parse(&model, "--config: code ON"),
            Intent::Config("code ON".into())
        );
        assert_eq!(
            parse(&model, "--config list"),
            Intent::Config("list".into())
        );
        assert_eq!(
            parse(&model, "--config: list"),
            Intent::Config("list".into())
        );
    }

    #[test]
    fn test_config_code_on_defaults_prompt_to_code_directive() {
        let model = Model::embedded().expect("embedded model");

        companion::with_test_config(
            companion::XsConfig {
                code: true,
                ..Default::default()
            },
            || {
                assert!(companion::is_config_code_on());

                // A plain request without --code prefix defaults to Intent::Coding!
                let intent = parse(&model, "add retry logic to the HTTP client");
                match intent {
                    Intent::Coding { prompt, guarded } => {
                        assert_eq!(prompt, "add retry logic to the HTTP client");
                        assert!(!guarded);
                    }
                    other => panic!("Expected Intent::Coding when code is ON, got {:?}", other),
                }

                // Directives like --bash still route to their own intent
                assert_eq!(
                    parse(&model, "--bash cargo check"),
                    Intent::Bash("cargo check".into())
                );
                assert_eq!(parse(&model, "--tools list"), Intent::Tool("list".into()));
            },
        );
        assert!(!companion::is_config_code_on());
    }

    #[test]
    fn explicit_find_files_directive_works_with_code_mode_enabled() {
        let model = Model::embedded().expect("embedded model");
        companion::with_test_config(
            companion::XsConfig {
                code: true,
                ..Default::default()
            },
            || {
                assert_eq!(
                    parse(&model, "--find-files **/*"),
                    Intent::Files("**/*".into())
                )
            },
        );
    }
}
