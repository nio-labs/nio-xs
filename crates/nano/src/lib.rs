//! # NioNano — MicroLM (μLM)
//!
//! NioNano combines **Symbolic AI** (deterministic code search, unified patch engine,
//! AST exploration, path sanitization, sandboxing) with a lightweight **Language Model (LM)**
//! (sub-megabyte embedded weights, multinomial n-gram distribution, <1ms CPU inference)
//! to provide sovereign pair-programming workflows without external API dependencies or GPU requirements.

pub mod analysis;
pub mod model;
pub mod patch;
pub use model::Model;
use nioxs_contracts::{
    ChatRequest, Execution, FunctionCall, Handoff, MODEL, Outcome, PROVIDER, SCHEMA_VERSION,
    ToolCall, digest,
};
pub use patch::{apply_patch_to_string, scaffold_template};
use serde_json::{Value, json};
use std::sync::OnceLock;

#[derive(Clone, Debug, PartialEq)]
pub enum Intent {
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
    Unsupported,
}
impl Intent {
    pub fn label(&self) -> &str {
        match self {
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
            Self::Help => "help",
            Self::Clarify(_) => "clarify",
            Self::Plan(_) => "plan",
            Self::AnalyzeFile(_) => "analyze_file",
            Self::AnalyzeDir(_) => "analyze_dir",
            Self::Unsupported => "unsupported",
        }
    }
}
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
    let lower = prompt.to_ascii_lowercase();
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
    if prompt.len() > 1024 || !prompt.is_ascii() {
        return Intent::Unsupported;
    }
    if let Some(parts) = split_compound_prompt(prompt) {
        let mut steps = Vec::new();
        for part in parts {
            if model::features(part).iter().any(|f| f == "<unsupported-risk>") {
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

pub fn parse_atomic(model: &Model, prompt: &str) -> Intent {
    if prompt.len() > 1024 || !prompt.is_ascii() {
        return Intent::Unsupported;
    }
    for word in prompt.split_whitespace() {
        let clean = word.trim_matches(|c: char| "`\"',;:!?<>".contains(c));
        if is_web_url(clean) {
            return Intent::WebFetch(clean.to_string());
        }
    }
    let lower = prompt.to_ascii_lowercase();
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
    ]
    .iter()
    .find_map(|p| {
        if lower.starts_with(p) && !lower.starts_with("where is symbol ") {
            Some(prompt[p.len()..].trim().trim_end_matches('?').trim())
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
            prompt[10..].trim()
        } else {
            prompt[5..].trim()
        };
        if !query.is_empty() {
            return Intent::Wiki(query.to_string());
        }
    }
    if lower.starts_with("github ") || lower.starts_with("git ") {
        let target = if lower.starts_with("github ") {
            prompt[7..].trim()
        } else {
            prompt[4..].trim()
        };
        if !target.is_empty() && target.contains('/') {
            return Intent::Github(target.to_string());
        }
    }
    if lower.starts_with("package ") || lower.starts_with("crate ") || lower.starts_with("npm ") {
        let pkg = if lower.starts_with("package ") {
            prompt[8..].trim()
        } else if lower.starts_with("crate ") {
            prompt[6..].trim()
        } else {
            prompt[4..].trim()
        };
        if !pkg.is_empty() && !pkg.contains(' ') {
            return Intent::Package(pkg.to_string());
        }
    }
    if lower.starts_with("scaffold ") || lower.starts_with("template ") {
        let tpl = prompt[9..].trim();
        if !tpl.is_empty() {
            return Intent::Scaffold(tpl.to_string());
        }
    }
    let trimmed = lower.trim().trim_end_matches(|c: char| "!?. \t\r\n".contains(c));
    if ["hey", "hello", "hi", "good morning", "good evening", "howdy", "sup", "yo"]
        .contains(&trimmed)
    {
        return Intent::Greeting;
    }
    if ["help", "what can you do", "commands", "usage", "capabilities"].contains(&trimmed) {
        return Intent::Help;
    }
    if ["search my codebase", "search the codebase", "search codebase", "search code"].contains(&trimmed) {
        return Intent::Clarify("Which literal text or symbol should I search for in the codebase?".into());
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
        if lower.starts_with(prefix) {
            let target = prompt[prefix.len()..].trim().trim_matches(|c: char| "`\"',:!?".contains(c));
            let lower_t = target.to_ascii_lowercase();
            if ["codebase", "the codebase", "my codebase", "project", "the project", "my project", "code", "the code", "."].contains(&lower_t.as_str()) {
                return Intent::AnalyzeDir(".".into());
            }
            if safe_path(target, true) && !target.is_empty() {
                if target.contains('.') && !target.ends_with('/') {
                    return Intent::AnalyzeFile(target.to_string());
                } else {
                    return Intent::AnalyzeDir(target.to_string());
                }
            }
        }
    }
    if ["analyze codebase", "review codebase", "audit codebase", "analyze project", "review project", "audit project", "analyze code", "code review", "code audit"].contains(&trimmed) {
        return Intent::AnalyzeDir(".".into());
    }
    let candidates = paths(prompt);
    match model.predict(prompt).label.as_str() {
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
                        && !["it", "that", "this", "the thing", "my codebase", "the codebase", "codebase"].contains(&q.as_str()) =>
                {
                    Intent::Search(q)
                }
                _ => Intent::Clarify("Which literal text or symbol should I search for in the codebase?".into()),
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
    }
}
struct Workflow<'a> {
    request: &'a ChatRequest,
    start: usize,
    key: String,
    execution: Execution,
    request_digest: String,
    intent: String,
    observed: Vec<ObservedAction>,
}
impl Workflow<'_> {
    fn decision(&self, content: Option<String>, call: Option<ToolCall>) -> Decision {
        Decision {
            content,
            call,
            execution: self.execution.clone(),
            request_digest: self.request_digest.clone(),
            result_digest: self.observed.last().map(|o| o.result_digest.clone()),
            intent: self.intent.clone(),
            step: self.observed.len(),
            observed: self.observed.clone(),
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
        self.end(
            reason.to_string(),
            Outcome::CapabilityLimitation,
        )
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
        if previous == "Which project-relative file should I read?" && paths(&prompt).len() == 1 {
            prompt = format!("read {prompt}");
        } else if previous == "Which literal text or symbol should I search for?" {
            prompt = format!("search code for {prompt}");
        } else if previous == "Which configured check should I run: test, build, or lint?" {
            prompt = format!("run {prompt}");
        }
    }
    let intent = parse(model, &prompt);
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
            model: MODEL.into(),
            outcome: Outcome::CapabilityLimitation,
            handoff: None,
        },
        request_digest,
        intent: intent.label().into(),
        observed: vec![],
    };
    match intent {
        Intent::Unsupported => Ok(w.limitation(
            "I'm specialized in project code inspection, tests, builds, scaffolds, and web research. I couldn't identify a programming action for that request. Try asking 'help' to view available capabilities, or specify a file or symbol.",
            false,
        )),
        Intent::Clarify(question) => Ok(w.end(question, Outcome::Clarification)),
        Intent::Read(path) => {
            let text = tool!(w, "read_file", json!({"path": path, "line_count": 80}));
            Ok(w.end(
                format!("Here is `{path}`:\n\n```\n{}\n```", excerpt(&text, 8000)),
                Outcome::Completed,
            ))
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
            let manifests: Vec<&String> = paths.iter().filter(|p| ["Cargo.toml", "package.json", "pyproject.toml", "go.mod"].contains(&p.as_str())).take(3).collect();
            let mut report = format!("Project Workspace ({} files discovered):\n{}\n", paths.len(), paths.iter().map(|p| format!("• `{p}`")).collect::<Vec<_>>().join("\n"));
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
                report.push_str("No standard root manifest found.\n");
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
            let report = run_async(nioxs_browser::research(&query, 3))
                .map_err(|e| format!("web research error: {e}"))?;

            if report.passages.is_empty() {
                Ok(w.end(
                    format!(
                        "I searched for '{query}' across web and documentation sources, but didn't find high-confidence cited passages. Try checking Wikipedia with `wiki {query}`."
                    ),
                    Outcome::Completed,
                ))
            } else {
                let top = &report.passages[0];
                let mut answer = format!("Based on findings for **{query}**:\n\n{}\n\n", top.excerpt.trim());
                if report.passages.len() > 1 {
                    answer.push_str("Key details from additional sources:\n");
                    for p in &report.passages[1..] {
                        let snippet = p.excerpt.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
                        if !snippet.is_empty() {
                            answer.push_str(&format!("• **{}**: {}\n", p.title, snippet));
                        }
                    }
                    answer.push('\n');
                }
                answer.push_str("Sources:\n");
                for (idx, passage) in report.passages.iter().enumerate() {
                    answer.push_str(&format!("{}. [{}]({})\n", idx + 1, passage.title, passage.url));
                }
                Ok(w.end(excerpt(&answer, 8000), Outcome::Completed))
            }
        }
        Intent::Wiki(query) => {
            let results = run_async(nioxs_browser::search_wikipedia(&query, 3))
                .map_err(|e| format!("wikipedia search error: {e}"))?;
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
                let mut out = format!(
                    "Wikipedia article: {} ({})\n\n{}",
                    top.title,
                    top.url,
                    excerpt(&article_text, 5000)
                );
                if results.len() > 1 {
                    out.push_str("\n\nRelated articles:\n");
                    for r in &results[1..] {
                        out.push_str(&format!("- {} ({})\n", r.title, r.url));
                    }
                }
                Ok(w.end(out, Outcome::Completed))
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
            Ok(w.end(
                "Hello! I'm NioNano, your sovereign pair programmer and MicroLM (μLM). I can inspect your project, search code, run test/build checks, scaffold templates, or research the web. What would you like to work on?".into(),
                Outcome::Completed,
            ))
        }
        Intent::Help => {
            Ok(w.end(
                "NioNano MicroLM (μLM) Capabilities:\n\n• Code Inspection: `read <file>`, `find <glob>`, `inspect project`\n• Code Analysis: `analyze <file>`, `review <file>`, `analyze codebase`\n• Code Search: `search code for <symbol>`\n• Checks: `run test check`, `run build check`, `run lint check`\n• Web Research: `wiki <topic>`, `github <repo>`, `what is <term>?`\n• Package Registry: `crate <name>`, `npm <name>`\n• Templates: `scaffold gitignore`, `scaffold github-ci`, `scaffold license-mit`".into(),
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
        Intent::Plan(steps) => {
            let mut summary = format!("Decomposed and executing {} task steps:\n\n", steps.len());
            for (idx, step) in steps.iter().enumerate() {
                let step_num = idx + 1;
                match step {
                    Intent::Read(path) => {
                        let text = tool!(w, "read_file", json!({"path": path, "line_count": 80}));
                        summary.push_str(&format!("### Step {step_num}: Read {path}\n{}\n\n", excerpt(&text, 2000)));
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
    fn request(prompt: &str, tools: bool) -> ChatRequest {
        serde_json::from_value(json!({"model":MODEL,"messages":[{"role":"user","content":prompt}],"tools":if tools{vec![json!({"type":"function","function":{"name":"read_file"}})]}else{vec![]}})).unwrap()
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
        assert_eq!(
            parse(model, "what is Jev?"),
            Intent::Research("Jev".into())
        );
        assert_eq!(parse(model, "hey"), Intent::Greeting);
        assert_eq!(parse(model, "help"), Intent::Help);
        assert_eq!(
            parse(model, "search my codebase"),
            Intent::Clarify("Which literal text or symbol should I search for in the codebase?".into())
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
}
