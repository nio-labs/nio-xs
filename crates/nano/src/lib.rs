pub mod model;
pub use model::Model;
use nioxs_contracts::{
    ChatRequest, Execution, FunctionCall, Handoff, MODEL, Outcome, PROVIDER, SCHEMA_VERSION,
    ToolCall, digest,
};
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
    Clarify(String),
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
            Self::Clarify(_) => "clarify",
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
pub fn parse(model: &Model, prompt: &str) -> Intent {
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
    let query_prefix = [
        "search the web for ",
        "search web for ",
        "web search for ",
        "web search ",
        "research ",
    ]
    .iter()
    .find_map(|p| {
        if lower.starts_with(p) {
            Some(prompt[p.len()..].trim())
        } else {
            None
        }
    });
    if let Some(query) = query_prefix
        && !query.is_empty()
    {
        return Intent::Research(query.to_string());
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
                        && !["it", "that", "this", "the thing"].contains(&q.as_str()) =>
                {
                    Intent::Search(q)
                }
                _ => Intent::Clarify("Which literal text or symbol should I search for?".into()),
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
            format!("NioNano cannot complete this workflow: {reason}."),
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
    match intent{
 Intent::Unsupported=>Ok(w.limitation("unsupported request or low classifier confidence; automatic LLM continuation is planned for Phase 4",true)),
 Intent::Clarify(question)=>Ok(w.end(question,Outcome::Clarification)),
 Intent::Read(path)=>{let text=tool!(w,"read_file",json!({"path":path,"line_count":80}));Ok(w.end(format!("File excerpt from {path}:\n\n{}",excerpt(&text,8000)),Outcome::Completed))},
 Intent::Files(glob)=>{let text=tool!(w,"find_files",json!({"path":".","glob":glob,"limit":50}));let(paths,limited)=files(&text)?;Ok(w.end(format!("Files matching {glob}:\n{}\n{}",paths.iter().map(|p|format!("- {p}")).collect::<Vec<_>>().join("\n"),if limited{"Inventory incomplete; narrow the glob."}else if paths.is_empty(){"No matches returned."}else{""}),Outcome::Completed))},
 Intent::Search(query)=>{
 let text=tool!(w,"search_code",json!({"query":query,"path":".","regex":false,"context_lines":1,"limit":30}));let value:Value=serde_json::from_str(&text).map_err(|_|"invalid search result")?;
 let mut ranked:Vec<(usize,String)>=value["matches"].as_array().ok_or("missing matches")?.iter().map(|entry|{
 let path=entry["path"].as_str().filter(|p|safe_path(p,false)).ok_or("invalid search path")?;let line=entry["line"].as_u64().filter(|n|*n>0).ok_or("invalid search line")?;let snippet=entry["excerpt"].as_str().ok_or("missing excerpt")?;
 Ok((snippet.to_ascii_lowercase().matches(&query.to_ascii_lowercase()).count(),format!("{path}:{line}\n{snippet}"))) }).collect::<Result<Vec<_>,String>>()?;
 ranked.sort_by(|a,b|b.0.cmp(&a.0).then(a.1.cmp(&b.1)));let limited=value["scan_limited"]==true||!value["next_offset"].is_null();
 Ok(w.end(format!("Literal code matches for {query}:\n\n{}\n{}",ranked.iter().map(|(_,row)|row.as_str()).collect::<Vec<_>>().join("\n\n"),if limited{"Search coverage incomplete; narrow the query."}else if ranked.is_empty(){"No matches returned."}else{""}),Outcome::Completed))},
 Intent::Inspect=>{
 let text=tool!(w,"find_files",json!({"path":".","limit":50}));let(paths,limited)=files(&text)?;
 let manifests:Vec<&String>=paths.iter().filter(|p|["Cargo.toml","package.json","pyproject.toml","go.mod"].contains(&p.as_str())).take(3).collect();
 let mut report=format!("Project files (up to 50):\n{}\n",paths.iter().map(|p|format!("- {p}")).collect::<Vec<_>>().join("\n"));
 for path in &manifests{let content=tool!(w,"read_file",json!({"path":path,"line_count":200}));report.push_str(&format!("\nManifest evidence from {path}:\n{}\n",excerpt(&content,2200)));
 if path.as_str()=="package.json"&&let Ok(package)=serde_json::from_str::<Value>(payload(&content)){report.push_str(&format!("Package name: {}; version: {}.\n",package["name"].as_str().unwrap_or("not declared"),package["version"].as_str().unwrap_or("not declared")));if let Some(deps)=package["dependencies"].as_object(){report.push_str(&format!("Declared dependencies: {}.\n",deps.keys().take(20).cloned().collect::<Vec<_>>().join(", ")));}}}
 if manifests.is_empty(){report.push_str("No supported root manifest returned; project type unknown.\n");}
if limited{report.push_str("File inventory incomplete.\n");}Ok(w.end(excerpt(&report,8000),Outcome::Completed))},
 Intent::Checks(kind)=>{
 if !request.permits_tool("run_command"){return Ok(w.limitation("commands require an offered run_command tool and Nio's Build-mode approval",false));}
 let text=tool!(w,"find_files",json!({"path":".","limit":50}));let(paths,limited)=files(&text)?;
 let(command,source):(String,String)=if paths.iter().any(|p|p==".nioxs.json"){
 let text=tool!(w,"read_file",json!({"path":".nioxs.json","line_count":200}));let config:Value=serde_json::from_str(payload(&text)).map_err(|_|"invalid or truncated .nioxs.json")?;
 if config["schema_version"]!=1{return Err("unsupported .nioxs.json schema".into());}
 let Some(command)=config["commands"][&kind].as_str()else{return Ok(w.end(format!("No {kind} command configured in .nioxs.json."),Outcome::Clarification));};
 let allowed=["cargo test --locked","cargo build --locked","cargo clippy --locked -- -D warnings","npm test","npm run test","npm run build","npm run lint","python3 -m unittest","python3 -m pytest","go test ./...","go build ./...","go vet ./..."];
 if !allowed.contains(&command){return Ok(w.limitation("configured command outside Nano's supported named-command set",false));}(command.into(),".nioxs.json".into())
 }else if paths.iter().any(|p|p=="Cargo.toml"){
 (match kind.as_str(){"test"=>"cargo test --locked","build"=>"cargo build --locked",_=>"cargo clippy --locked -- -D warnings"}.into(),"Cargo.toml (fixed Rust mapping)".into())
 }else if paths.iter().any(|p|p=="package.json"){
 let text=tool!(w,"read_file",json!({"path":"package.json","line_count":200}));let package:Value=serde_json::from_str(payload(&text)).map_err(|_|"invalid or truncated package.json")?;
 if package["scripts"][&kind].as_str().is_none(){return Ok(w.end(format!("package.json has no {kind} script."),Outcome::Clarification));}(format!("npm run {kind}"),"package.json scripts".into())
 }else{return Ok(w.end(format!("No supported {kind} command found{}; configure .nioxs.json.",if limited{" in incomplete inventory"}else{""}),Outcome::Clarification));};
 let result=tool!(w,"run_command",json!({"command":command}));Ok(w.end(format!("{kind} check completed with exit code 0.\nCommand: {command}\nConfiguration: {source}\n\n{}",excerpt(&result,6500)),Outcome::Completed))}
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
                        "No cited passages found for '{query}'. Visited {} candidate sources.",
                        report.sources_visited.len()
                    ),
                    Outcome::Completed,
                ))
            } else {
                let mut report_text = format!("Research report for '{query}':\n\n");
                for (idx, passage) in report.passages.iter().enumerate() {
                    report_text.push_str(&format!(
                        "### Source [{}] ({})\nTitle: {}\n\n{}\n\n",
                        idx + 1,
                        passage.url,
                        passage.title,
                        passage.excerpt
                    ));
                }
                Ok(w.end(excerpt(&report_text, 8000), Outcome::Completed))
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
    }
}
