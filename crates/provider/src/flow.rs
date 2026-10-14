//! Bounded NioFlow orchestration. Specialists share the configured upstream;
//! each stage is a separate call with its own role and carried-forward evidence.
use super::*;
use nioxs_contracts::{
    AtomContract, HelmGateVerdict, HelmVerdictContract, MODEL_ATOM, MODEL_ECHO, MODEL_FLOW,
    MODEL_HELM, MODEL_NANO, MODEL_NEON, MODEL_NOVA, MODEL_QUILL, MODEL_SAGE, MODEL_SCOUT,
    MODEL_THINKER, NeonReviewContract,
};
use nioxs_nano::profiles;
use nioxs_storage::NioDb;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use std::time::Instant;

const MAX_TOOL_ROUNDS: usize = 16;
const MAX_INTERNAL_ROUNDS: usize = 28;
/// Durable handoff lifetime; matches the in-memory retention window.
const FLOW_SESSION_TTL: u64 = 600;
/// Serialized handoff budget, kept under the 16 KiB NioDB record cap.
const FLOW_RECORD_BUDGET: usize = 15 * 1024;
/// Stages that gather project evidence directly. A plan without one of these
/// cannot back a filesystem claim, so an inspection request is routed to one.
const READ_TOOL_STAGES: [&str; 5] = [
    MODEL_NANO,
    MODEL_NEON,
    MODEL_SCOUT,
    MODEL_THINKER,
    MODEL_ATOM,
];
/// Assertions that a project listing or scan actually happened.
const INSPECTION_MARKERS: [&str; 16] = [
    "directory scan",
    "file scan",
    "filesystem scan",
    "file system scan",
    "directory listing",
    "file listing",
    "filesystem listing",
    "file system listing",
    "listing shows",
    "scan completed",
    "scan shows",
    "comprehensive listing",
    "was inspected",
    "was scanned",
    "i inspected",
    "i scanned",
];
const PATH_EXTENSIONS: [&str; 18] = [
    "rs", "md", "toml", "json", "jsonl", "ts", "tsx", "js", "py", "go", "yaml", "yml", "txt",
    "lock", "html", "css", "sh", "java",
];
#[derive(Clone, Deserialize, Serialize)]
struct Stage {
    model: String,
    task: String,
}
#[derive(Clone, Deserialize, Serialize)]
struct Plan {
    goal: String,
    success_criteria: Vec<String>,
    needs_context: bool,
    stages: Vec<Stage>,
}
#[derive(Clone, Deserialize, Serialize)]
struct Session {
    fingerprint: String,
    helm: bool,
    plan: Option<Plan>,
    stage: usize,
    drafts: Vec<Value>,
    feedback: Vec<Value>,
    pending: Vec<String>,
    stage_rounds: usize,
    total_rounds: usize,
    repairs: usize,
    validation_failures: usize,
    reasoning_escalated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    atom_contract: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    neon_contract: Option<Value>,
    /// Monotonic last-touch time. Never serialized: a restored handoff starts a
    /// fresh in-memory window, bounded by the durable record's own expiry.
    #[serde(skip, default = "Instant::now")]
    touched: Instant,
}
static SESSIONS: OnceLock<Mutex<HashMap<String, Session>>> = OnceLock::new();
fn sessions() -> &'static Mutex<HashMap<String, Session>> {
    SESSIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// A size-bounded, serializable snapshot of a handoff session. Returns `None`
/// when even a trimmed snapshot exceeds the record budget, so an oversized
/// handoff stays in memory only instead of being stored partially.
fn persistable_session(session: &Session) -> Option<Value> {
    let mut value = serde_json::to_value(session).ok()?;
    if serde_json::to_vec(&value).ok()?.len() <= FLOW_RECORD_BUDGET {
        return Some(value);
    }
    // Tool results are per-stage and re-supplied by the caller on continuation.
    value["feedback"] = json!([]);
    if serde_json::to_vec(&value).ok()?.len() <= FLOW_RECORD_BUDGET {
        return Some(value);
    }
    // Drop the oldest carried-forward drafts until the snapshot fits.
    let mut dropped = false;
    while serde_json::to_vec(&value).ok()?.len() > FLOW_RECORD_BUDGET {
        if value["drafts"].as_array_mut()?.is_empty() {
            return None;
        }
        value["drafts"].as_array_mut()?.remove(0);
        dropped = true;
    }
    if dropped {
        value["truncated"] = json!(true);
    }
    Some(value)
}

/// Tool results a turn actually supplied, the only basis on which a filesystem
/// claim may be accepted as evidence.
struct ToolEvidence {
    text: String,
    results: usize,
}

/// Path-like tokens in a sentence: `src/lib.rs`, `nio-guru/apps/server`.
///
/// A token only counts as a path claim when it has a positive path shape: a
/// known file extension, an explicit path prefix, or at least two separators.
/// Prose that joins two hyphenated concepts with a slash
/// (`speech-to-text/text-to-speech`) is not a filesystem claim, and treating it
/// as one withheld an answer that was in fact grounded in a read file.
fn path_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| c.is_whitespace() || "()[]{}\"'`,;".contains(c))
        // Trailing sentence punctuation only: trimming a leading dot turned
        // `./README.md` into `/readme.md`, which no tool result can contain.
        .map(|token| token.trim_end_matches(|c: char| ".,;:!?*".contains(c)))
        .filter(|token| !token.is_empty() && looks_like_path(token))
        .map(|token| token.to_ascii_lowercase())
        .collect()
}

fn looks_like_path(token: &str) -> bool {
    if token.starts_with("./") || token.starts_with("../") || token.starts_with('/') {
        return true;
    }
    if token.matches('/').count() >= 2 {
        return true;
    }
    token
        .rsplit_once('.')
        .is_some_and(|(_, ext)| PATH_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
}

/// Nouns and verbs that mark the token next to them as a folder being asked about.
const FOLDER_CUES: [&str; 18] = [
    "folder",
    "directory",
    "dir",
    "repo",
    "project",
    "workspace",
    "package",
    "module",
    "check",
    "checking",
    "inspect",
    "explore",
    "summarize",
    "contents",
    "inside",
    "under",
    "list",
    "files",
];

/// The caller-project folder a request asks about, when one is named.
///
/// `find_files` returns a bounded list of files, so an unscoped `**/*` sample is
/// a poor evidence basis for a folder question and a directory glob matches no
/// file at all: both look like "the folder is not there". Scoping the listing to
/// the named folder returns its actual files instead.
fn named_folder(prompt: &str) -> Option<String> {
    let words: Vec<&str> = prompt
        .split(|c: char| c.is_whitespace() || ",.;:!?()[]{}\"'`".contains(c))
        .filter(|word| !word.is_empty())
        .collect();
    for (index, word) in words.iter().enumerate() {
        let token = word
            .trim_matches('*')
            .trim_end_matches('/')
            .trim_start_matches("./");
        // A folder name looks like a project path: hyphen, underscore, or
        // separator, no file extension, and never a leading flag.
        let folderish = !token.is_empty()
            && token.len() <= 64
            && !token.starts_with('-')
            && !token.contains('.')
            && (token.contains('/') || token.contains('-') || token.contains('_'));
        if !folderish {
            continue;
        }
        let cue_before = index
            .checked_sub(1)
            .and_then(|i| words.get(i))
            .is_some_and(|word| FOLDER_CUES.contains(&word.to_ascii_lowercase().as_str()));
        let cue_after = words
            .get(index + 1)
            .is_some_and(|word| FOLDER_CUES.contains(&word.to_ascii_lowercase().as_str()));
        if cue_before || cue_after {
            return Some(token.to_string());
        }
    }
    None
}

/// Phrases that assert something is missing from the caller's project.
const ABSENCE_MARKERS: [&str; 10] = [
    "does not exist",
    "doesn't exist",
    "not present",
    "no such",
    "was not found",
    "could not find",
    "couldn't find",
    "cannot find",
    "not found in",
    "does not contain",
];

/// Lines that claim a path or folder is absent.
fn absence_claims(content: &str) -> Vec<String> {
    let lower = content.to_ascii_lowercase();
    let mut claims: Vec<String> = lower
        .lines()
        .filter(|line| ABSENCE_MARKERS.iter().any(|marker| line.contains(marker)))
        .map(|line| line.trim().chars().take(200).collect())
        .filter(|line: &String| !line.is_empty())
        .collect();
    claims.dedup();
    claims
}

/// Lines that assert an inspection or listing happened.
fn inspection_claims(content: &str) -> Vec<String> {
    let lower = content.to_ascii_lowercase();
    let mut claims: Vec<String> = lower
        .lines()
        .filter(|line| {
            INSPECTION_MARKERS
                .iter()
                .any(|marker| line.contains(marker))
        })
        .map(|line| line.trim().chars().take(200).collect())
        .filter(|line: &String| !line.is_empty())
        .collect();
    claims.dedup();
    claims
}

/// Drop NioFlow's internal `[<model> completed stage: <task>]` headers from an
/// answer. They are loop bookkeeping; a model can echo one when a prior draft is
/// the last assistant turn, and it must never reach the user.
fn strip_stage_markers(content: &str) -> String {
    let mut kept = String::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[')
            && trimmed.ends_with(']')
            && trimmed.contains(" completed stage: ")
        {
            continue;
        }
        kept.push_str(line);
        kept.push('\n');
    }
    kept.trim().to_string()
}

/// The deterministic answer emitted in place of an unbacked Helmet verdict.
fn withheld_helm_answer(unverified: &[String], tool_results: usize) -> String {
    let mut answer = String::from(
        "⚠️ **NioHelm withheld this verdict.** The synthesized answer asserted filesystem facts that this turn never received as tool results, so NioFlow did not accept them as evidence.\n\n**Withheld assertions (unverified):**\n",
    );
    for claim in unverified {
        answer.push_str(&format!("- {claim}\n"));
    }
    // State what actually happened: a turn with results is not a turn without
    // tools, and telling the caller to add read tools would be wrong.
    if tool_results == 0 {
        answer.push_str(
            "\nThis turn supplied no tool result at all. Run the request through a stage with read tools, or share the listing, and ask again.\n",
        );
    } else {
        answer.push_str(&format!(
            "\n{tool_results} tool result(s) were supplied this turn, but none contained these paths. Ask again with those files read, or share the listing.\n"
        ));
    }
    answer
}

struct Runner {
    consult: CodingConsult,
    offered: Vec<Value>,
    session: Session,
    key: String,
    id: String,
    turn: String,
    metadata: Value,
    guard: TurnGuard,
    queue: VecDeque<String>,
    done: bool,
    announced: bool,
    internal_rounds: usize,
    db: NioDb,
    deployment: String,
    persist_key: String,
}

pub(super) async fn respond(
    consult: CodingConsult,
    request: &ChatRequest,
    id: String,
    metadata: Value,
    guard: TurnGuard,
    turn: String,
    instance: &str,
    db: &NioDb,
    deployment: &str,
    fingerprint: &str,
) -> Response {
    let key = format!("{instance}:{turn}:{}", request.model);
    let persist_key = format!("{deployment}:{turn}:{}", request.model);
    let start = request.validate().expect("validated request");
    let thinker = request.model == MODEL_THINKER;
    let source_prompt = request.messages[start].content.as_str().unwrap_or("");
    let helm = if thinker {
        profiles::thinker_options(source_prompt).1
    } else {
        profiles::flow_options(source_prompt).1
    };
    let continuing = request.messages[start + 1..]
        .iter()
        .any(|message| !message.tool_calls.is_empty());
    let saved = sessions().lock().ok().and_then(|mut sessions| {
        sessions
            .retain(|_, session| session.touched.elapsed() < Duration::from_secs(FLOW_SESSION_TTL));
        sessions.get(&key).cloned()
    });
    let mut session = if let Some(session) = saved {
        if session.fingerprint != fingerprint {
            return coding_json(
                &id,
                json!({"role":"assistant","content":"⚠️ NioFlow's task context changed during a handoff; no further tools were dispatched. Start a fresh task with the current evidence."}),
                "stop",
                metadata,
            );
        }
        session
    } else if continuing {
        // In-memory state is gone: retention expired or the provider restarted.
        // Resume only from an unexpired durable handoff whose fingerprint still
        // matches; otherwise refuse without replaying a captured tool.
        match db.flow_session_get(&persist_key, now_unix()).await {
            Ok(Some(record)) => {
                match serde_json::from_value::<Session>(record["session"].clone()) {
                    Ok(session) if session.fingerprint == fingerprint => session,
                    Ok(_) => {
                        return coding_json(
                            &id,
                            json!({"role":"assistant","content":"⚠️ NioFlow's task context changed during a handoff; no further tools were dispatched. Start a fresh task with the current evidence."}),
                            "stop",
                            metadata,
                        );
                    }
                    Err(_) => {
                        return coding_json(
                            &id,
                            json!({"role":"assistant","content":"⚠️ NioFlow's handoff state expired or the provider restarted. No tool was replayed. Start a fresh task using the results already collected."}),
                            "stop",
                            metadata,
                        );
                    }
                }
            }
            _ => {
                return coding_json(
                    &id,
                    json!({"role":"assistant","content":"⚠️ NioFlow's handoff state expired or the provider restarted. No tool was replayed. Start a fresh task using the results already collected."}),
                    "stop",
                    metadata,
                );
            }
        }
    } else {
        Session {
            fingerprint: fingerprint.into(),
            helm,
            plan: thinker.then(|| Plan {
                goal: consult.prompt.clone(),
                success_criteria: vec!["Answer the original question using evidence; resolve objections or state remaining uncertainty".into()],
                needs_context: false,
                stages: {
                    let mut stages = thinker_stages(&consult.prompt);
                    if helm { stages.push(Stage { model: MODEL_HELM.into(), task: "Give the user-requested final Go / No-Go after Thinker has finished all three reflection phases".into() }); }
                    stages
                },
            }),
            stage: 0,
            drafts: vec![],
            feedback: vec![],
            pending: vec![],
            stage_rounds: 0,
            total_rounds: 0,
            repairs: 0,
            validation_failures: 0,
            reasoning_escalated: false,
            atom_contract: None,
            neon_contract: None,
            touched: Instant::now(),
        }
    };
    let mut announced = continuing;
    if !session.pending.is_empty() {
        let complete = session.pending.iter().all(|id| {
            request.messages[start + 1..]
                .iter()
                .any(|message| message.role == "tool" && message.tool_call_id.as_ref() == Some(id))
        });
        if !complete {
            return coding_json(
                &id,
                json!({"role":"assistant","content":"Waiting for the caller to return the pending tool results; no tool was replayed."}),
                "stop",
                metadata,
            );
        }
        if session
            .plan
            .as_ref()
            .and_then(|plan| plan.stages.get(session.stage))
            .is_some_and(|stage| stage.model == MODEL_NANO)
        {
            session.stage += 1;
            session.stage_rounds = 0;
            announced = false;
        }
        session.pending.clear();
    }
    let offered = request
        .tools
        .iter()
        .filter(|tool| {
            request.permits_tool(tool["function"]["name"].as_str().unwrap_or(""))
                && companion::is_coding_tool(tool["function"]["name"].as_str().unwrap_or(""))
        })
        .cloned()
        .collect();
    let mut runner = Runner {
        consult,
        offered,
        session,
        key,
        id,
        turn,
        metadata,
        guard,
        queue: VecDeque::new(),
        done: false,
        announced,
        internal_rounds: 0,
        db: db.clone(),
        deployment: deployment.to_string(),
        persist_key,
    };
    runner.metadata["progress"] = json!([]);
    let model = runner.model().to_string();
    runner.metadata["model"] = json!(model);
    runner.metadata["routing"]["selected_model"] = json!(model);
    runner.metadata["routing"]["method"] = json!(if model == MODEL_THINKER {
        "rust_bounded_reflection"
    } else {
        "rust_flow_cascade"
    });
    if announced {
        runner
            .queue
            .push_back(status_chunk(&runner.id, &model, None, "thinking"));
    }
    if request.stream {
        let output = stream::unfold(runner, |mut runner| async move {
            loop {
                if let Some(payload) = runner.queue.pop_front() {
                    return Some((Ok::<_, Infallible>(Event::default().data(payload)), runner));
                }
                if runner.done {
                    return None;
                }
                if let Err(error) = runner.advance().await {
                    runner.fail(&error).await;
                }
            }
        });
        Sse::new(output).into_response()
    } else {
        while !runner.done {
            if let Err(error) = runner.advance().await {
                runner.fail(&error).await;
            }
            runner.queue.clear();
        }
        let message = runner.metadata["flow_message"].take();
        let finish = runner.metadata["flow_finish"].take();
        coding_json(
            &runner.id,
            message,
            finish.as_str().unwrap_or("stop"),
            runner.metadata.clone(),
        )
    }
}

impl Runner {
    fn model(&self) -> &str {
        self.session
            .plan
            .as_ref()
            .and_then(|plan| plan.stages.get(self.session.stage))
            .map_or(MODEL_FLOW, |stage| stage.model.as_str())
    }
    fn announce(&mut self) {
        let model = self.model().to_string();
        let previous = self.metadata["routing"]["selected_model"]
            .as_str()
            .unwrap_or(MODEL_FLOW)
            .to_string();
        let label = if model == MODEL_THINKER {
            let task = &self.session.plan.as_ref().unwrap().stages[self.session.stage].task;
            if task.starts_with("Challenge:") {
                "🧠 NioThinker · Challenging"
            } else if task.starts_with("Refine:") {
                "🧠 NioThinker · Refining"
            } else {
                "🧠 NioThinker · Drafting"
            }
        } else {
            companion::model_stage(&model).unwrap_or(&model)
        };
        let message = if model != MODEL_FLOW && model != MODEL_HELM && previous != model {
            format!(
                "↪ {}\n",
                label.trim_start_matches(|c: char| c == '↪' || c.is_whitespace())
            )
        } else {
            format!("{label}\n")
        };
        self.metadata["model"] = json!(model);
        self.metadata["routing"]["selected_model"] = json!(model);
        self.metadata["routing"]["skill"] = json!(profiles::skill_name(&model));
        self.metadata["routing"]["method"] = json!(if model == MODEL_THINKER {
            "rust_bounded_reflection"
        } else {
            "rust_flow_cascade"
        });
        self.metadata["progress"].as_array_mut().unwrap().push(json!({"kind":if model == MODEL_FLOW {"planning"} else {"model_handoff"},"from":previous,"to":model,"message":message}));
        self.queue
            .push_back(status_chunk(&self.id, &model, Some(message), "routing"));
        self.queue
            .push_back(status_chunk(&self.id, &model, None, "thinking"));
        if model == MODEL_FLOW && !self.session.helm {
            let skipped = format!("{}\n", companion::TRAIL_SKIP_HELM);
            self.metadata["progress"]
                .as_array_mut()
                .unwrap()
                .push(json!({"kind":"helm_skipped","message":skipped}));
            self.queue
                .push_back(status_chunk(&self.id, &model, Some(skipped), "routing"));
        }
        self.announced = true;
    }
    fn is_final(&self) -> bool {
        self.session
            .plan
            .as_ref()
            .is_some_and(|plan| self.session.stage + 1 == plan.stages.len())
    }
    fn messages(&self, stage: &Stage) -> Vec<Value> {
        let mut base = self.consult.messages.clone();
        // Replace Flow's instructions, rather than stacking conflicting roles.
        base.retain(|message| {
            !(message["role"] == "system"
                && message["content"].as_str().is_some_and(|content| {
                    content.contains(profiles::instructions(MODEL_FLOW).unwrap_or("unmatchable"))
                        || content.contains(
                            profiles::instructions(MODEL_THINKER).unwrap_or("unmatchable"),
                        )
                }))
        });
        let mut messages = profiles::messages(&stage.model, base);
        messages.extend(self.session.drafts.clone());
        messages.extend(self.session.feedback.clone());
        messages.push(json!({"role":"system","content":format!("NioFlow assigned this bounded stage: {}. The current user request, and the only task you answer, is: \"{}\". Plan goal: {}. Success criteria: {}. Treat earlier user turns as context only; never answer a previous question in their place. Reuse all supplied evidence. Complete only your assigned work, preserve the caller persona and permissions, and do not claim another role ran. If a tool is unavailable or its budget is exhausted, finish with the limitation. Do not call another specialist. Never repeat NioFlow's internal stage headers in your answer.", stage.task, self.consult.prompt, self.session.plan.as_ref().map_or(self.consult.prompt.as_str(), |plan| plan.goal.as_str()), self.session.plan.as_ref().map_or(String::new(), |plan| plan.success_criteria.join("; ")))}));
        if stage.model == MODEL_ATOM {
            messages.push(json!({"role":"system","content":"Implement the requested changes using only offered tools. Conclude your report with a structured contract block in JSON (`json:atom`) containing summary, patch_status, modified_files, checks_executed, and unverified_items."}));
        }
        if stage.model == MODEL_NEON {
            messages.push(json!({"role":"system","content":"Review the diff, code, and verification results. Conclude your review with a structured review block in JSON (`json:neon`) containing verdict (approve, request_changes, comment), summary, verification_passed, and findings."}));
        }
        if stage.model == MODEL_HELM {
            messages.push(json!({"role":"system","content":"Synthesize a useful final answer to the original user request from the specialist drafts, then give a concise Go / Conditional Go / No-Go grounded in actual results formatted with structured verdict fields (verdict, summary, decisive_evidence, unmet_criteria, residual_risks). Do not call another specialist or ask for tools."}));
        }
        if self.is_final()
            && !self.session.helm
            && self.metadata["routing"]["requested_model"] != MODEL_THINKER
        {
            messages.push(json!({"role":"system","content":"The user selected --no-helm. You are the final specialist stage. Deliver a complete answer to the original user request using the specialist drafts and actual tool results. Do not imply Helm reviewed the task, do not invent a Go / No-Go gate, and do not ask for another model handoff."}));
        }
        messages
    }
    async fn query(&mut self, messages: &[Value], tools: &[Value]) -> Result<Value, String> {
        self.internal_rounds += 1;
        if self.internal_rounds > MAX_INTERNAL_ROUNDS {
            return Err("NioFlow's bounded stage budget was exhausted".into());
        }
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|error| error.to_string())?;
        let cancelled = self
            .guard
            .active
            .0
            .lock()
            .ok()
            .and_then(|active| active.get(&self.guard.id).map(|sender| sender.subscribe()));
        let reasons = complexity_evidence(
            &self.consult.prompt,
            self.session.plan.as_ref(),
            self.session.validation_failures,
            messages,
        );
        // The planner starts on Mini Code; escalation persists across later handoffs.
        self.session.reasoning_escalated |= self.session.plan.is_some() && reasons.len() >= 3;
        let previous_upstream = self.metadata["upstream_model"].as_str().map(str::to_string);
        let request = nioxs_nano::upstream::send_chat_for_complexity(
            &client,
            &mut self.consult.config,
            messages,
            tools,
            false,
            self.session.reasoning_escalated,
        );
        let response = if let Some(mut cancelled) = cancelled {
            if *cancelled.borrow() { return Err("NioFlow cancelled; no further action dispatched".into()); }
            tokio::select! { result = request => result, _ = cancelled.changed() => return Err("NioFlow cancelled; no further action dispatched".into()) }
        } else { request.await }.map_err(|error| format!("upstream request failed: {error}"))?;
        self.metadata["upstream_model"] = json!(self.consult.config.model);
        self.metadata["upstream_routing"] = json!({"selected_model":self.consult.config.model,"high_complexity":self.session.reasoning_escalated,"evidence":reasons,"automatic":self.consult.config.automatic_model});
        let upstream_transition = previous_upstream
            .as_ref()
            .is_some_and(|previous| previous != &self.consult.config.model)
            && (self.consult.config.model.starts_with("stepfun/")
                || previous_upstream
                    .as_ref()
                    .is_some_and(|previous| previous.starts_with("stepfun/")));
        if upstream_transition {
            let message = format!(
                "↪ Upstream: {} → {}{}\n",
                previous_upstream.as_deref().unwrap_or("default"),
                self.consult.config.model,
                if self.consult.config.model.starts_with("stepfun/") {
                    " (high task complexity)"
                } else {
                    " (availability fallback)"
                }
            );
            self.metadata["progress"].as_array_mut().unwrap().push(json!({"kind":"upstream_switched","message":message,"from":previous_upstream,"to":self.consult.config.model}));
            self.queue.push_back(status_chunk(
                &self.id,
                self.model(),
                Some(message),
                "routing",
            ));
        }
        let fallback_message = record_upstream_fallback(&self.consult.config, &mut self.metadata);
        if !upstream_transition && let Some(message) = fallback_message {
            self.queue.push_back(status_chunk(
                &self.id,
                self.model(),
                Some(message),
                "routing",
            ));
        }
        if !response.status().is_success() {
            return Err(format!("upstream returned {}", response.status()));
        }
        let mut response = response;
        let mut body = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| format!("upstream body failed: {error}"))?
        {
            if body.len() + chunk.len() > 256 * 1024 {
                return Err("upstream completion exceeded the bounded stage limit".into());
            }
            body.extend_from_slice(&chunk);
        }
        serde_json::from_slice(&body)
            .map_err(|error| format!("invalid upstream completion: {error}"))
    }
    async fn advance(&mut self) -> Result<(), String> {
        if !self.announced {
            self.announce();
            return Ok(());
        }
        if self.session.plan.is_none() {
            let stage = Stage {
                model: MODEL_FLOW.into(),
                task: "Prepare the plan only".into(),
            };
            let mut messages = self.messages(&stage);
            messages.push(json!({"role":"system","content":"Return only a JSON object with goal (string), success_criteria (array of strings), needs_context (boolean), and stages (array of {model, task}). Choose only the useful models from nio-echo for direct answers, nio-sage for teaching, nio-quill for writing, nio-scout for sourced research, nio-thinker for challenging assumptions, nio-nova for ideas, nio-atom for implementation, nio-neon for review/diagnosis. Do not include nio-flow, nio-nano, or nio-helm; Rust inserts bounded local evidence when needed and the final Helm gate. No tools or project inspections at this planning stage. Keep at most three specialist stages, no repeated model, and use the user's actual requested scope and offered capabilities. Set needs_context=false for casual questions, general knowledge, explanations, new-project ideas, and feasibility questions unless the answer requires this caller's existing files. Prefer Echo for casual answers. Do not turn requests for suggestions or feasibility into a request to build or approve a project."}));
            let result = self.query(&messages, &[]).await?;
            let content = result["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("");
            let mut plan = parse_plan(content, &self.consult.prompt);
            // `parse_plan` already inserts a native listing stage for an
            // inspection request, so add one only when the plan has none: two
            // listing stages made the caller run the same scan twice.
            let already_lists = plan.stages.iter().any(|stage| stage.model == MODEL_NANO);
            if plan.needs_context
                && !already_lists
                && self
                    .offered
                    .iter()
                    .any(|tool| tool["function"]["name"] == "find_files")
            {
                plan.stages.insert(
                    0,
                    Stage {
                        model: MODEL_NANO.into(),
                        task: "Gather a bounded caller-project file listing".into(),
                    },
                );
            }
            if self.session.helm {
                plan.stages.push(Stage { model: MODEL_HELM.into(), task: "Answer the original request and give the final Go / No-Go after all applicable stages".into() });
            }
            self.metadata["flow"] = json!({"goal":plan.goal,"success_criteria":plan.success_criteria,"stages":plan.stages,"stage_index":0,"helm_enabled":self.session.helm});
            self.session.plan = Some(plan);
            self.announced = false;
            return Ok(());
        }
        let stage = self
            .session
            .plan
            .as_ref()
            .unwrap()
            .stages
            .get(self.session.stage)
            .ok_or("invalid flow stage")?
            .clone();
        let mut flow_meta = json!({"goal":self.session.plan.as_ref().unwrap().goal,"stages":self.session.plan.as_ref().unwrap().stages,"stage_index":self.session.stage,"completed_stages":self.session.stage,"helm_enabled":self.session.helm});
        if let Some(ref atom) = self.session.atom_contract {
            flow_meta["atom_contract"] = atom.clone();
        }
        if let Some(ref neon) = self.session.neon_contract {
            flow_meta["neon_contract"] = neon.clone();
        }
        self.metadata["flow"] = flow_meta;
        if stage.model == MODEL_NANO {
            // Scope the listing to a folder the caller named, so the evidence is
            // about that folder instead of a bounded sample of the whole project.
            let glob = match named_folder(&self.consult.prompt) {
                Some(folder) => format!("{folder}/**/*"),
                None => "**/*".to_string(),
            };
            self.metadata["flow"]["evidence_glob"] = json!(glob);
            let directive = format!("--find-files {glob}");
            let native: ChatRequest = serde_json::from_value(json!({"model":MODEL_NANO,"messages":[{"role":"user","content":directive}],"tools":self.offered.iter().filter(|tool| tool["function"]["name"] == "find_files").collect::<Vec<_>>(),"tool_choice":"auto"})).map_err(|_| "cannot construct Nano evidence request")?;
            let decision = match nioxs_nano::decide_with_turn(
                &native,
                "nioflow-evidence",
                Some(&self.turn),
            ) {
                Ok(decision) => decision,
                Err(error) => {
                    self.session.drafts.push(json!({"role":"assistant","content":format!("Optional Nano file listing unavailable: {error}. No files were read; continue with supplied evidence and report any necessary gap.")}));
                    self.next_stage();
                    return Ok(());
                }
            };
            let Some(native) = decision.call else {
                self.session.drafts.push(json!({"role":"assistant","content":"Optional Nano file listing produced no tool action. No files were read; continue with supplied evidence and report any necessary gap."}));
                self.next_stage();
                return Ok(());
            };
            self.metadata["flow"]["nano_intent"] = json!(decision.intent);
            let calls = [RequestedCall {
                id: native.id,
                name: native.function.name,
                arguments: native.function.arguments,
            }];
            let relayed = relay_offered_coding_calls(
                &calls,
                &self.turn,
                &self.offered,
                &self.consult.messages,
            )?;
            return self.handoff_tools(relayed).await;
        }
        let messages = self.messages(&stage);
        let tools: Vec<Value> = if stage.model == MODEL_HELM
            || self.session.total_rounds >= MAX_TOOL_ROUNDS
            || self.session.stage_rounds >= profiles::tool_rounds(&stage.model)
        {
            vec![]
        } else {
            self.offered
                .iter()
                .filter(|tool| {
                    if tool["function"]["name"] != "web_fetch"
                        && !profiles::project_evidence_relevant(&self.consult.prompt)
                    {
                        return false;
                    }
                    if profiles::direct_answer(&stage.model) {
                        let refining =
                            stage.model == MODEL_THINKER && stage.task.starts_with("Refine:");
                        !refining
                            && profiles::answer_tool(
                                &stage.model,
                                tool["function"]["name"].as_str().unwrap_or(""),
                            )
                    } else {
                        stage.model == MODEL_ATOM
                            || is_read_tool(tool["function"]["name"].as_str().unwrap_or(""))
                    }
                })
                .cloned()
                .collect()
        };
        let result = self.query(&messages, &tools).await?;
        let content = result["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let mut calls = message_tool_calls(&result);
        if calls.is_empty() {
            calls = extract_xml_tool_calls(&content);
        }
        if !calls.is_empty() {
            match relay_offered_coding_calls(&calls, &self.turn, &tools, &messages) {
                Ok(relayed) => {
                    let mut fresh = vec![];
                    for call in relayed {
                        if is_read_tool(&call.name) && self.session.repairs < 2 {
                            if let Some(result) = cached_result(&call, &self.consult.messages) {
                                let id = format!(
                                    "flow_cached_{}_{}_{}",
                                    self.session.stage,
                                    self.internal_rounds,
                                    self.session.feedback.len()
                                );
                                self.feedback(&id, &call.name, &call.arguments, &result);
                                continue;
                            }
                        }
                        fresh.push(call);
                    }
                    if !fresh.is_empty() {
                        return self.handoff_tools(fresh).await;
                    }
                    self.session.repairs += 1;
                    self.session.stage_rounds += 1;
                    return Ok(());
                }
                Err(error) if self.session.repairs < 2 && !tools.is_empty() => {
                    self.session.repairs += 1;
                    self.session.validation_failures += 1;
                    self.session.feedback.push(json!({"role":"system","content":format!("Your last tool-call proposal was rejected before execution: {error}. Correct the required arguments or finish from existing evidence. No tool was executed.")}));
                    return Ok(());
                }
                Err(error) => {
                    self.session.drafts.push(json!({"role":"assistant","content":format!("{} stage limitation: {error}. No tool was executed.", stage.model)}));
                    if self.is_final() {
                        self.finish(format!(
                            "The final stage could not complete: {error}. No tool was executed."
                        ));
                        return Ok(());
                    }
                    self.next_stage();
                    return Ok(());
                }
            }
        }
        if stage.model == MODEL_ATOM {
            if let Some(contract) = AtomContract::parse_from_text(&content) {
                let val = json!(contract);
                self.session.atom_contract = Some(val.clone());
                self.metadata["flow"]["atom_contract"] = val;
            }
        }
        if stage.model == MODEL_NEON {
            if let Some(contract) = NeonReviewContract::parse_from_text(&content) {
                let val = json!(contract);
                self.session.neon_contract = Some(val.clone());
                self.metadata["flow"]["neon_contract"] = val;
            }
        }
        if self.is_final() {
            if content.trim().is_empty() {
                return Err("The final stage returned no answer; completion is unverified".into());
            }
            let content = if stage.model == MODEL_HELM {
                self.guard_helm(content)
            } else {
                content
            };
            // Internal stage headers are loop bookkeeping, never user output.
            let stripped = strip_stage_markers(&content);
            self.finish(if stripped.trim().is_empty() {
                content
            } else {
                stripped
            });
        } else {
            self.session.drafts.push(json!({"role":"assistant","content":format!("[{} completed stage: {}]\n{}", stage.model, stage.task, content.chars().take(12000).collect::<String>())}));
            self.next_stage();
        }
        Ok(())
    }
    fn feedback(&mut self, id: &str, name: &str, arguments: &str, content: &str) {
        self.session.feedback.push(json!({"role":"assistant","content":null,"tool_calls":[{"id":id,"type":"function","function":{"name":name,"arguments":arguments}}]}));
        self.session.feedback.push(json!({"role":"tool","tool_call_id":id,"content":content.chars().take(16000).collect::<String>()}));
    }
    fn next_stage(&mut self) {
        self.session.stage += 1;
        self.session.stage_rounds = 0;
        self.session.repairs = 0;
        self.session.feedback.clear();
        self.announced = false;
    }
    async fn handoff_tools(&mut self, calls: Vec<ClientCall>) -> Result<(), String> {
        self.session.pending = calls.iter().map(|call| call.id.clone()).collect();
        self.session.total_rounds += 1;
        self.session.stage_rounds += 1;
        self.session.touched = Instant::now();
        {
            let mut stored = sessions().lock().map_err(|_| "flow state unavailable")?;
            if stored.len() >= 32 && !stored.contains_key(&self.key) {
                if let Some(oldest) = stored
                    .iter()
                    .min_by_key(|(_, session)| session.touched)
                    .map(|(key, _)| key.clone())
                {
                    stored.remove(&oldest);
                }
            }
            stored.insert(self.key.clone(), self.session.clone());
        }
        self.persist().await;
        let model = self.model().to_string();
        for (index, call) in calls.iter().enumerate() {
            self.queue
                .push_back(call_delta(&self.id, &model, call, index));
        }
        let tool_calls: Vec<Value> = calls.iter().map(|call| json!({"id":call.id,"type":"function","function":{"name":call.name,"arguments":call.arguments}})).collect();
        self.metadata["flow_message"] =
            json!({"role":"assistant","content":null,"tool_calls":tool_calls});
        self.metadata["flow_finish"] = json!("tool_calls");
        self.queue.push_back(coding_terminal(
            &self.id,
            "tool_calls",
            &public_metadata(&self.metadata),
        ));
        self.queue.push_back("[DONE]".into());
        self.done = true;
        Ok(())
    }
    fn finish(&mut self, mut answer: String) {
        answer.push_str(&companion::coding_footer(&self.consult.config.model));
        self.queue.push_back(sse_chunk(
            &self.id,
            self.model(),
            json!({"content":answer}),
            Value::Null,
        ));
        self.metadata["flow"]["completed_stages"] = json!(self.session.stage + 1);
        if let Some(ref atom) = self.session.atom_contract {
            self.metadata["flow"]["atom_contract"] = atom.clone();
        }
        if let Some(ref neon) = self.session.neon_contract {
            self.metadata["flow"]["neon_contract"] = neon.clone();
        }
        self.metadata["flow_message"] = json!({"role":"assistant","content":answer});
        self.metadata["flow_finish"] = json!("stop");
        self.queue.push_back(coding_terminal(
            &self.id,
            "stop",
            &public_metadata(&self.metadata),
        ));
        self.queue.push_back("[DONE]".into());
        if let Ok(mut sessions) = sessions().lock() {
            sessions.remove(&self.key);
        }
        self.clear_persisted();
        self.done = true;
    }
    async fn fail(&mut self, error: &str) {
        self.metadata["outcome"] = json!("capability_limitation");
        let retryable = error == "The final stage returned no answer; completion is unverified";
        self.metadata["flow"]["status"] = json!("failed");
        self.metadata["flow"]["error"] = json!(error);
        self.metadata["flow"]["retryable"] = json!(retryable);
        self.queue.push_back(sse_chunk(&self.id, self.model(), json!({"content":format!("⚠️ {error}. NioFlow stopped without dispatching further tools.")}), Value::Null));
        self.metadata["flow_message"] = json!({"role":"assistant","content":format!("⚠️ {error}. NioFlow stopped without dispatching further tools.")});
        self.metadata["flow_finish"] = json!("stop");
        self.queue.push_back(coding_terminal(
            &self.id,
            "stop",
            &public_metadata(&self.metadata),
        ));
        self.queue.push_back("[DONE]".into());
        if let Ok(mut sessions) = sessions().lock() {
            if retryable {
                self.session.touched = Instant::now();
                if sessions.len() >= 32 && !sessions.contains_key(&self.key) {
                    if let Some(oldest) = sessions
                        .iter()
                        .min_by_key(|(_, session)| session.touched)
                        .map(|(key, _)| key.clone())
                    {
                        sessions.remove(&oldest);
                    }
                }
                sessions.insert(self.key.clone(), self.session.clone());
            } else {
                sessions.remove(&self.key);
            }
        }
        if retryable {
            self.persist().await;
        } else {
            self.clear_persisted();
        }
        self.done = true;
    }
    /// Best-effort durable snapshot of the current handoff. A storage failure or
    /// an oversized record leaves the in-memory state authoritative; the turn
    /// never fails because persistence did.
    async fn persist(&self) {
        let Some(session) = persistable_session(&self.session) else {
            return;
        };
        let _ = self
            .db
            .flow_session_set(
                &self.deployment,
                &self.persist_key,
                &session,
                now_unix(),
                FLOW_SESSION_TTL,
            )
            .await;
    }
    /// Tool results this turn supplied, as the evidence basis for Helm.
    fn supplied_tool_evidence(&self) -> ToolEvidence {
        let mut text = String::new();
        let mut results = 0usize;
        for message in self
            .consult
            .messages
            .iter()
            .chain(self.session.feedback.iter())
        {
            if message["role"] != "tool" {
                continue;
            }
            let Some(content) = message["content"]
                .as_str()
                .filter(|content| !content.trim().is_empty())
            else {
                continue;
            };
            results += 1;
            text.push_str(content);
            text.push('\n');
        }
        ToolEvidence {
            text: text.to_ascii_lowercase(),
            results,
        }
    }

    /// Rejects a Helm answer that asserts filesystem facts this turn was never
    /// given evidence for. Returns the answer to emit: the original when every
    /// claim is backed by a supplied tool result, otherwise a deterministic
    /// refusal that replaces it. Helm runs without tools, so this is the one
    /// stage where a fabricated inspection could otherwise reach the user.
    fn guard_helm(&mut self, content: String) -> String {
        let evidence = self.supplied_tool_evidence();
        let mut unverified: Vec<String> = Vec::new();
        if evidence.results == 0 {
            unverified.extend(inspection_claims(&content));
        }
        let verdict = HelmVerdictContract::parse_from_text(&content);
        if let Some(verdict) = &verdict {
            for supporting in &verdict.decisive_evidence {
                for path in path_tokens(supporting) {
                    if !evidence.text.contains(&path) {
                        unverified.push(format!("path `{path}` asserted as evidence"));
                    }
                }
            }
        }
        // `find_files` lists files, so an empty result says "no files matched",
        // never "no such folder". A claim that something is missing, supported
        // only by an empty listing, is not evidence.
        if evidence.text.contains("\"files\":[]") || evidence.text.contains("\"files\": []") {
            unverified.extend(absence_claims(&content));
        }
        if unverified.is_empty() {
            if let Some(verdict) = verdict {
                self.metadata["flow"]["helm_verdict"] = json!(verdict);
            }
            return content;
        }
        unverified.sort();
        unverified.dedup();
        self.metadata["flow"]["helm_verdict"] = json!(HelmVerdictContract {
            verdict: HelmGateVerdict::NoGo,
            summary:
                "Withheld: the final answer asserted filesystem facts with no supporting tool result"
                    .into(),
            decisive_evidence: Vec::new(),
            unmet_criteria: unverified.clone(),
            residual_risks: vec![
                "NioHelm's synthesized answer was replaced by a deterministic evidence guard".into(),
            ],
            reversibility: Some("high".into()),
        });
        self.metadata["flow"]["helm_guard"] =
            json!({"withheld_claims": unverified, "tool_results": evidence.results});
        withheld_helm_answer(&unverified, evidence.results)
    }

    /// Best-effort durable cleanup once a turn ends.
    fn clear_persisted(&self) {
        let db = self.db.clone();
        let key = self.persist_key.clone();
        tokio::spawn(async move {
            let _ = db.flow_session_remove(&key).await;
        });
    }
}
fn public_metadata(metadata: &Value) -> Value {
    let mut value = metadata.clone();
    value.as_object_mut().unwrap().remove("flow_message");
    value.as_object_mut().unwrap().remove("flow_finish");
    value
}
fn is_read_tool(name: &str) -> bool {
    matches!(
        name,
        "read_file"
            | "find_files"
            | "search_code"
            | "read_skill_file"
            | "git_status"
            | "git_diff"
            | "web_fetch"
    )
}
fn cached_result(call: &ClientCall, messages: &[Value]) -> Option<String> {
    let start = messages
        .iter()
        .rposition(|message| message["role"] == "user")
        .map_or(0, |index| index + 1);
    let messages = &messages[start..];
    let wanted: Value = serde_json::from_str(&call.arguments).ok()?;
    for message in messages.iter().rev() {
        let Some(calls) = message["tool_calls"].as_array() else {
            continue;
        };
        for prior in calls {
            let name = prior["function"]["name"].as_str()?;
            if !is_read_tool(name) {
                return None;
            }
            let arguments: Value =
                serde_json::from_str(prior["function"]["arguments"].as_str()?).ok()?;
            if name == call.name && arguments == wanted {
                let id = prior["id"].as_str()?;
                let result = messages.iter().rev().find(|result| result["role"] == "tool" && result["tool_call_id"] == id)?["content"].as_str()?;
                if result.trim_start().starts_with("Error") || result.trim_start().starts_with("⚠")
                {
                    return None;
                }
                return Some(result.to_string());
            }
        }
    }
    None
}
fn parse_plan(content: &str, prompt: &str) -> Plan {
    let clean = content
        .trim()
        .strip_prefix("```json")
        .or_else(|| content.trim().strip_prefix("```"))
        .unwrap_or(content.trim())
        .trim()
        .trim_end_matches("```")
        .trim();
    let parsed = serde_json::from_str::<Plan>(clean).ok().filter(|plan| {
        !plan.goal.trim().is_empty()
            && plan.goal.len() <= 2048
            && !plan.success_criteria.is_empty()
            && plan.success_criteria.len() <= 6
            && plan
                .success_criteria
                .iter()
                .all(|criterion| criterion.len() <= 512)
            && !plan.stages.is_empty()
            && plan.stages.len() <= 3
            && plan.stages.iter().all(|stage| {
                matches!(
                    stage.model.as_str(),
                    MODEL_ECHO
                        | MODEL_SAGE
                        | MODEL_QUILL
                        | MODEL_SCOUT
                        | MODEL_THINKER
                        | MODEL_NOVA
                        | MODEL_ATOM
                        | MODEL_NEON
                ) && !stage.task.trim().is_empty()
                    && stage.task.len() <= 1024
            })
    });
    let mut plan = parsed.unwrap_or_else(|| {
        let lower = prompt.to_ascii_lowercase();
        let build = ["implement", "fix", "refactor", "add", "build", "create"]
            .iter()
            .any(|word| {
                lower
                    .split(|c: char| !c.is_alphanumeric())
                    .any(|token| token == *word)
            });
        let creative = ["suggest", "idea", "creative", "brainstorm", "design"]
            .iter()
            .any(|word| lower.contains(word));
        Plan {
            goal: prompt.chars().take(1000).collect(),
            success_criteria: vec![
                "Address the user's request with evidence and preserve its scope and permissions"
                    .into(),
            ],
            needs_context: profiles::project_evidence_relevant(prompt),
            stages: vec![Stage {
                model: if build && profiles::project_evidence_relevant(prompt) {
                    MODEL_ATOM
                } else if creative {
                    MODEL_NOVA
                } else if profiles::project_evidence_relevant(prompt) {
                    MODEL_NEON
                } else {
                    MODEL_ECHO
                }
                .into(),
                task: prompt.chars().take(1000).collect(),
            }],
        }
    });
    let project_evidence = profiles::project_evidence_relevant(prompt);
    let inspection = profiles::inspection_request(prompt);
    // An inspection request ("check the nio-guru folder") must READ the project
    // before any answer exists. An answering profile placed first would answer
    // from memory, and its draft then primes every later stage, so a read-tool
    // stage is moved (or inserted) to the front.
    if inspection {
        match plan
            .stages
            .iter()
            .position(|stage| READ_TOOL_STAGES.contains(&stage.model.as_str()))
        {
            Some(index) if index > 0 => {
                let read_stage = plan.stages.remove(index);
                plan.stages.insert(0, read_stage);
            }
            None => plan.stages.insert(
                0,
                Stage {
                    model: MODEL_NANO.into(),
                    task: "Gather a bounded caller-project listing as evidence for this request"
                        .into(),
                },
            ),
            _ => {}
        }
    }
    plan.needs_context = (plan.needs_context && project_evidence) || inspection;
    let mut seen = std::collections::HashSet::new();
    plan.stages.retain(|stage| seen.insert(stage.model.clone()));
    if plan.stages.iter().any(|stage| stage.model == MODEL_ATOM)
        && !plan.stages.iter().any(|stage| stage.model == MODEL_NEON)
    {
        plan.stages.push(Stage {
            model: MODEL_NEON.into(),
            task: "Review the implementation draft, actual edits, and check results".into(),
        });
    }
    plan.stages = plan
        .stages
        .into_iter()
        .flat_map(|stage| {
            if stage.model == MODEL_THINKER {
                thinker_stages(&stage.task)
            } else {
                vec![stage]
            }
        })
        .collect();
    plan
}

fn complexity_evidence(
    prompt: &str,
    plan: Option<&Plan>,
    validation_failures: usize,
    messages: &[Value],
) -> Vec<String> {
    let mut reasons = Vec::new();
    if let Some(plan) = plan {
        if plan
            .stages
            .iter()
            .map(|stage| &stage.model)
            .collect::<std::collections::HashSet<_>>()
            .len()
            >= 3
        {
            reasons.push("Three or more distinct workflow roles".into());
        }
        if plan.success_criteria.len() >= 4 {
            reasons.push("Four or more acceptance criteria".into());
        }
        if plan.stages.iter().any(|stage| stage.model == MODEL_ATOM)
            && plan.stages.iter().any(|stage| stage.model == MODEL_NEON)
        {
            reasons.push("Implementation requires subsequent code review".into());
        }
    }
    let lower = prompt.to_ascii_lowercase();
    for (label, terms) in [
        (
            "Security or privacy constraints",
            &["security", "privacy", "authentication", "authorization"] as &[&str],
        ),
        (
            "Concurrency or distributed coordination",
            &["concurrency", "concurrent", "distributed", "race condition"],
        ),
        (
            "Architecture or migration constraints",
            &[
                "architecture",
                "migration",
                "backward compat",
                "cross-platform",
            ],
        ),
    ] {
        if terms.iter().any(|term| lower.contains(term)) {
            reasons.push(label.into());
        }
    }
    if validation_failures >= 2 {
        reasons.extend([
            "Two rejected tool proposals".into(),
            "Repeated correction required".into(),
        ]);
    }
    if messages.iter().any(|message| {
        message["role"] == "tool"
            && message["content"]
                .as_str()
                .and_then(|content| serde_json::from_str::<Value>(content).ok())
                .is_some_and(|result| result["exit_code"].as_i64().is_some_and(|code| code != 0))
    }) {
        reasons.extend([
            "Caller reported a failed command check".into(),
            "Repair must account for actual failure evidence".into(),
        ]);
    }
    reasons
}

fn thinker_stages(task: &str) -> Vec<Stage> {
    [
        ("Draft", "Develop a candidate answer and identify assumptions and missing evidence"),
        ("Challenge", "Check the candidate against evidence, counterexamples, and the strongest objection; do not treat the draft as verified"),
        ("Refine", "Resolve supported objections and deliver the best supported answer with remaining uncertainty; no tools or additional phases"),
    ].into_iter().map(|(phase, instruction)| Stage {
        model: MODEL_THINKER.into(),
        task: format!("{phase}: {instruction}. Assigned question: {task}"),
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn answer(content: &str) -> Value {
        json!({"choices":[{"message":{"role":"assistant","content":content}}]})
    }
    fn call(name: &str, arguments: Value) -> Value {
        json!({"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"upstream_reused","type":"function","function":{"name":name,"arguments":arguments.to_string()}}]}}]})
    }
    fn plan(context: bool, stages: &[&str]) -> Value {
        answer(&json!({"goal":"Address the task","success_criteria":["Use actual evidence"],"needs_context":context,"stages":stages.iter().map(|model| json!({"model":model,"task":format!("Perform {model} work")})).collect::<Vec<_>>()}).to_string())
    }
    async fn gateway(replies: Vec<Value>) -> (String, tokio::task::JoinHandle<Vec<Value>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let mut requests = vec![];
            for reply in replies {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut bytes = vec![];
                let mut chunk = [0; 4096];
                let (body, length) = loop {
                    let read = socket.read(&mut chunk).await.unwrap();
                    assert!(read > 0);
                    bytes.extend_from_slice(&chunk[..read]);
                    if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&bytes[..end]);
                        let length: usize = headers
                            .lines()
                            .find_map(|line| {
                                let (name, value) = line.split_once(':')?;
                                if name.eq_ignore_ascii_case("content-length") {
                                    value.trim().parse().ok()
                                } else {
                                    None
                                }
                            })
                            .unwrap();
                        break (end + 4, length);
                    }
                };
                while bytes.len() < body + length {
                    let read = socket.read(&mut chunk).await.unwrap();
                    assert!(read > 0);
                    bytes.extend_from_slice(&chunk[..read]);
                }
                requests.push(serde_json::from_slice(&bytes[body..body + length]).unwrap());
                let body = reply.to_string();
                socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
            }
            requests
        });
        (base, server)
    }
    fn request(stream: bool) -> ChatRequest {
        serde_json::from_value(json!({"model":"nio-flow","stream":stream,"messages":[{"role":"system","content":"Persona & Guidelines (always active):\nIdentity: Your name is Morgan."},{"role":"user","content":"--helm Suggest and implement a dashboard"}],"tools":companion::coding_tool_schemas()})).unwrap()
    }
    fn request_default(stream: bool) -> ChatRequest {
        serde_json::from_value(json!({"model":"nio-flow","stream":stream,"messages":[{"role":"system","content":"Persona & Guidelines (always active):\nIdentity: Your name is Morgan."},{"role":"user","content":"Suggest and implement a dashboard"}],"tools":companion::coding_tool_schemas()})).unwrap()
    }
    #[tokio::test]
    async fn empty_final_answer_is_retryable_without_replanning_or_replaying_tools() {
        let (base, server) = gateway(vec![
            plan(true, &[MODEL_ECHO]),
            answer(""),
            answer("Recovered final answer"),
        ])
        .await;
        let mut request = request_default(false);
        request.messages.last_mut().unwrap().content = json!("--no-helm inspect this project");
        let first = run(&base, &request, "turn_empty_final_retry").await;
        assert_eq!(first["choices"][0]["finish_reason"], "tool_calls");
        replay(&mut request, &first, r#"{"files":["README.md"]}"#);
        let failed = run(&base, &request, "turn_empty_final_retry").await;
        assert_eq!(failed["nioxs"]["flow"]["status"], "failed");
        assert_eq!(failed["nioxs"]["flow"]["retryable"], true);
        assert_eq!(failed["nioxs"]["flow"]["helm_enabled"], false);
        let recovered = run(&base, &request, "turn_empty_final_retry").await;
        assert!(
            recovered["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .contains("Recovered final answer")
        );
        assert!(
            recovered["choices"][0]["message"]
                .get("tool_calls")
                .is_none()
        );
        assert_eq!(recovered["nioxs"]["flow"]["helm_enabled"], false);
        let requests = tokio::time::timeout(std::time::Duration::from_secs(3), server)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            requests.len(),
            3,
            "plan once, then retry only the failed final stage"
        );
    }
    async fn run(base: &str, request: &ChatRequest, turn: &str) -> Value {
        let db = NioDb::new("http://127.0.0.1:9", "test".into()).unwrap();
        run_flow(&db, base, "flow-tests", request, turn).await
    }
    async fn run_flow(
        db: &NioDb,
        base: &str,
        instance: &str,
        request: &ChatRequest,
        turn: &str,
    ) -> Value {
        let consult = CodingConsult {
            config: companion::CodingConfig {
                base_url: base.into(),
                api_key: None,
                model: "mock/model".into(),
                mode: "ask".into(),
                auto_approve: false,
                automatic_model: false,
            },
            prompt: "Suggest and implement a dashboard".into(),
            messages: profiles::messages(
                request.model.as_str(),
                companion::upstream_messages(
                    "Suggest and implement a dashboard",
                    &request.messages,
                ),
            ),
            tools: request.tools.clone(),
            files: vec![],
            guarded: false,
        };
        let response = respond(consult, request, "chat-test".into(), json!({"model":request.model,"routing":{"requested_model":request.model,"selected_model":request.model},"progress":[],"status":{}}), TurnGuard { active:ActiveTurns::default(), id:turn.into() }, turn.into(), instance, db, "test", "stable-task").await;
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        if request.stream {
            let text = String::from_utf8(bytes.to_vec()).unwrap();
            let events: Vec<Value> = text
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter(|line| *line != "[DONE]")
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            json!({"events":events})
        } else {
            serde_json::from_slice(&bytes).unwrap()
        }
    }
    fn replay(request: &mut ChatRequest, response: &Value, result: &str) {
        request
            .messages
            .push(serde_json::from_value(response["choices"][0]["message"].clone()).unwrap());
        for call in response["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap()
        {
            request.messages.push(
                serde_json::from_value(
                    json!({"role":"tool","tool_call_id":call["id"],"content":result}),
                )
                .unwrap(),
            );
        }
    }

    fn session_snapshot() -> Session {
        Session {
            fingerprint: "f".into(),
            helm: true,
            plan: None,
            stage: 0,
            drafts: vec![],
            feedback: vec![],
            pending: vec![],
            stage_rounds: 0,
            total_rounds: 0,
            repairs: 0,
            validation_failures: 0,
            reasoning_escalated: false,
            atom_contract: None,
            neon_contract: None,
            touched: Instant::now(),
        }
    }

    #[test]
    fn persistable_session_bounds_the_durable_snapshot() {
        let value = persistable_session(&session_snapshot()).expect("a small session fits");
        assert!(value.get("truncated").is_none());

        // A large per-stage tool result is dropped before carried-forward work.
        let mut feedback = session_snapshot();
        feedback.feedback = vec![json!({"role":"tool","content":"x".repeat(20_000)})];
        let value = persistable_session(&feedback).expect("feedback dropped to fit");
        assert_eq!(value["feedback"].as_array().unwrap().len(), 0);
        assert!(value.get("truncated").is_none());

        // Drafts are trimmed only as far as the budget requires, and flagged.
        let mut drafts = session_snapshot();
        drafts.drafts = (0..3)
            .map(|i| json!({"role":"assistant","content":format!("{i}").repeat(8_000)}))
            .collect();
        let value = persistable_session(&drafts).expect("drafts trimmed to fit");
        assert_eq!(value["truncated"], json!(true));
        assert!(value["drafts"].as_array().unwrap().len() < 3);

        // When even an empty snapshot is too large, nothing is persisted.
        let mut oversized = session_snapshot();
        oversized.fingerprint = "f".repeat(20_000);
        assert!(persistable_session(&oversized).is_none());
    }

    #[tokio::test]
    async fn flow_handoff_survives_a_provider_restart_without_replaying_tools() {
        let (upstream, server) = gateway(vec![
            plan(false, &[MODEL_NEON]),
            call("read_file", json!({"path":"README.md"})),
            answer("Neon reviewed the actual README"),
            answer("Go based on the actual README"),
        ])
        .await;
        let (db_url, store) = crate::testdb::mock_niodb().await;
        let db = NioDb::new(&db_url, "test".into()).unwrap();
        let mut request = request(false);

        let first = run_flow(&db, &upstream, "flow-tests", &request, "turn_restart").await;
        assert_eq!(first["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            first["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "read_file"
        );
        assert_eq!(store.lock().await.len(), 1, "the handoff is stored durably");

        replay(&mut request, &first, "Actual README contents for the task");
        // A restarted provider has an empty in-memory map: the continuation must
        // resume from the durable handoff instead of replanning from scratch.
        let second = run_flow(&db, &upstream, "flow-restarted", &request, "turn_restart").await;
        assert_eq!(second["choices"][0]["finish_reason"], "stop");
        let content = second["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert!(
            content.contains("Go based on the actual README"),
            "resumed answer: {content}"
        );
        // plan + read + Neon + Helm: the restart resumed, it did not replan.
        assert_eq!(server.await.unwrap().len(), 4);
    }

    #[tokio::test]
    async fn flow_handoff_with_a_changed_fingerprint_is_refused_after_restart() {
        let (upstream, _server) = gateway(vec![
            plan(false, &[MODEL_NEON]),
            call("read_file", json!({"path":"README.md"})),
        ])
        .await;
        let (db_url, _store) = crate::testdb::mock_niodb().await;
        let db = NioDb::new(&db_url, "test".into()).unwrap();
        let mut request = request(false);
        let first = run_flow(&db, &upstream, "flow-tests", &request, "turn_changed").await;
        replay(&mut request, &first, "Actual README contents for the task");

        // Same turn and model, but the caller's task context changed.
        let db_restart = db.clone();
        let response = respond(
            CodingConsult {
                config: companion::CodingConfig {
                    base_url: upstream.clone(),
                    api_key: None,
                    model: "mock/model".into(),
                    mode: "ask".into(),
                    auto_approve: false,
                    automatic_model: false,
                },
                prompt: "Suggest and implement a dashboard".into(),
                messages: profiles::messages(
                    request.model.as_str(),
                    companion::upstream_messages(
                        "Suggest and implement a dashboard",
                        &request.messages,
                    ),
                ),
                tools: request.tools.clone(),
                files: vec![],
                guarded: false,
            },
            &request,
            "chat-test".into(),
            json!({}),
            TurnGuard {
                active: ActiveTurns::default(),
                id: "turn_changed".into(),
            },
            "turn_changed".into(),
            "flow-restarted",
            &db_restart,
            "test",
            "different-task",
        )
        .await;
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        let content = value["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert!(
            content.contains("task context changed"),
            "unexpected refusal: {content}"
        );
    }

    #[tokio::test]
    async fn folder_inspection_dispatches_a_read_tool_before_answering() {
        // The planner put an answering profile first; the inspection must still
        // read the project before any answer is produced.
        let (upstream, server) = gateway(vec![plan(false, &[MODEL_ECHO, MODEL_SCOUT])]).await;
        let db = NioDb::new("http://127.0.0.1:9", "test".into()).unwrap();
        let prompt = "I meant checking nio-guru folder pls.";
        let request: ChatRequest = serde_json::from_value(json!({
            "model": "nio-flow",
            "stream": false,
            "messages": [
                {"role":"system","content":"Identity: Your name is Morgan."},
                {"role":"user","content":prompt}
            ],
            "tools": companion::coding_tool_schemas()
        }))
        .unwrap();
        let consult = CodingConsult {
            config: companion::CodingConfig {
                base_url: upstream.clone(),
                api_key: None,
                model: "mock/model".into(),
                mode: "ask".into(),
                auto_approve: false,
                automatic_model: false,
            },
            prompt: prompt.into(),
            messages: profiles::messages(
                request.model.as_str(),
                companion::upstream_messages(prompt, &request.messages),
            ),
            tools: request.tools.clone(),
            files: vec![],
            guarded: false,
        };
        let response = respond(
            consult,
            &request,
            "chat-test".into(),
            json!({"model":request.model,"routing":{"requested_model":request.model,"selected_model":request.model},"progress":[],"status":{}}),
            TurnGuard { active: ActiveTurns::default(), id: "turn_inspect".into() },
            "turn_inspect".into(),
            "flow-inspect",
            &db,
            "test",
            "stable-task",
        )
        .await;
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        // The inspection is routed to a native read-tool stage instead of the
        // answering profile, so a call is dispatched before any answer exists.
        assert_eq!(value["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            value["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "find_files"
        );
        // The listing stage is native Rust: the plan was the only upstream call.
        assert_eq!(server.await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn flow_stage_pins_the_current_request_over_earlier_turns() {
        let (upstream, server) = gateway(vec![
            plan(false, &[MODEL_ECHO]),
            answer("Answer for the current request"),
            answer("Go based on the current request"),
        ])
        .await;
        let db = NioDb::new("http://127.0.0.1:9", "test".into()).unwrap();
        let request = request(false);
        let _ = run_flow(&db, &upstream, "flow-anchor", &request, "turn_anchor").await;
        let requests = server.await.unwrap();
        // requests[0] is the plan; the specialist stages follow.
        for index in [1, 2] {
            let messages = requests[index]["messages"].to_string();
            assert!(
                messages.contains("earlier user turns as context only"),
                "stage {index} lost the anchor: {messages}"
            );
            assert!(
                messages.contains("Suggest and implement a dashboard"),
                "stage {index} lost the request: {messages}"
            );
        }
    }

    #[tokio::test]
    async fn flow_strips_internal_stage_markers_from_the_final_answer() {
        let (upstream, _server) = gateway(vec![
            plan(false, &[MODEL_ECHO]),
            answer("Here is an answer"),
            answer("[nio-echo completed stage: Synthesize] \nReal final answer\n\n```json:helm\n{\"verdict\":\"go\",\"summary\":\"ok\",\"decisive_evidence\":[],\"unmet_criteria\":[],\"residual_risks\":[],\"reversibility\":\"high\"}\n```"),
        ])
        .await;
        let db = NioDb::new("http://127.0.0.1:9", "test".into()).unwrap();
        let request = request(false);
        let result = run_flow(&db, &upstream, "flow-strip", &request, "turn_strip").await;
        let content = result["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert!(
            !content.contains("completed stage:"),
            "marker leaked: {content}"
        );
        assert!(
            content.contains("Real final answer"),
            "answer lost: {content}"
        );
        assert_eq!(result["nioxs"]["flow"]["helm_verdict"]["verdict"], "go");
    }

    #[tokio::test]
    async fn helm_withholds_a_fabricated_scan_with_no_tool_results() {
        let fabricated = "Directory scan completed and returned comprehensive project structure.\n\n```json:helm\n{\"verdict\":\"no_go\",\"summary\":\"Cannot check nio-guru folder as it does not exist in the workspace\",\"decisive_evidence\":[\"File system listing shows only nio/ and nio-cm/ directories\",\"No nio-guru folder present in any location\"],\"unmet_criteria\":[\"Target folder existence confirmed missing\"],\"residual_risks\":[\"Potential confusion about project structure\"],\"reversibility\":\"high\"}\n```";
        let (upstream, _server) = gateway(vec![
            plan(false, &[MODEL_ECHO]),
            answer("Here is the assessment"),
            answer(fabricated),
        ])
        .await;
        let db = NioDb::new("http://127.0.0.1:9", "test".into()).unwrap();
        let request = request(false);
        let result = run_flow(&db, &upstream, "flow-guard", &request, "turn_guard").await;
        let content = result["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert!(
            content.contains("withheld"),
            "guard did not fire: {content}"
        );
        assert_eq!(result["nioxs"]["flow"]["helm_verdict"]["verdict"], "no_go");
        assert!(
            !result["nioxs"]["flow"]["helm_guard"]["withheld_claims"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn helm_keeps_a_verdict_backed_by_an_actual_listing() {
        let verdict = "The listing was reviewed.\n\n```json:helm\n{\"verdict\":\"go\",\"summary\":\"nio-guru contains the expected files\",\"decisive_evidence\":[\"find_files returned nio-guru/src/lib.rs\"],\"unmet_criteria\":[],\"residual_risks\":[],\"reversibility\":\"high\"}\n```";
        let (upstream, _server) = gateway(vec![
            plan(false, &[MODEL_NEON]),
            call("find_files", json!({"glob":"**/*"})),
            answer("Neon has the listing"),
            answer(verdict),
        ])
        .await;
        let db = NioDb::new("http://127.0.0.1:9", "test".into()).unwrap();
        let mut request = request(false);
        let first = run_flow(&db, &upstream, "flow-guard", &request, "turn_backed").await;
        assert_eq!(first["choices"][0]["finish_reason"], "tool_calls");
        replay(
            &mut request,
            &first,
            "nio-guru/src/lib.rs\naio-cm/src/lib.rs",
        );
        let second = run_flow(&db, &upstream, "flow-guard", &request, "turn_backed").await;
        let content = second["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert!(!content.contains("withheld"), "guard misfired: {content}");
        assert_eq!(second["nioxs"]["flow"]["helm_verdict"]["verdict"], "go");
    }

    #[test]
    fn a_named_folder_scopes_the_evidence_listing() {
        assert_eq!(
            named_folder("I meant checking nio-guru folder pls.").as_deref(),
            Some("nio-guru")
        );
        assert_eq!(
            named_folder("check the nio-cm directory").as_deref(),
            Some("nio-cm")
        );
        assert_eq!(named_folder("inspect nio-xs").as_deref(), Some("nio-xs"));
        // No folder named, or a bare word with no folder shape.
        assert_eq!(named_folder("summarize this project").as_deref(), None);
        assert_eq!(
            named_folder("what do you think of our project?").as_deref(),
            None
        );
        // A file path is not a folder request.
        assert_eq!(named_folder("read src/main.rs").as_deref(), None);
    }

    #[tokio::test]
    async fn a_flow_continuation_forwards_the_callers_tool_result_to_the_stage() {
        // The caller answered the relayed listing on the continuation. The stage
        // must see it, or the model re-asks and concludes the folder is missing.
        let (upstream, server) = gateway(vec![
            plan(true, &[MODEL_ECHO]),
            answer("Echo answered from the listing"),
        ])
        .await;
        let db = NioDb::new("http://127.0.0.1:9", "test".into()).unwrap();
        let prompt = "I meant checking nio-guru folder pls.";
        let mut request: ChatRequest = serde_json::from_value(json!({
            "model": "nio-flow",
            "stream": false,
            "messages": [
                {"role":"system","content":"Identity: Your name is Morgan."},
                {"role":"user","content":prompt}
            ],
            "tools": companion::coding_tool_schemas()
        }))
        .unwrap();
        let first = run_flow(&db, &upstream, "flow-tools", &request, "turn_tools").await;
        assert_eq!(
            first["choices"][0]["finish_reason"], "tool_calls",
            "the listing stage must dispatch a scan: {first}"
        );
        replay(
            &mut request,
            &first,
            "{\"files\":[\"nio-guru/README.md\"],\"next_offset\":null,\"scan_limited\":false}",
        );
        let second = run_flow(&db, &upstream, "flow-tools", &request, "turn_tools").await;
        assert_eq!(second["choices"][0]["finish_reason"], "stop");

        let requests = server.await.unwrap();
        assert_eq!(requests.len(), 2);
        // plan, then the stage that runs after the continuation.
        let stage = requests
            .iter()
            .skip(1)
            .find(|request| {
                request["messages"].as_array().is_some_and(|messages| {
                    messages.iter().any(|message| message["role"] == "tool")
                })
            })
            .expect("a stage after the continuation must carry the caller's tool result");
        let tool = stage["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|message| message["role"] == "tool")
            .unwrap();
        assert!(
            tool["content"]
                .as_str()
                .unwrap()
                .contains("nio-guru/README.md"),
            "the caller's listing must reach the stage: {tool}"
        );
    }

    #[tokio::test]
    async fn a_folder_request_lists_that_folder_instead_of_a_whole_project_sample() {
        let (upstream, server) = gateway(vec![plan(true, &[MODEL_ECHO])]).await;
        let db = NioDb::new("http://127.0.0.1:9", "test".into()).unwrap();
        let prompt = "I meant checking nio-guru folder pls.";
        let request: ChatRequest = serde_json::from_value(json!({
            "model": "nio-flow",
            "stream": false,
            "messages": [
                {"role":"system","content":"Identity: Your name is Taylor."},
                {"role":"user","content":prompt}
            ],
            "tools": companion::coding_tool_schemas()
        }))
        .unwrap();
        let consult = CodingConsult {
            config: companion::CodingConfig {
                base_url: upstream.clone(),
                api_key: None,
                model: "mock/model".into(),
                mode: "ask".into(),
                auto_approve: false,
                automatic_model: false,
            },
            prompt: prompt.into(),
            messages: profiles::messages(
                request.model.as_str(),
                companion::upstream_messages(prompt, &request.messages),
            ),
            tools: request.tools.clone(),
            files: vec![],
            guarded: false,
        };
        let response = respond(
            consult,
            &request,
            "chat-test".into(),
            json!({"model":request.model,"routing":{"requested_model":request.model,"selected_model":request.model},"progress":[],"status":{}}),
            TurnGuard { active: ActiveTurns::default(), id: "turn_scope".into() },
            "turn_scope".into(),
            "flow-scope",
            &db,
            "test",
            "stable-task",
        )
        .await;
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        let call = &value["choices"][0]["message"]["tool_calls"][0]["function"];
        assert_eq!(call["name"], "find_files");
        assert_eq!(
            serde_json::from_str::<Value>(call["arguments"].as_str().unwrap()).unwrap()["glob"],
            "nio-guru/**/*",
            "the listing must be scoped to the folder the caller named"
        );
        // One listing stage only: a duplicated scan wasted a caller round.
        let stages = value["nioxs"]["flow"]["stages"].as_array().unwrap();
        assert_eq!(
            stages
                .iter()
                .filter(|stage| stage["model"] == MODEL_NANO)
                .count(),
            1,
            "the listing stage must not be duplicated: {stages:?}"
        );
        // The plan was the only upstream call; the listing stage is native Rust.
        assert_eq!(server.await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn helm_withholds_a_missing_folder_claim_built_on_an_empty_listing() {
        let verdict = "nio-guru does not exist in this project.\n\n```json:helm\n{\"verdict\":\"no_go\",\"summary\":\"The nio-guru folder does not exist in the current project structure.\",\"decisive_evidence\":[\"Directory scan of the project root found only nio/ and nio-cm/ directories; no nio-guru folder present\"],\"unmet_criteria\":[\"Requested folder nio-guru was not found in the project\"],\"residual_risks\":[\"The folder may exist in a different project path or have been renamed\"],\"reversibility\":\"high\"}\n```";
        let (upstream, _server) = gateway(vec![
            plan(false, &[MODEL_NEON]),
            call("find_files", json!({"glob":"**/*"})),
            answer("Neon has the listing"),
            answer(verdict),
        ])
        .await;
        let db = NioDb::new("http://127.0.0.1:9", "test".into()).unwrap();
        let mut request = request(false);
        let first = run_flow(&db, &upstream, "flow-absent", &request, "turn_absent").await;
        // The caller's listing matched no files, which is not proof of absence.
        replay(
            &mut request,
            &first,
            "{\"files\":[],\"next_offset\":null,\"scan_limited\":false}",
        );
        let second = run_flow(&db, &upstream, "flow-absent", &request, "turn_absent").await;
        let content = second["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert!(
            content.contains("withheld"),
            "an empty listing must not support a missing-folder claim: {content}"
        );
        assert_eq!(second["nioxs"]["flow"]["helm_verdict"]["verdict"], "no_go");
        let withheld = second["nioxs"]["flow"]["helm_guard"]["withheld_claims"]
            .as_array()
            .unwrap()
            .clone();
        assert!(
            withheld.iter().any(|claim| claim
                .as_str()
                .is_some_and(|claim| claim.contains("does not exist"))),
            "the absence claim must be listed: {withheld:?}"
        );
    }

    #[test]
    fn concept_pairs_are_not_path_claims() {
        // A slash-joined pair of concepts is prose, not a filesystem claim.
        assert!(path_tokens("The README covers speech-to-text/text-to-speech.").is_empty());
        assert!(path_tokens("cost/benefit and read/write windows").is_empty());
        // Real path shapes are still claims.
        assert_eq!(
            path_tokens("Read src/lib.rs and nio-guru/apps/server"),
            vec!["src/lib.rs", "nio-guru/apps/server"]
        );
        assert_eq!(path_tokens("See ./README.md"), vec!["./readme.md"]);
    }

    #[tokio::test]
    async fn helm_keeps_a_verdict_that_names_concepts_with_a_slash() {
        let verdict = "nio-guru documents its features.\n\n```json:helm\n{\"verdict\":\"go\",\"summary\":\"nio-guru covers speech-to-text/text-to-speech dictation\",\"decisive_evidence\":[\"README.md documents speech-to-text/text-to-speech support\"],\"unmet_criteria\":[],\"residual_risks\":[],\"reversibility\":\"high\"}\n```";
        let (upstream, _server) = gateway(vec![
            plan(false, &[MODEL_NEON]),
            call("read_file", json!({"path":"nio-guru/README.md"})),
            answer("Neon read the README"),
            answer(verdict),
        ])
        .await;
        let db = NioDb::new("http://127.0.0.1:9", "test".into()).unwrap();
        let mut request = request(false);
        let first = run_flow(&db, &upstream, "flow-concepts", &request, "turn_concepts").await;
        assert_eq!(first["choices"][0]["finish_reason"], "tool_calls");
        // The real nio-guru/README.md text the model was working from: the two
        // concepts are separate bullets, so the joined pair is the model's own.
        replay(
            &mut request,
            &first,
            "nio-guru/README.md\n- **Speech-to-Text Voice Dictation (Composer Mic):** Dictate your prompts hands-free directly into the composer with live browser Speech Recognition.\n- **Text-to-Speech Message Reading (Assistant Speak):** Listen to Guru responses with one-click voice reading powered by SpeechSynthesis.",
        );
        let second = run_flow(&db, &upstream, "flow-concepts", &request, "turn_concepts").await;
        let content = second["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert!(
            !content.contains("withheld"),
            "a concept pair must not read as a path claim: {content}"
        );
        assert!(content.contains("speech-to-text/text-to-speech"));
        assert_eq!(second["nioxs"]["flow"]["helm_verdict"]["verdict"], "go");
    }

    #[tokio::test]
    async fn helm_withholds_a_fabricated_path_even_when_other_results_were_supplied() {
        let verdict = "The listing was reviewed.\n\n```json:helm\n{\"verdict\":\"go\",\"summary\":\"nio-guru contains the expected files\",\"decisive_evidence\":[\"The listing shows nio-guru/src/ghost.rs\"],\"unmet_criteria\":[],\"residual_risks\":[],\"reversibility\":\"high\"}\n```";
        let (upstream, _server) = gateway(vec![
            plan(false, &[MODEL_NEON]),
            call("find_files", json!({"glob":"**/*"})),
            answer("Neon has the listing"),
            answer(verdict),
        ])
        .await;
        let db = NioDb::new("http://127.0.0.1:9", "test".into()).unwrap();
        let mut request = request(false);
        let first = run_flow(&db, &upstream, "flow-ghost", &request, "turn_ghost").await;
        replay(&mut request, &first, "nio-guru/README.md\nnio-guru/PLAN.md");
        let second = run_flow(&db, &upstream, "flow-ghost", &request, "turn_ghost").await;
        let content = second["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert!(
            content.contains("withheld"),
            "guard did not fire: {content}"
        );
        let withheld = second["nioxs"]["flow"]["helm_guard"]["withheld_claims"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(
            withheld,
            vec![json!("path `nio-guru/src/ghost.rs` asserted as evidence")]
        );
        // With results supplied, the reason must not claim that none were.
        assert!(
            content.contains("tool result(s) were supplied this turn"),
            "misleading reason: {content}"
        );
        assert!(!content.contains("supplied no tool result at all"));
    }

    #[tokio::test]
    async fn thinker_carries_draft_objections_and_actual_evidence_across_tool_handoffs() {
        let (base, server) = gateway(vec![
            answer("Candidate: cache every read; assumption: files never change"),
            call("read_file", json!({"path":"README.md"})),
            answer("Objection: actual evidence says files can change; invalidate on edits"),
            answer("Use an invalidated read cache. Remaining uncertainty: concurrent edits"),
        ])
        .await;
        let mut request = request(false);
        request.model = MODEL_THINKER.into();
        request.tools.extend([
            json!({"type":"function","function":{"name":"patch_file"}}),
            json!({"type":"function","function":{"name":"run_command"}}),
        ]);
        let first = run(&base, &request, "turn_thinker_tools").await;
        assert_eq!(first["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            first["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "read_file"
        );
        replay(
            &mut request,
            &first,
            "Actual project evidence: files change after edits",
        );
        let final_result = run(&base, &request, "turn_thinker_tools").await;
        assert_eq!(final_result["model"], MODEL_THINKER);
        assert_eq!(final_result["nioxs"]["flow"]["completed_stages"], 3);
        assert_eq!(final_result["nioxs"]["flow"]["helm_enabled"], false);
        assert!(
            final_result["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .contains("Remaining uncertainty")
        );
        let requests = server.await.unwrap();
        assert_eq!(requests.len(), 4);
        for request in &requests[..3] {
            assert!(
                request["tools"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|tool| profiles::answer_tool(
                        MODEL_THINKER,
                        tool["function"]["name"].as_str().unwrap()
                    ))
            );
        }
        let final_messages = requests[3]["messages"].to_string();
        for evidence in [
            "Candidate:",
            "Objection:",
            "Actual project evidence:",
            "Morgan",
        ] {
            assert!(final_messages.contains(evidence), "missing {evidence}");
        }
        assert!(
            requests[3]
                .get("tools")
                .is_none_or(|tools| tools.as_array().unwrap().is_empty())
        );
    }

    #[tokio::test]
    async fn thinker_stream_reports_three_phases_without_leaking_drafts() {
        let (base, server) = gateway(vec![
            answer("PRIVATE_DRAFT"),
            answer("PRIVATE_CRITIQUE"),
            answer("Supported final answer"),
        ])
        .await;
        let mut request = request(true);
        request.model = MODEL_THINKER.into();
        let output = run(&base, &request, "turn_thinker_stream").await;
        let events = output["events"].as_array().unwrap();
        let text = events
            .iter()
            .filter_map(|event| event["choices"][0]["delta"]["content"].as_str())
            .collect::<String>();
        for phase in ["Drafting", "Challenging", "Refining"] {
            assert!(text.contains(phase));
        }
        assert!(text.contains("Supported final answer"));
        assert!(!text.contains("PRIVATE_DRAFT"));
        assert!(!text.contains("PRIVATE_CRITIQUE"));
        assert!(!text.contains("NioFlow"));
        assert_eq!(
            events
                .iter()
                .filter(|event| event["nioxs"]["status"]["phase"] == "thinking")
                .count(),
            3
        );
        assert_eq!(server.await.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn direct_thinker_runs_helm_only_after_explicit_user_opt_in() {
        let (base, server) = gateway(vec![
            answer("Candidate"),
            answer("Challenge"),
            answer("Refined answer"),
            answer("User-requested final verdict"),
        ])
        .await;
        let mut request = request(false);
        request.model = MODEL_THINKER.into();
        request.messages.last_mut().unwrap().content = json!("--with-helm check this reasoning");
        let output = run(&base, &request, "turn_thinker_optin_gate").await;
        assert_eq!(output["model"], MODEL_HELM);
        assert_eq!(output["nioxs"]["flow"]["helm_enabled"], true);
        assert_eq!(output["nioxs"]["flow"]["completed_stages"], 4);
        let requests = server.await.unwrap();
        assert_eq!(requests.len(), 4);
        for request in &requests[..3] {
            assert!(
                !request["messages"]
                    .to_string()
                    .contains("Synthesize a useful final answer")
            );
        }
        assert!(
            requests[3]["messages"]
                .to_string()
                .contains("Synthesize a useful final answer")
        );
    }

    #[tokio::test]
    async fn flow_owned_thinker_runs_three_phases_before_the_final_helm_gate() {
        let (base, server) = gateway(vec![
            plan(false, &[MODEL_THINKER]),
            answer("Candidate answer"),
            answer("Challenge to the candidate"),
            answer("Refined answer"),
            answer("Conditional Go: evidence is incomplete"),
        ])
        .await;
        let output = run(&base, &request(false), "turn_flow_thinker").await;
        assert_eq!(output["model"], MODEL_HELM);
        let stages = output["nioxs"]["flow"]["stages"].as_array().unwrap();
        assert_eq!(stages.len(), 4);
        assert!(
            stages[..3]
                .iter()
                .all(|stage| stage["model"] == MODEL_THINKER)
        );
        assert_eq!(stages[3]["model"], MODEL_HELM);
        let requests = server.await.unwrap();
        assert_eq!(requests.len(), 5);
        assert!(
            requests[4]["messages"]
                .to_string()
                .contains("Refined answer")
        );
    }

    #[tokio::test]
    async fn flow_can_handoff_to_each_new_answer_specialist() {
        for (index, model) in [MODEL_ECHO, MODEL_SAGE, MODEL_QUILL, MODEL_SCOUT]
            .into_iter()
            .enumerate()
        {
            let (base, server) = gateway(vec![
                plan(false, &[model]),
                answer("Specialist answer"),
                answer("Final evidence assessment"),
            ])
            .await;
            let output = run(
                &base,
                &request(false),
                &format!("turn_answer_specialist_{index}"),
            )
            .await;
            assert_eq!(output["model"], MODEL_HELM);
            assert_eq!(output["nioxs"]["flow"]["stages"][0]["model"], model);
            let requests = server.await.unwrap();
            assert!(
                !requests[1]["messages"]
                    .to_string()
                    .contains("Synthesize a useful final answer")
            );
            assert!(
                requests[2]["messages"]
                    .to_string()
                    .contains("Synthesize a useful final answer")
            );
            assert!(
                requests[1]["messages"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|message| message["content"].as_str().is_some_and(
                        |content| content.contains(profiles::instructions(model).unwrap())
                    ))
            );
        }
    }

    #[tokio::test]
    async fn invokes_nova_atom_neon_and_helm_with_carried_forward_evidence() {
        let (base, server) = gateway(vec![
            plan(false, &[MODEL_NOVA, MODEL_ATOM, MODEL_NEON]),
            answer("Nova's creative proposal"),
            answer("Atom's implementation draft; no edit tools offered"),
            answer("Neon's evidence review"),
            answer("Conditional Go: proposal is ready; implementation is unverified"),
        ])
        .await;
        let output = run(&base, &request(true), "turn_all_specialists").await;
        let requests = server.await.unwrap();
        assert_eq!(requests.len(), 5);
        for request in &requests[..4] {
            assert!(
                !request["messages"]
                    .to_string()
                    .contains("Synthesize a useful final answer")
            );
        }
        assert!(
            requests[4]["messages"]
                .to_string()
                .contains("Synthesize a useful final answer")
        );
        for (request, model) in requests
            .iter()
            .zip([MODEL_FLOW, MODEL_NOVA, MODEL_ATOM, MODEL_NEON, MODEL_HELM])
        {
            assert!(
                request["messages"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|message| message["content"]
                        .as_str()
                        .is_some_and(|text| text.contains(profiles::instructions(model).unwrap()))),
                "missing real {model} invocation"
            );
            assert!(request["messages"].to_string().contains("Morgan"));
        }
        let final_messages = requests[4]["messages"].to_string();
        assert!(final_messages.contains("Nova's creative proposal"));
        assert!(final_messages.contains("Atom's implementation draft"));
        assert!(final_messages.contains("Neon's evidence review"));
        assert!(
            requests[4]["tools"].is_null(),
            "Helm cannot start another tool stage"
        );
        let labels: Vec<&str> = output["events"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|event| event["choices"][0]["delta"]["content"].as_str())
            .filter(|text| companion::stage_model(text).is_some())
            .collect();
        assert_eq!(
            labels,
            vec![
                "🌊 NioFlow · Planning\n",
                "↪ NioNova · Exploring\n",
                "↪ NioAtom · Building\n",
                "↪ NioNeon · Reviewing\n",
                "⚖ NioHelm · Go / No-Go\n"
            ]
        );
        let final_event = output["events"].as_array().unwrap().last().unwrap();
        assert_eq!(
            final_event["nioxs"]["routing"]["requested_model"],
            MODEL_FLOW
        );
        assert_eq!(
            final_event["nioxs"]["routing"]["selected_model"],
            MODEL_HELM
        );
        assert_eq!(
            final_event["nioxs"]["flow"]["helm_verdict"]["verdict"],
            "conditional_go"
        );
        assert_eq!(
            final_event["nioxs"]["flow"]["helm_verdict"]["summary"],
            "proposal is ready; implementation is unverified"
        );
        assert_eq!(
            final_event["nioxs"]["flow"]["atom_contract"]["patch_status"],
            "patch_only"
        );
        assert_eq!(
            final_event["nioxs"]["flow"]["neon_contract"]["verdict"],
            "comment"
        );
    }

    #[tokio::test]
    async fn helm_attaches_structured_json_verdict_to_flow_metadata() {
        let helm_answer = r#"
Here is the synthesized final assessment.

```json:helm
{
  "verdict": "go",
  "summary": "Implementation verified and all tests pass",
  "decisive_evidence": ["cargo test passed with exit code 0"],
  "unmet_criteria": [],
  "residual_risks": [],
  "reversibility": "high"
}
```
"#;
        let (base, _server) = gateway(vec![
            plan(false, &[MODEL_NOVA]),
            answer("Nova design"),
            answer(helm_answer),
        ])
        .await;
        let output = run(&base, &request(true), "turn_structured_helm").await;
        let final_event = output["events"].as_array().unwrap().last().unwrap();
        assert_eq!(
            final_event["nioxs"]["flow"]["helm_verdict"]["verdict"],
            "go"
        );
        assert_eq!(
            final_event["nioxs"]["flow"]["helm_verdict"]["summary"],
            "Implementation verified and all tests pass"
        );
        assert_eq!(
            final_event["nioxs"]["flow"]["helm_verdict"]["decisive_evidence"][0],
            "cargo test passed with exit code 0"
        );
        assert_eq!(
            final_event["nioxs"]["flow"]["helm_verdict"]["reversibility"],
            "high"
        );
    }

    #[tokio::test]
    async fn atom_and_neon_structured_contracts_are_recorded_in_metadata() {
        let atom_answer = r#"
Implementation complete.
```json:atom
{
  "summary": "Added connection retry logic",
  "patch_status": "applied_cleanly",
  "modified_files": ["src/client.rs"],
  "checks_executed": [
    {
      "command": "cargo test test_retry",
      "exit_code": 0,
      "passed": true,
      "output_summary": "ok"
    }
  ],
  "unverified_items": []
}
```
"#;
        let neon_answer = r#"
Review complete.
```json:neon
{
  "verdict": "approve",
  "summary": "Retry logic is safe and bounded",
  "verification_passed": true,
  "findings": []
}
```
"#;
        let (base, _server) = gateway(vec![
            plan(false, &[MODEL_ATOM, MODEL_NEON]),
            answer(atom_answer),
            answer(neon_answer),
            answer("Go: verified and approved"),
        ])
        .await;
        let output = run(&base, &request(true), "turn_atom_neon_contracts").await;
        let final_event = output["events"].as_array().unwrap().last().unwrap();
        assert_eq!(
            final_event["nioxs"]["flow"]["atom_contract"]["summary"],
            "Added connection retry logic"
        );
        assert_eq!(
            final_event["nioxs"]["flow"]["atom_contract"]["patch_status"],
            "applied_cleanly"
        );
        assert_eq!(
            final_event["nioxs"]["flow"]["atom_contract"]["modified_files"][0],
            "src/client.rs"
        );
        assert_eq!(
            final_event["nioxs"]["flow"]["neon_contract"]["verdict"],
            "approve"
        );
        assert_eq!(
            final_event["nioxs"]["flow"]["neon_contract"]["verification_passed"],
            true
        );
    }

    #[tokio::test]
    async fn no_helm_persists_across_nano_continuation_and_last_specialist_answers() {
        let (base, server) = gateway(vec![
            plan(true, &[MODEL_NOVA, MODEL_NEON]),
            answer("Nova's design"),
            answer("Final recommendation: one Pages site, reviewed by Neon"),
        ])
        .await;
        let mut request = request(false);
        request.messages.last_mut().unwrap().content = json!("--no-helm suggest a site");
        let first = run(&base, &request, "turn_no_helm").await;
        assert!(
            first["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .contains("Helm skipped (--no-helm)")
        );
        assert_eq!(first["nioxs"]["flow"]["helm_enabled"], false);
        replay(&mut request, &first, "Actual project listing");
        let last = run(&base, &request, "turn_no_helm").await;
        assert_eq!(last["model"], MODEL_NEON);
        assert_eq!(last["choices"][0]["finish_reason"], "stop");
        assert_eq!(last["nioxs"]["flow"]["helm_enabled"], false);
        let requests = server.await.unwrap();
        assert_eq!(requests.len(), 3, "Helm must not be invoked");
        assert!(
            requests[2]["messages"]
                .to_string()
                .contains("The user selected --no-helm")
        );
        assert!(!requests.iter().any(|request| {
            request["messages"]
                .as_array()
                .unwrap()
                .iter()
                .any(|message| {
                    message["content"].as_str().is_some_and(|content| {
                        content.contains(profiles::instructions(MODEL_HELM).unwrap())
                    })
                })
        }));
    }

    #[tokio::test]
    async fn nano_evidence_handoff_resumes_without_replanning() {
        let (base, server) = gateway(vec![
            plan(true, &[MODEL_NOVA]),
            answer("Nova used the real project listing"),
            answer("Go: use one Pages site"),
        ])
        .await;
        let mut request = request(false);
        let first = run(&base, &request, "turn_nano_resume").await;
        assert_eq!(first["model"], MODEL_NANO);
        assert_eq!(
            first["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "find_files"
        );
        let text = first["choices"][0]["message"]["content"].as_str().unwrap();
        assert_eq!(text.matches("NioFlow · Planning").count(), 1);
        replay(&mut request, &first, "nio/README.md\nnio-xs/README.md");
        let second = run(&base, &request, "turn_nano_resume").await;
        assert_eq!(second["model"], MODEL_HELM);
        assert_eq!(second["choices"][0]["finish_reason"], "stop");
        let requests = server.await.unwrap();
        assert_eq!(
            requests.len(),
            3,
            "do not re-run the planner after tool results"
        );
        assert!(
            requests[1]["messages"]
                .to_string()
                .contains("nio-xs/README.md")
        );
        assert!(
            !second["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .contains("NioFlow · Planning")
        );
    }
    #[tokio::test]
    async fn malformed_search_is_repaired_before_any_client_execution() {
        let (base, server) = gateway(vec![
            plan(false, &[MODEL_NEON]),
            call("search_code", json!({"glob":"**/package.json"})),
            call(
                "search_code",
                json!({"query":"name","glob":"**/package.json"}),
            ),
        ])
        .await;
        let response = run(&base, &request(false), "turn_repair_query").await;
        assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
        let calls = response["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(calls.len(), 1);
        let args: Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["query"], "name");
        let requests = server.await.unwrap();
        assert!(
            requests[2]["messages"]
                .to_string()
                .contains("missing required argument")
        );
    }
    #[tokio::test]
    async fn atom_relays_edits_and_checks_but_review_and_gate_cannot_mutate() {
        let (base, server) = gateway(vec![
            plan(false, &[MODEL_ATOM]),
            call(
                "patch_file",
                json!({"path":"src/main.rs","old_content":"old","new_content":"new"}),
            ),
            call("run_command", json!({"command":"cargo check"})),
            answer("Atom finished the patch and caller checks"),
            answer("Neon reviewed the actual patch and check results"),
            answer("Go: verified by caller tool results"),
        ])
        .await;
        let mut request = request(false);
        request.tools.extend([
            json!({"type":"function","function":{"name":"patch_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"old_content":{"type":"string"},"new_content":{"type":"string"}},"required":["path","old_content","new_content"]}}}),
            json!({"type":"function","function":{"name":"run_command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}),
        ]);
        let first = run(&base, &request, "turn_atom_edit").await;
        assert_eq!(
            first["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "patch_file"
        );
        replay(
            &mut request,
            &first,
            "Edited src/main.rs using the caller's approved patch",
        );
        let second = run(&base, &request, "turn_atom_edit").await;
        assert_eq!(
            second["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "run_command"
        );
        replay(&mut request, &second, "cargo check passed; exit code 0");
        let final_result = run(&base, &request, "turn_atom_edit").await;
        assert_eq!(final_result["model"], MODEL_HELM);
        let requests = server.await.unwrap();
        assert_eq!(requests.len(), 6);
        assert!(
            requests[1]["tools"]
                .as_array()
                .unwrap()
                .iter()
                .any(|tool| tool["function"]["name"] == "patch_file")
        );
        assert!(
            !requests[4]["tools"]
                .as_array()
                .unwrap()
                .iter()
                .any(|tool| matches!(
                    tool["function"]["name"].as_str(),
                    Some("patch_file" | "run_command")
                ))
        );
        assert!(requests[5]["tools"].is_null());
        assert!(
            requests[5]["messages"]
                .to_string()
                .contains("cargo check passed; exit code 0")
        );
    }

    #[test]
    fn cache_is_invalidated_after_a_mutation_and_does_not_reuse_prior_tasks() {
        let call = ClientCall {
            id: "new".into(),
            name: "read_file".into(),
            arguments: "{\"path\":\"README.md\"}".into(),
            hint: String::new(),
        };
        let mut messages = vec![
            json!({"role":"user","content":"first task"}),
            json!({"role":"assistant","tool_calls":[{"id":"old","function":{"name":"read_file","arguments":call.arguments}}]}),
            json!({"role":"tool","tool_call_id":"old","content":"old README"}),
        ];
        assert_eq!(
            cached_result(&call, &messages).as_deref(),
            Some("old README")
        );
        messages.push(json!({"role":"assistant","tool_calls":[{"id":"write","function":{"name":"patch_file","arguments":"{}"}}]}));
        assert!(cached_result(&call, &messages).is_none());
        messages.pop();
        messages.push(json!({"role":"user","content":"new task: refresh the README"}));
        assert!(cached_result(&call, &messages).is_none());
    }

    #[tokio::test]
    async fn repeated_reads_reuse_results_without_another_client_read() {
        let read = call("read_file", json!({"path":"README.md"}));
        let (base, server) = gateway(vec![
            plan(false, &[MODEL_NEON]),
            read.clone(),
            read,
            answer("Neon has enough context"),
            answer("Go based on the actual README"),
        ])
        .await;
        let mut request = request(false);
        let first = run(&base, &request, "turn_cached_read").await;
        replay(&mut request, &first, "Actual README contents for the task");
        let second = run(&base, &request, "turn_cached_read").await;
        assert_eq!(second["choices"][0]["finish_reason"], "stop");
        assert!(second["choices"][0]["message"]["tool_calls"].is_null());
        let requests = server.await.unwrap();
        assert!(
            requests[3]["messages"]
                .as_array()
                .unwrap()
                .iter()
                .any(|message| message["role"] == "tool"
                    && message["content"] == "Actual README contents for the task")
        );
    }

    #[tokio::test]
    async fn flow_defaults_to_running_without_helm() {
        let (base, server) = gateway(vec![
            plan(false, &[MODEL_NOVA, MODEL_NEON]),
            answer("Nova's plan"),
            answer("Neon's final review and answer"),
        ])
        .await;
        let request = request_default(false);
        let result = run(&base, &request, "turn_default_no_helm").await;
        assert_eq!(result["nioxs"]["flow"]["helm_enabled"], false);
        assert_eq!(result["model"], MODEL_NEON);
        assert_eq!(result["choices"][0]["finish_reason"], "stop");
        let requests = server.await.unwrap();
        assert_eq!(
            requests.len(),
            3,
            "Helm must not be invoked under default settings"
        );
    }
}
