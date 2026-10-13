//! Bounded NioFlow orchestration. Specialists share the configured upstream;
//! each stage is a separate call with its own role and carried-forward evidence.
use super::*;
use nioxs_contracts::{
    MODEL_ATOM, MODEL_ECHO, MODEL_FLOW, MODEL_HELM, MODEL_NANO, MODEL_NEON, MODEL_NOVA,
    MODEL_QUILL, MODEL_SAGE, MODEL_SCOUT, MODEL_THINKER,
};
use nioxs_nano::profiles;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use std::time::Instant;

const MAX_TOOL_ROUNDS: usize = 16;
const MAX_INTERNAL_ROUNDS: usize = 28;
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
#[derive(Clone)]
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
    touched: Instant,
}
static SESSIONS: OnceLock<Mutex<HashMap<String, Session>>> = OnceLock::new();
fn sessions() -> &'static Mutex<HashMap<String, Session>> {
    SESSIONS.get_or_init(|| Mutex::new(HashMap::new()))
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
}

pub(super) async fn respond(
    consult: CodingConsult,
    request: &ChatRequest,
    id: String,
    metadata: Value,
    guard: TurnGuard,
    turn: String,
    instance: &str,
    fingerprint: &str,
) -> Response {
    let key = format!("{instance}:{turn}:{}", request.model);
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
        sessions.retain(|_, session| session.touched.elapsed() < Duration::from_secs(600));
        sessions.get(&key).cloned()
    });
    let mut session = match saved {
        Some(session) if session.fingerprint == fingerprint => session,
        Some(_) => {
            return coding_json(
                &id,
                json!({"role":"assistant","content":"⚠️ NioFlow's task context changed during a handoff; no further tools were dispatched. Start a fresh task with the current evidence."}),
                "stop",
                metadata,
            );
        }
        None if continuing => {
            return coding_json(
                &id,
                json!({"role":"assistant","content":"⚠️ NioFlow's handoff state expired or the provider restarted. No tool was replayed. Start a fresh task using the results already collected."}),
                "stop",
                metadata,
            );
        }
        None => Session {
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
            touched: Instant::now(),
        },
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
                    runner.fail(&error);
                }
            }
        });
        Sse::new(output).into_response()
    } else {
        while !runner.done {
            if let Err(error) = runner.advance().await {
                runner.fail(&error);
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
        messages.push(json!({"role":"system","content":format!("NioFlow assigned this bounded stage: {}. Original goal: {}. Success criteria: {}. Reuse all supplied evidence. Complete only your assigned work, preserve the caller persona and permissions, and do not claim another role ran. If a tool is unavailable or its budget is exhausted, finish with the limitation. Do not call another specialist.", stage.task, self.session.plan.as_ref().map_or(self.consult.prompt.as_str(), |plan| plan.goal.as_str()), self.session.plan.as_ref().map_or(String::new(), |plan| plan.success_criteria.join("; ")))}));
        if stage.model == MODEL_HELM {
            messages.push(json!({"role":"system","content":"Synthesize a useful final answer to the original user request from the specialist drafts, then give a concise Go / Conditional Go / No-Go grounded in actual results. Do not call another specialist or ask for tools."}));
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
            if plan.needs_context
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
        self.metadata["flow"] = json!({"goal":self.session.plan.as_ref().unwrap().goal,"stages":self.session.plan.as_ref().unwrap().stages,"stage_index":self.session.stage,"completed_stages":self.session.stage,"helm_enabled":self.session.helm});
        if stage.model == MODEL_NANO {
            let native: ChatRequest = serde_json::from_value(json!({"model":MODEL_NANO,"messages":[{"role":"user","content":"--find-files **/*"}],"tools":self.offered.iter().filter(|tool| tool["function"]["name"] == "find_files").collect::<Vec<_>>(),"tool_choice":"auto"})).map_err(|_| "cannot construct Nano evidence request")?;
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
            return self.handoff_tools(relayed);
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
                        return self.handoff_tools(fresh);
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
        if self.is_final() {
            if content.trim().is_empty() {
                return Err("The final stage returned no answer; completion is unverified".into());
            }
            self.finish(content);
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
    fn handoff_tools(&mut self, calls: Vec<ClientCall>) -> Result<(), String> {
        self.session.pending = calls.iter().map(|call| call.id.clone()).collect();
        self.session.total_rounds += 1;
        self.session.stage_rounds += 1;
        self.session.touched = Instant::now();
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
        drop(stored);
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
        self.done = true;
    }
    fn fail(&mut self, error: &str) {
        self.metadata["outcome"] = json!("capability_limitation");
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
            sessions.remove(&self.key);
        }
        self.done = true;
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
    plan.needs_context &= profiles::project_evidence_relevant(prompt);
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
        serde_json::from_value(json!({"model":"nio-flow","stream":stream,"messages":[{"role":"system","content":"Persona & Guidelines (always active):\nIdentity: Your name is Morgan."},{"role":"user","content":"Suggest and implement a dashboard"}],"tools":companion::coding_tool_schemas()})).unwrap()
    }
    async fn run(base: &str, request: &ChatRequest, turn: &str) -> Value {
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
        let response = respond(consult, request, "chat-test".into(), json!({"model":request.model,"routing":{"requested_model":request.model,"selected_model":request.model},"progress":[],"status":{}}), TurnGuard { active:ActiveTurns::default(), id:turn.into() }, turn.into(), "flow-tests", "stable-task").await;
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
}
