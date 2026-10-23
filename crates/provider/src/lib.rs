use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use futures_util::{StreamExt, stream};
use nioxs_contracts::{
    ChatRequest, DecideRequest, DurableEvent, MAX_REQUEST_BYTES, MODEL, MODELS, Outcome, PROVIDER,
    SCHEMA_VERSION, ToolCall,
};
use nioxs_nano::companion::{self, CodingConsult};
use nioxs_nano::{Model, decide_with_model};
mod flow;
pub mod journal;
pub mod mcp;
pub mod niodb_supervisor;
mod research;
#[cfg(test)]
mod testdb;
use nioxs_storage::NioDb;
use serde_json::{Value, json};
use std::{
    collections::{HashMap, VecDeque},
    convert::Infallible,
    sync::{Arc, Mutex},
    time::Duration,
};

#[derive(Clone, Default)]
pub struct ActiveTurns(Arc<Mutex<HashMap<String, tokio::sync::watch::Sender<bool>>>>);

struct TurnGuard {
    active: ActiveTurns,
    id: String,
}
impl Drop for TurnGuard {
    fn drop(&mut self) {
        if let Ok(mut active) = self.active.0.lock() {
            active.remove(&self.id);
        }
    }
}

#[derive(Clone)]
pub struct App {
    pub db: NioDb,
    pub deployment: String,
    pub api_token: Option<String>,
    pub active: ActiveTurns,
    pub model: Arc<Model>,
    pub ray_engine: Option<nioxs_engine::RayEngine>,
    pub instance: String,
}

pub fn router(app: App) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/v1/models", get(models))
        .route("/v1/capabilities", get(capabilities))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/decide", post(decide))
        .route("/v1/turns/:turn_id/cancel", post(cancel))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BYTES))
        .with_state(Arc::new(app))
}

fn error(status: StatusCode, kind: &str, message: &str) -> Response {
    (
        status,
        Json(json!({"error":{"message":message,"type":kind,"code":kind,"param":null}})),
    )
        .into_response()
}

fn authorized(app: &App, headers: &HeaderMap) -> bool {
    app.api_token.as_ref().is_none_or(|token| {
        headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.strip_prefix("Bearer ") == Some(token.as_str()))
    })
}

async fn health() -> Json<Value> {
    Json(json!({"status":"ok","provider":PROVIDER,"version":env!("CARGO_PKG_VERSION")}))
}

async fn ready(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    if !authorized(&app, &headers) {
        return error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "provider authentication required",
        );
    }
    match app.db.ready().await {
        // The research cache and its quota are reported here so a caller can see
        // how full the bounded NioDB research collections are.
        Ok(_) => {
            let research = app.db.research_usage().await.unwrap_or(Value::Null);
            Json(json!({"status":"ready","storage":"ready","research":research,"nano":"trained_naive_bayes","model_version":app.model.model_version,"schema_version":SCHEMA_VERSION})).into_response()
        }
        Err(_) => error(
            StatusCode::SERVICE_UNAVAILABLE,
            "storage_unavailable",
            "NioDB is not ready",
        ),
    }
}

async fn models(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    if !authorized(&app, &headers) {
        return error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "provider authentication required",
        );
    }
    let names = [
        "NioCuda — Tactical Swarm Conductor & Orchestrator",
        "NioRay — Keystroke Predator (Streaming Reflex 24L, < 86 MB RAM)",
        "NioViper — Tactical Striker (Shell Reflex 8L, < 47 MB RAM)",
        "NioManta — Conversational & AST Companion (16L, < 110 MB RAM)",
        "NioNano — Intent Reflex & MicroLM (~0.5M, sub-0.1ms)",
        "NioMicro — Contextual Policy Decision Arbiter (5.0M / cluster)",
        "NioMask — Byte-Stream Privacy & Secret Redactor (1.1M, sub-0.2ms)",
        "NioMini — Predictive Defect & Quality Forecaster (7.1M, sub-5ms)",
    ];
    let data: Vec<Value> = MODELS.iter().zip(names).map(|(id, name)| json!({"id":id,"object":"model","created":0,"owned_by":PROVIDER,"name":name,"type":"language","free":true})).collect();
    Json(json!({"object":"list","data":data})).into_response()
}

async fn capabilities(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    if !authorized(&app, &headers) {
        return error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "provider authentication required",
        );
    }
    Json(json!({"schema_version":SCHEMA_VERSION,"provider":PROVIDER,"model":MODEL,"default_model":MODEL,"models":MODELS,"implementation":"trained_naive_bayes_and_typed_workflows","profiles":"shared_engine_with_rust_routing_budgets_and_guards","orchestration":"bounded_rust_flow_cascade","flow_handoff_state":"bounded_memory_32_tasks_10_minutes","reflection":{"model":"nio-thinker","phases":["draft","challenge","refine"],"read_only":true,"refine_tools":false},"legacy_aliases":[nioxs_contracts::MODEL_AUTO],"upstream_default":companion::DEFAULT_CODING_MODEL,"upstream_fallback":companion::FALLBACK_CODING_MODEL,"upstream_catalog_check":true,"model_version":app.model.model_version,"trained":true,"confidence_threshold":app.model.threshold,"workflows":nioxs_nano::SHIPPED_WORKFLOWS,"client_tools":["read_file","find_files","search_code","run_command","edit_file","web_fetch"],"tool_registry":nioxs_nano::tools::registry(),"workflow_registry":nioxs_nano::tools::workflow_registry(),"server_tools":[],"streaming":true,"automatic_fallback":true,"browser_execution":true,"mcp":true,"decision_contract":true,"storage":"niodb_immutable_action_journal"})).into_response()
}

async fn cancel(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    if !authorized(&app, &headers) {
        return error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "provider authentication required",
        );
    }
    let Ok(active) = app.active.0.lock() else {
        return error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "turn registry unavailable",
        );
    };
    match active.get(&id) {
        Some(signal) => {
            let _ = signal.send(true);
            Json(json!({"turn_id":id,"status":"cancellation_requested"})).into_response()
        }
        None => error(
            StatusCode::NOT_FOUND,
            "turn_not_active",
            "turn is not active; no action was undone",
        ),
    }
}

/// Advisory decision modeling: score candidate proposals and return the choice.
/// This endpoint never dispatches an action and never touches storage, so it is
/// safe to call before a turn exists.
///
/// `async` because the handler must be a future for axum; the call is pure CPU.
async fn decide(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    payload: Result<Json<DecideRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if !authorized(&app, &headers) {
        return error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "provider authentication required",
        );
    }
    let request = match payload {
        Ok(Json(request)) => request,
        Err(_) => {
            return error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "invalid or unsupported request fields",
            );
        }
    };
    if let Err(message) = request.validate() {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", &message);
    }
    Json(nioxs_nano::decision::decide(&app.model, &request)).into_response()
}

async fn microlm_chat_handler(
    model_name: &str,
    request: ChatRequest,
    preloaded: Option<&nioxs_engine::RayEngine>,
) -> Response {
    let engine = if model_name == "nio-ray" || model_name == "ray" {
        if let Some(eng) = preloaded {
            eng.clone()
        } else {
            match nioxs_engine::RayEngine::for_model(model_name) {
                Ok(engine) => engine,
                Err(e) => {
                    return error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "engine_unavailable",
                        &format!(
                            "In-process MicroLM engine for '{model_name}' could not be loaded: {e}. Run 'nioxs pull {model_name}'."
                        ),
                    );
                }
            }
        }
    } else {
        match nioxs_engine::RayEngine::for_model(model_name) {
            Ok(engine) => engine,
            Err(e) => {
                return error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "engine_unavailable",
                    &format!(
                        "In-process MicroLM engine for '{model_name}' could not be loaded: {e}. Run 'nioxs pull {model_name}'."
                    ),
                );
            }
        }
    };

    let mut prompt = String::new();
    let system_prompt = match model_name {
        "nio-manta" | "manta" => {
            "You are Nio, an intelligent AI coding assistant. Answer concisely and directly in 1-2 short sentences unless the user explicitly requests full code."
        }
        "nio-viper" | "viper" => {
            "You are Nio, a lightning-fast terminal coding assistant. Keep replies brief and focused."
        }
        "nio-cuda" | "cuda" => {
            "You are Nio, a concise terminal coding assistant. Keep replies focused and to the point."
        }
        _ => {
            "You are Nio, a lightweight AI coding assistant for the terminal. You are helpful, polite, and very concise. For greetings, give a warm 1-sentence greeting and ask what to code. Do not output unrequested code, ray tracing, or headers."
        }
    };

    let tools_xml = if !request.tools.is_empty() {
        let mut t_str = String::from("\n\n# Tools\n\nYou may call one or more functions to inspect the project or perform tasks.\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n");
        for t in &request.tools {
            if let Ok(serialized) = serde_json::to_string(t) {
                t_str.push_str(&serialized);
                t_str.push('\n');
            }
        }
        t_str.push_str("</tools>\n\nWhen you need a tool, output ONLY the tool call within <tool_call></tool_call> tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
        t_str
    } else {
        String::new()
    };

    let client_system = request
        .messages
        .iter()
        .filter(|m| m.role == "system")
        .filter_map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");

    let full_system = if client_system.is_empty() {
        format!("{system_prompt}{tools_xml}")
    } else {
        format!("{client_system}{tools_xml}")
    };

    prompt.push_str(&format!("<|im_start|>system\n{full_system}<|im_end|>\n"));
    for msg in &request.messages {
        if msg.role == "system" {
            continue;
        }
        let role = &msg.role;
        if role == "tool" {
            let content = msg.content.as_str().unwrap_or("");
            prompt.push_str(&format!("<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"));
        } else if role == "assistant" && !msg.tool_calls.is_empty() {
            let mut tool_calls_xml = String::new();
            for call in &msg.tool_calls {
                let name = &call.function.name;
                let args = &call.function.arguments;
                tool_calls_xml.push_str(&format!("<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>\n"));
            }
            let content = msg.content.as_str().unwrap_or("");
            if content.is_empty() {
                prompt.push_str(&format!("<|im_start|>assistant\n{tool_calls_xml}<|im_end|>\n"));
            } else {
                prompt.push_str(&format!("<|im_start|>assistant\n{content}\n{tool_calls_xml}<|im_end|>\n"));
            }
        } else {
            let content = msg.content.as_str().unwrap_or("");
            prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
        }
    }
    prompt.push_str("<|im_start|>assistant\n");

    let max_gen_tokens = request.max_tokens.unwrap_or(512).clamp(1, 4096);

    let id = format!(
        "chatcmpl-{}-{}",
        model_name.replace("nio-", ""),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let model = match model_name {
        "nio-manta" | "manta" => nioxs_contracts::MODEL_MANTA,
        "nio-viper" | "viper" => nioxs_contracts::MODEL_VIPER,
        "nio-cuda" | "cuda" => nioxs_contracts::MODEL_CUDA,
        _ => nioxs_contracts::MODEL_RAY,
    };

    if !request.stream {
        match engine.generate(prompt, max_gen_tokens, 0.7, 0.9).await {
            Ok(content) => {
                let tool_calls = extract_xml_tool_calls(&content);
                if !tool_calls.is_empty() {
                    let calls_json: Vec<Value> = tool_calls
                        .into_iter()
                        .enumerate()
                        .map(|(i, c)| {
                            let args_val = serde_json::from_str::<Value>(&c.arguments).unwrap_or(json!({}));
                            json!({
                                "index": i,
                                "id": c.id,
                                "type": "function",
                                "function": {
                                    "name": c.name,
                                    "arguments": args_val.to_string()
                                }
                            })
                        })
                        .collect();
                    Json(json!({
                        "id": id,
                        "object": "chat.completion",
                        "created": 0,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": Value::Null,
                                "tool_calls": calls_json
                            },
                            "finish_reason": "tool_calls"
                        }]
                    }))
                    .into_response()
                } else {
                    Json(json!({
                        "id": id,
                        "object": "chat.completion",
                        "created": 0,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop"
                        }]
                    }))
                    .into_response()
                }
            }
            Err(e) => error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference_error",
                &format!("MicroLM inference for '{model_name}' failed: {e}"),
            ),
        }
    } else {
        match engine.generate_stream(prompt, max_gen_tokens, 0.7, 0.9).await {
            Ok(rx) => {
                let id_clone = id.clone();
                let accumulated = Arc::new(tokio::sync::Mutex::new(String::new()));
                let acc_clone = accumulated.clone();
                let live = stream::unfold(rx, move |mut rx| {
                    let id_item = id_clone.clone();
                    let acc = acc_clone.clone();
                    async move {
                        match rx.recv().await {
                            Some(chunk) => {
                                acc.lock().await.push_str(&chunk);
                                let payload = sse_chunk(
                                    &id_item,
                                    model,
                                    json!({"content": chunk}),
                                    Value::Null,
                                );
                                Some((Ok::<_, Infallible>(Event::default().data(payload)), rx))
                            }
                            None => None,
                        }
                    }
                });
                let id_tail = id.clone();
                let tail = stream::once(async move {
                    let final_text = accumulated.lock().await.clone();
                    let tool_calls = extract_xml_tool_calls(&final_text);
                    if !tool_calls.is_empty() {
                        let calls_json: Vec<Value> = tool_calls
                            .into_iter()
                            .enumerate()
                            .map(|(i, c)| {
                                let args_val = serde_json::from_str::<Value>(&c.arguments).unwrap_or(json!({}));
                                json!({
                                    "index": i,
                                    "id": c.id,
                                    "type": "function",
                                    "function": {
                                        "name": c.name,
                                        "arguments": args_val.to_string()
                                    }
                                })
                            })
                            .collect();
                        vec![
                            Ok(Event::default().data(sse_chunk(
                                &id_tail,
                                model,
                                json!({"tool_calls": calls_json}),
                                json!("tool_calls"),
                            ))),
                            Ok(Event::default().data("[DONE]")),
                        ]
                    } else {
                        vec![
                            Ok(Event::default().data(sse_chunk(&id_tail, model, json!({}), json!("stop")))),
                            Ok(Event::default().data("[DONE]")),
                        ]
                    }
                })
                .flat_map(stream::iter);
                Sse::new(live.chain(tail)).into_response()
            }
            Err(e) => error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference_error",
                &format!("MicroLM stream for '{model_name}' failed: {e}"),
            ),
        }
    }
}

async fn chat(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    payload: Result<Json<ChatRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if !authorized(&app, &headers) {
        return error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "provider authentication required",
        );
    }
    let request = match payload {
        Ok(Json(request)) => request,
        Err(_) => {
            return error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "invalid or unsupported request fields",
            );
        }
    };
    let model_norm = request.model.strip_prefix("nioxs::").unwrap_or(&request.model);
    if request.tools.is_empty() {
        match model_norm {
            "nio-ray" | "ray" => {
                return microlm_chat_handler("nio-ray", request, app.ray_engine.as_ref()).await;
            }
            "nio-manta" | "manta" => return microlm_chat_handler("nio-manta", request, None).await,
            "nio-viper" | "viper" => return microlm_chat_handler("nio-viper", request, None).await,
            "nio-cuda" | "cuda" => return microlm_chat_handler("nio-cuda", request, None).await,
            _ => {}
        }
    }
    let turn_override = match headers.get("x-nioxs-turn-id") {
        Some(value) => match value.to_str() {
            Ok(id) => Some(id),
            Err(_) => {
                return error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "invalid turn header",
                );
            }
        },
        None => None,
    };
    let mut decision = match decide_with_model(&request, &app.deployment, turn_override, &app.model)
    {
        Ok(d) => d,
        Err(message) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", &message),
    };
    // Scout's caller-executed fetches are the only place the provider sees
    // research content, so cache and record them here, best effort.
    research::record_web_fetches(
        &app.db,
        &app.deployment,
        &decision.execution.turn_id,
        &app.deployment,
        &request.messages,
    )
    .await;
    let mut event = DurableEvent {
        schema_version: SCHEMA_VERSION,
        deployment_id: app.deployment.clone(),
        turn_id: decision.execution.turn_id.clone(),
        event: format!("{:?}", decision.execution.outcome).to_lowercase(),
        model: decision.execution.model.clone(),
        request_digest: decision.request_digest.clone(),
        action_id: decision.call.as_ref().map(|call| call.id.clone()),
        result_digest: decision.result_digest.clone(),
        // Immutable event expiry is an explicit field; cleanup is a later implementation.
        expires_at_unix: 0,
    };
    let (sender, mut receiver) = tokio::sync::watch::channel(false);
    {
        let Ok(mut active) = app.active.0.lock() else {
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                "turn registry unavailable",
            );
        };
        if active.contains_key(&event.turn_id) {
            return error(
                StatusCode::CONFLICT,
                "turn_in_progress",
                "this turn already has an active request",
            );
        }
        active.insert(event.turn_id.clone(), sender);
    }
    let guard = TurnGuard {
        active: app.active.clone(),
        id: event.turn_id.clone(),
    };
    let recorded = tokio::select! {
        result=async {
            journal::record(&app,&mut decision).await?;
            event.event=serde_json::to_value(&decision.execution.outcome).map_err(|_|"invalid outcome")?.as_str().unwrap_or("unknown").into();
            event.action_id=decision.call.as_ref().map(|call|call.id.clone());
            app.db.append_event(&event).await.map(|_|())
        }=>result,
        _=receiver.changed()=>return error(StatusCode::CONFLICT,"turn_cancelled","turn cancelled; no client action was dispatched"),
    };
    if *receiver.borrow() {
        return error(
            StatusCode::CONFLICT,
            "turn_cancelled",
            "turn cancelled; no client action was dispatched",
        );
    }
    if let Err(reason) = recorded {
        eprintln!("RECORDED ERROR: {reason}");
        if reason == "workflow_revision_conflict" || reason == "NioDB returned HTTP 409" {
            return error(
                StatusCode::CONFLICT,
                "workflow_conflict",
                "workflow transition conflicts with durable revision; no action dispatched",
            );
        }
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "storage_unavailable",
            "durable tracking unavailable; no action was dispatched",
        );
    }
    let id = format!("chatcmpl-{}", decision.execution.turn_id);
    // Carry the turn's decision alongside execution metadata so a client can see
    // what NioNano chose, and why it refused, without a second round trip.
    let nioxs_value = {
        let mut value = serde_json::to_value(&decision.execution).unwrap_or_else(|_| json!({}));
        let dispatch = decision.call.as_ref().map(|call| {
            (
                call.function.name.as_str(),
                call.function.arguments.as_str(),
            )
        });
        let evidence: Vec<String> = decision
            .observed
            .iter()
            .map(|observed| observed.result_digest.clone())
            .collect();
        let reason = decision
            .execution
            .handoff
            .as_ref()
            .map(|handoff| handoff.reason.clone())
            .or_else(|| match decision.execution.outcome {
                Outcome::Clarification | Outcome::ToolFailed => decision.content.clone(),
                _ => None,
            });
        value["decision"] = json!(nioxs_nano::decision::summarize(
            &decision.intent,
            &decision.execution.outcome,
            dispatch,
            evidence,
            reason,
        ));
        let route = nioxs_nano::profiles::route(
            &request.model,
            request.messages[request.validate().expect("validated request")]
                .content
                .as_str()
                .unwrap_or(""),
        );
        value["routing"] = json!({"requested_model":request.model,"selected_model":route.selected,"skill":nioxs_nano::profiles::skill_name(route.selected),"reason":route.reason,"method":"explicit_profile_or_legacy_alias"});
        value["progress"] = routing_progress(&request, route.selected);
        let start = request
            .messages
            .iter()
            .rposition(|message| message.role == "user")
            .unwrap_or(0);
        value["status"] = json!({"fallback_announced": request.messages[start + 1..].iter().any(|message| message.role == "assistant" && message.content.as_str().is_some_and(|content| content.lines().any(companion::is_fallback_status)))});
        value
    };
    // A `--code:` turn hands back a pending upstream consultation instead of a
    // pre-computed answer. The model is asked with the caller's file tools and
    // its own deltas are forwarded live, so it decides what to read and the
    // client executes those reads in the directory it was started in.
    if let Some(consult) = decision.coding.clone() {
        let turn_id = decision.execution.turn_id.clone();
        if request.model == nioxs_contracts::MODEL_THINKER
            || (matches!(
                request.model.as_str(),
                nioxs_contracts::MODEL_CUDA
                    | nioxs_contracts::MODEL_FLOW
                    | nioxs_contracts::MODEL_AUTO
            ) && (!request.messages[request.validate().expect("validated request")]
                .content
                .as_str()
                .unwrap_or("")
                .trim()
                .starts_with("--")
                || nioxs_nano::profiles::is_flow_directive(
                    request.messages[request.validate().expect("validated request")]
                        .content
                        .as_str()
                        .unwrap_or(""),
                )))
        {
            return flow::respond(
                consult,
                &request,
                id,
                nioxs_value,
                guard,
                turn_id,
                &app.instance,
                &app.db,
                &app.deployment,
                &decision.request_digest,
            )
            .await;
        }
        if request.stream {
            return coding_sse(
                consult,
                decision.stream_segments.clone(),
                id,
                nioxs_value,
                guard,
                turn_id,
            )
            .await;
        }
        return coding_response(consult, id, nioxs_value, turn_id).await;
    }
    let finish = if decision.call.is_some() {
        "tool_calls"
    } else {
        "stop"
    };
    let mut message = json!({"role":"assistant","content":decision.content});
    prepend_progress(&mut message, &nioxs_value);
    if let Some(call) = &decision.call {
        message["tool_calls"] = json!([call]);
    }
    if !request.stream {
        return Json(json!({"id":id,"object":"chat.completion","created":0,"model":decision.execution.model,"choices":[{"index":0,"message":message,"finish_reason":finish}],"nioxs":nioxs_value})).into_response();
    }
    let chunk = |delta: Value, finish: Value| json!({"id":id,"object":"chat.completion.chunk","created":0,"model":decision.execution.model,"choices":[{"index":0,"delta":delta,"finish_reason":finish}]});
    let mut terminal = chunk(json!({}), json!(finish));
    terminal["nioxs"] = nioxs_value;
    // A successful `--code:` turn carries pre-ordered segments: the step trail
    // then the chunked answer. Emit each as its own SSE delta so the client
    // renders progress live. Every other intent keeps the single-chunk shape.
    if !decision.stream_segments.is_empty() {
        let payloads = trail_payloads(
            &decision.stream_segments,
            decision.call.as_ref(),
            &terminal,
            chunk,
        );
        let events: Vec<std::result::Result<Event, Infallible>> = payloads
            .into_iter()
            .map(|payload| Ok(Event::default().data(payload)))
            .collect();
        return Sse::new(stream::iter(events)).into_response();
    }
    let mut delta = message;
    if let Some(call) = &decision.call {
        delta["tool_calls"] =
            json!([{"index":0,"id":call.id,"type":"function","function":call.function}]);
    }
    let events = vec![
        Ok::<_, Infallible>(Event::default().data(chunk(delta, Value::Null).to_string())),
        Ok(Event::default().data(terminal.to_string())),
        Ok(Event::default().data("[DONE]")),
    ];
    Sse::new(stream::iter(events)).into_response()
}

/// Text and tool requests collected while forwarding an upstream `--code:`
/// stream.
#[derive(Default)]
struct CodingSink {
    text: String,
    error: Option<String>,
    calls: Vec<RequestedCall>,
}

/// One tool call the upstream model asked for, assembled from streamed deltas.
#[derive(Clone, Default, Debug)]
struct RequestedCall {
    id: String,
    name: String,
    arguments: String,
}

/// A tool call relayed to the client, carrying the id the client echoes back.
#[derive(Clone, Debug)]
struct ClientCall {
    id: String,
    name: String,
    arguments: String,
    #[allow(dead_code)]
    hint: String,
}

/// Folds one streamed payload's `tool_calls` fragments into `calls`.
fn collect_sse_tool_calls(data: &str, calls: &mut Vec<RequestedCall>) {
    let Ok(value) = serde_json::from_str::<Value>(data) else {
        return;
    };
    let Some(entries) = value["choices"][0]["delta"]["tool_calls"].as_array() else {
        return;
    };
    for entry in entries {
        let index = entry["index"].as_u64().unwrap_or(0) as usize;
        while calls.len() <= index {
            calls.push(RequestedCall::default());
        }
        let call = &mut calls[index];
        if let Some(id) = entry["id"].as_str() {
            call.id = id.to_string();
        }
        if let Some(name) = entry["function"]["name"].as_str() {
            call.name = name.to_string();
        }
        if let Some(arguments) = entry["function"]["arguments"].as_str() {
            call.arguments.push_str(arguments);
        }
    }
}

/// Reads `message.tool_calls` from a non-streamed upstream answer.
fn message_tool_calls(value: &Value) -> Vec<RequestedCall> {
    value["choices"][0]["message"]["tool_calls"]
        .as_array()
        .map(|entries| {
            entries
                .iter()
                .map(|entry| RequestedCall {
                    id: entry["id"].as_str().unwrap_or_default().to_string(),
                    name: entry["function"]["name"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                    arguments: entry["function"]["arguments"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_param_value(val_str: &str) -> Value {
    let trimmed = val_str.trim();
    if let Ok(num) = trimmed.parse::<i64>() {
        return json!(num);
    }
    if let Ok(num) = trimmed.parse::<f64>() {
        return json!(num);
    }
    if let Ok(b) = trimmed.parse::<bool>() {
        return json!(b);
    }
    if (trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'))
    {
        if let Ok(json_val) = serde_json::from_str::<Value>(trimmed) {
            return json_val;
        }
    }
    if (trimmed.starts_with('"') && trimmed.ends_with('"'))
        || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
    {
        if trimmed.len() >= 2 {
            return json!(&trimmed[1..trimmed.len() - 1]);
        }
    }
    json!(trimmed)
}

fn xml_tool_leak_pos(text: &str) -> Option<usize> {
    let lower = text.to_ascii_lowercase();
    let tags = [
        "<dots_function_call",
        "<invoke",
        "<tool_call",
        "<function_call",
        "<function=",
        "<function name=",
        "<|tool_call",
    ];
    tags.iter().filter_map(|tag| lower.find(tag)).min()
}

/// Parses pythonic tool call syntax like `[read_file(path='README.md', limit=50)]` or `read_file(...)`.
fn parse_pythonic_calls(input: &str) -> Vec<(String, Value)> {
    let mut results = Vec::new();
    let trimmed = input.trim();
    let content = if trimmed.starts_with('[') && trimmed.ends_with(']') {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    };

    let bytes = content.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        // Skip whitespace, commas, brackets
        while i < len && (bytes[i].is_ascii_whitespace() || bytes[i] == b',' || bytes[i] == b'[') {
            i += 1;
        }
        if i >= len {
            break;
        }

        // Identifier start: [a-zA-Z_]
        if !bytes[i].is_ascii_alphabetic() && bytes[i] != b'_' {
            i += 1;
            continue;
        }

        let name_start = i;
        while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'-')
        {
            i += 1;
        }
        let name = &content[name_start..i];

        // Skip whitespace before '('
        while i < len && bytes[i].is_ascii_whitespace() {
            i += 1;
        }

        if i >= len || bytes[i] != b'(' {
            continue;
        }
        i += 1; // skip '('

        // Parse arguments inside (...)
        let mut args_map = serde_json::Map::new();
        while i < len {
            // Skip whitespace and commas
            while i < len && (bytes[i].is_ascii_whitespace() || bytes[i] == b',') {
                i += 1;
            }
            if i >= len || bytes[i] == b')' {
                break;
            }

            // Arg name: [a-zA-Z0-9_]
            let arg_start = i;
            while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let arg_name = &content[arg_start..i];
            if arg_name.is_empty() {
                while i < len && bytes[i] != b',' && bytes[i] != b')' {
                    i += 1;
                }
                continue;
            }

            // Skip whitespace before '='
            while i < len && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < len && bytes[i] == b'=' {
                i += 1; // skip '='
            } else {
                while i < len && bytes[i] != b',' && bytes[i] != b')' {
                    i += 1;
                }
                continue;
            }

            // Skip whitespace before value
            while i < len && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i >= len {
                break;
            }

            // Parse value
            let parsed_val = if bytes[i] == b'\'' || bytes[i] == b'"' {
                let quote = bytes[i];
                i += 1;
                let mut escaped = false;
                let mut s = String::new();
                while i < len {
                    let b = bytes[i];
                    if escaped {
                        match b {
                            b'n' => s.push('\n'),
                            b'r' => s.push('\r'),
                            b't' => s.push('\t'),
                            b'\\' => s.push('\\'),
                            q if q == quote => s.push(quote as char),
                            other => {
                                s.push('\\');
                                s.push(other as char);
                            }
                        }
                        escaped = false;
                    } else if b == b'\\' {
                        escaped = true;
                    } else if b == quote {
                        break;
                    } else {
                        s.push(b as char);
                    }
                    i += 1;
                }
                if i < len && bytes[i] == quote {
                    i += 1;
                }
                Value::String(s)
            } else if bytes[i] == b'{' || bytes[i] == b'[' {
                let open = bytes[i];
                let close = if open == b'{' { b'}' } else { b']' };
                let val_start = i;
                let mut depth = 0;
                let mut in_str: Option<u8> = None;
                let mut esc = false;
                while i < len {
                    let b = bytes[i];
                    if let Some(q) = in_str {
                        if esc {
                            esc = false;
                        } else if b == b'\\' {
                            esc = true;
                        } else if b == q {
                            in_str = None;
                        }
                    } else if b == b'\'' || b == b'"' {
                        in_str = Some(b);
                    } else if b == open {
                        depth += 1;
                    } else if b == close {
                        depth -= 1;
                        if depth == 0 {
                            i += 1;
                            break;
                        }
                    }
                    i += 1;
                }
                let raw_val = &content[val_start..i];
                serde_json::from_str::<Value>(raw_val)
                    .unwrap_or_else(|_| Value::String(raw_val.to_string()))
            } else {
                let val_start = i;
                while i < len && bytes[i] != b',' && bytes[i] != b')' && bytes[i] != b']' {
                    i += 1;
                }
                let raw = content[val_start..i].trim();
                let lower = raw.to_ascii_lowercase();
                if lower == "true" {
                    Value::Bool(true)
                } else if lower == "false" {
                    Value::Bool(false)
                } else if lower == "none" || lower == "null" {
                    Value::Null
                } else if let Ok(num) = raw.parse::<i64>() {
                    json!(num)
                } else if let Ok(num) = raw.parse::<f64>() {
                    json!(num)
                } else {
                    Value::String(raw.to_string())
                }
            };

            args_map.insert(arg_name.to_string(), parsed_val);
        }

        if i < len && bytes[i] == b')' {
            i += 1; // skip ')'
        }

        results.push((name.to_string(), Value::Object(args_map)));
    }

    results
}

/// Recovers tool calls emitted as XML text (such as `<dots_function_call>` or `<invoke>`).
fn extract_xml_tool_calls(text: &str) -> Vec<RequestedCall> {
    let mut calls = Vec::new();
    let mut cursor = 0;

    // 1. Look for <invoke ...> tags (covers <dots_function_call> and standalone <invoke>)
    while let Some(invoke_pos) = text[cursor..].find("<invoke") {
        let abs_invoke = cursor + invoke_pos;
        let Some(tag_open_end) = text[abs_invoke..].find('>').map(|i| abs_invoke + i) else {
            break;
        };
        let tag_header = &text[abs_invoke..tag_open_end];
        let name = if let Some(name_pos) = tag_header.find("name=") {
            let after_name = &tag_header[name_pos + 5..];
            let trimmed = after_name
                .trim()
                .trim_start_matches(|c| c == '"' || c == '\'');
            let end = trimmed
                .find(|c: char| c.is_whitespace() || c == '"' || c == '\'')
                .unwrap_or(trimmed.len());
            trimmed[..end].trim().to_string()
        } else {
            String::new()
        };

        let block_end = match text[tag_open_end + 1..].find("</invoke>") {
            Some(i) => tag_open_end + 1 + i,
            None => match text[tag_open_end + 1..].find("</dots_function_call>") {
                Some(i) => tag_open_end + 1 + i,
                None => text.len(),
            },
        };
        let block_content = &text[tag_open_end + 1..block_end];

        if !name.is_empty() {
            let mut args_map = serde_json::Map::new();
            let mut param_cursor = 0;
            while let Some(param_pos) = block_content[param_cursor..]
                .find("<parameter=")
                .or_else(|| block_content[param_cursor..].find("<parameter name="))
            {
                let abs_param = param_cursor + param_pos;
                let is_attr = block_content[abs_param..].starts_with("<parameter name=");
                let tag_start = if is_attr {
                    abs_param + 16
                } else {
                    abs_param + 11
                };
                let Some(tag_end) = block_content[tag_start..].find('>') else {
                    break;
                };
                let param_name = block_content[tag_start..tag_start + tag_end]
                    .trim()
                    .trim_matches(|c| c == '"' || c == '\'');

                let val_start = tag_start + tag_end + 1;
                let val_end = match block_content[val_start..].find("</parameter>") {
                    Some(closing) => val_start + closing,
                    None => block_content.len(),
                };
                let val_str = &block_content[val_start..val_end];
                args_map.insert(param_name.to_string(), parse_param_value(val_str));

                param_cursor = val_end;
                if param_cursor >= block_content.len() {
                    break;
                }
            }

            if args_map.is_empty() {
                if let Some(b_start) = block_content.find('{') {
                    if let Some(b_end) = block_content.rfind('}') {
                        if b_end > b_start {
                            if let Ok(Value::Object(obj)) =
                                serde_json::from_str(&block_content[b_start..=b_end])
                            {
                                args_map = obj;
                            }
                        }
                    }
                }
            }

            calls.push(RequestedCall {
                id: format!("call_xml_{}", calls.len()),
                name,
                arguments: Value::Object(args_map).to_string(),
            });
        }

        cursor = if block_end < text.len() {
            block_end + 9
        } else {
            text.len()
        };
    }

    // 2. Also support <function=...> or <function name=...> if no <invoke> was found
    if calls.is_empty() {
        let mut cur = 0;
        while let Some(func_pos) = text[cur..]
            .find("<function=")
            .or_else(|| text[cur..].find("<function name="))
        {
            let abs_func = cur + func_pos;
            let is_name_attr = text[abs_func..].starts_with("<function name=");
            let tag_name_start = if is_name_attr {
                abs_func + 15
            } else {
                abs_func + 10
            };
            let Some(tag_open_end) = text[tag_name_start..].find('>').map(|i| tag_name_start + i)
            else {
                break;
            };
            let name_raw = &text[tag_name_start..tag_open_end];
            let name = name_raw.trim().trim_matches(|c| c == '"' || c == '\'');

            let block_end = match text[tag_open_end + 1..].find("</function>") {
                Some(i) => tag_open_end + 1 + i,
                None => text.len(),
            };
            let block_content = &text[tag_open_end + 1..block_end];

            if !name.is_empty() {
                let mut args_map = serde_json::Map::new();
                let mut param_cursor = 0;
                while let Some(param_pos) = block_content[param_cursor..]
                    .find("<parameter=")
                    .or_else(|| block_content[param_cursor..].find("<parameter name="))
                {
                    let abs_param = param_cursor + param_pos;
                    let is_attr = block_content[abs_param..].starts_with("<parameter name=");
                    let tag_start = if is_attr {
                        abs_param + 16
                    } else {
                        abs_param + 11
                    };
                    let Some(tag_end) = block_content[tag_start..].find('>') else {
                        break;
                    };
                    let param_name = block_content[tag_start..tag_start + tag_end]
                        .trim()
                        .trim_matches(|c| c == '"' || c == '\'');

                    let val_start = tag_start + tag_end + 1;
                    let val_end = match block_content[val_start..].find("</parameter>") {
                        Some(closing) => val_start + closing,
                        None => block_content.len(),
                    };
                    let val_str = &block_content[val_start..val_end];
                    args_map.insert(param_name.to_string(), parse_param_value(val_str));

                    param_cursor = val_end;
                    if param_cursor >= block_content.len() {
                        break;
                    }
                }

                if args_map.is_empty() {
                    if let Some(b_start) = block_content.find('{') {
                        if let Some(b_end) = block_content.rfind('}') {
                            if b_end > b_start {
                                if let Ok(Value::Object(obj)) =
                                    serde_json::from_str(&block_content[b_start..=b_end])
                                {
                                    args_map = obj;
                                }
                            }
                        }
                    }
                }

                calls.push(RequestedCall {
                    id: format!("call_xml_{}", calls.len()),
                    name: name.to_string(),
                    arguments: Value::Object(args_map).to_string(),
                });
            }

            cur = if block_end < text.len() {
                block_end + 11
            } else {
                text.len()
            };
        }
    }

    // 3. Also support <tool_call> JSON if neither <invoke> nor <function> was found
    if calls.is_empty() {
        let mut cur = 0;
        while let Some(start) = text[cur..].find("<tool_call>") {
            let abs_start = cur + start + 11;
            let end = match text[abs_start..].find("</tool_call>") {
                Some(i) => abs_start + i,
                None => text.len(),
            };
            let block = text[abs_start..end].trim();
            let mut found_call = false;
            if block.starts_with('{') {
                if let Ok(val) = serde_json::from_str::<Value>(block) {
                    if let Some(name) = val.get("name").and_then(Value::as_str) {
                        let arguments = val.get("arguments").cloned().unwrap_or(json!({}));
                        calls.push(RequestedCall {
                            id: format!("call_xml_{}", calls.len()),
                            name: name.to_string(),
                            arguments: arguments.to_string(),
                        });
                        found_call = true;
                    }
                }
            } else if block.starts_with('[') {
                if let Ok(Value::Array(items)) = serde_json::from_str::<Value>(block) {
                    for item in items {
                        if let Some(name) = item.get("name").and_then(Value::as_str) {
                            let arguments = item.get("arguments").cloned().unwrap_or(json!({}));
                            calls.push(RequestedCall {
                                id: format!("call_xml_{}", calls.len()),
                                name: name.to_string(),
                                arguments: arguments.to_string(),
                            });
                            found_call = true;
                        }
                    }
                }
            }
            if !found_call {
                for (name, args) in parse_pythonic_calls(block) {
                    calls.push(RequestedCall {
                        id: format!("call_xml_{}", calls.len()),
                        name,
                        arguments: args.to_string(),
                    });
                }
            }
            cur = if end < text.len() {
                end + 12
            } else {
                text.len()
            };
        }
    }

    // 4. Also support ChatML / Qwen <|tool_call_start|> ... <|tool_call_end|>
    if calls.is_empty() {
        let mut cur = 0;
        while let Some(start) = text[cur..]
            .find("<|tool_call_start|>")
            .or_else(|| text[cur..].find("<|tool_call"))
        {
            let abs_start = cur + start;
            let tag_open_end = match text[abs_start..].find('>') {
                Some(i) => abs_start + i + 1,
                None => abs_start + 19,
            };
            let end = match text[tag_open_end..].find("<|tool_call_end|>") {
                Some(i) => tag_open_end + i,
                None => match text[tag_open_end..].find("<|tool_call_end") {
                    Some(i) => tag_open_end + i,
                    None => match text[tag_open_end..].find("|>") {
                        Some(i) => tag_open_end + i,
                        None => text.len(),
                    },
                },
            };
            let block = text[tag_open_end..end].trim();
            let mut found_call = false;
            if block.starts_with('{') {
                if let Ok(val) = serde_json::from_str::<Value>(block) {
                    if let Some(name) = val.get("name").and_then(Value::as_str) {
                        let arguments = val.get("arguments").cloned().unwrap_or(json!({}));
                        calls.push(RequestedCall {
                            id: format!("call_xml_{}", calls.len()),
                            name: name.to_string(),
                            arguments: arguments.to_string(),
                        });
                        found_call = true;
                    }
                }
            } else if block.starts_with('[') {
                if let Ok(Value::Array(items)) = serde_json::from_str::<Value>(block) {
                    for item in items {
                        if let Some(name) = item.get("name").and_then(Value::as_str) {
                            let arguments = item.get("arguments").cloned().unwrap_or(json!({}));
                            calls.push(RequestedCall {
                                id: format!("call_xml_{}", calls.len()),
                                name: name.to_string(),
                                arguments: arguments.to_string(),
                            });
                            found_call = true;
                        }
                    }
                }
            }
            if !found_call {
                for (name, args) in parse_pythonic_calls(block) {
                    calls.push(RequestedCall {
                        id: format!("call_xml_{}", calls.len()),
                        name,
                        arguments: args.to_string(),
                    });
                }
            }
            cur = if end < text.len() {
                end + 17
            } else {
                text.len()
            };
        }
    }

    calls
}

/// Turns the model's tool requests into calls the caller can execute.
///
/// Only the caller's own file tools are relayed, and every path must stay inside
/// the caller's project, so a model can never talk the client into reading
/// outside it. An unusable request is reported instead of being silently sent.
fn relay_coding_calls(calls: &[RequestedCall], turn: &str) -> Result<Vec<ClientCall>, String> {
    let mut relayed = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (index, call) in calls.iter().enumerate() {
        if !companion::is_coding_tool(&call.name) {
            return Err(format!(
                "the upstream model requested unsupported tool `{}`",
                call.name
            ));
        }
        let Ok(mut arguments) = serde_json::from_str::<Value>(&call.arguments) else {
            return Err(format!(
                "`{}` was called with unparseable arguments",
                call.name
            ));
        };
        // Normalize search_code query if model passed alternative keys
        if call.name == "search_code" && arguments.get("query").is_none() {
            if let Some(val) = arguments
                .get("pattern")
                .or_else(|| arguments.get("text"))
                .or_else(|| arguments.get("search"))
                .or_else(|| arguments.get("term"))
            {
                if let Some(s) = val.as_str() {
                    arguments["query"] = json!(s);
                }
            }
        }
        let serialized_args = arguments.to_string();
        // Deduplicate duplicate identical tool calls in the same batch
        let key = (call.name.clone(), serialized_args.clone());
        if !seen.insert(key) {
            continue;
        }
        let path = arguments["path"].as_str().unwrap_or(".");
        if call.name != "read_skill_file"
            && call.name != "git_status"
            && !nioxs_nano::safe_path(path, false)
        {
            return Err(format!("`{path}` is outside the caller's project"));
        }
        let hint = if call.name == "read_file" {
            if arguments["path"].as_str().is_none() {
                return Err("read_file was called without a path".into());
            }
            format!("📄 read_file {path}\n")
        } else if call.name == "read_skill_file" {
            let skill_name = arguments["name"].as_str().unwrap_or("skill");
            format!("🧠 read_skill {skill_name}\n")
        } else if call.name == "search_code" {
            let query = arguments["query"].as_str().unwrap_or("");
            format!("🔍 search_code {query}\n")
        } else if call.name == "git_status" {
            "📊 git_status\n".to_string()
        } else if call.name == "git_diff" {
            let p = arguments["path"].as_str().unwrap_or(".");
            format!("📝 git_diff {p}\n")
        } else {
            let glob = arguments["glob"].as_str().unwrap_or("**/*");
            format!("📁 find_files {glob}\n")
        };
        relayed.push(ClientCall {
            id: format!(
                "call_nano_{turn}_{}",
                &nioxs_contracts::digest(
                    format!("{turn}:{index}:{}:{}", call.name, serialized_args).as_bytes()
                )[..16]
            ),
            name: call.name.clone(),
            arguments: serialized_args,
            hint,
        });
    }
    Ok(relayed)
}

/// Builds the SSE payloads for a turn that streams a progress trail: the trail
/// lines first, then the pending tool call, then the terminal chunk.
///
/// A `--code:` turn reads the caller's project one client action at a time,
/// so a trail round still has to carry the call the client executes next.
fn trail_payloads(
    segments: &[String],
    call: Option<&ToolCall>,
    terminal: &Value,
    chunk: impl Fn(Value, Value) -> Value,
) -> Vec<String> {
    let mut payloads: Vec<String> = segments
        .iter()
        .map(|segment| chunk(json!({"content": segment}), Value::Null).to_string())
        .collect();
    if let Some(call) = call {
        payloads.push(
            chunk(
                json!({"tool_calls":[{"index":0,"id":call.id,"type":"function","function":call.function}]}),
                Value::Null,
            )
            .to_string(),
        );
    }
    payloads.push(terminal.to_string());
    payloads.push("[DONE]".into());
    payloads
}

fn relay_offered_coding_calls(
    calls: &[RequestedCall],
    turn: &str,
    tools: &[Value],
    messages: &[Value],
) -> Result<Vec<ClientCall>, String> {
    if tools.is_empty() && !calls.is_empty() {
        return Err("the project-inspection tool budget is exhausted; answer using the evidence already provided".into());
    }
    for call in calls {
        if !tools
            .iter()
            .any(|tool| tool["function"]["name"] == call.name)
        {
            return Err(format!(
                "the upstream model requested tool `{}` that was not offered this turn",
                call.name
            ));
        }
    }
    let calls = normalize_caller_paths(calls, messages)?;
    let mut relayed = relay_coding_calls(&calls, turn)?;
    // A call identifies one occurrence, not just its arguments. The history
    // changes after each result, so repeated reads cannot collide across rounds.
    let round = nioxs_contracts::digest(
        &serde_json::to_vec(messages).map_err(|_| "cannot encode tool history")?,
    );
    for call in &mut relayed {
        let schema = tools
            .iter()
            .find(|tool| tool["function"]["name"] == call.name)
            .expect("offered tool");
        let arguments: Value =
            serde_json::from_str(&call.arguments).map_err(|_| "invalid tool arguments")?;
        if let Some(required) = schema["function"]["parameters"]["required"].as_array() {
            for field in required.iter().filter_map(Value::as_str) {
                if arguments.get(field).is_none_or(Value::is_null) {
                    return Err(format!(
                        "{} is missing required argument `{field}`; no tool was executed",
                        call.name
                    ));
                }
            }
        }
        if let Some(properties) = schema["function"]["parameters"]["properties"].as_object() {
            for (field, property) in properties {
                if let Some(value) = arguments.get(field) {
                    let valid = match property["type"].as_str() {
                        Some("string") => value.is_string(),
                        Some("integer") => value.is_i64() || value.is_u64(),
                        Some("boolean") => value.is_boolean(),
                        Some("object") => value.is_object(),
                        Some("array") => value.is_array(),
                        _ => true,
                    };
                    if !valid {
                        return Err(format!(
                            "{} has an invalid `{field}` argument; no tool was executed",
                            call.name
                        ));
                    }
                }
            }
        }
        let occurrence = nioxs_contracts::digest(format!("{}:{round}", call.id).as_bytes());
        call.id = format!("call_nano_{turn}_{}", &occurrence[..16]);
    }
    Ok(relayed)
}

fn is_caller_path_absolute(path: &std::path::Path) -> bool {
    if path.is_absolute() || path.has_root() {
        return true;
    }
    let s = path.to_string_lossy();
    let b = s.as_bytes();
    if b.starts_with(b"/") || b.starts_with(b"\\") {
        return true;
    }
    if b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1] == b':' && (b[2] == b'/' || b[2] == b'\\')
    {
        return true;
    }
    false
}

/// Convert only paths underneath the root declared in the caller's system
/// context. Do not consult the daemon's filesystem or widen caller permissions.
fn normalize_caller_paths(
    calls: &[RequestedCall],
    messages: &[Value],
) -> Result<Vec<RequestedCall>, String> {
    let root = messages
        .iter()
        .filter(|message| message["role"] == "system")
        .filter_map(|message| message["content"].as_str())
        .find_map(|content| {
            content
                .split_once("Project overview:\nRoot: ")
                .and_then(|(_, rest)| rest.lines().next())
        })
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(std::path::Path::new)
        .filter(|root| is_caller_path_absolute(root));
    let mut normalized = calls.to_vec();
    for call in &mut normalized {
        let Ok(mut arguments) = serde_json::from_str::<Value>(&call.arguments) else {
            continue;
        };
        let Some(path) = arguments["path"].as_str() else {
            continue;
        };
        let absolute = std::path::Path::new(path);
        if !is_caller_path_absolute(absolute) {
            continue;
        }
        let Some(root) = root else { continue };
        if absolute
            .components()
            .any(|part| matches!(part, std::path::Component::ParentDir))
        {
            return Err(format!(
                "`{path}` contains parent traversal; use a project-relative path"
            ));
        }
        let rel_str = if let Ok(rel) = absolute.strip_prefix(root) {
            rel.to_string_lossy().replace('\\', "/")
        } else {
            let clean_root = root.to_string_lossy().replace('\\', "/");
            let clean_path = absolute.to_string_lossy().replace('\\', "/");
            let root_clean = clean_root.trim_end_matches('/');
            if clean_path == root_clean {
                String::new()
            } else if clean_path.starts_with(&format!("{root_clean}/")) {
                clean_path[root_clean.len() + 1..].to_string()
            } else if cfg!(windows)
                && clean_path
                    .to_ascii_lowercase()
                    .starts_with(&format!("{}/", root_clean.to_ascii_lowercase()))
            {
                clean_path[root_clean.len() + 1..].to_string()
            } else {
                return Err(format!("`{path}` is outside the caller's project"));
            }
        };
        let rel_trimmed = rel_str.trim_matches('/');
        arguments["path"] = json!(if rel_trimmed.is_empty() {
            "."
        } else {
            rel_trimmed
        });
        call.arguments = arguments.to_string();
    }
    Ok(normalized)
}

fn routing_progress(request: &ChatRequest, selected: &str) -> Value {
    let previous = request
        .messages
        .iter()
        .rev()
        .filter(|message| message.role == "assistant")
        .filter_map(|message| message.content.as_str())
        .find_map(|content| content.lines().rev().find_map(companion::stage_model));
    let start = request
        .messages
        .iter()
        .rposition(|message| message.role == "user")
        .unwrap_or(0);
    let continuing = request.messages[start + 1..]
        .iter()
        .any(|message| !message.tool_calls.is_empty());
    if previous == Some(selected) || (previous.is_none() && continuing) {
        return json!([]);
    }
    let message = format!("{}\n", companion::model_stage(selected).unwrap_or(selected));
    json!([{"kind":if previous.is_some() {"profile_switched"} else {"profile_selected"},"from":previous,"to":selected,"message":message}])
}

fn progress_text(value: &Value) -> Vec<String> {
    value["progress"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|event| event["message"].as_str().map(str::to_string))
        .collect()
}

fn prepend_progress(message: &mut Value, metadata: &Value) {
    let prefix = progress_text(metadata).concat();
    if !prefix.is_empty() {
        message["content"] = json!(format!(
            "{prefix}{}",
            message["content"].as_str().unwrap_or("")
        ));
    }
}

fn record_upstream_fallback(
    config: &companion::CodingConfig,
    metadata: &mut Value,
) -> Option<String> {
    if !config.automatic_model
        || config.model != companion::FALLBACK_CODING_MODEL
        || metadata["status"]["fallback_announced"] == true
    {
        return None;
    }
    let message = "↪ Using Kilo Auto Free\n".to_string();
    if !metadata["progress"].is_array() {
        metadata["progress"] = json!([]);
    }
    metadata["progress"].as_array_mut().unwrap().push(json!({"kind":"upstream_fallback","from":companion::DEFAULT_CODING_MODEL,"to":config.model,"message":message}));
    metadata["status"]["fallback_announced"] = json!(true);
    Some(message)
}

fn sse_chunk(id: &str, model: &str, delta: Value, finish: Value) -> String {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })
    .to_string()
}

fn content_event(id: &str, model: &str, content: String) -> Result<Event, Infallible> {
    if companion::stage_model(&content).is_some() || companion::is_fallback_status(&content) {
        return Ok(Event::default().data(status_chunk(id, model, Some(content), "routing")));
    }
    Ok(Event::default().data(sse_chunk(
        id,
        model,
        json!({"content": content}),
        Value::Null,
    )))
}

fn status_chunk(id: &str, model: &str, content: Option<String>, phase: &str) -> String {
    let delta = content.map_or_else(|| json!({}), |text| json!({"content":text}));
    let mut chunk: Value =
        serde_json::from_str(&sse_chunk(id, model, delta, Value::Null)).expect("valid chunk");
    chunk["nioxs"] = json!({"status":{"phase":phase,"message":if phase == "thinking" { "Thinking…" } else { "Routing" }}});
    chunk.to_string()
}

/// Splits buffered SSE bytes into complete, non-terminal `data:` payloads.
fn take_sse_data_lines(buffer: &mut String) -> Vec<String> {
    let mut payloads = Vec::new();
    while let Some(position) = buffer.find('\n') {
        let line: String = buffer.drain(..=position).collect();
        let line = line.trim_end_matches(['\n', '\r']);
        if let Some(data) = line.strip_prefix("data:") {
            let data = data.trim();
            if !data.is_empty() && data != "[DONE]" {
                payloads.push(data.to_string());
            }
        }
    }
    payloads
}

/// Extracts assistant text from one OpenAI-compatible SSE payload.
fn sse_delta_content(data: &str) -> Option<String> {
    let value: Value = serde_json::from_str(data).ok()?;
    let choice = &value["choices"][0];
    choice["delta"]["content"]
        .as_str()
        .or_else(|| choice["message"]["content"].as_str())
        .or_else(|| choice["text"].as_str())
        .map(str::to_string)
        .filter(|text| !text.is_empty())
}

/// Streams the progress trail, then the upstream model's own deltas, then a
/// guarded verdict — all as OpenAI `delta.content` events.
async fn coding_sse(
    consult: CodingConsult,
    trail: Vec<String>,
    id: String,
    nioxs_value: Value,
    guard: TurnGuard,
    turn_id: String,
) -> Response {
    // Yield routing status before waiting for the upstream HTTP response.
    let model = nioxs_value["model"].as_str().unwrap_or(MODEL);
    let prefix: Vec<Result<axum::body::Bytes, axum::Error>> = progress_text(&nioxs_value)
        .into_iter()
        .map(|text| {
            Ok(axum::body::Bytes::from(format!(
                "data: {}\n\n",
                status_chunk(&id, model, Some(text), "routing")
            )))
        })
        .chain(std::iter::once(Ok(axum::body::Bytes::from(format!(
            "data: {}\n\n",
            status_chunk(&id, model, None, "thinking")
        )))))
        .collect();
    let body = stream::once(async move {
        coding_sse_inner(consult, trail, id, nioxs_value, guard, turn_id)
            .await
            .into_body()
            .into_data_stream()
    })
    .flatten();
    let mut response = Response::new(axum::body::Body::from_stream(
        stream::iter(prefix).chain(body),
    ));
    response
        .headers_mut()
        .insert("content-type", "text/event-stream".parse().unwrap());
    response
        .headers_mut()
        .insert("cache-control", "no-cache".parse().unwrap());
    response
}

async fn coding_sse_inner(
    mut consult: CodingConsult,
    mut trail: Vec<String>,
    id: String,
    mut nioxs_value: Value,
    guard: TurnGuard,
    turn_id: String,
) -> Response {
    let model = nioxs_value["model"].as_str().unwrap_or(MODEL).to_string();
    let url = companion::upstream_chat_url(&consult.config);
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            return coding_static_sse(
                id,
                trail,
                format!("⚠️ Failed to build the upstream HTTP client: {error}"),
                nioxs_value,
            );
        }
    };
    let response = match nioxs_nano::upstream::send_chat(
        &client,
        &mut consult.config,
        &consult.messages,
        &consult.tools,
        true,
    )
    .await
    {
        Ok(response) if response.status().is_success() => response,
        Ok(response) => {
            let status = response.status();
            let detail: String = response
                .text()
                .await
                .unwrap_or_default()
                .chars()
                .take(600)
                .collect();
            return coding_static_sse(
                id,
                trail,
                format!(
                    "⚠️ Upstream `{}` returned {status}: {detail}",
                    consult.config.model
                ),
                nioxs_value,
            );
        }
        Err(error) => {
            return coding_static_sse(
                id,
                trail,
                format!("⚠️ Request to `{url}` failed: {error}"),
                nioxs_value,
            );
        }
    };
    nioxs_value["upstream_model"] = json!(consult.config.model);
    if let Some(message) = record_upstream_fallback(&consult.config, &mut nioxs_value) {
        trail.push(message);
    }
    // The completed answer already carries an "answered by" footer. Repeating
    // the consultation banner on every tool-result continuation is noise.
    trail.retain(|line| !line.starts_with(companion::TRAIL_CONSULT));

    // Forward the model's deltas as they arrive. `sink` accumulates the answer
    // so the guarded check and error reporting can run once the stream ends.
    let sink = Arc::new(Mutex::new(CodingSink::default()));
    let live_sink = sink.clone();
    let live = stream::unfold(
        (response, String::new(), VecDeque::<String>::new(), false),
        move |(mut response, mut buffer, mut queue, mut done)| {
            let sink = live_sink.clone();
            async move {
                loop {
                    if let Some(delta) = queue.pop_front() {
                        return Some((delta, (response, buffer, queue, done)));
                    }
                    if done {
                        return None;
                    }
                    match response.chunk().await {
                        Ok(Some(bytes)) => {
                            buffer.push_str(&String::from_utf8_lossy(&bytes));
                            for payload in take_sse_data_lines(&mut buffer) {
                                if let Ok(mut sink) = sink.lock() {
                                    if let Some(delta) = sse_delta_content(&payload) {
                                        let prev_len = sink.text.len();
                                        sink.text.push_str(&delta);
                                        match xml_tool_leak_pos(&sink.text) {
                                            Some(leak_pos) => {
                                                if leak_pos > prev_len {
                                                    let safe_len = leak_pos - prev_len;
                                                    queue.push_back(delta[..safe_len].to_string());
                                                }
                                            }
                                            None => {
                                                queue.push_back(delta);
                                            }
                                        }
                                    }
                                    // The model may instead ask the caller to read
                                    // something; those fragments arrive last.
                                    collect_sse_tool_calls(&payload, &mut sink.calls);
                                }
                            }
                        }
                        Ok(None) => done = true,
                        Err(error) => {
                            if let Ok(mut sink) = sink.lock() {
                                sink.error =
                                    Some(format!("⚠️ Upstream stream interrupted: {error}"));
                            }
                            done = true;
                        }
                    }
                }
            }
        },
    );

    let id_trail = id.clone();
    let trail_events: Vec<Result<Event, Infallible>> = trail
        .into_iter()
        .map(|segment| content_event(&id_trail, &model, segment))
        .collect();
    let id_live = id.clone();
    let live_model = model.clone();
    let live_events = live.map(move |delta| content_event(&id_live, &live_model, delta));

    let tail_sink = sink.clone();
    let tail_consult = consult.clone();
    let tail_turn = turn_id;
    let tail_value = nioxs_value;
    let id_tail = id.clone();
    let tail = stream::once(async move {
        let (text, error, mut calls) = match tail_sink.lock() {
            Ok(sink) => (sink.text.clone(), sink.error.clone(), sink.calls.clone()),
            Err(_) => return Vec::new(),
        };
        if calls.is_empty() {
            calls = extract_xml_tool_calls(&text);
        }
        // The model owns exploration: when it asks for files, relay its calls to
        // the caller instead of answering, so the next round reaches it with the
        // real contents and it can decide what else it needs.
        if !calls.is_empty() {
            return match relay_offered_coding_calls(
                &calls,
                &tail_turn,
                &tail_consult.tools,
                &tail_consult.messages,
            ) {
                Ok(relayed) => {
                    let mut events: Vec<Result<Event, Infallible>> = Vec::new();
                    for (index, call) in relayed.iter().enumerate() {
                        events.push(Ok(
                            Event::default().data(call_delta(&id_tail, &model, call, index))
                        ));
                    }
                    events.push(Ok(Event::default().data(coding_terminal(
                        &id_tail,
                        "tool_calls",
                        &tail_value,
                    ))));
                    events.push(Ok(Event::default().data("[DONE]")));
                    events
                }
                Err(reason) if tail_consult.tools.is_empty() => {
                    let mut recovery_messages = tail_consult.messages.clone();
                    recovery_messages.push(json!({
                        "role": "user",
                        "content": "The tool inspection budget is reached. You cannot execute more tools. Provide your complete final answer now using the files and evidence already inspected above."
                    }));
                    let client = reqwest::Client::new();
                    let mut config = tail_consult.config.clone();
                    if let Ok(res) = nioxs_nano::upstream::send_chat(&client, &mut config, &recovery_messages, &[], false).await {
                        if let Ok(val) = res.json::<Value>().await {
                            if let Some(ans) = val["choices"][0]["message"]["content"].as_str().filter(|s| !s.trim().is_empty()) {
                                let mut events = vec![content_event(&id_tail, &model, ans.to_string())];
                                events.push(Ok(Event::default().data(coding_terminal(
                                    &id_tail,
                                    "stop",
                                    &tail_value,
                                ))));
                                events.push(Ok(Event::default().data("[DONE]")));
                                return events;
                            }
                        }
                    }
                    let mut events = vec![content_event(
                        &id_tail,
                        &model,
                        format!("\n\n⚠️ {reason}; no additional files were read."),
                    )];
                    events.push(Ok(Event::default().data(coding_terminal(
                        &id_tail,
                        "stop",
                        &tail_value,
                    ))));
                    events.push(Ok(Event::default().data("[DONE]")));
                    events
                }
                Err(reason) => {
                    let mut events = vec![content_event(
                        &id_tail,
                        &model,
                        format!("\n\n⚠️ {reason}; no additional files were read."),
                    )];
                    events.push(Ok(Event::default().data(coding_terminal(
                        &id_tail,
                        "stop",
                        &tail_value,
                    ))));
                    events.push(Ok(Event::default().data("[DONE]")));
                    events
                }
            };
        }
        let mut closing = String::new();
        if let Some(error) = error {
            closing.push_str(&format!("\n\n{error}"));
        }
        if tail_consult.guarded
            && let Some(alert) = companion::detect_hallucinations(&text, &[], &tail_consult.files)
        {
            closing.push_str(&format!(
                "\n\n---\n\n### ⚠️ [Anti-Hallucination Guard Activated]\n\n{}",
                alert.trim()
            ));
        }
        // Option C: Append subtle footer at the end of the answer only when answer text is non-empty.
        if !text.trim().is_empty() {
            closing.push_str(&companion::coding_footer(&tail_consult.config.model));
        }
        let mut events = Vec::new();
        if !closing.is_empty() {
            events.push(content_event(&id_tail, &model, closing));
        }
        events.push(Ok(Event::default().data(coding_terminal(
            &id_tail,
            "stop",
            &tail_value,
        ))));
        events.push(Ok(Event::default().data("[DONE]")));
        events
    })
    .flat_map(stream::iter);

    // Hold the turn guard for the lifetime of the response so a cancel request
    // can still reach a turn that is mid-stream.
    let keeper = Arc::new(guard);
    let events = stream::iter(trail_events)
        .chain(live_events)
        .chain(tail)
        .map(move |event| {
            let _ = &keeper;
            event
        });
    Sse::new(events).into_response()
}

/// Builds the terminal chunk of a coding stream with the given finish reason.
fn coding_terminal(id: &str, finish: &str, nioxs_value: &Value) -> String {
    let mut terminal: Value = serde_json::from_str(&sse_chunk(
        id,
        nioxs_value["model"].as_str().unwrap_or(MODEL),
        json!({}),
        json!(finish),
    ))
    .expect("valid terminal chunk");
    terminal["nioxs"] = nioxs_value.clone();
    if finish == "tool_calls" {
        terminal["nioxs"]["status"]["phase"] = json!("awaiting_tools");
        terminal["nioxs"]["status"]["target"] = json!("caller");
    }
    terminal.to_string()
}

/// One client-visible `tool_calls` delta for a relayed model request.
fn call_delta(id: &str, model: &str, call: &ClientCall, index: usize) -> String {
    sse_chunk(
        id,
        model,
        json!({"tool_calls":[{"index":index,"id":call.id,"type":"function","function":{"name":call.name,"arguments":call.arguments}}]}),
        Value::Null,
    )
}

/// Consults the upstream model for a caller that does not stream.
///
/// The answer is returned as one message, or as the model's own tool calls when
/// it wants to read the caller's project first.
async fn coding_response(
    mut consult: CodingConsult,
    id: String,
    mut nioxs_value: Value,
    turn_id: String,
) -> Response {
    let url = companion::upstream_chat_url(&consult.config);
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            return coding_json(
                &id,
                json!({"role":"assistant","content":format!("⚠️ Failed to build the upstream HTTP client: {error}")}),
                "stop",
                nioxs_value,
            );
        }
    };
    let value: Value = match nioxs_nano::upstream::send_chat(
        &client,
        &mut consult.config,
        &consult.messages,
        &consult.tools,
        false,
    )
    .await
    {
        Ok(response) if response.status().is_success() => match response.json().await {
            Ok(value) => value,
            Err(error) => {
                return coding_json(
                    &id,
                    json!({"role":"assistant","content":format!("⚠️ Invalid upstream response: {error}")}),
                    "stop",
                    nioxs_value,
                );
            }
        },
        Ok(response) => {
            let status = response.status();
            let detail: String = response
                .text()
                .await
                .unwrap_or_default()
                .chars()
                .take(600)
                .collect();
            return coding_json(
                &id,
                json!({"role":"assistant","content":format!("⚠️ Upstream `{}` returned {status}: {detail}", consult.config.model)}),
                "stop",
                nioxs_value,
            );
        }
        Err(error) => {
            return coding_json(
                &id,
                json!({"role":"assistant","content":format!("⚠️ Request to `{url}` failed: {error}")}),
                "stop",
                nioxs_value,
            );
        }
    };
    nioxs_value["upstream_model"] = json!(consult.config.model);

    record_upstream_fallback(&consult.config, &mut nioxs_value);

    let mut calls = message_tool_calls(&value);
    if calls.is_empty() {
        if let Some(content) = value["choices"][0]["message"]["content"].as_str() {
            calls = extract_xml_tool_calls(content);
        }
    }
    if !calls.is_empty() {
        return match relay_offered_coding_calls(&calls, &turn_id, &consult.tools, &consult.messages)
        {
            Ok(relayed) => {
                let tool_calls: Vec<Value> = relayed
                    .iter()
                    .enumerate()
                    .map(|(index, call)| {
                        json!({"index":index,"id":call.id,"type":"function","function":{"name":call.name,"arguments":call.arguments}})
                    })
                    .collect();
                coding_json(
                    &id,
                    json!({"role":"assistant","content":null,"tool_calls":tool_calls}),
                    "tool_calls",
                    nioxs_value,
                )
            }
            Err(reason) if consult.tools.is_empty() => {
                let mut recovery_messages = consult.messages.clone();
                recovery_messages.push(json!({
                    "role": "user",
                    "content": "The tool inspection budget is reached. You cannot execute more tools. Provide your complete final answer now using the files and evidence already inspected above."
                }));
                let mut config = consult.config.clone();
                if let Ok(res) = nioxs_nano::upstream::send_chat(
                    &client,
                    &mut config,
                    &recovery_messages,
                    &[],
                    false,
                )
                .await
                {
                    if let Ok(val) = res.json::<Value>().await {
                        if let Some(ans) = val["choices"][0]["message"]["content"]
                            .as_str()
                            .filter(|s| !s.trim().is_empty())
                        {
                            return coding_json(
                                &id,
                                json!({"role":"assistant","content":ans}),
                                "stop",
                                nioxs_value,
                            );
                        }
                    }
                }
                coding_json(
                    &id,
                    json!({"role":"assistant","content":format!("⚠️ {reason}; no additional files were read.")}),
                    "stop",
                    nioxs_value,
                )
            }
            Err(reason) => coding_json(
                &id,
                json!({"role":"assistant","content":format!("⚠️ {reason}; no additional files were read.")}),
                "stop",
                nioxs_value,
            ),
        };
    }

    let mut content = value["choices"][0]["message"]["content"]
        .as_str()
        .or_else(|| value["choices"][0]["text"].as_str())
        .unwrap_or_default()
        .to_string();
    if consult.guarded
        && let Some(alert) = companion::detect_hallucinations(&content, &[], &consult.files)
    {
        content.push_str(&format!(
            "\n\n---\n\n### ⚠️ [Anti-Hallucination Guard Activated]\n\n{}",
            alert.trim()
        ));
    }
    if !content.trim().is_empty() {
        content.push_str(&companion::coding_footer(&consult.config.model));
    }
    coding_json(
        &id,
        json!({"role":"assistant","content":content}),
        "stop",
        nioxs_value,
    )
}

/// Wraps one assistant message in a non-streamed completion envelope.
fn coding_json(id: &str, mut message: Value, finish: &str, mut nioxs_value: Value) -> Response {
    prepend_progress(&mut message, &nioxs_value);
    if finish == "tool_calls" {
        nioxs_value["status"]["phase"] = json!("awaiting_tools");
        nioxs_value["status"]["target"] = json!("caller");
    }
    Json(json!({"id":id,"object":"chat.completion","created":0,"model":nioxs_value["model"].as_str().unwrap_or(MODEL),"choices":[{"index":0,"message":message,"finish_reason":finish}],"nioxs":nioxs_value})).into_response()
}

/// Emits the trail, one explanatory message, and the terminal chunk. Used when
/// the upstream call fails before streaming starts.
fn coding_static_sse(
    id: String,
    trail: Vec<String>,
    message: String,
    nioxs_value: Value,
) -> Response {
    let model = nioxs_value["model"].as_str().unwrap_or(MODEL);
    let mut events: Vec<Result<Event, Infallible>> = trail
        .into_iter()
        .map(|segment| content_event(&id, model, segment))
        .collect();
    events.push(content_event(&id, model, format!("\n\n{message}")));
    let mut terminal: Value =
        serde_json::from_str(&sse_chunk(&id, model, json!({}), json!("stop")))
            .expect("valid terminal chunk");
    terminal["nioxs"] = nioxs_value;
    events.push(Ok(Event::default().data(terminal.to_string())));
    events.push(Ok(Event::default().data("[DONE]")));
    Sse::new(stream::iter(events)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use futures_util::StreamExt;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use tower::ServiceExt;

    #[test]
    fn caller_absolute_root_and_children_are_relayed_as_relative_paths() {
        let messages = vec![
            json!({"role":"system","content":"Project overview:\nRoot: /home/user/project\nFiles (partial listing):\nnio/README.md"}),
        ];
        let calls = [
            RequestedCall {
                id: "root".into(),
                name: "find_files".into(),
                arguments: json!({"path":"/home/user/project"}).to_string(),
            },
            RequestedCall {
                id: "file".into(),
                name: "read_file".into(),
                arguments: json!({"path":"/home/user/project/nio/src/main.rs"}).to_string(),
            },
        ];
        let relayed = relay_offered_coding_calls(
            &calls,
            "turn_root",
            &companion::coding_tool_schemas(),
            &messages,
        )
        .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&relayed[0].arguments).unwrap()["path"],
            "."
        );
        assert_eq!(
            serde_json::from_str::<Value>(&relayed[1].arguments).unwrap()["path"],
            "nio/src/main.rs"
        );
        for path in [
            "/home/user/project-other/file.rs",
            "/home/user/project/../private/file.rs",
            "/etc/passwd",
        ] {
            let bad = [RequestedCall {
                id: "bad".into(),
                name: "read_file".into(),
                arguments: json!({"path":path}).to_string(),
            }];
            assert!(
                relay_offered_coding_calls(
                    &bad,
                    "turn_root",
                    &companion::coding_tool_schemas(),
                    &messages
                )
                .is_err()
            );
        }
        assert!(
            relay_offered_coding_calls(&calls, "turn_root", &companion::coding_tool_schemas(), &[])
                .is_err(),
            "no root must not permit an absolute path"
        );
    }

    #[test]
    fn caller_windows_absolute_paths_are_relayed_as_relative_paths() {
        let messages = vec![
            json!({"role":"system","content":"Project overview:\nRoot: C:\\Users\\mn\\nio-labs\nFiles (partial listing):\nnio/README.md"}),
        ];
        let calls = [
            RequestedCall {
                id: "root".into(),
                name: "find_files".into(),
                arguments: json!({"path":"C:\\Users\\mn\\nio-labs"}).to_string(),
            },
            RequestedCall {
                id: "file".into(),
                name: "read_file".into(),
                arguments: json!({"path":"C:\\Users\\mn\\nio-labs\\nio\\src\\main.rs"}).to_string(),
            },
        ];
        let relayed = relay_offered_coding_calls(
            &calls,
            "turn_root",
            &companion::coding_tool_schemas(),
            &messages,
        )
        .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&relayed[0].arguments).unwrap()["path"],
            "."
        );
        assert_eq!(
            serde_json::from_str::<Value>(&relayed[1].arguments).unwrap()["path"],
            "nio/src/main.rs"
        );
    }

    #[test]
    fn relay_ids_are_unique_across_repeated_read_rounds() {
        let calls = [RequestedCall {
            id: "upstream-reused-id".into(),
            name: "read_file".into(),
            arguments: "{\"path\":\"server/src/main.rs\"}".into(),
        }];
        let tools = companion::coding_tool_schemas();
        let first = relay_offered_coding_calls(&calls, "turn_repeat", &tools, &[]).unwrap();
        let history = vec![
            json!({"role":"assistant","tool_calls":[{"id":first[0].id,"type":"function","function":{"name":calls[0].name,"arguments":calls[0].arguments}}]}),
            json!({"role":"tool","tool_call_id":first[0].id,"content":"fn main() {}"}),
        ];
        let second = relay_offered_coding_calls(&calls, "turn_repeat", &tools, &history).unwrap();
        assert_ne!(first[0].id, second[0].id);
        assert!(second[0].id.starts_with("call_nano_turn_repeat_"));
    }

    #[test]
    fn routing_status_reports_changes_without_repeating_on_tool_rounds() {
        let mut request: ChatRequest = serde_json::from_value(
            json!({"model":"nio-flow","messages":[{"role":"user","content":"fix this"}]}),
        )
        .unwrap();
        let first = routing_progress(&request, "nio-flow");
        assert_eq!(first[0]["kind"], "profile_selected");
        request.messages.push(
            serde_json::from_value(json!({"role":"assistant","content":first[0]["message"]}))
                .unwrap(),
        );
        assert_eq!(routing_progress(&request, "nio-flow"), json!([]));
        let switched = routing_progress(&request, "nio-atom");
        assert_eq!(switched[0]["kind"], "profile_switched");
        assert_eq!(switched[0]["from"], "nio-flow");
        assert_eq!(switched[0]["to"], "nio-atom");
        assert!(
            companion::strip_trail_segments(switched[0]["message"].as_str().unwrap()).is_empty()
        );
    }

    #[test]
    fn fallback_status_is_announced_once_and_tool_handoff_is_structured() {
        let config = companion::CodingConfig {
            base_url: String::new(),
            api_key: None,
            model: companion::FALLBACK_CODING_MODEL.into(),
            mode: "auto".into(),
            auto_approve: false,
            automatic_model: true,
        };
        let mut metadata =
            json!({"model":"nio-flow","progress":[],"status":{"fallback_announced":false}});
        assert!(record_upstream_fallback(&config, &mut metadata).is_some());
        assert!(record_upstream_fallback(&config, &mut metadata).is_none());
        assert_eq!(metadata["progress"].as_array().unwrap().len(), 1);
        let terminal: Value =
            serde_json::from_str(&coding_terminal("test", "tool_calls", &metadata)).unwrap();
        assert_eq!(terminal["nioxs"]["status"]["phase"], "awaiting_tools");
        assert_eq!(terminal["nioxs"]["status"]["target"], "caller");
    }

    #[test]
    fn take_sse_data_lines_splits_only_complete_payloads() {
        let mut buffer = String::from("data: {\"a\":1}\n\ndata: {\"partial");
        assert_eq!(
            take_sse_data_lines(&mut buffer),
            vec!["{\"a\":1}".to_string()]
        );
        assert_eq!(buffer, "data: {\"partial");
        buffer.push_str("\":2}\n\ndata: [DONE]\n\n");
        assert_eq!(
            take_sse_data_lines(&mut buffer),
            vec!["{\"partial\":2}".to_string()]
        );
        assert!(buffer.is_empty(), "remaining: {buffer:?}");
    }

    #[tokio::test]
    async fn routing_banner_arrives_before_upstream_request() {
        let consult = CodingConsult {
            config: companion::CodingConfig {
                base_url: "http://127.0.0.1:1/v1".into(),
                api_key: None,
                model: "mock/model".into(),
                mode: "auto".into(),
                auto_approve: false,
                automatic_model: false,
            },
            prompt: "fix code".into(),
            messages: vec![],
            tools: vec![],
            files: vec![],
            guarded: false,
        };
        let metadata = json!({"model":"nio-flow","progress":[{"kind":"profile_selected","message":"🌊 NioFlow · Planning\n"}]});
        let response = coding_sse(
            consult,
            vec![],
            "test".into(),
            metadata,
            TurnGuard {
                active: ActiveTurns::default(),
                id: "turn_banner".into(),
            },
            "turn_banner".into(),
        )
        .await;
        let mut body = response.into_body().into_data_stream();
        let first = tokio::time::timeout(Duration::from_secs(1), body.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let text = String::from_utf8(first.to_vec()).unwrap();
        assert!(text.contains("🌊 NioFlow · Planning"));
        assert!(
            !text.contains("Request to"),
            "upstream errors must follow the routing event"
        );
        let thinking = body.next().await.unwrap().unwrap();
        let thinking = String::from_utf8(thinking.to_vec()).unwrap();
        let thinking: Value =
            serde_json::from_str(thinking.trim().strip_prefix("data: ").unwrap()).unwrap();
        assert_eq!(thinking["nioxs"]["status"]["phase"], "thinking");
        assert!(
            thinking["choices"][0]["delta"]["content"].is_null(),
            "loading must not become answer text"
        );
    }

    /// A trail round must still carry the tool call the client executes next,
    /// otherwise a tool-driven `--code:` turn stalls after the scan line.
    #[test]
    fn trail_rounds_carry_their_tool_call() {
        let call = ToolCall {
            id: "call_nano_turn_test_abc".into(),
            kind: "function".into(),
            function: nioxs_contracts::FunctionCall {
                name: "find_files".into(),
                arguments: "{\"path\":\".\",\"limit\":50}".into(),
            },
        };
        let terminal = json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]});
        let payloads = trail_payloads(
            &["🛠️ scanning workspace\n".to_string()],
            Some(&call),
            &terminal,
            sse_chunk_value,
        );
        assert_eq!(payloads.len(), 4, "payloads: {payloads:?}");
        let first: Value = serde_json::from_str(&payloads[0]).unwrap();
        assert_eq!(
            first["choices"][0]["delta"]["content"],
            "🛠️ scanning workspace\n"
        );
        let second: Value = serde_json::from_str(&payloads[1]).unwrap();
        assert_eq!(
            second["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "find_files"
        );
        assert_eq!(second["choices"][0]["finish_reason"], Value::Null);
        assert_eq!(payloads.last().unwrap(), "[DONE]");
    }

    /// The chunk envelope used by both the handler and `trail_payloads`.
    fn sse_chunk_value(delta: Value, finish: Value) -> Value {
        serde_json::from_str(&sse_chunk("chatcmpl-test", MODEL, delta, finish)).unwrap()
    }

    /// Kilo streams tool calls as fragments: the id and name arrive once, the
    /// JSON arguments in pieces. They must be reassembled before relaying.
    #[test]
    fn streamed_tool_calls_are_assembled_from_fragments() {
        let mut calls = Vec::new();
        collect_sse_tool_calls(
            &json!({"choices":[{"delta":{"content":null,"tool_calls":[{"index":0,"id":"call_up_1","type":"function","function":{"name":"read_file","arguments":""}}]}}]})
                .to_string(),
            &mut calls,
        );
        collect_sse_tool_calls(
            &json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\": \"src/main.rs\""}}]}}]})
                .to_string(),
            &mut calls,
        );
        collect_sse_tool_calls(
            &json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]}}]})
                .to_string(),
            &mut calls,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments, "{\"path\": \"src/main.rs\"}");
        // Content-only payloads must not disturb the assembly.
        collect_sse_tool_calls(
            &json!({"choices":[{"delta":{"content":"hello"}}]}).to_string(),
            &mut calls,
        );
        assert_eq!(calls.len(), 1);
    }

    /// The model may only ask for the caller's own file tools, inside its own
    /// project, and every relayed call must be traceable back to the turn.
    #[test]
    fn relayed_calls_stay_inside_the_callers_project() {
        let call = |name: &str, arguments: &str| RequestedCall {
            id: "call_up_1".into(),
            name: name.into(),
            arguments: arguments.into(),
        };

        let relayed = relay_coding_calls(
            &[
                call("read_file", "{\"path\":\"src/main.rs\"}"),
                call("find_files", "{\"glob\":\"src/**/*.rs\"}"),
                call("search_code", "{\"query\":\"fn main\"}"),
                call("git_status", "{}"),
                call("git_diff", "{}"),
            ],
            "turn_abc",
        )
        .unwrap();
        assert_eq!(relayed.len(), 5);
        assert_eq!(relayed[0].hint, "📄 read_file src/main.rs\n");
        assert_eq!(relayed[1].hint, "📁 find_files src/**/*.rs\n");
        assert_eq!(relayed[2].hint, "🔍 search_code fn main\n");
        assert_eq!(relayed[3].hint, "📊 git_status\n");
        assert_eq!(relayed[4].hint, "📝 git_diff .\n");
        assert!(
            relayed[0].id.starts_with("call_nano_turn_abc_"),
            "the client must be able to recover the turn: {}",
            relayed[0].id
        );

        for bad in [
            call("delete_everything", "{\"command\":\"rm -rf /\"}"),
            call("read_file", "{\"path\":\"../../etc/passwd\"}"),
            call("read_file", "{\"path\":\"/etc/passwd\"}"),
            call("read_file", "{}"),
            call("read_file", "not json"),
        ] {
            let name = bad.name.clone();
            let arguments = bad.arguments.clone();
            assert!(
                relay_coding_calls(&[bad], "turn_abc").is_err(),
                "{name} {arguments} must not reach the client"
            );
        }
    }

    #[test]
    fn sse_delta_content_reads_openai_shapes() {
        assert_eq!(
            sse_delta_content("{\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}"),
            Some("hi".into())
        );
        assert_eq!(
            sse_delta_content("{\"choices\":[{\"message\":{\"content\":\"full\"}}]}"),
            Some("full".into())
        );
        assert_eq!(sse_delta_content("{\"choices\":[{\"delta\":{}}]}"), None);
        assert_eq!(sse_delta_content("not json"), None);
    }

    /// The provider must forward the upstream model's deltas as they arrive
    /// rather than buffering the whole answer. The mock upstream sends one event
    /// and then blocks until the test has observed it.
    #[tokio::test]
    async fn coding_stream_forwards_upstream_deltas_before_completion() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (release, waiting) = std::sync::mpsc::channel::<()>();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            let mut header_end = None;
            while header_end.is_none() {
                let n = stream.read(&mut chunk).unwrap();
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
                header_end = buf.windows(4).position(|w| w == b"\r\n\r\n");
            }
            let header_end = header_end.unwrap();
            let headers = String::from_utf8_lossy(&buf[..header_end]).into_owned();
            let content_length: usize = headers
                .lines()
                .find_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    key.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse().ok())?
                })
                .unwrap();
            let body_start = header_end + 4;
            while buf.len() < body_start + content_length {
                let n = stream.read(&mut chunk).unwrap();
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
            }
            let request: Value =
                serde_json::from_slice(&buf[body_start..body_start + content_length]).unwrap();
            assert_eq!(
                request["stream"], true,
                "upstream call must request streaming"
            );

            let first = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"First line.\\n\"}}]}\n\n";
            let rest = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Second line.\\n\"}}]}\n\ndata: [DONE]\n\n";
            let full = format!("{first}{rest}");
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        full.len()
                    )
                    .as_bytes(),
                )
                .unwrap();
            stream.write_all(first.as_bytes()).unwrap();
            stream.flush().unwrap();
            let _ = waiting.recv_timeout(Duration::from_secs(5));
            stream.write_all(rest.as_bytes()).unwrap();
            stream.flush().unwrap();
        });

        let consult = CodingConsult {
            config: companion::CodingConfig {
                base_url: format!("http://{addr}"),
                api_key: None,
                model: "mock/model".into(),
                mode: "coding".into(),
                auto_approve: false,
                automatic_model: false,
            },
            prompt: "analyze my project".into(),
            messages: companion::upstream_messages("analyze my project", &[]),
            tools: companion::coding_tool_schemas(),
            files: Vec::new(),
            guarded: false,
        };
        let trail = vec![companion::trail_consult_segment("mock/model")];
        let response = coding_sse(
            consult,
            trail,
            "chatcmpl-test".into(),
            json!({"intent": "coding", "model": "nio-neon"}),
            TurnGuard {
                active: ActiveTurns::default(),
                id: "turn_stream_test".into(),
            },
            "turn_stream_test".into(),
        )
        .await;

        let mut body = response.into_body().into_data_stream();
        let mut raw = String::new();
        let mut parse = String::new();
        let mut deltas: Vec<String> = Vec::new();
        let mut answered_before_completion = false;
        let mut released = false;
        while let Some(chunk) = body.next().await {
            let text = String::from_utf8_lossy(&chunk.unwrap()).into_owned();
            raw.push_str(&text);
            parse.push_str(&text);
            for payload in take_sse_data_lines(&mut parse) {
                let value: Value = serde_json::from_str(&payload).unwrap();
                assert_eq!(
                    value["model"], "nio-neon",
                    "every stream chunk preserves the profile"
                );
                if let Some(content) = value["choices"][0]["delta"]["content"].as_str() {
                    deltas.push(content.to_string());
                    if content.contains("First line.") && !released {
                        answered_before_completion = true;
                        released = true;
                        release.send(()).unwrap();
                    }
                }
            }
        }
        server.join().unwrap();

        assert!(
            answered_before_completion,
            "a delta must reach the client before the upstream answer completes; deltas: {deltas:?}"
        );
        assert!(raw.contains("[DONE]"), "stream must terminate");
        assert!(raw.contains("\"finish_reason\":\"stop\""));
        assert!(raw.contains("\"coding\""), "decision payload must survive");
        assert_eq!(
            trail_len(&deltas),
            0,
            "no progress banner is emitted: {deltas:?}"
        );
        assert!(
            !deltas
                .iter()
                .any(|delta| delta.starts_with(companion::TRAIL_CONSULT))
        );
        assert!(deltas.contains(&"First line.\n".to_string()), "{deltas:?}");
        assert!(deltas.contains(&"Second line.\n".to_string()), "{deltas:?}");
        assert_eq!(
            deltas.iter().position(|d| d.contains("First line.")),
            deltas
                .iter()
                .position(|d| d.contains("Second line."))
                .map(|p| p - 1),
            "upstream order must be preserved: {deltas:?}"
        );
    }

    #[test]
    fn test_extract_xml_tool_calls_recovers_dots_function_call() {
        let sample = r#"
<dots_function_call>
<invoke name="find_files">
<parameter name="glob">
src/prepare*
</parameter>
</invoke>
</dots_function_call>
<dots_function_call>
<invoke name="find_files">
<parameter name="glob">
src/engine*
</parameter>
</invoke>
</dots_function_call>
"#;
        let calls = extract_xml_tool_calls(sample);
        assert_eq!(calls.len(), 2, "must extract both invoke calls: {calls:?}");
        assert_eq!(calls[0].name, "find_files");
        assert_eq!(calls[0].arguments, "{\"glob\":\"src/prepare*\"}");
        assert_eq!(calls[1].name, "find_files");
        assert_eq!(calls[1].arguments, "{\"glob\":\"src/engine*\"}");

        let single =
            r#"<invoke name="read_file"><parameter name="path">src/mcp.rs</parameter></invoke>"#;
        let read_calls = extract_xml_tool_calls(single);
        assert_eq!(read_calls.len(), 1);
        assert_eq!(read_calls[0].name, "read_file");
        assert_eq!(read_calls[0].arguments, "{\"path\":\"src/mcp.rs\"}");

        // ChatML / Qwen <|tool_call_start|> pythonic syntax
        let chatml_sample =
            "<|tool_call_start|>[read_file(path='README.md', limit=50)]<|tool_call_end|>";
        let chatml_calls = extract_xml_tool_calls(chatml_sample);
        assert_eq!(chatml_calls.len(), 1);
        assert_eq!(chatml_calls[0].name, "read_file");
        assert_eq!(
            chatml_calls[0].arguments,
            "{\"limit\":50,\"path\":\"README.md\"}"
        );

        // Multiple pythonic calls in <|tool_call_start|>
        let multi_sample = "<|tool_call_start|>[read_file(path='a.txt'), list_dir(path='crates')]<|tool_call_end|>";
        let multi_calls = extract_xml_tool_calls(multi_sample);
        assert_eq!(multi_calls.len(), 2);
        assert_eq!(multi_calls[0].name, "read_file");
        assert_eq!(multi_calls[1].name, "list_dir");

        // JSON inside <|tool_call_start|>
        let json_chatml = r#"<|tool_call_start|>[{"name": "read_file", "arguments": {"path": "README.md", "limit": 50}}]<|tool_call_end|>"#;
        let json_calls = extract_xml_tool_calls(json_chatml);
        assert_eq!(json_calls.len(), 1);
        assert_eq!(json_calls[0].name, "read_file");
        assert_eq!(
            json_calls[0].arguments,
            "{\"limit\":50,\"path\":\"README.md\"}"
        );
    }

    fn trail_len(deltas: &[String]) -> usize {
        deltas
            .iter()
            .take_while(|delta| !delta.contains("First line.") && !delta.contains("Second line."))
            .count()
    }
    fn app() -> Router {
        router(App {
            db: NioDb::new("http://127.0.0.1:9", "test".into()).unwrap(),
            deployment: "test".into(),
            api_token: None,
            active: ActiveTurns::default(),
            model: Arc::new(Model::embedded().unwrap()),
            ray_engine: None,
            instance: "unit-test".into(),
        })
    }
    #[tokio::test]
    async fn model_catalog_exposes_profiles_and_auto() {
        let response = app()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let value: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 8192).await.unwrap()).unwrap();
        let ids: Vec<&str> = value["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, MODELS);
    }

    #[test]
    fn profile_identity_survives_terminal_stream_chunks() {
        let value: Value = serde_json::from_str(&coding_terminal("id", "tool_calls", &json!({"model":"nio-atom","routing":{"selected_model":"nio-atom"},"upstream_model":"cohere/north-mini-code:free"}))).unwrap();
        assert_eq!(value["model"], "nio-atom");
        assert_eq!(value["nioxs"]["routing"]["selected_model"], "nio-atom");
        assert_eq!(
            value["nioxs"]["upstream_model"],
            "cohere/north-mini-code:free"
        );
    }
    #[test]
    fn upstream_cannot_relay_tools_that_were_not_offered() {
        let call = RequestedCall {
            name: "read_file".into(),
            arguments: "{\"path\":\"README.md\"}".into(),
            ..Default::default()
        };
        assert!(relay_offered_coding_calls(&[call.clone()], "turn_test", &[], &[]).is_err());
        assert!(
            relay_offered_coding_calls(
                &[call],
                "turn_test",
                &companion::coding_tool_schemas(),
                &[]
            )
            .is_ok()
        );
    }
    #[tokio::test]
    async fn unavailable_storage_does_not_dispatch_a_tool() {
        let response = app().oneshot(Request::builder().method("POST").uri("/v1/chat/completions").header("content-type","application/json")
            .body(Body::from(json!({"model":MODEL,"messages":[{"role":"user","content":"read README.md"}],"tools":[{"type":"function","function":{"name":"read_file"}}]}).to_string())).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(response.into_body(), 8192).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["code"], "storage_unavailable");
        assert!(value.get("choices").is_none());
    }
    #[tokio::test]
    async fn ready_reports_the_research_cache_and_quota() {
        let (db_url, _store) = crate::testdb::mock_niodb().await;
        let secured = router(App {
            db: NioDb::new(&db_url, "test".into()).unwrap(),
            deployment: "test".into(),
            api_token: None,
            active: ActiveTurns::default(),
            model: Arc::new(Model::embedded().unwrap()),
            ray_engine: None,
            instance: "unit-test".into(),
        });
        let response = secured
            .oneshot(
                Request::builder()
                    .uri("/ready")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 8192).await.unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["research"]["cache_records"], 0);
        assert_eq!(value["research"]["evidence_records"], 0);
        assert_eq!(
            value["research"]["cache_limit"],
            nioxs_storage::RESEARCH_CACHE_LIMIT
        );
    }
    #[tokio::test]
    async fn catalog_is_honest_about_the_baseline_and_rejects_auth_failures() {
        let response = app()
            .oneshot(
                Request::builder()
                    .uri("/v1/capabilities")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = to_bytes(response.into_body(), 16384).await.unwrap();
        let caps: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(caps["trained"], true);
        // The advertised surface must match what actually shipped, including MCP.
        assert_eq!(caps["mcp"], true);
        let workflows = caps["workflows"].as_array().expect("workflows array");
        for expected in [
            "coding",
            "git_summary",
            "project_suggestions",
            "task_plan",
            "capabilities",
            "git_inspect",
            "diagnostics",
            "project_tasks",
        ] {
            assert!(
                workflows.iter().any(|w| w == expected),
                "capabilities must advertise the shipped workflow '{expected}': {workflows:?}"
            );
        }
        let registry = caps["tool_registry"]
            .as_array()
            .expect("typed tool registry");
        assert!(registry.iter().all(|entry| entry["execution"] == "caller"));
        assert!(
            registry
                .iter()
                .any(|entry| entry["schema"]["function"]["name"] == "run_command"
                    && entry["permission"] == "command_approval")
        );
        assert_eq!(caps["workflow_registry"].as_array().unwrap().len(), 13);
        for name in [
            "symbol_definitions",
            "symbol_references",
            "symbol_callers",
            "symbol_implementations",
        ] {
            assert!(
                caps["workflow_registry"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|w| w["name"] == name && w["permission"] == "read")
            );
        }
        let secure = router(App {
            db: NioDb::new("http://127.0.0.1:9", "test".into()).unwrap(),
            deployment: "test".into(),
            api_token: Some("private".into()),
            active: ActiveTurns::default(),
            model: Arc::new(Model::embedded().unwrap()),
            ray_engine: None,
            instance: "unit-test".into(),
        });
        assert_eq!(
            secure
                .oneshot(
                    Request::builder()
                        .uri("/v1/models")
                        .body(Body::empty())
                        .unwrap()
                )
                .await
                .unwrap()
                .status(),
            StatusCode::UNAUTHORIZED
        );
    }
    async fn post(path: &str, body: Value) -> (StatusCode, Value) {
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(path)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 65536).await.unwrap();
        let value = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).unwrap_or(Value::Null)
        };
        (status, value)
    }

    #[tokio::test]
    async fn decide_endpoint_scores_proposals_without_executing_them() {
        let (status, value) = post(
            "/v1/decide",
            json!({
                "schema_version": 1,
                "prompt": "read README.md",
                "proposals": [
                    {"id": "read", "action": "read_file", "arguments": {"path": "README.md"}},
                    {"id": "answer", "action": "answer"}
                ],
                "allowed_actions": ["read_file"],
                "required_fields": ["path"]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(value["outcome"], "execute");
        assert_eq!(value["chosen"]["action"], "read_file");
        assert_eq!(value["chosen"]["id"], "read");
        assert!(value["confidence"].as_f64().unwrap() >= 0.5);
        assert!(value["abstained_reason"].is_null());
        // Advisory only: no completion envelope and no dispatched action.
        assert!(value.get("choices").is_none());
        assert!(value.get("tool_calls").is_none());
    }

    #[tokio::test]
    async fn decide_endpoint_abstains_on_policy_and_clarifies_missing_fields() {
        let (status, blocked) = post(
            "/v1/decide",
            json!({
                "schema_version": 1,
                "prompt": "run tests",
                "proposals": [{"id": "run", "action": "run_command", "arguments": {"command": "cargo test"}}],
                "allowed_actions": ["read_file"]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(blocked["outcome"], "abstain");
        assert_eq!(blocked["policy"]["denied"][0], "run_command");
        assert!(blocked["chosen"].is_null());

        let (status, clarify) = post(
            "/v1/decide",
            json!({
                "schema_version": 1,
                "prompt": "read it",
                "proposals": [{"id": "read", "action": "read_file", "arguments": {}}],
                "allowed_actions": ["read_file"],
                "required_fields": ["path"]
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(clarify["outcome"], "clarify");
        assert!(clarify["chosen"].is_null());
        assert!(
            clarify["abstained_reason"]
                .as_str()
                .unwrap()
                .contains("path")
        );
    }

    #[tokio::test]
    async fn decide_endpoint_rejects_invalid_requests_and_requires_auth() {
        let (status, value) = post(
            "/v1/decide",
            json!({"schema_version": 2, "prompt": "read README.md"}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(value["error"]["code"], "invalid_request_error");

        let secure = router(App {
            db: NioDb::new("http://127.0.0.1:9", "test".into()).unwrap(),
            deployment: "test".into(),
            api_token: Some("private".into()),
            active: ActiveTurns::default(),
            model: Arc::new(Model::embedded().unwrap()),
            ray_engine: None,
            instance: "unit-test".into(),
        });
        let response = secure
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/decide")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({"schema_version": 1, "prompt": "read README.md"}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
}
