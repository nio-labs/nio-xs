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
    ChatRequest, DecideRequest, DurableEvent, MAX_REQUEST_BYTES, MODEL, Outcome, PROVIDER,
    SCHEMA_VERSION, ToolCall,
};
use nioxs_nano::companion::{self, CodingConsult};
use nioxs_nano::{Model, decide_with_model};
pub mod journal;
pub mod mcp;
pub mod niodb_supervisor;
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
        Ok(_) => Json(json!({"status":"ready","storage":"ready","nano":"trained_naive_bayes","model_version":app.model.model_version,"schema_version":SCHEMA_VERSION})).into_response(),
        Err(_) => error(StatusCode::SERVICE_UNAVAILABLE,"storage_unavailable","NioDB is not ready"),
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
    Json(json!({"object":"list","data":[{"id":MODEL,"object":"model","created":0,"owned_by":PROVIDER,"name":"NioNano MicroLM (μLM)","type":"language","free":true}]})).into_response()
}

async fn capabilities(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    if !authorized(&app, &headers) {
        return error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "provider authentication required",
        );
    }
    Json(json!({"schema_version":SCHEMA_VERSION,"provider":PROVIDER,"model":MODEL,"implementation":"trained_naive_bayes_and_typed_workflows","model_version":app.model.model_version,"trained":true,"confidence_threshold":app.model.threshold,"workflows":nioxs_nano::SHIPPED_WORKFLOWS,"client_tools":["read_file","find_files","search_code","run_command","web_fetch"],"server_tools":[],"streaming":true,"automatic_fallback":false,"browser_execution":true,"mcp":true,"decision_contract":true,"storage":"niodb_immutable_action_journal"})).into_response()
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
    let mut event = DurableEvent {
        schema_version: SCHEMA_VERSION,
        deployment_id: app.deployment.clone(),
        turn_id: decision.execution.turn_id.clone(),
        event: format!("{:?}", decision.execution.outcome).to_lowercase(),
        model: MODEL.into(),
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
        value
    };
    // A `--code:` turn hands back a pending upstream consultation instead of a
    // pre-computed answer. The model is asked with the caller's file tools and
    // its own deltas are forwarded live, so it decides what to read and the
    // client executes those reads in the directory it was started in.
    if let Some(consult) = decision.coding.clone() {
        let turn_id = decision.execution.turn_id.clone();
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
    if let Some(call) = &decision.call {
        message["tool_calls"] = json!([call]);
    }
    if !request.stream {
        return Json(json!({"id":id,"object":"chat.completion","created":0,"model":MODEL,"choices":[{"index":0,"message":message,"finish_reason":finish}],"nioxs":nioxs_value})).into_response();
    }
    let chunk = |delta: Value, finish: Value| json!({"id":id,"object":"chat.completion.chunk","created":0,"model":MODEL,"choices":[{"index":0,"delta":delta,"finish_reason":finish}]});
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
    ];
    tags.iter()
        .filter_map(|tag| lower.find(tag))
        .min()
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
            let trimmed = after_name.trim().trim_start_matches(|c| c == '"' || c == '\'');
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
            let Some(tag_open_end) = text[tag_name_start..].find('>').map(|i| tag_name_start + i) else {
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
                    let tag_start = if is_attr { abs_param + 16 } else { abs_param + 11 };
                    let Some(tag_end) = block_content[tag_start..].find('>') else { break; };
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
                    if param_cursor >= block_content.len() { break; }
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

            cur = if block_end < text.len() { block_end + 11 } else { text.len() };
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
            if block.starts_with('{') {
                if let Ok(val) = serde_json::from_str::<Value>(block) {
                    if let Some(name) = val.get("name").and_then(Value::as_str) {
                        let arguments = val.get("arguments").cloned().unwrap_or(json!({}));
                        calls.push(RequestedCall {
                            id: format!("call_xml_{}", calls.len()),
                            name: name.to_string(),
                            arguments: arguments.to_string(),
                        });
                    }
                }
            }
            cur = if end < text.len() { end + 12 } else { text.len() };
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

fn sse_chunk(id: &str, delta: Value, finish: Value) -> String {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })
    .to_string()
}

fn content_event(id: &str, content: String) -> Result<Event, Infallible> {
    Ok(Event::default().data(sse_chunk(id, json!({"content": content}), Value::Null)))
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
    let url = companion::upstream_chat_url(&consult.config);
    let body =
        companion::upstream_chat_body(&consult.config, &consult.messages, &consult.tools, true);
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
    let mut request = client.post(&url).json(&body);
    if let Some(key) = consult.config.api_key.as_deref() {
        request = request.bearer_auth(key);
    }
    let response = match request.send().await {
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
        .map(|segment| content_event(&id_trail, segment))
        .collect();
    let id_live = id.clone();
    let live_events = live.map(move |delta| content_event(&id_live, delta));

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
            return match relay_coding_calls(&calls, &tail_turn) {
                Ok(relayed) => {
                    let mut events: Vec<Result<Event, Infallible>> = Vec::new();
                    for (index, call) in relayed.iter().enumerate() {
                        events.push(Ok(Event::default().data(call_delta(&id_tail, call, index))));
                    }
                    events.push(Ok(Event::default().data(coding_terminal(
                        &id_tail,
                        "tool_calls",
                        &tail_value,
                    ))));
                    events.push(Ok(Event::default().data("[DONE]")));
                    events
                }
                Err(reason) => {
                    let mut events = vec![content_event(
                        &id_tail,
                        format!("\n\n⚠️ {reason}; no files were read."),
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
            events.push(content_event(&id_tail, closing));
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
    let mut terminal: Value = serde_json::from_str(&sse_chunk(id, json!({}), json!(finish)))
        .expect("valid terminal chunk");
    terminal["nioxs"] = nioxs_value.clone();
    terminal.to_string()
}

/// One client-visible `tool_calls` delta for a relayed model request.
fn call_delta(id: &str, call: &ClientCall, index: usize) -> String {
    sse_chunk(
        id,
        json!({"tool_calls":[{"index":index,"id":call.id,"type":"function","function":{"name":call.name,"arguments":call.arguments}}]}),
        Value::Null,
    )
}

/// Consults the upstream model for a caller that does not stream.
///
/// The answer is returned as one message, or as the model's own tool calls when
/// it wants to read the caller's project first.
async fn coding_response(
    consult: CodingConsult,
    id: String,
    nioxs_value: Value,
    turn_id: String,
) -> Response {
    let url = companion::upstream_chat_url(&consult.config);
    let body =
        companion::upstream_chat_body(&consult.config, &consult.messages, &consult.tools, false);
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
    let mut request = client.post(&url).json(&body);
    if let Some(key) = consult.config.api_key.as_deref() {
        request = request.bearer_auth(key);
    }
    let value: Value = match request.send().await {
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

    let mut calls = message_tool_calls(&value);
    if calls.is_empty() {
        if let Some(content) = value["choices"][0]["message"]["content"].as_str() {
            calls = extract_xml_tool_calls(content);
        }
    }
    if !calls.is_empty() {
        return match relay_coding_calls(&calls, &turn_id) {
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
            Err(reason) => coding_json(
                &id,
                json!({"role":"assistant","content":format!("⚠️ {reason}; no files were read.")}),
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
fn coding_json(id: &str, message: Value, finish: &str, nioxs_value: Value) -> Response {
    Json(json!({"id":id,"object":"chat.completion","created":0,"model":MODEL,"choices":[{"index":0,"message":message,"finish_reason":finish}],"nioxs":nioxs_value})).into_response()
}

/// Emits the trail, one explanatory message, and the terminal chunk. Used when
/// the upstream call fails before streaming starts.
fn coding_static_sse(
    id: String,
    trail: Vec<String>,
    message: String,
    nioxs_value: Value,
) -> Response {
    let mut events: Vec<Result<Event, Infallible>> = trail
        .into_iter()
        .map(|segment| content_event(&id, segment))
        .collect();
    events.push(content_event(&id, format!("\n\n{message}")));
    let mut terminal: Value = serde_json::from_str(&sse_chunk(&id, json!({}), json!("stop")))
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
        serde_json::from_str(&sse_chunk("chatcmpl-test", delta, finish)).unwrap()
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
            call("run_command", "{\"command\":\"rm -rf /\"}"),
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
            json!({"intent": "coding"}),
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
            deltas[..trail_len(&deltas)],
            ["💭 consulting with LLM: mock/model\n".to_string()][..],
            "trail lines must stream first, one per event: {deltas:?}"
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

        let single = r#"<invoke name="read_file"><parameter name="path">src/mcp.rs</parameter></invoke>"#;
        let read_calls = extract_xml_tool_calls(single);
        assert_eq!(read_calls.len(), 1);
        assert_eq!(read_calls[0].name, "read_file");
        assert_eq!(read_calls[0].arguments, "{\"path\":\"src/mcp.rs\"}");
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
            instance: "unit-test".into(),
        })
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
        let bytes = to_bytes(response.into_body(), 8192).await.unwrap();
        let caps: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(caps["trained"], true);
        // The advertised surface must match what actually shipped, including MCP.
        assert_eq!(caps["mcp"], true);
        let workflows = caps["workflows"].as_array().expect("workflows array");
        for expected in ["coding", "git_summary", "project_suggestions", "task_plan"] {
            assert!(
                workflows.iter().any(|w| w == expected),
                "capabilities must advertise the shipped workflow '{expected}': {workflows:?}"
            );
        }
        let secure = router(App {
            db: NioDb::new("http://127.0.0.1:9", "test".into()).unwrap(),
            deployment: "test".into(),
            api_token: Some("private".into()),
            active: ActiveTurns::default(),
            model: Arc::new(Model::embedded().unwrap()),
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
