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
use futures_util::stream;
use nioxs_contracts::{
    ChatRequest, DecideRequest, DurableEvent, MAX_REQUEST_BYTES, MODEL, Outcome, PROVIDER,
    SCHEMA_VERSION,
};
use nioxs_nano::{Model, decide_with_model};
mod journal;
pub mod mcp;
pub mod niodb_supervisor;
use nioxs_storage::NioDb;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    convert::Infallible,
    sync::{Arc, Mutex},
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
    Json(json!({"schema_version":SCHEMA_VERSION,"provider":PROVIDER,"model":MODEL,"implementation":"trained_naive_bayes_and_typed_workflows","model_version":app.model.model_version,"trained":true,"confidence_threshold":app.model.threshold,"workflows":["read_file","inspect_project","find_files","search_code","run_checks","web_fetch","research_sources"],"client_tools":["read_file","find_files","search_code","run_command","web_fetch"],"server_tools":[],"streaming":true,"automatic_fallback":false,"browser_execution":true,"mcp":false,"decision_contract":true,"storage":"niodb_immutable_action_journal"})).into_response()
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
    let _guard = TurnGuard {
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
    let mut delta = message;
    if let Some(call) = &decision.call {
        delta["tool_calls"] =
            json!([{"index":0,"id":call.id,"type":"function","function":call.function}]);
    }
    let chunk = |delta: Value, finish: Value| json!({"id":id,"object":"chat.completion.chunk","created":0,"model":MODEL,"choices":[{"index":0,"delta":delta,"finish_reason":finish}]});
    let mut terminal = chunk(json!({}), json!(finish));
    terminal["nioxs"] = nioxs_value;
    let events = vec![
        Ok::<_, Infallible>(Event::default().data(chunk(delta, Value::Null).to_string())),
        Ok(Event::default().data(terminal.to_string())),
        Ok(Event::default().data("[DONE]")),
    ];
    Sse::new(stream::iter(events)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use tower::ServiceExt;
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
        assert_eq!(
            serde_json::from_slice::<Value>(&bytes).unwrap()["trained"],
            true
        );
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
