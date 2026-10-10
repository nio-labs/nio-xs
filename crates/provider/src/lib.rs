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
    ChatRequest, DurableEvent, MAX_REQUEST_BYTES, MODEL, PROVIDER, SCHEMA_VERSION,
};
use nioxs_nano::{Model, decide_with_model};
mod journal;
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
    Json(json!({"object":"list","data":[{"id":MODEL,"object":"model","created":0,"owned_by":PROVIDER,"name":"NioNano (trained project workflows)","type":"language","free":true}]})).into_response()
}

async fn capabilities(State(app): State<Arc<App>>, headers: HeaderMap) -> Response {
    if !authorized(&app, &headers) {
        return error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "provider authentication required",
        );
    }
    Json(json!({"schema_version":SCHEMA_VERSION,"provider":PROVIDER,"model":MODEL,"implementation":"trained_naive_bayes_and_typed_workflows","model_version":app.model.model_version,"trained":true,"confidence_threshold":app.model.threshold,"workflows":["read_file","inspect_project","find_files","search_code","run_checks","web_fetch","research_sources"],"client_tools":["read_file","find_files","search_code","run_command","web_fetch"],"server_tools":[],"streaming":true,"automatic_fallback":false,"browser_execution":true,"mcp":false,"storage":"niodb_immutable_action_journal"})).into_response()
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
        return Json(json!({"id":id,"object":"chat.completion","created":0,"model":MODEL,"choices":[{"index":0,"message":message,"finish_reason":finish}],"nioxs":decision.execution})).into_response();
    }
    let mut delta = message;
    if let Some(call) = &decision.call {
        delta["tool_calls"] =
            json!([{"index":0,"id":call.id,"type":"function","function":call.function}]);
    }
    let chunk = |delta: Value, finish: Value| json!({"id":id,"object":"chat.completion.chunk","created":0,"model":MODEL,"choices":[{"index":0,"delta":delta,"finish_reason":finish}]});
    let mut terminal = chunk(json!({}), json!(finish));
    terminal["nioxs"] = json!(decision.execution);
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
}
