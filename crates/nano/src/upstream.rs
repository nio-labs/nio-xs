//! Kilo catalog selection and pre-stream availability fallback.
use crate::companion::{
    CodingConfig, DEFAULT_CODING_MODEL, FALLBACK_CODING_MODEL, upstream_chat_body,
    upstream_chat_url,
};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

struct CatalogCache {
    model: String,
    stepfun: Option<String>,
    expires: Instant,
}
static CACHE: OnceLock<Mutex<HashMap<(String, String), CatalogCache>>> = OnceLock::new();

fn catalog_model(value: &Value) -> &'static str {
    if value["data"]
        .as_array()
        .is_some_and(|models| models.iter().any(|m| m["id"] == DEFAULT_CODING_MODEL))
    {
        DEFAULT_CODING_MODEL
    } else {
        FALLBACK_CODING_MODEL
    }
}

/// Only select a free StepFun entry actually returned by this gateway.
fn catalog_stepfun(value: &Value) -> Option<String> {
    let zero = |price: &Value| {
        price
            .as_f64()
            .or_else(|| price.as_str().and_then(|s| s.parse::<f64>().ok()))
            == Some(0.0)
    };
    value["data"]
        .as_array()?
        .iter()
        .filter(|model| {
            let id = model["id"].as_str().unwrap_or("");
            id.starts_with("stepfun/")
                && (id.ends_with(":free")
                    || id.ends_with("-free")
                    || id.ends_with("/free")
                    || (zero(&model["pricing"]["prompt"]) && zero(&model["pricing"]["completion"])))
                && model["supported_parameters"]
                    .as_array()
                    .is_some_and(|parameters| {
                        parameters.iter().any(|parameter| parameter == "tools")
                    })
        })
        .max_by_key(|model| model["context_length"].as_u64().unwrap_or(0))
        .and_then(|model| model["id"].as_str())
        .map(str::to_string)
}

/// Checks the catalog with a short timeout. Cache successes for five minutes,
/// failures for thirty seconds; never hold a lock across a network request.
pub async fn select_model(client: &reqwest::Client, config: &mut CodingConfig) {
    if !config.automatic_model {
        return;
    }
    let endpoint = format!("{}/models", config.base_url.trim_end_matches('/'));
    let credential_digest =
        nioxs_contracts::digest(config.api_key.as_deref().unwrap_or("").as_bytes());
    let key = (endpoint.clone(), credential_digest);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(cached) = cache.lock()
        && let Some(entry) = cached.get(&key)
        && Instant::now() < entry.expires
    {
        config.model = entry.model.clone();
        return;
    }
    let mut request = client.get(&endpoint).timeout(Duration::from_secs(3));
    if let Some(key) = &config.api_key {
        request = request.bearer_auth(key);
    }
    let catalog = match request.send().await {
        Ok(response) if response.status().is_success() => response
            .json::<Value>()
            .await
            .ok()
            .filter(|value| value["data"].is_array()),
        _ => None,
    };
    config.model = catalog
        .as_ref()
        .map(catalog_model)
        .unwrap_or(FALLBACK_CODING_MODEL)
        .into();
    if let Ok(mut cached) = cache.lock() {
        cached.retain(|_, entry| Instant::now() < entry.expires);
        if cached.len() >= 16 {
            cached.clear();
        }
        cached.insert(
            key,
            CatalogCache {
                model: config.model.clone(),
                stepfun: catalog.as_ref().and_then(catalog_stepfun),
                expires: Instant::now()
                    + Duration::from_secs(if catalog.is_some() { 300 } else { 30 }),
            },
        );
    }
}

async fn post(
    client: &reqwest::Client,
    config: &CodingConfig,
    messages: &[Value],
    tools: &[Value],
    stream: bool,
) -> Result<reqwest::Response, reqwest::Error> {
    let mut request = client
        .post(upstream_chat_url(config))
        .json(&upstream_chat_body(config, messages, tools, stream));
    if let Some(key) = &config.api_key {
        request = request.bearer_auth(key);
    }
    request.send().await
}

/// Retry unavailable North Mini Code once using Auto Free, before any response
/// bytes or tool calls are relayed. Explicit model choices are never replaced.
pub async fn send_chat(
    client: &reqwest::Client,
    config: &mut CodingConfig,
    messages: &[Value],
    tools: &[Value],
    stream: bool,
) -> Result<reqwest::Response, reqwest::Error> {
    send_chat_for_complexity(client, config, messages, tools, stream, false).await
}

pub async fn send_chat_for_complexity(
    client: &reqwest::Client,
    config: &mut CodingConfig,
    messages: &[Value],
    tools: &[Value],
    stream: bool,
    high_complexity: bool,
) -> Result<reqwest::Response, reqwest::Error> {
    select_model(client, config).await;
    if high_complexity && config.automatic_model {
        let key = (
            format!("{}/models", config.base_url.trim_end_matches('/')),
            nioxs_contracts::digest(config.api_key.as_deref().unwrap_or("").as_bytes()),
        );
        if let Ok(cache) = CACHE.get_or_init(|| Mutex::new(HashMap::new())).lock()
            && let Some(stepfun) = cache.get(&key).and_then(|entry| entry.stepfun.as_ref())
        {
            config.model = stepfun.clone();
        }
    }
    let response = post(client, config, messages, tools, stream).await;
    let unavailable = match &response {
        Ok(r) => matches!(
            r.status().as_u16(),
            402 | 404 | 410 | 429 | 500 | 502 | 503 | 504
        ),
        Err(_) => true,
    };
    if config.automatic_model
        && (config.model == DEFAULT_CODING_MODEL || config.model.starts_with("stepfun/"))
        && unavailable
    {
        let failed_stepfun = config.model.starts_with("stepfun/");
        config.model = FALLBACK_CODING_MODEL.into();
        // Avoid selecting the unavailable preferred model again on the next tool round.
        if let Ok(mut cached) = CACHE.get_or_init(|| Mutex::new(HashMap::new())).lock()
            && let Some(entry) = cached.get_mut(&(
                format!("{}/models", config.base_url.trim_end_matches('/')),
                nioxs_contracts::digest(config.api_key.as_deref().unwrap_or("").as_bytes()),
            ))
        {
            if failed_stepfun {
                entry.stepfun = None;
            }
            entry.model = config.model.clone();
            entry.expires = Instant::now() + Duration::from_secs(30);
        }
        return post(client, config, messages, tools, stream).await;
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn gateway(
        replies: Vec<(u16, Value)>,
    ) -> (String, tokio::task::JoinHandle<Vec<(String, Value)>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let mut requests = Vec::new();
            for (status, body) in replies {
                let (mut socket, _) =
                    tokio::time::timeout(Duration::from_secs(5), listener.accept())
                        .await
                        .unwrap()
                        .unwrap();
                let mut bytes = Vec::new();
                let (headers, offset, length) = loop {
                    let mut chunk = [0u8; 4096];
                    let n = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut chunk))
                        .await
                        .unwrap()
                        .unwrap();
                    assert!(n > 0);
                    bytes.extend_from_slice(&chunk[..n]);
                    if let Some(offset) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                        let headers = String::from_utf8(bytes[..offset].to_vec()).unwrap();
                        let length = headers
                            .lines()
                            .find_map(|line| {
                                line.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(|v| v.trim().parse::<usize>().unwrap())
                            })
                            .unwrap_or(0);
                        break (headers, offset + 4, length);
                    }
                };
                while bytes.len() < offset + length {
                    let mut chunk = [0u8; 4096];
                    let n = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut chunk))
                        .await
                        .unwrap()
                        .unwrap();
                    assert!(n > 0);
                    bytes.extend_from_slice(&chunk[..n]);
                }
                let path = headers
                    .lines()
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .nth(1)
                    .unwrap()
                    .to_string();
                let request_body = if length == 0 {
                    Value::Null
                } else {
                    serde_json::from_slice(&bytes[offset..offset + length]).unwrap()
                };
                requests.push((path, request_body));
                let body = body.to_string();
                let response = format!(
                    "HTTP/1.1 {status} Mock\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
            requests
        });
        (base, server)
    }

    fn config(base_url: String, automatic_model: bool) -> CodingConfig {
        // Avoid test-cache collisions when the OS quickly reuses a loopback port.
        let api_key = Some(format!("test-{base_url}"));
        CodingConfig {
            base_url,
            api_key,
            model: DEFAULT_CODING_MODEL.into(),
            mode: "coding".into(),
            auto_approve: false,
            automatic_model,
        }
    }

    #[tokio::test]
    async fn catalog_selection_is_cached_and_preserves_tool_traffic() {
        let (base, server) = gateway(vec![
            (200, json!({"data":[{"id":DEFAULT_CODING_MODEL}]})),
            (200, json!({})),
            (200, json!({})),
        ])
        .await;
        let mut cfg = config(base, true);
        let client = reqwest::Client::new();
        let messages = json!([{"role":"user","content":"build auth"},{"role":"assistant","tool_calls":[{"id":"a"}]},{"role":"tool","tool_call_id":"a","content":"real file"}]);
        let tools = json!([{"type":"function","function":{"name":"read_file"}}]);
        for stream in [true, false] {
            assert!(
                send_chat(
                    &client,
                    &mut cfg,
                    messages.as_array().unwrap(),
                    tools.as_array().unwrap(),
                    stream
                )
                .await
                .unwrap()
                .status()
                .is_success()
            );
        }
        let requests = server.await.unwrap();
        assert_eq!(requests[0].0, "/models");
        assert_eq!(requests.len(), 3);
        assert_eq!(requests[1].1["model"], DEFAULT_CODING_MODEL);
        assert_eq!(requests[1].1["messages"], messages);
        assert_eq!(requests[1].1["tools"], tools);
        assert_eq!(requests[1].1["stream"], true);
        assert_eq!(requests[2].1["stream"], false);
    }

    #[tokio::test]
    async fn missing_or_failed_catalog_uses_auto_free() {
        for (status, catalog) in [
            (200, json!({"data":[]})),
            (503, json!({"error":"down"})),
            (200, json!({"unexpected":[]})),
        ] {
            let (base, server) = gateway(vec![(status, catalog), (200, json!({}))]).await;
            let mut cfg = config(base, true);
            send_chat(&reqwest::Client::new(), &mut cfg, &[], &[], false)
                .await
                .unwrap();
            let requests = server.await.unwrap();
            assert_eq!(requests[1].1["model"], FALLBACK_CODING_MODEL);
        }
    }

    #[tokio::test]
    async fn unavailable_preferred_model_falls_back_once_before_streaming() {
        let (base, server) = gateway(vec![
            (200, json!({"data":[{"id":DEFAULT_CODING_MODEL}]})),
            (503, json!({"error":"down"})),
            (200, json!({})),
        ])
        .await;
        let mut cfg = config(base, true);
        assert!(
            send_chat(&reqwest::Client::new(), &mut cfg, &[], &[], true)
                .await
                .unwrap()
                .status()
                .is_success()
        );
        let requests = server.await.unwrap();
        assert_eq!(requests[1].1["model"], DEFAULT_CODING_MODEL);
        assert_eq!(requests[2].1["model"], FALLBACK_CODING_MODEL);
        assert_eq!(requests[2].1["stream"], true);
        assert_eq!(cfg.model, FALLBACK_CODING_MODEL);
    }

    #[tokio::test]
    async fn high_complexity_selects_catalogued_free_tool_capable_stepfun() {
        let step = "stepfun/step-5-preview:free";
        let (base, server) = gateway(vec![
            (
                200,
                json!({"data":[
                    {"id":DEFAULT_CODING_MODEL},
                    {"id":"stepfun/paid", "supported_parameters":["tools"]},
                    {"id":step, "supported_parameters":["tools"], "context_length":128000}
                ]}),
            ),
            (200, json!({})),
        ])
        .await;
        let mut cfg = config(base, true);
        send_chat_for_complexity(&reqwest::Client::new(), &mut cfg, &[], &[], false, true)
            .await
            .unwrap();
        let requests = server.await.unwrap();
        assert_eq!(requests[1].1["model"], step);
        assert_eq!(cfg.model, step);
    }

    #[tokio::test]
    async fn unavailable_stepfun_falls_back_to_auto_free_before_response() {
        let step = "stepfun/step-5-preview:free";
        let (base, server) = gateway(vec![
            (
                200,
                json!({"data":[{"id":DEFAULT_CODING_MODEL},{"id":step,"supported_parameters":["tools"]}]}),
            ),
            (503, json!({"error":"down"})),
            (200, json!({})),
        ])
        .await;
        let mut cfg = config(base, true);
        send_chat_for_complexity(&reqwest::Client::new(), &mut cfg, &[], &[], false, true)
            .await
            .unwrap();
        let requests = server.await.unwrap();
        assert_eq!(requests[1].1["model"], step);
        assert_eq!(requests[2].1["model"], FALLBACK_CODING_MODEL);
        assert_eq!(cfg.model, FALLBACK_CODING_MODEL);
    }

    #[tokio::test]
    async fn explicit_models_bypass_catalog_and_fallback() {
        let (base, server) = gateway(vec![(503, json!({"error":"down"}))]).await;
        let mut cfg = config(base, false);
        cfg.model = "custom/model".into();
        assert_eq!(
            send_chat(&reqwest::Client::new(), &mut cfg, &[], &[], false)
                .await
                .unwrap()
                .status()
                .as_u16(),
            503
        );
        let requests = server.await.unwrap();
        assert_eq!(requests[0].0, "/chat/completions");
        assert_eq!(requests[0].1["model"], "custom/model");
    }
    #[test]
    fn selects_only_the_preferred_free_model() {
        assert_eq!(
            catalog_model(
                &json!({"data":[{"id":"stepfun/step-5-preview-free"},{"id":DEFAULT_CODING_MODEL}]})
            ),
            DEFAULT_CODING_MODEL
        );
        for value in [
            json!({"data":[{"id":"cohere/north-mini-code"}]}),
            json!({"data":[]}),
            json!({"error":"unavailable"}),
        ] {
            assert_eq!(catalog_model(&value), FALLBACK_CODING_MODEL);
        }
    }
}
