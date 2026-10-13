//! A minimal in-memory NioDB over raw HTTP for provider tests.
//!
//! POST stores a record, GET lists the records of one collection, DELETE removes
//! one. Records carry their collection so a `?collection=` filter behaves like
//! the real adapter.
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Returns the mock base URL and the record store it serves.
pub async fn mock_niodb() -> (String, std::sync::Arc<tokio::sync::Mutex<Vec<Value>>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let store = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::<Value>::new()));
    let handle = store.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                break;
            };
            let mut bytes = vec![];
            let mut chunk = [0u8; 4096];
            let (head, body_start) = loop {
                let read = match socket.read(&mut chunk).await {
                    Ok(0) | Err(_) => return,
                    Ok(read) => read,
                };
                bytes.extend_from_slice(&chunk[..read]);
                if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
                    break (String::from_utf8_lossy(&bytes[..end]).to_string(), end + 4);
                }
            };
            let length = head
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            while bytes.len() < body_start + length {
                let read = match socket.read(&mut chunk).await {
                    Ok(0) | Err(_) => return,
                    Ok(read) => read,
                };
                bytes.extend_from_slice(&chunk[..read]);
            }
            let mut head_lines = head.lines();
            let request_line = head_lines.next().unwrap_or("");
            let mut parts = request_line.split_whitespace();
            let method = parts.next().unwrap_or("");
            let target = parts.next().unwrap_or("");
            let path = target.split('?').next().unwrap_or("");
            let collection = target
                .split('?')
                .nth(1)
                .and_then(|query| {
                    query
                        .split('&')
                        .find_map(|part| part.strip_prefix("collection="))
                })
                .unwrap_or("")
                .to_string();
            let body = &bytes[body_start..body_start + length];
            let response = {
                let mut store = handle.lock().await;
                if method == "POST" && path == "/api/v1/records" {
                    let value: Value = serde_json::from_slice(body).unwrap();
                    let record = json!({
                        "id": format!("rec_{}", store.len() + 1),
                        "revision": 1,
                        "collection": value["collection"],
                        "data": value["data"],
                    });
                    store.push(record.clone());
                    record
                } else if method == "GET" && path == "/api/v1/records" {
                    let items: Vec<Value> = store
                        .iter()
                        .filter(|record| record["collection"] == json!(collection))
                        .cloned()
                        .collect();
                    json!({ "items": items })
                } else if method == "DELETE" && path.starts_with("/api/v1/records/") {
                    let id = path.trim_start_matches("/api/v1/records/");
                    store.retain(|record| record["id"] != id);
                    json!({})
                } else {
                    json!({})
                }
            };
            let body = response.to_string();
            let _ = socket
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await;
        }
    });
    (base, store)
}
