//! Records public research evidence from caller-executed `web_fetch` results.
//!
//! Research is caller-driven: NioXS relays the `web_fetch` tool and the client
//! performs the fetch, so the returned content is the only place the provider
//! can observe it. Recording is best effort — a cache or quota failure never
//! fails the turn — and challenge, login, or denied pages are never cached.
use nioxs_browser::{PageStatus, classify, extract_text, extract_title};
use nioxs_contracts::Message;
use nioxs_storage::{NioDb, RESEARCH_CACHE_LIMIT, RESEARCH_TTL_SECONDS, ResearchEntry};
use serde_json::Value;
use std::collections::HashMap;

/// Stored excerpt cap, comfortably under the 16 KiB NioDB record limit.
const MAX_EXCERPT_CHARS: usize = 4_000;

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Cache and record evidence for every `web_fetch` result in `messages`.
///
/// Returns the number of entries stored. Reads no database state when the
/// history contains no `web_fetch` result.
pub async fn record_web_fetches(
    db: &NioDb,
    deployment: &str,
    turn: &str,
    scope: &str,
    messages: &[Message],
) -> usize {
    // Pair each `web_fetch` tool call with the URL it requested.
    let mut urls: HashMap<&str, String> = HashMap::new();
    for message in messages {
        for call in &message.tool_calls {
            if call.function.name != "web_fetch" {
                continue;
            }
            let url = serde_json::from_str::<Value>(&call.function.arguments)
                .ok()
                .and_then(|arguments| arguments["url"].as_str().map(str::to_string));
            if let Some(url) = url {
                urls.insert(call.id.as_str(), url);
            }
        }
    }
    if urls.is_empty() {
        return 0;
    }

    let mut stored = 0;
    for message in messages {
        if message.role != "tool" {
            continue;
        }
        let Some(url) = message.tool_call_id.as_deref().and_then(|id| urls.get(id)) else {
            continue;
        };
        let Some(content) = message
            .content
            .as_str()
            .filter(|text| !text.trim().is_empty())
        else {
            continue;
        };
        // A challenge, login, or access-denied page is not evidence.
        if classify(content) != PageStatus::Readable {
            continue;
        }
        let excerpt: String = extract_text(content)
            .chars()
            .take(MAX_EXCERPT_CHARS)
            .collect();
        if excerpt.trim().is_empty() {
            continue;
        }
        let title = extract_title(content).unwrap_or_default();
        let entry = ResearchEntry {
            deployment_id: deployment,
            turn_id: turn,
            scope,
            canonical_url: url,
            title: &title,
            excerpt: &excerpt,
            now_unix: now_unix(),
            ttl_seconds: RESEARCH_TTL_SECONDS,
            quota: RESEARCH_CACHE_LIMIT,
        };
        if db.research_put(&entry).await.is_ok() {
            stored += 1;
        }
    }
    stored
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testdb::mock_niodb;
    use serde_json::json;

    fn fetch_call(id: &str, url: &str) -> Message {
        serde_json::from_value(json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": id,
                "type": "function",
                "function": {"name": "web_fetch", "arguments": json!({"url": url}).to_string()},
            }],
        }))
        .unwrap()
    }

    fn tool_result(id: &str, content: &str) -> Message {
        serde_json::from_value(json!({
            "role": "tool",
            "tool_call_id": id,
            "content": content,
        }))
        .unwrap()
    }

    fn readable_page(body: &str) -> String {
        format!("<html><head><title>Rust Guide</title></head><body>{body}</body></html>")
    }

    async fn db() -> NioDb {
        let (url, _store) = mock_niodb().await;
        NioDb::new(&url, "test-token".into()).unwrap()
    }

    #[tokio::test]
    async fn web_fetch_results_are_cached_and_recorded_as_evidence() {
        let db = db().await;
        let messages = vec![
            fetch_call("call_1", "https://example.com/rust"),
            tool_result(
                "call_1",
                &readable_page("<p>Tokio provides an async runtime for Rust.</p>"),
            ),
        ];

        let stored = record_web_fetches(&db, "test", "turn_1", "workspace", &messages).await;
        assert_eq!(stored, 1);

        let now = now_unix();
        let cached = db
            .research_get("workspace", "https://example.com/rust", now)
            .await
            .unwrap()
            .expect("cached excerpt");
        assert_eq!(cached["title"], "Rust Guide");
        assert!(
            cached["excerpt"]
                .as_str()
                .unwrap()
                .contains("async runtime")
        );
        assert_eq!(cached["access_scope"], "workspace");

        let usage = db.research_usage().await.unwrap();
        assert_eq!(usage["cache_records"], 1);
        assert_eq!(usage["evidence_records"], 1);
    }

    #[tokio::test]
    async fn challenge_pages_are_never_cached() {
        let db = db().await;
        let messages = vec![
            fetch_call("call_1", "https://blocked.example.com"),
            tool_result(
                "call_1",
                "<form class='g-recaptcha'>Verify you are human</form>",
            ),
        ];

        let stored = record_web_fetches(&db, "test", "turn_1", "workspace", &messages).await;
        assert_eq!(stored, 0);
        assert!(
            db.research_get("workspace", "https://blocked.example.com", now_unix())
                .await
                .unwrap()
                .is_none()
        );
        let usage = db.research_usage().await.unwrap();
        assert_eq!(usage["cache_records"], 0);
        assert_eq!(usage["evidence_records"], 0);
    }

    #[tokio::test]
    async fn cached_excerpt_expires_after_its_ttl() {
        let db = db().await;
        let url = "https://example.com/expiring";
        let messages = vec![
            fetch_call("call_1", url),
            tool_result("call_1", &readable_page("<p>Fresh documentation.</p>")),
        ];
        assert_eq!(
            record_web_fetches(&db, "test", "turn_1", "workspace", &messages).await,
            1
        );

        let stored_at = now_unix();
        assert!(
            db.research_get("workspace", url, stored_at)
                .await
                .unwrap()
                .is_some()
        );
        assert!(
            db.research_get("workspace", url, stored_at + RESEARCH_TTL_SECONDS + 1)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn re_fetching_a_url_refreshes_instead_of_taking_a_second_quota_slot() {
        let db = db().await;
        let messages = vec![
            fetch_call("call_1", "https://example.com/repeat"),
            tool_result("call_1", &readable_page("<p>Repeated observation.</p>")),
        ];

        assert_eq!(
            record_web_fetches(&db, "test", "turn_1", "workspace", &messages).await,
            1
        );
        assert_eq!(
            record_web_fetches(&db, "test", "turn_1", "workspace", &messages).await,
            1
        );
        assert_eq!(
            db.research_usage().await.unwrap()["cache_records"],
            1,
            "a refetched URL refreshes one entry rather than inflating the quota"
        );
    }

    #[tokio::test]
    async fn writes_fail_closed_at_the_quota() {
        let db = db().await;
        let now = now_unix();
        fn entry(url: &str, now: u64) -> ResearchEntry<'_> {
            ResearchEntry {
                deployment_id: "test",
                turn_id: "turn_1",
                scope: "workspace",
                canonical_url: url,
                title: "Doc",
                excerpt: "bounded excerpt",
                now_unix: now,
                ttl_seconds: RESEARCH_TTL_SECONDS,
                quota: 1,
            }
        }

        assert!(
            db.research_put(&entry("https://example.com/a", now))
                .await
                .is_ok()
        );
        let refused = db.research_put(&entry("https://example.com/b", now)).await;
        assert_eq!(refused.unwrap_err(), "research cache quota reached");
        assert!(db.research_usage().await.unwrap()["cache_records"] == 1);
    }

    #[tokio::test]
    async fn cache_scopes_are_isolated() {
        let db = db().await;
        let messages = vec![
            fetch_call("call_1", "https://example.com/scoped"),
            tool_result("call_1", &readable_page("<p>Scoped evidence.</p>")),
        ];
        assert_eq!(
            record_web_fetches(&db, "test", "turn_1", "workspace-a", &messages).await,
            1
        );

        let now = now_unix();
        assert!(
            db.research_get("workspace-a", "https://example.com/scoped", now)
                .await
                .unwrap()
                .is_some()
        );
        assert!(
            db.research_get("workspace-b", "https://example.com/scoped", now)
                .await
                .unwrap()
                .is_none()
        );

        assert_eq!(db.research_clear("workspace-a").await.unwrap(), 1);
        assert!(
            db.research_get("workspace-a", "https://example.com/scoped", now)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn a_write_failure_does_not_fail_the_turn() {
        // An unreachable NioDB must degrade to "nothing cached", not an error.
        let db = NioDb::new("http://127.0.0.1:9", "test-token".into()).unwrap();
        let messages = vec![
            fetch_call("call_1", "https://example.com/offline"),
            tool_result("call_1", &readable_page("<p>Offline evidence.</p>")),
        ];

        assert_eq!(
            record_web_fetches(&db, "test", "turn_1", "workspace", &messages).await,
            0
        );
    }

    #[tokio::test]
    async fn unrelated_tool_results_are_ignored() {
        let db = db().await;
        let messages = vec![
            tool_result("call_unknown", "plain output with no fetch call"),
            tool_result("call_1", ""),
        ];
        assert_eq!(
            record_web_fetches(&db, "test", "turn_1", "workspace", &messages).await,
            0
        );
        assert_eq!(
            db.research_usage().await.unwrap()["cache_records"],
            0,
            "no fetch call means no database writes"
        );
    }
}
