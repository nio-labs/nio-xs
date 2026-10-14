use futures_util::StreamExt;
use nioxs_contracts::{DurableEvent, digest};
use serde_json::{Value, json};
use std::time::Duration;

/// Bounded record count for the research collections (`nano_cache`,
/// `nano_evidence`); scans fail closed at this limit.
pub const RESEARCH_CACHE_LIMIT: usize = 2000;
/// Default expiry for a cached public excerpt.
pub const RESEARCH_TTL_SECONDS: u64 = 24 * 3600;
/// Extraction pipeline version, so a cache entry can be invalidated on change.
pub const EXTRACTION_VERSION: u32 = 1;

/// One public research excerpt to cache and record as evidence.
pub struct ResearchEntry<'a> {
    pub deployment_id: &'a str,
    pub turn_id: &'a str,
    /// Isolation scope for the cache. Entries are never read across scopes.
    pub scope: &'a str,
    pub canonical_url: &'a str,
    pub title: &'a str,
    pub excerpt: &'a str,
    pub now_unix: u64,
    pub ttl_seconds: u64,
    /// Maximum cache records allowed; the write fails closed at the limit.
    pub quota: usize,
}

/// Stable scope+URL cache key. Hashed so records are matched without exposing
/// the URL as an identifier.
fn research_key(scope: &str, canonical_url: &str) -> String {
    digest(format!("{scope}:{canonical_url}").as_bytes())
}

#[derive(Clone)]
pub struct NioDb {
    client: reqwest::Client,
    base: reqwest::Url,
    token: String,
}

impl NioDb {
    pub fn new(base: &str, token: String) -> Result<Self, String> {
        let base = reqwest::Url::parse(&format!("{}/", base.trim_end_matches('/')))
            .map_err(|_| "invalid NioDB URL")?;
        if !matches!(base.scheme(), "http" | "https")
            || base.host_str().is_none()
            || !base.username().is_empty()
            || base.password().is_some()
            || base.query().is_some()
            || base.fragment().is_some()
        {
            return Err("NioDB URL must be HTTP(S) without credentials/query/fragment".into());
        }
        if token.trim().is_empty() {
            return Err("NioDB token is required".into());
        }
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|_| "cannot build storage client")?;
        Ok(Self {
            client,
            base,
            token,
        })
    }

    async fn response(&self, request: reqwest::RequestBuilder) -> Result<Value, String> {
        let response = request
            .bearer_auth(&self.token)
            .send()
            .await
            .map_err(|_| "NioDB unavailable")?;
        let status = response.status();
        if !status.is_success() {
            return Err(format!("NioDB returned HTTP {}", status.as_u16()));
        }
        let mut bytes = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|_| "NioDB response interrupted")?;
            if bytes.len() + chunk.len() > 256 * 1024 {
                return Err("NioDB response exceeds limit".into());
            }
            bytes.extend_from_slice(&chunk);
        }
        serde_json::from_slice(&bytes).map_err(|_| "invalid NioDB response".into())
    }

    pub async fn ready(&self) -> Result<Value, String> {
        // Authenticated CRUD scope is checked too; public /health alone is insufficient.
        self.response(
            self.client
                .get(self.base.join("api/v1/records").unwrap())
                .query(&[("collection", "nano_events"), ("limit", "1")]),
        )
        .await
    }

    pub async fn append_event(&self, event: &DurableEvent) -> Result<Value, String> {
        let data = serde_json::to_value(event).map_err(|_| "cannot encode event")?;
        let key = format!(
            "nano:{}",
            digest(&serde_json::to_vec(&data).map_err(|_| "cannot encode event")?)
        );
        self.response(
            self.client
                .post(self.base.join("api/v1/records").unwrap())
                .header("Idempotency-Key", key)
                .json(&json!({"collection":"nano_events","data":data})),
        )
        .await
    }

    pub async fn records(&self, collection: &str, turn: &str) -> Result<Vec<Value>, String> {
        let mut records = Vec::new();
        for offset in (0..2000).step_by(50) {
            let page = self
                .response(
                    self.client
                        .get(self.base.join("api/v1/records").unwrap())
                        .query(&[
                            ("collection", collection.to_string()),
                            ("limit", "50".into()),
                            ("offset", offset.to_string()),
                        ]),
                )
                .await?;
            let items = page["items"].as_array().ok_or("invalid record page")?;
            for item in items {
                if item["data"]["turn_id"] == turn {
                    if item["revision"] != 1 || item["data"]["schema_version"] != 2 {
                        return Err("workflow_revision_conflict".into());
                    }
                    records.push(item.clone());
                }
            }
            if items.len() < 50 {
                return Ok(records);
            }
        }
        Err("workflow journal scan quota reached".into())
    }

    pub async fn append_record(
        &self,
        collection: &str,
        key: &str,
        data: &Value,
    ) -> Result<Value, String> {
        if serde_json::to_vec(data)
            .map_err(|_| "invalid record")?
            .len()
            > 16 * 1024
        {
            return Err("record exceeds quota".into());
        }
        let result = self
            .response(
                self.client
                    .post(self.base.join("api/v1/records").unwrap())
                    .header(
                        "Idempotency-Key",
                        format!("nano2:{}", digest(key.as_bytes())),
                    )
                    .json(&json!({"collection":collection,"data":data})),
            )
            .await?;
        if result["revision"] != 1 || result["data"] != *data {
            return Err("workflow_revision_conflict".into());
        }
        Ok(result)
    }

    pub async fn read(&self, id: &str) -> Result<Value, String> {
        if id.is_empty() || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            return Err("invalid record ID".into());
        }
        self.response(
            self.client
                .get(self.base.join(&format!("api/v1/records/{id}")).unwrap()),
        )
        .await
    }

    /// Read an unexpired entry from `nano_cache`.
    /// Returns Ok(Some(data)) if present and unexpired, Ok(None) if absent or expired.
    pub async fn cache_get(&self, key: &str, now_unix: u64) -> Result<Option<Value>, String> {
        let digest_key = digest(key.as_bytes());
        for offset in (0..1000).step_by(50) {
            let page = self
                .response(
                    self.client
                        .get(self.base.join("api/v1/records").unwrap())
                        .query(&[
                            ("collection", "nano_cache"),
                            ("limit", "50"),
                            ("offset", &offset.to_string()),
                        ]),
                )
                .await?;
            let items = page["items"].as_array().ok_or("invalid record page")?;
            for item in items {
                if item["data"]["cache_key"] == digest_key {
                    let expires_at = item["data"]["expires_at_unix"].as_u64().unwrap_or(0);
                    if expires_at > now_unix {
                        return Ok(Some(item["data"].clone()));
                    } else {
                        return Ok(None);
                    }
                }
            }
            if items.len() < 50 {
                break;
            }
        }
        Ok(None)
    }

    /// Store a 24-hour TTL entry in `nano_cache`.
    pub async fn cache_set(
        &self,
        deployment_id: &str,
        key: &str,
        data_payload: &Value,
        now_unix: u64,
        ttl_seconds: u64,
    ) -> Result<Value, String> {
        let digest_key = digest(key.as_bytes());
        let record = json!({
            "schema_version": 2,
            "deployment_id": deployment_id,
            "cache_key": digest_key,
            "created_at_unix": now_unix,
            "expires_at_unix": now_unix + ttl_seconds,
            "data": data_payload,
        });
        self.append_record("nano_cache", &format!("cache:{digest_key}"), &record)
            .await
    }

    /// Read an unexpired NioFlow handoff session by its stable key.
    ///
    /// `Ok(Some(record))` is a live record (its `session` field carries the
    /// serialized handoff), `Ok(None)` is absent or expired, and `Err` means
    /// NioDB is unavailable — the caller then keeps its in-memory state and
    /// refuses to resume rather than guessing.
    pub async fn flow_session_get(
        &self,
        key: &str,
        now_unix: u64,
    ) -> Result<Option<Value>, String> {
        let digest_key = digest(key.as_bytes());
        for offset in (0..1000).step_by(50) {
            let page = self
                .response(
                    self.client
                        .get(self.base.join("api/v1/records").unwrap())
                        .query(&[
                            ("collection", "nano_flow"),
                            ("limit", "50"),
                            ("offset", &offset.to_string()),
                        ]),
                )
                .await?;
            let items = page["items"].as_array().ok_or("invalid record page")?;
            for item in items {
                if item["data"]["session_key"] == digest_key {
                    let expires_at = item["data"]["expires_at_unix"].as_u64().unwrap_or(0);
                    if expires_at > now_unix {
                        return Ok(Some(item["data"].clone()));
                    }
                    return Ok(None);
                }
            }
            if items.len() < 50 {
                break;
            }
        }
        Ok(None)
    }

    /// Store a NioFlow handoff session with a bounded TTL.
    ///
    /// The record is capped by `append_record`; an oversized session fails here
    /// and the caller keeps that handoff in memory only.
    pub async fn flow_session_set(
        &self,
        deployment_id: &str,
        key: &str,
        session: &Value,
        now_unix: u64,
        ttl_seconds: u64,
    ) -> Result<Value, String> {
        let digest_key = digest(key.as_bytes());
        let record = json!({
            "schema_version": 2,
            "deployment_id": deployment_id,
            "session_key": digest_key,
            "created_at_unix": now_unix,
            "expires_at_unix": now_unix + ttl_seconds,
            "session": session,
        });
        self.append_record("nano_flow", &format!("flow:{digest_key}"), &record)
            .await
    }

    /// Delete a NioFlow handoff session by key. Best effort: a missing record is
    /// not an error, and a stale record is still bounded by its TTL.
    pub async fn flow_session_remove(&self, key: &str) -> Result<(), String> {
        let digest_key = digest(key.as_bytes());
        let mut ids = Vec::new();
        for offset in (0..1000).step_by(50) {
            let page = self
                .response(
                    self.client
                        .get(self.base.join("api/v1/records").unwrap())
                        .query(&[
                            ("collection", "nano_flow"),
                            ("limit", "50"),
                            ("offset", &offset.to_string()),
                        ]),
                )
                .await?;
            let items = page["items"].as_array().ok_or("invalid record page")?;
            for item in items {
                if item["data"]["session_key"] == digest_key
                    && let Some(id) = item["id"].as_str()
                {
                    ids.push(id.to_string());
                }
            }
            if items.len() < 50 {
                break;
            }
        }
        for id in ids {
            self.delete_record(&id).await?;
        }
        Ok(())
    }

    /// Bounded scan of one collection, capped at `limit` records.
    async fn scan_collection(&self, collection: &str, limit: usize) -> Result<Vec<Value>, String> {
        let mut found = Vec::new();
        let mut offset = 0usize;
        while offset < limit {
            let page = self
                .response(
                    self.client
                        .get(self.base.join("api/v1/records").unwrap())
                        .query(&[
                            ("collection", collection),
                            ("limit", "50"),
                            ("offset", &offset.to_string()),
                        ]),
                )
                .await?;
            let items = page["items"].as_array().ok_or("invalid record page")?;
            found.extend(items.iter().cloned());
            if items.len() < 50 {
                break;
            }
            offset += 50;
        }
        Ok(found)
    }

    /// Read an unexpired cached research excerpt for a scope and canonical URL.
    pub async fn research_get(
        &self,
        scope: &str,
        canonical_url: &str,
        now_unix: u64,
    ) -> Result<Option<Value>, String> {
        let key = research_key(scope, canonical_url);
        for record in self
            .scan_collection("nano_cache", RESEARCH_CACHE_LIMIT)
            .await?
        {
            if record["data"]["cache_key"] == key {
                let expires_at = record["data"]["expires_at_unix"].as_u64().unwrap_or(0);
                if expires_at > now_unix {
                    return Ok(Some(record["data"].clone()));
                }
                return Ok(None);
            }
        }
        Ok(None)
    }

    /// Store a bounded, public research excerpt and its evidence record.
    ///
    /// Fails closed when the quota, the per-record cap, or `append_record`
    /// rejects the write. Callers treat that as a best-effort miss.
    pub async fn research_put(&self, entry: &ResearchEntry<'_>) -> Result<Value, String> {
        self.evict_expired_research(entry.now_unix).await?;
        let key = research_key(entry.scope, entry.canonical_url);
        // Re-fetching a URL refreshes its entry instead of adding a duplicate, so
        // the quota counts distinct cached URLs and reuse cannot inflate it.
        let existing = self
            .scan_collection("nano_cache", RESEARCH_CACHE_LIMIT)
            .await?;
        let mut replaced = false;
        if let Some(record) = existing
            .iter()
            .find(|record| record["data"]["cache_key"] == json!(key))
            && let Some(id) = record["id"].as_str()
        {
            self.delete_record(id).await?;
            replaced = true;
        }
        if existing.len() - usize::from(replaced) >= entry.quota {
            return Err("research cache quota reached".into());
        }
        let data = json!({
            "schema_version": 2,
            "deployment_id": entry.deployment_id,
            "turn_id": entry.turn_id,
            "cache_key": key,
            "access_scope": entry.scope,
            "canonical_url": entry.canonical_url,
            "title": entry.title,
            "extraction_version": EXTRACTION_VERSION,
            "retrieved_at_unix": entry.now_unix,
            "created_at_unix": entry.now_unix,
            "expires_at_unix": entry.now_unix + entry.ttl_seconds,
            "content_digest": digest(entry.excerpt.as_bytes()),
            "excerpt": entry.excerpt,
        });
        let cached = self
            .append_record("nano_cache", &format!("cache:{key}"), &data)
            .await?;
        // Evidence is a separate, append-only record. The retrieval time is part
        // of the key so a re-observation appends a new record instead of
        // colliding with the earlier one; a failure here still leaves a usable
        // cache entry, so it does not fail the put.
        let mut evidence = data.clone();
        evidence.as_object_mut().unwrap().remove("cache_key");
        let _ = self
            .append_record(
                "nano_evidence",
                &format!("evidence:{}:{}:{key}", entry.turn_id, entry.now_unix),
                &evidence,
            )
            .await;
        Ok(cached)
    }

    /// Delete every cached entry in a scope (clearing / invalidation).
    pub async fn research_clear(&self, scope: &str) -> Result<usize, String> {
        let mut removed = 0;
        for record in self
            .scan_collection("nano_cache", RESEARCH_CACHE_LIMIT)
            .await?
        {
            if record["data"]["access_scope"] == scope
                && let Some(id) = record["id"].as_str()
            {
                self.delete_record(id).await?;
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Cache and evidence record counts, for quota reporting.
    pub async fn research_usage(&self) -> Result<Value, String> {
        let cache = self
            .scan_collection("nano_cache", RESEARCH_CACHE_LIMIT)
            .await?;
        let evidence = self
            .scan_collection("nano_evidence", RESEARCH_CACHE_LIMIT)
            .await?;
        Ok(json!({
            "cache_records": cache.len(),
            "evidence_records": evidence.len(),
            "cache_limit": RESEARCH_CACHE_LIMIT,
        }))
    }

    /// Delete expired cache records before a quota check. Expiry is the only
    /// automatic invalidation this prototype performs.
    async fn evict_expired_research(&self, now_unix: u64) -> Result<(), String> {
        for id in self.scan_expired("nano_cache", now_unix).await? {
            self.delete_record(&id).await?;
        }
        Ok(())
    }

    /// Scan `nano_cache` or a collection and find IDs of records expired before `now_unix`.
    pub async fn scan_expired(
        &self,
        collection: &str,
        now_unix: u64,
    ) -> Result<Vec<String>, String> {
        let mut expired_ids = Vec::new();
        for offset in (0..1000).step_by(50) {
            let page = self
                .response(
                    self.client
                        .get(self.base.join("api/v1/records").unwrap())
                        .query(&[
                            ("collection", collection),
                            ("limit", "50"),
                            ("offset", &offset.to_string()),
                        ]),
                )
                .await?;
            let items = page["items"].as_array().ok_or("invalid record page")?;
            for item in items {
                let expires_at = item["data"]["expires_at_unix"].as_u64().unwrap_or(0);
                if expires_at > 0
                    && expires_at <= now_unix
                    && let Some(id) = item["id"].as_str()
                {
                    expired_ids.push(id.to_string());
                }
            }
            if items.len() < 50 {
                break;
            }
        }
        Ok(expired_ids)
    }

    /// Delete a record by ID via DELETE /api/v1/records/{id} if supported.
    pub async fn delete_record(&self, id: &str) -> Result<(), String> {
        if id.is_empty() || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            return Err("invalid record ID".into());
        }
        let response = self
            .client
            .delete(self.base.join(&format!("api/v1/records/{id}")).unwrap())
            .bearer_auth(&self.token)
            .send()
            .await
            .map_err(|_| "NioDB unavailable")?;
        let status = response.status();
        if !status.is_success() && status.as_u16() != 404 {
            return Err(format!("NioDB returned HTTP {}", status.as_u16()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_digest_is_deterministic() {
        let key = "https://en.wikipedia.org/wiki/Rust_(programming_language)";
        let digest1 = digest(key.as_bytes());
        let digest2 = digest(key.as_bytes());
        assert_eq!(digest1, digest2);
        assert_eq!(digest1.len(), 64);
    }

    #[test]
    fn test_expiry_ttl_calculation() {
        let now_unix = 1700000000;
        let ttl_seconds = 86400; // 24 hours
        let expires_at = now_unix + ttl_seconds;

        assert!(expires_at > now_unix);
        assert_eq!(expires_at - now_unix, 86400);

        // At 12 hours, unexpired
        let check_at_12h = now_unix + 43200;
        assert!(expires_at > check_at_12h);

        // At 25 hours, expired
        let check_at_25h = now_unix + 90000;
        assert!(expires_at <= check_at_25h);
    }
}
