use futures_util::StreamExt;
use nioxs_contracts::{DurableEvent, digest};
use serde_json::{Value, json};
use std::time::Duration;

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
}
