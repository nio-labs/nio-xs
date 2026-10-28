use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::process::Stdio;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::{Duration, timeout};

const CLASSIFIER_TIMEOUT: Duration = Duration::from_secs(10);
const FALLBACK_TIMEOUT: Duration = Duration::from_secs(120);

async fn run_json_command(executable: &str, input: &Value, limit: Duration) -> Result<Value> {
    let mut child = Command::new(executable)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .with_context(|| format!("cannot start configured command {executable:?}"))?;
    let mut stdin = child.stdin.take().ok_or_else(|| anyhow!("command stdin unavailable"))?;
    let bytes = serde_json::to_vec(input)?;
    stdin.write_all(&bytes).await?;
    drop(stdin);
    let output = timeout(limit, child.wait_with_output())
        .await
        .map_err(|_| anyhow!("configured command timed out"))??;
    if !output.status.success() {
        return Err(anyhow!("configured command exited with {}", output.status));
    }
    serde_json::from_slice(&output.stdout).context("configured command returned invalid JSON")
}

async fn is_allowed(direction: &str, content: &str) -> Result<bool> {
    let executable = std::env::var("NIOXS_SKATE_SAFETY_CLASSIFIER")
        .context("NIOXS_SKATE_SAFETY_CLASSIFIER is not configured")?;
    let result = run_json_command(&executable, &json!({"direction": direction, "content": content}), CLASSIFIER_TIMEOUT).await?;
    Ok(result.get("decision").and_then(Value::as_str) == Some("allow"))
}

async fn fallback(reason: &str, prompt: &str) -> Result<String> {
    let executable = std::env::var("NIOXS_SKATE_FALLBACK_COMMAND")
        .context("NIOXS_SKATE_FALLBACK_COMMAND is not configured; refusing to run Skate")?;
    let result = run_json_command(
        &executable,
        &json!({"reason": reason, "prompt": prompt}),
        FALLBACK_TIMEOUT,
    )
    .await?;
    let response = result.get("response").and_then(Value::as_str)
        .ok_or_else(|| anyhow!("fallback command must return a JSON string field named response"))?;
    if response.trim().is_empty() {
        return Err(anyhow!("fallback command returned an empty response"));
    }
    if !is_allowed("fallback_output", response).await? {
        return Err(anyhow!("safety classifier rejected fallback output"));
    }
    Ok(response.to_owned())
}

fn release_is_approved(model: &std::path::Path, tokenizer: &std::path::Path) -> Result<bool> {
    let report_path = std::env::var("NIOXS_SKATE_RELEASE_REPORT")
        .context("NIOXS_SKATE_RELEASE_REPORT is not configured")?;
    let report: Value = serde_json::from_slice(&std::fs::read(report_path)?)?;
    if report.get("release_approved").and_then(Value::as_bool) != Some(true) {
        return Ok(false);
    }
    let actual = format!("{:x}", Sha256::digest(std::fs::read(model)?));
    let tokenizer_hash = format!("{:x}", Sha256::digest(std::fs::read(tokenizer)?));
    Ok(report.get("model_sha256").and_then(Value::as_str) == Some(actual.as_str())
        && report.get("tokenizer_sha256").and_then(Value::as_str) == Some(tokenizer_hash.as_str()))
}

pub async fn generate(prompt: &str) -> Result<String> {
    if !is_allowed("user_input", prompt).await.unwrap_or(false) {
        return fallback("input_blocked_or_classifier_unavailable", prompt).await;
    }

    let engine = match nioxs_engine::RayEngine::for_model("nio-skate") {
        Ok(engine) => engine,
        Err(_) => return fallback("candidate_unavailable", prompt).await,
    };
    if !release_is_approved(engine.model_path(), engine.tokenizer_path()).unwrap_or(false) {
        return fallback("candidate_not_release_approved", prompt).await;
    }
    let policy = nioxs_engine::skate_system()?;
    let formatted = format!(
        "<|im_start|>system\n{policy}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    );
    let generated = engine.generate(formatted, 128, 0.65, 0.9).await?;
    if generated.trim().is_empty() || !is_allowed("model_output", &generated).await.unwrap_or(false) {
        return fallback("output_blocked_or_classifier_unavailable", prompt).await;
    }
    Ok(generated)
}
