use serde::Serialize;
use std::{path::Path, process::Stdio, time::Duration};
use tokio::{io::AsyncReadExt, process::Command};

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PageStatus {
    Readable,
    Challenge,
    AccessDenied,
    Empty,
}

pub fn classify(text: &str) -> PageStatus {
    let lower = text.to_lowercase();
    if [
        "g-recaptcha",
        "hcaptcha",
        "cf-chl-",
        "verify you are human",
        "unusual traffic from your computer",
        "our systems have detected unusual traffic",
        "captcha",
    ]
    .iter()
    .any(|s| lower.contains(s))
    {
        PageStatus::Challenge
    } else if ["access denied", "403 forbidden", "sign in to continue"]
        .iter()
        .any(|s| lower.contains(s))
    {
        PageStatus::AccessDenied
    } else if text.trim().is_empty() {
        PageStatus::Empty
    } else {
        PageStatus::Readable
    }
}

/// CLI feasibility reader, not the Phase 3 crawler. No alternative browser path.
pub async fn fetch(binary: &Path, url: &str) -> Result<String, String> {
    if !url.starts_with("https://") {
        return Err("probe only accepts HTTPS URLs".into());
    }
    let mut child = Command::new(binary)
        .args(["fetch", url, "--dump", "html", "--timeout", "12", "--quiet"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|_| "cannot start Obscura")?;
    let stdout = child.stdout.take().ok_or("Obscura stdout unavailable")?;
    let result = tokio::time::timeout(Duration::from_secs(18), async {
        let mut bytes = Vec::new();
        stdout
            .take(1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .await
            .map_err(|_| "cannot read Obscura output")?;
        if bytes.len() > 1024 * 1024 {
            return Err("Obscura output exceeded 1 MiB".into());
        }
        let status = child.wait().await.map_err(|_| "cannot reap Obscura")?;
        if !status.success() {
            return Err("Obscura navigation failed".into());
        }
        String::from_utf8(bytes).map_err(|_| "Obscura returned invalid UTF-8".into())
    })
    .await;
    match result {
        Ok(result) => result,
        Err(_) => {
            let _ = child.kill().await;
            Err("Obscura deadline exceeded".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn blocked_pages_are_not_evidence() {
        assert_eq!(
            classify("<form class='g-recaptcha'>Verify you are human</form>"),
            PageStatus::Challenge
        );
        assert_eq!(classify("403 Forbidden"), PageStatus::AccessDenied);
        assert_eq!(classify("   "), PageStatus::Empty);
        assert_eq!(
            classify("<h1>Project documentation</h1>"),
            PageStatus::Readable
        );
    }
}
