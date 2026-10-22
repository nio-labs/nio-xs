use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::Duration;

pub struct NioDbProcess {
    child: Option<Child>,
    pub url: String,
    pub token: String,
}

impl Drop for NioDbProcess {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Finds the `niodb` executable across standard locations, or auto-downloads it if missing.
pub fn find_or_download_niodb() -> Option<PathBuf> {
    find_niodb_binary().or_else(auto_download_niodb)
}

/// Auto-downloads NioDB binary into ~/.nioxs/bin if not found locally.
pub fn auto_download_niodb() -> Option<PathBuf> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()?;
    let bin_dir = Path::new(&home).join(".nioxs").join("bin");
    let _ = std::fs::create_dir_all(&bin_dir);

    let is_windows = cfg!(target_os = "windows");
    let binary_name = if is_windows { "niodb.exe" } else { "niodb" };
    let target_path = bin_dir.join(binary_name);
    if target_path.is_file() {
        return Some(target_path);
    }

    let os_str = if cfg!(target_os = "macos") {
        "darwin"
    } else if is_windows {
        "win32"
    } else {
        "linux"
    };

    let arch_str = if cfg!(target_arch = "aarch64") {
        "arm64"
    } else {
        "x64"
    };

    let ext = if is_windows { ".exe" } else { "" };
    let asset = format!("niodb-{os_str}-{arch_str}{ext}");
    let url = format!("https://github.com/nio-labs/nio-db/releases/latest/download/{asset}");

    eprintln!("📦 Auto-installing NioDB sovereign state engine ({asset})...");

    let downloaded = if is_windows {
        let script = format!(
            "Invoke-WebRequest -Uri '{url}' -OutFile '{}'",
            target_path.display()
        );
        Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", &script])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    } else {
        let curl_ok = Command::new("curl")
            .args([
                "--connect-timeout",
                "10",
                "--max-time",
                "120",
                "-fsSL",
                &url,
                "-o",
                target_path.to_str().unwrap_or(""),
            ])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        if curl_ok {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ =
                    std::fs::set_permissions(&target_path, std::fs::Permissions::from_mode(0o755));
            }
            true
        } else {
            false
        }
    };

    if downloaded && target_path.is_file() {
        eprintln!("✅ Installed NioDB to {}", target_path.display());
        Some(target_path)
    } else {
        None
    }
}

/// Finds the `niodb` executable across standard locations, or returns None.
pub fn find_niodb_binary() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("NIOXS_NIODB_BIN") {
        let path = PathBuf::from(explicit);
        if path.is_file() {
            return Some(path);
        }
    }

    if let Ok(home) = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")) {
        let home = Path::new(&home);
        let candidates = [
            home.join(".nioxs/bin/niodb"),
            home.join(".nioxs/bin/niodb.exe"),
            home.join(".local/bin/niodb"),
            home.join(".local/bin/niodb.exe"),
            home.join(".cargo/bin/niodb"),
            home.join(".cargo/bin/niodb.exe"),
        ];
        for candidate in &candidates {
            if candidate.is_file() {
                return Some(candidate.clone());
            }
        }
    }

    // Check system PATH
    which("niodb").or_else(|| which("niodb.exe"))
}

fn which(bin: &str) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&paths) {
        let full = dir.join(bin);
        if full.is_file() {
            return Some(full);
        }
    }
    None
}

/// Resolves existing token from disk or returns None
pub fn resolve_token(dir: &Path) -> Option<String> {
    // 0. Direct environment variable
    if let Ok(direct) = std::env::var("NIOXS_DB_TOKEN") {
        let trimmed = direct.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    // 1. Token file environment variable
    if let Ok(explicit) = std::env::var("NIOXS_DB_TOKEN_FILE") {
        if let Ok(content) = std::fs::read_to_string(explicit) {
            let trimmed = content.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }

    // 2. Client token files in common dirs
    let token_candidates = [
        dir.join("client-token"),
        dir.join("niodb-client.token"),
        dir.join("db.token"),
    ];

    for path in &token_candidates {
        if let Ok(content) = std::fs::read_to_string(path) {
            let trimmed = content.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }

    None
}

/// Initializes auth if credentials don't exist yet, returning the client_token.
pub fn ensure_niodb_auth(niodb_bin: &Path, data_dir: &Path) -> Result<String, String> {
    if let Some(existing) = resolve_token(data_dir) {
        return Ok(existing);
    }

    let auth_file = data_dir.join("auth.json");
    if auth_file.exists() {
        // auth.json exists but client token file was deleted: recreate a fresh auth bundle
        let _ = std::fs::remove_file(&auth_file);
    }

    let output = Command::new(niodb_bin)
        .args([
            "init-auth",
            "--dir",
            data_dir.to_str().unwrap_or("."),
            "--name",
            "nioxs",
        ])
        .output()
        .map_err(|e| format!("failed to run niodb init-auth: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "niodb init-auth failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout_str = String::from_utf8_lossy(&output.stdout);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&stdout_str) {
        if let Some(tok) = v.get("client_token").and_then(|t| t.as_str()) {
            let token_path = data_dir.join("client-token");
            let _ = std::fs::write(&token_path, tok);
            return Ok(tok.to_string());
        }
    }

    // Re-check client token
    resolve_token(data_dir).ok_or_else(|| "could not resolve client_token from NioDB auth".into())
}

/// Starts an integrated NioDB instance if one is not already running.
pub async fn ensure_niodb(
    requested_url: Option<&str>,
    token_file: Option<&str>,
) -> Result<(String, String, Option<NioDbProcess>), String> {
    let url = requested_url.unwrap_or("http://127.0.0.1:7432").to_string();

    // Check if an existing NioDB is already healthy
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(600))
        .build()
        .map_err(|e| e.to_string())?;

    let health_url = format!("{}/health", url.trim_end_matches('/'));
    let already_running = client.get(&health_url).send().await.is_ok();

    if already_running || requested_url.is_some() {
        // Resolve token from NIOXS_DB_TOKEN, token_file or default paths
        let token = if let Ok(direct) = std::env::var("NIOXS_DB_TOKEN") {
            let trimmed = direct.trim().to_string();
            if trimmed.is_empty() {
                return Err("NIOXS_DB_TOKEN cannot be empty".to_string());
            }
            trimmed
        } else if let Some(tf) = token_file {
            std::fs::read_to_string(tf)
                .map_err(|e| format!("failed reading token file {tf}: {e}"))?
                .trim()
                .to_string()
        } else {
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .unwrap_or_else(|_| ".".into());
            let default_dir = Path::new(&home).join(".nioxs/data");
            resolve_token(&default_dir).ok_or_else(|| {
                format!("NioDB is configured at {url}, but no token was provided (set NIOXS_DB_TOKEN or NIOXS_DB_TOKEN_FILE)")
            })?
        };
        return Ok((url, token, None));
    }

    // Not running: Locate or spawn bundled niodb
    let niodb_bin = find_or_download_niodb()
        .ok_or_else(|| "NioDB executable not found and auto-download failed. Please install via install.sh or run 'npx @nio-labs/nio-xs'.".to_string())?;

    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".into());
    let data_dir = Path::new(&home).join(".nioxs/data");
    let _ = std::fs::create_dir_all(&data_dir);

    let token = ensure_niodb_auth(&niodb_bin, &data_dir)?;

    let port = url
        .split(':')
        .last()
        .and_then(|p| p.split('/').next())
        .unwrap_or("7432");
    let listen_addr = format!("127.0.0.1:{port}");

    let child = Command::new(&niodb_bin)
        .args([
            "serve",
            "--dir",
            data_dir.to_str().unwrap_or("."),
            "--listen",
            &listen_addr,
            "--no-pg",
            "--no-demo",
            "--nio-bin",
            "disabled",
            "--node-bin",
            "disabled",
        ])
        .spawn()
        .map_err(|e| format!("failed to spawn bundled niodb: {e}"))?;

    // Wait for health check with retries
    let mut ready = false;
    for _ in 0..30 {
        tokio::time::sleep(Duration::from_millis(150)).await;
        if let Ok(res) = client.get(&health_url).send().await {
            if res.status().is_success() {
                ready = true;
                break;
            }
        }
    }

    if !ready {
        return Err("bundled NioDB process started but health check timed out".into());
    }

    let proc = NioDbProcess {
        child: Some(child),
        url: url.clone(),
        token: token.clone(),
    };

    Ok((url, token, Some(proc)))
}
