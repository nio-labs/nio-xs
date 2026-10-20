use nioxs_nano::Model;
use nioxs_provider::{ActiveTurns, App, router};
use nioxs_storage::NioDb;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("nioxs: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arg = std::env::args().nth(1);
    if matches!(arg.as_deref(), Some("--version")) {
        println!("nioxs {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    if matches!(arg.as_deref(), Some("--help" | "-h")) {
        println!(
            "Usage: nioxs [serve|ray <prompt>|mcp|init|undo|history]\n\
Commands:\n\
  serve     Start HTTP OpenAI-compatible AI gateway (default)\n\
  ray       Run local in-process NioRay predator inference on prompt\n\
  mcp       Start Model Context Protocol (MCP) JSON-RPC stdio server\n\
  init      Initialize configuration for coding agents (--aider, --cline, --opencode, --cursor)\n\
  undo      Rollback last applied multi-file patch (or specific patch_id)\n\
  history   Display local session turn journal from .nioxs/history.jsonl\n\
\n\
Environment: NIOXS_LISTEN (127.0.0.1:7460), NIOXS_DB_URL, NIOXS_DB_TOKEN_FILE, NIOXS_DEPLOYMENT_ID, NIOXS_API_TOKEN_FILE (optional on loopback)."
        );
        return Ok(());
    }
    if arg.as_deref() == Some("ray") {
        let prompt = std::env::args().skip(2).collect::<Vec<_>>().join(" ");
        if prompt.is_empty() {
            eprintln!("Usage: nioxs ray <prompt>");
            return Ok(());
        }
        let engine = nioxs_engine::RayEngine::new()?;
        let mut rx = engine.generate_stream(prompt, 512, 0.7, 0.9).await?;
        use std::io::Write;
        while let Some(chunk) = rx.recv().await {
            print!("{chunk}");
            let _ = std::io::stdout().flush();
        }
        println!();
        return Ok(());
    }
    if arg.as_deref() == Some("init") {
        let target = std::env::args().nth(2).unwrap_or_else(|| "--all".into());
        match target.as_str() {
            "--aider" => {
                let config = "openai-api-base: http://127.0.0.1:7460/v1\nmodel: openai/nio-flow\nopenai-api-key: nioxs-local\n";
                std::fs::write(".aider.conf.yml", config)?;
                println!("✔ Successfully created .aider.conf.yml for Aider");
            }
            "--cline" => {
                let config = serde_json::json!({
                    "mcpServers": {
                        "nioxs": {
                            "command": "nioxs",
                            "args": ["mcp"]
                        }
                    }
                });
                std::fs::write(
                    "cline_mcp_settings.json",
                    serde_json::to_string_pretty(&config)?,
                )?;
                println!("✔ Successfully created cline_mcp_settings.json for Cline");
            }
            "--opencode" => {
                let config = serde_json::json!({
                    "provider": "openai",
                    "baseURL": "http://127.0.0.1:7460/v1",
                    "apiKey": "nioxs-local",
                    "model": "nio-flow"
                });
                std::fs::write("opencode.json", serde_json::to_string_pretty(&config)?)?;
                println!("✔ Successfully created opencode.json for OpenCode");
            }
            "--cursor" => {
                println!("Cursor OpenAI Endpoint Configuration:");
                println!("  Base URL: http://127.0.0.1:7460/v1");
                println!("  API Key:  nioxs-local");
                println!("  Model:    nio-flow");
            }
            _ => {
                println!("Auto-configuring coding agent settings in current directory...");
                let aider_cfg = "openai-api-base: http://127.0.0.1:7460/v1\nmodel: openai/nio-flow\nopenai-api-key: nioxs-local\n";
                let _ = std::fs::write(".aider.conf.yml", aider_cfg);
                let cline_cfg = serde_json::json!({
                    "mcpServers": {
                        "nioxs": {
                            "command": "nioxs",
                            "args": ["mcp"]
                        }
                    }
                });
                let _ = std::fs::write(
                    "cline_mcp_settings.json",
                    serde_json::to_string_pretty(&cline_cfg)?,
                );
                println!("✔ Configured .aider.conf.yml and cline_mcp_settings.json");
            }
        }
        return Ok(());
    }
    if arg.as_deref() == Some("undo") {
        let current_dir = std::env::current_dir().unwrap_or_else(|_| ".".into());
        let patch_id = std::env::args().nth(2);
        let id_to_rollback = if let Some(id) = patch_id {
            id
        } else {
            let undo_dir = current_dir.join(".nioxs/undo");
            let mut latest_file = None;
            let mut latest_time = std::time::SystemTime::UNIX_EPOCH;
            if let Ok(entries) = std::fs::read_dir(&undo_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("json") {
                        if let Ok(meta) = entry.metadata() {
                            if let Ok(mod_time) = meta.modified() {
                                if mod_time > latest_time {
                                    latest_time = mod_time;
                                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                                        latest_file = Some(stem.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            match latest_file {
                Some(id) => id,
                None => {
                    println!("No undo snapshots found in .nioxs/undo/");
                    return Ok(());
                }
            }
        };

        match nioxs_nano::patch::rollback_patch(&current_dir, &id_to_rollback) {
            Ok(count) => {
                println!(
                    "✔ Successfully rolled back patch '{id_to_rollback}' ({count} files restored)."
                );
            }
            Err(e) => {
                eprintln!("Failed to rollback patch '{id_to_rollback}': {e}");
            }
        }
        return Ok(());
    }
    if arg.as_deref() == Some("history") {
        let current_dir = std::env::current_dir().unwrap_or_else(|_| ".".into());
        let limit = std::env::args()
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);
        let turns = nioxs_provider::journal::read_local_turns(&current_dir, limit);
        if turns.is_empty() {
            println!("No local turn history found in .nioxs/history.jsonl");
        } else {
            println!("Recent Turns ({} total):", turns.len());
            for (idx, turn) in turns.iter().enumerate() {
                println!(
                    "[{:2}] ts:{} | Intent: {:<12} | Outcome: {:<10}\n     Prompt: {}\n",
                    idx + 1,
                    turn.timestamp_unix,
                    turn.intent,
                    turn.outcome,
                    turn.prompt.chars().take(80).collect::<String>()
                );
            }
        }
        return Ok(());
    }
    if arg.as_deref() == Some("mcp") {
        return nioxs_provider::mcp::run_stdio_server().map_err(|e| e.into());
    }
    if arg.is_some() && arg.as_deref() != Some("serve") {
        return Err("unknown command; expected 'nioxs', 'nioxs serve', 'nioxs init', 'nioxs mcp', 'nioxs undo', or 'nioxs history'".into());
    }
    let listen: std::net::SocketAddr = if let Ok(listen_str) = std::env::var("NIOXS_LISTEN") {
        listen_str.parse()?
    } else if let Ok(port_str) = std::env::var("PORT") {
        let port: u16 = port_str.parse()?;
        std::net::SocketAddr::from(([0, 0, 0, 0], port))
    } else {
        "127.0.0.1:7460".parse()?
    };
    let (db_url, db_token, _niodb_proc) = nioxs_provider::niodb_supervisor::ensure_niodb(
        std::env::var("NIOXS_DB_URL").ok().as_deref(),
        std::env::var("NIOXS_DB_TOKEN_FILE").ok().as_deref(),
    )
    .await?;

    let api_token = if let Ok(direct) = std::env::var("NIOXS_API_TOKEN") {
        let trimmed = direct.trim().to_string();
        if trimmed.is_empty() {
            return Err("provider token cannot be empty".into());
        }
        Some(trimmed)
    } else if let Ok(file) = std::env::var("NIOXS_API_TOKEN_FILE") {
        let content = std::fs::read_to_string(file)?;
        let trimmed = content.trim().to_string();
        if trimmed.is_empty() {
            return Err("provider token cannot be empty".into());
        }
        Some(trimmed)
    } else {
        None
    };

    let allow_unauthenticated = std::env::var("NIOXS_ALLOW_UNAUTHENTICATED")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if !listen.ip().is_loopback() && api_token.is_none() && !allow_unauthenticated {
        return Err("non-loopback deployment requires NIOXS_API_TOKEN or NIOXS_API_TOKEN_FILE (or set NIOXS_ALLOW_UNAUTHENTICATED=true)".into());
    }
    let deployment = std::env::var("NIOXS_DEPLOYMENT_ID").unwrap_or_else(|_| "local".into());
    if deployment.is_empty()
        || deployment.len() > 64
        || !deployment
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
    {
        return Err("invalid deployment ID".into());
    }
    let db = NioDb::new(&db_url, db_token)?;
    let model = match std::env::var("NIOXS_MODEL_FILE") {
        Ok(file) => {
            let bytes = std::fs::read(file)?;
            let expected = std::env::var("NIOXS_MODEL_SHA256")
                .map_err(|_| "external model requires NIOXS_MODEL_SHA256")?;
            if nioxs_contracts::digest(&bytes) != expected {
                return Err("model checksum mismatch".into());
            }
            Model::load(&bytes)?
        }
        Err(_) => Model::embedded()?,
    };
    let instance = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    );
    let ray_engine = nioxs_engine::RayEngine::new().ok();
    let microlm_banner = if ray_engine.is_some() {
        "NioRay (In-Process Predator, Candle GGUF)"
    } else {
        "NioNano (<0.5ms decision triage)"
    };
    let listener = tokio::net::TcpListener::bind(listen).await?;
    let addr = listener.local_addr()?;
    eprintln!(
        "\n\
  ╭──────────────────────────────────────────────────────────╮\n\
  │  ⚡ NioXS v{:<8} Pure Coding Companion Gateway        │\n\
  ├──────────────────────────────────────────────────────────┤\n\
  │  • Gateway API : http://{:<33} │\n\
  │  • NioDB Store : {:<40} │\n\
  │  • MicroLM     : {:<40} │\n\
  │  • Directives  : --bash, --tools, --code, --test         │\n\
  ╰──────────────────────────────────────────────────────────╯\n",
        env!("CARGO_PKG_VERSION"),
        addr,
        db_url,
        microlm_banner,
    );
    axum::serve(
        listener,
        router(App {
            db,
            deployment,
            api_token,
            active: ActiveTurns::default(),
            model: Arc::new(model),
            ray_engine,
            instance,
        }),
    )
    .with_graceful_shutdown(async {
        let _ = tokio::signal::ctrl_c().await;
    })
    .await?;
    Ok(())
}
