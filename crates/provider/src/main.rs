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
            "Usage: nioxs [serve|mcp]\nCommands:\n  serve   Start HTTP OpenAI-compatible AI gateway (default)\n  mcp     Start Model Context Protocol (MCP) JSON-RPC stdio server\nEnvironment: NIOXS_LISTEN (127.0.0.1:7460), NIOXS_DB_URL, NIOXS_DB_TOKEN_FILE, NIOXS_DEPLOYMENT_ID, NIOXS_API_TOKEN_FILE (optional on loopback)."
        );
        return Ok(());
    }
    if arg.as_deref() == Some("mcp") {
        return nioxs_provider::mcp::run_stdio_server().map_err(|e| e.into());
    }
    if arg.is_some() && arg.as_deref() != Some("serve") {
        return Err("unknown command; expected 'nioxs', 'nioxs serve', or 'nioxs mcp'".into());
    }
    let listen: std::net::SocketAddr = std::env::var("NIOXS_LISTEN")
        .unwrap_or_else(|_| "127.0.0.1:7460".into())
        .parse()?;
    let token_file =
        std::env::var("NIOXS_DB_TOKEN_FILE").map_err(|_| "NIOXS_DB_TOKEN_FILE is required")?;
    let db_token = std::fs::read_to_string(token_file)?.trim().to_string();
    let api_token = std::env::var("NIOXS_API_TOKEN_FILE")
        .ok()
        .map(std::fs::read_to_string)
        .transpose()?
        .map(|s| s.trim().to_string());
    if api_token.as_ref().is_some_and(String::is_empty) {
        return Err("provider token cannot be empty".into());
    }
    if !listen.ip().is_loopback() && api_token.is_none() {
        return Err("non-loopback deployment requires NIOXS_API_TOKEN_FILE".into());
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
    let db = NioDb::new(
        &std::env::var("NIOXS_DB_URL").unwrap_or_else(|_| "http://127.0.0.1:7432".into()),
        db_token,
    )?;
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
    let listener = tokio::net::TcpListener::bind(listen).await?;
    eprintln!(
        "NioXS {} listening on {} (Phase 2 trained Nano)",
        env!("CARGO_PKG_VERSION"),
        listener.local_addr()?
    );
    axum::serve(
        listener,
        router(App {
            db,
            deployment,
            api_token,
            active: ActiveTurns::default(),
            model: Arc::new(model),
            instance,
        }),
    )
    .with_graceful_shutdown(async {
        let _ = tokio::signal::ctrl_c().await;
    })
    .await?;
    Ok(())
}
