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
            "Usage: nioxs [serve|pull <model>|viper <prompt>|ray <prompt>|manta <prompt>|mcp|init|undo|history]\n\
Commands:\n\
  serve          Start HTTP OpenAI-compatible AI gateway (default)\n\
  pull [model]   Download and verify local MicroLM weights (viper, ray, manta, cuda, all)\n\
  viper [-x|-y] <prompt> Run local NioViper 50M Swiss Gear reflex (-x executes natively with safety clutch)\n\
  viper plugin <install|list> Manage isolated Python plugins for NioViper sandbox\n\
  ray <prompt>   Run local in-process NioRay predator inference on prompt\n\
  manta <prompt> Run local in-process NioManta ONNX companion inference on prompt\n\
  mcp            Start Model Context Protocol (MCP) JSON-RPC stdio server\n\
  init           Initialize configuration for coding agents (--aider, --cline, --opencode, --cursor)\n\
  undo           Rollback last applied multi-file patch (or specific patch_id)\n\
  history        Display local session turn journal from .nioxs/history.jsonl\n\
\n\
Environment: NIOXS_LISTEN (127.0.0.1:7460), NIOXS_DB_URL, NIOXS_DB_TOKEN_FILE, NIOXS_DEPLOYMENT_ID, NIOXS_API_TOKEN_FILE (optional on loopback)."
        );
        return Ok(());
    }
    if arg.as_deref() == Some("pull") {
        let model_target = std::env::args().nth(2).unwrap_or_else(|| "all".into());
        let auto_yes = std::env::args().any(|a| a == "-y" || a == "--yes");
        let manifest = nioxs_artifacts::default_manifest();
        if model_target == "all" {
            println!("Pulling all production MicroLM models...");
            for entry in &manifest {
                let _ = nioxs_artifacts::ensure_model_ready(&entry.model_id, None)
                    .map_err(|e| format!("Failed to download {}: {e}", entry.model_id))?;
            }
            println!("✔ All production models downloaded and verified.");
            return Ok(());
        }
        let lowered = model_target.trim().to_ascii_lowercase();
        let normalized = match lowered.as_str() {
            "viper" | "nio-viper" => "nio-viper",
            "ray" | "nio-ray" => "nio-ray",
            "manta" | "nio-manta" => "nio-manta",
            "cuda" | "nio-cuda" => "nio-cuda",
            other => other,
        };
        let path = nioxs_artifacts::prompt_and_pull_if_needed(normalized, auto_yes)
            .map_err(|e| format!("Failed to pull {normalized}: {e}"))?;
        println!("✔ Model '{normalized}' ready: {}", path.display());
        return Ok(());
    }
    if arg.as_deref() == Some("viper") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!("Usage: nioxs viper [-x|--run] [-y|--yes] <prompt>");
            eprintln!("       nioxs viper plugin <install <pkg>|list>");
            return Ok(());
        }

        // Subcommand: Python Plugin Sandbox Manager
        if raw_args.first().map(|s| s.as_str()) == Some("plugin") {
            let action = raw_args.get(1).map(|s| s.as_str()).unwrap_or("help");
            let home = std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .unwrap_or_else(|_| ".".into());
            let plugin_dir = std::path::PathBuf::from(home).join(".nioxs").join("plugins");
            let venv_dir = plugin_dir.join("venv");
            std::fs::create_dir_all(&plugin_dir)?;

            match action {
                "install" => {
                    let pkg = raw_args.get(2).ok_or("Usage: nioxs viper plugin install <package_name>")?;
                    if !venv_dir.exists() {
                        println!("📦 Initializing isolated Viper plugin sandbox in {}...", venv_dir.display());
                        let venv_status = std::process::Command::new("python3")
                            .args(["-m", "venv", &venv_dir.to_string_lossy()])
                            .status()
                            .or_else(|_| {
                                std::process::Command::new("python")
                                    .args(["-m", "venv", &venv_dir.to_string_lossy()])
                                    .status()
                            })?;
                        if !venv_status.success() {
                            return Err("Failed to create Python virtualenv. Ensure python3 is installed.".into());
                        }
                    }
                    #[cfg(target_os = "windows")]
                    let pip_bin = venv_dir.join("Scripts").join("pip.exe");
                    #[cfg(not(target_os = "windows"))]
                    let pip_bin = venv_dir.join("bin").join("pip");

                    println!("📦 Installing plugin '{pkg}' into isolated Viper sandbox...");
                    let status = std::process::Command::new(pip_bin)
                        .args(["install", pkg])
                        .status()?;
                    if status.success() {
                        println!("✔ Plugin '{pkg}' successfully installed in Viper sandbox!");
                    } else {
                        return Err(format!("Failed to install plugin '{pkg}'").into());
                    }
                    return Ok(());
                }
                "list" => {
                    if !venv_dir.exists() {
                        println!("No plugins installed yet. Use 'nioxs viper plugin install <package>'");
                        return Ok(());
                    }
                    #[cfg(target_os = "windows")]
                    let pip_bin = venv_dir.join("Scripts").join("pip.exe");
                    #[cfg(not(target_os = "windows"))]
                    let pip_bin = venv_dir.join("bin").join("pip");

                    println!("📦 Installed NioViper Plugins (isolated in {}):", venv_dir.display());
                    let _ = std::process::Command::new(pip_bin)
                        .args(["list"])
                        .status()?;
                    return Ok(());
                }
                _ => {
                    println!("Usage:\n  nioxs viper plugin install <package>\n  nioxs viper plugin list");
                    return Ok(());
                }
            }
        }

        // Parse Swiss Gear execution flags: -x / --run, -y / --yes
        let mut execute = false;
        let mut auto_yes = false;
        let mut prompt_tokens = Vec::new();

        for arg in raw_args {
            if arg == "-x" || arg == "--run" {
                execute = true;
            } else if arg == "-y" || arg == "--yes" {
                auto_yes = true;
            } else {
                prompt_tokens.push(arg);
            }
        }

        let prompt = prompt_tokens.join(" ");
        if prompt.is_empty() {
            eprintln!("Usage: nioxs viper [-x|--run] [-y|--yes] <prompt>");
            return Ok(());
        }

        let engine = nioxs_engine::RayEngine::for_model("nio-viper")?;
        let max_tokens = if prompt.split_whitespace().count() <= 2 { 32 } else { 128 };
        let mut rx = engine.generate_stream(prompt.clone(), max_tokens, 0.7, 0.9).await?;
        use std::io::Write;
        let mut generated_cmd = String::new();
        while let Some(chunk) = rx.recv().await {
            print!("{chunk}");
            let _ = std::io::stdout().flush();
            generated_cmd.push_str(&chunk);
        }
        println!();

        let clean_cmd = ground_and_sanitize_command(&prompt, &generated_cmd);
        if clean_cmd != generated_cmd.trim() && !execute {
            println!("🎯 Grounded reflex: {clean_cmd}");
        }

        if execute {
            if clean_cmd.is_empty() {
                eprintln!("No executable command was generated.");
                return Ok(());
            }

            // Safety Clutch: Check for destructive mutations
            let is_destructive = clean_cmd.starts_with("rm ")
                || clean_cmd.contains("rm -rf")
                || clean_cmd.contains("kill -9")
                || clean_cmd.contains("pkill -9")
                || clean_cmd.contains("git reset --hard")
                || clean_cmd.contains("Stop-Process")
                || clean_cmd.contains("Remove-Item")
                || clean_cmd.contains("drop database");

            if is_destructive && !auto_yes {
                print!("⚡ Destructive command detected. Execute: \"{clean_cmd}\"? [Y/n] ");
                let _ = std::io::stdout().flush();
                let mut input = String::new();
                std::io::stdin().read_line(&mut input)?;
                let trimmed = input.trim().to_lowercase();
                if !trimmed.is_empty() && trimmed != "y" && trimmed != "yes" {
                    println!("Execution canceled.");
                    return Ok(());
                }
            }

            println!("🚀 Executing action: {clean_cmd}");
            #[cfg(target_os = "windows")]
            let mut child = std::process::Command::new("powershell")
                .args(["-NoProfile", "-Command", &clean_cmd])
                .spawn()?;

            #[cfg(not(target_os = "windows"))]
            let mut child = std::process::Command::new("sh")
                .args(["-c", &clean_cmd])
                .spawn()?;

            let status = child.wait()?;
            if !status.success() {
                eprintln!("Action finished with status: {status}");
            }
        }

        return Ok(());
    }
    if arg.as_deref() == Some("ray") {
        let prompt = std::env::args().skip(2).collect::<Vec<_>>().join(" ");
        if prompt.is_empty() {
            eprintln!("Usage: nioxs ray <prompt>");
            return Ok(());
        }
        let engine = nioxs_engine::RayEngine::new()?;
        let max_tokens = if prompt.split_whitespace().count() <= 2 { 32 } else { 128 };
        let mut rx = engine.generate_stream(prompt, max_tokens, 0.7, 0.9).await?;
        use std::io::Write;
        while let Some(chunk) = rx.recv().await {
            print!("{chunk}");
            let _ = std::io::stdout().flush();
        }
        println!();
        return Ok(());
    }
    if arg.as_deref() == Some("manta") {
        let raw_args: Vec<String> = std::env::args().skip(2).collect();
        if raw_args.is_empty() {
            eprintln!("Usage: nioxs manta <prompt> | nioxs manta --plan <prompt> | nioxs manta --benchmark");
            return Ok(());
        }

        let mut is_plan = false;
        let mut prompt_parts = Vec::new();
        for a in &raw_args {
            if a == "--plan" {
                is_plan = true;
            } else if a == "--benchmark" || a == "-b" {
                // Keep for python benchmark if needed
            } else if a != "-i" && a != "--interactive" {
                prompt_parts.push(a.as_str());
            }
        }
        let user_prompt = prompt_parts.join(" ");

        // Prefer native Rust RayEngine for nio-manta
        if let Ok(engine) = nioxs_engine::RayEngine::for_model("nio-manta") {
            let sys_msg = if is_plan {
                "You are Manta, Nio's Conversational Guru and Wasm Architect. Compile the user request into a machine-verifiable <manta_plan> XML DAG with step id, name, description, command, and rollback."
            } else {
                "You are Manta, Nio's Conversational Guru and Wasm Architect. Answer with high technical precision, empathy, and conciseness."
            };
            let formatted_prompt = format!("<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n");
            let max_tokens = if is_plan { 384 } else { 160 };
            let mut rx = engine.generate_stream(formatted_prompt, max_tokens, 0.2, 0.9).await?;
            use std::io::Write;
            while let Some(chunk) = rx.recv().await {
                print!("{chunk}");
                let _ = std::io::stdout().flush();
            }
            println!();
            return Ok(());
        }

        // Fallback: Locate runner script
        let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf()));
        let mut script_candidates = vec![
            std::path::PathBuf::from("scripts/run_niomanta_onnx.py"),
            std::path::PathBuf::from("/home/mn/nio-labs/nio-xs/scripts/run_niomanta_onnx.py"),
        ];
        if let Some(dir) = exe_dir {
            script_candidates.push(dir.join("../scripts/run_niomanta_onnx.py"));
            script_candidates.push(dir.join("scripts/run_niomanta_onnx.py"));
        }

        let script = script_candidates.into_iter().find(|p| p.exists());

        // Locate python interpreter
        let py_candidates = [
            std::env::var("NIOXS_PYTHON").ok(),
            Some("/home/mn/nio-labs/.venv/bin/python3".into()),
            Some(".venv/bin/python3".into()),
            Some("python3".into()),
        ];
        let python = py_candidates.into_iter().flatten().find(|p| {
            std::process::Command::new(p)
                .arg("--version")
                .output()
                .is_ok()
        }).unwrap_or_else(|| "python3".into());

        if let Some(script_path) = script {
            let mut cmd = std::process::Command::new(&python);
            cmd.arg(&script_path);

            let mut is_plan = false;
            let mut prompt_parts = Vec::new();

            for a in raw_args {
                if a == "--plan" {
                    is_plan = true;
                } else if a == "--benchmark" || a == "-b" {
                    cmd.arg("--benchmark");
                } else if a == "-i" || a == "--interactive" {
                    cmd.arg("--interactive");
                } else {
                    prompt_parts.push(a);
                }
            }

            if is_plan {
                cmd.arg("--plan");
            }
            let prompt = prompt_parts.join(" ");
            if !prompt.is_empty() {
                cmd.arg("-p").arg(prompt);
            }

            let status = cmd.status()?;
            if !status.success() {
                std::process::exit(status.code().unwrap_or(1));
            }
            return Ok(());
        } else {
            eprintln!("NioManta ONNX runner script not found. Expected at scripts/run_niomanta_onnx.py");
            return Ok(());
        }
    }
    if arg.as_deref() == Some("init") {
        let target = std::env::args().nth(2).unwrap_or_else(|| "--all".into());
        match target.as_str() {
            "--aider" => {
                let config = "openai-api-base: http://127.0.0.1:7460/v1\nmodel: openai/nio-cuda\nopenai-api-key: nioxs-local\n";
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
                    "model": "nio-cuda"
                });
                std::fs::write("opencode.json", serde_json::to_string_pretty(&config)?)?;
                println!("✔ Successfully created opencode.json for OpenCode");
            }
            "--cursor" => {
                println!("Cursor OpenAI Endpoint Configuration:");
                println!("  Base URL: http://127.0.0.1:7460/v1");
                println!("  API Key:  nioxs-local");
                println!("  Model:    nio-cuda");
            }
            _ => {
                println!("Auto-configuring coding agent settings in current directory...");
                let aider_cfg = "openai-api-base: http://127.0.0.1:7460/v1\nmodel: openai/nio-cuda\nopenai-api-key: nioxs-local\n";
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
    if let Some(engine) = &ray_engine {
        let _ = engine.ensure_loaded().await;
    }
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

fn ground_and_sanitize_command(prompt: &str, raw_cmd: &str) -> String {
    let mut clean_cmd = raw_cmd
        .trim()
        .trim_start_matches("```powershell")
        .trim_start_matches("```bash")
        .trim_start_matches("```sh")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim()
        .to_string();

    // 1. Sanitize trailing dangling flags (e.g. "ls -" -> "ls -la")
    if clean_cmd == "ls -" || clean_cmd.ends_with("ls -") {
        clean_cmd = clean_cmd.replace("ls -", "ls -la");
    } else if clean_cmd.ends_with(" -") {
        clean_cmd = clean_cmd.trim_end_matches(" -").trim().to_string();
    } else if clean_cmd.ends_with(" --") {
        clean_cmd = clean_cmd.trim_end_matches(" --").trim().to_string();
    }

    // 2. Extract quoted literal terms from prompt (e.g. 'TODO' or "TODO")
    let prompt_quoted: Vec<&str> = {
        let mut list = Vec::new();
        for quote_char in ['\'', '"'] {
            let mut start = None;
            for (i, c) in prompt.char_indices() {
                if c == quote_char {
                    if let Some(s) = start {
                        if i > s + 1 {
                            list.push(&prompt[s + 1..i]);
                        }
                        start = None;
                    } else {
                        start = Some(i);
                    }
                }
            }
        }
        list
    };

    // If user specified an explicit search term, align the pattern in the reflex
    if let Some(&user_term) = prompt_quoted.first() {
        // Handle -e '...' or -e "..."
        if let Some(pos) = clean_cmd.find("-e ") {
            let after = &clean_cmd[pos + 3..];
            if after.starts_with('\'') || after.starts_with('"') {
                let quote = after.chars().next().unwrap();
                if let Some(end_quote) = after[1..].find(quote) {
                    let full_flag = &clean_cmd[pos..pos + 3 + 1 + end_quote + 1];
                    let replacement = format!("-e '{user_term}'");
                    clean_cmd = clean_cmd.replace(full_flag, &replacement);
                }
            }
        }
        // Handle -Pattern '...' or -Pattern "..."
        if let Some(pos) = clean_cmd.find("-Pattern ") {
            let after = &clean_cmd[pos + 9..];
            if after.starts_with('\'') || after.starts_with('"') {
                let quote = after.chars().next().unwrap();
                if let Some(end_quote) = after[1..].find(quote) {
                    let full_flag = &clean_cmd[pos..pos + 9 + 1 + end_quote + 1];
                    let replacement = format!("-Pattern '{user_term}'");
                    clean_cmd = clean_cmd.replace(full_flag, &replacement);
                }
            }
        }
        // Handle grep without -e, e.g. grep -rn 'panic' or grep -rnw 'panic'
        if clean_cmd.starts_with("grep ") && !clean_cmd.contains("-e ") {
            for quote in ['\'', '"'] {
                if let Some(first_q) = clean_cmd.find(quote) {
                    if let Some(second_q) = clean_cmd[first_q + 1..].find(quote) {
                        let matched_slice = &clean_cmd[first_q..first_q + 1 + second_q + 1];
                        if !matched_slice.contains('/') && matched_slice != format!("{quote}.{quote}") {
                            let replacement = format!("'{user_term}'");
                            clean_cmd = clean_cmd.replace(matched_slice, &replacement);
                            break;
                        }
                    }
                }
            }
        }
    }

    // 3. Fallback system directory sanitization:
    // If prompt didn't specify system paths like /var/log, /opt, /var/cache, /data, /home/app,
    // don't scan them with grep/find, fallback to '.'
    let system_paths = [
        "'/var/log'", "\"/var/log\"",
        "'/var/cache'", "\"/var/cache\"",
        "'/home/app'", "\"/home/app\"",
        "'/opt'", "\"/opt\"",
        "'/data'", "\"/data\"",
    ];

    for path in system_paths {
        let stripped = path.trim_matches('\'').trim_matches('"');
        if clean_cmd.contains(path) && !prompt.contains(stripped) {
            clean_cmd = clean_cmd.replace(path, "'.'");
        }
    }

    // 4. Port number slot grounding:
    // If prompt mentions a port number (e.g. "port 3000" or "port 8080")
    for word in prompt.split_whitespace() {
        let clean_word = word.trim_matches(|c: char| !c.is_numeric());
        if let Ok(port_num) = clean_word.parse::<u16>() {
            if port_num >= 80 {
                let port_str = port_num.to_string();
                for default_port in ["3000", "8080", "5432", "8000", "9000"] {
                    if default_port != port_str && clean_cmd.contains(default_port) {
                        clean_cmd = clean_cmd.replace(default_port, &port_str);
                    }
                }
            }
        }
    }

    // 5. Code & Spider Search Hygiene (Binary noise suppression, directory exclusions, extension filter)
    if clean_cmd.starts_with("grep ") {
        // Prevent binary match spam (target/*.rmeta, *.rlib, *.so, *.exe)
        if clean_cmd.contains("-rn ") {
            clean_cmd = clean_cmd.replace("-rn ", "-rIn ");
        } else if clean_cmd.contains("-rni ") {
            clean_cmd = clean_cmd.replace("-rni ", "-rIni ");
        } else if clean_cmd.contains("-rnw ") {
            clean_cmd = clean_cmd.replace("-rnw ", "-rInw ");
        }

        // Exclude build artifacts and caches
        if !clean_cmd.contains("--exclude-dir") && !prompt.contains("target") && !prompt.contains(".git") {
            clean_cmd.push_str(" --exclude-dir={target,.git,node_modules}");
        }

        // File extension grounding (e.g. ".rs files", ".py files", ".ts files")
        if !clean_cmd.contains("--include") {
            for word in prompt.split_whitespace() {
                let clean_w = word.trim_matches(|c: char| c == '\'' || c == '"' || c == ',');
                if clean_w.starts_with('.') && clean_w.len() >= 2 && clean_w.len() <= 6 && clean_w[1..].chars().all(|c| c.is_alphanumeric()) {
                    let ext = clean_w.trim_start_matches('.');
                    clean_cmd.push_str(&format!(" --include=\"*.{ext}\""));
                    break;
                }
            }
        }

        // Ensure search directory anchor exists
        let parts: Vec<&str> = clean_cmd.split_whitespace().collect();
        let last = parts.last().copied().unwrap_or("");
        if last.starts_with('-') || clean_cmd.ends_with('}') || clean_cmd.ends_with('"') || clean_cmd.ends_with('\'') {
            clean_cmd.push_str(" .");
        }
    }

    clean_cmd
}
