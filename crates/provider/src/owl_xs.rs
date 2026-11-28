//! Owl XS (`vMicroLM Atlas`) CLI runner: 135M non-coding world-knowledge,
//! financial/data analysis (with deterministic Rust CSV/Math Guard), and storytelling.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use nioxs_engine::{AstEngine, DevicePreference};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwlFolio {
    Kids,
    Encyclopedia,
    Finance,
    Novel,
}

impl OwlFolio {
    pub fn slug(self) -> &'static str {
        match self {
            Self::Kids => "folio-kids",
            Self::Encyclopedia => "folio-encyclopedia",
            Self::Finance => "folio-finance",
            Self::Novel => "folio-novel",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Kids => "Kids Folio",
            Self::Encyclopedia => "Encyclopedia Folio",
            Self::Finance => "Finance / Analyst Folio",
            Self::Novel => "Novel Folio",
        }
    }

    pub fn system_prompt(self) -> &'static str {
        match self {
            Self::Kids => "You are Owl XS (Kids Folio), a warm children's storyteller created by Nio Labs.",
            Self::Encyclopedia => {
                "You are Owl XS (Encyclopedia Folio), created by Nio Labs. Answer strictly from the provided passage or context. If the context does not contain the answer, say 'Not supported by passage.'"
            }
            Self::Finance => {
                "You are Owl XS (Finance Folio), created by Nio Labs. Analyze the financial data accurately using only verified numbers from [Atlas Math Guard] and the provided context."
            }
            Self::Novel => {
                "You are Owl XS (Novel Folio), created by Nio Labs. Write vivid, emotionally grounded fiction consistent with any provided Story Bible."
            }
        }
    }
}

pub fn parse_folio_flag(flag: &str) -> Option<OwlFolio> {
    match flag.trim().to_ascii_lowercase().as_str() {
        "kids" | "folio-kids" | "story" => Some(OwlFolio::Kids),
        "encyclopedia" | "folio-encyclopedia" | "wiki" | "knowledge" => {
            Some(OwlFolio::Encyclopedia)
        }
        "finance" | "folio-finance" | "analyst" | "csv" => Some(OwlFolio::Finance),
        "novel" | "folio-novel" | "romance" | "fiction" => Some(OwlFolio::Novel),
        _ => None,
    }
}

pub fn parse_route_token(raw: &str) -> OwlFolio {
    let token = raw
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_matches(|c: char| !c.is_ascii_alphanumeric())
        .to_ascii_lowercase();
    match token.as_str() {
        "kids" => OwlFolio::Kids,
        "finance" => OwlFolio::Finance,
        "novel" => OwlFolio::Novel,
        _ => OwlFolio::Encyclopedia,
    }
}

async fn route_folio_dynamically(engine: &mut AstEngine, prompt: &str) -> OwlFolio {
    let route_prompt = format!(
        "<|im_start|>system\nClassify the user request into exactly one word: kids, encyclopedia, finance, or novel.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    );
    let Ok(mut rx) = engine.generate_stream(route_prompt, 4, 0.0, 1.0).await else {
        return OwlFolio::Encyclopedia;
    };
    let mut raw = String::new();
    while let Some(chunk) = rx.recv().await {
        raw.push_str(&chunk);
    }
    parse_route_token(&raw)
}

/// Deterministic Rust CSV/numeric evaluator for Finance Folio so 135M never relies on mental math.
pub fn compute_csv_guard(input: &str) -> Option<String> {
    let lines: Vec<&str> = input
        .lines()
        .flat_map(|l| l.split('|'))
        .map(str::trim)
        .filter(|l| l.contains(','))
        .collect();
    if lines.len() < 2 {
        return None;
    }

    let headers: Vec<&str> = lines[0].split(',').map(str::trim).collect();
    if headers.len() < 2 {
        return None;
    }

    let mut col_sums = vec![0.0_f64; headers.len()];
    let mut col_counts = vec![0_usize; headers.len()];
    let mut row_notes = Vec::new();

    for line in &lines[1..] {
        let cells: Vec<&str> = line.split(',').map(str::trim).collect();
        if cells.len() != headers.len() {
            continue;
        }
        let mut parsed_vals = Vec::new();
        for (idx, cell) in cells.iter().enumerate().skip(1) {
            let cleaned: String = cell
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                .collect();
            if let Ok(v) = cleaned.parse::<f64>() {
                col_sums[idx] += v;
                col_counts[idx] += 1;
                parsed_vals.push((headers[idx], *cell, v));
            }
        }
        if !parsed_vals.is_empty() {
            let row_sum: f64 = parsed_vals.iter().map(|(_, _, v)| *v).sum();
            row_notes.push(format!("- {}: row_sum={:.2}", cells[0], row_sum));
        }
    }

    if row_notes.is_empty() {
        return None;
    }

    let mut summary = String::from("[Atlas Math Guard — Verified Calculations]\n");
    for idx in 1..headers.len() {
        if col_counts[idx] > 0 {
            let avg = col_sums[idx] / col_counts[idx] as f64;
            summary.push_str(&format!(
                "- {} (n={}): sum={:.2}, avg={:.2}\n",
                headers[idx], col_counts[idx], col_sums[idx], avg
            ));
        }
    }
    for note in row_notes {
        summary.push_str(&note);
        summary.push('\n');
    }
    Some(summary)
}

fn resolve_owl_xs_paths(models_dir: &Path) -> Result<(PathBuf, PathBuf)> {
    let tok = [
        models_dir.join("orca-xs-tokenizer.json"),
        models_dir.join("baby-orca-tokenizer.json"),
    ]
    .into_iter()
    .find(|p| p.is_file())
    .ok_or_else(|| {
        anyhow::anyhow!(
            "Could not locate orca-xs-tokenizer.json in {} (run `nioxs pull owl-xs`)",
            models_dir.display()
        )
    })?;

    let model = [
        models_dir.join("owl-xs-135m-q4_k_m.gguf"),
        models_dir.join("orca-xs-lead-135m-q4_k_m.gguf"),
        models_dir.join("mink-xs-scout-135m-q4_k_m.gguf"),
        models_dir.join("baby-meerkat-scout-135m-q4_k_m.gguf"),
    ]
    .into_iter()
    .find(|p| p.is_file())
    .ok_or_else(|| {
        anyhow::anyhow!(
            "Could not locate Owl XS 135M GGUF in {} (run `nioxs pull owl-xs`)",
            models_dir.display()
        )
    })?;

    Ok((model, tok))
}

fn resolve_folio_lora(models_dir: &Path, folio: OwlFolio) -> Option<PathBuf> {
    let base = models_dir
        .join("owl_xs_loras")
        .join(format!("owl-xs-135m.{}", folio.slug()));
    if base.is_dir() {
        Some(base)
    } else {
        None
    }
}

pub async fn run_owl_xs_cli(args: Vec<String>) -> Result<()> {
    let mut explicit_folio: Option<OwlFolio> = None;
    let mut context_parts: Vec<String> = Vec::new();
    let mut prompt_parts: Vec<String> = Vec::new();
    let mut max_tokens: usize = 512;
    let mut verbose = false;
    let mut device_pref = DevicePreference::Auto;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!(
                    "Usage: nioxs owl-xs [--kids|--encyclopedia|--finance|--novel] [--folio <name>] [--context <text>] [--file <path>] [-n <max_tokens>] [--verbose] <prompt>"
                );
                return Ok(());
            }
            "--kids" => explicit_folio = Some(OwlFolio::Kids),
            "--encyclopedia" | "--wiki" => explicit_folio = Some(OwlFolio::Encyclopedia),
            "--finance" | "--analyst" => explicit_folio = Some(OwlFolio::Finance),
            "--novel" | "--romance" => explicit_folio = Some(OwlFolio::Novel),
            "--verbose" | "-v" => verbose = true,
            "--cpu" => device_pref = DevicePreference::Cpu,
            "--gpu" => device_pref = DevicePreference::Gpu,
            "--auto" => device_pref = DevicePreference::Auto,
            "--folio" if i + 1 < args.len() => {
                i += 1;
                explicit_folio = parse_folio_flag(&args[i]);
            }
            "--context" if i + 1 < args.len() => {
                i += 1;
                context_parts.push(args[i].clone());
            }
            "--file" if i + 1 < args.len() => {
                i += 1;
                let data = std::fs::read_to_string(&args[i])
                    .with_context(|| format!("Failed to read --file {}", args[i]))?;
                context_parts.push(data);
            }
            "--max-tokens" | "-n" if i + 1 < args.len() => {
                i += 1;
                max_tokens = args[i].parse::<usize>().unwrap_or(512);
            }
            other => prompt_parts.push(other.to_string()),
        }
        i += 1;
    }

    let prompt = prompt_parts.join(" ");
    if prompt.trim().is_empty() {
        anyhow::bail!("Empty prompt. Usage: nioxs owl-xs <prompt>");
    }

    let context_block = context_parts.join("\n\n");
    let combined_for_csv = if context_block.is_empty() {
        prompt.as_str()
    } else {
        context_block.as_str()
    };
    let csv_guard = compute_csv_guard(combined_for_csv);
    let models_dir = nioxs_artifacts::resolve_models_dir();
    let (model_path, tok_path) = resolve_owl_xs_paths(&models_dir)?;

    let mut engine = AstEngine::from_paths(model_path.clone(), tok_path);
    engine.set_device_preference(device_pref);

    let folio = match explicit_folio {
        Some(f) => f,
        None if csv_guard.is_some() => OwlFolio::Finance,
        None => route_folio_dynamically(&mut engine, &prompt).await,
    };
    let lora_path = resolve_folio_lora(&models_dir, folio);

    if verbose {
        eprintln!(
            "\x1b[1;36m[Owl XS]\x1b[0m Mode: {} | Base: {} | LoRA: {}",
            folio.label(),
            model_path.file_name().and_then(|n| n.to_str()).unwrap_or("135M"),
            lora_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "none (base 135M)".to_string())
        );
    }

    if let Some(ref lp) = lora_path {
        let _ = engine.attach_lora_head(&lp.to_string_lossy()).await;
    }

    let mut user_msg = String::new();
    if let Some(ref guard) = csv_guard {
        user_msg.push_str(guard);
        user_msg.push('\n');
    }
    if !context_block.is_empty() {
        user_msg.push_str("Context:\n");
        user_msg.push_str(&context_block);
        user_msg.push_str("\n\n");
    }
    user_msg.push_str(&prompt);

    let formatted = format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n",
        folio.system_prompt()
    );
    let is_grounded = explicit_folio.is_some_and(|f| matches!(f, OwlFolio::Encyclopedia | OwlFolio::Finance))
        || csv_guard.is_some()
        || !context_block.is_empty();
    let temp = if is_grounded { 0.0 } else { 0.60 };

    let mut rx = engine
        .generate_stream(formatted, max_tokens, temp, 0.90)
        .await?;
    while let Some(chunk) = rx.recv().await {
        print!("{chunk}");
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_guard_computes_exact_totals() {
        let csv = "Quarter,Revenue,COGS,OpEx | Q1,100M,40M,35M | Q2,120M,45M,38M | Q3,115M,50M,40M";
        let guard = compute_csv_guard(csv).expect("should parse csv");
        assert!(guard.contains("Revenue (n=3): sum=335.00, avg=111.67"));
        assert!(guard.contains("COGS (n=3): sum=135.00, avg=45.00"));
        assert!(guard.contains("OpEx (n=3): sum=113.00, avg=37.67"));
    }

    #[test]
    fn test_folio_flag_and_route_token() {
        assert_eq!(parse_folio_flag("kids"), Some(OwlFolio::Kids));
        assert_eq!(parse_folio_flag("analyst"), Some(OwlFolio::Finance));
        assert_eq!(parse_route_token("kids."), OwlFolio::Kids);
        assert_eq!(parse_route_token("finance"), OwlFolio::Finance);
        assert_eq!(parse_route_token("novel\n"), OwlFolio::Novel);
        assert_eq!(parse_route_token("encyclopedia"), OwlFolio::Encyclopedia);
    }
}
