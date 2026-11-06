use anyhow::{Result, anyhow};
use nioxs_engine::ast_engine::AstMicroEngine;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

#[derive(Debug, Serialize, Clone)]
struct BenchmarkResult {
    model_name: String,
    prompt_id: usize,
    category: String,
    prompt: String,
    prompt_tokens: usize,
    prefill_time_ms: f64,
    prefill_tps: f64,
    gen_tokens: usize,
    gen_time_s: f64,
    tokens_per_sec: f64,
    word_count: usize,
    words_per_sec: f64,
    vm_rss_mb: f64,
    vm_hwm_mb: f64,
    idle_rss_mb: f64,
    response: String,
}

#[derive(Debug, Serialize)]
struct ModelSummary {
    model: String,
    disk_size_mb: f64,
    num_queries: usize,
    avg_prefill_ms: f64,
    avg_prefill_tps: f64,
    avg_tokens_per_sec: f64,
    avg_words_per_sec: f64,
    peak_ram_mb: f64,
    avg_resident_ram_mb: f64,
    post_reclaim_ram_mb: f64,
    zero_code_compliance: bool,
    results: Vec<BenchmarkResult>,
}

fn memory_mb() -> (f64, f64) {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let get_val = |key: &str| -> f64 {
        status
            .lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0)
            / 1024.0
    };
    (get_val("VmRSS:"), get_val("VmHWM:"))
}

fn get_file_size_mb(path: &Path) -> f64 {
    std::fs::metadata(path)
        .map(|m| m.len() as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0)
}

fn benchmark_model(
    model_name: &str,
    model_path: &Path,
    tokenizer_path: &Path,
    system_prompt: &str,
    test_cases: &[(&str, &str)], // (category, user_prompt)
    max_gen_tokens: usize,
    temp: f32,
    top_p: f32,
) -> Result<ModelSummary> {
    println!("\n========================================================");
    println!("  BENCHMARKING: {model_name}");
    println!("  Model Path:     {}", model_path.display());
    println!("  Tokenizer Path: {}", tokenizer_path.display());
    println!("  Disk Size:      {:.2} MB", get_file_size_mb(model_path));
    println!("========================================================");

    let (start_rss, start_hwm) = memory_mb();
    println!(
        "Process Start RAM: VmRSS = {:.2} MB | VmHWM = {:.2} MB",
        start_rss, start_hwm
    );

    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!("{e}"))?;
    let t0 = Instant::now();
    let mut engine = AstMicroEngine::new(model_path)?;
    if model_name.to_lowercase().contains("skate") {
        engine.set_suppressed_tokens([3725, 47182]);
    } else if model_name.to_lowercase().contains("manta") {
        engine.set_suppressed_tokens([41233, 54275, 73594]);
    }
    let load_time = t0.elapsed().as_secs_f64();
    let (load_rss, load_hwm) = memory_mb();
    println!(
        "Engine loaded in {:.2}s | Post-load RAM: VmRSS = {:.2} MB | VmHWM = {:.2} MB\n",
        load_time, load_rss, load_hwm
    );

    let eos_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(49153);
    let mut results = Vec::new();
    let mut total_prefill_ms = 0.0;
    let mut total_prefill_tps = 0.0;
    let mut total_tps = 0.0;
    let mut total_wps = 0.0;
    let mut peak_ram_observed: f64 = load_hwm;
    let mut has_code_violation = false;

    for (idx, &(category, prompt)) in test_cases.iter().enumerate() {
        println!("--------------------------------------------------------");
        println!(
            "[Case {}/{}] Category: {} | Prompt: \"{}\"",
            idx + 1,
            test_cases.len(),
            category,
            prompt
        );

        // Reset cache before turn
        engine.reset_cache();
        engine.set_sampling(temp, top_p, 1.15, 64);

        let formatted = format!(
            "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        );
        let encoded = tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| anyhow!("{e}"))?;
        let prompt_tokens = encoded.get_ids().to_vec();

        // Prefill
        let t_prefill = Instant::now();
        let _ = engine.prefill(&prompt_tokens)?;
        let prefill_duration = t_prefill.elapsed();
        let prefill_ms = prefill_duration.as_secs_f64() * 1000.0;
        let prefill_tps = prompt_tokens.len() as f64 / prefill_duration.as_secs_f64();

        // Generation
        let mut generated_tokens = Vec::new();
        let t_gen = Instant::now();
        for _ in 0..max_gen_tokens {
            let next_tok = engine.next_token_with_history(None, &generated_tokens, false)?;
            if next_tok == eos_id || next_tok == 0 {
                break;
            }
            generated_tokens.push(next_tok);
            let _ = engine.forward_stream(
                &[next_tok],
                prompt_tokens.len() + generated_tokens.len() - 1,
            )?;
        }
        let gen_duration = t_gen.elapsed();
        let gen_time_s = gen_duration.as_secs_f64();
        let tps = if gen_time_s > 0.0 {
            generated_tokens.len() as f64 / gen_time_s
        } else {
            0.0
        };

        let decoded = tokenizer
            .decode(&generated_tokens, false)
            .unwrap_or_default();
        let word_count = decoded.split_whitespace().count();
        let wps = if gen_time_s > 0.0 {
            word_count as f64 / gen_time_s
        } else {
            0.0
        };

        let (rss_after, hwm_after) = memory_mb();
        if hwm_after > peak_ram_observed {
            peak_ram_observed = hwm_after;
        }

        // Post-turn reset and idle measurement
        engine.reset_cache();
        let (idle_rss, _) = memory_mb();

        // Check for code emissions if category is deflection
        if category.contains("deflect") || category.contains("Zero-Code") {
            if decoded.contains("```")
                || decoded.contains("def ")
                || decoded.contains("function(")
                || decoded.contains("function (")
                || decoded.contains("fn ")
                || decoded.contains("=>")
            {
                has_code_violation = true;
            }
        }

        println!(
            "  Tokens: {} | Time: {:.2}s | Speed: {:.1} tok/s | {:.1} words/s",
            generated_tokens.len(),
            gen_time_s,
            tps,
            wps
        );
        println!(
            "  Prefill: {:.1} ms ({:.1} t/s) | Active RSS: {:.2} MB | Peak HWM: {:.2} MB | Idle RSS: {:.2} MB",
            prefill_ms, prefill_tps, rss_after, hwm_after, idle_rss
        );
        let preview: String = decoded.chars().take(80).collect();
        println!("  Response preview: \"{}\"...", preview);

        total_prefill_ms += prefill_ms;
        total_prefill_tps += prefill_tps;
        total_tps += tps;
        total_wps += wps;

        results.push(BenchmarkResult {
            model_name: model_name.to_string(),
            prompt_id: idx + 1,
            category: category.to_string(),
            prompt: prompt.to_string(),
            prompt_tokens: prompt_tokens.len(),
            prefill_time_ms: prefill_ms,
            prefill_tps,
            gen_tokens: generated_tokens.len(),
            gen_time_s,
            tokens_per_sec: tps,
            word_count,
            words_per_sec: wps,
            vm_rss_mb: rss_after,
            vm_hwm_mb: hwm_after,
            idle_rss_mb: idle_rss,
            response: decoded,
        });
    }

    let n = test_cases.len() as f64;
    engine.reset_cache();
    engine.evict_all();
    let (final_idle_rss, _) = memory_mb();

    let summary = ModelSummary {
        model: model_name.to_string(),
        disk_size_mb: get_file_size_mb(model_path),
        num_queries: test_cases.len(),
        avg_prefill_ms: total_prefill_ms / n,
        avg_prefill_tps: total_prefill_tps / n,
        avg_tokens_per_sec: total_tps / n,
        avg_words_per_sec: total_wps / n,
        peak_ram_mb: peak_ram_observed,
        avg_resident_ram_mb: results.iter().map(|r| r.vm_rss_mb).sum::<f64>() / n,
        post_reclaim_ram_mb: final_idle_rss,
        zero_code_compliance: !has_code_violation,
        results,
    };

    println!("\n=== SUMMARY FOR {model_name} ===");
    println!(
        "  Avg Generation Speed:  {:.1} words/sec ({:.1} tokens/sec)",
        summary.avg_words_per_sec, summary.avg_tokens_per_sec
    );
    println!(
        "  Avg Prefill Speed:     {:.1} tokens/sec ({:.1} ms)",
        summary.avg_prefill_tps, summary.avg_prefill_ms
    );
    println!("  Peak Process RAM:      {:.2} MB", summary.peak_ram_mb);
    println!(
        "  Avg Resident RAM:      {:.2} MB",
        summary.avg_resident_ram_mb
    );
    println!(
        "  Post-Reclaim Idle RAM: {:.2} MB",
        summary.post_reclaim_ram_mb
    );
    println!(
        "  Zero-Code Invariant:   {}",
        if summary.zero_code_compliance {
            "PASS (100%)"
        } else {
            "FAIL (Emitted code)"
        }
    );

    Ok(summary)
}

fn main() -> Result<()> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/mn".into());
    let models_dir = PathBuf::from(home).join(".nioxs").join("models");

    let skate_model = models_dir.join("skate.vmicrolm-vector.gguf");
    let skate_tokenizer = models_dir.join("skate-tokenizer.json");
    let manta_model = models_dir.join("manta.vmicrolm-vector.gguf");
    let manta_tokenizer = models_dir.join("manta-tokenizer.json");

    if !skate_model.is_file() {
        return Err(anyhow!("Skate model missing at {:?}", skate_model));
    }
    if !manta_model.is_file() {
        return Err(anyhow!("Manta model missing at {:?}", manta_model));
    }

    let skate_system_prompt = "You are Skate, Nio's ambient companion. Be friendly, clear, grounded, and concise. Explain everyday topics accurately and express uncertainty when you do not know. Do not write code; direct programming requests to Viper or Manta. For medical concerns recommend an appropriate qualified professional and encourage emergency help when urgent. Respond to signs of crisis with empathy and encourage immediate human support and local emergency or crisis resources where appropriate.";

    let manta_system_prompt = "You are Manta, Nio's Conversational Guru and DAG Architect. You specialize in systems architecture, boundary analysis, trade-off evaluation, and operational planning. Decompose technical tasks into structured implementation roadmaps using <manta_plan>, <manta_step>, and <manta_risk> tags. When raw code is requested, provide high-level architecture and defer code generation to Cuda or Fin.";

    let skate_test_cases = [
        (
            "everyday_science",
            "What's the difference between an espresso, an Americano, and a latte?",
        ),
        (
            "physics_concept",
            "Why is the sky blue during the day but red at sunset?",
        ),
        (
            "desk_companion",
            "What is a good way to stretch after typing for hours at a desk?",
        ),
        (
            "stress_empathy",
            "I feel overwhelmed and stressed with this project deadline today.",
        ),
        (
            "coding_deflect",
            "Please write a Python function that sorts and deduplicates a list of dictionaries.",
        ),
        (
            "coding_deflect",
            "Can you fix my JavaScript fetch API code?",
        ),
    ];

    let manta_test_cases = [
        (
            "concurrency_tradeoffs",
            "Explain the difference between Arc<Mutex<T>> and RwLock<T> in Tokio. When does RwLock degrade performance?",
        ),
        (
            "dag_planning",
            "Plan a zero-downtime database migration for a multi-tenant PostgreSQL schema using <manta_plan> tags.",
        ),
        (
            "distributed_systems",
            "Design a resilient edge consensus mechanism for 5 nodes with intermittent satellite connectivity.",
        ),
        (
            "caching_architecture",
            "How do we prevent cache stampedes in high-throughput microservices using Redis and singleflight?",
        ),
        (
            "coding_deflect_architect",
            "Write me a Rust web server using Axum and SQLite.",
        ),
    ];

    let skate_summary = benchmark_model(
        "Skate (135M Ambient Companion)",
        &skate_model,
        &skate_tokenizer,
        skate_system_prompt,
        &skate_test_cases,
        80,
        0.65,
        0.90,
    )?;

    let manta_summary = benchmark_model(
        "Manta (494M Systems Guru & DAG Architect)",
        &manta_model,
        &manta_tokenizer,
        manta_system_prompt,
        &manta_test_cases,
        128,
        0.50,
        0.85,
    )?;

    let report = serde_json::json!({
        "timestamp": format!("{:?}", std::time::SystemTime::now()),
        "skate": skate_summary,
        "manta": manta_summary,
    });

    let output_path = PathBuf::from("skate_and_manta_production_benchmark.json");
    std::fs::write(&output_path, serde_json::to_string_pretty(&report)?)?;
    println!(
        "\n✅ Full benchmark report written to: {}",
        output_path.display()
    );

    Ok(())
}
