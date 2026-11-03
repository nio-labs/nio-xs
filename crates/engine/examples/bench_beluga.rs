use anyhow::{anyhow, Result};
use nioxs_engine::ast_engine::AstMicroEngine;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

#[derive(Debug, Serialize, Clone)]
struct BenchmarkResult {
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
    has_think_tags: bool,
    response: String,
}

#[derive(Debug, Serialize)]
struct BelugaSummary {
    model: String,
    disk_size_mb: f64,
    num_queries: usize,
    avg_prefill_ms: f64,
    avg_prefill_tps: f64,
    avg_tokens_per_sec: f64,
    peak_ram_mb: f64,
    sub_500mb_sla_passed: bool,
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

fn main() -> Result<()> {
    let model_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga.vmicrolm-vector.gguf");
    let tokenizer_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga-tokenizer.json");

    if !model_path.exists() {
        return Err(anyhow!("Beluga model missing at {:?}", model_path));
    }
    if !tokenizer_path.exists() {
        return Err(anyhow!("Beluga tokenizer missing at {:?}", tokenizer_path));
    }

    println!("\n========================================================");
    println!("  🐋 BENCHMARKING: NIOBELUGA (1.54B) v-MicroLM Companion");
    println!("  Model Path:     {}", model_path.display());
    println!("  Tokenizer Path: {}", tokenizer_path.display());
    println!("  Disk Size:      {:.2} MB", get_file_size_mb(&model_path));
    println!("========================================================");

    let (start_rss, start_hwm) = memory_mb();
    println!("Process Start RAM: VmRSS = {:.2} MB | VmHWM = {:.2} MB", start_rss, start_hwm);

    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("{e}"))?;
    let t0 = Instant::now();
    let mut engine = AstMicroEngine::new(&model_path)?;
    let load_time = t0.elapsed().as_secs_f64();
    let (load_rss, load_hwm) = memory_mb();
    println!("Engine loaded in {:.2}s | Post-load RAM: VmRSS = {:.2} MB | VmHWM = {:.2} MB\n", load_time, load_rss, load_hwm);

    let eos_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
    let system_prompt = "You are Beluga, Nio's Flagship Sovereign Cognitive Companion. You think before speaking using <think>...</think> tags. Reason deeply, verify edge cases, and deliver clear, authoritative architectural solutions. Defer raw low-level code generation to Cuda.";

    let test_cases = [
        ("systems_architecture", "Design an in-memory lock-free priority queue for high-frequency trading in Rust."),
        ("memory_safety", "Explain the memory safety implications and cache hazards of uninitialized memory in tensor runtimes."),
        ("consensus_protocols", "Compare Raft vs Multi-Paxos for distributed key-value stores under high network jitter."),
    ];

    let mut results = Vec::new();
    let mut total_prefill_ms = 0.0;
    let mut total_prefill_tps = 0.0;
    let mut total_tps = 0.0;
    let mut peak_ram_observed: f64 = load_hwm;

    for (idx, &(category, prompt)) in test_cases.iter().enumerate() {
        println!("--------------------------------------------------------");
        println!("[Case {}/{}] Category: {} | Prompt: \"{}\"", idx + 1, test_cases.len(), category, prompt);

        engine.reset_cache();
        engine.set_sampling(0.65, 0.90, 1.15, 64);

        let formatted = format!(
            "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        );
        let encoded = tokenizer.encode(formatted.as_str(), false).map_err(|e| anyhow!("{e}"))?;
        let prompt_tokens = encoded.get_ids().to_vec();

        // Prefill
        let t_prefill = Instant::now();
        let _ = engine.prefill(&prompt_tokens)?;
        let prefill_duration = t_prefill.elapsed();
        let prefill_ms = prefill_duration.as_secs_f64() * 1000.0;
        let prefill_tps = prompt_tokens.len() as f64 / prefill_duration.as_secs_f64();

        // Decode loop
        let t_gen = Instant::now();
        let mut gen_tokens = Vec::new();
        let max_gen_tokens = 64;

        for _ in 0..max_gen_tokens {
            let next_tok = engine.next_token_with_history(None, &gen_tokens, false)?;
            if next_tok == eos_id || next_tok == 0 {
                break;
            }
            gen_tokens.push(next_tok);
            let _ = engine.forward_stream(&[next_tok], prompt_tokens.len() + gen_tokens.len() - 1)?;
        }
        let gen_duration = t_gen.elapsed();
        let gen_time_s = gen_duration.as_secs_f64();
        let tps = if gen_time_s > 0.0 { gen_tokens.len() as f64 / gen_time_s } else { 0.0 };

        let decoded_response = tokenizer.decode(&gen_tokens, false).unwrap_or_default();
        let words = decoded_response.split_whitespace().count();
        let wps = if gen_time_s > 0.0 { words as f64 / gen_time_s } else { 0.0 };

        let (cur_rss, cur_hwm) = memory_mb();
        if cur_hwm > peak_ram_observed {
            peak_ram_observed = cur_hwm;
        }

        let has_think = decoded_response.contains("<think>") || decoded_response.contains("think");

        println!("  Prefill: {:.1} ms ({:.1} t/s) | Gen Tokens: {} | Gen Time: {:.2}s | Speed: {:.2} t/s | {:.2} w/s", prefill_ms, prefill_tps, gen_tokens.len(), gen_time_s, tps, wps);
        println!("  RAM: VmRSS = {:.2} MB | VmHWM = {:.2} MB (Peak: {:.2} MB)", cur_rss, cur_hwm, peak_ram_observed);
        println!("  Sample Output:\n{}\n", decoded_response.trim());

        total_prefill_ms += prefill_ms;
        total_prefill_tps += prefill_tps;
        total_tps += tps;

        results.push(BenchmarkResult {
            prompt_id: idx + 1,
            category: category.to_string(),
            prompt: prompt.to_string(),
            prompt_tokens: prompt_tokens.len(),
            prefill_time_ms: prefill_ms,
            prefill_tps,
            gen_tokens: gen_tokens.len(),
            gen_time_s,
            tokens_per_sec: tps,
            word_count: words,
            words_per_sec: wps,
            vm_rss_mb: cur_rss,
            vm_hwm_mb: cur_hwm,
            has_think_tags: has_think,
            response: decoded_response,
        });
    }

    let n = test_cases.len() as f64;
    let summary = BelugaSummary {
        model: "NioBeluga (1.54B Sovereign Cognitive Companion)".to_string(),
        disk_size_mb: get_file_size_mb(&model_path),
        num_queries: test_cases.len(),
        avg_prefill_ms: total_prefill_ms / n,
        avg_prefill_tps: total_prefill_tps / n,
        avg_tokens_per_sec: total_tps / n,
        peak_ram_mb: peak_ram_observed,
        sub_500mb_sla_passed: peak_ram_observed < 500.0,
        results,
    };

    println!("========================================================");
    println!("  🐋 BENCHMARK SUMMARY & SLA VERIFICATION");
    println!("  Peak Resident RAM (VmHWM): {:.2} MB", summary.peak_ram_mb);
    println!("  Sub-500MB SLA:             {}", if summary.sub_500mb_sla_passed { "✅ PASSED (< 500 MB)" } else { "❌ FAILED" });
    println!("  Avg Prefill TPS:           {:.2} t/s", summary.avg_prefill_tps);
    println!("  Avg Generation TPS:        {:.2} t/s", summary.avg_tokens_per_sec);
    println!("========================================================");

    let output_path = PathBuf::from("/home/mn/nio-labs/beluga_production_benchmark.json");
    std::fs::write(&output_path, serde_json::to_string_pretty(&summary)?)?;
    println!("✅ Report saved to: {}", output_path.display());

    Ok(())
}
