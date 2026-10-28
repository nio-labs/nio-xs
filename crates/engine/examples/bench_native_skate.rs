use anyhow::Result;
use nioxs_engine::ast_engine::AstMicroEngine;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

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

fn main() -> Result<()> {
    let model_path = PathBuf::from(std::env::args().nth(1).expect("model path"));
    let tok_path = PathBuf::from(std::env::args().nth(2).expect("tokenizer path"));

    println!("Loading tokenizer from {:?}...", tok_path);
    let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("Loading AstMicroEngine from {:?}...", model_path);
    let t0 = Instant::now();
    let mut engine = AstMicroEngine::new(&model_path)?;
    println!("Loaded engine in {:.2}s!", t0.elapsed().as_secs_f64());

    let (rss_load, hwm_load) = memory_mb();
    println!("Post-Load RAM: VmRSS: {:.2} MB | VmHWM: {:.2} MB", rss_load, hwm_load);

    let prompt = std::env::args().nth(3).expect("formatted prompt argument");
    let encoded = tokenizer.encode(prompt.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?;
    let prompt_tokens = encoded.get_ids().to_vec();

    println!("\nPrefilling {} prompt tokens...", prompt_tokens.len());
    let t_prefill = Instant::now();
    let _ = engine.prefill(&prompt_tokens)?;
    let prefill_time = t_prefill.elapsed().as_secs_f64();
    println!("Prefill done in {:.3}s ({:.1} t/s)", prefill_time, prompt_tokens.len() as f64 / prefill_time);

    let eos_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(49153);
    let mut generated_tokens = Vec::new();
    let max_gen = 64;

    let t_gen = Instant::now();
    for _ in 0..max_gen {
        let next_tok = engine.next_token_with_history(None, &generated_tokens, false)?;
        if next_tok == eos_id || next_tok == 0 {
            break;
        }
        generated_tokens.push(next_tok);
        let _ = engine.forward_stream(&[next_tok], prompt_tokens.len() + generated_tokens.len() - 1)?;
    }
    let gen_time = t_gen.elapsed().as_secs_f64();
    let tps = generated_tokens.len() as f64 / gen_time;

    let (rss_end, hwm_end) = memory_mb();
    let decoded = tokenizer.decode(&generated_tokens, false).unwrap_or_default();
    let words = decoded.split_whitespace().count();
    let wps = words as f64 / gen_time;

    println!("\nGenerated Text:\n{}", decoded);
    println!("\n=== NATIVE AST-MICRO ENGINE BENCHMARK ===");
    println!("Generated Tokens:     {}", generated_tokens.len());
    println!("Generation Time:      {:.3}s", gen_time);
    println!("Tokens per Second:    {:.1} t/s", tps);
    println!("Words per Second:     {:.1} w/s", wps);
    println!("Final Resident RAM:   {:.2} MB", rss_end);
    println!("Peak Process RAM:     {:.2} MB", hwm_end);

    Ok(())
}
