use anyhow::{Result, anyhow};
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
    let model_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga.vmicrolm-vector.gguf");
    let tokenizer_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga-tokenizer.json");

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("{e}"))?;
    tokenizer.with_padding(None);
    let mut engine = AstMicroEngine::new(&model_path)?;

    engine.set_resident_layers(7);
    engine.set_rolling_window(4);

    let eos_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
    let prompt = "Explain how Raft avoids split-brain during leader election when a network partition occurs.";
    let formatted = format!(
        "<|im_start|>system\nYou are Beluga, Nio's Flagship Sovereign Cognitive Companion. You think before speaking using <think>...</think> tags. Reason deeply, verify edge cases, and deliver clear, authoritative architectural solutions.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    );

    let encoded = tokenizer
        .encode(formatted.as_str(), false)
        .map_err(|e| anyhow!("{e}"))?;
    let prompt_tokens = encoded.get_ids().to_vec();

    engine.reset_cache();
    let _ = engine.prefill(&prompt_tokens)?;
    let first_tok = engine.next_token_with_history(None, &[], false)?;

    println!("Prefill done. Testing 20 pure AR tokens...");
    let mut curr = first_tok;
    let mut stream_pos = prompt_tokens.len();
    let mut tokens = vec![curr];

    let t_start = Instant::now();
    for _ in 0..20 {
        let t0 = Instant::now();
        engine.forward_stream(&[curr], stream_pos)?;
        stream_pos += 1;
        let tok = engine.next_token_with_history(None, &tokens, false)?;
        let dt = t0.elapsed();
        tokens.push(tok);
        curr = tok;
        println!(
            "  AR token: {} ({:?}) | dt: {:.2}ms",
            tok,
            tokenizer.decode(&[tok], false),
            dt.as_secs_f64() * 1000.0
        );
        if tok == eos_id {
            break;
        }
    }
    let total_dt = t_start.elapsed();
    let (rss, hwm) = memory_mb();
    let text = tokenizer.decode(&tokens, true).unwrap_or_default();
    let words = text.split_whitespace().count();

    println!("\n=== PURE AR RESULTS ===");
    println!("Tokens: {}", tokens.len());
    println!("Words: {}", words);
    println!("Total time: {:.3}s", total_dt.as_secs_f64());
    println!(
        "Token rate: {:.2} tok/s",
        (tokens.len() - 1) as f64 / total_dt.as_secs_f64()
    );
    println!(
        "Word rate: {:.2} words/sec",
        words as f64 / total_dt.as_secs_f64()
    );
    println!("VmRSS: {:.2} MB | VmHWM: {:.2} MB", rss, hwm);

    Ok(())
}
