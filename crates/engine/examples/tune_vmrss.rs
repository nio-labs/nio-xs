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

fn test_config(res: usize, win: usize) -> Result<()> {
    let model_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga.vmicrolm-vector.gguf");
    let tokenizer_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga-tokenizer.json");

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("{e}"))?;
    tokenizer.with_padding(None);
    let mut engine = AstMicroEngine::new(&model_path)?;

    engine.set_resident_layers(res);
    engine.set_rolling_window(win);

    let prompt = "Explain how Raft avoids split-brain.";
    let encoded = tokenizer
        .encode(prompt, false)
        .map_err(|e| anyhow!("{e}"))?;
    let prompt_tokens = encoded.get_ids().to_vec();

    engine.reset_cache();
    let _ = engine.prefill(&prompt_tokens)?;
    let mut tok = engine.next_token_with_history(None, &[], false)?;

    // Warmup 5 tokens
    for i in 0..5 {
        engine.forward_stream(&[tok], prompt_tokens.len() + i)?;
        tok = engine.next_token_with_history(None, &[], false)?;
    }

    let t0 = Instant::now();
    for i in 0..20 {
        engine.forward_stream(&[tok], prompt_tokens.len() + 5 + i)?;
        tok = engine.next_token_with_history(None, &[], false)?;
    }
    let dt = t0.elapsed().as_secs_f64();
    let (rss, hwm) = memory_mb();

    println!(
        "res={:2} win={:2} | 20 tokens: {:.3}s ({:5.2} tok/s, {:5.1} ms/tok) | VmRSS: {:5.1} MB (HWM: {:5.1} MB)",
        res,
        win,
        dt,
        20.0 / dt,
        dt / 20.0 * 1000.0,
        rss,
        hwm
    );

    Ok(())
}

fn main() -> Result<()> {
    test_config(7, 4)?;
    test_config(8, 4)?;
    test_config(9, 4)?;
    test_config(8, 6)?;
    test_config(9, 6)?;
    Ok(())
}
