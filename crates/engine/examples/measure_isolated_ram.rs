use anyhow::{Result, anyhow};
use nioxs_engine::ast_engine::AstMicroEngine;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

fn print_detailed_ram(label: &str) {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        let mut vmrss = 0.0;
        let mut vmhwm = 0.0;
        let mut rssanon = 0.0;
        let mut rssfile = 0.0;
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                vmrss = line
                    .split_whitespace()
                    .nth(1)
                    .unwrap()
                    .parse::<f64>()
                    .unwrap()
                    / 1024.0;
            } else if line.starts_with("VmHWM:") {
                vmhwm = line
                    .split_whitespace()
                    .nth(1)
                    .unwrap()
                    .parse::<f64>()
                    .unwrap()
                    / 1024.0;
            } else if line.starts_with("RssAnon:") {
                rssanon = line
                    .split_whitespace()
                    .nth(1)
                    .unwrap()
                    .parse::<f64>()
                    .unwrap()
                    / 1024.0;
            } else if line.starts_with("RssFile:") {
                rssfile = line
                    .split_whitespace()
                    .nth(1)
                    .unwrap()
                    .parse::<f64>()
                    .unwrap()
                    / 1024.0;
            }
        }
        println!(
            "[{label}] Peak VmHWM: {:6.1} MB | Current VmRSS: {:6.1} MB (Anon Heap: {:5.1} MB, File Mmap: {:5.1} MB)",
            vmhwm, vmrss, rssanon, rssfile
        );
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: measure_isolated_ram <model_path>");
        return Ok(());
    }

    let model_path = PathBuf::from(&args[1]);
    let tokenizer_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga-tokenizer.json");

    print_detailed_ram("Start of Process");

    let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("{e}"))?;
    tokenizer.with_padding(None);
    let mut engine = AstMicroEngine::new(&model_path)?;
    print_detailed_ram("Post Engine Load");

    let prompt = "Explain how Raft avoids split-brain during leader election when a network partition occurs.";
    let formatted = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
    let prompt_tokens = tokenizer
        .encode(formatted.as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();

    let _ = engine.prefill(&prompt_tokens)?;
    print_detailed_ram("Post Prefill");

    let mut current_tok = engine.next_token_with_history(None, &[], false)?;
    let mut gen_tokens = vec![current_tok];
    let mut stream_pos = prompt_tokens.len();

    let t0 = Instant::now();
    for _ in 0..80 {
        let _ = engine.forward_stream(&[current_tok], stream_pos)?;
        stream_pos += 1;
        let next_tok = engine.next_token_with_history(None, &gen_tokens, false)?;
        gen_tokens.push(next_tok);
        current_tok = next_tok;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    print_detailed_ram("During/Post Generation");

    println!(
        "Generated {} tokens in {:.2}s ({:.2} tok/s)",
        gen_tokens.len(),
        elapsed,
        gen_tokens.len() as f64 / elapsed
    );

    Ok(())
}
