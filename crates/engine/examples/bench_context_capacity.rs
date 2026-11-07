//! Native CPU context-capacity, retrieval, and memory probe. MODEL TOKENIZER [PROMPT_TOKENS]
use anyhow::{Result, anyhow};
use nioxs_engine::{AstEngine, AstMicroEngine, manta_system};
use serde_json::json;
use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Instant,
};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(args.len() >= 3, "expected MODEL TOKENIZER [PROMPT_TOKENS]");
    let target: usize = args.get(3).map_or(Ok(8160), |s| s.parse())?;
    let running = Arc::new(AtomicBool::new(true));
    let peak_anon = Arc::new(AtomicU64::new(0));
    let peak_rss = Arc::new(AtomicU64::new(0));
    let (alive, anon, rss) = (running.clone(), peak_anon.clone(), peak_rss.clone());
    let sampler = std::thread::spawn(move || {
        while alive.load(Ordering::Relaxed) {
            if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    let mut parts = line.split_whitespace();
                    let key = parts.next().unwrap_or("");
                    let value = parts
                        .next()
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(0);
                    if key == "RssAnon:" {
                        anon.fetch_max(value, Ordering::Relaxed);
                    }
                    if key == "VmRSS:" {
                        rss.fetch_max(value, Ordering::Relaxed);
                    }
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    });
    let start = Instant::now();
    let tokenizer = Tokenizer::from_file(&args[2]).map_err(|e| anyhow!("{e}"))?;
    let mut engine = AstMicroEngine::new(Path::new(&args[1]))?;
    AstEngine::validate_tokenizer(&tokenizer, &engine.tokenizer_tokens()?)?;
    engine.set_sampling(0.0, 1.0, 1.0, 64);
    let context = engine.context_length();
    anyhow::ensure!(target < context, "leave output space within context");
    let encode = |text: &str| -> Result<Vec<u32>> {
        Ok(tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("{e}"))?
            .get_ids()
            .to_vec())
    };
    let mut input = encode(&format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\nThe START marker is BRONZE.\n",
        manta_system(false)
    ))?;
    let filler = encode(" Routine archive entry: the service is healthy.\n")?;
    let middle = encode("\nThe MIDDLE marker is SILVER.\n")?;
    let end = encode(
        "\nThe END marker is GOLD.\nReturn the START, MIDDLE, and END markers in that order, using only their three names.<|im_end|>\n<|im_start|>assistant\n",
    )?;
    anyhow::ensure!(
        input.len() + middle.len() + end.len() < target,
        "target too small"
    );
    let mut padding = 0;
    while input.len() < target / 2 {
        input.push(filler[padding % filler.len()]);
        padding += 1;
    }
    input.extend(middle);
    while input.len() < target - end.len() {
        input.push(filler[padding % filler.len()]);
        padding += 1;
    }
    input.extend(end);
    anyhow::ensure!(input.len() == target, "incorrect fixture length");
    if let Some(path) = args.get(4) {
        std::fs::write(
            path,
            input
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" "),
        )?;
        eprintln!("Exported exact fixture IDs to {path}");
        running.store(false, Ordering::Relaxed);
        sampler
            .join()
            .map_err(|_| anyhow!("memory sampler failed"))?;
        return Ok(());
    }
    eprintln!("Loaded native CPU model; prefill starts: {target}/{context} tokens");
    let loaded_seconds = start.elapsed().as_secs_f64();
    let prefill_start = Instant::now();
    engine.prefill(&input)?;
    let prefill_seconds = prefill_start.elapsed().as_secs_f64();
    eprintln!("Prefill finished after {prefill_seconds:.2}s; decoding retrieval answer");
    let stops: Vec<_> = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
        .iter()
        .filter_map(|s| tokenizer.token_to_id(s))
        .collect();
    let budget = (context - input.len()).min(32);
    let mut generated = Vec::new();
    let mut position = input.len();
    let decode_start = Instant::now();
    while generated.len() < budget {
        let token = engine.next_token_with_history(None, &generated, false)?;
        if stops.contains(&token) {
            break;
        }
        generated.push(token);
        if generated.len() < budget {
            engine.forward_stream(&[token], position)?;
            position += 1;
        }
    }
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    let response = tokenizer
        .decode(&generated, false)
        .map_err(|e| anyhow!("{e}"))?;
    let pad = *input.last().unwrap();
    let boundary_target = if target >= context.saturating_sub(64) {
        context
    } else {
        position
    };
    while position < boundary_target {
        engine.forward_stream(&[pad], position)?;
        position += 1;
    }
    let overrun = engine
        .forward_stream(&[pad], context)
        .expect_err("context overrun must be rejected")
        .to_string();
    let oversized = engine
        .prefill(&vec![pad; context + 1])
        .expect_err("oversized prefill must be rejected")
        .to_string();
    engine.reset_cache();
    let recovery = encode(&format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\nWhat is a work queue?<|im_end|>\n<|im_start|>assistant\n",
        manta_system(false)
    ))?;
    engine.prefill(&recovery)?;
    let recovery_token = engine.next_token_with_history(None, &[], false)?;
    engine.park();
    running.store(false, Ordering::Relaxed);
    sampler
        .join()
        .map_err(|_| anyhow!("memory sampler failed"))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "device":"cpu", "runtime":"native autoregressive greedy; no output filtering or repetition penalty",
            "prompt_tokens":input.len(), "context_limit":context, "output_budget":budget,
        "cache_tokens_exercised":position,
            "load_seconds":loaded_seconds, "prefill_seconds":prefill_seconds,
            "decode_seconds":decode_seconds, "generated_tokens":generated, "response":response,
            "retrieval_contains_all_markers":(["BRONZE","SILVER","GOLD"].iter().all(|s| response.contains(s))),
            "context_overrun_error":overrun, "oversized_prefill_error":oversized,
            "recovery_token":recovery_token, "recovery_succeeded":true,
            "sampled_peak_anon_mib":peak_anon.load(Ordering::Relaxed) as f64/1024.0,
            "sampled_peak_rss_mib":peak_rss.load(Ordering::Relaxed) as f64/1024.0,
            "memory_method":"RssAnon and VmRSS sampled every 10 ms across loading, full prefill, decoding and recovery; brief peaks may be missed",
            "input_construction":"Tokenizer IDs assembled to exact length, with beginning/middle/end markers and repetitive filler. Capacity and synthetic retrieval probe; not a general long-context quality benchmark."
        }))?
    );
    Ok(())
}
