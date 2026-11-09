//! Measure model loading separately from repeated production generation requests.
use anyhow::{Result, bail};
use nioxs_engine::{AstEngine, DevicePreference};
use serde_json::json;
use std::time::Instant;

fn words(text: &str) -> usize {
    text.replace("<think>", "")
        .replace("</think>", "")
        .split_whitespace()
        .filter(|word| word.chars().any(char::is_alphanumeric))
        .count()
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut model = "beluga".to_owned();
    let mut max_tokens = 256usize;
    let mut repeat = 2usize;
    let mut prompt = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--model" {
            model = args
                .next()
                .ok_or_else(|| anyhow::anyhow!("--model requires a model name"))?;
        } else if arg == "--max-tokens" {
            max_tokens = args
                .next()
                .ok_or_else(|| anyhow::anyhow!("--max-tokens requires a count"))?
                .parse()?;
        } else if arg == "--repeat" {
            repeat = args
                .next()
                .ok_or_else(|| anyhow::anyhow!("--repeat requires a count"))?
                .parse()?;
        } else if prompt.is_none() {
            prompt = Some(arg);
        } else {
            bail!(
                "Usage: bench_beluga_latency [--model skate|manta|beluga|blue] [--max-tokens N] [--repeat N] [quoted prompt]"
            );
        }
    }
    if !matches!(model.as_str(), "skate" | "manta" | "beluga" | "blue") {
        bail!("Unsupported companion: {model}");
    }
    if max_tokens == 0 || repeat == 0 {
        bail!("--max-tokens and --repeat must be positive");
    }
    let prompt = prompt
        .unwrap_or_else(|| "Explain how a refrigerator moves heat in about 120 words.".to_owned());
    let start = Instant::now();
    let engine = AstEngine::for_model_with_device(&model, DevicePreference::Cpu)?;
    engine.ensure_loaded().await?;
    let load_seconds = start.elapsed().as_secs_f64();
    for request in 1..=repeat {
        let started = Instant::now();
        let mut stream = engine
            .generate_stream(prompt.clone(), max_tokens, 0.0, 1.0)
            .await?;
        let mut response = String::new();
        let mut first = None;
        let mut first_chunk_words = 0;
        let mut last_chunk_seconds = 0.0;
        while let Some(chunk) = stream.recv().await {
            if chunk.is_empty() {
                continue;
            }
            last_chunk_seconds = started.elapsed().as_secs_f64();
            if first.is_none() {
                first = Some(last_chunk_seconds);
                first_chunk_words = words(&chunk);
            }
            response.push_str(&chunk);
        }
        let seconds = started.elapsed().as_secs_f64();
        let words = words(&response);
        let streaming_seconds = first.map(|first| last_chunk_seconds - first);
        let words_after_first_chunk = words.saturating_sub(first_chunk_words);
        let streaming_words_per_second = streaming_seconds
            .filter(|s| *s > 0.0)
            .map(|s| words_after_first_chunk as f64 / s);
        println!(
            "{}",
            json!({
                "model": model, "device": "cpu", "prompt": prompt, "max_tokens": max_tokens,
                "request": request, "load_seconds": load_seconds,
                "request_seconds_including_prefill": seconds,
                "first_chunk_seconds_including_prefill": first,
                "visible_words": words,
                "first_chunk_words": first_chunk_words,
                "words_after_first_chunk": words_after_first_chunk,
                "streaming_seconds_first_to_last_chunk": streaming_seconds,
                "streaming_words_per_second": streaming_words_per_second,
                "words_per_second_including_prefill": words as f64 / seconds,
                "cold_words_per_second": if request == 1 { Some(words as f64 / (load_seconds + seconds)) } else { None },
                "response": response,
                "method": "Same loaded AstEngine; production decoding. Streaming rate divides words after the first chunk by time from first to last nonempty chunk; loading, prefill, first-chunk words and shutdown are excluded. Chunk buffering makes this approximate, not exact model-token decode speed. Displayed thought text counts as words."
            })
        );
    }
    Ok(())
}
