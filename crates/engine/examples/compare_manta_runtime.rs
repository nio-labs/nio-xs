//! Greedy, unfiltered Manta inference for comparison with a reference runtime.
//! Usage: compare_manta_runtime MODEL TOKENIZER SUITE [FORCED_TRACE | --layers]
use anyhow::{Context, Result};
use nioxs_engine::{AstEngine, AstMicroEngine, manta_system};
use serde_json::{Value, json};
use std::path::Path;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 4,
        "expected MODEL TOKENIZER SUITE [FORCED_TRACE]"
    );
    let mut tokenizer = Tokenizer::from_file(&args[2]).map_err(|e| anyhow::anyhow!("{e}"))?;
    tokenizer.with_padding(None);
    let mut engine = AstMicroEngine::new(Path::new(&args[1]))?;
    AstEngine::validate_tokenizer(&tokenizer, &engine.tokenizer_tokens()?)?;
    engine.set_sampling(0.0, 1.0, 1.0, 64);
    let suite: Vec<Value> = serde_json::from_slice(&std::fs::read(&args[3])?)?;
    let layer_trace = args.get(4).is_some_and(|s| s == "--layers");
    let forced: Option<Vec<Value>> = args
        .get(4)
        .filter(|_| !layer_trace)
        .map(|path| -> Result<_> { Ok(serde_json::from_slice(&std::fs::read(path)?)?) })
        .transpose()?;
    let stop_ids: Vec<_> = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
        .iter()
        .filter_map(|s| tokenizer.token_to_id(s))
        .collect();
    let mut records = Vec::new();
    for (index, case) in suite.iter().enumerate() {
        engine.reset_cache();
        let question = case["prompt"].as_str().context("missing prompt")?;
        let system = manta_system(case["plan"].as_bool().unwrap_or(false));
        let prompt = format!(
            "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        );
        let input = tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec();
        if layer_trace {
            let layers = engine.diagnostic_prefill_layers(&input)?;
            records.push(json!({"name":case["name"], "prompt_tokens":input, "layers":layers}));
            continue;
        }
        engine.prefill(&input)?;
        let force = forced
            .as_ref()
            .map(|rows| -> Result<Vec<u32>> {
                rows.get(index).context("missing forced case")?["generated_tokens"]
                    .as_array()
                    .context("missing forced tokens")?
                    .iter()
                    .take(32)
                    .map(|t| {
                        t.as_u64()
                            .and_then(|n| u32::try_from(n).ok())
                            .context("invalid token")
                    })
                    .collect()
            })
            .transpose()?;
        let mut generated = Vec::new();
        let mut trace = Vec::new();
        let steps = force.as_ref().map_or(128, Vec::len);
        for step in 0..steps {
            let logits = engine.diagnostic_logits()?;
            let mut ranked: Vec<_> = logits.iter().enumerate().collect();
            ranked.sort_unstable_by(|a, b| b.1.total_cmp(a.1));
            let top: Vec<_> = ranked
                .iter()
                .take(10)
                .map(|(id, score)| json!({"id":id, "logit":score}))
                .collect();
            let greedy = ranked[0].0 as u32;
            // Verify the production vocabulary kernel selects the same token.
            let sampled = engine.next_token_with_history(None, &[], false)?;
            anyhow::ensure!(
                sampled == greedy,
                "vocabulary kernel disagrees with diagnostic logits"
            );
            trace.push(json!({"step":step, "greedy":greedy, "top":top}));
            let next = force.as_ref().map_or(greedy, |tokens| tokens[step]);
            if force.is_none() && stop_ids.contains(&next) {
                break;
            }
            generated.push(next);
            if step + 1 < steps {
                engine.forward_stream(&[next], input.len() + step)?;
            }
        }
        let response = tokenizer
            .decode(&generated, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        records.push(json!({"name":case["name"], "formatted_prompt":prompt,
            "prompt_tokens":input, "stop_ids":stop_ids, "generated_tokens":generated,
            "response":response, "trace":trace, "forced":force.is_some()}));
    }
    println!("{}", serde_json::to_string_pretty(&records)?);
    Ok(())
}
