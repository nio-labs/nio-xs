use anyhow::Result;
use nioxs_engine::{AstMicroEngine, RayEngine, skate_system};
use serde_json::{json, Value};
use std::path::Path;
use tokenizers::Tokenizer;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let system_prompt = skate_system()?;
    let tokenizer = Tokenizer::from_file(&a[2]).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut engine = AstMicroEngine::new(Path::new(&a[1]))?;
    RayEngine::validate_tokenizer(&tokenizer, &engine.tokenizer_tokens()?)?;
    engine.set_sampling(0.0, 1.0, 1.0, 64);
    let records: Vec<Value> = serde_json::from_slice(&std::fs::read(&a[3])?)?;
    let mut out = Vec::new();
    for record in records {
        let question = record["prompt"].as_str().ok_or_else(|| anyhow::anyhow!("missing prompt"))?;
        let prompt = format!("<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n");
        let encoded = tokenizer.encode(prompt, false).map_err(|e| anyhow::anyhow!("{e}"))?;
        let input = encoded.get_ids();
        engine.prefill(input)?;
        let mut generated = Vec::new();
        let stop: Vec<_> = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"].iter().filter_map(|x| tokenizer.token_to_id(x)).collect();
        for _ in 0..96 {
            let tok = engine.next_token_with_history(None, &generated, false)?;
            if stop.contains(&tok) { break; }
            generated.push(tok);
            engine.forward_stream(&[tok], input.len()+generated.len()-1)?;
        }
        let native = tokenizer.decode(&generated, false).map_err(|e| anyhow::anyhow!("{e}"))?;
        out.push(json!({"prompt":question,"f16_reference":record["f16"],"quantized_reference":record["quantized"],"native":native}));
        engine.reset_cache();
    }
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
