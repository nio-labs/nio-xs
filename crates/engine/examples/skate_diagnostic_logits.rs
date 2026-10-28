use anyhow::Result;
use nioxs_engine::{AstMicroEngine, RayEngine, skate_system};
use serde_json::{Value,json};
use std::path::Path;
use tokenizers::Tokenizer;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let tok = Tokenizer::from_file(&args[2]).map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut engine = AstMicroEngine::new(Path::new(&args[1]))?;
    RayEngine::validate_tokenizer(&tok, &engine.tokenizer_tokens()?)?;
    let suite: Value = serde_json::from_slice(&std::fs::read(&args[3])?)?;
    let index: usize = args.get(4).map(|v| v.parse()).transpose()?.unwrap_or(0);
    let q = suite["held_out"][index].as_str().ok_or_else(|| anyhow::anyhow!("no held-out prompt"))?;
    let policy = skate_system()?;
    let prompt = format!("<|im_start|>system\n{policy}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n");
    let ids = tok.encode(prompt, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    engine.prefill(&ids)?;
    println!("{}", json!({"tokens":ids,"logits":engine.diagnostic_logits()?}));
    Ok(())
}
