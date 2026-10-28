use anyhow::Result;
use nioxs_engine::skate_system;
use serde_json::{Value, json};
use tokenizers::Tokenizer;
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let tokenizer=Tokenizer::from_file(&a[1]).map_err(|e| anyhow::anyhow!("{e}"))?;
    let suite: Value=serde_json::from_slice(&std::fs::read(&a[2])?)?;
    let question=suite["held_out"][0].as_str().ok_or_else(|| anyhow::anyhow!("missing prompt"))?;
    let policy=skate_system()?;
    let prompt=format!("<|im_start|>system\n{policy}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n");
    let ids=tokenizer.encode(prompt,false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
    println!("{}",json!(ids));
    Ok(())
}
