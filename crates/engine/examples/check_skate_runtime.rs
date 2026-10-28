use anyhow::{ensure, Result};
use nioxs_engine::{AstMicroEngine, RayEngine};
use std::{path::PathBuf, time::Duration};

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let cases: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[4])?)?;
    let prompt = cases["held_out"][0].as_str().ok_or_else(|| anyhow::anyhow!("missing held-out prompt"))?.to_owned();
    let model = PathBuf::from(&args[1]);
    let tokenizer = PathBuf::from(&args[2]);
    let mut ast = AstMicroEngine::new(&model)?;
    ensure!(ast.prefill(&vec![1; 4097]).is_err(), "long context accepted");
    ensure!(ast.prefill(&[u32::MAX]).is_err(), "invalid token accepted");
    drop(ast);
    let bad = RayEngine::from_paths(model.clone(), PathBuf::from(&args[3]));
    ensure!(bad.ensure_loaded().await.is_err(), "mismatched tokenizer accepted");
    let engine = RayEngine::from_paths(model, tokenizer);
    let rx = engine.generate_stream(prompt.clone(), 128, 0.65, 0.9).await?;
    drop(rx);
    let follow = tokio::time::timeout(Duration::from_secs(20), engine.generate(prompt.clone(), 12, 0.65, 0.9)).await??;
    ensure!(!follow.is_empty(), "empty follow-up after cancellation");
    let (a, b) = tokio::time::timeout(Duration::from_secs(30), async {
        tokio::join!(engine.generate(prompt.clone(), 12, 0.65, 0.9), engine.generate(prompt.clone(), 12, 0.65, 0.9))
    }).await?;
    ensure!(!a?.is_empty() && !b?.is_empty(), "empty concurrent response");
    println!("context limit, invalid token, tokenizer mismatch, cancellation and concurrent requests passed");
    Ok(())
}
