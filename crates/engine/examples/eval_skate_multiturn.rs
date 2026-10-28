use anyhow::Result;
use nioxs_engine::{RayEngine, skate_system};
use serde_json::{json, Value};
use std::{path::PathBuf, time::Instant};

#[tokio::main]
async fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let system_prompt = skate_system()?;
    let engine = RayEngine::from_paths(PathBuf::from(&a[1]), PathBuf::from(&a[2]));
    let cases: Value = serde_json::from_slice(&std::fs::read(&a[3])?)?;
    let mut records = Vec::new();
    for case in cases.as_array().ok_or_else(|| anyhow::anyhow!("expected array"))? {
        let mut conversation = format!("<|im_start|>system\n{system_prompt}<|im_end|>\n");
        let mut turns = Vec::new();
        for turn in case["turns"].as_array().ok_or_else(|| anyhow::anyhow!("missing turns"))? {
            let prompt = turn.as_str().ok_or_else(|| anyhow::anyhow!("invalid turn"))?;
            conversation.push_str(&format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"));
            let start = Instant::now();
            let answer = engine.generate(conversation.clone(), 128, 0.65, 0.9).await?;
            turns.push(json!({"prompt":prompt,"response":answer,"seconds":start.elapsed().as_secs_f64()}));
            conversation.push_str(&format!("{answer}<|im_end|>\n"));
        }
        records.push(json!({"case":case["case"],"turns":turns}));
    }
    println!("{}", serde_json::to_string_pretty(&records)?);
    Ok(())
}
