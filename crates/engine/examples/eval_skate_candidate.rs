use anyhow::Result;
use nioxs_engine::{RayEngine, skate_system};
use serde_json::{json, Value};
use std::{path::PathBuf, time::Instant};

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let system_prompt = skate_system()?;
    let engine = RayEngine::from_paths(PathBuf::from(&args[1]), PathBuf::from(&args[2]));
    let suite: Value = serde_json::from_slice(&std::fs::read(&args[3])?)?;
    let mut records = Vec::new();
    for (category, questions) in suite["aligned"].as_object().ok_or_else(|| anyhow::anyhow!("missing aligned suite"))? {
        for question in questions.as_array().ok_or_else(|| anyhow::anyhow!("invalid category"))? {
            let question = question.as_str().ok_or_else(|| anyhow::anyhow!("invalid prompt"))?;
            let prompt = format!("<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n");
            let start = Instant::now();
            let response = engine.generate(prompt, 128, 0.65, 0.9).await?;
            let record = json!({"category":category,"prompt":question,"response":response,"seconds":start.elapsed().as_secs_f64()});
            eprintln!("{}", record);
            records.push(record);
        }
    }
    println!("{}", serde_json::to_string_pretty(&records)?);
    Ok(())
}
