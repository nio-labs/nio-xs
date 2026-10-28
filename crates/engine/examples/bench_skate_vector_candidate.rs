use anyhow::Result;
use nioxs_engine::{RayEngine, skate_system, memory_mb};
use std::{path::PathBuf, time::Instant};
#[tokio::main]
async fn main() -> Result<()> {
 let a: Vec<String> = std::env::args().collect();
    let system_prompt = skate_system()?;
 let engine = RayEngine::from_paths(PathBuf::from(&a[1]), PathBuf::from(&a[2]));
 let fixture: serde_json::Value = serde_json::from_slice(&std::fs::read(&a[3])?)?;
 let questions = fixture["held_out"].as_array().ok_or_else(|| anyhow::anyhow!("missing held_out"))?;
 for q in questions {
  let q = q.as_str().ok_or_else(|| anyhow::anyhow!("invalid prompt"))?;
  let prompt = format!("<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n");
  let start=Instant::now(); let mut first=None; let mut text=String::new();
  let mut rx=engine.generate_stream(prompt,128,0.65,0.9).await?;
  while let Some(chunk)=rx.recv().await { if first.is_none() {first=Some(start.elapsed().as_secs_f64());} text.push_str(&chunk); }
  let total=start.elapsed().as_secs_f64(); let (rss,peak)=memory_mb();
  println!("{}",serde_json::json!({"prompt":q,"response":text,"total_s":total,"ttft_s":first,"words":text.split_whitespace().count(),"rss_mb":rss,"peak_mb":peak}));
 }
 Ok(())
}
