use reqwest::Client;
use serde_json::{json, Value};
use tokio::io::AsyncWriteExt;
use std::io::Write;

pub async fn consult_kilo_stream(prompt: &str) -> Result<String, String> {
    let client = Client::new();
    
    // Print the prompt context to terminal just like the screenshot showed, but real.
    println!("\n🤖 \x1b[1;36mKilo Code Consultant\x1b[0m streaming response for: '{}'\n", prompt);
    let _ = std::io::stdout().flush();

    let mut res = client
        .post("https://api.kilo.ai/api/gateway/chat/completions")
        .json(&json!({
            "model": "kilo-auto/free",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": true
        }))
        .send()
        .await
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    if !res.status().is_success() {
        return Err(format!("Kilo API Error: HTTP {}", res.status()));
    }

    let mut full_response = String::new();

    while let Ok(Some(chunk)) = res.chunk().await {
        let chunk_str = String::from_utf8_lossy(&chunk);
        for line in chunk_str.lines() {
            if line.starts_with("data: ") {
                let data = &line["data: ".len()..];
                if data == "[DONE]" {
                    break;
                }
                if let Ok(json) = serde_json::from_str::<Value>(data) {
                    if let Some(content) = json["choices"][0]["delta"]["content"].as_str() {
                        print!("{}", content);
                        let _ = std::io::stdout().flush();
                        full_response.push_str(content);
                    }
                }
            }
        }
    }
    
    println!("\n\n✓ Finished streaming from Kilo Code.");
    Ok(full_response)
}
