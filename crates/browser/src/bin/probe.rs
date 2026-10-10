use nioxs_browser::{classify, fetch};
use serde_json::json;
use std::path::Path;

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let Some(binary) = args.next() else {
        eprintln!("usage: probe OBSCURA_BINARY HTTPS_URL...");
        std::process::exit(2);
    };
    for url in args {
        match fetch(Path::new(&binary), &url).await {
            Ok(text) => println!(
                "{}",
                json!({"url":url,"status":classify(&text),"bytes":text.len(),"excerpt":text.chars().take(400).collect::<String>()})
            ),
            Err(error) => println!("{}", json!({"url":url,"status":"error","error":error})),
        }
    }
}
