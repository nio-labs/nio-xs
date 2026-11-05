use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

fn collect_files(dir: &Path, files: &mut Vec<PathBuf>, depth: usize) {
    if depth > 4 { return; }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            let s = p.to_string_lossy();
            if s.contains("/target") || s.contains("/.git") || s.contains("/node_modules") {
                continue;
            }
            if p.is_dir() {
                collect_files(&p, files, depth + 1);
            } else if let Some(ext) = p.extension() {
                if (ext == "md" || ext == "rs" || ext == "py") && entry.metadata().map(|m| m.len() < 500_000).unwrap_or(false) {
                    files.push(p);
                }
            }
        }
    }
}

fn main() -> Result<()> {
    let tokenizer_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/blue-tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path).unwrap();

    let mut bigrams: HashMap<u32, HashMap<u32, usize>> = HashMap::new();
    let mut trigrams: HashMap<(u32, u32), HashMap<u32, usize>> = HashMap::new();
    let mut total_tokens = 0usize;

    // 1. Mine Blue's exact training dataset (weighted 5x so Blue's native phrasing dominates)
    if let Ok(gold) = std::fs::read_to_string("/home/mn/nio-labs/nio-ideas/dataset/nio_blue_gold_dataset.jsonl") {
        for line in gold.lines() {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                let prompt = val.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
                let resp = val.get("response").and_then(|v| v.as_str()).unwrap_or("");
                for (text, weight) in [(prompt, 2usize), (resp, 5usize)] {
                    if let Ok(enc) = tokenizer.encode(text, false) {
                        let ids = enc.get_ids();
                        total_tokens += ids.len();
                        for w in ids.windows(2) {
                            *bigrams.entry(w[0]).or_default().entry(w[1]).or_default() += weight;
                        }
                        for w in ids.windows(3) {
                            *trigrams.entry((w[0], w[1])).or_default().entry(w[2]).or_default() += weight;
                        }
                    }
                }
            }
        }
    }

    // 2. Also scan workspace docs/code
    let mut files = Vec::new();
    collect_files(Path::new("/home/mn/nio-labs"), &mut files, 0);

    for f in &files {
        if let Ok(content) = std::fs::read_to_string(f) {
            if let Ok(enc) = tokenizer.encode(content.as_str(), false) {
                let ids = enc.get_ids();
                total_tokens += ids.len();
                for w in ids.windows(2) {
                    *bigrams.entry(w[0]).or_default().entry(w[1]).or_default() += 1;
                }
                for w in ids.windows(3) {
                    *trigrams.entry((w[0], w[1])).or_default().entry(w[2]).or_default() += 1;
                }
            }
        }
    }

    let is_special = |t: u32| t == 0 || t == 1 || t == 2 || t >= 32000;

    let mut out_bigrams: HashMap<u32, u32> = HashMap::new();
    for (&t1, nexts) in &bigrams {
        if is_special(t1) {
            continue;
        }
        let total: usize = nexts.values().sum();
        if total >= 8 {
            if let Some((&t2, &cnt)) = nexts.iter().max_by_key(|&(_, c)| *c) {
                if is_special(t2) {
                    continue;
                }
                let ratio = cnt as f64 / total as f64;
                if ratio >= 0.65 {
                    out_bigrams.insert(t1, t2);
                }
            }
        }
    }

    let mut out_trigrams: HashMap<u64, u32> = HashMap::new();
    for (&(t1, t2), nexts) in &trigrams {
        if is_special(t1) || is_special(t2) {
            continue;
        }
        let total: usize = nexts.values().sum();
        if total >= 10 {
            if let Some((&t3, &cnt)) = nexts.iter().max_by_key(|&(_, c)| *c) {
                if is_special(t3) {
                    continue;
                }
                let ratio = cnt as f64 / total as f64;
                if ratio >= 0.75 && cnt >= 8 {
                    let key = ((t1 as u64) << 32) | (t2 as u64);
                    out_trigrams.insert(key, t3);
                }
            }
        }
    }

    println!("Total tokens scanned: {}", total_tokens);
    println!("Blue bigrams saved: {}", out_bigrams.len());
    println!("Blue trigrams saved: {}", out_trigrams.len());

    let bi_bytes = serde_json::to_vec(&out_bigrams)?;
    std::fs::write("/home/mn/nio-labs/nio-xs/crates/engine/src/blue_subwords.json", &bi_bytes)?;

    let tri_bytes = serde_json::to_vec(&out_trigrams)?;
    std::fs::write("/home/mn/nio-labs/nio-xs/crates/engine/src/blue_trigrams.json", &tri_bytes)?;
    println!(
        "Wrote blue_subwords.json ({:.1} KB) and blue_trigrams.json ({:.1} KB)",
        bi_bytes.len() as f64 / 1024.0,
        tri_bytes.len() as f64 / 1024.0
    );

    Ok(())
}
