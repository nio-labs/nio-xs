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
    let tokenizer_path = PathBuf::from("/home/mn/nio-labs/nio-xs-models/beluga-tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path).unwrap();

    let mut bigrams: HashMap<u32, HashMap<u32, usize>> = HashMap::new();
    let mut total_tokens = 0usize;

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
            }
        }
    }

    let mut high_conf_pairs: Vec<(u32, u32, usize, f64)> = Vec::new();
    for (&t1, nexts) in &bigrams {
        // Exclude EOS / special tokens
        if t1 == 151643 || t1 == 151644 || t1 == 151645 {
            continue;
        }
        let total: usize = nexts.values().sum();
        if total >= 8 {
            if let Some((&t2, &cnt)) = nexts.iter().max_by_key(|&(_, c)| *c) {
                if t2 == 151643 || t2 == 151644 || t2 == 151645 {
                    continue;
                }
                let ratio = cnt as f64 / total as f64;
                if ratio >= 0.60 {
                    high_conf_pairs.push((t1, t2, cnt, ratio));
                }
            }
        }
    }

    println!("Total tokens scanned: {}", total_tokens);
    println!("High confidence pairs (>=60% dominant, cnt>=8): {}", high_conf_pairs.len());

    let mut out_map: HashMap<u32, u32> = HashMap::new();
    for &(t1, t2, _, _) in &high_conf_pairs {
        out_map.insert(t1, t2);
    }
    // Always include reasoning tag transitions
    out_map.insert(13708, 766);  // <th -> ink
    out_map.insert(766, 29);     // ink -> >
    out_map.insert(26418, 26865); // ...</ -> think
    out_map.insert(26865, 397);   // think -> >\n

    let json_bytes = serde_json::to_vec(&out_map)?;
    std::fs::write("/home/mn/nio-labs/nio-xs-models/universal_subwords.json", json_bytes)?;
    println!("Saved {} universal subwords to universal_subwords.json", out_map.len());

    Ok(())
}
