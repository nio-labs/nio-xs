"""
NioRay-v2 Dataset Generator
Generates high-density training pairs across 20 developer keystroke reflex tasks
formatted for Qwen-based distillation into a non-autoregressive MicroLM.
"""

import json
from typing import List, Dict

TASK_TEMPLATES = {
    "shell_completion": [
        ("find . -name \"*.rs\" -exec gr", "grep -Hn \"\" {} +"),
        ("git commit -m \"feat(engine):", " add AST streaming eviction\""),
        ("cargo test -- --nocapture", " --test-threads=1"),
        ("grep -rn \"TODO\" ./src", " --exclude-dir=target"),
        ("docker run -d --name niodb -p 7432:7432", " nio-labs/niodb:latest"),
    ],
    "function_signature": [
        ("pub fn calculate_velocity(tokens: usize, duration: Duration)", " -> f64 {\n    tokens as f64 / duration.as_secs_f64()\n}"),
        ("pub fn is_empty(&self)", " -> bool {\n    self.len() == 0\n}"),
        ("pub async fn fetch_turn(id: &str)", " -> Result<Turn, Error> {"),
        ("fn parse_header(line: &str)", " -> Option<(&str, &str)> {"),
    ],
    "match_arm": [
        ("match status {\n    Status::Idle => \"waiting\",\n    Status::Running =>", " \"in progress\",\n    Status::Done => \"completed\",\n    Status::Error(e) => \"failed\",\n}"),
        ("match result {\n    Ok(val) => val,\n    Err(e) =>", " return Err(e.into()),\n}"),
        ("match opt {\n    Some(x) => x * 2,\n    None =>", " 0,\n}"),
    ],
    "variable_rename": [
        ("let tmp_val = read_token();\nprocess(tmp_val);", "let cached_token = read_token();\nprocess(cached_token);"),
        ("let x = get_count();\nif x > 0 { print(x); }", "let total_items = get_count();\nif total_items > 0 { print(total_items); }"),
    ],
    "missing_import": [
        ("Missing import for AtomicBool::new(false)", "use std::sync::atomic::{AtomicBool, Ordering};"),
        ("Missing import for Instant::now()", "use std::time::Instant;"),
        ("Missing import for HashMap", "use std::collections::HashMap;"),
        ("Missing import for serde Serialize", "use serde::{Serialize, Deserialize};"),
    ],
    "if_to_match": [
        ("if code == 200 { \"OK\" } else { \"Error\" }", "match code {\n    200 => \"OK\",\n    _ => \"Error\",\n}"),
        ("if is_valid { 1 } else { 0 }", "match is_valid {\n    true => 1,\n    false => 0,\n}"),
    ],
    "unit_test": [
        ("fn add(a: i32, b: i32) -> i32 { a + b }", "#[test]\nfn test_add() {\n    assert_eq!(add(2, 3), 5);\n    assert_eq!(add(-1, 1), 0);\n}"),
        ("fn is_even(n: i64) -> bool { n % 2 == 0 }", "#[test]\nfn test_is_even() {\n    assert!(is_even(4));\n    assert!(!is_even(5));\n}"),
    ],
    "borrow_checker": [
        ("let mut s = String::from(\"hello\");\nlet r1 = &s;\ns.push_str(\" world\");\nprintln!(\"{}\", r1);", "let mut s = String::from(\"hello\");\nlet r1 = s.clone();\ns.push_str(\" world\");\nprintln!(\"{}\", r1);"),
    ],
    "error_handling": [
        ("fn read_config(path: &str) {\n    let s = std::fs::read_to_string(path);\n    s\n}", "fn read_config(path: &str) -> Result<String, std::io::Error> {\n    let s = std::fs::read_to_string(path)?;\n    Ok(s)\n}"),
    ],
    "loop_to_iterator": [
        ("let mut res = Vec::new();\nfor x in nums {\n    if x > 0 {\n        res.push(x * 2);\n    }\n}", "let res: Vec<_> = nums.into_iter().filter(|&x| x > 0).map(|x| x * 2).collect();"),
    ],
    "todo_comment": [
        ("fn is_even(n: i64) -> bool {\n    // TODO: check if n is even\n}", "fn is_even(n: i64) -> bool {\n    n % 2 == 0\n}"),
    ],
    "cli_parser": [
        ("match arg.as_str() {\n    \"-v\" | \"--version\" => println!(\"1.0\"),\n    \"-h\" | \"--help\" =>", " {\n        print_help();\n        return Ok(());\n    }"),
    ],
    "log_statement": [
        ("fn track(user_id: u64, latency_ms: u128) {", "fn track(user_id: u64, latency_ms: u128) {\n    eprintln!(\"[metrics] user={user_id} latency={latency_ms}ms\");"),
    ],
    "json_to_struct": [
        ("{\"name\": \"Alice\", \"age\": 30, \"active\": true}", "#[derive(Debug, serde::Deserialize, serde::Serialize)]\npub struct User {\n    pub name: String,\n    pub age: u32,\n    pub active: bool,\n}"),
    ],
    "regex_pattern": [
        ("Generate regex for IPv4 address", r"^((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.?\b){4}$"),
        ("Generate regex for email address", r"^[\w\.-]+@[\w\.-]+\.\w+$"),
    ],
    "sql_rewrite": [
        ("SELECT users.name, orders.total FROM users, orders WHERE users.id = orders.user_id", "SELECT users.name, orders.total\nFROM users\nINNER JOIN orders ON users.id = orders.user_id;"),
    ],
    "doc_comments": [
        ("pub fn reset_cache() -> bool", "/// Resets in-memory model caches and evicts unmapped pages.\n/// Returns `true` if reclamation succeeded.\npub fn reset_cache() -> bool"),
    ],
    "formatting": [
        ("fn foo() {\nlet x = 1;\n    let y = 2;\nreturn x + y;\n}", "fn foo() -> i32 {\n    let x = 1;\n    let y = 2;\n    x + y\n}"),
    ],
    "commit_message": [
        ("perf(engine): evict unused pages via", "perf(engine): evict unused pages via madvise(MADV_DONTNEED) to maintain < 86MB peak RAM"),
    ],
    "readme_snippet": [
        ("Install Nio via cargo", "### Installation\nInstall Nio directly using Cargo:\n```bash\ncargo install nio\n```"),
    ]
}

def generate_dataset(output_path: str = "nioray_v2_dataset.jsonl", samples_per_task: int = 50) -> List[Dict[str, str]]:
    dataset = []
    for task_name, pairs in TASK_TEMPLATES.items():
        for i in range(samples_per_task):
            idx = i % len(pairs)
            prompt, completion = pairs[idx]
            formatted_prompt = f"<|im_start|>system\nYou are NioRay-v2, a sub-second code reflex engine. Provide the completion directly.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            dataset.append({
                "task": task_name,
                "input": formatted_prompt,
                "target": completion
            })
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
            
    print(f"✔ Generated {len(dataset)} reflex examples to {output_path}")
    return dataset

if __name__ == "__main__":
    generate_dataset()
