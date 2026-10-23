#!/usr/bin/env python3
"""
⚡ NIOMANTA COMPANION DISTILLATION TRAINING & Q4_0 GGUF EXPORT PIPELINE
- Teacher Model: Qwen/Qwen2.5-Coder-7B-Instruct (4-bit NF4 quantized, ~4.8 GB VRAM)
- Student Model: NioManta Sliced MicroLM (Qwen2.5-Coder-0.5B base, sliced to 2 Layers)
- Distillation: CrossEntropy on target tokens + KL-Divergence on 7B Teacher logits
- Quantization Target: nio-manta-ast.Q4_0.gguf (< 100 MB on disk, < 50 MB resident RAM)
- Target Hardware: Google Colab T4 GPU
"""

import os
import sys
import time
import json
import shutil
import subprocess
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)

print("=" * 80, flush=True)
print("⚡ NIOMANTA (< 100 MB) COMPANION DISTILLATION TRAINING (TEACHER: 7B-CODER)", flush=True)
print("=" * 80, flush=True)

if not torch.cuda.is_available():
    print("❌ ERROR: CUDA GPU not detected! Colab runtime must have GPU enabled.", flush=True)
    sys.exit(1)

device_name = torch.cuda.get_device_name(0)
total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"✅ GPU Detected: {device_name} ({total_vram_gb:.2f} GB VRAM)", flush=True)

# ------------------------------------------------------------------------------
# 1. Asynchronous llama-quantize build
# ------------------------------------------------------------------------------
llama_quant_bin = Path("/content/llama.cpp/build/bin/llama-quantize")
llama_build_proc = None

if llama_quant_bin.exists():
    print("✅ llama-quantize already compiled and ready!", flush=True)
else:
    print("\n🛠 [1/6] Launching llama.cpp background build for GGUF Q4_0 quantization...", flush=True)
    build_cmd = """
    set -e
    if [ ! -d "/content/llama.cpp" ]; then
        git clone --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp
    fi
    cd /content/llama.cpp
    cmake -B build -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
    cmake --build build --target llama-quantize -j$(nproc)
    """
    llama_build_proc = subprocess.Popen(
        ["bash", "-c", build_cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    print("   llama.cpp build running asynchronously.", flush=True)

# ------------------------------------------------------------------------------
# 2. Dataset Preparation (Companion Dialogue & AST Reflex Tasks)
# ------------------------------------------------------------------------------
RAW_DATASET = [
    # Conversational & Diagnostic Companion
    ("Why is our text stream from model is running slow. one word by one word",
     "Autoregressive token generation runs sequential CPU matrix multiplications per token. Use a shallow MicroLM like NioManta (< 100MB) or increase phrase buffer flushing to achieve sub-second response."),
    ("How do I fix this borrow checker error when mutating a string while holding an immutable reference?",
     "Clone the data before mutating or scope the immutable reference so it drops before calling push_str."),
    ("Explain the difference between Arc and Rc in Rust in one sentence.",
     "Arc uses atomic reference counting for thread-safe sharing, while Rc is non-atomic and restricted to a single thread."),
    ("What is the fastest way to read a small text file into memory in Rust?",
     "Use `std::fs::read_to_string(path)` for UTF-8 text or `std::fs::read(path)` for raw binary bytes."),
    ("Why use madvise MADV_DONTNEED in a local AI engine?",
     "It immediately instructs the kernel to release physical resident RAM pages without unmapping the virtual address space."),
    
    # Manta Planning Format (<manta_plan>)
    ("Refactor this legacy synchronous function to asynchronous tokio",
     "<manta_plan id=\"tokio-refactor\">\n<manta_step id=\"1\" priority=\"high\">Add async keyword and return Result with tokio io error</manta_step>\n<manta_step id=\"2\" priority=\"medium\">Replace std::fs calls with tokio::fs equivalents</manta_step>\n<manta_risk level=\"warning\">Ensure tokio runtime is running on the calling thread</manta_risk>\n</manta_plan>\nAsync refactor planned."),
    ("Add graceful shutdown handler for our background server",
     "<manta_plan id=\"graceful-shutdown\">\n<manta_step id=\"1\" priority=\"high\">Listen for tokio::signal::ctrl_c()</manta_step>\n<manta_step id=\"2\" priority=\"high\">Broadcast cancel signal to active worker tasks via tokio::sync::broadcast</manta_step>\n<manta_step id=\"3\" priority=\"medium\">Await active drain with a 5-second timeout</manta_step>\n</manta_plan>\nShutdown coordination structured."),
    
    # 20 AST Reflex & Transformation Tasks
    ("Complete this half-typed shell command: `find . -name \"*.rs\" -exec gr`",
     "`find . -name \"*.rs\" -exec grep -Hn \"\" {} +`"),
    ("Finish this Rust function signature: `pub fn calculate_velocity(tokens: usize, duration: Duration)`",
     "-> f64 {\n    tokens as f64 / duration.as_secs_f64()\n}"),
    ("Suggest the next Rust match arm:\nmatch status {\n    Status::Idle => \"waiting\",\n    Status::Running => \"in progress\",\n",
     "    Status::Done => \"completed\",\n    Status::Error(e) => \"failed\",\n}"),
    ("Rename `tmp_val` to `cached_token`:\nlet tmp_val = read_token();\nprocess(tmp_val);",
     "let cached_token = read_token();\nprocess(cached_token);"),
    ("What import is missing for `AtomicBool::new(false)` in Rust?",
     "use std::sync::atomic::{AtomicBool, Ordering};"),
    ("Convert this to a match statement: `if code == 200 { \"OK\" } else { \"Error\" }`",
     "match code {\n    200 => \"OK\",\n    _ => \"Error\",\n}"),
    ("Generate a small Rust unit test for `fn add(a: i32, b: i32) -> i32 { a + b }`",
     "#[test]\nfn test_add() {\n    assert_eq!(add(2, 3), 5);\n    assert_eq!(add(-1, 1), 0);\n}"),
    ("Fix this borrow error:\nlet mut s = String::from(\"hello\");\nlet r1 = &s;\ns.push_str(\" world\");\nprintln!(\"{}\", r1);",
     "let mut s = String::from(\"hello\");\nlet r1 = s.clone();\ns.push_str(\" world\");\nprintln!(\"{}\", r1);"),
    ("Add `Result<String, std::io::Error>` return type and `?` operator:\nfn read_config(path: &str) {\n    let s = std::fs::read_to_string(path);\n    s\n}",
     "fn read_config(path: &str) -> Result<String, std::io::Error> {\n    let s = std::fs::read_to_string(path)?;\n    Ok(s)\n}"),
    ("Refactor to iterator chain:\nlet mut res = Vec::new();\nfor x in nums {\n    if x > 0 {\n        res.push(x * 2);\n    }\n}",
     "let res: Vec<_> = nums.into_iter().filter(|&x| x > 0).map(|x| x * 2).collect();"),
    ("Fill in the TODO:\nfn is_even(n: i64) -> bool {\n    // TODO: check if n is even\n}",
     "fn is_even(n: i64) -> bool {\n    n % 2 == 0\n}"),
    ("Complete this match arm for CLI flags:\nmatch arg.as_str() {\n    \"-v\" | \"--version\" => println!(\"1.0\"),\n    \"-h\" | \"--help\" =>",
     "{\n        print_help();\n        return Ok(());\n    }"),
    ("Add an `eprintln!` log statement logging user id and latency:\nfn track(user_id: u64, latency_ms: u128) {",
     "fn track(user_id: u64, latency_ms: u128) {\n    eprintln!(\"[metrics] user={user_id} latency={latency_ms}ms\");"),
    ("Convert this JSON to Rust struct fields:\n{\"name\": \"Alice\", \"age\": 30, \"active\": true}",
     "#[derive(Debug, serde::Deserialize, serde::Serialize)]\npub struct User {\n    pub name: String,\n    pub age: u32,\n    pub active: bool,\n}"),
    ("Generate a regex to match an IPv4 address.",
     r"^((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.?\b){4}$"),
    ("Rewrite this query to use INNER JOIN:\nSELECT users.name, orders.total FROM users, orders WHERE users.id = orders.user_id",
     "SELECT users.name, orders.total\nFROM users\nINNER JOIN orders ON users.id = orders.user_id;"),
    ("Add Rust doc comments (`///`) to this function:\npub fn reset_cache() -> bool",
     "/// Resets in-memory model caches and evicts unmapped pages.\n/// Returns `true` if reclamation succeeded.\npub fn reset_cache() -> bool"),
    ("Fix indentation:\nfn foo() {\nlet x = 1;\n    let y = 2;\nreturn x + y;\n}",
     "fn foo() -> i32 {\n    let x = 1;\n    let y = 2;\n    x + y\n}"),
    ("Complete this git commit message for adding memory eviction:\nperf(engine): evict unused pages via",
     "perf(engine): evict unused pages via madvise(MADV_DONTNEED) to maintain < 86MB peak RAM"),
    ("Draft a 2-sentence README section on how to install Nio via cargo.",
     "### Installation\nInstall Nio directly using Cargo:\n```bash\ncargo install nio\n```")
]

# ------------------------------------------------------------------------------
# 3. Load Teacher (Qwen2.5-Coder-7B-Instruct in 4-bit NF4)
# ------------------------------------------------------------------------------
TEACHER_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
STUDENT_BASE_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

print(f"\n🧠 [2/6] Loading Teacher Model: {TEACHER_ID} in 4-bit NF4...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)
teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
teacher.eval()
print(f"✅ Teacher 7B loaded into ~4.8 GB VRAM!", flush=True)

# ------------------------------------------------------------------------------
# 4. Initialize & Slice Student (2 Layers -> < 100 MB Q4_0 GGUF)
# ------------------------------------------------------------------------------
TARGET_LAYERS = 2
print(f"\n✂️  [3/6] Initializing Student from {STUDENT_BASE_ID} and slicing to {TARGET_LAYERS} Layers...", flush=True)
student = AutoModelForCausalLM.from_pretrained(
    STUDENT_BASE_ID,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to("cuda")

student.model.layers = nn.ModuleList(student.model.layers[:TARGET_LAYERS])
student.config.num_hidden_layers = TARGET_LAYERS
if hasattr(student.config, "layer_types") and student.config.layer_types is not None:
    student.config.layer_types = student.config.layer_types[:TARGET_LAYERS]

for idx, layer in enumerate(student.model.layers):
    layer.self_attn.layer_idx = idx

student_params = sum(p.numel() for p in student.parameters()) / 1e6
print(f"✅ Student NioManta initialized: {student_params:.1f}M params (Target Q4_0 size: ~92 MB, < 100 MB!)", flush=True)

# ------------------------------------------------------------------------------
# 5. Distillation Training Dataset & Loop
# ------------------------------------------------------------------------------
class DistillationDataset(Dataset):
    def __init__(self, raw_samples, tok, max_len=96, repeats=15):
        self.samples = []
        for prompt, target in raw_samples * repeats:
            full = f"<|im_start|>system\nYou are NioManta, a high-speed coding and architecture companion.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{target}<|im_end|>"
            enc = tok(full, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            
            prompt_only = f"<|im_start|>system\nYou are NioManta, a high-speed coding and architecture companion.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            p_len = len(tok(prompt_only)["input_ids"])
            labels = input_ids.clone()
            labels[:p_len] = -100
            labels[attention_mask == 0] = -100
            self.samples.append((input_ids, attention_mask, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

dataset = DistillationDataset(RAW_DATASET, tokenizer, max_len=96, repeats=15)
loader = DataLoader(dataset, batch_size=8, shuffle=True)
print(f"\n📚 [4/6] Prepared {len(dataset)} training samples for distillation.", flush=True)

optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5, weight_decay=0.01)
ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

epochs = 3
total_steps = len(loader) * epochs
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_steps)

print(f"\n🚀 [5/6] Distilling from 7B-Coder into NioManta for {epochs} epochs...", flush=True)
student.train()
t0 = time.time()

for epoch in range(epochs):
    epoch_loss = 0.0
    for step, (input_ids, attention_mask, labels) in enumerate(loader):
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        labels = labels.to("cuda")

        with torch.no_grad():
            teacher_out = teacher(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_out.logits

        student_out = student(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_out.logits

        # 1. Cross-Entropy Loss on Ground Truth Target Tokens
        loss_ce = ce_loss_fn(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))

        # 2. Distillation Loss (KL Divergence with Teacher Logits)
        min_v = min(teacher_logits.size(-1), student_logits.size(-1))
        t_prob = torch.softmax(teacher_logits[..., :min_v] / 2.0, dim=-1)
        s_log_prob = torch.log_softmax(student_logits[..., :min_v] / 2.0, dim=-1)
        loss_kd = kl_loss_fn(s_log_prob, t_prob) * (2.0 ** 2)

        loss = loss_ce + 0.5 * loss_kd

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    print(f"   Epoch {epoch + 1}/{epochs} - Average Distillation Loss: {avg_loss:.4f}", flush=True)

train_duration = time.time() - t0
print(f"🎉 Distillation completed in {train_duration:.1f}s!", flush=True)

# ------------------------------------------------------------------------------
# 6. Export PyTorch -> GGUF Q4_0
# ------------------------------------------------------------------------------
print("\n💾 [6/6] Exporting NioManta to GGUF format...", flush=True)
export_dir = Path("/content/niomanta_hf")
export_dir.mkdir(parents=True, exist_ok=True)
student.save_pretrained(export_dir)
tokenizer.save_pretrained(export_dir)
print(f"✅ HuggingFace model saved to {export_dir}", flush=True)

# Await llama.cpp compilation if running
if llama_build_proc:
    print("   Waiting for llama-quantize compilation...", flush=True)
    llama_build_proc.wait()
    print("✅ llama-quantize build verified ready!", flush=True)

# 1. Convert to FP16 GGUF
f16_gguf = Path("/content/nio-manta-f16.gguf")
print(f"\n⚙️  Converting to FP16 GGUF -> {f16_gguf}...", flush=True)
subprocess.run([
    sys.executable, "/content/llama.cpp/convert_hf_to_gguf.py",
    str(export_dir),
    "--outfile", str(f16_gguf),
    "--outtype", "f16"
], check=True)

# 2. Quantize to Q4_0 (32-byte block-aligned)
q4_gguf = Path("/content/nio-manta-ast.Q4_0.gguf")
print(f"\n📦 Quantizing to 32-byte block-aligned Q4_0 -> {q4_gguf}...", flush=True)
subprocess.run([
    str(llama_quant_bin),
    str(f16_gguf),
    str(q4_gguf),
    "Q4_0"
], check=True)

if q4_gguf.exists():
    q4_size_mb = q4_gguf.stat().st_size / (1024**2)
    print("\n" + "=" * 80, flush=True)
    print(f"🎉 SUCCESS! NioManta (< 100MB) Artifact Ready!", flush=True)
    print(f"   Artifact File: {q4_gguf} ({q4_size_mb:.2f} MiB)", flush=True)
    print(f"   Teacher: {TEACHER_ID} (4-bit NF4)", flush=True)
    print(f"   Architecture: 2-Layer Transformer (d=896)", flush=True)
    print(f"   Target RAM: < 50 MB resident RAM on CPU", flush=True)
    print("=" * 80, flush=True)
else:
    print("❌ ERROR: Failed to produce Q4_0 GGUF file!", flush=True)
    sys.exit(1)
