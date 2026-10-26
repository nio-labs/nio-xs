#!/usr/bin/env python3
"""
⚡ NIOMANTA ONNX ENGINE RUNNER & TEST HARNESS FOR NIO-XS
Official in-process execution script for NioManta (~166M INT8 ONNX MicroLM).
Companion to NioMini (7.1M ONNX) and NioMicro (5.0M ONNX).

Usage:
  python3 scripts/run_niomanta_onnx.py -p "How do I fix borrow checker error in Rust?"
  python3 scripts/run_niomanta_onnx.py --plan -p "Migrate SQLite to PostgreSQL"
  python3 scripts/run_niomanta_onnx.py -b
  python3 scripts/run_niomanta_onnx.py -i
"""

import os
import sys
import time
import argparse
import re
from pathlib import Path

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
except ImportError as e:
    print(f"❌ Missing required dependency: {e}")
    print("Run using: /home/mn/nio-labs/.venv/bin/python3 scripts/run_niomanta_onnx.py")
    sys.exit(1)


def get_memory_rss_mb() -> float:
    """Reads resident memory set size (VmRSS) from /proc/self/status on Linux."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def find_default_paths():
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"

    model_candidates = [
        models_dir / "niomanta_int8.onnx",
        models_dir / "nio-manta.onnx",
        Path.home() / ".nioxs" / "models" / "niomanta_int8.onnx",
        Path.home() / ".nioxs" / "models" / "nio-manta.onnx",
        base_dir.parent / "nio-xs-models" / "models" / "nio-manta" / "niomanta_int8.onnx",
        base_dir.parent / "nio-xs-models" / "models" / "nio-manta" / "nio-manta.onnx",
    ]
    model_path = next((p for p in model_candidates if p.exists()), None)

    tok_candidates = [
        models_dir / "nio-manta-tokenizer.json",
        models_dir / "tokenizer.json",
        base_dir.parent / "nio-ideas" / "models" / "tokenizer.json",
        base_dir.parent / "nio-xs-models" / "models" / "nio-manta" / "nio-manta-tokenizer.json",
        base_dir.parent / "nio-xs-models" / "models" / "nio-manta" / "tokenizer.json",
    ]
    tok_path = next((p for p in tok_candidates if p.exists()), None)

    return model_path, tok_path


class NioMantaOnnxEngine:
    def __init__(self, model_path: str, tokenizer_path: str, threads: int = 4):
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"NioManta ONNX model not found: {self.model_path}")
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {self.tokenizer_path}")

        # Initialize ONNX Runtime Session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Crucial for Edge: disable internal caching arena to keep RSS bounded
        sess_options.enable_cpu_mem_arena = False

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )

        # Initialize Tokenizer
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.eos_token_ids = {151645, 151643}  # <|im_end|>, <|endoftext|>

    def format_prompt(self, user_prompt: str, is_plan: bool = False) -> str:
        if is_plan:
            sys_msg = (
                "You are NioManta, a high-speed coding and architecture companion. "
                "Output your plan using <manta_plan id=\"...\"> with <manta_step> and <manta_risk> XML tags."
            )
        else:
            sys_msg = "You are NioManta, a high-speed coding and architecture companion."

        return (
            f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        is_plan: bool = False,
        stream_callback=None
    ):
        full_prompt = self.format_prompt(prompt, is_plan=is_plan)
        input_ids = self.tokenizer.encode(full_prompt).ids
        prompt_len = len(input_ids)

        tokens = list(input_ids)
        generated_tokens = []

        t_start = time.perf_counter()
        ttft_ms = 0.0

        for step in range(max_new_tokens):
            inp = np.array([tokens], dtype=np.int64)
            step_t0 = time.perf_counter()
            outputs = self.session.run(["logits"], {"input_ids": inp})
            logits = outputs[0][0, -1, :]  # Shape: (vocab_size,)

            if step == 0:
                ttft_ms = (time.perf_counter() - t_start) * 1000

            if temperature <= 1e-4:
                next_tok = int(np.argmax(logits))
            else:
                scaled = logits / temperature
                exp_logits = np.exp(scaled - np.max(scaled))
                probs = exp_logits / np.sum(exp_logits)
                next_tok = int(np.random.choice(len(probs), p=probs))

            if next_tok in self.eos_token_ids:
                break

            tokens.append(next_tok)
            generated_tokens.append(next_tok)

            if stream_callback:
                chunk_str = self.tokenizer.decode([next_tok])
                stream_callback(chunk_str)

        total_time = time.perf_counter() - t_start
        gen_time = max(0.0001, total_time - (ttft_ms / 1000.0))
        num_gen = len(generated_tokens)
        tok_per_sec = (num_gen - 1) / gen_time if num_gen > 1 else (num_gen / total_time)

        output_text = self.tokenizer.decode(generated_tokens)

        return {
            "text": output_text,
            "prompt_tokens": prompt_len,
            "generated_tokens": num_gen,
            "total_tokens": prompt_len + num_gen,
            "ttft_ms": ttft_ms,
            "total_time_s": total_time,
            "tokens_per_second": tok_per_sec,
            "rss_mb": get_memory_rss_mb(),
        }


def parse_manta_plan(text: str):
    """Extract and format <manta_plan> XML tags."""
    plan_match = re.search(r'<manta_plan\s+id="([^"]+)">', text)
    if not plan_match:
        return None

    plan_id = plan_match.group(1)
    steps = re.findall(r'<manta_step\s+id="([^"]+)"(?:\s+priority="([^"]+)")?>([^<]+)</manta_step>', text)
    risks = re.findall(r'<manta_risk(?:\s+level="([^"]+)")?>([^<]+)</manta_risk>', text)

    return {
        "id": plan_id,
        "steps": [{"id": s[0], "priority": s[1] or "normal", "description": s[2].strip()} for s in steps],
        "risks": [{"level": r[0] or "warning", "message": r[1].strip()} for r in risks],
    }


def run_benchmark(engine: NioMantaOnnxEngine):
    test_cases = [
        ("AST Reflex", "Complete this Rust function: pub fn add(a: i32, b: i32) -> i32 {"),
        ("Architecture", "Explain the difference between Arc and Rc in Rust in one sentence."),
        ("Diagnostics", "Why does mutating a String in a loop cause a borrow checker error?"),
        ("Manta Plan", "Plan a step-by-step migration from SQLite to PostgreSQL for FastAPI."),
    ]

    print("\n" + "=" * 78)
    print("⚡ NIOMANTA ONNX BENCHMARK (Edge CPU Execution)")
    print("=" * 78)
    print(f"{'Task':<16} | {'Prompt Tok':<10} | {'Gen Tok':<8} | {'TTFT (ms)':<10} | {'Tok/Sec':<10} | {'RSS (MB)':<8}")
    print("-" * 78)

    ttfts = []
    speeds = []

    for name, prompt in test_cases:
        res = engine.generate(prompt, max_new_tokens=24, is_plan=("Plan" in name))
        ttfts.append(res["ttft_ms"])
        speeds.append(res["tokens_per_second"])
        print(f"{name:<16} | {res['prompt_tokens']:<10} | {res['generated_tokens']:<8} | {res['ttft_ms']:<10.1f} | {res['tokens_per_second']:<10.2f} | {res['rss_mb']:<8.1f}")

    print("-" * 78)
    print(f"Average TTFT: {sum(ttfts)/len(ttfts):.1f} ms  |  Average Decode Speed: {sum(speeds)/len(speeds):.2f} tok/s")
    print(f"Working RAM: {get_memory_rss_mb():.1f} MB (Budget: < 200 MB)")
    print("=" * 78 + "\n")


def interactive_repl(engine: NioMantaOnnxEngine):
    print("\n" + "=" * 70)
    print("💬 NioManta ONNX Companion REPL (Type 'exit' or 'quit' to leave)")
    print(f"Model: {engine.model_path.name} | RAM: {get_memory_rss_mb():.1f} MB")
    print("=" * 70)

    while True:
        try:
            prompt = input("\n\033[1;36mYou:\033[0m ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit", "q"]:
                print("Exiting NioManta REPL.")
                break

            is_plan = prompt.lower().startswith("plan ") or prompt.lower().startswith("migrate ") or "--plan" in prompt
            prompt_clean = prompt.replace("--plan", "").strip()

            print("\033[1;32mNioManta:\033[0m ", end="", flush=True)

            res = engine.generate(
                prompt_clean,
                max_new_tokens=64,
                is_plan=is_plan,
                stream_callback=lambda token: print(token, end="", flush=True)
            )
            print()

            plan = parse_manta_plan(res["text"])
            if plan:
                print(f"\n📋 \033[1mStructured Plan [{plan['id']}]:\033[0m")
                for s in plan["steps"]:
                    print(f"  • Step {s['id']} [{s['priority']}]: {s['description']}")
                for r in plan["risks"]:
                    print(f"  ⚠️ Risk [{r['level']}]: {r['message']}")

            print(f"\n\033[2m[{res['generated_tokens']} tokens in {res['total_time_s']:.2f}s | TTFT: {res['ttft_ms']:.1f}ms | {res['tokens_per_second']:.1f} tok/s | RSS: {res['rss_mb']:.1f}MB]\033[0m")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting NioManta REPL.")
            break


def main():
    default_model, default_tok = find_default_paths()

    parser = argparse.ArgumentParser(description="NioManta ONNX Runner & Test Harness")
    parser.add_argument("--model", "-m", default=str(default_model) if default_model else None, help="Path to niomanta_int8.onnx")
    parser.add_argument("--tokenizer", "-t", default=str(default_tok) if default_tok else None, help="Path to nio-manta-tokenizer.json")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="Prompt to execute")
    parser.add_argument("--plan", action="store_true", help="Force <manta_plan> XML formatting")
    parser.add_argument("--tokens", "-n", type=int, default=48, help="Maximum generated tokens")
    parser.add_argument("--temp", "-T", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run automated benchmark suite")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive terminal companion REPL")
    parser.add_argument("--raw", action="store_true", help="Output raw text stream only (no headers or metrics)")

    args = parser.parse_args()

    if not args.model or not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("Expected model at 'models/niomanta_int8.onnx' or 'models/nio-manta.onnx'.")
        sys.exit(1)

    if not args.tokenizer or not Path(args.tokenizer).exists():
        print(f"❌ Tokenizer not found: {args.tokenizer}")
        sys.exit(1)

    if not args.raw:
        print(f"⚡ Loading NioManta ONNX: {Path(args.model).name} ({Path(args.model).stat().st_size / (1024**2):.1f} MB)...", flush=True)
    engine = NioMantaOnnxEngine(args.model, args.tokenizer)
    if not args.raw:
        print(f"✔ Engine ready! Initial resident RAM: {get_memory_rss_mb():.1f} MB", flush=True)

    if args.benchmark:
        run_benchmark(engine)
        return

    if args.interactive:
        interactive_repl(engine)
        return

    prompt = args.prompt or "Explain the difference between Arc and Rc in Rust in one sentence."
    if not args.raw:
        print(f"\nUser Query: {prompt}")
        print("NioManta: ", end="", flush=True)

    res = engine.generate(
        prompt,
        max_new_tokens=args.tokens,
        temperature=args.temp,
        is_plan=args.plan,
        stream_callback=lambda tok: print(tok, end="", flush=True)
    )
    if not args.raw:
        print()

    if not args.raw:
        plan = parse_manta_plan(res["text"])
        if plan:
            print(f"\n📋 Structured Plan [{plan['id']}]:")
            for s in plan["steps"]:
                print(f"  • Step {s['id']} [{s['priority']}]: {s['description']}")
            for r in plan["risks"]:
                print(f"  ⚠️ Risk [{r['level']}]: {r['message']}")

        print("\n" + "-" * 60)
        print(f"Metrics: {res['generated_tokens']} tokens | TTFT: {res['ttft_ms']:.1f} ms | Decode: {res['tokens_per_second']:.2f} tok/s | RSS: {res['rss_mb']:.1f} MB")
        print("-" * 60)


if __name__ == "__main__":
    main()
