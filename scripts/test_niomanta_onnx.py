#!/usr/bin/env python3
"""
⚡ NIOMANTA ONNX ENGINE INTEGRATION TESTS
Validates NioManta ONNX INT8 model and tokenizer execution for nio-xs edge deployments.
Parallels test harnesses in nio-mini and nio-micro.
"""

import sys
import time
from pathlib import Path
import numpy as np

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir / "scripts"))

try:
    import onnxruntime as ort
    from tokenizers import Tokenizer
    from run_niomanta_onnx import NioMantaOnnxEngine, find_default_paths
except ImportError as e:
    print(f"Skipping tests: missing dependency ({e})")
    sys.exit(0)


def test_model_artifacts_exist():
    model_path, tok_path = find_default_paths()
    assert model_path is not None and model_path.exists(), f"Missing NioManta ONNX model at {model_path}"
    assert tok_path is not None and tok_path.exists(), f"Missing Tokenizer JSON at {tok_path}"

    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✔ Artifacts verified: {model_path.name} ({file_size_mb:.2f} MiB)")


def test_onnx_graph_io():
    model_path, _ = find_default_paths()
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    assert len(inputs) == 1, f"Expected 1 input, got {len(inputs)}"
    assert inputs[0].name == "input_ids", f"Expected input 'input_ids', got {inputs[0].name}"

    assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"
    assert outputs[0].name == "logits", f"Expected output 'logits', got {outputs[0].name}"
    print("✔ ONNX Graph IO verified: input_ids -> logits")


def test_tokenizer_encoding():
    _, tok_path = find_default_paths()
    tok = Tokenizer.from_file(str(tok_path))

    vocab_size = tok.get_vocab_size()
    assert vocab_size >= 150000, f"Expected Qwen vocabulary (>150k), got {vocab_size}"

    text = "<|im_start|>system\nYou are NioManta.<|im_end|>"
    encoded = tok.encode(text)
    assert len(encoded.ids) > 0
    decoded = tok.decode(encoded.ids)
    assert "NioManta" in decoded
    print(f"✔ Tokenizer verified (Vocab size: {vocab_size:,})")


def test_prefill_and_generation():
    model_path, tok_path = find_default_paths()
    engine = NioMantaOnnxEngine(str(model_path), str(tok_path))

    prompt = "Explain Rust ownership"
    t0 = time.perf_counter()
    res = engine.generate(prompt, max_new_tokens=10)
    elapsed = time.perf_counter() - t0

    assert res["generated_tokens"] > 0, "No tokens generated"
    assert res["ttft_ms"] > 0, "Invalid TTFT measurement"
    print(f"✔ Generation verified: {res['generated_tokens']} tokens in {elapsed:.2f}s (TTFT: {res['ttft_ms']:.1f}ms)")


if __name__ == "__main__":
    print("=" * 60)
    print("Running NioManta ONNX Integration Tests...")
    print("=" * 60)
    test_model_artifacts_exist()
    test_onnx_graph_io()
    test_tokenizer_encoding()
    test_prefill_and_generation()
    print("=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
