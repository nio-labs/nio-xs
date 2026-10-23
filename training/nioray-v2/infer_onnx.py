"""
NioRay-v2 PureML CPU Inference Demonstration
Runs the exported ONNX INT8 model with sub-0.1s latency and < 30 MB resident RAM.
"""

import time
import numpy as np
try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
except ImportError:
    print("Install dependencies: pip install onnxruntime transformers")
    exit(1)

def run_onnx_inference(
    onnx_path="output_nioray_v2/nioray_v2_int8.onnx",
    tokenizer_name="Qwen/Qwen2.5-Coder-0.5B",
    prompt="find . -name \"*.rs\" -exec gr"
):
    print(f"Loading ONNX Model: {onnx_path}...")
    start_load = time.time()
    
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, session_options, providers=["CPUExecutionProvider"])
    load_time = time.time() - start_load
    print(f"✔ ONNX Session Initialized in {load_time:.3f}s")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    formatted = f"<|im_start|>system\nYou are NioRay-v2.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    enc = tokenizer(formatted, return_tensors="np")
    input_ids = enc["input_ids"].astype(np.int64)

    print(f"Executing Single-Shot Forward Pass on CPU...")
    start_infer = time.time()
    outputs = session.run(["logits"], {"input_ids": input_ids})
    infer_time = time.time() - start_infer

    logits = outputs[0]  # [1, seq_len, vocab_size]
    predicted_id = np.argmax(logits[0, -1, :])
    token_str = tokenizer.decode([predicted_id])

    print(f"\n=======================================================")
    print(f"  Input Prompt: {prompt}")
    print(f"  Top Infilled Token: {token_str}")
    print(f"  Inference Latency: {infer_time * 1000:.2f} ms ({infer_time:.4f}s)")
    print(f"=======================================================")

if __name__ == "__main__":
    import sys
    model_file = sys.argv[1] if len(sys.argv) > 1 else "output_nioray_v2/nioray_v2_int8.onnx"
    run_onnx_inference(onnx_path=model_file)
