# NioRay-v2 PureML MicroLM (ONNX + Google Colab POC)

**NioRay-v2** is a sub-second, ultra-compact code reflex MicroLM distilled directly from **Qwen2.5-Coder-0.5B**.

Instead of executing 24 layers autoregressively token-by-token (which takes 7s–15s on CPU), NioRay-v2 runs a **4-layer non-autoregressive sequence pass** in **< 0.1s (< 100ms)** with **< 30 MB resident RAM**.

---

## Architecture Overview

```
Teacher: Qwen2.5-Coder-0.5B (24 Layers, 490M params, GGUF)
                    │
                    ▼ Knowledge Distillation on 20 Developer Tasks
Student: NioRay-v2 (4 Layers, ~45M params)
                    │
                    ▼ Export & Dynamic INT8 Quantization
Artifact: nioray_v2_int8.onnx (~35 MB file, < 30 MB resident RAM)
```

---

## How to Train via Google Colab (Free T4 GPU)

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File -> Upload notebook** and select:
   `training/nioray-v2/NioRay_v2_Colab_Training.ipynb`
3. In Colab, go to **Runtime -> Change runtime type** and ensure **T4 GPU** is selected.
4. Click **Runtime -> Run all**.
5. The notebook will:
   * Load Qwen2.5-Coder-0.5B.
   * Train the 4-layer student model via logit matching on the 20 keystroke reflex tasks.
   * Export `nioray_v2.onnx` and quantize it to `nioray_v2_int8.onnx`.
   * Benchmark the CPU latency (shows **~35ms – 70ms** single-shot latency).
   * Automatically prompt you to download `nioray_v2_int8.onnx`.

---

## Local Pipeline Files

| File | Purpose |
| :--- | :--- |
| [`NioRay_v2_Colab_Training.ipynb`](file:///home/mn/nio-labs/nio-xs/training/nioray-v2/NioRay_v2_Colab_Training.ipynb) | Complete 1-click Google Colab notebook |
| [`dataset.py`](file:///home/mn/nio-labs/nio-xs/training/nioray-v2/dataset.py) | 20-task developer keystroke reflex dataset generator |
| [`model.py`](file:///home/mn/nio-labs/nio-xs/training/nioray-v2/model.py) | PyTorch `NioRayV2MicroLM` 4-layer sliced architecture + ONNX exporter |
| [`train_distill.py`](file:///home/mn/nio-labs/nio-xs/training/nioray-v2/train_distill.py) | Distillation training and dynamic INT8 quantization pipeline |
| [`infer_onnx.py`](file:///home/mn/nio-labs/nio-xs/training/nioray-v2/infer_onnx.py) | Standalone CPU inference runner verifying sub-0.1s latency |

---

## Integrating into NioXS

The resulting `nioray_v2_int8.onnx` can be placed into `~/.nioxs/models/` and executed via `tract-onnx` or `ort` in pure Rust with zero PyTorch dependencies.
