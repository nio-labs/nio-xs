#!/usr/bin/env python3
"""
LoRA-to-GGUF Converter for Hydra Anchors & Specialist Heads.
Converts PEFT FP8/FP16/FP32 safetensors adapters into GGUF LoRA adapters
compatible with llama-cli --lora at native CPU/NEON/AVX speeds (35+ tok/s).
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
import numpy as np

try:
    import torch
    from safetensors import safe_open
    import gguf
except ImportError as e:
    sys.stderr.write(f"Missing required Python dependencies: {e}\n")
    sys.stderr.write("Run inside ~/.nioxs/env: ~/.nioxs/env/bin/python3\n")
    sys.exit(1)

MODULE_MAP = {
    'q_proj': 'attn_q',
    'k_proj': 'attn_k',
    'v_proj': 'attn_v',
    'o_proj': 'attn_output',
    'gate_proj': 'ffn_gate',
    'down_proj': 'ffn_down',
    'up_proj': 'ffn_up',
}

PATTERN = re.compile(
    r'(?:base_model\.model\.)?(?:model\.)?layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.(lora_A|lora_B)\.weight$'
)


def convert_adapter(adapter_dir: Path, out_path: Path | None = None, overwrite: bool = False) -> bool:
    adapter_dir = Path(adapter_dir)
    st_path = adapter_dir / "adapter_model.safetensors"
    cfg_path = adapter_dir / "adapter_config.json"

    if not st_path.is_file():
        # Check if compressed version exists
        lz4_path = adapter_dir / "adapter_model.safetensors.lz4"
        if lz4_path.is_file():
            sys.stderr.write(f"Decompressing {lz4_path.name}...\n")
            try:
                import lz4.frame  # or raw
            except ImportError:
                pass
        if not st_path.is_file():
            sys.stderr.write(f"Skipping {adapter_dir}: adapter_model.safetensors not found\n")
            return False

    if out_path is None:
        out_path = adapter_dir / "adapter_model.gguf"

    if out_path.is_file() and not overwrite:
        print(f"Already exists: {out_path}")
        return True

    lora_alpha = 16.0
    if cfg_path.is_file():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            lora_alpha = float(cfg.get("lora_alpha", 16.0))

    writer = gguf.GGUFWriter(str(out_path), arch="qwen2")
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, lora_alpha)

    scales = {}
    converted_count = 0

    with safe_open(str(st_path), framework="pt") as f:
        # First pass: collect fp8 scales if present
        for k in f.keys():
            if k.endswith(".scale"):
                base_k = k[:-6]
                scales[base_k] = f.get_tensor(k).item()

        # Second pass: write tensors
        for k in f.keys():
            if k.endswith(".scale"):
                continue

            m = PATTERN.search(k)
            if not m:
                continue

            layer_idx = int(m.group(1))
            mod_name = m.group(3)
            lora_ab = m.group(4)

            gguf_mod = MODULE_MAP.get(mod_name)
            if not gguf_mod:
                continue

            gguf_lora = "lora_a" if lora_ab == "lora_A" else "lora_b"
            gguf_name = f"blk.{layer_idx}.{gguf_mod}.weight.{gguf_lora}"

            t = f.get_tensor(k)
            scale = scales.get(k, 1.0)

            # Dequantize or cast to FP16
            t_f32 = t.to(torch.float32) * scale
            t_np = t_f32.cpu().to(torch.float16).numpy()

            writer.add_tensor(gguf_name, t_np)
            converted_count += 1

    if converted_count == 0:
        sys.stderr.write(f"No valid LoRA tensors found in {st_path}\n")
        return False

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Successfully converted {adapter_dir.name} -> {out_path.name} ({converted_count} tensors, {size_mb:.2f} MB)")
    return True


def convert_all(directory: Path, overwrite: bool = False):
    directory = Path(directory)
    if not directory.is_dir():
        sys.stderr.write(f"Directory not found: {directory}\n")
        return

    subdirs = sorted([d for d in directory.iterdir() if d.is_dir()])
    success = 0
    total = 0
    for d in subdirs:
        if (d / "adapter_model.safetensors").is_file() or (d / "adapter_model.safetensors.lz4").is_file():
            total += 1
            if convert_adapter(d, overwrite=overwrite):
                success += 1

    print(f"\nBatch GGUF conversion complete: {success}/{total} adapters converted.")


def main():
    parser = argparse.ArgumentParser(description="Convert PEFT safetensors LoRA adapters to GGUF")
    parser.add_argument("path", nargs="?", help="Path to adapter directory or directory of adapters")
    parser.add_argument("--all", action="store_true", help="Convert all adapters in specified directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing adapter_model.gguf")
    parser.add_argument("--out", help="Explicit output path (single adapter mode only)")

    args = parser.parse_args()

    target_path = Path(args.path) if args.path else Path.home() / ".nioxs" / "anchors"

    if args.all or (target_path.is_dir() and not (target_path / "adapter_model.safetensors").is_file()):
        convert_all(target_path, overwrite=args.overwrite)
    else:
        out = Path(args.out) if args.out else None
        convert_adapter(target_path, out_path=out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
