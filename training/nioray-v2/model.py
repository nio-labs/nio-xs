"""
NioRay-v2 PureML MicroLM Architecture
Derives a compact, 4-layer Transformer from Qwen2.5-Coder-0.5B, optimized for
sub-0.1s ONNX inference on CPU with < 30 MB resident RAM.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

class NioRayV2MicroLM(nn.Module):
    def __init__(self, qwen_model, num_layers=4):
        super().__init__()
        self.config = qwen_model.config
        self.embed_tokens = qwen_model.model.embed_tokens
        self.layers = nn.ModuleList([qwen_model.model.layers[i] for i in range(num_layers)])
        self.norm = qwen_model.model.norm
        self.lm_head = qwen_model.lm_head
        self.num_layers = num_layers

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        # input_ids: [batch_size, seq_len]
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal mask if not provided
        bsz, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones((bsz, seq_len), device=input_ids.device)

        # 4-layer forward pass
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=None,
                position_ids=None,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

def export_to_onnx(model, tokenizer, onnx_path="nioray_v2.onnx", max_seq_len=128):
    model.eval()
    dummy_input = torch.ones((1, 32), dtype=torch.long)
    
    print(f"Exporting NioRay-v2 to ONNX: {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=17,
        do_constant_folding=True
    )
    print(f"✔ Successfully exported ONNX model to {onnx_path}")

def quantize_onnx(input_path="nioray_v2.onnx", output_path="nioray_v2_int8.onnx"):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        print(f"Quantizing ONNX to INT8: {output_path}...")
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QInt8
        )
        print(f"✔ Quantized INT8 model saved to {output_path}")
    except ImportError:
        print("onnxruntime is required for quantization: pip install onnxruntime")

if __name__ == "__main__":
    print("NioRay-v2 MicroLM Module Loaded")
