"""
NioRay-v2 Distillation & ONNX Export Pipeline
Designed to run locally or in Google Colab (Free T4 GPU).
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset import generate_dataset
from model import NioRayV2MicroLM, export_to_onnx, quantize_onnx

class ReflexDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_len=128):
        self.samples = []
        for item in data_list:
            full_text = item["input"] + item["target"]
            enc = tokenizer(full_text, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"].squeeze(0)
            
            # Mask out input prompt for loss calculation
            input_enc = tokenizer(item["input"], truncation=True, max_length=max_len)
            prompt_len = len(input_enc["input_ids"])
            labels = input_ids.clone()
            labels[:prompt_len] = -100  # ignore prompt loss
            
            self.samples.append((input_ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def train_and_export(
    teacher_name="Qwen/Qwen2.5-Coder-0.5B",
    num_layers=4,
    epochs=3,
    lr=5e-5,
    batch_size=8,
    output_dir="./output_nioray_v2"
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading Tokenizer & Teacher ({teacher_name})...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    ).to(device)
    teacher.eval()

    print(f"Initializing NioRay-v2 Student ({num_layers} layers derived from Qwen)...")
    student = NioRayV2MicroLM(teacher, num_layers=num_layers).to(device)
    student.train()

    raw_data = generate_dataset(output_path=os.path.join(output_dir, "dataset.jsonl"), samples_per_task=25)
    train_ds = ReflexDataset(raw_data, tokenizer)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    mse_loss_fn = nn.MSELoss()

    print("Starting Distillation Training...")
    for epoch in range(epochs):
        total_loss = 0.0
        for step, (input_ids, labels) in enumerate(loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(input_ids).logits

            student_logits = student(input_ids)

            # Combined Task Loss + Knowledge Distillation Logit Matching
            ce_loss = ce_loss_fn(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
            kd_loss = mse_loss_fn(student_logits, teacher_logits)
            loss = ce_loss + 0.5 * kd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.4f}")

    print("✔ Distillation training complete!")

    # Export to ONNX (CPU mode for clean serialization)
    student.to("cpu")
    onnx_path = os.path.join(output_dir, "nioray_v2.onnx")
    quant_path = os.path.join(output_dir, "nioray_v2_int8.onnx")
    
    export_to_onnx(student, tokenizer, onnx_path=onnx_path)
    quantize_onnx(input_path=onnx_path, output_path=quant_path)

    print(f"\n=======================================================")
    print(f"  NioRay-v2 MicroLM Artifacts Ready in {output_dir}")
    print(f"  • ONNX FP32: {onnx_path}")
    print(f"  • ONNX INT8: {quant_path}")
    print(f"=======================================================")

if __name__ == "__main__":
    train_and_export()
