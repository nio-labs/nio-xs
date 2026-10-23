# NioManta: Teacher-Student Distillation Architecture (< 100 MB MicroLM)

**Author:** Nio Engineering Team  
**Status:** Architecture Blueprint & Implementation Guide  
**Target Artifact:** `nio-manta-ast.Q4_0.gguf`  
**Target Hardware:** Consumer Terminal / Embedded CPU (< 100 MB Disk, < 50 MB RAM, < 1.5s Latency)

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Bottleneck of Standard Autoregressive LLMs
Modern small language models (such as `Qwen2.5-Coder-0.5B` at 24 layers) deliver impressive reasoning, but running them on terminal CPUs presents a severe tradeoff:
- **Sequential Memory Bandwidth Starvation:** CPU autoregression must sequentially fetch all 24 layers of weights across system RAM for every generated token. On a standard laptop CPU running at ~2 tokens/second, a typical 30-token developer response takes **15–20 seconds**.
- **Memory Footprint:** Full 0.5B models consume **280 MB to 380 MB** on disk and demand **180 MB to 300 MB** of active RAM, violating the strict sovereign constraint of terminal tools remaining lightweight background companions.

```
Standard Autoregressive (24 Layers):
[Token N] ──> [L0] ──> [L1] ──> ... ──> [L23] ──> Latency: ~750ms/token (~18s total)
Memory Footprint: ~350 MB on disk | ~220 MB RAM

NioManta Distilled Student (Shallow MicroLM):
[Token N] ──> [L0] ──> [L1] ──> [L2] ... ──> Latency: ~40ms/token (< 1.5s total)
Memory Footprint: < 100 MB on disk | < 50 MB RAM
```

### 1.2 The NioManta Solution
**NioManta** solves this bottleneck through **Knowledge Distillation from an Enterprise Teacher**:
1. **Teacher:** `Qwen/Qwen2.5-Coder-7B-Instruct` provides deep programming synthesis, high-accuracy AST understanding, and architectural foresight.
2. **Student:** A surgically depth-sliced Transformer ($d=896$, 2 to 6 layers) trained specifically to mimic the probability distribution and structured reasoning of the 7B teacher.
3. **Artifact:** A 32-byte SIMD block-aligned `Q4_0 GGUF` file strictly bounded **under 100 MB**, executing via `AstMicroEngine` with zero-copy `mmap` and page-level memory reclamation.

---

## 2. Architectural Blueprint

```mermaid
flowchart TD
    subgraph Teacher["Teacher Model (Training VM / Cloud T4)"]
        T_Base["Qwen2.5-Coder-7B-Instruct (7 Billion Params)"]
        T_Quant["4-bit NF4 Quantization via bitsandbytes (~4.8 GB VRAM)"]
        T_Logits["Teacher Output Distribution (Logits z_t)"]
        T_Base --> T_Quant --> T_Logits
    end

    subgraph Data["Companion & Reflex Curriculum"]
        D1["Conversational Diagnostics & Explanations"]
        D2["Structured Plan Generation (<manta_plan>, <manta_step>)"]
        D3["Developer AST Reflexes (Borrow Checker, Regex, Refactor)"]
    end

    subgraph Student["Student Model (NioManta MicroLM)"]
        S_Init["Qwen2.5-Coder Base ($d_{model}=896$)"]
        S_Slice["Depth Slicing: 24L ➔ 2-6 Layers (~90M Params)"]
        S_Logits["Student Output Distribution (Logits z_s)"]
        S_Init --> S_Slice --> S_Logits
    end

    subgraph Loss["Distillation Objective"]
        L_CE["Task Cross-Entropy Loss"]
        L_KL["KL Divergence Loss (Logit Matching)"]
        Loss_Total["Total Loss = L_CE + 0.5 * τ² * L_KL"]
    end

    subgraph Target["Runtime Artifact (< 100 MB)"]
        Export["GGUF Conversion (llama.cpp)"]
        Quant["32-Byte Block-Aligned Q4_0 Quantization"]
        Final["nio-manta-ast.Q4_0.gguf (< 100 MB, < 50 MB RAM)"]
        Export --> Quant --> Final
    end

    T_Logits --> L_KL
    S_Logits --> L_KL
    S_Logits --> L_CE
    Data --> Teacher
    Data --> Student
    L_CE --> Loss_Total
    L_KL --> Loss_Total
    Loss_Total -->|Backpropagation (AdamW)| S_Slice
    S_Slice --> Export
```

---

## 3. Component Specifications

### 3.1 Teacher Model Specification
- **Identifier:** `Qwen/Qwen2.5-Coder-7B-Instruct`
- **Parameter Count:** 7.61 Billion parameters
- **Precision:** 4-bit NormalFloat4 (NF4) with double quantization
- **VRAM Footprint:** **~4.8 GB** (comfortably fits on consumer RTX 3060 or free Google Colab Tesla T4 with 15 GB VRAM)
- **Role:** Generates continuous soft probabilities over developer tokens, capturing nuance, syntax correctness, and structural intent that hard ground-truth labels omit.

### 3.2 Student Model Specification (NioManta)
- **Base Architecture:** Sliced Qwen2 Transformer
- **Hidden Dimension ($d_{model}$):** `896` (cleanly divisible by 32: $896 / 32 = 28$ blocks, achieving 100% vector SIMD alignment in Q4_0)
- **Attention Heads:** 14 Query Heads, 2 Key-Value Heads (Grouped-Query Attention)
- **Intermediate Dimension:** 4,864 (SwiGLU)
- **Layer Depth Options:**
  - **Option A (2 Layers, Full 152k Vocab):**  
    - Embedding Table ($151,936 \times 896 \times 0.5\text{ B}$): $\sim 68$ MB  
    - 2 Transformer Layers ($2 \times 12\text{ MB}$): $\sim 24$ MB  
    - Total Q4_0 Model Size: **$\approx 92$ MB** ($\mathbf{< 100\text{ MB}}$ target achieved)  
    - Latency: **$< 0.04$s per token** ($< 1.2$s full response on CPU)
  - **Option B (6-8 Layers, Compact 32k Lexicon):**  
    - Lexicon: Specialized 32,000 Developer & Shell tokens (as proven in NioViper)  
    - Embedding Table ($32,000 \times 896 \times 0.5\text{ B}$): $\sim 14.3$ MB  
    - 6 Transformer Layers ($6 \times 10.5\text{ MB}$): $\sim 63$ MB  
    - Total Q4_0 Model Size: **$\approx 78$ MB** ($\mathbf{< 100\text{ MB}}$ target achieved)  
    - Latency: **$< 0.08$s per token** ($< 2.0$s full response on CPU)

---

## 4. Distillation Formulation

The student is optimized using a dual-objective loss function combining hard ground-truth Cross-Entropy with soft logit knowledge distillation:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\hat{y}_s, y) + \alpha \cdot \tau^2 \cdot \mathcal{D}_{\text{KL}}\left(\sigma\left(\frac{z_s}{\tau}\right) \parallel \sigma\left(\frac{z_t}{\tau}\right)\right)$$

Where:
- $z_s$ and $z_t$ are the raw output logits of the student and teacher models respectively.
- $\tau$ is the distillation temperature ($\tau = 2.0$), which softens the teacher probability distribution over candidate syntax tokens.
- $\sigma(\cdot)$ is the softmax operation across the vocabulary dimension.
- $\alpha = 0.5$ balances imitation with exact task precision.
- **Vocabulary Alignment:** For tokens where vocabulary sizes differ (Teacher: 152,064 vs Student: 151,936 or 32,000), the loss is computed over the common subset $V_{\text{common}} = \min(|V_s|, |V_t|)$.

---

## 5. Domain Curriculum & Structured Formats

NioManta is trained on three core companion domains:

### 5.1 Conversational Engineering & Diagnostic Dialogue
Concise technical answers explaining systems programming nuances (Tokio async runtimes, Rust ownership and lifetime mechanics, zero-copy buffer strategies, OS page reclamation).

### 5.2 Structured Architectural Plans (`<manta_plan>`)
NioManta outputs machine-parseable plan structures enabling `nio` to execute multi-step CLI workflows with risk assessment:
```xml
<manta_plan id="graceful-shutdown">
  <manta_step id="1" priority="high">Listen for tokio::signal::ctrl_c()</manta_step>
  <manta_step id="2" priority="high">Broadcast cancel signal to workers via broadcast channel</manta_step>
  <manta_step id="3" priority="medium">Await active task drain with a 5-second timeout</manta_step>
  <manta_risk level="warning">Ensure worker channels are drained before aborting process</manta_risk>
</manta_plan>
```

### 5.3 AST & Keystroke Reflex Transformations
Instant single-line completions, syntax fixes, Rust match arms, regex formulations, and shell command triage executed sub-100ms.

---

## 6. Runtime Integration: AST-Micro Engine

Once trained and exported to `nio-manta-ast.Q4_0.gguf`, NioManta is executed natively inside `nioxs-engine` via `AstMicroEngine`:

### 6.1 Zero-Copy Streaming Pipeline
```
[mmap2::Mmap] ──> Virtual Address Space (~95 MB)
                        │
                        ▼ (Per-Token Forward Pass)
            Load Layer Weights on Demand
                        │
                        ▼
            libc::madvise(MADV_DONTNEED)  <── Releases physical pages immediately
                        │
                        ▼
       Active Resident Set (VmRSS): < 45 MB RAM
```

### 6.2 Key Runtime Optimizations
1. **Dynamic Block Count:** `AstMicroEngine` detects the layer depth dynamically from `qwen2.block_count` in GGUF metadata, transparently supporting 2L, 6L, or 16L models.
2. **Metadata Header Pruning:** Embedded tokenizer strings and merge tables are purged from memory immediately after initialization (`libc::malloc_trim(0)`).
3. **Chunked Phrase Buffer:** Generation uses token phrase grouping (flushing on sentence boundaries, punctuation, or 4-word thresholds) to deliver fluent, non-jittery terminal display.

---

## 7. Automated Training Pipeline (Google Colab CLI)

The distillation and quantization process is automated via `/home/mn/nio-labs/nio-xs/training/niomanta/train_niomanta_colab.py`.

### 7.1 Execution Steps
1. **Launch Cloud GPU Environment:**
   ```bash
   colab new --gpu T4 -s niomanta
   ```
2. **Install Remote Prerequisites:**
   ```bash
   colab install -s niomanta transformers accelerate bitsandbytes "numpy<2.3" gguf
   ```
3. **Execute Training & Export Pipeline:**
   ```bash
   colab exec -s niomanta -f /home/mn/nio-labs/nio-xs/training/niomanta/train_niomanta_colab.py --timeout 1200
   ```
4. **Retrieve Generated Artifact:**
   ```bash
   colab download -s niomanta nio-manta-ast.Q4_0.gguf -o ~/.nioxs/models/
   ```
5. **Teardown Instance:**
   ```bash
   colab stop -s niomanta
   ```

---

## 8. Performance Benchmark Expectations

| Metric | Unquantized Base (0.5B-24L) | NioRay Lite (Q4_0-24L) | **NioManta (Distilled 2L/6L)** |
| :--- | :--- | :--- | :--- |
| **Model Size on Disk** | 988 MB (FP16) | 353 MB (Q4_0) | **$\mathbf{78\text{ – }92}$ MB (Q4_0)** |
| **Peak RAM (VmRSS)** | ~550 MB | ~81 MB | **$\mathbf{< 45}$ MB** |
| **Time to First Token (TTFT)** | ~0.05s | ~0.002s | **$\mathbf{< 0.001}$s** |
| **Generation Latency (30 tokens)** | ~18.5s | ~15.2s | **$\mathbf{0.8\text{ – }1.4}$s** |
| **Task Speedup Factor** | $1.0\times$ | $1.2\times$ | **$\mathbf{12.0\times\text{ faster}}$** |
| **Target Hardware** | High-end CPU | Modern CPU | **Any Terminal / Edge CPU** |
