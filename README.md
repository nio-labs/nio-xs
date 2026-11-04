# NioXS (Nio Axis)

<p align="center">
  <strong>The Sub-10ms Native MicroLM Edge Suite, Terminal Swiss Gear & Local Coding Gateway</strong>
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/@nio-labs/nio-xs"><img src="https://img.shields.io/npm/v/@nio-labs/nio-xs.svg?style=flat-square" alt="npm version"></a>
  <a href="https://github.com/nio-labs/nio-xs/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/nio-labs/nio-xs/ci.yml?branch=main&style=flat-square&label=CI" alt="CI Status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License: MIT"></a>
</p>

> **AXIS = AI + XS**: NioXS is a high-performance, edge-first coding companion built in native Rust. It embeds dedicated task-specialized Micro Learning Models (MicroLMs) running directly on your CPU in sub-10ms and under 28MB RAM, alongside an OpenAI-compatible gateway (`:7460`) for coding agents like Aider, Cursor, Cline, and OpenCode.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 THE NIOXS EDGE RUNTIME                                 │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
        ┌───────────────────────────────────┼────────────────────────────────────┐
        ▼                                   ▼                                    ▼
 🐍 Viper (100M)                     🦈 Skate (135M)                     🌊 Manta (494M)
 Tactical Terminal Striker           Ambient Desk Companion              Conversational Guru
 • Sub-20µs Latency · ~61.7 MB disk  • < 68 MB RAM · ~82.9 MB disk       • < 240 MB Peak RAM · ~317.9 MB disk
 • < 27 MB Peak Resident RAM         • 55+ w/s (66.5 tok/s AVX2)         • < 80 MB Idle Reclaim
 • Destructive Safety Clutch         • Friendly Banter & Active Listen   • <manta_plan> Execution DAGs
 • Pure POSIX & PowerShell Pipes     • Zero Code Bloat (Masked)          • Architectural Synthesis
```

---

## ⚡ 1-Minute Quick Start

### 1. Instant Run via Cargo / Release Binary
```bash
# Terminal Tactical Reflex
nioxs viper "kill whatever process is holding port 3000"

# Direct Execution with Safety Clutch (-x)
nioxs viper -x "crawl directory for text 'TODO'"

# Ambient Companion Query
nioxs skate "Let's plan out our focus sprint for the morning"

# Conversational Guru & Architecture Planning
nioxs manta --plan "Add zero-downtime NOT NULL column to postgres table"

# Start the OpenAI-Compatible Local Gateway
nioxs serve
```

### 2. Universal Install Script (Linux, macOS, WSL)
```bash
curl -fsSL https://raw.githubusercontent.com/nio-labs/nio-xs/main/install.sh | bash
```

### 3. Native Windows (PowerShell)
```powershell
irm https://raw.githubusercontent.com/nio-labs/nio-xs/main/install.ps1 | iex
```

---

## 🐍 Viper: The Tactical Terminal Striker (100M MicroLM)

Viper is a locked **100,000,000 parameter** reflex model trained on a curated terminal & DevOps curriculum across POSIX Shell and Windows PowerShell.

### 1. Core Capabilities
* **Sub-20µs Sampling:** Evaluates candidate commands in microseconds on standard laptop CPUs.
* **Strict Memory Cap:** Runs in **strictly < 27 MB peak resident RAM (`VmHWM`)** using 32-byte SIMD block-aligned `IQ3_XXS` / `Q4_0` GGUF (~61.7 MB file).
* **Cross-Platform Native:** Auto-detects your OS and generates native commands:
  - **Linux / macOS:** `lsof`, `kill -9`, `grep -rnw`, `find`, `du -sh`
  - **Windows PowerShell:** `Stop-Process`, `Get-ChildItem -Recurse | Select-String`, `Get-NetTCPConnection`
* **Compound Command Chaining:** Handles multi-step terminal actions via shell operators (`&&`, `;`, `|`).

### 2. Direct Execution (`-x`) & Safety Clutch
```bash
# 1. Read-only actions run immediately:
nioxs viper -x "crawl folder for database credentials"

# 2. Destructive actions trigger the safety clutch:
nioxs viper -x "kill port 3000"
# -> Output:
#    lsof -ti:3000 | xargs kill -9
#    ⚡ Destructive command detected. Execute: "lsof -ti:3000 | xargs kill -9"? [Y/n]

# 3. Bypass confirmation in scripts or CI/CD:
nioxs viper -x -y "kill port 3000"
```

---

## ⚡ The Production Predator Hierarchy

| Model | Parameters | Role | Disk Format | Net Model RAM | Process Peak (`VmHWM`) | Speed (CPU) | Cognitive Mode |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Viper** | **100M** | Tactical Terminal Striker | **~61.7 MB** (`Q4_0`) | **< 27 MB** | **~124 MB** | Sub-20µs | Pure CLI Pipes |
| **Dolphin** | **135M** | Universal Engram Memory (MCP) | **~75 MB** (`IQ4_NL`) | **< 28 MB** | **~125 MB** | Sub-5ms recall | Hebbian Fast Weights & NioDB |
| **Skate** | **135M** | Ambient Desk Companion | **~75 MB** (`IQ4_NL`) | **< 28 MB** | **~125 MB** | 55+ w/s | Friendly Banter & Zero-Code Focus |
| **Mako** | **150M** | Live Web Intelligence & Scout | **~65.4 MB** (`IQ3_XXS`) | **< 22 MB** | **~120 MB** | 200+ tok/s | DOM Noise Stripping & Wire DSL |
| **Manta** | **494M** | Systems Guru & DAG Architect | **~254 MB** (`Q4_0`) | **~95 MB** | **~195 MB** | 20–35 w/s | `<manta_plan>` DAGs & Wasm Engine |
| **Beluga** | **1.54B** | Sovereign Lead Architect | **~604 MB** (`IQ3_XXS`) | **~76 MB** | **~295 MB** | 14–22 tok/s | Deep `<think>` CoT Invariant Proofs |

---

## 🧬 The v-µLM (V-MicroLM) Architecture: Reflex, Vector & Companion

Nio's proprietary **v-µLM (V-MicroLM)** architecture eliminates reliance on brittle synthetic dataset fine-tuning by organizing execution across three complementary mathematical pillars:

| Pillar | Role | Key Models | Operational SLA |
| :--- | :--- | :--- | :--- |
| **`v-µLM Reflex`** | Instant Grammar & In-Process Decision Engine | **Nano (~0.5M)**, **Mask (1.1M)**, **Micro (5.0M)**, **Mini (7.1M)**, **Viper (100M)** | `< 28 MB RAM`, sub-0.1ms to sub-20µs sampling |
| **`v-µLM Vector`** | Vector Codebook Lattice & Web Scout | **Mako (150M)** & **µCuda / Cuda (3.0B–6.0B)** | `92 MB disk`, >0.999 cosine fidelity, wire scraping |
| **`v-µLM Companion`** | Sovereign Edge Companion Fleet | **Skate (135M)**, **Manta (494M)** & **Beluga (1.54B)** | `< 50 MB to < 295 MB peak RAM`, pure CPU, zero GPU |

---

## 🤝 State of our v-MicroLM Companion Fleet (Skate, Manta & Beluga)

The **v-MicroLM Companion Fleet** represents Nio's tiered, zero-cloud edge intelligence architecture. By pairing custom `v-MicroLM Vector` codebooks (`IQ3_XXS` / `IQ4_NL`) with dynamic layer streaming and lock-free AVX2 parallelism, each companion tier delivers uncompromising domain specialization strictly within ultralight memory envelopes:

- **Tier 1: Skate (135M)** — **Ambient Desk Companion**. Always-on background desk-mate running in `< 50 MB` active peak RAM with strict zero-code deflection.
- **Tier 2: Manta (494M)** — **Systems Guru & DAG Architect**. High-throughput architectural and RFC planner generating structured `<manta_plan>` execution DAGs in `< 235 MB` peak RAM.
- **Tier 3: Beluga (1.54B)** — **Sovereign Lead Architect**. Flagship cognitive companion delivering bit-exact `<think>` Chain-of-Thought reasoning proofs and system synthesis in `< 295 MB` peak RAM.

### 📊 v-MicroLM Companion Benchmark Suite

| Metric / Specification | 🛹 Skate | 🦈 Manta | 🐋 Beluga |
| :--- | :--- | :--- | :--- |
| **Model Parameters** | **135M** (134.5M) | **494M** | **1.54B** (28 Layers, 12 Heads) |
| **Primary Specialization** | Ambient Desk Companion & Edge Chat | Systems Guru & DAG Task Planner | Sovereign Lead Architect & CoT Prover |
| **Quantization & Format** | `IQ3_XXS` Vector / `IQ4_NL` | `IQ3_XXS` (imatrix conditioned) | `IQ3_XXS` Layers + `IQ4_NL` Vocab |
| **On-Disk Binary Size** | **~75 – 83 MB** | **~254 – 318 MB** | **604 MB** |
| **Idle Model RAM (Heap)** | **< 28 MB** (reclaims to 27.6 MB) | **~75 MB** (reclaims to 74.8 MB) | **~76 MB** |
| **Active Process Peak (`VmHWM`)**| **< 50 MB** (40.8 MB default) | **~195 – 234 MB** | **295.1 MB** *(vs 1,312 MB stock)* |
| **Inference Rate (Pure CPU)** | **35–42 tok/s** (55+ w/s default) | **20–35 words/sec** (~30–45 tok/s) | **21.58 tok/s** (14.4 words/sec) |
| **Peak Throughput / Burst** | **115–120.8 tok/s** (`--turbo`) | **~50 tok/s** burst | **39.7 tok/s** (27.4 w/s on spec hits) |
| **Cognitive Invariant** | Strict Zero-Code (deflects to Manta/Fin) | Strict Zero-Code Planner (`<manta_plan>`) | Bit-exact `<think>...</think>` CoT |
| **Target Hardware SLA** | Always-on background daemon / 128MB RAM | 512MB RAM edge nodes & containers | 512MB–1GB RAM systems (Zero GPU) |
| **CLI Direct Invocation** | `nioxs skate "<prompt>"` | `nioxs manta "<prompt>"` | `nioxs beluga "<prompt>"` |

---

## 🐋 Beluga vs Stock Qwen 2.5 (1.5B): Side-by-Side Comparison

| Technical Dimension | Stock Qwen 2.5 1.5B<br>*(Standard Ollama / llama.cpp)* | Beluga<br>*(Sovereign NioXS Engine)* |
| :--- | :--- | :--- |
| **Model Parameters** | 1.54 Billion (28 Layers, 12 Heads, 2 KV Heads) | 1.54 Billion (28 Layers, 12 Heads, 2 KV Heads) |
| **Vocabulary Coverage** | 151,936 BPE Tokens | 151,936 BPE Tokens (100% stock identical) |
| **Reasoning Engine** | `<think>...</think>` Chain-of-Thought | `<think>...</think>` Chain-of-Thought (100% bit-exact) |
| **Weight Quantization** | `Q4_K_M` (Standard 4.5-bit block quantization) | `IQ3_XXS` (Layers) + `IQ4_NL` (Vocab codebook) |
| **Disk Storage Footprint** | **1,020 MB (1.02 GB)** | **604 MB (0.60 GB)** *(41% smaller)* |
| **True Peak Process RAM (`VmHWM`)** | **1,312 MB (1.31 GB)** | **295.1 MB (0.29 GB)** *(4.4× lower RAM)* |
| **Minimum Hardware RAM Required** | **2 GB – 4 GB free RAM** *(OOMs below 2 GB)* | **< 350 MB free RAM** *(Runs on 512MB systems)* |
| **Runtime Engine** | GGML / `llama.cpp` runtime | Custom Sovereign `AST-Micro` Engine (Pure Rust) |
| **CPU Execution Strategy** | Standard Autoregressive (1 token per pass) | Speculative Trio + Asymmetric Zero-Penalty Verification |
| **CPU Parallelism** | OpenMP / standard thread pool | Lock-Free SpinThreadPool pinned to physical P-cores (`NUM_WORKERS = 6`) |
| **Memory Management** | Full GGUF file mmapped and held in physical RAM | Dynamic Layer Streaming with aggressive OS page eviction (`MADV_DONTNEED`) |
| **KV Cache Architecture** | Full static context buffer (112 MB for 4K context) | Rolling localized cache with zero-allocation reuse |
| **Measured Decode Rate (Pure CPU)** | **22.30 tok/s (16.06 words/sec)** | **21.58 tok/s (14.40 words/sec)** |
| **Peak Burst Rate** | Flat 22.3 tok/s (No multi-token burst) | Up to **39.7 tok/s (27.4 words/sec)** on subword hits |
| **Zero GPU Offload** | Supported, but requires > 1.5 GB RAM | Supported, guaranteed strictly < 350 MB RAM |

---

## 🔌 One-Click Agent Setup (`nioxs init`)

NioXS acts as a private, local OpenAI gateway (`http://127.0.0.1:7460/v1`). Automatically configure coding agents:

```bash
# Auto-configure all detected agents in the current project
nioxs init

# Or target specific agents
nioxs init --aider       # Writes .aider.conf.yml
nioxs init --cline       # Writes cline_mcp_settings.json
nioxs init --opencode    # Writes opencode.json
nioxs init --cursor      # Configures custom endpoint
```

---

## 🎯 Model Pull & Artifact Verification

Pull and verify model weights directly from the official model registry:
```bash
# Pull individual model weights
nioxs pull viper
nioxs pull skate
nioxs pull manta

# Pull all production weights
nioxs pull all
```

---

## 🛡️ License & Privacy

- **Engine & Runtime:** 100% Open Source under the permissive **MIT License**.
- **Model Distribution:** Standard 32-byte SIMD block-aligned GGUF (`IQ3_XXS` / `Q4_0`).
- **Privacy First:** 100% local in-process inference. Zero data telemetry, zero cloud token fees.
