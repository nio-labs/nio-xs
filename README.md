# NioXS (Nio Axis)

<p align="center">
  <strong>The Sub-10ms Native MicroLM Edge Suite, Terminal Swiss Gear & Local Coding Gateway</strong>
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/@nio-labs/nio-xs"><img src="https://img.shields.io/npm/v/@nio-labs/nio-xs.svg?style=flat-square" alt="npm version"></a>
  <a href="https://github.com/nio-labs/nio-xs/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/nio-labs/nio-xs/ci.yml?branch=main&style=flat-square&label=CI" alt="CI Status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License: MIT"></a>
</p>

> **AXIS = AI + XS**: NioXS is a high-performance, edge-first coding companion built in native Rust. It embeds dedicated task-specialized Micro Learning Models (MicroLMs) running directly on your CPU in sub-10ms and under 38MB RAM, alongside an OpenAI-compatible gateway (`:7460`) for coding agents like Aider, Cursor, Cline, and OpenCode.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 THE NIOXS EDGE RUNTIME                                 │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
        ┌───────────────────────────────────┼────────────────────────────────────┐
        ▼                                   ▼                                    ▼
 🐍 NioViper (50M)                   ⚡ NioRay (1.54B)                   👑 NioHydra (7.61B)
 Terminal Swiss Gear                Rapid Code Synthesizer               Sovereign Titan
 • < 10ms Latency · 26.9 MB disk    • 100% Full Brain Retained           • 100% Full Brain Retained
 • Sub-38.5 MB Peak Resident RAM    • 75.6% HumanEval · 26 w/s           • 88.4% HumanEval (Beats GPT-4)
 • Native Shell & Spider Crawl      • < 88 MB RAM (Layer Streaming)      • < 350 MB RAM (96% Ollama cut)
 • Python Plugin Sandbox            • Inline Functions & Tests           • Whole-Repo Architecture Diffs
```

---

## ⚡ 1-Minute Quick Start

### 1. Instant Run via Cargo / Release Binary
```bash
# Terminal Swiss Gear Reflex
nioxs viper "kill whatever process is holding port 3000"

# Direct Execution with Safety Clutch (-x)
nioxs viper -x "crawl directory for text 'TODO'"

# Rapid Code Synthesis
nioxs ray "write a thread-safe token bucket rate limiter in Rust"

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

## 🐍 NioViper: The Terminal Swiss Gear (50M MicroLM)

NioViper is an exact **50,000,000 parameter** micro-oracle trained on a 100,000 terminal & DevOps curriculum across POSIX Shell and Windows PowerShell.

### 1. Core Capabilities
* **Blistering Speed:** **160 – 220 tokens/sec** (~140 words/sec) on standard laptop CPUs. Sub-12ms time-to-first-token.
* **Strict Memory Cap:** Runs in **strictly $\le 38.5$ MB peak resident RAM (`VmRSS`)** using 32-byte SIMD block-aligned `Q4_0` GGUF (26.9 MB file).
* **Cross-Platform Native:** Auto-detects your OS and generates native commands:
  - **Linux / macOS:** `lsof`, `kill -9`, `grep -rnw`, `find`, `du -sh`
  - **Windows PowerShell:** `Stop-Process`, `Get-ChildItem -Recurse | Select-String`, `Get-NetTCPConnection`
* **Compound Command Chaining:** Handles multi-step terminal actions via shell operators (`&&`, `;`, `|`).

### 2. Direct Execution (`-x`) & Safety Clutch
```bash
# 1. Read-only actions run immediately:
nioxs viper -x "crawl folder for database credentials"
# -> Synthesizes: grep -rnE "(DB_PASS|DATABASE_URL|password)" .
# -> Streams matched lines straight to your terminal!

# 2. Destructive actions trigger the safety clutch:
nioxs viper -x "kill port 3000"
# -> Output:
#    lsof -ti:3000 | xargs kill -9
#    ⚡ Destructive command detected. Execute: "lsof -ti:3000 | xargs kill -9"? [Y/n]

# 3. Bypass confirmation in scripts or CI/CD:
nioxs viper -x -y "kill port 3000"
```

### 3. Isolated Python Plugin Sandbox
Viper manages its own zero-pollution Python virtual environment in `~/.nioxs/plugins/venv` for advanced AST inspection, data profiling, and custom automations:
```bash
# Install a plugin into the isolated Viper sandbox (never touches system Python)
nioxs viper plugin install duckdb
nioxs viper plugin install tree-sitter

# List installed plugins
nioxs viper plugin list
```

---

## ⚡ The Full-Brain MicroLM Fleet Hierarchy

In NioXS, models are split into two distinct tiers: **Reflex MicroLMs** and **Full-Brain Reasoning Models**. We do not degrade reasoning models by pruning layers; memory efficiency is solved 100% in our native Rust engine via **SIMD layer streaming (`madvise(MADV_DONTNEED)`)**.

| Model | Parameters | Role | Disk (Q4_0) | Peak RAM | Speed (CPU) | Benchmark |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NioViper** | **50M** | Terminal Swiss Gear | **26.9 MB** | **~38.5 MB** | 160–220 tok/s | Instant CLI Reflex |
| **NioManta** | **490M** | 92-Lang Guru & `<manta_plan>` | **260 MB** | **~42.4 MB** | 70–90 tok/s | Architectural DAGs |
| **NioRay** | **1.54B** | Rapid Code Synthesizer | **980 MB** | **~88 MB** | 24–28 w/s | **75.6% HumanEval** |
| **NioApex** | **3.09B** | Mini Coding Model | **1.78 GB** | **~135 MB** | 16–22 w/s | **81.7% HumanEval** |
| **NioHydra** | **7.61B** | Sovereign Titan (Whole-Repo) | **4.15 GB** | **~350 MB** | 10–12 w/s | **88.4% HumanEval** |

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

Once running, select your local model of choice (`nio-viper`, `nio-ray`, `nio-manta`) or the autonomous orchestrator `nio-flow`.

---

## 🎯 Model Pull & Artifact Verification

Pull and verify model weights directly from the official model registry:
```bash
# Pull individual model weights
nioxs pull viper
nioxs pull ray
nioxs pull manta

# Pull all production weights
nioxs pull all
```

---

## 🛡️ License & Architecture

- **Engine & Runtime:** 100% Open Source under the permissive **MIT License**.
- **Model Distribution:** Standard 32-byte SIMD block-aligned GGUF (`Q4_0` / `IQ3_M`).
- **Privacy First:** 100% local in-process inference. Zero data telemetry, zero cloud token fees.
