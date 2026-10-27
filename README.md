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
 🐍 Viper (100M)                     🦈 Skate (200M)                     🌊 Manta (494M)
 Tactical Terminal Striker           Ambient Desk Companion              Conversational Guru
 • Sub-20µs Latency · ~61.7 MB disk  • < 68 MB RAM · ~93.4 MB disk       • < 95 MB RAM · ~252 MB disk
 • < 27 MB Peak Resident RAM         • 50+ w/s (AVX2 SIMD)               • 21 NioGuru Personas
 • Destructive Safety Clutch         • Friendly Banter & Active Listen   • <manta_plan> Execution DAGs
 • Pure POSIX & PowerShell Pipes     • Zero Code Bloat                   • Multi-File Architecture
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

| Model | Parameters | Role | Disk (`IQ3_XXS`) | Peak RAM | Speed (CPU) | Cognitive Mode |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Viper** | **100M** | Tactical Terminal Striker | **~61.7 MB** | **< 27 MB** | Sub-20µs | Pure CLI Pipes |
| **Skate** | **200M** | Ambient Desk Companion | **~93.4 MB** | **< 68 MB** | 50–55 w/s | Friendly Banter & Focus |
| **µSail** | **270M** | Symbolic Invariant Reflex | **~120 MB** | **< 32 MB** | < 15 ms | Symbolic Safety Proofs |
| **Manta** | **494M** | Full Conversational Guru | **~252 MB** | **< 95 MB** | 35–45 w/s | `<manta_plan>` DAGs |
| **Sail** | **1.54B** | Lean CoT Reasoner | **~650 MB** | **< 95 MB** | 31–36 w/s | Dense `<think>` Logic |
| **Fin** | **Council** | Frontier Coding Genius | **~1.8GB – 28GB** | **~3.1 GB** | 30–90 t/s | Compiler-Verified Diff |
| **Gar** | **14.7B** | Encyclopedic Polymath | **~6.8 GB** | **< 850 MB** | 8–12 w/s | Universal Knowledge |

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
