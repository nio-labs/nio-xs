# NioXS (Nio Axis)

**The Lightweight Coding Companion & AI Gateway powered by NioNano MicroLM (Micro Learning Model).**

> **AXIS = AI + XS**: NioXS serves as the operational axis and in-TUI coding companion—providing OpenAI-compatible gateway routing, an in-session bash runner, Python tool plugins, compiler diagnostic triage, and anti-hallucination guardrails for coding agents.

```
                      ┌────────────────────────────────────────┐
                      │    Coding Agent / TUI / IDE of Choice  │
                      │  (Aider, OpenCode, Cline, Cursor, Roo) │
                      └───────────────────┬────────────────────┘
                                          │ OpenAI /v1/chat/completions
                                          ▼
                      ┌────────────────────────────────────────┐
                      │          NioXS Companion Gateway       │
                      │  • OpenAI-Compatible Proxy (:7460)     │
                      │  • Prefix Router (--bash, --tools, etc)│
                      │  • Anti-Hallucination Reality Check    │
                      └───────────────┬────────────────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
   ┌───────────────────────────┐             ┌───────────────────────────┐
   │  NioNano (MicroLM: 52 KB) │             │    Upstream LLM (Kilo)    │
   │  • Micro Learning Model   │             │  • Triggered via --coding │
   │  • <1ms CPU inference     │             │  • Kilo Auto Free / Mini  │
   │  • AST compiler triage    │             │  • Context-bundled from   │
   │  • Pure native scraping   │             │    local files & folders  │
   │  • UI label & style fixes │             │  • Guarded against ghost  │
   │  • Decision & safety gate │             │    symbols & fake paths   │
   └───────────────────────────┘             └───────────────────────────┘
```

NioNano acts as a **Micro Learning Model (µLM)**—not a bloated generative model, but a specialized, deterministic 52 KB model that assists the developer directly in their TUI. For heavy generation, NioXS acts as an intelligent middleman to upstream LLMs while validating generated code against the local project AST to prevent hallucinations.

---

## ⚡ 1-Minute Quick Start

### 1. Instant Run via NPX (Zero Install)
Runs on any machine with Node.js (Linux, macOS, Windows, Termux):
```bash
npx @nio-labs/nio-xs
```

### 2. Universal Install Script (Linux, macOS, Termux)
```bash
curl -fsSL https://raw.githubusercontent.com/nio-labs/nio-xs/main/install.sh | bash
```

### 3. Native Windows (PowerShell)
```powershell
irm https://raw.githubusercontent.com/nio-labs/nio-xs/main/install.ps1 | iex
```

### 4. Docker
```bash
docker run -d --name nioxs -p 7460:7460 ghcr.io/nio-labs/nioxs:latest
```

---

## 🔌 Plug into Your Favorite Coding Agent

Once NioXS is running on `http://127.0.0.1:7460/v1`, select `nio-nano` in your coding agent:

* **Aider**:
  ```bash
  aider --openai-api-base http://127.0.0.1:7460/v1 --openai-api-key none --model nio-nano
  ```
* **OpenCode / Cline / Continue / Cursor / Roo Code**:
  Set OpenAI API Base URL to `http://127.0.0.1:7460/v1` and Model ID to `nio-nano`.

---

## 🎯 In-TUI Companion Directives

Developers can switch to NioNano along their project lifecycle or trigger in-session directives:

| Directive | Purpose | Behavior |
|---|---|---|
| *(Direct / Unprefixed)* | **MicroLM Assist** | Sub-millisecond edits: change button labels, font colors, CSS tweaks, AST compiler diagnostics triage, and keyless web lookups. |
| `--bash: <cmd>` | **In-TUI Terminal** | Run shell commands (e.g. `--bash: cargo test`) directly in conversation without exiting or minimizing the coding agent TUI. |
| `--tools: <task>` | **Python Plugins** | Specialized automation (e.g. SVG-to-favicon generator, asset resizers). Prompts for download consent before installing new plugins. |
| `--coding: <prompt>` | **Kilo LLM Middleman** | Bundles relevant local files/folders and consults Kilo models (`kilo auto free`, `kilo mini code`). |
| `--coding:guard <req>`<br/>*or* `--coding!: <req>` | **Anti-Hallucination Guard** | Verifies Kilo's generated code against local AST symbols. Catches ghost APIs and non-existent files; prompts to reroute with live docs. |

---

## 🔬 Taxonomy: What is a MicroLM?

* **MicroLM = Micro Learning Model (µLM)**: Unlike Large Language Models (LLMs) that require gigabytes of VRAM and hallucinate text, NioNano is a **52 KB deterministic learning model** trained specifically for code navigation, intent classification, and decision arbiter logic.
* **CPU-Native**: Runs locally with `<1ms` inference latency on standard CPUs with zero GPU requirements.
* **Binary Footprint**: Standalone single-binary executable is only **5.5 MB** with zero external dependencies.

---

## Architectural Principles

1. **MicroLM-First Autonomous Execution**:
   NioNano handles codebase inspection, symbol searches, checks, and native internet research autonomously with zero GPU requirements and zero API costs.
2. **Decision Modeling, Not Replacement**:
   NioNano is a decision layer, not a smaller LLM. It turns natural language into typed intent, selects among bounded workflows and installed plugins, and dispatches verified actions. The LLM remains the reasoning and authoring partner; NioNano owns routing, policy, evidence, and execution.
3. **Pure Rust Native Web Engine**:
   Zero external browser binaries (no Chromium, no Obscura), zero search API keys. NioXS curls and scrapes DuckDuckGo Lite, Wikipedia, and GitHub directly with a built-in BM25 lexical ranker and bot-challenge defenses.
4. **Durable & Idempotent**:
   Every turn, action claim, and execution state is cryptographically journaled in NioDB to prevent duplicate tool dispatches.
