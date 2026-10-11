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

---

- **v0.1.0 (Current)**: Rock-solid, non-LLM MicroLM foundation — plus the Phase A correctness patch (gated edits, truthful telemetry, no risk-word blocklist):
  - Rust-native intent router and classifier (`nioxs-nano`).
  - Read, find, search, check workflows.
  - Pure native Rust scraping engine (`nioxs-browser`) for DuckDuckGo Lite, Wikipedia, and GitHub.
  - NioDB durability and frozen holdout safety verification.
  - Note: the coding engine, dataset hub, and local MCP stdio server are in-tree ahead of their gates and are re-scoped by the v0.2.0 plan below.
- **v0.2.0 (Next)**: The Measured MicroLM — [plan](NIOXS_V0.2.0.md):
  - Real-phrasing corpus (Stack Overflow / GitHub issue titles) with frozen holdout and CI eval.
  - Trained hashed word+char-ngram logistic router (~20 hierarchical labels, ≤500 KB artifact, <1 ms p99).
  - Trained CRF argument extraction replacing string heuristics.
  - Calibrated abstention: clarify as a guarantee, not a threshold guess.
  - On-disk hash-pinned knowledge packs with BM25 + MMR citations, additive without recompiling.
  - Still 100% non-LLM: no runtime LLM, no neural networks, no LLM-generated training data.
- **v1.0.0**: Decision Layer & Plugin Ecosystem:
  - Versioned plugin packs (Rust or Python) fetched, verified, and executed under MicroLM policy.
  - Decision modeling over external LLM candidate answers: MicroLM scores, verifies, and selects an executable plan instead of delegating execution.
  - Public knowledge-pack repository distributed through GitHub Releases (see [NIOXS_NEXT.md](NIOXS_NEXT.md)).

---

## Planning & Documentation

- [Next milestone — Decision Layer, Plugins, and Knowledge Packs](NIOXS_NEXT.md)
- [Full design and flows](NIOXS_PLAN.md)
- [Five-phase roadmap and acceptance gates](NIOXS_ROADMAP.md)
- [v0.2.0 plan — The Measured MicroLM](NIOXS_V0.2.0.md)
- [Local development and startup](docs/DEVELOPMENT.md)
- [Provider contracts](contracts/README.md)
- [Storage contract](contracts/storage.md)
- [Phase 1 evidence](evaluation/PHASE1_EVIDENCE.md)
- [Phase 2 evidence and limitations](evaluation/PHASE2_EVIDENCE.md)
- [Dependency pins](dependency-pins.json)
- [License inventory](THIRD_PARTY.md)
