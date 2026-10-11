# NioXS (Nio Axis)

**The AI Gateway & Sovereign Router for MicroLMs ($\mu$LMs) and Frontier Models.**

> **AXIS = AI + XS**: NioAI brings the intelligent agent interface; NioXS anchors the operational axis (gateway routing, state durability, safety guardrails, and tool dispatch).

```
                      ┌────────────────────────────────────────┐
                      │              NioAI (Agent)             │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │           NioXS (The Gateway)          │
                      │  • OpenAI-Compatible Proxy (/v1/chat)  │
                      │  • Sovereign Tool & Policy Broker      │
                      │  • NioDB Durability & State Journal    │
                      └───────────────┬────────────────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
   ┌───────────────────────────┐             ┌───────────────────────────┐
   │    NioNano (MicroLM)      │             │    LLM of your choice     │
   │  • Sub-megabyte model     │             │  • Reasoning & authoring  │
   │  • <1ms CPU inference     │             │  • DeepSeek / Claude /    │
   │  • 100% deterministic     │             │    Ollama / OpenAI        │
   │  • Pure native scraping   │             │  • Client-side session    │
   │  • Decides & executes     │             │  • Plans & explains       │
   └───────────────────────────┘             └───────────────────────────┘
```

NioNano and an LLM cooperate instead of competing: the LLM reasons and writes,
NioNano decides, routes, verifies, and executes against the real machine. There
is no runtime LLM call inside NioXS.

---

## The Identity & Taxonomy

- **The Product**: **NioAI** (`nio`) — Sovereign AI pair programmer and agent CLI.
- **The Gateway**: **NioXS** (`nioxs`) — The high-performance, self-hosted gateway router (similar in interface to OpenRouter, but optimized for local-first tool execution and durable journal governance).
- **The MicroLM ($\mu$LM)**: **NioNano** (`nio-nano`) — Not an LLM, but a hyper-compact, sub-megabyte **MicroLM** running deterministic calibrated inference on CPU in under 1ms.
- **The Future Model Family**:
  - `NioNano` ($\mu$LM): Sub-megabyte edge router for code search, read, check, and web research.
  - `NioMini` / `NioCore` ($\mu$LM+): Enhanced local patch and template models.
  - `NioMax` / `NioUltra`: Hybrid orchestrators.
- **Persistence**: **NioDB** through authenticated REST storage contracts (`nano_events`, `nano_actions`, `nano_runs`, and `nano_cache`).

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

## Installation & Quick Start

### 1. Instant Run via NPX (Zero Install)
Runs anywhere with Node.js installed (Linux, macOS, Windows, Termux):
```bash
npx @nio-labs/nio-xs
```

### 2. Universal Install Script (Linux, macOS, Termux)
```bash
curl -fsSL https://raw.githubusercontent.com/nio-labs/nio-xs/main/install.sh | bash
```
*Supports x86_64, ARM64 (Apple Silicon, Raspberry Pi), and Android Termux.*

### 3. Native Windows (PowerShell)
```powershell
irm https://raw.githubusercontent.com/nio-labs/nio-xs/main/install.ps1 | iex
```

### 4. Run with Docker
```bash
docker run -d --name nioxs \
  -p 7460:7460 \
  -v ~/.nioxs:/root/.nioxs \
  ghcr.io/nio-labs/nioxs:latest
```

Verify readiness:
```bash
curl http://127.0.0.1:7460/v1/models
```

### Option 2: Connect to Any Coding Agent (OpenAI Compatible)
NioXS serves on `http://127.0.0.1:7460/v1`. Configure your coding agent:

* **Aider**:
  ```bash
  aider --openai-api-base http://127.0.0.1:7460/v1 --openai-api-key none --model nio-nano
  ```
* **OpenCode / Cline / Continue**:
  Set OpenAI API Base URL to `http://127.0.0.1:7460/v1` with Model ID `nio-nano`.

### In-TUI Directives:
* `--bash: <cmd>`: Run bash commands in-session without exiting the TUI.
* `--tools: <task>`: Run specialized Python plugins (e.g. SVG-to-favicon) with interactive download consent.
* `--coding: <req>`: Fast middleman consultation to Kilo models.
* `--coding:guard <req>` (or `--coding!:`): Strict guarded mode with anti-hallucination reality checks against your project AST.

---

## Open Source & Closed Source Architecture (Approach A)

* **NioXS (Open Source)**: The public gateway (`crates/provider`), browser scraper (`crates/browser`), and CLI tools are open source.
* **NioNano (Proprietary / Closed Source)**: The sub-megabyte MicroLM weights (`models/nano.json`), decision arbiter, and Code Assist Brain (`crates/nano`) are proprietary and compiled directly into official release binaries.
* **Release Artifacts**: Pre-compiled standalone binaries (~5.5 MB) and Docker images include the embedded NioNano engine ready to run out of the box with zero external model downloads.

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
