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
   │    NioNano (MicroLM)      │             │    Consulted Frontier     │
   │  • Sub-megabyte model     │             │    LLMs (v1.0.0+)         │
   │  • <1ms CPU inference     │             │  • DeepSeek / Claude /    │
   │  • 100% deterministic     │             │    Ollama / OpenAI        │
   │  • Pure native scraping   │             │  • Consult & Advise only  │
   │  • Codebase inspection    │             │  • Zero tool execution    │
   └───────────────────────────┘             └───────────────────────────┘
```

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
2. **Consult & Execute Paradigm (v1.0.0)**:
   Rather than handing off entire sessions to third-party LLMs, NioXS will consult frontier models for advice or strategy, while execution stays sovereignly inside NioNano and NioXS.
3. **Pure Rust Native Web Engine**:
   Zero external browser binaries (no Chromium, no Obscura), zero search API keys. NioXS curls and scrapes DuckDuckGo Lite, Wikipedia, and GitHub directly with a built-in BM25 lexical ranker and bot-challenge defenses.
4. **Durable & Idempotent**:
   Every turn, action claim, and execution state is cryptographically journaled in NioDB to prevent duplicate tool dispatches.

---

## Milestone Roadmap

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
- **v1.0.0**: Hybrid Gateway & Frontier LLM Consultation:
  - NioXS routes high-ambiguity planning queries to external LLMs while keeping tool execution strictly governed by MicroLM policies.

---

## Planning & Documentation

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
