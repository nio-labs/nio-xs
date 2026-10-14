# NioXS (Nio Axis)

<p align="center">
  <strong>The Lightweight Coding Companion & AI Gateway orchestrated by NioFlow, with NioNano MicroLM for local evidence and actions.</strong>
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/@nio-labs/nio-xs"><img src="https://img.shields.io/npm/v/@nio-labs/nio-xs.svg?style=flat-square" alt="npm version"></a>
  <a href="https://github.com/nio-labs/nio-xs/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/nio-labs/nio-xs/ci.yml?branch=main&style=flat-square&label=CI" alt="CI Status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="https://railway.app/new/template?template=https%3A%2F%2Fgithub.com%2Fnio-labs%2Fnio-xs"><img src="https://railway.app/button.svg" alt="Deploy on Railway" height="30"></a>
  &nbsp;&nbsp;
  <a href="https://app.koyeb.com/deploy?type=git&repository=github.com/nio-labs/nio-xs&branch=main&name=nioxs"><img src="https://www.koyeb.com/static/images/deploy/button.svg" alt="Deploy to Koyeb" height="30"></a>
</p>

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
   │   NioFlow (Orchestrator)  │────────────▶│    Upstream LLM (Kilo)    │
   │  • Plans and hands off    │             │  • OpenRouter / Free      │
   │  • Default runs no Helm   │             │  • StepFun for hard tasks │
   │  • --helm for final gate  │             │  • Auto Free fallback     │
   └──────────────┬────────────┘             └───────────────────────────┘
                  │
        ┌─────────┼─────────────┬───────────────┐
        ▼         ▼             ▼               ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐     ┌─────────┐
   │ NioNano │ │ NioNova │ │ NioAtom │ ... │ NioHelm │
   │ evidence│ │ explore │ │  build  │     │final gate│
   └─────────┘ └─────────┘ └─────────┘     └─────────┘
```

NioNano acts as a **Micro Learning Model (µLM)**—not a bloated generative model, but a specialized, deterministic 46 KB model that assists the developer directly in their TUI. For heavy generation, NioXS acts as an intelligent middleman to upstream LLMs while validating generated code against the local project AST to prevent hallucinations.

<p align="center"><a href="docs/PLAN.md">AXIS plan: model roles, NioFlow handoffs, and advancement priorities</a></p>

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

### 4. Docker & Cloud Deployment
```bash
docker run -d --name nioxs -p 7460:7460 ghcr.io/nio-labs/nioxs:latest
```

Deploy instantly to cloud platforms:
- **Railway**: Click the [Deploy on Railway](https://railway.app/new/template?template=https%3A%2F%2Fgithub.com%2Fnio-labs%2Fnio-xs) button or deploy using the included `railway.json`.
- **Koyeb**: Click the [Deploy to Koyeb](https://app.koyeb.com/deploy?type=git&repository=github.com/nio-labs/nio-xs&branch=main&name=nioxs) button or deploy using `koyeb.yaml`.

---

## 🔌 One-Click Agent Setup (`nioxs init`)

Automatically configure your coding agents to route through NioXS:

```bash
# Auto-configure all detected agents in the current project
nioxs init

# Or target specific agents
nioxs init --aider       # Writes .aider.conf.yml
nioxs init --cline       # Writes cline_mcp_settings.json
nioxs init --opencode    # Writes opencode.json
nioxs init --cursor      # Displays Cursor OpenAI custom endpoint settings
```

Once running on `http://127.0.0.1:7460/v1`, select `nio-flow` in your agent. NioFlow is the default orchestrator; `nio-nano` remains available for direct bounded local workflows:

* **Aider**:
  ```bash
  aider --openai-api-base http://127.0.0.1:7460/v1 --openai-api-key none --model nio-flow
  ```
* **OpenCode / Cline / Continue / Cursor / Roo Code**:
  Set OpenAI API Base URL to `http://127.0.0.1:7460/v1` and Model ID to `nio-flow`.

---

## 🎯 In-TUI Companion Directives

This is the **authoritative** directive table; other docs point here instead of duplicating it.

Developers can switch to NioNano along their project lifecycle or trigger in-session directives:

| Directive | Purpose | Behavior |
|---|---|---|
| *(Direct / Unprefixed)* | **NioFlow** | Default plan-first model profile. Runs without Helm verification by default. Use `--helm` to opt in to Helm review. |
| `--helm <task>` / `--with-helm <task>` | **Flow with Helm** | Opts in to final Helm Go / No-Go verification after Flow's specialist stages. |
| `--no-helm <task>` | **Flow without Helm** | Explicitly runs without Helm (already the default). Also accepts `<task> --no-helm`. |
| `--bash: <cmd>` | **In-TUI Terminal** | Run shell commands (e.g. `--bash: cargo test`) directly in conversation without exiting or minimizing the coding agent TUI. |
| `--test: [pattern]` | **Auto Test Runner** | Automatically detects test suite (`cargo test`, `npm test`, `pytest`, `go test`), runs matching pattern, and feeds back trimmed traces. |
| `--git` / `--git:status` / `--git:diff` | **Git Summary** | Runs `git status -s` and/or `git diff --stat` (bare `--git` runs both); an unknown subcommand clarifies instead of guessing. |
| `--tools: <task>` | **Python Plugins** | Specialized automation (e.g. real SVG-to-favicon generator via `resvg`, honest CSS color extraction, Pillow-based image scaling). Prompts for download consent before installing new plugins; a reply of `yes` confirms the plugin that prompt named. |
| `--code: <prompt>` | **Kilo LLM Middleman** | Hands the request to Kilo with the caller's available tools. Tool calls return to the client for execution in its project directory, and results are handed back to the model. OpenRouter / Free is preferred; Auto Free is the availability fallback. Progress identifies availability fallbacks, and the final `answered by` footer names the responding upstream model. The model reuses supplied file contents; general questions do not trigger project inspection. |
| `--code guard <req>`<br/>*or* `--code! <req>` | **Anti-Hallucination Guard** | Verifies generated code against Rust, TypeScript, Python, and Go AST symbols. Catches ghost APIs and non-existent files; prompts to reroute. |
| `--config <key> <on\|off>` | **Session Config** | Toggles `code`, `guard`, `auto`, or `mode` for the current session. |
| `--help [topic]` | **Help** | Prints the directive list, or focused help for one topic (e.g. `--help code`). |

---

## Model profiles and automatic routing

Select a model through your client's model picker or the `model` field of
`POST /v1/chat/completions`. `GET /v1/models` lists all eleven options:

| Model ID | Purpose | Rust-controlled behavior |
|---|---|---|
| `nio-flow` | Plan and hand off | Rust validates a plan and invokes the useful specialists (runs without Helm by default; opt in via `--helm`) |
| `nio-nano` | Assist | Existing trained intent classification and bounded workflows |
| `nio-echo` | Answer | Direct Q&A and conversation; 2 read-only inspection rounds |
| `nio-sage` | Explain | Teaching and examples; read-only project evidence |
| `nio-quill` | Write | Draft and polish prose; returns text or proposed changes |
| `nio-scout` | Research | Read/search available sources with attribution; 6 inspection rounds |
| `nio-thinker` | Reflect | Rust runs Drafting, Challenging, and Refining as separate calls; read-only evidence |
| `nio-nova` | Invent | Creative alternatives and prototypes; 4 exploration rounds |
| `nio-atom` | Build | Bounded implementation through caller tools; code guard |
| `nio-neon` | Inspect | Evidence-based review and diagnosis; 6 exploration rounds, code guard |
| `nio-helm` | Final Go / No-Go | Evidence-linked decision after applicable work and review stages |

NioFlow starts with Kilo North Mini Code when available and Kilo Auto Free as the
availability fallback. When task evidence indicates high complexity, Flow checks
Kilo's live `/models` catalog and switches to a free StepFun model only if the
catalog advertises tool support. If that model is unavailable before it responds,
Flow falls back to Auto Free. Explicit upstream model settings are respected.

Flow prepares a bounded plan and hands off only to useful specialists. It asks
Nano for a native project listing when project context is needed or when the
request inspects the caller's own files or directories ("check the nio-guru
folder") and the caller offers the required file tool, so the inspection reads
the project before any answer is produced instead of an answering profile
guessing. If that optional listing fails, Flow continues with available evidence
and reports the gap. Every stage is pinned to the caller's current request
(earlier user turns are context only), and NioFlow's internal
`[<model> completed stage: …]` headers are stripped from the final answer. General questions,
explanations, and feasibility requests do not trigger workspace scans. Prior
drafts and actual tool results follow the task. Helm gives the final Go / No-Go
by default. Caller tool permissions still decide what can run: Atom receives
only offered tools; Neon and Nova receive read tools; Helm receives no tools.
Echo, Sage, and Quill receive project inspection tools only; Scout and Thinker
can also use an offered `web_fetch` tool. These five profiles cannot execute
commands or apply edits. Because Helm runs without tools, NioXS withholds its
verdict when the synthesized answer asserts a path or scan that the turn never
received as a tool result: the answer is replaced by a deterministic refusal and
`nioxs.flow.helm_guard` lists the withheld assertions. Only tokens with a
positive path shape count — a known file extension, an explicit `./`, `../`, or
`/` prefix, or at least two separators — so prose that joins two hyphenated
concepts with a slash (`speech-to-text/text-to-speech`) is not treated as a
filesystem claim. The refusal states how many tool results the turn did supply,
so a turn with results is never reported as one without tools. An empty file
listing never supports a claim that a path or folder is missing: `find_files`
lists files, so an empty result means no file matched. When a request names a
folder, NioFlow scopes its native listing to that folder instead of sampling the
whole project, so a folder that exists cannot be missed by a bounded sample.
The caller's persona block closes the upstream system prompt, after the
environment and tool sections, so the role and stage messages that follow cannot
bury it.

Selecting Nano, Echo, Sage, Quill, Scout, Nova, Atom, or Neon directly runs the selected role without scheduling Helm. Shared stage instructions keep final approval guidance exclusive to Helm; each specialist returns its own deliverable. Selecting Thinker directly runs a bounded three-phase review of its own answer, with visible Drafting → Challenging → Refining statuses and carried-forward evidence. It does not run Helm by default; add `--with-helm` before or after the task to request Helm’s final verdict. Only the refined answer is shown; it states unresolved uncertainty rather than promising 100% correctness. A Flow-owned Thinker stage uses the same three phases and also runs without Helm by default. These profiles share the configured upstream and are not separate trained model weights.

NioFlow runs without Helm by default; the last specialist supplies the final answer directly without extra delay. To opt in to Helm verification for a task, enter `--helm suggest a dashboard` or `--with-helm suggest a dashboard` in the NioFlow chat (or add `--helm` after the task). The UI shows `⚖ NioHelm · Go / No-Go`. The choice persists through that task's tool rounds and resets for the next task.
Existing `nio-auto` requests remain a compatibility alias for `nio-flow`.

Routing status names the active model and role: `🌊 NioFlow · Planning`,
`↪ NioAtom · Building`, `↪ NioNeon · Reviewing`, or `⚖ NioHelm · Go / No-Go`.
An upstream model switch is shown in progress (for example,
`↪ Upstream: openrouter/free → stepfun/...:free (high task complexity)`);
an availability fallback displays `↪ Using Kilo Auto Free`. Tool-result
continuations do not repeat the stage label. These lines report actual
stage selections and specialist invocations, including automatic Flow handoffs.
API clients can read `nioxs.progress` and `nioxs.status` for routing events and
the `awaiting_tools` handoff to the caller.

Flow keeps bounded handoff state in memory for 10 minutes (up to 32 tasks) and
snapshots it to the NioDB `nano_flow` collection (600-second expiry), so a
provider restart can resume a continuation instead of dropping it. A resumed
handoff must still match the current task fingerprint. If the durable record is
absent, expired, unreadable, or the context changed, Flow stops the continuation
without replaying tools and asks for a fresh task using the collected evidence.
A handoff snapshot too large for the 16 KiB record cap stays in memory only.
NioScout research is caller-executed: NioXS relays `web_fetch` and the client
fetches, so the provider caches and records only the tool results it actually
observes. Cached excerpts live in NioDB `nano_cache` with evidence in
`nano_evidence`, expire after 24 hours, are isolated per access scope, and fail
closed at 2,000 records; challenge, login, and denied pages are never cached.
Every write is best effort, so an unavailable database degrades to no cache
rather than a failed turn, and `GET /ready` reports cache/evidence counts
against the limit. Live remote research observations remain open work.
Stage drafts are buffered; the final answer is emitted after the specialist stage (or Helm when opted in) completes.
Direct specialist requests retain their existing live upstream streaming. The
caller's persona stays active through planning, handoffs, and final answers;
model names describe workflow roles and do not replace that persona.

Nio keeps an animated `Thinking…` indicator with elapsed time while waiting
for model output, including after routing notices and between tool rounds.
Loading state is sent as SSE metadata rather than saved answer text. The
caller's current persona takes precedence over server-local persona settings;
model role names describe the workflow and do not change that persona.

Profiles share Rust workflows and role instructions; they are not separate
trained language-model weights. Explicit `--` directives retain their existing
behavior. NioAtom edits and runs checks only through tools advertised by the
caller and remains subject to that caller's approval policy. Completions expose the requested model, selected profile,
associated skill, and routing reason in `nioxs.routing`; the completion's
`model` identifies the selected profile. Model role skills live under
`.agents/skills/`, with role guidance loaded automatically for the profiles.

With no upstream model override, NioXS checks Kilo's
`https://api.kilo.ai/api/gateway/models` and prefers
`openrouter/free`. If that ID is absent or the catalog check fails,
it falls back to `cohere/north-mini-code:free` and `kilo-auto/free`. For high-complexity NioFlow work, the catalog is also
checked for a free StepFun model that advertises tool support. Catalog checks
have a 3-second timeout; successful results are cached for 5 minutes and
failures for 30 seconds. If North Mini Code or a selected StepFun model is
unavailable before responding, NioXS tries Auto Free; it does not restart a
partial response. `nioxs.upstream_model` records the requested Kilo model ID on
proxied completions. Auto Free chooses its underlying model server-side.
`NIOXS_CODING_MODEL` and `--config model <id>` override automatic model
selection; custom gateway URLs retain their configured model.

Kilo's [API terms](https://kilo.ai/terms) allow application integration subject
to end-user compliance and applicable model/provider terms. Its
[free-model guidance](https://kilo.ai/docs/getting-started/using-kilo-for-free)
warns that Auto Free may use trial endpoints that log prompts and outputs;
do not send personal or confidential data through that tier. North Mini Code's
[Apache 2.0 model license](https://docs.cohere.com/docs/north-mini-code-1.0)
does not replace hosted-service terms. Nio profile names describe our workflows;
the upstream identity remains visible in response metadata and answer footers.

## 🔬 Taxonomy: What is a MicroLM?

* **MicroLM = Micro Learning Model (µLM)**: Unlike Large Language Models (LLMs) that require gigabytes of VRAM and hallucinate text, NioNano is a **46 KB deterministic learning model** trained specifically for code navigation, intent classification, and decision arbiter logic.
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
