# NioXS Next — Companion Coding Assist, Prefix Directives, and Decision Gate

Status: Revised roadmap for v0.2.0. NioXS is positioned as a **Pure Coding Assist Companion** (OpenAI-compatible drop-in for coding agents/TUIs) rather than an autonomous agent.

---

## 0. Direction in one paragraph

NioXS and NioNano operate as a **lightweight in-TUI companion and decision layer**, not an autonomous agent that takes over the keyboard. Because NioXS exposes standard OpenAI-compatible endpoints (`/v1/chat/completions`), developers can plug it into any existing coding agent (Aider, OpenCode, Cline, Continue, Roo, Cursor). Coders switch to `nio-nano` along their project lifecycle for micro-edits (label changes, font-colors, CSS tweaks, AST compiler triage) or trigger specialized workflows using explicit prefixes:
- `--tools: <task>`: executes pre-defined specialized tools and Python plugins (e.g., SVG-to-Favicon generation), prompting the user for consent before downloading external plugins.
- `--bash: <cmd>`: executes shell commands directly within the coding agent session so the user never has to leave their TUI.
- `--coding: <req>`: acts as an intelligent middleman by bundling local relevant files/folders and consulting upstream LLMs via the Kilo provider (Kilo Auto Free / Mini Code).
- **Decision Model**: acts as the safety arbiter and download/bash permission gate.

---

## 1. Positioning: Companion Assist & Decision Arbiter

The differentiator is that NioXS is **complementary to the developer and their coding agent**:

| | Coding Agent / Host LLM | NioNano Companion |
|---|---|---|
| Role | Orchestrate large session, edit loops | Companion assist, rapid micro-edits, prefix routing |
| Placement | Host agent / TUI (Aider, Cline, etc.) | Selected model or middleman gateway (`nio-nano`) |
| In-TUI Terminal | Typically requires switching out to terminal | `--bash:` runs directly in conversation |
| Specialized Tools | Static tools or heavy tool definitions | `--tools:` triggers Python plugins (with download consent) |
| Large Coding Tasks | Handles directly or via `--coding:` | Acts as middleman to Kilo LLM with local files/folders bundled |
| Micro Edits & Fixes | High token cost, slow turnarounds | Local MicroLM (<1ms), label/color/CSS edits, AST compiler triage |
| Safety Gate | Unchecked edits unless gated | Decision Model arbitrates policy, bash safety & plugin downloads |

**Interaction model**

1. The coder interacts in their coding agent TUI.
2. If they need instant micro-assistance (e.g. "change button label to 'Submit'", "fix rustc diagnostic", "change heading font-color to blue"), NioNano handles it instantly without high token fees.
3. If they enter `--bash: cargo test`, NioXS runs the command and streams stdout/stderr back without exiting the TUI.
4. If they enter `--tools: convert icon.svg to favicon.ico`, NioXS detects the required Python plugin, checks if it is installed, requests user approval if download is required, and executes the script.
5. If they enter `--coding: refactor auth module`, NioXS bundles the relevant local files and folders and queries the Kilo provider (`kilo auto free` / `kilo mini code`). Before returning the code, NioXS performs a **Hallucination & Reality Check** (inspecting generated symbols, imports, and file paths against the project AST). If an invented symbol or invalid API is found, NioXS automatically reroutes (e.g. augmenting context with web/AST docs and re-querying Kilo) or prompts the user whether they wish to reroute or continue.

**Deliverable (implemented):** a `nioxs.decision` object on the completions
response — `{ chosen, ranked, confidence, evidence, policy, abstained_reason }`
— plus a `POST /v1/decide` endpoint that accepts candidate proposals and returns
the same object without executing anything. Scoring, the abstention gates, and
the request validation live in `crates/nano/src/decision.rs` and are covered by
unit tests plus provider endpoint tests. Training data for the decision model
still comes only from frozen decision benchmarks, never from live user turns.

Open follow-ups for this workstream: learn the weights instead of the current
hand-set constants, calibrate the confidence, and feed `past_outcomes` from real
NioDB journal records rather than caller-supplied counts.

**Acceptance gates:** zero unauthorized actions on the decision benchmark;
measured choose/abstain accuracy on a frozen holdout; every executed decision
carries an evidence reference and a durable journal record.

---

## 2. Workstream A — Plugin system (Rust and Python)

Goal: add tools and domain logic without recompiling NioXS, under the same
permission model as built-in workflows.

**One contract, two runtimes.** Plugins talk a strict JSON-over-stdio protocol
(mirroring the existing MCP server) rather than linking into the process:

- `runtime = "rust"`: a self-contained binary built from the plugin repo,
  invoked as a subprocess. No `cdylib`, no ABI coupling, no `unsafe` in core.
- `runtime = "python"`: a pinned interpreter running the plugin entrypoint as a
  subprocess (a managed venv when declared).
- `runtime = "mcp"`: an existing MCP server, so MCP support and plugins share
  one loader.

Process isolation is deliberate: Node and Python stay development-only
dependencies for NioJS/inference, and a plugin crash or hang can never take
down the gateway.

**Manifest (`plugin.toml`)** — required fields:

```toml
id = "nio/http-probe"
version = "0.1.0"
runtime = "rust"                 # rust | python | mcp
entrypoint = "bin/http-probe"
min_nioxs = "0.3.0"
license = "Apache-2.0"
determinism = "network"          # deterministic | network | side_effecting
sha256 = "…"                     # pinned at install time

[capabilities]
tools = ["http_head", "http_timing"]

[permissions]
fs_read = []                     # project-relative globs, deny by default
network = ["https://*"]          # explicit host allowlist
exec = false
```

**Permissions are deny-by-default:** no filesystem access beyond the trusted
project root, no network without an explicit allowlist, no shell execution
unless the manifest declares it and the user approves in Build mode. Plugins
never read ambient credentials. CPU time, wall time, and memory are capped; a
plugin that exceeds its budget is killed and recorded as `plugin_failed`.

**Lifecycle:**

```sh
nio plugin search <term>          # reads the plugin index
nio plugin install <id>[@<ver>]   # fetch, sha256-verify, store, record consent
nio plugin list                   # installed, versions, capabilities, digests
nio plugin remove <id>
```

Installed plugins live in the project's `.nioxs/plugins/` (or the user store),
are listed in the journal with their digest, and are never auto-installed or
auto-upgraded. An unverified digest blocks the install.

**Rollout order:** (1) manifest + local load/unload + subprocess supervisor;
(2) permissions + limits + audit; (3) registry fetch with hash pinning;
(4) Python venv management; (5) MCP mapping on top of the same loader.

**Acceptance gates:** a plugin cannot read outside its granted globs, reach a
non-allowlisted host, or run after its budget; every install/run/removal is
journaled; a crashed plugin yields a bounded error, never a hung gateway.

---

## 3. Workstream B — Knowledge packs and the public dataset repository

Goal: a large body of domain knowledge, fetched in ~100 KB units on demand,
stored locally, and served from NioDB with citations.

### 3.1 Repository topology

The dataset lives in a **separate public repository** (`nio-datasets`), never in
this one. Recommended layout, and why:

| Option | Shape | Trade-off |
|---|---|---|
| A. One repo, many releases | `nio-datasets` with one release per family, each release carrying that family's packs | Simplest to operate; one CI pipeline; easiest index generation. Recommended start. |
| B. One repo per family | `nio-datasets-systems`, `nio-datasets-security`, … each with its own releases | Independent versioning and ownership; more repos to maintain. Good once families have separate maintainers. |

Both satisfy "public repo, released on GitHub, packs downloaded on demand". The
index is a single small JSON asset (`index.json`, plus `index.json.sha256`)
published to the newest release and mirrored in each repo for option B. The
index is the only thing fetched at startup; packs are fetched per prompt.

### 3.2 Pack format (≤ 100 KB)

```
pack.json      # metadata + keyword index
content.jsonl  # entries (Q/A pairs, snippets, checklists, templates)
ATTRIBUTION    # required when the license needs it
```

`pack.json` fields: `id`, `family`, `version`, `schema_version`, `title`,
`description`, `language`, `keywords`, `license` (SPDX), `sources` (upstream
URLs), `provenance` (`{generator, generator_version, built_at, git_commit}`),
`sha256`, `entries`, `bytes`. Content is data only — packs are never executed.

Sizes: target 60–100 KB per pack, gzip-compressed for transport. A ~2 GB corpus
is therefore roughly 20,000 packs; ship the most-used families first instead of
publishing a monolithic archive.

### 3.3 On-demand resolution flow

1. Prompt → intent + keywords (existing router features).
2. Local index (cached, ETag revalidated) → candidate pack ids.
3. Miss → fetch the pack from the release asset URL, check `sha256`, reject on
   mismatch.
4. Store in NioDB collection `nano_datasets`, update the in-memory BM25 index
   incrementally, and serve results with pack id + source citations.
5. Enforce a local quota (default 256 MB) with LRU eviction; everything works
   offline from cache, and a failed download degrades to "pack unavailable",
   never to a silent wrong answer.

Users can also pin packs explicitly: `nio dataset install <id>` /
`nio dataset list --installed`. No prompt text, project code, or telemetry is
ever uploaded to the dataset repo.

### 3.4 Licensing and provenance (do this first)

Only redistribute content whose license permits it, and record it in the pack:

- Prefer CC0, public domain, MIT, Apache-2.0, CC-BY, and self-authored content.
- Attribute CC-BY/CC-BY-SA sources in `ATTRIBUTION` and in `pack.json.sources`.
- Do not bulk-import Stack Overflow or other ShareAlike material without
  attribution review; keep a per-pack license gate in CI.
- Every pack ships `provenance` so a pack can be reproduced from its generator
  commit. Adding embedding models or neural artifacts is out of scope: packs
  stay lexical and non-LLM so they remain inspectable and offline.

### 3.5 Build and publish pipeline (in this repo)

`training/datasets/` generates packs deterministically, validates schema, size,
license, and checksums, and writes a `publish-manifest.json`.
A release workflow (GitHub Actions, same style as the existing smoke scripts)
attaches packs to a dataset repo release and regenerates `index.json`. The
trainer and the runtime consume the exact same pack artifacts, so a published
pack is exactly what evaluation tested.

**Acceptance gates:** every pack ≤ 100 KB and schema-valid; `index.json`
checksums reproduce byte-for-byte; an end-to-end fetch installs a pack into
NioDB and answers a prompt with a citation; offline reuse works from cache;
license CI rejects a pack without an approved SPDX identifier.

---

### 3.6 Locked decisions (scaffold shipped)

The dataset repository scaffold now exists at `../nio-xs-dataset` as a separate
public repository (tooling under CC0-1.0). Decisions taken there and adopted
here:

- **Topology: option A** — one public repo, releases per version, one
  `index.json` as the canonical map. Option B (a repo per family) stays available
  if families later gain separate maintainers.
- **Granularity: pack ≤ 100 KiB, shard ≤ 10 MiB** measured on uncompressed
  content, so a ~1 GB corpus is ~10,000 packs across ~100 shard assets. Releases
  are not used as a CDN and byte-range requests are not relied upon: a shard is
  downloaded whole.
- **Two pinned digests** — the shard file digest and each pack's `content.jsonl`
  digest. A consumer verifies the shard before extracting and the pack content
  after, refusing to extract on mismatch.
- **Deterministic shards** — sorted tar entries, `--mtime=@0`, zeroed gzip mtime,
  so identical inputs reproduce identical digests.
- **License gate** — every pack declares an approved SPDX identifier; content
  outside the approved set is rejected at build time.

## 4. Sequencing

| Milestone | Contents | Exit evidence |
|---|---|---|
| **v0.3 (now)** | Retire consulting (done); professional NLP responses (done); `dev.sh` reload runner (done); decision contract implemented (scoring, abstention, `/v1/decide`, `nioxs.decision`) | No `consult_llm`/Kilo code; synthesis has no raw markdown links; all four decision outcomes covered by tests |
| **v0.4** | Learned decision weights and calibration on frozen benchmarks; plugin manifest, supervisor, permissions, limits, audit | Plugin escapes blocked; decision benchmark numbers recorded |
| **v0.5** | Dataset index, on-demand fetch, quota/eviction; first 25 packs published publicly | Live fetch → NioDB → cited answer; license CI green |
| **v0.6** | Plugin registry, Python venv management, MCP mapping; decision model training on frozen benchmarks | Reproducible packs; plugin install verified by digest |
| **v0.7** | Plugin SDK docs, dataset SDK docs, evaluation report | Third-party plugin builds against the contract unmodified |

---

## 5. Risks and open questions

- **Sandbox escapes (high).** A subprocess is isolation, not a security proof.
  Mitigate with least-privilege allowlists, resource caps, no credentials, and
  Build-mode approval; revisit OS-level sandboxing before any third-party
  plugin marketplace.
- **GitHub Releases as a CDN (medium).** Availability and rate limits.
  Mitigate with a local cache, a mirrored index, and an optional custom index
  URL. Keep packs small so partial failures are cheap.
- **Licensing (high, legal).** One bad bulk import can taint the whole corpus.
  Gate per pack in CI, keep attribution files, and prefer permissive licenses.
- **Decision-model quality (medium).** A wrong "execute" is worse than an
  abstention. Ship abstention first, calibrate on a frozen holdout, and require
  evidence references for every executed decision.
- **Still open:** whether Python plugins require a venv by default; whether the
  runtime mirrors the dataset index to object storage once download volume grows.
  The dataset repo topology and pack/shard granularity are settled in §3.6.
