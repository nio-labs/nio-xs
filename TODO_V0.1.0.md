# NioXS + NioNano v0.1.0 TODO

Status: Phases 1–2 complete with evidence; Phase 3 research engine implemented in-tree with its NioDB cache/quota/smoke gates still open; Phase 4 partial (a local MCP stdio server and a bounded edit/verify loop exist, but edit paths bypass the tool gate, and consult_llm, feedback records, and handoff continuation are missing); Phase 5 open. Optional Phase 1 Crawl4AI probe remains deferred. Target: v0.1.0. See [roadmap](NIOXS_ROADMAP.md) and [full design](NIOXS_PLAN.md).

Check tasks only after their acceptance evidence exists. Each phase depends on the previous phase. Nano is the only model in this release.

**Tracking sync 2026-10-10.** Work shipped after this checklist was written (coding engine, dataset hub, MCP stdio server, package registry inspector, code-health analysis, git review helpers, rotating responses) outpaced the roadmap. Interim feature labels (v0.7.x, v0.8.x) used in those commits were dropped 2026-10-10; the workspace is v0.1.0-alpha.2 and no 0.8.x release exists. The work is recorded below under "Delivered beyond the original v0.1.0 scope" instead of being silently folded into phase gates. Two invariant violations in that work must be fixed first — [Nano Phase A](NIOXS_NANO_PHASES.md): `coder.rs:208,242` `fs::write` and `coder.rs:254` direct command spawn bypass the Nio tool gate, and the classifier risk-word blocklist (`model.rs`) forces common prompts to unsupported. Capability metadata at `crates/provider/src/lib.rs` still advertises the original seven workflows and `mcp:false`; it is stale and must be regenerated from the real intent list before release.

## Phase 1 — Foundations, NioDB, and provider connection

Goal: establish the private Rust project, prove browser feasibility, connect Nio to NioXS, and define NioDB persistence. NioJS is excluded.

- [x] Establish the private repository and Rust workspace for provider, Nano, browser, and artifact components; keep proprietary code outside public Nio.
- [x] Pin toolchain/dependency versions and record distribution notices.
- [x] Pin and start a tested NioDB service with private credentials and persistent storage; verify core CRUD and readiness without Nio assistance or SQL fallback.
- [x] Define the Rust HTTP storage adapter and collection schemas for runs, actions, evidence/cache, and optional feedback.
- [x] Specify ownership, revisions, idempotency, retention, quotas, restart recovery, and storage-outage behavior; prove one durable round trip.
- [x] Define provider ID nioxs, model ID nio-nano, version identifiers, and capability/readiness schema.
- [x] Define request, workflow state, client action, tool result, progress, cancellation, and handoff contracts with example payloads.
- [x] Specify how standard API clients receive capability limitations and how Nio recognizes the handoff extension.
- [x] Specify client-versus-provider tool ownership and request permissions, including provider-side enforcement of no-tools mode.
- [x] Create the supported/unsupported/ambiguous intent dataset specification, fixed evaluation split, and initial acceptance thresholds.
- [x] Probe Obscura search-result extraction and destination reads on representative queries, including Google and alternative engines.
- [x] Verify CAPTCHA detection and unsupported-page behavior without Chromium; record observed engine compatibility.
- [ ] Probe optional Crawl4AI via Obscura CDP; deferred. Crawl4AI is excluded until this compatibility check passes; no Chromium was installed.
- [x] Implement the minimum NioXS service: health/readiness, model catalog, bounded text Chat Completions and a tool round trip.
- [x] Register NioXS in Nio's provider resolution/catalog and verify nioxs::nio-nano selection.

Exit evidence: one request goes Nio → NioXS → client tool call → Nio result → completed response. Cancellation stops the request. Obscura feasibility findings and contract examples are written down. NioXS can write/read a scoped durable record and reports NioDB unavailability clearly.

Evidence: [Phase 1 report](evaluation/PHASE1_EVIDENCE.md), [smoke results](evaluation/reports/phase1-smoke.json), and [browser observations](evaluation/reports/phase1-browser.json). Contracts and later collection schemas are defined; mutable workflows, retention enforcement, trained inference, and handoff execution remain later-phase implementation. Dependency license inventory is preliminary; release notices are Phase 5. The repository is local and unpublished; no remote visibility is configured.

## Phase 2 — NioNano model and project workflows

Goal: deliver an actual trained non-LLM classifier with useful project workflows.

- [x] Implement the rule baseline and typed intent/argument validation for the initial workflows.
- [x] Prepare labeled training and held-out examples, including unrelated requests, missing arguments, and unfamiliar tasks.
- [x] Train and calibrate a small classifier in Python; export features, weights, labels, and artifact metadata.
- [x] Implement matching feature extraction and model inference in Rust, including abstention for unsupported input.
- [x] Compare classifier quality with the baseline; document unsupported-task acceptance and performance by workflow.
- [x] Load and validate versioned model artifacts; expose model readiness and capability limits through NioXS.
- [x] Implement project inspection from manifests and files with evidence-backed output.
- [x] Implement file/code search with ranked matches, source paths, and exact snippets.
- [x] Implement approved test/build/lint workflows using supported configured commands and bounded parsed output.
- [x] Implement structured/template responses and workflow-specific completion checks.
- [x] Persist correlated run/action transitions through NioDB with validated record revisions and supported idempotency.
- [x] Record dispatch intent and observed outcomes; recover completed actions after restart and mark uncertain outcomes unknown instead of automatically replaying commands.
- [x] Test concurrent continuation, revision conflict, restart, and storage outage; block side-effect dispatch when durable tracking is unavailable.
- [x] Verify Ask/Plan/Build, project trust, no-project-tools, no-tools, command approvals, output limits, and cancellation.

Exit evidence: trained Rust inference agrees with exported-model fixtures; supported project workflows succeed on the held-out corpus, unsupported requests abstain, and no workflow gains extra permissions. NioDB restart/outage tests do not replay side effects.

Evidence: [Phase 2 report](evaluation/PHASE2_EVIDENCE.md), [frozen model evaluation](evaluation/reports/phase2-frozen-holdout.json), and [real workflow checks](evaluation/reports/phase2-smoke.json). Checkmarks cover bounded prototype workflows; production evaluation, retention cleanup, and distribution remain later gates.

## Phase 3 — Keyless adaptive browser research (Pure Native Rust)

Goal: search, visit useful websites, and return evidence with complexity-based resource limits using pure native Rust without external browser dependencies or Chromium.

- [x] Implement pure native Rust in-process HTTP and HTML extraction in `nioxs-browser` with bounded timeouts and zero external binaries.
- [x] Implement keyless search adapter (DuckDuckGo Lite) with title, destination URL, and snippet parsing.
- [x] Skip URLs when destination pages are blocked; exclude CAPTCHA, bot challenge, and access-denied pages from extracted evidence.
- [x] Extract clean readable text, headers, and code blocks from HTML pages; strip boilerplate, styles, and scripts.
- [x] Implement passage extraction with lexical term-overlap and BM25 ranking.
- [x] Implement source-linked research reports and cited excerpts in `nioxs-browser` and integrate into `nioxs-nano`.
- [x] Expose `web_fetch` and `research_sources` workflows and client-tool compatibility in `nioxs-provider`.
- [ ] Store bounded evidence records and research cache in NioDB with timestamps, normalized keys, extraction versions, and access scope. (Storage adapter primitives exist — `cache_get`/`cache_set`/`scan_expired` in `nioxs-storage` — but research workflows are not yet wired to them.)
- [ ] Implement expiry, quotas, invalidation, and clearing in the storage adapter; exclude challenge pages, credentials, and sensitive content from shared cache.
- [ ] Test cache reuse/expiry, workspace isolation, quota enforcement, and cache-write failure behavior.
- [ ] Confirm clean execution never installs, launches, or falls back to Chromium or external browser processes.

Exit evidence: a lookup stops when its required fields are found, a comparison expands only when needed, blocked pages cannot create loops, and reported excerpts match linked sources.

## Phase 4 — Consult & Execute, automatic handoff, and API/MCP interoperability

Goal: empower NioNano as a small coding agent via LLM consultation ("Consult & Execute") while preserving sovereign execution, with handoff as an escape hatch, API compatibility, and local MCP.

- [ ] Define the typed client tool contract for `consult_llm` (prompt, context excerpts, patch/instruction schema) executed via Nio's provider broker.
- [ ] Implement NioNano's coding agent loop: inspect failure/code → assemble compact context → dispatch `consult_llm` → parse advisory patch.
- [ ] Implement bounded, verified code modification: safe file editing (`edit_file` / `apply_patch`) validating against `safe_path` and project trust. (An edit engine exists — `crates/nano/src/patch.rs` and the coder edit path — but `coder.rs:208,242` call `fs::write` directly, bypassing the tool gate; gated on Nano Phase A.)
- [ ] Implement the verification loop: run configured checks (`run_checks`) post-edit and iterate or terminate within the 8-action budget. (An edit-and-test loop exists in `coder.rs` but runs on the ungated edit path and spawns commands directly; gated on Nano Phase A.)
- [ ] Implement fallback policy and handoff triggers: unsupported conversational tasks, classifier abstention, or exhausted budget after consultation.
- [ ] Keep missing-input clarification separate from escalation; do not escalate just because a page is blocked or a tool fails once.
- [ ] Persist consultation/handoff state and actual backend path in scoped NioDB run records while keeping credentials in Nio.
- [ ] Add opt-in feedback records with retention and explicit export; no automatic prompt upload or retraining.
- [ ] Assemble bounded handoff packets with request context, evidence, tool-call/result groups, completed side effects, gaps, and current permissions.
- [ ] Implement the NioXS handoff extension and Nio's automatic continuation on the user-configured LLM provider when full delegation is needed.
- [ ] Keep user provider credentials in Nio; show NioNano → configured LLM with the reason and actual backend path.
- [ ] Handle disabled/missing fallback, provider failures, cancellation, follow-up turns, and one-handoff-per-turn limits.
- [ ] Verify no replayed edits/commands and no handoff after denied approvals or cancellation.
- [ ] Complete the documented text Chat Completions subset, SSE conformance, errors, model validation, and capability metadata.
- [ ] Verify one external OpenAI-compatible client against the supported API subset and document unsupported parameters.
- [x] Expose basic local MCP stdio research/page-reading tools with the same resource and permission rules. (Implemented as `nioxs mcp`, `crates/provider/src/mcp.rs`; tools share the browser worker.)
- [ ] Connect Nio's broker to one configured local MCP server; map known schemas to Nano workflows and abstain on unknown tools.
- [ ] Test external API and MCP tool loops, cancellation, history continuity, and no-tools enforcement.

Exit evidence: coding requests successfully consult the LLM, apply bounded edits, and verify tests without losing Nano's execution sovereignty; unsupported conversational requests hand off cleanly; external API and MCP tools work under documented scope.

## Phase 5 — Packaging, evaluation, and v0.1.0 release

Goal: deliver verified artifacts and a release that can be installed and used on a clean machine.

- [ ] Run final held-out workflow/model evaluation and end-to-end Nio, API, browser, MCP, and handoff checks.
- [ ] Measure installation size, peak memory, startup/research latency, source usefulness, and fallback reasons; publish measured limitations.
- [ ] Build and pin runtime, Nano model, Obscura, NioDB, optional extraction dependencies, and wire/model/record compatibility versions.
- [ ] Package or document NioDB installation, startup, readiness, private credentials, persistent volume, and shutdown.
- [ ] Validate schema migration and rollback compatibility, full backup/restore, database restart, retention cleanup, and disk/storage failure behavior.
- [ ] Produce initial Linux x86_64 artifacts; include other platforms only after their full checks pass.
- [ ] Package required licenses/notices, including Crawl4AI attribution if included.
- [ ] Generate and sign the release manifest with immutable versioned artifacts, hashes, sizes, platform, and compatibility metadata.
- [ ] Configure R2 or S3 publishing and artifact access without embedding write credentials in clients.
- [ ] Implement download verification, safe extraction, atomic installation/update, version pinning, and rollback.
- [ ] Verify corrupted/incompatible artifacts, interrupted updates, expired links where applicable, and signature rejection.
- [ ] Document local deployment, endpoint configuration, provider authentication, hosted deployment boundaries, and troubleshooting.
- [ ] Verify clean-machine installation and execution with no Chromium dependency or search API key.
- [ ] Prepare v0.1.0 release notes, capability matrix, known limitations, and release checklist.
- [ ] Publish the reviewed v0.1.0 artifacts when release checks pass and release authorization is in place.
- [ ] Verify the NioNano default meets production acceptance before release; the development default was activated during Phase 1 at the user’s request. Preserve existing explicit model preferences and local-only settings.

Exit evidence: the released bundle installs on the supported platform, verified updates and rollback work, no release blockers remain, and default activation does not override existing model choices.

## After v0.1.0

- NioMini, NioCore, NioMax, and NioUltra.
- Remote MCP transport and expanded external tool mappings.
- Responses API compatibility and additional modalities.
- Broader platform support and public hosted-service launch.
- Vector retrieval and separately generated embeddings; NioDB assistance/SQL are not Nano prerequisites.
- Richer document/data workflows and predefined repairs.

## Delivered beyond the original v0.1.0 scope (sync 2026-10-10)

Features shipped in-tree that this checklist never tracked. They exceed the v0.1.0 phase plan and carry no phase evidence; several are v0.1.5 roadmap items arriving early. Decision required before release: fold them into a re-baselined v0.1.5 scope with their own gates, or explicitly descope them so v0.1.0 claims stay honest.

- Code-health analysis (`analyze file` / `analyze codebase` with complexity and safety findings) — `crates/nano/src/analysis.rs`.
- Coding engine: symbol definition/reference navigation, `replace … in …` refactor, autonomous edit-and-test loop, test scaffolder, `git diff` review, commit-message suggestion — `crates/nano/src/coder.rs`.
- Deterministic patch engine and scaffold templates — `crates/nano/src/patch.rs` plus `scaffold <template>`.
- Dataset hub: 36 static knowledge packs (~100 KB payload target each), prompt router, and on-demand persistence into NioDB collection `nano_datasets` — `crates/nano/src/dataset.rs`. Primarily a v0.1.5 capability; no cache or evidence report accompanies it.
- Natural-language prompt normalization and rich response formatting — `normalize_prompt` in `lib.rs`, `crates/nano/src/synthesize.rs`.
- Autonomous task planner and dynamic project suggestions — `TaskPlan` / `ProjectSuggestions` intents in `lib.rs` and their handlers in `coder.rs`.
- Local MCP stdio server (`nioxs mcp`) — `crates/provider/src/mcp.rs`.

## Tracking summary

| Phase | Status |
| --- | --- |
| 1 — Foundations/NioDB/provider | Complete with evidence; optional Crawl4AI probe deferred |
| 2 — Trained Nano/project workflows | Complete with evidence |
| 3 — Adaptive browser research | Engine implemented in-tree; NioDB cache/quota/pruning, live smoke tests, and an evidence report remain open |
| 4 — Handoff/API/MCP | Partial: MCP stdio server and bounded edit/verify loop exist; edit gating, consult_llm, feedback, handoff continuation open |
| 5 — Packaging/release | Open; stale capability metadata (`crates/provider/src/lib.rs`) must be fixed first |
| Beyond scope | Coding engine, dataset hub, code analysis, git helpers, rotating responses — shipped, untracked, awaiting v0.1.5 scope decision |

Update phase status and attach acceptance evidence as work is completed. Phase 1 checkmarks cover its limited foundation scope, not Phase 2–5 capabilities.
