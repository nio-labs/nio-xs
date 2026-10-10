# NioXS + NioNano v0.1.0 TODO

Status: Phase 2 core implemented and tested; optional Phase 1 Crawl4AI probe deferred. Target: v0.1.0. See [roadmap](NIOXS_ROADMAP.md) and [full design](NIOXS_PLAN.md).

Check tasks only after their acceptance evidence exists. Each phase depends on the previous phase. Nano is the only model in this release.

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

## Phase 3 — Keyless adaptive browser research

Goal: search, visit useful websites, and return evidence with complexity-based resource limits.

- [ ] Integrate the pinned Obscura worker with bounded startup, navigation, idle shutdown, crash handling, and cancellation.
- [ ] Implement tested browser search adapters, configurable engine order, query encoding, link extraction, and redirect normalization.
- [ ] Switch engines when a search results page is blocked; skip URLs when destination pages are blocked.
- [ ] Exclude CAPTCHA/login/access-denied pages from extracted evidence and apply bounded domain cooldowns.
- [ ] Extract readable passages and metadata; use normal HTTP for suitable pages and Obscura for rendered content.
- [ ] Deduplicate pages and rank passages with a measured lexical baseline.
- [ ] Estimate query complexity and start with small batches; expand based on missing fields and source coverage.
- [ ] Enforce page, candidate, engine-switch, deadline, concurrency, redirect, link-following, and byte limits.
- [ ] Implement early stopping for complete lookups and diminishing returns; label incomplete results and conflicts.
- [ ] Implement source-linked research reports and package comparison with explicit missing fields and access times.
- [ ] Store bounded evidence records and research cache in NioDB with timestamps, normalized keys, extraction versions, and access scope.
- [ ] Implement expiry, quotas, invalidation, and clearing in the storage adapter; exclude challenge pages, credentials, and sensitive content from shared cache.
- [ ] Test cache reuse/expiry, workspace isolation, quota enforcement, and cache-write failure behavior.
- [ ] Test simple lookups, comparisons, repetitive results, redirects, CAPTCHA fixtures, outages, and exhausted budgets.
- [ ] Confirm clean execution never installs, launches, or falls back to Chromium; include Crawl4AI only if Phase 1 feasibility passed.

Exit evidence: a lookup stops when its required fields are found, a comparison expands only when needed, blocked pages cannot create loops, and reported excerpts match linked sources.

## Phase 4 — Automatic handoff, API compatibility, and local MCP

Goal: continue unsupported work on the configured LLM and support initial external integrations.

- [ ] Implement fallback policy and triggers: unsupported intent, abstention, broader reasoning needs, and repeated lack of useful progress.
- [ ] Keep missing-input clarification separate from escalation; do not escalate just because a page is blocked or a tool fails once.
- [ ] Persist handoff state and actual backend path in scoped NioDB run records while keeping credentials in Nio.
- [ ] Add opt-in feedback records with retention and explicit export; no automatic prompt upload or retraining.
- [ ] Assemble bounded handoff packets with request context, evidence, tool-call/result groups, completed side effects, gaps, and current permissions.
- [ ] Implement the NioXS handoff extension and Nio's automatic continuation on the user-configured LLM provider.
- [ ] Keep user provider credentials in Nio; show NioNano → configured LLM with the reason and actual backend path.
- [ ] Handle disabled/missing fallback, provider failures, cancellation, follow-up turns, and one-handoff-per-turn limits.
- [ ] Verify no replayed edits/commands and no handoff after denied approvals or cancellation.
- [ ] Complete the documented text Chat Completions subset, SSE conformance, errors, model validation, and capability metadata.
- [ ] Verify one external OpenAI-compatible client against the supported API subset and document unsupported parameters.
- [ ] Expose basic local MCP stdio research/page-reading tools with the same resource and permission rules.
- [ ] Connect Nio's broker to one configured local MCP server; map known schemas to Nano workflows and abstain on unknown tools.
- [ ] Test external API and MCP tool loops, cancellation, history continuity, and no-tools enforcement.

Exit evidence: an unsupported request continues on the configured LLM without losing progress; a supported request stays on Nano; an external API client and both initial local MCP roles work under the documented scope.

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

## Tracking summary

| Phase | Status |
| --- | --- |
| 1 — Foundations/NioDB/provider | Core checks passed; 14/15 tasks, optional Crawl4AI probe deferred |
| 2 — Trained Nano/project workflows | Core implemented; model and real workflow checks passed |
| 3 — Adaptive browser research | Not started |
| 4 — Handoff/API/MCP | Not started |
| 5 — Packaging/release | Not started |

Update phase status and attach acceptance evidence as work is completed. Phase 1 checkmarks cover its limited foundation scope, not Phase 2–5 capabilities.
