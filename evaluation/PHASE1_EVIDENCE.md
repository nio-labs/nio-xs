# Phase 1 acceptance evidence

Recorded 2026-10-09, runtime 0.1.0-alpha.1. Core foundation checks passed. The optional Crawl4AI/Obscura CDP probe is deferred; Crawl4AI is excluded from this implementation.

## Implemented

Rust workspace: provider, contracts, rules-only Nano baseline, NioDB HTTP storage adapter, Obscura probe, and artifact descriptor schema. HTTP health/readiness, authenticated catalog/capabilities, bounded text JSON/SSE completions, no-tools enforcement, correlated read_file results, explicit cancellation, and unsupported-task handoff metadata. Public Nio now defaults to nioxs::nio-nano at the user’s request. NioXS and Kilo are built in; saved preferences and explicit overrides take precedence. The local user preference was intentionally changed from Kilo to Nano.

## Verified

The [real smoke report](reports/phase1-smoke.json) records 13 passing checks. Nio CLI → NioXS → Nio read_file → completed response ran against a dedicated actual NioDB instance. Core records work with Nio assistance and SQL unavailable, idempotent retries return the same record, conflicting payloads are rejected, revisions increment on update, invalid credentials fail, and a record survives restart. Provider events contain digests rather than project contents. Explicit cancellation stops pending dispatch. A storage outage returns 503 without a tool action.

Five Rust unit tests cover tool correlation/abstention, browser challenge classification, authentication, and fail-closed storage. Workspace fmt and clippy checks pass. Nio provider integration and launcher checks are recorded in the development session; the live smoke itself runs the real CLI rather than a mocked provider.

## Browser feasibility

The [live report](reports/phase1-browser.json) used pinned Obscura 0.2.4 without Chromium or a search API key. DuckDuckGo returned usable official Rust links; the probe followed one to doc.rust-lang.org and read its content. Wikipedia and example.com were readable. Google returned a challenge and was skipped. Bing loaded but produced no accepted destination links. These are observations from one query on one machine, not permanent engine guarantees. CAPTCHA detection is a baseline heuristic with fixture coverage. Engine-specific extraction and adaptive budgets remain Phase 3.

## Storage boundary

NioDB PATCH increments revisions but its tested API does not implement expected-revision compare-and-swap. Phase 1 uses immutable, payload-hashed idempotent events rather than pretending mutable workflow updates are race-safe. Ownership is a dedicated deployment/database workspace. See [storage contract](../contracts/storage.md). Expiry 0 is a development exception; production retention/quota enforcement and recovery of mutating commands remain Phase 2/3.

## Deferred

No trained classifier, arbitrary command workflow, chat browser workflow, automatic LLM switching, MCP, downloadable model weights, installer, signed release, or hosted service is delivered in this phase. The 42-record seed corpus specifies family-separated splits for future classifier work; it is not an accuracy evaluation. Node is only used for development scripts. Python training is Phase 2. The optional Crawl4AI CDP probe was not performed because no Python runtime is installed and Obscura already supports this feasibility check.

Reproduction instructions: [DEVELOPMENT.md](../docs/DEVELOPMENT.md). Dependencies and external binary hashes: [dependency-pins.json](../dependency-pins.json).
