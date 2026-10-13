# NioDB adapter and persistence contract

The tested development binary is pinned by version and SHA-256 in `dependency-pins.json`. The provider uses core HTTP records only: authenticated GET /api/v1/records, POST /api/v1/records with Idempotency-Key, and GET /api/v1/records/{id}. No assist, chat, SQL, query, vector, or worker endpoint is used.

One NioDB server has one workspace. For this prototype dedicate the instance/workspace to a NioXS deployment; do not expose the provider token as a database token. Phase 1 uses operator-owned credentials, not per-end-user row isolation. Credentials are read from private token files and never stored in records. Requests cannot override the database endpoint or deployment ID.

## Phase 1 event schema

Collection: nano_events. Immutable append-only events contain schema_version, deployment_id, turn_id, event, model, request_digest, action_id, result_digest, expires_at_unix. Request/file content is not persisted. The adapter hashes the complete stable event payload into the idempotency key; identical retry payloads return the same record, while NioDB rejects key reuse with different data.

The prototype records a durable intent/result before emitting a response. An outage returns 503 before a client action is emitted. Recorded intent is not proof of client execution. Nio's own conversation history remains the authority for client tool results. A database write accepted before cancellation may remain as intent; this does not imply the file was read.

## Verified database limitation

The inspected record PATCH handler increments revision but does not compare an expected revision or If-Match header. Thus the adapter deliberately does not use PATCH for workflow concurrency or claim compare-and-swap. Phase 2 must serialize mutable transitions or use a separately verified conditional-update mechanism. The smoke fixture exercises PATCH for basic CRUD, not concurrency protection.

No exactly-once external command guarantee is possible from the journal alone. On restart, inspect uncertain action outcomes before allowing another dispatch. Side-effect workflows are not implemented in Phase 1.

## Phase 2/3 collection schemas

| Collection | Required application fields | Initial policy |
| --- | --- | --- |
| nano_runs | schema_version, deployment_id, turn_id, state, intent, model_version, created_at, updated_at, expires_at | Retain up to 7 days; explicit recovery state |
| nano_actions | schema_version, deployment_id, turn_id, action_id, tool, arguments_digest, status, result_digest, expires_at | Match run retention; unknown is a real status |
| nano_evidence | schema_version, deployment_id, turn_id, canonical_url, retrieved_at, extraction_version, excerpt, content_digest, expires_at | Bounded excerpts only, no secrets |
| nano_cache | schema_version, deployment_id, cache_key, access_scope, canonical_url, extraction_version, retrieved_at, expires_at, excerpt | Public data only; expire after 24 hours by default |
| nano_feedback | schema_version, deployment_id, turn_id, consent, outcome, created_at, expires_at | Opt-in; retain up to 30 days; no automatic training |
| nano_flow | schema_version, deployment_id, session_key, created_at_unix, expires_at_unix, session | Bounded serialized handoff snapshot; 600-second expiry; oversized snapshots are not stored |

Policies are specified here, not implemented by the Phase 1 service. For immutable prototype nano_events, expires_at_unix=0 explicitly means a development retention exception; use isolated disposable storage. Production cleanup/quotas are required before release. Proposed per-record data cap: 16 KiB; cache cap: 64 MiB or 2,000 records, whichever is reached first; feedback cap: 1,000 records. Revisit measured limits before Phase 3.

No automatic NioDB TTL, row concurrency, vector generation, or LLM assistance is assumed. Expiry and quotas belong to the NioXS adapter. Authentication/revision/idempotency/readiness behavior is verified against the pinned binary by the local smoke test. Database backup/restore and schema rollback are Phase 5 release gates.

## Implemented Phase 2 journal

Collection nano_actions contains dispatch_intent and completed/failed outcome records. Schema version 2 includes deployment_id, turn_id, action_id, logical revision, model_version, tool, arguments_digest, request_digest, status, result_digest for outcomes, created_at_unix and expires_at_unix. Claims also include a provider instance owner. nano_runs contains immutable state transitions with correlated completed action IDs and intent. Database record revision must remain 1; an external PATCH causes a conflict. Logical revisions are 2*step+1 for dispatch and 2*step+2 for observed outcomes. Idempotency keys coordinate each turn/step claim and each action result. Cross-instance owners conflict instead of both issuing a command.

Before a command dispatch, existing claims/outcomes are read. An incomplete previous command becomes unknown_action; a recorded result prevents another command dispatch even if raw output was lost from the client history. Raw output is not persisted; a client retaining its history can recover exact answers. Conflicting argument/result digests stop continuation. Record data is capped at 16 KiB and bounded collection scans at 2,000 records; exceeding a scan fails closed. Dedicated deployments must plan cleanup before reaching this development quota.

7-day expiry metadata is stored, but deletion is not implemented yet. Phase 3 cleanup must preserve command tombstones or reject continuations for expired turns rather than permit replay. Historical Phase 1 nano_events remain compatible and retain their original development expiry exception.

## Implemented research cache and quota

Collections nano_cache and nano_evidence implement the Phase 3 research cache. Research is caller-driven: NioXS relays web_fetch and the client fetches, so the provider caches and records only what it actually observes in a returned tool result. A record stores schema version, deployment_id, turn_id, a hashed cache key over scope+canonical URL, access_scope, canonical_url, title, extraction_version, retrieved_at/created_at/expires_at, content_digest, and a bounded excerpt capped at 4,000 characters. Evidence is a second append-only record of the same payload; a failure writing evidence leaves a usable cache entry and does not fail the put.

Cache entries expire 24 hours after retrieval. Each put first deletes expired cache records, then refreshes an existing entry for the same scope and URL instead of adding a duplicate, so the quota counts distinct cached URLs. It fails closed with `research cache quota reached` at 2,000 cache records, matching the record cap and mirroring the 16 KiB per-record limit; an oversized or rejected write is a cache miss, never a turn failure. Evidence records are keyed by turn and retrieval time, so a re-observation appends rather than colliding. Reads are scope-isolated: `research_get` returns an entry only for the same scope and canonical URL and only while unexpired, so one workspace can never read another's excerpts. `research_clear` deletes every cached entry in a scope for explicit invalidation. All of this is best effort — an unreachable NioDB degrades to "nothing cached" and the turn continues. Challenge, login, and access-denied pages are classified before extraction and are never cached. `GET /ready` reports the current `cache_records`, `evidence_records`, and `cache_limit` so quota use is observable.

## Implemented flow handoff

Collection nano_flow stores a bounded snapshot of an in-progress NioFlow handoff so a restarted provider can resume a continuation without replaying tools. A record holds schema version, deployment ID, a hashed session key, created/expiry timestamps, and the serialized session. The snapshot is kept under the 16 KiB record cap: per-stage tool results are dropped first, then the oldest carried-forward drafts, and the snapshot is flagged truncated; a snapshot that still cannot fit is not persisted and that handoff stays in memory only. Expiry is 600 seconds and matches the in-memory retention window. A continuation is resumed only when the stored session fingerprint matches the current task; otherwise it is refused. The record is removed best-effort once the turn ends, and the expiry bounds any record left behind.
