# Phase 2 provider contract

Schema version: 1. Provider: `nioxs`. Default model: `nio-flow`. Runtime: `0.1.0-alpha.2`.

NioNano uses trained Naive Bayes inference and typed workflow validation. Read, inspect, filename search, literal code search, and configured checks are supported. The client must offer the required tools and allow automatic choice. Every actual file read/command is performed by Nio’s permission broker. No provider-side browser/shell/filesystem tools are exposed by chat.

Tool expansion Phases 1–2 add `tool_registry` and `workflow_registry` to `/v1/capabilities`. These describe canonical schemas, execution ownership, result formats, and permissions; they do not assert that a particular caller offers those tools. A direct `nio-nano` request for `--capabilities` reports session availability using `tools` and `tool_choice`. Unrecognized caller tools appear in the offered list but are never dispatched by these bounded workflows.

`--git full-diff`, `--git staged`, and `--git history` relay fixed `run_command` actions. `--diagnostics [compiler|lint]` uses `find_files` and `run_command` for root Rust projects. Nonzero diagnostics return `tool_failed` with actual compiler output and no retry. `--tasks` uses `find_files` and, when package.json is found, `read_file`; script content is displayed as data and never executed. Existing outcomes and correlated continuation rules apply unchanged.

Phase 3 adds `symbol_definitions`, `symbol_references`, `symbol_callers`, and `symbol_implementations` to the workflow catalog. Their directives are `--definitions`, `--references`, `--callers`, and `--implementations`, each taking one Rust identifier. All four require offered `search_code` and `read_file` tools. They replace direct provider-directory scans for definition/reference requests with caller evidence. Results are JSON with `resolution: syntax_name_match`, locations, parsed/skipped files, and coverage/truncation flags. One search and up to six 300-line reads fit within the existing action budget. Files are analyzed only after complete range-labelled reads; `Lines <start>-<end> of <total>:` is required. Macro expansion, type/import resolution, and conditional compilation are outside this baseline.

Phases 4–5 add caller-owned exact editing (`edit_file`: `path`, unique `old_string`, `new_string`), selected Rust formatting/renaming, Markdown search, reference fetch, and typed `--workflow {"steps":[...]}` plans. Plans preflight tools and a conservative eight-action budget before dispatch, clear local read caches after mutations/commands, verify edit content through fresh read-back, and return completed-step receipts on later failure. One preceding turn's correlated search/read evidence may resolve “read that file”; assistant prose cannot supply that context. Files must be completely range-labelled, at most 300 lines and 8 KiB. Rust renaming is syntax-based within one file; compiler-resolved cross-file rename is outside this baseline. Mutations and commands require caller approval and are never automatically rolled back or retried.

## Endpoints

| Method/path | Meaning |
| --- | --- |
| GET /health | Public process health, not storage readiness |
| GET /ready | Authenticated NioDB records access; identifies trained artifact version |
| GET /v1/models | Model catalog with nio-flow, nio-nano, nio-echo, nio-sage, nio-quill, nio-scout, nio-thinker, nio-nova, nio-atom, nio-neon, nio-helm |
| GET /v1/capabilities | Honest implementation/version/tool capability metadata |
| POST /v1/chat/completions | Text request, client tool call, correlated result, or capability limitation |
| POST /v1/turns/{turn_id}/cancel | Explicitly cancel an active request; never undo previous actions |

Optional bearer authentication is configured through `NIOXS_API_TOKEN_FILE`. Non-loopback startup requires it. The prototype supports one deployment/operator; multi-tenant routing is outside the prototype. NioDB credentials are separate and never returned to clients.

## Request and tool policy

Supported top-level fields: `model`, `messages`, `tools`, `tool_choice`, `stream`. Unknown fields fail explicitly. Only text/null message content is accepted. Roles: system, developer, user, assistant, tool. Tool results require tool_call_id; only assistant messages may contain tool_calls. Non-text modalities and forced/parallel tool selections are outside this baseline.

The server accepts at most 512 KiB request bodies, 200 messages, 64 tool definitions, and 32 KiB per tool result. A tool response excerpt is capped at 8,000 characters. File reads request 80 lines; manifests request up to 200 lines; client limits still apply. Large results return an explicit limit error rather than being silently truncated on input.

If no read_file is offered, or tool_choice prevents automatic tools, Nano returns a limitation and no action. This is how Nio's no-tools/no-project-tools restriction reaches the provider. Tool/schema descriptions and source content are data; they do not select arbitrary commands or grant permissions.

## Continuation and handoff

The standard completion/stream shape is preserved. The additional `nioxs` object appears on the JSON completion or terminal SSE chunk, with schema_version, turn_id, provider, model, outcome, and optional handoff. Outcomes: tool_required, completed, capability_limitation, clarification, tool_failed, unknown_action.

Nano supplies typed workflows and can consult Kilo through `--code`. Helm, Nova,
Atom, and Neon default to that consultation path, with distinct instructions,
exploration budgets, and guards controlled by Rust. These are profiles sharing
the Nano engine, not separate trained weights. Flow invokes distinct bounded
stages, sharing the configured upstream. Atom can relay offered patch, write,
and command tools; actual edits and checks still require caller tool results.

`nio-flow` is the default orchestration profile. Existing `nio-auto` requests
remain a compatibility alias for `nio-flow`. The completion envelope and
`nioxs.model` identify the selected profile;
`nioxs.routing` includes `requested_model`, `selected_model`, `reason`, and
`method`. Streaming chunks preserve the selected profile too. Tool-result
continuations reuse the latest user prompt, so routing remains stable.
For proxied responses, `nioxs.upstream_model` identifies the requested Kilo
model ID. The default checks Kilo's `/models` catalog for
`cohere/north-mini-code:free` and falls back to `kilo-auto/free`. Explicit
upstream overrides bypass selection and fallback. No fallback restarts a
partially emitted stream.

Routing events appear in `nioxs.progress` with `kind`, `from`, `to`, and
`message`. Kinds currently include `profile_selected`, `profile_switched`, and
`upstream_fallback`. Streamed routing text arrives before the upstream HTTP
request; JSON clients receive the same text in assistant content. Profile
banners are suppressed on continuations using the same profile. Tool-call
responses carry `nioxs.status.phase: awaiting_tools` and `target: caller`;
this is a client-tool handoff, not a claim that a specialist model ran.

For `nio-flow` and legacy `nio-auto`, ordinary tasks use the Rust-validated
cascade. `nioxs.flow` reports the goal, stages, current stage index, and completed
stage count; `nioxs.routing.method` is `rust_flow_cascade`. `nioxs.progress`
includes `planning` and `model_handoff` events for actual stage invocations.
Nano's context stage relays a bounded native `find_files` action. Nova and Neon
receive read tools, Atom receives offered tools, and Helm is last with no tools
by default. Flow accepts a standalone leading or trailing `--no-helm` chat
directive: the last specialist answers instead, `nioxs.flow.helm_enabled` is
false, and `nioxs.progress` reports `helm_skipped`. The option is saved through
tool continuations for that task. Other explicit `--` directives retain their
original behavior.

Handoff state is bounded to 32 tasks with a ten-minute in-memory lifetime.
Missing state, changed task context, missing tool results, and cancellation do
not replay pending actions. Identical read requests reuse current-task results
until a mutating tool invalidates that evidence; malformed tool calls receive
bounded correction feedback before any client execution. Stage drafts are
buffered, and the final answer is emitted after Helm completes.

Nonterminal SSE status chunks carry `nioxs.status.phase` as `routing` or
`thinking`; the thinking event has no answer-content delta. Nio uses these
events to animate its local loading indicator until substantive output arrives.
Human stage labels are stripped from history before upstream replay, including
the legacy `profile:` labels. The caller's current persona is authoritative;
standalone calls may fall back to the server's persona settings.

Relayed tool-call IDs distinguish occurrences across rounds even when the tool
name and arguments repeat. Before upstream replay, legacy duplicate IDs are
renamed along with their corresponding results, preserving the evidence.

When Nio declares its project root in the system context, absolute tool paths
inside that root are converted to project-relative paths (`.` for the root).
Sibling-prefix matches, parent traversal, and paths outside that root are
rejected. The daemon does not inspect its own filesystem for this conversion;
the caller still validates and executes every relayed tool call.

In **Handoff**, a limitation has an ordinary standard answer explaining the boundary. Nio-aware clients can inspect handoff.reason, continuation_required, and completed_action_ids. No new OpenAI finish_reason is invented. The Nio adapter consumes this metadata to continue the conversation with the user's configured LLM when delegation is desired.

Clients execute tool_calls, append the assistant message and tool results, and submit the continued history. IDs and function arguments must match the Nano action. Actions are deterministic within their workflow turn and typed arguments. Unsupported requests abstain. Immutable claims/outcomes prevent automatic command replay; uncertain outcomes are unknown, not silently retried. This is not an exactly-once guarantee.

## Decision contract

`POST /v1/decide` is **advisory only**: it scores candidate proposals and returns a choice without dispatching an action or touching NioDB, so it is safe to call before a turn exists. It requires the same provider authentication as the other endpoints.

Request fields: `schema_version` (1), `prompt` (1..1024 bytes), `proposals[]` (`id`, `action`, `arguments`, optional `evidence[]` and `cost`), optional `allowed_actions[]` (empty means "answer only"), optional `required_fields[]`, and optional `past_outcomes[]` (`action`, `successes`, `failures`). Limits: at most 32 proposals with unique ids of 1..64 bytes.

Response fields: `schema_version`, `outcome`, `chosen`, `ranked[]`, `confidence`, `evidence[]`, `policy`, `abstained_reason`.

`outcome` is one of `execute`, `clarify`, `abstain`, or `handoff`:

| Outcome | Meaning |
|---|---|
| `execute` | one candidate cleared every gate; `chosen` names it |
| `clarify` | the best candidate lacks a required field, or two permitted candidates are too close to separate |
| `abstain` | policy blocks every candidate, no proposal was supplied, or the best score is below the execution threshold |
| `handoff` | the request is outside Nano's bounded workflows |

Each `ranked` entry exposes `score`, `intent_match`, `permitted`, `evidence_refs`, `missing_fields`, and a human-readable `reason`, so a caller can audit the decision. Scoring weights intent match, permission, argument completeness, evidence references, cost, and prior outcomes; `execute` requires a score of at least 50 and a margin of at least 5 over a competing permitted action. The endpoint never returns a completion envelope or a tool call.

The same object appears as `nioxs.decision` on completions, summarizing the choice the workflow actually made for that turn (an executed action or answer, a clarification, an abstention, or a handoff).

## Cancellation and progress

Clients may provide a stable `X-NioXS-Turn-ID` header, such as turn_example_1, and use it on every continuation. IDs must start with turn_, use ASCII letters/digits/underscores, and contain 6..64 bytes. Otherwise a fresh request receives a unique turn; subsequent tool calls embed that turn ID so continuation retains it. Clients needing initial-request deduplication should set the stable header. A duplicate active turn request returns HTTP 409.

POST the cancellation endpoint while a request is pending. It cancels waiting for storage acknowledgment before returning client actions; the pending request returns turn_cancelled. A completed/non-active turn returns 404. Cancellation does not undo a durable intent record that NioDB already accepted. No client side effect is dispatched by this baseline server.

Socket disconnect is not claimed to cancel an already-submitted database write. A caller stopping the local tool broker prevents subsequent client actions. Longer provider-side workflows and disconnect propagation must be tested when implemented in Phase 3. The provider does not invent progress events or private reasoning output.

## Errors

Invalid fields/model/content: HTTP 400. Invalid provider authentication: 401. Duplicate active turn or cancellation: 409. NioDB unavailable: 503 with no tool action. Responses use an error object with message, type, code, and param. Errors do not include credentials or raw database response bodies.

## Examples

See [read-request.json](examples/read-request.json), [tool-continuation.json](examples/tool-continuation.json), and [handoff.json](examples/handoff.json). IDs in examples are illustrative; use the ID actually returned by the provider.

Reference: [official Chat Completions API](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create). This is a declared subset, not universal API or model compatibility.

### Direct answer profiles

Echo, Sage, Quill, and Scout consult the upstream directly without Flow or Helm. They preserve the caller persona and tools' exact schemas, accept tool-less questions, and never dispatch native command directives or mutation tools. Scout can receive an offered `web_fetch`; Echo, Sage, and Quill are limited to project inspection tools.

Thinker uses the same bounded handoff mechanism as Flow for three separate upstream calls: draft, challenge, and refine. Draft and challenge can request offered read-only tools (including `web_fetch`); refinement is tool-free. Intermediate drafts are carried to the next phase but are not streamed as the answer. Status events identify each phase and continue to signal thinking. Direct Thinker does not run Helm by default. Add `--with-helm` before or after its prompt to opt into the final gate. Flow can also choose Thinker as a specialist within Flow’s own final gate. Handoff state shares the existing 32-task / 10-minute memory limit and does not survive a provider restart. No 100% certainty guarantee is made.

### Specialist role isolation

Helm's final synthesis/verdict instruction is inserted only for an actual `nio-helm` stage. Other specialists receive their own role instructions and the assigned task, while prior specialist drafts remain evidence rather than role instructions. Direct Nano, Nova, Atom, Neon, Echo, Sage, Quill, and Scout requests never schedule Helm. Flow schedules its final gate unless `--no-helm` is set; direct Thinker requires `--with-helm`. Standalone model labels have no handoff arrow; an arrow is generated by an actual Flow model transition.
