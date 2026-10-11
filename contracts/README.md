# Phase 2 provider contract

Schema version: 1. Provider: `nioxs`. Model: `nio-nano`. Runtime: `0.1.0-alpha.2`.

NioNano uses trained Naive Bayes inference and typed workflow validation. Read, inspect, filename search, literal code search, and configured checks are supported. The client must offer the required tools and allow automatic choice. Every actual file read/command is performed by Nio’s permission broker. No provider-side browser/shell/filesystem tools are exposed by chat.

## Endpoints

| Method/path | Meaning |
| --- | --- |
| GET /health | Public process health, not storage readiness |
| GET /ready | Authenticated NioDB records access; identifies trained artifact version |
| GET /v1/models | Model catalog with nio-nano |
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

Nano is a decision layer, not an LLM proxy: it does not call an external model and exposes no `consult_llm` tool. Prose that requires reasoning or authoring is answered by the client's own LLM, which may use Nano through this API for typed, verified actions.

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
