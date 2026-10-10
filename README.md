# NioXS

**The axis of Nio's non-LLM models.**

AXIS = AI + XS: NioAI carries AI; NioXS carries XS.

## Project identity

- Product: NioAI (nio).
- Required model provider: NioXS (provider ID: nioxs).
- First non-LLM model: NioNano (model ID: nio-nano).
- Future family: NioMini, NioCore, NioMax, NioUltra.
- Runtime and training implementation: closed source.
- Persistence: NioDB through its core HTTP records API.
- NioJS is outside v0.1.0; the provider and Nano runtime remain Rust.

## Planning documents

- [Full design and flows](NIOXS_PLAN.md)
- [Five-phase roadmap and acceptance gates](NIOXS_ROADMAP.md)
- [v0.1.0 task checklist](TODO_V0.1.0.md)

## Execution flow

NioAI → NioXS provider → NioNano → supported workflows and tools.

Planned Phase 4 behavior: when a task exceeds Nano's capability, NioXS returns a handoff and Nio continues on the user's configured LLM provider. File and command execution remain under Nio's permission broker.

## Core decisions

- NioXS is mandatory for access to NioNano, hosted or self-hosted.
- Obscura is the only browser engine; no Chromium.
- Browser search needs no search API key; blocked pages and CAPTCHAs are skipped.
- Research starts with a small page batch and expands according to complexity and coverage.
- NioDB stores bounded workflow records, research cache, and optional feedback; its Nio/LLM assistance is not used by Nano.
- Binaries and model artifacts are distributed through R2 or S3.
- OpenAI-compatible provider API first, MCP support follows the roadmap.

## Current status

Nio now defaults to `nioxs::nio-nano` in the development build. Kilo remains built in as another provider. Start the local NioDB and NioXS services before testing Nano; use `:provider` to customize the endpoint.

Phase 2 core implementation is available at `0.1.0-alpha.2`: a trained non-LLM intent classifier with Rust inference, file reading, project inspection, filename/code search, and approved configured checks. Natural prompts such as “now read readme.md” work. NioDB journals actions and prevents automatic replay of uncertain commands. Automatic LLM switching, chat browser research, and MCP remain later phases. The optional Crawl4AI probe is deferred.

- [Local development and startup](docs/DEVELOPMENT.md)
- [Provider contracts](contracts/README.md)
- [Storage contract](contracts/storage.md)
- [Phase 1 evidence](evaluation/PHASE1_EVIDENCE.md)
- [Phase 2 evidence and limitations](evaluation/PHASE2_EVIDENCE.md)
- [Dependency pins](dependency-pins.json)
- [License inventory](THIRD_PARTY.md)
