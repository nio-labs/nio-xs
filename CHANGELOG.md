# Changelog

Notable changes to NioXS / NioNano. The version matches the workspace
`Cargo.toml`; the git tag is `v<version>`.

## v0.4.0

Release theme: **orchestrated model roles with bounded, evidence-led handoffs.**

### Added
- NioFlow as the default profile, with bounded planning and model-specialist handoffs.
- NioEcho, NioSage, NioQuill, NioScout, NioThinker, NioNova, NioAtom,
  NioNeon, and NioHelm workflow profiles alongside NioNano.
- Optional high-complexity routing from Kilo North Mini Code to a free,
  tool-capable StepFun model discovered from Kilo's live `/models` catalog.
- Auto Free availability fallback and visible upstream-switch progress.
- Role skills, caller-persona preservation, bounded Thinker reflection, and
  Flow's `--no-helm` directive.
- Responsive GitHub Pages documentation using the NioJS documentation layout.

### Fixed
- General questions and feasibility prompts no longer trigger project scans.
- NioNano's explicit `--find-files` directive works with code mode enabled;
  Flow continues with a clear limitation when optional evidence gathering fails.
- Specialist prompts retain the caller persona and keep Helm's decision role
  out of other profiles' answers.

## v0.3.3

Release theme: **a tested protocol surface and an honest benchmark.** No
runtime behavior changes; this release hardens what v0.3.x already shipped.

### Added
- Acceptance coverage for the MCP stdio server in `crates/provider/src/mcp.rs`:
  the `initialize` handshake (protocol version, `tools` capability, non-empty
  server version), silent `notifications/initialized`, `tools/list` (including
  every `plugin_*` tool), `prompts/list`, a real `tools/call`, and the two error
  paths (unknown method `-32601`, malformed JSON `-32700`).

### Fixed
- `evaluate` was not hermetic: it read the ambient `.nioxs/config.json`, so a
  local `code: true` routed every corpus prompt to Coding and reported
  `acceptance_passed: false` with `f1 0.0`. It now runs the corpus under the
  default config, so the benchmark measures the model rather than the machine.

### Changed
- `docs/ROADMAP.md` gained an authoritative release & phase map; `docs/NEXT.md`
  and `docs/TODO_V0.1.0.md` describe the same state in its terms. Two stale
  claims were corrected: the upstream consultation is live on `--code` (only the
  `--coding:` spelling is retired), and edit paths are `safe_path`-gated.

## v0.3.2

Release theme: **Termux binaries.**

### Fixed
- Termux/Android binaries failed to build because the workflow set only the
  cargo linker and `cc-rs` could not find the NDK clang. The Android branch now
  also exports `CC`/`AR` and adds the NDK toolchain `bin` to `PATH`.

## v0.3.1

Release theme: **complete binaries.** v0.3.0 shipped only the linux-amd64,
darwin-arm64, and windows-amd64 assets because the cross-targets failed.

### Fixed
- The release workflow installed cross-compilation targets into `stable` while
  cargo built with the channel pinned in `rust-toolchain.toml`, so linux-arm64,
  darwin-amd64, termux (amd64/arm64), and windows-arm64 failed with
  "can't find crate for `core`". The setup action is now pinned to the same
  channel, so every artifact `bin/nioxs.js` requests is produced.

## v0.3.0

Release theme: **safe writes and an honest surface.**

### Added
- `--code` handoff for unbound code-modification verbs: `refactor`, `debug`,
  `rename`, `implement`, `migrate`, and `rewrite` with no bounded local workflow
  now return a clarification naming `--code` instead of dead-ending as
  `unsupported`.
- `nioxs_nano::SHIPPED_WORKFLOWS`, the single source of truth for the workflow
  list advertised by `/v1/capabilities`.

### Changed
- `/v1/capabilities` is regenerated from the shipped intent set (23 workflows)
  and now reports `mcp: true`.
- `--git` honors its subcommand: bare `--git` runs status+diff, `--git:status`
  and `--git:diff` run one each, and an unknown subcommand clarifies.
- README's directive table is the authoritative list; `docs/ROADMAP.md` and
  `docs/PLAN.md` point to it instead of duplicating it.
- `docs/NEXT.md` and `docs/TODO_V0.1.0.md` no longer claim the Kilo consultation
  is retired: it is live on the `--code` directive (only the `--coding:` spelling
  is retired).

### Fixed
- Every `coder.rs` edit path writes through a single `safe_path`-gated helper.
  Untrusted targets (`../`, absolute paths, `.env`, `.ssh`, `.key`, `.token`)
  are refused before any read or write, and rollbacks use the same gate.
- Intent telemetry: `ProjectSuggestions` and `TaskPlan` report their real labels
  instead of `unsupported`.

### Notes
- The upstream Kilo consultation remains available via `--code`; the retired
  name is `--coding`, not the feature.
- Deferred to a later release: the MCP acceptance gate, the real-phrasing
  (Phase B) evaluation corpus, and CI matrix validation for the Termux and
  windows-arm64 targets.
