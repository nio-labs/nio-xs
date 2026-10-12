# Changelog

Notable changes to NioXS / NioNano. The version matches the workspace
`Cargo.toml`; the git tag is `v<version>`.

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
