# Nano tool expansion

Phases 1–5 are implemented within the bounded scopes described below.

| Phase | Scope | Acceptance |
|---|---|---|
| 1: Foundation | Canonical registry with argument schemas, permission metadata, execution ownership, result formats, and per-session discovery | `--capabilities` reflects caller tool policy and identifies missing tools; the HTTP capability catalog exposes the registry |
| 2: Inspection | Full/staged Git diffs, recent history, Rust compiler/lint diagnostics, Cargo and package.json task discovery | Caller tool results provide actual evidence; failures retain output; discovery never executes scripts |
| 3: Navigation | Definitions, references, callers, implementations | Locate symbols and usages in Rust first, then extend language support |
| 4: Editing | Patches, selected-file formatting, symbol renaming | Caller-approved edits produce reviewable diffs and validation evidence |
| 5: Connected workflows | Follow-up context, chaining, documentation search and reference fetching | Complete a find/inspect/fix/test request with correlated evidence |

## Phases 1–2 usage

Select `nio-nano` directly and use:

```text
--capabilities
--git full-diff
--git staged
--git history
--diagnostics compiler
--diagnostics lint
--tasks
```

Natural-language equivalents include “what changed?”, “show recent commits”,
“check compiler errors”, “show lint diagnostics”, and “list project tasks”.
“What failed?” runs a fresh compiler check; it does not retrieve historical failures.

All new inspection actions execute through caller tools. Commands need an
offered `run_command` tool and caller approval. Results use the existing text
contract: `exit code: <integer>` followed by a newline and command output.
Nonzero command results stop the workflow with `tool_failed` and a bounded
excerpt, including compiler diagnostics. Nano does not retry commands.

Git full-diff means the unstaged diff of tracked files; staged changes use the
separate staged view. History is capped at 20 commits. Diffs disable external
diff helpers and text conversion. Reports are bounded excerpts and explicitly
mark truncation.

Diagnostics support root Rust manifests and fixed Cargo commands. Task discovery
supports root Cargo.toml and package.json, displays incomplete-inventory status,
and reports malformed manifests rather than guessing. Cargo tasks are built-in
mappings, not discovered custom aliases; package.json scripts are actual manifest
data. Makefiles and other languages remain future extensions.

The registry describes supported primitive contracts and these new workflows.
Caller-offered plugins are listed for visibility, but listing does not grant
permission or automatically enable arbitrary dispatch. New primitive support
requires an explicit workflow implementation and registry entry.

## Phase 3: Rust navigation

```text
--definitions parse_atomic
--references parse_atomic
--callers parse_atomic
--implementations Service
```

Natural-language requests include “where is parse_atomic defined?”, “find
references to parse_atomic”, “who calls parse_atomic?”, and “who implements
Service?”. Use one Rust identifier; raw identifiers such as `r#type` are supported.
Qualified paths and arbitrary expressions require clarification.

These workflows use caller-owned `search_code` and `read_file`; they never scan
the provider's filesystem or execute commands. Candidate files come from a
literal search restricted to `**/*.rs`. Repeated paths are deduplicated. Rust
source is parsed with `syn`; comments, string contents, and longer identifiers
do not become symbol matches. Definitions and usages are separate; callers
include function and method call sites with enclosing function names. Trait and
inherent implementation blocks are reported for matching traits or types.

Each turn performs one search and at most six read windows of 300 lines. Large
files are read in successive windows and analyzed only when complete. Read
results must use `Lines <start>-<end> of <total>:` followed by the source. Missing
or inconsistent coverage headers, changed line totals, invalid syntax, truncated
searches, or exhausted read budgets produce explicit coverage warnings and
skipped-file reasons. Reports remain valid JSON within the output budget and
mark omitted results. No matches means none in the fully read, parsed evidence;
it does not establish absence throughout the project.

This is syntax-based name matching, not compiler-resolved navigation. Same-name
symbols may belong to different scopes or types. Imports and receiver types are
not resolved, macros are not expanded, and conditional compilation is not
evaluated. Calls through function pointers are usages rather than proven calls
to the original function. Module-qualified symbol selection, other languages,
and language-server integration remain future extensions.

## Phase 4: caller-approved editing

```text
--patch {"path":"src/lib.rs","old":"old_name","new":"new_name"}
--format src/lib.rs
--rename old_name new_name src/lib.rs
```

Existing natural-language exact replacement requests also use this caller-owned
editing path. Nano reads the selected file, validates a unique replacement,
relays `edit_file` with `path`, `old_string`, and `new_string`, and reads again
to verify the proposed content. The old string is the complete observed file
contents, so a conforming caller rejects missing or ambiguous target matches.
Concurrent changes outside the matched text can still occur and are detected
by read-back where observable; this is not a transactional edit guarantee.
Caller edit approval remains required. Changed Rust source
must parse before dispatch. Reports contain reviewable whole-file diffs.

Formatting relays `rustfmt --edition 2024 --config skip_children=true -- '<file>'`
through caller command approval, then reads the selected file again. It does
not format the entire workspace. Rename changes same-name Rust identifier
tokens within one selected file, preserving comments and strings. It requires
one matching declaration and rejects detected shadowing, destination collisions,
and macro-token usages. It is a file-scoped syntax rename: imports, receiver
types, and usages in other files are not resolved. Compiler-aware project-wide
rename remains future work.

Editing supports complete files up to 300 lines and 8 KiB. Incomplete reads
stop before editing. Failed or mismatched read-back stops with an unverified
outcome; commands and mutations are not retried. Successful read-back validates
content; compiler/test validation requires an explicit check step. Completed
changes remain applied if a later step fails; there is no automatic rollback.

## Phase 5: connected workflows and follow-up context

```text
--workflow {"steps":[{"action":"search","query":"old_name"},{"action":"read","path":"$selected"},{"action":"patch","path":"$selected","old":"old_name","new":"new_name"},{"action":"check","kind":"test"}]}
--docs routing
--fetch https://example.com/reference
```

Supported steps are `search`/`docs` with `query`, `read`/`format` with `path`,
`patch`/`rename` with `path`, `old`, and `new`, `check` with `kind`, and `fetch`
with `url`. Checks support Rust `check`, `test`, `build`, and `lint` through
fixed Cargo commands after confirming a root Cargo.toml. Docs searches are
restricted to caller Markdown files; reference fetching requires offered
`web_fetch` and does not fall back to a provider-side browser.

All required tools and a conservative action budget are checked before the
first action. Plans support up to six steps and eight tool actions; edits cost
up to three actions and checks cost two. Ambiguous or incomplete searches do
not choose a file. `$selected` carries a unique evidence-backed path; after a
completed search or read, “read that file” also works in the next user turn.
Follow-ups derive paths from correlated tool history, never assistant prose,
and read fresh contents. Within a workflow, repeated reads may reuse observed
contents; every mutation and validation command clears that read cache.

Continuation calls replay the same typed plan and matching action receipts.
Unknown JSON fields, unsafe paths, tool denials, malformed coverage, or failed
commands stop the workflow. Reports preserve completed-step evidence and
explicitly flag bounded excerpts. A later failure does not imply earlier edits
were rolled back. Free-form diagnosis and authored fixes still use Flow/`--code`;
the local engine executes explicit edits rather than inventing replacements.
