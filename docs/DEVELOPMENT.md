# Phase 2 local development

Build with the pinned Rust toolchain:

```sh
cargo build --workspace --locked
cargo test --workspace --locked
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
```

Node is used only by development smoke scripts. It is not NioJS, a provider runtime, or an inference dependency. Python is used only for offline training and evaluation; inference runs entirely in Rust.

## Local provider

Start a dedicated pinned NioDB instance. Provision credentials with its init-auth command; redirect its token-pair output to a private file because it prints credentials. Save only the client token to a mode-0600 token file. Use `--nio-bin nioxs-assistance-disabled --node-bin nioxs-sql-disabled --no-demo --no-pg` for core-record-only fixtures; no database-assisted inference is needed.

Set NIOXS_DB_URL and NIOXS_DB_TOKEN_FILE, then run:

```sh
NIOXS_DB_URL=http://127.0.0.1:7432 NIOXS_DB_TOKEN_FILE=/private/path/db.token cargo run -p nioxs-provider -- serve
```

NIOXS_LISTEN defaults to 127.0.0.1:7460. NIOXS_DEPLOYMENT_ID defaults to local. Use NIOXS_API_TOKEN_FILE for provider authentication; it is mandatory for non-loopback listening. Do not put tokens in command arguments or source files. `/health` is process health; `/ready` checks authenticated database access.

Nio now defaults to nioxs::nio-nano with a built-in endpoint of http://127.0.0.1:7460/v1. Kilo remains built in as another provider. Use :provider only to customize the NioXS endpoint or credentials. Existing saved model choices and explicit overrides take precedence; set `nio config set model nioxs::nio-nano` to change a saved choice. Start NioDB and NioXS, then run `read README.md` in a trusted project, or:

```sh
nio run --mode ask --trust-project "read README.md"
```

Nano recognizes natural variants and supports read/inspect/find/search/configured-check workflows. It returns actual bounded evidence. It does not explain arbitrary code or automatically switch to an LLM yet.

## Reproducible isolated checks

```sh
node scripts/phase2-smoke.mjs --niodb /path/to/niodb --nio /path/to/nio --provider ./target/debug/nioxs --report ./evaluation/reports/local-smoke.json
node scripts/browser-feasibility.mjs /path/to/obscura ./evaluation/reports/live-browser.json
```

The smoke script creates isolated temporary database files, private credentials, provider configuration, and projects; it stops its child processes. It does not alter normal Nio configuration or existing NioDB data. Temporary credential files remain private; remove the temporary fixture directory when no longer needed. Report files contain results, not tokens. Live browser checks require internet access and do not solve CAPTCHAs.

## Current limitations

Trained classifier plus typed workflows; no automatic fallback, chat browser execution, edits, or MCP. English/ASCII requests and relative paths without spaces. Immutable journal transitions, one deployment/operator, eight actions per workflow. Expiry metadata exists; retention cleanup and production hardening remain later phase work.

Crawl4AI is not integrated: the environment has no Python runtime, its browser compatibility is not proven here, and Obscura already exports the content needed for this spike. Its optional CDP runtime probe is explicitly deferred. No Python or Chromium was installed for the browser spike.

## Offline training and evaluation

```sh
python3 training/train.py
cargo test --workspace --locked
cargo run -p nioxs-nano --bin evaluate -- evaluation/phase2-frozen-holdout.jsonl
```

Training requires Python 3.13 and no third-party Python packages. A checksum-verified portable Python was used for this development run; provenance is recorded in dependency-pins.json. The separate frozen corpus must not become training input. Regeneration of the artifact should reproduce the recorded digest. The development test was used for feature diagnosis; acceptance uses the separate frozen corpus.

An external artifact requires both NIOXS_MODEL_FILE and NIOXS_MODEL_SHA256. The default artifact is embedded. Version/schema mismatch prevents startup.

## Configured checks

Cargo projects map test/build/lint to `cargo test --locked`, `cargo build --locked`, and `cargo clippy --locked -- -D warnings`. Node projects require the corresponding package.json script. An optional .nioxs.json takes precedence:

```json
{"schema_version":1,"commands":{"test":"npm run test","build":"npm run build","lint":"npm run lint"}}
```

Only the fixed named command set in the workflow controller is accepted. Use Build mode and approve the command through Nio:

```sh
nio run --mode build --trust-project "run tests"
```

Ask/Plan, absent tools, and denied approval cannot run commands. The development smoke uses --auto only for safe isolated fixture scripts.
